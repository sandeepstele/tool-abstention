"""Deterministic model-output parsing and five-class evaluation."""

import json
import re
import unicodedata

from pydantic import ValidationError

from tool_abstention.records import (
    AnswerExpected,
    CallExpected,
    ClarifyExpected,
    DomainAnswerValidator,
    EvaluationRecord,
    ExactAnswerValidator,
    NoopExpected,
    NormalizedTextAnswerValidator,
    NumericAnswerValidator,
    ParsedToolCall,
    PredictionRecord,
    RefuseExpected,
    SetAnswerValidator,
    TaskRecord,
)
from tool_abstention.taxonomy import DecisionClass

TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
REFUSAL = re.compile(
    r"\b(cannot|can't|unable|no available tool|no tool|not equipped|unsupported|"
    r"none of (?:the )?(?:provided|available) (?:functions|tools))\b",
    re.IGNORECASE,
)
GENERIC_NOOP = (
    "already complete",
    "already completed",
    "already closed",
    "no action needed",
    "nothing to do",
)


def normalize_text(value: str) -> str:
    """Normalize Unicode, whitespace, case, and edge punctuation."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = " ".join(normalized.split())
    return normalized.strip(" \t\n\r.,!?;:'\"")


def _decode_arguments(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _call_payload(value: object) -> object:
    if not isinstance(value, dict):
        return value
    if "tool_call" in value:
        return value["tool_call"]
    calls = value.get("tool_calls")
    if isinstance(calls, list) and len(calls) == 1:
        call = calls[0]
        if isinstance(call, dict) and "function" in call:
            return call["function"]
        return call
    if "function" in value and isinstance(value["function"], dict):
        return value["function"]
    return value


def parse_tool_call_text(raw_text: str) -> ParsedToolCall | None:
    """Parse plain, wrapped, OpenAI-style, or Qwen XML tool-call JSON."""
    text = raw_text.strip()
    match = TOOL_BLOCK.fullmatch(text)
    if match:
        text = match.group(1)
    if not text.startswith("{"):
        return None
    try:
        payload = _call_payload(json.loads(text))
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        arguments = _decode_arguments(payload.get("arguments"))
        return ParsedToolCall.model_validate({"name": name, "arguments": arguments})
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def looks_like_tool_call(raw_text: str) -> bool:
    """Detect a malformed or valid tool-call attempt."""
    lowered = raw_text.casefold()
    return "<tool_call" in lowered or any(
        marker in lowered for marker in ('"tool_call"', '"tool_calls"', '"arguments"')
    )


def _answer_is_correct(expected: AnswerExpected, response: str) -> bool:
    validator = expected.validator
    if isinstance(validator, ExactAnswerValidator):
        if validator.case_sensitive:
            return response.strip() == validator.value.strip()
        return normalize_text(response) == normalize_text(validator.value)
    if isinstance(validator, NormalizedTextAnswerValidator):
        return normalize_text(response) == normalize_text(validator.value)
    if isinstance(validator, NumericAnswerValidator):
        match = NUMBER.search(response.replace(",", ""))
        if match is None:
            return False
        if validator.unit and normalize_text(validator.unit) not in normalize_text(
            response
        ):
            return False
        return (
            abs(float(match.group()) - validator.value) <= validator.absolute_tolerance
        )
    if isinstance(validator, SetAnswerValidator):
        pieces = re.split(r"\s*(?:,|;|\band\b|\n|•)\s*", response, flags=re.IGNORECASE)
        pieces = [piece for piece in pieces if piece]
        if validator.case_sensitive:
            return set(pieces) == set(validator.values)
        return {normalize_text(piece) for piece in pieces} == {
            normalize_text(value) for value in validator.values
        }
    if isinstance(validator, DomainAnswerValidator):
        return False
    raise AssertionError("unreachable answer validator")


def _answer_is_semantically_correct(expected: AnswerExpected, response: str) -> bool:
    """Accept a validated atomic answer inside an otherwise natural response."""
    validator = expected.validator
    if isinstance(validator, ExactAnswerValidator):
        if validator.case_sensitive:
            return validator.value.strip() in response
        return normalize_text(validator.value) in normalize_text(response)
    if isinstance(validator, NormalizedTextAnswerValidator):
        return normalize_text(validator.value) in normalize_text(response)
    return _answer_is_correct(expected, response)


def _mentions_missing_slot(expected: ClarifyExpected, response: str) -> bool:
    normalized = normalize_text(response).replace("_", " ")
    return any(slot.replace("_", " ") in normalized for slot in expected.missing_slots)


def _classify_text(task: TaskRecord, response: str) -> DecisionClass | None:
    text = response.strip()
    if not text:
        return None
    if REFUSAL.search(text):
        return DecisionClass.REFUSE
    if (
        isinstance(task.expected, ClarifyExpected)
        and (
            "?" in text or normalize_text(text).startswith(("which ", "what ", "who "))
        )
        and _mentions_missing_slot(task.expected, text)
    ):
        return DecisionClass.CLARIFY
    markers = list(GENERIC_NOOP)
    if isinstance(task.expected, NoopExpected):
        markers.extend(task.expected.allowed_markers)
    normalized = normalize_text(text)
    if any(normalize_text(marker) in normalized for marker in markers):
        return DecisionClass.NOOP
    return DecisionClass.ANSWER


def evaluate_prediction(
    task: TaskRecord, prediction: PredictionRecord
) -> EvaluationRecord:
    """Classify and deterministically score one stored prediction."""
    if prediction.task_id != task.id:
        raise ValueError("prediction task_id does not match task")
    if prediction.inference_error is not None:
        return EvaluationRecord(
            task_id=task.id,
            predicted_class=None,
            behavior_correct=False,
            semantic_correct=False,
            protocol_correct=False,
            correct=False,
            reason_code="inference_error",
        )
    parsed = prediction.tool_call or parse_tool_call_text(prediction.raw_text)
    if parsed is not None:
        behavior_correct = task.label is DecisionClass.CALL
        semantic_correct = (
            isinstance(task.expected, CallExpected)
            and parsed.name == task.expected.tool_name
            and parsed.arguments == task.expected.arguments
        )
        correct = behavior_correct and semantic_correct
        return EvaluationRecord(
            task_id=task.id,
            predicted_class=DecisionClass.CALL,
            behavior_correct=behavior_correct,
            semantic_correct=semantic_correct,
            protocol_correct=True,
            correct=correct,
            reason_code="correct_tool_call" if correct else "incorrect_tool_call",
            parsed_tool_call=parsed,
        )
    if looks_like_tool_call(prediction.raw_text):
        return EvaluationRecord(
            task_id=task.id,
            predicted_class=DecisionClass.CALL,
            behavior_correct=task.label is DecisionClass.CALL,
            semantic_correct=False,
            protocol_correct=False,
            correct=False,
            reason_code="malformed_tool_call",
        )
    predicted_class = _classify_text(task, prediction.raw_text)
    behavior_correct = predicted_class is task.label
    semantic_correct = False
    reason = "wrong_behavior"
    if behavior_correct:
        if isinstance(task.expected, AnswerExpected):
            semantic_correct = _answer_is_semantically_correct(
                task.expected, prediction.raw_text
            )
            reason = "correct_answer" if semantic_correct else "incorrect_answer"
        elif isinstance(task.expected, ClarifyExpected):
            semantic_correct = _mentions_missing_slot(
                task.expected, prediction.raw_text
            )
            reason = (
                "correct_clarification"
                if semantic_correct
                else "incorrect_clarification"
            )
        elif isinstance(task.expected, RefuseExpected):
            semantic_correct = bool(REFUSAL.search(prediction.raw_text))
            reason = "correct_refusal" if semantic_correct else "incorrect_refusal"
        elif isinstance(task.expected, NoopExpected):
            normalized = normalize_text(prediction.raw_text)
            semantic_correct = any(
                normalize_text(marker) in normalized
                for marker in task.expected.allowed_markers
            )
            reason = "correct_noop" if semantic_correct else "incorrect_noop"
    if predicted_class is None:
        reason = "empty_output"
    correct = behavior_correct and semantic_correct
    return EvaluationRecord(
        task_id=task.id,
        predicted_class=predicted_class,
        behavior_correct=behavior_correct,
        semantic_correct=semantic_correct,
        protocol_correct=bool(prediction.raw_text.strip()),
        correct=correct,
        reason_code=reason,
    )
