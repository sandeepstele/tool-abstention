"""Contract tests for taxonomy and canonical artifact records."""

import math
from typing import Any

import pytest
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
    TaskPair,
    TaskRecord,
    ToolDefinition,
)
from tool_abstention.taxonomy import (
    DatasetSplit,
    DecisionClass,
    PerturbationType,
    TaskVariant,
)


def tool() -> ToolDefinition:
    return ToolDefinition(
        name="close_ticket",
        description="Close a support ticket.",
        parameters={
            "type": "object",
            "properties": {"ticket_id": {"type": "integer", "minimum": 1}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    )


def act_task(*, pair_id: str = "productivity-001") -> TaskRecord:
    return TaskRecord(
        id=f"{pair_id}-act",
        pair_id=pair_id,
        domain="productivity",
        split=DatasetSplit.TRAIN,
        variant=TaskVariant.ACT,
        generator_version="1.0.0",
        query="Close ticket 7.",
        tools=(tool(),),
        environment={"tickets": {"7": "open"}},
        label=DecisionClass.CALL,
        perturbation=None,
        expected=CallExpected(
            tool_name="close_ticket",
            arguments={"ticket_id": 7},
            expected_result={"ticket_id": 7, "status": "closed"},
        ),
    )


def abstain_expected(label: DecisionClass) -> tuple[PerturbationType, object]:
    if label is DecisionClass.ANSWER:
        return (
            PerturbationType.ANSWER_PROVIDED,
            AnswerExpected(validator=ExactAnswerValidator(value="Ticket 7 is closed.")),
        )
    if label is DecisionClass.CLARIFY:
        return (
            PerturbationType.REQUIRED_ARGUMENT_REMOVED,
            ClarifyExpected(missing_slots=("ticket_id",)),
        )
    if label is DecisionClass.REFUSE:
        return (
            PerturbationType.TOOL_REMOVED,
            RefuseExpected(
                unavailable_capability="close_ticket", reason="missing_tool"
            ),
        )
    if label is DecisionClass.NOOP:
        return (
            PerturbationType.ALREADY_SATISFIED,
            NoopExpected(
                state_assertion="Ticket 7 is already closed.",
                allowed_markers=("already closed", "no action needed"),
            ),
        )
    raise AssertionError(f"unsupported abstain label: {label}")


def abstain_task(
    label: DecisionClass, *, pair_id: str = "productivity-001"
) -> TaskRecord:
    perturbation, expected = abstain_expected(label)
    tools = () if label is DecisionClass.REFUSE else (tool(),)
    return TaskRecord.model_validate(
        {
            "id": f"{pair_id}-abstain",
            "pair_id": pair_id,
            "domain": "productivity",
            "split": "train",
            "variant": "abstain",
            "generator_version": "1.0.0",
            "query": "Do not close ticket 7.",
            "tools": tools,
            "environment": {"tickets": {"7": "closed"}},
            "label": label,
            "perturbation": perturbation,
            "expected": expected,
        }
    )


@pytest.mark.parametrize(
    "label",
    [
        DecisionClass.ANSWER,
        DecisionClass.CLARIFY,
        DecisionClass.REFUSE,
        DecisionClass.NOOP,
    ],
)
def test_valid_task_and_pair_for_each_abstention_class(label: DecisionClass) -> None:
    act = act_task()
    abstain = abstain_task(label)
    pair = TaskPair(pair_id=act.pair_id, act=act, abstain=abstain)
    assert pair.abstain.label is label


@pytest.mark.parametrize(
    ("validator", "kind"),
    [
        (ExactAnswerValidator(value="Paris"), "exact"),
        (NormalizedTextAnswerValidator(value="Paris, France"), "normalized_text"),
        (
            NumericAnswerValidator(value=3.14, absolute_tolerance=0.01, unit="m"),
            "numeric",
        ),
        (SetAnswerValidator(values=("red", "blue")), "set"),
        (
            DomainAnswerValidator(
                validator_id="ticket_status", parameters={"status": "closed"}
            ),
            "domain",
        ),
    ],
)
def test_answer_validator_variants_round_trip(validator: object, kind: str) -> None:
    expected = AnswerExpected.model_validate({"validator": validator})
    restored = AnswerExpected.model_validate_json(expected.model_dump_json())
    assert restored.validator.kind == kind
    assert restored == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_numeric_validator_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        NumericAnswerValidator(value=value, absolute_tolerance=0)


def test_numeric_validator_rejects_negative_tolerance() -> None:
    with pytest.raises(ValidationError):
        NumericAnswerValidator(value=1, absolute_tolerance=-0.1)


def test_duplicate_set_values_ignore_case() -> None:
    with pytest.raises(ValidationError, match="unique ignoring case"):
        SetAnswerValidator(values=("Red", "red"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "wrong-act"}, "task id must be"),
        ({"label": "ANSWER"}, "expected behavior kind must match"),
        ({"perturbation": "tool_removed"}, "act tasks cannot declare"),
        ({"variant": "abstain"}, "task id must be"),
    ],
)
def test_invalid_act_invariants(change: dict[str, Any], message: str) -> None:
    data = act_task().model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError, match=message):
        TaskRecord.model_validate(data)


def test_abstain_rejects_call_label() -> None:
    data = act_task().model_dump(mode="json")
    data.update(
        id="productivity-001-abstain",
        variant="abstain",
        perturbation="tool_removed",
    )
    with pytest.raises(ValidationError, match="abstain tasks cannot use"):
        TaskRecord.model_validate(data)


def test_act_rejects_matching_non_call_behavior() -> None:
    data = act_task().model_dump(mode="json")
    data["label"] = "ANSWER"
    data["expected"] = {
        "kind": "ANSWER",
        "validator": {"kind": "exact", "value": "Already complete."},
    }
    with pytest.raises(ValidationError, match="act tasks must use the CALL label"):
        TaskRecord.model_validate(data)


def test_abstain_requires_perturbation() -> None:
    data = abstain_task(DecisionClass.ANSWER).model_dump(mode="json")
    data["perturbation"] = None
    with pytest.raises(ValidationError, match="must declare a perturbation"):
        TaskRecord.model_validate(data)


def test_perturbation_must_match_label() -> None:
    data = abstain_task(DecisionClass.ANSWER).model_dump(mode="json")
    data["perturbation"] = "tool_removed"
    with pytest.raises(ValidationError, match="does not produce"):
        TaskRecord.model_validate(data)


def test_tool_definition_rejects_invalid_schema() -> None:
    with pytest.raises(ValidationError, match="invalid Draft 2020-12"):
        ToolDefinition(
            name="bad_tool",
            description="Invalid schema.",
            parameters={"type": "not-a-json-schema-type"},
        )


def test_tool_definition_requires_object_root() -> None:
    with pytest.raises(ValidationError, match="root type must be 'object'"):
        ToolDefinition(
            name="bad_tool",
            description="Array schema.",
            parameters={"type": "array"},
        )


def test_expected_tool_must_be_visible() -> None:
    data = act_task().model_dump(mode="json")
    data["tools"] = []
    with pytest.raises(ValidationError, match="expected tool must exist"):
        TaskRecord.model_validate(data)


def test_expected_arguments_must_match_tool_schema() -> None:
    data = act_task().model_dump(mode="json")
    data["expected"]["arguments"] = {"ticket_id": "seven"}
    with pytest.raises(ValidationError, match="arguments are invalid"):
        TaskRecord.model_validate(data)


def test_duplicate_tool_names_are_rejected() -> None:
    data = act_task().model_dump(mode="json")
    data["tools"] = [data["tools"][0], data["tools"][0]]
    with pytest.raises(ValidationError, match="tool names must be unique"):
        TaskRecord.model_validate(data)


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"kind": "CLARIFY", "missing_slots": ["id", "id"]}, "must be unique"),
        (
            {
                "kind": "NOOP",
                "state_assertion": "Done.",
                "allowed_markers": ["Done", "done"],
            },
            "unique ignoring case",
        ),
    ],
)
def test_duplicate_expected_markers_are_rejected(
    expected: dict[str, Any], message: str
) -> None:
    model = ClarifyExpected if expected["kind"] == "CLARIFY" else NoopExpected
    with pytest.raises(ValidationError, match=message):
        model.model_validate(expected)


def test_nested_non_finite_json_is_rejected() -> None:
    data = act_task().model_dump(mode="json")
    data["environment"] = {"nested": [1, math.inf]}
    with pytest.raises(ValidationError, match="non-finite"):
        TaskRecord.model_validate(data)


@pytest.mark.parametrize("field", ["domain", "split", "generator_version"])
def test_pair_members_must_share_metadata(field: str) -> None:
    act = act_task()
    data = abstain_task(DecisionClass.NOOP).model_dump(mode="json")
    replacements: dict[str, object] = {
        "domain": "finance",
        "split": "test",
        "generator_version": "2.0.0",
    }
    data[field] = replacements[field]
    abstain = TaskRecord.model_validate(data)
    with pytest.raises(ValidationError, match=f"must share {field}"):
        TaskPair(pair_id=act.pair_id, act=act, abstain=abstain)


def test_pair_id_and_member_variants_are_validated() -> None:
    act = act_task()
    abstain = abstain_task(DecisionClass.ANSWER)
    with pytest.raises(ValidationError, match="pair_id values must match"):
        TaskPair(pair_id="different-001", act=act, abstain=abstain)
    with pytest.raises(ValidationError, match="act member must use"):
        TaskPair(pair_id=act.pair_id, act=abstain, abstain=abstain)
    with pytest.raises(ValidationError, match="abstain member must use"):
        TaskPair(pair_id=act.pair_id, act=act, abstain=act)


def test_prediction_success_and_failure_states() -> None:
    success = PredictionRecord(
        task_id="productivity-001-act",
        raw_text="",
        tool_call=ParsedToolCall(name="close_ticket", arguments={"ticket_id": 7}),
        latency_ms=12.5,
        input_tokens=20,
        output_tokens=8,
    )
    failure = PredictionRecord(
        task_id="productivity-001-act",
        raw_text="",
        latency_ms=1,
        inference_error="backend unavailable",
    )
    assert success.tool_call is not None
    assert failure.inference_error == "backend unavailable"


@pytest.mark.parametrize(
    "change",
    [
        {"latency_ms": -1},
        {"input_tokens": -1},
        {"latency_ms": math.inf},
        {"inference_error": "failed", "raw_text": "partial"},
        {
            "inference_error": "failed",
            "tool_call": {"name": "close_ticket", "arguments": {}},
        },
    ],
)
def test_prediction_rejects_invalid_states(change: dict[str, Any]) -> None:
    data: dict[str, Any] = {
        "task_id": "productivity-001-act",
        "raw_text": "answer",
        "latency_ms": 1,
    }
    if "inference_error" in change:
        data["raw_text"] = ""
    data.update(change)
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(data)


def test_evaluation_record_contract() -> None:
    valid = EvaluationRecord(
        task_id="productivity-001-act",
        predicted_class=DecisionClass.CALL,
        correct=True,
        reason_code="correct_tool_call",
    )
    assert valid.correct
    with pytest.raises(
        ValidationError, match="unclassified prediction cannot be correct"
    ):
        EvaluationRecord(
            task_id="productivity-001-act",
            predicted_class=None,
            correct=True,
            reason_code="parse_error",
        )


def test_task_json_round_trip_preserves_information() -> None:
    original = act_task()
    restored = TaskRecord.model_validate_json(original.model_dump_json())
    assert restored == original
