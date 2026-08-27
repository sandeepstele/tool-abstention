"""Tests and adversarial calibration for deterministic evaluation."""

import pytest

from tool_abstention.evaluator import (
    evaluate_prediction,
    looks_like_tool_call,
    normalize_text,
    parse_tool_call_text,
)
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.records import (
    AnswerExpected,
    CallExpected,
    DomainAnswerValidator,
    ExactAnswerValidator,
    NormalizedTextAnswerValidator,
    NumericAnswerValidator,
    ParsedToolCall,
    PredictionRecord,
    SetAnswerValidator,
    TaskPair,
    TaskRecord,
)
from tool_abstention.taxonomy import DatasetSplit, DecisionClass


def pairs() -> list[TaskPair]:
    return generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=1,
            generator_version="1.0.0",
            split=DatasetSplit.TRAIN,
        )
    )


def prediction(task: TaskRecord, text: str) -> PredictionRecord:
    return PredictionRecord(task_id=task.id, raw_text=text, latency_ms=1)


def correct_prediction(task: TaskRecord) -> PredictionRecord:
    expected = task.expected
    if isinstance(expected, CallExpected):
        return PredictionRecord(
            task_id=task.id,
            raw_text="",
            tool_call=ParsedToolCall(
                name=expected.tool_name, arguments=expected.arguments
            ),
            latency_ms=1,
        )
    if task.label is DecisionClass.ANSWER:
        assert isinstance(expected, AnswerExpected)
        assert isinstance(expected.validator, ExactAnswerValidator)
        return prediction(task, expected.validator.value)
    if task.label is DecisionClass.CLARIFY:
        return prediction(task, "Which recipient should I send the message to?")
    if task.label is DecisionClass.REFUSE:
        return prediction(
            task, "I cannot do that because no available tool supports it."
        )
    if task.label is DecisionClass.NOOP:
        return prediction(task, "The ticket is already closed; no action needed.")
    raise AssertionError("unsupported task")


@pytest.mark.parametrize(
    "raw",
    [
        '{"name":"close_ticket","arguments":{"ticket_id":7}}',
        '<tool_call>{"name":"close_ticket","arguments":{"ticket_id":7}}</tool_call>',
        '{"tool_call":{"name":"close_ticket","arguments":{"ticket_id":7}}}',
        (
            '{"tool_calls":[{"function":{"name":"close_ticket",'
            '"arguments":"{\\"ticket_id\\":7}"}}]}'
        ),
        ('{"function":{"name":"close_ticket","arguments":{"ticket_id":7}}}'),
        '{"tool_calls":[{"name":"close_ticket","arguments":{"ticket_id":7}}]}',
    ],
)
def test_parse_supported_tool_call_formats(raw: str) -> None:
    assert parse_tool_call_text(raw) == ParsedToolCall(
        name="close_ticket", arguments={"ticket_id": 7}
    )


@pytest.mark.parametrize(
    "raw",
    [
        "ordinary response",
        "[]",
        '{"name":"close_ticket"}',
        '<tool_call>{"name":</tool_call>',
        '{"tool_calls":[]}',
        '{"tool_calls":[{},{}]}',
        '{"tool_call":[]}',
    ],
)
def test_unparseable_tool_calls_return_none(raw: str) -> None:
    assert parse_tool_call_text(raw) is None


def test_tool_attempt_detection_and_normalization() -> None:
    assert looks_like_tool_call('<tool_call>{"bad": true}')
    assert looks_like_tool_call('{"arguments": {}}')
    assert not looks_like_tool_call("A normal answer")
    assert normalize_text("  PARIS,\n France! ") == "paris, france"


def test_every_correct_productivity_behavior_passes() -> None:
    for pair in pairs():
        for task in (pair.act, pair.abstain):
            evaluation = evaluate_prediction(task, correct_prediction(task))
            assert evaluation.correct, (task.id, evaluation)
            assert evaluation.predicted_class is task.label


def test_wrong_and_malformed_calls_are_call_failures() -> None:
    task = pairs()[0].act
    wrong = PredictionRecord(
        task_id=task.id,
        raw_text="",
        tool_call=ParsedToolCall(name="send_email", arguments={}),
        latency_ms=1,
    )
    wrong_result = evaluate_prediction(task, wrong)
    assert wrong_result.predicted_class is DecisionClass.CALL
    assert not wrong_result.correct
    assert wrong_result.reason_code == "incorrect_tool_call"

    malformed = evaluate_prediction(
        task, prediction(task, '<tool_call>{"name":</tool_call>')
    )
    assert malformed.predicted_class is DecisionClass.CALL
    assert malformed.reason_code == "malformed_tool_call"


def test_task_id_and_inference_errors() -> None:
    task = pairs()[0].act
    with pytest.raises(ValueError, match="task_id does not match"):
        evaluate_prediction(
            task, PredictionRecord(task_id="different-act", raw_text="", latency_ms=1)
        )
    failed = PredictionRecord(
        task_id=task.id, raw_text="", latency_ms=1, inference_error="timeout"
    )
    result = evaluate_prediction(task, failed)
    assert result.predicted_class is None
    assert result.reason_code == "inference_error"


def test_empty_wrong_behavior_and_missing_slot_failures() -> None:
    answer_task = pairs()[0].abstain
    empty = evaluate_prediction(answer_task, prediction(answer_task, ""))
    assert empty.predicted_class is None
    assert empty.reason_code == "empty_output"

    clarify_task = pairs()[1].abstain
    vague = evaluate_prediction(
        clarify_task, prediction(clarify_task, "Could you clarify?")
    )
    assert vague.predicted_class is DecisionClass.ANSWER
    assert not vague.correct
    refusal = evaluate_prediction(
        clarify_task, prediction(clarify_task, "I cannot send this.")
    )
    assert refusal.predicted_class is DecisionClass.REFUSE
    assert refusal.reason_code == "wrong_behavior"


def answer_task_with(validator: object) -> TaskRecord:
    task = pairs()[0].abstain
    data = task.model_dump(mode="json")
    data["expected"] = AnswerExpected.model_validate(
        {"kind": "ANSWER", "validator": validator}
    ).model_dump(mode="json")
    return TaskRecord.model_validate(data)


@pytest.mark.parametrize(
    ("validator", "response", "correct"),
    [
        (ExactAnswerValidator(value="Paris", case_sensitive=True), "Paris", True),
        (
            ExactAnswerValidator(value="Paris", case_sensitive=True),
            "The answer is Paris.",
            True,
        ),
        (ExactAnswerValidator(value="Paris", case_sensitive=True), "paris", False),
        (ExactAnswerValidator(value="Paris", case_sensitive=False), " PARIS! ", True),
        (NormalizedTextAnswerValidator(value="Café au lait"), "Café au lait.", True),
        (
            NumericAnswerValidator(value=42, absolute_tolerance=0.1, unit="kg"),
            "42.05 kg",
            True,
        ),
        (
            NumericAnswerValidator(value=42, absolute_tolerance=0.1, unit="kg"),
            "42.2 kg",
            False,
        ),
        (
            NumericAnswerValidator(value=42, absolute_tolerance=0.1, unit="kg"),
            "42.0",
            False,
        ),
        (SetAnswerValidator(values=("red", "blue")), "Blue and red", True),
        (
            SetAnswerValidator(values=("Red", "blue"), case_sensitive=True),
            "blue, Red",
            True,
        ),
        (SetAnswerValidator(values=("red", "blue")), "red, green", False),
        (
            DomainAnswerValidator(
                validator_id="future_validator", parameters={"expected": "x"}
            ),
            "x",
            False,
        ),
    ],
)
def test_answer_validator_behavior(
    validator: object, response: str, correct: bool
) -> None:
    task = answer_task_with(validator)
    result = evaluate_prediction(task, prediction(task, response))
    assert result.correct is correct


def test_numeric_answer_without_a_number_fails() -> None:
    task = answer_task_with(
        NumericAnswerValidator(value=42, absolute_tolerance=0.1, unit=None)
    )
    assert not evaluate_prediction(task, prediction(task, "forty-two")).correct


def test_refusal_and_noop_require_expected_intent() -> None:
    refuse_task = pairs()[2].abstain
    calibrated = evaluate_prediction(
        refuse_task,
        prediction(refuse_task, "None of the provided functions can do that."),
    )
    assert calibrated.predicted_class is DecisionClass.REFUSE
    assert calibrated.behavior_correct
    assert calibrated.semantic_correct
    assert calibrated.protocol_correct
    assert calibrated.correct
    assert not evaluate_prediction(
        refuse_task, prediction(refuse_task, "I do not want to do that.")
    ).correct
    noop_task = pairs()[3].abstain
    result = evaluate_prediction(noop_task, prediction(noop_task, "Nothing to do."))
    assert result.predicted_class is DecisionClass.NOOP
    assert not result.correct


def test_200_case_construction_calibration_matrix() -> None:
    all_pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=10,
            generator_version="1.0.0",
            split=DatasetSplit.TRAIN,
        )
    )
    acts = [pair.act for pair in all_pairs]
    abstains = [pair.abstain for pair in all_pairs]
    tasks = acts[:8]
    for label in (
        DecisionClass.ANSWER,
        DecisionClass.CLARIFY,
        DecisionClass.REFUSE,
        DecisionClass.NOOP,
    ):
        tasks.extend([task for task in abstains if task.label is label][:8])
    assert len(tasks) == 40
    cases: list[PredictionRecord] = []
    for task in tasks[:40]:
        cases.extend(
            (
                correct_prediction(task),
                prediction(task, ""),
                prediction(task, '<tool_call>{"name":</tool_call>'),
                prediction(
                    task, "I cannot continue because no available tool can help."
                ),
                prediction(task, "This is already closed; no action needed."),
            )
        )
    assert len(cases) == 200
    results = []
    for index, task in enumerate(tasks[:40]):
        block = cases[index * 5 : index * 5 + 5]
        block_results = [evaluate_prediction(task, case) for case in block]
        assert [result.predicted_class for result in block_results] == [
            task.label,
            None,
            DecisionClass.CALL,
            DecisionClass.REFUSE,
            DecisionClass.NOOP,
        ]
        assert [result.correct for result in block_results] == [
            True,
            False,
            False,
            task.label is DecisionClass.REFUSE,
            task.label is DecisionClass.NOOP,
        ]
        results.extend(block_results)
    assert len(results) == 200
    assert sum(result.correct for result in results) == 56
