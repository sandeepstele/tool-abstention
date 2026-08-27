"""Tests for metrics and stored-prediction evaluation."""

import json
from pathlib import Path

import pytest

from tool_abstention.harness import evaluate_files
from tool_abstention.metrics import compute_metrics
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.records import EvaluationRecord, TaskRecord
from tool_abstention.taxonomy import DatasetSplit, DecisionClass
from tool_abstention.util.jsonl import write_jsonl

from .test_evaluator import correct_prediction


def tasks_and_evaluations() -> tuple[list[TaskRecord], list[EvaluationRecord]]:
    pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=1,
            generator_version="1.0.0",
            split=DatasetSplit.TRAIN,
        )
    )
    tasks = [task for pair in pairs for task in (pair.act, pair.abstain)]
    evaluations = [
        EvaluationRecord(
            task_id=task.id,
            predicted_class=task.label,
            correct=True,
            reason_code="correct",
        )
        for task in tasks
    ]
    return tasks, evaluations


def test_perfect_metrics() -> None:
    tasks, evaluations = tasks_and_evaluations()
    summary = compute_metrics(tasks, evaluations)
    assert summary.task_count == 8
    assert summary.pair_count == 4
    assert summary.accuracy == 1
    assert summary.paired_accuracy == 1
    assert summary.macro_f1 == 1
    assert summary.act_accuracy == 1
    assert summary.abstention_accuracy == 1
    assert summary.tool_hallucination_rate == 0


def test_known_imperfect_metrics() -> None:
    tasks, evaluations = tasks_and_evaluations()
    abstain_index = 1
    evaluations[abstain_index] = EvaluationRecord(
        task_id=evaluations[abstain_index].task_id,
        predicted_class=DecisionClass.CALL,
        correct=False,
        reason_code="incorrect_tool_call",
    )
    summary = compute_metrics(tasks, evaluations)
    assert summary.accuracy == 7 / 8
    assert summary.paired_accuracy == 3 / 4
    assert summary.act_accuracy == 1
    assert summary.abstention_accuracy == 3 / 4
    assert summary.tool_hallucination_rate == 1 / 4
    assert summary.per_class[DecisionClass.ANSWER].recall == 0


def test_metric_input_validation() -> None:
    tasks, evaluations = tasks_and_evaluations()
    with pytest.raises(ValueError, match="empty task"):
        compute_metrics([], [])
    with pytest.raises(ValueError, match="ids must match"):
        compute_metrics(tasks, evaluations[:-1])
    with pytest.raises(ValueError, match="ids must be unique"):
        compute_metrics(tasks, [*evaluations, evaluations[0]])
    with pytest.raises(ValueError, match="both act and abstain"):
        compute_metrics(tasks[::2], evaluations[::2])
    with pytest.raises(ValueError, match="exactly two"):
        compute_metrics(tasks[:-1], evaluations[:-1])


def test_evaluate_files_writes_replayable_outputs(tmp_path: Path) -> None:
    raw_tasks, _ = tasks_and_evaluations()
    tasks = list(raw_tasks)
    task_path = tmp_path / "tasks.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output = tmp_path / "results"
    write_jsonl(task_path, [task.model_dump(mode="json") for task in tasks])
    write_jsonl(
        prediction_path,
        [correct_prediction(task).model_dump(mode="json") for task in tasks],
    )
    metrics = evaluate_files(task_path, prediction_path, output)
    assert metrics.accuracy == 1
    assert len((output / "evaluations.jsonl").read_text().splitlines()) == 8
    stored = json.loads((output / "metrics.json").read_text())
    assert stored["paired_accuracy"] == 1


def test_evaluate_files_rejects_prediction_id_problems(tmp_path: Path) -> None:
    raw_tasks, _ = tasks_and_evaluations()
    tasks = list(raw_tasks)
    task_path = tmp_path / "tasks.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    write_jsonl(task_path, [task.model_dump(mode="json") for task in tasks])
    first = correct_prediction(tasks[0])
    write_jsonl(
        prediction_path,
        [first.model_dump(mode="json"), first.model_dump(mode="json")],
    )
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_files(task_path, prediction_path, tmp_path / "output")
    foreign = first.model_copy(update={"task_id": "foreign-act"})
    write_jsonl(prediction_path, [foreign.model_dump(mode="json")])
    with pytest.raises(ValueError, match="must belong to the task file"):
        evaluate_files(task_path, prediction_path, tmp_path / "output")
