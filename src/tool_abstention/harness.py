"""Evaluate stored raw predictions independently of inference."""

from pathlib import Path
from typing import Any

from tool_abstention.evaluator import evaluate_prediction
from tool_abstention.metrics import MetricsSummary, compute_metrics
from tool_abstention.records import EvaluationRecord, PredictionRecord, TaskRecord
from tool_abstention.util.hashing import canonical_json_bytes
from tool_abstention.util.jsonl import read_jsonl, write_jsonl


def evaluate_files(
    task_path: Path, prediction_path: Path, output_directory: Path
) -> MetricsSummary:
    """Evaluate stored predictions and write replayable results."""
    tasks = [TaskRecord.model_validate(value) for value in read_jsonl(task_path)]
    predictions = [
        PredictionRecord.model_validate(value) for value in read_jsonl(prediction_path)
    ]
    prediction_by_id = {prediction.task_id: prediction for prediction in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("prediction task ids must be unique")
    if set(prediction_by_id) != {task.id for task in tasks}:
        raise ValueError("prediction ids must match task ids exactly")
    evaluations: list[EvaluationRecord] = [
        evaluate_prediction(task, prediction_by_id[task.id]) for task in tasks
    ]
    metrics = compute_metrics(tasks, evaluations)
    write_jsonl(
        output_directory / "evaluations.jsonl",
        [evaluation.model_dump(mode="json") for evaluation in evaluations],
    )
    metrics_value: dict[str, Any] = metrics.model_dump(mode="json")
    (output_directory / "metrics.json").write_bytes(
        canonical_json_bytes(metrics_value) + b"\n"
    )
    return metrics
