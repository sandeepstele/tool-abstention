"""Tests for blinded human-calibration packet handling."""

import csv
from pathlib import Path

import pytest

from tool_abstention.calibration import (
    ANNOTATION_FIELDS,
    agreement_summary,
    annotation_summary,
    evaluator_agreement_summary,
    export_calibration_packet,
    load_annotations,
)
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.records import EvaluationRecord, PredictionRecord, TaskRecord
from tool_abstention.taxonomy import DatasetSplit, DecisionClass


def calibration_records() -> tuple[list[TaskRecord], list[PredictionRecord]]:
    pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=1,
            generator_version="1.0.0",
            split=DatasetSplit.VALIDATION,
        )
    )
    tasks = [task for pair in pairs for task in (pair.act, pair.abstain)]
    predictions = [
        PredictionRecord(task_id=task.id, raw_text="review me", latency_ms=1)
        for task in tasks
    ]
    return tasks, predictions


def test_export_is_balanced_blinded_and_deterministic(tmp_path: Path) -> None:
    tasks, predictions = calibration_records()
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = export_calibration_packet(tasks, predictions, first, per_cell=1)
    export_calibration_packet(tasks, predictions, second, per_cell=1)
    assert manifest["item_count"] == 5
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    page = (first / "annotate.html").read_text(encoding="utf-8")
    assert "Download completed CSV" in page
    assert "expected" not in page.casefold()
    assert "task_id" not in page


def test_export_rejects_nonvalidation_missing_predictions_and_bad_size(
    tmp_path: Path,
) -> None:
    tasks, predictions = calibration_records()
    train_task = tasks[0].model_copy(update={"split": DatasetSplit.TRAIN})
    with pytest.raises(ValueError, match="validation tasks only"):
        export_calibration_packet([train_task], predictions, tmp_path / "train")
    with pytest.raises(ValueError, match="missing selected"):
        export_calibration_packet(tasks, [], tmp_path / "missing", per_cell=1)
    with pytest.raises(ValueError, match="positive"):
        export_calibration_packet(tasks, predictions, tmp_path / "size", per_cell=0)


def test_completed_annotations_validate_and_summarize(tmp_path: Path) -> None:
    path = tmp_path / "completed.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "audit_id": "audit-001",
                "predicted_behavior": "ANSWER",
                "semantic_correctness": "YES",
                "format_acceptable": "NO",
                "notes": "meaning is right",
            }
        )
    rows = load_annotations(path, {"audit-001"})
    summary = annotation_summary(rows)
    assert summary["item_count"] == 1
    assert summary["semantic_correctness"] == {"YES": 1}
    agreement = agreement_summary(rows, rows)
    assert agreement["fields"]["predicted_behavior"]["exact_agreement"] == 1
    assert agreement["fields"]["predicted_behavior"]["cohen_kappa"] == 1
    evaluation = EvaluationRecord(
        task_id="productivity-001-act",
        predicted_class=DecisionClass.ANSWER,
        behavior_correct=True,
        semantic_correct=True,
        protocol_correct=True,
        correct=True,
        reason_code="correct_answer",
    )
    evaluator_agreement = evaluator_agreement_summary(
        rows, {"audit-001": evaluation.task_id}, [evaluation]
    )
    assert evaluator_agreement["behavior_agreement"] == 1
    assert evaluator_agreement["semantic_agreement"] == 1
    assert evaluator_agreement["protocol_agreement"] == 0
    assert evaluator_agreement["disagreements"][0]["axes"] == "protocol"


def test_annotations_reject_blank_duplicate_foreign_and_bad_columns(
    tmp_path: Path,
) -> None:
    blank = tmp_path / "blank.csv"
    blank.write_text(
        ",".join(ANNOTATION_FIELDS) + "\naudit-001,,,,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_annotations(blank, {"audit-001"})

    duplicate = tmp_path / "duplicate.csv"
    row = "audit-001,ANSWER,YES,YES,\n"
    duplicate.write_text(
        ",".join(ANNOTATION_FIELDS) + "\n" + row + row, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unique"):
        load_annotations(duplicate, {"audit-001"})

    valid = tmp_path / "valid.csv"
    valid.write_text(",".join(ANNOTATION_FIELDS) + "\n" + row, encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_annotations(valid, {"audit-001", "audit-002"})

    bad = tmp_path / "bad.csv"
    bad.write_text("audit_id,answer\naudit-001,YES\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_annotations(bad, {"audit-001"})

    other = valid.read_text(encoding="utf-8").replace("audit-001", "audit-002")
    other_path = tmp_path / "other.csv"
    other_path.write_text(other, encoding="utf-8")
    with pytest.raises(ValueError, match="same audit ids"):
        agreement_summary(
            load_annotations(valid, {"audit-001"}),
            load_annotations(other_path, {"audit-002"}),
        )
    with pytest.raises(ValueError, match="annotations and mapping"):
        evaluator_agreement_summary(load_annotations(valid, {"audit-001"}), {}, [])
    with pytest.raises(ValueError, match="missing mapped"):
        evaluator_agreement_summary(
            load_annotations(valid, {"audit-001"}),
            {"audit-001": "productivity-001-act"},
            [],
        )
