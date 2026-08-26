"""Tests for the command-line entry point."""

from pathlib import Path

import pytest

from tool_abstention.cli import main
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.taxonomy import DatasetSplit
from tool_abstention.util.jsonl import write_jsonl

from .test_evaluator import correct_prediction
from .test_records import act_task


def test_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "Train and evaluate tool-use abstention" in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert "validate-config" in capsys.readouterr().out


def test_validate_config_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "project.yaml"
    config_path.write_text("name: demo\nseed: 7\n", encoding="utf-8")
    main(["validate-config", str(config_path)])
    assert '"name": "demo"' in capsys.readouterr().out


def test_export_schemas(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "schemas"
    main(["export-schemas", str(output)])
    assert (output / "task.schema.json").is_file()
    assert "evaluation:" in capsys.readouterr().out


def test_validate_record_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "task.json"
    path.write_text(act_task().model_dump_json(), encoding="utf-8")
    main(["validate-record", "task", str(path)])
    assert '"label": "CALL"' in capsys.readouterr().out


def test_validate_record_reports_useful_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["validate-record", "task", str(path)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "pair_id" in captured.err


def test_generate_and_audit_productivity_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "productivity.yaml"
    config_path.write_text(
        "seed: 0\npairs_per_class: 1\ngenerator_version: 1.0.0\nsplit: train\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    main(
        [
            "generate-productivity",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ]
    )
    assert "generated 4 pairs / 8 tasks" in capsys.readouterr().out
    main(["audit-pairs", str(output / "tasks.jsonl")])
    audit = capsys.readouterr().out
    assert audit.count("PAIR productivity-") == 4


def test_evaluate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=1,
            generator_version="1.0.0",
            split=DatasetSplit.TRAIN,
        )
    )
    tasks = [task for pair in pairs for task in (pair.act, pair.abstain)]
    task_path = tmp_path / "tasks.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output = tmp_path / "results"
    write_jsonl(task_path, [task.model_dump(mode="json") for task in tasks])
    write_jsonl(
        prediction_path,
        [correct_prediction(task).model_dump(mode="json") for task in tasks],
    )
    main(
        [
            "evaluate",
            "--tasks",
            str(task_path),
            "--predictions",
            str(prediction_path),
            "--output",
            str(output),
        ]
    )
    assert '"paired_accuracy": 1.0' in capsys.readouterr().out
