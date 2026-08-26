"""Tests for the command-line entry point."""

from pathlib import Path

import pytest

from tool_abstention.cli import main

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
