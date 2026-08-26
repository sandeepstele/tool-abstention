"""Tests for the command-line entry point."""

from pathlib import Path

import pytest

from tool_abstention.cli import main


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
