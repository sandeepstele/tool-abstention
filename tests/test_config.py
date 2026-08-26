"""Tests for strict YAML configuration loading."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tool_abstention.config import ProjectConfig, load_yaml_config


def write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_load_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "valid.yaml"
    write_yaml(path, {"name": "demo", "seed": 42})
    config = load_yaml_config(path, ProjectConfig)
    assert config == ProjectConfig(name="demo", seed=42, schema_version=1)


@pytest.mark.parametrize(
    "value",
    [
        {"seed": 0},
        {"name": "demo", "seed": "not-an-integer"},
        {"name": "demo", "seed": 0, "unknown": True},
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, value: object) -> None:
    path = tmp_path / "invalid.yaml"
    write_yaml(path, value)
    with pytest.raises(ValidationError):
        load_yaml_config(path, ProjectConfig)


@pytest.mark.parametrize("text", ["", "- item\n- item\n"])
def test_config_root_must_be_mapping(tmp_path: Path, text: str) -> None:
    path = tmp_path / "invalid-root.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml_config(path, ProjectConfig)
