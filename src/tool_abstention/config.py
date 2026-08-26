"""Strict typed configuration loading."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ProjectConfig(BaseModel):
    """Top-level project configuration used by foundation commands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    seed: int = Field(ge=0, le=2**32 - 1)
    schema_version: int = Field(default=1, ge=1)


def load_yaml_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load YAML from *path* and validate it with *model*."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return model.model_validate(value)
