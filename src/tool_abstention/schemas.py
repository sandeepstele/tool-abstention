"""Schema export and JSON record validation."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from tool_abstention.records import (
    EvaluationRecord,
    PredictionRecord,
    TaskPair,
    TaskRecord,
)
from tool_abstention.util.hashing import canonical_json_bytes

type SchemaKind = Literal["task", "pair", "prediction", "evaluation"]

SCHEMA_MODELS: dict[SchemaKind, type[BaseModel]] = {
    "task": TaskRecord,
    "pair": TaskPair,
    "prediction": PredictionRecord,
    "evaluation": EvaluationRecord,
}


def export_schemas(output_directory: Path) -> dict[str, Path]:
    """Export all public artifact schemas using canonical JSON bytes."""
    output_directory.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for name, model in SCHEMA_MODELS.items():
        path = output_directory / f"{name}.schema.json"
        path.write_bytes(canonical_json_bytes(model.model_json_schema()) + b"\n")
        exported[name] = path
    return exported


def validate_record(path: Path, kind: SchemaKind) -> BaseModel:
    """Read one JSON document and validate it as the requested record kind."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return SCHEMA_MODELS[kind].model_validate(value)
