"""Internal-only protocol stress generation and repair-SFT assembly."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tool_abstention.config import load_yaml_config
from tool_abstention.records import (
    CallExpected,
    ClarifyExpected,
    TaskPair,
    TaskRecord,
    ToolDefinition,
)
from tool_abstention.sft import format_sft_example
from tool_abstention.taxonomy import (
    DatasetSplit,
    DecisionClass,
    PerturbationType,
    TaskVariant,
)
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

PROTOCOL_GENERATOR_VERSION = "1.0.0"


class ProtocolStressConfig(BaseModel):
    """Strict sizes for a deterministic internal protocol stress slice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0, le=2**32 - 1)
    train_pairs: int = Field(gt=0)
    validation_pairs: int = Field(gt=0)
    generator_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")


def _case(index: int) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Return a tool name, request verb, schema, and typed arguments."""
    variant = index % 4
    arguments: dict[str, Any]
    schema: dict[str, Any]
    if variant == 0:
        arguments = {
            "order": {
                "id": f"ord-{1000 + index}",
                "items": [
                    {"quantity": 2, "sku": f"sku-{index:03d}"},
                    {"quantity": 1, "sku": f"sku-{index + 1:03d}"},
                ],
                "priority": index % 2 == 0,
            },
            "tags": ["fragile", f"batch-{index % 5}"],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "order": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "quantity": {"type": "integer", "minimum": 1},
                                    "sku": {"type": "string"},
                                },
                                "required": ["quantity", "sku"],
                            },
                        },
                        "priority": {"type": "boolean"},
                    },
                    "required": ["id", "items", "priority"],
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["order", "tags"],
        }
        return "submit_order", "Submit this structured order", schema, arguments
    if variant == 1:
        arguments = {
            "filters": {
                "active": True,
                "regions": ["north", "west"],
                "threshold": round(0.25 + index / 100, 2),
            },
            "format": "json",
            "limit": 25 + index,
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "filters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "active": {"type": "boolean"},
                        "regions": {"type": "array", "items": {"type": "string"}},
                        "threshold": {"type": "number"},
                    },
                    "required": ["active", "regions", "threshold"],
                },
                "format": {"type": "string", "enum": ["json", "csv"]},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["filters", "format", "limit"],
        }
        return "run_report", "Run this filtered report", schema, arguments
    if variant == 2:
        arguments = {
            "metadata": {"note": None, "owner": f"owner-{index}"},
            "points": [
                {"lat": 41.8 + index / 100, "lon": -87.7},
                {"lat": 41.9 + index / 100, "lon": -87.6},
            ],
            "round_trip": False,
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "note": {"type": ["string", "null"]},
                        "owner": {"type": "string"},
                    },
                    "required": ["note", "owner"],
                },
                "points": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "lat": {"type": "number"},
                            "lon": {"type": "number"},
                        },
                        "required": ["lat", "lon"],
                    },
                },
                "round_trip": {"type": "boolean"},
            },
            "required": ["metadata", "points", "round_trip"],
        }
        return "plan_route", "Plan this multi-point route", schema, arguments
    arguments = {
        "layers": [
            {"opacity": 0.75, "text": f"Layer {index}"},
            {"opacity": 1.0, "text": "✓ Unicode"},
        ],
        "options": {"background": "#112233", "transparent": False},
        "size": {"height": 720, "width": 1280},
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "layers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                        "text": {"type": "string"},
                    },
                    "required": ["opacity", "text"],
                },
            },
            "options": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "background": {"type": "string"},
                    "transparent": {"type": "boolean"},
                },
                "required": ["background", "transparent"],
            },
            "size": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "height": {"type": "integer"},
                    "width": {"type": "integer"},
                },
                "required": ["height", "width"],
            },
        },
        "required": ["layers", "options", "size"],
    }
    return "render_asset", "Render this layered asset", schema, arguments


def generate_protocol_pairs(config: ProtocolStressConfig) -> list[TaskPair]:
    """Generate complex CALL/CLARIFY pairs without external benchmark content."""
    pairs: list[TaskPair] = []
    total = config.train_pairs + config.validation_pairs
    for offset in range(total):
        index = config.seed * total + offset
        split = (
            DatasetSplit.TRAIN
            if offset < config.train_pairs
            else DatasetSplit.VALIDATION
        )
        pair_id = f"protocol-{offset + 1:03d}"
        tool_name, verb, schema, arguments = _case(index)
        tool = ToolDefinition(
            name=tool_name,
            description=f"Execute internal protocol stress operation {tool_name}.",
            parameters=schema,
        )
        serialized = canonical_json_bytes(arguments).decode("utf-8")
        common: dict[str, Any] = {
            "pair_id": pair_id,
            "domain": "protocol",
            "split": split,
            "generator_version": config.generator_version,
            "tools": (tool,),
            "environment": {"source": "internal_generated", "case": index},
        }
        act = TaskRecord(
            **common,
            id=f"{pair_id}-act",
            variant=TaskVariant.ACT,
            query=f"{verb} using these exact arguments: {serialized}",
            label=DecisionClass.CALL,
            perturbation=None,
            expected=CallExpected(
                tool_name=tool_name,
                arguments=arguments,
                expected_result={"accepted": True},
            ),
        )
        missing = next(iter(arguments))
        abstain_arguments = {
            key: value for key, value in arguments.items() if key != missing
        }
        abstain = TaskRecord(
            **common,
            id=f"{pair_id}-abstain",
            variant=TaskVariant.ABSTAIN,
            query=(
                f"{verb} using these arguments: "
                f"{canonical_json_bytes(abstain_arguments).decode('utf-8')}"
            ),
            label=DecisionClass.CLARIFY,
            perturbation=PerturbationType.REQUIRED_ARGUMENT_REMOVED,
            expected=ClarifyExpected(missing_slots=(missing,)),
        )
        pairs.append(TaskPair(pair_id=pair_id, act=act, abstain=abstain))
    return pairs


def build_protocol_stress(config_path: Path, output: Path) -> dict[str, Any]:
    """Write deterministic train/validation tasks with internal-only provenance."""
    config = load_yaml_config(config_path, ProtocolStressConfig)
    pairs = generate_protocol_pairs(config)
    train = [
        task
        for pair in pairs
        for task in (pair.act, pair.abstain)
        if task.split is DatasetSplit.TRAIN
    ]
    validation = [
        task
        for pair in pairs
        for task in (pair.act, pair.abstain)
        if task.split is DatasetSplit.VALIDATION
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "train.jsonl", [task.model_dump(mode="json") for task in train]
    )
    write_jsonl(
        output / "validation.jsonl",
        [task.model_dump(mode="json") for task in validation],
    )
    manifest = {
        "schema_version": 1,
        "source_kind": "internal_generated",
        "external_sources": [],
        "generator_version": PROTOCOL_GENERATOR_VERSION,
        "config_hash": sha256_file(config_path),
        "train_count": len(train),
        "validation_count": len(validation),
        "train_hash": sha256_file(output / "train.jsonl"),
        "validation_hash": sha256_file(output / "validation.jsonl"),
        "ids_hash": sha256_object([task.id for task in (*train, *validation)]),
        "test_consumed": False,
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def build_protocol_repair_sft(
    base_sft: Path, stress_root: Path, output: Path
) -> dict[str, Any]:
    """Append internal stress examples while refusing external or test provenance."""
    stress_manifest = json.loads(
        (stress_root / "manifest.json").read_text(encoding="utf-8")
    )
    if stress_manifest.get("source_kind") != "internal_generated":
        raise ValueError("protocol repair requires internal_generated stress data")
    if stress_manifest.get("external_sources") != []:
        raise ValueError("external benchmark records cannot enter repair SFT")
    if stress_manifest.get("test_consumed") is not False:
        raise ValueError("repair SFT provenance must prove test_consumed=false")
    base_manifest = json.loads((base_sft / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("test_consumed") is not False:
        raise ValueError("base SFT provenance must prove test_consumed=false")
    stress_train = [
        TaskRecord.model_validate(value)
        for value in read_jsonl(stress_root / "train.jsonl")
    ]
    stress_valid = [
        TaskRecord.model_validate(value)
        for value in read_jsonl(stress_root / "validation.jsonl")
    ]
    base_train = list(read_jsonl(base_sft / "train.jsonl"))
    base_valid = list(read_jsonl(base_sft / "valid.jsonl"))
    base_ids = {str(value["id"]) for value in (*base_train, *base_valid)}
    stress_ids = {task.id for task in (*stress_train, *stress_valid)}
    if base_ids & stress_ids:
        raise ValueError("base and protocol-repair example ids must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    train = [*base_train, *(format_sft_example(task) for task in stress_train)]
    valid = [*base_valid, *(format_sft_example(task) for task in stress_valid)]
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "valid.jsonl", valid)
    manifest = {
        "schema_version": 1,
        "formatter_version": "protocol-repair-1.0.0",
        "source_splits": ["train", "validation"],
        "external_sources": [],
        "test_consumed": False,
        "base_manifest_hash": sha256_file(base_sft / "manifest.json"),
        "stress_manifest_hash": sha256_file(stress_root / "manifest.json"),
        "train_count": len(train),
        "validation_count": len(valid),
        "train_hash": sha256_file(output / "train.jsonl"),
        "valid_hash": sha256_file(output / "valid.jsonl"),
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest
