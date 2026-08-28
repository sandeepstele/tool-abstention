"""Strict internal-only preference contracts and deterministic generation."""

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tool_abstention.config import load_yaml_config
from tool_abstention.evaluator import evaluate_prediction, parse_tool_call_text
from tool_abstention.records import (
    CallExpected,
    ContractModel,
    PredictionRecord,
    Slug,
    TaskRecord,
)
from tool_abstention.sft import assistant_text
from tool_abstention.taxonomy import DatasetSplit, DecisionClass, TaskVariant
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

PREFERENCE_GENERATOR_VERSION = "1.0.0"


class NegativeType(StrEnum):
    """Controlled reason that the rejected response is inferior."""

    WRONG_DECISION_ABSTAIN = "wrong_decision_abstain"
    WRONG_ABSTENTION_CLASS = "wrong_abstention_class"
    UNNECESSARY_CALL = "unnecessary_call"
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENTS = "wrong_arguments"
    MALFORMED_TOOL_CALL = "malformed_tool_call"
    SCHEMA_COPYING = "schema_copying"


class PreferenceProvenance(ContractModel):
    """Proof that one preference is internal and benchmark-free."""

    source_kind: Literal["internal_task"] = "internal_task"
    source_task_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    sft_init_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_sources: tuple[str, ...] = ()
    test_consumed: Literal[False] = False

    @field_validator("external_sources")
    @classmethod
    def forbid_external_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("preference training cannot use external sources")
        return value


class PreferenceRecord(ContractModel):
    """One chosen/rejected response pair tied to an internal task."""

    id: Slug
    task_id: Slug
    pair_id: Slug
    split: DatasetSplit
    target_class: DecisionClass
    negative_type: NegativeType
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    provenance: PreferenceProvenance

    @model_validator(mode="after")
    def validate_contract(self) -> "PreferenceRecord":
        if self.id != f"pref-{self.task_id}":
            raise ValueError("preference id must be 'pref-<task_id>'")
        if self.split is DatasetSplit.TEST:
            raise ValueError("test records are prohibited from preference data")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected responses must differ")
        return self


class PreferenceConfig(ContractModel):
    """Frozen generator identity and selected SFT initialization."""

    generator_version: Literal["1.0.0"]
    sft_init_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def _tool_call(name: str, arguments: object) -> str:
    payload = canonical_json_bytes({"name": name, "arguments": arguments}).decode(
        "utf-8"
    )
    return f"<tool_call>{payload}</tool_call>"


def _malform(call: str) -> str:
    marker = "}</tool_call>"
    if not call.endswith(marker):
        raise ValueError("cannot malform a non-canonical tool call")
    return f"{call[: -len(marker)]}</tool_call>"


def _wrong_abstention(task: TaskRecord) -> str:
    if task.label is DecisionClass.REFUSE:
        return "No action is needed; the request is already complete."
    return "I cannot complete this request because no visible tool supports it."


def _act_negative(task: TaskRecord, index: int) -> tuple[NegativeType, str]:
    if not isinstance(task.expected, CallExpected):
        raise ValueError("act preference requires CALL expected behavior")
    chosen = assistant_text(task)
    negative = (
        NegativeType.WRONG_DECISION_ABSTAIN,
        NegativeType.WRONG_TOOL,
        NegativeType.WRONG_ARGUMENTS,
        NegativeType.MALFORMED_TOOL_CALL,
        NegativeType.SCHEMA_COPYING,
    )[index % 5]
    if negative is NegativeType.WRONG_DECISION_ABSTAIN:
        return negative, "I cannot complete this request with the visible tools."
    if negative is NegativeType.WRONG_TOOL:
        other = next(
            (tool.name for tool in task.tools if tool.name != task.expected.tool_name),
            "invented_tool",
        )
        return negative, _tool_call(other, task.expected.arguments)
    if negative is NegativeType.WRONG_ARGUMENTS:
        return negative, _tool_call(task.expected.tool_name, {})
    if negative is NegativeType.MALFORMED_TOOL_CALL:
        return negative, _malform(chosen)
    tool = next(tool for tool in task.tools if tool.name == task.expected.tool_name)
    return negative, _tool_call(task.expected.tool_name, tool.parameters)


def _abstain_negative(
    task: TaskRecord, paired_act: TaskRecord, index: int
) -> tuple[NegativeType, str]:
    act_call = assistant_text(paired_act)
    negative = (
        NegativeType.UNNECESSARY_CALL,
        NegativeType.WRONG_ABSTENTION_CLASS,
        NegativeType.MALFORMED_TOOL_CALL,
    )[index % 3]
    if negative is NegativeType.UNNECESSARY_CALL:
        return negative, act_call
    if negative is NegativeType.WRONG_ABSTENTION_CLASS:
        return negative, _wrong_abstention(task)
    return negative, _malform(act_call)


def validate_preference(record: PreferenceRecord, task: TaskRecord) -> None:
    """Validate chosen/rejected semantics against the deterministic evaluator."""
    if record.task_id != task.id or record.pair_id != task.pair_id:
        raise ValueError("preference source ids do not match the task")
    if record.split is not task.split or record.target_class is not task.label:
        raise ValueError("preference label or split does not match the task")
    if record.provenance.source_task_hash != sha256_object(
        task.model_dump(mode="json")
    ):
        raise ValueError("preference source task hash does not match")
    chosen = evaluate_prediction(
        task,
        PredictionRecord(task_id=task.id, raw_text=record.chosen, latency_ms=0),
    )
    rejected = evaluate_prediction(
        task,
        PredictionRecord(task_id=task.id, raw_text=record.rejected, latency_ms=0),
    )
    if not chosen.correct or not chosen.protocol_correct:
        raise ValueError("chosen response must be semantically and protocol correct")
    if rejected.correct:
        raise ValueError("rejected response must fail deterministic evaluation")
    if (
        record.negative_type is NegativeType.MALFORMED_TOOL_CALL
        and rejected.protocol_correct
    ):
        raise ValueError("malformed_tool_call rejection must fail protocol validity")
    if (
        record.negative_type is not NegativeType.MALFORMED_TOOL_CALL
        and not rejected.protocol_correct
    ):
        raise ValueError("non-malformed rejection must remain protocol valid")
    if (
        record.negative_type
        in {
            NegativeType.WRONG_TOOL,
            NegativeType.WRONG_ARGUMENTS,
            NegativeType.SCHEMA_COPYING,
        }
        and parse_tool_call_text(record.rejected) is None
    ):
        raise ValueError("tool-call rejection must contain valid call syntax")


def generate_preferences(
    tasks: list[TaskRecord], *, sft_init_hash: str
) -> list[PreferenceRecord]:
    """Generate one controlled preference for every internal train/validation task."""
    ordered = sorted(tasks, key=lambda task: task.id)
    by_pair: dict[str, dict[TaskVariant, TaskRecord]] = {}
    for task in ordered:
        if task.split is DatasetSplit.TEST:
            raise ValueError("test records are prohibited from preference generation")
        by_pair.setdefault(task.pair_id, {})[task.variant] = task
    records: list[PreferenceRecord] = []
    for index, task in enumerate(ordered):
        members = by_pair[task.pair_id]
        if set(members) != {TaskVariant.ACT, TaskVariant.ABSTAIN}:
            raise ValueError(f"incomplete preference source pair: {task.pair_id}")
        if task.variant is TaskVariant.ACT:
            negative_type, rejected = _act_negative(task, index)
        else:
            negative_type, rejected = _abstain_negative(
                task, members[TaskVariant.ACT], index
            )
        record = PreferenceRecord(
            id=f"pref-{task.id}",
            task_id=task.id,
            pair_id=task.pair_id,
            split=task.split,
            target_class=task.label,
            negative_type=negative_type,
            chosen=assistant_text(task),
            rejected=rejected,
            provenance=PreferenceProvenance(
                source_task_hash=sha256_object(task.model_dump(mode="json")),
                generator_version=PREFERENCE_GENERATOR_VERSION,
                sft_init_hash=sft_init_hash,
            ),
        )
        validate_preference(record, task)
        records.append(record)
    return records


def build_preference_dataset(
    config_path: Path, internal_root: Path, output: Path
) -> dict[str, object]:
    """Build preferences from train/validation paths; test is unreachable by design."""
    config = load_yaml_config(config_path, PreferenceConfig)
    tasks: list[TaskRecord] = []
    for filename, split in (
        ("train.jsonl", DatasetSplit.TRAIN),
        ("validation.jsonl", DatasetSplit.VALIDATION),
    ):
        records = [
            TaskRecord.model_validate(value)
            for value in read_jsonl(internal_root / filename)
        ]
        if any(record.split is not split for record in records):
            raise ValueError(f"{filename} contains records from the wrong split")
        tasks.extend(records)
    preferences = generate_preferences(tasks, sft_init_hash=config.sft_init_hash)
    train = [record for record in preferences if record.split is DatasetSplit.TRAIN]
    valid = [
        record for record in preferences if record.split is DatasetSplit.VALIDATION
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "train.jsonl", [record.model_dump(mode="json") for record in train]
    )
    write_jsonl(
        output / "valid.jsonl", [record.model_dump(mode="json") for record in valid]
    )
    distribution = Counter(record.negative_type.value for record in preferences)
    class_distribution = Counter(record.target_class.value for record in preferences)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator_version": PREFERENCE_GENERATOR_VERSION,
        "config_hash": sha256_file(config_path),
        "sft_init_hash": config.sft_init_hash,
        "source_kind": "internal_task",
        "source_splits": ["train", "validation"],
        "external_sources": [],
        "test_consumed": False,
        "train_count": len(train),
        "validation_count": len(valid),
        "negative_distribution": dict(sorted(distribution.items())),
        "class_distribution": dict(sorted(class_distribution.items())),
        "source_train_hash": sha256_file(internal_root / "train.jsonl"),
        "source_validation_hash": sha256_file(internal_root / "validation.jsonl"),
        "train_hash": sha256_file(output / "train.jsonl"),
        "valid_hash": sha256_file(output / "valid.jsonl"),
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_preference(path: Path) -> PreferenceRecord:
    """Load one preference JSON object for CLI validation."""
    return PreferenceRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
