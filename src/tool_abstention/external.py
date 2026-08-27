"""Pinned external benchmark ingestion, leakage checks, and decision evaluation."""

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tool_abstention.config import load_yaml_config
from tool_abstention.evaluator import looks_like_tool_call, parse_tool_call_text
from tool_abstention.records import PredictionRecord, TaskRecord
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

REVISION = r"^[0-9a-f]{40}$"
ADAPTER_VERSION = "1.0.0"
BFCL_FILES = {
    "BFCL_v3_simple.json": "CALL",
    "BFCL_v3_irrelevance.json": "ABSTAIN",
}
EXTERNAL_TOOL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


def _file_inventory(root: Path) -> list[dict[str, str]]:
    """Hash a directory tree without including downloader cache metadata."""
    return [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    ]


class SourceUsage(StrEnum):
    EXTERNAL_EVAL = "external_eval"
    TRAINING = "training"


class ExternalDecision(StrEnum):
    CALL = "CALL"
    ABSTAIN = "ABSTAIN"


class SourceProvenance(BaseModel):
    """Immutable record-level source and licensing provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str = Field(min_length=1)
    source_uri: str = Field(pattern=r"^https://")
    revision: str = Field(pattern=REVISION)
    license: Literal["Apache-2.0", "CC-BY-4.0", "MIT"]
    usage: SourceUsage
    benchmark_only: bool = True
    original_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    transformations: tuple[str, ...]
    attribution: str = Field(min_length=1)

    @model_validator(mode="after")
    def prohibit_benchmark_training(self) -> "SourceProvenance":
        if self.benchmark_only and self.usage is SourceUsage.TRAINING:
            raise ValueError("benchmark-only sources cannot be used for training")
        return self


class ExternalSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str = Field(min_length=1)
    revision: str = Field(pattern=REVISION)
    license: Literal["Apache-2.0", "CC-BY-4.0"]
    files: tuple[str, ...] | None = None
    code_revision: str | None = Field(default=None, pattern=REVISION)
    expected_records: dict[str, int] | None = None
    expected_pairs: int | None = Field(default=None, gt=0)
    expected_environments: int | None = Field(default=None, gt=0)


class ExternalDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bfcl: ExternalSourceConfig
    agentabstain: ExternalSourceConfig


class ExternalMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ExternalFunction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def valid_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        Draft202012Validator.check_schema(value)
        return value


class ExternalDecisionRecord(BaseModel):
    """Native external decision example, intentionally separate from TaskRecord."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    category: Literal["simple", "irrelevance"]
    expected_decision: ExternalDecision
    messages: tuple[ExternalMessage, ...] = Field(min_length=1)
    functions: tuple[ExternalFunction, ...] = Field(min_length=1)
    provenance: SourceProvenance


class ExternalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    expected_decision: ExternalDecision
    predicted_decision: ExternalDecision | None
    correct: bool
    protocol_correct: bool
    reason_code: str


class ExternalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_version: Literal["1.1.0"] = "1.1.0"
    task_count: int = Field(gt=0)
    decision_accuracy: float = Field(ge=0, le=1)
    call_accuracy: float = Field(ge=0, le=1)
    abstain_accuracy: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    tool_call_rate: float = Field(ge=0, le=1)
    malformed_call_rate: float = Field(ge=0, le=1)
    per_category: dict[str, float]


TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
}
ALLOWED_TYPES = {
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}


def normalize_bfcl_schema(value: Any) -> Any:
    """Recursively convert BFCL Python-style types to JSON Schema types."""
    if isinstance(value, list):
        return [normalize_bfcl_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key == "type" and isinstance(item, str):
            if item == "any":
                continue
            converted = TYPE_MAP.get(item, item)
            if converted not in ALLOWED_TYPES:
                raise ValueError(f"unsupported BFCL schema type: {item}")
            normalized[key] = converted
        else:
            normalized[key] = normalize_bfcl_schema(item)
    return normalized


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise ValueError("external id cannot normalize to an empty slug")
    return slug


def _messages(value: object) -> tuple[ExternalMessage, ...]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError("BFCL question must contain exactly one conversation")
    return tuple(ExternalMessage.model_validate(item) for item in value[0])


def parse_bfcl_file(
    path: Path,
    *,
    revision: str,
    license_name: Literal["Apache-2.0", "CC-BY-4.0", "MIT"],
) -> list[ExternalDecisionRecord]:
    """Parse one pinned BFCL JSONL category into strict external records."""
    if path.name not in BFCL_FILES:
        raise ValueError(f"unsupported BFCL file: {path.name}")
    decision = ExternalDecision(BFCL_FILES[path.name])
    category: Literal["simple", "irrelevance"] = (
        "simple" if decision is ExternalDecision.CALL else "irrelevance"
    )
    digest = sha256_file(path)
    records: list[ExternalDecisionRecord] = []
    seen: set[str] = set()
    for raw in read_jsonl(path):
        original_id = raw.get("id")
        functions = raw.get("function")
        if not isinstance(original_id, str) or not isinstance(functions, list):
            raise ValueError(f"{path}: BFCL record requires id and function list")
        record_id = f"bfcl-{_slug(original_id)}"
        if record_id in seen:
            raise ValueError(f"duplicate BFCL id: {record_id}")
        seen.add(record_id)
        parsed_functions = []
        for function in functions:
            if not isinstance(function, dict):
                raise ValueError(f"{original_id}: function must be an object")
            normalized = dict(function)
            normalized["parameters"] = normalize_bfcl_schema(
                normalized.get("parameters")
            )
            parsed_functions.append(ExternalFunction.model_validate(normalized))
        provenance = SourceProvenance(
            source_name="BFCL",
            source_uri=(
                "https://huggingface.co/datasets/"
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
            ),
            revision=revision,
            license=license_name,
            usage=SourceUsage.EXTERNAL_EVAL,
            original_id=original_id,
            source_file=path.name,
            source_sha256=digest,
            adapter_version=ADAPTER_VERSION,
            transformations=("python-types-to-json-schema",),
            attribution="Berkeley Function Calling Leaderboard (BFCL)",
        )
        records.append(
            ExternalDecisionRecord(
                id=record_id,
                category=category,
                expected_decision=decision,
                messages=_messages(raw.get("question")),
                functions=tuple(parsed_functions),
                provenance=provenance,
            )
        )
    return sorted(records, key=lambda record: record.id)


def fetch_external(config_path: Path, output: Path) -> dict[str, Any]:
    """Download only the pinned declared snapshots through huggingface-hub."""
    from huggingface_hub import snapshot_download

    config = load_yaml_config(config_path, ExternalDataConfig)
    output.mkdir(parents=True, exist_ok=True)
    bfcl_files = config.bfcl.files or ()
    snapshot_download(
        config.bfcl.dataset,
        repo_type="dataset",
        revision=config.bfcl.revision,
        allow_patterns=list(bfcl_files),
        local_dir=output / "bfcl",
    )
    snapshot_download(
        config.agentabstain.dataset,
        repo_type="dataset",
        revision=config.agentabstain.revision,
        allow_patterns=["tasks/**", "environments/**", "README.md"],
        local_dir=output / "agentabstain",
    )
    missing = [
        filename
        for filename in bfcl_files
        if not (output / "bfcl" / filename).is_file()
    ]
    if missing:
        raise ValueError(f"BFCL download missing declared files: {missing}")
    manifest = {
        "schema_version": 1,
        "bfcl_revision": config.bfcl.revision,
        "agentabstain_revision": config.agentabstain.revision,
        "bfcl_files": _file_inventory(output / "bfcl"),
        "agentabstain_files_hash": sha256_object(
            _file_inventory(output / "agentabstain")
        ),
    }
    (output / "fetch-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def normalize_query(value: str) -> str:
    """Canonical query representation for cross-corpus comparisons."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _five_grams(value: str) -> set[str]:
    padded = f"  {normalize_query(value)}  "
    return {padded[index : index + 5] for index in range(max(1, len(padded) - 4))}


def near_duplicate(left: str, right: str) -> tuple[bool, str | None, float]:
    """Apply the declared exact, five-gram Jaccard, and sequence thresholds."""
    left_normalized = normalize_query(left)
    right_normalized = normalize_query(right)
    if left_normalized == right_normalized:
        return True, "exact", 1.0
    left_grams = _five_grams(left)
    right_grams = _five_grams(right)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 1.0
    if jaccard >= 0.80:
        return True, "five_gram_jaccard", jaccard
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    if sequence >= 0.90:
        return True, "sequence_match", sequence
    return False, None, max(jaccard, sequence)


def external_query(record: ExternalDecisionRecord) -> str:
    return "\n".join(
        message.content for message in record.messages if message.role == "user"
    )


def valid_external_tool_call(raw_text: str) -> bool:
    """Validate call syntax without imposing internal lowercase ID restrictions."""
    text = raw_text.strip()
    match = EXTERNAL_TOOL_BLOCK.fullmatch(text)
    if match:
        text = match.group(1)
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if "function" in payload and isinstance(payload["function"], dict):
        payload = payload["function"]
    name = payload.get("name")
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False
    return isinstance(name, str) and bool(name.strip()) and isinstance(arguments, dict)


def quarantine_leakage(
    records: list[ExternalDecisionRecord], internal_tasks: list[TaskRecord]
) -> tuple[list[ExternalDecisionRecord], list[dict[str, Any]]]:
    """Exclude and report external records overlapping any internal split."""
    kept: list[ExternalDecisionRecord] = []
    report: list[dict[str, Any]] = []
    for record in records:
        query = external_query(record)
        match = None
        for task in internal_tasks:
            duplicate, method, score = near_duplicate(query, task.query)
            if duplicate:
                match = {
                    "external_id": record.id,
                    "internal_id": task.id,
                    "internal_split": task.split.value,
                    "method": method,
                    "score": score,
                }
                break
        if match is None:
            kept.append(record)
        else:
            report.append(match)
    return kept, report


def catalog_agentabstain(
    root: Path, *, revision: str, code_revision: str
) -> dict[str, Any]:
    """Catalog native AgentAbstain pairs and environments without conversion."""
    task_root = root / "tasks"
    environment_root = root / "environments"
    environments = sorted(path for path in environment_root.iterdir() if path.is_dir())
    environment_names = {path.name for path in environments}
    pairs: list[str] = []
    for scenario in sorted(path for path in task_root.iterdir() if path.is_dir()):
        for candidate in sorted(path for path in scenario.iterdir() if path.is_dir()):
            act = candidate / "act"
            abstain = candidate / "abstain"
            if act.is_dir() != abstain.is_dir():
                raise ValueError(f"incomplete AgentAbstain pair: {candidate}")
            if act.is_dir():
                for side in (act, abstain):
                    if not (side / "task.yaml").is_file():
                        raise ValueError(f"missing AgentAbstain task.yaml: {side}")
                    state_root = side / "initial_states"
                    references = (
                        {path.stem for path in state_root.glob("*.json")}
                        if state_root.is_dir()
                        else set()
                    )
                    missing = references - environment_names
                    if missing:
                        raise ValueError(
                            f"unknown AgentAbstain environment in {side}: "
                            f"{sorted(missing)}"
                        )
                pairs.append(str(candidate.relative_to(task_root)))
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate AgentAbstain pair ids")
    return {
        "schema_version": 1,
        "source": "antiquality/agentabstain",
        "revision": revision,
        "code_revision": code_revision,
        "license": "CC-BY-4.0",
        "usage": "external_eval",
        "pair_count": len(pairs),
        "environment_count": len(environments),
        "pairs_hash": sha256_object(pairs),
        "environments_hash": sha256_object([path.name for path in environments]),
        "content_hash": sha256_object(_file_inventory(root)),
        "attribution": "AgentAbstain: Do LLM Agents Know When Not to Act?",
    }


def prepare_external(
    config_path: Path, raw_root: Path, internal_root: Path, output: Path
) -> dict[str, Any]:
    """Prepare BFCL decisions, quarantine leakage, and catalog AgentAbstain."""
    config = load_yaml_config(config_path, ExternalDataConfig)
    records: list[ExternalDecisionRecord] = []
    for filename in config.bfcl.files or ():
        parsed = parse_bfcl_file(
            raw_root / "bfcl" / filename,
            revision=config.bfcl.revision,
            license_name=config.bfcl.license,
        )
        expected = (config.bfcl.expected_records or {}).get(filename)
        if expected is not None and len(parsed) != expected:
            raise ValueError(
                f"{filename}: expected {expected} records, found {len(parsed)}"
            )
        records.extend(parsed)
    internal_tasks = [
        TaskRecord.model_validate(value)
        for split in ("train", "validation", "test")
        for value in read_jsonl(internal_root / f"{split}.jsonl")
    ]
    kept, quarantined = quarantine_leakage(records, internal_tasks)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "bfcl.jsonl", [record.model_dump(mode="json") for record in kept]
    )
    write_jsonl(output / "quarantine.jsonl", quarantined)
    agent_catalog = catalog_agentabstain(
        raw_root / "agentabstain",
        revision=config.agentabstain.revision,
        code_revision=config.agentabstain.code_revision or "",
    )
    expected_pairs = config.agentabstain.expected_pairs
    if expected_pairs is not None and agent_catalog["pair_count"] != expected_pairs:
        raise ValueError(
            f"AgentAbstain: expected {expected_pairs} pairs, found "
            f"{agent_catalog['pair_count']}"
        )
    expected_environments = config.agentabstain.expected_environments
    if (
        expected_environments is not None
        and agent_catalog["environment_count"] != expected_environments
    ):
        raise ValueError(
            f"AgentAbstain: expected {expected_environments} environments, found "
            f"{agent_catalog['environment_count']}"
        )
    (output / "agentabstain-manifest.json").write_bytes(
        canonical_json_bytes(agent_catalog) + b"\n"
    )
    counts = Counter(record.expected_decision.value for record in kept)
    manifest = {
        "schema_version": 1,
        "adapter_version": ADAPTER_VERSION,
        "source_count": len(records),
        "prepared_count": len(kept),
        "quarantined_count": len(quarantined),
        "decision_counts": dict(sorted(counts.items())),
        "bfcl_hash": sha256_file(output / "bfcl.jsonl"),
        "quarantine_hash": sha256_file(output / "quarantine.jsonl"),
        "agentabstain": agent_catalog,
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    card = (
        "# External Benchmark Card\n\n"
        "- Usage: external evaluation only; prohibited from training.\n"
        f"- BFCL prepared: {len(kept)}; quarantined: {len(quarantined)}.\n"
        f"- AgentAbstain pairs: {agent_catalog['pair_count']}; environments: "
        f"{agent_catalog['environment_count']}.\n"
        "- BFCL license: Apache-2.0; AgentAbstain dataset: CC BY 4.0.\n"
    )
    (output / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    return manifest


def evaluate_external_records(
    records: list[ExternalDecisionRecord], predictions: list[PredictionRecord]
) -> tuple[list[ExternalEvaluation], ExternalMetrics]:
    """Evaluate CALL versus ABSTAIN decisions independently of exact arguments."""
    prediction_by_id = {prediction.task_id: prediction for prediction in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("external prediction ids must be unique")
    if set(prediction_by_id) != {record.id for record in records}:
        raise ValueError("external prediction ids must match records")
    evaluations: list[ExternalEvaluation] = []
    for record in records:
        prediction = prediction_by_id[record.id]
        if prediction.inference_error is not None:
            evaluations.append(
                ExternalEvaluation(
                    task_id=record.id,
                    expected_decision=record.expected_decision,
                    predicted_decision=None,
                    correct=False,
                    protocol_correct=False,
                    reason_code="inference_error",
                )
            )
            continue
        parsed = prediction.tool_call or parse_tool_call_text(prediction.raw_text)
        external_protocol = valid_external_tool_call(prediction.raw_text)
        attempted = (
            parsed is not None
            or external_protocol
            or looks_like_tool_call(prediction.raw_text)
        )
        predicted = ExternalDecision.CALL if attempted else ExternalDecision.ABSTAIN
        protocol = (
            (parsed is not None or external_protocol)
            if attempted
            else bool(prediction.raw_text.strip())
        )
        evaluations.append(
            ExternalEvaluation(
                task_id=record.id,
                expected_decision=record.expected_decision,
                predicted_decision=predicted,
                correct=predicted is record.expected_decision,
                protocol_correct=protocol,
                reason_code=(
                    "correct_decision"
                    if predicted is record.expected_decision
                    else "incorrect_decision"
                ),
            )
        )
    call = [
        item for item in evaluations if item.expected_decision is ExternalDecision.CALL
    ]
    abstain = [
        item
        for item in evaluations
        if item.expected_decision is ExternalDecision.ABSTAIN
    ]
    if not call or not abstain:
        raise ValueError("external metrics require CALL and ABSTAIN records")
    call_accuracy = sum(item.correct for item in call) / len(call)
    abstain_accuracy = sum(item.correct for item in abstain) / len(abstain)
    by_record = {record.id: record for record in records}
    categories = sorted({record.category for record in records})
    metrics = ExternalMetrics(
        task_count=len(records),
        decision_accuracy=sum(item.correct for item in evaluations) / len(records),
        call_accuracy=call_accuracy,
        abstain_accuracy=abstain_accuracy,
        balanced_accuracy=(call_accuracy + abstain_accuracy) / 2,
        tool_call_rate=sum(
            item.predicted_decision is ExternalDecision.CALL for item in evaluations
        )
        / len(records),
        malformed_call_rate=sum(
            item.predicted_decision is ExternalDecision.CALL
            and not item.protocol_correct
            for item in evaluations
        )
        / len(records),
        per_category={
            category: sum(
                item.correct
                for item in evaluations
                if by_record[item.task_id].category == category
            )
            / sum(record.category == category for record in records)
            for category in categories
        },
    )
    return evaluations, metrics
