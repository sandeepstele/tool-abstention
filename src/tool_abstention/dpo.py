"""DPO preparation, numerical contracts, and reference-cache validation."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import Field, field_validator, model_validator

from tool_abstention.inference import task_messages
from tool_abstention.mlx_sft_runner import token_ids
from tool_abstention.preference import NegativeType, PreferenceRecord
from tool_abstention.records import ContractModel, Slug, TaskRecord
from tool_abstention.taxonomy import DatasetSplit, DecisionClass
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

DPO_FORMATTER_VERSION = "1.0.0"


class DpoExample(ContractModel):
    """One common prompt with chosen and rejected assistant completions."""

    id: Slug
    task_id: Slug
    split: DatasetSplit
    messages: tuple[dict[str, str], ...] = Field(min_length=1)
    tools: tuple[dict[str, Any], ...] = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    negative_type: NegativeType
    preference_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sft_init_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_example(self) -> "DpoExample":
        if self.id != f"dpo-{self.task_id}":
            raise ValueError("DPO id must be 'dpo-<task_id>'")
        if self.split is DatasetSplit.TEST:
            raise ValueError("test examples are prohibited from DPO")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("DPO chosen and rejected completions must differ")
        return self


class DpoPrepareConfig(ContractModel):
    """Strict deterministic selection for prepared DPO data."""

    formatter_version: Literal["1.0.0"]
    sft_init_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_limit: int | None = Field(default=None, gt=0)
    validation_limit: int | None = Field(default=None, gt=0)


class DpoTrainingConfig(ContractModel):
    """Pinned custom MLX DPO training configuration."""

    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sft_adapter_path: str = Field(min_length=1)
    sft_init_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    beta: float = Field(gt=0)
    label_smoothing: float = Field(ge=0, lt=0.5)
    logp_normalization: Literal["sum", "mean"] = "sum"
    chosen_sft_weight: float = Field(default=0.0, ge=0)
    batch_size: Literal[1]
    grad_accumulation_steps: int = Field(gt=0)
    iters: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    rank: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    scale: float = Field(gt=0)
    steps_per_report: int = Field(gt=0)
    steps_per_eval: int = Field(gt=0)
    save_every: int = Field(gt=0)
    peak_memory_limit_gb: float = Field(gt=0)
    required_mlx_version: Literal["0.32.2"]
    required_mlx_lm_version: Literal["0.29.1"]

    @model_validator(mode="after")
    def validate_schedule(self) -> "DpoTrainingConfig":
        if self.iters % self.grad_accumulation_steps:
            raise ValueError("iterations must align with gradient accumulation steps")
        return self


class ReferenceLogps(ContractModel):
    """Frozen reference log probabilities for one prepared example."""

    id: Slug
    example_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chosen_logp: float
    rejected_logp: float
    chosen_tokens: int = Field(gt=0)
    rejected_tokens: int = Field(gt=0)

    @field_validator("chosen_logp", "rejected_logp")
    @classmethod
    def finite_logp(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reference log probabilities must be finite")
        return value


class ReferenceCacheManifest(ContractModel):
    """Identity of a complete frozen-reference cache."""

    schema_version: Literal[1] = 1
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    examples_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_seq_length: int = Field(gt=0)
    record_count: int = Field(gt=0)
    records_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TokenizerBoundary(Protocol):
    """Minimal tokenizer behavior used by deterministic pair tokenization."""

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TokenizedDpoPair:
    """Token IDs and completion offsets for one DPO pair."""

    id: str
    chosen: tuple[int, ...]
    rejected: tuple[int, ...]
    chosen_offset: int
    rejected_offset: int
    chosen_truncated: bool
    rejected_truncated: bool


def _openai_tools(task: TaskRecord) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in task.tools
    )


def prepare_dpo_examples(
    config: DpoPrepareConfig,
    tasks: list[TaskRecord],
    preferences: list[PreferenceRecord],
) -> list[DpoExample]:
    """Join validated preferences to their internal prompt records."""
    task_by_id = {task.id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("DPO source task ids must be unique")
    if any(task.split is DatasetSplit.TEST for task in tasks):
        raise ValueError("test tasks are prohibited from DPO preparation")
    seen: set[str] = set()
    examples: list[DpoExample] = []
    for preference in sorted(preferences, key=lambda item: item.id):
        if preference.id in seen:
            raise ValueError("preference ids must be unique")
        seen.add(preference.id)
        task = task_by_id.get(preference.task_id)
        if task is None:
            raise ValueError(f"preference has no internal task: {preference.task_id}")
        task_hash = sha256_object(task.model_dump(mode="json"))
        if preference.provenance.source_task_hash != task_hash:
            raise ValueError("preference task hash is stale")
        if preference.provenance.sft_init_hash != config.sft_init_hash:
            raise ValueError("preference SFT initialization hash does not match")
        examples.append(
            DpoExample(
                id=f"dpo-{task.id}",
                task_id=task.id,
                split=task.split,
                messages=tuple(task_messages(task)),
                tools=_openai_tools(task),
                chosen=preference.chosen,
                rejected=preference.rejected,
                negative_type=preference.negative_type,
                preference_hash=sha256_object(preference.model_dump(mode="json")),
                task_hash=task_hash,
                sft_init_hash=config.sft_init_hash,
            )
        )
    return examples


def _select_balanced(examples: list[DpoExample], limit: int | None) -> list[DpoExample]:
    if limit is None or len(examples) <= limit:
        return examples
    if limit % 2:
        raise ValueError("balanced DPO limits must contain complete pairs")
    by_pair: dict[str, list[DpoExample]] = {}
    for example in examples:
        pair_id, separator, variant = example.task_id.rpartition("-")
        if not separator or variant not in {"act", "abstain"}:
            raise ValueError(f"invalid DPO task variant: {example.task_id}")
        by_pair.setdefault(pair_id, []).append(example)
    if any(
        len(members) != 2
        or {member.task_id.rpartition("-")[2] for member in members}
        != {"act", "abstain"}
        for members in by_pair.values()
    ):
        raise ValueError("balanced DPO selection requires complete unique pairs")

    strata: dict[tuple[str, str], list[str]] = {}
    abstention_labels = tuple(
        label for label in DecisionClass if label is not DecisionClass.CALL
    )
    for pair_id in by_pair:
        matched = next(
            (
                (pair_id.split(f"-{label.value.casefold()}-", 1)[0], label.value)
                for label in abstention_labels
                if f"-{label.value.casefold()}-" in pair_id
            ),
            None,
        )
        if matched is None or not matched[0]:
            raise ValueError(f"cannot derive DPO domain/class stratum: {pair_id}")
        strata.setdefault(matched, []).append(pair_id)
    for pair_ids in strata.values():
        pair_ids.sort()

    selected_pairs: list[str] = []
    pair_limit = limit // 2
    ordered_strata = sorted(strata)
    while len(selected_pairs) < pair_limit:
        progressed = False
        for stratum in ordered_strata:
            if strata[stratum] and len(selected_pairs) < pair_limit:
                selected_pairs.append(strata[stratum].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("not enough complete DPO pairs for requested limit")
    return sorted(
        (member for pair_id in selected_pairs for member in by_pair[pair_id]),
        key=lambda item: item.id,
    )


def build_dpo_dataset(
    config_path: Path,
    internal_root: Path,
    preference_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Prepare train/validation examples without opening test or external data."""
    from tool_abstention.config import load_yaml_config

    config = load_yaml_config(config_path, DpoPrepareConfig)
    tasks: list[TaskRecord] = []
    preferences: list[PreferenceRecord] = []
    for task_name, preference_name, split in (
        ("train.jsonl", "train.jsonl", DatasetSplit.TRAIN),
        ("validation.jsonl", "valid.jsonl", DatasetSplit.VALIDATION),
    ):
        split_tasks = [
            TaskRecord.model_validate(value)
            for value in read_jsonl(internal_root / task_name)
        ]
        split_preferences = [
            PreferenceRecord.model_validate(value)
            for value in read_jsonl(preference_root / preference_name)
        ]
        if any(item.split is not split for item in split_tasks) or any(
            item.split is not split for item in split_preferences
        ):
            raise ValueError("DPO input contains records from the wrong split")
        tasks.extend(split_tasks)
        preferences.extend(split_preferences)
    examples = prepare_dpo_examples(config, tasks, preferences)
    train = _select_balanced(
        [item for item in examples if item.split is DatasetSplit.TRAIN],
        config.train_limit,
    )
    valid = _select_balanced(
        [item for item in examples if item.split is DatasetSplit.VALIDATION],
        config.validation_limit,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "train.jsonl", [item.model_dump(mode="json") for item in train]
    )
    write_jsonl(
        output / "valid.jsonl", [item.model_dump(mode="json") for item in valid]
    )
    manifest = {
        "schema_version": 1,
        "formatter_version": DPO_FORMATTER_VERSION,
        "config_hash": sha256_file(config_path),
        "sft_init_hash": config.sft_init_hash,
        "source_splits": ["train", "validation"],
        "external_sources": [],
        "test_consumed": False,
        "train_count": len(train),
        "validation_count": len(valid),
        "train_hash": sha256_file(output / "train.jsonl"),
        "valid_hash": sha256_file(output / "valid.jsonl"),
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def tokenize_dpo_example(
    example: DpoExample,
    tokenizer: TokenizerBoundary,
    *,
    max_seq_length: int,
) -> TokenizedDpoPair:
    """Tokenize a shared prompt and reject loss-destroying truncation."""
    kwargs = {"tools": list(example.tools), "tokenize": True}
    prompt = token_ids(
        tokenizer.apply_chat_template(
            list(example.messages), add_generation_prompt=True, **kwargs
        )
    )

    def completion(value: str) -> tuple[tuple[int, ...], bool]:
        messages = [*example.messages, {"role": "assistant", "content": value}]
        tokens = token_ids(tokenizer.apply_chat_template(messages, **kwargs))
        if tokens[: len(prompt)] != prompt:
            raise ValueError(f"{example.id}: chosen/rejected prompt tokens differ")
        truncated = tuple(tokens[:max_seq_length])
        if len(truncated) <= len(prompt):
            raise ValueError(f"{example.id}: truncation removed the completion")
        return truncated, len(tokens) > max_seq_length

    chosen, chosen_truncated = completion(example.chosen)
    rejected, rejected_truncated = completion(example.rejected)
    if chosen == rejected:
        raise ValueError(f"{example.id}: truncation erased the preference signal")
    return TokenizedDpoPair(
        id=example.id,
        chosen=chosen,
        rejected=rejected,
        chosen_offset=len(prompt),
        rejected_offset=len(prompt),
        chosen_truncated=chosen_truncated,
        rejected_truncated=rejected_truncated,
    )


def numpy_sequence_logps(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy completion log probabilities for fixed-vector tests."""
    if (
        logits.ndim != 3
        or targets.shape != logits.shape[:2]
        or mask.shape != targets.shape
    ):
        raise ValueError("logits, targets, and mask shapes are incompatible")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    gathered = np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    counts = mask.sum(axis=-1)
    if np.any(counts <= 0):
        raise ValueError("each sequence must contain completion tokens")
    return (gathered * mask).sum(axis=-1), counts


def numpy_dpo_metrics(
    policy_chosen: np.ndarray,
    policy_rejected: np.ndarray,
    reference_chosen: np.ndarray,
    reference_rejected: np.ndarray,
    *,
    beta: float,
    label_smoothing: float,
    chosen_tokens: np.ndarray | None = None,
    rejected_tokens: np.ndarray | None = None,
    normalization: Literal["sum", "mean"] = "sum",
) -> dict[str, float]:
    """Compute standard DPO loss and reward metrics using stable NumPy math."""
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if beta < 0 or not 0 <= label_smoothing < 0.5:
        raise ValueError("invalid DPO beta or label smoothing")
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("DPO log probabilities must be finite")
    if normalization == "mean":
        if chosen_tokens is None or rejected_tokens is None:
            raise ValueError("mean DPO requires chosen and rejected token counts")
        if np.any(chosen_tokens <= 0) or np.any(rejected_tokens <= 0):
            raise ValueError("DPO token counts must be positive")
        policy_chosen = policy_chosen / chosen_tokens
        reference_chosen = reference_chosen / chosen_tokens
        policy_rejected = policy_rejected / rejected_tokens
        reference_rejected = reference_rejected / rejected_tokens
    logits = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    scaled = beta * logits
    positive = np.logaddexp(0.0, -scaled)
    negative = np.logaddexp(0.0, scaled)
    losses = (1 - label_smoothing) * positive + label_smoothing * negative
    chosen_rewards = beta * (policy_chosen - reference_chosen)
    rejected_rewards = beta * (policy_rejected - reference_rejected)
    margins = chosen_rewards - rejected_rewards
    return {
        "loss": float(losses.mean()),
        "chosen_reward": float(chosen_rewards.mean()),
        "rejected_reward": float(rejected_rewards.mean()),
        "reward_accuracy": float((margins > 0).mean()),
        "reward_margin": float(margins.mean()),
    }


def numpy_anchored_loss(
    dpo_loss: float,
    policy_chosen: np.ndarray,
    chosen_tokens: np.ndarray,
    *,
    chosen_sft_weight: float,
) -> dict[str, float]:
    """Combine DPO with mean chosen-completion negative log likelihood."""
    if chosen_sft_weight < 0:
        raise ValueError("chosen SFT weight must be non-negative")
    if not np.all(np.isfinite(policy_chosen)) or not np.all(chosen_tokens > 0):
        raise ValueError("chosen log probabilities and token counts must be valid")
    anchor = float((-policy_chosen / chosen_tokens).mean())
    return {
        "dpo_loss": dpo_loss,
        "chosen_sft_loss": anchor,
        "loss": dpo_loss + chosen_sft_weight * anchor,
    }


def validate_reference_cache(
    examples_path: Path,
    records_path: Path,
    manifest_path: Path,
    *,
    config: DpoTrainingConfig,
) -> tuple[list[DpoExample], dict[str, ReferenceLogps]]:
    """Reject incomplete, duplicate, stale, or mismatched reference caches."""
    examples = [DpoExample.model_validate(value) for value in read_jsonl(examples_path)]
    records = [
        ReferenceLogps.model_validate(value) for value in read_jsonl(records_path)
    ]
    manifest = ReferenceCacheManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("reference cache ids must be unique")
    ids = {example.id for example in examples}
    if set(by_id) != ids:
        raise ValueError("reference cache ids must match DPO examples exactly")
    for example in examples:
        expected = sha256_object(example.model_dump(mode="json"))
        if by_id[example.id].example_hash != expected:
            raise ValueError(f"reference cache example hash is stale: {example.id}")
    if manifest.model != config.model or manifest.revision != config.revision:
        raise ValueError("reference cache model identity does not match")
    if manifest.adapter_hash != config.sft_init_hash:
        raise ValueError("reference cache adapter hash does not match")
    if manifest.max_seq_length != config.max_seq_length:
        raise ValueError("reference cache sequence length does not match")
    if manifest.examples_hash != sha256_file(examples_path):
        raise ValueError("reference cache examples hash is stale")
    if manifest.records_hash != sha256_file(records_path):
        raise ValueError("reference cache records hash is stale")
    if manifest.record_count != len(records):
        raise ValueError("reference cache record count does not match")
    return examples, by_id
