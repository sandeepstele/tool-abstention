"""Deterministic SFT formatting and pinned MLX-LoRA launch contracts."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tool_abstention.config import load_yaml_config
from tool_abstention.inference import SYSTEM_PROMPT, task_messages
from tool_abstention.records import (
    AnswerExpected,
    CallExpected,
    ClarifyExpected,
    DomainAnswerValidator,
    ExactAnswerValidator,
    NoopExpected,
    NormalizedTextAnswerValidator,
    NumericAnswerValidator,
    SetAnswerValidator,
    TaskRecord,
)
from tool_abstention.taxonomy import DatasetSplit
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl


class SftTrainingConfig(BaseModel):
    """Strict configuration for one reproducible MLX LoRA run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    batch_size: int = Field(gt=0)
    grad_accumulation_steps: int = Field(gt=0)
    iters: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    num_layers: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    val_batches: int = Field(default=-1, ge=-1)
    steps_per_report: int = Field(gt=0)
    steps_per_eval: int = Field(gt=0)
    save_every: int = Field(gt=0)
    rank: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    scale: float = Field(gt=0)
    mask_prompt: bool = True
    grad_checkpoint: bool = False


def _assistant_text(task: TaskRecord) -> str:
    expected = task.expected
    if isinstance(expected, CallExpected):
        payload = {"name": expected.tool_name, "arguments": expected.arguments}
        rendered = canonical_json_bytes(payload).decode("utf-8")
        return f"<tool_call>{rendered}</tool_call>"
    if isinstance(expected, AnswerExpected):
        validator = expected.validator
        if isinstance(validator, (ExactAnswerValidator, NormalizedTextAnswerValidator)):
            return validator.value
        if isinstance(validator, NumericAnswerValidator):
            unit = f" {validator.unit}" if validator.unit else ""
            return f"{validator.value:g}{unit}"
        if isinstance(validator, SetAnswerValidator):
            return ", ".join(validator.values)
        if isinstance(validator, DomainAnswerValidator):
            raise ValueError(
                f"{task.id}: domain validator has no deterministic training response"
            )
    if isinstance(expected, ClarifyExpected):
        slots = ", ".join(expected.missing_slots)
        return f"Could you provide the missing {slots}?"
    if isinstance(expected, NoopExpected):
        return expected.allowed_markers[0]
    return (
        "I cannot complete that request because the required capability "
        f"'{expected.unavailable_capability}' is not available in the visible tools."
    )


def format_sft_example(task: TaskRecord) -> dict[str, Any]:
    """Render one canonical task as an MLX chat example with native tools."""
    messages = task_messages(task)
    messages.append({"role": "assistant", "content": _assistant_text(task)})
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in task.tools
    ]
    return {"id": task.id, "messages": messages, "tools": tools}


def _load_split(path: Path, expected_split: DatasetSplit) -> list[TaskRecord]:
    records = [TaskRecord.model_validate(value) for value in read_jsonl(path)]
    wrong = [record.id for record in records if record.split is not expected_split]
    if wrong:
        raise ValueError(f"{path}: records have wrong split: {wrong[:3]}")
    return sorted(records, key=lambda record: record.id)


def build_sft_dataset(internal_root: Path, output: Path) -> dict[str, Any]:
    """Export train/validation only; held-out test is structurally unreachable."""
    train = _load_split(internal_root / "train.jsonl", DatasetSplit.TRAIN)
    validation = _load_split(
        internal_root / "validation.jsonl", DatasetSplit.VALIDATION
    )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", [format_sft_example(task) for task in train])
    write_jsonl(
        output / "valid.jsonl", [format_sft_example(task) for task in validation]
    )
    manifest = {
        "schema_version": 1,
        "formatter_version": "1.0.0",
        "system_prompt_hash": sha256_object(SYSTEM_PROMPT),
        "source_splits": ["train", "validation"],
        "test_consumed": False,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_ids_hash": sha256_object([task.id for task in train]),
        "validation_ids_hash": sha256_object([task.id for task in validation]),
        "train_hash": sha256_file(output / "train.jsonl"),
        "valid_hash": sha256_file(output / "valid.jsonl"),
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def training_command(
    config: SftTrainingConfig,
    *,
    model_path: Path,
    data_path: Path,
    adapter_path: Path,
) -> list[str]:
    """Build the pinned installed MLX-LM command without invoking it."""
    command = [
        sys.executable,
        "-m",
        "tool_abstention.mlx_sft_runner",
        "--train",
        "--model",
        str(model_path),
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--optimizer",
        "adamw",
        "--mask-prompt",
        "--batch-size",
        str(config.batch_size),
        "--grad-accumulation-steps",
        str(config.grad_accumulation_steps),
        "--iters",
        str(config.iters),
        "--learning-rate",
        str(config.learning_rate),
        "--num-layers",
        str(config.num_layers),
        "--max-seq-length",
        str(config.max_seq_length),
        "--val-batches",
        str(config.val_batches),
        "--steps-per-report",
        str(config.steps_per_report),
        "--steps-per-eval",
        str(config.steps_per_eval),
        "--save-every",
        str(config.save_every),
        "--seed",
        str(config.seed),
    ]
    if config.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def run_sft_training(
    config_path: Path, data_path: Path, adapter_path: Path
) -> dict[str, Any]:
    """Resolve the immutable model snapshot and launch installed MLX-LM."""
    from huggingface_hub import snapshot_download

    config = load_yaml_config(config_path, SftTrainingConfig)
    resolved = Path(
        snapshot_download(
            config.model,
            revision=config.revision,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
    )
    adapter_path.mkdir(parents=True, exist_ok=True)
    mlx_config = {
        "lora_parameters": {
            "rank": config.rank,
            "dropout": config.dropout,
            "scale": config.scale,
        }
    }
    runtime_config = adapter_path / "mlx-config.json"
    runtime_config.write_text(json.dumps(mlx_config), encoding="utf-8")
    command = training_command(
        config, model_path=resolved, data_path=data_path, adapter_path=adapter_path
    )
    command.extend(["--config", str(runtime_config)])
    log_path = adapter_path / "training.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            check=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.is_file():
        raise ValueError("MLX training completed without adapters.safetensors")
    manifest = {
        "schema_version": 1,
        "model": config.model,
        "revision": config.revision,
        "config_hash": sha256_file(config_path),
        "data_manifest_hash": sha256_file(data_path / "manifest.json"),
        "adapter_hash": sha256_file(adapter_file),
        "training_log_hash": sha256_file(log_path),
        "command": command,
    }
    (adapter_path / "run_manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )
    return manifest
