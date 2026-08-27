"""Resumable local inference with a lazily imported MLX backend."""

import json
import os
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from tool_abstention.config import load_yaml_config
from tool_abstention.evaluator import parse_tool_call_text
from tool_abstention.records import PredictionRecord, TaskRecord, ToolDefinition
from tool_abstention.taxonomy import DecisionClass, TaskVariant
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl


class InferenceConfig(BaseModel):
    """Pinned deterministic local-inference configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    max_tokens: int = Field(gt=0, le=2048)
    temperature: float = Field(default=0, ge=0)
    prompt_variant: "PromptVariant" = Field(
        default_factory=lambda: PromptVariant.NATIVE_FULL
    )


class PromptVariant(StrEnum):
    """Controlled prompt strategies used by the baseline diagnostic."""

    NATIVE_FULL = "native-full"
    EMBEDDED_TOOLS = "embedded-tools"
    NATIVE_SHORT = "native-short"


class InferenceBackend(Protocol):
    """Minimal backend boundary used by the resumable runner."""

    def predict(self, task: TaskRecord) -> PredictionRecord: ...


SYSTEM_PROMPT = """You are a tool-use decision agent. Choose exactly one behavior:
- Call one visible tool only when it is necessary and all required arguments exist.
- Answer directly when the answer is already present in the request or context.
- Ask a concise follow-up question when a required argument is missing.
- Explicitly refuse when no visible tool can provide the requested capability.
- State that no action is needed when the environment shows the task is complete.
Never invent a tool, argument, or result."""

SHORT_SYSTEM_PROMPT = """Call one visible tool only when it is necessary and all
required arguments are present. Otherwise answer, clarify, refuse, or state that no
action is needed. Never invent tools or arguments."""

EMBEDDED_TOOL_INSTRUCTION = """Available tools (JSON): {tools}
When a tool is necessary, output exactly
<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>.
Otherwise respond directly with no tool-call tags."""


def _openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def task_messages(
    task: TaskRecord,
    variant: PromptVariant = PromptVariant.NATIVE_FULL,
) -> list[dict[str, str]]:
    """Build the stable system/user messages for one canonical task."""
    context = json.dumps(
        task.environment,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    system_prompt = (
        SHORT_SYSTEM_PROMPT if variant is PromptVariant.NATIVE_SHORT else SYSTEM_PROMPT
    )
    if variant is PromptVariant.EMBEDDED_TOOLS:
        serialized_tools = canonical_json_bytes(
            [_openai_tool(tool) for tool in task.tools]
        ).decode("utf-8")
        system_prompt = (
            f"{system_prompt}\n\n"
            f"{EMBEDDED_TOOL_INSTRUCTION.replace('{tools}', serialized_tools)}"
        )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Environment state: {context}\n\nRequest: {task.query}",
        },
    ]


def prompt_policy(variant: PromptVariant) -> str:
    """Return only the stable policy text, excluding task-specific content."""
    if variant is PromptVariant.NATIVE_SHORT:
        return SHORT_SYSTEM_PROMPT
    if variant is PromptVariant.EMBEDDED_TOOLS:
        return f"{SYSTEM_PROMPT}\n\n{EMBEDDED_TOOL_INSTRUCTION}"
    return SYSTEM_PROMPT


class MlxBackend:
    """Greedy MLX-LM inference loaded only when explicitly instantiated."""

    def __init__(self, config: InferenceConfig) -> None:
        import mlx.core as mx
        from mlx_lm import generate, load

        self.config = config
        self._mx = mx
        self._generate = generate
        loaded = load(
            config.model,
            revision=config.revision,
            tokenizer_config={"trust_remote_code": False},
            return_config=False,
        )
        self._model, self._tokenizer = cast("tuple[Any, Any]", loaded)
        mx.random.seed(config.seed)

    def predict(self, task: TaskRecord) -> PredictionRecord:
        """Generate one greedy prediction and capture cost metadata."""
        template_arguments: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": False,
        }
        if self.config.prompt_variant is not PromptVariant.EMBEDDED_TOOLS:
            template_arguments["tools"] = [_openai_tool(tool) for tool in task.tools]
        prompt = self._tokenizer.apply_chat_template(
            task_messages(task, self.config.prompt_variant), **template_arguments
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template must return text when tokenize=False")
        input_tokens = len(self._tokenizer.encode(prompt, add_special_tokens=False))
        if hasattr(self._mx, "reset_peak_memory"):
            self._mx.reset_peak_memory()
        started = time.perf_counter()
        response = self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        output_tokens = len(self._tokenizer.encode(response, add_special_tokens=False))
        parsed = parse_tool_call_text(response)
        return PredictionRecord(
            task_id=task.id,
            raw_text=response,
            tool_call=parsed,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            peak_memory_gb=float(self._mx.get_peak_memory()) / 1e9,
        )


def _append_prediction(path: Path, prediction: PredictionRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(prediction.model_dump(mode="json")) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_inference(
    tasks: Sequence[TaskRecord],
    backend: InferenceBackend,
    output_path: Path,
    *,
    limit: int | None = None,
) -> list[PredictionRecord]:
    """Run or resume inference, durably appending one prediction at a time."""
    existing = (
        [PredictionRecord.model_validate(value) for value in read_jsonl(output_path)]
        if output_path.exists()
        else []
    )
    by_id = {prediction.task_id: prediction for prediction in existing}
    if len(by_id) != len(existing):
        raise ValueError("existing predictions contain duplicate task ids")
    selected = list(tasks[:limit] if limit is not None else tasks)
    selected_ids = {task.id for task in selected}
    if not set(by_id).issubset(selected_ids):
        raise ValueError("existing predictions do not belong to the selected tasks")
    for task in selected:
        if task.id in by_id:
            continue
        try:
            prediction = backend.predict(task)
        except Exception as error:  # inference failures must be persisted and auditable
            prediction = PredictionRecord(
                task_id=task.id,
                raw_text="",
                latency_ms=0,
                inference_error=f"{type(error).__name__}: {error}",
            )
        _append_prediction(output_path, prediction)
        by_id[task.id] = prediction
    return [by_id[task.id] for task in selected]


def load_tasks(path: Path) -> list[TaskRecord]:
    """Load canonical tasks from one split artifact."""
    return [TaskRecord.model_validate(value) for value in read_jsonl(path)]


def select_stratified_smoke(tasks: Sequence[TaskRecord]) -> list[TaskRecord]:
    """Select one complete pair for each abstention class in stable order."""
    by_pair: dict[str, list[TaskRecord]] = {}
    for task in tasks:
        by_pair.setdefault(task.pair_id, []).append(task)
    selected: list[TaskRecord] = []
    needed = {
        DecisionClass.ANSWER,
        DecisionClass.CLARIFY,
        DecisionClass.REFUSE,
        DecisionClass.NOOP,
    }
    for members in by_pair.values():
        abstain = next(
            (task for task in members if task.variant is TaskVariant.ABSTAIN), None
        )
        if abstain is None or abstain.label not in needed:
            continue
        if len(members) != 2:
            raise ValueError(f"smoke pair is incomplete: {abstain.pair_id}")
        selected.extend(members)
        needed.remove(abstain.label)
        if not needed:
            return selected
    raise ValueError(
        f"could not select smoke pairs for: {sorted(label.value for label in needed)}"
    )


def load_inference_config(path: Path) -> InferenceConfig:
    """Load the strict pinned inference configuration."""
    return load_yaml_config(path, InferenceConfig)


def write_run_manifest(
    config: InferenceConfig,
    tasks: Sequence[TaskRecord],
    prediction_path: Path,
) -> Path:
    """Write deterministic inference provenance beside predictions."""
    manifest = {
        "schema_version": 1,
        "model": config.model,
        "revision": config.revision,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "prompt_variant": config.prompt_variant,
        "task_count": len(tasks),
        "task_ids_hash": sha256_object([task.id for task in tasks]),
        "prompt_policy_hash": sha256_object(prompt_policy(config.prompt_variant)),
        "rendered_prompts_hash": sha256_object(
            [task_messages(task, config.prompt_variant) for task in tasks]
        ),
        "predictions_hash": sha256_file(prediction_path),
    }
    path = prediction_path.with_name("run_manifest.json")
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path
