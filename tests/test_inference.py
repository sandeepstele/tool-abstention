"""Tests for prompt formatting and resumable inference without importing MLX."""

from pathlib import Path

import pytest

from tool_abstention.inference import (
    InferenceConfig,
    PromptExample,
    PromptVariant,
    load_inference_config,
    load_tasks,
    prompt_policy,
    run_inference,
    run_prompt_inference,
    select_stratified_smoke,
    task_messages,
    write_run_manifest,
)
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.records import PredictionRecord, TaskRecord
from tool_abstention.taxonomy import DatasetSplit, DecisionClass
from tool_abstention.util.jsonl import write_jsonl

from .test_records import act_task


class FakeBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def predict(self, task: TaskRecord) -> PredictionRecord:
        self.calls.append(task.id)
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        return PredictionRecord(
            task_id=task.id,
            raw_text="synthetic answer",
            latency_ms=2,
            input_tokens=10,
            output_tokens=2,
        )

    def predict_prompt(self, example: PromptExample) -> PredictionRecord:
        self.calls.append(example.id)
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        return PredictionRecord(
            task_id=example.id,
            raw_text="external answer",
            latency_ms=2,
            input_tokens=10,
            output_tokens=2,
        )


def two_tasks() -> list[TaskRecord]:
    return [act_task(pair_id="productivity-001"), act_task(pair_id="productivity-002")]


def test_task_messages_include_state_request_and_policy() -> None:
    messages = task_messages(act_task())
    assert messages[0]["role"] == "system"
    assert "Never invent" in messages[0]["content"]
    assert '"tickets"' in messages[1]["content"]
    assert "Close ticket 7" in messages[1]["content"]


def test_prompt_variants_are_distinct_and_embed_tools_canonically() -> None:
    task = act_task()
    short = task_messages(task, PromptVariant.NATIVE_SHORT)
    embedded = task_messages(task, PromptVariant.EMBEDDED_TOOLS)
    assert short[0]["content"] != task_messages(task)[0]["content"]
    assert "Available tools (JSON):" in embedded[0]["content"]
    assert '"name":"close_ticket"' in embedded[0]["content"]
    assert (
        '<tool_call>{"name":"tool_name","arguments":{}}</tool_call>'
        in embedded[0]["content"]
    )
    assert "{tools}" in prompt_policy(PromptVariant.EMBEDDED_TOOLS)
    strict = task_messages(task, PromptVariant.PROTOCOL_STRICT)
    assert "balanced braces" in strict[0]["content"]
    assert "schema definitions" in prompt_policy(PromptVariant.PROTOCOL_STRICT)
    assert strict[0]["content"] != task_messages(task)[0]["content"]


def test_run_inference_resumes_and_preserves_order(tmp_path: Path) -> None:
    tasks = two_tasks()
    output = tmp_path / "predictions.jsonl"
    first_backend = FakeBackend()
    first = run_inference(tasks, first_backend, output, limit=1)
    assert len(first) == 1
    assert first_backend.calls == [tasks[0].id]

    resumed_backend = FakeBackend()
    resumed = run_inference(tasks, resumed_backend, output)
    assert [prediction.task_id for prediction in resumed] == [task.id for task in tasks]
    assert resumed_backend.calls == [tasks[1].id]
    assert len(output.read_text().splitlines()) == 2


def test_backend_failure_is_persisted(tmp_path: Path) -> None:
    task = act_task()
    predictions = run_inference(
        [task], FakeBackend(fail=True), tmp_path / "predictions.jsonl"
    )
    assert predictions[0].inference_error == "RuntimeError: synthetic backend failure"


def test_prompt_inference_resumes_and_persists_failures(tmp_path: Path) -> None:
    examples = [
        PromptExample(
            id=f"external-{index}",
            messages=({"role": "user", "content": "hello"},),
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {"type": "object"},
                    },
                },
            ),
        )
        for index in (1, 2)
    ]
    output = tmp_path / "external.jsonl"
    first = FakeBackend()
    run_prompt_inference(examples, first, output, limit=1)
    assert first.calls == ["external-1"]
    resumed = FakeBackend()
    predictions = run_prompt_inference(examples, resumed, output)
    assert resumed.calls == ["external-2"]
    assert len(predictions) == 2
    failed_output = tmp_path / "failed.jsonl"
    failed = run_prompt_inference(examples[:1], FakeBackend(fail=True), failed_output)
    assert failed[0].inference_error == "RuntimeError: synthetic backend failure"


def test_resume_rejects_duplicate_and_foreign_ids(tmp_path: Path) -> None:
    task = act_task()
    prediction = PredictionRecord(task_id=task.id, raw_text="x", latency_ms=1)
    output = tmp_path / "predictions.jsonl"
    write_jsonl(
        output,
        [prediction.model_dump(mode="json"), prediction.model_dump(mode="json")],
    )
    with pytest.raises(ValueError, match="duplicate task ids"):
        run_inference([task], FakeBackend(), output)

    foreign = PredictionRecord(task_id="foreign-act", raw_text="x", latency_ms=1)
    write_jsonl(output, [foreign.model_dump(mode="json")])
    with pytest.raises(ValueError, match="do not belong"):
        run_inference([task], FakeBackend(), output)


def test_config_and_task_loading(tmp_path: Path) -> None:
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(
        "model: local/model\n"
        f"revision: {'a' * 40}\n"
        "seed: 0\nmax_tokens: 32\ntemperature: 0.0\n",
        encoding="utf-8",
    )
    config = load_inference_config(config_path)
    assert config == InferenceConfig(
        model="local/model", revision="a" * 40, seed=0, max_tokens=32, temperature=0
    )
    task_path = tmp_path / "tasks.jsonl"
    write_jsonl(task_path, [act_task().model_dump(mode="json")])
    assert load_tasks(task_path) == [act_task()]


def test_run_manifest_records_pinned_provenance(tmp_path: Path) -> None:
    task = act_task()
    prediction_path = tmp_path / "predictions.jsonl"
    run_inference([task], FakeBackend(), prediction_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"adapter")
    config = InferenceConfig(
        model="local/model",
        revision="a" * 40,
        seed=0,
        max_tokens=32,
        temperature=0,
        adapter_path=str(adapter),
    )
    path = write_run_manifest(config, [task], prediction_path)
    content = path.read_text(encoding="utf-8")
    assert '"revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' in content
    assert '"predictions_hash"' in content
    assert '"prompt_variant":"native-full"' in content
    assert '"rendered_prompts_hash"' in content
    assert '"adapter_hash"' in content


def test_stratified_smoke_selects_four_complete_classes() -> None:
    pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=2,
            generator_version="1.0.0",
            split=DatasetSplit.VALIDATION,
        )
    )
    selected = select_stratified_smoke(
        [task for pair in pairs for task in (pair.act, pair.abstain)]
    )
    assert len(selected) == 8
    assert {
        task.label for task in selected if task.label is not DecisionClass.CALL
    } == {
        DecisionClass.ANSWER,
        DecisionClass.CLARIFY,
        DecisionClass.REFUSE,
        DecisionClass.NOOP,
    }


def test_stratified_smoke_rejects_missing_classes() -> None:
    with pytest.raises(ValueError, match="could not select smoke pairs"):
        select_stratified_smoke([act_task()])
