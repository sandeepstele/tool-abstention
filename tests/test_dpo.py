"""Tests for DPO preparation, tokenization, math, and cache contracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from tool_abstention.dpo import (
    DpoExample,
    DpoPrepareConfig,
    DpoTrainingConfig,
    ReferenceCacheManifest,
    ReferenceLogps,
    build_dpo_dataset,
    numpy_dpo_metrics,
    numpy_sequence_logps,
    prepare_dpo_examples,
    tokenize_dpo_example,
    validate_reference_cache,
)
from tool_abstention.dpo_runtime import _check_versions, _validate_adapter_config
from tool_abstention.preference import generate_preferences
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.taxonomy import DatasetSplit
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import write_jsonl

SFT_HASH = "a" * 64


class FakeTokenizer:
    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        del kwargs
        prompt = [1, 2, 3]
        if messages[-1]["role"] != "assistant":
            return prompt
        content = messages[-1]["content"]
        return [*prompt, 9, *(10 + ord(char) % 31 for char in content), 4]


def task_records(split: DatasetSplit) -> list[Any]:
    pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=2,
            generator_version="1.0.0",
            split=split,
        )
    )
    return [task for pair in pairs for task in (pair.act, pair.abstain)]


def example() -> DpoExample:
    tasks = task_records(DatasetSplit.TRAIN)
    preferences = generate_preferences(tasks, sft_init_hash=SFT_HASH)
    return prepare_dpo_examples(
        DpoPrepareConfig(formatter_version="1.0.0", sft_init_hash=SFT_HASH),
        tasks,
        preferences,
    )[0]


def training_config() -> DpoTrainingConfig:
    return DpoTrainingConfig(
        model="local/model",
        revision="b" * 40,
        sft_adapter_path="checkpoints/sft",
        sft_init_hash=SFT_HASH,
        seed=0,
        beta=0.1,
        label_smoothing=0,
        logp_normalization="sum",
        batch_size=1,
        grad_accumulation_steps=1,
        iters=2,
        learning_rate=1e-5,
        max_seq_length=128,
        num_layers=4,
        rank=8,
        dropout=0,
        scale=16,
        steps_per_report=1,
        steps_per_eval=1,
        save_every=1,
        peak_memory_limit_gb=12,
        required_mlx_version="0.32.2",
        required_mlx_lm_version="0.29.1",
    )


def test_prepare_and_tokenize_share_prompt_and_preserve_signal() -> None:
    item = example()
    pair = tokenize_dpo_example(item, FakeTokenizer(), max_seq_length=512)
    assert pair.chosen[: pair.chosen_offset] == pair.rejected[: pair.rejected_offset]
    assert pair.chosen != pair.rejected
    assert pair.chosen_offset == 3
    assert not pair.chosen_truncated
    assert not pair.rejected_truncated


def test_tokenization_rejects_completion_removal_and_signal_erasure() -> None:
    item = example()
    with pytest.raises(ValueError, match="removed the completion"):
        tokenize_dpo_example(item, FakeTokenizer(), max_seq_length=3)
    with pytest.raises(ValueError, match="erased the preference"):
        tokenize_dpo_example(item, FakeTokenizer(), max_seq_length=4)


def test_numpy_sequence_logps_fixed_vector() -> None:
    logits = np.array([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    targets = np.array([[0, 1, 0]])
    mask = np.array([[False, True, True]])
    logps, counts = numpy_sequence_logps(logits, targets, mask)
    expected = -np.log1p(np.exp(-2.0)) - np.log(2.0)
    assert logps == pytest.approx([expected])
    assert counts.tolist() == [2]
    with pytest.raises(ValueError, match="completion tokens"):
        numpy_sequence_logps(logits, targets, np.zeros_like(mask))


def test_numpy_dpo_invariants_and_smoothing() -> None:
    zeros = np.zeros(2)
    equal = numpy_dpo_metrics(zeros, zeros, zeros, zeros, beta=0.1, label_smoothing=0)
    assert equal["loss"] == pytest.approx(np.log(2))
    assert equal["reward_margin"] == 0
    preferred = numpy_dpo_metrics(
        np.ones(2), zeros, zeros, zeros, beta=0.1, label_smoothing=0
    )
    assert preferred["loss"] < equal["loss"]
    assert preferred["reward_accuracy"] == 1
    swapped = numpy_dpo_metrics(
        zeros, np.ones(2), zeros, zeros, beta=0.1, label_smoothing=0
    )
    assert swapped["loss"] > equal["loss"]
    smoothed = numpy_dpo_metrics(
        np.ones(2), zeros, zeros, zeros, beta=0.1, label_smoothing=0.1
    )
    assert preferred["loss"] < smoothed["loss"] < swapped["loss"]
    beta_zero = numpy_dpo_metrics(zeros, zeros, zeros, zeros, beta=0, label_smoothing=0)
    assert beta_zero["loss"] == pytest.approx(np.log(2))
    mean = numpy_dpo_metrics(
        np.array([-2.0]),
        np.array([-3.0]),
        np.array([-6.0]),
        np.array([-4.0]),
        beta=0.1,
        label_smoothing=0,
        chosen_tokens=np.array([2]),
        rejected_tokens=np.array([1]),
        normalization="mean",
    )
    assert mean["reward_margin"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="requires"):
        numpy_dpo_metrics(
            zeros,
            zeros,
            zeros,
            zeros,
            beta=0.1,
            label_smoothing=0,
            normalization="mean",
        )


def test_dpo_dataset_is_deterministic_internal_and_balanced(tmp_path: Path) -> None:
    internal = tmp_path / "internal"
    preferences = tmp_path / "preferences"
    internal.mkdir()
    preferences.mkdir()
    train = task_records(DatasetSplit.TRAIN)
    valid = [
        task.model_copy(
            update={
                "pair_id": f"validation-{task.pair_id}",
                "id": f"validation-{task.id}",
            }
        )
        for task in task_records(DatasetSplit.VALIDATION)
    ]
    write_jsonl(
        internal / "train.jsonl", [item.model_dump(mode="json") for item in train]
    )
    write_jsonl(
        internal / "validation.jsonl", [item.model_dump(mode="json") for item in valid]
    )
    write_jsonl(
        preferences / "train.jsonl",
        [
            item.model_dump(mode="json")
            for item in generate_preferences(train, sft_init_hash=SFT_HASH)
        ],
    )
    write_jsonl(
        preferences / "valid.jsonl",
        [
            item.model_dump(mode="json")
            for item in generate_preferences(valid, sft_init_hash=SFT_HASH)
        ],
    )
    (internal / "test.jsonl").write_text("invalid and unread\n", encoding="utf-8")
    config = tmp_path / "dpo.yaml"
    config.write_text(
        f"formatter_version: 1.0.0\nsft_init_hash: {SFT_HASH}\n"
        "train_limit: 8\nvalidation_limit: 8\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_dpo_dataset(config, internal, preferences, first)
    build_dpo_dataset(config, internal, preferences, second)
    assert manifest["external_sources"] == []
    assert manifest["test_consumed"] is False
    assert manifest["train_count"] == 8
    selected = [
        json.loads(line) for line in (first / "train.jsonl").read_text().splitlines()
    ]
    selected_pairs = {item["task_id"].rpartition("-")[0] for item in selected}
    assert len(selected_pairs) == 4
    assert all(
        sum(item["task_id"].startswith(pair_id) for item in selected) == 2
        for pair_id in selected_pairs
    )
    assert {
        label
        for label in ("answer", "clarify", "refuse", "noop")
        if any(f"-{label}-" in pair_id for pair_id in selected_pairs)
    } == {"answer", "clarify", "refuse", "noop"}
    for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _write_cache(tmp_path: Path) -> tuple[Path, Path, Path]:
    examples_path = tmp_path / "valid.jsonl"
    records_path = tmp_path / "reference.jsonl"
    item = example()
    write_jsonl(examples_path, [item.model_dump(mode="json")])
    record = ReferenceLogps(
        id=item.id,
        example_hash=sha256_object(item.model_dump(mode="json")),
        chosen_logp=-2.0,
        rejected_logp=-3.0,
        chosen_tokens=2,
        rejected_tokens=3,
    )
    write_jsonl(records_path, [record.model_dump(mode="json")])
    config = training_config()
    manifest = ReferenceCacheManifest(
        model=config.model,
        revision=config.revision,
        adapter_hash=config.sft_init_hash,
        examples_hash=sha256_file(examples_path),
        tokenizer_hash="c" * 64,
        max_seq_length=config.max_seq_length,
        record_count=1,
        records_hash=sha256_file(records_path),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    return examples_path, records_path, manifest_path


def test_reference_cache_validates_identity_completeness_and_hashes(
    tmp_path: Path,
) -> None:
    examples, records, manifest = _write_cache(tmp_path)
    loaded, by_id = validate_reference_cache(
        examples, records, manifest, config=training_config()
    )
    assert loaded[0].id in by_id
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["examples_hash"] = "d" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="examples hash is stale"):
        validate_reference_cache(examples, records, manifest, config=training_config())


def test_contracts_reject_test_nonfinite_and_bad_versions() -> None:
    item = example()
    with pytest.raises(ValidationError, match="DPO id"):
        DpoExample.model_validate({**item.model_dump(mode="json"), "id": "dpo-wrong"})
    with pytest.raises(ValidationError, match="test examples"):
        DpoExample.model_validate(
            {**item.model_dump(mode="json"), "split": DatasetSplit.TEST}
        )
    with pytest.raises(ValidationError, match="must differ"):
        DpoExample.model_validate(
            {**item.model_dump(mode="json"), "rejected": item.chosen}
        )


def test_dpo_rejects_leakage_stale_sources_and_invalid_math() -> None:
    item = example()
    tasks = task_records(DatasetSplit.TRAIN)
    preferences = generate_preferences(tasks, sft_init_hash=SFT_HASH)
    config = DpoPrepareConfig(formatter_version="1.0.0", sft_init_hash=SFT_HASH)
    with pytest.raises(ValueError, match="task ids"):
        prepare_dpo_examples(config, [tasks[0], tasks[0]], preferences[:1])
    test_task = tasks[0].model_copy(update={"split": DatasetSplit.TEST})
    with pytest.raises(ValueError, match="test tasks"):
        prepare_dpo_examples(config, [test_task], [])
    with pytest.raises(ValueError, match="preference ids"):
        prepare_dpo_examples(config, tasks, [preferences[0], preferences[0]])
    missing = preferences[0].model_copy(update={"task_id": "missing-task"})
    with pytest.raises(ValueError, match="no internal task"):
        prepare_dpo_examples(config, tasks, [missing])
    stale_provenance = preferences[0].provenance.model_copy(
        update={"source_task_hash": "b" * 64}
    )
    stale = preferences[0].model_copy(update={"provenance": stale_provenance})
    with pytest.raises(ValueError, match="task hash is stale"):
        prepare_dpo_examples(config, tasks, [stale])
    wrong_sft = preferences[0].model_copy(
        update={
            "provenance": preferences[0].provenance.model_copy(
                update={"sft_init_hash": "b" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="initialization hash"):
        prepare_dpo_examples(config, tasks, [wrong_sft])
    logits = np.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="shapes"):
        numpy_sequence_logps(logits, np.zeros((1, 3)), np.zeros((1, 3)))
    zeros = np.zeros(1)
    with pytest.raises(ValueError, match="invalid DPO"):
        numpy_dpo_metrics(zeros, zeros, zeros, zeros, beta=-1, label_smoothing=0)
    with pytest.raises(ValueError, match="must be finite"):
        numpy_dpo_metrics(
            np.array([np.nan]), zeros, zeros, zeros, beta=0.1, label_smoothing=0
        )
    with pytest.raises(ValueError, match="complete pairs"):
        from tool_abstention.dpo import _select_balanced

        _select_balanced(prepare_dpo_examples(config, tasks, preferences), 3)
    with pytest.raises(ValidationError, match="finite"):
        ReferenceLogps(
            id=item.id,
            example_hash=sha256_object(item.model_dump(mode="json")),
            chosen_logp=float("nan"),
            rejected_logp=-1,
            chosen_tokens=1,
            rejected_tokens=1,
        )


def test_training_config_and_runtime_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training_config()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "num_layers": 4,
                "lora_parameters": {"rank": 8, "dropout": 0, "scale": 16},
            }
        ),
        encoding="utf-8",
    )
    _validate_adapter_config(config, adapter)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="LoRA configuration"):
        _validate_adapter_config(config, adapter)
    monkeypatch.setattr(
        "tool_abstention.dpo_runtime.importlib.metadata.version",
        lambda package: "0.32.2" if package == "mlx" else "0.29.1",
    )
    _check_versions(config)
    monkeypatch.setattr(
        "tool_abstention.dpo_runtime.importlib.metadata.version", lambda _: "0.0.0"
    )
    with pytest.raises(ValueError, match="MLX version mismatch"):
        _check_versions(config)
    invalid = config.model_dump()
    invalid["grad_accumulation_steps"] = 3
    with pytest.raises(ValidationError, match="align"):
        DpoTrainingConfig.model_validate(invalid)
