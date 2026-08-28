"""Tests for deterministic internal-only SFT preparation and launch."""

import sys
import types
from pathlib import Path

import pytest

from tool_abstention.mlx_sft_runner import token_ids
from tool_abstention.records import (
    AnswerExpected,
    AnswerValidator,
    DomainAnswerValidator,
    NormalizedTextAnswerValidator,
    NumericAnswerValidator,
    SetAnswerValidator,
)
from tool_abstention.sft import (
    SftTrainingConfig,
    build_sft_dataset,
    format_sft_example,
    run_sft_training,
    training_command,
)
from tool_abstention.taxonomy import DatasetSplit, DecisionClass
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

from .test_records import abstain_task, act_task


def test_token_ids_accepts_lists_and_batch_mappings() -> None:
    assert token_ids([1, 2]) == [1, 2]
    assert token_ids({"input_ids": [3, 4], "attention_mask": [1, 1]}) == [3, 4]
    with pytest.raises(TypeError, match="flat integer"):
        token_ids({"input_ids": [[1, 2]]})


def test_formatter_covers_call_and_each_abstention_class() -> None:
    call = format_sft_example(act_task())
    assert call["messages"][-1]["content"].startswith("<tool_call>")
    assert call["tools"][0]["type"] == "function"
    expected_fragments = {
        DecisionClass.ANSWER: "Ticket 7 is closed.",
        DecisionClass.CLARIFY: "Could you provide",
        DecisionClass.REFUSE: "not available",
        DecisionClass.NOOP: "already closed",
    }
    for label, fragment in expected_fragments.items():
        example = format_sft_example(abstain_task(label))
        assert fragment in example["messages"][-1]["content"]


def test_formatter_covers_structured_answer_validators() -> None:
    task = abstain_task(DecisionClass.ANSWER)
    validators: list[tuple[AnswerValidator, str]] = [
        (NormalizedTextAnswerValidator(value="Paris, France"), "Paris, France"),
        (
            NumericAnswerValidator(value=3.14, absolute_tolerance=0.01, unit="m"),
            "3.14 m",
        ),
        (SetAnswerValidator(values=("red", "blue")), "red, blue"),
    ]
    for validator, expected in validators:
        changed = task.model_copy(
            update={"expected": AnswerExpected(validator=validator)}
        )
        assert format_sft_example(changed)["messages"][-1]["content"] == expected
    domain = task.model_copy(
        update={
            "expected": AnswerExpected(
                validator=DomainAnswerValidator(validator_id="custom")
            )
        }
    )
    with pytest.raises(ValueError, match="no deterministic training response"):
        format_sft_example(domain)


def test_sft_export_is_deterministic_and_never_reads_test(tmp_path: Path) -> None:
    internal = tmp_path / "internal"
    internal.mkdir()
    train = act_task().model_copy(update={"split": DatasetSplit.TRAIN})
    valid = abstain_task(DecisionClass.ANSWER).model_copy(
        update={"split": DatasetSplit.VALIDATION}
    )
    write_jsonl(internal / "train.jsonl", [train.model_dump(mode="json")])
    write_jsonl(internal / "validation.jsonl", [valid.model_dump(mode="json")])
    (internal / "test.jsonl").write_text("this is deliberately invalid\n")
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_sft_dataset(internal, first)
    build_sft_dataset(internal, second)
    assert manifest["test_consumed"] is False
    assert manifest["source_splits"] == ["train", "validation"]
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert len(list(read_jsonl(first / "train.jsonl"))) == 1
    assert not (first / "test.jsonl").exists()


def test_sft_export_rejects_wrong_split(tmp_path: Path) -> None:
    internal = tmp_path / "internal"
    internal.mkdir()
    wrong = act_task().model_copy(update={"split": DatasetSplit.TEST})
    write_jsonl(internal / "train.jsonl", [wrong.model_dump(mode="json")])
    write_jsonl(internal / "validation.jsonl", [])
    with pytest.raises(ValueError, match="wrong split"):
        build_sft_dataset(internal, tmp_path / "output")


def test_training_command_is_pinned_and_masks_prompts(tmp_path: Path) -> None:
    config = SftTrainingConfig(
        model="model/name",
        revision="a" * 40,
        seed=0,
        batch_size=1,
        grad_accumulation_steps=2,
        iters=3,
        learning_rate=1e-4,
        num_layers=4,
        max_seq_length=512,
        steps_per_report=1,
        steps_per_eval=2,
        save_every=2,
        rank=8,
        dropout=0,
        scale=16,
        grad_checkpoint=True,
    )
    command = training_command(
        config,
        model_path=tmp_path / "snapshot",
        data_path=tmp_path / "data",
        adapter_path=tmp_path / "adapter",
    )
    assert "--mask-prompt" in command
    assert command[command.index("--model") + 1].endswith("snapshot")
    assert command[command.index("--grad-accumulation-steps") + 1] == "2"
    assert "--grad-checkpoint" in command


def test_training_launcher_resolves_revision_and_records_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "manifest.json").write_text("{}\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model: model/name\n"
        f"revision: {'a' * 40}\n"
        "seed: 0\nbatch_size: 1\ngrad_accumulation_steps: 1\n"
        "iters: 2\nlearning_rate: 0.0001\nnum_layers: 2\n"
        "max_seq_length: 256\nval_batches: 1\nsteps_per_report: 1\n"
        "steps_per_eval: 1\nsave_every: 1\nrank: 4\ndropout: 0\n"
        "scale: 8\nmask_prompt: true\n"
    )
    seen: dict[str, object] = {}

    def fake_snapshot_download(
        model: str, *, revision: str, allow_patterns: list[str]
    ) -> str:
        seen.update(model=model, revision=revision, patterns=allow_patterns)
        return str(model_path)

    def fake_run(
        command: list[str],
        *,
        check: bool,
        stdout: object,
        stderr: object,
        text: bool,
    ) -> None:
        assert check
        assert stdout is not None and stderr is not None and text
        adapter = Path(command[command.index("--adapter-path") + 1])
        (adapter / "adapters.safetensors").write_bytes(b"adapter")

    fake_hub = types.ModuleType("huggingface_hub")
    monkeypatch.setattr(
        fake_hub, "snapshot_download", fake_snapshot_download, raising=False
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr("tool_abstention.sft.subprocess.run", fake_run)
    output = tmp_path / "adapter"
    manifest = run_sft_training(config_path, data_path, output)
    assert seen["revision"] == "a" * 40
    assert manifest["revision"] == "a" * 40
    assert manifest["adapter_hash"]
    assert (output / "run_manifest.json").is_file()
