"""Tests for strict deterministic preference generation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tool_abstention.cli import main
from tool_abstention.preference import (
    NegativeType,
    PreferenceProvenance,
    PreferenceRecord,
    build_preference_dataset,
    generate_preferences,
    validate_preference,
)
from tool_abstention.productivity import ProductivityConfig, generate_productivity_pairs
from tool_abstention.records import TaskRecord
from tool_abstention.taxonomy import DatasetSplit
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

from .test_records import act_task

SFT_HASH = "a" * 64


def tasks(split: DatasetSplit = DatasetSplit.TRAIN) -> list[TaskRecord]:
    pairs = generate_productivity_pairs(
        ProductivityConfig(
            seed=0,
            pairs_per_class=2,
            generator_version="1.0.0",
            split=split,
        )
    )
    return [task for pair in pairs for task in (pair.act, pair.abstain)]


def test_generation_covers_all_negative_types_and_validates_semantics() -> None:
    source = tasks()
    records = generate_preferences(source, sft_init_hash=SFT_HASH)
    assert len(records) == len(source)
    assert {record.negative_type for record in records} == set(NegativeType)
    by_id = {task.id: task for task in source}
    for record in records:
        validate_preference(record, by_id[record.task_id])
        assert record.provenance.external_sources == ()
        assert record.provenance.test_consumed is False
        assert record.provenance.sft_init_hash == SFT_HASH


def test_record_rejects_test_identical_and_external_data() -> None:
    source = act_task()
    provenance = PreferenceProvenance(
        source_task_hash="b" * 64,
        generator_version="1.0.0",
        sft_init_hash=SFT_HASH,
    )
    with pytest.raises(ValidationError, match="test records are prohibited"):
        PreferenceRecord(
            id=f"pref-{source.id}",
            task_id=source.id,
            pair_id=source.pair_id,
            split=DatasetSplit.TEST,
            target_class=source.label,
            negative_type=NegativeType.WRONG_ARGUMENTS,
            chosen="same",
            rejected="different",
            provenance=provenance,
        )
    with pytest.raises(ValidationError, match="must differ"):
        PreferenceRecord(
            id=f"pref-{source.id}",
            task_id=source.id,
            pair_id=source.pair_id,
            split=DatasetSplit.TRAIN,
            target_class=source.label,
            negative_type=NegativeType.WRONG_ARGUMENTS,
            chosen="same",
            rejected=" same ",
            provenance=provenance,
        )
    with pytest.raises(ValidationError, match="external sources"):
        PreferenceProvenance(
            source_task_hash="b" * 64,
            generator_version="1.0.0",
            sft_init_hash=SFT_HASH,
            external_sources=("bfcl",),
        )


def test_validation_rejects_tampering_and_wrong_declared_type() -> None:
    source = tasks()
    records = generate_preferences(source, sft_init_hash=SFT_HASH)
    record = next(
        item
        for item in records
        if item.negative_type is NegativeType.MALFORMED_TOOL_CALL
    )
    task = next(task for task in source if task.id == record.task_id)
    with pytest.raises(ValueError, match="source task hash"):
        validate_preference(
            record.model_copy(
                update={
                    "provenance": record.provenance.model_copy(
                        update={"source_task_hash": "c" * 64}
                    )
                }
            ),
            task,
        )
    with pytest.raises(ValueError, match="non-malformed rejection"):
        validate_preference(
            record.model_copy(update={"negative_type": NegativeType.WRONG_ARGUMENTS}),
            task,
        )


def test_generation_rejects_test_and_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match="test records"):
        generate_preferences(
            tasks(DatasetSplit.TEST),
            sft_init_hash=SFT_HASH,
        )
    with pytest.raises(ValueError, match="incomplete"):
        generate_preferences([act_task()], sft_init_hash=SFT_HASH)


def _write_internal(root: Path) -> None:
    root.mkdir()
    train = tasks(DatasetSplit.TRAIN)
    valid = tasks(DatasetSplit.VALIDATION)
    write_jsonl(root / "train.jsonl", [task.model_dump(mode="json") for task in train])
    write_jsonl(
        root / "validation.jsonl",
        [task.model_dump(mode="json") for task in valid],
    )
    (root / "test.jsonl").write_text("deliberately invalid\n", encoding="utf-8")


def test_dataset_is_deterministic_balanced_and_never_reads_test(tmp_path: Path) -> None:
    internal = tmp_path / "internal"
    _write_internal(internal)
    config = tmp_path / "preferences.yaml"
    config.write_text(
        f"generator_version: 1.0.0\nsft_init_hash: {SFT_HASH}\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_preference_dataset(config, internal, first)
    build_preference_dataset(config, internal, second)
    assert manifest["test_consumed"] is False
    assert manifest["external_sources"] == []
    assert manifest["negative_distribution"]
    for filename in ("train.jsonl", "valid.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert not (first / "test.jsonl").exists()


def test_cli_builds_and_schema_validates_preference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    internal = tmp_path / "internal"
    _write_internal(internal)
    config = tmp_path / "preferences.yaml"
    config.write_text(
        f"generator_version: 1.0.0\nsft_init_hash: {SFT_HASH}\n",
        encoding="utf-8",
    )
    output = tmp_path / "preferences"
    main(
        [
            "build-preferences",
            "--config",
            str(config),
            "--internal",
            str(internal),
            "--output",
            str(output),
        ]
    )
    first = next(read_jsonl(output / "train.jsonl"))
    record_path = tmp_path / "preference.json"
    import json

    record_path.write_text(json.dumps(first), encoding="utf-8")
    main(["validate-record", "preference", str(record_path)])
    captured = capsys.readouterr().out
    assert '"test_consumed": false' in captured
