"""Tests for multi-domain generation, splitting, and manifests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tool_abstention.dataset import (
    build_full_dataset,
    build_pairs,
    family_split,
    template_family,
    validate_no_leakage,
)
from tool_abstention.domains import execute_domain_tool, generate_domain_pairs
from tool_abstention.productivity import ProductivityConfig
from tool_abstention.records import TaskPair, TaskRecord
from tool_abstention.taxonomy import DatasetSplit


def config(count: int = 25) -> ProductivityConfig:
    return ProductivityConfig(
        seed=0,
        pairs_per_class=count,
        generator_version="1.0.0",
        split=DatasetSplit.TRAIN,
    )


def test_builds_balanced_300_pair_corpus_without_leakage() -> None:
    pairs = build_pairs(config())
    assert len(pairs) == 300
    assert {pair.act.domain for pair in pairs} == {
        "productivity",
        "finance",
        "weather",
    }
    assert {
        split: sum(pair.act.split is split for pair in pairs) for split in DatasetSplit
    } == {
        DatasetSplit.TRAIN: 180,
        DatasetSplit.VALIDATION: 60,
        DatasetSplit.TEST: 60,
    }
    validate_no_leakage(pairs)
    assert all(family_split(template_family(pair)) is pair.act.split for pair in pairs)


def test_full_dataset_is_byte_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "full.yaml"
    config_path.write_text(
        "seed: 0\npairs_per_class: 25\ngenerator_version: 1.0.0\nsplit: train\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_full_dataset(config_path, first)
    second_manifest = build_full_dataset(config_path, second)
    assert first_manifest == second_manifest
    assert first_manifest["pair_count"] == 300
    assert first_manifest["task_count"] == 600
    assert first_manifest["domain_counts"] == {
        "finance": 100,
        "productivity": 100,
        "weather": 100,
    }
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "DATASET_CARD.md",
        "manifest.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_domain_generation_and_executor_error_paths() -> None:
    assert len(generate_domain_pairs("finance", config(1))) == 4
    assert len(generate_domain_pairs("weather", config(1))) == 4
    with pytest.raises(ValueError, match="unsupported domain"):
        generate_domain_pairs("unknown", config(1))
    assert execute_domain_tool(
        "finance", "list_transactions", {"account_id": "acct-1"}, {}
    ) == {"account_id": "acct-1", "transactions": []}
    with pytest.raises(ValueError, match="unknown finance tool"):
        execute_domain_tool("finance", "missing", {}, {})


def test_leakage_guards_reject_duplicate_pair_and_family_split() -> None:
    pair = build_pairs(config(1))[0]
    with pytest.raises(ValueError, match="duplicate pair id"):
        validate_no_leakage([pair, pair])
    data = pair.abstain.model_dump(mode="json")
    data["split"] = "test"
    with pytest.raises(ValidationError, match="must share split"):
        TaskPair(
            pair_id=pair.pair_id,
            act=pair.act,
            abstain=TaskRecord.model_validate(data),
        )
