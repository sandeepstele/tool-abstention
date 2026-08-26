"""Tests for deterministic productivity tools and paired generation."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tool_abstention.productivity import (
    CONTACTS,
    PRODUCTIVITY_TOOLS,
    ProductivityConfig,
    audit_pairs,
    build_productivity_dataset,
    execute_tool,
    generate_productivity_pairs,
    load_pairs,
    semantic_pair_changes,
    validate_semantic_pair,
)
from tool_abstention.records import CallExpected, TaskPair, TaskRecord
from tool_abstention.taxonomy import DatasetSplit, DecisionClass, PerturbationType
from tool_abstention.util.hashing import sha256_file
from tool_abstention.util.jsonl import write_jsonl


def config(*, seed: int = 0, count: int = 10) -> ProductivityConfig:
    return ProductivityConfig(
        seed=seed,
        pairs_per_class=count,
        generator_version="1.0.0",
        split=DatasetSplit.TRAIN,
    )


def write_config(path: Path, *, seed: int = 0, count: int = 10) -> None:
    path.write_text(
        f"seed: {seed}\npairs_per_class: {count}\n"
        "generator_version: 1.0.0\nsplit: train\n",
        encoding="utf-8",
    )


def test_generation_produces_40_balanced_executable_pairs() -> None:
    pairs = generate_productivity_pairs(config())
    assert len(pairs) == 40
    counts: dict[DecisionClass, int] = {
        label: 0 for label in DecisionClass if label is not DecisionClass.CALL
    }
    for pair in pairs:
        counts[pair.abstain.label] += 1
        expected = pair.act.expected
        assert isinstance(expected, CallExpected)
        assert (
            execute_tool(expected.tool_name, expected.arguments, pair.act.environment)
            == expected.expected_result
        )
    assert set(counts.values()) == {10}
    assert {tool.name for tool in PRODUCTIVITY_TOOLS} == {
        "search_contacts",
        "create_event",
        "close_ticket",
        "send_email",
    }
    clarify_queries = [
        pair.abstain.query
        for pair in pairs
        if pair.abstain.label is DecisionClass.CLARIFY
    ]
    assert all(not query.endswith('".') for query in clarify_queries)


def test_generation_is_seeded_and_repeatable() -> None:
    first = generate_productivity_pairs(config(seed=11, count=4))
    repeated = generate_productivity_pairs(config(seed=11, count=4))
    changed = generate_productivity_pairs(config(seed=12, count=4))
    first_json = [pair.model_dump_json() for pair in first]
    assert first_json == [pair.model_dump_json() for pair in repeated]
    assert first_json != [pair.model_dump_json() for pair in changed]


def test_every_pair_changes_only_declared_semantic_dimension() -> None:
    expected = {
        PerturbationType.ANSWER_PROVIDED: frozenset({"query"}),
        PerturbationType.REQUIRED_ARGUMENT_REMOVED: frozenset({"query"}),
        PerturbationType.TOOL_REMOVED: frozenset({"tools"}),
        PerturbationType.ALREADY_SATISFIED: frozenset({"environment"}),
    }
    for pair in generate_productivity_pairs(config()):
        assert pair.abstain.perturbation is not None
        assert semantic_pair_changes(pair) == expected[pair.abstain.perturbation]


def test_semantic_validator_detects_second_change() -> None:
    pair = generate_productivity_pairs(config(count=1))[0]
    abstain_data = pair.abstain.model_dump(mode="json")
    abstain_data["environment"] = {"unexpected": True}
    invalid = TaskPair(
        pair_id=pair.pair_id,
        act=pair.act,
        abstain=TaskRecord.model_validate(abstain_data),
    )
    with pytest.raises(ValueError, match="semantic changes"):
        validate_semantic_pair(invalid)


def test_tool_execution_error_paths_and_noop_result() -> None:
    with pytest.raises(ValueError, match="unknown productivity tool"):
        execute_tool("missing", {}, {})
    with pytest.raises(ValueError, match="invalid arguments"):
        execute_tool("close_ticket", {"ticket_id": "bad"}, {})
    assert execute_tool("search_contacts", {"name": "Nobody"}, {"contacts": []}) is None
    assert execute_tool(
        "close_ticket", {"ticket_id": 7}, {"tickets": {"7": "closed"}}
    ) == {"ticket_id": 7, "status": "closed", "changed": False}


def test_tool_execution_does_not_mutate_input_state() -> None:
    state: dict[str, Any] = {"tickets": {"7": "open"}}
    execute_tool("close_ticket", {"ticket_id": 7}, state)
    assert state == {"tickets": {"7": "open"}}


def test_build_is_byte_identical_with_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "productivity.yaml"
    write_config(config_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build_productivity_dataset(config_path, first_dir)
    second = build_productivity_dataset(config_path, second_dir)
    assert first == second
    for filename in ("tasks.jsonl", "manifest.json"):
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()
    assert first["pair_count"] == 40
    assert first["task_count"] == 80
    assert first["artifacts"]["tasks.jsonl"]["content_hash"] == sha256_file(
        first_dir / "tasks.jsonl"
    )


def test_load_and_audit_all_pairs(tmp_path: Path) -> None:
    config_path = tmp_path / "productivity.yaml"
    write_config(config_path)
    output = tmp_path / "output"
    build_productivity_dataset(config_path, output)
    pairs = load_pairs(output / "tasks.jsonl")
    audit = audit_pairs(pairs)
    assert len(pairs) == 40
    assert audit.count("PAIR productivity-") == 40
    assert audit.count("  ACT:") == 40
    assert audit.count("  ABSTAIN:") == 40
    assert all(pair.pair_id in audit for pair in pairs)


def test_load_rejects_duplicate_and_incomplete_pairs(tmp_path: Path) -> None:
    pair = generate_productivity_pairs(config(count=1))[0]
    duplicate_path = tmp_path / "duplicate.jsonl"
    write_jsonl(
        duplicate_path,
        [pair.act.model_dump(mode="json"), pair.act.model_dump(mode="json")],
    )
    with pytest.raises(ValueError, match="duplicate act task"):
        load_pairs(duplicate_path)

    incomplete_path = tmp_path / "incomplete.jsonl"
    write_jsonl(incomplete_path, [pair.act.model_dump(mode="json")])
    with pytest.raises(ValueError, match="incomplete pair"):
        load_pairs(incomplete_path)


def test_config_rejects_more_entities_than_available() -> None:
    assert len(CONTACTS) == 10
    with pytest.raises(ValidationError):
        ProductivityConfig(
            seed=0,
            pairs_per_class=11,
            generator_version="1.0.0",
            split=DatasetSplit.TRAIN,
        )
