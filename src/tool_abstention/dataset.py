"""Combined multi-domain dataset construction and leakage-safe splitting."""

from collections import Counter
from pathlib import Path
from typing import Any

from tool_abstention.config import load_yaml_config
from tool_abstention.domains import execute_domain_tool, generate_domain_pairs
from tool_abstention.productivity import (
    ProductivityConfig,
    execute_tool,
    generate_productivity_pairs,
    validate_semantic_pair,
)
from tool_abstention.records import CallExpected, TaskPair, TaskRecord
from tool_abstention.taxonomy import DatasetSplit
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import write_jsonl


def template_family(pair: TaskPair) -> str:
    """Return the controlled template group used for split isolation."""
    index = int(pair.pair_id.rsplit("-", 1)[1])
    return f"{pair.act.domain}-{pair.abstain.label.value.casefold()}-{index % 5}"


def family_split(family: str) -> DatasetSplit:
    """Map five template groups to an exact 60/20/20 partition."""
    group = int(family.rsplit("-", 1)[1])
    if group < 3:
        return DatasetSplit.TRAIN
    if group == 3:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def _with_split(pair: TaskPair, split: DatasetSplit) -> TaskPair:
    act = TaskRecord.model_validate(
        {**pair.act.model_dump(mode="json"), "split": split}
    )
    abstain = TaskRecord.model_validate(
        {**pair.abstain.model_dump(mode="json"), "split": split}
    )
    return TaskPair(pair_id=pair.pair_id, act=act, abstain=abstain)


def build_pairs(config: ProductivityConfig) -> list[TaskPair]:
    """Build and validate the 300-pair corpus."""
    pairs = generate_productivity_pairs(config)
    pairs.extend(generate_domain_pairs("finance", config))
    pairs.extend(generate_domain_pairs("weather", config))
    assigned = [
        _with_split(pair, family_split(template_family(pair))) for pair in pairs
    ]
    for pair in assigned:
        validate_semantic_pair(pair)
        expected = pair.act.expected
        assert isinstance(expected, CallExpected)
        if pair.act.domain == "productivity":
            result = execute_tool(
                expected.tool_name, expected.arguments, pair.act.environment
            )
        else:
            result = execute_domain_tool(
                pair.act.domain,
                expected.tool_name,
                expected.arguments,
                pair.act.environment,
            )
        if result != expected.expected_result:
            raise ValueError(f"{pair.pair_id}: executor result mismatch")
    return assigned


def validate_no_leakage(pairs: list[TaskPair]) -> None:
    """Reject pair, family, or normalized-query leakage across splits."""
    family_splits: dict[str, DatasetSplit] = {}
    query_splits: dict[str, DatasetSplit] = {}
    seen_pairs: set[str] = set()
    for pair in pairs:
        if pair.pair_id in seen_pairs:
            raise ValueError(f"duplicate pair id: {pair.pair_id}")
        seen_pairs.add(pair.pair_id)
        family = template_family(pair)
        previous = family_splits.setdefault(family, pair.act.split)
        if previous is not pair.act.split:
            raise ValueError(f"template family leaked across splits: {family}")
        for task in (pair.act, pair.abstain):
            normalized = " ".join(task.query.casefold().split())
            previous_query = query_splits.setdefault(normalized, task.split)
            if previous_query is not task.split:
                raise ValueError("normalized query leaked across splits")


def build_full_dataset(config_path: Path, output: Path) -> dict[str, Any]:
    """Write split artifacts, dataset card, and deterministic manifest."""
    config = load_yaml_config(config_path, ProductivityConfig)
    pairs = build_pairs(config)
    validate_no_leakage(pairs)
    by_split: dict[DatasetSplit, list[dict[str, Any]]] = {
        split: [] for split in DatasetSplit
    }
    for pair in pairs:
        by_split[pair.act.split].extend(
            (pair.act.model_dump(mode="json"), pair.abstain.model_dump(mode="json"))
        )
    artifacts: dict[str, dict[str, Any]] = {}
    for split, records in by_split.items():
        path = output / f"{split.value}.jsonl"
        write_jsonl(path, records)
        artifacts[path.name] = {
            "content_hash": sha256_file(path),
            "tasks": len(records),
        }
    domain_counts = Counter(pair.act.domain for pair in pairs)
    class_counts = Counter(pair.abstain.label.value for pair in pairs)
    card = (
        "# Generated Dataset Card\n\n"
        f"- Pairs: {len(pairs)}\n- Tasks: {len(pairs) * 2}\n"
        f"- Domains: {dict(sorted(domain_counts.items()))}\n"
        f"- Abstention classes: {dict(sorted(class_counts.items()))}\n"
        "- Split policy: template-family grouped 60/20/20\n"
        "- Limitations: synthetic, single-turn, template-generated development corpus\n"
    )
    card_path = output / "DATASET_CARD.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    artifacts[card_path.name] = {"content_hash": sha256_file(card_path)}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator_version": config.generator_version,
        "config_hash": sha256_object(config.model_dump(mode="json")),
        "pair_count": len(pairs),
        "task_count": len(pairs) * 2,
        "test_set_hash": artifacts["test.jsonl"]["content_hash"],
        "domain_counts": dict(sorted(domain_counts.items())),
        "abstention_class_counts": dict(sorted(class_counts.items())),
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest
