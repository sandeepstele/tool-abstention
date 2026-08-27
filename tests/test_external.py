"""Tests for pinned external benchmark preparation and evaluation."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from tool_abstention.external import (
    ExternalDecision,
    ExternalDecisionRecord,
    ExternalFunction,
    ExternalMessage,
    SourceProvenance,
    SourceUsage,
    catalog_agentabstain,
    evaluate_external_records,
    fetch_external,
    near_duplicate,
    normalize_bfcl_schema,
    parse_bfcl_file,
    prepare_external,
    quarantine_leakage,
)
from tool_abstention.records import PredictionRecord
from tool_abstention.taxonomy import DatasetSplit
from tool_abstention.util.jsonl import write_jsonl

from .test_records import act_task


def provenance(original_id: str = "simple_0") -> SourceProvenance:
    return SourceProvenance(
        source_name="BFCL",
        source_uri="https://example.com/dataset",
        revision="a" * 40,
        license="Apache-2.0",
        usage=SourceUsage.EXTERNAL_EVAL,
        original_id=original_id,
        source_file="BFCL_v3_simple.json",
        source_sha256="b" * 64,
        adapter_version="1.0.0",
        transformations=("python-types-to-json-schema",),
        attribution="BFCL",
    )


def external_record(
    record_id: str, decision: ExternalDecision
) -> ExternalDecisionRecord:
    return ExternalDecisionRecord(
        id=record_id,
        category="simple" if decision is ExternalDecision.CALL else "irrelevance",
        expected_decision=decision,
        messages=(ExternalMessage(role="user", content=f"request {record_id}"),),
        functions=(
            ExternalFunction(
                name="lookup",
                description="Look up a value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            ),
        ),
        provenance=provenance(record_id),
    )


def bfcl_row(row_id: str = "simple_0") -> dict[str, Any]:
    return {
        "id": row_id,
        "question": [[{"role": "user", "content": "Look up alpha."}]],
        "function": [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "value": {"type": "str"},
                        "weight": {"type": "float"},
                        "flags": {"type": "list", "items": {"type": "bool"}},
                    },
                    "required": ["value"],
                },
            }
        ],
    }


def test_provenance_is_strict_and_prohibits_training() -> None:
    assert provenance().usage is SourceUsage.EXTERNAL_EVAL
    data = provenance().model_dump(mode="json")
    data["usage"] = "training"
    with pytest.raises(ValidationError, match="cannot be used for training"):
        SourceProvenance.model_validate(data)
    data = provenance().model_dump(mode="json")
    data["revision"] = "main"
    with pytest.raises(ValidationError):
        SourceProvenance.model_validate(data)
    data = provenance().model_dump(mode="json")
    data["unknown"] = True
    with pytest.raises(ValidationError):
        SourceProvenance.model_validate(data)


def test_schema_normalization_is_recursive_and_fails_closed() -> None:
    normalized = normalize_bfcl_schema(bfcl_row()["function"][0]["parameters"])
    assert normalized["type"] == "object"
    assert normalized["properties"]["weight"]["type"] == "number"
    assert normalized["properties"]["flags"]["items"]["type"] == "boolean"
    assert normalize_bfcl_schema({"type": "tuple"}) == {"type": "array"}
    assert normalize_bfcl_schema({"type": "any", "description": "opaque"}) == {
        "description": "opaque"
    }
    with pytest.raises(ValueError, match="unsupported BFCL schema type"):
        normalize_bfcl_schema({"type": "complex"})


def test_fetch_external_uses_declared_files_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "external.yaml"
    config.write_text(
        "bfcl:\n"
        "  dataset: bfcl/test\n"
        f"  revision: {'a' * 40}\n"
        "  license: Apache-2.0\n"
        "  files: [BFCL_v3_simple.json, BFCL_v3_irrelevance.json]\n"
        "agentabstain:\n"
        "  dataset: agent/test\n"
        f"  revision: {'b' * 40}\n"
        "  license: CC-BY-4.0\n"
        f"  code_revision: {'c' * 40}\n",
        encoding="utf-8",
    )
    module = ModuleType("huggingface_hub")

    def fake_snapshot_download(
        dataset: str,
        *,
        repo_type: str,
        revision: str,
        allow_patterns: list[str],
        local_dir: Path,
    ) -> None:
        assert repo_type == "dataset"
        assert revision in {"a" * 40, "b" * 40}
        local_dir.mkdir(parents=True, exist_ok=True)
        if dataset == "bfcl/test":
            for filename in allow_patterns:
                (local_dir / filename).write_text(f"{filename}\n")
        else:
            assert allow_patterns == ["tasks/**", "environments/**", "README.md"]
            (local_dir / "README.md").write_text("AgentAbstain\n")

    module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    output = tmp_path / "raw"
    first = fetch_external(config, output)
    first_bytes = (output / "fetch-manifest.json").read_bytes()
    second = fetch_external(config, output)
    assert first == second
    assert (output / "fetch-manifest.json").read_bytes() == first_bytes
    assert len(first["bfcl_files"]) == 2


@pytest.mark.parametrize(
    ("filename", "decision", "category"),
    [
        ("BFCL_v3_simple.json", ExternalDecision.CALL, "simple"),
        ("BFCL_v3_irrelevance.json", ExternalDecision.ABSTAIN, "irrelevance"),
    ],
)
def test_bfcl_parsing_is_strict_and_deterministic(
    tmp_path: Path,
    filename: str,
    decision: ExternalDecision,
    category: str,
) -> None:
    path = tmp_path / filename
    write_jsonl(path, [bfcl_row("row_2"), bfcl_row("row_1")])
    records = parse_bfcl_file(path, revision="a" * 40, license_name="Apache-2.0")
    assert [record.id for record in records] == ["bfcl-row-1", "bfcl-row-2"]
    assert records[0].expected_decision is decision
    assert records[0].category == category
    assert records[0].functions[0].parameters["type"] == "object"


def test_bfcl_parsing_rejects_bad_files_duplicates_and_shape(tmp_path: Path) -> None:
    unsupported = tmp_path / "other.json"
    write_jsonl(unsupported, [bfcl_row()])
    with pytest.raises(ValueError, match="unsupported BFCL file"):
        parse_bfcl_file(unsupported, revision="a" * 40, license_name="Apache-2.0")
    path = tmp_path / "BFCL_v3_simple.json"
    write_jsonl(path, [bfcl_row(), bfcl_row()])
    with pytest.raises(ValueError, match="duplicate BFCL id"):
        parse_bfcl_file(path, revision="a" * 40, license_name="Apache-2.0")
    malformed = bfcl_row()
    malformed["question"] = []
    write_jsonl(path, [malformed])
    with pytest.raises(ValueError, match="exactly one conversation"):
        parse_bfcl_file(path, revision="a" * 40, license_name="Apache-2.0")
    invalid = bfcl_row()
    invalid.pop("id")
    write_jsonl(path, [invalid])
    with pytest.raises(ValueError, match="requires id and function list"):
        parse_bfcl_file(path, revision="a" * 40, license_name="Apache-2.0")
    invalid = bfcl_row()
    invalid["function"] = ["not an object"]
    write_jsonl(path, [invalid])
    with pytest.raises(ValueError, match="function must be an object"):
        parse_bfcl_file(path, revision="a" * 40, license_name="Apache-2.0")


def test_near_duplicate_thresholds_and_quarantine() -> None:
    assert near_duplicate(" Hello, WORLD! ", "hello world")[:2] == (True, "exact")
    duplicate, method, _ = near_duplicate(
        "Find the current weather forecast for Chicago",
        "Find the current weather forecast for Chicago today",
    )
    assert duplicate and method in {"five_gram_jaccard", "sequence_match"}
    assert not near_duplicate("close ticket seven", "weather in paris")[0]
    record = external_record("external-one", ExternalDecision.CALL).model_copy(
        update={"messages": (ExternalMessage(role="user", content="Close ticket 7"),)}
    )
    internal = act_task().model_copy(update={"split": DatasetSplit.TEST})
    kept, report = quarantine_leakage([record], [internal])
    assert not kept
    assert report[0]["internal_split"] == "test"


def test_external_evaluation_covers_calls_abstention_and_malformed() -> None:
    records = [
        external_record("external-call", ExternalDecision.CALL),
        external_record("external-abstain", ExternalDecision.ABSTAIN),
    ]
    predictions = [
        PredictionRecord(
            task_id="external-call",
            raw_text='<tool_call>{"name":"lookup","arguments":{"value":"x"}}</tool_call>',
            latency_ms=1,
        ),
        PredictionRecord(
            task_id="external-abstain", raw_text="I can answer directly.", latency_ms=1
        ),
    ]
    evaluations, metrics = evaluate_external_records(records, predictions)
    assert all(item.correct for item in evaluations)
    assert metrics.balanced_accuracy == 1
    assert metrics.call_accuracy == 1
    assert metrics.abstain_accuracy == 1
    assert metrics.tool_call_rate == 0.5
    assert metrics.malformed_call_rate == 0

    malformed = predictions[0].model_copy(
        update={"raw_text": '<tool_call>{"name":</tool_call>'}
    )
    evaluations, metrics = evaluate_external_records(
        records, [malformed, predictions[1]]
    )
    assert evaluations[0].predicted_decision is ExternalDecision.CALL
    assert evaluations[0].correct
    assert not evaluations[0].protocol_correct
    assert metrics.malformed_call_rate == 0.5


def test_external_evaluation_rejects_ids_and_single_class() -> None:
    call = external_record("external-call", ExternalDecision.CALL)
    prediction = PredictionRecord(task_id=call.id, raw_text="direct", latency_ms=1)
    with pytest.raises(ValueError, match="CALL and ABSTAIN"):
        evaluate_external_records([call], [prediction])
    with pytest.raises(ValueError, match="must match"):
        evaluate_external_records([call], [])


def test_external_evaluation_persists_inference_failure() -> None:
    records = [
        external_record("external-call", ExternalDecision.CALL),
        external_record("external-abstain", ExternalDecision.ABSTAIN),
    ]
    failed = PredictionRecord(
        task_id="external-call",
        raw_text="",
        latency_ms=0,
        inference_error="timeout",
    )
    abstain = PredictionRecord(
        task_id="external-abstain", raw_text="No tool is relevant.", latency_ms=1
    )
    evaluations, metrics = evaluate_external_records(records, [failed, abstain])
    assert evaluations[0].predicted_decision is None
    assert evaluations[0].reason_code == "inference_error"
    assert metrics.call_accuracy == 0


def test_agentabstain_catalog_validates_pairs(tmp_path: Path) -> None:
    root = tmp_path / "agent"
    (root / "environments" / "calendar").mkdir(parents=True)
    pair = root / "tasks" / "ambiguous" / "preview-001"
    (pair / "act").mkdir(parents=True)
    (pair / "abstain").mkdir()
    (pair / "act" / "task.yaml").write_text("instruction: act\n")
    (pair / "abstain" / "task.yaml").write_text("instruction: abstain\n")
    catalog = catalog_agentabstain(root, revision="a" * 40, code_revision="b" * 40)
    assert catalog["pair_count"] == 1
    assert catalog["environment_count"] == 1
    incomplete = root / "tasks" / "ambiguous" / "preview-002"
    (incomplete / "act").mkdir(parents=True)
    with pytest.raises(ValueError, match="incomplete AgentAbstain pair"):
        catalog_agentabstain(root, revision="a" * 40, code_revision="b" * 40)
    (incomplete / "abstain").mkdir()
    with pytest.raises(ValueError, match=r"missing AgentAbstain task\.yaml"):
        catalog_agentabstain(root, revision="a" * 40, code_revision="b" * 40)
    for side in ("act", "abstain"):
        side_root = incomplete / side
        (side_root / "task.yaml").write_text("instruction: test\n")
        (side_root / "initial_states").mkdir()
        (side_root / "initial_states" / "unknown.json").write_text("{}\n")
    with pytest.raises(ValueError, match="unknown AgentAbstain environment"):
        catalog_agentabstain(root, revision="a" * 40, code_revision="b" * 40)


def test_prepare_external_writes_reproducible_manifests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    bfcl = raw / "bfcl"
    bfcl.mkdir(parents=True)
    write_jsonl(bfcl / "BFCL_v3_simple.json", [bfcl_row("simple_0")])
    irrelevant = bfcl_row("irrelevance_0")
    irrelevant["question"] = [[{"role": "user", "content": "Unrelated request."}]]
    write_jsonl(bfcl / "BFCL_v3_irrelevance.json", [irrelevant])
    agent = raw / "agentabstain"
    (agent / "environments" / "calendar").mkdir(parents=True)
    pair = agent / "tasks" / "scenario" / "preview-001"
    (pair / "act").mkdir(parents=True)
    (pair / "abstain").mkdir()
    (pair / "act" / "task.yaml").write_text("instruction: act\n")
    (pair / "abstain" / "task.yaml").write_text("instruction: abstain\n")

    internal = tmp_path / "internal"
    internal.mkdir()
    task = act_task()
    for split in ("train", "validation", "test"):
        split_task = task.model_copy(update={"split": DatasetSplit(split)})
        write_jsonl(internal / f"{split}.jsonl", [split_task.model_dump(mode="json")])
    config = tmp_path / "external.yaml"
    config.write_text(
        "bfcl:\n"
        "  dataset: bfcl/test\n"
        f"  revision: {'a' * 40}\n"
        "  license: Apache-2.0\n"
        "  files: [BFCL_v3_simple.json, BFCL_v3_irrelevance.json]\n"
        "agentabstain:\n"
        "  dataset: agent/test\n"
        f"  revision: {'b' * 40}\n"
        "  license: CC-BY-4.0\n"
        f"  code_revision: {'c' * 40}\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = prepare_external(config, raw, internal, first)
    prepare_external(config, raw, internal, second)
    assert manifest["source_count"] == 2
    assert manifest["prepared_count"] == 2
    assert manifest["agentabstain"]["pair_count"] == 1
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / "DATASET_CARD.md").is_file()
