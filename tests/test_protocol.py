"""Tests for internal-only protocol stress and repair data."""

import json
from pathlib import Path

import pytest

from tool_abstention.cli import main
from tool_abstention.protocol import (
    ProtocolStressConfig,
    build_protocol_repair_sft,
    build_protocol_stress,
    generate_protocol_pairs,
)
from tool_abstention.records import CallExpected
from tool_abstention.sft import build_sft_dataset
from tool_abstention.taxonomy import DatasetSplit, DecisionClass
from tool_abstention.util.jsonl import read_jsonl, write_jsonl

from .test_records import abstain_task, act_task


def config() -> ProtocolStressConfig:
    return ProtocolStressConfig(
        seed=0, train_pairs=4, validation_pairs=2, generator_version="1.0.0"
    )


def test_protocol_pairs_are_typed_complex_and_internal() -> None:
    pairs = generate_protocol_pairs(config())
    assert len(pairs) == 6
    assert all(pair.act.label is DecisionClass.CALL for pair in pairs)
    assert all(pair.abstain.label is DecisionClass.CLARIFY for pair in pairs)
    assert all(pair.act.environment["source"] == "internal_generated" for pair in pairs)
    assert all(isinstance(pair.act.expected, CallExpected) for pair in pairs)
    assert any(
        isinstance(pair.act.expected, CallExpected)
        and isinstance(pair.act.expected.arguments.get("order"), dict)
        for pair in pairs
    )
    assert sum(pair.act.split is DatasetSplit.TRAIN for pair in pairs) == 4


def test_protocol_export_is_byte_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 0\ntrain_pairs: 4\nvalidation_pairs: 2\ngenerator_version: 1.0.0\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_protocol_stress(config_path, first)
    build_protocol_stress(config_path, second)
    assert manifest["external_sources"] == []
    assert manifest["test_consumed"] is False
    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_protocol_cli_builds_stress_and_repair_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _base_sft(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 0\ntrain_pairs: 1\nvalidation_pairs: 1\ngenerator_version: 1.0.0\n",
        encoding="utf-8",
    )
    stress = tmp_path / "stress"
    main(
        [
            "generate-protocol-stress",
            "--config",
            str(config_path),
            "--output",
            str(stress),
        ]
    )
    repair = tmp_path / "repair"
    main(
        [
            "build-protocol-repair-sft",
            "--base-sft",
            str(base),
            "--stress",
            str(stress),
            "--output",
            str(repair),
        ]
    )
    assert (repair / "manifest.json").is_file()
    assert '"external_sources": []' in capsys.readouterr().out


def _base_sft(tmp_path: Path) -> Path:
    internal = tmp_path / "internal"
    internal.mkdir()
    train = act_task().model_copy(update={"split": DatasetSplit.TRAIN})
    valid = abstain_task(DecisionClass.ANSWER).model_copy(
        update={"split": DatasetSplit.VALIDATION}
    )
    write_jsonl(internal / "train.jsonl", [train.model_dump(mode="json")])
    write_jsonl(internal / "validation.jsonl", [valid.model_dump(mode="json")])
    base = tmp_path / "base"
    build_sft_dataset(internal, base)
    return base


def test_repair_sft_appends_stress_and_records_no_external_sources(
    tmp_path: Path,
) -> None:
    base = _base_sft(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 0\ntrain_pairs: 1\nvalidation_pairs: 1\ngenerator_version: 1.0.0\n",
        encoding="utf-8",
    )
    stress = tmp_path / "stress"
    build_protocol_stress(config_path, stress)
    output = tmp_path / "repair"
    manifest = build_protocol_repair_sft(base, stress, output)
    assert manifest["train_count"] == 3
    assert manifest["validation_count"] == 3
    assert manifest["external_sources"] == []
    assert len(list(read_jsonl(output / "train.jsonl"))) == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_kind", "external", "internal_generated"),
        ("external_sources", ["bfcl"], "external benchmark"),
        ("test_consumed", True, "test_consumed=false"),
    ],
)
def test_repair_sft_rejects_contaminated_provenance(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    base = _base_sft(tmp_path)
    stress = tmp_path / "stress"
    stress.mkdir()
    manifest = {
        "source_kind": "internal_generated",
        "external_sources": [],
        "test_consumed": False,
    }
    manifest[field] = value
    (stress / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_protocol_repair_sft(base, stress, tmp_path / "output")
