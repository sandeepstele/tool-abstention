"""Tests for deterministic schema export and record validation."""

import json
from pathlib import Path

from tool_abstention.records import TaskRecord
from tool_abstention.schemas import SCHEMA_MODELS, export_schemas, validate_record
from tool_abstention.util.hashing import sha256_file

from .test_records import act_task

EXPECTED_SCHEMA_HASHES = {
    "evaluation": "831ae1069c11d590534ff75a6bca7f91f5d61ccd349ea51b0de19970b322162d",
    "pair": "43a8ff104809562b2b312e6ed399931795f3c27f616ef7c4f62b031be661c104",
    "prediction": "88bf10153139ad3d208a9abeb6df029de34e0fd288c1cfdbac69eb11e826a056",
    "task": "d314fdeac154547b37185f6b7125a15e42c66d5630b9868c940c57d5789c6049",
}


def test_schema_export_is_byte_identical_and_valid_json(tmp_path: Path) -> None:
    first = export_schemas(tmp_path / "first")
    second = export_schemas(tmp_path / "second")
    assert set(first) == set(SCHEMA_MODELS)
    for name in SCHEMA_MODELS:
        assert first[name].read_bytes() == second[name].read_bytes()
        assert sha256_file(first[name]) == sha256_file(second[name])
        assert sha256_file(first[name]) == EXPECTED_SCHEMA_HASHES[name]
        assert isinstance(json.loads(first[name].read_text(encoding="utf-8")), dict)


def test_validate_record_returns_requested_model(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(act_task().model_dump_json(), encoding="utf-8")
    validated = validate_record(path, "task")
    assert isinstance(validated, TaskRecord)
