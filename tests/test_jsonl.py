"""Tests for JSON Lines input and output."""

from pathlib import Path
from typing import Any

import pytest

from tool_abstention.util.jsonl import JsonlDecodeError, read_jsonl, write_jsonl


def test_unicode_round_trip_is_canonical(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    records: list[dict[str, Any]] = [
        {"text": "héllo", "id": 1},
        {"id": 2, "active": True},
    ]
    write_jsonl(path, records)
    assert path.read_bytes() == (
        b'{"id":1,"text":"h\xc3\xa9llo"}\n{"active":true,"id":2}\n'
    )
    assert list(read_jsonl(path)) == records


def test_empty_file_has_no_records(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    assert list(read_jsonl(path)) == []


def test_malformed_json_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot json\n', encoding="utf-8")
    with pytest.raises(JsonlDecodeError) as error:
        list(read_jsonl(path))
    assert error.value.path == path
    assert error.value.line_number == 2
    assert str(error.value).startswith(f"{path}:2:")


def test_non_object_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "array.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(JsonlDecodeError, match="record must be an object") as error:
        list(read_jsonl(path))
    assert error.value.detail == "record must be an object"
