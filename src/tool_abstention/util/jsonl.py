"""Validated UTF-8 JSON Lines input and output."""

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from tool_abstention.util.hashing import canonical_json_bytes


class JsonlDecodeError(ValueError):
    """A JSONL record could not be decoded or was not an object."""

    def __init__(self, path: Path, line_number: int, detail: str) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = detail
        super().__init__(f"{path}:{line_number}: {detail}")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield object records from a UTF-8 JSONL file.

    Blank lines are ignored. Invalid JSON and non-object values include the path
    and one-based line number in the raised error.
    """
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise JsonlDecodeError(path, line_number, error.msg) from error
            if not isinstance(value, dict):
                raise JsonlDecodeError(path, line_number, "record must be an object")
            yield value


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records as canonical UTF-8 JSONL, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json_bytes(record))
            stream.write(b"\n")
