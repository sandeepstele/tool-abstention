"""Tests for canonical serialization and hashing."""

from pathlib import Path

import pytest

from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)


def test_canonical_json_and_object_hash_fixed_vector() -> None:
    value = {"b": "é", "a": 1}
    assert canonical_json_bytes(value) == b'{"a":1,"b":"\xc3\xa9"}'
    assert sha256_object(value) == (
        "09ad9fd2fb648cb2f62141215828ea00a62c299db05d20aa9ade2f527a301cc6"
    )


def test_non_finite_number_is_rejected() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"invalid": float("nan")})


def test_streaming_file_hash_fixed_vector(tmp_path: Path) -> None:
    path = tmp_path / "content.txt"
    path.write_bytes(b"hello\n")
    assert sha256_file(path, chunk_size=2) == (
        "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    )


def test_file_hash_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(tmp_path / "unused", chunk_size=0)
