"""Canonical serialization and SHA-256 helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CHUNK_SIZE = 1024 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value to stable UTF-8 bytes."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def sha256_object(value: Any) -> str:
    """Return the lowercase SHA-256 digest of a JSON-compatible value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Hash a file incrementally without loading it all into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
