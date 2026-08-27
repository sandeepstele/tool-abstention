"""Deterministic analysis of malformed external tool-call attempts."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tool_abstention.evaluator import looks_like_tool_call
from tool_abstention.external import EXTERNAL_TOOL_BLOCK, valid_external_tool_call
from tool_abstention.records import PredictionRecord
from tool_abstention.util.hashing import canonical_json_bytes, sha256_file
from tool_abstention.util.jsonl import read_jsonl


def malformed_category(
    prediction: PredictionRecord, *, max_tokens: int = 256
) -> str | None:
    """Return one mutually exclusive syntax-failure category for a call attempt."""
    raw = prediction.raw_text.strip()
    if valid_external_tool_call(raw) or not looks_like_tool_call(raw):
        return None
    if prediction.output_tokens is not None and prediction.output_tokens >= max_tokens:
        return "max_token_truncation"
    match = EXTERNAL_TOOL_BLOCK.fullmatch(raw)
    text = match.group(1) if match else raw
    if not text.startswith("{"):
        return "prose_or_wrapper"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if text.count("{") > text.count("}"):
            return "truncated_json"
        if re.search(r"\b(?:true|false|null)\.\d+", text):
            return "invalid_literal"
        if re.search(r'"[^"\\]+"\s*=', text):
            return "invalid_separator"
        if re.search(r'\{\s*"[^"\\]+"\s*,\s*"', text):
            return "set_literal"
        return "other_json_syntax"
    arguments = payload.get("arguments")
    if isinstance(arguments, dict) and "name" in arguments and "name" not in payload:
        return "name_inside_arguments"
    if "name" not in payload:
        return "missing_name"
    if "arguments" not in payload:
        return "missing_arguments"
    return "invalid_structure"


def _classified(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in read_jsonl(path):
        prediction = PredictionRecord.model_validate(value)
        category = malformed_category(prediction)
        if category is not None:
            result[prediction.task_id] = category
    return result


def analyze_malformed_calls(
    base_path: Path, sft_path: Path, output: Path
) -> dict[str, Any]:
    """Compare stored base and SFT malformed attempts without model inference."""
    base = _classified(base_path)
    sft = _classified(sft_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "base_predictions_hash": sha256_file(base_path),
        "sft_predictions_hash": sha256_file(sft_path),
        "base_malformed_count": len(base),
        "sft_malformed_count": len(sft),
        "new_after_sft_count": len(set(sft) - set(base)),
        "resolved_after_sft_count": len(set(base) - set(sft)),
        "persistent_count": len(set(base) & set(sft)),
        "base_categories": dict(sorted(Counter(base.values()).items())),
        "sft_categories": dict(sorted(Counter(sft.values()).items())),
        "new_after_sft_ids": sorted(set(sft) - set(base)),
        "resolved_after_sft_ids": sorted(set(base) - set(sft)),
        "persistent_ids": sorted(set(base) & set(sft)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    return report
