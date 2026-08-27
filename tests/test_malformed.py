"""Tests for deterministic external malformed-call analysis."""

from pathlib import Path

import pytest

from tool_abstention.malformed import analyze_malformed_calls, malformed_category
from tool_abstention.records import PredictionRecord
from tool_abstention.util.jsonl import write_jsonl


@pytest.mark.parametrize(
    ("raw", "tokens", "category"),
    [
        ('<tool_call>{"arguments":{}}</tool_call>', 10, "missing_name"),
        (
            '<tool_call>{"arguments":{"name":"lookup"}}</tool_call>',
            10,
            "name_inside_arguments",
        ),
        ('<tool_call>{"name":"x"}</tool_call>', 10, "missing_arguments"),
        ('<tool_call>{"name":</tool_call>', 10, "truncated_json"),
        ('<tool_call>{"name"="x","arguments":{}}</tool_call>', 10, "invalid_separator"),
        (
            '<tool_call>{"name":"x","arguments":{"v":true.0}}</tool_call>',
            10,
            "invalid_literal",
        ),
        ("prefix <tool_call>", 256, "max_token_truncation"),
        ('prose {"arguments":{}}', 10, "prose_or_wrapper"),
        (
            '<tool_call>{"name":"x","arguments":{"v":{"a","b"}}}</tool_call>',
            10,
            "set_literal",
        ),
        (
            '<tool_call>{"name":"x","arguments":{},}</tool_call>',
            10,
            "other_json_syntax",
        ),
        ('<tool_call>{"name":1,"arguments":{}}</tool_call>', 10, "invalid_structure"),
    ],
)
def test_malformed_categories(raw: str, tokens: int, category: str) -> None:
    prediction = PredictionRecord(
        task_id="external-one", raw_text=raw, latency_ms=1, output_tokens=tokens
    )
    assert malformed_category(prediction) == category


def test_analysis_is_deterministic_and_reports_transitions(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    sft_path = tmp_path / "sft.jsonl"
    base = PredictionRecord(
        task_id="task-one",
        raw_text='<tool_call>{"arguments":{}}</tool_call>',
        latency_ms=1,
    )
    resolved = base.model_copy(
        update={"raw_text": '<tool_call>{"name":"x","arguments":{}}</tool_call>'}
    )
    new = PredictionRecord(
        task_id="task-two", raw_text='<tool_call>{"name":</tool_call>', latency_ms=1
    )
    write_jsonl(base_path, [base.model_dump(mode="json")])
    write_jsonl(
        sft_path, [resolved.model_dump(mode="json"), new.model_dump(mode="json")]
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    report = analyze_malformed_calls(base_path, sft_path, first)
    analyze_malformed_calls(base_path, sft_path, second)
    assert report["new_after_sft_ids"] == ["task-two"]
    assert report["resolved_after_sft_ids"] == ["task-one"]
    assert first.read_bytes() == second.read_bytes()
