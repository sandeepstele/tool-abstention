"""Deterministic, offline consolidation of stored experiment results."""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tool_abstention.util.hashing import canonical_json_bytes, sha256_file
from tool_abstention.util.jsonl import read_jsonl

T_CRITICAL_DF2 = 4.3026527299


class AnalysisModel(BaseModel):
    """Strict immutable analysis configuration base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentSpec(AnalysisModel):
    """One stored evaluation included in the final comparison."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    family: Literal["1.5b", "0.5b"]
    method: str = Field(min_length=1)
    split: Literal["internal-validation", "protocol-stress", "bfcl"]
    metrics: Path
    evaluations: Path
    status: Literal["baseline", "selected", "rejected"]

    @model_validator(mode="after")
    def prohibit_held_out_test(self) -> ExperimentSpec:
        for path in (self.metrics, self.evaluations):
            if "test" in {part.casefold() for part in path.parts}:
                raise ValueError("held-out test artifacts are prohibited")
        return self


class PairedComparisonSpec(AnalysisModel):
    """A declared paired accuracy comparison."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    baseline: str
    candidate: str


class FinalAnalysisConfig(AnalysisModel):
    """Complete source declaration for a final report build."""

    schema_version: Literal[1] = 1
    bootstrap_seed: int = 2027
    bootstrap_samples: int = Field(default=10_000, ge=100)
    experiments: tuple[ExperimentSpec, ...] = Field(min_length=1)
    paired_comparisons: tuple[PairedComparisonSpec, ...] = ()

    @model_validator(mode="after")
    def unique_and_referenced(self) -> FinalAnalysisConfig:
        ids = [item.id for item in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment IDs must be unique")
        known = set(ids)
        for comparison in self.paired_comparisons:
            if comparison.baseline not in known or comparison.candidate not in known:
                raise ValueError(f"unknown comparison experiment in {comparison.id}")
            by_id = {item.id: item for item in self.experiments}
            if by_id[comparison.baseline].split != by_id[comparison.candidate].split:
                raise ValueError(f"comparison {comparison.id} mixes evaluation splits")
        return self


def mean_ci95(values: Sequence[float]) -> dict[str, float | int]:
    """Return mean, sample SD, and a t interval for repeated seeds."""
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("mean CI requires at least two finite values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sample_std = math.sqrt(variance)
    half_width = T_CRITICAL_DF2 * sample_std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def paired_bootstrap_ci(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the paired candidate-minus-baseline accuracy difference."""
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired evaluations must have identical non-empty task IDs")
    ids = sorted(baseline)
    differences = [float(candidate[item]) - float(baseline[item]) for item in ids]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    estimates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in ids) / len(ids)
        for _ in range(samples)
    )
    low = estimates[math.floor(0.025 * (samples - 1))]
    high = estimates[math.ceil(0.975 * (samples - 1))]
    return {
        "task_count": len(ids),
        "samples": samples,
        "seed": seed,
        "difference": observed,
        "ci95_low": low,
        "ci95_high": high,
    }


def _load_yaml(path: Path) -> FinalAnalysisConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("analysis config must be a YAML mapping")
    return FinalAnalysisConfig.model_validate(value)


def _load_metrics(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metrics must be an object: {path}")
    return value


def _accuracy_rows(path: Path) -> dict[str, bool]:
    rows: dict[str, bool] = {}
    for value in read_jsonl(path):
        task_id = value.get("task_id")
        correct = value.get("correct")
        if not isinstance(task_id, str) or not isinstance(correct, bool):
            raise ValueError(f"invalid evaluation row in {path}")
        if task_id in rows:
            raise ValueError(f"duplicate evaluation task ID {task_id}")
        rows[task_id] = correct
    return rows


def _metric(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _required_metric(metrics: dict[str, object], key: str) -> float:
    value = _metric(metrics, key)
    if value is None:
        raise ValueError(f"missing numeric metric {key}")
    return value


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    fields = [
        "id",
        "family",
        "method",
        "split",
        "status",
        "task_count",
        "accuracy",
        "act_accuracy",
        "abstention_accuracy",
        "balanced_accuracy",
        "call_accuracy",
        "malformed_call_rate",
        "protocol_compliance_rate",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    return stream.getvalue().encode("utf-8")


def _comparison_markdown(rows: list[dict[str, object]]) -> bytes:
    lines = [
        "# Canonical experiment comparison",
        "",
        "Generated offline from committed metrics and per-example evaluations.",
        "",
        "| Experiment | Model | Split | Status | Accuracy | Act/CALL | "
        "Abstain | Protocol |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        accuracy = row.get("accuracy", "")
        act = row.get("act_accuracy", row.get("call_accuracy", ""))
        abstain = row.get("abstention_accuracy", "")
        protocol = row.get("protocol_compliance_rate", "")

        def render(value: object) -> str:
            if isinstance(value, int | float):
                return f"{100 * float(value):.2f}%"
            return "—"

        lines.append(
            f"| {row['method']} | {row['family']} | {row['split']} | {row['status']} "
            f"| {render(accuracy)} | {render(act)} | {render(abstain)} "
            f"| {render(protocol)} |"
        )
    lines.extend(
        [
            "",
            "Rejected experiments are retained as negative results, "
            "not promoted models.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _svg(rows: list[dict[str, object]], family: str) -> bytes:
    selected = [row for row in rows if row["family"] == family and "accuracy" in row]
    width, height = 900, 90 + 54 * len(selected)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="20">'
        f"{family} stored evaluation accuracy</text>",
    ]
    for index, row in enumerate(selected):
        y = 58 + index * 54
        raw_value = row["accuracy"]
        if not isinstance(raw_value, int | float):
            raise ValueError("chart accuracy must be numeric")
        value = float(raw_value)
        parts.extend(
            [
                f'<text x="20" y="{y + 17}" font-family="sans-serif" '
                f'font-size="13">{row["id"]}</text>',
                f'<rect x="280" y="{y}" width="{560 * value:.3f}" '
                'height="22" fill="#2563eb"/>',
                f'<text x="850" y="{y + 17}" text-anchor="end" '
                f'font-family="sans-serif" font-size="13">'
                f"{100 * value:.2f}%</text>",
            ]
        )
    parts.append("</svg>")
    return ("\n".join(parts) + "\n").encode("utf-8")


def build_final_analysis(config_path: Path, output: Path) -> dict[str, object]:
    """Build deterministic final artifacts without opening predictions or test data."""
    config = _load_yaml(config_path)
    rows: list[dict[str, object]] = []
    evaluations: dict[str, dict[str, bool]] = {}
    inputs: list[dict[str, str]] = []
    metrics_by_id: dict[str, dict[str, object]] = {}
    for spec in config.experiments:
        if not spec.metrics.is_file() or not spec.evaluations.is_file():
            raise ValueError(f"missing declared artifact for {spec.id}")
        metrics = _load_metrics(spec.metrics)
        metrics_by_id[spec.id] = metrics
        evaluations[spec.id] = _accuracy_rows(spec.evaluations)
        inputs.extend(
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in (spec.metrics, spec.evaluations)
        )
        accuracy_key = "decision_accuracy" if spec.split == "bfcl" else "accuracy"
        row: dict[str, object] = {
            "id": spec.id,
            "family": spec.family,
            "method": spec.method,
            "split": spec.split,
            "status": spec.status,
            "task_count": metrics.get("task_count", len(evaluations[spec.id])),
            "accuracy": _metric(metrics, accuracy_key),
        }
        for key in (
            "act_accuracy",
            "abstention_accuracy",
            "balanced_accuracy",
            "call_accuracy",
            "malformed_call_rate",
            "protocol_compliance_rate",
        ):
            value = _metric(metrics, key)
            if value is not None:
                row[key] = value
        external_abstain = _metric(metrics, "abstain_accuracy")
        if external_abstain is not None:
            row["abstention_accuracy"] = external_abstain
        rows.append(row)

    seed_ids = [f"sft-seed-{seed}-internal" for seed in range(3)]
    seed_stats: dict[str, object] = {}
    for key in ("accuracy", "act_accuracy", "abstention_accuracy", "paired_accuracy"):
        seed_stats[key] = mean_ci95(
            [_required_metric(metrics_by_id[item], key) for item in seed_ids]
        )
    comparisons = {
        spec.id: paired_bootstrap_ci(
            evaluations[spec.baseline],
            evaluations[spec.candidate],
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + index,
        )
        for index, spec in enumerate(config.paired_comparisons)
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "scope": "offline-stored-validation-and-bfcl-only",
        "held_out_test_opened": False,
        "external_data_used_for_training": False,
        "inputs": sorted(inputs, key=lambda item: item["path"]),
        "sft_1.5b_three_seed_statistics": seed_stats,
        "paired_bootstrap": comparisons,
        "selected_model": "1.5B SFT three-seed baseline",
        "preference_branch": "closed-after-behavior-gate-failures",
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary.json": canonical_json_bytes(summary) + b"\n",
        "experiments.csv": _csv_bytes(rows),
        "comparison.md": _comparison_markdown(rows),
        "accuracy-1.5b.svg": _svg(rows, "1.5b"),
        "accuracy-0.5b.svg": _svg(rows, "0.5b"),
    }
    for name, content in artifacts.items():
        (output / name).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "config_sha256": sha256_file(config_path),
        "artifacts": {name: sha256_file(output / name) for name in sorted(artifacts)},
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def audit_final_release(root: Path) -> dict[str, object]:
    """Verify committed release artifacts and their complete hash chain."""
    required = (
        "LICENSE",
        "README.md",
        "uv.lock",
        ".github/workflows/ci.yml",
        "configs/analysis/final.yaml",
        "assets/architecture.svg",
        "assets/result-summary.svg",
        "docs/26-final-analysis.md",
        "docs/engineering-article.md",
        "reports/final/manifest.json",
        "reports/final/summary.json",
    )
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        raise ValueError(f"missing release files: {', '.join(missing)}")

    manifest = _load_metrics(root / "reports/final/manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("release manifest has no artifacts")
    verified_artifacts = 0
    for name, expected_hash in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValueError("release manifest artifact entries must be strings")
        path = root / "reports/final" / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"release artifact hash mismatch: {name}")
        verified_artifacts += 1
    expected_config_hash = manifest.get("config_sha256")
    if expected_config_hash != sha256_file(root / "configs/analysis/final.yaml"):
        raise ValueError("analysis config hash mismatch")

    summary = _load_metrics(root / "reports/final/summary.json")
    if summary.get("held_out_test_opened") is not False:
        raise ValueError("release must attest that the held-out test stayed sealed")
    if summary.get("external_data_used_for_training") is not False:
        raise ValueError("release must prohibit external benchmark training")
    inputs = summary.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("release summary has no input provenance")
    verified_inputs = 0
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("invalid release input entry")
        relative = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("invalid release input fields")
        path = Path(relative)
        if path.is_absolute() or "test" in {part.casefold() for part in path.parts}:
            raise ValueError(f"prohibited release input path: {relative}")
        if not (root / path).is_file() or sha256_file(root / path) != expected_hash:
            raise ValueError(f"release input hash mismatch: {relative}")
        verified_inputs += 1
    return {
        "status": "pass",
        "verified_artifacts": verified_artifacts,
        "verified_inputs": verified_inputs,
        "held_out_test_opened": False,
        "external_data_used_for_training": False,
    }
