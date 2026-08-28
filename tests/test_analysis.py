"""Tests for deterministic final analysis."""

import json
from pathlib import Path

import pytest
import yaml

from tool_abstention.analysis import (
    FinalAnalysisConfig,
    audit_final_release,
    build_final_analysis,
    mean_ci95,
    paired_bootstrap_ci,
)
from tool_abstention.util.hashing import sha256_file
from tool_abstention.util.jsonl import write_jsonl


def test_mean_ci95_fixed_vector() -> None:
    result = mean_ci95([0.9, 0.8, 1.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(0.9)
    assert result["sample_std"] == pytest.approx(0.1)
    assert result["ci95_low"] == pytest.approx(0.651586, abs=1e-6)
    assert result["ci95_high"] == pytest.approx(1.148414, abs=1e-6)
    with pytest.raises(ValueError, match="two finite"):
        mean_ci95([1.0])
    with pytest.raises(ValueError, match="two finite"):
        mean_ci95([1.0, float("nan")])


def test_paired_bootstrap_is_deterministic_and_validates_ids() -> None:
    baseline = {"a": False, "b": True, "c": False, "d": True}
    candidate = {"a": True, "b": True, "c": True, "d": False}
    first = paired_bootstrap_ci(baseline, candidate, samples=200, seed=7)
    assert first == paired_bootstrap_ci(baseline, candidate, samples=200, seed=7)
    assert first["difference"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap_ci(baseline, {"a": True}, samples=100, seed=0)


def _write_experiment(root: Path, name: str, values: list[bool]) -> tuple[Path, Path]:
    metrics = root / f"{name}-metrics.json"
    evaluations = root / f"{name}-evaluations.jsonl"
    accuracy = sum(values) / len(values)
    metrics.write_text(
        f'{{"accuracy":{accuracy},"act_accuracy":{accuracy},'
        f'"abstention_accuracy":{accuracy},"paired_accuracy":{accuracy},'
        f'"task_count":{len(values)}}}\n',
        encoding="utf-8",
    )
    write_jsonl(
        evaluations,
        [
            {"task_id": f"task-{index}", "correct": value}
            for index, value in enumerate(values)
        ],
    )
    return metrics, evaluations


def _config(tmp_path: Path) -> Path:
    experiments: list[dict[str, object]] = []
    for seed, values in enumerate(([True, False], [True, True], [False, True])):
        metrics, evaluations = _write_experiment(tmp_path, f"seed-{seed}", values)
        experiments.append(
            {
                "id": f"sft-seed-{seed}-internal",
                "family": "1.5b",
                "method": f"SFT seed {seed}",
                "split": "internal-validation",
                "metrics": str(metrics),
                "evaluations": str(evaluations),
                "status": "selected",
            }
        )
    path = tmp_path / "analysis.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "bootstrap_seed": 3,
                "bootstrap_samples": 100,
                "experiments": experiments,
                "paired_comparisons": [
                    {
                        "id": "seed1-minus-seed0",
                        "baseline": "sft-seed-0-internal",
                        "candidate": "sft-seed-1-internal",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_final_analysis_is_byte_identical(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_final_analysis(config, first)
    assert manifest == build_final_analysis(config, second)
    assert sorted(path.name for path in first.iterdir()) == [
        "accuracy-0.5b.svg",
        "accuracy-1.5b.svg",
        "comparison.md",
        "experiments.csv",
        "manifest.json",
        "summary.json",
    ]
    for path in first.iterdir():
        assert sha256_file(path) == sha256_file(second / path.name)
    assert b'held_out_test_opened":false' in (first / "summary.json").read_bytes()


def test_analysis_rejects_test_paths_and_bad_references(tmp_path: Path) -> None:
    base = {
        "id": "one",
        "family": "1.5b",
        "method": "method",
        "split": "internal-validation",
        "metrics": "results/test/metrics.json",
        "evaluations": "evaluations.jsonl",
        "status": "baseline",
    }
    with pytest.raises(ValueError, match="held-out test"):
        FinalAnalysisConfig.model_validate({"schema_version": 1, "experiments": [base]})
    base["metrics"] = "metrics.json"
    with pytest.raises(ValueError, match="unknown comparison"):
        FinalAnalysisConfig.model_validate(
            {
                "schema_version": 1,
                "experiments": [base],
                "paired_comparisons": [
                    {"id": "bad", "baseline": "one", "candidate": "missing"}
                ],
            }
        )


def test_analysis_reports_missing_and_duplicate_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    content = config.read_text(encoding="utf-8").replace(
        str(tmp_path / "seed-0-metrics.json"), str(tmp_path / "missing.json")
    )
    config.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="missing declared"):
        build_final_analysis(config, tmp_path / "output")

    config = _config(tmp_path)
    evaluation = tmp_path / "seed-0-evaluations.jsonl"
    write_jsonl(
        evaluation,
        [{"task_id": "same", "correct": True}, {"task_id": "same", "correct": False}],
    )
    with pytest.raises(ValueError, match="duplicate evaluation"):
        build_final_analysis(config, tmp_path / "output")


def test_release_audit_verifies_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in (
        "LICENSE",
        "README.md",
        "uv.lock",
        ".github/workflows/ci.yml",
        "assets/architecture.svg",
        "assets/result-summary.svg",
        "docs/26-final-analysis.md",
        "docs/engineering-article.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("release\n", encoding="utf-8")
    config = _config(tmp_path)
    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert isinstance(config_value, dict)
    experiments = config_value["experiments"]
    assert isinstance(experiments, list)
    for experiment in experiments:
        assert isinstance(experiment, dict)
        experiment["metrics"] = Path(str(experiment["metrics"])).name
        experiment["evaluations"] = Path(str(experiment["evaluations"])).name
    canonical_config = tmp_path / "configs/analysis/final.yaml"
    canonical_config.parent.mkdir(parents=True)
    canonical_config.write_text(
        yaml.safe_dump(config_value, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    build_final_analysis(canonical_config, tmp_path / "reports/final")
    report = audit_final_release(tmp_path)
    assert report["status"] == "pass"
    assert report["verified_artifacts"] == 5
    assert report["verified_inputs"] == 6
    (tmp_path / "reports/final/comparison.md").write_text("tampered\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        audit_final_release(tmp_path)

    build_final_analysis(canonical_config, tmp_path / "reports/final")
    manifest_path = tmp_path / "reports/final/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no artifacts"):
        audit_final_release(tmp_path)

    build_final_analysis(canonical_config, tmp_path / "reports/final")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="config hash mismatch"):
        audit_final_release(tmp_path)


@pytest.mark.parametrize(
    ("summary_update", "message"),
    [
        ({"held_out_test_opened": True}, "test stayed sealed"),
        ({"external_data_used_for_training": True}, "external benchmark"),
        ({"inputs": []}, "no input provenance"),
        ({"inputs": ["bad"]}, "invalid release input entry"),
        ({"inputs": [{"path": 1, "sha256": 2}]}, "invalid release input fields"),
        (
            {"inputs": [{"path": "results/test/data.json", "sha256": "0" * 64}]},
            "prohibited release input path",
        ),
        (
            {"inputs": [{"path": "missing.json", "sha256": "0" * 64}]},
            "release input hash mismatch",
        ),
    ],
)
def test_release_audit_rejects_invalid_provenance(
    tmp_path: Path, summary_update: dict[str, object], message: str
) -> None:
    source_root = Path.cwd()
    for relative in (
        "LICENSE",
        "README.md",
        "uv.lock",
        ".github/workflows/ci.yml",
        "assets/architecture.svg",
        "assets/result-summary.svg",
        "docs/26-final-analysis.md",
        "docs/engineering-article.md",
        "configs/analysis/final.yaml",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
    report_dir = tmp_path / "reports/final"
    report_dir.mkdir(parents=True)
    for path in (source_root / "reports/final").iterdir():
        (report_dir / path.name).write_bytes(path.read_bytes())
    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(summary_update)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["summary.json"] = sha256_file(summary_path)
    manifest["config_sha256"] = sha256_file(tmp_path / "configs/analysis/final.yaml")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        audit_final_release(tmp_path)


def test_release_audit_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing release files"):
        audit_final_release(tmp_path)
