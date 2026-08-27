# tool-abstention

**Teaching small language models *when not to call a tool* — a reproducible post-training + evaluation pipeline for agentic abstention.**

> Most tool-use benchmarks (BFCL, τ-bench) measure whether a model calls the *correct* tool with the *correct* arguments. This project measures and trains the opposite half of the problem: recognizing when **no tool should be called at all** — because the answer is already in the prompt/parametric knowledge, because a required parameter is missing, or because no available tool can help.

## Why this exists

The "fine-tune a small model to call tools with LoRA + DPO/GRPO" project is now crowded (ToolBrain, RLFactory, Hugging Face TRL, and a dozen tutorials ship it). But a cluster of very recent papers (2025–2026) has identified a live, under-explored failure mode: **agents over-call and hallucinate tools**, and abstention is a distinct capability that scales *independently* of task-solving ability. Those papers release benchmarks and data but **almost none release a clean, reproducible small-model training pipeline.**

This project fills that gap:

- an **extended abstention taxonomy** (act vs. four ways to abstain),
- a **paired-task benchmark** (should-act vs. should-abstain),
- a **training pipeline** (SFT → DPO / ORPO / RPO, LoRA) runnable on a 24 GB Apple-Silicon laptop,
- a **verifiable evaluation harness** (no LLM-as-judge required for the core metrics).

## Core hypothesis

> Preference optimization with explicit negative examples (wrong tool call, fabricated tool, missed follow-up) can teach a ≤1.5B model to abstain correctly **without degrading** its tool-call accuracy — and we can measure that trade-off precisely.

## Documentation

All planning lives in [`docs/`](docs/). Read in order:

| # | Doc | Answers |
|---|-----|---------|
| 01 | [vision.md](docs/01-vision.md) | What, why, non-goals, success criteria |
| 02 | [literature-review.md](docs/02-literature-review.md) | What exists, where the gap is, citations |
| 03 | [research-questions.md](docs/03-research-questions.md) | Hypotheses + the research/paper angle |
| 04 | [architecture.md](docs/04-architecture.md) | Repo layout, components, data flow |
| 05 | [data-plan.md](docs/05-data-plan.md) | Taxonomy, dataset construction, sources |
| 06 | [training-plan.md](docs/06-training-plan.md) | Stages, models, hyperparameters, MLX |
| 07 | [evaluation-plan.md](docs/07-evaluation-plan.md) | Metrics, harness, baselines, testing |
| 08 | [roadmap.md](docs/08-roadmap.md) | Phases, milestones, exit criteria |
| 09 | [experiment-tracking.md](docs/09-experiment-tracking.md) | Reproducibility, configs, logging |
| 10 | [paper-plan.md](docs/10-paper-plan.md) | Research contribution + target venue |
| 11 | [implementation-plan.md](docs/11-implementation-plan.md) | Build order, interfaces, verification, deliverables |
| 12 | [baseline-diagnostics.md](docs/12-baseline-diagnostics.md) | Local model and evaluator calibration evidence |
| 13 | [external-data.md](docs/13-external-data.md) | BFCL provenance, leakage controls, and baseline results |
| 14 | [sft-baseline.md](docs/14-sft-baseline.md) | SFT training evidence, results, and failure analysis |

Engineering activity and decisions are recorded in [`WORKLOG.md`](WORKLOG.md). The
logging convention is defined in the implementation plan and applies to every
future implementation and experiment session.

## Status

**Phase 2 — SFT baseline complete for seed 0.** The v1 300-pair data pipeline is
complete. Controlled 0.5B prompt diagnostics and a pinned 1.5B capacity diagnostic
have executed locally on Metal. The frozen 1.5B `native-full` baseline scored
62.5% calibrated accuracy, 83.33% act accuracy, 41.67% abstention accuracy, and
25% paired accuracy on all 120 validation tasks. The calibrated evaluator agrees
with all 60 owner-verified adjudications across behavior, semantics, and protocol
validity. Held-out test data remains untouched. See
[`docs/12-baseline-diagnostics.md`](docs/12-baseline-diagnostics.md),
[`docs/13-external-data.md`](docs/13-external-data.md),
[`docs/08-roadmap.md`](docs/08-roadmap.md) and
[`docs/11-implementation-plan.md`](docs/11-implementation-plan.md).

## Development setup

The foundation targets Python 3.12 and uses
[`uv`](https://docs.astral.sh/uv/) for locked dependency management:

```bash
make setup
make check
make data
make baseline-smoke
make prompt-diagnostic
make capacity-diagnostic
make baseline-validation
make external-fetch       # networked, one-time pinned snapshots
make external-prepare     # network-free normalization and leakage audit
make external-baseline    # local Metal inference + stored evaluation
make sft-data             # internal train/validation only
make sft-smoke            # 0.5B/20-step Metal training check
make sft-train            # 1.5B seed-0 LoRA training
make sft-validation       # adapter-aware internal validation
uv run python -m tool_abstention --help
uv run tool-abstention validate-config configs/project.yaml
uv run tool-abstention export-schemas /tmp/tool-abstention-schemas
uv run tool-abstention validate-record task path/to/task.json
uv run tool-abstention audit-pairs data/raw/productivity/tasks.jsonl
uv run tool-abstention evaluate \
  --tasks data/raw/productivity/tasks.jsonl \
  --predictions path/to/predictions.jsonl \
  --output results/local-eval
```

`make check` runs Ruff formatting and lint checks, strict mypy type checking,
and the pytest suite with coverage. GitHub Actions runs the same CPU-only command;
foundation checks do not download model weights or require Apple Silicon.

Canonical task, pair, prediction, and evaluation records are strict immutable
Pydantic models. JSON Schema exports are deterministic and unknown record fields
are rejected. Tool parameter schemas use JSON Schema Draft 2020-12.

`make data` deterministically generates 300 pairs / 600 tasks across productivity,
finance, and weather/geo. Every domain contains all four abstention classes, and
template families remain isolated within train, validation, or test. Generated data
is ignored by Git and accompanied by a content-hashed manifest and dataset card.

Stored predictions can be evaluated without rerunning inference. The evaluator
parses plain/OpenAI/Qwen-style tool calls, validates class-specific behavior, and
writes per-example judgments plus accuracy, paired accuracy, macro-F1, act and
abstention accuracy, hallucination rate, and per-class metrics.

Human evaluator calibration uses a deterministic, blinded 60-item packet balanced
across all five classes and three domains. Open
[`calibration/round-1/annotate.html`](calibration/round-1/annotate.html), complete
the labels, and download `annotations.completed.csv`. Validate returned labels with:

```bash
uv run tool-abstention validate-calibration \
  --annotations path/to/annotations.completed.csv \
  --mapping calibration/round-1/mapping.jsonl
```

Two independent annotation files can be compared with `calibration-agreement`,
which reports exact agreement and Cohen's kappa for every judgment axis.

The smoke model is pinned to
`mlx-community/Qwen2.5-0.5B-Instruct-4bit@53a32aee5e9447773fd2b85988395066aef3700a`.
The capacity diagnostic pins
`mlx-community/Qwen2.5-1.5B-Instruct-4bit@8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.
MLX dependencies are isolated in the `inference` dependency group, and CPU-only CI
does not import or download them.

Public benchmark records are external evaluation only and never enter SFT data.
The pinned BFCL slice contains 400 CALL and 240 ABSTAIN records; preparation found
no overlap with internal train, validation, or test queries. The 1.5B native-full
baseline scored 88.44% decision accuracy and 84.83% balanced accuracy. AgentAbstain
is retained in its native multi-turn layout and cataloged as 263 pairs across 42
environments; it is not converted or executed by the single-turn harness.

## Stack (planned)

- **Models:** Qwen2.5-1.5B-Instruct (primary), Qwen2.5-0.5B / Qwen3-0.6B (iteration), Llama 3.2 3B (stretch)
- **Training:** MLX (`mlx_lm.lora` for SFT, `mlx_lm.dpo` for DPO; `mlx_lm_lora` community toolkit for ORPO/CPO)
- **Evaluation:** PyTorch / pure-Python harness (inference + metrics), portable off Apple Silicon
- **Hardware target:** MacBook M5, 24 GB unified memory

## License

Code and docs to be released under Apache-2.0 (matching the When2Call benchmark we build on). Third-party datasets retain their own licenses — see [`docs/05-data-plan.md`](docs/05-data-plan.md).
