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

Engineering activity and decisions are recorded in [`WORKLOG.md`](WORKLOG.md). The
logging convention is defined in the implementation plan and applies to every
future implementation and experiment session.

## Status

**Phase 1 — data pipeline in progress. Repository foundation, canonical record
contracts, and a 40-pair productivity development slice are implemented. Domain
expansion and dataset splits remain.** See
[`docs/08-roadmap.md`](docs/08-roadmap.md) and
[`docs/11-implementation-plan.md`](docs/11-implementation-plan.md).

## Development setup

The foundation targets Python 3.12 and uses
[`uv`](https://docs.astral.sh/uv/) for locked dependency management:

```bash
make setup
make check
make data
uv run python -m tool_abstention --help
uv run tool-abstention validate-config configs/project.yaml
uv run tool-abstention export-schemas /tmp/tool-abstention-schemas
uv run tool-abstention validate-record task path/to/task.json
uv run tool-abstention audit-pairs data/raw/productivity/tasks.jsonl
```

`make check` runs Ruff formatting and lint checks, strict mypy type checking,
and the pytest suite with coverage. GitHub Actions runs the same CPU-only command;
foundation checks do not download model weights or require Apple Silicon.

Canonical task, pair, prediction, and evaluation records are strict immutable
Pydantic models. JSON Schema exports are deterministic and unknown record fields
are rejected. Tool parameter schemas use JSON Schema Draft 2020-12.

`make data` deterministically generates 40 productivity pairs (80 tasks) across
`ANSWER`, `CLARIFY`, `REFUSE`, and `NOOP`. Generated data is ignored by Git and
accompanied by a content-hashed manifest.

## Stack (planned)

- **Models:** Qwen2.5-1.5B-Instruct (primary), Qwen2.5-0.5B / Qwen3-0.6B (iteration), Llama 3.2 3B (stretch)
- **Training:** MLX (`mlx_lm.lora` for SFT, `mlx_lm.dpo` for DPO; `mlx_lm_lora` community toolkit for ORPO/CPO)
- **Evaluation:** PyTorch / pure-Python harness (inference + metrics), portable off Apple Silicon
- **Hardware target:** MacBook M5, 24 GB unified memory

## License

Code and docs to be released under Apache-2.0 (matching the When2Call benchmark we build on). Third-party datasets retain their own licenses — see [`docs/05-data-plan.md`](docs/05-data-plan.md).
