# 04 — Architecture

> Repo layout, components, and data flow. The foundation and canonical record
> contracts are implemented; generators, training, and evaluation remain planned
> (see `08-roadmap.md`).

## 1. Design principles

1. **Determinism first.** Labels and core metrics are rule-based; no LLM-as-judge in the critical path. Any stochasticity (seeds, sampling) is pinned and logged.
2. **MLX for training, portable Python for eval.** Training is MLX-only (Apple Silicon); the eval harness is pure Python/PyTorch so results are checkable off-Mac.
3. **Data is a pipeline, not a blob.** Every dataset artifact is regenerable from `src/data` + a config; generated artifacts carry a content hash.
4. **One method = one config.** SFT / DPO / ORPO / RPO / K-DPO differ only by a config file, not by bespoke scripts.

## 2. Repo layout

```
tool-abstention/
├── README.md
├── LICENSE
├── pyproject.toml                # deps + entry points (uv)
├── Makefile                      # top-level targets: data, train, eval, test
├── configs/
│   ├── data/
│   │   ├── base.yaml             # taxonomy, domains, sizes, seeds
│   │   └── <domain>.yaml         # per-domain tool schemas + generators
│   ├── models/
│   │   ├── sft.yaml
│   │   ├── dpo.yaml
│   │   ├── orpo.yaml
│   │   ├── rpo.yaml
│   │   └── kdpo.yaml
│   └── eval/
│       └── harness.yaml
├── src/
│   ├── data/
│   │   ├── taxonomy.py           # the 5 decision classes (single source of truth)
│   │   ├── tools.py              # tool schema registry + mock executors
│   │   ├── generate.py           # paired act/abstain task generation
│   │   ├── label.py              # deterministic label assignment
│   │   ├── prompts.py            # prompt templating (system + tools + query)
│   │   ├── build_sft.py          # -> SFT records
│   │   ├── build_pref.py         # -> preference pairs (chosen/rejected)
│   │   └── split.py              # train/val/test, contamination guard
│   ├── train/
│   │   ├── sft.py                # mlx_lm.lora wrapper
│   │   ├── dpo.py                # mlx_lm.dpo wrapper
│   │   ├── orpo.py               # mlx_lm_lora wrapper
│   │   ├── rpo.py                # RPO objective (negative-example pref opt)
│   │   ├── kdpo.py               # K-DPO variant (knowledge-aware pairs)
│   │   └── launch.py             # CLI: read config -> run -> log
│   ├── eval/
│   │   ├── harness.py            # inference loop (deterministic decode)
│   │   ├── metrics.py            # accuracy, macro-F1, halluc-rate, abstain-acc
│   │   ├── judge.py              # rule-based act/abstain classifier
│   │   ├── baselines.py          # untrained + prompt-only + SFT baselines
│   │   └── probe.py              # (stretch) linear probe for H5
│   ├── util/
│   │   ├── seed.py               # global RNG seeding
│   │   ├── hash.py               # artifact content hashing
│   │   └── logging.py            # JSONL experiment log
│   └── __init__.py
├── tests/
│   ├── test_taxonomy.py          # class invariants, label determinism
│   ├── test_metrics.py           # metric correctness on known cases
│   ├── test_generate.py          # paired-task invariants (see below)
│   └── test_harness.py           # end-to-end smoke on a tiny model
├── data/                         # generated artifacts (gitignored, hashed)
│   ├── raw/
│   ├── processed/
│   └── manifest.json             # path -> hash -> config hash
├── checkpoints/                  # LoRA adapters (gitignored)
├── results/                      # metrics JSONL + summaries (committed)
└── docs/                         # this planning suite
```

## 3. Components

### 3.1 `data/` — construction
- `src/tool_abstention/taxonomy.py` now defines the five decision classes and
  pairing enums. `records.py` is the strict Pydantic source of truth for tool,
  task, pair, prediction, and evaluation artifacts; `schemas.py` exports their
  deterministic JSON Schemas and validates JSON records.
- `src/tool_abstention/productivity.py` implements the first vertical slice: four
  deterministic mock tools, seeded generation, semantic one-perturbation checks,
  executable `CALL` verification, manifests, loading, and human-readable audits.
- `taxonomy.py` is the **single source of truth** for the five classes (see `05-data-plan.md`). Everything else imports from it; no magic strings.
- `tools.py` holds a registry of *mock* tool schemas (OpenAI function-calling JSON-Schema style) plus deterministic executors so a "tool call" can actually be *executed* for `should-act` tasks (keeps the harness honest).
- `generate.py` produces **paired tasks**: each task has a `should-act` and a `should-abstain` variant via a controlled perturbation (mirroring AgentAbstain's paired design, but single-turn and rule-labeled).
- `label.py` assigns the ground-truth class *by construction*, so labels never depend on a model or an LLM judge.

### 3.2 `train/` — training
- Thin wrappers around MLX: `sft.py` (`mlx_lm.lora`), `dpo.py` (`mlx_lm.dpo`), `orpo.py` (`mlx_lm_lora`), `rpo.py` / `kdpo.py` (custom preference construction + `mlx_lm.dpo`).
- `launch.py` reads a `configs/models/*.yaml`, seeds everything, runs, and writes to `results/`.

### 3.3 `eval/` — evaluation
- `harness.py` runs greedy/decode over the test set and captures the *raw* model output (token text) — never a pre-baked score.
- `judge.py` classifies output into one of the five classes using rules only (tool-call syntax parse, refusal/ask lexicons, answer non-emptiness, no-op markers).
- `metrics.py` computes the report metrics (see `07-evaluation-plan.md`).
- `probe.py` (stretch) trains a linear probe on hidden states for H5.

## 4. Data flow

```
configs/data/*.yaml
      │
      ▼
  generate.py ──► paired tasks (JSONL, raw/)
      │
      ▼
  label.py ──► five-class labels (deterministic)
      │
      ├─► build_sft.py ──► SFT records ──► mlx_lm.lora ──► adapter (sft)
      │
      └─► build_pref.py ──► chosen/rejected pairs ──► dpo/orpo/rpo/kdpo
                                                           │
                                                           ▼
                                                    adapter (method)
                                                           │
                                                           ▼
  harness.py ──► raw outputs ──► judge.py ──► metrics.py ──► results/*.jsonl
```

## 5. Key interfaces

### Dataset record (processed)
```json
{
  "id": "finance-042-act",
  "domain": "finance",
  "pair_id": "finance-042",
  "variant": "act",              // "act" | "abstain"
  "tools": [ /* JSON-schema tool defs */ ],
  "query": "…",
  "label": "CALL",               // one of the five classes
  "expected_tool": "get_balance" // only for CALL; null otherwise
}
```

### Preference pair (for DPO-family)
```json
{
  "prompt": "…",                 // system + tools + query
  "chosen": "…",                 // correct behavior (e.g. refuse/answer/clarify)
  "rejected": "…",               // the paired negative (wrong call / fabricated / missed follow-up)
  "abstain_class": "REFUSE"
}
```

### Metrics record (result)
```json
{
  "run_id": "…",
  "method": "dpo",
  "seed": 0,
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "tool_hallucination_rate": 0.0,
  "abstention_accuracy": 0.0,
  "per_class": {"CALL": {…}, "ANSWER": {…}, "CLARIFY": {…}, "REFUSE": {…}, "NOOP": {…}}
}
```

## 6. Tech stack mapping

| Concern | Choice | Rationale |
|---|---|---|
| Package/deps | `uv` + `pyproject.toml` | fast, lockfile-reproducible |
| Training SFT | `mlx_lm.lora` | native LoRA on Apple Silicon |
| Training DPO | `mlx_lm.dpo` | native DPO |
| Training ORPO/CPO | `mlx_lm_lora` (community) | reference-free objective |
| Training RPO/K-DPO | `mlx_lm.dpo` + custom pair builder | the objective is data-side, not trainer-side |
| Eval | pure Python + PyTorch (inference) | portable off Mac |
| Logging | JSONL + optional W&B | no judge dependency, diffable artifacts |
| Hashing | SHA-256 manifest | regenerate-with-confidence |

## 7. Portability & reproduction

- **Training** requires MLX (macOS only) — documented as such.
- **Eval** runs anywhere with PyTorch: load the same LoRA adapter (exported to safetensors) or, if unavailable, evaluate the *base* model + a published adapter.
- Every artifact is hashed and tied to its generating config hash (see `09-experiment-tracking.md`), so a third party can verify a result is a *regeneration*, not an edit.
