# 06 — Training Plan

> Stages, models, hyperparameters, and the MLX execution path. All runs fit in 24 GB unified memory (MacBook M5).

## 1. Models

| Model | Role | Size | Why |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | **primary** | 1.5B | strong base tool-calling, MLX-friendly, standard comparison point |
| Qwen2.5-0.5B / Qwen3-0.6B | iteration | ~0.5B | fast hyperparameter sweeps before committing 1.5B runs |
| Llama 3.2 3B | stretch | 3B | test whether the result scales past the 1.5B claim |

All are instruction-tuned (not base) models so the "abstention" signal is a delta on an already-competent tool-caller, not cold-start capability.

## 2. Training stages

```
SFT (positive examples)  ──►  preference optimization (negative examples)
        │                              │
        │  common LoRA init            ├─ DPO
        │                              ├─ ORPO (reference-free, no SFT)
        │                              ├─ RPO  (reward-aware)
        └──────────────────────────────┴─ K-DPO (knowledge-aware variant)
```

- **SFT** is both a *baseline method* and the *init* for DPO/RPO/K-DPO.
- **ORPO** is reference-free and runs *without* a separate SFT stage (its own baseline+preference in one).
- **RPO / K-DPO** are data-side variants of the DPO objective (different pair construction), so they reuse `mlx_lm.dpo` with `build_pref.py` producing the pairs.

### 2.1 Data assembly per stage

| Stage | Input | Chosen | Rejected |
|---|---|---|---|
| SFT | all `act` tasks + all `abstain` tasks (positive only) | correct behavior (CALL/ANSWER/CLARIFY/REFUSE/NOOP) | — |
| DPO | preference pairs | correct behavior | wrong behavior of the *same* class family (e.g. fabricated call vs REFUSE) |
| ORPO | same pairs as DPO | correct | wrong (in-batch, no reference model) |
| RPO | negative-example pairs | correct | explicit negatives: wrong tool, fabricated tool, missed follow-up |
| K-DPO | knowledge-aware pairs | correct answer + minimal/no tool | correct answer + unnecessary tool call |

The negative examples in `build_pref.py` are **synthesized** (not model-sampled) so the "wrong" side is controlled and interpretable: a fabricated `get_grades` call, a wrong-but-plausible tool, a missed clarification.

## 3. Hyperparameters (starting points)

### LoRA (all methods)
| Param | Value | Note |
|---|---|---|
| rank `r` | 16 | sweep {8, 16, 32} on 0.5B |
| alpha | 32 | = 2×r |
| dropout | 0.05 | |
| target modules | `q_proj, k_proj, v_proj, o_proj` (attn only) | MLX `mlx_lm.lora` default; add `gate/up/down` only if needed |
| trainable params | ~0.3–0.5% | keeps 24 GB comfortable |

### SFT
| Param | Value |
|---|---|
| optimizer | AdamW |
| lr | 2e-5 (cosine to 1e-6) |
| batch size | 4 × 4 grad-accum (effective 16) |
| epochs | 3 (early-stop on val abstention accuracy) |
| seq len | 2048 |
| warmup | 5% of steps |

### Preference (DPO / RPO / K-DPO)
| Param | Value |
|---|---|
| lr | 5e-6 |
| beta | 0.1 (sweep {0.05, 0.1, 0.2}) |
| epochs | 1–2 |
| batch | 4 × 4 accum |
| reference model | the SFT adapter (frozen) |

### ORPO
| Param | Value |
|---|---|
| lr | 5e-6 |
| lambda (orpo weight) | 0.1 (sweep {0.05, 0.1, 0.5}) |
| epochs | 2 |
| reference model | none (reference-free) |

> All numbers are starting points; the 0.5B sweep precedes the 1.5B commit. Final values are pinned in `configs/models/*.yaml` and logged (see `09-experiment-tracking.md`).

## 4. Memory & compute budget (24 GB)

| Operation | Est. peak | Safe? |
|---|---|---|
| Qwen2.5-1.5B fp16 load | ~3 GB | ✓ |
| LoRA forward/backward (rank 16) | ~6–8 GB | ✓ |
| 3B (stretch) LoRA | ~12–16 GB | ✓ (tight) |
| Full fine-tune >3B | >24 GB | ✗ (explicit non-goal) |

- Sequence length 2048, batch 16 effective, on an M5 this is ~30–60 min/stage for 1.5B — well within a single evening per method.
- If memory pressure appears, drop to rank 8 and seq 1536; never compromise on ≥3 seeds.

## 5. Method-comparison matrix

| Method | Reference model? | Separate SFT? | Negative examples? | Notes |
|---|---|---|---|---|
| SFT | — | — | ✗ | positive-only baseline |
| DPO | ✓ | ✓ | ✓ | the standard preference baseline |
| ORPO | ✗ | ✗ | ✓ | reference-free; cheapest pipeline |
| RPO | ✓ | ✓ | ✓ (reward-aware) | When2Call's method |
| K-DPO | ✓ | ✓ | ✓ (knowledge-aware) | Tool-Overuse Illusion's method |

Each method × {0.5B sweep, 1.5B primary} × ≥3 seeds. Total 1.5B runs: 5 methods × 3 seeds = 15 adapters (SFT shared as init across DPO/RPO/K-DPO). This is the scale the method comparison lives at.

## 6. Seeding & reproducibility

- **Global seed** drives: data generation, split assignment, model init, batch order, and dropout. One seed per run, passed via config → `util/seed.py`.
- **Seed set:** {0, 1, 2} for the 1.5B comparison (≥3 seeds for error bars per the v2 success criterion); {0} for 0.5B sweeps.
- **No `Date.now`/non-determinism in the path** — all sampling uses an explicit RNG seeded at launch.
- Every run logs: config hash, data manifest hash, model checkpoint path, adapter hash, and full hyperparameters (see `09-experiment-tracking.md`).

## 7. Execution CLI

```bash
make data                          # build dataset (seed-pinned)
make train METHOD=dpo SEED=0       # -> checkpoints/dpo/seed0/
make eval  METHOD=dpo SEED=0       # -> results/dpo/seed0/metrics.jsonl
make test                          # unit + smoke tests
```

- `src/train/launch.py` is the only training entry point; methods differ by config file, never by ad-hoc script.
- Adapters are exported to **safetensors** after training so eval (PyTorch) can load them off-Mac.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ORPO unstable at 1.5B | keep `lambda` small; fall back to CPO (community toolkit) if divergence |
| DPO overfitting to synthetic negatives | early-stop on val; monitor tool-call accuracy, not just abstention |
| K-DPO pairs too easy (answer-vs-call is trivially separable) | sample "hard" negatives (correct answer *with* tool vs *without*) |
| MLX `dpo` API drift | pin `mlx-lm` version in lockfile; wrap in `dpo.py` so the rest of the repo is insulated |
