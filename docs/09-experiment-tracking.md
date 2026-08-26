# 09 — Experiment Tracking & Reproducibility

> How we make every result regenerable and every claim checkable. The rule: **a result you can't reproduce from a hash isn't a result.**

## 1. Config as the source of truth

Every run is defined by a **single config file** under `configs/`. No hyperparameter is hard-coded in a script; scripts read configs. A run's identity is the **SHA-256 of its config + data manifest + code commit**:

```
run_id = sha256(config.yaml ‖ data_manifest.json ‖ git_commit)
```

This means: same `run_id` ⇒ same inputs ⇒ same output (given determinism guarantees in `06-training-plan.md` §6).

## 2. The reproducibility manifest

`data/manifest.json` maps every artifact to its provenance:

```json
{
  "data/processed/sft/train.jsonl": {
    "content_hash": "sha256:…",
    "config_hash": "sha256:…",
    "seed": 0,
    "generator": "build_sft.py",
    "built_at": "…"
  }
}
```

Checkpoints and results carry the same convention:

| Artifact | Provenance recorded |
|---|---|
| dataset split | config hash + seed |
| LoRA adapter (`checkpoints/`) | model + data manifest hash + config hash + seed |
| metrics (`results/`) | adapter hash + harness config + test-suite commit hash |

## 3. Logging (two layers)

### 3.1 JSONL experiment log (always on)
`results/<method>/<seed>/events.jsonl` — one line per run with: `run_id`, `method`, `seed`, full flattened hyperparameters, wall-clock, and the metrics payload. Diffable, greppable, no external service required.

### 3.2 W&B (optional)
If available, mirror the JSONL to Weights & Biases for the trade-off scatter and training curves. **W&B is a view, never the source of truth** — the JSONL is.

## 4. Determinism guarantees

| Source of nondeterminism | Control |
|---|---|
| Data generation | explicit RNG seeded at launch |
| Split assignment | seeded shuffle |
| Model init | seeded (MLX `mlx.core.random.seed`) |
| Batch order / dropout | seeded |
| Inference (core metrics) | greedy decode (no sampling) |
| `Date.now` / wall-clock | **never** in the compute path (only in log timestamps) |

## 5. Environment pinning

- **Python deps:** `uv lock` → `uv.lock` committed (exact versions).
- **Model weights:** base model pinned by HF revision/commit hash, recorded in the manifest.
- **Training libs:** `mlx-lm`, `mlx`, `mlx_lm_lora` pinned to exact versions (API drift risk — see `06-training-plan.md` §8).
- **Hardware note:** `platform.machine()`, macOS version, and `mlx.core.metal.device_info()` logged (unified-memory headroom varies by machine).

## 6. The "regenerate to verify" contract

A third party can verify any published number by:

```bash
git checkout <commit>          # exact code
uv sync                        # exact deps (uv.lock)
make data    # → data/manifest.json must match published hash
make train METHOD=dpo SEED=0   # → adapter hash must match
make eval  METHOD=dpo SEED=0   # → metrics must match (within float tolerance)
```

If the manifest hash doesn't match, the result is **not** reproduced — full stop. This is the bar for the "reproducible public repo" claim in `01-vision.md`.

## 7. Result storage & retention

- `data/` and `checkpoints/` are **gitignored** (large, regenerable); only their **hashes** are committed.
- `results/*.jsonl` and `results/summary.md` are **committed** (small, the actual findings).
- Adapters are exported to safetensors and archived alongside the manifest (local + optionally HF).

## 8. Experiment naming convention

```
<method>__<model>__<seed>   e.g.  dpo__qwen2.5-1.5b__2
```

Names are derived from config, not hand-typed, so a rename can't silently change provenance.
