# 08 — Roadmap

> Phases, milestones, and exit criteria. Each phase has a hard "go/no-go" gate; nothing proceeds past a gate that isn't met. Times are calendar estimates, not promises.

## Status

**Phase 1 — data pipeline: in progress.** Planning, repository foundation,
canonical contracts, the 40-pair productivity development slice, and deterministic
stored-prediction evaluation are complete. Finance/weather expansion, leakage-safe
splits, and the full dataset manifest are not yet implemented.

---

## Phase 0 — Planning & framing ✅

**Deliverable:** the planning suite (`docs/01`–`docs/10`), literature verified, hypothesis falsifiable, stack chosen.

**Exit criteria:** ✓ README + 10 docs consistent; ✓ citations verified; ✓ H0 decomposed into testable H1–H5.

---

## Phase 1 — Data pipeline *(~1 week)*

**Build:** `src/data/` — taxonomy, tool registry, paired-task generator, deterministic labels, splits, manifest, unit tests.

**Milestone:** `make data` produces the full dataset (≥600 pairs / 1,200 tasks) deterministically.

**Exit criteria (all required):**
- [ ] Regeneration with same seed → byte-identical output.
- [ ] `test_taxonomy`, `test_generate` pass (pair integrity, class invariants, no leakage).
- [ ] Class balance ≥ 10% per class; CALL present in every domain.
- [ ] Every `CALL` label executes against its mock tool and returns the expected value.

---

## Phase 2 — SFT baseline *(~1 week)*

**Build:** `src/train/sft.py` + `launch.py`; `src/eval/harness.py`, `judge.py`, `metrics.py`.

**Milestone:** an untrained + SFT model evaluated end-to-end with the full metric set, logged and reproducible.

**Exit criteria:**
- [ ] `test_harness`, `test_metrics`, `test_judge` pass (incl. judge calibration ≥ 95% on 200 samples).
- [ ] Untrained baseline numbers recorded and sensible (not ~100% or ~0% everywhere).
- [ ] SFT improves *something* (act-accuracy, at minimum) over untrained — confirms the harness is sensitive.

---

## Phase 3 — Preference optimization *(~2 weeks)*

**Build:** `build_pref.py` + `dpo.py` / `orpo.py` / `rpo.py` / `kdpo.py`.

**Milestone:** the full method comparison (5 methods × 3 seeds × 1.5B) run and logged.

**Exit criteria:**
- [ ] All 15 adapters train to completion without OOM on 24 GB.
- [ ] Per-method results reproducible from config + seed alone.
- [ ] At least one preference method beats SFT on abstention accuracy.

---

## Phase 4 — Evaluation & analysis *(~1 week)*

**Build:** trade-off scatter, per-class deltas, significance tests, `results/summary.md`.

**Milestone:** the metric tables in `01-vision.md` goals 4–5 are generated and answerable.

**Exit criteria:**
- [ ] v1 success criteria (see `01-vision.md`) met — or a *negative result* cleanly characterized.
- [ ] ≥3-seed error bars on the 1.5B comparison.
- [ ] *(stretch)* probe AUROC pre/post (H5) computed.

---

## Phase 5 — Write-up & release *(~1 week)*

**Build:** `10-paper-plan.md` → actual write-up; repo polish; results README.

**Milestone:** public repo + short paper/technical report.

**Exit criteria:**
- [ ] Reproducibility manifest complete (config/data/adapter hashes).
- [ ] Write-up states the trade-off honestly, including any tool-call regression.
- [ ] Apache-2.0 clean (third-party data licenses documented).

---

## Timeline summary

| Phase | Estimate | Gate |
|---|---|---|
| 0 Planning | done | — |
| 1 Data | 1 wk | deterministic, tested, balanced |
| 2 SFT baseline | 1 wk | harness sensitive, judge calibrated |
| 3 Preference | 2 wk | 15 adapters, reproducible |
| 4 Eval/analysis | 1 wk | v1 criteria or clean negative |
| 5 Write-up | 1 wk | public, honest, reproducible |

**Total: ~6 weeks** of part-time work.

## Exit-early / pivot rules

- If **Phase 2** shows the harness can't detect *any* difference (untrained ≈ SFT on everything), stop and fix the harness before spending Phase 3 compute.
- If **Phase 3** shows *no* preference method beats SFT, that is **still a publishable negative result** (H1 falsified) — proceed to Phase 5 and write it up, don't chase a positive.
- If memory pressure forces rank 8 / seq 1536, that's an acceptable trade *only if* documented in the config and manifest — never silently.
