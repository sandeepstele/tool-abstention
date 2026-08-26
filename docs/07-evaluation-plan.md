# 07 — Evaluation Plan

> Metrics, harness, baselines, and testing. The core metrics are **rule-based and judge-free** (a stated non-goal is LLM-judge dependence); the harness is deterministic and portable off-Mac.

**Implementation status:** stored-prediction parsing, five-class deterministic
judgment, core aggregate metrics, paired metrics, and replayable result output are
implemented. A 200-case construction-based adversarial matrix passes. The separate
200-sample human-labeled calibration on real model outputs remains required before
reporting evaluator agreement.

## 1. Metrics

### 1.1 Headline metric — paired accuracy
A pair is **correct** iff the model is right on *both* the `act` and `abstain` variants:

```
paired_accuracy = mean( correct(act_i) ∧ correct(abstain_i) )
```

This is the strictest, most meaningful number (matches AgentAbstain's definition) and directly punishes "always-call" or "never-call" degeneracy.

### 1.2 Report metrics (from `01-vision.md` success criteria)

| Metric | Definition | What it captures |
|---|---|---|
| **accuracy** | mean correct over all tasks | overall decision quality |
| **macro-F1** | unweighted mean F1 across the **five classes** | balance — the "before/after table across five decision classes" (goal 4) |
| **tool-hallucination rate** | fraction of tasks where a tool is called but the correct label is an abstain class (esp. REFUSE/ANSWER) | the *fabrication/over-calling* failure |
| **abstention accuracy** | accuracy computed only over the four abstain classes | the isolated "learned to abstain?" signal |

### 1.3 The trade-off metric (H2)
- **act-accuracy** = accuracy over `CALL`-labeled tasks (did it call correctly when it should?).
- **abstain-accuracy** = accuracy over the four abstain classes (did it abstain correctly when it should?).
- **Report both on the same axes** (scatter, ≥3 seeds → error bars). The success criterion is: *abstain-accuracy up, act-accuracy within ≤ ~3 points.*

### 1.4 Per-class metric (H3/H4)
- **Per-class F1** for each of the five classes, before/after training. This is the raw material for "which classes are learned at different rates."

## 2. Harness design

### 2.1 Deterministic inference
- **Greedy decoding** (temperature 0 / argmax) for the core metrics — no sampling noise, no judge noise.
- *(Optional)* `temperature ∈ {0.2, 0.6, 1.0}` for a robustness sub-analysis only; core numbers are greedy.
- Input = `system` (tools + abstention instruction) + `query`, templated by `prompts.py`.

### 2.2 Rule-based classification (`judge.py`)
The raw model output is classified into one of the five classes **by rules only**:

| Rule | Class |
|---|---|
| Valid parsed tool call + correct tool + correct args | CALL |
| No tool call + non-empty answer + task answerable-by-construction | ANSWER |
| No tool call + interrogative + references the missing slot | CLARIFY |
| No tool call + refusal lexicon (`cannot`/`unable`/`no tool`/`can't help`) | REFUSE |
| No tool call + no-op/acknowledgment marker + task pre-satisfied | NOOP |

The classifier is **pure Python, fully unit-tested**, and its false-positive/false-negative behavior is characterized on a hand-labeled calibration set (target ≥ 95% agreement on 200 samples — a one-time, offline check, not a runtime judge).

### 2.3 Optional LLM-judge (off the critical path)
An LLM judge *may* be run **post-hoc** on the free-form subset to sanity-check the rule classifier, but it is **never the primary metric** and never gates a result.

## 3. Baselines (the comparison set)

| Baseline | What it is | Why it's there |
|---|---|---|
| **untrained** | the base instruct model, greedy | the "before" row |
| **prompt-only** | base model + a stronger/weaker abstention instruction in the system prompt | isolates promptable vs *trained* abstention (When2Tool's prompt-only baseline) |
| **SFT** | positive-only LoRA | the crowded default; the thing we claim preference beats |
| **DPO / ORPO / RPO / K-DPO** | the preference methods | the actual contribution |

Every method is evaluated on the **identical, held-out test set** with identical greedy decoding.

## 4. Statistical rigor

- **≥ 3 seeds** for the 1.5B method comparison → report mean ± std (error bars).
- **Seed = data seed + train seed**, so the error bars cover both data-split and optimization variance.
- **Pairwise significance** (paired bootstrap / McNemar over pairs, since act/abstain are paired) between methods where a claim of superiority is made.
- **Multiple-comparison awareness**: with 5 methods × pairwise, we report raw + (optionally) Holm-corrected p-values; the headline claim is a *frontier*, not a single winner.

## 5. Testing (unit + smoke)

| Test | What it guards |
|---|---|
| `test_taxonomy.py` | five classes mutually exclusive/exhaustive; label invariants |
| `test_generate.py` | paired-task integrity (every abstain has an act twin, exactly one δ); determinism under fixed seed |
| `test_metrics.py` | metric correctness on hand-computed cases (e.g. a known 2×2 confusion matrix → known F1) |
| `test_judge.py` | classifier agreement on the calibration set ≥ threshold; no judge in the metric path |
| `test_harness.py` | end-to-end smoke: tiny model + 8 tasks → metrics emit, no crash, results written |

**CI intent:** `make test` must pass before any experiment result is considered reportable; results are tagged with the test-suite commit hash.

## 6. What "good" looks like (tieing back to success criteria)

- **v1 minimum:** abstention-accuracy ↑ vs. untrained **and** vs. SFT; act-accuracy within ≤3 points; paired-accuracy ↑; tool-hallucination rate ↓.
- **v2 target:** a readable **Pareto frontier** of (act-accuracy, abstain-accuracy) across the 5 methods with error bars; a per-class delta table showing *which* abstain class each method buys; a probe AUROC (stretch, H5) pre/post.

## 7. Reporting artifact

`results/summary.md` (committed) auto-generated from `results/*.jsonl`:
- the five-class before/after table,
- the method-comparison table (all metrics × all methods),
- the trade-off scatter (act vs abstain accuracy),
- per-class F1 deltas,
- the reproducibility manifest (config + data + adapter hashes).
