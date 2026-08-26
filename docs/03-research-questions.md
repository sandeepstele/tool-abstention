# 03 — Research Questions & Hypotheses

> The falsifiable claims the whole project is organized around. Each hypothesis maps to a metric in `07-evaluation-plan.md` and to a figure in `10-paper-plan.md`.

## 1. Core hypothesis (from `01-vision.md`)

> **H0.** Preference optimization with explicit negative examples (wrong tool call, fabricated tool, missed follow-up) can teach a ≤1.5B model to abstain correctly **without degrading** its tool-call accuracy — and we can measure that trade-off precisely.

Everything below is a decomposition of H0 into testable parts.

## 2. Hypotheses

### Abstention is learnable (not just promptable)

**H1 — Preference beats SFT on abstention.** Training on *negative* abstain examples (DPO/RPO/K-DPO/ORPO) yields a larger abstention-accuracy gain than SFT on positive examples alone, at matched compute and matched tool-call accuracy.

- *Why it's interesting:* When2Call showed RPO > SFT at 8B; nobody has shown this holds at ≤1.5B, or that it's *not* just a scale effect.
- *Falsified by:* SFT ≥ best preference method on abstention accuracy at every seed.

**H2 — The capability–reliability trade-off is real but bounded.** Abstention training degrades tool-call accuracy by a *small, measurable* amount (we target ≤ ~3 points), and this cost differs by method.

- *Why it's interesting:* SimpleToolHalluBench/Reasoning-Trap predicts the trade-off exists; we claim it can be made *small and predictable* via method choice, rather than accepting "reliability always costs capability".
- *Falsified by:* no method keeps tool-call regression within the margin, *or* every method regresses identically (no method signal).

### The taxonomy matters (not all abstention is the same)

**H3 — The four abstain classes are not equally trainable.** "Refuse (no tool)" and "clarify (missing arg)" are easier to teach than "answer-from-knowledge" and "no-op", because the former have structural surface cues and the latter require epistemic self-assessment.

- *Why it's interesting:* this is the K-DPO insight (epistemic boundary is the hard part) generalized. If true, it tells the field *where* to spend data budget.
- *Falsified by:* roughly uniform per-class improvement, or the "answer-from-knowledge" class improving as easily as "refuse".

**H4 — Unifying the taxonomy helps, not hurts.** Training on the *combined* five-class taxonomy (When2Call's "can't answer" + K-DPO's "answer from knowledge") does not harm performance on either, relative to training each in isolation.

- *Why it's interesting:* it is the concrete way we "unify" the two lines of work; if combining them degrades either, that's a real negative result about taxonomy design.
- *Falsified by:* combined training underperforms single-class training on the class it "inherited".

### The signal is already there (mechanistic prediction)

**H5 — Act/abstain is linearly decodable, and training sharpens it.** The act/abstain decision is recoverable by a linear probe of the hidden state *before* training, and preference optimization increases the margin (probe accuracy / AUROC) rather than relocating the decision.

- *Why it's interesting:* When2Tool (AUROC 0.89–0.96) predicts the pre-training signal exists; we test whether training merely *surfaces* it. This is the single cheapest "why does it work" experiment and a natural Figure.
- *Falsified by:* no pre-training decodability, or post-training decodability *decreases* while behavior improves.
- *Status:* **stretch** — a diagnostic, not a gating deliverable.

## 3. Research questions (RQ) — the paper's spine

| # | Question | Answered by | Maps to |
|---|---|---|---|
| RQ1 | Can a ≤1.5B model learn to abstain at all (vs. its own untrained baseline)? | pre/post Δ on abstention metrics | H1 |
| RQ2 | Which method (SFT / DPO / ORPO / RPO / K-DPO) gives the best abstention gain *per unit of tool-call regression*? | method-comparison table, efficiency frontier | H1, H2 |
| RQ3 | What is the *shape* of the capability–reliability trade-off across methods — does a Pareto frontier emerge? | act-accuracy vs abstain-accuracy scatter, ≥3 seeds | H2 |
| RQ4 | Are the four abstain classes learned at different rates, and does the ordering match our epistemic-difficulty hypothesis? | per-class macro-F1 deltas | H3 |
| RQ5 | Does a unified taxonomy train as well as its parts in isolation? | ablation: combined vs per-class training | H4 |
| RQ6 | Is the act/abstain decision linearly decodable before training, and does training sharpen the decision boundary? | probe AUROC pre/post | H5 (stretch) |

## 4. Expected findings (a prior, stated to keep us honest)

1. **DPO-family > SFT** on abstention, consistent with When2Call's RPO result — but the *size* of the gap shrinks at 1.5B. We expect RPO/K-DPO to lead on "refuse" and "answer-from-knowledge" respectively.
2. **A visible trade-off**: abstention up, tool-call accuracy down a few points; ORPO (reference-free, no separate SFT stage) likely trades most cheaply but with higher variance.
3. **Heterogeneous class difficulty**: "clarify" and "refuse" improve fastest; "answer-from-knowledge" is the long tail.
4. **Decodability confirmed** (per When2Tool): pre-training AUROC already high; training widens the margin.

## 5. What a negative result would still teach us

- **If H0 fails** (abstention gain always costs >3 points of tool-call accuracy): we publish the *failure frontier* — a measured, quantitative "you can't have both" for small models. Still useful, still honest, still the thing the Reasoning-Trap paper observed but never costed at small scale.
- **If H3 fails** (uniform class improvement): the taxonomy is flatter than the papers imply — a real simplification result for benchmark design.
- **If H4 fails** (unified taxonomy hurts): a concrete warning that "answer-from-knowledge" and "can't-answer" are *opposed* training signals, not a single axis.

Every one of these is a citable finding. The project is structured so that **the honest answer is the deliverable**, not a specific hoped-for result.

## 6. Out of scope (not asked, to avoid scope creep)

- RLVR/GRPO with a learned or environment reward (a *verifiable-reward* GRPO extension is a stretch goal only — see `01-vision.md` non-goals).
- Causal attribution of *why* preference optimization works at the mechanistic level (probe = descriptive; no interventions/ablations on circuits).
- Multi-turn agentic trajectories with irreversible state changes (AgentAbstain's "post-hoc abstention" is *cited*, not reproduced — our benchmark is single-turn decision).
