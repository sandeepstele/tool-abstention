# 10 — Paper Plan

> The research contribution and target venue. This is the "why anyone should care" doc — it exists so Phase 5 isn't a scramble.

## 1. Contribution statement (one paragraph)

Tool-using agents are evaluated almost exclusively on *choosing the right tool*, but production failures are dominated by the opposite decision — *acting when they should not*. Recent benchmarks (When2Call, AgentAbstain, ToolFailBench) have measured this "agentic abstention" gap, and recent methods (RPO, K-DPO) have proposed fixes, but **no open work unifies the abstention taxonomy and ships a reproducible small-model training pipeline**. We do both: a five-class extended taxonomy (act + four ways to abstain), a paired, rule-labeled benchmark, and a head-to-head comparison of SFT / DPO / ORPO / RPO / K-DPO at ≤1.5B — runnable end-to-end on a laptop — that measures the *capability–reliability trade-off* rather than papering over it.

## 2. What's genuinely new (the reviewer's "so what")

1. **A unified taxonomy.** When2Call treats "direct answer" as always-wrong; K-DPO treats "answer from knowledge" as the goal. We show these are two *different* abstain classes that can — and should — be trained together.
2. **The method comparison the papers omit.** RPO (When2Call) and K-DPO (Tool-Overuse Illusion) were each validated in isolation on different taxonomies. We run all five methods on *identical data* and report a Pareto frontier, not a single winning number.
3. **Small-scale, honest, reproducible.** The field's abstention results are all at 4B–70B+. We show what's achievable at 1.5B and *what it costs* (the ≤~3-point tool-call regression), with ≥3-seed error bars and hash-reproducible artifacts.

## 3. Positioning vs. closest work

| Paper | Our delta |
|---|---|
| When2Call (2025) | + training code, + ≤1.5B, + "answer-from-knowledge" class, + method comparison |
| AgentAbstain (2026) | + released training pipeline; single-turn + rule-labeled (simpler, judge-free) |
| Tool-Overuse Illusion / K-DPO (2026) | + small-model reproduction, + all four abstain classes, not just knowledge-abstention |
| Reasoning Trap / SimpleToolHalluBench (2025) | + we *train* the mitigation across methods and cost the trade-off, rather than only observe it |

## 4. Target venues

- **Primary (short, realistic):** a workshop with a post-training/eval slant (e.g. ICML/NeurIPS agentic-eval or tool-use workshops), or **arXiv + ACL SRW / Findings** if the comparison table is clean.
- **Stretch:** NAACL Findings (When2Call's venue — a natural citation home).
- **Honest fallback:** a well-crafted arXiv technical report + blog — better than a wrong-venue rejection.

## 5. Paper outline

1. **Intro** — the negative case is under-measured; the gap; our two contributions (taxonomy + pipeline).
2. **Related work** — condense `02-literature-review.md` into the four-layer framing + synthesis table.
3. **The extended abstention taxonomy** — the five classes, why four abstain classes, the paired-task construction (Fig 1).
4. **Method** — data generation (rule-labeled, paired, deterministic); training (SFT/DPO/ORPO/RPO/K-DPO, LoRA); evaluation (rule-based, judge-free).
5. **Experiments** — main comparison table; the trade-off scatter (Fig 2); per-class delta table (Fig 3); ablation (unified vs per-class, H4).
6. **Analysis** — which classes are learnable (H3); the shape of the trade-off (H2); *(stretch)* probe decodability (H5).
7. **Limitations** — single-turn, synthetic data, small model, no learned reward; how each bounds the claims.
8. **Conclusion** — the honest finding, positive or negative.

## 6. Figure & table plan

| # | Artifact | Shows |
|---|---|---|
| Fig 1 | Five-class taxonomy + paired construction (schematic) | the design |
| Fig 2 | Act-accuracy vs abstain-accuracy scatter, 5 methods, error bars | the **trade-off frontier** (H2) — the money figure |
| Fig 3 | Per-class F1 deltas, before/after, heatmap | *which* classes each method buys (H3) |
| Tab 1 | Full metric table (paired-acc, macro-F1, halluc-rate, abstain-acc, act-acc) × 5 methods | the headline numbers |
| Tab 2 | Unified vs per-class training ablation | H4 |
| *(stretch)* Fig 4 | Linear-probe AUROC pre/post | H5 mechanistic signal |

## 7. The honest-results commitment

We commit *in advance* to publishing whichever of these is true:

- abstention trainable at small scale with bounded cost (**H0 holds**), or
- the measured **failure frontier** — the quantitative point at which reliability stops being free (**H0 fails**).

Either way the deliverable is the *measurement*, not the hoped-for outcome. This is what distinguishes the project from a "fine-tune and brag" repo, and it's the sentence that carries the paper.
