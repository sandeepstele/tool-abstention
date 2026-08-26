# 01 — Vision & Goals

## One-paragraph thesis

Agentic systems fail in production not only by choosing the *wrong* tool, but by choosing to act when they should not: calling a tool that doesn't exist, inventing arguments, or fetching data they already have. This capability — **agentic abstention** — is under-measured (existing benchmarks reward confident action) and under-trained (almost no open pipeline teaches it). This project builds the missing piece: a benchmark and a reproducible LoRA-based post-training pipeline that teaches a ≤1.5B model *when not to call a tool*, and measures the act/abstain trade-off with verifiable metrics.

## Problem statement

Tool-calling evaluation (BFCL, τ-bench, ToolEval) is **action-optimized**: it scores how well a model calls the right tool with the right arguments. Three failure modes fall through this frame:

1. **Over-calling / unnecessary tools** — the model calls a tool for a question answerable from the prompt or parametric knowledge (the "Tool-Overuse Illusion", K-DPO).
2. **Fabrication** — no suitable tool exists, but the model hallucinates one (When2Call, SimpleToolHalluBench, AgentProp-Bench).
3. **Missed clarification** — a required parameter is missing, but the model invents it instead of asking (When2Call follow-up class).

Recent work (see `02-literature-review.md`) shows these are **real, measurable, and largely independent of general task-solving ability**. Frontier agents still score ~59% on paired act/abstain tasks. This is a live gap, not a solved problem.

## What exists today (and the gap we fill)

| Existing asset | What it provides | What it does **not** provide |
|---|---|---|
| When2Call (NAACL 2025) | 4-way benchmark + SFT/preference data + eval scripts | No training code; uses 4B/8B models; treats "direct answer" as *always* wrong |
| AgentAbstain (2026) | Paired act/abstain taxonomy + 263 tasks | No released training pipeline |
| When2Tool / ToolFailBench / SimpleToolHalluBench | Diagnostic benchmarks | Benchmarks only, no training |
| K-DPO / Tool-Overuse Illusion | Method for "trust internal knowledge" | No open small-model reproduction |
| ToolBrain / RLFactory / TRL | Generic tool-use post-training | Ignore the *abstain* dimension |

**Our gap-fill:** a single, clean, open pipeline that (a) runs end-to-end on a laptop, (b) unifies the "no tool / can't answer" abstention of When2Call with the "answer from knowledge" abstention of K-DPO, and (c) publishes the *training* code and method-comparison that the papers omit.

## Goals

1. Build a **paired benchmark**: for each task, a *should-act* variant and a *should-abstain* variant, with verifiable labels (no LLM-judge needed for core metrics).
2. Implement a **three-stage training pipeline** (SFT → preference optimization, LoRA) runnable in MLX on 24 GB unified memory.
3. **Compare methods**: SFT vs DPO vs ORPO vs RPO (K-DPO as a variant) on the same data.
4. Produce a **before/after metric table** across the five decision classes.
5. Package everything as a **reproducible, public repo** with a results README and a short write-up.

## Non-goals (explicitly out of scope)

- Training a foundation model from scratch, or any full fine-tune >3B.
- Building a production agent framework / new RAG system.
- Claiming SOTA on BFCL tool-calling accuracy.
- Multi-modal (vision/audio) tool use.
- RLHF with a learned reward model (a *verifiable-reward* GRPO extension is a stretch goal, not core).
- Building an LLM-judge-gated metric as the *primary* signal (we avoid judge dependency for core metrics).

## Success criteria

**Minimum (v1):**
- Paired benchmark of ≥ 500 tasks across ≥ 3 tool domains, with deterministic labels.
- A LoRA SFT model and at least one preference-optimized model (DPO or ORPO) trained on an M5/24GB.
- A reproducible eval harness reporting: accuracy, macro-F1, tool-hallucination rate, and abstention accuracy.
- A measurable improvement in abstention metrics vs. baseline, with tool-call accuracy held within a small margin (≤ ~3 points).

**Target (v2):**
- Method comparison across SFT / DPO / ORPO / RPO with ≥ 3 seeds and error bars.
- A finding on the **capability–reliability trade-off** (does abstention training degrade tool-call quality? by how much?).
- A public repo + a technical write-up / short paper (see `10-paper-plan.md`).

## Key differentiators vs. a "generic fine-tune" project

1. **Measures the negative case** — the thing most benchmarks and repos ignore.
2. **Owns the labels** — paired, verifiable ground truth rather than a borrowed dataset + borrowed metric.
3. **Method comparison, not a single run** — DPO vs ORPO vs RPO on identical data is the actual contribution.
4. **Honest about the trade-off** — we report *both* abstention gain and any tool-call regression, which is the part hiring managers and reviewers respect.
