# 02 — Literature Review

> What exists, where the gap is, citations. Companion to `01-vision.md`; the paper plan in `10-paper-plan.md` positions us against the closest rows below.

## 0. How this review is organized

We split the landscape into four layers, because the field's blind spot lives *between* layers:

1. **Action-optimized baselines** — benchmarks that reward confident action and never score the "should I act?" decision.
2. **Diagnostic benchmarks** — the recent (2025–2026) cluster that *does* measure abstention / over-calling / hallucination.
3. **Training methods** — preference-optimization recipes that actually teach "when not to act".
4. **Generic tool-use training stacks** — the crowded fine-tuning space that ignores the abstain dimension.

The gap we fill sits at the intersection of layers 2 and 3: **nobody releases a small-model training pipeline on top of the diagnostic taxonomies.**

---

## 1. Action-optimized baselines (the frame we're leaving)

| Benchmark | What it scores | Why it misses the negative case |
|---|---|---|
| **BFCL** (Berkeley Function Calling Leaderboard, Yan et al. 2024) | Correct tool + correct args, plus multi-turn/simple/parallel/relevance subsets | Its "relevance" / "irrelevance" subsets are the closest thing to abstention, but the headline score is action-optimized; irrelevance is a small add-on, not the point |
| **τ-bench** (Sierra, Yao et al. 2024) | Task completion in tool-mediated multi-turn dialogues against a policy | Rewards *getting the task done*; never rewards correctly deciding a tool call is unnecessary or impossible |
| **ToolEval** (Qin et al. 2023) | `pass`/`win` rate of tool calls vs. a reference | Reference is always "call the tool"; no "should not call" reference exists |

**Takeaway:** these benchmarks share an implicit prior — *a good agent is one that calls the right tool*. The three failure modes in `01-vision.md` (over-calling, fabrication, missed clarification) are structurally invisible to them, because each is scored as "just not a successful call" and lumped into the same error bucket.

---

## 2. Diagnostic benchmarks (the evidence our gap is real)

### 2.1 When2Call — the closest ancestor
- **Paper:** Ross, Mahabaleshwarkar, Suhara, *"When2Call: When (not) to Call Tools"*, NAACL 2025. arXiv:2504.18851. [aclanthology](https://aclanthology.org/2025.naacl-long.174/).
- **What it is:** reframes tool-calling as a 4-way decision — **(a) direct answer, (b) tool call, (c) follow-up question, (d) unable to answer** — exposing over-calling and abstention failure directly.
- **Data:** `nvidia/When2Call` on HF (CC-BY-4.0). 15k SFT examples + 9k preference pairs (train); 3,652 MCQ + 300 LLM-as-judge free-form (test). Synthetic, auto-labeled, generation scripts on GitHub.
- **Headline results:** Llama-3.1-8B scores **~0%** on "unable to answer"; RPO-trained Mistral-NeMo-Minitron-8B reaches **+8.6%** over SFT and **87.1%** on BFCL-Irrelevance (vs. 56.0% for Llama-3.1-8B-Instruct).
- **What it does *not* provide:** no training code; uses 4B/8B models (not ≤1.5B); and its "direct answer" class is treated as **always wrong** — it cannot distinguish "answer from knowledge (correct)" from "hallucinate an answer when a tool *is* needed (wrong)". That distinction is exactly what K-DPO captures and we unify (see §3.1).

### 2.2 AgentAbstain — the paired-act/abstain design we adopt
- **Paper:** *"AgentAbstain: Do LLM Agents Know When Not to Act?"*, arXiv:2607.10059 (July 2026). [Project page](https://agentabstain.github.io/).
- **What it is:** the first *systematic* agentic-abstention benchmark. **263 paired tasks** (a *should-act* + a *should-abstain* variant) across **42 executable MCP sandboxes** and **541 tools** (246 lookup / 119 verify / 176 commit — only commit tools mutate state). Each pair is a *controlled perturbation* of one of three dimensions (query, environment state, or tool inventory/runtime).
- **Taxonomy:** 8 scenarios across two phases — *pre-execution* (S1 missing parameter, S2 ambiguous action, S3 conflicting constraints, S4 high-stakes action, S5 insufficient tools) and *runtime* (S6 tool failure, S7 conflicting evidence, S8 emergent risk). Successful abstention is *strictly* an explicit refuse/ask at termination — silently halting, or acting then hedging, does not count.
- **AbstainGen:** a pipeline that *synthesizes* environments + paired tasks end-to-end (seeds from 8 upstream benchmarks, deterministic DAG replay validation, cross-family LLM critics); 94–98% of a stratified 100-task sample rated well-designed by 3 human annotators.
- **Headline result:** best of **17 frontier models** (Gemini 3.1 Pro) reaches only **59.5% paired accuracy**; 13/17 are below 50%. Abstention capability scales **largely independently** of task-solving. A signature failure is **post-hoc abstention** (commit, then claim refusal).
- **What it does *not* provide:** no released *training* pipeline. It is a benchmark + generation method, not a post-training recipe.

### 2.3 When2Tool — the decision *can* be read from hidden state
- **Paper:** Sun, Liu, Yan, Wang, Weng, *"LLM Agents Already Know When to Call Tools — Even Without Reasoning"*, arXiv:2605.09252. [Code](https://github.com/Trustworthy-ML-Lab/when2tool).
- **What it is:** 18 environments, 1,080 train / 2,700 test tasks, three tool-necessity categories (**A** computational scale, **B** knowledge boundary, **C** execution reliability) × 3 difficulty levels.
- **Key finding:** tool necessity is **linearly decodable** from the last-token hidden state (AUROC **0.89–0.96** across six models), far exceeding the model's own verbal reasoning. Their **Probe&Prefill** method cuts tool calls **48%** for **1.7%** accuracy loss.
- **Relevance to us:** it *mechanistically explains why* preference optimization on "should I call?" should work — the signal is already in the model; training is about surfacing it, not creating it. It also gives us a cheap *diagnostic* we can optionally run (does a linear probe on our LoRA model's hidden state recover the act/abstain decision?).

### 2.4 ToolFailBench — the failure-mode taxonomy we mirror
- **Paper:** Soni, *"ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents"*, arXiv:2607.04686 (ICML 2026 AIWILD/FAGEN workshops). [Traces](https://huggingface.co/datasets/SoHarshh/toolfailbench-traces).
- **What it is:** 1,000 tasks, 5 domains (finance/medicine/law/cybersecurity/real-estate), each domain 150 tool-required + 50 no-tool control. Tool-required tasks are "parametric traps" (the mock tool returns a value contradicting a plausible memorized value).
- **Taxonomy (per-response, single label):** tool-required → *Clean Tool-Use Rate (CTUR)*, **Tool-Skip (TS)**, **Result-Ignore (RI)**, **Output-Fabrication (OF)**; control → **Unnecessary-Tool-Use (UTU)**, *Wrong-Answer*. Hybrid rule-based + two-LLM-judge labeling (Fleiss' κ=0.693).
- **Headline:** best model **86.33% CTUR**; Llama-3.1 shows an "always-call" hyperactive pattern, differing from Qwen2.5-72B by **89 points** on control-task accuracy.
- **Relevance to us:** its *single-label, per-response* failure taxonomy is the cleanest template for our "five decision classes" (see `05-data-plan.md`). We adopt its spirit but push to **fully rule-based labels** (no LLM judge) for the core metric.

### 2.5 SimpleToolHalluBench + the Reasoning Trap — the reliability–capability trade-off
- **Paper:** Yin et al., *"The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination"*, arXiv:2510.22977. [Code](https://github.com/albert-y1n/Reasoning_Trap). **SimpleToolHalluBench** is its diagnostic benchmark: two tasks — **NTA** (no tool available, query still demands one → does it fabricate?) and **DT** (distractor tool present → does it wrongly use/hallucinate?).
- **Key findings:** reasoning RL *increases* tool hallucination proportionally to task gains; it's not overfitting; it's method-agnostic; and the mitigation (prompting / DPO) reveals a **fundamental reliability–capability trade-off** — reducing hallucination consistently degrades utility.
- **Relevance to us:** this is the single most important *prior* on our core hypothesis. It predicts our result **will** show a trade-off, and tells us to measure it honestly (see success criteria: "tool-call accuracy held within ≤ ~3 points"). Our contribution is measuring it *cleanly at small scale* and comparing methods, not just noting it exists.

### 2.6 AgentProp-Bench & FAIL-TaLMs — supporting evidence
- **AgentProp-Bench** (Gurram et al., *"Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation"*, arXiv:2604.16706, TMLR under review; [code](https://github.com/bhaskargurram-ai/agenthallu-bench)): 2,000 core + 300 retail tasks, controlled parameter injection. Shows **fabricated tool use** ranges **5–100%** across models (Gemini-2.0-Flash fabricates results in 37.5% of traces), and that rejection/recovery are *independent* capabilities.
- **FAIL-TaLMs** (Ngai et al., NAACL 2025, [aclanthology](https://aclanthology.org/2025.naacl-long.149/)): 1,749 examples, 906 tools; studies under-specified queries and non-available tools — the two "should not call" conditions we reuse.

---

## 3. Training methods (what we actually reimplement)

### 3.1 K-DPO — teach the model to trust its knowledge
- **Paper:** Zeng et al., *"The Tool-Overuse Illusion: Why Does LLM Prefer External Tools over Internal Knowledge?"*, arXiv:2604.19749 (HIT + Huawei + PKU).
- **Diagnosis:** models make **0.93 unnecessary tool calls/query**; tool use on internally-solvable questions *drops* accuracy **3.29–14.48%** (tool context = noise); even in high-knowledge regions (avg@1024 > 0.8) Qwen3-8B averages 2.2 calls.
- **Method:** **K-DPO** constructs preference pairs rewarding *correct answer + minimal tool use* over *correct answer + excessive tool use*, aligning perceived vs. actual knowledge boundary.
- **Result:** **82.8%** overuse reduction (32B) with **~3%** accuracy *improvement*.
- **Complementary finding:** outcome-only RLVR rewards *incentivize* tool use (+65% calls); a **balanced reward** (correctness + efficiency) cuts unnecessary calls **66.7% (7B) / 60.7% (32B)**.
- **Why it matters:** this is the *methodological* piece we generalize. K-DPO targets one abstain class (answer-from-knowledge); we extend the preference construction to *all four* abstain classes.

### 3.2 RPO — reward-aware preference optimization on negative examples
- **Paper:** same When2Call paper (Ross et al. 2025). **RPO** trains on *negative* behaviors (wrong call, fabricated call, missed follow-up) as well as positive ones, in a reward-aware preference objective.
- **Result:** +8.6% When2Call accuracy over SFT; RPO is the reference *preference* method we reimplement at ≤1.5B (see `06-training-plan.md`, where "RPO" in our SFT/DPO/ORPO/**RPO** comparison refers to this, and K-DPO is a variant of the DPO/RPO family).

### 3.3 The comparison we run (and why it's the contribution)
- **SFT** (positive-only baseline — the crowded default),
- **DPO** (paired preference, negative abstain examples),
- **ORPO** (reference-free, no separate SFT stage),
- **RPO** (reward-aware, When2Call's method),
- **K-DPO** (knowledge-aware variant, Tool-Overuse Illusion's method).

Identical data, identical model, identical seeds → the difference *is* the method. None of the papers above published this head-to-head at small scale on a unified taxonomy.

---

## 4. Generic tool-use training stacks (ignore the abstain dimension)

| Stack | What it gives | Gap |
|---|---|---|
| **ToolBrain** (2024) | Annotated tool-use preferences + DPO data | No "don't call" class; action-optimized |
| **RLFactory** (2025) | Verifiable-reward RL for tool use | Rewards *task success*, not correct abstention |
| **Hugging Face TRL** | DPO/ORPO/KTO trainers | Objective-agnostic; no abstention data or taxonomy |
| **MLX** (`mlx_lm.lora`, `mlx_lm.dpo`) | On-device SFT + DPO | Objective-agnostic; we build the data + eval around it |

---

## 5. Synthesis: the gap

| Existing asset | Provides | Does **not** provide |
|---|---|---|
| When2Call | 4-way benchmark + SFT/RPO data + eval scripts | Training code; ≤1.5B models; distinguishes "answer from knowledge" from "hallucinate" |
| AgentAbstain | Paired act/abstain taxonomy, 263 tasks, AbstainGen | Any released training pipeline |
| When2Tool | Necessity-decodability + Probe&Prefill | Post-training recipe |
| ToolFailBench / SimpleToolHalluBench / AgentProp-Bench / FAIL-TaLMs | Diagnostic failure taxonomies | No training |
| K-DPO / Tool-Overuse Illusion | Method for "trust internal knowledge" | No open small-model reproduction; single abstain class |
| RPO (When2Call) | Method for negative-example preference opt | No unified, method-comparative small-scale reproduction |
| ToolBrain / RLFactory / TRL | Generic tool-use post-training | Ignore the abstain dimension entirely |

**Our gap-fill, stated once:** a single, clean, laptop-runnable pipeline that (a) unifies When2Call's "can't answer / no tool" abstention with K-DPO's "answer from knowledge" abstention into **one five-class taxonomy**, (b) publishes **paired, verifiable, rule-labeled** data, and (c) ships the **training code + method comparison** (SFT/DPO/ORPO/RPO/K-DPO) the papers omit.

---

## 6. Citation index (quick reference)

| Short name | Reference | Venue | arXiv |
|---|---|---|---|
| When2Call | Ross, Mahabaleshwarkar, Suhara | NAACL 2025 | 2504.18851 |
| RPO | (same as When2Call) | NAACL 2025 | 2504.18851 |
| AgentAbstain | — | arXiv 2026 | 2607.10059 |
| Tool-Overuse Illusion / K-DPO | Zeng et al. | arXiv 2026 | 2604.19749 |
| When2Tool | Sun, Liu, Yan, Wang, Weng | arXiv 2026 | 2605.09252 |
| ToolFailBench | Soni | ICML 2026 W/S | 2607.04686 |
| Reasoning Trap / SimpleToolHalluBench | Yin et al. | arXiv 2025 | 2510.22977 |
| AgentProp-Bench | Gurram et al. | TMLR (u.r.) | 2604.16706 |
| FAIL-TaLMs | Ngai et al. | NAACL 2025 | — |
| BFCL | Yan et al. | ICLR 2024 | 2402.09712 |
| τ-bench | Yao et al. | — | 2406.12045 |
| ToolEval | Qin et al. | ACL 2024 | 2307.16789 |
| ToolBrain | — | 2024 | 2410.18559 |
| RLFactory | — | 2025 | 2505.10032 |

> Final arXiv IDs / venues for the last four rows should be re-verified at write-up time; the ones with live links above were confirmed during planning.
