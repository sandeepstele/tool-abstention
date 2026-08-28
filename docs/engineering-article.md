# Reward accuracy is not agent reliability

## Engineering a small language model that knows when not to call tools

Function calling is usually framed as selection: given a request and functions,
choose the right function and arguments. Production agents face a harder upstream
question. Should they invoke a function at all?

A model that calls a valid function for every request may still be unsafe and
expensive. It can search for information already present, invent an unavailable
capability, act without a required parameter, or repeat an action whose goal is
already true. This project turns those cases into explicit, testable behavior.

## From a binary label to a behavior contract

“Abstain” is not one response. The correct behavior may be to answer directly,
request missing information, refuse an unavailable capability, or do nothing when
the requested state already holds. I represented the decision as five classes:

- `CALL`: invoke a visible tool with schema-valid arguments;
- `ANSWER`: respond without a tool when the answer is available;
- `CLARIFY`: request a required missing slot;
- `REFUSE`: identify an unavailable capability;
- `NOOP`: recognize that no state change is needed.

Every record is a strict immutable contract. Tool definitions use Draft 2020-12
JSON Schema. Expected calls are validated against the visible function schema.
Unknown fields, duplicate tools, invalid IDs, non-finite values, and contradictory
prediction states fail closed.

The dataset uses controlled pairs. An act task and its abstain counterpart share a
domain and intent while one declared capability condition changes. This makes
paired accuracy meaningful: the model must handle both sides rather than exploit a
topic cue.

## Keeping the evaluator honest

Exact-string grading is brittle, but an unbounded model judge would make the core
measurement expensive and difficult to reproduce. The evaluator instead parses
plain, OpenAI-style, and Qwen-style calls and applies class-specific semantic rules.
It reports semantic correctness, predicted behavior, protocol validity, paired
accuracy, and hallucinated-tool rate separately.

That separation mattered. A blinded, balanced 60-item calibration packet was
owner-adjudicated before training conclusions were accepted. All 60 final labels
agreed with the calibrated evaluator. Later experiments showed that protocol
compliance can remain perfect while decision behavior fails. Valid syntax is
necessary, but it is not agent reliability.

## SFT worked—and varied by seed

The primary model was a pinned 4-bit Qwen2.5-1.5B-Instruct checkpoint trained with
LoRA on Apple Metal. The base model scored 62.50% on the 120-task internal
validation set. Across three SFT seeds, mean validation accuracy reached 94.72%
with a 0.96% sample standard deviation. Mean act and abstention accuracy were
95.00% and 94.44%.

Seed variation was operationally important. Seed 0 was unusually favorable,
especially on external abstention. Reporting only that run would overstate the
result. The final analysis therefore includes three-seed sample statistics and
deterministic paired bootstrap comparisons.

On the same internal examples, SFT seed 0 improved accuracy over the base model by
33.33 percentage points, with a paired-bootstrap 95% interval of 25.00 to 41.67
points. On 640 non-overlapping BFCL decision records, its gain was 5.16 points,
with a 2.97 to 7.34 point interval.

## External data was a test, never a teacher

BFCL and AgentAbstain were added with immutable revisions, licenses, source hashes,
original IDs, transformations, and usage restrictions. A leakage detector compares
external queries with every internal split using exact normalization, character
five-gram Jaccard, and sequence similarity. Overlaps are quarantined.

The prepared BFCL slice contains 400 simple CALL and 240 irrelevance ABSTAIN
records. BFCL was never used for training, hyperparameter selection, checkpoint
selection, or retry decisions. AgentAbstain remains cataloged in its native
multi-turn environment because flattening it would erase benchmark semantics.

## Why I implemented DPO instead of pretending it existed

The pinned MLX-LM release did not provide a DPO trainer. The project therefore owns
the numerical boundary: shared-prompt tokenization, completion-only masks, sequence
log probabilities, reference-adjusted DPO loss, label smoothing, and reward
metrics. Fixed-vector NumPy tests cover the math before Metal execution.

Reference log probabilities are precomputed once from the frozen SFT adapter. Each
cache entry binds the model revision, adapter, tokenizer and prompt, example,
completion boundaries, and token counts. Training rejects stale, missing,
duplicate, non-finite, or mismatched entries. Only LoRA parameters are trainable.

This design uses one policy model during optimization instead of keeping a second
reference model in memory. It also makes the reference calculation independently
auditable.

## The metric that lied

The 1.5B DPO run was numerically clean. Losses and gradients were finite, reload was
exact, and preference reward accuracy reached 100%. Yet free-generation act
accuracy fell from 100% to 0%, while overall internal accuracy fell to 42.50%.

The optimizer learned to separate chosen and rejected completions under teacher
forced scoring. It did not preserve the policy’s willingness to emit tool calls
during generation. A preference metric that looked perfect concealed a useless
agent.

I tested bounded explanations rather than launching an open-ended sweep. Mean-log
probability normalization removed length-sum bias. Pair selection was corrected to
balance domains and all four abstention classes. A competent 0.5B SFT initializer
replaced an inadequate smoke adapter. Finally, DPO was combined with a
chosen-completion supervised anchor at two predeclared weights.

Every candidate still failed its frozen behavior gates. The anchors reduced the
collapse but did not preserve enough act accuracy. The branch was stopped, the
adapters were rejected, and BFCL was skipped whenever internal promotion failed.

## What this demonstrates as an engineering project

The useful artifact is not merely a fine-tuned adapter. It is a system that can
reject an attractive but broken experiment:

1. contracts make malformed data impossible to ignore;
2. leakage rules keep benchmarks out of training;
3. numerical tests validate the optimization implementation;
4. generation-based gates prevent proxy metrics from promoting bad agents;
5. per-example outputs and hashes make conclusions replayable;
6. negative results remain visible instead of being rewritten as success.

The public release is verifiable with `make release`. That CPU-only command
rebuilds the statistical report, runs 210 tests with strict lint and typing, and
checks the hash chain for the canonical artifacts and their 30 source files.

## Limitations and next experiment

The internal data is synthetic, templated, and limited to three domains. BFCL tests
the external decision boundary but exact argument accuracy is outside this slice.
Three seeds reveal variance without tight population estimates. The study covers
one primary model family on local hardware.

The next credible experiment is not another small hyperparameter sweep. It needs a
larger and more linguistically diverse licensed training corpus, plus a conservative
objective whose checkpoints are selected by free-generation behavior. Until then,
the defensible result is straightforward: SFT substantially improved tool
abstention, while the tested preference objectives optimized their proxy and broke
the agent behavior they were supposed to improve.
