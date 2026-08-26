# 11 — Implementation Plan

> The execution contract for turning the research plan into a working MLE
> project. This document defines build order, interfaces, verification, and the
> evidence required before a milestone is called complete.

## 1. Delivery principle

Build a narrow, tested vertical slice before expanding breadth:

1. one domain,
2. all five decision classes,
3. executable mock tools,
4. deterministic task generation,
5. rule-based evaluation,
6. base-model inference,
7. only then training and additional domains.

This order exposes taxonomy and evaluator defects before expensive experiments.
The first recruiting-quality release contains **base, prompt-only, SFT, and DPO**.
ORPO, RPO, K-DPO, and the linear probe are extensions, not blockers for v1.

## 2. Definition of done

The first complete release must provide all of the following:

- `make setup`, `make data`, `make test`, `make eval`, and `make train` have
  documented behavior and useful failure messages.
- At least three domains and all five classes are represented.
- Every generated pair differs along exactly one declared perturbation.
- Every `CALL` example executes against a deterministic mock tool.
- Every free-form class has an explicit output-validation contract.
- Base, prompt-only, SFT, and DPO use the same held-out test set and inference
  settings.
- Final 1.5B experiments use three seeds and report mean and standard deviation.
- Results include raw predictions, aggregate metrics, per-class errors, and a
  manually reviewed evaluator-calibration sample.
- A clean clone can reproduce data and evaluation from pinned configuration.
- `README.md` contains actual results and limitations, not projected claims.

## 3. Planned implementation sequence

### Milestone A — repository foundation

**Build**

- `pyproject.toml` with a supported Python range and separated core, training,
  and development dependencies.
- `uv.lock`, `Makefile`, `.gitignore`, `LICENSE`, and package skeleton.
- `configs/` with schema-validated YAML files.
- `src/tool_abstention/util/` for seeding, canonical JSON serialization,
  SHA-256 hashing, and JSONL I/O.
- CI running lint, type checks, and unit tests on CPU.

**Decisions to make during implementation**

- Confirm current MLX and Transformers APIs before pinning dependencies.
- Decide whether training is MLX-only or whether a PyTorch/TRL path is supported.
- Choose one configuration library; do not mix ad-hoc YAML access throughout the
  codebase.

**Acceptance checks**

- Fresh environment installation succeeds from the lockfile.
- `python -m tool_abstention --help` exits successfully.
- Seed and hash utilities pass fixed-vector tests.
- CI passes without model downloads.

### Milestone B — taxonomy and record contracts

**Build**

- A `DecisionClass` enum: `CALL`, `ANSWER`, `CLARIFY`, `REFUSE`, `NOOP`.
- Typed records for tool schemas, environment state, task pairs, expected
  behavior, predictions, and evaluation results.
- JSON Schema export for generated artifacts.
- A validator that rejects incomplete or contradictory records.

**Required record fields**

Each task must include:

- stable `id`, `pair_id`, `domain`, split, variant, and generator version;
- query, visible tool inventory, and visible environment context;
- label and perturbation type;
- expected tool name and normalized arguments for `CALL`;
- expected answer or answer validator for `ANSWER`;
- missing slot names for `CLARIFY`;
- unavailable capability/reason code for `REFUSE`;
- pre-satisfied state and allowed response markers for `NOOP`.

**Acceptance checks**

- The five labels are mutually exclusive.
- Invalid combinations fail with specific validation errors.
- Serialization round-trips without information loss.
- Tests cover at least one valid and invalid record per class.

### Milestone C — first vertical slice: productivity domain

Productivity is first because it naturally exercises calls, missing arguments,
removed tools, and already-completed state without relying on changing real-world
knowledge.

**Build**

- Tools such as `search_contacts`, `create_event`, `close_ticket`, and
  `send_email` using in-memory deterministic state.
- Hand-written task templates and seeded entity generation.
- All four act-to-abstain perturbations.
- Approximately 40 development pairs, intentionally small enough to inspect.
- A human-readable audit command that prints pair diffs and labels.

**Acceptance checks**

- Same seed produces byte-identical JSONL and manifest hashes.
- Each pair has one act and one abstain member.
- The pair-diff validator detects any undeclared second perturbation.
- Every `CALL` executes and matches the expected result.
- All 40 pairs receive manual inspection recorded in the worklog.

### Milestone D — deterministic evaluator

**Build**

- Parsers for the model's native function-call format and a canonical internal
  call representation.
- Class-specific validators rather than a single loose response regex.
- Accuracy, paired accuracy, macro-F1, per-class precision/recall/F1,
  abstention accuracy, act accuracy, and tool-hallucination rate.
- Confusion matrices and per-example error records.

**Important correctness rule**

`ANSWER` must not mean merely “non-empty text.” It must compare against an exact,
normalized, set-valued, numeric-tolerance, or domain-specific deterministic
validator declared by the task. `REFUSE`, `CLARIFY`, and `NOOP` must each check
both the absence of a tool call and evidence of the expected intent.

**Calibration**

- Create a frozen set of at least 200 hand-labeled outputs, including adversarial
  overlaps and incorrect answers.
- Measure evaluator agreement against those labels.
- Investigate every disagreement; report class-wise agreement in addition to the
  overall target of at least 95%.

**Acceptance checks**

- Metrics match hand-computed fixtures.
- Tool syntax errors cannot be misclassified as abstention success.
- Incorrect direct answers fail `ANSWER` validation.
- Calibration results and known limitations are committed.

### Milestone E — data expansion and split hygiene

**Build**

- Finance and weather/geo domains after the first domain passes its gate.
- A target of 300–600 pairs, adjusted after measuring template diversity.
- Grouped splitting by `pair_id`, template family, and semantic entity pattern.
- Exact and normalized near-duplicate detection across splits.
- Dataset cards containing class/domain distributions and generation limitations.

**Acceptance checks**

- No pair or template family crosses a split boundary.
- No class is below 10% overall; `CALL` exists in every domain.
- Test data is frozen before training begins.
- A random stratified sample is manually reviewed and logged.

### Milestone F — baseline inference

**Build**

- Model adapter interface returning raw text, parsed calls, token counts, latency,
  and errors.
- Qwen chat-template and tool-schema formatting verified against the pinned model.
- Greedy base-model and prompt-only runs.
- Raw predictions saved before classification so evaluator changes can be replayed
  without rerunning inference.

**Acceptance checks**

- Identical inference settings across all methods.
- Resume support does not duplicate or skip examples.
- Base scores are plausible and manually spot-checked.
- At least 25 failures are categorized and discussed before training.

### Milestone G — SFT

**Build**

- Training-example formatter covering all five response types.
- LoRA training wrapper, checkpoint metadata, resume behavior, and adapter export.
- Training curves and validation metrics logged locally in JSONL.
- A tiny-model smoke test; large model training stays outside CI.

**Acceptance checks**

- A 10–20-example overfit test succeeds, proving the training path learns.
- Adapter loading produces different outputs from the base model on known cases.
- No test example is consumed during training or model selection.
- SFT is evaluated through the identical baseline harness.

### Milestone H — DPO

**Build**

- Preference pairs with explicit negative-type metadata.
- Validation preventing identical chosen/rejected responses and malformed calls.
- DPO initialized from the frozen SFT checkpoint.
- Small beta sweep on the iteration model, then one locked configuration for the
  final model.

**Acceptance checks**

- Preference data distribution is published by class and negative type.
- Chosen/rejected reward margin is logged during training.
- DPO beats or clearly fails to beat SFT under the predeclared metric; either
  outcome is retained.
- Act-accuracy regression is reported, never hidden by aggregate accuracy.

### Milestone I — final experiment and release

**Build**

- Three-seed SFT and DPO final runs on the 1.5B model.
- Bootstrap confidence intervals or paired significance tests over task pairs.
- Auto-generated `results/summary.md`, plots, and reproducibility manifest.
- A concise demo and a README section explaining one representative success and
  one representative failure.

**Acceptance checks**

- Every reported number links to raw predictions, configuration, data hash, code
  version, and adapter metadata.
- Results regenerate from the documented commands within stated numeric tolerance.
- Claims are rewritten to match observed evidence.
- Known limitations cover synthetic data, single-turn scope, evaluator boundaries,
  small-model generality, and hardware-specific training.

### Milestone J — optional research extensions

Only start these after the recruiting-quality v1 release:

- ORPO comparison;
- RPO negative construction;
- K-DPO knowledge-focused pairs;
- unified-versus-per-class ablation;
- hidden-state linear probe;
- 3B scaling experiment;
- external benchmark transfer evaluation.

Each extension needs its own hypothesis, config, baseline, compute estimate, and
worklog entry before work begins.

## 4. Interface boundaries

The implementation should preserve these boundaries:

```text
domain templates -> validated task records -> immutable dataset artifacts
                                           -> training formatters
model adapter -> raw prediction records -> deterministic evaluator -> metrics
config + data hash + code version -> run identity -> checkpoints and results
```

- Domain generators do not import model code.
- Evaluators consume stored predictions and can be rerun independently.
- Training formatters never mutate canonical dataset records.
- Metrics never infer ground truth from model prose.
- Experimental configuration contains all behavior-changing parameters.

## 5. Documentation protocol

Every work session that changes the project must update `WORKLOG.md` in the same
change. Each entry records:

- date and objective;
- files added, modified, or removed;
- implementation decisions and alternatives considered;
- commands/tests run and their exact outcomes;
- generated artifacts and hashes when applicable;
- unresolved issues and the next concrete action.

Additional documentation rules:

- User-facing setup or command changes update `README.md` immediately.
- Architecture or interface changes update the relevant numbered document.
- Every experiment writes a machine-readable event record; the worklog links to
  it instead of copying large metric payloads.
- Failed experiments are logged with the same care as successful ones.
- Secrets, machine-specific credentials, private paths, and large generated data
  are never copied into documentation.
- “Passed” is not sufficient: record the command and test count or key output.

## 6. Immediate next actions

The next implementation session should perform only Milestone A:

1. initialize repository metadata if desired by the owner;
2. create the package/config/test skeleton;
3. choose and pin the initial dependency set after API verification;
4. add seed, hashing, JSONL, and configuration primitives;
5. establish CPU-only CI;
6. update the worklog with commands and observed results.

No task generation should begin until the foundation tests pass.
