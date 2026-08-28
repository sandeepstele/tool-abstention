# Worklog

This is the human-readable engineering journal for the project. It complements
machine-readable experiment logs and Git history. New entries are added in reverse
chronological order and follow the protocol in
[`docs/11-implementation-plan.md`](docs/11-implementation-plan.md#5-documentation-protocol).

## 2026-08-26 — Begin Milestone F local Qwen baseline inference

### Objective

Install a separately locked MLX inference stack, implement resumable greedy
inference, download a revision-pinned 0.5B Qwen model, and complete a representative
eight-task Metal smoke run before spending compute on the validation split.

### Changes

- Added an `inference` dependency group pinned by `uv.lock`: MLX-LM 0.29.1, MLX
  0.32.2, MLX Metal 0.32.2, and Transformers 5.16.1.
- Pinned `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at revision
  `53a32aee5e9447773fd2b85988395066aef3700a` with greedy decoding and 256 tokens.
- Added a lazy MLX backend, stable system/environment/tool prompt formatting,
  durable per-record append, resume validation, persisted backend errors, token and
  latency accounting, peak Metal memory, and deterministic run manifests.
- Added stratified smoke selection: one complete pair for each abstention class.
- Added `infer`, `baseline-smoke`, fake-backend tests, provenance tests, and partial
  stored-prediction evaluation support.
- Added optional peak-memory metadata to the prediction contract and updated its
  fixed JSON Schema hash.

### Attempts, failures, and fixes

- Verified installed APIs before coding: `load(..., revision=...)` and
  `generate(model, tokenizer, prompt=...)` are supported by MLX-LM 0.29.1.
- Attempt 1 downloaded the approximately 290 MB model but all eight predictions
  persisted `BatchEncoding` errors. Transformers 5 returned a `BatchEncoding` for
  `tokenize=True`; changed the chat template to `tokenize=False` and passed text to
  MLX-LM, matching its current documented path.
- Attempt 2 ran successfully but selected only `ANSWER` pairs because validation is
  class ordered. Added tested stratified pair selection rather than presenting the
  sample as representative.
- Attempt 3 completed the stratified run but predated run-manifest and peak-memory
  capture. Added both and reran from a fresh output directory.
- Diagnostic attempt directories were preserved during debugging, then moved out of
  the repository before commit; their failure modes and fixes are recorded here.

### Final smoke result

- Eight tasks / four complete pairs; all four abstention classes represented.
- Zero inference errors; mean latency 246.62 ms, median 243.20 ms, peak Metal memory
  0.8591 GB; 5,711 input and 312 output tokens total.
- Accuracy 0.0, paired accuracy 0.0, act accuracy 0.0, abstention accuracy 0.0,
  macro-F1 0.1455, tool-hallucination rate 0.75.
- Seven outputs attempted tools. Most emitted malformed double-braced Qwen tool JSON;
  one abstention response asked for a ticket ID already visible in the request.
- The pinned tokenizer exposes tool delimiters through its template but MLX-LM did
  not infer a `tool_parser` for this older Qwen2 template. The raw malformed output
  remains a failure; no brace-stripping heuristic was added to inflate the score.
- Prediction hash `b7c77925...b444d9`; prompt-policy hash `9c461034...d4dcb`;
  task-selection hash `fc9a0984...dd570`.

### Verification and next action

- Definitive CPU suite before the final run: 129 tests passed, Ruff clean, strict
  mypy clean across 31 files, and 96.15% coverage.
- The negative smoke result is reportable evidence that the infrastructure works and
  the 0.5B model strongly over-calls/malforms tools under the current prompt.
- Next: characterize the tokenizer-native tool parser and prompt schema on a tiny
  diagnostic set, then run the full validation split only after deciding whether
  malformed double braces are model behavior or a correctable template mismatch.

## 2026-08-26 — Implement Milestone E multi-domain dataset

### Objective

Expand the executable corpus to finance and weather/geo, reach 300 paired tasks,
create leakage-safe deterministic splits, and freeze test identity before inference.

### Changes and decisions

- Expanded productivity to 25 examples per abstention class and added 25 finance
  and 25 weather/geo examples per class: 300 pairs / 600 tasks total.
- Added four tools per new domain and deterministic execution for every generated
  `CALL` expectation.
- Added template-family grouping with exact 60/20/20 pair splits, normalized-query
  leakage rejection, duplicate-pair rejection, and test-set hashing.
- Added deterministic train/validation/test JSONL, a dataset card, and a provenance
  manifest with artifact hashes and class/domain distributions.
- Changed `make data` to build the complete corpus and added the
  `generate-dataset` CLI command.
- Kept generated artifacts ignored; only source/configuration is committed.
- Marked the 300-pair v1 pipeline complete while retaining the original 600-pair
  roadmap target as an explicit later expansion rather than overstating progress.

### Commands, failures, and verification

- Ruff formatted the new domain modules; strict mypy identified invariant JSON
  container types, which were corrected with explicit `JsonValue` annotations.
- The first full build stopped on normalized-query leakage because finance FX
  prompts were repeated across splits. Added deterministic unique currency pairs;
  the unchanged leakage guard then passed.
- The first regression run exposed an outdated 10-contact test and missing new-code
  coverage. Updated the capacity check and added multi-domain, executor, split,
  leakage, determinism, manifest, and dataset-card tests.
- Definitive `make check`: 42 files formatted, Ruff clean, strict mypy clean across
  29 source files, 120 tests passed, 98.79% coverage.
- Generated counts: train 360 tasks, validation 120, test 120.
- Hashes: train `df308fe4...da2b9`, validation `563b28c9...790c8`, frozen test
  `76bbac17...59bc8`, manifest `62a6797f...3311c`.

### Open issues and next action

- The corpus remains synthetic and template-generated; real-model failure analysis
  must guide any further expansion.
- Human calibration of 200 real outputs remains required.
- Next: baseline inference adapter, pinned small Qwen model, greedy local inference,
  raw prediction persistence, evaluator replay, and 200-output human calibration.

## 2026-08-26 — Implement Milestone D deterministic evaluator

### Objective

Implement replayable, judge-free scoring before any model inference: parse stored
outputs, classify five behaviors, validate correctness, compute paired metrics, and
regression-test the entire repository.

### Changes

- Added parsing for canonical JSON, wrapped tool calls, single-item OpenAI
  `tool_calls`, direct function objects, and Qwen-style `<tool_call>` blocks.
- Added explicit malformed-tool-attempt detection so broken calls cannot pass as
  direct answers or abstention.
- Added deterministic text normalization and correctness checks for exact,
  normalized-text, numeric-tolerance, set-valued, clarification, refusal, and no-op
  behavior.
- Added inference-error and task-ID consistency handling with machine-readable
  reason codes on every evaluation.
- Added accuracy, strict paired accuracy, macro-F1, per-class precision/recall/F1,
  act accuracy, abstention accuracy, and abstention-denominated tool hallucination.
- Added a stored-prediction harness that writes canonical `evaluations.jsonl` and
  `metrics.json`, allowing evaluator changes without rerunning inference.
- Added the `evaluate` CLI command and documented its use.
- Added a deterministic 200-case adversarial construction matrix spanning all five
  classes, plus parser, validator, metric, harness, and CLI tests.

### Decisions and rationale

- **Malformed tool attempts count as `CALL` failures.** Treating them as prose would
  undercount hallucination and inflate abstention.
- **Prediction storage remains separate from evaluation.** Raw inference can be
  replayed after a rule fix without consuming model compute again.
- **Text classification uses task-aware structural requirements.** A clarification
  must be interrogative and mention the declared missing slot; generic hedging does
  not qualify.
- **Domain validators fail closed for now.** No arbitrary validator ID is executed
  until a reviewed registry is implemented.
- **Calibration claims remain conservative.** The 200-case construction matrix is a
  deterministic regression suite, not a substitute for human labeling real model
  outputs; the ≥95% human-agreement gate remains open.

### Commands and outcomes

- Ruff initially requested evaluator and test formatting; all required layouts were
  applied without weakening rules.
- Strict mypy found float-compatible ratio typing and an unannotated calibration
  collection; both were made explicit.
- The first expanded suite found one incorrect test assumption: direct
  `{name, arguments}` inside a single `tool_calls` item is intentionally supported.
  The case was moved to the supported-format matrix.
- Final `make check` passed: 39 files formatted, Ruff clean, strict mypy clean across
  26 source files, and 116 tests passed with 99.22% coverage.
- `uv sync --locked` checked all 27 installed packages.
- Regenerated the productivity dataset and confirmed byte-identical hashes:
  `f15950c3...e4c904` for tasks and `78f967ce...f261c` for the manifest.
- Exported all four public schemas into a temporary directory and confirmed the
  expected file count; `git diff --check` also passed.

### Open issues

- The required 200-output human calibration must be performed on actual baseline
  model responses before evaluator agreement is reported.
- Domain answer-validator execution remains intentionally disabled.
- Model-specific streaming/multi-call parsing is outside this single-call v1 scope.

### Next action

Implement Milestone E: add finance and weather/geo domains, expand template
diversity, create grouped leakage-safe train/validation/test splits, freeze the test
set, and produce the full dataset card and manifest. Baseline local inference follows.

## 2026-08-26 — Implement Milestone C productivity vertical slice

### Objective

Build the first executable dataset slice: 40 deterministic productivity pairs
covering all four abstention transformations, with artifact hashing and a complete
human-readable audit. Do not download or run a language model.

### Changes

- Added four Draft 2020-12 productivity tools and deterministic executors:
  `search_contacts`, `create_event`, `close_ticket`, and `send_email`.
- Added a strict seeded generator configuration and `make data` target.
- Added ten pairs per abstention class, producing 40 pairs / 80 task records.
- Added semantic pair-diff validation: query-only for `ANSWER`/`CLARIFY`,
  tool-inventory-only for `REFUSE`, and environment-only for `NOOP`.
- Made generation execute every `CALL` and compare the result with its declared
  expected result before writing artifacts.
- Added deterministic `tasks.jsonl` and `manifest.json` generation with config and
  content hashes; timestamps and absolute paths are excluded from compute artifacts.
- Added pair reconstruction, duplicate/incomplete-pair detection, and a complete
  audit renderer.
- Added `generate-productivity` and `audit-pairs` CLI commands.
- Added 11 generator, executor, artifact, semantic-validation, and CLI tests.
- Anchored generated-data ignore rules to the repository root so the checked-in
  `configs/data/` directory is not accidentally ignored.

### Decisions and rationale

- **Four template families map one-to-one to abstention classes.** This makes the
  controlled perturbation visible and testable before adding linguistic variety.
- **Executors operate on deep copies.** Test fixtures and expected state cannot be
  mutated by execution order.
- **Manifests omit build time.** The same config and seed produce byte-identical
  task and manifest artifacts.
- **Development slice stays in the train split.** Final grouped splits belong to
  the multi-domain expansion milestone and must not be implied by this audit set.

### Commands and outcomes

- Initial static checks requested Ruff formatting, import ordering, and line-length
  changes; all were applied without weakening rules.
- Strict mypy caught invariant `dict` value types and string-versus-enum test
  inputs; explicit JSON-value and `DatasetSplit` types fixed them.
- `make check` passed 78 tests with 99.27% coverage, clean Ruff, and strict mypy
  across 21 source files before the final audit correction.
- `make data` generated 40 pairs / 80 tasks locally using CPU only.
- Printed and inspected all 40 pairs. Every pair showed exactly its declared
  semantic dimension; the audit exposed doubled punctuation in all clarification
  prompts, which was corrected and protected by a regression assertion.
- The same audit revealed `configs/data/` was hidden by an unanchored ignore rule;
  changed it to `/data/` so only generated root data is ignored.
- Regenerated and re-audited the corrected dataset: all 40 pair headers were
  present, clarification punctuation was clean, and the artifact hashes were
  `f15950c3...e4c904` for `tasks.jsonl` and `78f967ce...f261c` for the manifest.
- Definitive `git diff --check` and `make check` passed: 34 files formatted, Ruff
  clean, strict mypy clean across 21 source files, and 78 tests passed with 99.27%
  coverage.

### Open issues

- Templates are deliberately narrow and are not yet a general benchmark.
- The development slice has no train/validation/test split and must not be used for
  final reported metrics.
- The audit verifies construction quality, not natural-language evaluator quality;
  evaluator calibration remains Milestone D.

### Next action

Implement Milestone D: model-output parsing, class-specific deterministic judges,
metrics, adversarial calibration records, and stored raw prediction evaluation.

## 2026-08-26 — Implement Milestone B taxonomy and record contracts

### Objective

Freeze the strict public contracts shared by future data generation, training,
inference, and evaluation without implementing domain data or model behavior.

### Changes

- Added canonical enums for five decision classes, task variants, dataset splits,
  and the four act-to-abstain perturbations.
- Added immutable, unknown-field-rejecting Pydantic contracts for tools, task
  records, task pairs, expected behavior, parsed calls, predictions, and per-example
  evaluations.
- Modeled expected behavior as discriminated `CALL`, `ANSWER`, `CLARIFY`, `REFUSE`,
  and `NOOP` variants; added five deterministic answer-validator descriptions.
- Added Draft 2020-12 tool-schema validation and expected-call argument validation
  using the pinned `jsonschema` dependency.
- Added recursive finite JSON-value validation for environment state, arguments,
  expected results, and domain-validator parameters.
- Added deterministic JSON Schema export and JSON record validation APIs.
- Added `export-schemas` and `validate-record` CLI commands with concise errors.
- Added 46 contract/schema/CLI tests, bringing the suite to 66 tests.
- Updated README, architecture, and roadmap status to reflect implemented contracts.

### Decisions and rationale

- **Pydantic is the source of truth.** JSON Schemas are derived, canonicalized
  artifacts rather than independently maintained definitions.
- **Expected behavior is a discriminated union.** Invalid combinations cannot be
  represented as a bag of nullable fields.
- **Perturbations map to exactly one abstention label.** This prevents a generator
  from silently labeling a removed-tool example as `ANSWER`, for example.
- **Tool arguments are validated now.** Catching schema/argument contradictions at
  record construction keeps invalid `CALL` examples out of future datasets.
- **Semantic one-change pair comparison is deferred.** Milestone B validates pair
  identity and shared metadata; Milestone C will understand domain semantics.

### Commands and outcomes

- `uv lock` resolved 27 packages after adding `jsonschema`, then 28 after adding
  `types-jsonschema`; CPython 3.12.13 remained the selected interpreter.
- Initial format check reported required layouts in the CLI and record models;
  those layouts were applied without weakening formatting rules.
- Ruff then reported `RUF036` for union ordering; moved `None` to the end.
- Strict mypy reported missing jsonschema stubs; pinned `types-jsonschema` and mypy
  passed across 16 source files before tests were added.
- The first full contract `make check` requested two test formatting changes; they
  were applied manually.
- The next `make check` passed 66 tests with 98.94% coverage, clean Ruff, and strict
  mypy across 19 source files.
- Exported all four schemas into a temporary directory and recorded fixed SHA-256
  vectors in `test_schemas.py`; no generated schemas were added to the repository.
- Final `uv sync --locked`, `git diff --check`, and CLI help smoke test passed.
- Definitive `make check` passed: 32 files formatted, Ruff clean, strict mypy clean
  across 19 source files, and 67 tests passed with 99.20% total coverage; the
  public record models themselves reached 100% statement coverage.

### Open issues

- Registered domain answer validators are declarative only; their runtime registry
  belongs to the deterministic evaluator milestone.
- Pair validation does not yet prove semantic single-perturbation equivalence; that
  requires the productivity domain state and tool semantics in Milestone C.
- Prediction records preserve parsed calls but no model-specific parser exists yet.

### Next action

Implement Milestone C: deterministic in-memory productivity tools, seeded templates,
all four perturbations, approximately 40 development pairs, pair-diff auditing, and
manual inspection recorded here.

## 2026-08-26 — Implement Milestone A repository foundation

### Objective

Create a Python 3.12 package foundation with locked dependencies, deterministic
utilities, strict configuration loading, CPU-only CI, and acceptance tests. Do not
add model, data-generation, or training dependencies.

### Changes

- Initialized a local Git repository on branch `main` using the existing Git author
  configuration and truthful commit timestamps.
- Added Apache-2.0 licensing, Python 3.12 selection, generated-artifact and secret
  ignore rules, Hatchling packaging, and a `uv` lockfile.
- Added the `tool-abstention` console command and `python -m tool_abstention`
  entry point using `argparse`.
- Added a frozen Pydantic project configuration model, strict unknown-key rejection,
  YAML loading, and a checked example configuration.
- Added deterministic Python/NumPy seeding, canonical UTF-8 JSON serialization,
  object/file SHA-256 helpers, and validated JSONL I/O.
- Added Ruff, strict mypy, pytest, coverage, Make targets, and CPU-only GitHub
  Actions CI.
- Added 20 tests covering the CLI, configuration failures, fixed hash vectors,
  invalid numeric values, streaming file hashing, Unicode JSONL round trips, empty
  and malformed JSONL, line-numbered errors, and repeatable random sequences.
- Updated the README with the actual setup, validation, and check commands.

### Decisions and rationale

- **Python is constrained to `>=3.12,<3.13`.** The installed 3.12.13 interpreter
  supports the target ML ecosystem while preventing accidental 3.14 environments.
- **Configuration uses Pydantic plus PyYAML.** This provides typed validation and
  rejects misspelled keys before they affect experiments.
- **JSONL writes canonical bytes.** Stable key order, compact separators, UTF-8,
  and rejection of non-finite floats make content hashes reproducible.
- **Blank JSONL lines are ignored.** They do not represent records; malformed
  nonblank lines fail with the path and one-based line number.
- **CI contains no ML dependencies.** The foundation remains fast and portable;
  training dependencies will be selected only after their APIs are verified.

### Commands and outcomes

- `git config --get user.name` and `git config --get user.email` found the existing
  author identity; no identity was invented.
- Initial sandboxed `git init -b main` failed because `.git` metadata required
  explicit permission. The approved retry succeeded.
- Created planning commit `6e43436` and foundation commit `fd3a729`, both with real
  timestamps.
- Initial sandboxed `uv lock` could not access the user cache. The approved retry
  used CPython 3.12.13 and resolved 22 packages successfully.
- First `make check` found Ruff rule `UP047`; changed the generic loader to Python
  3.12 type-parameter syntax.
- Second `make check` found a formatter mismatch; applied Ruff's required layout.
- Third `make check` found a strict-mypy fixture inference error; explicitly typed
  the heterogeneous JSON records.
- Final `make check` passed: Ruff formatting and lint clean, mypy reported no issues
  across 13 source files, and all 20 pytest cases passed with 96.81% coverage.
- `uv sync --locked` succeeded with 21 installed packages checked.
- Both `python -m tool_abstention --help` and the installed
  `tool-abstention validate-config configs/project.yaml` command succeeded.
- `git check-ignore` confirmed that `data/`, `checkpoints/`, `.venv/`, and `.env`
  artifacts are ignored.

### Open issues

- GitHub Actions is configured but cannot execute until the repository is pushed to
  GitHub; local `make check` exercises the same check command.
- Training backend and training dependency groups remain intentionally undecided
  until the training API-verification milestone.
- Dataset and decision-class schemas belong to Milestone B and are not part of this
  foundation.

### Next action

Implement Milestone B: define the five-class taxonomy and validated canonical task,
tool, environment, expected-behavior, prediction, and evaluation record contracts.

## 2026-08-26 — Convert research roadmap into an implementation contract

### Objective

Translate the existing research plan into a buildable sequence suitable for a
Summer 2027 MLE portfolio project, and establish the requirement that future work
is documented as it happens.

### Changes

- Added `docs/11-implementation-plan.md` with ten gated milestones.
- Prioritized a one-domain vertical slice before broad dataset generation.
- Defined explicit record requirements for each of the five decision classes.
- Made deterministic answer validation a requirement instead of accepting any
  non-empty direct answer.
- Reduced the recruiting-quality v1 comparison to base, prompt-only, SFT, and DPO.
- Moved ORPO, RPO, K-DPO, probing, and scaling experiments behind the v1 gate.
- Added this worklog and linked both documents from `README.md`.
- Updated the README status to distinguish research planning from implementation.

### Decisions and rationale

- **Productivity is the first domain.** It can cover all five decisions using
  deterministic state and avoids reliance on time-sensitive factual knowledge.
- **Evaluator before training.** Training results are meaningless until output
  correctness is validated reliably.
- **Store raw predictions.** This permits evaluator fixes without repeating costly
  inference.
- **Narrow v1 method scope.** Two well-tested training methods produce stronger MLE
  evidence than five incomplete integrations.

### Verification

- Read `README.md` and all documents under `docs/` before planning.
- Confirmed the repository contained planning documents only; no implementation
  files were present.
- Confirmed the working directory was not recognized as a Git repository by the
  local `git` command, so no commit metadata was available to inspect.
- Verified the README links to both new documents and checked the rendered source
  sections with `rg`, `wc`, and `sed`; all expected references were present.

### Open issues

- MLX, Transformers, and TRL versions/APIs must be verified before dependency
  pinning because they are time-sensitive.
- The owner must decide whether a portable PyTorch training path is part of v1 or
  whether MLX remains the only training backend.
- Final dataset size should be based on measured template diversity, not an
  arbitrary count alone.

### Next action

Implement Milestone A from `docs/11-implementation-plan.md`: repository foundation,
configuration validation, deterministic utilities, tests, and CPU-only CI.
## 2026-08-26 — Diagnose baseline prompts and model capacity

### Objective

Determine whether the failed 0.5B smoke baseline came from prompt formatting,
strict parsing, or insufficient model capacity before running full validation.

### Changes and decisions

- Added strict `native-full`, `embedded-tools`, and `native-short` prompt variants.
- Added CLI prompt selection and task-specific plus policy-only prompt hashes to
  each run manifest.
- Kept the evaluator and parser unchanged; malformed braces were not repaired.
- Added reproducible 0.5B prompt and pinned 1.5B capacity diagnostic targets.
- Pinned Qwen2.5-1.5B-Instruct-4bit revision
  `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` after a read-only remote lookup.
- Froze `native-full` for full validation: it ties for best strict accuracy and
  retains the complete decision policy.

### Commands and outcomes

- The first sandboxed `uv` command could not read the external cache; the approved
  retry used the existing cache.
- The first `make check` found two long prompt lines. After wrapping them, pytest
  caught literal JSON braces being interpreted by `str.format`; replacing only the
  `{tools}` marker fixed the defect.
- The corrected `make check` passed Ruff, strict mypy over 31 source files, and 130
  tests with 95.94% coverage.
- `make prompt-diagnostic` ran 24 predictions with the cached 0.5B model and no
  inference errors. All variants scored 0% strict accuracy. Native-full and
  native-short hallucinated tools on 75% and 100% of abstention cases;
  embedded-tools made no required calls and had 0% tool hallucination.
- `make capacity-diagnostic` downloaded the pinned approximately 869 MB 1.5B model
  and ran 24 predictions locally on Metal with no inference errors. Native-full
  and native-short scored 50% strict accuracy, 100% act accuracy, 0% abstention
  accuracy, 0% paired accuracy, and 75% abstention tool hallucination. Native-full
  peaked at 1.54 GB Metal memory and averaged 592 ms per example.
- Embedded-tools at 1.5B scored 0% strict accuracy with 50% abstention tool
  hallucination. It was not selected.
- `make baseline-validation` ran the frozen 1.5B/native-full configuration on all
  120 validation tasks. It completed with no inference errors: 41.67% strict
  accuracy, 0% paired accuracy, 83.33% act accuracy, 0% abstention accuracy,
  25.79% macro-F1, and 58.33% abstention tool hallucination. Mean/median latency
  was 483/464 ms and peak Metal memory was 1.54 GB.
- Domain accuracy was 50.0% productivity, 37.5% finance, and 37.5% weather.
- Manually inspected all 70 strict failures: 35 unsafe tool calls, 10 missed
  required calls, 15 semantically correct but exact-format-mismatched answers, and
  10 semantically appropriate refusals missed by the plain-text classifier.

### Unresolved issues and next action

- MLX emits a deprecation warning for `mx.metal.device_info`; it originates in the
  pinned dependency and does not affect results.
- Calibrate answer and refusal judgments against stratified human labels, add
  regression cases for accepted corrections, and re-evaluate the stored baseline
  without rerunning the model. Do not inspect the held-out test split yet.
## 2026-08-26 — Build blinded human evaluator calibration

### Objective

Replace agent-assisted failure impressions with a reproducible workflow for real
human labels before changing evaluator semantics or beginning training.

### Changes and decisions

- Added a deterministic validation-only sampler with four examples per
  domain/class cell: 60 total items across three domains and five labels.
- Generated `calibration/round-1` from the frozen 1.5B validation predictions.
- Blinded the browser interface to task IDs, gold labels, and evaluator results.
- Added browser-local progress, six behavior choices including `UNCLEAR`, ternary
  semantic correctness, binary format acceptability, notes, and CSV export.
- Added a strict CSV loader rejecting incomplete fields, unknown values, duplicate
  IDs, foreign IDs, and schema changes.
- Added summaries plus two-annotator exact agreement and Cohen's kappa reporting.
- Kept semantic correctness separate from protocol compliance; no evaluator rule
  has been loosened before human evidence exists.

### Commands and outcomes

- Initial `make check` found long embedded HTML/JavaScript lines and one unused
  import. Used a file-level E501 exemption for the embedded asset and removed the
  import; all other Ruff rules remain active.
- The next run caught an incorrect test assumption that the private mapping would
  not contain domain-bearing task IDs. Corrected the test to verify that IDs and
  expected results are absent from the annotator-facing HTML.
- A focused four-test run passed its cases but correctly failed the repository's
  whole-suite 95% coverage threshold; verification therefore uses `make check`.
- Exported 60 real items with selection hash
  `4c1f1612009f7bb68a621cd3d86a06bf6ba7726bc041fcc88bb5a13c97907c92`
  and selected-predictions hash
  `efeaaff204129e9bea4f989c56d0e08586faab5d78203594f0b0510738ee6185`.

### Manual blocker and next action

- A real person must complete `calibration/round-1/annotate.html`; agent-generated
  labels will not be represented as human annotations.
- After the completed CSV returns, validate it, calculate agreement if a second
  annotator participates, adjudicate uncertain cases, then change evaluator rules
  only where labels demonstrate systematic disagreement.
## 2026-08-26 — Complete AI adjudication of calibration round 1

### Objective

Complete the 60-item semantic review at the owner's explicit request without
misrepresenting AI-generated judgments as independent human annotations.

### Actions and outcomes

- Reviewed every blinded packet item and wrote `annotations.agent.csv`.
- Added a provenance sidecar declaring `annotator_type: ai` and
  `independent_human_annotation: false`.
- Classified 39 outputs as CALL, 13 as ANSWER, and 8 as REFUSE.
- Judged 31 outputs semantically correct and 29 incorrect; all 60 used acceptable
  response syntax even when the selected action was unsafe or unavailable.
- `validate-calibration` accepted all 60 rows with no missing, duplicate, foreign,
  or invalid fields.

### Interpretation and next action

- The adjudication supports two evaluator changes to test explicitly: recognize
  “none of the provided/available functions” as REFUSE, and separate semantic
  answer correctness from exact protocol compliance.
- These labels can drive provisional regression tests and evaluator development,
  but must be described as AI adjudication in reports. Independent human agreement
  remains optional follow-up evidence, not a completed claim.
## 2026-08-26 — Calibrate evaluator against owner-verified adjudication

### Objective

Correct demonstrated evaluator false negatives while preserving unsafe-call
failures and keeping raw model predictions immutable.

### Changes

- Recorded owner verification in the AI-adjudication provenance sidecar; the file
  remains explicitly non-independent and AI-generated.
- Extended each evaluation record with behavior, semantic, and protocol axes.
- Defined headline correctness as correct behavior plus semantic correctness.
- Accepted case-aware exact atomic answers inside natural response sentences.
- Recognized “none of the provided/available functions/tools” as REFUSE.
- Added behavior, semantic, and protocol aggregate rates.
- Versioned the calibrated metric policy as evaluator `2.0.0` in every metrics
  artifact so scoring changes remain distinct from model changes.
- Added a deterministic CLI comparison between verified annotations and evaluator
  outputs, with explicit disagreement records.
- Updated the fixed public evaluation-schema hash and regression tests.

### Verification and results

- Re-evaluated 120 stored validation predictions without running inference.
- Accuracy changed from 41.67% to 62.5%, abstention accuracy from 0% to 41.67%,
  paired accuracy from 0% to 25%, and macro-F1 from 25.79% to 44.79%.
- Act accuracy remained 83.33% and abstention tool hallucination remained 58.33%;
  genuine unsafe behavior was not forgiven by calibration.
- The calibrated evaluator achieved 100% behavior, semantic, and protocol
  agreement on all 60 owner-verified items with zero disagreements.
- Re-evaluated all six prompt/capacity diagnostics under the new schema without
  rerunning either model.
- `make check` passed before artifact refresh: Ruff clean, strict mypy clean over
  33 source files, and 136 tests passed with 96.02% coverage. A final gate follows
  documentation updates.

### Next action

Implement provenance-aware public dataset adapters and leakage controls. Keep BFCL
and AgentAbstain external evaluation partitions out of training.

## 2026-08-27 — Milestone H: external provenance and BFCL evaluation

### Objective and boundaries

- Add pinned BFCL evaluation and a native AgentAbstain catalog without admitting
  public benchmark records to SFT or preference training.
- Preserve the internal test split: it was read only for leakage comparison; no
  internal test prediction was generated or inspected.
- Keep CI CPU-only and network-free through local fixtures and an optional
  `external-data` dependency group.

### Contracts and implementation

- Added immutable, extra-forbidding provenance and external-decision contracts.
  Provenance requires a 40-character revision, recognized SPDX license, original
  ID/file, source SHA-256, adapter version, transformations, attribution, and usage.
  Benchmark-only provenance rejects `training` usage.
- Added pinned external source configuration and expected real-source counts.
- Added `fetch-external`, `prepare-external`, `infer-external`, and
  `evaluate-external`, plus reproducible Make targets.
- Fetching is restricted to the two declared BFCL files and AgentAbstain's README,
  tasks, and environments. Canonical fetch manifests omit downloader cache metadata.
- Added recursive BFCL schema normalization. The fixture-supported mappings were
  extended after real preparation exposed `tuple` and explicit `any`: tuple maps to
  JSON Schema array; `any` faithfully omits a type constraint while retaining its
  metadata. All unknown types still fail closed.
- Added exact, normalized five-gram Jaccard (0.80), and SequenceMatcher (0.90)
  leakage checks across internal train, validation, and test data. Matches are
  quarantined and retained in a separate report.
- Added native AgentAbstain pair/environment cataloging, side integrity checks,
  required task files, environment-reference validation, and content hashes. No
  multi-turn conversion or execution was added.
- Generalized the inference boundary with `PromptExample` and resumable prompt
  prediction while preserving the existing `TaskRecord` path.
- The external evaluator records decision correctness and protocol validity
  separately. A malformed call attempt is CALL behavior and is counted in the
  malformed-call rate.

### Source fetch and factual correction

- Initial `uv lock` with `huggingface-hub<1` downgraded the verified Transformers
  stack; changed the group to `huggingface-hub>=1,<2`, restoring
  huggingface-hub 1.28.0, Transformers 5.16.1, and tokenizers 0.23.1.
- Fetched BFCL revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` and
  AgentAbstain revision `842228426c2a703347396501af61c7890972c7ee` without
  credentials. The hub emitted only an unauthenticated-rate-limit warning.
- The requested count of 239 BFCL irrelevance records was incorrect. The pinned
  file has IDs `irrelevance_0` through `irrelevance_239`, and official BFCL
  documentation states 240. Retained all official records instead of silently
  deleting one: 400 CALL plus 240 ABSTAIN, 640 total.
- Raw hashes: BFCL simple
  `fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158`,
  BFCL irrelevance
  `975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690`,
  AgentAbstain content
  `76f4d15cbbc27a6806ef7ee5530f93f084eeb14dbe5242bf07562351cb9b248d`.
- Two cached fetches produced the same manifest hash
  `414e7d9cdcc40f50a74345c1faea9824be18a539e3bdf6c22168c7d491278d6a`.
- Preparation found zero internal overlaps. The prepared BFCL hash is
  `89f296ce30834a665a679583e434b6ffa1a2dcfaa54e664451c2117bed112303`.
  Repeated preparations produced manifest hash
  `e718a771f3331cb0a893e8cbd223b5d053b6b33bbc669be584a7affba09fb1b3`.
- AgentAbstain matched 263 pairs and 42 environments exactly.

### Local baseline and results

- The first inference smoke inside the restricted sandbox failed with
  `No Metal device available`. Reran with approved local Metal access; eight real
  records completed successfully.
- Ran all 640 non-overlapping BFCL records with pinned
  `mlx-community/Qwen2.5-1.5B-Instruct-4bit@8b403126fc14f14cfc99bb4cfa72ecbc129ea677`,
  seed 0, temperature 0, and `native-full`. All predictions completed with zero
  inference errors; mean latency was 676.65 ms per record.
- Results: 88.44% decision accuracy, 99.25% CALL accuracy, 70.42% ABSTAIN
  accuracy, 84.83% balanced accuracy, 73.12% tool-call rate, and 3.91% malformed
  call rate.
- Prediction hash:
  `c2085f581f3936424977d02331f0ef5df60eb8ce4969305beb0928db0a28d222`.
  Raw output and latency are retained. The installed MLX generation boundary did
  not expose useful token or peak-memory counters, so optional fields remain absent.

### Verification record

- An intermediate `make check` passed Ruff and strict mypy and all 149 tests, but
  aggregate coverage was 94.79%, below the 95% gate. Added real catalog integrity
  branch tests before the final gate.
- A targeted one-test run passed its assertion but correctly failed the global
  coverage threshold; isolated tests are diagnostic only, and acceptance uses the
  complete suite.
- The session could not write the default global uv cache, so verification used
  `UV_CACHE_DIR=/private/tmp/tool-abstention-uv-cache` without changing the lock or
  environment semantics.
- Documentation now records provenance, commands, source-count correction,
  evaluation scope, results, and limitations. Raw datasets and model artifacts stay
  ignored; reproducible prediction/evaluation evidence is tracked.
- The first final gate caught two Ruff formatting changes; the project formatter
  corrected them. The next lint pass caught an unescaped regex dot in a test and it
  was made explicit.
- The first fixture fetch test reused its output directory without `exist_ok=True`
  and failed on the second reproducibility pass. Corrected the fake downloader; this
  was test scaffolding only, not production fetch behavior.
- Final `make check` passed: Ruff formatting/lint clean, strict mypy clean over 35
  source files, and all 150 tests passed with 95.28% aggregate coverage.

## 2026-08-27 — Milestone I: seed-0 SFT baseline

### Data and implementation

- Added strict SFT training configuration, deterministic five-class assistant-target
  formatting, `build-sft` and `train-sft` commands, Make targets, adapter-aware
  internal/external inference, and network-free fixture tests.
- Exported 360 internal train and 120 validation examples. The exporter structurally
  reads only `train.jsonl` and `validation.jsonl`; a fixture proves an invalid test
  file is never opened. BFCL and AgentAbstain remain external-only.
- Repeated export was byte-identical. After the final CLARIFY correction, train hash
  is `a42910aa5db7b4525544e131fba09e304313a9b05616d905c26467bd5fd63d93`
  and validation hash is
  `b4c9b978b84b4edef12b1b481b29311864862fe8c1d99f86f8856f860c41b5f7`.
- Resolved the configured model revision to a local immutable snapshot before
  launching MLX because the installed LoRA CLI does not accept `revision` directly.
- Real smoke exposed a Transformers 5 / mlx-lm 0.29 boundary: chat templates return
  `BatchEncoding` rather than a flat token list. Added a narrow compatibility runner
  that extracts `input_ids`, validates them, and retains prompt masking.

### Training attempts and failures

- The 0.5B/20-step Metal smoke completed. Validation loss fell 3.782 → 0.979;
  adapter hash was `9fba9c2d9ffeb8fb937bfa65ca34ba28bb775ee4815933a7095f8812fd30eba9`;
  peak memory was 2.04 GB.
- Initial 1.5B batch-4 training reached step 75, then Metal OOMed at 19.30 GB. Its
  step-45 checkpoint was preserved but not promoted.
- Restarted with batch 2, accumulation 8, and gradient checkpointing. Effective
  batch remained 16 and memory stabilized at 4.20 GB. This run reached the complete
  step-180 first epoch before an intentional stop/interruption; its checkpoint was
  evaluated diagnostically.
- Diagnostic validation was 87.5% overall / 75% paired, but CLARIFY recall was 0%.
  Raw outputs exactly matched the target `Please provide the missing <slot>.`; the
  evaluator correctly classified this imperative as ANSWER rather than a question.
- Corrected the target to `Could you provide the missing <slot>?`, regenerated data,
  and restarted from the base model in a new directory. No checkpoint was overwritten.

### Selected completed run and evaluation

- Completed one epoch (180 microbatches) of corrected 1.5B SFT with seed 0. Full
  validation loss ended at 0.005 and peak memory at 4.20 GB. Selected adapter hash:
  `88841d6959a751cea2b60b88788b3552c283fc82acdfb9ce43ca08988a582556`.
- Internal validation: 95.83% accuracy, 100% act accuracy, 91.67% abstention
  accuracy, 91.67% paired accuracy, 95.2% macro-F1, 100% protocol compliance, and
  8.33% abstention hallucination. All ANSWER, CLARIFY, and REFUSE examples passed;
  five NOOP examples over-called.
- Ran all 640 non-overlapping BFCL records with the selected adapter. Decision
  accuracy was 93.59%, CALL 99.25%, ABSTAIN 84.17%, and balanced accuracy 91.71%.
- Compared with base BFCL, balanced accuracy improved 6.88 points and ABSTAIN
  accuracy improved 13.75 points without lowering CALL decision accuracy.
- Malformed-call rate worsened from 3.91% to 11.56%. This protocol regression is a
  real negative result and must be analyzed before DPO; it is not hidden by the
  improved decision metric.
- Refreshed inference manifests to include the exact adapter SHA-256. Raw adapter
  weights and intermediate checkpoints remain ignored; predictions, evaluations,
  metrics, and manifests are retained.

### Verification and remaining gate

- Final implementation gate before documentation: Ruff clean, strict mypy clean
  over 38 source files, and all 157 tests passed at 95.01% aggregate coverage.
- The internal held-out test split remains untouched.
- Seed 0 is evidence, not a final multi-seed result. Next: classify malformed BFCL
  calls, then repeat the frozen SFT recipe for seeds 1 and 2 before DPO.
- After adapter-aware manifest hashing and its fixture regression test, the final
  gate passed with all 157 tests and 95.02% aggregate coverage; Ruff and strict
  mypy remained clean.

## 2026-08-27 — BFCL malformed-call analysis

### Evaluator audit and correction

- Analyzed stored base and SFT predictions without running either model.
- Found the external evaluator reused the internal lowercase-slug parsed-call
  contract. Valid native BFCL names such as `calculate_NPV` and `fMRI.analyze`
  were therefore falsely marked protocol-invalid.
- Added a separate external JSON protocol validator that permits non-empty native
  function names while retaining strict object/arguments structure. Versioned the
  corrected external metric policy as 1.1.0 and added regression fixtures.
- Replayed both stored runs. CALL/ABSTAIN decision metrics did not change. Base
  malformed rate corrected from 3.91% to 1.72% (11/640); SFT corrected from 11.56%
  to 9.69% (62/640).

### Deterministic failure taxonomy

- Added `analyze-malformed`, a canonical stored-prediction comparison with source
  hashes, mutually exclusive categories, and new/resolved/persistent task IDs.
- Base failures: 4 max-token truncations and 7 prose/wrapper violations.
- SFT failures: 49 truncated JSON structures, 7 other JSON syntax errors, 3
  max-token truncations, 2 prose/wrapper violations, and 1 invalid literal.
- SFT introduced 61 malformed task IDs, resolved 10 base failures, and retained one
  persistent failure.
- Manual inspection of the dominant category shows schema fields copied into
  `arguments`, the function name nested inside that object, and a missing root
  closing brace. Only 3/62 failures reached the token limit, so increasing
  `max_tokens` is not the primary fix.
- Decision: do not relax the parser to accept structurally invalid JSON. Freeze the
  seed-0 training/evaluation recipe, run seeds 1 and 2, then test protocol-focused
  interventions as explicit ablations without training on BFCL.
- Final gate passed: Ruff clean, strict mypy clean over 40 source files, and all
  170 tests passed with 95.05% aggregate coverage.

## 2026-08-27 — Frozen SFT seeds 1 and 2

### Execution

- Added seed-specific training configurations that differ from the selected
  seed-0 recipe only in `seed`. Kept the pinned model revision, shared SFT data
  manifest, one-epoch schedule, batch/accumulation, LoRA, and evaluation settings.
- Trained seed 1 for all 180 iterations on local Metal. Final validation loss was
  0.013, peak memory was 4.204 GB, and adapter SHA-256 is
  `388e04107c329fd55efca2d3be92817e81575c09c3dc4b4e60072df83710e6f4`.
- The first seed-2 process was interrupted at iteration 155 by an attached tool
  session ending. Preserved it as ignored `seed-2-interrupted`; did not evaluate or
  report its iteration-135 checkpoint. Restarted from the base model and completed
  all 180 iterations. Its repeated loss sequence matched the interrupted run.
- Complete seed 2 final validation loss was 0.012, peak memory was 4.204 GB, and
  adapter SHA-256 is
  `0edbc9d1dcb9dd6f23c0742d2caf1c018b3ebecd0321d17573991e068f91f0f0`.

### Evaluation and interpretation

- Seed 1 internal accuracy was 94.17% (act 88.33%, abstain 100.00%, paired
  88.33%); seed 2 was 94.17% (act 96.67%, abstain 91.67%, paired 88.33%).
- Seed 1 BFCL decision/balanced accuracy was 91.25%/88.75%, with 5.47% malformed
  calls. Seed 2 was 91.41%/88.71%, with 4.06% malformed calls.
- Three-seed SFT mean ± sample SD is 94.72 ± 0.96% internal accuracy, 95.00 ±
  6.01% act accuracy, 94.44 ± 4.81% internal abstention, 92.08 ± 1.31% BFCL
  decision accuracy, and 89.72 ± 1.72% BFCL balanced accuracy.
- The BFCL malformed-call mean is 6.41 ± 2.93%, still above the 1.72% base rate.
  Seed 0 was unusually strong on external abstention and unusually weak on syntax;
  the per-seed table is retained instead of presenting only aggregate values.
- Wrote a canonical aggregate artifact at `results/sft/1.5b/summary.json` and a
  human-readable analysis in `docs/16-sft-multiseed.md`.
- The internal held-out test split remained sealed. No public benchmark record was
  used for training or intervention design.

### Verification

- The first `make check` attempt could not initialize the sandboxed default uv
  cache. Repeated with `UV_CACHE_DIR=/private/tmp/tool-abstention-uv-cache`; Ruff
  formatting and lint passed, strict mypy passed over 40 source files, and all 170
  tests passed with 95.05% aggregate coverage.
- `git diff --check` passed. The held-out test file remained byte-identical with
  SHA-256 `76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.
- Confirmed each new result directory contains raw predictions, evaluations,
  metrics, inference manifest, and malformed-call analysis. The two training
  configs differ from seed 0 only by their declared seed.

## 2026-08-27 — Internal-only protocol-repair ablation

### Implementation and tests

- Added a deterministic protocol-stress generator with 16 train and 4 validation
  CALL/CLARIFY pairs. Arguments cover nested objects, arrays, booleans, nulls,
  numeric values, Unicode, and long structured payloads.
- Added repair-SFT assembly that appends 32 train and 8 validation stress records
  to the original 360/120 corpus. It rejects non-internal provenance, any declared
  external source, test consumption, and ID collisions.
- Added a `protocol-strict` inference prompt and CLI commands for generating stress
  records and assembling repair data. Added fixture coverage for determinism,
  provenance rejection, CLI behavior, and prompt rendering.
- Generated manifests record zero external sources and `test_consumed=false`.
  The held-out test hash remained
  `76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.
- Pre-training gate passed: Ruff clean, strict mypy clean over 42 source files, and
  177 tests passed at 95.22% coverage.

### Controlled experiments

- Prompt-only strict JSON instructions failed the internal stress gate: exact
  accuracy stayed 25.0% and protocol compliance fell from 62.5% to 50.0%. No BFCL
  run was performed for this rejected prompt.
- Trained one 196-batch seed-0 repair adapter from the pinned base model on the
  392-example augmented corpus. Adapter SHA-256 is
  `05fed69862ae6188bec4d81974d6a804432d8d5ea41637a8099ff581b7c76ee9`.
- Repair SFT improved internal stress exact accuracy from 25.0% to 62.5% and
  protocol compliance from 62.5% to 100.0%. It preserved original internal
  validation accuracy at 95.83%, act at 100.0%, and abstention at 91.67%.
- Because both internal gates passed, ran one BFCL external report. Malformed calls
  fell from 62/640 (9.69%) to 13/640 (2.03%), but ABSTAIN accuracy collapsed from
  84.17% to 58.33% and balanced accuracy from 91.71% to 79.17%.
- Rejected the repair adapter. It fixes syntax largely by shifting toward CALL
  behavior, violating the predeclared no-decision-regression criterion. DPO must
  not initialize from it. The original three-seed SFT baseline remains selected.
- Final gate passed again after documentation: Ruff clean, strict mypy clean over
  42 source files, and all 177 tests passed with 95.22% aggregate coverage.

## 2026-08-27 — Preference-data contracts and generator

### Implementation

- Added strict immutable preference records with internal task/pair identity,
  split, target class, chosen/rejected responses, seven controlled negative types,
  source-task hash, generator version, and selected SFT adapter hash.
- Enforced no test records, no external provenance, no identical responses, a
  correct protocol-valid chosen response, an evaluator-failing rejected response,
  and protocol-valid syntax for every non-malformed negative.
- Added deterministic generation for wrong decisions, wrong abstention class,
  unnecessary calls, wrong tools, wrong arguments, malformed calls, and schema
  copying. Paired act responses provide realistic unnecessary-call negatives.
- Added `build-preferences`, public preference JSON Schema export/validation,
  `make preferences`, fixed-vector schema hashing, and tests for tampering,
  contamination, incomplete pairs, determinism, CLI behavior, and evaluator
  semantics.

### Generated artifact and trainer audit

- Generated 360 train and 120 validation preferences. Distribution: 128 malformed
  calls, 80 unnecessary calls, 80 wrong abstention classes, and 48 each of schema
  copying, wrong arguments, wrong decision abstention, and wrong tool.
- Manifest hashes the internal train/validation sources and records
  `external_sources=[]`, `test_consumed=false`, and selected SFT adapter SHA-256
  `88841d6959a751cea2b60b88788b3552c283fc82acdfb9ce43ca08988a582556`.
- The held-out test remained unchanged at SHA-256
  `76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.
- Audited pinned `mlx-lm 0.29` source and official MLX documentation. It supports
  LoRA/DoRA/full SFT but has no released DPO trainer or preference dataset. The
  official DPO feature request remains open and contains only a proposed loss.
- Decision: do not claim a normal LoRA run is DPO and do not add an unverified
  third-party trainer. Implement a narrow numerically tested MLX DPO runner next,
  then require a 0.5B Metal smoke before primary experiments.
- Final verification passed: Ruff clean, strict mypy clean over 44 source files,
  and all 183 tests passed with 95.07% aggregate coverage.

## 2026-08-27 — Numerically verified MLX DPO and 0.5B smoke

### Implementation and corrections

- Added strict prepared DPO examples, deterministic balanced smoke selection,
  shared-prompt token enforcement, completion-only masking, truncation rejection,
  source hashes, and separate 0.5B/1.5B manifests pinned to their actual SFT
  adapter hashes.
- Added independent NumPy fixed-vector implementations for sequence log
  probabilities, reference-adjusted DPO loss, label smoothing, rewards, reward
  accuracy, and margin.
- Added frozen-reference caches with model, revision, adapter, tokenizer, example,
  maximum-length, record, and completion-token identities. Missing, duplicate,
  stale, mismatched, and non-finite cache data fail closed.
- Added the version-gated project-owned MLX runner and `prepare-dpo`,
  `cache-dpo-reference`, and `train-dpo` commands. The runner keeps one policy
  model, freezes base weights, trains LoRA only, accumulates gradients, saves
  checkpoints, reloads the final adapter, and writes canonical audit artifacts.
- The first static pass found wrong MLX namespace and return-shape assumptions;
  corrected log-softmax/log-sigmoid operations and typed the MLX-LM load boundary
  before any Metal training.
- The first full gate passed all 192 tests but failed coverage at 88.74% because
  the Metal-only runner cannot import in CPU-only CI. Excluded only that hardware
  module from CPU coverage, retained pure numerical/cache/leakage tests and fake
  CLI boundary tests, and added missing failure-path coverage without lowering the
  95% threshold.
- Direct `tool-abstention` invocation was unavailable on the shell PATH. Repeated
  through the locked `uv run tool-abstention` entry point; no generated input or
  test data changed during the failed attempt.

### Preparation and Metal smoke

- Generated the smoke preference manifest (360 train / 120 validation before
  selection), then prepared exactly 16 train and 8 validation DPO examples. Also
  prepared the primary 360/120 manifest without training it.
- Computed frozen 0.5B reference caches for all 24 smoke examples. Both caches use
  tokenizer hash `a126ebff…e74d`; train/validation records hashes are
  `134bf0bc…91a0` and `780f2f4e…68c7`.
- The initial smoke passed, but audit review found the run manifest lacked embedded
  canonical-config/exact-command hashes and truncation counts. Added them and
  reran the full gate and smoke rather than documenting the earlier incomplete
  artifact.
- Final 20-step smoke passed: validation loss 0.064053, reward accuracy 100%,
  reward margin 3.576737, zero train/validation truncations, exact reload metric
  reproduction, 3.172 GB peak memory, and output adapter hash `1989858a…17f7`.
  Runtime was 16.03 seconds excluding reference caching.
- The held-out internal test stayed hash-identical at
  `76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.
  No BFCL, AgentAbstain, or rejected protocol-repair artifact was opened.
- Final CPU gate: Ruff clean, strict mypy clean over 47 source files, and all 193
  tests passed at 95.14% aggregate coverage.

## 2026-08-27 — 1.5B DPO seed-0 experiment

### Frozen-reference cache and training

- After the smoke commit was pushed, computed the pinned 1.5B frozen-reference
  caches: 360 train records (`a1c15ad…7153`) and 120 validation records
  (`d3e0d5b…012c`). Both used original SFT adapter `88841d69…2556`, tokenizer hash
  `a126ebff…e74d`, maximum length 2048, and internal-only prepared examples.
- Ran the one authorized 360-example seed-0 epoch with the predeclared config. No
  retry, retuning, early stopping, or checkpoint selection occurred. Training and
  four scheduled validation passes completed with finite values.
- Runtime was 2,455.13 seconds, peak memory 12.327 GB, and no chosen/rejected
  completion was truncated. Output adapter SHA-256 is `f643063f…5485`.
- Final preference validation loss was 0.000464, reward margin 15.2616, and reward
  accuracy 100%. A fresh adapter reload reproduced all metrics exactly.

### Internal gate and rejection

- Evaluated all 120 original internal validation tasks with the frozen inference
  config. Accuracy collapsed from SFT seed 0's 95.83% to 42.50%; act accuracy fell
  from 100% to 0%, abstention accuracy fell from 91.67% to 85%, paired accuracy
  fell to 0%, and protocol compliance stayed 100%.
- Evaluated the eight internal protocol-stress tasks. Accuracy was 0%, act accuracy
  0%, and hallucinated-call rate 0%. Inspection showed four direct answers and
  four refusals, not calls: the apparent syntax improvement was complete
  over-abstention.
- Rejected the DPO adapter because three internal promotion conditions failed.
  BFCL was not opened or run. No external result influenced training, selection,
  or retry behavior, and no retry was performed.
- The held-out test remained byte-identical at
  `76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.

## 2026-08-27 — Mean-normalized 0.5B DPO diagnostic

- Added strict `sum`/`mean` completion-log-probability normalization to the DPO
  config and numerical/runtime implementations. Historical configurations retain
  `sum` by default. Mean mode requires positive chosen/rejected token counts and
  normalizes policy and frozen-reference values identically.
- Added a fixed-vector mean-normalization test. The first expected margin was
  incorrectly set to 0.1 even though chosen and rejected normalized rewards were
  both 0.1; the full gate caught it. Corrected the policy vector to encode a real
  0.1 margin, then all 193 tests passed at 95.12% coverage with Ruff and strict
  mypy clean.
- Predeclared one bounded diagnostic: 0.5B SFT smoke initialization, 16/8
  preferences, mean normalization, 8 updates, `5e-6`, and no smoothing.
- Before DPO, generated the missing full behavioral baseline for the exact 0.5B
  initializer. It scored only 5% overall, 0% act, 10% abstention, and 0% stress
  accuracy. Decision: the old adapter is a compatibility smoke, not a valid
  behavioral promotion model.
- The mean-normalized run completed in 8.66 seconds at 3.168 GB peak memory with
  exact reload reproduction. Reward accuracy was 87.5%, chosen reward 0.0199,
  rejected reward -0.0463, and reward margin 0.0662; the loss did not saturate.
- Free generation improved overall accuracy from 5% to 10% and abstention from
  10% to 20%, but act accuracy remained 0% and stress accuracy remained 0%.
  Rejected the adapter; did not run BFCL or authorize 1.5B.
- Decision: train and validate a competent one-epoch 0.5B SFT initializer before
  any further DPO behavioral claim. The internal test remains sealed.

## 2026-08-28 — Competent 0.5B SFT screening initializer

- Predeclared a replacement for the inadequate 20-step SFT smoke using the same
  pinned 0.5B model and internal SFT corpus: 180 batches, batch 2, accumulation 8,
  learning rate `2e-5`, 16 LoRA layers, rank 16, and maximum length 1024.
- Training completed on local Metal. Validation loss was 0.029 at iteration 135;
  peak memory was approximately 7.13 GB. Final adapter SHA-256 is
  `f788f9f7048282380ae2a2815018bb331a470ded0c6c51bcd8ef47a95a4677f4`.
- Full original validation scored 80.83% overall, 63.33% act, 98.33% abstention,
  83.33% behavior accuracy, and 100% protocol compliance. This is sufficiently
  competent to detect loss of tool-calling behavior.
- Internal protocol stress remained difficult: 0% exact, 25% behavior accuracy,
  87.5% protocol compliance, and 50% hallucinated-call rate. This supplies an
  improvement target without consulting BFCL.
- Froze behavioral DPO gates: at most five points act regression, two points
  overall/abstention regression, 100% original protocol compliance, improved
  stress behavior, non-regressed stress protocol, and lower stress hallucination.
- No external or held-out test record was used. The adapter is a screening
  initializer only; the selected research baseline remains 1.5B SFT.

## 2026-08-28 — Mean-DPO behavioral screen and selection audit

- Pinned new internal preferences to competent 0.5B adapter `f788f9f…677f4` and
  prepared 32 train / 16 validation examples. Frozen caches used pinned model,
  adapter, tokenizer, and example hashes; the held-out test stayed unchanged.
- Ran one mean-normalized 32-update screen at `2e-6`. It completed in 63.6 seconds
  with 5.255 GB peak memory, exact reload reproduction, zero truncations, 100%
  preference reward accuracy, and 1.2927 reward margin.
- The frozen behavioral gate failed: original accuracy fell 80.83%→40%, act
  63.33%→0%, abstention 98.33%→80%, and protocol 100%→85.83%. Stress behavior
  improved 25%→50%, but protocol collapsed 87.5%→12.5% and hallucination worsened
  50%→75%. Rejected the adapter and skipped BFCL.
- Audited the subset. CALL/abstain counts were balanced 16/16, but 30/32 records
  were finance and almost every pair was ANSWER-family. Root cause: the selector
  seeds rejection types then fills lexicographically; it does not balance domain
  or target abstention class.
- Decision: preserve the negative result, fix selection to operate on complete
  pairs round-robin across domain/class, and keep optimizer settings unchanged for
  at most one bounded rerun.

## 2026-08-28 — Stratified DPO rerun and branch stop

- Replaced the small-subset selector with deterministic complete-pair selection
  round-robin across domain and ANSWER/CLARIFY/REFUSE/NOOP strata. Odd limits,
  invalid variants, incomplete/duplicate pairs, and unknown strata fail closed.
- Added tests proving an eight-record subset contains four complete pairs and all
  four abstention classes. Focused Ruff, strict mypy, and nine DPO tests passed.
- Regenerated the 32/16 subset. Training contained exactly four pairs per
  abstention class and represented finance, productivity, and weather. Recomputed
  frozen reference caches because example hashes changed.
- Ran the one allowed rerun with unchanged mean-DPO optimizer settings. It
  completed in 55.5 seconds at 8.991 GB peak memory, with exact reload, zero
  truncations, 100% reward accuracy, and 0.6191 reward margin.
- Behavioral gate failed again: original accuracy 80.83%→45.83%, act
  63.33%→0%, and abstention 98.33%→91.67%. Protocol remained 100%. Stress
  behavior fell 25%→12.5%; zero hallucination reflected suppressed tool calling.
- Rejected the adapter, skipped BFCL, and stopped the standard-DPO branch. Subset
  skew was real but not sufficient to explain collapse. No additional learning
  rate/beta sweep and no new 1.5B DPO run are authorized.
- Decision: retain the original SFT baseline. Any future preference work requires
  a separately planned method change with supervised or explicit conservative/KL
  anchoring and generation-based checkpoint gates.

## 2026-08-28 — Supervised-anchored DPO terminal screen

- Implemented `chosen_sft_weight` and the combined completion-only objective
  `DPO + λ * chosen NLL`. Added independent NumPy fixed vectors and failure tests;
  runtime now logs total, DPO, and chosen-SFT losses separately.
- Strict mypy initially rejected an MLX `stack` tuple; changed it to the typed list
  form before Metal execution. The first full test run passed all 194 tests but
  coverage was 94.98%; added the missing non-finite/zero-token anchor failure path
  instead of lowering the threshold. Final pre-run gate passed at 95.02%.
- Predeclared exactly two candidates from identical initializer, stratified data,
  cache, seed, beta, learning rate, and eight-update budget: λ=0.5 and λ=1.0.
- Both trained with finite component losses, zero truncations, exact reload, and
  8.891 GB peak memory. Reward accuracy was 87.5%; margins were 0.0810 and 0.0716.
- λ=0.5 scored 65.83% overall, 35% act, 96.67% abstention, and 100% protocol.
  λ=1.0 scored 60.83% overall, 28.33% act, 93.33% abstention, and 100% protocol.
  Both had 12.5% stress behavior, 87.5% stress protocol, and 25% hallucination.
- Both failed frozen act/overall/stress gates. Rejected both, skipped BFCL, and did
  not authorize 1.5B. No adaptive anchor/update sweep was performed.
- Terminal decision: stop preference optimization for this project version and
  retain the three-seed 1.5B SFT baseline. Move next to consolidated comparison,
  limitations, reproducibility accounting, and technical-report preparation.

## 2026-08-28 — Deterministic final analysis

- Inventoried every retained internal-validation, protocol-stress, and BFCL
  metrics/evaluation artifact. Confirmed the analysis requires no model execution,
  network access, prediction parsing, external raw data, or held-out test access.
- Added a strict offline analysis contract and `build-final-analysis` CLI. Declared
  inputs are content-hashed; missing files, unknown experiment references,
  mismatched paired task IDs, duplicate rows, and any path with a `test` component
  fail closed.
- Added three-seed sample statistics with t-based 95% intervals and deterministic
  paired bootstrap intervals with 10,000 resamples. Separated 1.5B primary results
  from 0.5B screening failures and retained rejected adapters visibly.
- Added canonical JSON, CSV, Markdown, and dependency-free SVG outputs plus an
  artifact manifest. Ran `make analysis` twice; all output hashes were identical.
- Added fixed-vector, determinism, leakage, bad-reference, duplicate, missing-file,
  and CLI delegation tests. Focused Ruff and strict mypy checks passed; all focused
  tests passed. The repository-wide quality gate is recorded below after the final
  documentation update.
- Decision: select the three-seed 1.5B SFT baseline. Treat preference optimization
  as a characterized negative result. Do not spend more compute until training data
  diversity and the optimization method materially change.
- Final `make check` passed: Ruff formatting and lint clean, strict mypy clean,
  and all 200 tests passed at 95.05% coverage. The sealed internal-test SHA-256
  remained `76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.
