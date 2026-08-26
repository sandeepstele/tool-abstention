# Worklog

This is the human-readable engineering journal for the project. It complements
machine-readable experiment logs and Git history. New entries are added in reverse
chronological order and follow the protocol in
[`docs/11-implementation-plan.md`](docs/11-implementation-plan.md#5-documentation-protocol).

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
