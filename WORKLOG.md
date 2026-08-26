# Worklog

This is the human-readable engineering journal for the project. It complements
machine-readable experiment logs and Git history. New entries are added in reverse
chronological order and follow the protocol in
[`docs/11-implementation-plan.md`](docs/11-implementation-plan.md#5-documentation-protocol).

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
