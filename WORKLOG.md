# Worklog

This is the human-readable engineering journal for the project. It complements
machine-readable experiment logs and Git history. New entries are added in reverse
chronological order and follow the protocol in
[`docs/11-implementation-plan.md`](docs/11-implementation-plan.md#5-documentation-protocol).

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
