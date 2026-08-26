# 05 — Data Plan

> The five-class taxonomy, paired-task construction, domains, sources, and label determinism. This is the project's scientific core — everything else (training, eval, paper) is downstream of the data.

## 1. The five decision classes (extended abstention taxonomy)

We extend When2Call's 4-way scheme with K-DPO's "answer from knowledge" and AgentAbstain's "no action needed", giving **one act class + four abstain classes**:

| # | Class | Behavior | Positive example | The failure it counteracts |
|---|---|---|---|---|
| 1 | **CALL** | Emit the correct tool call with correct arguments | `get_weather(city="Paris")` | — (the act case) |
| 2 | **ANSWER** | Answer directly from the prompt / parametric knowledge; no tool | "The capital of France is Paris." | *over-calling* (Tool-Overuse Illusion / K-DPO) |
| 3 | **CLARIFY** | Ask a follow-up question because a required argument is missing/ambiguous | "Which account number did you mean?" | *missed clarification* (When2Call follow-up) |
| 4 | **REFUSE** | Explicitly decline because no available tool can help | "I can't answer that — no available tool provides it." | *fabrication* (When2Call unable-to-answer) |
| 5 | **NOOP** | Take no action because the correct action is *none* (task already satisfied / not actionable) | "There are no pending actions." | *unnecessary action* (AgentAbstain high-stakes / emergent-risk) |

**Why four abstain classes and not three:** When2Call's "direct answer" is conflated — it treats *any* direct answer as wrong, but answering-from-knowledge is often *correct* (K-DPO's entire point). Splitting **ANSWER** out and adding **NOOP** lets us (a) unify the two lines of work, and (b) test H3 — that the four classes are learned at different rates.

### 1.1 Class invariants (enforced in `tests/test_taxonomy.py`)
- The five classes are **mutually exclusive and exhaustive** for every task.
- **ANSWER** vs **REFUSE** vs **NOOP** all produce *no tool call* but are distinguished by *why*, captured in the label at construction time.
- **CALL** is the only class whose correctness depends on a tool *execution* result; the other four are judged by surface behavior + construction label.

## 2. Paired-task construction (the core design decision)

Mirroring AgentAbstain's paired act/abstain design, **every task has a `should-act` and a `should-abstain` variant**, produced by a *controlled perturbation* δ of exactly one dimension:

| Perturbation target | act → abstain transformation | Resulting class |
|---|---|---|
| Query (add answerable content) | "What's the balance?" → "What's the balance? (I already told you it's $42)" | ANSWER |
| Query (drop a required arg) | "transfer $10 to acct 1234" → "transfer $10" | CLARIFY |
| Tool inventory (remove the tool) | keep `get_weather`, drop nothing → remove `get_weather` | REFUSE |
| Environment state (make it already-done) | "close issue #7" (open) → "close issue #7" (already closed) | NOOP |

**Rationale:** paired design makes the benchmark *jointly* measure over-abstention (failing to act when it should) and under-abstention (acting when it shouldn't). A model is correct only if it is right on **both** sides of the pair (paired accuracy — our headline metric, matching AgentAbstain).

**Paired-accuracy definition:** `correct(act) ∧ correct(abstain)` for a given pair. This is a stricter, more meaningful signal than raw accuracy because it punishes models that simply *always* or *never* call.

## 3. Tool domains (≥ 3, target 4)

| Domain | Example tools | Abstain stress tested |
|---|---|---|
| **Finance** | `get_balance`, `transfer`, `list_transactions`, `fx_rate` | REFUSE (remove tool), CLARIFY (missing acct), ANSWER (trivia about fees) |
| **Weather / Geo** | `get_weather`, `get_time`, `geocode`, `distance` | REFUSE (remove `get_time`), ANSWER (capital cities), NOOP (already-set) |
| **Productivity / CRM** | `search_contacts`, `create_event`, `close_ticket`, `send_email` | NOOP (already-closed), CLARIFY (ambiguous contact) |
| *(stretch)* **Knowledge / General** | `lookup_wiki`, `calculator` | ANSWER-heavy (K-DPO's home turf) |

Each domain contributes ~150 pairs (act + abstain), giving **~600 pairs / 1,200 labeled tasks** — comfortably above the ≥500-task minimum.

## 4. Size & splits

| Split | Size | Purpose |
|---|---|---|
| **train** | ~360 pairs (720 tasks) | SFT + preference data |
| **val** | ~120 pairs (240 tasks) | early stopping + method selection |
| **test** | ~120 pairs (240 tasks) | held-out, reported once |

- **No task leakage across variants:** a pair's `act` and `abstain` never straddle a split boundary — both live in the same split (a model can't "cheat" by memorizing the twin).
- **Contamination guard** (`split.py`): dedupe near-duplicate queries (min-edit / embedding) so test queries are never paraphrase-adjacent to train queries.

## 5. Sources & construction method

We build our own data (not borrow a dataset) because **owning the labels** is a stated differentiator (`01-vision.md`). The construction is *synthetic but grounded*:

1. **Schema grounding.** Tool schemas are hand-written per domain (JSON-Schema, OpenAI function-calling format). Mock executors return deterministic values, so a `CALL` task is genuinely executable.
2. **Template seeding.** Query/state templates are written per (domain, class), then *programmatically instantiated* with sampled entities (names, cities, amounts, IDs) — no LLM needed, fully deterministic given a seed.
3. **Perturbation.** The paired variant is generated by applying δ (table above) to the seeded task.
4. **Label assignment.** `label.py` assigns the class *by construction* — the generator knows which class it produced, so the label is exact, never judged.
5. *(Stretch only)* **LLM augmentation.** Optionally expand query surface (paraphrases) using an LLM, then *filter* through the deterministic rules; the LLM never touches labels.

**Why not start from When2Call / AgentAbstain data directly:**
- When2Call's "direct answer" label is *always-wrong*, which would bake in the very conflation we're fixing.
- AgentAbstain is multi-turn/agentic with irreversible state — heavier than our single-turn decision, and its 8 scenarios don't map cleanly onto our 5-class scheme.
- Licensing: When2Call is CC-BY-4.0 (fine to *build on*), but a from-scratch generator keeps us Apache-2.0-clean and reproducible.

## 6. Label determinism & verifiability (no LLM judge)

Each abstain class is deterministically decidable from **construction + surface output**:

| Class | Deterministic check (`judge.py`) |
|---|---|
| CALL | parsed tool call + correct tool name + correct args (diff against expected) |
| ANSWER | no tool call + non-empty answer + answerable-by-construction |
| CLARIFY | no tool call + interrogative + references the missing arg (regex over slot names) |
| REFUSE | no tool call + refusal lexicon ("cannot", "unable", "no tool", "can't help") |
| NOOP | no tool call + no-op/acknowledgment marker + task pre-satisfied by construction |

The *only* genuinely ambiguous axis — ANSWER vs REFUSE when the model is confidently wrong — is resolved by the construction label, not by reading the output's prose. This keeps the **core metric fully rule-based** (a stated non-goal is judge-dependence).

## 7. Output format & manifest

- **Raw:** `data/raw/<domain>/tasks.jsonl` — one line per task (act/abstain variants, tools, query, label, expected_tool).
- **Processed:** `data/processed/sft/*.jsonl` (SFT records) and `data/processed/pref/*.jsonl` (chosen/rejected pairs).
- **Manifest:** `data/manifest.json` maps every artifact path → `{content_hash, config_hash, seed}` (see `09-experiment-tracking.md`).

## 8. Data-quality acceptance tests (gate Phase 1 → Phase 2)

- **Determinism:** regenerate with the same seed → byte-identical output.
- **Class balance:** no class < 10% of tasks; CALL present in every domain.
- **Pair integrity:** every `abstain` has a matching `act` twin (same `pair_id`, exactly one δ applied).
- **Executability:** every `CALL` label, when executed against the mock tool, returns the expected value.
- **No leakage:** cross-split near-duplicate rate ≈ 0 (below a min-edit threshold).
