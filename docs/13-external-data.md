# 13 — External Data Provenance and BFCL Evaluation

## Scope and isolation

Milestone H adds public data only as external evaluation evidence. No BFCL or
AgentAbstain record is copied into SFT or preference data, and no prediction is
generated for the internal test split. Strict provenance records include immutable
source revisions, SPDX licenses, source-file hashes, original IDs, adapter version,
transformations, attribution, and permitted usage. Benchmark-only provenance rejects
training use.

## Pinned sources

- BFCL dataset `gorilla-llm/Berkeley-Function-Calling-Leaderboard` at revision
  `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, Apache-2.0.
- AgentAbstain dataset `antiquality/agentabstain` at revision
  `842228426c2a703347396501af61c7890972c7ee`, CC-BY-4.0.
- AgentAbstain code at revision
  `f581249704b26804e28a39e37396f1be00b71a4d`, MIT.

Only declared BFCL files and the native AgentAbstain task/environment trees are
fetched. Raw snapshots live under ignored `data/external/raw`; preparation emits
canonical manifests, hashes, a dataset card, normalized BFCL records, and an
auditable quarantine report under ignored `data/external/prepared`.

## Source-count correction

The implementation request stated 239 BFCL irrelevance records. The pinned file and
BFCL's own dataset documentation contain 240 (`irrelevance_0` through
`irrelevance_239`). The adapter therefore fails unless it sees the truthful official
counts: 400 simple CALL plus 240 irrelevance ABSTAIN. It does not silently discard a
record to match the mistaken total.

AgentAbstain acceptance matched the declared 263 native pairs and 42 environments.
Each pair must contain both `act` and `abstain` task files, and referenced initial
state environments must exist.

## Normalization and leakage

BFCL Python-style schema types are recursively converted to Draft 2020-12 forms.
`dict`, `float`, `int`, `str`, `bool`, `list`, and `tuple` have explicit mappings;
BFCL's explicit `any` becomes an unconstrained schema with its metadata retained.
Unknown types fail preparation.

Every external query is compared with internal train, validation, and test data by:

1. canonical case/whitespace/punctuation-insensitive exact matching;
2. normalized character five-gram Jaccard at or above 0.80;
3. normalized `SequenceMatcher` similarity at or above 0.90.

The real pinned preparation produced 640 records, quarantined zero, and hashed the
prepared BFCL file as
`89f296ce30834a665a679583e434b6ffa1a2dcfaa54e664451c2117bed112303`.

## Local 1.5B baseline

The complete non-overlapping set ran locally on Metal using
`mlx-community/Qwen2.5-1.5B-Instruct-4bit` at revision
`8b403126fc14f14cfc99bb4cfa72ecbc129ea677`, deterministic decoding, and the
`native-full` prompt. All 640 predictions completed with no inference errors.

| Metric | Result |
|---|---:|
| Decision accuracy | 88.44% |
| CALL accuracy | 99.25% |
| ABSTAIN accuracy | 70.42% |
| Balanced accuracy | 84.83% |
| Tool-call rate | 73.12% |
| Malformed-call rate | 1.72% |

The predictions hash is
`c2085f581f3936424977d02331f0ef5df60eb8ce4969305beb0928db0a28d222`.
A syntactically malformed call attempt counts as CALL behavior but fails protocol
validity separately. This benchmark measures CALL versus ABSTAIN decisions, not
exact BFCL arguments. The current MLX generation API did not expose useful token or
peak-memory counters, so those optional prediction fields remain absent; latency and
raw output are retained per record.

AgentAbstain is not run here. Its multi-turn sandbox and judge are intentionally
outside this deterministic single-turn milestone.

The malformed rate was corrected by external evaluator 1.1.0 after separating
native BFCL function-name syntax from the internal lowercase stable-ID contract.
