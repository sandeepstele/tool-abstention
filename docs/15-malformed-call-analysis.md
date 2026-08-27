# 15 — BFCL Malformed-Call Analysis

## Evaluator correction

External evaluator 1.0 reused the internal `ParsedToolCall` contract, whose tool
names are lowercase stable IDs. Native BFCL names such as `calculate_NPV` and
`fMRI.analyze` are valid external function names but were rejected as malformed.
External evaluator 1.1 separates JSON protocol validity from internal ID policy.
Replaying stored predictions changed only protocol metrics, never CALL-versus-
ABSTAIN decisions:

| Model | Previously reported | Corrected |
|---|---:|---:|
| Base 1.5B | 3.91% | 1.72% (11/640) |
| SFT seed 0 | 11.56% | 9.69% (62/640) |

## Deterministic taxonomy

The analysis uses stored raw predictions only. It assigns one mutually exclusive
category and records source prediction hashes plus new/resolved/persistent IDs.

| Category | Base | SFT |
|---|---:|---:|
| Truncated JSON structure | 0 | 49 |
| Other JSON syntax | 0 | 7 |
| Max-token truncation | 4 | 3 |
| Prose/wrapper violation | 7 | 2 |
| Invalid JSON literal | 0 | 1 |
| **Total** | **11** | **62** |

There are 61 newly malformed IDs after SFT, 10 base failures resolved by SFT, and
one persistent failure. The canonical report is stored beside the SFT BFCL metrics.

## Interpretation

The regression is not mainly caused by the 256-token decoding ceiling: only three
SFT failures hit that limit. Forty-nine outputs begin a tool-call object but fail to
close the root JSON structure. Manual inspection shows a recurring learned pattern:
schema material is copied into `arguments`, the function `name` is placed inside
that object, and the final root brace is omitted. This is consistent with limited
generalization from simple internal schemas to BFCL's more varied nested schemas.

The base model more often emits prose or Markdown around otherwise recognizable
calls; SFT largely removes that behavior but introduces rigid malformed JSON on
unfamiliar schemas. Parser relaxation is therefore not the appropriate fix. The
next SFT seeds should use the frozen recipe to measure variance. Afterward, candidate
interventions should be explicit ablations: more diverse internal schemas,
grammar-constrained decoding, or a small protocol-focused training slice. BFCL must
remain evaluation-only.
