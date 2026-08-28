# 16 — Three-Seed SFT Baseline

## Frozen experiment

Seeds 0, 1, and 2 use the same pinned 1.5B model revision, 360 training tasks,
120 validation tasks, one-epoch schedule, LoRA settings, and greedy inference
configuration. Only the training seed changes. The shared training-data manifest
hash is `1a4c888ee27c520837a1f6f9f48a161404e99da05f3eae26f80cd4191ef94478`.
Public BFCL records remain evaluation-only and the internal test split remains
sealed. The canonical machine-readable result is
[`results/sft/1.5b/summary.json`](../results/sft/1.5b/summary.json).

## Internal validation

| Seed | Accuracy | Act | Abstain | Paired |
|---:|---:|---:|---:|---:|
| 0 | 95.83% | 100.00% | 91.67% | 91.67% |
| 1 | 94.17% | 88.33% | 100.00% | 88.33% |
| 2 | 94.17% | 96.67% | 91.67% | 88.33% |
| Mean ± sample SD | 94.72 ± 0.96% | 95.00 ± 6.01% | 94.44 ± 4.81% | 89.44 ± 1.92% |

The overall score is stable, but act/abstain trade-offs vary materially by seed.
Reporting only seed 0 would overstate act accuracy and understate uncertainty.

## External BFCL decision evaluation

| Seed | Decision | CALL | ABSTAIN | Balanced | Malformed |
|---:|---:|---:|---:|---:|---:|
| 0 | 93.59% | 99.25% | 84.17% | 91.71% | 9.69% |
| 1 | 91.25% | 98.75% | 78.75% | 88.75% | 5.47% |
| 2 | 91.41% | 99.50% | 77.92% | 88.71% | 4.06% |
| Mean ± sample SD | 92.08 ± 1.31% | 99.17 ± 0.38% | 80.28 ± 3.39% | 89.72 ± 1.72% | 6.41 ± 2.93% |

The three-seed conclusion is less flattering than seed 0 alone. SFT reliably
preserves BFCL CALL behavior, and all seeds beat the 70.42% base ABSTAIN accuracy,
but external abstention varies and malformed syntax remains worse than the 1.72%
base rate. Seeds 1 and 2 resolve all 11 base malformed cases while introducing
35 and 26 new failures respectively; most are truncated or otherwise invalid JSON.

## Decision

The SFT baseline is complete and suitable as the initialization/control for the
next milestone. The next work is a protocol-focused intervention designed only
from internal training data, followed by preference-data contracts and a small DPO
smoke. BFCL must remain evaluation-only, and no internal test predictions should
be generated until the final locked comparison.
