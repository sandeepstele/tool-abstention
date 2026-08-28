# Competent-initializer mean-DPO screen

## Predeclared run

This first DPO screen from the competent one-epoch 0.5B SFT adapter used 32 train
and 16 validation preferences, mean completion log probabilities, 32 updates,
learning rate `2e-6`, `beta=0.1`, and no smoothing. Behavioral thresholds were
frozen before training.

Numerically, the run completed cleanly: 100% preference reward accuracy, reward
margin 1.2927, exact reload reproduction, zero truncations, 63.6 seconds runtime,
and 5.255 GB peak memory. The output adapter hash is
`1bb85f0ceecfd4dbbb95bc1b866a7efb04b9bcbe999199aaff8421ddaf7b9012`.

## Behavioral failure

| Gate metric | SFT initializer | DPO candidate | Required | Pass? |
|---|---:|---:|---:|---:|
| Original accuracy | 80.83% | 40.00% | ≥78.83% | No |
| Original act accuracy | 63.33% | 0.00% | ≥58.33% | No |
| Original abstention | 98.33% | 80.00% | ≥96.33% | No |
| Original protocol | 100.00% | 85.83% | 100% | No |
| Stress behavior | 25.00% | 50.00% | >25% | Yes |
| Stress protocol | 87.50% | 12.50% | ≥87.5% | No |
| Stress hallucination | 50.00% | 75.00% | <50% | No |

The candidate was rejected before BFCL and cannot authorize a 1.5B run.

## Selection audit

The 32-record training subset contained 16 CALL and 16 abstain examples, but its
lexicographic selection was badly skewed: 30 records were finance tasks and 30
were ANSWER-family pairs, with only one CLARIFY pair and no REFUSE or NOOP pair.
The selector seeded each rejection type and then filled from sorted IDs; it did
not stratify domain or target class.

This is an experimental-design defect. The result remains retained because it
faithfully measures the declared subset, but optimizer conclusions should not be
generalized from it. The next implementation must select complete pairs and
round-robin across domain and abstention class while maintaining deterministic
ordering. No learning-rate or beta change is justified before that correction.
