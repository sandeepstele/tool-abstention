# Supervised-anchored DPO screen

## Method

The final bounded preference experiment added a chosen-completion supervised
anchor to mean-normalized DPO:

```text
loss = DPO_mean + λ * (-mean_logp_policy(chosen completion))
```

The anchor is completion-only and shares the prompt mask with DPO. Fixed-vector
tests cover the combined loss, negative weights, non-finite values, and zero token
counts. Runtime logs total, DPO, and chosen-SFT loss separately.

Two candidates were predeclared: `λ=0.5` and `λ=1.0`. Both used the competent
0.5B SFT initializer, corrected stratified 32/16 preference subset, `beta=0.1`,
learning rate `2e-6`, and exactly eight updates. Neither candidate influenced the
other's configuration.

## Numerical results

| Metric | λ=0.5 | λ=1.0 |
|---|---:|---:|
| Reward accuracy | 87.5% | 87.5% |
| Reward margin | 0.0810 | 0.0716 |
| Peak memory | 8.891 GB | 8.891 GB |
| Runtime | 27.1 s | 28.9 s |
| Reload reproduction | exact | exact |

Both outputs differed from initialization, all values were finite, and no
completion was truncated.

## Frozen behavioral gate

| Metric | SFT initializer | λ=0.5 | λ=1.0 | Required |
|---|---:|---:|---:|---:|
| Original accuracy | 80.83% | 65.83% | 60.83% | ≥78.83% |
| Original act accuracy | 63.33% | 35.00% | 28.33% | ≥58.33% |
| Original abstention | 98.33% | 96.67% | 93.33% | ≥96.33% |
| Original protocol | 100.00% | 100.00% | 100.00% | 100% |
| Stress behavior | 25.00% | 12.50% | 12.50% | >25% |
| Stress protocol | 87.50% | 87.50% | 87.50% | ≥87.5% |
| Stress hallucination | 50.00% | 25.00% | 25.00% | <50% |

The anchor materially reduced collapse compared with unanchored DPO, but neither
candidate preserved act or overall accuracy, and neither improved stress behavior.
Lower hallucination again coincided with weaker correct CALL behavior.

## Terminal decision

Reject both adapters. Do not run BFCL or 1.5B anchored DPO. The predeclared matrix
is exhausted, and further weight/update tuning would be an adaptive sweep after
multiple failures.

Preference optimization is stopped for this project version. The selected model
remains the three-seed 1.5B SFT baseline. The negative sequence is itself a useful
result: perfect or improving offline preference metrics repeatedly failed to
predict generation-level act/abstain behavior, even after mean normalization,
balanced selection, a competent initializer, reduced update budget, and a chosen
SFT anchor.

The next project milestone is final analysis: consolidate SFT uncertainty,
protocol-repair trade-offs, DPO/anchored-DPO failures, compute and artifact costs,
and limitations into one reproducible comparison and technical report.
