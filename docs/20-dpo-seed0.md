# 1.5B DPO seed-0 negative result

## Predeclared experiment

The 0.5B compatibility smoke passed, authorizing one 1.5B seed-0 experiment. The
policy initialized from the original selected SFT adapter
`88841d69…2556`, not the rejected protocol-repair adapter. Frozen reference caches
contained all 360 internal training preferences and 120 internal validation
preferences. BFCL, AgentAbstain, and the held-out internal test split were not
training inputs.

The fixed configuration used `beta=0.1`, no label smoothing, batch size 1,
gradient accumulation 8, learning rate `5e-6`, 16 LoRA layers, maximum length
2048, and one 360-example epoch. No retry, early stop, checkpoint selection, or
hyperparameter change was made after observing intermediate metrics.

## Numerical training result

Training completed in 2,455 seconds (40.9 minutes) on local Apple Metal. Peak
unified memory was 12.327 GB. All losses and gradients were finite; no completion
was truncated. The output adapter hash is
`f643063f5764ff8cfb2d3bcbb4139d82b00d36dbc051a6dfa18d0262f8e35485`.
A fresh reload reproduced every final DPO validation metric exactly.

| Preference metric | Initial | Final |
|---|---:|---:|
| DPO loss | 0.693147 | 0.000464 |
| Chosen reward | 0.0000 | -3.9224 |
| Rejected reward | 0.0000 | -19.1839 |
| Reward margin | 0.0000 | 15.2616 |
| Reward accuracy | 0% | 100% |

The objective successfully separated chosen from rejected completions, primarily
by reducing rejected likelihood much more strongly. This numerical success did
not imply acceptable generation behavior.

## Internal promotion gate

| Gate | Required | SFT seed 0 | DPO seed 0 | Pass? |
|---|---:|---:|---:|---:|
| Original act accuracy | ≥97% | 100.00% | 0.00% | No |
| Original abstention accuracy | ≥91.67% | 91.67% | 85.00% | No |
| Protocol compliance | 100% | 100.00% | 100.00% | Yes |
| Overall drop from SFT | ≤2 points | 95.83% | 42.50% | No |
| Stress hallucinated-call rate | improve | 100.00% | 0.00% | Yes |

The model produced no tool calls on either internal evaluation. On the original
validation set its 120 predictions were 55 `ANSWER`, 39 `REFUSE`, 16 `NOOP`, and
10 `CLARIFY`. On the eight stress tasks it emitted four answers and four refusals.
Thus the apparent hallucination improvement is an over-abstention collapse, not a
usable protocol improvement. Paired accuracy fell to zero.

## Decision

Reject the DPO adapter. The internal promotion gate failed before external
evaluation, so BFCL was not opened and no external metric was produced. The
original three-seed SFT baseline remains selected.

This result exposes an important distinction: completion-level preference
separation can look perfect while free generation crosses the act/abstain decision
boundary catastrophically. Any follow-up must be a separately predeclared
ablation, justified from internal data only. Candidate mechanisms to test include
per-token rather than summed completion log-probability normalization, explicit
CALL-class weighting, and a materially smaller optimization budget. They should
first pass a richer 0.5B behavioral smoke, not merely the reward-margin gate.

The held-out test file remained byte-identical at SHA-256
`76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.
