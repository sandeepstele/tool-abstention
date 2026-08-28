# Stratified mean-DPO screen

## Selector correction

The DPO subset selector now requires complete unique act/abstain pairs and even
limits. It deterministically round-robins across domain and all four abstention
classes. The corrected 32-record training subset contains four pairs each for
ANSWER, CLARIFY, REFUSE, and NOOP, with all three internal domains represented.
The 16-record validation subset contains two pairs per abstention class across its
available domains.

This correction is covered by deterministic tests. The optimizer configuration
was intentionally unchanged from the failed skewed-subset run: mean normalization,
32 updates, learning rate `2e-6`, `beta=0.1`, and no smoothing.

## Numerical result

Training completed in 55.5 seconds with 8.991 GB peak memory. All values were
finite, no sequence was truncated, the adapter reloaded exactly, reward accuracy
was 100%, and reward margin was 0.6191. Output adapter SHA-256 is
`fa4f5194a4d51906251d03954337e31f5bbb8280c8a0180935eb37be0defc6d7`.

## Behavioral result

| Gate metric | SFT initializer | DPO candidate | Required | Pass? |
|---|---:|---:|---:|---:|
| Original accuracy | 80.83% | 45.83% | ≥78.83% | No |
| Original act accuracy | 63.33% | 0.00% | ≥58.33% | No |
| Original abstention | 98.33% | 91.67% | ≥96.33% | No |
| Original protocol | 100.00% | 100.00% | 100% | Yes |
| Stress behavior | 25.00% | 12.50% | >25% | No |
| Stress protocol | 87.50% | 87.50% | ≥87.5% | Yes |
| Stress hallucination | 50.00% | 0.00% | <50% | Yes |

The zero hallucination rate again comes from suppressing tool calls, not solving
the paired tasks. Paired accuracy remained zero. The candidate was rejected and
BFCL was not run.

## Conclusion

Subset skew was a real defect, but it was not the sole cause of collapse. With a
competent initializer, mean normalization, conservative learning rate, bounded
updates, and balanced paired data, standard offline DPO still destroyed correct
CALL generation while its preference reward metrics looked perfect.

No additional standard-DPO sweep or 1.5B run was justified. The original SFT
baseline remained selected. The subsequent bounded supervised-anchor experiment
also failed its generation gates and closed the preference branch for this release;
see `25-anchored-dpo-screen.md`.
