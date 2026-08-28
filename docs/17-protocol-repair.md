# 17 — Protocol-Repair Ablation

## Question and boundary

Can a small internal-only protocol curriculum reduce malformed tool calls without
damaging the act/abstain decision? BFCL remained external evaluation only. The
held-out internal test split was never opened. Intervention selection used a new
deterministic stress validation slice, not BFCL.

The generator creates 16 train and 4 validation CALL/CLARIFY pairs. CALL arguments
exercise nested objects, arrays of objects, booleans, nulls, numbers, Unicode, and
long structured payloads. Each CLARIFY twin removes one required top-level field.
The repair corpus contains the original 360/120 train/validation examples plus
32/8 stress records. Manifests require `source_kind=internal_generated`, an empty
external-source list, and `test_consumed=false`.

## Prompt-only ablation

The strict prompt explicitly required balanced strict JSON and prohibited schema
copying and prose. On the eight internal stress records it left exact accuracy at
25.0% and reduced protocol compliance from 62.5% to 50.0%. It was rejected before
BFCL evaluation. More instructions did not repair the learned behavior.

## Repair-SFT ablation

The seed-0 1.5B model was retrained from the pinned base revision for one epoch
(196 batches) on the augmented internal corpus. Adapter SHA-256 is
`05fed69862ae6188bec4d81974d6a804432d8d5ea41637a8099ff581b7c76ee9`.

| Gate | Original SFT seed 0 | Repair SFT | Delta |
|---|---:|---:|---:|
| Stress exact accuracy | 25.00% | 62.50% | +37.50 |
| Stress protocol compliance | 62.50% | 100.00% | +37.50 |
| Stress abstention accuracy | 0.00% | 25.00% | +25.00 |
| Original internal accuracy | 95.83% | 95.83% | 0.00 |
| Original internal act accuracy | 100.00% | 100.00% | 0.00 |
| Original internal abstention accuracy | 91.67% | 91.67% | 0.00 |
| BFCL balanced accuracy | 91.71% | 79.17% | -12.54 |
| BFCL CALL accuracy | 99.25% | 100.00% | +0.75 |
| BFCL ABSTAIN accuracy | 84.17% | 58.33% | -25.83 |
| BFCL malformed-call rate | 9.69% | 2.03% | -7.66 |

The syntax intervention worked, but the model learned a stronger propensity to
call tools. It resolved 53 of seed 0's 62 malformed cases and produced only 13
malformed cases, yet its BFCL call rate rose to 78.12% and external abstention
collapsed. This is not an acceptable repair.

## Decision

Reject both interventions for model selection and do not initialize DPO from the
repair adapter. Retain the original three-seed SFT baseline. The next preference
milestone must explicitly separate two negative types: malformed protocol output
and semantically wrong CALL decisions. Chosen/rejected construction must contain
matched abstention counterexamples so syntax repair cannot be achieved by shifting
the model toward always calling. BFCL records remain prohibited from training.
