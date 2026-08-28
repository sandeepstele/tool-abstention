# Canonical experiment comparison

Generated offline from committed metrics and per-example evaluations.

| Experiment | Model | Split | Status | Accuracy | Act/CALL | Abstain | Protocol |
|---|---:|---|---|---:|---:|---:|---:|
| Base native-full | 1.5b | internal-validation | baseline | 62.50% | 83.33% | 41.67% | 100.00% |
| Base native-full | 1.5b | bfcl | baseline | 88.44% | 99.25% | 70.42% | — |
| SFT seed 0 | 1.5b | internal-validation | selected | 95.83% | 100.00% | 91.67% | 100.00% |
| SFT seed 1 | 1.5b | internal-validation | selected | 94.17% | 88.33% | 100.00% | 100.00% |
| SFT seed 2 | 1.5b | internal-validation | selected | 94.17% | 96.67% | 91.67% | 100.00% |
| SFT seed 0 | 1.5b | bfcl | selected | 93.59% | 99.25% | 84.17% | — |
| SFT seed 1 | 1.5b | bfcl | selected | 91.25% | 98.75% | 78.75% | — |
| SFT seed 2 | 1.5b | bfcl | selected | 91.41% | 99.50% | 77.92% | — |
| Protocol repair | 1.5b | internal-validation | rejected | 95.83% | 100.00% | 91.67% | 100.00% |
| Protocol repair | 1.5b | protocol-stress | rejected | 62.50% | 100.00% | 25.00% | 100.00% |
| Protocol repair | 1.5b | bfcl | rejected | 84.38% | 100.00% | 58.33% | — |
| Standard DPO | 1.5b | internal-validation | rejected | 42.50% | 0.00% | 85.00% | 100.00% |
| SFT screen | 0.5b | internal-validation | baseline | 80.83% | 63.33% | 98.33% | 100.00% |
| Anchored DPO lambda 0.5 | 0.5b | internal-validation | rejected | 65.83% | 35.00% | 96.67% | 100.00% |
| Anchored DPO lambda 1.0 | 0.5b | internal-validation | rejected | 60.83% | 28.33% | 93.33% | 100.00% |

Rejected experiments are retained as negative results, not promoted models.
