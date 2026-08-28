# Mean-normalized DPO diagnostic

## Motivation and predeclaration

The rejected 1.5B run used summed completion log probabilities and achieved a
large preference margin while collapsing into over-abstention. This bounded
diagnostic tested one mechanism only: divide each chosen/rejected sequence log
probability by its own completion-token count before computing the policy/reference
gap. It used the existing 0.5B SFT smoke adapter, 16 train preferences, 8 validation
preferences, 8 updates, learning rate `5e-6`, and no label smoothing.

The diagnostic was predeclared as non-promotable if the 0.5B initializer lacked
usable act behavior.

## Baseline discovery

Full generation evaluation showed that the original 20-step 0.5B SFT smoke was
only a runtime compatibility artifact, not a competent behavioral baseline:

| Metric | 0.5B SFT smoke |
|---|---:|
| Overall accuracy | 5.0% |
| Act accuracy | 0.0% |
| Abstention accuracy | 10.0% |
| Protocol compliance | 100.0% |
| Stress accuracy | 0.0% |

This means it cannot detect preservation of tool calling. The earlier numerical
smoke gate was therefore insufficient even though it correctly validated MLX
training, memory, adapter persistence, and reward math.

## Mean-normalized result

Training took 8.66 seconds with 3.168 GB peak memory. No sequence was truncated,
all values remained finite, and a fresh reload reproduced the final metrics.
Preference reward accuracy was 87.5%, chosen reward was positive `0.0199`, rejected
reward was `-0.0463`, and margin was `0.0662`. Unlike summed DPO, the objective did
not immediately saturate.

| Behavioral metric | SFT smoke | Mean-DPO | Delta |
|---|---:|---:|---:|
| Overall accuracy | 5.0% | 10.0% | +5.0 |
| Act accuracy | 0.0% | 0.0% | 0.0 |
| Abstention accuracy | 10.0% | 20.0% | +10.0 |
| Protocol compliance | 100.0% | 100.0% | 0.0 |
| Stress accuracy | 0.0% | 0.0% | 0.0 |

## Decision

Mean normalization is numerically healthier but this adapter is rejected: it
still never calls a tool. No 1.5B run and no BFCL evaluation are authorized from
this result.

The next prerequisite is a competent 0.5B SFT initializer trained for a real
internal epoch and evaluated on original validation plus protocol stress. Future
DPO behavioral gates must require preservation of that initializer's act accuracy,
not merely positive reward margin.
