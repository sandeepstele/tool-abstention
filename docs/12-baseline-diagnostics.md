# Baseline Prompt and Capacity Diagnostics

## Experimental control

Both diagnostics use the same eight validation tasks: one complete pair for each
abstention class. Generation is greedy with seed 0 and a 256-token limit. The task
selection hash is
`fc9a0984eea1140adcad1f7ee13cf7120f04d7b3227ba2fbea775137e9ddd570`.
The evaluator and strict parser are unchanged between runs.

The prompt variants are:

- `native-full`: tokenizer-native tools and the complete five-way policy.
- `embedded-tools`: canonical tool JSON and exact call syntax in the system text.
- `native-short`: tokenizer-native tools and a shorter decision policy.

## Results

| Model | Prompt | Accuracy | Act | Abstain | Paired | Hallucination | Peak GB |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B 4-bit | native-full | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 0.87 |
| Qwen2.5 0.5B 4-bit | embedded-tools | 0.25 | 0.00 | 0.50 | 0.00 | 0.00 | 0.79 |
| Qwen2.5 0.5B 4-bit | native-short | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.84 |
| Qwen2.5 1.5B 4-bit | native-full | 0.625 | 1.00 | 0.25 | 0.25 | 0.75 | 1.54 |
| Qwen2.5 1.5B 4-bit | embedded-tools | 0.125 | 0.00 | 0.25 | 0.00 | 0.50 | 1.46 |
| Qwen2.5 1.5B 4-bit | native-short | 0.625 | 1.00 | 0.25 | 0.25 | 0.75 | 1.50 |

## Decision

Freeze `native-full` for the baseline. It ties for best strict accuracy, retains
the explicit five-way decision policy, and has the same observed hallucination
rate as `native-short`. The 1.5B result proves that the task and parser can score
valid required calls, while its 0% abstention accuracy leaves a clear target for
fine-tuning. The embedded variant changes behavior but does not improve strict
accuracy, so it is not selected.

These eight-task diagnostics select a configuration; they are not a performance
claim.

## Full validation

The frozen configuration was run on all 120 validation tasks (60 complete pairs):

| Metric | Result |
|---|---:|
| Calibrated accuracy | 0.6250 |
| Behavior accuracy | 0.6250 |
| Semantic accuracy | 0.6250 |
| Protocol compliance | 1.0000 |
| Paired accuracy | 0.2500 |
| Act accuracy | 0.8333 |
| Abstention accuracy | 0.4167 |
| Macro-F1 | 0.4479 |
| Abstention tool hallucination | 0.5833 |
| Mean / median latency | 483 / 464 ms |
| Peak Metal memory | 1.54 GB |

Domain accuracy was 50.0% for productivity and 37.5% each for finance and
weather. There were no inference errors.

The original strict evaluator produced 70 failures. A blinded 60-item calibration
packet was AI-adjudicated and owner-verified. The calibrated evaluator has 100%
agreement on behavior, semantics, and surface protocol validity for that packet.
It leaves 45 substantive failures across the full validation split:

- 35 genuine unsafe actions: the model called a tool on CLARIFY or NOOP tasks, or
  invented an unavailable calendar tool on productivity REFUSE tasks.
- 10 genuine missed actions: finance and weather CALL tasks received unsupported
  direct answers instead of required state lookups.
The 15 natural-language direct answers and 10 capability refusals are now scored
correctly. Raw outputs were not regenerated. Held-out test data remains untouched.
