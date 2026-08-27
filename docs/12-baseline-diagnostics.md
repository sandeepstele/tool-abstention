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
| Qwen2.5 0.5B 4-bit | embedded-tools | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.79 |
| Qwen2.5 0.5B 4-bit | native-short | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.84 |
| Qwen2.5 1.5B 4-bit | native-full | 0.50 | 1.00 | 0.00 | 0.00 | 0.75 | 1.54 |
| Qwen2.5 1.5B 4-bit | embedded-tools | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.46 |
| Qwen2.5 1.5B 4-bit | native-short | 0.50 | 1.00 | 0.00 | 0.00 | 0.75 | 1.50 |

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
| Strict accuracy | 0.4167 |
| Paired accuracy | 0.0000 |
| Act accuracy | 0.8333 |
| Abstention accuracy | 0.0000 |
| Macro-F1 | 0.2579 |
| Abstention tool hallucination | 0.5833 |
| Mean / median latency | 483 / 464 ms |
| Peak Metal memory | 1.54 GB |

Domain accuracy was 50.0% for productivity and 37.5% each for finance and
weather. There were no inference errors.

All 70 strict failures were manually inspected, exceeding the planned 25-output
audit. They fall into four repeated categories:

- 35 genuine unsafe actions: the model called a tool on CLARIFY or NOOP tasks, or
  invented an unavailable calendar tool on productivity REFUSE tasks.
- 10 genuine missed actions: finance and weather CALL tasks received unsupported
  direct answers instead of required state lookups.
- 15 semantically correct direct answers rejected by exact answer formatting.
- 10 semantically appropriate capability refusals classified as the wrong plain
  text behavior by the current heuristic.

The last two categories are evaluator-calibration candidates, not evidence that
the model abstains reliably. The evaluator must be calibrated against human labels
before training comparisons; raw outputs remain immutable so corrections do not
require rerunning inference. Held-out test data remains untouched.
