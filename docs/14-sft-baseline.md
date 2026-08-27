# 14 — SFT Baseline

## Boundary and data

SFT consumes exactly 360 internal train tasks and 120 internal validation tasks.
The formatter never opens the internal test file, and BFCL/AgentAbstain provenance
prohibits training use. Repeated exports are byte-identical. Assistant targets use
native Qwen tool calls plus deterministic ANSWER, CLARIFY, REFUSE, and NOOP text.

## Implementation findings

The installed Transformers 5 chat template returns `BatchEncoding`, while
`mlx-lm 0.29` expects a flat token list. A narrow tested compatibility boundary
extracts `input_ids` and preserves prompt masking. The model revision is resolved to
an immutable local Hugging Face snapshot before the MLX subprocess starts.

A 0.5B/20-step smoke reduced full validation loss from 3.782 to 0.979 and produced
an adapter at 2.04 GB peak memory. The first 1.5B configuration (batch 4) reached
step 75 but Metal OOMed after memory rose to 19.30 GB. Batch 2, accumulation 8, and
gradient checkpointing preserved effective batch 16 and stabilized at 4.20 GB.

The first safe one-epoch checkpoint exposed a target bug: CLARIFY was trained as an
imperative rather than a question, causing 0% CLARIFY recall despite exact target
reproduction. The target was corrected to `Could you provide the missing <slot>?`
and training restarted from the base model. No flawed adapter was promoted.

## Selected seed-0 run

- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Revision: `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- One epoch: 180 microbatches, batch 2, accumulation 8
- LoRA: rank 16, scale 32, dropout 0.05, final 16 layers
- Learning rate: `2e-5`; max sequence length: 2048
- Adapter SHA-256:
  `88841d6959a751cea2b60b88788b3552c283fc82acdfb9ce43ca08988a582556`
- Final full-validation loss: 0.005; peak memory: 4.20 GB

One epoch was selected because validation loss had saturated; the earlier attempted
three-epoch run was interrupted and is not represented as complete.

## Results

| Metric | Base 1.5B | SFT seed 0 | Delta |
|---|---:|---:|---:|
| Internal accuracy | 62.50% | 95.83% | +33.33 |
| Internal act accuracy | 83.33% | 100.00% | +16.67 |
| Internal abstention accuracy | 41.67% | 91.67% | +50.00 |
| Internal paired accuracy | 25.00% | 91.67% | +66.67 |
| Internal hallucination rate | 58.33% | 8.33% | -50.00 |
| BFCL decision accuracy | 88.44% | 93.59% | +5.16 |
| BFCL CALL accuracy | 99.25% | 99.25% | 0.00 |
| BFCL ABSTAIN accuracy | 70.42% | 84.17% | +13.75 |
| BFCL balanced accuracy | 84.83% | 91.71% | +6.88 |
| BFCL malformed-call rate | 1.72% | 9.69% | +7.97 |

The result supports the core SFT premise for seed 0 and generalizes beyond the
synthetic templates. It also reveals a protocol regression: SFT preserves the CALL
decision rate but increases malformed tool-call syntax. External evaluator 1.1.0
correctly permits native BFCL names containing uppercase letters and dots; the
remaining regression and its deterministic taxonomy are documented in
[`15-malformed-call-analysis.md`](15-malformed-call-analysis.md).

The internal held-out test split remains untouched. Seeds 1 and 2 are required
before reporting uncertainty or treating SFT as a finalized multi-seed baseline.
