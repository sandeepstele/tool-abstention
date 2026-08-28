# Numerically verified MLX DPO smoke

## Scope

Milestone J adds a project-owned DPO implementation for the pinned `mlx==0.32.2`
and `mlx-lm==0.29.1` stack. MLX-LM does not provide a released DPO trainer. The
implementation initializes policy and reference behavior from the same selected
SFT adapter, precomputes the frozen reference log probabilities, and keeps only
one policy model resident during optimization.

BFCL, AgentAbstain, the rejected protocol-repair adapter, and the internal test
split are prohibited inputs. The prepared manifests declare only internal train
and validation sources. The held-out test SHA-256 remained
`76bbac17a10e87c9cb58aaaacf1b2be8c5dccbd22790c19e8e01a04c49f59bc8`.

## Objective and masking

For chosen completion `y+`, rejected completion `y-`, policy `π`, frozen
reference `ref`, and inverse-temperature `β`, the per-example objective is:

```text
-logsigmoid(β * ((log π(y+|x) - log π(y-|x))
                    - (log ref(y+|x) - log ref(y-|x))))
```

Optional conservative label smoothing mixes the positive and reversed terms.
Sequence log probabilities sum only assistant-completion tokens. Preparation
requires byte-identical prompt-token prefixes, records the prompt boundary, and
rejects truncation that removes a completion, changes the boundary, or erases the
chosen/rejected distinction.

## Reference cache and failure policy

Each cache record contains chosen/rejected log probabilities, completion-token
counts, and the canonical example hash. Its manifest pins the model revision,
initial adapter hash, tokenizer/prompt identity, input-file hash, maximum length,
record count, and records hash. Missing, duplicate, stale, non-finite, or
mismatched entries fail before training.

The runner freezes the quantized base and verifies that every trainable parameter
is LoRA. It fails on incompatible package versions, adapter hashes or LoRA layout,
non-finite loss/gradients, zero completion tokens, accumulation misalignment,
missing output adapters, reload disagreement, or excessive peak memory.

CPU CI measures the pure contracts, numerical fixed vectors, cache validation,
leakage controls, and a fake CLI/runtime boundary. The Metal-only module is
excluded from CPU coverage because importing MLX requires an available Metal
device; it is exercised by the recorded compatibility smoke.

## 0.5B compatibility smoke

Configuration: 16 deterministic train preferences, 8 validation preferences,
`β=0.1`, no smoothing, batch 1, 20 iterations, learning rate `1e-5`, four LoRA
layers, and maximum length 1024. Initialization was the existing 0.5B SFT adapter
`9fba9c2…eba9`.

| Gate | Result |
|---|---:|
| Initial validation reward margin | 0.000000 |
| Final validation reward margin | 3.576737 |
| Final validation reward accuracy | 100% |
| Final validation DPO loss | 0.064053 |
| Train / validation truncations | 0 / 0 |
| Peak unified memory | 3.172 GB |
| Reload maximum metric difference | 0.0 |
| Output adapter SHA-256 | `1989858a…17f7` |
| Promotion gate | **PASS** |

All observed losses and gradients were finite. The reloaded adapter reproduced
every final validation metric exactly, and its hash differs from initialization.
The training path took about 16 seconds after reference caching on the local Apple
Metal device.

## Limitations and next gate

The smoke set is deliberately tiny and demonstrates numerical/runtime
compatibility, not research effectiveness. A positive chosen reward is not itself
required: DPO depends on the chosen-versus-rejected reward margin. Here both moved,
but rejected completions were penalized much more strongly.

The passed smoke permits one predeclared 1.5B seed-0 run from the original SFT
adapter. Internal validation and protocol-stress gates must pass before BFCL is
opened. BFCL remains reporting-only and cannot select or retry the checkpoint.
