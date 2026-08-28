# 18 — Preference Data Foundation

## Contract

`PreferenceRecord` is a strict immutable contract tying one chosen/rejected pair to
an internal task, pair, split, target decision class, controlled negative type,
source-task hash, generator version, and selected SFT initialization hash. It
forbids unknown fields, test records, identical responses, and external provenance.

Generation validates both responses with the deterministic evaluator. Chosen
responses must be semantically correct and protocol-valid. Rejected responses must
fail, and only `malformed_tool_call` negatives may fail protocol validity. Tool
semantic negatives must retain parseable call syntax.

## Negative taxonomy

- `wrong_decision_abstain`
- `wrong_abstention_class`
- `unnecessary_call`
- `wrong_tool`
- `wrong_arguments`
- `malformed_tool_call`
- `schema_copying`

The first generated dataset contains 360 train and 120 validation preferences,
one per internal task. Its 480 negatives contain 128 malformed calls (26.7%), 160
wrong/unnecessary decision examples, and 192 other semantic negatives. Syntax
therefore cannot dominate the objective as it did in the rejected repair ablation.

The manifest pins SFT adapter
`88841d6959a751cea2b60b88788b3552c283fc82acdfb9ce43ca08988a582556`,
records zero external sources and `test_consumed=false`, and hashes both source
splits and generated artifacts. Regeneration is byte-identical.

## DPO trainer boundary

The pinned `mlx-lm 0.29` installation contains SFT/LoRA dataset and trainer paths
but no DPO command, preference dataset, or supported DPO loss. The official MLX-LM
LoRA documentation lists LoRA, DoRA, and full fine-tuning; the MLX examples DPO
request remains an open enhancement with a proposed loss sketch rather than a
released trainer.

Primary references: [MLX-LM LoRA documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
and [open MLX DPO feature request](https://github.com/ml-explore/mlx-examples/issues/513).

Therefore no model smoke was mislabeled as DPO in this milestone. The next step is
a narrow local runner with fixed chosen/rejected token masking, a frozen SFT
reference model, tested log-probability and DPO-loss vectors, reward-margin logging,
and a 0.5B/10–20-step Metal smoke. It must be isolated from the public evaluator
and must not proceed to 1.5B until its numerical tests and adapter reload pass.
