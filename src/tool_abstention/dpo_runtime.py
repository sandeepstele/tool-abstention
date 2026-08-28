"""Lazy MLX runtime for frozen-reference caching and custom DPO training."""

import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, cast

from tool_abstention.config import load_yaml_config
from tool_abstention.dpo import (
    DpoExample,
    DpoTrainingConfig,
    ReferenceCacheManifest,
    ReferenceLogps,
    TokenizedDpoPair,
    numpy_dpo_metrics,
    tokenize_dpo_example,
    validate_reference_cache,
)
from tool_abstention.util.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_object,
)
from tool_abstention.util.jsonl import read_jsonl, write_jsonl


def _check_versions(config: DpoTrainingConfig) -> None:
    found_mlx = importlib.metadata.version("mlx")
    found_mlx_lm = importlib.metadata.version("mlx-lm")
    if found_mlx != config.required_mlx_version:
        raise ValueError(
            "MLX version mismatch: expected "
            f"{config.required_mlx_version}, found {found_mlx}"
        )
    if found_mlx_lm != config.required_mlx_lm_version:
        raise ValueError(
            "MLX-LM version mismatch: expected "
            f"{config.required_mlx_lm_version}, found {found_mlx_lm}"
        )


def _tokenizer_hash(tokenizer: Any) -> str:
    return sha256_object(
        {
            "class": type(tokenizer).__name__,
            "chat_template": getattr(tokenizer, "chat_template", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
    )


def _mlx_sequence_logp(
    model: Any, tokens: tuple[int, ...], offset: int
) -> tuple[Any, int]:
    import mlx.core as mx

    if len(tokens) <= offset:
        raise ValueError("DPO sequence has zero completion tokens")
    sequence = mx.array(tokens)
    logits = model(sequence[None, :-1]).astype(mx.float32)
    targets = sequence[None, 1:]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    gathered = mx.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    start = offset - 1
    if start < 0 or start >= gathered.shape[1]:
        raise ValueError("invalid DPO completion offset")
    value = gathered[:, start:].sum()
    count = int(gathered.shape[1] - start)
    if count <= 0:
        raise ValueError("DPO sequence has zero completion tokens")
    return value, count


def _load_examples(path: Path) -> list[DpoExample]:
    examples = [DpoExample.model_validate(value) for value in read_jsonl(path)]
    if not examples:
        raise ValueError("DPO example file is empty")
    if len({item.id for item in examples}) != len(examples):
        raise ValueError("DPO example ids must be unique")
    return examples


def _validate_adapter_config(config: DpoTrainingConfig, adapter: Path) -> None:
    raw = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    lora = raw.get("lora_parameters", {})
    expected = {
        "num_layers": config.num_layers,
        "rank": config.rank,
        "dropout": config.dropout,
        "scale": config.scale,
    }
    actual = {
        "num_layers": raw.get("num_layers"),
        "rank": lora.get("rank"),
        "dropout": lora.get("dropout"),
        "scale": lora.get("scale"),
    }
    if actual != expected:
        raise ValueError("SFT adapter LoRA configuration does not match DPO config")


def cache_reference_logps(
    config_path: Path, examples_path: Path, output: Path
) -> dict[str, Any]:
    """Compute frozen SFT reference log probabilities on local Metal."""
    import mlx.core as mx
    from mlx_lm import load

    config = load_yaml_config(config_path, DpoTrainingConfig)
    _check_versions(config)
    adapter = Path(config.sft_adapter_path)
    if sha256_file(adapter / "adapters.safetensors") != config.sft_init_hash:
        raise ValueError("SFT adapter hash does not match DPO configuration")
    _validate_adapter_config(config, adapter)
    examples = _load_examples(examples_path)
    if any(item.sft_init_hash != config.sft_init_hash for item in examples):
        raise ValueError("DPO examples use a different SFT initialization")
    model, tokenizer = cast(Any, load)(
        config.model,
        revision=config.revision,
        adapter_path=str(adapter),
        tokenizer_config={"trust_remote_code": False},
        return_config=False,
    )
    model.freeze()
    model.eval()
    records: list[ReferenceLogps] = []
    for example in examples:
        pair = tokenize_dpo_example(
            example, tokenizer, max_seq_length=config.max_seq_length
        )
        chosen, chosen_tokens = _mlx_sequence_logp(
            model, pair.chosen, pair.chosen_offset
        )
        rejected, rejected_tokens = _mlx_sequence_logp(
            model, pair.rejected, pair.rejected_offset
        )
        mx.eval(chosen, rejected)
        records.append(
            ReferenceLogps(
                id=example.id,
                example_hash=sha256_object(example.model_dump(mode="json")),
                chosen_logp=float(chosen.item()),
                rejected_logp=float(rejected.item()),
                chosen_tokens=chosen_tokens,
                rejected_tokens=rejected_tokens,
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    canonical_config_path = output / "config.json"
    canonical_config_path.write_bytes(
        canonical_json_bytes(config.model_dump(mode="json")) + b"\n"
    )
    records_path = output / "records.jsonl"
    write_jsonl(records_path, [item.model_dump(mode="json") for item in records])
    manifest = ReferenceCacheManifest(
        model=config.model,
        revision=config.revision,
        adapter_hash=config.sft_init_hash,
        examples_hash=sha256_file(examples_path),
        tokenizer_hash=_tokenizer_hash(tokenizer),
        max_seq_length=config.max_seq_length,
        record_count=len(records),
        records_hash=sha256_file(records_path),
    )
    (output / "manifest.json").write_bytes(
        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    return manifest.model_dump(mode="json")


def _pair_logps(model: Any, pair: TokenizedDpoPair) -> tuple[Any, Any, int]:
    chosen, chosen_tokens = _mlx_sequence_logp(model, pair.chosen, pair.chosen_offset)
    rejected, rejected_tokens = _mlx_sequence_logp(
        model, pair.rejected, pair.rejected_offset
    )
    return chosen, rejected, chosen_tokens + rejected_tokens


def _evaluate_policy(
    model: Any,
    pairs: list[TokenizedDpoPair],
    reference: dict[str, ReferenceLogps],
    config: DpoTrainingConfig,
) -> dict[str, float]:
    import mlx.core as mx

    policy_chosen: list[float] = []
    policy_rejected: list[float] = []
    ref_chosen: list[float] = []
    ref_rejected: list[float] = []
    model.eval()
    for pair in pairs:
        chosen, rejected, _ = _pair_logps(model, pair)
        mx.eval(chosen, rejected)
        policy_chosen.append(float(chosen.item()))
        policy_rejected.append(float(rejected.item()))
        ref_chosen.append(reference[pair.id].chosen_logp)
        ref_rejected.append(reference[pair.id].rejected_logp)
    import numpy as np

    return numpy_dpo_metrics(
        np.asarray(policy_chosen),
        np.asarray(policy_rejected),
        np.asarray(ref_chosen),
        np.asarray(ref_rejected),
        beta=config.beta,
        label_smoothing=config.label_smoothing,
    )


def train_dpo(
    config_path: Path,
    train_examples_path: Path,
    valid_examples_path: Path,
    train_cache: Path,
    valid_cache: Path,
    output: Path,
) -> dict[str, Any]:
    """Train LoRA policy with cached-reference DPO and reload verification."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    config = load_yaml_config(config_path, DpoTrainingConfig)
    _check_versions(config)
    adapter = Path(config.sft_adapter_path)
    if "protocol-repair" in adapter.parts:
        raise ValueError("the rejected protocol-repair adapter is prohibited from DPO")
    if sha256_file(adapter / "adapters.safetensors") != config.sft_init_hash:
        raise ValueError("SFT adapter hash does not match DPO configuration")
    _validate_adapter_config(config, adapter)
    train_examples, train_reference = validate_reference_cache(
        train_examples_path,
        train_cache / "records.jsonl",
        train_cache / "manifest.json",
        config=config,
    )
    valid_examples, valid_reference = validate_reference_cache(
        valid_examples_path,
        valid_cache / "records.jsonl",
        valid_cache / "manifest.json",
        config=config,
    )
    model, tokenizer = cast(Any, load)(
        config.model,
        revision=config.revision,
        tokenizer_config={"trust_remote_code": False},
        return_config=False,
    )
    model.freeze()
    load_adapters(model, str(adapter))
    tokenizer_hash = _tokenizer_hash(tokenizer)
    for cache_path in (train_cache, valid_cache):
        cache_manifest = ReferenceCacheManifest.model_validate_json(
            (cache_path / "manifest.json").read_text(encoding="utf-8")
        )
        if cache_manifest.tokenizer_hash != tokenizer_hash:
            raise ValueError("reference cache tokenizer hash does not match")
    trainable = cast(list[tuple[str, Any]], tree_flatten(model.trainable_parameters()))
    if not trainable or any("lora" not in name.casefold() for name, _ in trainable):
        raise ValueError("DPO policy must expose only trainable LoRA parameters")
    train_pairs = [
        tokenize_dpo_example(item, tokenizer, max_seq_length=config.max_seq_length)
        for item in train_examples
    ]
    valid_pairs = [
        tokenize_dpo_example(item, tokenizer, max_seq_length=config.max_seq_length)
        for item in valid_examples
    ]
    initial_metrics = _evaluate_policy(model, valid_pairs, valid_reference, config)
    optimizer = optim.AdamW(learning_rate=config.learning_rate)

    def loss_fn(policy: Any, pair: TokenizedDpoPair) -> tuple[Any, Any]:
        chosen, rejected, token_count = _pair_logps(policy, pair)
        reference = train_reference[pair.id]
        logit = (chosen - rejected) - (reference.chosen_logp - reference.rejected_logp)
        scaled = config.beta * logit
        positive_log_sigmoid = -mx.logaddexp(mx.array(0.0), -scaled)
        negative_log_sigmoid = -mx.logaddexp(mx.array(0.0), scaled)
        loss = -(1 - config.label_smoothing) * positive_log_sigmoid
        loss -= config.label_smoothing * negative_log_sigmoid
        return loss, mx.array(token_count)

    value_and_grad = cast(Any, nn).value_and_grad(model, loss_fn)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(train_pairs)).tolist()
    grad_accum: Any = None
    metrics_rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    canonical_config_path = output / "config.json"
    canonical_config_path.write_bytes(
        canonical_json_bytes(config.model_dump(mode="json")) + b"\n"
    )
    started = time.perf_counter()
    model.train()
    for iteration in range(1, config.iters + 1):
        pair = train_pairs[order[(iteration - 1) % len(order)]]
        (loss_and_tokens, gradients) = value_and_grad(model, pair)
        loss, token_count = loss_and_tokens
        if grad_accum is None:
            grad_accum = gradients
        else:
            grad_accum = tree_map(
                lambda left, right: left + right, grad_accum, gradients
            )
        mx.eval(loss, token_count, grad_accum)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise ValueError("DPO training produced non-finite loss")
        if any(
            not bool(mx.all(mx.isfinite(value)).item())
            for _, value in cast(list[tuple[str, Any]], tree_flatten(grad_accum))
        ):
            raise ValueError("DPO training produced non-finite gradients")
        if iteration % config.grad_accumulation_steps == 0:
            scaled_grad = tree_map(
                lambda value: value / config.grad_accumulation_steps, grad_accum
            )
            optimizer.update(model, scaled_grad)
            mx.eval(model.state, optimizer.state)
            grad_accum = None
        if iteration % config.steps_per_report == 0 or iteration == config.iters:
            row = {
                "iteration": iteration,
                "loss": loss_value,
                "completion_tokens": int(token_count.item()),
                "peak_memory_gb": float(mx.get_peak_memory()) / 1e9,
            }
            metrics_rows.append(row)
            print(json.dumps(row), flush=True)
        if iteration % config.steps_per_eval == 0 or iteration == config.iters:
            evaluation = _evaluate_policy(model, valid_pairs, valid_reference, config)
            metrics_rows.append({"iteration": iteration, "validation": evaluation})
            model.train()
        if iteration % config.save_every == 0:
            weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(
                str(output / f"{iteration:07d}_adapters.safetensors"), weights
            )
    if grad_accum is not None:
        raise ValueError("iterations must align with gradient accumulation steps")
    adapter_file = output / "adapters.safetensors"
    mx.save_safetensors(
        str(adapter_file), dict(tree_flatten(model.trainable_parameters()))
    )
    shutil.copy2(adapter / "adapter_config.json", output / "adapter_config.json")
    write_jsonl(output / "metrics.jsonl", metrics_rows)
    final_metrics = _evaluate_policy(model, valid_pairs, valid_reference, config)
    peak_memory = float(mx.get_peak_memory()) / 1e9
    if peak_memory >= config.peak_memory_limit_gb:
        raise ValueError("DPO smoke exceeded its peak-memory limit")
    reloaded, _ = cast(Any, load)(
        config.model,
        revision=config.revision,
        adapter_path=str(output),
        tokenizer_config={"trust_remote_code": False},
        return_config=False,
    )
    reload_metrics = _evaluate_policy(reloaded, valid_pairs, valid_reference, config)
    if any(
        abs(final_metrics[key] - reload_metrics[key]) > 1e-5 for key in final_metrics
    ):
        raise ValueError("reloaded DPO adapter metrics do not reproduce")
    gate_passed = (
        final_metrics["reward_margin"] > initial_metrics["reward_margin"]
        and final_metrics["reward_margin"] > 0
        and final_metrics["reward_accuracy"] >= 0.6
        and sha256_file(adapter_file) != config.sft_init_hash
    )
    command = [
        sys.executable,
        "-m",
        "tool_abstention",
        "train-dpo",
        "--config",
        str(config_path),
        "--train-examples",
        str(train_examples_path),
        "--valid-examples",
        str(valid_examples_path),
        "--train-cache",
        str(train_cache),
        "--valid-cache",
        str(valid_cache),
        "--output",
        str(output),
    ]
    manifest = {
        "schema_version": 1,
        "config_hash": sha256_file(config_path),
        "canonical_config_hash": sha256_file(canonical_config_path),
        "train_examples_hash": sha256_file(train_examples_path),
        "valid_examples_hash": sha256_file(valid_examples_path),
        "train_cache_hash": sha256_file(train_cache / "manifest.json"),
        "valid_cache_hash": sha256_file(valid_cache / "manifest.json"),
        "initial_adapter_hash": config.sft_init_hash,
        "output_adapter_hash": sha256_file(adapter_file),
        "metrics_hash": sha256_file(output / "metrics.jsonl"),
        "command": command,
        "train_truncation_count": sum(
            pair.chosen_truncated + pair.rejected_truncated for pair in train_pairs
        ),
        "validation_truncation_count": sum(
            pair.chosen_truncated + pair.rejected_truncated for pair in valid_pairs
        ),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "reload_metrics": reload_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory,
        "gate_passed": gate_passed,
    }
    (output / "run_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest
