"""Compatibility boundary for MLX-LM chat training with Transformers 5."""

from typing import Any


def token_ids(value: Any) -> list[int]:
    """Normalize legacy token lists and Transformers 5 BatchEncoding values."""
    if hasattr(value, "keys") and "input_ids" in value:
        result = value["input_ids"]
    else:
        result = value
    if not isinstance(result, list) or any(
        not isinstance(item, int) for item in result
    ):
        raise TypeError("chat template must produce a flat integer token list")
    return result


def main() -> None:  # pragma: no cover - exercised by the real Metal smoke
    from mlx_lm import lora
    from mlx_lm.tuner import datasets

    def process(dataset: Any, datum: dict[str, Any]) -> tuple[list[int], int]:
        messages = datum[dataset.chat_key]
        tools = datum.get("tools")
        tokens = token_ids(dataset.tokenizer.apply_chat_template(messages, tools=tools))
        if not dataset.mask_prompt:
            return tokens, 0
        add_generation_prompt = messages[-1].get("role") == "assistant"
        prompt = dataset.tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        return tokens, len(token_ids(prompt))

    datasets.ChatDataset.process = process  # type: ignore[method-assign]
    lora.main()  # type: ignore[no-untyped-call]


if __name__ == "__main__":  # pragma: no cover
    main()
