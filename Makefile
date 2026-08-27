.PHONY: setup data baseline-smoke prompt-diagnostic capacity-diagnostic baseline-validation lint typecheck test check

setup:
	uv sync --locked

data:
	uv run tool-abstention generate-dataset \
		--config configs/data/full.yaml \
		--output data/processed

baseline-smoke: data
	uv run --group inference tool-abstention infer \
		--config configs/models/qwen-smoke.yaml \
		--tasks data/processed/validation.jsonl \
		--output results/base/smoke/predictions.jsonl \
		--stratified-smoke
	uv run tool-abstention evaluate \
		--tasks data/processed/validation.jsonl \
		--predictions results/base/smoke/predictions.jsonl \
		--output results/base/smoke

prompt-diagnostic: data
	uv run --group inference tool-abstention infer --config configs/models/qwen-smoke.yaml --tasks data/processed/validation.jsonl --output results/base/prompt-diagnostic/native-full/predictions.jsonl --stratified-smoke --prompt-variant native-full
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/prompt-diagnostic/native-full/predictions.jsonl --output results/base/prompt-diagnostic/native-full
	uv run --group inference tool-abstention infer --config configs/models/qwen-smoke.yaml --tasks data/processed/validation.jsonl --output results/base/prompt-diagnostic/embedded-tools/predictions.jsonl --stratified-smoke --prompt-variant embedded-tools
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/prompt-diagnostic/embedded-tools/predictions.jsonl --output results/base/prompt-diagnostic/embedded-tools
	uv run --group inference tool-abstention infer --config configs/models/qwen-smoke.yaml --tasks data/processed/validation.jsonl --output results/base/prompt-diagnostic/native-short/predictions.jsonl --stratified-smoke --prompt-variant native-short
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/prompt-diagnostic/native-short/predictions.jsonl --output results/base/prompt-diagnostic/native-short

capacity-diagnostic: data
	uv run --group inference tool-abstention infer --config configs/models/qwen-1.5b-diagnostic.yaml --tasks data/processed/validation.jsonl --output results/base/capacity-diagnostic/native-full/predictions.jsonl --stratified-smoke --prompt-variant native-full
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/capacity-diagnostic/native-full/predictions.jsonl --output results/base/capacity-diagnostic/native-full
	uv run --group inference tool-abstention infer --config configs/models/qwen-1.5b-diagnostic.yaml --tasks data/processed/validation.jsonl --output results/base/capacity-diagnostic/embedded-tools/predictions.jsonl --stratified-smoke --prompt-variant embedded-tools
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/capacity-diagnostic/embedded-tools/predictions.jsonl --output results/base/capacity-diagnostic/embedded-tools
	uv run --group inference tool-abstention infer --config configs/models/qwen-1.5b-diagnostic.yaml --tasks data/processed/validation.jsonl --output results/base/capacity-diagnostic/native-short/predictions.jsonl --stratified-smoke --prompt-variant native-short
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/capacity-diagnostic/native-short/predictions.jsonl --output results/base/capacity-diagnostic/native-short

baseline-validation: data
	uv run --group inference tool-abstention infer --config configs/models/qwen-1.5b-diagnostic.yaml --tasks data/processed/validation.jsonl --output results/base/validation/predictions.jsonl --prompt-variant native-full
	uv run tool-abstention evaluate --tasks data/processed/validation.jsonl --predictions results/base/validation/predictions.jsonl --output results/base/validation

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
