.PHONY: setup data baseline-smoke lint typecheck test check

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

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
