.PHONY: setup data lint typecheck test check

setup:
	uv sync --locked

data:
	uv run tool-abstention generate-dataset \
		--config configs/data/full.yaml \
		--output data/processed

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
