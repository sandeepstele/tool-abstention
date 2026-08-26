.PHONY: setup data lint typecheck test check

setup:
	uv sync --locked

data:
	uv run tool-abstention generate-productivity \
		--config configs/data/productivity.yaml \
		--output data/raw/productivity

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
