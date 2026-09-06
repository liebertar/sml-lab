.PHONY: env check test lint baseline

env:
	uv sync --all-extras

check:
	uv run python scripts/check_env.py

test:
	uv run pytest -q

lint:
	uv run ruff check .

baseline:
	uv run python experiments/00-env-baseline/run.py $(ARGS)
