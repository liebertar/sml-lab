.PHONY: env check test lint

env:
	uv sync --all-extras

check:
	uv run python scripts/check_env.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
