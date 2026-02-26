.PHONY: format check develop test

format:
	uv run ruff format ppca
	uv run ruff check ppca --fix

check:
	uv run ruff format ppca --check
	uv run ruff check ppca

develop:
	uv run maturin develop

test:
	uv run pytest -v ppca
