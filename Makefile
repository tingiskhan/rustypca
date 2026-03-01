.PHONY: format check develop test

format:
	cargo fmt
	uv run ruff format rustypca
	uv run ruff check rustypca --fix

check:
	cargo fmt --check
	uv run ruff format rustypca --check
	uv run ruff check rustypca

develop:
	uv run maturin develop

test:
	uv run pytest -v rustypca
