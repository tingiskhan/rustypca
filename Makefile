.PHONY: format check develop test

format:
	ruff format ppca
	ruff check ppca --fix

check:
	ruff format ppca --check
	ruff check ppca

develop:
	maturin develop

test:
	pytest -v ppca
