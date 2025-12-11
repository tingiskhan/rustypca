PHONY: format check develop

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
