SHELL := /bin/sh

.PHONY: help setup test test-q lint format typecheck check

help:
	@echo "Targets:"
	@echo "  make setup     Install uv (if missing) and sync Python deps"
	@echo "  make test      Run pytest"
	@echo "  make test-q    Run pytest in quiet mode"
	@echo "  make lint      Run ruff checks"
	@echo "  make format    Run ruff format"
	@echo "  make typecheck Run mypy"
	@echo "  make check     Run lint, typecheck, and tests"

setup:
	@./tools/setup-python.sh

test:
	@uv run pytest

test-q:
	@uv run pytest -q

lint:
	@uv run ruff check .

format:
	@uv run ruff format .

typecheck:
	@uv run mypy src

check: lint typecheck test-q
