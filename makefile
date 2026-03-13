.PHONY: all test lint format docs-serve docs-build

all: format lint test

test:
	@echo "Running tests..."
	@uv run pytest

lint:
	@echo "Linting with ruff..."
	@uv run ruff check .

format:
	@echo "Formatting with ruff..."
	@uv run ruff format .

docs-serve:
	@mkdocs serve

docs-build:
	@mkdocs build
