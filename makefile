.PHONY: all test lint format docs-serve docs-build book book-compile

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

BOOK_FILES = book/01-hello-lingo.md book/02-messages-and-context.md book/03-the-engine.md book/04-flows.md book/05-tools.md book/06-skills-and-routing.md book/07-state.md book/08-patterns.md

book-compile:
	@echo "Compiling book..."
	@illiterate -d book $(BOOK_FILES)

book: book-compile
	@echo "Testing book..."
	@uv run pytest book/tests/ -v
