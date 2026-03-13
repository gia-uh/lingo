.PHONY: all test lint format docs-serve docs-build

all: test lint

test:
	@echo "Running tests..."

docs-serve:
	@mkdocs serve

docs-build:
	@mkdocs build
