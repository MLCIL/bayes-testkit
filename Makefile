.PHONY: setup test test-coverage help
.DEFAULT_GOAL := help

setup:  ## Install development dependencies
	@# check if uv is installed
	@uv --version >/dev/null 2>&1 || (echo "uv is not installed, please install it" && exit 1)

	@# install dependencies
	uv sync --group dev --group test
	uv run pre-commit install

test:  ## Run tests without regression
	uv run ruff check
	uv run pytest tests -m "not slow"

fix-and-test:
	$(MAKE) ruff-fix
	$(MAKE) test-all

test-all:
	uv run pytest tests

test-coverage:  ## Run tests and calculate test coverage
	uv run pytest --cov=btk tests

ruff-fix:
	uv run ruff check --fix

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
