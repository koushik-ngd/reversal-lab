.PHONY: install lint format typecheck test check experiment build

install:
	uv sync --frozen --all-groups --extra service

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

experiment:
	uv run reversallab validate-estimators --config configs/experiments/001_estimator_validation.yaml

build:
	uv build --wheel
