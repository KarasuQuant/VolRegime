.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run black .

.PHONY: style
style: format lint

.PHONY: typecheck
typecheck:
	uv run mypy .

.PHONY: test
test:
	uv run pytest

.PHONY: fetch
fetch:
	uv run volregime fetch