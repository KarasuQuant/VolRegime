.PHONY: lint format style typecheck test

lint:
	uv run ruff check .

format:
	uv run black .

style: format lint

typecheck:
	uv run mypy .

test:
	uv run pytest