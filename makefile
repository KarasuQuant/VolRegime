.PHONY: lint format style typecheck

lint:
	uv run ruff check .

format:
	uv run black .

style: format lint

typecheck:
	uv run mypy .