.PHONY: sync test lint

# Python files changed since the last commit (tracked and untracked); override with `make lint FILES="a.py b.py"`.
FILES ?= $(shell { git diff --name-only HEAD; git ls-files --others --exclude-standard; } | grep '\.py$$' | grep -v '/results/' | sort -u)

sync:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check $(FILES)
	uv run pyright $(FILES)
