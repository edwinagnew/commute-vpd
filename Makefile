.PHONY: sync test lint

# On the cluster .venv is a link to the one venv every clone shares (fleet/link-data), built with the CUDA torch group
# by cluster/setup.sbatch. A bare `uv run` or `uv sync` there reinstalls the default (CPU) torch and points the venv's
# editable install at this clone, which broke every GPU job on 2026-10-27. So `uv run` never syncs, and the targets
# import this clone's own src/ (pytest's pythonpath; pyright adds src/ itself).
export UV_NO_SYNC := 1

# Python files changed since the last commit (tracked and untracked); override with `make lint FILES="a.py b.py"`.
FILES ?= $(shell { git diff --name-only HEAD; git ls-files --others --exclude-standard; } | grep '\.py$$' | grep -v '/results/' | sort -u)

sync:
	@if [ -L .venv ]; then echo "make sync: .venv is the shared cluster venv; rebuild it with: sbatch cluster/setup.sbatch" >&2; exit 1; fi
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check $(FILES)
	uv run pyright $(FILES)
