"""The size check of qwen_files.fetch on a tiny fake directory; nothing here calls W&B."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from commute import qwen_files


def _fake(dest: Path) -> None:
    dest.mkdir()
    (dest / "a.safetensors").write_bytes(b"x" * 5)
    (dest / "launch_config.yaml").write_text("seed: 0\n")
    files = {"a.safetensors": {"sha256": "", "bytes": 5}}
    (dest / "index.json").write_text(json.dumps({"files": files}))


def test_complete_directory_needs_no_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _fake(tmp_path / "q")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    qwen_files.fetch(tmp_path / "q", log=lambda _: None)  # would raise if it went to W&B


def test_wrong_size_or_missing_file_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "q"
    _fake(dest)
    (dest / "a.safetensors").write_bytes(b"x" * 4)
    assert not qwen_files.complete(dest)
    _fake(tmp_path / "r")
    (tmp_path / "r" / "launch_config.yaml").unlink()
    assert not qwen_files.complete(tmp_path / "r")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        qwen_files.fetch(dest)
