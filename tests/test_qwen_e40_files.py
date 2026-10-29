"""qwen_e40_files' sha256 check on tiny fake files; nothing here calls W&B."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from commute import qwen_e40_files as e


def test_verified_and_fetch_skips_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dw = tmp_path / "dW"
    dw.mkdir()
    for name in e.FILES:
        (dw / name).write_bytes(name.encode())
    (dw / e.SUMS).write_text(subprocess.run(["sha256sum", *e.FILES], cwd=dw, capture_output=True, text=True,
                                            check=True).stdout)
    assert e.verified(tmp_path) and set(e.sums(tmp_path)) == set(e.FILES)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    e.fetch(tmp_path)  # would raise if it went to W&B
    (dw / e.FILES[0]).write_bytes(b"changed")
    assert not e.verified(tmp_path)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        e.fetch(tmp_path)
