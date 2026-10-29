"""qwen_e40_files' sha256 checks on tiny fake files; nothing here calls W&B."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from commute import qwen_e40_files as e


def test_verified_needs_the_committed_sums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dw = tmp_path / "dW"
    dw.mkdir()
    for name in e.FILES:
        (dw / name).write_bytes(name.encode())
    (dw / e.SUMS).write_text(subprocess.run(["sha256sum", *e.FILES], cwd=dw, capture_output=True, text=True,
                                            check=True).stdout)
    committed = tmp_path / "committed"
    monkeypatch.setattr(e, "COMMITTED", committed)
    assert not e.verified(tmp_path)  # no committed sums: never the cluster's files
    committed.write_text((dw / e.SUMS).read_text())
    assert e.verified(tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    e.fetch(tmp_path, "v0")  # would raise if it went to W&B
    committed.write_text((dw / e.SUMS).read_text().replace(e.FILES[0], "other.safetensors"))
    assert not e.verified(tmp_path)  # sums that self-agree but differ from the committed ones
    committed.write_text((dw / e.SUMS).read_text())
    (dw / e.FILES[0]).write_bytes(b"changed")
    assert not e.verified(tmp_path)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        e.fetch(tmp_path, "v0")
