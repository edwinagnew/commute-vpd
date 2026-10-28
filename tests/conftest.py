import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    """experiments/qwen_sft/run.py's log() appends to the tracked results/run.log. The tests load run.py under several
    names (`run`, `qwen_sft_run`), so every loaded module's `_LOG` is pointed at tmp_path for the test."""
    for mod in list(sys.modules.values()):
        logs = getattr(mod, "_LOG", None)
        if isinstance(logs, list) and Path(getattr(mod, "__file__", None) or "/").is_relative_to(EXPERIMENTS):
            monkeypatch.setattr(mod, "_LOG", [tmp_path / "run.log"])
