import sys
from pathlib import Path
from types import ModuleType

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    """experiments/qwen_sft/run.py's log() appends to the tracked results/run.log. The tests load run.py under several
    names (`run`, `qwen_sft_run`), so every loaded module's `_LOG` is pointed at tmp_path for the test. Another
    experiment's run.py can take over the name `run` while a test module still holds the old one, which then logged to the
    tracked file in a full run though not in a run of its file alone (friction 191), so modules held by others count too."""
    loaded = list(sys.modules.values())
    held = [v for m in loaded if (getattr(m, "__name__", "") or "").startswith("test_") for v in vars(m).values()]
    for mod in {id(m): m for m in loaded + [v for v in held if isinstance(v, ModuleType)]}.values():
        logs = getattr(mod, "_LOG", None)
        if isinstance(logs, list) and Path(getattr(mod, "__file__", None) or "/").is_relative_to(EXPERIMENTS):
            monkeypatch.setattr(mod, "_LOG", [tmp_path / "run.log"])
