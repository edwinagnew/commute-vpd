"""commute.jobs: another experiment's file by path, with its sibling imports, and the exit after the final write."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from commute import jobs


@pytest.fixture
def exps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Two experiments that share file names, as qwen_maps and qwen_pass_test share run.py."""
    for exp in ("a", "b"):
        (tmp_path / exp).mkdir()
        (tmp_path / exp / "run.py").write_text(f"WHO = {exp!r}\n")
        (tmp_path / exp / "amend.py").write_text("import run\nWHO = run.WHO\n")
    (tmp_path / "b" / "only_b.py").write_text("WHO = 'only_b'\n")
    (tmp_path / "a" / "uses_b.py").write_text("import only_b\nimport run\nWHO = (run.WHO, only_b.WHO)\n")
    monkeypatch.setattr(jobs, "EXPERIMENTS", tmp_path)
    yield tmp_path
    for name in [n for n in sys.modules if n.startswith(("a__", "b__"))]:
        del sys.modules[name]


def test_siblings_come_from_the_named_folder(exps: Path):
    assert jobs.load_experiment("a", "amend").WHO == "a"
    assert jobs.load_experiment("b", "amend").WHO == "b"


def test_the_callers_module_of_that_name_is_put_back(exps: Path, monkeypatch: pytest.MonkeyPatch):
    mine = sys.modules["pytest"]
    monkeypatch.setitem(sys.modules, "run", mine)
    path = list(sys.path)
    jobs.load_experiment("a", "amend")
    assert sys.modules["run"] is mine and sys.path == path


def test_one_file_is_one_module(exps: Path):
    run = jobs.load_experiment("a", "run")
    run.WHO = "changed"
    assert jobs.load_experiment("a", "amend").WHO == "changed"
    assert jobs.load_experiment("a", "run") is run


def test_also_folders_come_after_the_named_one(exps: Path):
    assert jobs.load_experiment("a", "uses_b", "b").WHO == ("a", "only_b")


def test_a_missing_file_is_named(exps: Path):
    with pytest.raises(AssertionError, match="no nope.py in experiments/a"):
        jobs.load_experiment("a", "nope")


def test_hard_exit_flushes_and_does_not_wait_for_teardown():
    """The hang at exit is stood in for by a live non-daemon thread and an atexit handler that never return."""
    script = textwrap.dedent("""
        import atexit, sys, threading, time
        from commute import jobs
        atexit.register(lambda: time.sleep(600))
        threading.Thread(target=time.sleep, args=(600,)).start()
        print("final line")  # block-buffered: stdout is a pipe
        jobs.hard_exit(3)
    """)
    r = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "final line" in r.stdout and "peak host RSS" in r.stdout, (r.returncode, r.stdout, r.stderr)
