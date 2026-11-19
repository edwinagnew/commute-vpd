"""commute.jobs: another experiment's file by path, with its sibling imports, and the exit after the final write."""

from __future__ import annotations

import hashlib
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


def _folders(tmp_path, monkeypatch, **files):
    """experiments/<folder>/<name>.py from keyword arguments folder__name=source."""
    for key, source in files.items():
        folder, name = key.split("__")
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / f"{name}.py").write_text(source)
    monkeypatch.setattr(jobs, "EXPERIMENTS", tmp_path)


def test_a_file_reached_through_also_does_not_poison_its_own_folder(tmp_path, monkeypatch):
    _folders(tmp_path, monkeypatch, pa__run="WHO = 'a'\n", pa__amend="import helper\n",
             pb__run="WHO = 'b'\n", pb__helper="import run\nWHO = run.WHO\n")
    assert jobs.load_experiment("pa", "amend", "pb").helper.WHO == "a"
    assert jobs.load_experiment("pb", "helper").WHO == "b"


def test_a_file_the_caller_already_imported_is_the_same_module(tmp_path, monkeypatch):
    _folders(tmp_path, monkeypatch, pc__run="import sys\nsys.ran_pc = getattr(sys, 'ran_pc', 0) + 1\n", pc__cnt="import run\n")
    monkeypatch.syspath_prepend(str(tmp_path / "pc"))
    monkeypatch.delitem(sys.modules, "run", raising=False)
    monkeypatch.delattr(sys, "ran_pc", raising=False)
    import run  # pyright: ignore[reportMissingImports]

    assert jobs.load_experiment("pc", "cnt").run is run and sys.ran_pc == 1  # pyright: ignore[reportAttributeAccessIssue]
    assert sys.modules["run"] is run
    del sys.modules["run"]


def test_run_items_goes_on_past_a_stopped_item():
    ran, lines = [], []

    def one(i: int) -> None:
        if i == 1:
            sys.exit("STOPPED: item 1's early check")
        ran.append(i)

    assert jobs.run_items([0, 1, 2], one, lines.append) == 1
    assert ran == [0, 2] and any("1 item(s) stopped" in x for x in lines)
    assert jobs.run_items([0, 2], one, lines.append) == 0


@pytest.mark.parametrize("code", [0, jobs.MORE_TO_DO])
def test_run_items_ends_the_job_on_a_wall_cut_or_exit_0(code: int):
    ran = []

    def one(i: int) -> None:
        if i == 0:
            sys.exit(code)
        ran.append(i)

    with pytest.raises(SystemExit) as e:
        jobs.run_items([0, 1], one, print)
    assert e.value.code == code and ran == []


def _filed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: bool) -> Path:
    """A repo root (tmp_path) with experiments/x/predictions.json; with `git`, an empty .git (a pod's failed clone)."""
    exps = tmp_path / "experiments"
    (exps / "x").mkdir(parents=True)
    (path := exps / "x" / "predictions.json").write_text("{}\n")
    if git:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(jobs, "EXPERIMENTS", exps)
    for name in ("PRED_SHA", "E120_PRED_SHA"):
        monkeypatch.delenv(name, raising=False)
    return path


@pytest.mark.parametrize("git", [False, True])
@pytest.mark.parametrize("name", ["PRED_SHA", "E120_PRED_SHA"])
def test_require_filed_takes_the_sha_when_set_even_beside_a_git(tmp_path, monkeypatch, git: bool, name: str):
    path = _filed(tmp_path, monkeypatch, git)
    monkeypatch.setenv(name, hashlib.sha256(path.read_bytes()).hexdigest())
    jobs.require_filed(path)  # friction 756, 767: a .git with no commit used to refuse here
    monkeypatch.setenv(name, "0" * 64)
    with pytest.raises(SystemExit, match="must equal PRED_SHA"):
        jobs.require_filed(path)


def test_require_filed_refuses_with_no_git_and_no_sha(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="unset"):
        jobs.require_filed(_filed(tmp_path, monkeypatch, False))


def test_stop_writes_a_stopped_marker_and_a_passing_rerun_removes_it(tmp_path: Path):
    path = tmp_path / "stage0_p5s1.json"
    with pytest.raises(SystemExit) as e:
        jobs.stop(path, {"a": 1}, "stage 0: a miss", log=lambda _m: None)
    assert e.value.code == 1 and (tmp_path / "stage0_p5s1.stopped").read_text() == "stage 0: a miss\n"
    jobs.write_stage(path, {"a": 1, "done": True})
    assert not (tmp_path / "stage0_p5s1.stopped").exists()
    assert jobs.stopped_marker(tmp_path / "fact.stopped.json").name == "fact.stopped"
