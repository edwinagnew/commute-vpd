"""Small parts of an experiment script's life that src/commute's science modules do not cover.

`load_experiment` runs another experiment's file without PYTHONPATH, `python -P` or a pyright suppression;
`hard_exit` ends a job whose final write is done without waiting for the interpreter's teardown."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def load_experiment(folder: str, name: str, *also: str) -> Any:
    """experiments/<folder>/<name>.py as a module, with its bare sibling imports (`from run import ...`) taken from
    <folder> and then from the folders in `also`, whatever the caller's own folder, sys.path or sys.modules hold under
    those names; the caller's are put back afterwards. Experiments share file names (run.py, amend2.py, job.py), so
    import order and PYTHONPATH order decided which one a script got (friction 299, 369, 378, 381, 382, 383, 388).

    A file reached first through its own folder is one module, kept in sys.modules as `<folder>__<stem>`, so two calls
    see the same `run`, and a module the caller already imported from that file is that module. A file reached through
    `also` binds its sibling imports to <folder> first, so it is a separate module per <folder>, `<folder>+<its folder>__<stem>`.
    Its bare name exists only during the load: an `import run` inside a function called later fails or finds the caller's,
    and pickle cannot find its classes. Importing a file runs its top level, so it must not need more than that."""
    dirs = [EXPERIMENTS / d for d in (folder, *also)]
    owner = {f.stem: d for d in reversed(dirs) for f in d.glob("*.py")}
    assert name in {f.stem for f in dirs[0].glob("*.py")}, f"no {name}.py in experiments/{folder}"
    keys = {s: f"{folder}__{s}" if d == dirs[0] else f"{folder}+{d.name}__{s}" for s, d in owner.items()}
    for s, key in keys.items():
        mine = getattr(sys.modules.get(s), "__file__", None)
        if key not in sys.modules and owner[s] == dirs[0] and mine and Path(mine).resolve() == (dirs[0] / f"{s}.py").resolve():
            sys.modules[key] = sys.modules[s]
    held = {s: sys.modules.pop(s) for s in owner if s in sys.modules}
    sys.modules.update({s: sys.modules[key] for s, key in keys.items() if key in sys.modules})
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return importlib.import_module(name)
    finally:
        del sys.path[: len(dirs)]
        for s, key in keys.items():
            if s in sys.modules:
                sys.modules[key] = sys.modules.pop(s)
        sys.modules.update(held)


def hard_exit(code: int = 0) -> NoReturn:
    """A job's last line, after its final write: print the peak host memory, flush, and leave with os._exit. Jobs hung
    for 6 to 10+ min in interpreter teardown with their results already on disk and held a slot until scancel (friction
    150, 296). It skips atexit handlers and open files' close: everything the job needs must be written and closed first."""
    try:
        print(f"peak host RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB", flush=True)  # friction 132
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def write_result(path: Path, out: dict[str, Any], decomp_dir: Path) -> None:
    """Write `out` as JSON to `path` with out["decomp"] = the run id of `decomp_dir` (`qwen_vpd.decomp_id`). Raises if
    `path` already holds a result stamped with a different id (a resume after swapping --decomp). For amendment
    scripts; frozen scripts keep their own writers."""
    # imported here: qwen_vpd imports torch, which only a caller that stamps a Qwen result needs
    from commute.qwen_vpd import decomp_id

    did = decomp_id(decomp_dir)
    if path.is_file():
        old = json.loads(path.read_text()).get("decomp")
        if old is not None and old != did:
            raise RuntimeError(f"{path} holds decomposition {old}; this run reads {did}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**out, "decomp": did}, indent=1))


def require_filed(path: Path) -> None:
    """Exit unless `path` (a predictions file) is tracked by git and unchanged from HEAD: filed before the run.
    A working tree with no `.git` (a pod) compares the file's sha256 with the environment variable E120_PRED_SHA
    instead (E120 Amendment 1, friction 599); unset or different refuses."""
    repo, rel = str(EXPERIMENTS.parent), str(path.resolve().relative_to(EXPERIMENTS.parent.resolve()))
    if not (EXPERIMENTS.parent / ".git").exists():
        want, got = os.environ.get("E120_PRED_SHA"), hashlib.sha256(path.read_bytes()).hexdigest()
        if want != got:
            sys.exit(f"no .git here: {rel} sha256 {got} must equal E120_PRED_SHA ({want or 'unset'})")
        return
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False)
    if git("diff", "--quiet", "HEAD", "--", rel).returncode or not git("ls-files", rel).stdout.strip():
        sys.exit(f"{rel} is not committed unchanged: commit it first")
