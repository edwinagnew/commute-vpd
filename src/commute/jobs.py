"""Small parts of an experiment script's life that src/commute's science modules do not cover.

`load_experiment` runs another experiment's file without PYTHONPATH, `python -P` or a pyright suppression;
`hard_exit` ends a job whose final write is done without waiting for the interpreter's teardown."""

from __future__ import annotations

import importlib
import logging
import os
import resource
import sys
from pathlib import Path
from typing import Any, NoReturn

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def load_experiment(folder: str, name: str, *also: str) -> Any:
    """experiments/<folder>/<name>.py as a module, with its bare sibling imports (`from run import ...`) taken from
    <folder> and then from the folders in `also`, whatever the caller's own folder, sys.path or sys.modules hold under
    those names; the caller's are put back afterwards. Experiments share file names (run.py, amend2.py, job.py), so
    import order and PYTHONPATH order decided which one a script got (friction 299, 369, 378, 381, 382, 383, 388).

    A file is one module however it is reached, kept in sys.modules as `<folder>__<stem>`, so two calls see the same
    `run`. Its bare name exists only during the load: an `import run` inside a function called later fails or finds the
    caller's, and pickle cannot find its classes. Importing a file runs its top level, so it must not need more than that."""
    dirs = [EXPERIMENTS / d for d in (folder, *also)]
    owner = {f.stem: d.name for d in reversed(dirs) for f in d.glob("*.py")}
    assert name in {f.stem for f in dirs[0].glob("*.py")}, f"no {name}.py in experiments/{folder}"
    held = {s: sys.modules.pop(s) for s in owner if s in sys.modules}
    sys.modules.update({s: sys.modules[f"{d}__{s}"] for s, d in owner.items() if f"{d}__{s}" in sys.modules})
    sys.path[:0] = [str(d) for d in dirs]
    try:
        return importlib.import_module(name)
    finally:
        del sys.path[: len(dirs)]
        for s, d in owner.items():
            if s in sys.modules:
                sys.modules[f"{d}__{s}"] = sys.modules.pop(s)
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
