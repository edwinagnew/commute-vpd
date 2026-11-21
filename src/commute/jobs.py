"""Small parts of an experiment script's life that src/commute's science modules do not cover.

`load_experiment` runs another experiment's file without PYTHONPATH, `python -P` or a pyright suppression;
`hard_exit` ends a job whose final write is done without waiting for the interpreter's teardown; `run_items` runs a
job's list of items so that one item's stop does not end the rest; `stop` and `write_stage` write a stage's JSON with a
`.stopped` marker file beside it whenever the stage stopped."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import resource
import subprocess
import sys
from collections.abc import Callable, Iterable
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
        _write_inputs()  # os._exit skips atexit (record_inputs)
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


MORE_TO_DO = 3  # the exit code fleet/resubmit reads as "resubmit the same line"


def run_items[T](items: Iterable[T], run_one: Callable[[T], object], log: Callable[[str], object]) -> int:
    """Each item in turn, for a job that runs a list of items. An item that stops, by `sys.exit` with a message or
    code 1 as an experiment's `stop()` does after writing its record, ends that item only and the next one runs: one
    stop in E139 ended its job and the items after it never ran (friction 671). A wall cut (exit 3) or exit 0 ends the
    job as before, and any other exception propagates. Returns 1 when any item stopped, else 0."""
    stopped = []
    for item in items:
        try:
            run_one(item)
        except SystemExit as e:
            if e.code in (0, None, MORE_TO_DO):
                raise
            stopped.append(item)
            log(f"item {item} stopped ({e.code}); going on to the next item")
    if stopped:
        log(f"{len(stopped)} item(s) stopped: {stopped}")
    return 1 if stopped else 0


def stopped_marker(path: Path) -> Path:
    """The `.stopped` marker file beside a stage JSON: results/stage0_p5s1.json -> results/stage0_p5s1.stopped (an
    item's `<item>.stopped.json` -> `<item>.stopped`)."""
    return path.with_name(path.name.removesuffix(".json").removesuffix(".stopped") + ".stopped")


def write_stage(path: Path, obj: dict[str, Any]) -> None:
    """Write a stage's JSON (indent 1), and keep its `.stopped` marker in step: written (holding the reason) when the
    JSON carries a "stopped" key, removed otherwise, so a stage re-run that now passes leaves no stale marker. A runner
    told to watch for a stop can wait on the file; the key alone made one wait out its cap (friction 861)."""
    path.write_text(json.dumps(obj, indent=1))
    marker = stopped_marker(path)
    if "stopped" in obj:
        marker.write_text(f"{obj['stopped']}\n")
    else:
        marker.unlink(missing_ok=True)


def stop(path: Path, out: dict[str, Any], why: str, log: Callable[[str], object] = print) -> NoReturn:
    """A stage's stop: `out` plus {"stopped": why} to `path`, the `.stopped` marker beside it (`write_stage`), a STOP
    line in the log, exit 1."""
    write_stage(path, out | {"stopped": why})
    log(f"STOP: {why}")
    sys.exit(1)


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
    When the environment variable PRED_SHA is set (or E120_PRED_SHA, its old name, still read for briefs in flight;
    friction 727), the file's sha256 must equal it, whether or not a `.git` exists: a pod's clone step can leave a
    `.git` whose HEAD lacks the local commit, and git then refused (friction 756, 767). A working tree with no `.git`
    (a pod) and no PRED_SHA refuses (E120 Amendment 1, friction 599). The brief computes it on the dev node,
    `git show HEAD:<path> | sha256sum`, and the pod's run line sets it, `PRED_SHA=<sha> python run.py ...`;
    `cluster/pod_setup.sh` says so at its top and in its log (friction 680)."""
    repo, rel = str(EXPERIMENTS.parent), str(path.resolve().relative_to(EXPERIMENTS.parent.resolve()))
    want = os.environ.get("PRED_SHA") or os.environ.get("E120_PRED_SHA")
    if want or not (EXPERIMENTS.parent / ".git").exists():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if want != got:
            sys.exit(f"{rel} sha256 {got} must equal PRED_SHA ({want or 'unset, and no .git here to ask'})")
        return
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False)
    if git("diff", "--quiet", "HEAD", "--", rel).returncode or not git("ls-files", rel).stdout.strip():
        sys.exit(f"{rel} is not committed unchanged: commit it first")


def config_from_settings[C](cls: type[C], settings: dict[str, Any]) -> C:
    """`cls(**settings)` for a predictions.json "settings" block, with note keys allowed: a key ending `_source` (where a
    value came from, e.g. `lr_source`) is dropped, and any other key `cls` does not take exits naming it, before a GPU
    is held. A note key added after the dry run killed a pod at start-up through a bare `Config(**settings)` (friction
    819). For new run.py files; frozen scripts keep their own line."""
    import dataclasses
    import inspect

    taken = ({f.name for f in dataclasses.fields(cls) if f.init} if dataclasses.is_dataclass(cls)
             else set(inspect.signature(cls).parameters))
    kept = {k: v for k, v in settings.items() if not k.endswith("_source")}
    if unknown := sorted(set(kept) - taken):
        sys.exit(f"settings key(s) {unknown} are not fields of {cls.__name__}; a note key must end in _source")
    return cls(**kept)


_INPUTS: set[str] = set()  # absolute paths opened for reading, while record_inputs is active
_RECORDING = False


def _write_inputs() -> None:
    """Write the sorted input list to $INPUT_MANIFEST (see `record_inputs`); called at exit and by `hard_exit`."""
    out = os.environ.get("INPUT_MANIFEST")
    if not (_RECORDING and out):
        return
    repo = str(EXPERIMENTS.parent)
    files = sorted(os.path.relpath(p, repo) if p.startswith(repo + os.sep) else p for p in _INPUTS if os.path.isfile(p))
    Path(out).write_text("".join(f"{p}\n" for p in files))


def _cuda_on_cpu() -> None:
    """For a CPU dry run only (INPUT_MANIFEST_CPU=1, see `record_inputs`): a global TorchFunctionMode rewrites a "cuda"
    device argument of any torch call (factories, Tensor.to, so Module.to too) to "cpu" and makes Tensor.cuda() a no-op,
    and torch.cuda answers as one present device (is_available, device_count, synchronize, memory stats, empty_cache).
    Not covered: torch.Generator(device="cuda") and CUDA-only kernels (flash attention, bitsandbytes)."""
    import torch
    from torch.overrides import TorchFunctionMode

    def cpu(a: Any) -> Any:
        if isinstance(a, str) and a.startswith("cuda"):
            return "cpu"
        if isinstance(a, torch.device) and a.type == "cuda":
            return torch.device("cpu")
        return a

    class CudaOnCpu(TorchFunctionMode):
        def __torch_function__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
            if func is torch.Tensor.cuda:
                return args[0]
            kwargs = {k: cpu(v) if k == "device" else v for k, v in (kwargs or {}).items()}
            return func(*(cpu(a) for a in args), **kwargs)

    for name, stub in {"is_available": lambda: True, "device_count": lambda: 1, "current_device": lambda: 0,
                       "synchronize": lambda *_a, **_k: None, "empty_cache": lambda: None,
                       "reset_peak_memory_stats": lambda *_a, **_k: None, "max_memory_allocated": lambda *_a, **_k: 0,
                       "memory_allocated": lambda *_a, **_k: 0, "max_memory_reserved": lambda *_a, **_k: 0,
                       "is_current_stream_capturing": lambda: False,  # Adam.step asks (E234 dry run)
                       "get_device_name": lambda *_a, **_k: "cpu (INPUT_MANIFEST_CPU dry run)"}.items():
        setattr(torch.cuda, name, stub)
    CudaOnCpu().__enter__()  # for the rest of the process
    print("INPUT_MANIFEST_CPU: cuda is the CPU for this dry run; its numbers mean nothing", file=sys.stderr, flush=True)


def record_inputs() -> None:
    """When $INPUT_MANIFEST names a file, record every file the process opens for reading under the repo or
    ~/commute-data (an audit hook on "open") and write the sorted list there at exit, repo paths relative to the repo:
    the input list a pod brief quotes, from a short CPU dry run (friction 832, 842). Does nothing when unset. Runs at
    `commute.jobs` import; `python -m commute.jobs <script> [args]` also covers what a script opens before importing
    it. Code (.py) is left out, and files read by native code without Python's open (safetensors' mmap) are not seen.

    With INPUT_MANIFEST_CPU=1 as well, the dry run also works with no GPU (fleet/cpu-run): `_cuda_on_cpu` puts every
    tensor and module asked for on "cuda" on the CPU, so a script that hard-codes "cuda" passes its model load and
    goes on to open its later inputs (E217, E218 dry runs stopped at the load, friction 906). Its numbers mean nothing;
    only the manifest counts. Give the job memory for the model in its CPU dtype."""
    global _RECORDING
    if _RECORDING or not os.environ.get("INPUT_MANIFEST"):
        return
    if os.environ.get("INPUT_MANIFEST_CPU") == "1":
        _cuda_on_cpu()
    data = os.path.expanduser("~/commute-data")
    roots = tuple({str(EXPERIMENTS.parent) + os.sep, data + os.sep, os.path.realpath(data) + os.sep})
    writes = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & writes):
            return
        p = os.path.abspath(os.fsdecode(args[0]))
        if p.startswith(roots) and not p.endswith((".py", ".pyc")):  # code is synced with the tree
            _INPUTS.add(p)

    import atexit

    _RECORDING = True
    sys.addaudithook(hook)
    atexit.register(_write_inputs)


if __name__ == "__main__":  # python -m commute.jobs <script> [args]: run the script with record_inputs active
    import runpy

    importlib.import_module("commute.jobs")  # the module the script will import; its import installs the hook
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
else:
    record_inputs()
