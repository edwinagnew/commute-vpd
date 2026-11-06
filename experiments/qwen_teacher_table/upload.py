"""E111's inputs on W&B, for a pod (the dev node's uplink is ~80 KB/s). Imports no torch. Each δW E93 scored is its own
artifact, `qwen3-1.7b-e111-dW-<state>` (type delta), one file `<state>.safetensors` exactly as saved (fp32, u2 bf16:
the run reproduces E93's stored reads, which these files gave); it counts only when its sha256 is the one in E93's
results/<state>/state.json. The small inputs, `qwen3-1.7b-e111-inputs` (type inputs): E45's anchor-1 items and the two
TOFU files `qwen_tofu.held_rows` reads; their sums are committed in results/inputs_SHA256SUMS. E76's cut and W0's g
statistics come from E80's artifact (experiments/qwen_d1_adversary/upload.py), the decomposition from
`commute.qwen_files`. From the pod, `offsets` uploads the trained offsets (results never pass through the dev node).

  upload (cluster CPU job, cluster/qwen_teacher_table_upload.sbatch):  python experiments/qwen_teacher_table/upload.py upload inputs R_s70 ...
  fetch (any pod):  PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_teacher_table/upload.py fetch ROOT R_s70 ...
  offsets (the pod, after a stage):  .venv/bin/python experiments/qwen_teacher_table/upload.py offsets ROOT/large
Each needs WANDB_API_KEY; never prints it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from commute import qwen_files
from commute.qwen_files import ENTITY, PROJECT

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
E93_RESULTS = EXPERIMENTS / "qwen_d1_table" / "results"
COMMITTED = HERE / "results" / "inputs_SHA256SUMS"
INPUTS, VERSION = "qwen3-1.7b-e111-inputs", "v0"
TOFU_DIR = EXPERIMENTS.parent / "artifacts" / "tofu"
INPUT_SOURCES = {"a1_rows.json": EXPERIMENTS / "qwen_updates" / "results" / "large" / "P" / "a1_rows.json",
                 "retain_perturbed.json": TOFU_DIR / "retain_perturbed.json",
                 "forget10_perturbed.json": TOFU_DIR / "forget10_perturbed.json"}


def e93(state: str) -> dict[str, str]:
    rec = json.loads((E93_RESULTS / state / "state.json").read_text())
    return {"file": rec["file"], "sha256": rec["sha256"]}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def committed() -> dict[str, str]:
    return {name: h for h, name in (line.split() for line in COMMITTED.read_text().splitlines() if line)}


def inputs_ok(d: Path) -> bool:
    return all((d / n).is_file() and digest(d / n) == h for n, h in committed().items())


def state_ok(state: str) -> Callable[[Path], bool]:
    return lambda d: (d / f"{state}.safetensors").is_file() and digest(d / f"{state}.safetensors") == e93(state)["sha256"]


def push(name: str, kind: str, files: dict[str, Path], meta: dict[str, object]) -> None:
    import wandb

    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
    t0 = time.time()
    art = wandb.Artifact(name, type=kind, metadata=meta, description="E111 (experiments/qwen_teacher_table/upload.py)")
    for n, p in files.items():
        art.add_file(str(p), name=n, skip_cache=True, policy="immutable")
    logged = run.log_artifact(art).wait()
    size = sum(p.stat().st_size for p in files.values())
    print(f"[upload] {logged.qualified_name}: {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min; {json.dumps(meta)}", flush=True)
    run.finish()


def upload(what: list[str]) -> None:
    for w in what:
        if w == "inputs":
            want = committed()
            assert set(want) == set(INPUT_SOURCES), "the sums and the sources name different files"
            for n, p in INPUT_SOURCES.items():
                assert digest(p) == want[n], f"{p} does not match {COMMITTED}"
            push(INPUTS, "inputs", INPUT_SOURCES, {"sha256": want})
            continue
        rec = e93(w)
        src = EXPERIMENTS / rec["file"]
        got = digest(src)
        assert got == rec["sha256"], f"{src}: sha256 {got}, E93 read {rec['sha256']}"
        push(f"qwen3-1.7b-e111-dW-{w}", "delta", {f"{w}.safetensors": src}, {"sha256": got, "e93_file": rec["file"]})


def fetch(root: Path, states: list[str], log: Any = print) -> None:
    """ROOT/inputs and ROOT/dW/<state>.safetensors; each is kept when already complete."""
    qwen_files.fetch(root / "inputs", log, artifact=INPUTS, kind="inputs", ok=inputs_ok, version=VERSION)
    for s in states:
        qwen_files.fetch(root / "dW", log, artifact=f"qwen3-1.7b-e111-dW-{s}", kind="delta", ok=state_ok(s), version=VERSION)


def offsets(d: Path) -> None:
    files = {p.name: p for p in sorted(d.glob("*_offsets.safetensors"))}
    assert files, f"no offsets in {d}"
    push("qwen3-1.7b-e111-offsets", "offsets", files, {"sha256": {n: digest(p) for n, p in files.items()}})


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "upload":
        upload(rest)
    elif cmd == "fetch":
        fetch(Path(rest[0]).expanduser().resolve(), rest[1:])
    else:
        assert cmd == "offsets", cmd
        offsets(Path(rest[0]).expanduser().resolve())
