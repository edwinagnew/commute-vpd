"""E80's small inputs on W&B (artifact qwen3-1.7b-e80-inputs, type inputs, ~440 MB): E76's A offsets and cut, and the W0 g statistics that fix the alive set
of the in-span state. A pod fetches them from W&B (30-45 MB/s; sync() from the dev node is ~10 MB/s, measured 2026-11-11). Files count only when they match the committed results/inputs_SHA256SUMS. Imports no torch:
  upload (cluster CPU job, cluster/qwen_d1_adversary_upload.sbatch):  python experiments/qwen_d1_adversary/upload.py upload
  fetch (any pod):  PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_d1_adversary/upload.py fetch DIR v0
Both need WANDB_API_KEY; never prints it.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

from commute import qwen_files
from commute.qwen_files import ENTITY, PROJECT

ARTIFACT, ARTIFACT_TYPE = "qwen3-1.7b-e80-inputs", "inputs"
VERSION = "v0"
COMMITTED = Path(__file__).resolve().parent / "results" / "inputs_SHA256SUMS"
LARGE = Path.home() / "commute-data" / "results-large"
SOURCES = {"A_offsets.safetensors": LARGE / "qwen_masked_teacher" / "A_offsets.safetensors",
           "cut.safetensors": LARGE / "qwen_masked_teacher" / "cut.safetensors",
           "W0_maths.safetensors": LARGE / "qwen_sft" / "nowarmup" / "gstats" / "W0_maths.safetensors",
           "W0_fineweb.safetensors": LARGE / "qwen_sft" / "nowarmup" / "gstats" / "W0_fineweb.safetensors"}


def committed() -> dict[str, str]:
    pairs = (line.split() for line in COMMITTED.read_text().splitlines() if line)
    return {name: h for h, name in pairs}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def verified(d: Path) -> bool:
    """Every file of the committed sums is in `d` and matches."""
    return all((d / n).is_file() and digest(d / n) == h for n, h in committed().items())


def upload() -> None:
    import wandb

    want = committed()
    assert set(want) == set(SOURCES), "the sums and the sources name different files"
    for name, path in SOURCES.items():
        assert digest(path) == want[name], f"{path} does not match {COMMITTED}"
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
    t0 = time.time()
    art = wandb.Artifact(ARTIFACT, type=ARTIFACT_TYPE, metadata={"sha256": want},
                         description="E80's inputs: E76's A offsets and cut, W0's g statistics (experiments/qwen_d1_adversary/upload.py)")
    for name, path in SOURCES.items():
        art.add_file(str(path), name=name, skip_cache=True, policy="immutable")
    logged = run.log_artifact(art).wait()
    print(f"[upload] {logged.qualified_name} in {(time.time() - t0) / 60:.1f} min; sha256 {want}", flush=True)
    run.finish()


def fetch(d: Path, version: str = VERSION) -> None:
    qwen_files.fetch(d, artifact=ARTIFACT, kind=ARTIFACT_TYPE, ok=verified, version=version)


if __name__ == "__main__":
    upload() if sys.argv[1] == "upload" else fetch(Path(sys.argv[2]).expanduser().resolve(), sys.argv[3])
