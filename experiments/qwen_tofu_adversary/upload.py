"""E81's TOFU update on W&B (artifact qwen3-1.7b-e81-tofu-dW, type delta, 5.64 GB, one file): the fp32 δW at step 1250, ‖δW‖ 17.998. The uplink of the dev node is ~80 KB/s, so a pod fetches
it. The file counts only when it matches the committed results/inputs_SHA256SUMS. Imports no torch:
  upload (cluster CPU job, cluster/qwen_tofu_adversary_upload.sbatch):  python experiments/qwen_tofu_adversary/upload.py upload
  fetch (any pod):  PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_tofu_adversary/upload.py fetch DIR v0
Both need WANDB_API_KEY; never prints it.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from commute import qwen_files
from commute.qwen_e40_files import sums
from commute.qwen_files import ENTITY, PROJECT

ARTIFACT, ARTIFACT_TYPE, VERSION = "qwen3-1.7b-e81-tofu-dW", "delta", "v0"
NAME = "step_1250.safetensors"
COMMITTED = Path(__file__).resolve().parent / "results" / "inputs_SHA256SUMS"
SOURCE = Path.home() / "commute-data" / "results-large" / "qwen_tofu" / "dW" / "a2_2e-5" / NAME


def verified(d: Path) -> bool:
    """The file is in `d` and matches the committed sum."""
    return (d / NAME).is_file() and subprocess.run(["sha256sum", "-c", "--quiet", str(COMMITTED)], check=False, cwd=d).returncode == 0


def upload() -> None:
    import wandb

    assert verified(SOURCE.parent), f"{SOURCE} does not match {COMMITTED}"
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
    t0 = time.time()
    art = wandb.Artifact(ARTIFACT, type=ARTIFACT_TYPE, metadata={"sha256": sums(COMMITTED)},
                         description="E81's TOFU update, fp32 δW at step 1250 (experiments/qwen_tofu_adversary/upload.py)")
    art.add_file(str(SOURCE), name=NAME, skip_cache=True, policy="immutable")
    logged = run.log_artifact(art).wait()
    print(f"[upload] {logged.qualified_name} in {(time.time() - t0) / 60:.1f} min; sha256 {sums(COMMITTED)}", flush=True)
    run.finish()


def fetch(d: Path, version: str = VERSION) -> None:
    qwen_files.fetch(d, artifact=ARTIFACT, kind=ARTIFACT_TYPE, ok=verified, version=version)


if __name__ == "__main__":
    upload() if sys.argv[1] == "upload" else fetch(Path(sys.argv[2]).expanduser().resolve(), sys.argv[3])
