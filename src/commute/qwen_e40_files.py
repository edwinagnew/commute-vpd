"""E40's no-warm-up δW of Qwen3-1.7B on W&B (artifact qwen3-1.7b-e40-nowarmup-dW, type delta): steps 1, 10, 100 in
bf16 and 100 in fp32, as remade on the cluster by experiments/qwen_maps/regen.py, with their dW/SHA256SUMS (written by
`sha256sum` in cluster/qwen_e40_regen.sbatch; also in the artifact's metadata). Imports no torch, as `qwen_files`:
  upload (cluster CPU job, cluster/qwen_e40_upload.sbatch):  python -m commute.qwen_e40_files upload DIR
  fetch (any pod):  PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files fetch DIR
DIR holds dW/<files>, E40's layout, and is what E44's run.py takes as --e40. Both need WANDB_API_KEY.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from commute import qwen_files
from commute.qwen_files import ENTITY, PROJECT

ARTIFACT, ARTIFACT_TYPE, SUMS = "qwen3-1.7b-e40-nowarmup-dW", "delta", "SHA256SUMS"
FILES = ("step_1.safetensors", "step_10.safetensors", "step_100.safetensors", "step_100_fp32.safetensors")


def sums(d: Path) -> dict[str, str]:
    """d/dW/SHA256SUMS as file name -> sha256."""
    pairs = (line.split() for line in (d / "dW" / SUMS).read_text().splitlines() if line)
    return {name.lstrip("*"): h for h, name in pairs}


def verified(d: Path) -> bool:
    """Every file of FILES under d/dW, each matching SHA256SUMS (`sha256sum -c`, ~1 min for the 14 GB)."""
    if not (d / "dW" / SUMS).is_file() or set(sums(d)) != set(FILES):
        return False
    return subprocess.run(["sha256sum", "-c", "--quiet", SUMS], check=False, cwd=d / "dW").returncode == 0


def upload(d: Path) -> None:
    import wandb

    assert verified(d), f"{d}/dW does not match its {SUMS}"
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
    t0 = time.time()
    art = wandb.Artifact(ARTIFACT, type=ARTIFACT_TYPE, metadata={"sha256": sums(d)},
                         description="E40 no-warm-up SFT δW remade on a cluster L40 (experiments/qwen_maps/regen.py)")
    for name in (*FILES, SUMS):
        art.add_file(str(d / "dW" / name), name=f"dW/{name}", skip_cache=True, policy="immutable")
    logged = run.log_artifact(art).wait()
    print(f"[upload] {logged.qualified_name} in {(time.time() - t0) / 60:.1f} min; sha256 {sums(d)}", flush=True)
    run.finish()


def fetch(d: Path) -> None:
    qwen_files.fetch(d, artifact=ARTIFACT, kind=ARTIFACT_TYPE, ok=verified)


if __name__ == "__main__":
    {"upload": upload, "fetch": fetch}[sys.argv[1]](Path(sys.argv[2]).expanduser().resolve())
