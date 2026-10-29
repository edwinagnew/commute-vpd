"""E40's no-warm-up δW of Qwen3-1.7B on W&B (artifact qwen3-1.7b-e40-nowarmup-dW, type delta): steps 1, 10, 100 in
bf16 and 100 in fp32, as remade on the cluster by experiments/qwen_maps/regen.py, with their dW/SHA256SUMS (written by
`sha256sum` in cluster/qwen_e40_regen.sbatch, committed as experiments/qwen_maps/results/regen_SHA256SUMS, and in the
artifact's metadata). Files count as the cluster's only when they match the committed sums. Imports no torch:
  upload (cluster CPU job, cluster/qwen_e40_upload.sbatch):  python -m commute.qwen_e40_files upload DIR
  fetch (any pod):  PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files fetch DIR v0
DIR holds dW/<files>, E40's layout, and is what E44's run.py takes as --e40. Both need WANDB_API_KEY.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from commute import qwen_files
from commute.qwen_files import ENTITY, PROJECT

ARTIFACT, ARTIFACT_TYPE, SUMS = "qwen3-1.7b-e40-nowarmup-dW", "delta", "SHA256SUMS"
FILES = ("step_1.safetensors", "step_10.safetensors", "step_100.safetensors", "step_100_fp32.safetensors")
RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "qwen_maps" / "results"  # the editable install's repo
COMMITTED, REGEN = RESULTS / "regen_SHA256SUMS", RESULTS / "regen.json"


def sums(f: Path) -> dict[str, str]:
    """A sha256sum file as file name -> sha256."""
    pairs = (line.split() for line in f.read_text().splitlines() if line)
    return {name.lstrip("*"): h for h, name in pairs}


def verified(d: Path) -> bool:
    """d/dW's SHA256SUMS equals the committed one, lists exactly FILES, and every file matches it (`sha256sum -c`)."""
    f = d / "dW" / SUMS
    if not (f.is_file() and COMMITTED.is_file()) or sums(f) != sums(COMMITTED) or set(sums(f)) != set(FILES):
        return False
    return subprocess.run(["sha256sum", "-c", "--quiet", SUMS], check=False, cwd=d / "dW").returncode == 0


def upload(d: Path) -> None:
    import wandb

    rec = json.loads(REGEN.read_text())
    assert rec.get("done") and rec.get("pass") and "stopped" not in rec, f"{REGEN}: the regen run did not pass"
    assert verified(d), f"{d}/dW does not match {COMMITTED}"
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
    t0 = time.time()
    art = wandb.Artifact(ARTIFACT, type=ARTIFACT_TYPE, metadata={"sha256": sums(COMMITTED), "kl": rec["kl"]},
                         description="E40 no-warm-up SFT δW remade on a cluster L40 (experiments/qwen_maps/regen.py)")
    for name in (*FILES, SUMS):
        art.add_file(str(d / "dW" / name), name=f"dW/{name}", skip_cache=True, policy="immutable")
    logged = run.log_artifact(art).wait()
    print(f"[upload] {logged.qualified_name} in {(time.time() - t0) / 60:.1f} min; sha256 {sums(COMMITTED)}", flush=True)
    run.finish()


def fetch(d: Path, version: str) -> None:
    qwen_files.fetch(d, artifact=ARTIFACT, kind=ARTIFACT_TYPE, ok=verified, version=version)


if __name__ == "__main__":
    d = Path(sys.argv[2]).expanduser().resolve()
    upload(d) if sys.argv[1] == "upload" else fetch(d, sys.argv[3])
