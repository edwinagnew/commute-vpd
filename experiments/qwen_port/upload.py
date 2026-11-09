"""Upload the converted Qwen3-1.7B decomposition (the directory `convert.py` writes: fp32 safetensors, `index.json`,
`launch_config.yaml`) as one W&B artifact, so a RunPod pod can fetch it with `commute.qwen_files.fetch` (route in
reports/analyses/2026-10-27_analysis_qwen-compute.md section 2). Run once, from the cluster: the dev node's uplink is
~80 KB/s.

The files are added with `policy="immutable"` and `skip_cache=True`: W&B hashes them in place instead of copying 50 GB
into its staging and cache directories, so the directory must not change while the upload runs. Needs WANDB_API_KEY
in the environment; never prints it. The project is `edwinagnew-mats/commute` (the key's default entity), created on
first use.

The artifact name comes from the decomposition id in the directory's index.json (`qwen_vpd.decomp_id`) through
`qwen_files.DECOMP_ARTIFACT`; an id not in that mapping stops the upload, so one decomposition can never overwrite
another's artifact.

Usage (repo root; CPU Slurm job, the directory is optional and defaults to the old decomposition's):
  sbatch cluster/qwen_upload.sbatch ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch
or directly (CPU job):
  ~/commute-agents/fleet/cpu-run -c 8 --mem 16G -t 300 -- \
    .venv/bin/python experiments/qwen_port/upload.py ~/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch
"""

import sys
import time
from pathlib import Path

import wandb

from commute.qwen_files import ARTIFACT_TYPE, DECOMP_ARTIFACT, ENTITY, PROJECT
from commute.qwen_vpd import decomp_id


def main(src: Path) -> None:
    ident = decomp_id(src)
    assert ident in DECOMP_ARTIFACT, f"{src}: decomposition {ident} is not in qwen_files.DECOMP_ARTIFACT"
    name = DECOMP_ARTIFACT[ident]
    total = sum(p.stat().st_size for p in src.iterdir() if p.is_file())
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
    print(f"[upload] {src} -> {run.entity}/{PROJECT}/{name}: {total / 1e9:.2f} GB", flush=True)
    t0 = time.time()
    art = wandb.Artifact(name, type=ARTIFACT_TYPE, description="experiments/qwen_port/convert.py output")
    art.add_dir(str(src), skip_cache=True, policy="immutable")
    print(f"[upload] hashed in {time.time() - t0:.0f} s", flush=True)
    logged = run.log_artifact(art).wait()
    dt = time.time() - t0
    print(f"[upload] {logged.qualified_name}: {total / 1e9:.2f} GB in {dt / 60:.1f} min, {total / 1e6 / dt:.0f} MB/s")
    run.finish()


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser().resolve())
