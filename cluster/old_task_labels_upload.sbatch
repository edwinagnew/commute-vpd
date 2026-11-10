#!/bin/bash
# E142 (P1.7): upload old_task_labels' inputs (the 24 items' saved fine-tunes, the token files `commute.old_tasks` registers, and the
# cached Pile rows `make_sets` and E53's token maps read) to the W&B artifact e142-old-task-inputs at their repo-relative paths, so a
# RunPod pod can fetch them (the pod uplink is ~80 KB/s). CPU job. Submit from the clone root with WANDB_API_KEY set:
#   sbatch --comment='E142 stage: input upload for RunPod move' cluster/old_task_labels_upload.sbatch
#SBATCH --job-name=commute-old_task_labels_upload
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
mkdir -p "$HOME/commute-data/scratch/definition-lead/e142-runpod"
cat > "$HOME/commute-data/scratch/definition-lead/e142-runpod/files.txt" <<'LIST'
experiments/induction_cue/results/large/copy_s0.pt
experiments/induction_cue/results/large/copy_s1.pt
experiments/induction_cue/results/large/lookup_s0.pt
experiments/induction_cue/results/large/lookup_s1.pt
experiments/induction_cue/results/large/windows.pt
experiments/open_pair/results/large/amend1/copy_s0.pt
experiments/open_pair/results/large/amend1/copy_s1.pt
experiments/open_pair/results/large/amend1/windows_copy_trained.pt
experiments/open_pair/results/large/amend1/windows_copy_new.pt
experiments/open_pair/results/large/cipher_s0.pt
experiments/open_pair/results/large/cipher_s1.pt
experiments/open_pair/results/large/windows_cipher_trained.pt
experiments/open_pair/results/large/windows_cipher_new.pt
experiments/open_pair/results/large/amend1/rank1_s1.pt
experiments/open_pair/results/large/amend1/rank1_s0.pt
experiments/open_pair/results/large/amend1/windows_rank1_trained.pt
experiments/open_pair/results/large/amend1/windows_rank1_new.pt
experiments/kv_task/results/large/rand1/dW_ft_step10.pt
experiments/kv_task/results/large/windows_rand1.pt
experiments/performed_anchor/results/large/dW_cipher_s0_step300.pt
experiments/performed_anchor/results/large/dW_cipher_s1_step300.pt
experiments/performed_anchor/results/large/dW_broad_cipher_s0_step300.pt
experiments/performed_anchor/results/large/dW_broad_cipher_s1_step300.pt
experiments/performed_anchor/results/large/dW_lowercase_s0_step300.pt
experiments/performed_anchor/results/large/dW_lowercase_s1_step300.pt
experiments/case_label/results/large/dW_real_s0.pt
experiments/case_label/results/large/dW_real_s1.pt
experiments/case_label/results/large/dW_twin_s0.pt
experiments/case_label/results/large/dW_twin_s1.pt
experiments/case_label/results/large/task.pt
experiments/fresh_reuse_pair/results/large/dW_real_s0.pt
experiments/fresh_reuse_pair/results/large/dW_real_s1.pt
experiments/fresh_reuse_pair/results/large/dW_twin_s0.pt
experiments/fresh_reuse_pair/results/large/dW_twin_s1.pt
experiments/fresh_reuse_pair/results/large/task.pt
experiments/recruitment_test/results/large/states/emoticon_s0.pt
experiments/recruitment_test/results/large/states/emoticon_s1.pt
experiments/transport_test/results/large/windows.pt
artifacts/sets/train_0_32768.npy
artifacts/sets/train_32768_8192.npy
artifacts/sets/val_0_2000.npy
artifacts/sets/train_0_4096.npy
LIST
.venv/bin/python - <<'PY'
import hashlib, os
from pathlib import Path
import wandb
from commute.qwen_files import ENTITY, PROJECT
files = Path(os.environ["HOME"], "commute-data/scratch/definition-lead/e142-runpod/files.txt").read_text().split()
run = wandb.init(entity=ENTITY, project=PROJECT, job_type="upload", dir=Path.home() / "commute-data/wandb")
art = wandb.Artifact("e142-old-task-inputs", type="inputs", description="E142 old_task_labels inputs at repo-relative paths")
for f in files:
    real = Path(f).resolve(strict=True)
    h = hashlib.sha256()
    with open(real, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    print(f"{h.hexdigest()}  {f}", flush=True)
    art.add_file(str(real), name=f, skip_cache=True)
print("[upload]", run.log_artifact(art).wait().qualified_name, flush=True)
run.finish()
PY
