#!/bin/bash
# P2.69 eapig_overlap (E___): builds the second venv, Python 3.10 pinned to FinetuneCircuits fdaeeaac63d5's
# requirements.txt (experiments/eapig_overlap/requirements-fc.txt: every pin but pygraphviz, which needs graphviz headers
# the cluster lacks). CPU only, no GPU. The venv lives on NFS and is reused by every run; rebuild only when the pins change.
# Usage (from the clone root): sbatch --comment='E___ venv' cluster/eapig_overlap_venv.sbatch
#SBATCH --job-name=commute-eapig-venv
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
VENV=$HOME/commute-data/venvs/fc310-fdaeeaa   # = predictions.json settings "venv"
uv python install 3.10
rm -rf "$VENV"
uv venv --python 3.10 "$VENV"
uv pip install --python "$VENV/bin/python" -r experiments/eapig_overlap/requirements-fc.txt
"$VENV/bin/python" -I -c "import sys, torch, transformer_lens, transformers; from importlib.metadata import version as v; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformer_lens', v('transformer_lens'), 'transformers', transformers.__version__, 'cuda build', torch.version.cuda)"
echo "=== VENV DONE: $VENV ==="
