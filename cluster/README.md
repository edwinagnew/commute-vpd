# MATS cluster for `commute`

Scripts for running this project on the MATS Slurm cluster (one node `l40-worker`, 8× L40 46 GB) and for
using a cluster GPU as the Jupyter kernel for notebooks open in VS Code on the laptop. Rules come from
`../../guides/mats_cluster_handbook.md`; this file covers only what is specific to this project.

- Login (controller, for submitting jobs and light file operations only): `ssh e.agnew@185.141.218.207`
- Slurm: account `c11`, partition `compute`. Code: one git clone per agent, `/mnt/nw/home/e.agnew/commute-agents/<role>`
  (NFS, persistent). Submit every sbatch from the clone root: jobs source `$SLURM_SUBMIT_DIR/cluster/env.sh`, which
  sets `COMMUTE_REPO` to that clone. The big gitignored files are shared in `/mnt/nw/home/e.agnew/commute-data` and
  linked into each clone by `fleet/link-data`: venv `.venv` (Python 3.13, `uv sync --locked --no-default-groups --group dev --group gpu`: CUDA torch; the laptop default is the CPU build), `artifacts/` (VPD decomposition + target model),
  `cluster/logs/` and `experiments/*/results/large/`.
- Caches and temp files go to `/ephemeral/$USER` (`env.sh`). That includes `HF_HOME`, so the tokenizer and dataset
  caches can be re-downloaded and do not count as results.

| File | Where it runs | What it does |
| --- | --- | --- |
| `env.sh` | cluster | `source` it: caches/TMPDIR → `/ephemeral/$USER`, uv on PATH |
| `setup.sbatch` | GPU node | (re)builds `.venv` with the `gpu` torch group, checks torch/CUDA, runs `gpu-sanity.sh` |
| `gpu-sanity.sh` | GPU node | the GPU guide's sanity script (topology file moved off `/tmp`) |
| `jupyter.sbatch` | GPU node | Jupyter server on 1 GPU; writes `.jupyter-session` (node, port, token; mode 600, gitignored) |
| `jupyter.sh` | laptop | `start` / `connect` / `status` / `stop` the Jupyter job |
| `kernel_check.py` | laptop | smoke test: starts a kernel through the tunnel, runs `torch.cuda.get_device_name()` |
| `sync.sh` | laptop | retired 2026-10-26: code moves by git; big files stay in `commute-data` |

Slurm logs go to `cluster/logs/<job-name>-<jobid>.out` on the cluster, and every job appends a line to
`cluster/logs/jobs.log`. Note that `sacct` and `sacctmgr` do not work from the controller (slurmdbd refuses the
connection; checked 2026-10-14). Use `squeue` and `scontrol show job <id>` instead, and keep job IDs yourself.

## Limits (read live from `scontrol show assoc_mgr flags=qos` on 2026-10-14)

| QoS | Max walltime | GPUs | Notes |
| --- | --- | --- | --- |
| `debug` | **2 h** (120 min) | 1 per job | priority 1000. The default for `jupyter.sh start` |
| `normal` | **24 h** (1440 min) | 6 per user (no per-job cap set) | also 124 CPUs / 384 GB per user |

The handbook gives both 1 h and 2 h for debug, and both 24 h and 7 days for normal. The values above are
what the scheduler enforces, and jobs over the limit are rejected when you submit them.

`sbatch --test-only` is useless for `debug` jobs: it ignores the QoS priority. On 2026-10-19 (13:58, ~90 jobs
pending) it put a 20-min debug job's start 24 h away, while the two debug smoke jobs submitted that hour (22366,
22368) started 6 min and 3 min after submission. For `normal` jobs it is the only estimate there is (the same
query put a 2-h normal job 16 days out). Policy (2026-10-20): only runs of ≤ 30 min come here, on
`debug`, with at most 10 min of waiting for a start; longer runs go straight to RunPod (`.claude/agents/pod-runner.md`).

## Notebook in VS Code on a cluster GPU

1. Start the server (laptop, repo root): `cluster/jupyter.sh start`.
   For longer than 2 h: `cluster/jupyter.sh start normal 06:00:00`.
2. Open the tunnel: `cluster/jupyter.sh connect`. It waits until the job is running and the server is answering,
   prints `http://localhost:<port>/?token=…` and then holds the tunnel open. Keep that terminal open.
3. In VS Code, open the notebook, then **Select Kernel → Select Another Kernel… → Existing Jupyter Server →
   Enter the URL of the running Jupyter Server**, paste the URL, and pick **Python 3 (ipykernel)**. The kernel
   is the project venv on the cluster.
4. When you are done: `cluster/jupyter.sh stop`. Closing VS Code or the tunnel does **not** free the GPU.

The notebook file stays on the laptop, and VS Code saves it there. The kernel runs on the node with its working
directory under the cluster clone (`COMMUTE_CLUSTER_REPO`, default `commute-agents/lead`), so code reads and
writes that clone's tree (`artifacts/`, `experiments/*/results/`). Push the code first and `git pull` in the clone;
commit results there to bring them back. `src/commute` is an editable install, so after a pull the next kernel
restart picks up the new code. If `pyproject.toml` or `uv.lock` changed, rebuild the venv with
`ssh e.agnew@185.141.218.207 'cd /mnt/nw/home/e.agnew/commute-agents/lead && sbatch cluster/setup.sbatch'`.

The kernel's working directory follows the notebook's relative path (for example, `notebooks/` on the cluster), so
in cells use `commute.env.REPO_ROOT` / `env.VPD_DIR` or absolute `/mnt/nw/...` paths rather than relative ones.

To check the connection without VS Code: `.venv/bin/python cluster/kernel_check.py '<url>'`. That check (start a
kernel, run `torch.cuda.get_device_name()`, get `NVIDIA L40`) passed through the tunnel on 2026-10-14. The VS Code
"Existing Jupyter Server" dialog itself was not exercised, but it uses the same URL and REST/websocket API.

### How the tunnel gets in (tested 2026-10-14)

From the controller, every port on `l40-worker` (10.0.4.14) except slurmd's is firewalled, including 22 and
the Jupyter port. So a plain `ssh -L …:10.0.4.14:<port>` or a `ProxyJump` times out, and there is no
`pam_slurm_adopt`. `connect` instead goes in through the job itself. Its `ProxyCommand` runs
`srun --jobid=<id> --overlap --gres=none --unbuffered -c1 --mem=100M nc 127.0.0.1 22` on the controller, which
relays stdio to the node's own sshd. Your usual key works there because the home directory is on the shared NFS.
The tunnel then forwards to the Jupyter server, which listens on the node's `127.0.0.1` only. Three flags matter:
`--gres=none` (otherwise the step waits for the GPU the server holds), `--unbuffered` (otherwise slurmstepd
line-buffers the binary SSH stream and the handshake stalls), and a small `--mem` (otherwise the step asks for
all the job's memory and waits). The tunnel dies with the job. The host key is pinned under the alias
`mats-l40-worker` in `~/.ssh/known_hosts`.

**VS Code Remote-SSH to the node works the same way, but don't use it for GPU work.** It needs a `~/.ssh/config`
entry with that `ProxyCommand` and the current job ID. An SSH session on the node is outside the Slurm job: it has
no `CUDA_VISIBLE_DEVICES`, and `nvidia-smi` shows all 8 L40s, so code started from it lands on GPU 0, which is
probably someone else's. Its processes also outlive the job. Use the Jupyter kernel (it inherits the job's GPU)
or `sbatch`.

## Caveats

- **Walltime kills the kernel.** On `debug`, the job and the kernel with all its state are killed at 2 h with no
  warning in the notebook. `connect` prints the end time, and `status` shows the time left. For longer
  work, use `normal` with a realistic `--time` (≤ 24 h).
- **An idle interactive job still holds a GPU.** The node has 8 L40s shared by every fellow, and the queue often
  has people waiting. Cancel the job (`cluster/jupyter.sh stop` or `scancel <id>`) as soon as you stop working.
  Do not leave a `normal` session open overnight "just in case".
- **Checkpoints and results go to `/mnt/nw`** (that is, the repo tree, e.g. `experiments/<name>/results/large/`).
  Never use `/tmp`. `/ephemeral` is a disposable cache on the node.
- Long training runs should be resumable `sbatch` jobs (copy the header of `setup.sbatch`), not cells in a
  notebook kernel that dies at the walltime.
- **GPU isolation is by `CUDA_VISIBLE_DEVICES` only.** There is no device cgroup, so inside a job `nvidia-smi`
  lists all 8 GPUs. Never overwrite `CUDA_VISIBLE_DEVICES` in a notebook or script, and don't pick devices by
  index from `nvidia-smi`.
- The Jupyter server listens on the node's localhost, but any user on the node could reach it there, so the
  random token in `.jupyter-session` (mode 600) is the only protection. Don't paste the URL anywhere. The server
  log in `cluster/logs/` also contains it.
- `gpu-sanity.sh` gives two harmless warnings here. The repo directory is not itself a mount point (`/mnt/nw`
  is), and the Cloudflare download test reads about 0 Mbps (it seems to be blocked), while pip/uv downloads are
  fast.
