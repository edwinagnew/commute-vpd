# Faster pod set-up (2026-10-22)

A fresh RunPod pod used to take five to fifteen minutes before an experiment could start. A timing pod found the
main cause: the plugin's default 20 GB volume put the code, the venv and the caches on RunPod's network storage.
The pod now keeps everything on its own disk, clones the code from GitHub, and sets itself up with one script; none
of this has been run together yet.

## What the timing pod measured (setup_timing, 4090, $0.24)

| Step | Time |
|---|---|
| `start()` returns | 34 s |
| first cell | 43 s |
| first `sync()`, whole tree (~19 MB compressed) | 3 min 38 s |
| second `sync()`, nothing changed | ≤ 7 s |
| uv 0.12.9 + Python 3.13 | 4 s |
| VPD fetch, 3.2 GB, alongside `uv sync` | 10 min (140 s on 2026-10-19) |
| `uv sync` | failed: disk quota on `/workspace` |

`/workspace` was a 20 GB MooseFS mount (`mfs#eur-no-1.runpod.net`), the plugin's default `volume-gb = 20`. The
uv cache (14 GB) and `.venv` (4 GB) filled its quota; the full volume corrupted the plugin's lease file there, so
`terminate()` was refused and the runner deleted the pod with `runpodctl remove pod`.

## Changes

- `remote-kernels.toml`: `volume-gb = 0`, `workdir = "/root/work"` (container disk, 50 GB); `UV_CACHE_DIR` back to
  `/root/uv`; `GIT_PAT` added to `inherit-env`.
- `cluster/pod_setup.sh`: pinned uv, Python 3.13, `uv sync` with the VPD fetch in parallel, a timestamp per step.
- `src/commute/vpd_files.py`: the W&B names and `fetch()`, torch-free so the fetch runs before the venv exists;
  `commute.vpd` re-exports them.
- `.claude/agents/pod-runner.md`: clone `edwinagnew/commute-vpd` with `GIT_PAT` before the first `sync()`, then
  `pod_setup.sh`; the job's last log line says it finished and where its results are.

## Result (second timing pod, 4090, $0.05)

Set-up from the `start()` call to a working CUDA venv took about 3 min, against over 15 min (never finished) on the
previous pod. The plugin accepted `volume-gb = 0` and `workdir`; the token stayed out of `.git/config`.

| Step | Now | Previous pod |
|---|---|---|
| first cell after `start()` | 76 s | 43 s |
| clone with `GIT_PAT` | 2 s | n/a |
| first `sync()` | ~19 s | 3 min 38 s |
| `uv sync` | 34 s | failed: disk quota |
| VPD fetch, 3.2 GB, alongside | done by +46 s | 10 min |
| `pod_setup.sh` total | 51 s | did not finish |

The slowest step left is pod creation (76 s), which we don't control. A Docker image would save at most the ~50 s of
`pod_setup.sh`; not worth building now. `sync-include` in the toml still sends nothing; inputs go with
`sync(include=[<directory>])`.

Web search (2026-10-22) found nothing RunPod-specific beyond "hosts cache images; big images pull slowly". The pod
still uses the plugin's default `runpod/pytorch:2.1.0` CUDA 11.8 devel image, though our venv brings its own CUDA. A
leaner image might shave part of the 76 s, but an uncached one could be slower and must still run sshd for the plugin:
one timing pod would tell, worth it only if pods start many times a day.

## Not run

- Docker image: not built (see above).
- Reusing a live pod with `pod_setup.sh` (should report the VPD files `present`): untested.
