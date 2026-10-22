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

## Next session

`GIT_PAT` is only visible to a session started after it was exported. Hand the new session this file and
`.claude/agents/pod-runner.md`, push first, then run one pod-runner with the same brief as setup_timing (start,
clone, `sync()` ×2, `bash cluster/pod_setup.sh`, `df -h /root/work /root`, terminate; $1). Expected: first `sync()`
well under a minute, fetch near its 2026-10-19 140 s, `uv sync` 1–2 min, whole set-up under ~5 min.

## Not run

- The clone, `workdir = "/root/work"` and `volume-gb = 0`: untested on a pod; whether plugin 0.3.9 accepts
  `volume-gb = 0` is unchecked.
- `pod_setup.sh` has not reached its end.
- Docker image: not built; only worth it if the next timing still shows installs dominating.
