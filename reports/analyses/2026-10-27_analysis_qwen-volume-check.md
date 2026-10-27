# Qwen network volume check (2026-10-27)

We started two RunPod pods in US-MO-1 that both mounted the new 100 GB network volume `ydavm7ct4x`, using
remote-kernels 0.3.9 with `workdir = "/root/work"`. We wrote and read 2 GiB files on the volume from both pods, then
terminated the first pod and read the data back from the second. The volume works for the Qwen line: both pods ran
at the same time without a lease conflict, the data outlived the pod that wrote it, and large files move at about
160-310 MB/s per stream.

## Set-up

- Pod A: `01M5YXJKRASDXA3GCPAC7GKPGQ` (RunPod `xw6hunftqdwqq0`), L40S, $1.09/h. From `start()` to a working kernel:
  under 1 min.
- Pod B: `01M5YXRKTXAT3M6CQ6NXFBT2E0` (RunPod `ef6sq1p6b3xrxw`), H100 80GB HBM3, $3.99/h. The L40S request failed
  with "RunPod API error (400): There are no longer any instances available with the requested specifications", so
  pod B ran on an H100 as the brief allowed.
- RunPod shows both pods under the config's existing name `commute-recruitment_test`, because `remote-kernels.toml`
  was not edited.

## Results

| Check | Result |
| --- | --- |
| Mount | `/workspace`, `mfs#us-mo-1.runpod.net:9421`, type `fuse` (MooseFS) |
| Size shown by `df` | 448T total, 114T free: the whole shared MooseFS cluster, not the 100 GB quota (the quota does not appear in `df`) |
| Kernel cwd | `/root/work`, on the 100 GB container overlay, not under `/workspace` |
| Plugin files on the volume | only `/workspace/.remote-kernels/machines/<machine-id>/` (`lease.json`, `lease.lock`, watchdog, kernel-output); sync() was not called |
| Local source (urandom to `/root`) | 405 MB/s |
| Write to volume, `oflag=direct`, a.bin | 181 MB/s (11.8 s for 2 GiB) |
| Write to volume, `oflag=direct`, b.bin | 160 MB/s (13.4 s) |
| Write to volume, buffered + fsync | 179 MB/s (12.0 s) |
| Read on A, `iflag=direct`, a.bin | 210 MB/s (10.2 s) |
| Read on A, buffered, b.bin | 135 MB/s (15.9 s) |
| 4 parallel readers on A (a, a, b, b; direct) | 63-68 MB/s each; 8 GiB in 34.2 s, about 240 MiB/s in total |
| Cold read on B, `iflag=direct`, a.bin | 311 MB/s (6.9 s) |
| sha256 a.bin (`e08cab82...b6af57`) | checked on A, then OK on B with both pods running, and OK on B after A was terminated |
| `sha256sum` of 2 GiB on the volume | 17.2 s (A), 13.4 s (B) |

`drop_caches` is not allowed in the container (read-only `/proc/sys`). The reads used `iflag=direct`, and pod B had
never touched the file, so B's 311 MB/s read did not come from B's page cache.

## Lease and fencing

Each machine keeps its own lease in its own directory (`machines/<id>/lease.json`, generation 1, state `active`,
same owner session). With both pods running, `status()` showed both as Running, both kernels executed, and neither
logged a lease or fence error. The 2026-10-19 failure, where the second pod read the first pod's lease and fenced
itself, did not happen.

After A was terminated, B's kernel kept working. A's state directory stays on the volume with `lease.json` state
`finalizing`, action `terminate`. Both machines' directories were left in `.remote-kernels` (8.8 MB in total), so
these stale per-machine directories will build up over time.

## End state

`/workspace/voltest` was removed. The volume holds only `.remote-kernels/` (`du -sh /workspace` = 8.8M). Both pods
were terminated, and `status()` reports no machine records. Session cost from `status()`: $0.21.

## Not run

- Two pods reading the volume at the same moment (concurrent throughput across pods).
- Throughput with many small files (uv venv, `import torch` from the volume). The 2026-10-19 tests found MooseFS
  slow for these.
- A check that the 100 GB quota is enforced (we did not fill the volume).
- Two L40S pods: pod B ran on an H100 because L40S stock ran out.
