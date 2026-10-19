# Decision: queue cap of 30 min, and whether to keep one RunPod pod running all day

*2026-10-19. Source: the user's request ("don't wait more than 30 minutes on the cluster before resorting to RunPod";
"consider keeping a single RunPod running throughout the day, maybe paused"). Evidence from the cluster scheduler,
this repo's pod notebooks and reports, and RunPod's documentation.*

## 1. What changed today

`.claude/agents/pod-runner.md`, `.claude/skills/pod-runner/SKILL.md`, `.claude/CLAUDE.md`, `cluster/README.md`,
`decisions.md`. The rule:

- **Cluster first; RunPod once the cluster has not started the job within 30 min.** Short runs wait less: the run's
  own expected time, floored at 10 min (a 3-min smoke waits 10 min, a 20-min check 20 min).
- **`debug` jobs (≤ 2 h, 1 GPU) are submitted and waited on.** `sbatch --test-only` is not consulted for them.
- **`normal` jobs (> 2 h) use `--test-only`**, and an estimated start more than 30 min out counts as blocked.
- When the cap passes, the pod-runner cancels the pending cluster job *before* starting the pod, and reports both IDs.
- The brief's Platform line now pre-authorises RunPod at 30 min by default, so the agent does not bounce back to a lead
  that cannot answer mid-run. The $10 RunPod cap still applies: the rule now routes long scored runs to RunPod as
  well, and a 6-h 4090 run is ~$5–6.
- The key-value plan's "RunPod if the queue holds it > 45 min" (`reports/2026-10-19_plan_key-value-task.md`) is
  superseded by this; the plan file is left as written.

### Why `--test-only` had to go for debug jobs

Read at 13:58 UTC today, with about 90 jobs pending on the one 8×L40 node:

| Query | `--test-only` estimated start | What happened |
| --- | --- | --- |
| debug, 1 GPU, 20 min | 2026-10-20 14:33 (24 h away) | the two debug smoke jobs submitted that hour, 22366 and 22368, started **6 min** and **3 min** after submission |
| normal, 1 GPU, 2 h | 2026-11-04 (16 days away) | no normal job submitted today to compare |

The estimator ignores the `debug` QoS priority (1000 against `normal`'s 0, `scontrol show assoc_mgr flags=qos`).
Under the old agent text ("check `--test-only`; over ~1 h, report") every smoke run today would have been sent to
RunPod, which is the gap the user noticed. (The 2026-10-14 chain of five debug jobs, 20266–20289, ran within an
hour, which is consistent with this but not evidence: `jobs.log` records start times, not submission times.)

What the 30-min cap is blind to: a debug job that waits 25 min and then runs for 3 min. That is still faster than the
~10 min a fresh pod costs end to end only when the lead is doing something else meanwhile; the shorter cap for short
runs (expected runtime, floor 10 min) is there for that case.

## 2. Keep one pod running all day?

### Evidence on what a fresh pod costs

From the saved pod notebooks (`remote-kernels/*.ipynb`) and reports:

| Step | Time | Where it is measured |
| --- | --- | --- |
| `uv sync` (CUDA torch 847 MB) | 1–2 min | 2026-10-14 pods: 61 s; 2026-10-19 pods download torch, longer |
| VPD fetch from W&B (3.2 GB) | 1–1.7 min | `[fetch] model_400000.pth: 2908.6 MB in 26–78 s` |
| provision + SSH + kernel | the rest of the ~5 min the user observes | not logged |
| Billing of the overhead | ~$0.07–0.13 | the PGD check: 1 min of GPU billed $0.13 (`reports/2026-10-19_review_pgd-local.md`) |

Pods per day so far: 2026-10-14 about four, 2026-10-15 four, 2026-10-19 five by 14:51 (notebook count; some
notebooks are a second kernel on the same pod). Session spend attributed to today's other sessions at 14:00: $7.17.
Rate: $0.89/h for a 4090 on Secure Cloud.

So the dollar case for keeping a pod running does not exist: setup bills about a dime, an idle hour bills $0.89. The
case is latency, roughly 5 min × 5 pods ≈ 25 min of waiting per busy day, bought with $7–9 of idle 4090 time over an
8–10 h day. That is most of a session's $10 budget on idleness.

### "Paused" (stopped) is not an option

RunPod's docs ([manage pods](https://docs.runpod.io/pods/manage-pods),
[zero GPUs on resume](https://docs.runpod.io/pods/troubleshooting/zero-gpus),
[pricing](https://docs.runpod.io/pods/pricing)):

- A stopped pod stays pinned to its physical host and **its GPU is released**. If anyone else rents that GPU, the
  resume comes back with zero GPUs (a CPU-only start meant for data recovery). The fix RunPod offers is to terminate and
  redeploy, which is what we do anyway.
- The **container disk is wiped on stop**. Our pods keep the VPD artifacts and the uv cache under `/root/` (container
  disk), so a resume re-fetches the 3 GB and re-syncs regardless. Only `/workspace` (volume disk) survives, billed at
  $0.20/GB/month while stopped.
- The 2026-10-19 task-ladder pod already showed the plugin side of a resume: the SSH port moved and the pod was lost
  mid-transfer (`CLAUDE.md`).

A paused pod therefore saves nothing on resume and adds a failure mode. The agent file now says never to `stop()` a pod
meaning to resume it.

### Keeping it *running*

Pros: no provisioning, venv and artifacts in place, so the next run starts in seconds; one pod ID for the day; fewer
"download before terminate" hand-offs.

Cons:
- Idle cost ($0.89/h) against a $10 session budget; the plugin's `cleanup = "terminate"` ends the pod when the owning
  session ends, so a day pod needs its owning session alive all day or `cleanup = "stop"` (which brings back the resume
  problem).
- One pod is one session's: remote-kernels machines are single-controller, and today's running pod belongs to another
  session. Pod-runner agents in several sessions would have to hand control around with `attach(force=true)`, which
  the plugin treats as a takeover.
- One GPU: two runs at once contend for the 24 GB and the compute, so "many quick experiments of different sizes" would
  serialise anyway.
- The runs of different sizes all fit a 4090 so far (every pod in this repo was a 4090 or L40S), so size is not the
  reason for separate pods; the reason has been that each run lives in its own pod-runner.

### What I recommend

1. **Reuse within a session, which the agent now does by itself.** Before `start()`, the pod-runner calls `status()`;
   an idle `commute-*` pod owned by the same session is attached and reused. This is free and covers the common case
   of a smoke run followed by the scored run.
2. **Keep running only on request.** The brief's new `Pod:` line can say "keep running after, the next run follows
   within ~1 h". Then the pod-runner leaves it up and reports the pod ID, rate and idle-since time. Break-even against a
   fresh pod is about one idle hour in dollars, and 5 min in latency, so the rule of thumb is: next run known and under
   an hour away, keep it; otherwise terminate. The default stays terminate.
3. **(Superseded by section 3: tried the same afternoon, does not work with the plugin.)** The real fix for the 5 min
   looked like a RunPod network volume, not a long-lived pod. Put the VPD artifacts, the HF
   cache and the uv cache on a volume (50 GB ≈ $3.50/month at $0.07/GB/month, billed whether or not a pod is up).
   `remote-kernels.toml` already has `network-volume-id` and `data-center-ids` fields for it. Then every `start()` is a
   fresh, clean pod that skips the fetch and most of the sync, about 2–3 of the 5 minutes. The cost is pinning one data
   center: if that center has no 4090 free, the volume becomes the bottleneck (GPU guide, "Network Volume placement").
   The volume has to be created in the MATS team RunPod account, which is the user's call.

What is *not* changed: `cleanup = "terminate"`, the $10 cap, and the rule that a pod is terminated as soon as it is
idle unless the brief says otherwise.

## 3. Network volume: tried, not adopted (same day)

The user created volume `697hr92me8` (50 GB, Standard, EU-RO-1). A pod-runner filled it from one pod (RunPod id
`edqi6sizcts60y`, 4090, 25 min, $0.36) and started a second to test persistence. Measured on pod 1, venv and caches on
the volume (`/persist`, MooseFS):

| Step | On the volume | Previous pods (container disk) |
| --- | --- | --- |
| `uv sync` (CUDA torch) | 576 s | 61 s |
| `import torch`, cold / warm | 45 s / 22 s | a few s |
| `vpd.fetch()` 3.2 GB from W&B | 140 s | 60–100 s |
| `lora_matched all --smoke` | 208 s | 88 s on the cluster L40 |

Two things rule it out with remote-kernels 0.3.8:

1. **The plugin's working directory is the volume mount.** `volume-mount-path` is where `sync()` lands the tree, where
   kernels run, and where the plugin keeps its lease and watchdog (`<workdir>/.remote-kernels/`). Pod 2 read pod 1's
   lease, reported itself fenced, and had to be deleted through the REST API (the plugin refused `terminate()` on a
   machine it thought was self-cleaning). Only the Kubernetes runtime has a separate `workdir` option.
2. **Small-file I/O on the volume is slow** (table above), so the venv and uv cache cannot live there anyway. The only
   saving left is the fetch, about 2 min, against a layout the plugin cannot run twice.

Decision: `remote-kernels.toml` has the volume lines commented out and `[env]` pointing at `/root/...` on the
container disk. Pods go to any data center again. The volume still exists (29 GB written, about $2/month); delete it in
the console or keep it for a plugin version with a RunPod `workdir` option. Worth asking the plugin author for that
option: the Kubernetes runtime already has it.

Other things the test found: the server reads `remote-kernels.toml` only at start-up, so a config edit needs `/mcp`
reconnect in every session (the first attempt placed a pod in US-MO-1 with none of the new config); the cheaper cards
listed first in `gpu-type-ids` were not in stock in EU-RO-1 and both pods were 4090s; `sync()` deletes remote files
without a local copy; `sync(include=["<dir>"])` is how gitignored inputs go up.

## Not run

- No timing of a full fresh pod from `start()` to first cell; the ~5 min is the user's observation plus the logged
  `uv sync` and fetch.
- No test of `attach()` to an idle pod from a pod-runner (the only running pod belongs to another session).
- No network volume created or tried; the 2–3 min saving is inferred from which setup steps it would remove.
