# commute

## Remote GPU work (remote-kernels, RunPod)

- Batch runs over ~30 min go here directly, on the fastest card that helps; only runs of ≤ 30 min try the MATS
  cluster's `debug` QoS first, and only when it can start them at once (`.claude/CLAUDE.md` → Rules). The user has
  plenty of credits and little time (2026-11-11): a pod is the default whenever the cluster would make results wait.
- Transfer speeds from the dev node, measured 2026-11-11 on a RunPod 4090: `sync()` sent a fresh 500 MB file in 54 s (~9.5 MB/s) and a 2.6 GB tree at ~12.6 MB/s; `download()` brought 500 MB back in 38 s (~13 MB/s); E153's 270 MB sync took 40 s; W&B fetches on a pod run 30-45 MB/s. So a GB costs about 2 min each way. Estimate any transfer as size ÷ 10 MB/s and show the sum; never cite a transfer time you did not compute this way or measure. The old "~80 KB/s" figure had no measurement behind it and is retired (it made a 268 MB upload look like an hour, friction 716, 720).
- Pods use `cleanup = "terminate"` and have no network volume, so nothing on the machine survives. After every run that produces checkpoints, results, or logs, `download()` them into the project right away, before starting the next run.
- Lessons from the 2026-10-19 task-ladder pod. All 1.4 GB of states were left for one download at the end. A RunPod
  "resume" moved the SSH port mid-transfer, the plugin could not reconnect, and the pod had to be terminated with the
  states unsaved.
  - Save only what the analysis reads: the scored step, and small dtypes.
  - Download each task's outputs as soon as they are written, while the next task runs, not in one batch at the end.
  - `download()` of a directory nested it (`results/results/`). Download into the matching parent, or download the
    files individually, and check the local layout after the first download.
  - `sync-include` did not send gitignored files; `sync(include=["<directory>"])` does (file names alone do not,
    2026-10-19). Check sha256 on both sides. The directory named must be the gitignored one itself
    (`.../results/large`): a file or a subdirectory under it is never walked, and the sync still reports success
    (2026-10-23, two pods idled 20-25 min each). To send a few files from a large directory, hard-link them into a
    staging path ending in `results/large/` and name that path; start the sync before set-up.
    Delete the staging directory afterwards. Since `~/commute-agents/fleet/link-data`, every `results/large` is a symlink and arrives
    on the pod dangling (2026-10-27): always stage by hard link, and replace the symlink on the pod with the real
    directory. The W&B fetch of `artifacts/vpd` works again (2026-10-27, ~45 MB/s, 65 s).
  - If the plugin loses the pod (connection refused, port changed), stop and tell the user at once. A `/mcp`
    reconnect of remote-kernels may fix it. Don't retry in a loop while the pod bills.
- Lesson from the 2026-10-19 lora_commute pod. The fork downloaded after each arm as told, but the FINAL JSON (score, last
  arms) and a late extra file were still not on disk when the machine was terminated; the run's log saved the numbers.
  Before `terminate()`: download the final results file, verify its sha256 on both sides, then terminate. Make the run
  print every scored number to the log as well, so a lost file is a summary loss, not a run loss.
- Lesson from the 2026-10-26 teacher-training amendments (three pods lost, ~$9 of reruns). With `cleanup =
  "terminate"` the machine terminates itself when the session disconnects (plugin README, "Automatic cleanup"). Two
  pods died within a minute of a runner rewriting `remote-kernels.toml` and starting a third; one died 75 min into a
  run with no config change; one survived a WebSocket closure because its runner opened a new kernel within two
  minutes. Rules: never edit `remote-kernels.toml` while a machine of the session runs (label pods through `start()`'s
  argument); no `execute()` or `wait()` over 5 min, re-arm instead; on a WebSocket closure, `status(instance=...)`
  then `create_kernel(instance=...)` at once; download each stage as it is written; pass `instance` to every
  machine-scoped call when several machines run.
- Lesson from the 2026-10-26 E36 Amendment 6 pod (~$40 lost) and E34 Amendment 5 pod (idle 40 min). Each
  `results/large/` is a symlink into `~/commute-data/results-large/`, outside the project: `download()` there is
  refused, and the `mv` into it waits at a permission prompt, all night when nobody is there. Terminate the moment the
  results JSON's sha256 matches, before any local file move. Download large files to a project path (e.g.
  `results/<run>_dl/`) and move them into `large/` after the pod is gone.
- Stop or terminate machines as soon as they are idle. The $100/session `REMOTE_KERNELS_BUDGET` (2026-11-03; $40 before) is the budget and the backstop that terminates machines.
