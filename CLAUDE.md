# commute

## Remote GPU work (remote-kernels, RunPod)

- Batch runs over ~30 min go here directly, on a 4090 or faster card; only runs of ≤ 30 min try the MATS cluster's
  `debug` QoS first (`.claude/CLAUDE.md` → Rules, user 2026-10-20).
- Pods use `cleanup = "terminate"` and have no network volume, so nothing on the machine survives. After every run that produces checkpoints, results, or logs, `download()` them into the project right away, before starting the next run.
- Lessons from the 2026-10-19 task-ladder pod. All 1.4 GB of states were left for one download at the end. A RunPod
  "resume" moved the SSH port mid-transfer, the plugin could not reconnect, and the pod had to be terminated with the
  states unsaved.
  - Save only what the analysis reads: the scored step, and small dtypes.
  - Download each task's outputs as soon as they are written, while the next task runs, not in one batch at the end.
  - `download()` of a directory nested it (`results/results/`). Download into the matching parent, or download the
    files individually, and check the local layout after the first download.
  - `sync-include` did not send gitignored files; `sync(include=["<directory>"])` does (file names alone do not,
    2026-10-19). Check sha256 on both sides.
  - If the plugin loses the pod (connection refused, port changed), stop and tell the user at once. A `/mcp`
    reconnect of remote-kernels may fix it. Don't retry in a loop while the pod bills.
- Lesson from the 2026-10-19 lora_commute pod. The fork downloaded after each arm as told, but the FINAL JSON (score, last
  arms) and a late extra file were still not on disk when the machine was terminated; the run's log saved the numbers.
  Before `terminate()`: download the final results file, verify its sha256 on both sides, then terminate. Make the run
  print every scored number to the log as well, so a lost file is a summary loss, not a run loss.
- Stop or terminate machines as soon as they are idle. The $40/session `REMOTE_KERNELS_BUDGET` is the budget and the backstop that terminates machines.
