# commute

## Remote GPU work (remote-kernels, RunPod)

- Interactive Jupyter dev only. Single-GPU batch jobs (1–24h) go to the MATS cluster first (see ../CLAUDE.md).
- Pods use `cleanup = "terminate"` and have no network volume, so nothing on the machine survives. After every run that produces checkpoints, results, or logs, `download()` them into the project right away, before starting the next run.
- Stop or terminate machines as soon as they are idle. The $20/session `REMOTE_KERNELS_BUDGET` is only a backstop.
