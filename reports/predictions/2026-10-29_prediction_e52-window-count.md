# E52 predictions: window count note (2026-10-29, before the scored run)

`experiments/baseline_neurons/results/predictions.json` → `settings.windows` says each task has 100 held-out windows,
50 per half, and that the gain G is the mean over all 100. The saved `windows_<task>.pt` files hold 97 held-out windows
per task (found when job 44056 stopped on a loader check, before any measurement). The run therefore uses 97: fit half
49 (even indices), scoring half 48 (odd indices), G over all 97. No rule, threshold, direction or prediction changes;
the parity split and the demonstration-key seeding are as filed. The filed file is left as it was.
