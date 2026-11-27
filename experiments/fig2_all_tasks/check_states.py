"""E325 CPU check, no model, first on the pod (or `fleet/cpu-run --mem 8G python3 experiments/fig2_all_tasks/check_states.py`).
It checks the windows Fig 2 reads, the definition's D (`old_tasks.task_windows`), not E314's held-out rows: which new
tasks have an old_tasks entry; that entry's saved states (seeds 0-1) are known_states' files; each entry's D is 512
windows of 8 tokens (E298's n_windows and window_len); every new state's update file loads as 24 module-keyed
matrices; then the filed config loads (it stops while any new task has no entry)."""

import json
from types import SimpleNamespace

from commute import items as IT
from commute import known_states as KS
from commute import old_tasks as OT
from commute.jobs import EXPERIMENTS, load_experiment

M = load_experiment("fig2_all_tasks", "run")
S = json.loads(M.PRED.read_text())["settings"]
new = [s for s in KS.states() if s.task in S["tasks"]]
print("NO OLD_TASKS ENTRY", sorted({s.task for s in new if s.old_task is None}), flush=True)
for s in new:
    if s.old_task is None:
        continue
    saved = OT.TASKS[s.old_task].states.get(s.seed)
    assert saved is None or saved.resolve() == (EXPERIMENTS.parent / s.file).resolve(), (s.name, saved, s.file)
    print("STATE", s.name, s.old_task, s.file, len(IT.load_dw(IT.Item(s.name, s.file, s.sha256))), flush=True)
for ot in sorted({s.old_task for s in new if s.old_task}):
    D = M.definition_D(ot, SimpleNamespace(n_read=S["n_read"], read_seed=S["read_seed"]))
    print("D", ot, tuple(D.shape), flush=True)
    assert tuple(D.shape) == (512, 8), (ot, D.shape)
M.load_config()
print("CONFIG OK", flush=True)
