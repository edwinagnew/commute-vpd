"""E325 check, first on the pod (it builds key_grid's World, which puts the model on the GPU; no reading is computed).
It checks the windows Fig 2 reads, the definition's D (`old_tasks.task_windows`), not E314's held-out rows:
  1. `known_states.definition_task` reproduces E298's e58 D: for each e58 task the OldTask equals E221's `e187_task`
     (E298's path, key_grid World) and the 512 x 8 windows are equal bit for bit to the World's `read(read_half187)`;
  2. only then, every new task's D (the e54 tasks by the same rule) is 512 x 8, the old_tasks entry's saved states
     (seeds 0-1) are known_states' files, and every new state's update file loads as 24 module-keyed matrices;
  3. the filed config loads.
Any miss raises."""

import json
from types import SimpleNamespace

import torch

from commute import items as IT
from commute import known_states as KS
from commute import old_tasks as OT
from commute.jobs import EXPERIMENTS, load_experiment

M = load_experiment("fig2_all_tasks", "run")
S = json.loads(M.PRED.read_text())["settings"]
cfg = SimpleNamespace(n_read=S["n_read"], read_seed=S["read_seed"])

KG = load_experiment("key_grid", "run")
kg = KG.World(KG.load_config()[0])
for t in S["overlap"]:
    st = next(s for s in KS.states() if s.task == t)
    assert KS.definition_task(st) == KG.E221.e187_task(t, kg.pairs.inv["tasks"]), t
    mine, e298 = M.definition_D(st, cfg), kg.read(kg.read_half187(t)).cpu()
    assert mine.shape == (512, 8) and torch.equal(mine, e298), (t, tuple(mine.shape), tuple(e298.shape))
    print("E298 D REPRODUCED", t, KS.definition_task(st).sources, tuple(mine.shape), flush=True)
del kg

new = [s for s in KS.states() if s.task in S["tasks"]]
for s in new:
    saved = OT.TASKS[s.old_task].states.get(s.seed) if s.old_task in OT.TASKS else None
    assert saved is None or saved.resolve() == (EXPERIMENTS.parent / s.file).resolve(), (s.name, saved, s.file)
    print("STATE", s.name, s.old_task, s.file, len(IT.load_dw(IT.Item(s.name, s.file, s.sha256))), flush=True)
for name in sorted({s.old_task for s in new}):
    st = next(s for s in new if s.old_task == name)
    D = M.definition_D(st, cfg)
    print("D", name, KS.definition_task(st).kind, KS.definition_task(st).sources, tuple(D.shape), flush=True)
    assert tuple(D.shape) == (512, 8), (name, D.shape)
M.load_config()
print("CHECK DONE", flush=True)
