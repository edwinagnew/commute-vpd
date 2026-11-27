"""E325 CPU check, no model, first on the pod (or `fleet/cpu-run --mem 8G python3 experiments/fig2_all_tasks/check_states.py`):
the filed config loads, commute.known_states gives E314's 48 states (same file and rows), every new state's update
file loads as 24 module-keyed matrices, and each read task's rows equal E314's (E136 load_read, target column cut)."""

import torch

from commute import items as IT
from commute import known_states as KS
from commute.jobs import load_experiment

M = load_experiment("fig2_all_tasks", "run")
cfg = M.load_config()
print("ITEMS", len(cfg.items()), cfg.items()[:3], flush=True)
E314 = load_experiment("sae_proxies", "run")
old = {s.name: s for s in E314.states()}
for s in KS.states():
    o = old[s.name]
    assert (o.file, o.rows) == (s.file, s.rows), s.name
print("known_states == E314's states (file, rows) on all 48", flush=True)
for t, e in cfg.overlap.items():
    for sd in cfg.seeds:
        assert KS.by_name()[f"{t}_s{sd}"].file == f"experiments/refit_96key/results/large/dW_{e}_s{sd}.pt"
seen = set()
for n in cfg.new_states():
    s = KS.by_name()[n]
    dw = IT.load_dw(IT.Item(n, s.file, s.sha256))
    print("STATE", n, s.rows, s.trained_by, len(dw), flush=True)
    if s.rows not in seen:
        seen.add(s.rows)
        r = KS.rows(s)
        o = E314.W.load_read(old[n].rows, old[n].spec)
        ot = o.toks[:, : o.ci_len] if o.ci_len is not None else o.toks
        print("ROWS", s.rows, tuple(r.toks.shape), "pos", r.pos, "same as E314:", torch.equal(ot, r.toks) and o.half == r.pos, flush=True)
