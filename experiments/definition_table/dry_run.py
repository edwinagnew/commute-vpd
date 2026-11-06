"""E116 dry run on CPU, no model (`cpu-run --mem 8G python experiments/definition_table/dry_run.py`): builds every task and
readout from the saved files, checks each target lies in its task's value set and each scoring window's class in its
readout's values, and runs the Pile position finders that need no model (copy, identity) on the probe rows."""

from __future__ import annotations

import json
from pathlib import Path

import readouts as RO
import tasks as TK
import torch

from commute.data import pile_rows

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "results" / "predictions.json").read_text())
S = P["settings"]


class C:  # the settings as attributes, as run.Config gives them
    pass


for k, v in S.items():
    setattr(C, k, v)

F = TK.SITES.Files()
TK.add_records(F)
T, R = TK.build(F, C)
V = 50277
for n, t in T.items():
    for part, s in (("fit", t.fit), ("held", t.held)):
        if s is None:
            continue
        miss = int((RO.index_of(t.values, s.tgt, V) < 0).sum())
        print(f"{n} {part}: {len(s.tgt)} windows of {s.toks.shape[1]} tokens, |S| {len(t.values)}, targets outside S {miss}, states {sorted(t.states)}")
        assert miss == 0
print("states", sum(len(t.states) for t in T.values()))
rows = pile_rows(*S["probe_rows"])
for r in R:
    if r.kind == "guess":
        print(f"{r.name}: m {len(r.values)}, score windows {len(r.score.tgt)} (labels need W0)")
        continue
    y = RO.score_labels(None, r, V)  # type: ignore[arg-type]
    toks, py, types = RO.pile_positions(None, r, rows, rows[:0], C, V)  # type: ignore[arg-type]
    cov = int((torch.bincount(py, minlength=len(r.values)) > 0).sum())
    print(f"{r.name}: m {len(r.values)}, score windows {len(y)} majority {float(torch.bincount(y).max()) / len(y):.3f}; "
          f"Pile windows {len(py)} {tuple(toks.shape)}, values with a position {cov}, types {len(set(types.tolist()))}")
    if r.kind == "copy":  # each window: token at 19 occurs once in 0..18 and the token after it is the label
        for w, lab in zip(toks[:200], py[:200], strict=True):
            hits = (w[:19] == w[19]).nonzero()[:, 0]
            assert len(hits) == 1 and int(w[hits[0] + 1]) == int(r.values[lab]), (w, lab)
        print("copy windows checked")
