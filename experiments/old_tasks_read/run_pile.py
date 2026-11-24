"""E292 (cell P2.74 run 2, whip-lead 11-24): the Pile loss change of E272's realistic-panel states, so step 3 sets the
old and realistic tasks on the same axes. E272 saved each state's KL and held gain but not its Pile loss change.

Read with run.py's `Extra` (the E292 old-task reading: Pile val rows 0-63, mean log P at W0 minus at W0 + dW, and
KL from W0 on the same rows), on every E272 state that step 3 plots: the wide set's kl_hi pick per task and seed
(results/wide/selection.json), and each seed's saves in results/train/<task>.json (for the steps to half gain).
Do-nothing first: W0 (dW = 0) must read KL 0 and Pile loss change 0 exactly, else the job stops. Beside each state,
E272's own KL for that file (not a stop: E272 measured it on its own kl rows).

    sbatch --comment=E292 cluster/old_tasks_read.sbatch pile        -> results/e272_pile.json (written as it goes)
"""

from __future__ import annotations

import json
from typing import Any

import torch

from commute import steering as ST
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, write_stage
from commute.open_pair import jsonable

RUN: Any = load_experiment("old_tasks_read", "run")
REPO = EXPERIMENTS.parent
E272 = EXPERIMENTS / "convergence_panel" / "results"
OUT = RUN.RESULTS / "e272_pile.json"


def files() -> dict[str, float | None]:
    """Every state file to read -> E272's KL for it."""
    out: dict[str, float | None] = {}
    sel = json.loads((E272 / "wide" / "selection.json").read_text())["kl"]["hi"]["tasks"]
    for by in sel.values():
        for s in by.values():
            out[s["file"]] = s["kl"]
    for task in sel:
        for saves in json.loads((E272 / "train" / f"{task}.json").read_text()).get("seeds", {}).values():
            for s in saves if isinstance(saves, list) else []:
                if "file" in s:
                    out.setdefault(s["file"], s.get("kl"))
    return out


def main() -> int:
    C, _ = RUN.load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    ex = RUN.Extra(net, C)
    d: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": "E292 run_pile", "states": {}}
    zero = ex.of(RUN.zero(net))  # run.py stage 0's do-nothing
    d["do_nothing"] = zero
    if zero["kl"] != 0.0 or zero["pile_loss_change"] != 0.0:
        write_stage(OUT, jsonable(d | {"stopped": f"W0 reads {zero}, not exactly 0"}))
        RUN.log(f"STOP: do-nothing {zero}")
        return 1
    todo = files()
    RUN.log(f"run_pile: do-nothing {zero}; {len(todo)} E272 state files, {len(d['states'])} already read")
    for k, (f, kl272) in enumerate(sorted(todo.items())):
        if f in d["states"]:
            continue
        dW = {n: v.float() for n, v in torch.load(REPO / f, weights_only=False).items()}
        d["states"][f] = ex.of(dW) | {"kl_e272": kl272}
        write_stage(OUT, jsonable(d))
        RUN.log(f"STATE {k + 1} of {len(todo)} DONE")
    RUN.log("E292 run_pile DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main())
