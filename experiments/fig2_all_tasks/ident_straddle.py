"""Neurons identity diagnostic for E326's stage-0 anomaly (cell P1.66): the identity swap (x' = x, every variable) read
under the old neuron hook (x's activation at the swapped layer taken from the rerun) and under the fixed one
(`needed_neurons.Swap`, x's activation taken from F's cached run), on the same states, windows and pairs.

The old hook also counts straddles: entries it splits where the rerun's activation and the cached one fall on opposite
sides of the neuron's bound. Expected (filed here, committed before the run): old max_dev 0.02715 on e62/copy_s0 and 0.0692 on e46/ident32_s0 with >= 1 straddle each, ~5e-5 with 0
straddles on e58/copy_s0 and c3/copy_s0; fixed max_dev under 1e-4 on all four, 0 flips.

    PRED_SHA=<sha> python experiments/fig2_all_tasks/ident_straddle.py   # writes results/ident_straddle.json
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

from commute import needed as IX
from commute import needed_neurons as NN
from commute.jobs import hard_exit, load_experiment, require_filed

RUN: Any = load_experiment("fig2_all_tasks", "run")
STATES = ("e62/copy_s0", "e46/ident32_s0", "e58/copy_s0", "c3/copy_s0")


class OldSwap(NN.Swap):
    """E326's hook as it ran (x's activation from the rerun), counting straddles of the entries it splits."""

    straddles = 0
    max_noise = 0.0

    def mlp(self, _m: Any, inp: tuple[Tensor, ...]) -> tuple[Tensor] | None:
        assert self.var is not None and self.ex is not None and self.pt is not None
        a, ex, pt = inp[0], self.ex, self.pt
        neu = ((self.var >= NN.H) & (self.var < NN.H + NN.NEUR)).nonzero().squeeze(1)
        rem = (self.var == NN.H + NN.NEUR).nonzero().squeeze(1)
        if not len(neu) and not len(rem):
            return None
        rows = torch.cat([neu, rem])
        OldSwap.max_noise = max(OldSwap.max_noise, float((a[rows] - self.a[ex[rows]]).abs().max()))
        a = a.clone()
        if len(neu):
            i = self.var[neu] - NN.H
            th = self.hi[i][:, None]
            cur, new = a[neu, :, i], self.a[pt[neu], :, i]
            OldSwap.straddles += int(((cur > th) != (self.a[ex[neu], :, i] > th)).sum())
            a[neu, :, i] = (cur - NN.strong(cur, th)) + NN.strong(new, th)
        if len(rem):
            new = self.a[pt[rem]]
            OldSwap.straddles += int(((a[rem] > self.hi) != (self.a[ex[rem]] > self.hi)).sum())
            a[rem] = NN.strong(a[rem], self.hi) + (new - NN.strong(new, self.hi))
        return (a,)


def one(rd: Any, sts: Any, cfg: Any, item: str) -> dict[str, Any]:
    F, toks, at, _ = sts.make(item)
    st = rd.prep(F, sts.M, toks)
    run = rd.cache(st, F, toks, at)
    ex = torch.arange(min(cfg.examples, len(toks)))
    rows = IX.rows_per_forward(cfg.rows, cfg.logit_rows, run.top.shape[1])
    new = rd.scan(st, F, run, at, ex, ex)
    OldSwap.straddles, OldSwap.max_noise = 0, 0.0
    sw = [OldSwap(run, l, rd.feat.hi[l]) for l in range(len(F.net.h))]
    old = IX.scan_sites(F.net, sw, run, at, ex, ex, rows, len(F.net.h) * NN.PER_LAYER, RUN.log)
    return {"item": item, "old_max_dev": old.max_dev, "old_flips": int(old.flips.sum()), "straddles": OldSwap.straddles,
            "rerun_noise_max": OldSwap.max_noise, "fixed_max_dev": new.max_dev, "fixed_flips": int(new.flips.sum()), "cases": new.rows}  # fmt: skip


def main() -> int:
    cfg = RUN.load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E298
    rd = RUN.Neurons(cfg)
    sts = RUN.States(cfg, RUN.World.load("cuda"), RUN.RESULTS / "state_files" / "neurons_ident_straddle.json")
    out = []
    for item in STATES:
        out.append(one(rd, sts, cfg, item))
        RUN.log(f"STRADDLE {json.dumps(out[-1])}")
        (RUN.RESULTS / "ident_straddle.json").write_text(json.dumps(out, indent=1))
    RUN.log("=== FINISHED ident_straddle ===")
    return 0


if __name__ == "__main__":
    require_filed(RUN.PRED)
    hard_exit(main())
