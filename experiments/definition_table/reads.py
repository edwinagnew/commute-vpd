"""E116 stages 1 and 2: a state's reads on a task's windows. P1 = the state's top-1 within the task's value set S against
the target, on held-out windows; P3's margin = the target's log-odds within S (`case_label.set_log_odds`, E112's margin at
m = 2; lead's decision 2), full-vocabulary log-odds beside; P2 = g on the readout's class at the site (identity or table,
lead's decision 1). GPU only."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from readouts import index_of, site_classes
from tasks import SITES, Site, Task
from torch import Tensor

from commute import case_label as CL
from commute import removal as RM
from commute import steering as ST


@torch.no_grad()
def set_reads(net: ST.Net, s: Site, values: Tensor, dW: ST.DW | None = None, edit: ST.Edit | None = None,
              batch: int = 64) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    """(logits over the values (N, m), full-vocabulary log-odds of the target (N,), full-vocabulary argmax (N,)), CPU."""
    sl, lo, am = [], [], []
    with net.at(dW):
        for i in range(0, len(s.toks), batch):
            lg = net.run(s.toks[i : i + batch], edit)[0]
            sl.append(lg[:, s.half][:, values.to(lg.device)].cpu())
            lo.append(ST.logodds_at(lg, s.half, s.tgt[i : i + batch]))
            am.append(lg[:, s.half].argmax(-1).cpu())
    return torch.cat(sl), torch.cat(lo), torch.cat(am)


def margin(t: Task, s: Site, set_logits: Tensor) -> Tensor:
    y = index_of(t.values, s.tgt, int(max(t.values.max(), s.tgt.max())) + 1)
    assert bool((y >= 0).all()), f"{t.name}: a target is not in the task's value set"
    return CL.set_log_odds(set_logits, y)


def answers(net: ST.Net, t: Task, s: Site, dW: ST.DW | None) -> Tensor:
    """(N,) the state's top-1 token within S."""
    return t.values[set_reads(net, s, t.values, dW)[0].argmax(1)]


def p1(net: ST.Net, t: Task, dW: ST.DW | None) -> dict[str, Any]:
    if t.held is None or t.p1_na:
        return {"P1": math.nan, "na": t.p1_na}
    sl, _, am = set_reads(net, t.held, t.values, dW)
    return {"P1": CL.agreement(t.values[sl.argmax(1)], t.held.tgt), "P1_full_vocab": CL.agreement(am, t.held.tgt), "n": len(am)}


def w0_margins(net: ST.Net, t: Task) -> tuple[Tensor, Tensor]:
    sl, lo, _ = set_reads(net, t.fit, t.values)
    return margin(t, t.fit, sl), lo


def shares(net: ST.Net, t: Task, dW: ST.DW, block: int, bases: dict[str, Tensor], floor: float, w0: tuple[Tensor, Tensor]) -> dict[str, Any]:
    """On the task's fit windows: the gain G (within-S log-odds, and full-vocabulary beside) and, per basis, the share of
    it the removal of the basis from block `block`'s output at the read index takes away (`case_label.removed_share`)."""
    sl, lo, _ = set_reads(net, t.fit, t.values, dW)
    ms = margin(t, t.fit, sl)
    out: dict[str, Any] = {"gain": float((ms - w0[0]).mean()), "gain_full_vocab": float((lo - w0[1]).mean()), "n": len(ms)}
    for name, u in bases.items():
        rs, rlo, _ = set_reads(net, t.fit, t.values, dW, SITES.AP.removal_edit(block, t.fit.half, u))
        out[name] = CL.removed_share(ms, margin(t, t.fit, rs), w0[0], floor)
        out[f"{name}_full_vocab"] = CL.removed_share(lo, rlo, w0[1], floor)
    return out


def p2(net: ST.Net, t: Task, probe: dict[str, Any], block: int, dW: ST.DW) -> dict[str, Any]:
    """g = the identity on classes (no fit) where the answer is the readout's own value; else `removal.table_accuracy`'s
    table fitted on the fit windows, read on the held-out windows (an unseen class counts wrong)."""
    if t.held is None:
        return {"P2": math.nan, "na": "no held-out windows"}
    cls, ans = site_classes(net, probe, block, t.held), answers(net, t, t.held, dW)
    if t.g == "identity":
        return {"P2": CL.agreement(cls, ans), "g": "identity", "n": len(ans)}
    fc, fa = site_classes(net, probe, block, t.fit), answers(net, t, t.fit, dW)
    tab = RM.table_accuracy(fc.numpy().astype(np.float64), fa.numpy(), cls.numpy().astype(np.float64), ans.numpy())
    return {"P2": tab["acc"], "g": "table", "floor": tab["floor"], "unseen": tab["unseen"], "n": len(ans)}
