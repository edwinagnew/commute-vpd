"""E112's model-side reads of the frozen reuse definition (`decisions.md` rows 81, 87, 164), with the prediction
position and the window length passed in instead of E112's fixed HALF and 41 tokens (moved for E124, 2026-11-06;
`experiments/case_label/run.py` keeps its own copies as it ran). Same calls in the same order as E112's `label_diff`,
`margin`, `removal_edit`, `resid_at`, `removal_share`, `state_reads`, `prune` and `reverse`; the labels y (N,) of each
window are passed in, since E124's content word sits at a different place in every window. Rules come only from
`commute.case_label` (`fit_threshold`, `apply_threshold`, `removed_share`).

Every `toks` here is exactly what the model sees (the prediction position `pos` is a column of it); targets, when
needed, are a separate (N,) tensor.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from commute import case_label as CL
from commute import readout as RO
from commute import removal as RM
from commute import revival_kv as RK
from commute import steering as ST
from commute.copying import with_target
from commute.data import LAST, Windows
from commute.finetune import ArmAConfig, StopTraining, finetune_arm_a
from commute.identity import log_odds

BATCH = 64


@torch.no_grad()
def logits_at(net: ST.Net, toks: Tensor, pos: int, dW: ST.DW | None = None, edit: ST.Edit | None = None) -> Tensor:
    """(N, V) fp32 CPU logits at `pos`."""
    with net.at(dW):
        return torch.cat([net.run(toks[s : s + BATCH], edit)[0][:, pos].cpu() for s in range(0, len(toks), BATCH)])


def label_diff(net: ST.Net, toks: Tensor, ab: tuple[int, int], pos: int, dW: ST.DW | None = None, edit: ST.Edit | None = None) -> Tensor:
    """(N,) float64: logit A - logit B at `pos` (E112's `label_diff`)."""
    lg = logits_at(net, toks, pos, dW, edit)
    return (lg[:, ab[0]] - lg[:, ab[1]]).double()


def full_log_odds(net: ST.Net, toks: Tensor, target: Tensor, pos: int, dW: ST.DW | None = None) -> Tensor:
    """(N,) float64: log-odds of each window's target over the whole vocabulary at `pos` (the route split's T unit)."""
    return log_odds(logits_at(net, toks, pos, dW), target).double()


def accuracy(d: Tensor, y: Tensor) -> float:
    return float(((d > 0).long() == y).double().mean()) if len(y) else float("nan")


def margin(d: Tensor, y: Tensor) -> Tensor:
    return d * (2 * y - 1)


def removal_edit(block: int, u: Tensor, pos: int) -> ST.Edit:
    """Remove the unit direction u (d, 1) from block `block`'s output residual at `pos` only."""

    def f(x: Tensor) -> Tensor:
        y = x.clone()
        y[:, pos] = RM.remove(x[:, pos], u.to(x.device, x.dtype))
        return y

    return {RO.RESID[block]: f}


@torch.no_grad()
def resid_at(net: ST.Net, toks: Tensor, pos: int) -> Tensor:
    """(4, N, d) fp32 CPU: block outputs at `pos` at W0."""
    return torch.cat([RO.resid(net, toks[s : s + BATCH])[1][:, :, pos].cpu() for s in range(0, len(toks), BATCH)], 1)


@torch.no_grad()
def next_ce(net: ST.Net, toks: Tensor, pos: int, edit: ST.Edit | None = None) -> float:
    """W0's mean next-token loss at `pos`; toks carries the true next token as its last column (not fed)."""
    ce = [F.cross_entropy(net.run(toks[s : s + BATCH, :-1], edit)[0][:, pos], toks[s : s + BATCH, -1].to(net.device), reduction="sum")
          for s in range(0, len(toks), BATCH)]  # fmt: skip
    return float(torch.stack(ce).sum()) / len(toks)


def variance_share(h: Tensor, u: Tensor) -> float:
    """The share of the residual variance of rows h (N, d) along the unit direction u (d, 1)."""
    hc = h.double() - h.double().mean(0)
    return float((hc @ u.double()).pow(2).sum() / hc.pow(2).sum())


def removal_share(net: ST.Net, toks: Tensor, ab: tuple[int, int], y: Tensor, dW: ST.DW, block: int, u: Tensor, pos: int,
                  floor: float) -> dict[str, float]:  # fmt: skip
    """E112's P3: the label log-odds gain on the windows and the share of it the removal of u at `block` takes away."""
    m0, ms = margin(label_diff(net, toks, ab, pos), y), margin(label_diff(net, toks, ab, pos, dW), y)
    mr = margin(label_diff(net, toks, ab, pos, dW, removal_edit(block, u, pos)), y)
    return {"gain": float((ms - m0).mean()), "share": CL.removed_share(ms, mr, m0, floor)}


def state_reads(net: ST.Net, R: Mapping[str, Any], fit: tuple[Tensor, Tensor], held: tuple[Tensor, Tensor], ab: tuple[int, int],
                dW: ST.DW, l_star: int, w: Tensor, b: Tensor, h0: Mapping[str, Tensor], pos: int) -> dict[str, Any]:  # fmt: skip
    """E112's P1-P3 of one state; fit and held are (toks, y). P2's g is a threshold on W0's readout score at l*, fitted to
    the state's own labels on the fit windows."""
    dh, df = label_diff(net, held[0], ab, pos, dW), label_diff(net, fit[0], ab, pos, dW)
    score = {k: ((h0[k][l_star].double() @ w) + b).numpy() for k in ("fit", "held")}
    theta, sign = CL.fit_threshold(score["fit"], (df > 0).long().numpy())
    p3 = removal_share(net, fit[0], ab, fit[1], dW, l_star, RM.unit_basis(w), pos, R["gain_floor"])
    return {"P1": accuracy(dh, held[1]), "P2": float((CL.apply_threshold(score["held"], theta, sign) == (dh > 0).long().numpy()).mean()),
            "g": {"theta": theta, "sign": sign}, "gain": p3["gain"], "P3": p3["share"], "P3_no_gain": not p3["gain"] > R["gain_floor"]}  # fmt: skip


def prune(net: ST.Net, fit: Tensor, held: tuple[Tensor, Tensor], ab: tuple[int, int], dW: ST.DW, pos: int, ks: Sequence[int],
          n_random: int, seed: int) -> dict[str, Any]:  # fmt: skip
    """Beside, no rule (E112's `prune`): held-out label accuracy after zeroing the top-K MLP units by grad . w of the fit
    windows' KL to W0 (Jain App. D Alg. 3, `revival_kv`), against K random units."""
    ref = RK.make_ref(net, fit, pos)
    scores, n = RK.unit_scores(net, ref, dW), ST.N_LAYERS * ST.D_MLP
    acc = lambda units: accuracy(label_diff(net, held[0], ab, pos, dW, RK.zero_units(units)), held[1])
    out: dict[str, Any] = {"acc": accuracy(label_diff(net, held[0], ab, pos, dW), held[1])}
    for k in ks:
        rng = np.random.default_rng([seed, k])
        out[str(k)] = {"top": acc(torch.topk(scores, k).indices.tolist()), "random": [acc(rng.choice(n, k, replace=False).tolist()) for _ in range(n_random)]}
    return out


def reverse(net: ST.Net, C: Any, train: tuple[Tensor, Tensor], held: tuple[Tensor, Tensor], ab: tuple[int, int], dW: ST.DW,
            pos: int, deadline: float) -> dict[str, Any]:  # fmt: skip
    """Beside, no rule (E112's `reverse`): E62's recipe on the train windows' own next tokens (train = (toks, text's next
    token)), from the state; held-out label accuracy every `reverse_every` steps until <= the bar, the cap or the deadline."""
    curve: list[tuple[int, float]] = []
    end = "cap"

    def on_step(step: int, _p: Any) -> None:
        nonlocal end
        if step % C.reverse_every == 0:
            curve.append((step, accuracy(label_diff(net, held[0], ab, pos), held[1])))
            end = "bar" if curve[-1][1] <= C.reverse_bar else "deadline" if time.monotonic() > deadline else end
            if end != "cap":
                raise StopTraining

    n = train[0].shape[1]
    cfg = ArmAConfig(o_token=LAST, kl_lambda=C.kl_lambda, lr=C.lr, batch=C.batch, steps=C.reverse_cap, checkpoints=(), grad_clip=C.grad_clip,
                     seed=C.reverse_seed, ci_len=n)  # fmt: skip
    w = with_target(wins(train[0], pos), train[1])
    with net.at(dW):
        curve.append((0, accuracy(label_diff(net, held[0], ab, pos), held[1])))
        try:
            finetune_arm_a(net.model, w, w, cfg, net.device, lambda _m: None, on_step)
        except StopTraining:
            pass
    return {"curve": curve, "end": end, "steps_to_bar": CL.reverse_steps(curve, C.reverse_bar)}


def wins(toks: Tensor, pos: int) -> Windows:
    n = len(toks)
    return Windows(toks, torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), pos)
