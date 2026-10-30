"""E54's key-dependent pair (elicit map and cipher): the fine-tunes, the matched-size step and the two readings (steering's
S and the route split's r), shared since 2026-10-30 with E56 (`experiments/keydependent_patch`). Moved, not changed,
from `experiments/keydependent_elicit/run.py`, which now calls them; `elicit_ft` and `cipher_ft` are the bodies of its
`finetunes` loops, one seed each, the same calls in the same order.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import anchors as AN
from commute import elicit as EL
from commute import steering as ST
from commute.faithfulness import model_state

E12_CHECKS = (10, 30, 100, 300)
CIPHER_CHECKS = (*range(1, 31), 100, 300)
P_LEARN, BATCH = 0.5, 64


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def geo_rate(lp: Tensor) -> float:
    """Geometric mean over windows of P(target): exp(mean log P)."""
    return math.exp(float(lp.mean()))


@dataclass
class Ctx:
    net: ST.Net
    lift: dict[str, Any]
    comps0: dict[str, tuple[Tensor, Tensor]]
    P: dict[str, Any]
    large: Path


def make_ctx(device: str, alive_path: Path, P: dict[str, Any], large: Path) -> Ctx:
    from commute import vpd

    net = ST.Net(device)
    alive = torch.load(alive_path, weights_only=False)["alive"]
    lift = AN.lift_context(net.model, {k: v.cpu().bool() for k, v in alive.items()})
    return Ctx(net, lift, vpd.component_weights(net.model), P, large)


# ----------------------------------------------------------------------------- the two readings


def base_of(ctx: Ctx, t: ST.Task) -> dict[str, Any]:
    from commute.local_faith import route_cells

    return route_cells(ctx.net.model, t.toks, t.half, t.table, BATCH)


def route_r(ctx: Ctx, t: ST.Task, dW: ST.DW, base: dict[str, Any]) -> dict[str, Any]:
    """r of the least-squares split state: target W0 + delta-W, components lifted; r = (L01 - L00) / (T1 - T0)."""
    from commute.local_faith import route_split

    m = ctx.net.model
    target = {k: w + dW[k].to(w.device, w.dtype) for k, w in ctx.net.W0.items()}
    with model_state(m, target=target, components=AN.lifted(m, ctx.lift, dW)):
        rs = route_split(m, t.toks, t.half, t.table, base, ctx.comps0, BATCH)
    return {"r": rs["shares"]["r"], "G_logodds": rs["gain"], "cells": rs["cells"]}


def steer_reading(ctx: Ctx, t: ST.Task, dW: ST.DW, g: dict[str, float]) -> dict[str, Any]:
    """Steering's S (E52's B3) of one state, its block and its block-0 share. Taken out of `readings` (2026-10-30, for
    experiments/regating_truth_table), the same calls in the same order."""
    b = ST.b3(ctx.net, t, dW, g)
    # the filed edge case: a share whose gain is within 1e-6 of 0 is nan (here, not in steering.py)
    return {"S": EL.share_or_nan(b["S"], g["G_score"]), "layer": b["layer"], "S_block0": b["layers"][0]["score"], "b3_layers": b["layers"]}


def readings(ctx: Ctx, t: ST.Task, dW: ST.DW, g: dict[str, float], base: dict[str, Any]) -> dict[str, Any]:
    s = steer_reading(ctx, t, dW, g)
    rr = route_r(ctx, t, dW, base)
    return {**s, **rr, "r": EL.share_or_nan(rr["r"], rr["G_logodds"])}


# ----------------------------------------------------------------------------- the fine-tunes


def finetune(ctx: Ctx, w: dict[str, Any], seed: int, checks: tuple[int, ...]) -> Any:
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=w["table"], lr=AN.FT_LR, steps=E12_CHECKS[-1], checkpoints=checks, seed=seed)
    return finetune_arm_a(ctx.net.model, w["train"], w["heldout"], cfg, ctx.net.device, log)


def scored_step(evals: dict[int, dict[str, float]]) -> tuple[int, bool]:
    """E12's rule: the first of 10, 30, 100, 300 with held-out P(target) >= 0.5, else 300 ('did not learn')."""
    return next(((s, True) for s in E12_CHECKS if evals[s]["p_o"] >= P_LEARN), (E12_CHECKS[-1], False))


def elicit_ft(ctx: Ctx, w: dict[str, Any], seed: int) -> tuple[ST.DW, dict[str, Any]]:
    """The elicit map's fine-tune at E12's scored step, and its record."""
    ft = finetune(ctx, w, seed, E12_CHECKS)
    step, learned = scored_step(ft.evals)
    return ft.deltas[step], {"step": step, "learned": learned, "norm": EL.summed_norm(ft.deltas[step]),
                             "p_heldout": {k: e["p_o"] for k, e in ft.evals.items()}}


def cipher_ft(ctx: Ctx, w: dict[str, Any], seed: int, target_norm: float) -> tuple[ST.DW, ST.DW, dict[str, Any], dict[str, Any]]:
    """The cipher's fine-tune read at the step in 1..30 whose ||delta-W|| is nearest `target_norm` (the elicit
    fine-tune's) and at its own E12 step: (matched delta-W, own delta-W, matched record, own record)."""
    ft = finetune(ctx, w, seed, CIPHER_CHECKS)
    norms = {k: EL.summed_norm(ft.deltas[k]) for k in range(1, 31)}
    m = EL.matched_step(norms, target_norm)
    own, learned = scored_step(ft.evals)
    meta = {"step": m, "norm": norms[m], "at_bound": m == 30, "same_state": m == own, "norms": norms,
            "p_heldout": {k: e["p_o"] for k, e in ft.evals.items()}}
    return ft.deltas[m], ft.deltas[own], meta, {"step": own, "learned": learned, "norm": EL.summed_norm(ft.deltas[own])}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
