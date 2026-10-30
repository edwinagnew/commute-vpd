"""The held-out-key label of a key -> value fine-tune and E54's readings beside it (E58, `experiments/open_pair`).
Moved, not changed, from `experiments/open_pair/run.py` on 2026-10-30 so that E58 Amendment 1 (`amend1.py`) calls the
same code; the one edit: `pool_of` drops repeated values (a no-op on E58's maps, whose 32 values are distinct; the
rank-1 map of Amendment 1 can repeat a value). Plan: reports/plans/2026-10-30_plan_e58-open-pair.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import keydep as KD
from commute import open_pair as OP
from commute import steering as ST
from commute.data import Windows
from commute.identity import log_odds
from commute.kv import KVTask

PARTS = ("trained", "new")


def keyed(w: Windows, keys: list[int]) -> Windows:
    return w.subset(torch.isin(w.tokens[:, w.half], torch.tensor(keys)))


def split_task(src: Path, trained: list[int], held: list[int], large: Path, name: str) -> dict[str, str]:
    """E54's windows file cut by key: `<name>_trained` = the trained keys' train and held-out windows and their table;
    `<name>_new` = every window of the held-out keys (both of E54's row splits; none is ever trained on here) and their
    table. Saved; their sha256."""
    w = torch.load(src, weights_only=False)
    task, V = w["task"], int(w["table"].shape[0])
    val = dict(zip(task.keys, task.values, strict=True))
    sub = {p: KVTask(f"{name}_{p}", task.marker, tuple(ks), tuple(val[k] for k in ks)) for p, ks in (("trained", trained), ("new", held))}
    tr, nw = keyed(w["train"], held), keyed(w["heldout"], held)
    files = {"trained": {"train": keyed(w["train"], trained), "heldout": keyed(w["heldout"], trained)},
             "new": {"heldout": Windows(torch.cat([tr.tokens, nw.tokens]), torch.cat([tr.g, nw.g]), torch.cat([tr.row, nw.row]),
                                        torch.cat([tr.pos, nw.pos]), tr.half)}}
    assert len(files["trained"]["train"]) + len(tr) == len(w["train"]) and len(files["trained"]["heldout"]) + len(nw) == len(w["heldout"])
    sha = {}
    for p, f in files.items():
        path = large / f"windows_{name}_{p}.pt"
        torch.save({**f, "task": sub[p], "table": sub[p].table(V)}, path)
        sha[p] = KD.sha256(path)
    return sha


def gain(net: ST.Net, t: ST.Task, dW: ST.DW, readout: ST.Readout = ST.logodds_at) -> Tensor:
    """Per window, the fine-tuned model's log-odds of the target minus W0's (target model only)."""
    return ST.read(net, t, t.toks, dW, readout=readout) - ST.read(net, t, t.toks, readout=readout)


def others(values: list[int]) -> ST.Readout:
    """Per window, the mean log-odds of the held-out values other than the window's own target (the prior-shift check)."""
    vals = torch.tensor(values)

    def f(logits: Tensor, pos: int, tgt: Tensor) -> Tensor:
        lo = torch.stack([log_odds(logits[:, pos], v) for v in values], 1).double().cpu()
        keep = (vals[None, :] != tgt.cpu()[:, None]).double()
        return (lo * keep).sum(1) / keep.sum(1)

    return f


def pool_of(t: dict[str, ST.Task]) -> list[int]:
    """The task's values (trained keys' then held-out keys'), each value once."""
    return list(dict.fromkeys(int(t[p].table[k]) for p in PARTS for k in t[p].keys))


def specific(net: ST.Net, t: ST.Task, dW: ST.DW, pool: list[int]) -> tuple[Tensor, Tensor]:
    """Per window (raw gain of its own target, specific gain = raw minus the mean gain of the pool's other values)
    (E58 amendment 2026-10-30, after job 46259: the cipher fine-tune lifts the whole value pool)."""
    raw = gain(net, t, dW)
    return raw, raw - gain(net, t, dW, others(pool))


def behaviour(net: ST.Net, t: dict[str, ST.Task], dW: ST.DW, th: dict[str, Any]) -> dict[str, Any]:
    """The label on specific gains: G_spec on the trained keys' held-out windows, H_spec on the held-out keys' windows,
    x = H_spec / G_spec, its row-bootstrap s (scored) and key-bootstrap s_key (reported); raw G, H, x reported beside, and
    H_other (the other held-out values' gain at the held-out windows) and H_spec per held-out key."""
    pool, new, tr = pool_of(t), t["new"], t["trained"]
    dG, dGs = specific(net, tr, dW, pool)
    dH, dHs = specific(net, new, dW, pool)
    dO = gain(net, new, dW, others([int(new.table[k]) for k in new.keys]))
    G, H, Gs, Hs = (float(d.mean()) for d in (dG, dH, dGs, dHs))
    s = OP.boot_sd(dHs, dGs, th["n_boot"], th["boot_seed"])
    at = new.toks[:, new.half]
    s_key = OP.boot_sd_keys(dHs, at, dGs, tr.toks[:, tr.half], th["n_boot"], th["boot_seed"])
    return {"G_spec": Gs, "H_spec": Hs, "x": Hs / Gs if Gs != 0 else math.nan, "s": s, "s_key": s_key, "label": OP.label(Hs, Gs, s, th),
            "G": G, "H": H, "x_raw": H / G if G != 0 else math.nan, "H_other": float(dO.mean()), "n_G": len(dG), "n_H": len(dH),
            "H_spec_per_key": {str(k): float(dHs[at == k].mean()) for k in new.keys}}


def say_label(n: str, b: dict[str, Any]) -> None:
    KD.log(f"[label {n}] G_spec {b['G_spec']:.4f} H_spec {b['H_spec']:.4f} x {b['x']:.4f} 3s {3 * b['s']:.4f} -> {b['label']} | s_key {b['s_key']:.4f} "
           f"raw G {b['G']:.4f} H {b['H']:.4f} x_raw {b['x_raw']:.4f} H_other {b['H_other']:.4f} (n_G {b['n_G']}, n_H {b['n_H']}) "
           f"per key {json.dumps({k: round(v, 3) for k, v in b['H_spec_per_key'].items()})}")


def vpd_read(ctx: KD.Ctx, t: ST.Task, dW: ST.DW, base: dict[str, Any]) -> dict[str, Any]:
    """E54's `readings` (S, r and the route cells), plus the stand-in ratio from the same cells: (L11 - L00) / (U1 - U0),
    nan when the subcomponent gain U1 - U0 is under E53's floor."""
    g = ST.gains(ctx.net, t, dW)
    rd = KD.readings(ctx, t, dW, g, base)
    c, floor = rd["cells"], ctx.P["thresholds"]["ratio_floor"]
    piece = c["U1"] - c["U0"]
    return {"r": rd["r"], "S": rd["S"], "layer": rd["layer"], "S_block0": rd["S_block0"], "G_score_logP": g["G_score"], "G_logodds_route": rd["G_logodds"],
            "cells": c, "piece_gain": piece, "ratio": (c["L11"] - c["L00"]) / piece if piece >= floor else math.nan}
