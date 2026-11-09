"""E137's stage 1: the exact readings at each pair's known witness and the first wording's equalities."""

from __future__ import annotations

import json
from typing import Any

import pieces as PC
import rules as RU
import search as SE
import torch
from common import dump, f_edit_of, load, log
from torch import Tensor

from commute.hidden_maps import chunks, merged, run, tv, write_last
from commute.steering import Net

PAIRS, SITES = PC.PAIRS, SE.SITES


def mean_f(net: Net, windows: Tensor, p: PC.Pieces, cfg: PC.Config, pair: str) -> float:
    if pair == "XOR":
        return float(PC.xor_f(p, windows).mean())
    tot = 0.0
    for t in chunks(windows, cfg.read_batch):
        tot += float(PC.cue_f(p, run(net, t, sites=("in1",))[1]["in1"][:, -1]).sum())
    return tot / windows.shape[0]


def witness_read(net: Net, pair: str, p: PC.Pieces, read: Tensor, gap: float, cfg: PC.Config) -> dict[str, Any]:
    """The named (phi, s) of each pair, read as in appendix D. Always on: phi = Q, s = v + b, closes everything. Cue: phi = Q; G's s is
    v + mean(f) b (the best a single shift can do, below 1) and H's is v + f(z) b (1). XOR: phi = Q and s = v + mean(f) b for G; there is no affine H
    witness. Table: phi = the whole residual, H's s = h + C z0 + c (1); G is not applicable at k = 4."""
    fedit, bv = f_edit_of(net, p, pair), p.b
    if pair == "Table":
        Q: Tensor | None = None
        s_g = None
        s_h = lambda v, seen: v + seen["in0"][:, -1] @ p.C.T + p.c
    else:
        Q = p.Q
        fbar = 1.0 if pair == "Always on" else mean_f(net, read, p, cfg, pair)
        s_g = lambda v, _seen: v + fbar * bv
        if pair == "Always on":
            s_h = s_g
        elif pair == "Cue":
            s_h = lambda v, seen: v + PC.cue_f(p, seen["in1"][:, -1])[:, None] * bv
        else:
            s_h = None
    g = SE.read_pair(net, read, "in2", Q, s_g, fedit, cfg) if s_g is not None else None
    h = SE.read_pair(net, read, "in2", Q, s_h, fedit, cfg) if s_h is not None else None
    use = h if h is not None else g
    assert use is not None
    return {"G": 1 - g["d"] / gap if g else None, "H": 1 - h["d"] / gap if h else None, "reversion": 1 - use["d_rev"] / gap,
            "mismatch": use["mismatch"], "v_var": use["v_var"], "phi": "Q" if Q is not None else "whole residual"}


def first_wording(net: Net, pair: str, p: PC.Pieces, E: Tensor, read: Tensor, tol_tv: float, cfg: PC.Config) -> dict[str, Any]:
    """The first wording's exact checks at phi = Q (the only named read-out of dimension 4), as equalities to the float tolerance (per-window max
    distance): is phi a variable of M on E, F[phi<-v](x) = M[phi<-v](x) at other inputs' values and random values, the two consequences, and "does
    not count" (M[phi<-v](x) the same for every x at a fixed v). Table's phi is Q too; its witness of `rewired` needs no search."""
    Q, fedit = p.Q, f_edit_of(net, p, pair)
    gen = torch.Generator().manual_seed(137_700)

    def write(values: Tensor) -> Any:
        return {"in2": write_last(Q, values)}

    def vals(toks: Tensor) -> Tensor:
        return run(net, toks, sites=("in2",))[1]["in2"][:, -1] @ Q

    def perm_of(n: int) -> Tensor:
        return torch.randperm(n, generator=gen)

    D = read[: cfg.first_n]
    perm_e = perm_of(E.shape[0])
    m_e = torch.cat([run(net, t)[0][:, -1] for t in chunks(E, cfg.read_batch)])
    v_e = torch.cat([vals(t) for t in chunks(E, cfg.read_batch)])
    swapped = torch.cat([run(net, E[i : i + cfg.read_batch], edit=write(v_e[perm_e][i : i + cfg.read_batch]))[0][:, -1]
                         for i in range(0, E.shape[0], cfg.read_batch)])
    variable = float(tv(m_e, swapped).max())

    v_d = torch.cat([vals(t) for t in chunks(D, cfg.read_batch)])
    v_rand = v_d.mean(0) + v_d.std(0) * torch.randn(v_d.shape, generator=gen).to(v_d.device, v_d.dtype)
    worst: dict[str, float] = {}
    for label, v in (("other_inputs", v_d[perm_of(D.shape[0])]), ("random_values", v_rand)):
        d_max = 0.0
        for i in range(0, D.shape[0], cfg.read_batch):
            t, vi = D[i : i + cfg.read_batch], v[i : i + cfg.read_batch]
            lm = run(net, t, edit=write(vi))[0][:, -1]
            lf = run(net, t, edit=_after(fedit(t), Q, vi))[0][:, -1]
            d_max = max(d_max, float(tv(lm, lf).max()))
        worst[label] = d_max

    c1, c2 = 0.0, 0.0
    vary: list[Tensor] = []
    for i in range(0, D.shape[0], cfg.read_batch):
        t = D[i : i + cfg.read_batch]
        lm, seen_m = run(net, t, sites=("in2",))
        lf, seen_f = run(net, t, sites=("in2",), edit=fedit(t))
        vm, vf = seen_m["in2"][:, -1] @ Q, seen_f["in2"][:, -1] @ Q
        c1 = max(c1, float(tv(run(net, t, edit=write(vf))[0][:, -1], lf[:, -1]).max()))
        c2 = max(c2, float(tv(run(net, t, edit=_after(fedit(t), Q, vm))[0][:, -1], lm[:, -1]).max()))
        vary.append(run(net, t, edit=write(v_d[:1].expand(t.shape[0], -1)))[0][:, -1])
    fixed = torch.cat(vary)
    does_not_count = float(tv(fixed, fixed[:1].expand(fixed.shape[0], -1)).max())
    ok = lambda x: x <= tol_tv
    return {"variable_of_M_max_d": variable, "variable_of_M": variable > tol_tv,
            "reuse_equal_max_d": worst, "reuse_equal": ok(worst["other_inputs"]) and ok(worst["random_values"]),
            "cons_M_vF_max_d": c1, "cons_F_vM_max_d": c2, "cons_equal": ok(c1) and ok(c2),
            "fixed_value_fixes_output_max_d": does_not_count, "does_not_count": ok(does_not_count)}


def _after(e: Any, Q: Tensor, v: Tensor) -> Any:
    return merged(e, {"in2": write_last(Q, v)})


def stage1(net: Net, cfg: PC.Config, sets: PC.Sets, task: PC.Task, p: PC.Pieces, st0: dict[str, Any], filed: dict[str, Any]) -> dict[str, Any]:
    done = load(cfg, "stage1.json")
    if done is not None:
        log("stage 1 read from file")
        return done
    out: dict[str, Any] = {"pairs": {}}
    for pair in PAIRS:
        info = st0["pairs"][pair]
        if not info["reading"]:
            out["pairs"][pair] = {"no_reading": True, "gap": info["gap"]}
            log(f"stage 1 {pair}: no reading")
            continue
        _, read = PC.pair_windows(cfg, pair, sets, task)
        wit = witness_read(net, pair, p, read, info["gap"], cfg)
        first = first_wording(net, pair, p, sets.E, read, st0["tol_tv"], cfg)
        out["pairs"][pair] = {"witness": wit, "first_wording": first, "gap": info["gap"],
                              "tol_share": RU.tol_share(st0["tol_tv"], info["gap"]), "tol_mismatch": RU.tol_mismatch(st0["tol_h"], wit["v_var"])}
        log(f"stage 1 {pair}: {json.dumps(out['pairs'][pair], default=float)}")
    dump(cfg, "stage1.json", out)
    print("STAGE 1 DONE", flush=True)
    return out
