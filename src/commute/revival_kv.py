"""E55 Stage C3: pruning revival, reverse fine-tuning and the possession probe on the 67M key -> value states, and the
rules that place each column's elicit map on the rank-1-to-cipher axis. Plan: reports/plans/2026-10-31_plan_e55-stage-c-
open-pair.md (§§ 3, 5, 6); filed rules: experiments/open_pair_instruments/results/predictions.json. Not
`commute.revival` (C1/C1b, minGPT and PCFG only; its callers are frozen): nothing is copied from it.

The old behaviour is W0's whole next-token distribution at the answer position; kappa(m) = the mean over windows of
KL(p_W0 || p_m) there, in nats, fp32 forwards (float64 sums). D = kappa(state).
- Pruning revival (Jain et al. 2024, Alg. 3): unit j of block l (`steering.Net` site `mlp<l>`, flat id l * 3072 + j)
  scored by grad . w on its input row (and bias, if any) of the fit-half mean kappa at the state; the top K zeroed at
  every position (equal to zeroing the row, as GELU(0) = 0) against n_random random K-unit removals.
- Reverse fine-tuning (`finetune.finetune_to_reference`): KL to W0 at every position of the train windows, from W0 +
  delta-W; kappa on the score half after every step; t_half = the interpolated step at which kappa first reaches D / 2.
- The probe (reference): W0's logit lens per block on the map's own value against its other values.
- The planted wrapper (the instruments' positive control): 13 block units of W0 whose down_proj columns push away from
  the rank-1 values, scaled by bisection.
The rules at the bottom are pure arithmetic, CPU-tested in tests/test_revival_kv.py.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from commute.steering import D_MLP, DW, N_LAYERS, Edit, Net
from commute.vpd import module_keys, target_linear

POLES = ("rank1", "cipher")
SCORED_MAPS = ("rank1", "cipher", "elicit")  # the maps p is built from; copy is reported beside
MAPS = ("rank1", "cipher", "elicit", "copy")


# ----------------------------------------------------------------------------- kappa


@dataclass
class Ref:
    """The windows every state is read on, with W0's log-softmax at the answer position (on the device)."""

    fit: Tensor  # (Nf, L) tokens, even held-out indices
    score: Tensor  # (Ns, L) odd
    half: int
    lp0: dict[str, Tensor]  # "fit" / "score" -> (N, V) float64


def answer_lp(net: Net, toks: Tensor, half: int, dW: DW | None = None, edit: Edit | None = None) -> Tensor:
    with net.at(dW):
        return torch.log_softmax(net.run(toks, edit)[0][:, half].double(), -1)


def make_ref(net: Net, heldout: Tensor, half: int) -> Ref:
    fit, score = heldout[0::2], heldout[1::2]
    return Ref(fit, score, half, {"fit": answer_lp(net, fit, half), "score": answer_lp(net, score, half)})


def kl_rows(lp0: Tensor, lp: Tensor) -> Tensor:
    return (lp0.exp() * (lp0 - lp)).sum(-1)


def kappa(net: Net, ref: Ref, dW: DW | None = None, edit: Edit | None = None, part: str = "score") -> float:
    toks = ref.fit if part == "fit" else ref.score
    return float(kl_rows(ref.lp0[part], answer_lp(net, toks, ref.half, dW, edit)).mean())


@torch.no_grad()
def off_answer_kl(net: Net, ref: Ref, dW: DW) -> float:
    """The mean KL(W0 || state) over the score half's positions other than the answer position."""
    l0 = torch.log_softmax(net.run(ref.score)[0].double(), -1)
    with net.at(dW):
        l1 = torch.log_softmax(net.run(ref.score)[0].double(), -1)
    kl = kl_rows(l0, l1)
    keep = torch.ones(kl.shape[1], dtype=torch.bool, device=kl.device)
    keep[ref.half] = False
    return float(kl[:, keep].mean())


# ----------------------------------------------------------------------------- pruning


def up_key(model: Any, layer: int) -> str:
    """The MLP's input matrix of block `layer` (the decomposed MLP matrix that is not down_proj)."""
    ks = [k for k in module_keys(model) if k.startswith(f"h.{layer}.mlp.") and not k.endswith("down_proj")]
    assert len(ks) == 1, ks
    return ks[0]


def zero_units(units: Sequence[int]) -> Edit:
    """Zero the flat MLP units at every position (site mlp<l>)."""
    idx, edit = torch.tensor(sorted(units), dtype=torch.long), {}
    for i in range(N_LAYERS):
        sel = idx[(idx >= i * D_MLP) & (idx < (i + 1) * D_MLP)] - i * D_MLP
        if len(sel):
            edit[f"mlp{i}"] = lambda x, sel=sel: x.index_fill(2, sel.to(x.device), 0.0)
    return edit


def unit_scores(net: Net, ref: Ref, dW: DW | None) -> Tensor:
    """Alg. 3's grad . w per MLP unit, flat (4 * 3072,): the gradient of the fit-half mean kappa at the state, summed
    over the unit's row of the MLP input matrix (and its bias, if any). Removing a unit with a large score lowers kappa
    most, to first order."""
    lins = [target_linear(net.model, up_key(net.model, i)) for i in range(N_LAYERS)]
    params = [p for lin in lins for p in (lin.weight, lin.bias) if p is not None]
    with net.at(dW):
        try:
            for p in params:
                p.requires_grad_(True)
                p.grad = None
            out = net.net(ref.fit.to(net.device))
            logits = (out[0] if isinstance(out, tuple) else out)[:, ref.half]
            kl_rows(ref.lp0["fit"], torch.log_softmax(logits.double(), -1)).mean().backward()
            s = [(lin.weight.grad * lin.weight).sum(1) + (lin.bias.grad * lin.bias if lin.bias is not None else 0.0) for lin in lins]  # type: ignore[operator]
        finally:
            for p in params:
                p.requires_grad_(False)
                p.grad = None
    return torch.cat(s).detach().double().cpu()


def prune_readings(net: Net, ref: Ref, dW: DW | None, ks: Sequence[int], n_random: int, seed: int,
                   scores: Tensor | None) -> dict[str, Any]:  # fmt: skip
    """kappa of the state, and per K: kappa after each of n_random uniform K-unit removals (`default_rng([seed, K])`,
    the same draws for every state) and, when `scores` is given, after removing the top K (ids saved)."""
    out: dict[str, Any] = {"kappa": kappa(net, ref, dW)}
    n = N_LAYERS * D_MLP
    for k in ks:
        rng = np.random.default_rng([seed, k])
        rec: dict[str, Any] = {"random": [kappa(net, ref, dW, zero_units(rng.choice(n, k, replace=False).tolist())) for _ in range(n_random)]}
        if scores is not None:
            top = torch.topk(scores, k).indices.tolist()
            rec |= {"units": top, "top": kappa(net, ref, dW, zero_units(top))}
        out[str(k)] = rec
    return out


# ----------------------------------------------------------------------------- reverse fine-tuning


def reverse_curve(net: Net, ref: Ref, dW: DW, w0_net: nn.Module, train: Tensor, cfg: Any,
                  deadline: float | None = None) -> dict[str, Any]:  # fmt: skip
    """kappa after every step of `finetune_to_reference` from W0 + delta-W, until it reaches half its start or the cap.
    `end` is "half", "cap" or "deadline" (a deadline exit is not a reading). The model is W0 again afterwards."""
    from commute.data import Windows
    from commute.finetune import StopTraining, finetune_to_reference

    curve, end = [], "cap"

    def on_step(step: int, _params: Any) -> None:
        nonlocal end
        curve.append(kappa(net, ref))
        if curve[-1] <= curve[0] / 2:
            end = "half"
            raise StopTraining
        if deadline is not None and time.monotonic() > deadline:
            end = "deadline"
            raise StopTraining

    n = len(train)
    w = Windows(train, torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), ref.half)
    with net.at(dW):
        curve.append(kappa(net, ref))
        if curve[0] > 0:
            finetune_to_reference(net.model, w0_net, w, cfg, net.device, on_step, log=lambda _m: None)
    return {"curve": curve, "end": end, "t_half": half_life(curve) if end != "deadline" else None}


def w0_copy(net: Net) -> nn.Module:
    """A frozen copy of the target model at W0, the reverse fine-tune's reference."""
    return copy.deepcopy(net.net).eval().requires_grad_(False)


# ----------------------------------------------------------------------------- the probe (reference) and the prior


@torch.no_grad()
def lens_q(net: Net, toks: Tensor, half: int, own: Tensor, values: Sequence[int]) -> dict[str, Any]:
    """Per block l = 1-4, the mean over windows of [log P_lens,l(own value) - mean over the other values of log
    P_lens,l(value)], P_lens,l = W0's final norm and unembedding on the residual after block l; q = the max over
    blocks. `lens_check` = max |block-4 lens - W0's log-softmax| at the answer position."""
    logits, c = net.run(toks, capture=True)
    vals = torch.tensor(list(values))
    keep = (vals[None, :] != own.cpu()[:, None]).double()
    res = [c[f"in{i}"] for i in range(1, N_LAYERS)] + [c[f"out{N_LAYERS - 1}"]]
    lps = [torch.log_softmax(net.net.lm_head(net.net.ln_f(r[:, half])).double(), -1).cpu() for r in res]
    per = [float((lp.gather(1, own.cpu()[:, None]).squeeze(1) - (lp[:, vals] * keep).sum(1) / keep.sum(1)).mean()) for lp in lps]
    check = float((lps[-1] - torch.log_softmax(logits[:, half].double(), -1).cpu()).abs().max())
    return {"per_block": per, "q": max(per), "lens_check": check}


# ----------------------------------------------------------------------------- the planted wrapper


def planted_units(net: Net, ref: Ref, block: int, n: int) -> dict[str, Any]:
    """Among block `block`'s units whose mean GELU activation at the answer position (fit half, W0) is above the
    block's median and above 0 (the edit divides by it), the n whose single zeroing on W0 costs least (fit-half
    kappa; ties to the lower id). Flat ids, their mean activations and costs."""
    a = net.run(ref.fit, capture=True)[1][f"mlp{block}"][:, ref.half].double().mean(0).cpu()
    cand = [j for j in range(D_MLP) if a[j] > a.median() and a[j] > 0]
    cost = {j: kappa(net, ref, edit=zero_units([block * D_MLP + j]), part="fit") for j in cand}
    pick = sorted(cand, key=lambda j: (cost[j], j))[:n]
    return {"units": [block * D_MLP + j for j in pick], "abar": [float(a[j]) for j in pick], "cost": [cost[j] for j in pick],
            "n_candidates": len(cand)}


def rank1_direction(net: Net, fit_values: Tensor) -> Tensor:
    """The unit vector of the mean over fit-half windows of (u_value - u_mean), u = the unembedding rows."""
    u = net.net.lm_head.weight.detach().double()
    w = (u[fit_values.to(u.device)] - u.mean(0)).mean(0)
    return (w / w.norm()).float()


def planted_dw(net: Net, units: Sequence[int], abar: Sequence[float], w: Tensor, alpha: float) -> DW:
    """Zero everywhere except block's down_proj: column j += -alpha * w / abar_j."""
    dW = {k: torch.zeros_like(v) for k, v in net.W0.items()}
    for u, a in zip(units, abar, strict=True):
        layer, j = divmod(u, D_MLP)
        dW[f"h.{layer}.mlp.down_proj"][:, j] -= alpha * w.to(net.W0[f"h.{layer}.mlp.down_proj"].device) / a
    return dW


def fit_alpha(kap: Callable[[float], float], off: Callable[[float], float], target: float, bound: float,
              start: float = 1.0, iters: int = 40, max_doublings: int = 60) -> dict[str, Any]:  # fmt: skip
    """alpha by doubling then bisection, assuming kap and off rise with alpha: the smallest alpha with kap >= target if
    off <= bound there; else ("fallback") the largest alpha found with off <= bound, where kap < target."""
    lo, hi = 0.0, start
    for _ in range(max_doublings):
        if kap(hi) >= target or off(hi) > bound:
            break
        lo, hi = hi, 2 * hi
    else:
        return {"alpha": lo, "kappa": kap(lo), "off": off(lo), "fallback": True}
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kap(mid) < target and off(mid) <= bound else (lo, mid)
    a = hi if off(hi) <= bound else lo
    k = kap(a)
    return {"alpha": a, "kappa": k, "off": off(a), "fallback": k < target}


# ----------------------------------------------------------------------------- per-state arithmetic (pure)


def mlp_share(dW: Mapping[str, Tensor]) -> float:
    """(sum of ||delta-W||^2 over the MLP matrices) / (sum over all 24)."""
    sq = {k: float(v.double().pow(2).sum()) for k, v in dW.items()}
    tot = sum(sq.values())
    return sum(v for k, v in sq.items() if ".mlp." in k) / tot if tot > 0 else math.nan


def sd(v: Sequence[float]) -> float:
    """Sample standard deviation (n - 1); nan under 2 values."""
    return float(np.std(v, ddof=1)) if len(v) >= 2 else math.nan


def revival(rec: Mapping[str, Any], ks: Sequence[int], D: float) -> dict[str, Any]:
    """Per K: Delta_K = (kappa - kappa_top) - mean(kappa - kappa_random), and the random spread sd((kappa -
    kappa_random) / D); R = max over K of Delta_K / D (nan when D <= 0 or a top reading is missing)."""
    k0, delta, spread = rec["kappa"], {}, {}
    for k in ks:
        r = rec[str(k)]
        drop = [k0 - x for x in r["random"]]
        spread[str(k)] = sd([d / D for d in drop]) if D > 0 else math.nan
        delta[str(k)] = (k0 - r["top"]) - float(np.mean(drop)) if "top" in r else math.nan
    R = max(delta[str(k)] / D for k in ks) if D > 0 and all(map(math.isfinite, delta.values())) else math.nan
    return {"Delta": delta, "spread": spread, "R": R}


def half_life(curve: Sequence[float]) -> float | None:
    """The step at which kappa first falls to curve[0] / 2, linear between the two steps that straddle it (curve[i]
    is kappa after step i); None if it never does (censored) or curve[0] <= 0."""
    if not curve or curve[0] <= 0:
        return None
    t = curve[0] / 2
    for i in range(1, len(curve)):
        if curve[i] <= t:
            a, b = curve[i - 1], curve[i]
            return i - 1 + (a - t) / (a - b) if a != b else float(i)
    return None


def planted_pass(top13: Sequence[int], planted: Sequence[int], R: float, ceiling: float, found_min: int, frac: float) -> dict[str, Any]:
    found = len(set(top13) & set(planted))
    return {"found": found, "R": R, "R_min": frac * ceiling, "pass": found >= found_min and math.isfinite(R) and R >= frac * ceiling}


# ----------------------------------------------------------------------------- the rules (§ 6), pure

Column = Mapping[str, Sequence[float | None]]


def clean(v: Sequence[float | None]) -> list[float]:
    return [float(x) for x in v if x is not None and math.isfinite(x)]


def gap_status(col: Column, direction: int | None, floor: float, mult: float) -> dict[str, Any]:
    """Rule 1. noise = max(the within-map seed sd over rank-1, cipher and elicit, floor); gap = mean rank-1 - mean
    cipher; "ok" iff d * gap >= mult * noise and gap != 0, "reversed" iff -d * gap >= mult * noise and gap != 0, else
    "no gap"; direction None = undirected (|gap|). "void" if one of the three maps has fewer than 2 values."""
    v = {m: clean(col.get(m, [])) for m in MAPS}
    if any(len(v[m]) < 2 for m in SCORED_MAPS):
        return {"status": "void", "gap": math.nan, "noise": math.nan, "n": {m: len(v[m]) for m in MAPS}}
    noise = max(*(sd(v[m]) for m in SCORED_MAPS), floor)
    gap = float(np.mean(v["rank1"]) - np.mean(v["cipher"]))
    bar = mult * noise
    if gap == 0:
        status = "no gap"
    elif direction is None:
        status = "ok" if abs(gap) >= bar else "no gap"
    else:
        status = "ok" if direction * gap >= bar else "reversed" if -direction * gap >= bar else "no gap"
    return {"status": status, "gap": gap, "noise": noise, "n": {m: len(v[m]) for m in MAPS}}


def position(means: Mapping[str, float], of: str = "elicit") -> float:
    """(mean_of - mean_cipher) / (mean_rank1 - mean_cipher); nan on a zero denominator or a missing map."""
    if of not in means:
        return math.nan
    den = means["rank1"] - means["cipher"]
    return (means[of] - means["cipher"]) / den if den != 0 else math.nan


def side(p: float, vpd_min: float, label_max: float) -> str | None:
    """Rule 3: "VPD" iff p >= vpd_min (p > 1 included), "label" iff p <= label_max (p < 0 included), else "between";
    ties go to the outer band; None for nan."""
    if not math.isfinite(p):
        return None
    return "VPD" if p >= vpd_min else "label" if p <= label_max else "between"


def bootstrap_p(col: Column, draws: int, seed: int) -> np.ndarray:
    """Rule 4's draws: per draw, each of rank-1, cipher and elicit resampled over its seeds with replacement; p (nan on
    a zero denominator)."""
    rng = np.random.default_rng(seed)
    v = {m: np.asarray(clean(col[m])) for m in SCORED_MAPS}
    means = {m: v[m][rng.integers(0, len(v[m]), (draws, len(v[m])))].mean(1) for m in SCORED_MAPS}
    den = means["rank1"] - means["cipher"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, (means["elicit"] - means["cipher"]) / np.where(den != 0, den, 1.0), np.nan)


def percentiles(boot: np.ndarray, central: float) -> tuple[float, float]:
    """The (1 - central) / 2 and (1 + central) / 2 percentiles, without interpolation (lower and higher), with every
    nan draw (zero denominator) placed outside every band: at -inf for the lower, +inf for the upper."""
    q = (1 - central) / 2
    lo = float(np.quantile(np.where(np.isnan(boot), -np.inf, boot), q, method="lower"))
    hi = float(np.quantile(np.where(np.isnan(boot), np.inf, boot), 1 - q, method="higher"))
    return lo, hi


def firm(point: str | None, lo: float, hi: float, vpd_min: float, label_max: float) -> str | None:
    """Rule 4: "VPD" holds iff lo >= vpd_min, "label" iff hi <= label_max; otherwise "between"."""
    if point == "VPD":
        return "VPD" if lo >= vpd_min else "between"
    if point == "label":
        return "label" if hi <= label_max else "between"
    return point


def column_reading(col: Column, direction: int | None, floor: float, C: Mapping[str, Any]) -> dict[str, Any]:
    """Rules 1, 3 and 4 on one column: gap, p, p_copy, the bootstrap's percentiles and the firm side (None unless the
    gap is "ok")."""
    rec = gap_status(col, direction, floor, C["gap_mult"])
    means = {m: float(np.mean(clean(col[m]))) for m in MAPS if len(clean(col.get(m, []))) >= 1}
    rec |= {"means": means, "p": math.nan, "p_copy": math.nan, "p_lo": math.nan, "p_hi": math.nan, "side": None}
    if rec["status"] == "void":
        return rec
    b, B = C["bands"], C["bootstrap"]
    rec["p"], rec["p_copy"] = position(means), position(means, "copy") if len(clean(col.get("copy", []))) >= 2 else math.nan
    rec["p_lo"], rec["p_hi"] = percentiles(bootstrap_p(col, B["draws"], B["rng_seed"]), B["central"])
    if rec["status"] == "ok":
        rec["side"] = firm(side(rec["p"], b["vpd_min"], b["label_max"]), rec["p_lo"], rec["p_hi"], b["vpd_min"], b["label_max"])
    return rec


def tracks(rd: Mapping[str, Any], base: Mapping[str, Any], tol: float) -> bool:
    """Rules 5 and 6: fires iff both gaps are "ok" and the firm sides agree or |p - p_base| <= tol."""
    if rd["status"] != "ok" or base["status"] != "ok":
        return False
    return rd["side"] == base["side"] or abs(rd["p"] - base["p"]) <= tol


def verdict(rd: Mapping[str, Any], dist: Mapping[str, Any], prior: Mapping[str, Any], C: Mapping[str, Any]) -> str:
    """Rule 7, per instrument: void / censored / no gap / reversed; else the firm side, replaced by "distance",
    "prior" or "distance and prior" when rules 5 or 6 fire."""
    if rd["status"] != "ok":
        return rd["status"]
    d, p = tracks(rd, dist, C["distance_tol"]), tracks(rd, prior, C["prior_tol"])
    return "distance and prior" if d and p else "distance" if d else "prior" if p else str(rd["side"])


def joint_verdict(v: Mapping[str, str]) -> str:
    """Rule 7, joint, over two instruments: "shared" iff both VPD; "shared (by X)" iff X reads VPD and the other no gap,
    void or censored; "VPD alone" / "VPD alone (by X)" likewise with label; "conflict" iff one reads VPD and the other
    label or reversed; else "not settled", naming each."""
    (a, va), (b, vb) = sorted(v.items())
    quiet = {"no gap", "void", "censored"}
    for side_, name in (("VPD", "shared"), ("label", "VPD alone")):
        if va == vb == side_:
            return name
        for x, vx, vy in ((a, va, vb), (b, vb, va)):
            if vx == side_ and vy in quiet:
                return f"{name} (by {x})"
    if "VPD" in (va, vb) and {va, vb} & {"label", "reversed"}:
        return "conflict"
    return f"not settled ({a}: {va}; {b}: {vb})"


def t_half_column(per_map: Mapping[str, Sequence[float | str | None]]) -> tuple[dict[str, list[float | None]], bool]:
    """Rule 2: "censored" entries dropped; censored is True iff a pole seed is censored (the column is then void)."""
    cens = any(x == "censored" for m in POLES for x in per_map.get(m, []))
    col: dict[str, list[float | None]] = {m: [None if isinstance(x, str) or x is None else float(x) for x in v] for m, v in per_map.items()}
    return col, cens


DIRECTIONS: dict[str, int | None] = {"R": 1, "t_half": -1, "q": 1, "r": 1, "ratio": 1, "S": 1, "x": 1,
                                     "norm": None, "D": None, "mlp_share": None, "prior": None}  # fmt: skip
INSTRUMENTS = {"pruning": "R", "reverse": "t_half"}
CHECKED = ("R", "t_half", "r", "ratio")  # put through rules 5 and 6


def score_columns(cols: Mapping[str, Column], floors: Mapping[str, float], void: Mapping[str, str],
                  C: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Rules 1-7 on every column: each column's reading (`void` overrides a status: a failed planted control or lens
    check, or "censored"), the verdicts of R, t_half, r and the stand-in ratio, and the joint verdict."""
    rd = {k: column_reading(c, DIRECTIONS[k], floors.get(k, 0.0), C) for k, c in cols.items()}
    for k, why in void.items():
        rd[k] = {**rd[k], "status": why, "side": None}
    v = {k: verdict(rd[k], rd["D"], rd["prior"], C) for k in CHECKED if k in rd}
    return {"readings": rd, "verdicts": v, "joint": joint_verdict({i: v[k] for i, k in INSTRUMENTS.items()})}


def probe_reference(q: Mapping[str, float]) -> dict[str, Any]:
    """The filed reference (not scored): cipher q within +-0.5 nats, elicit q >= 1, rank-1 q >= elicit; p_q beside."""
    checks = {"cipher_null": abs(q["cipher"]) <= 0.5, "elicit_ge_1": q["elicit"] >= 1, "rank1_ge_elicit": q["rank1"] >= q["elicit"]}
    return {**checks, "all": all(checks.values()), "p_q": position(q), "p_q_copy": position(q, "copy")}
