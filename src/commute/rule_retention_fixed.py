"""Causal-rule retention under the fixed choices (plan `reports/plans/2026-11-17_plan_rule-retention-fixed-choices.md`,
"Definitions in code"; the definition is `~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`,
"Choices fixed for the test", tested, not adopted). Added 2026-11-17 for `experiments/rule_retention_fixed` (E195). It
builds on `commute.rule_retention` (P1.24), which it calls and does not change. Run 2 of P1.25
(`experiments/rule_retention_pairs`) also calls it, reading the contrast's KL at the last position only (`at`, added
then; its default, every position, is what run 1 ran).

A unit is an MLP neuron (level A) or a head after o_proj or a whole MLP block (level B, the side reading). Its output on
M's inputs to its layer: neuron i, f_i(x) = a_i(x) w_i (activation times its output column). For F's unit i and M's
unit j, summed over every window, position and residual dimension:
    cross_ij = <f^F_i, f^M_j>   (neurons: G_ij (w^F_i . w^M_j), G_ij = sum_x a^F_i(x) a^M_j(x))
    cos2_ij  = cross_ij^2 / (|f^F_i|^2 |f^M_j|^2)
Per layer and kind, `linear_sum_assignment` pairs F's non-zero units one-to-one with M's units for the greatest total
cos2 (a block pairs with itself, as a layer has one). Under the pairing: r = cos2, gain g = cross / |f^M|^2 (signed),
`same` = 1 - |f^F - f^M|^2 / (|f^F|^2 + |f^M|^2). A unit of F that writes zero has no r and is in no set; a non-zero unit
left without a partner has r = 0. A unit is changed when it is unpaired or `same` < 1 - tol (a gain change counts).

The hybrid (`forward`): F with the units in `revert` put back to their partner's M rule in F's slot. A reverted neuron
uses its partner's M input row and output column (M's norm) on the hybrid's own residual; a reverted head is the partner
head of M (its rms_1, attention and o_proj columns); a reverted block is M's block. A reverted unpaired unit is
removed. Zero units of F stay zero.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from commute import rule_retention as R

Unit = R.Unit  # ("neuron", layer, i) with i >= 3072 an appended neuron; ("head", layer, h); ("mlp", layer, 0)
KINDS = {"A": ("neuron",), "B": ("head", "mlp")}


@dataclass
class Pairing:
    """One kind of unit in one layer; index = F's unit."""

    partner: Tensor  # long, M's unit or -1 (zero or unpaired)
    zero: Tensor  # bool, F's unit writes zero on every window
    r: Tensor  # double, cos2 with the partner; 0 unpaired; nan zero
    gain: Tensor  # double, cross / |f^M|^2; nan when no partner
    same: Tensor  # double, 1 - |f^F - f^M|^2 / (|f^F|^2 + |f^M|^2); nan when no partner


def pair(cross: Tensor, nF: Tensor, nM: Tensor) -> Pairing:
    """The re-pairing of one layer's units of one kind, from the cross sums (nF x nM) and the squared norms."""
    cross, nF, nM = cross.double().cpu(), nF.double().cpu(), nM.double().cpu()
    den = torch.outer(nF, nM)
    cos2 = torch.where(den > 0, cross.pow(2) / den.clamp(min=1e-300), torch.zeros_like(den))
    nz = (nF > 0).nonzero().squeeze(1)
    rows, cols = linear_sum_assignment(cos2[nz].numpy(), maximize=True)
    partner = torch.full((len(nF),), -1, dtype=torch.long)
    partner[nz[torch.as_tensor(rows)]] = torch.as_tensor(cols, dtype=torch.long)
    nan = torch.full((len(nF),), float("nan"), dtype=torch.float64)
    r, gain, same = torch.where(nF > 0, 0.0, nan), nan.clone(), nan.clone()
    i = (partner >= 0).nonzero().squeeze(1)
    j = partner[i]
    c, a, b = cross[i, j], nF[i], nM[j]
    r[i], gain[i] = cos2[i, j], torch.where(b > 0, c / b.clamp(min=1e-300), nan[i])
    same[i] = 1 - ((a - 2 * c + b).clamp(min=0) / (a + b)).clamp(max=1)
    return Pairing(partner, nF <= 0, r, gain, same)


Profile = dict[tuple[str, int], Pairing]


@torch.no_grad()
def profile(F: R.Model, M: R.Model, batches: Iterable[Tensor]) -> Profile:
    """Every unit of F at both levels, paired with M's, read on M's inputs over every window and position of `batches`."""
    s: dict[tuple[str, int], Tensor] = {}
    for toks in batches:
        for l, (x, y) in enumerate(R.m_inputs(M, toks)):
            aF, aM = R.acts(F, l, y).double().flatten(0, 1), R.acts(M, l, y).double().flatten(0, 1)
            hF, hM = R.heads(F.net.h[l], x).double(), R.heads(M.net.h[l], x).double()
            oF, oM = R.block(F, l, y).double(), R.block(M, l, y).double()
            for k, v in {"G": aF.T @ aM, "aFF": aF.pow(2).sum(0), "aMM": aM.pow(2).sum(0),
                         "hx": torch.einsum("bthd,btkd->hk", hF, hM), "hF": hF.pow(2).sum((0, 1, 3)), "hM": hM.pow(2).sum((0, 1, 3)),
                         "bx": (oF * oM).sum().view(1, 1), "bF": oF.pow(2).sum().view(1), "bM": oM.pow(2).sum().view(1)}.items():  # fmt: skip
                s[k, l] = s[k, l] + v if (k, l) in s else v
    out: Profile = {}
    for l in range(len(M.net.h)):
        WF, WM = R.cols(F, l).double(), R.cols(M, l).double()
        out["neuron", l] = pair(s["G", l] * (WF.T @ WM), s["aFF", l] * WF.pow(2).sum(0), s["aMM", l] * WM.pow(2).sum(0))
        out["head", l] = pair(s["hx", l], s["hF", l], s["hM", l])
        out["mlp", l] = pair(s["bx", l], s["bF", l], s["bM", l])
    return out


def units(prof: Profile, level: str) -> list[Unit]:
    """F's non-zero units at a level."""
    return [(k, l, i) for (k, l), p in prof.items() if k in KINDS[level] for i in (~p.zero).nonzero().squeeze(1).tolist()]


def changed(prof: Profile, level: str, tol: float) -> list[Unit]:
    """Non-zero units of F that are unpaired or whose output differs from their partner's: same < 1 - tol."""
    return [u for u in units(prof, level) if get(prof, u, "partner") < 0 or not get(prof, u, "same") >= 1 - tol]


def get(prof: Profile, u: Unit, what: str) -> float:
    return float(getattr(prof[u[0], u[1]], what)[u[2]])


# ---- the hybrid -----------------------------------------------------------------------------------------------------


def _attn(F: R.Model, M: R.Model, l: int, x: Tensor, partner: Tensor, rev: list[int]) -> Tensor:
    bF = F.net.h[l]
    if not rev:
        return bF.attn(bF.rms_1(x))
    hF, hM = R.heads(bF, x), R.heads(M.net.h[l], x)
    keep = torch.ones(R.N_HEAD, device=x.device, dtype=hF.dtype)
    keep[rev] = 0
    out = (hF * keep[:, None]).sum(2)
    p = [int(partner[i]) for i in rev if partner[i] >= 0]
    return out + hM[:, :, p].sum(2) if p else out


def _mlp(F: R.Model, M: R.Model, l: int, y: Tensor, partner: Tensor, rev: list[int]) -> Tensor:
    aF, WF = R.acts(F, l, y), R.cols(F, l)
    if not rev:
        return aF @ WF.T
    keep = torch.ones(aF.shape[-1], device=y.device, dtype=aF.dtype)
    keep[rev] = 0
    out = (aF * keep) @ WF.T
    p = partner[rev]
    p = p[p >= 0].to(y.device)
    if not len(p):
        return out
    bM = M.net.h[l]
    aM = bM.mlp.gelu(bM.rms_2(y) @ bM.mlp.c_fc.weight[p].T)
    return out + aM @ bM.mlp.down_proj.weight[:, p].T


@torch.no_grad()
def forward(F: R.Model, M: R.Model, toks: Tensor, prof: Profile, revert: Iterable[Unit] = (), at: slice = slice(None)) -> Tensor:
    """F's logits at positions `at` with the units in `revert` put back to their partner's M rule in F's slot."""
    rv: dict[tuple[str, int], list[int]] = {}
    for k, l, i in revert:
        rv.setdefault((k, l), []).append(i)
    x = F.net.wte(toks)
    for l in range(len(F.net.h)):
        x = x + _attn(F, M, l, x, prof["head", l].partner, rv.get(("head", l), []))
        x = x + (R.block(M, l, x) if ("mlp", l) in rv else _mlp(F, M, l, x, prof["neuron", l].partner, rv.get(("neuron", l), [])))
    return F.net.lm_head(F.net.ln_f(x[:, at])).float()


def gate(F: R.Model, M: R.Model, toks: Tensor, prof: Profile) -> float:
    """Max over windows of max |hybrid with nothing put back - native logit| / max |native logit|."""
    native = F.logits(toks)
    return float(((forward(F, M, toks, prof) - native).abs().amax((1, 2)) / native.abs().amax((1, 2))).max())


# ---- retention for the contrast ---------------------------------------------------------------------------------------


def eliminate(closure: Callable[[set[Unit]], float], order: list[Unit], bar: float, chunk: int) -> tuple[list[Unit], int]:
    """Backward elimination along `order` by chunks: try dropping the next `c` units (put back to M's rule); keep the drop
    when the rest still closes `bar`. On a failure halve c; a single unit that fails stays in the set. After a drop c
    doubles again, up to `chunk`. Returns the set left and the number of closure evaluations."""
    S, i, c, n = set(order), 0, chunk, 0
    while i < len(order):
        part = set(order[i : i + c])
        n += 1
        if closure(S - part) >= bar:
            S, i, c = S - part, i + len(part), min(2 * c, chunk)
        elif c > 1:
            c //= 2
        else:
            i += 1
    return [u for u in order if u in S], n


def contrast(F: R.Model, M: R.Model, toks: Tensor, prof: Profile, level: str, th: dict[str, Any], log: Callable[[str], None],
             at: slice = slice(None)) -> dict[str, Any]:
    """The retention for the contrast F against M on `toks` at one level, KL read at positions `at` (every position by
    default, as E195 run 1 read it): per seeded order, the smallest set found, its mean r, the gains on it, and the
    evaluations and seconds the order took."""
    ch = changed(prof, level, th["r_tol"])
    lF, lM = forward(F, M, toks, prof, at=at), M.logits(toks)[:, at]
    kl0 = float(R.kl(lF, lM).mean())
    out: dict[str, Any] = {"kl": kl0, "n_units": len(units(prof, level)), "n_changed": len(ch)}
    if not ch:
        return out | {"reading": "no change"}
    if kl0 <= th["kl_floor"]:
        return out | {"reading": "no contrast"}

    def closure(S: set[Unit]) -> float:
        return 1 - float(R.kl(lF, forward(F, M, toks, prof, [u for u in ch if u not in S], at)).mean()) / kl0

    gen, orders = torch.Generator().manual_seed(th["order_seed"]), []
    for k in range(th["orders"]):
        t0 = time.time()
        S, n = eliminate(closure, [ch[i] for i in torch.randperm(len(ch), generator=gen).tolist()], th["closure_bar"], th["chunk"])
        rec = {"evals": n, "seconds": time.time() - t0, "size": len(S), "closure": closure(set(S)),
               "retention": sum(get(prof, u, "r") for u in S) / len(S) if S else None,
               "units": [{"unit": R.name(u), "r": get(prof, u, "r"), "gain": get(prof, u, "gain")} for u in S]}  # fmt: skip
        g = [x["gain"] for x in rec["units"] if x["gain"] == x["gain"]]  # an unpaired unit has no gain
        rec |= {"gain_min": min(g), "gain_mean": sum(g) / len(g), "gain_max": max(g)} if g else {}
        orders.append(rec)
        log(f"  level {level} order {k}: set of {len(S)}, retention {rec['retention']}, closure {rec['closure']:.4f}, {n} evaluations, {rec['seconds']:.1f} s")
    ret = [o["retention"] for o in orders]
    out |= {"closure_none_kept": closure(set()), "orders": orders, "retention": ret}
    if any(v is None for v in ret):
        return out | {"reading": "no set"}
    return out | {"reading": "read", "min": min(ret), "max": max(ret)}
