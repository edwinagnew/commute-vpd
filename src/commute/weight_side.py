"""E67's weight-side read, do-nothing side: per matrix, on W0's activations a, with C1 = C0, the re-switching term

    R = (g1 - g0) C0 a

and its overlap share with the update's own output change, S_R = E<R, delta-W a> / E||delta-W a||^2. g is Γ's value per position and
subcomponent, on the model it is read from. No dense component matrix: C0 a = (g * (a V)) U, in param-decomp's layout, V (d_in, C), U (C, d_out).
Plan: reports/plans/2026-11-02_plan_e67-weight-side-read.md. Rules: experiments/weight_side_read/results/predictions.json.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from commute import split
from commute.faithfulness import model_state


def dot(x: Tensor, y: Tensor) -> float:
    return float((x * y).sum(dtype=torch.float64))


@dataclass
class Matrix:
    """One matrix's sums for one δW: S_R's numerator and denominator per chunk, |g1 - g0| over positions and subcomponents."""

    r: list[float] = field(default_factory=list)
    den: list[float] = field(default_factory=list)
    abs_dg: float = 0.0
    positions: int = 0
    C: int = 0

    def add(self, a: Tensor, g0: Tensor, g1: Tensor, U: Tensor, V: Tensor, dW: Tensor) -> None:
        """One chunk, flattened to (N, ...)."""
        dg, Ma = g1 - g0, a @ dW.T
        self.r.append(dot((dg * (a @ V)) @ U, Ma))
        self.den.append(dot(Ma, Ma))
        self.abs_dg += float(dg.abs().sum(dtype=torch.float64))
        self.positions, self.C = self.positions + a.shape[0], g0.shape[1]

    def is_zero(self) -> bool:
        """Every number of a read with nothing changed: exact zeros."""
        return self.abs_dg == 0 and not any(self.den) and not any(self.r)


def mean_abs_dg(ms: dict[str, Matrix]) -> float:
    """|g1 - g0| per subcomponent and position, pooled over every subcomponent."""
    return sum(m.abs_dg for m in ms.values()) / sum(m.positions * m.C for m in ms.values())


def leave_one_chunk_out(r: list[float], den: list[float]) -> dict[str, float]:
    """S_R with each chunk left out in turn: how far the pooled ratio rests on a few rows. Needs two chunks; the job has 8 and 4."""
    each = [(sum(r) - x) / (sum(den) - y) for x, y in zip(r, den, strict=True)]
    return {"min": min(each), "max": max(each), "spread": max(each) - min(each)}


def pooled(ms: dict[str, Matrix]) -> dict[str, Any]:
    """Numerators and denominators summed over the matrices, per chunk and whole. A state with a zero denominator raises: W1 = W0 is read by `is_zero`."""
    r = [math.fsum(x) for x in zip(*(m.r for m in ms.values()), strict=True)]
    den = [math.fsum(x) for x in zip(*(m.den for m in ms.values()), strict=True)]
    return {"S_R": sum(r) / sum(den), "mean_abs_dg": mean_abs_dg(ms), "den": sum(den), "chunks": {"r": r, "den": den}, "leave_one_chunk_out": leave_one_chunk_out(r, den)}


def per_matrix(ms: dict[str, Matrix]) -> dict[str, Any]:
    return {k: {"S_R": sum(m.r) / sum(m.den), "mean_abs_dg": m.abs_dg / (m.positions * m.C), "den": sum(m.den)} for k, m in ms.items()}


# ----------------------------------------------------------------------------- reads of the model


@torch.no_grad()
def inputs_and_gates(model: Any, rows: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """(a, g) for one chunk of rows, each (N, dim) on the model's device: Γ on the model as it stands. Full precision: no autocast."""
    device = next(model.parameters()).device
    out = model(rows.to(device), cache_type="input")
    ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
    a = {k: v.reshape(-1, v.shape[-1]).float() for k, v in out.cache.items()}
    g = {k: v.reshape(-1, v.shape[-1]).float() for k, v in ci.lower_leaky.items()}
    return a, g


def read_state(model: Any, W0: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], dW: dict[str, Tensor], chunks: list[Tensor]) -> dict[str, Matrix]:
    """One δW over the chunks. W0's pass and W1's run per chunk, so a chunk's a and g are all that is held."""
    device = next(model.parameters()).device
    dW = {k: v.float().to(device) for k, v in dW.items()}
    W1 = {k: W0[k] + dW[k] for k in W0}
    ms = {k: Matrix() for k in comps}
    for rows in chunks:
        a, g0 = inputs_and_gates(model, rows)
        with model_state(model, target=W1):
            _, g1 = inputs_and_gates(model, rows)
        for k, m in ms.items():
            m.add(a[k], g0[k], g1[k], comps[k][0], comps[k][1], dW[k])
        del a, g0, g1
    return ms


def matched_gaussian(like: dict[str, Tensor], sigma: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    """Per matrix iid normal entries, scaled so E||G a||^2 = E||δW a||^2 under E[a a^T] = sigma. Keys in sorted order, so the seed fixes the draw."""
    gen, out = torch.Generator().manual_seed(seed), {}
    for k in sorted(like):
        G = torch.randn(like[k].shape, generator=gen, dtype=torch.float64)
        S, M = sigma[k].double().cpu(), like[k].double().cpu()
        out[k] = (G * (split.weighted_energy(M, S) / split.weighted_energy(G, S)) ** 0.5).float()
    return out


# ----------------------------------------------------------------------------- the filed rules (results/predictions.json)


def pair_verdict(code: float, copy: float, th: dict[str, float]) -> str:
    """One learned-nothing reading against one copy reading. keep and drop cannot both hold, since keep's gap is at least keep_copy_min - keep_code_max."""
    if abs(code - copy) <= th["drop_band"] + th["tie_eps"]:
        return "drop"
    if code > copy:
        return "reversed"
    if copy < th["keep_copy_min"] - th["tie_eps"] or code > th["keep_code_max"] + th["tie_eps"]:
        return "between"
    return "code_negative" if code < -th["keep_code_max"] - th["tie_eps"] else "keep"


def overall(primary: dict[str, str], second: dict[str, str], spread: float, th: dict[str, float]) -> str:
    """The label from the pairs alone. The two step-10 pairs decide; step 100 can only take a keep back."""
    labels = set(primary.values())
    if len(labels) > 1:
        return "between" if spread < th["seed_spread_min"] - th["tie_eps"] else "seeds_disagree"
    name = labels.pop()
    return "keep_at_step10_only" if name == "keep" and set(second.values()) != {"keep"} else name


def verdict(s: dict[str, float], th: dict[str, float]) -> dict[str, Any]:
    """The filed verdict from the S_R of code_step10, code_step100, ident32_s0, ident32_s1, the Gaussian's beside it, and code_step10 on the rows where r failed,
    which can only take back a label that would adopt the read. ident32_s0_held is printed, never read here."""
    primary = {x: pair_verdict(s["code_step10"], s[x], th) for x in ("ident32_s0", "ident32_s1")}
    second = {x: pair_verdict(s["code_step100"], s[x], th) for x in primary}
    spread = abs(s["ident32_s0"] - s["ident32_s1"])
    name = overall(primary, second, spread, th)
    held_fail = name in ("keep", "keep_at_step10_only") and s["code_step10_held"] > th["keep_code_max"] + th["tie_eps"]
    small = max(abs(s["code_step10"]), abs(s["ident32_s0"]), abs(s["ident32_s1"])) <= th["keep_code_max"]
    copy_mean = (s["ident32_s0"] + s["ident32_s1"]) / 2
    return {"verdict": "between" if held_fail else name, "held_rows_fail": held_fail, "pairs_step10": primary, "pairs_step100": second, "drop_both_small": name == "drop" and small,
            "copy_seed_spread": spread, "gaussian_like_copy": abs(s["gaussian"] - copy_mean) <= th["drop_band"] + th["tie_eps"], "read": s}
