"""E67's weight-side read: per matrix, on W0's activations a, the masked output change g1 C1 a - g0 C0 a as

    D = g0 (C1 - C0) a        direct
    R = (g1 - g0) C0 a        re-switching
    I = (g1 - g0)(C1 - C0) a  interaction
    E = delta-W a - (D + R + I)  residual

and each term's overlap share S_X = E<X, delta-W a> / E||delta-W a||^2, which sum to 1 by linearity. g is Γ's value per position and
subcomponent, on the model it is read from. No dense component matrix: C a = (g * (a V)) U, in param-decomp's layout, V (d_in, C), U (C, d_out).
Differences of masked outputs are formed from the offsets (dU, dV), so fp32 does not have to resolve a 1% change against the whole output.
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

TERMS = ("D", "R", "I", "E")


@dataclass
class Offsets:
    """One matrix's C1 - C0, every column (zeros where nothing moved), param-decomp's layout."""

    dU: Tensor  # (C, d_out)
    dV: Tensor  # (d_in, C)


def offsets_from(states: dict[str, tuple[Tensor, Tensor]], uv: dict[str, tuple[Tensor, Tensor]], alive: dict[str, Tensor]) -> dict[str, Offsets]:
    """Alive (U', V') in theory layout, as `split.split_state` or `saved_states.offsets_state` give them, as `Offsets`, on CPU in fp32."""
    out = {}
    for k, (U2, V2) in states.items():
        U, V = uv[k]
        d_out, d_in, C = U.shape[0], V.shape[0], alive[k].numel()
        dU, dV = torch.zeros(C, d_out), torch.zeros(d_in, C)
        dU[alive[k]] = (U2.float() - U.float()).T
        dV[:, alive[k]] = V2.float() - V.float()
        out[k] = Offsets(dU, dV)
    return out


def changed(a: Tensor, w: Tensor, U: Tensor, V: Tensor, off: Offsets) -> Tensor:
    """sum_c w_c (C1_c - C0_c) a, with (a V1)_c U1_c - (a V)_c U_c = (a dV)_c U1_c + (a V)_c dU_c. a (N, d_in), w (N, C)."""
    return (w * (a @ off.dV)) @ (U + off.dU) + (w * (a @ V)) @ off.dU


def dot(x: Tensor, y: Tensor) -> float:
    return float((x * y).sum(dtype=torch.float64))


@dataclass
class Matrix:
    """One matrix's running sums over positions, for one δW. `split` holds the least-squares state's numerators; `nothing` do-nothing's."""

    den: float = 0.0  # E||δW a||^2, as a sum
    abs_dg: float = 0.0  # |g1 - g0|, summed over positions and subcomponents
    positions: int = 0
    C: int = 0
    split: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TERMS, 0.0))
    nothing: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TERMS, 0.0))

    def add(self, a: Tensor, g0: Tensor, g1: Tensor, U: Tensor, V: Tensor, dW: Tensor, off: Offsets) -> None:
        """One chunk, flattened to (N, ...). The residuals come from the whole masked change, not from 1 minus the rest, so the shares' sum is a check."""
        N, dg, acts = a.shape[0], g1 - g0, a @ V
        Ma = a @ dW.T
        R = (dg * acts) @ U
        D, I = changed(a, g0, U, V, off), changed(a, dg, U, V, off)
        whole = (g1 * (a @ (V + off.dV))) @ (U + off.dU) - (g0 * acts) @ U
        self.den += dot(Ma, Ma)
        self.abs_dg += float(dg.abs().sum(dtype=torch.float64))
        self.positions, self.C = self.positions + N, g0.shape[1]
        for term, x in (("D", D), ("R", R), ("I", I), ("E", Ma - whole)):
            self.split[term] += dot(x, Ma)
        self.nothing["R"] = self.split["R"]
        self.nothing["E"] += dot(Ma - R, Ma)

    def is_zero(self) -> bool:
        """Every number of a read with nothing changed: exact zeros."""
        return self.den == 0 and self.abs_dg == 0 and not any(self.split.values()) and not any(self.nothing.values())


def shares(num: dict[str, float], den: float) -> dict[str, float | None]:
    """S_X, and their sum. None on a zero denominator (W0): 0/0 has no share, and the numerators are read instead."""
    if den == 0:
        return dict.fromkeys(("S_D", "S_R", "S_I", "S_E", "sum"), None)
    out: dict[str, float | None] = {f"S_{k}": num[k] / den for k in TERMS}
    out["sum"] = sum(num[k] / den for k in TERMS)
    return out


def pooled(ms: dict[str, Matrix]) -> dict[str, Any]:
    """Numerators and denominators summed over the matrices; |g1 - g0| per subcomponent and position, pooled over every subcomponent."""
    den = sum(m.den for m in ms.values())
    return {"split": shares({t: sum(m.split[t] for m in ms.values()) for t in TERMS}, den),
            "nothing": shares({t: sum(m.nothing[t] for m in ms.values()) for t in TERMS}, den),
            "mean_abs_dg": sum(m.abs_dg for m in ms.values()) / sum(m.positions * m.C for m in ms.values()), "den": den}


def per_matrix(ms: dict[str, Matrix]) -> dict[str, Any]:
    return {k: {"split": shares(m.split, m.den), "nothing": shares(m.nothing, m.den), "mean_abs_dg": m.abs_dg / (m.positions * m.C), "den": m.den} for k, m in ms.items()}


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


@torch.no_grad()
def second_moments(model: Any, chunks: list[Tensor]) -> dict[str, Tensor]:
    """E[a a^T] per matrix on the model as it stands, float64 on CPU. As `split.input_second_moments`, without its bf16 autocast."""
    device, acc, n = next(model.parameters()).device, {}, 0
    for rows in chunks:
        cache = model(rows.to(device), cache_type="input").cache
        for k, v in cache.items():
            x = v.reshape(-1, v.shape[-1]).double()
            acc[k] = acc.get(k, 0) + (x.T @ x).cpu()
        n += rows.numel()
    return {k: v / n for k, v in acc.items()}


def read_state(model: Any, W0: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], dW: dict[str, Tensor], off: dict[str, Offsets], chunks: list[Tensor]) -> dict[str, Matrix]:
    """One δW over the chunks. W0's pass and W1's run per chunk, so a chunk's a and g are all that is held. A matrix with no entry in `off` has C1 = C0."""
    device = next(model.parameters()).device
    dW = {k: v.float().to(device) for k, v in dW.items()}
    moved = {k: Offsets(torch.zeros_like(U), torch.zeros_like(V)) if k not in off else Offsets(off[k].dU.to(device), off[k].dV.to(device)) for k, (U, V) in comps.items()}
    W1 = {k: W0[k] + dW[k] for k in W0}
    ms = {k: Matrix() for k in comps}
    for rows in chunks:
        a, g0 = inputs_and_gates(model, rows)
        with model_state(model, target=W1):
            _, g1 = inputs_and_gates(model, rows)
        for k, m in ms.items():
            m.add(a[k], g0[k], g1[k], comps[k][0], comps[k][1], dW[k], moved[k])
        del a, g0, g1
    return ms


def rel_norm_max(dW: dict[str, Tensor], W0: dict[str, Tensor]) -> float:
    """The largest ||δW|| / ||W|| over the matrices: the size E12 Amendment 1's table reports, `rel_norm_max`."""
    return max(float(dW[k].double().norm() / W0[k].double().norm().cpu()) for k in dW)


def matched_gaussian(like: dict[str, Tensor], sigma: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    """Per matrix iid normal entries, scaled so E||G a||^2 = E||δW a||^2 under E[a a^T] = sigma. Keys in sorted order, so the seed fixes the draw."""
    gen, out = torch.Generator().manual_seed(seed), {}
    for k in sorted(like):
        G = torch.randn(like[k].shape, generator=gen, dtype=torch.float64)
        S, M = sigma[k].double().cpu(), like[k].double().cpu()
        out[k] = (G * (split.weighted_energy(M, S) / split.weighted_energy(G, S)) ** 0.5).float()
    return out


def split_identity(M: Tensor, b: split.Basis, dU: Tensor, dV: Tensor, S: Tensor) -> dict[str, float]:
    """With every gate on, the least-squares state's change on the data, in the terms of `split.weighted_fractions`.
    With O = O(M), I = M - O and Q = dU dV^T, the state's change is I + Q, so in shares of E||M a||^2

        E||(I + Q) a||^2 = 1 - omega - 2 cross + second,  cross = E<I a, O a>,  second = E||Q a||^2 + 2 E<I a, Q a>

    `cross` is why 1 - omega alone is not the answer: O and I share the write side outside col U when Σ is not white. dU, dV in theory layout."""
    M, S = M.double(), S.double()
    O, Q = split.obstruction(M, b), dU.double() @ dV.double().T
    Ins = M - O
    den = split.weighted_energy(M, S)
    omega = split.weighted_fractions(M, b, S)["omega_act"]
    cross, second = float(((Ins @ S) * O).sum()) / den, (split.weighted_energy(Q, S) + 2 * float(((Ins @ S) * Q).sum())) / den
    return {"one_minus_omega": 1 - omega, "cross": cross, "second": second, "predicted": 1 - omega - 2 * cross + second, "den": den}


# ----------------------------------------------------------------------------- the filed rules (results/predictions.json)


def pair_verdict(code: float, copy: float, th: dict[str, float]) -> str:
    """One learned-nothing reading against one copy reading. keep and drop cannot both hold, since keep's gap is at least keep_copy_min - keep_code_max."""
    if not (math.isfinite(code) and math.isfinite(copy)):
        return "unreadable"
    if abs(code - copy) <= th["drop_band"] + th["tie_eps"]:
        return "drop"
    if code > copy:
        return "reversed"
    if copy < th["keep_copy_min"] - th["tie_eps"] or code > th["keep_code_max"] + th["tie_eps"]:
        return "between"
    return "code_negative" if code < -th["keep_code_max"] - th["tie_eps"] else "keep"


def verdict(s_r: dict[str, float | None], th: dict[str, float]) -> dict[str, Any]:
    """The filed verdict from the do-nothing S_R of code_step10, code_step100, ident32_s0, ident32_s1, and the Gaussian's reading beside it.
    code_step10 decides, as it matches the copy fine-tunes' step; code_step100 can only take a keep back."""
    read = {k: float("nan" if s_r.get(k) is None else s_r[k]) for k in ("code_step10", "code_step100", "ident32_s0", "ident32_s1", "gaussian")}  # type: ignore[arg-type]
    primary = {s: pair_verdict(read["code_step10"], read[s], th) for s in ("ident32_s0", "ident32_s1")}
    second = {s: pair_verdict(read["code_step100"], read[s], th) for s in primary}
    labels = set(primary.values())
    if "unreadable" in labels or "unreadable" in set(second.values()):
        name = "unreadable"
    elif len(labels) > 1:
        name = "seeds_disagree"
    else:
        name = labels.pop()
        if name == "keep" and set(second.values()) != {"keep"}:
            name = "keep_at_step10_only"
    small = max(abs(read["code_step10"]), abs(read["ident32_s0"]), abs(read["ident32_s1"])) <= th["keep_code_max"]
    copy_mean = (read["ident32_s0"] + read["ident32_s1"]) / 2
    return {"verdict": name, "pairs_step10": primary, "pairs_step100": second, "drop_both_small": name == "drop" and small, "copy_seed_spread": abs(read["ident32_s0"] - read["ident32_s1"]),
            "gaussian_like_copy": abs(read["gaussian"] - copy_mean) <= th["drop_band"] + th["tie_eps"], "read": read}
