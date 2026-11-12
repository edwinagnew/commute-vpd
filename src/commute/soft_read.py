"""Reading a change F of the 67M under the third reuse wording (cell P1.19, E173; testing only, not adopted): a variable V
of a hand-named causal model H of M, fixed on M alone, is checked by interchange interventions (carry c and reversion r),
and the change in V is graded by a least-squares fit in a stated map class. Added 2026-11-12. Plan:
reports/plans/2026-11-12_plan_soft-intervention-wording.md and its appendix (section D gives every reading used here).

A variable is a site and an orthonormal Q (d, k) at the last position. A trigger is the one named variable the one-variable
class may add: either a scalar, the projection of M's activation at (site, pos) on a unit direction, which interchange
interventions can swap; or a gain, a (n, k) regressor read from a token's own row of a table, which they cannot.
Nothing here changes a function another experiment calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from commute import variable_read as VR
from commute.hidden_maps import Edit, Fn, chunks, merged, run, stash_last, tv
from commute.steering import Net

EPS = 1e-9  # float rounding at a bar: a share exactly at the bar passes
NO_READING = "no reading"  # the gap is under the floor
NO_VARIABLE = "no carrying variable among the named ones"  # min(c, r) misses the bar at every site
PURE = "pure re-setting"
NEW_COMPUTATION = "new computation"


def new_trigger(cls: str) -> str:
    return f"new trigger ({cls} class)"


# ---- the alignment, fixed on M --------------------------------------------------------------------------------------


def class_dir(net: Net, windows: Tensor, label: Callable[[Tensor], Tensor], site: str, pos: int,
              batch: int) -> tuple[Tensor, float, float]:
    """M's class-mean direction at (site, pos) between windows of label 1 and label 0 (label -1: not counted), as a unit
    vector d, with the two classes' mean projections p0, p1 on it. `planted.class_means` is this at in2, the last position,
    unnormalised. Raises on an empty class (that item then has no one-variable reading; it should not occur)."""
    sums: list[Tensor] = []
    n = [0, 0]
    for t in chunks(windows, batch):
        h = run(net, t, sites=(site,))[1][site][:, pos].double()
        y = label(t).to(h.device)
        if not sums:
            sums = [h.new_zeros(h.shape[1]), h.new_zeros(h.shape[1])]
        for k in (0, 1):
            m = y == k
            sums[k] += h[m].sum(0)
            n[k] += int(m.sum())
    if min(n) == 0:
        raise ValueError(f"an empty class at {site}, position {pos}: {n}")
    mu0, mu1 = (sums[k] / n[k] for k in (0, 1))
    d = (mu1 - mu0) / (mu1 - mu0).norm()
    return d.float(), float(mu0 @ d), float(mu1 @ d)


def write_at(pos: int, Q: Tensor, value: Tensor) -> Fn:
    """`hidden_maps.write_last` at any position: the subspace Q (d, k)'s value at `pos` becomes `value` (B, k)."""

    def f(x: Tensor) -> Tensor:
        x = x.clone()
        h = x[:, pos].clone()
        x[:, pos] = h + (value - h @ Q) @ Q.T
        return x

    return f


# ---- F built from an edit of M --------------------------------------------------------------------------------------


def caller_first(net: Net, piece: Callable[[Tensor], Edit], write_site: str) -> VR.Target:
    """F = M with `piece(toks)`. At `write_site` the piece runs before the caller's edit (so writing M's value there reverts
    it); at every other site the caller's edit runs first, so a piece that reads an earlier site (a stash) reads what the
    caller wrote there. `VR.piece_target` runs the piece first everywhere, and its Cue then ignores a swapped trigger."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        e = edit or {}
        before = {s: fn for s, fn in e.items() if s != write_site}
        after = {s: fn for s, fn in e.items() if s == write_site}
        return run(net, toks, sites=sites, edit=merged(before, piece(toks), after))

    return f


def planted_constant(net: Net, unit: Tensor) -> Callable[[float], VR.Target]:
    """s -> F = M + s unit at in2, last position, on every window."""
    return lambda s: VR.label_target(net, lambda t: torch.ones(t.shape[0]), s * unit)


def planted_trigger(net: Net, unit: Tensor, d: Tensor, p0: float, p1: float) -> Callable[[float], VR.Target]:
    """s -> F = M + s l unit at in2, last position, l = (P - p0) / (p1 - p0), P the value at in1 (last position) along the
    unit direction d: E157's ceiling with the read moved one block earlier. The stash runs after any caller's edit at in1."""

    def make(s: float) -> VR.Target:
        def piece(_toks: Tensor) -> Edit:
            store: dict[str, Tensor] = {}

            def add(x: Tensor) -> Tensor:
                x = x.clone()
                x[:, -1] = x[:, -1] + (((store["h"] @ d - p0) / (p1 - p0)) * s)[:, None] * unit
                return x

            return {"in1": stash_last(store, "h"), "in2": add}

        return caller_first(net, piece, "in2")

    return make


# ---- the readings ---------------------------------------------------------------------------------------------------


def sweep(net: Net, F: VR.Target, windows: Tensor, Qs: dict[str, Tensor], batch: int, extra_sites: tuple[str, ...],
          extra: Callable[[Tensor, dict[str, Tensor]], dict[str, Tensor]], interchange: bool) -> dict[str, Any]:
    """One pass over `windows`: V_M and V_F at each site of `Qs` (n, k), extra(toks, M's activations) per row, and, with
    `interchange`, the gap and the carry and reversion distances per site: d_c = mean TV(M[V <- V_F], F) and
    d_r = mean TV(F[V <- V_M], M), at the last position. Everything returned is float64 on the CPU."""
    sites = tuple(Qs)
    vm: dict[str, list[Tensor]] = {s: [] for s in sites}
    vf: dict[str, list[Tensor]] = {s: [] for s in sites}
    ex: dict[str, list[Tensor]] = {}
    tot = {"gap": 0.0, **{f"c_{s}": 0.0 for s in sites}, **{f"r_{s}": 0.0 for s in sites}}
    for toks in chunks(windows, batch):
        lm, sm = run(net, toks, sites=sites + extra_sites)
        lf, sf = F(toks, sites)
        for k, v in extra(toks, sm).items():
            ex.setdefault(k, []).append(v.double().cpu() if v.is_floating_point() else v.cpu())
        for s, Q in Qs.items():
            a, b = sm[s][:, -1] @ Q, sf[s][:, -1] @ Q
            vm[s].append(a.double().cpu())
            vf[s].append(b.double().cpu())
            if interchange:
                lc = run(net, toks, edit={s: write_at(-1, Q, b)})[0]
                lr = F(toks, edit={s: write_at(-1, Q, a)})[0]
                tot[f"c_{s}"] += float(tv(lc[:, -1], lf[:, -1]).sum())
                tot[f"r_{s}"] += float(tv(lr[:, -1], lm[:, -1]).sum())
        if interchange:
            tot["gap"] += float(tv(lf[:, -1], lm[:, -1]).sum())
    n = windows.shape[0]
    return {"VM": {s: torch.cat(v) for s, v in vm.items()}, "VF": {s: torch.cat(v) for s, v in vf.items()},
            "extra": {k: torch.cat(v) for k, v in ex.items()}, "mean": {k: v / n for k, v in tot.items()}}


def carry_reversion(mean: dict[str, float], sites: Iterable[str]) -> dict[str, dict[str, float]]:
    """c = 1 - d_c / gap and r = 1 - d_r / gap per site, from `sweep`'s means. The caller has checked the gap is above the floor."""
    g = mean["gap"]
    return {s: {"c": 1 - mean[f"c_{s}"] / g, "r": 1 - mean[f"r_{s}"] / g} for s in sites}


def best_site(cr: dict[str, dict[str, float]]) -> str:
    """The site of the largest min(c, r); ties to the earlier site (dicts keep the order of `Qs`)."""
    top = max(min(v.values()) for v in cr.values())
    return next(s for s, v in cr.items() if min(v.values()) >= top)


def delta_fit(fit: tuple[Tensor, Tensor, Tensor | None], read: tuple[Tensor, Tensor, Tensor | None]) -> dict[str, float]:
    """Least squares of Delta (n, k) on X (n, p) (the caller includes the intercept or a set of columns that spans it), plus,
    when g (n, k) is given, one scalar gain on g shared by the k outputs (fitted by Frisch-Waugh: Delta and g residualised
    on X). Fitted on `fit`, read on `read`: f = 1 - SSE / SS, SS centred per output, summed over outputs. Returns f, the
    gain or the last column's coefficient as `beta` (the trigger's, when the caller puts it last), and SS and sum Delta^2
    on the read half (for the constant rule)."""
    D, X, g = fit
    if g is None:
        theta = _lstsq(X, D)
        beta = float(theta[-1, 0]) if D.shape[1] == 1 else float("nan")
        gain = 0.0
    else:
        both = _lstsq(X, torch.cat([D, g], 1))
        rd, rg = D - X @ both[:, : D.shape[1]], g - X @ both[:, D.shape[1] :]
        den = float((rg * rg).sum())
        gain = float((rg * rd).sum()) / den if den > 0 else 0.0
        theta, beta = _lstsq(X, D - gain * g), gain
    Dr, Xr, gr = read
    pred = Xr @ theta + (gain * gr if gr is not None else 0.0)
    ss = float(((Dr - Dr.mean(0)) ** 2).sum())
    sse = float(((Dr - pred) ** 2).sum())
    return {"f": 1 - sse / ss if ss > 0 else float("nan"), "beta": beta, "ss": ss, "sum_sq": float((Dr * Dr).sum())}


def _lstsq(X: Tensor, Y: Tensor) -> Tensor:
    """Minimum-norm least squares in float64 through the normal equations (gelsd): a column that is all zero, or a set
    of columns that spans another, gets the minimum-norm share."""
    A, B = X.T @ X, X.T @ Y
    return torch.linalg.lstsq(A, B, rcond=1e-12, driver="gelsd").solution


def trigger_interchange(net: Net, F: VR.Target, toks: Tensor, t_tgt: Tensor, t_src: Tensor, d_tgt: Tensor,
                        trig: tuple[str, int, Tensor], var: tuple[str, Tensor], beta: float, batch: int) -> dict[str, float]:
    """i = sum sign(t' - t) (Delta_sw - Delta) / (beta sum |t' - t|): in F and in M, the target windows `toks` get the
    source's trigger value t' (written at trig = (site, pos, unit d)), V is read at var = (site, Q) with Q (d, 1), and
    Delta_sw = V_F - V_M. `d_tgt` is the targets' own Delta. NaN (not applicable) when beta or sum |t' - t| is 0."""
    site, pos, d = trig
    vsite, Q = var
    den = beta * float((t_src - t_tgt).abs().sum())
    if beta == 0 or den == 0:
        return {"i": float("nan"), "n_pairs": toks.shape[0]}
    num, at = 0.0, 0
    for tk in chunks(toks, batch):
        b = tk.shape[0]
        value = t_src[at : at + b].to(d.device, torch.float32)[:, None]
        e = {site: write_at(pos, d[:, None], value)}
        vm = run(net, tk, sites=(vsite,), edit=e)[1][vsite][:, -1] @ Q
        vf = F(tk, (vsite,), e)[1][vsite][:, -1] @ Q
        sw = (vf - vm)[:, 0].double().cpu()
        num += float((torch.sign(t_src[at : at + b] - t_tgt[at : at + b]) * (sw - d_tgt[at : at + b])).sum())
        at += b
    return {"i": num / den, "n_pairs": toks.shape[0]}


# ---- the grade (pure) -----------------------------------------------------------------------------------------------


def grade(gap: float, cr: dict[str, dict[str, float]], fits: dict[str, Any], i: float | None, bar: float, floor_gap: float,
          constant_rel: float, classes: tuple[str, ...]) -> str:
    """Section D's grade at bar B. `fits` holds, at the best site, "own", "one-variable" and each further class of
    `classes` (in order, e.g. ("per-token",) or ("named-variables", "per-token")), each {"f", "ss", "sum_sq"}. NaN fails
    every bar (an i not applicable: no pair, beta 0). `i` None: the trigger is the input token's own row (the key tables),
    which no interchange can swap, so the one-variable branch reads f_one alone. A constant needs a non-zero change:
    SS <= constant_rel sum Delta^2 with sum Delta^2 > 0."""
    if gap < floor_gap:
        return NO_READING
    if max(min(v.values()) for v in cr.values()) < bar - EPS:
        return NO_VARIABLE
    own = fits["own"]
    if (own["sum_sq"] > 0 and own["ss"] <= constant_rel * own["sum_sq"]) or own["f"] >= bar - EPS:
        return PURE
    if fits["one-variable"]["f"] >= bar - EPS and (i is None or i >= bar - EPS):
        return new_trigger("one-variable")
    for cls in classes:
        if fits[cls]["f"] >= bar - EPS:
            return new_trigger(cls)
    return NEW_COMPUTATION
