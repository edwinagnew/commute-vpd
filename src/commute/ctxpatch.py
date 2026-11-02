"""E66's context patch and units share on the 67M (experiments/interchange_67m; plan
reports/plans/2026-11-02_plan_e66-interchange-67m.md, "Amendment 1: feasibility probe"). Target model only, fp32.

R(m) = metric in model m on x_lo with W0's block-l input at the answer position taken from x_hi, minus the same taken
from x_lo. T = mean R(state) / mean R(W0). T_off is T for a target the map does not train, with the map's own target
out of the odds, so a state that makes its target dominate does not move it. Q = T / T_off cancels global sharpening.
U = 1 - G_abl / G, the share of the state's gain lost when W0's top K neurons are mean-ablated in both models.
The rules below the rule are pure, so tests/test_ctxpatch.py reads every edge case on hand-made numbers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import finetune as FT
from commute import perturb as PT
from commute.data import Windows
from commute.identity import log_odds
from commute.steering import D_MLP, DW, N_LAYERS, Edit, Net, add_at

BLOCKS = (1, 2, 3)  # block 0's input is the embedding, which no update touches
EditOf = Callable[[slice], Edit]


@dataclass(frozen=True)
class Pairs:
    hi: Tensor  # (N, 41) donor contexts
    lo: Tensor  # (N, 41) recipient contexts, same key and marker
    tgt: Tensor  # (N,) token read
    mask: Tensor | None  # (N, 2) tokens out of the odds, off-map reads only: the map's target on x_hi and on x_lo (they differ on the copy map)


def read(net: Net, toks: Tensor, tgt: Tensor, half: int, dW: DW | None = None, edit_of: EditOf | None = None,
         mask: Tensor | None = None, batch: int = 64) -> Tensor:  # fmt: skip
    """(N,) float64: log-odds of tgt at `half` in W0 + dW, under the edit, with `mask`'s tokens (N, m) out of the odds."""
    out = []
    with net.at(dW):
        for s in range(0, len(toks), batch):
            i = slice(s, s + batch)
            lg = net.run(toks[i], edit_of(i) if edit_of else None)[0][:, half]
            if mask is not None:
                lg = lg.scatter(1, mask[i].to(lg.device), -math.inf)
            out.append(log_odds(lg, tgt[i]).double().cpu())
    return torch.cat(out)


def sites(net: Net, toks: Tensor, half: int, batch: int = 64) -> dict[int, Tensor]:
    """Per block of BLOCKS, (N, 768): block l's input at `half` in the model as it stands."""
    parts: dict[int, list[Tensor]] = {l: [] for l in BLOCKS}
    for s in range(0, len(toks), batch):
        seen = net.run(toks[s : s + batch], capture=True)[1]
        for l in BLOCKS:
            parts[l].append(seen[f"in{l}"][:, half])
    return {l: torch.cat(v) for l, v in parts.items()}


def set_at(l: int, pos: int, vecs: Tensor) -> EditOf:
    """Block l's input at `pos` takes each row's vector."""

    def edit_of(i: slice) -> Edit:
        def f(x: Tensor) -> Tensor:
            x = x.clone()
            x[:, pos] = vecs[i].to(x.device, x.dtype)
            return x

        return {f"in{l}": f}

    return edit_of


@dataclass
class Arm:
    pairs: Pairs
    hi: dict[int, Tensor]  # W0's sites on x_hi
    lo: dict[int, Tensor]  # W0's sites on x_lo
    R0: dict[int, Tensor]  # per block, (N,) R(W0)


def responses(net: Net, a: Pairs, hi: dict[int, Tensor], lo: dict[int, Tensor], half: int, dW: DW | None = None) -> dict[int, Tensor]:
    """Per block, (N,) R(W0 + dW)."""
    arm = lambda l, d: read(net, a.lo, a.tgt, half, dW, set_at(l, half, d[l]), a.mask)
    return {l: arm(l, hi) - arm(l, lo) for l in BLOCKS}


def make_arm(net: Net, p: Pairs, half: int) -> Arm:
    hi, lo = sites(net, p.hi, half), sites(net, p.lo, half)
    return Arm(p, hi, lo, responses(net, p, hi, lo, half))


def transplant(net: Net, a: Arm, l: int, half: int) -> EditOf:
    """Block l's input at every position takes x_hi's, with the arm's stored x_hi site at `half`, so a misaligned site shows."""

    def edit_of(i: slice) -> Edit:
        seen = net.run(a.pairs.hi[i], capture=True)[1][f"in{l}"]

        def f(_x: Tensor) -> Tensor:
            y = seen.clone()
            y[:, half] = a.hi[l][i].to(y.device, y.dtype)
            return y

        return {f"in{l}": f}

    return edit_of


def transplant_max(net: Net, a: Arm, half: int) -> float:
    """W0: the largest |read of x_lo with x_hi's block-l input written in - read of x_hi| over the pairs and blocks. Nothing but
    the stored site and the edit path can make it differ from rounding. nan propagates."""
    p = a.pairs
    want = read(net, p.hi, p.tgt, half, mask=p.mask)
    return nan_max([float((read(net, p.lo, p.tgt, half, None, transplant(net, a, l, half), p.mask) - want).abs().max()) for l in BLOCKS])


def steer_shares(net: Net, toks: Tensor, tgt: Tensor, half: int, dW: DW) -> dict[int, float]:
    """Steering in log-odds, per block: the mean shift of block l's input over the even windows, added to W0; the rise on
    the odd windows over the state's gain there. E52's B3 for a target per window (E62's states have no filed share)."""
    fit, score, tscore = toks[::2], toks[1::2], tgt[1::2]
    c0 = sites(net, fit, half)
    with net.at(dW):
        c1 = sites(net, fit, half)
    base = read(net, score, tscore, half)
    G = float((read(net, score, tscore, half, dW) - base).mean())
    rise = lambda l: float((read(net, score, tscore, half, None, lambda _i: {f"in{l}": add_at(half, (c1[l] - c0[l]).mean(0))}) - base).mean())
    return {l: rise(l) / G if G > 0 else math.nan for l in BLOCKS}


# ----------------------------------------------------------------------------- neurons


def batch_scores(net: Net, toks: Tensor, tgt: Tensor, half: int) -> Tensor:
    """(12288,) float64: gradient x activation of the log-odds at `half`, summed over the batch and positions. The gradient is
    the total derivative: a zero leaf added to each MLP's activations, so every path through later neurons counts."""
    seen: dict[int, tuple[Tensor, Tensor]] = {}

    def hook(l: int) -> Any:
        def f(_m: Any, a: tuple[Tensor, ...]) -> tuple[Tensor]:
            seen[l] = (a[0].detach(), torch.zeros_like(a[0]).requires_grad_(True))
            return (a[0] + seen[l][1],)
        return f

    handles = [b.mlp.down_proj.register_forward_pre_hook(hook(l)) for l, b in enumerate(net.net.h)]
    try:
        with torch.enable_grad():
            out = net.net(toks.to(net.device))
            lg = (out[0] if isinstance(out, tuple) else out).float()
            grads = torch.autograd.grad(log_odds(lg[:, half], tgt).sum(), [seen[l][1] for l in range(N_LAYERS)])
    finally:
        for h in handles:
            h.remove()
    return torch.cat([(g * seen[l][0]).sum((0, 1)).double().cpu() for l, g in enumerate(grads)])


def neuron_scores(net: Net, toks: Tensor, tgt: Tensor, half: int, batch: int = 64) -> Tensor:
    """(12288,): `batch_scores` at W0, mean over windows."""
    total = torch.zeros(N_LAYERS * D_MLP, dtype=torch.float64)
    for s in range(0, len(toks), batch):
        total += batch_scores(net, toks[s : s + batch], tgt[s : s + batch], half)
    return total / len(toks)


def neuron_stats(net: Net, toks: Tensor, batch: int = 64) -> tuple[Tensor, Tensor]:
    """(12288,) twice: W0's mean activation and mean |activation|, every position."""
    mean, mabs = torch.zeros(N_LAYERS * D_MLP, dtype=torch.float64), torch.zeros(N_LAYERS * D_MLP, dtype=torch.float64)
    for s in range(0, len(toks), batch):
        seen = net.run(toks[s : s + batch], capture=True)[1]
        a = torch.cat([seen[f"mlp{l}"].double() for l in range(N_LAYERS)], -1).cpu()
        mean, mabs = mean + a.sum((0, 1)), mabs + a.abs().sum((0, 1))
    return mean / toks.numel(), mabs / toks.numel()


def alignment(net: Net, targets: Tensor) -> Tensor:
    """(12288,): per neuron, the mean over the targets of |down_proj column . unembedding row|, W0."""
    lm = net.net.lm_head.weight.detach().float()[targets.unique().to(net.device)]
    return torch.cat([(lm @ net.W0[f"h.{l}.mlp.down_proj"]).abs().mean(0).double().cpu() for l in range(N_LAYERS)])


def ablate(idx: Tensor, means: Tensor) -> EditOf:
    """Mean ablation of the flat neuron indices at every position."""
    edit: Edit = {}
    for l in range(N_LAYERS):
        sel = idx[(idx >= l * D_MLP) & (idx < (l + 1) * D_MLP)] - l * D_MLP
        if len(sel):
            def f(x: Tensor, sel: Tensor = sel, m: Tensor = means[l * D_MLP : (l + 1) * D_MLP]) -> Tensor:
                x = x.clone()
                x[..., sel.to(x.device)] = m[sel].to(x.device, x.dtype)
                return x

            edit[f"mlp{l}"] = f
    return lambda _i: edit


def gain_under(net: Net, m: Map, dW: DW, ablated: EditOf | None = None) -> float:
    """The state's gain over W0 on the map's U windows, with the same ablation, if any, in both."""
    toks, tgt = m.u
    return float((read(net, toks, tgt, m.half, dW, ablated) - read(net, toks, tgt, m.half, None, ablated)).mean())


@dataclass
class Map:
    """One map's pool of contexts, its on-map and off-map pairs with W0's responses, and its neuron sets."""

    name: str
    toks: Tensor  # (N, 41)
    keys: Tensor  # (N,)
    tgt: Tensor  # (N,)
    on: Arm
    off: Arm
    lstar: int | None
    half: int
    every: int  # U reads every `every`-th window of the pool
    score: Tensor | None = None  # gradient x activation at W0
    top: Tensor | None = None
    draws: list[Tensor] | None = None
    fell: list[int] | None = None  # per draw, how many neurons left their cell

    @property
    def u(self) -> tuple[Tensor, Tensor]:
        return self.toks[:: self.every], self.tgt[:: self.every]


def make_map(net: Net, name: str, toks: Tensor, keys: Tensor, tgt: Tensor, off: Tensor, on: Pairs | None, S: dict[str, Any],
             th: dict[str, Any]) -> Map:  # fmt: skip
    """Pairs chosen from W0 alone. `on` is given for the copy map (a window and that window with B gone); otherwise ranked by
    W0's log-odds of the target. Off-map pairs are ranked by W0's log-odds of the off-map target, the map's target masked.
    Blind to: pairs are picked on W0's own metric, so R(W0) carries selection inflation and T can sit below 1 for any state,
    the more so as the update grows; that feeds the reading of T against the update's size."""
    half = S["half"]
    assert bool((off != tgt).all()), f"{name}: an off-map target is the map's own"
    if on is None:
        hi, lo = quarter_pairs(read(net, toks, tgt, half), keys)
        on = Pairs(toks[hi], toks[lo], tgt[lo], None)
    hi, lo = quarter_pairs(read(net, toks, off, half, mask=tgt[:, None]), keys)
    a_on, a_off = make_arm(net, on, half), make_arm(net, Pairs(toks[hi], toks[lo], off[lo], torch.stack([tgt[hi], tgt[lo]], 1)), half)
    star = l_star(a_on.R0, th, S["seed"]) if enough_pairs(len(on.tgt), len(keys.unique()), th) else None
    return Map(name, toks, keys, tgt, a_on, a_off, star, half, S["u_every"])


def prefix_share(a: Arm, half: int) -> tuple[float, float]:
    """The share of x_hi and of x_lo windows holding the target before the marker: what the pairs vary (finding 5)."""
    has = lambda t: float((t[:, : half - 1] == a.pairs.tgt[:, None]).any(1).double().mean())
    return has(a.pairs.hi), has(a.pairs.lo)


def patch_row(net: Net, m: Map, dW: DW, th: dict[str, Any], seed: int) -> dict[int, dict[str, float]]:
    """Per block: mean R, T, T_off, Q and their bootstrap sds."""
    R, R_off = (responses(net, a.pairs, a.hi, a.lo, m.half, dW) for a in (m.on, m.off))
    rec = {}
    for l in BLOCKS:
        T, d = ratio(R[l], m.on.R0[l], th, seed)
        T_off, d_off = ratio(R_off[l], m.off.R0[l], th, seed)
        Q, s_Q = quotient(T, d, T_off, d_off)
        rec[l] = {"R": float(R[l].mean()), "T": T, "s": float(d.std()), "T_off": T_off, "s_off": float(d_off.std()), "Q": Q, "s_Q": s_Q}
    return rec


def top_abs(score: Tensor, K: int) -> Tensor:
    return score.abs().argsort(descending=True, stable=True)[:K]


def neuron_sets(net: Net, m: Map, K: int, S: dict[str, Any]) -> None:
    """W0's top K for the map, and n_draws null sets matched on activation and alignment deciles (finding 4)."""
    toks, tgt = m.u
    m.score = neuron_scores(net, toks, tgt, m.half)
    m.top = top_abs(m.score, K)
    if not null_has_room(K):
        return
    a, b = deciles(neuron_stats(net, toks)[1]), deciles(alignment(net, tgt))
    gen = torch.Generator().manual_seed(S["seed"])
    m.draws, m.fell = (list(v) for v in zip(*(matched_draw(m.top, a, b, gen) for _ in range(S["n_draws"])), strict=True))


def units(net: Net, m: Map, K: int | None, means: Tensor, dW: DW, th: dict[str, Any]) -> dict[str, Any]:
    """G, U, the null draws' mean and sd, and how many neurons left their cell per draw. Void under the gain floor, with no K,
    or with no room for a null."""
    G = gain_under(net, m, dW)
    if K is None or m.top is None or m.draws is None or G < th["gain_floor"]:
        return {"G": G, "void": True}
    share = lambda idx: units_share(G, gain_under(net, m, dW, ablate(idx, means)), th["gain_floor"])
    nulls = torch.tensor([share(d) for d in m.draws], dtype=torch.float64)
    return {"G": G, "U": share(m.top), "null": float(nulls.mean()), "null_sd": float(nulls.std()), "fell": m.fell, "void": False}


# ----------------------------------------------------------------------------- the built controls


def with_matrices(net: Net, parts: dict[str, Tensor]) -> DW:
    return {k: parts[k].float().cpu() if k in parts else torch.zeros_like(w).cpu() for k, w in net.W0.items()}


def matched(net: Net, m: Map, build: Callable[[float], DW], Gt: float, cap: float, tol: float) -> tuple[DW, dict[str, Any]]:
    """`perturb.match` on the map's gain; status "cap" keeps the largest alpha, reported."""
    found = PT.match(lambda a: gain_under(net, m, build(a)), Gt, cap, tol)
    return build(found["x"]), {"alpha": found["x"], "G": found["f"], "status": found["status"], "G_target": Gt}


def write_state(net: Net, m: Map, Gt: float, tol: float) -> tuple[DW, dict[str, Any]]:
    """E46's write moved to block 3's MLP output, downstream of every l*: new by construction. It needs one answer per key,
    so the copy map has none."""
    acts = torch.cat([net.run(m.toks[s : s + 64], capture=True)[1]["mlp3"][:, m.half] for s in range(0, len(m.toks), 64)]).cpu()
    keys = m.keys.unique().tolist()
    A = torch.stack([acts[m.keys == k].mean(0) for k in keys], 1)
    vals = torch.stack([m.tgt[m.keys == k][0] for k in keys])
    lm = net.net.lm_head.weight.detach().float().cpu()
    X = PT.key_write(A, lm[vals].T)
    dW, info = matched(net, m, lambda a: with_matrices(net, {"h.3.mlp.down_proj": a * X}), Gt, PT.ALPHA_MAX, tol)
    return dW, info | {"fit": PT.write_fit(X, A, acts, lm[m.tgt])}


def amplify_state(net: Net, m: Map, K: int, Gt: float, cap: float, tol: float) -> tuple[DW, dict[str, Any]]:
    """W0's K most positive-scoring neurons at blocks >= l*, their down_proj columns x (1 + alpha): reuse by algebra, and
    downstream of the patch, so the patch can see it."""
    assert m.score is not None and m.lstar is not None
    start = m.lstar * D_MLP
    top = m.score[start:].argsort(descending=True, stable=True)[:K] + start
    top = top[m.score[top] > 0]
    by_block = {l: top[(top >= l * D_MLP) & (top < (l + 1) * D_MLP)] - l * D_MLP for l in range(m.lstar, N_LAYERS)}

    def build(a: float) -> DW:
        parts = {}
        for l, sel in by_block.items():
            w = net.W0[f"h.{l}.mlp.down_proj"].cpu()
            parts[f"h.{l}.mlp.down_proj"] = torch.zeros_like(w).index_copy(1, sel, a * w[:, sel])
        return with_matrices(net, parts)

    dW, info = matched(net, m, build, Gt, cap, tol)
    overlap = float(torch.isin(top, m.top).double().mean()) if m.top is not None and len(top) else math.nan
    return dW, info | {"n_neurons": len(top), "share_in_U_top": overlap}


def sharpened(net: Net, spans: Windows, S: dict[str, Any], log: Any) -> DW:
    """The sharpening fine-tune: `finetune_to_reference` toward W0's own logits divided by the temperature."""

    class Tempered(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = copy.deepcopy(net.net).eval().requires_grad_(False)

        def forward(self, toks: Tensor) -> Tensor:
            out = self.net(toks)
            return (out[0] if isinstance(out, tuple) else out) / S["tau"]

    kept: DW = {}

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        if step == S["steps"]:
            kept.update({k: (p.detach().float() - net.W0[k]).cpu() for k, p in params.items()})

    cfg = FT.ReferenceConfig(lr=S["lr"], batch=S["batch"], steps=S["steps"], grad_clip=S["grad_clip"], seed=S["seed"])
    FT.finetune_to_reference(net.model, Tempered().to(net.device), spans, cfg, net.device, on_step, log)
    return kept


# ----------------------------------------------------------------------------- the filed rules (pure arithmetic)


def finite(*v: float) -> bool:
    return all(isinstance(x, int | float) and math.isfinite(x) for x in v)


def quarter_pairs(scores: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
    """(hi, lo) window indices: per key, its windows by score, i-th highest with i-th lowest over the top and bottom
    quarters (stable, so ties follow window order). A key with under 4 windows gives no pair."""
    hi, lo = [], []
    for k in keys.unique().tolist():
        at = (keys == k).nonzero().squeeze(1)
        order = at[torch.argsort(scores[at], descending=True, stable=True)]
        q = len(order) // 4
        if q:
            hi.append(order[:q])
            lo.append(order.flip(0)[:q])
    empty = torch.empty(0, dtype=torch.long)
    return (torch.cat(hi), torch.cat(lo)) if hi else (empty, empty)


def enough_pairs(n_pairs: int, n_keys: int, th: dict[str, Any]) -> bool:
    return n_pairs >= th["min_pairs"] and n_keys >= th["min_keys"]


def boot_index(n: int, n_boot: int, seed: int) -> Tensor:
    return torch.randint(n, (n_boot, n), generator=torch.Generator().manual_seed(seed))


def l_star(R0: dict[int, Tensor], th: dict[str, Any], seed: int) -> int | None:
    """The earliest block whose mean R(W0) is at least the floor and at least 3 bootstrap sds (ties count); None if no
    block qualifies (the map is void, never low). A negative mean cannot qualify."""
    for l in sorted(R0):
        r = R0[l].double()
        if not len(r):
            return None
        sd = float(r[boot_index(len(r), th["n_boot"], seed)].mean(1).std())
        if float(r.mean()) >= th["floor"] and float(r.mean()) >= th["z"] * sd:
            return l
    return None


def ratio(R: Tensor, R0: Tensor, th: dict[str, Any], seed: int) -> tuple[float, Tensor]:
    """T = mean R / mean R0 and its bootstrap draws (pairs resampled together). nan unless R0 clears the floor at 3 sds, so
    a zero or negative denominator never gives a reading."""
    nan = torch.full((th["n_boot"],), math.nan, dtype=torch.float64)
    if not len(R0):
        return math.nan, nan
    i = boot_index(len(R0), th["n_boot"], seed)
    m0 = R0.double()[i].mean(1)
    if not (float(R0.mean()) >= th["floor"] and float(R0.mean()) >= th["z"] * float(m0.std())):
        return math.nan, nan
    return float(R.mean() / R0.mean()), R.double()[i].mean(1) / m0


def quotient(T: float, d: Tensor, T_off: float, d_off: Tensor) -> tuple[float, float]:
    """Q = T / T_off and its sd. nan when T_off is void, not positive, or crosses 0 in a resample (the ratio has no meaning)."""
    if not finite(T, T_off) or T_off <= 0 or not bool((d_off > 0).all()) or not bool(torch.isfinite(d).all()):
        return math.nan, math.nan
    return T / T_off, float((d / d_off).std())


def null_has_room(K: int) -> bool:
    """A null of K neurons outside the top K needs 2 K <= 12288; a tie has room."""
    return 2 * K <= N_LAYERS * D_MLP


def nan_max(v: list[float]) -> float:
    """The largest value, nan if any is nan (Python's max depends on the order)."""
    return float(torch.tensor(v, dtype=torch.float64).max())


def first_k(falls: dict[int, float], need: float) -> int | None:
    """The smallest K whose fall reaches `need` (a tie counts), None if none does."""
    return next((k for k in sorted(falls) if finite(falls[k]) and falls[k] >= need), None)


def deciles(x: Tensor) -> Tensor:
    """0..9 by rank, ties in index order."""
    rank = torch.empty_like(x, dtype=torch.long)
    rank[torch.argsort(x, stable=True)] = torch.arange(len(x))
    return rank * 10 // len(x)


def matched_draw(top: Tensor, a: Tensor, b: Tensor, gen: torch.Generator) -> tuple[Tensor, int]:
    """One neuron per neuron of `top`, none in `top` and none twice, from the same cell of the decile grids a, b; an empty
    cell falls back to the same decile of a, then to any free neuron. Returns the draw and how many fell back."""
    free = torch.ones(len(a), dtype=torch.bool)
    free[top] = False
    out, fell = [], 0
    for t in top.tolist():
        same_a = free & (a == a[t])
        cand = same_a & (b == b[t])
        if not bool(cand.any()):
            fell += 1
            cand = same_a if bool(same_a.any()) else free
        at = cand.nonzero().squeeze(1)
        pick = at[torch.randint(len(at), (1,), generator=gen)]
        free[pick] = False
        out.append(pick)
    return torch.cat(out), fell


def units_share(G: float, G_abl: float, floor: float) -> float:
    """1 - G_abl / G; nan under the gain floor (a tie passes), so a zero update and W0 have no share. Negative is kept."""
    return 1 - G_abl / G if finite(G, G_abl) and G >= floor else math.nan


def stage0_status(worst: float, lstars: dict[str, int | None], th: dict[str, Any]) -> str:
    """"transplant" on a max-abs over the tolerance (a tie passes, nan fails), "void_maps" with half the maps or more void at
    every block (a tie stops), else "pass"."""
    if math.isnan(worst) or worst > th["transplant_tol"]:
        return "transplant"
    return "void_maps" if 2 * sum(v is None for v in lstars.values()) >= len(lstars) else "pass"


def near_one(q: float, s: float, th: dict[str, Any]) -> str:
    """|Q - 1| within q_band: "near_1", a tie counts; "off" outside; "void" on a nan, or when 3 s is over the band (the reading
    cannot tell near from far; a tie reads)."""
    if not finite(q, s) or th["z"] * s > th["q_band"]:
        return "void"
    return "near_1" if abs(q - 1) <= th["q_band"] else "off"


def t_band(T: float, s: float, th: dict[str, Any]) -> str:
    """The write's T: "as_expected" in [write_t_min, 1 + 3 s], ties count; "low" below; "high" above; "void" on a nan."""
    if not finite(T, s):
        return "void"
    return "low" if T < th["write_t_min"] else "high" if T > 1 + th["z"] * s else "as_expected"


def above(x: float, s: float, ref: float, s_ref: float, th: dict[str, Any]) -> str:
    """"above" iff x - ref > 3 max(s, s_ref), strict; "not_above" otherwise; "void" on a nan."""
    if not finite(x, s, ref, s_ref):
        return "void"
    return "above" if x - ref > th["z"] * max(s, s_ref) else "not_above"


def write_conditioned(fit: dict[str, float], th: dict[str, Any]) -> bool:
    """The write holds on single windows as E46's did: cond(A) and the mean relative error within their bars (ties pass, nan fails)."""
    return finite(fit["cond_A"], fit["rel_err_mean"]) and fit["cond_A"] <= th["write_cond_max"] and fit["rel_err_mean"] <= th["write_rel_err_max"]


def stage1_status(amp: dict[str, str]) -> str:
    """"stop" when no map can be read, or amplify's T misses on half the readable maps or more (a tie stops); else "pass"."""
    readable = [v for v in amp.values() if v != "void"]
    return "stop" if not readable or 2 * sum(v == "not_above" for v in readable) >= len(readable) else "pass"


def blind_mark(shares: Mapping[int, float | None], lstar: int | None, th: dict[str, Any]) -> str:
    """Finding 1 on the state: "blind" when steering reaches blind_share at any block up to l* (a tie counts), since the
    patch at l* overwrites what those blocks carry; "unknown" with a share missing or no l*; else "readable"."""
    if lstar is None:
        return "unknown"
    up_to = [shares.get(l) for l in BLOCKS if l <= lstar]
    if any(v is None or not finite(v) for v in up_to):
        return "unknown"
    return "blind" if max(v for v in up_to if v is not None) >= th["blind_share"] else "readable"


def control_verdicts(ctrl: dict[str, Any], write: dict[str, Any], l: int, th: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on one map at its l*. `write` is the map's own write, or the lookup's for the copy map. Amplify is read
    against the write only when its gain matched and the write is conditioned; else void, so stage 1's stop does not count it.
    Sharpening is read against 1, and its Q only once its T shows it sharpened."""
    w, a, sh = write["row"][l], ctrl["amplify"]["row"][l], ctrl["sharpen"]["row"][l]
    ok = ctrl["amplify"]["info"]["status"] == "ok" and write_conditioned(write["info"]["fit"], th)
    sharp = above(sh["T"], sh["s"], 1.0, 0.0, th)

    def u_rule(u: dict[str, Any]) -> str:
        return "void" if u["void"] else above(u["U"], 0.0, u["null"], u["null_sd"], th)

    return {"write_T": t_band(w["T"], w["s"], th), "write_Q": near_one(w["Q"], w["s_Q"], th),
            "write_conditioned": write_conditioned(write["info"]["fit"], th),
            "amplify_T": above(a["T"], a["s"], w["T"], w["s"], th) if ok else "void",
            "amplify_Q": above(a["Q"], a["s_Q"], w["Q"], w["s_Q"], th) if ok else "void",
            "sharpen_T": sharp, "sharpen_Q": near_one(sh["Q"], sh["s_Q"], th) if sharp == "above" else "void",
            "write_U": u_rule(write["units"]), "amplify_U": u_rule(ctrl["amplify"]["units"]),
            "amplify_share_in_U_top": ctrl["amplify"]["info"]["share_in_U_top"]}
