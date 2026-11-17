"""Causal-rule retention at a declared level, on the 67M (plan `reports/plans/2026-11-17_plan_causal-rule-retention.md`;
the definition is `~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`, tested, not adopted).
Added 2026-11-17 for `experiments/rule_retention`; nothing else imports it.

The 67M (param-decomp's LlamaSimpleMLP, `artifacts/vpd/t-9d2b8f02/model_config.yaml`): 4 blocks, each
x + attn(rms_1(x)) then x + mlp(rms_2(x)); attention has 6 heads of 128 (d 768, rotary, no bias), its heads concatenated
into o_proj; the MLP is 768 -> 3072 -> 768 with NewGELU and no bias; the embedding is tied to the unembedding.

Two levels. A: a unit is one MLP neuron; its output is its activation times its down_proj column. Appended neurons
(`Extra`) are units of F with no pair in M. Attention, embeddings and the unembedding are equations between units.
B: a unit is one head read after o_proj (its 128 columns of o_proj), or one whole MLP block, appended neurons included.

A unit's rule in network N is N's unit applied to M's input at that layer (`m_inputs`). d(u) = min(1, |f_F - f_M|^2 /
(|f_F|^2 + |f_M|^2)), summed over windows, positions and residual dimensions; r = 1 - d; an unpaired unit has r = 0;
a unit whose two outputs are both zero on every window has r = 1 (the two rules agree there).

F with a set of units put back to M's rule (`forward(F, M, toks, revert)`): a reverted neuron uses M's norm, input row
and output column on the hybrid's own residual; a reverted appended neuron is removed; a reverted head uses M's rms_1
and attention weights for that head; a reverted MLP block is M's block. Everything else is F's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

N_HEAD, D_HEAD, D_MLP = 6, 128, 3072
Unit = tuple[str, int, int]  # ("neuron", layer, j) with j >= D_MLP an appended neuron; ("head", layer, h); ("mlp", layer, 0)


def name(u: Unit) -> str:
    kind, l, j = u
    if kind == "neuron":
        return f"L{l}.n{j}" if j < D_MLP else f"L{l}.new{j - D_MLP}"
    return f"L{l}.h{j}" if kind == "head" else f"L{l}.mlp"


class Extra(nn.Module):
    """n neurons appended to one MLP layer: gelu(h W_in^T) W_out^T on that layer's normed input h."""

    def __init__(self, n: int, d: int, gelu: nn.Module, std_in: float, std_out: float, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w_in = nn.Parameter(torch.randn(n, d, generator=gen) * std_in)
        self.w_out = nn.Parameter(torch.randn(d, n, generator=gen) * std_out)
        self.gelu = gelu

    def acts(self, h: Tensor) -> Tensor:
        return self.gelu(h @ self.w_in.T)

    def forward(self, h: Tensor) -> Tensor:
        return self.acts(h) @ self.w_out.T


@dataclass
class Model:
    """A LlamaSimpleMLP and its appended neurons by layer. `append` hooks them into net.h[l].mlp, so `net(toks)` (the
    native forward the faithfulness gate compares with) includes them."""

    net: Any
    extra: dict[int, Extra] = field(default_factory=dict)

    def append(self, layer: int, e: Extra) -> None:
        self.extra[layer] = e
        self.net.h[layer].mlp.register_forward_hook(lambda _m, a, o: o + e(a[0]))

    def logits(self, toks: Tensor) -> Tensor:
        return self.net(toks)[0].float()


# ---- unit outputs on a given residual ------------------------------------------------------------------------------


def heads(block: Any, x: Tensor) -> Tensor:
    """(B, T, 6, d) each head's output after o_proj on residual x, from `block`'s own rms_1 and attention."""
    z: list[Tensor] = []
    h = block.attn.o_proj.register_forward_pre_hook(lambda _m, a: z.append(a[0]))
    try:
        block.attn(block.rms_1(x))
    finally:
        h.remove()
    B, T, d = x.shape
    W = block.attn.o_proj.weight.view(d, N_HEAD, D_HEAD)
    return torch.einsum("bthk,dhk->bthd", z[0].view(B, T, N_HEAD, D_HEAD), W)


def acts(N: Model, l: int, y: Tensor) -> Tensor:
    """Layer l's neuron activations (B, T, 3072 + appended) on residual y."""
    b = N.net.h[l]
    h = b.rms_2(y)
    a = b.mlp.gelu(b.mlp.c_fc(h))
    return torch.cat([a, N.extra[l].acts(h)], -1) if l in N.extra else a


def cols(N: Model, l: int) -> Tensor:
    """Layer l's output columns (d, 3072 + appended), in the order of `acts`."""
    W = N.net.h[l].mlp.down_proj.weight
    return torch.cat([W, N.extra[l].w_out], -1) if l in N.extra else W


def block(N: Model, l: int, y: Tensor) -> Tensor:
    return acts(N, l, y) @ cols(N, l).T


@torch.no_grad()
def m_inputs(M: Model, toks: Tensor) -> list[tuple[Tensor, Tensor]]:
    """M's input to each layer: (the block's input residual x, the residual before its MLP y)."""
    out, x = [], M.net.wte(toks)
    for b in M.net.h:
        y = x + b.attn(b.rms_1(x))
        out.append((x, y))
        x = y + b.mlp(b.rms_2(y))
    return out


# ---- F, M and the hybrid -----------------------------------------------------------------------------------------


def _attn(F: Model, M: Model, l: int, x: Tensor, rev: set[int], by_heads: bool) -> Tensor:
    bF = F.net.h[l]
    if not rev and not by_heads:
        return bF.attn(bF.rms_1(x))
    hF = heads(bF, x)
    if rev:
        idx = torch.tensor(sorted(rev), device=x.device)
        hF = hF.index_copy(2, idx, heads(M.net.h[l], x).index_select(2, idx))
    return hF.sum(2)


def _mlp(F: Model, M: Model, l: int, y: Tensor, rev: set[int], whole: bool) -> Tensor:
    if whole:
        return block(M, l, y)
    aF, WF = acts(F, l, y), cols(F, l)
    if not rev:
        return aF @ WF.T
    keep = torch.ones(aF.shape[-1], device=y.device, dtype=aF.dtype)
    keep[sorted(rev)] = 0
    back = 1 - keep[:D_MLP]
    aM, WM = acts(M, l, y)[..., :D_MLP], cols(M, l)[:, :D_MLP]
    return (aF * keep) @ WF.T + (aM * back) @ WM.T


@torch.no_grad()
def forward(F: Model, M: Model, toks: Tensor, revert: Iterable[Unit] = (), by_heads: bool = False) -> Tensor:
    """F's logits computed unit by unit, with the units in `revert` put back to M's rule. With no revert and
    `by_heads`, the attention output is rebuilt as the sum of its heads (the Level B faithfulness gate)."""
    rv = set(revert)
    x = F.net.wte(toks)
    for l in range(len(F.net.h)):
        x = x + _attn(F, M, l, x, {j for k, ll, j in rv if k == "head" and ll == l}, by_heads)
        x = x + _mlp(F, M, l, x, {j for k, ll, j in rv if k == "neuron" and ll == l}, ("mlp", l, 0) in rv)
    return F.net.lm_head(F.net.ln_f(x)).float()


def gate(N: Model, toks: Tensor) -> dict[str, float]:
    """Max over windows of max |rebuilt - native logit| / max |native logit|, per level, N rebuilt against itself."""
    native = N.logits(toks)
    scale = native.abs().amax((1, 2))
    return {lv: float(((forward(N, N, toks, by_heads=lv == "B") - native).abs().amax((1, 2)) / scale).max()) for lv in "AB"}


def kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """KL(p || q) per window, summed over the vocabulary and averaged over positions, fp64."""
    lp, lq = torch.log_softmax(p_logits.double(), -1), torch.log_softmax(q_logits.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean(-1)


# ---- the retention profile ----------------------------------------------------------------------------------------


def _r(diff: Tensor, nF: Tensor, nM: Tensor) -> Tensor:
    den = nF + nM
    d = torch.where(den > 0, (diff.clamp(min=0) / den.clamp(min=1e-300)).clamp(max=1), torch.zeros_like(den))
    return 1 - d


@torch.no_grad()
def profile(F: Model, M: Model, batches: Iterable[Tensor]) -> dict[str, dict[Unit, float]]:
    """r of every unit of F at both levels, read on M's inputs over every window and position of `batches`."""
    s: dict[tuple[str, int], Tensor] = {}

    def add(k: str, l: int, v: Tensor) -> None:
        s[k, l] = s[k, l] + v if (k, l) in s else v

    for toks in batches:
        for l, (x, y) in enumerate(m_inputs(M, toks)):
            aF, aM = acts(F, l, y).double().flatten(0, 1), acts(M, l, y).double().flatten(0, 1)
            hF, hM = heads(F.net.h[l], x).double(), heads(M.net.h[l], x).double()
            oF, oM = block(F, l, y).double(), block(M, l, y).double()
            for k, v in {"aFF": (aF * aF).sum(0), "aFM": (aF[:, :D_MLP] * aM).sum(0), "aMM": (aM * aM).sum(0),
                         "hd": (hF - hM).pow(2).sum((0, 1, 3)), "hF": hF.pow(2).sum((0, 1, 3)), "hM": hM.pow(2).sum((0, 1, 3)),
                         "bd": (oF - oM).pow(2).sum(), "bF": oF.pow(2).sum(), "bM": oM.pow(2).sum()}.items():  # fmt: skip
                add(k, l, v)
    out: dict[str, dict[Unit, float]] = {"A": {}, "B": {}}
    for l in range(len(M.net.h)):
        WF, WM = cols(F, l).double(), cols(M, l).double()
        nF, nM = s["aFF", l] * WF.pow(2).sum(0), s["aMM", l] * WM.pow(2).sum(0)  # |f_F|^2, |f_M|^2 per neuron
        diff = nF[:D_MLP] - 2 * s["aFM", l] * (WF[:, :D_MLP] * WM).sum(0) + nM  # |a_F w_F - a_M w_M|^2 summed
        rA = torch.cat([_r(diff, nF[:D_MLP], nM), torch.zeros_like(nF[D_MLP:])])  # appended neurons: no pair, r = 0
        out["A"] |= {("neuron", l, j): v for j, v in enumerate(rA.tolist())}
        out["B"] |= {("head", l, h): v for h, v in enumerate(_r(s["hd", l], s["hF", l], s["hM", l]).tolist())}
        out["B"][("mlp", l, 0)] = float(_r(s["bd", l], s["bF", l], s["bM", l]))
    return out


# ---- retention for the contrast -----------------------------------------------------------------------------------


def changed(r: dict[Unit, float], tol: float) -> list[Unit]:
    return [u for u, v in r.items() if v < 1 - tol]


def smallest_sets(closure: Callable[[list[Unit]], float], units: list[Unit], orders: int, bar: float, seed: int) -> list[list[Unit]]:
    """Greedy backward elimination, once per seeded random order: start from every changed unit and drop each in turn
    (put it back to M's rule) when the rest still closes at least `bar` of KL(F || M)."""
    gen, sets = torch.Generator().manual_seed(seed), []
    for _ in range(orders):
        S = list(units)
        for i in torch.randperm(len(units), generator=gen).tolist():
            rest = [v for v in S if v != units[i]]
            if closure(rest) >= bar:
                S = rest
        sets.append(S)
    return sets


def contrast(F: Model, M: Model, toks: Tensor, r: dict[Unit, float], th: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """The retention for the contrast F against M on `toks` at one level: each order's smallest set and its mean r."""
    units = changed(r, th["r_tol"])
    lF, lM = forward(F, M, toks), M.logits(toks)
    kl0 = float(kl(lF, lM).mean())
    out: dict[str, Any] = {"kl": kl0, "n_changed": len(units)}
    if not units:
        return out | {"reading": "no change"}
    if kl0 <= th["kl_floor"]:
        return out | {"reading": "no contrast"}

    def closure(S: list[Unit]) -> float:
        return 1 - float(kl(lF, forward(F, M, toks, [u for u in units if u not in S])).mean()) / kl0

    sets = smallest_sets(closure, units, th["orders"], th["closure_bar"], th["order_seed"])
    ret = [sum(r[u] for u in S) / len(S) if S else None for S in sets]
    out |= {"closure_none_kept": closure([]), "sets": [[name(u) for u in S] for S in sets], "closure": [closure(S) for S in sets], "retention": ret}
    if any(v is None for v in ret):
        return out | {"reading": "no set"}  # the contrast lies in the equations between units
    out |= {"reading": "read", "min": min(ret), "max": max(ret)}  # type: ignore[type-var]
    log(f"  contrast: KL {kl0:.6g}, {len(units)} changed, sets of {[len(S) for S in sets]}, retention {min(ret):.4f}..{max(ret):.4f}")  # type: ignore[type-var]
    return out
