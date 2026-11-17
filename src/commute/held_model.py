"""The held model H of causal-rule retention (cell P1.26, E197; plan `reports/plans/2026-11-18_plan_held-model-retention.md`,
"Frozen specification" and "Settled"; tested, not adopted). Added 2026-11-17 for `experiments/held_model_retention`.

H is M with only its output weights replaced: at each layer, the MLP's `down_proj` and the attention's `o_proj`. A site is
one of those two matrices at one layer. At a site, M's unit activations are a^M (the 3,072 neuron activations of the MLP,
or the six heads' 128-wide outputs before `o_proj`, concatenated), computed on M's input to that layer; F's activations
a^F are F's own units (F's norms and input weights) on the same M input. F's write at the site is a^F W_F^T. Every
reading writes a^M A^T, so H at a site is one matrix A in place of W_M:
    reading 1  units are neurons (columns) and heads (128-column blocks); F's unit i is paired one-to-one within the site
               with M's unit j by the least error left after fitting the gain (the score cross_ij^2 / |f^M_j|^2,
               maximised by linear assignment); its gain is g = cross_ij / |f^M_j|^2, any sign; A = W_M with M's unit j's
               columns times its partner's gain, 0 for an M unit with no non-zero partner. F units that write zero are
               left out, so a switched-off unit stays off.
    reading 2  as reading 1, with the whole MLP block one unit (heads as reading 1); F's block includes its appended
               neurons (`rule_retention.Extra`, read through `rule_retention.acts` and `cols`), so it can be wider than M's.
    reading 3  the least-squares map A = C_FM C_MM^+ from M's unit activations to F's write, C_FM = W_F G_FM.
All three come from three fp64 second moments per site, summed over every fit position: G_FF = a^F' a^F, G_FM = a^F' a^M,
G_MM = a^M' a^M. The write of a unit (k consecutive columns) is f_i = a_i W_i^T, so <f^F_i, f^M_j> is the block sum of
G_FM * (W_F' W_M) over unit i's rows and j's columns. A site's local fit residual, for any A, is
|a^F W_F' - a^M A'|^2 / |a^F W_F'|^2 = (n_F - 2 <A, W_F G_FM> + <A G_MM, A>) / n_F.
Embeddings, the unembedding (tied) and the final norm are M's in H; `with_ends` puts F's there for the side number.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from commute import rule_retention as R

Site = tuple[str, int]  # ("mlp", layer) or ("attn", layer)
READINGS = ("1", "2", "3")
UNIT = {"1": {"mlp": 1, "attn": R.D_HEAD}, "2": {"mlp": None, "attn": R.D_HEAD}}  # columns per unit; None = the whole site
ENDS = ("wte.weight", "lm_head.weight", "ln_f.weight")  # weights that are no unit's


def out_weight(N: R.Model, site: Site) -> Tensor:
    """The site's output matrix, appended neurons' columns included (in M and H, the down_proj parameter itself)."""
    kind, l = site
    return R.cols(N, l) if kind == "mlp" else N.net.h[l].attn.o_proj.weight


def z(block: Any, x: Tensor) -> Tensor:
    """(B, T, 768) the six heads' outputs before o_proj, concatenated, from `block`'s own rms_1 and attention on residual x."""
    seen: list[Tensor] = []
    h = block.attn.o_proj.register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    try:
        block.attn(block.rms_1(x))
    finally:
        h.remove()
    return seen[0]


@torch.no_grad()
def moments(F: R.Model, M: R.Model, batches: Iterable[Tensor]) -> dict[Site, dict[str, Tensor]]:
    """Per site, G_FF, G_FM and G_MM in fp64 over every position of `batches`, on M's input to each layer."""
    s: dict[Site, dict[str, Tensor]] = {}
    for toks in batches:
        for l, (x, y) in enumerate(R.m_inputs(M, toks)):
            acts = {("mlp", l): (R.acts(F, l, y), R.acts(M, l, y)), ("attn", l): (z(F.net.h[l], x), z(M.net.h[l], x))}
            for site, (aF, aM) in acts.items():
                aF, aM = aF.flatten(0, 1).double(), aM.flatten(0, 1).double()
                for k, v in {"FF": aF.T @ aF, "FM": aF.T @ aM, "MM": aM.T @ aM}.items():
                    g = s.setdefault(site, {})
                    g[k] = g[k] + v if k in g else v
    return s


def _blocks(T: Tensor, kr: int, kc: int) -> Tensor:
    n, m = T.shape
    return T.view(n // kr, kr, m // kc, kc).sum((1, 3))


def unit_sums(G: dict[str, Tensor], WF: Tensor, WM: Tensor, k: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """Units of k columns (None: each model's whole site one unit): cross_ij = <f^F_i, f^M_j>, |f^F_i|^2 and |f^M_j|^2,
    summed over every fit position."""
    WF, WM = WF.double(), WM.double()
    kF, kM = k or WF.shape[1], k or WM.shape[1]
    cross = _blocks(G["FM"] * (WF.T @ WM), kF, kM)
    return cross, _blocks(G["FF"] * (WF.T @ WF), kF, kF).diagonal(), _blocks(G["MM"] * (WM.T @ WM), kM, kM).diagonal()


@dataclass
class Pairing:
    """One site's units; index = F's unit."""

    partner: Tensor  # long, M's unit, -1 for an F unit that writes zero
    gain: Tensor  # double, cross / |f^M|^2 (0 when the partner writes zero); nan for a zero F unit
    err: Tensor  # double, the error left after the gain over |f^F|^2, 1 - cross^2 / (|f^F|^2 |f^M|^2); nan for a zero F unit


def pair(cross: Tensor, nF: Tensor, nM: Tensor) -> Pairing:
    """Pair F's non-zero units one-to-one with M's for the least total error after the fitted gain."""
    cross, nF, nM = cross.double().cpu(), nF.double().cpu(), nM.double().cpu()
    score = torch.where(nM > 0, cross.pow(2) / nM.clamp(min=1e-300), torch.zeros_like(cross))  # |f^F_i|^2 - error
    nz = (nF > 0).nonzero().squeeze(1)
    rows, cols = linear_sum_assignment(score[nz].numpy(), maximize=True)
    i, j = nz[torch.as_tensor(rows)], torch.as_tensor(cols, dtype=torch.long)
    partner = torch.full((len(nF),), -1, dtype=torch.long)
    gain, err = torch.full((len(nF),), float("nan"), dtype=torch.float64), torch.full((len(nF),), float("nan"), dtype=torch.float64)
    partner[i] = j
    gain[i] = torch.where(nM[j] > 0, cross[i, j] / nM[j].clamp(min=1e-300), torch.zeros_like(nF[i]))
    err[i] = 1 - score[i, j] / nF[i]
    return Pairing(partner, gain, err)


def held_matrix(p: Pairing, WM: Tensor, k: int | None) -> Tensor:
    """W_M with each M unit's k columns (None: all) times its partner's gain; 0 for an M unit with no non-zero partner."""
    k = k or WM.shape[1]
    g = torch.zeros(WM.shape[1] // k, dtype=torch.float64)
    i = (p.partner >= 0).nonzero().squeeze(1)
    g[p.partner[i]] = p.gain[i]
    return WM.double() * g.repeat_interleave(k).to(WM.device)[None, :]


def least_squares(G: dict[str, Tensor], WF: Tensor) -> Tensor:
    """A = C_FM C_MM^+ in fp64, the pseudo-inverse at fp64 machine tolerance (torch's default rtol)."""
    return WF.double() @ G["FM"] @ torch.linalg.pinv(G["MM"], hermitian=True)


def residual(G: dict[str, Tensor], WF: Tensor, A: Tensor) -> float:
    """|a^F W_F' - a^M A'|^2 / |a^F W_F'|^2 over the fit positions; nan when F's write is zero."""
    WF = WF.double()
    nF = float((G["FF"] * (WF.T @ WF)).sum())
    err = nF - 2 * float((A * (WF @ G["FM"])).sum()) + float(((A @ G["MM"]) * A).sum())
    return err / nF if nF > 0 else float("nan")


@dataclass
class Fit:
    pairs: dict[str, dict[Site, Pairing]]  # readings 1 and 2
    maps: dict[str, dict[Site, Tensor]]  # every reading: the fp64 matrix that replaces W_M at the site
    residual: dict[str, dict[Site, float]]


def fit(F: R.Model, M: R.Model, batches: Iterable[Tensor]) -> Fit:
    """The three readings' H matrices for every site, from one pass over the fit windows."""
    out = Fit({r: {} for r in UNIT}, {r: {} for r in READINGS}, {r: {} for r in READINGS})
    for site, G in moments(F, M, batches).items():
        WF, WM = out_weight(F, site), out_weight(M, site)
        for r, k in ((r, UNIT[r][site[0]]) for r in UNIT):
            out.pairs[r][site] = p = pair(*unit_sums(G, WF, WM, k))
            out.maps[r][site] = held_matrix(p, WM, k)
        out.maps["3"][site] = least_squares(G, WF)
        for r in READINGS:
            out.residual[r][site] = residual(G, WF, out.maps[r][site])
    return out


@torch.no_grad()
def held(M: R.Model, maps: dict[Site, Tensor]) -> R.Model:
    """H: a copy of M with each site's output matrix replaced, run end to end by its own forward."""
    H = R.Model(copy.deepcopy(M.net).requires_grad_(False))
    for site, A in maps.items():
        W = out_weight(H, site)
        W.copy_(A.to(W.device, W.dtype))
    return H


@torch.no_grad()
def with_ends(H: R.Model, F: R.Model) -> R.Model:
    """A copy of H with F's embedding, unembedding and final norm (the side number for full fine-tunes)."""
    out = R.Model(copy.deepcopy(H.net).requires_grad_(False))
    pH, pF = dict(out.net.named_parameters(remove_duplicate=False)), dict(F.net.named_parameters(remove_duplicate=False))
    for k in ENDS:
        pH[k].copy_(pF[k])
    return out


def ends_differ(F: R.Model, M: R.Model) -> bool:
    pF, pM = dict(F.net.named_parameters(remove_duplicate=False)), dict(M.net.named_parameters(remove_duplicate=False))
    return any(not torch.equal(pF[k], pM[k]) for k in ENDS)


def kl_last(p_logits: Tensor, q_logits: Tensor) -> float:
    """Mean over windows of KL(p || q) at the last position, fp64."""
    return float(R.kl(p_logits[:, -1:], q_logits[:, -1:]).mean())


def logit_error(h_logits: Tensor, f_logits: Tensor) -> float:
    """Max over windows of max |H logit - F logit| / max |F logit|, every position."""
    return float(((h_logits - f_logits).abs().amax((1, 2)) / f_logits.abs().amax((1, 2))).max())


def survival(kl_fh: float, kl_fm: float, floor: float) -> float | None:
    """S = 1 - KL(F || H) / KL(F || M); None ('no change') when KL(F || M) is at or under the floor."""
    return 1 - kl_fh / kl_fm if kl_fm > floor else None
