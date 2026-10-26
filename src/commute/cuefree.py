"""Cue-free models of the change of Gamma's importance values (E35 `experiments/regating_cuefree`; E36 to follow).

Plan: reports/2026-10-26_plan_regating-without-cues.md. Inputs are per-row values (one row per window at the trigger,
or per position), (N, C) per matrix, fp32 CPU (`shift_anatomy.Vals`): g0 under W0 and g1 under the do-nothing state.
The change d = g1 - g0 is kept on alive subcomponents only (0 on dead, as `recruit.mean_change`). Every pattern is
clip(g0(x) + c(x), 0, 1), with c fitted on training rows only, except the fixed re-weighting clip(a g0(x) + b):
  groups   c(x) = mean d over the training rows with x's label (cue word, k-means cluster of g0, or a random label)
  near     c(x) = mean d over the `k` training rows nearest x in g0 (Euclidean over alive subcomponents, all matrices)
  refit    per subcomponent, least squares g1 ~ a g0 + b over training rows; where g0 varies by less than 0.01 (sd)
           over them, a = 1 and b = mean d (`line`); dead subcomponents a = 1, b = 0
"""

from __future__ import annotations

import torch
from torch import Tensor

from commute.shift_anatomy import Vals, flat, group_means, per_window

CHUNK = 1024  # rows per distance / gather block


def change(g1: Vals, g0: Vals, alive: dict[str, Tensor]) -> Vals:
    return {k: torch.where(alive[k].cpu().bool(), g1[k] - g0[k], 0.0) for k in g0}


def apply(g0: Vals, c: Vals) -> Vals:
    return {k: (v + c[k]).clamp(0.0, 1.0) for k, v in g0.items()}


def broadcast(vec: dict[str, Tensor], n: int) -> Vals:
    return {k: v.expand(n, -1) for k, v in vec.items()}


def by_group(d_tr: Vals, lab_tr: Tensor, lab_he: Tensor) -> Vals:
    """Each held-out row gets the mean d of the training rows with its label; every held-out label must occur."""
    missing = set(lab_he.tolist()) - set(lab_tr.tolist())
    assert not missing, f"held-out labels without training rows: {sorted(missing)[:10]}"
    return per_window(group_means(d_tr, lab_tr), lab_he)


def _sqdist(a: Tensor, b: Tensor) -> Tensor:
    return (a * a).sum(1, keepdim=True) - 2 * a @ b.T + (b * b).sum(1)[None, :]


def nearest(g0_tr: Vals, g0_he: Vals, d_tr: Vals, alive: dict[str, Tensor], k: int, device: str) -> Vals:
    """Each held-out row gets the mean d of its k nearest training rows in g0."""
    X = flat(g0_tr, alive).to(device)
    Y = flat(g0_he, alive)
    idx = torch.cat([_sqdist(Y[s : s + CHUNK].to(device), X).topk(k, largest=False).indices.cpu()
                     for s in range(0, Y.shape[0], CHUNK)])
    blocks = range(0, idx.shape[0], CHUNK)
    return {m: torch.cat([v[idx[s : s + CHUNK]].mean(1) for s in blocks]) for m, v in d_tr.items()}


def kmeans(X: Tensor, K: int, seed: int, iters: int = 50) -> Tensor:
    """(K, D) centroids of the rows of X (on X's device): k-means++ seeding, then Lloyd's steps until no row moves or
    `iters`; an emptied cluster keeps its centroid."""
    gen = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    assert K <= n, f"K {K} > {n} rows"
    cent = X[torch.randint(n, (1,), generator=gen).to(X.device)]
    d2 = _sqdist(X, cent).squeeze(1).clamp_min(0)
    for _ in range(1, K):
        j = torch.multinomial((d2 / d2.sum()).double().cpu(), 1, generator=gen)
        cent = torch.cat([cent, X[j.to(X.device)]])
        d2 = torch.minimum(d2, _sqdist(X, cent[-1:]).squeeze(1).clamp_min(0))
    lab = assign(X, cent)
    for _ in range(iters):
        sums = torch.zeros_like(cent).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=K)
        cent = torch.where(cnt[:, None] > 0, sums / cnt.clamp_min(1)[:, None], cent)
        new = assign(X, cent)
        if torch.equal(new, lab):
            break
        lab = new
    return cent


def assign(X: Tensor, cent: Tensor) -> Tensor:
    return torch.cat([_sqdist(X[s : s + CHUNK], cent).argmin(1) for s in range(0, X.shape[0], CHUNK)])


def clusters(g0_tr: Vals, g0_he: Vals, alive: dict[str, Tensor], K: int, seed: int,
             device: str) -> tuple[Tensor, Tensor]:
    """k-means labels of the training rows and the held-out rows' labels by nearest centroid, both on CPU. Centroids
    no training row is nearest to are dropped first, so every held-out label has training rows."""
    X = flat(g0_tr, alive).to(device)
    cent = kmeans(X, K, seed)
    cent = cent[assign(X, cent).unique()]
    return assign(X, cent).cpu(), assign(flat(g0_he, alive).to(device), cent).cpu()


def purity(lab: Tensor, truth: Tensor) -> float:
    """Share of rows whose cluster's most common true label is their own (1: every cluster holds one true label)."""
    hits = sum(int(torch.bincount(truth[lab == c]).max()) for c in lab.unique())
    return hits / lab.numel()


FLAT_VAR = 1e-4  # a subcomponent whose g0 varies by less than 0.01 (sd) over the fit rows gets no slope of its own


def line(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Per column of (N, C) rows, least squares y ~ a x + b; a column with var(x) <= FLAT_VAR gets a = 1, b = mean
    change (its few varying rows would otherwise set a slope of any size)."""
    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean(0)
    var, cov = ((x - mx) ** 2).mean(0), ((x - mx) * (y - my)).mean(0)
    flatline = var <= FLAT_VAR
    a = torch.where(flatline, 1.0, cov / var.clamp_min(FLAT_VAR))
    return a.float(), torch.where(flatline, my - mx, my - a * mx).float()


def refit(g0_tr: Vals, g1_tr: Vals, alive: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """(a, b) per matrix, (C,) each: the per-subcomponent `line` g1 ~ a g0 + b over the training rows."""
    a, b = {}, {}
    for k, x in g0_tr.items():
        ak, bk = line(x, g1_tr[k])
        live = alive[k].cpu().bool()
        a[k] = torch.where(live, ak, 1.0).float()
        b[k] = torch.where(live, bk, 0.0).float()
    return a, b


def reweighted(g0: Vals, a: dict[str, Tensor], b: dict[str, Tensor]) -> Vals:
    return {k: (a[k] * v + b[k]).clamp(0.0, 1.0) for k, v in g0.items()}
