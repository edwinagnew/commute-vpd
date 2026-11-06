"""E112's case-label pair (plan reports/plans/2026-11-06_plan_e112-case-label.md; rules filed in
experiments/case_label/results/predictions.json): the word pool, the two label tables, the probe's Pile positions and the
pure rules that turn the readings into a verdict under `decisions.md` rows 81 and 87. No model is read here; the pure
rules are tested in tests/test_case_label.py.

Label convention: y = 1 for label word A, 0 for B. A window's margin is (logit A - logit B) signed toward its own label,
so it is right when the margin is > 0. Real task: A iff the word before the trigger is the capitalised member of its pair.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from commute import removal as RM

Pair = tuple[int, int]  # (capitalised token, lowercase token)

# ----------------------------------------------------------------------------- the pool and the labels


def case_pairs(decode: Callable[[list[int]], str], vocab: int) -> list[Pair]:
    """Every (" Word", " word") pair of single tokens: a space, then letters only, the lowercase member all lower case
    and the capitalised member the same word with its first letter upper case."""
    ids: dict[str, int] = {}
    for t in range(vocab):
        try:
            s = decode([t])
        except (IndexError, KeyError, ValueError, TypeError):  # an id past the tokenizer's range
            continue
        if len(s) >= 2 and s[0] == " " and s[1:].isalpha():
            ids.setdefault(s, t)
    out = []
    for s, low in ids.items():
        w = s[1:]
        cap = " " + w[0].upper() + w[1:]
        if w.islower() and cap != s and cap in ids:
            out.append((ids[cap], low))
    return out


def top_pairs(pairs: Sequence[Pair], counts: Mapping[int, int], n: int, exclude: set[int]) -> list[Pair]:
    """The n pairs whose rarer form is most frequent (ties to the lower lowercase id); a pair with either form in
    `exclude`, or either form absent, is out."""
    ok = [p for p in pairs if not set(p) & exclude and min(counts.get(p[0], 0), counts.get(p[1], 0)) > 0]
    ok.sort(key=lambda p: (-min(counts[p[0]], counts[p[1]]), p[1]))
    assert len(ok) >= n, f"{len(ok)} case pairs, need {n}"
    return ok[:n]


def split_pairs(pairs: Sequence[Pair], n_train: int, seed: int) -> tuple[list[Pair], list[Pair]]:
    """A seeded split by pair: (train pairs, held-out pairs)."""
    perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    return [pairs[i] for i in perm[:n_train]], [pairs[i] for i in perm[n_train:]]


def pick_labels(cands: Sequence[int], exclude: set[int], seed: int) -> tuple[int, int]:
    """Two label tokens (A, B) drawn, seeded, from the candidates not in `exclude`."""
    ok = [c for c in cands if c not in exclude]
    assert len(ok) >= 2, f"{len(ok)} label candidates left"
    perm = torch.randperm(len(ok), generator=torch.Generator().manual_seed(seed)).tolist()
    return ok[perm[0]], ok[perm[1]]


def real_y(pairs: Sequence[Pair]) -> dict[int, int]:
    """Real task: y = 1 (label A) for the capitalised form, 0 for the lowercase form."""
    return {t: int(i == 0) for p in pairs for i, t in enumerate(p)}


def twin_y(tokens: Sequence[int], seed: int) -> dict[int, int]:
    """Twin task: each token (both forms separately) gets A or B by a seeded shuffle, half each (one more B if odd)."""
    perm = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(seed)).tolist()
    return {tokens[i]: int(r < len(tokens) // 2) for r, i in enumerate(perm)}


def control_y(pairs: Sequence[Pair], seed: int) -> dict[int, int]:
    """Hewitt-Liang control: one seeded random bit per pair, the same for both of its forms (a fixed map)."""
    bits = torch.randint(0, 2, (len(pairs),), generator=torch.Generator().manual_seed(seed)).tolist()
    return {t: b for p, b in zip(pairs, bits, strict=True) for t in p}


def table(y: Mapping[int, int], a: int, b: int, vocab: int) -> Tensor:
    """(vocab,) int64: the label token of each pool token, -1 elsewhere."""
    t = torch.full((vocab,), -1, dtype=torch.long)
    for tok, yy in y.items():
        t[tok] = a if yy else b
    return t


# ----------------------------------------------------------------------------- the probe's Pile positions


def probe_positions(rows: Tensor, pairs: Sequence[Pair], half: int, cap: int, seed: int) -> tuple[Tensor, Tensor]:
    """(row, t) of Pile positions whose previous token is a word of `pairs`, with a full window row[t - half : t + half
    + 1]. Per pair, n = min(cap, hits of each form) positions of each form, drawn seeded: the classes are balanced."""
    lo, hi = half, rows.shape[1] - half  # t in [lo, hi)
    prev = rows[:, lo - 1 : hi - 1]
    gen = torch.Generator().manual_seed(seed)
    rs, ts = [], []
    for p in pairs:
        hits = [torch.nonzero(prev == tok, as_tuple=False) for tok in p]
        n = min(cap, *(len(h) for h in hits))
        for h in hits:
            pick = h[torch.randperm(len(h), generator=gen)[:n]]
            rs.append(pick[:, 0])
            ts.append(pick[:, 1] + lo)
    return torch.cat(rs), torch.cat(ts)


def value_probes(h: Tensor, y: Tensor, hh: Tensor, yh: Tensor, m: int, lam_frac: float) -> tuple[Tensor, Tensor, list[float]]:
    """P0's probes over a value set of m values (indices 0..m-1), moved from E112's `probes` loop (E116): per block, one
    closed-form ridge per value (one-vs-rest on the value's indicator, `removal.fit_ridge`) on the Pile residuals h
    (blocks, N, d) with labels y (N,), read on hh (blocks, M, d) as the argmax over the values (ties to the lowest index)
    against yh (M,). A value with no Pile position scores -inf, so it is never the argmax and its windows count wrong. For
    m = 2 the two probes are complements (score_0 = 1 - score_1), so the argmax is E112's "score_1 > 0.5". Returns the
    weights W (blocks, m, d), intercepts B (blocks, m), float64 on h's device, and the accuracy per block (nan when no
    value has a position or yh is empty)."""
    onehot = F.one_hot(y.long(), m).double().to(h.device)
    present = onehot.sum(0) > 0
    fits = [RM.fit_ridge(h[i], onehot, lam_frac) for i in range(len(h))]
    W, B = torch.stack([w.T for w, _ in fits]), torch.stack([b for _, b in fits])
    acc = []
    for i in range(len(h)):
        score = (hh[i].to(W.device).double() @ W[i].T + B[i]).masked_fill(~present, -math.inf)
        acc.append(float((score.argmax(1).cpu() == yh.long().cpu()).double().mean()) if present.any() else math.nan)
    return W, B, acc


# ----------------------------------------------------------------------------- the rules


def first_passing_block(acc: Sequence[float], sel: Sequence[float], acc_bar: float, sel_bar: float) -> int | None:
    """P0: l* = the earliest block before the last (0-based) with accuracy >= acc_bar and selectivity >= sel_bar, both
    inclusive; None when no such block (the last block never counts, a nan never passes)."""
    return next((i for i in range(len(acc) - 1) if acc[i] >= acc_bar and sel[i] >= sel_bar), None)


def scored_step(acc: Mapping[int, float], checkpoints: Sequence[int], bar: float) -> tuple[int, bool]:
    """The first checkpoint whose train-word accuracy reaches `bar` (inclusive), else the last ("did not learn")."""
    return next(((s, True) for s in checkpoints if s in acc and acc[s] >= bar), (checkpoints[-1], False))


def fit_threshold(score: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """g: the threshold and sign (+1: A when score > theta; -1: A when score < theta) that best agree with y on the fit
    windows. Thresholds at -inf and at each midpoint between adjacent distinct scores; ties go to the first found
    (lowest theta, sign +1 before -1). With one class only, theta = -inf and the sign that returns that class."""
    s = np.unique(score)
    thetas = np.concatenate([[-np.inf], (s[:-1] + s[1:]) / 2])
    best = (-1.0, -np.inf, 1)
    for th in thetas:
        for sign in (1, -1):
            agree = float((apply_threshold(score, th, sign) == y).mean())
            if agree > best[0]:
                best = (agree, float(th), sign)
    return best[1], best[2]


def apply_threshold(score: np.ndarray, theta: float, sign: int) -> np.ndarray:
    return (score > theta).astype(int) if sign == 1 else (score <= theta).astype(int)


def removed_share(m_state: Tensor, m_removed: Tensor, m_w0: Tensor, floor: float) -> float:
    """P3: the share of the label log-odds gain G = mean(state - W0) taken away by the removal, mean(state - removed) / G.
    nan ("no gain") when G <= floor nats."""
    gain = float((m_state - m_w0).double().mean())
    return float((m_state - m_removed).double().mean()) / gain if gain > floor else math.nan


def meets(p1: float, p2: float, p3: float, bars: Mapping[str, float]) -> dict[str, bool]:
    """Each of P1-P3 against the real column's bar (inclusive); a nan meets nothing."""
    return {"P1": p1 >= bars["P1"], "P2": p2 >= bars["P2"], "P3": p3 >= bars["P3"]}


def verdict(l_star: int | None, p1: float, p2: float, p3: float, bars: Mapping[str, float]) -> str:
    """The filed rule: possession fails -> "unreadable"; P1, P2 and P3 all meet the real column -> "reuses C"; else
    "not through C" (a state between the bars, a state that did not learn, and a "no gain" P3 all read so)."""
    if l_star is None:
        return "unreadable"
    return "reuses C" if all(meets(p1, p2, p3, bars).values()) else "not through C"


def do_nothing_ok(shares: Sequence[float], bar: float) -> bool:
    """The random direction's share <= bar on every state (inclusive; as filed, not |share|). A nan (no gain) is no
    reading there and does not stop the run."""
    return all(not s > bar for s in shares)


def reverse_steps(curve: Sequence[tuple[int, float]], bar: float) -> int | None:
    """Reverse fine-tuning, reported beside: the first step whose held-out label accuracy is <= bar, else None."""
    return next((s for s, a in curve if a <= bar), None)


# ----------------------------------------------------------------------------- Amendment 1 (results/predictions_amend1.json)


def pair_basis(a: Tensor, b: Tensor) -> Tensor:
    """An orthonormal basis (d, 2), fp32, of the span of two weight vectors (d,): a's unit direction, then b's part at
    right angles to it (QR in float64). Two parallel vectors have no rank-2 basis: assert."""
    q, r = torch.linalg.qr(torch.stack([a.double(), b.double()], 1))
    assert abs(float(r[1, 1])) > 1e-6 * float(b.double().norm()), "the two directions are parallel: no rank-2 basis"
    return q.float()


# ----------------------------------------------------------------------------- Amendment 2 (results/predictions_amend2.json)


def inlp(fit_x: Tensor, fit_y: Tensor, read_x: Tensor, read_y: Tensor, lam_frac: float, target: float, cap: int) -> tuple[Tensor, list[float], bool]:
    """Iterative nullspace projection (Ravfogel et al. 2020) with stage 0's ridge probe (`removal.fit_ridge`, y in {0, 1}):
    fit the probe on rows fit_x with the basis so far removed, read it on rows read_x with the same removal (score > 0.5,
    as stage 0 reads its probe) against read_y; stop when that accuracy is <= target (inclusive) or `cap` directions are
    removed, else add the probe's unit direction (made orthogonal to the basis) and refit. Returns the orthonormal basis
    (d, k) in float64, the accuracies after 0, 1, ..., k removals (k + 1 numbers) and whether the target was reached."""
    fx, rx, ry = fit_x.double(), read_x.double(), read_y.to(read_x.device).long()
    q = fx.new_zeros((fx.shape[1], 0))
    accs: list[float] = []
    while True:
        w, b = RM.fit_ridge(RM.remove(fx, q), fit_y.to(fx.device), lam_frac)
        accs.append(float(((RM.remove(rx, q) @ w + b > 0.5).long() == ry).double().mean()))
        if accs[-1] <= target or q.shape[1] >= cap:
            return q, accs, accs[-1] <= target
        w = w - q @ (q.T @ w)
        assert w.norm() > 0, "the probe has no weight outside the basis: nothing to remove"
        q = torch.cat([q, (w / w.norm())[:, None]], 1)


def merge_bases(bases: Sequence[Tensor], tol: float) -> Tensor:
    """One orthonormal basis (d, r), fp32, of the joint span of several bases (d, k_i): QR in float64 of their columns side
    by side, dropping each column whose part at right angles to the columns before it has norm <= tol (near-dependent),
    then QR of the columns kept. Every input empty: (d, 0)."""
    m = torch.cat([b.double().cpu() for b in bases], 1)
    assert m.shape[1] <= m.shape[0], f"{m.shape[1]} columns in {m.shape[0]} dimensions"
    if m.shape[1] == 0:
        return m.float()
    keep = torch.linalg.qr(m)[1].diagonal().abs() > tol
    return torch.linalg.qr(m[:, keep])[0].float()
