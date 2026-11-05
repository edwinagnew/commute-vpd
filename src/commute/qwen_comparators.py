"""E92's three cheap non-VPD readings of a Qwen3-1.7B state, and their rules: update size, the rank of the trained answer,
one steering vector. No decomposition is read. Plan: reports/plans/2026-11-05_plan_e92-comparators.md; the run:
experiments/qwen_comparators/run.py.

A task's rows are packed sequences (prompt, then answer). A position "reads" the token after it, so `Rows.answer` marks the
positions that predict an answer token, and `Rows.first` the one that predicts each answer's first token. A row's position 0
and an answer cut by the row's end are dropped by the mask, not treated apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute.qwen_sft import Seq, pack, set_weights
from commute.qwen_vpd import parse_site

BATCH = 16  # rows per forward; 16 x 256 positions of logits are under 3 GB on the card


@dataclass(frozen=True)
class Rows:
    name: str
    tokens: Tensor  # (N, L)
    answer: Tensor  # (N, L) bool: this position predicts an answer token
    first: Tensor  # (N, L) bool: this position predicts an answer's first token

    def half(self, which: str) -> Rows:
        """The fit half is the even rows and the scoring half the odd rows, as `steering.Task.part`."""
        s = slice(0, None, 2) if which == "fit" else slice(1, None, 2)
        return Rows(f"{self.name}_{which}", self.tokens[s], self.answer[s], self.first[s])


def _reads(marks: Tensor) -> Tensor:
    out = torch.zeros_like(marks)
    out[:, :-1] = marks[:, 1:]
    return out


def rows_of(name: str, seqs: Sequence[Seq], pack_len: int, keep: int, n_rows: int | None = None, even: int | None = None) -> Rows:
    """`seqs` packed end to end into rows of `pack_len` (n_rows None: every whole row), `even` of them taken evenly if given,
    each cut to `keep`. A sequence whose prompt length is its whole length marks no answer: it stays in the rows as context."""
    ids = [s for s, _ in seqs]
    answer = [[int(i >= p) for i in range(len(s))] for s, p in seqs]
    first = [[int(i == p) for i in range(len(s))] for s, p in seqs]
    packed = [pack(x, n_rows, pack_len) for x in (ids, answer, first)]
    pick = [i * len(packed[0]) // even for i in range(even)] if even else list(range(len(packed[0])))
    toks, ans, fst = (torch.tensor([x[i][:keep] for i in pick]) for x in packed)
    return Rows(name, toks, _reads(ans.bool()), _reads(fst.bool()))


ROW_LEN, TOFU_ROWS = 256, 128  # E68's row_len; E86's TOFU rows


def a1_rows(tok: Any, test: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]]) -> Rows:
    """E57's anchor-1 rows as `qwen_reading.a1_rows`: the prompted model's greedy answers to test 256-511 (`items`, E45's
    large/P/a1_rows.json) under the unprompted chat, 64 rows of 512, cut to 256. (Moved from qwen_comparators/run.py, E92.)"""
    from commute.qwen_eval import prompt

    seqs: list[Seq] = []
    for r in items:
        p = tok(prompt(tok, test[r["idx"]]["question"]), add_special_tokens=False)["input_ids"]
        seqs.append((p + r["ids"], len(p)))
    return rows_of("a1", seqs, 512, ROW_LEN, n_rows=64)


def tofu_rows(tok: Any, end_id: int, kept: set[int]) -> Rows:
    """E86's 128 TOFU rows: the 800 rows in TOFU's own wording, pair 0 of the five E84 trained, packed in rows of 256,
    taken evenly. Rows not in `kept` (the 13 E84 dropped) stay as context and mark no answer. (Moved from E92's run.py.)"""
    from commute import qwen_tofu as tofu

    seqs = []
    for i, r in enumerate(tofu.held_rows()):
        ids, p = tofu.seq(tok, end_id, r["question"] + tofu.ASK, r["answer"])
        seqs.append((ids, p if i in kept else len(ids)))
    return rows_of("tofu", seqs, ROW_LEN, ROW_LEN, even=TOFU_ROWS)


def question_seen(R: Rows, user_start: int, user_word: int) -> Tensor:
    """Per answer position, what its row holds of its question: 0 the whole chat turn, 1 cut (the row opens inside the
    prompt), 2 none (the row opens inside the answer). (Moved from qwen_comparators/dissect.py, E92.)"""
    out = []
    for i in range(len(R.tokens)):
        toks, first = R.tokens[i].tolist(), R.first[i].tolist()
        where = {p + 1 for p in range(len(toks) - 1) if toks[p] == user_start and toks[p + 1] == user_word}
        state, turn_seen = 2, False
        for p in range(len(toks)):
            turn_seen = turn_seen or p in where
            if first[p]:
                state, turn_seen = (0 if turn_seen else 1), False
            if R.answer[i, p]:
                out.append(state)
    return torch.tensor(out, dtype=torch.int64)


# ----------------------------------------------------------------------------- 1, update size


def _size(dw2: float, w02: float) -> dict[str, float]:
    """Each unit named: `norm` the root of the summed squares, `ratio_norm` a norm over a norm, `share_squared` its square."""
    return {"norm": math.sqrt(dw2), "w0_norm": math.sqrt(w02), "ratio_norm": math.sqrt(dw2 / w02), "share_squared": dw2 / w02}


def update_size(dW: Mapping[str, Tensor], W0: Mapping[str, Tensor]) -> dict[str, Any]:
    """‖δW‖ over ‖W0‖ per block and in total, on the decomposed matrices only (the embedding, norms and head are not in δW).
    The total's `norm` is E91's definition (`qwen_size_grid/run.py`, read_state)."""
    d2, w2 = {}, {}
    for s, x in dW.items():
        b = parse_site(s)[0]
        d2[b] = d2.get(b, 0.0) + float(x.double().pow(2).sum())
        w2[b] = w2.get(b, 0.0) + float(W0[s].double().pow(2).sum())
    return {"total": _size(sum(d2.values()), sum(w2.values())), "blocks": {str(b): _size(d2[b], w2[b]) for b in sorted(d2)}}


# ----------------------------------------------------------------------------- 2, rank and log-probability


@dataclass(frozen=True)
class Read:
    rank: Tensor  # (K,) int32, 1 = top of the whole vocabulary, ties to the best place
    logp: Tensor  # (K,) float32
    row_of: Tensor  # (K,) int64
    is_first: Tensor  # (K,) bool

    def where(self, keep: Tensor) -> Read:
        return Read(self.rank[keep], self.logp[keep], self.row_of[keep], self.is_first[keep])


def token_reads(logits: Tensor, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Rank and log P of the next token at the masked positions."""
    next_tok = torch.zeros_like(tokens)
    next_tok[:, :-1] = tokens[:, 1:]
    lg, tgt = logits[mask].float(), next_tok[mask][:, None]
    return (lg > lg.gather(1, tgt)).sum(1).int() + 1, torch.log_softmax(lg, -1).gather(1, tgt).squeeze(1)


@torch.no_grad()
def read(model: Any, rows: Rows, add: tuple[int, Tensor] | None = None) -> Read:
    """The weights as they stand, on every row; `add` = (l, v) adds v to block l's input at the answer positions."""
    dev, rank, logp = next(model.parameters()).device, [], []
    for s in range(0, len(rows.tokens), BATCH):
        toks, mask = rows.tokens[s : s + BATCH].to(dev), rows.answer[s : s + BATCH].to(dev)
        handle = None
        if add is not None:
            handle = model.model.layers[add[0]].register_forward_pre_hook(
                lambda _m, a, m=mask, v=add[1]: (a[0] + m[..., None] * v, *a[1:]))
        try:
            logits = model(toks, use_cache=False).logits
        finally:
            if handle is not None:
                handle.remove()
        r, lp = token_reads(logits, toks, mask)
        rank.append(r.cpu())
        logp.append(lp.cpu())
    return Read(torch.cat(rank), torch.cat(logp), rows.answer.nonzero()[:, 0], rows.first[rows.answer])


def summary(r: Read) -> dict[str, Any]:
    """The two summaries of a multi-token answer: its first token, and the teacher-forced mean over every answer token."""
    out: dict[str, Any] = {}
    for key, part in (("first", r.where(r.is_first)), ("all", r)):
        n = len(part.rank)
        out[key] = {"n": n, "median_rank": float(part.rank.float().median()) if n else None,
                    "mean_logp": float(part.logp.double().mean()) if n else None}
    return out


def row_mean_se(x: Tensor, row_of: Tensor) -> tuple[float, float, int]:
    """The pooled mean of x, the standard error of its per-row means (inf under two rows), and the number of rows."""
    rows = row_of.unique()
    if len(rows) == 0:
        return 0.0, math.inf, 0
    means = torch.stack([x[row_of == i].double().mean() for i in rows])
    return float(x.double().mean()), float(means.std() / math.sqrt(len(rows))) if len(rows) > 1 else math.inf, len(rows)


def share(rise: Tensor, gain: Tensor, row_of: Tensor) -> dict[str, Any]:
    """rise / gain, pooled, by rows. The filed rule: undefined (None) unless the gain is further from 0 than twice its row
    standard error, so an empty pool, one row, a zero update and a gain lost in row noise all read undefined."""
    G, se, n = row_mean_se(gain, row_of)
    out: dict[str, Any] = {"gain": G, "gain_se": se if n > 1 else None, "rows": n, "share": None, "share_se": None}
    if n > 1 and abs(G) > 2 * se:
        R, rse, _ = row_mean_se(rise, row_of)
        out |= {"rise": R, "share": R / G, "share_se": rse / abs(G)}
    return out


def separates(a1: Sequence[float | None], a2: Sequence[float | None]) -> str:
    """The plan's stop rule for a column on the anchors: "counts" if the gap between the anchors' means is strictly over the
    larger of the two same-anchor seed spreads, else "does not count"; a missing or non-finite seed reads "unread"."""
    vals = [*a1, *a2]
    if len(a1) != 2 or len(a2) != 2 or any(v is None or not math.isfinite(v) for v in vals):
        return "unread"
    x, y = [v for v in a1 if v is not None], [v for v in a2 if v is not None]
    return "counts" if abs(sum(x) - sum(y)) / 2 > max(abs(x[0] - x[1]), abs(y[0] - y[1])) else "does not count"


# ----------------------------------------------------------------------------- 3, one steering vector


@torch.no_grad()
def inputs_at(model: Any, rows: Rows) -> list[Tensor]:
    """Per block, its input at the answer positions, (K, d), on the card."""
    dev, seen = next(model.parameters()).device, [[] for _ in model.model.layers]
    for s in range(0, len(rows.tokens), BATCH):
        mask = rows.answer[s : s + BATCH].to(dev)
        handles = [layer.register_forward_pre_hook(lambda _m, a, i=i, m=mask: seen[i].append(a[0][m]))
                   for i, layer in enumerate(model.model.layers)]
        try:
            model(rows.tokens[s : s + BATCH].to(dev), use_cache=False)
        finally:
            for h in handles:
                h.remove()
    return [torch.cat(x) for x in seen]


def variance_share(X: Tensor, u: Tensor) -> float:
    """The share of X's total variance along u."""
    c = X - X.mean(0)
    return float((c @ (u / u.norm())).double().var() / c.double().var(0).sum())


def random_direction(X: Tensor, norm: float, seed: int) -> Tensor:
    """A random direction inside X's own spread, at `norm`: a Gaussian mix of the centred rows of X. Isotropic noise would
    read 0 by construction, since almost none of it lies where the model's residuals vary."""
    g = torch.randn(len(X), generator=torch.Generator(X.device).manual_seed(seed), device=X.device, dtype=X.dtype)
    r = (X - X.mean(0)).T @ g
    return r * (norm / r.norm())


def pick_layer(fit_shares: Sequence[float]) -> int:
    """The largest fit-half share, ties to the lower layer (`steering.b3`'s rule)."""
    return max(range(len(fit_shares)), key=lambda i: (fit_shares[i], -i))


def steered_share(model: Any, rows: Rows, add: tuple[int, Tensor], w0: Read, st: Read) -> dict[str, Any]:
    """On the scoring half, with the weights at W0: the steered rise over W0 as a share of the state's gain, both summaries."""
    keep = w0.row_of % 2 == 1
    w, s, got = w0.where(keep), st.where(keep), read(model, rows.half("score"), add)
    assert len(got.rank) == len(w.rank)
    rise, gain, f = got.logp - w.logp, s.logp - w.logp, w.is_first
    return {"all": share(rise, gain, w.row_of), "first": share(rise[f], gain[f], w.row_of[f])}


def steer(model: Any, rows: Rows, w0: Read, st: Read, W: Mapping[str, Tensor], seed: int) -> dict[str, Any]:
    """`W` the state's weights; the model stands at W0. v_l = the mean over the fit half's answer positions of (state − W0) block-l input, added to W0 at every
    answer position. The layer is measured: the largest fit-half share of the teacher-forced gain. Undefined on either half:
    no forward runs and no layer is named, so the do-nothing state reads None without a share to compare."""
    fit, keep = rows.half("fit"), w0.row_of % 2 == 0
    w, s = w0.where(keep), st.where(keep)
    G_fit = share(s.logp - w.logp, s.logp - w.logp, w.row_of)
    G_score = share(st.logp[~keep] - w0.logp[~keep], st.logp[~keep] - w0.logp[~keep], w0.row_of[~keep])
    out: dict[str, Any] = {"layer": None, "gain_fit": G_fit["gain"], "gain_score": G_score["gain"], "steer": None, "random": None}
    if G_fit["share"] is None or G_score["share"] is None:
        return out
    X0 = inputs_at(model, fit)
    with set_weights(model, dict(W)):
        Xs = inputs_at(model, fit)
    vs = [(a - b).mean(0) for a, b in zip(Xs, X0, strict=True)]
    del Xs
    fit_shares = [float((read(model, fit, (i, v)).logp - w.logp).double().mean()) / G_fit["gain"] for i, v in enumerate(vs)]
    best = pick_layer(fit_shares)
    v, X = vs[best], X0[best]
    r = random_direction(X, float(v.norm()), seed)
    return out | {"layer": best, "fit_shares": fit_shares, "v_norm": float(v.norm()), "input_rms": float(X.pow(2).sum(1).mean().sqrt()),
                  "variance_share": {"steer": variance_share(X, v), "random": variance_share(X, r)},
                  "steer": steered_share(model, rows, (best, v), w0, st), "random": steered_share(model, rows, (best, r), w0, st)}
