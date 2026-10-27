"""E38's piece census (`experiments/recruitment_test`, stage 3): where the original model turns each subcomponent on in
ordinary text, and what it predicts there. Plan: reports/plans/2026-10-27_plan_regating-recruitment.md, "Stage 3".

Over Pile rows at W0, with `vpd.compute_ci`, for each listed subcomponent ("piece"): the positions where its importance
g exceeds ON are its on-positions. Positions 0..T-2 of each row are read (the last has no next token). Per piece:
  on_rate     on-positions / positions read;
  top_next    the tokens most over-represented as the NEXT token at its on-positions, by the ratio of their rate there
              to their base rate over all positions read (only tokens seen at least MIN_COUNT times at its
              on-positions, so one stray occurrence cannot top the list);
  top_cur     the same for the CURRENT token.
A piece with fewer than MIN_ON on-positions is 'rare'. Classes: 'target' (the target among the top three next tokens),
'trigger' (a trigger token among the top three current tokens), 'both', 'neither', 'rare'.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.recruit import Sel, layer_or_none

ON, MIN_ON, MIN_COUNT, TOP, TOP_CLASS = 0.5, 50, 5, 10, 3
CLASSES = ("target", "trigger", "both", "neither", "rare")


class Census:
    """Counters for the pieces of `sel` (matrix -> indices): on-positions, and next / current token counts there."""

    def __init__(self, sel: Sel, vocab: int, device: str | torch.device) -> None:
        self.keys = sorted(k for k, v in sel.items() if v.numel())
        self.idx = {k: sel[k].to(device) for k in self.keys}
        self.pieces = [(k, int(c)) for k in self.keys for c in sel[k].tolist()]
        P, self.V = len(self.pieces), vocab
        self.n_on = torch.zeros(P, dtype=torch.long, device=device)
        self.next = torch.zeros(P * vocab, dtype=torch.long, device=device)
        self.cur = torch.zeros(P * vocab, dtype=torch.long, device=device)
        self.base_next = torch.zeros(vocab, dtype=torch.long, device=device)
        self.base_cur = torch.zeros(vocab, dtype=torch.long, device=device)
        self.n_all = 0

    def add(self, g: dict[str, Tensor], rows: Tensor) -> None:
        """One batch: g[k] (B, T, C) from `compute_ci`, rows (B, T) token ids."""
        rows = rows.to(self.n_on.device)
        cur, nxt = rows[:, :-1], rows[:, 1:]
        on = torch.cat([g[k][:, :-1][..., self.idx[k]] > ON for k in self.keys], dim=-1)  # (B, T-1, P)
        self.n_on += on.sum(dim=(0, 1))
        b, t, p = on.nonzero(as_tuple=True)
        ones = torch.ones_like(p)
        self.next.index_add_(0, p * self.V + nxt[b, t], ones)
        self.cur.index_add_(0, p * self.V + cur[b, t], ones)
        self.base_next += torch.bincount(nxt.reshape(-1), minlength=self.V)
        self.base_cur += torch.bincount(cur.reshape(-1), minlength=self.V)
        self.n_all += int(cur.numel())


def run_census(model: Any, rows: Tensor, sel: Sel, batch: int = 8, log: Any = print) -> Census:
    """One pass of `compute_ci` over the rows at the model's current weights, counting for every piece of `sel`."""
    from commute.vpd import compute_ci

    device = next(model.parameters()).device
    c: Census | None = None
    for s in range(0, rows.shape[0], batch):
        ci = compute_ci(model, rows[s : s + batch], chunk=batch)
        c = c or Census(sel, int(ci.logits.shape[-1]), device)  # the vocabulary is the logits' width
        c.add(ci.g, rows[s : s + batch])
        del ci
        if (s // batch) % 50 == 0:
            log(f"[census] rows {s + batch}/{rows.shape[0]}")
    assert c is not None, "no rows"
    return c


def enriched(counts: Tensor, n_on: int, base: Tensor, n_all: int, top: int = TOP,
             min_count: int = MIN_COUNT) -> list[tuple[int, float, int]]:
    """(token id, ratio of its rate at the on-positions to its base rate, count there) for the `top` largest ratios
    among tokens with at least `min_count` occurrences at the on-positions."""
    counts, base = counts.double().cpu(), base.double().cpu()
    ratio = torch.where((counts >= min_count) & (base > 0), (counts / max(n_on, 1)) / (base / n_all), torch.tensor(-1.0))
    order = ratio.argsort(descending=True)[:top]
    return [(int(i), float(ratio[i]), int(counts[i])) for i in order if ratio[i] > 0]


def pieces_table(c: Census) -> dict[tuple[str, int], dict[str, Any]]:
    """Per piece: matrix, layer, on_rate, n_on, top_next and top_cur (token id, ratio, count)."""
    out = {}
    for i, (k, idx) in enumerate(c.pieces):
        n_on = int(c.n_on[i])
        sl = slice(i * c.V, (i + 1) * c.V)
        out[(k, idx)] = {"matrix": k, "index": idx, "layer": layer_or_none(k), "n_on": n_on, "on_rate": n_on / c.n_all,
                         "top_next": enriched(c.next[sl], n_on, c.base_next, c.n_all),
                         "top_cur": enriched(c.cur[sl], n_on, c.base_cur, c.n_all)}
    return out


def classify(row: dict[str, Any], target: int, triggers: set[int]) -> str:
    if row["n_on"] < MIN_ON:
        return "rare"
    is_t = target in [t for t, _, _ in row["top_next"][:TOP_CLASS]]
    is_g = bool(triggers & {t for t, _, _ in row["top_cur"][:TOP_CLASS]})
    return "both" if is_t and is_g else "target" if is_t else "trigger" if is_g else "neither"


def class_shares(table: dict[tuple[str, int], dict[str, Any]], sel: Sel, target: int, triggers: set[int]) -> dict[str, Any]:
    """Each class's count and share among the pieces of `sel`, and the share of pieces whose class names the target
    ('target' or 'both')."""
    labels = [classify(table[(k, int(c))], target, triggers) for k, v in sel.items() for c in v.tolist()]
    n = max(len(labels), 1)
    counts = {c: labels.count(c) for c in CLASSES}
    return {"n": len(labels), "counts": counts, "shares": {c: v / n for c, v in counts.items()},
            "target_share": (counts["target"] + counts["both"]) / n}
