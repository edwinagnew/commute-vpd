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

Amendment 2 (2026-10-27) adds the target census at the bottom: on-positions and next-token counts at a few ids for
every subcomponent, and each piece's target ratio (P8).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.recruit import ContrastAcc, Sel, layer_or_none

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


# ----------------------------------------------------------------------------- E38 Amendment 2: the target census
# reports/plans/2026-10-27_plan_e38-amend2.md. Only what P8 reads: per subcomponent of every matrix its on-positions
# (g > ON) and how many of them have each of a few token ids as the NEXT token; the same counts over all positions
# read; positions 0..T-2 of each row, as `Census`. The target ratio of a piece is (its on-positions followed by the
# target / its on-positions) / (positions followed by the target / positions), None for a rare piece (< MIN_ON).

# P8 as amended before any run (results/predictions_amend2_p8.json): a piece is target-enriched with >= MIN_ON
# on-positions, >= MIN_HITS of them followed by the target and target ratio >= MIN_RATIO; a set passes when its
# enriched fraction (over all its pieces, rare ones counting as not enriched) is >= P8_TIMES x the pooled random sets'
# and >= P8_MIN. The same rule for the contrast control, which unscores a task when it fails.
MIN_HITS, MIN_RATIO, P8_TIMES, P8_MIN = 5, 3.0, 2.0, 0.10


class TargetCounts:
    def __init__(self, ids: list[int]) -> None:
        self.ids = list(ids)
        self.n_on: dict[str, Tensor] = {}  # matrix -> (C,) long
        self.next: dict[str, Tensor] = {}  # matrix -> (C, len(ids)) long
        self.base = torch.zeros(len(self.ids), dtype=torch.long)
        self.n_all = 0

    def add(self, g: dict[str, Tensor], rows: Tensor) -> None:
        """One batch: g[k] (B, T, C) from `compute_ci`, rows (B, T) token ids."""
        nxt = rows.to(next(iter(g.values())).device)[:, 1:]
        hit = torch.stack([nxt == i for i in self.ids], dim=-1).float()  # (B, T-1, J), exact 0/1
        for k, v in g.items():
            on = (v[:, :-1] > ON).float()
            n_on, nxt_k = on.sum(dim=(0, 1)).long().cpu(), torch.einsum("btc,btj->cj", on, hit).round().long().cpu()
            self.n_on[k] = self.n_on[k] + n_on if k in self.n_on else n_on
            self.next[k] = self.next[k] + nxt_k if k in self.next else nxt_k
        self.base += hit.sum(dim=(0, 1)).long().cpu()
        self.n_all += int(nxt.numel())

    def ratio(self, k: str, c: int, token: int) -> float | None:
        n_on, j = int(self.n_on[k][c]), self.ids.index(token)
        if n_on < MIN_ON:
            return None
        return (int(self.next[k][c, j]) / n_on) / (int(self.base[j]) / self.n_all)

    def ratios(self, sel: Sel, token: int) -> list[float]:
        """The target ratios of the non-rare pieces of `sel`."""
        out = [self.ratio(k, int(c), token) for k, v in sel.items() for c in v.tolist()]
        return [r for r in out if r is not None]

    def enriched(self, k: str, c: int, token: int) -> bool:
        r = self.ratio(k, c, token)
        return r is not None and int(self.next[k][c, self.ids.index(token)]) >= MIN_HITS and r >= MIN_RATIO

    def flags(self, sel: Sel, token: int) -> list[bool]:
        """Target-enriched or not, for every piece of `sel` (rare pieces are not)."""
        return [self.enriched(k, int(c), token) for k, v in sel.items() for c in v.tolist()]

    def pieces(self, sel: Sel) -> dict[str, dict[str, Any]]:
        """'matrix:index' -> n_on and the next-token counts at `ids`, for the pieces of `sel` (what the JSON keeps)."""
        return {f"{k}:{int(c)}": {"n_on": int(self.n_on[k][c]), "next": self.next[k][c].tolist()}
                for k, v in sel.items() for c in v.tolist()}


def target_census(model: Any, rows: Tensor, ids: list[int], masks: dict[str, Tensor], batch: int = 8,
                  log: Any = print) -> tuple[TargetCounts, dict[str, ContrastAcc]]:
    """One `compute_ci` pass over the rows at the model's current weights: the `TargetCounts` at `ids`, and per mask
    (B, T) a `recruit.ContrastAcc` (mean g at the masked positions minus mean g over all positions)."""
    from commute.vpd import compute_ci

    counts, accs = TargetCounts(ids), {n: ContrastAcc() for n in masks}
    for s in range(0, rows.shape[0], batch):
        ci = compute_ci(model, rows[s : s + batch], chunk=batch)
        counts.add(ci.g, rows[s : s + batch])
        for n, acc in accs.items():
            acc.add(ci.g, masks[n][s : s + batch])
        del ci
        if (s // batch) % 50 == 0:
            log(f"[census] rows {s + batch}/{rows.shape[0]}")
    return counts, accs


def p8_task(counts: TargetCounts, S: Sel, random: list[Sel], control: Sel, token: int) -> dict[str, Any]:
    """One task's P8 (amended rule): target-enriched fractions of S, of the random sets' pieces pooled and of the
    control; the control passing scores the task, S passing is P8 there. Median target ratios over the non-rare
    pieces are reported only."""
    import statistics

    def frac(x: list[bool]) -> float:
        return sum(x) / len(x) if x else float("nan")

    def med(x: list[float]) -> float:
        return statistics.median(x) if x else float("nan")

    def passes(f: float, f_random: float) -> bool:
        return f >= P8_TIMES * f_random and f >= P8_MIN

    f_S, f_c = frac(counts.flags(S, token)), frac(counts.flags(control, token))
    f_r = frac([x for s in random for x in counts.flags(s, token)])
    control_ok = passes(f_c, f_r)
    r_S, r_c = counts.ratios(S, token), counts.ratios(control, token)
    r_r = [x for s in random for x in counts.ratios(s, token)]
    return {"frac_S": f_S, "frac_random": f_r, "frac_control": f_c, "control_ok": control_ok,
            "pass": passes(f_S, f_r) if control_ok else "unscored",
            "median_S": med(r_S), "median_random": med(r_r), "median_control": med(r_c),
            "n_nonrare": [len(r_S), len(r_r), len(r_c)]}
