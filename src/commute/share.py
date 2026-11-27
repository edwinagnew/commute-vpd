"""The re-gating share r of an update on a task's read rows (the Terms' "re-gating share"), one definition for the 67M
and for Qwen3-1.7B, with its settings as required arguments (the reading refactor, P2.79 move 4; one share for both
models, P2.90).

r = mean over answers of (L01 - L00) / mean over answers of (T1 - T0), each read in one unit. L is the CI-masked model
(Delta out) with the released components and Gamma's gates from W0 (L00) or from W0 + dW (L01); T the target model at
W0 (T0) or W0 + dW (T1).
  answer_logodds  the share's unit: the whole-answer log-odds log P(ans) - log(1 - P(ans)), log P(ans) the sum of the
                  answer tokens' teacher-forced log-probs (`Spans` say where each answer sits). A one-token answer
                  reads `steering.logodds_at` exactly.
  cells           T0, T1, L00, L01 per item and unit from any model's four forwards (`Pass`): the 67M's below, Qwen's
                  `share_reading.lab_passes` (experiments/qwen_reading/lab.py `Lab.passes`).
  share_cells     the 67M's: `vpd.compute_ci` and `local_faith.masked_logits` (bf16 autocast) in batches of `batch`
                  (`masked_reads.share_cells` calls it; `route_cells_by` and E136's `route_split` make the same
                  forwards). The batch moves bf16 numbers: E136 read E53's tasks at 32, masked_reads at 64.
  share           the ratio over a gain the caller names: `cells_gain` (T1 - T0 of the same forwards; E136's `.trained`,
                  E140, E211, E272) or an fp32 gain (the target model alone; E136's untrained shares).
  Floor           |gain| >= 1e-6 (`route_split`, two-sided) or gain >= 1 (`MIN_GAIN`, `masked_reads.share`, one-sided).
Units: a `steering.Readout` per name, (logits, pos, targets) -> (N,) float64 CPU: `answer_at` (the share's unit on
one-token rows), `steering.logodds_at` (the same numbers) or `steering.contrast_at(ab)` (E140's two-label contrast).

Not here: `local_faith.route_split`, the four-cell diagnostic (d, r, i and the two leftovers). The KL ratio
(K00 - K01) / (K00 - K11) that App H's Qwen share was read in until P2.90 stays in `qwen_reading.shares` for the runs
that read it (experiments/qwen_library_share and its followers), and is no longer the Qwen share.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute import steering as ST
from commute import vpd
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.local_faith import masked_logits
from commute.windows import Rows

Cells = dict[str, dict[str, Tensor]]  # cell (T0, T1, L00, L01) -> unit -> (items,) float64 CPU
CELLS = ("T0", "L00", "T1", "L01")
Pass = Callable[[int, int], Iterator[tuple[str, Tensor]]]  # rows [s, e) -> (cell, logits (e - s, L, V)) per cell
Read = Callable[[Tensor, int, int], Tensor]  # (logits of rows [s, e), s, e) -> (items of those rows,) float64 CPU


# ----------------------------------------------------------------------------- the unit


@dataclass(frozen=True)
class Spans:
    """The answers read on a set of rows: answer a is ids[a, :n[a]], predicted from positions start[a] .. start[a] +
    n[a] - 1 of row[a] (teacher-forced: the answer's own tokens are the row's next tokens, or, on the 67M, a target
    the row does not hold). ids is padded with 0 past n."""

    row: Tensor  # (A,) int64
    start: Tensor  # (A,) int64
    ids: Tensor  # (A, J) int64
    n: Tensor  # (A,) int64, each >= 1

    def __post_init__(self) -> None:
        A = len(self.row)
        assert self.start.shape == self.n.shape == (A,) and self.ids.shape[0] == A, "one start, length and id row per answer"
        assert A == 0 or (int(self.n.min()) >= 1 and int(self.n.max()) <= self.ids.shape[1]), "every answer 1..J tokens"

    @staticmethod
    def at(pos: int, tgt: Tensor) -> Spans:
        """One one-token answer per row: tgt[i], read at pos (the 67M's rows)."""
        N = len(tgt)
        return Spans(torch.arange(N), torch.full((N,), pos), tgt.long().cpu()[:, None], torch.ones(N, dtype=torch.long))

    def take(self, s: int, e: int) -> Spans:
        """The answers of rows [s, e), their rows counted from s."""
        k = (self.row >= s) & (self.row < e)
        return Spans(self.row[k] - s, self.start[k], self.ids[k], self.n[k])


def spans_from_marks(toks: Tensor, answer: Tensor, first: Tensor, end_id: int) -> Spans:
    """The whole answers marked in packed rows (`qwen_comparators.Rows`: answer[i, p] when position p predicts an
    answer token, first[i, p] when it predicts an answer's first token). An answer is a run of marked positions that
    starts at a `first` position and whose last position predicts end_id (the trained response ends with it); a run cut
    by the row's start or end is not a whole answer and is left out."""
    row, start, n, ids = [], [], [], []
    for i in range(len(toks)):
        a, f, t = answer[i].tolist(), first[i].tolist(), toks[i].tolist()
        p = 0
        while p < len(a):
            if not a[p]:
                p += 1
                continue
            q = p
            while q + 1 < len(a) and a[q + 1] and not f[q + 1]:
                q += 1
            if f[p] and q + 1 < len(t) and t[q + 1] == end_id:
                row.append(i), start.append(p), n.append(q - p + 1), ids.append(t[p + 1 : q + 2])
            p = q + 1
    J = max(n, default=1)
    pad = torch.zeros(len(ids), J, dtype=torch.long)
    for k, x in enumerate(ids):
        pad[k, : len(x)] = torch.tensor(x)
    return Spans(torch.tensor(row, dtype=torch.long), torch.tensor(start, dtype=torch.long), pad, torch.tensor(n, dtype=torch.long))


def answer_logodds(logits: Tensor, spans: Spans) -> Tensor:
    """log P(ans) - log(1 - P(ans)) per answer, (A,) float64 CPU, from logits (rows, L, V) of the rows spans.row counts.
    Per token t, d_t = -`identity.log_odds` (exact near p = 1); log p_t = -softplus(d_t), log(1 - p_t) =
    -softplus(-d_t); log(1 - P) builds up as 1 - P_{<=t} = (1 - P_{<t}) + P_{<t} (1 - p_t), in log space, so it stays
    exact near P = 1 (log(-expm1(log P)) would lose it when log P rounds to 0). A one-token answer reads log_odds
    itself, as `steering.logodds_at`."""
    dev, J = logits.device, spans.ids.shape[1]
    row, start = spans.row.to(dev), spans.start.to(dev)
    one = log_odds(logits[row, start], spans.ids[:, 0].to(dev)).double().cpu()
    if J == 1:
        return one
    j = torch.arange(J)
    pos = (spans.start[:, None] + j).clamp(max=logits.shape[1] - 1).to(dev)
    d = -log_odds(logits[row[:, None], pos], spans.ids.to(dev)).double().cpu()  # (A, J)
    on = j[None] < spans.n[:, None]
    log_p = torch.zeros(len(d), dtype=torch.float64)
    log_q = torch.full((len(d),), float("-inf"), dtype=torch.float64)  # log(1 - P) of the tokens so far
    for t in range(J):
        log_q = torch.where(on[:, t], torch.logaddexp(log_q, log_p - F.softplus(-d[:, t])), log_q)
        log_p = torch.where(on[:, t], log_p - F.softplus(d[:, t]), log_p)
    return torch.where(spans.n == 1, one, log_p - log_q)


def answer_at(logits: Tensor, pos: int, tgt: Tensor) -> Tensor:
    """`answer_logodds` as a `steering.Readout` on one-token rows (equal to `steering.logodds_at`)."""
    return answer_logodds(logits, Spans.at(pos, tgt))


# ----------------------------------------------------------------------------- the cells


@dataclass(frozen=True)
class Floor:
    """The gain under which a share is not read (nan)."""

    value: float
    two_sided: bool  # |gain| >= value, or gain >= value

    def passes(self, gain: float) -> bool:
        return (abs(gain) if self.two_sided else gain) >= self.value and gain != 0


@torch.no_grad()
def cells(passes: Pass, n_rows: int, batch: int, units: Mapping[str, Read]) -> Cells:
    """T0, T1, L00, L01 per item and unit: `passes` over rows [s, s + batch), each cell's logits read by every unit
    as it comes (so at most the two cells of one weight set are held)."""
    out: dict[str, list[dict[str, Tensor]]] = {c: [] for c in CELLS}
    for s in range(0, n_rows, batch):
        e = min(n_rows, s + batch)
        for c, lg in passes(s, e):
            out[c].append({u: f(lg, s, e) for u, f in units.items()})
    return {c: {u: torch.cat([p[u] for p in parts]) for u in units} for c, parts in out.items()}


def model_passes(model: Any, W0: Mapping[str, Tensor], dW: ST.DW, toks: Tensor) -> Pass:
    """The 67M's four forwards of rows toks[s:e]: Gamma and the target at W0 (T0) and W0 + dW (T1), then the masked
    model under each one's gates (L00, L01)."""
    device = next(model.parameters()).device
    W1 = {k: w + dW[k].to(w.device, w.dtype) for k, w in W0.items()}

    def passes(s: int, e: int) -> Iterator[tuple[str, Tensor]]:
        x = toks[s:e].to(device)
        for T, L, W in (("T0", "L00", None), ("T1", "L01", W1)):
            with model_state(model, target=W):  # W None: the model as it stands, W0
                ci = vpd.compute_ci(model, x, chunk=x.shape[0])
                m = masked_logits(model, x, ci.g, None, None)
            yield T, ci.logits
            yield L, m

    return passes


def share_cells(model: Any, W0: Mapping[str, Tensor], dW: ST.DW, rows: Rows, units: Mapping[str, ST.Readout], batch: int) -> Cells:
    """T0, T1, L00, L01 per row of rows.toks and unit, read at rows.pos (see the module docstring)."""
    tgt = rows.tgt
    assert tgt is not None, f"{rows.task} {rows.kind}: no target per row"
    reads = {u: (lambda lg, s, e, f=f: f(lg, rows.pos, tgt[s:e])) for u, f in units.items()}
    return cells(model_passes(model, W0, dW, rows.toks), len(rows.toks), batch, reads)


def cells_gain(cells: Cells, unit: str) -> float:
    """mean(T1 - T0) of the cells' own forwards."""
    return float((cells["T1"][unit] - cells["T0"][unit]).mean())


def numerator(cells: Cells, unit: str) -> float:
    """mean(L01 - L00) in `unit`: the share's numerator."""
    return float((cells["L01"][unit] - cells["L00"][unit]).mean())


def share(cells: Cells, unit: str, gain: float, floor: Floor) -> float:
    """r = mean(L01 - L00) / gain in `unit`; nan when the gain does not pass `floor`."""
    return numerator(cells, unit) / gain if floor.passes(gain) else float("nan")
