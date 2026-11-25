"""The re-gating share r of an update on a task's read rows (the Terms' "re-gating share"), one definition with its
settings as required arguments (the reading refactor, P2.79 move 4; plan reports/plans/2026-11-25_plan_reading-refactor.md,
appendix A4).

r = mean(L01 - L00) / gain, read at rows.pos in one unit. L is the CI-masked model (Delta out) with the released
components and Gamma's gates from W0 (L00) or from W0 + dW (L01); T the target model at W0 (T0) or W0 + dW (T1).
  share_cells     T0, T1, L00, L01 per row and unit, from `vpd.compute_ci` and `local_faith.masked_logits` (bf16
                  autocast) in batches of `batch` (`masked_reads.route_cells_by`'s calls; E136's `route_split` makes
                  the same forwards). The batch moves bf16 numbers: E136 read E53's tasks at 32, masked_reads at 64.
  share           the ratio over a gain the caller names: `cells_gain` (T1 - T0 of the same bf16 forwards; E136's
                  `.trained`, E140, E211, E272) or an fp32 gain (the target model alone; E136's untrained shares).
  Floor           |gain| >= 1e-6 (`route_split`, two-sided) or gain >= 1 (`MIN_GAIN`, `masked_reads.share`, one-sided).
  untrained_share the same do-nothing form on W0's SVD and its rotation (`untrained.do_nothing_cells`, fp32 hooks,
                  E136's top-k_m gates), over the fp32 gain of those forwards.
Units: a `steering.Readout` per name, (logits, pos, targets) -> (N,) float64 CPU: `steering.logodds_at` ("lo", the
share's unit) or `steering.contrast_at(ab)` (E140's two-label contrast).

Not here: `local_faith.route_split`, the four-cell diagnostic (d, r, i and the two leftovers), and the Qwen re-gating
share, a KL ratio (K00 - K01) / (K00 - K11), a different quantity named `qwen_reading.kl_share`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from commute import steering as ST
from commute import untrained as U
from commute import vpd
from commute.data import LAST
from commute.faithfulness import model_state
from commute.local_faith import masked_logits
from commute.windows import Rows

Cells = dict[str, dict[str, Tensor]]  # cell (T0, T1, L00, L01) -> unit -> (N,) float64 CPU


@dataclass(frozen=True)
class Floor:
    """The gain under which a share is not read (nan)."""

    value: float
    two_sided: bool  # |gain| >= value, or gain >= value

    def passes(self, gain: float) -> bool:
        return (abs(gain) if self.two_sided else gain) >= self.value and gain != 0


@torch.no_grad()
def share_cells(model: Any, W0: Mapping[str, Tensor], dW: ST.DW, rows: Rows, units: Mapping[str, ST.Readout], batch: int) -> Cells:
    """T0, T1, L00, L01 per row of rows.toks and unit, read at rows.pos (see the module docstring)."""
    assert rows.tgt is not None, f"{rows.task} {rows.kind}: no target per row"
    device = next(model.parameters()).device
    W1 = {k: w + dW[k].to(w.device, w.dtype) for k, w in W0.items()}
    out: dict[str, list[dict[str, Tensor]]] = {c: [] for c in ("T0", "L00", "T1", "L01")}
    for s in range(0, len(rows.toks), batch):
        x, tgt = rows.toks[s : s + batch].to(device), rows.tgt[s : s + batch]
        for T, L, W in (("T0", "L00", None), ("T1", "L01", W1)):
            with model_state(model, target=W):  # W None: the model as it stands, W0
                ci = vpd.compute_ci(model, x, chunk=x.shape[0])
                m = masked_logits(model, x, ci.g, None, None)
            out[T].append({u: f(ci.logits, rows.pos, tgt) for u, f in units.items()})
            out[L].append({u: f(m, rows.pos, tgt) for u, f in units.items()})
    return {c: {u: torch.cat([p[u] for p in parts]) for u in units} for c, parts in out.items()}


def cells_gain(cells: Cells, unit: str) -> float:
    """mean(T1 - T0) of the cells' own forwards (bf16 autocast)."""
    return float((cells["T1"][unit] - cells["T0"][unit]).mean())


def numerator(cells: Cells, unit: str) -> float:
    """mean(L01 - L00) in `unit`: the share's numerator."""
    return float((cells["L01"][unit] - cells["L00"][unit]).mean())


def share(cells: Cells, unit: str, gain: float, floor: Floor) -> float:
    """r = mean(L01 - L00) / gain in `unit`; nan when the gain does not pass `floor`."""
    return numerator(cells, unit) / gain if floor.passes(gain) else float("nan")


def untrained_share(model: Any, W0: Mapping[str, Tensor], dW: ST.DW, rows: Rows, decomps: Mapping[str, U.Decomposition], k_m: Mapping[str, int],
                    unit: Literal["lo"] | ST.Readout, batch: int, floor: Floor) -> dict[str, float]:  # fmt: skip
    """r per decomposition of W0 that needs no training, over the fp32 gain of the same forwards ("gain_fp32" beside).
    unit "lo" is `do_nothing_cells`' own log-odds (summed on the device, E136's path); a Readout reads that unit."""
    assert rows.tgt is not None, f"{rows.task} {rows.kind}: no target per row"
    w = torch.cat([rows.toks, rows.tgt[:, None]], 1)  # the target rides as a last column that no forward reads
    u = None if unit == "lo" else unit
    c0 = U.do_nothing_cells(model, w, rows.pos, LAST, decomps, k_m, batch, rows.toks.shape[1], None, u)
    target = {k: x + dW[k].to(x.device) for k, x in W0.items()}
    c1 = U.do_nothing_cells(model, w, rows.pos, LAST, decomps, k_m, batch, rows.toks.shape[1], target, u)
    gain = c1["T"] - c0["T"]
    return {n: (c1[n] - c0[n]) / gain if floor.passes(gain) else float("nan") for n in decomps} | {"gain_fp32": gain}
