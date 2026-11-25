"""W0's standing on a task's read rows before any update: the prior (the reading refactor, P2.79 move 2; plan
reports/plans/2026-11-25_plan_reading-refactor.md, appendix A9). "Prior" names two quantities in the project, so they
get two names here:

  prior_geo        P, the geometric mean over rows of W0's P(target) at the read position (`keydep.geo_rate` of fp32
                   log P; E133 `prior_vs_share.forward_prior`). The scorecard plots prior_log10_geo(P), log10 of it;
                   several of its P are copied from older JSON, so the log10 takes a P, computed or saved.
  prior_mean_lo    the mean over rows of W0's reading in a unit (natural log-odds, or the two-label contrast): E211
                   `realistic_reads.Reader.state` "prior", E272 `convergence_panel` "prior", procedural_rl.
The forwards are the fp32 target model (`steering.Net.run`) in batches of `batch` (E133 read all rows at once).
"""

from __future__ import annotations

import math

import torch

from commute import steering as ST
from commute.keydep import geo_rate
from commute.windows import Rows


def _at_w0(net: ST.Net, rows: Rows, unit: ST.Readout, batch: int) -> torch.Tensor:
    """(N,) float64 CPU: `unit` of W0's logits at rows.pos against each row's target."""
    assert rows.tgt is not None, f"{rows.task} {rows.kind}: no target per row"
    n = len(rows.toks)
    return torch.cat([unit(net.run(rows.toks[s : s + batch])[0], rows.pos, rows.tgt[s : s + batch]) for s in range(0, n, batch)])


def prior_geo(net: ST.Net, rows: Rows, batch: int) -> float:
    """P: exp of the mean over rows of W0's log P(target) at rows.pos."""
    return geo_rate(_at_w0(net, rows, ST.logp_at, batch))


def prior_log10_geo(p: float) -> float:
    """The scorecard's prior axis: log10 of a P from `prior_geo` or a saved one (E133 `point`)."""
    return math.log10(p)


def prior_mean_lo(net: ST.Net, rows: Rows, unit: ST.Readout, batch: int) -> float:
    """The mean over rows of W0's `unit` at rows.pos (`steering.logodds_at`: natural log-odds; `contrast_at(ab)`)."""
    return float(_at_w0(net, rows, unit, batch).mean())
