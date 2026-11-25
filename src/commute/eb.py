"""E_b, the definition's kept share, at any featurization (the reading refactor, P2.79 move 6; plan
reports/plans/2026-11-25_plan_reading-refactor.md, appendix A5).

`write_sums`: one write pass over D in F's own run (`stand_in.write_change`) on any site list with `module`, `idx`,
`gain_sums`, `change_sums` and `leftover` (`stand_in.VPDSite`, `needed_neurons.HeadSite` / `MLPSite`,
`sae_feat.SAESite`), with the featurization's changed set. `eb`: `stand_in.explained_share` over K & changed, with or
without the leftover variables (each matrix's or layer's remainder, each SAE site's error term). The set read may be
the swap test's K (`needed.needed_set`) or the old K* (`kstar_vpd.search`, E263's SAE row): the caller names which.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import stand_in as SI


@dataclass(frozen=True)
class Sums:
    num: Tensor  # (n,) fp64, sum over every position of D of <write_F_u, write_M_u>
    den: Tensor  # (n,) fp64, sum of |write_M_u|^2
    dd: Tensor  # (n,) fp64, sum of |write_F_u - write_M_u|^2
    changed: Tensor  # (n,) bool: the variables whose mechanism F changed
    leftover: Tensor | None  # (n,) bool: remainders and error terms; None for sums saved without their sites


@torch.no_grad()
def write_sums(net: Any, sites: Sequence[Any], toks: Tensor, batch: int, changed: Tensor) -> Sums:
    """The sums over every position of `toks` (D, on net's device) in F's run `net`, `batch` windows per forward;
    changed (n,) bool, n every variable of the featurization (the sites may leave some out, e.g. VPD's embedding)."""
    n = len(changed)
    num, den, dd = SI.write_change(net, list(sites), n, list(toks.split(batch)))
    left = torch.zeros(n, dtype=torch.bool)
    left[[s.leftover for s in sites if s.leftover is not None]] = True
    return Sums(num, den, dd, changed.cpu().bool(), left)


def eb(sums: Sums, K: Tensor, leftovers: bool) -> float:
    """E_b over K & changed; leftovers False drops the remainders and error terms (E277's `E_b_K_no_rem`, E298's form)."""
    sel = K.bool() & sums.changed
    if not leftovers:
        assert sums.leftover is not None, "these sums carry no leftover mask"
        sel = sel & ~sums.leftover
    return SI.explained_share(sums.num, sums.den, sums.dd, sel)
