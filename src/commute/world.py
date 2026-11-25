"""The 67M world every reading rests on, built once by one call (the reading refactor, P2.79 move 1; plan
reports/plans/2026-11-25_plan_reading-refactor.md, appendix A1). It replaces reaching the model through the chain of
experiment run.py files (E214 -> E221 -> E226 -> E242 -> E257 -> E268 `world()`).

`World.load(device)`: one ComponentModel, requires_grad off (`vpd.load_component_model`), wrapped as `steering.Net`;
W0 from the checkpoint file, checked bit for bit against the live model's weights; the released components (uv) and
weight deltas; `bases(tol)` the least-squares split's fp64 SVD bases (`kstar_vpd.bases`), tol a required argument.
`weights()`: W0 and uv from the checkpoint with no model built, for a CPU job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import vpd

DW = dict[str, Tensor]
UV = dict[str, tuple[Tensor, Tensor]]


def weights() -> tuple[DW, UV]:
    """W0 and (U (C, d_out), V (d_in, C)) per decomposed matrix from the checkpoint file, CPU fp32 (`vpd.checkpoint_weights`)."""
    return vpd.checkpoint_weights()


@dataclass
class World:
    net: Any  # steering.Net: the ComponentModel and its target model, run and edited
    W0: DW  # on the device, equal bit for bit to the checkpoint's
    uv: UV  # the released components, live
    delta: DW  # W_target - sum_c U_c V_c per matrix (`vpd.weight_deltas`)

    @classmethod
    def load(cls, device: str) -> World:
        from commute import steering as ST

        net = ST.Net(device)
        W0 = weights()[0]
        assert sorted(W0) == sorted(net.W0) and all(torch.equal(W0[k], net.W0[k].cpu()) for k in W0), "checkpoint W0 differs from the live model's"
        return cls(net, {k: w.to(device) for k, w in W0.items()}, vpd.component_weights(net.model), vpd.weight_deltas(net.model))

    @property
    def model(self) -> Any:
        return self.net.model

    def bases(self, tol: float) -> dict[str, Any]:
        """The split's SVD bases per matrix over every subcomponent, fp64; singular values under tol x max are zero."""
        return KV.bases(self.uv, tol)
