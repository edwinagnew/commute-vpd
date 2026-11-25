"""The output change of an update, in nats per token, in its two directions (the reading refactor, P2.79 move 3; plan
reports/plans/2026-11-25_plan_reading-refactor.md, appendix A7). Two names for two quantities: KL is not symmetric,
and on the same state the two directions differ (kl_control part A reads the first, part B a form of the second).

  kl_base_to_ft  KL(W0 || W0 + dW): fp32 per position, summed in fp64, the mean over every (row, position) of `toks`.
                 = rl_67m run_stage1 `kl_from_base`, `panel.kl_from_w0`, `rl_arms.kl_to_ref` (E256/E260's `kl_saved`,
                 E272's `kl`).
  kl_ft_to_base  KL(W0 + dW || W0): fp64 per position (`rule_retention.kl`), the mean over `positions` per row, then
                 over rows. The direction gating_vpd's `kl_fm` reads (there on M, the masked model, not W0).
Rows, batch and positions are required: the forwards run in batches of `batch`, and fp32 logits move with it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from commute import steering as ST
from commute.rule_retention import kl as kl_rows_fp64
from commute.vpd import kl_per_position


@torch.no_grad()
def kl_base_to_ft(net: ST.Net, toks: Tensor, dW: ST.DW, batch: int) -> float:
    """KL(W0 || W0 + dW) in nats per token, the mean over every position of every row of `toks`."""
    tot, n = 0.0, 0
    for b in toks.split(batch):
        p = net.run(b)[0]
        with net.at(dW):
            q = net.run(b)[0]
        kl = kl_per_position(p, q)
        tot, n = tot + float(kl.double().sum()), n + kl.numel()
    return tot / n


@torch.no_grad()
def kl_ft_to_base(net: ST.Net, toks: Tensor, dW: ST.DW, batch: int, positions: slice) -> float:
    """KL(W0 + dW || W0) in nats per token, fp64: per row the mean over `positions`, then the mean over rows."""
    per: list[Tensor] = []
    for b in toks.split(batch):
        base = net.run(b)[0][:, positions]
        with net.at(dW):
            per.append(kl_rows_fp64(net.run(b)[0][:, positions], base).cpu())
    return float(torch.cat(per).mean())
