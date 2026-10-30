"""VPD's two readings of a PCFG fine-tune (E55 C2, plan reports/plans/2026-10-31_plan_e55-c2-vpd.md): the route
split's re-gating share r and the stand-in ratio, on a decomposition of LOW (`decompose.py`).

Both come from `local_faith.route_split` of the least-squares split state (`anchors.lifted`), as E54/E58 read them
(`heldout_keys.vpd_read`): r = (L01 - L00) / (T1 - T0), ratio = (L11 - L00) / (U1 - U0), with T the target, U the
unmasked and L the CI-masked model's log-odds of the answer at `=`, index 0 at W0 with LOW's components and 1 at the
state. A window is a natural row cut after its answer, (N, ANS + 1) tokens, its answer at ANS set to the label the
reading asks about (`windows`), so the target is each window's own next token (`data.NEXT`) at half = ART_POS.

The update read is the 24 decomposed matrices' change only (`matrix_update`); every other parameter of the minGPT
(embeddings, LayerNorms, biases, the output head) stays at LOW's value in every cell. `full_gain` gives the state's own
log-odds gain, so `matrix_share` = (T1 - T0) / full gain says how much of the fine-tune those matrices carry.

The cells are means over windows, so they are computed on chunks of `chunk` windows and averaged (`combine`): the gates
route_cells keeps for 258-token windows and 38K subcomponents are ~40 MB per window.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from commute import pcfg

CELLS = ("L00", "L01", "L10", "L11", "U0", "U1", "T0", "T1")
GAIN_MIN = 1e-6  # r is nan when |T1 - T0| is under this (local_faith.route_split's rule)


def windows(rows: np.ndarray, answers: np.ndarray) -> Tensor:
    """(N, ANS + 1) int64: the rows up to and including the answer, the answer replaced by `answers` (token ids)."""
    w = rows[:, : pcfg.ANS + 1].copy()
    w[:, pcfg.ANS] = answers
    return torch.from_numpy(w.astype(np.int64))


def matrix_update(state: nn.Module, low: nn.Module, keys: list[str]) -> dict[str, Tensor]:
    """W_state - W_LOW for each decomposed matrix (fp32, on LOW's device)."""
    s, b = dict(state.named_parameters()), dict(low.named_parameters())
    return {k: (s[f"{k}.weight"].detach().float() - b[f"{k}.weight"].detach().float()) for k in keys}


def outside_norm_share(state: nn.Module, low: nn.Module, keys: list[str]) -> float:
    """The share of the fine-tune's squared parameter change outside the decomposed matrices' weights."""
    inside = {f"{k}.weight" for k in keys}
    b = dict(low.named_parameters())
    sq = {n: float((p.detach().double() - b[n].detach().double()).pow(2).sum()) for n, p in state.named_parameters()}
    total = sum(sq.values())
    return sum(v for n, v in sq.items() if n not in inside) / total if total > 0 else math.nan


@torch.no_grad()
def answer_log_odds(model: nn.Module, w: Tensor, batch: int = 256) -> float:
    """Mean over windows of the plain minGPT's log-odds of each window's answer at `=`."""
    from commute.identity import log_odds

    device = next(model.parameters()).device
    tot = 0.0
    for s in range(0, w.shape[0], batch):
        x = w[s : s + batch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(x[:, : pcfg.ANS])[:, pcfg.ART_POS]
        tot += float(log_odds(lg, x[:, pcfg.ANS]).double().sum())
    return tot / w.shape[0]


def combine(parts: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Window-weighted mean of per-chunk cells: (n windows, cells) -> cells."""
    n = sum(k for k, _ in parts)
    assert n > 0 and all(k > 0 for k, _ in parts)
    return {c: sum(k * cells[c] for k, cells in parts) / n for c in CELLS}


def shares(cells: dict[str, float], ratio_floor: float) -> dict[str, float]:
    """r, the stand-in ratio and their denominators from the cells; r nan when |gain| < GAIN_MIN, ratio nan when the
    subcomponent gain U1 - U0 is under `ratio_floor` (E53's floor; inclusive at the floor, as heldout_keys.vpd_read)."""
    gain, piece = cells["T1"] - cells["T0"], cells["U1"] - cells["U0"]
    return {"gain": gain, "piece_gain": piece,
            "r": (cells["L01"] - cells["L00"]) / gain if abs(gain) >= GAIN_MIN else math.nan,
            "ratio": (cells["L11"] - cells["L00"]) / piece if piece >= ratio_floor else math.nan}  # fmt: skip


def cells_of(model: Any, lift: dict[str, Any], comps0: dict[str, tuple[Tensor, Tensor]], W0: dict[str, Tensor],
             dW: dict[str, Tensor], w: Tensor, chunk: int, batch: int) -> dict[str, float]:  # fmt: skip
    """The eight route cells of the update dW on windows w (chunked, then `combine`d)."""
    from commute.anchors import lifted
    from commute.data import NEXT
    from commute.faithfulness import model_state
    from commute.local_faith import route_cells, route_split

    target = {k: W0[k] + dW[k].to(W0[k].device, W0[k].dtype) for k in W0}
    comps1 = lifted(model, lift, dW)
    parts = []
    for s in range(0, w.shape[0], chunk):
        x = w[s : s + chunk]
        with model_state(model, target=W0, components=comps0):
            base = route_cells(model, x, pcfg.ART_POS, NEXT, batch)
        with model_state(model, target=target, components=comps1):
            rs = route_split(model, x, pcfg.ART_POS, NEXT, base, comps0, batch)
        parts.append((x.shape[0], rs["cells"]))
        del base, rs
    return combine(parts)
