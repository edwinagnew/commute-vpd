"""E52's neuron patching (B2) and its two apparatus controls, shared since 2026-10-30 with E56
(`experiments/keydependent_patch`). Moved, not changed, from `experiments/baseline_neurons/run.py` (`flat_mlp`,
`neuron_edit`, the ranking and grid of `b2`, and stage 0's identity and full residual patches), which now calls them;
E52's idle share stays in its run.py.

Target model only, fp32. Neurons are the 4 x 3072 MLP GELU outputs (`steering.Net` site `mlp<l>`), flat index =
layer x 3072 + neuron. Ranked by |mean over the fit half of (tuned - W0) activation at the answer position|; the top k
take the tuned model's activations (its own forward on the same window) at every position of a W0 forward; share = the
rise of log P(target) over W0 on the scoring half divided by G_score.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from commute.steering import D_MLP, DW, N_LAYERS, Edit, Net, Task, read, share


def flat_mlp(c: dict[str, Tensor], pos: int) -> Tensor:
    return torch.cat([c[f"mlp{i}"][:, pos] for i in range(N_LAYERS)], -1)


def neuron_edit(c1: dict[str, Tensor], idx: Tensor) -> Edit:
    edit: Edit = {}
    for i in range(N_LAYERS):
        sel = (idx[(idx >= i * D_MLP) & (idx < (i + 1) * D_MLP)] - i * D_MLP).to(c1[f"mlp{i}"].device)
        if len(sel):
            edit[f"mlp{i}"] = lambda x, src=c1[f"mlp{i}"], sel=sel: x.index_copy(2, sel, src.index_select(2, sel))
    return edit


def neuron_patch(net: Net, t: Task, dW: DW, G_score: float, grid: list[int]) -> dict[str, Any]:
    """B2's grid: per k, the share of the top-k neurons' patch; k50 = the smallest k on the grid with share >= 0.5
    (inf if none), k50_index its grid index (len(grid) if none). Also the ranking (`order`, `shift`) and W0's fit-half
    activations at the answer position (`a0`), which E52's idle share reads."""
    f, s = t.toks[t.part("fit")], t.toks[t.part("score")]
    with net.at(dW):
        _, c1f = net.run(f, capture=True)
        _, c1s = net.run(s, capture=True)
    a0 = flat_mlp(net.run(f, capture=True)[1], t.half)
    shift = (flat_mlp(c1f, t.half) - a0).double().mean(0)
    order = torch.sort(shift.abs().cpu(), descending=True, stable=True).indices
    shares = {k: share(read(net, t, s, edit=neuron_edit(c1s, order[:k])), t.lp0[t.part("score")], G_score) for k in grid}
    i50 = next((i for i, k in enumerate(grid) if shares[k] >= 0.5), len(grid))
    return {"shares": shares, "k50": grid[i50] if i50 < len(grid) else math.inf, "k50_index": i50,
            "order": order, "shift": shift, "a0": a0}


def patch_controls(net: Net, t: Task, dW: DW, G_score: float) -> dict[str, float]:
    """E52's stage 0 on one state, scoring half: W0 with all 12288 neurons replaced by W0's own activations from a
    separate forward (max |log P - log P_W0| in nats, and as a share of G_score), and W0 with block 3's output
    replaced at every position by the tuned model's (share of G_score)."""
    s, lp0s = t.toks[t.part("score")], t.lp0[t.part("score")]
    _, c0 = net.run(s, capture=True)
    ident = read(net, t, s, edit={f"mlp{i}": (lambda _x, v=c0[f"mlp{i}"]: v) for i in range(N_LAYERS)})
    with net.at(dW):
        _, c1 = net.run(s, capture=True)
    full = share(read(net, t, s, edit={"out3": lambda _x, v=c1["out3"]: v}), lp0s, G_score)
    return {"identity_max_abs": float((ident - lp0s).abs().max()), "identity_share": share(ident, lp0s, G_score),
            "full_patch_share": full}
