"""P2.18's top-set overlap (E155, `experiments/neuron_overlap`) on Qwen3-1.7B, for cell P3.6 (E-number in
`experiments/qwen_cheap_readings/`). No decomposition is read. Descriptive: on the 67M the check that a high overlap means
"the same neurons firing harder" failed, so the overlap is not evidence of reuse.

Per block l, at every position of a row set that predicts an answer token (`qwen_comparators.Rows.answer`), the block's MLP
firing (down_proj's input, width 6144) with the decomposed matrices at W0 and at W, and per position the K neurons of
largest |firing| in each (`masked_reads.top_overlap`, E155's sets; K = 480, P2.18's share 240/3072 = 7.8125% of 6144):
  overlap   Jaccard of the two top-K sets, mean over positions, then over the 28 blocks;
  on_both   the squared firing change summed over neurons in both sets, over that summed over every neuron, pooled over
            positions and blocks (nan when W changes no firing).
W equal to W0 reads overlap exactly 1.0 and on_both nan.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from commute import masked_reads as MR
from commute.qwen_comparators import Rows
from commute.qwen_sft import set_weights
from commute.qwen_vpd import N_LAYERS

D_MLP = 6144
K = D_MLP * 240 // 3072  # 480: P2.18's 240 of the 67M's 3072
BATCH = 16  # rows per forward; firing at answer positions only, about 1.7 GB per forward at 16 x 256


def firing(model: Any, tokens: Tensor, mask: Tensor) -> dict[str, Tensor]:
    """Per block, down_proj's input at the positions in `mask`: {"mlp<l>": (P, 1, 6144) fp32}, the shape
    `masked_reads.top_overlap` reads at position 0. The base model only: no logits are formed."""
    out: dict[str, Tensor] = {}

    def keep(l: int) -> Any:
        return lambda _m, args: out.__setitem__(f"mlp{l}", args[0][mask].float().unsqueeze(1))

    hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(keep(l)) for l in range(N_LAYERS)]
    try:
        model.model(input_ids=tokens, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return out


@torch.no_grad()
def overlap(model: Any, rows: Rows, W: Mapping[str, Tensor], k: int = K) -> dict[str, Any]:
    """The model's own weights (W0) against W on `rows`' answer positions: overlap, on_both and their per-block parts."""
    dev = next(model.parameters()).device
    acc: dict[int, dict[str, Any]] = {l: {"jaccard": [], "d2_both": 0.0, "d2_all": 0.0} for l in range(N_LAYERS)}
    for s in range(0, len(rows.tokens), BATCH):
        x, m = rows.tokens[s : s + BATCH].to(dev), rows.answer[s : s + BATCH].to(dev)
        f0 = firing(model, x, m)
        with set_weights(model, W):
            f1 = firing(model, x, m)
        for l, r in MR.top_overlap(f0, f1, dict.fromkeys(range(N_LAYERS), k), 0).items():
            acc[l]["jaccard"].append(r["jaccard"])
            acc[l]["d2_both"] += float(r["d2_both"].sum())
            acc[l]["d2_all"] += float(r["d2_all"].sum())
        del f0, f1
    return summarise(acc)


def summarise(blocks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    jac = {l: torch.cat(a["jaccard"]) for l, a in blocks.items()}
    both, every = sum(a["d2_both"] for a in blocks.values()), sum(a["d2_all"] for a in blocks.values())
    return {"overlap": statistics.fmean(float(j.mean()) for j in jac.values()), "on_both": both / every if every > 0 else math.nan,
            "positions": len(jac[0]),
            "per_block": {l: {"overlap": float(jac[l].mean()), "overlap_min": float(jac[l].min()), "d2_both": a["d2_both"], "d2_all": a["d2_all"]}
                          for l, a in blocks.items()}}  # fmt: skip


def site_norms(path: Path) -> dict[str, float]:
    """Frobenius norm of each matrix of a saved update (safetensors keyed by site), read one tensor at a time, fp64."""
    with safe_open(str(path), framework="pt") as f:
        return {s: float(f.get_tensor(s).double().norm()) for s in f.keys()}  # noqa: SIM118


def random_weights(W0: Mapping[str, Tensor], norms: Mapping[str, float], seed: int) -> dict[str, Tensor]:
    """W0 + R, R iid Gaussian per matrix scaled to `norms[site]` (zero where that is 0): the matched-norm random floor.
    One CPU generator, sites in sorted order, so a draw is the same on any card."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for s in sorted(W0):
        r = torch.randn(W0[s].shape, generator=gen)
        out[s] = W0[s] + (r * (norms[s] / float(r.norm()))).to(W0[s].device, W0[s].dtype)
    return out
