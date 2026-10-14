"""Diagnostic for the adversarial-KL gap (reports/2026-10-14_decision_pgd-scope.md): is the 20-step shared-adversary
KL of the released decomposition on W0 a property of our rows or of our environment?

Measures, with the same settings as faithfulness.py (20 steps of 0.1, random init, Delta included, one adversary per
block of 128 rows in sub-batches of 8): the KL on val blocks 1..3 (rows 128..511) beside block 0; and on block 0 the
per-position KL distribution under the final adversarial masks (quantiles, share of the mean carried by the top 1% of
positions, per-row means). Writes results/large/logs/pgd_check.json. Read-only; changes nothing.

Usage: python experiments/transport_test/check_pgd.py [--blocks 1 2 3] [--steps 20]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "large" / "logs" / "pgd_check.json"


def shared_adversary_per_position(model, batches, steps: int, step_size: float = 0.1, seed: int = 0):
    """The authors' multi-batch PGD loop, returning the per-position KL under the final masks."""
    from param_decomp.metrics.pgd_utils import (
        _construct_mask_infos_from_adv_sources,
        get_pgd_init_tensor,
    )
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl

    from commute.vpd import compute_ci, kl_per_position, weight_deltas

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    wd = weight_deltas(model)
    adv = {k: get_pgd_init_tensor("random", (1, 1, model.module_to_c[k] + 1), device).requires_grad_(True) for k in model.target_module_paths}
    cis = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches:
            ci = compute_ci(model, b, chunk=b.shape[0])
            cis.append((ci.g, ci.logits))

    def masked_logits(i: int, b: torch.Tensor) -> torch.Tensor:
        g, _ = cis[i]
        mi = _construct_mask_infos_from_adv_sources(adv, g, wd, "all", tuple(g[next(iter(g))].shape[:-1]))
        return model(b, mask_infos=mi)

    for _ in range(steps):
        grads = {k: torch.zeros_like(v) for k, v in adv.items()}
        for i, b in enumerate(batches):
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = masked_logits(i, b)
                loss, _ = recon_loss_kl(out, cis[i][1])
            gs = torch.autograd.grad(loss, list(adv.values()))
            for k, gk in zip(adv, gs):
                grads[k] += gk
        with torch.no_grad():
            for k in adv:
                adv[k].add_(step_size * grads[k].sign()).clamp_(0.0, 1.0)
    per_pos = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, b in enumerate(batches):
            per_pos.append(kl_per_position(cis[i][1], masked_logits(i, b)).float().cpu())
    return torch.cat(per_pos)  # (rows, 512)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    from commute.data import batches, pile_rows
    from commute.vpd import load_component_model

    model = load_component_model("cuda")
    res = {"steps": args.steps, "blocks": {}}
    for blk in args.blocks:
        t0 = time.time()
        bs = [b.cuda() for b in batches(pile_rows("val", 128 * blk, 128), 8)]
        kl = shared_adversary_per_position(model, bs, args.steps)
        flat = kl.flatten()
        srt = torch.sort(flat, descending=True).values
        top1 = int(0.01 * flat.numel())
        res["blocks"][blk] = {
            "rows": f"{128 * blk}..{128 * blk + 127}",
            "mean_kl": float(flat.mean()),
            "median_kl": float(flat.median()),
            "quantiles": {q: float(torch.quantile(flat, q)) for q in (0.5, 0.9, 0.99, 0.999)},
            "max_kl": float(flat.max()),
            "share_of_mean_from_top_1pct_positions": float(srt[:top1].sum() / flat.sum()),
            "row_means_sorted_desc_top10": [round(x, 3) for x in torch.sort(kl.mean(1), descending=True).values[:10].tolist()],
            "row_means_median": float(kl.mean(1).median()),
            "seconds": time.time() - t0,
        }
        print(blk, json.dumps(res["blocks"][blk]), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
        del bs, kl
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
