"""The paper's adversary at the authors' size on the unchanged Qwen3-1.7B decomposition. Plan:
reports/plans/2026-10-27_plan_qwen-adversary-128.md; rules in results/predictions.json (the run refuses to start
without it).

On 4 rows the 20-step adversary read ~20 nats (port report); the authors evaluate it on 128 rows of 512 tokens with
one source per site shared by all of them. This run reads it at that size. Stages, cheapest first, each written to
results/adversary.json as it finishes and every number printed:
  1  control: seeds 0-2 on rows 0-3 (the qwen_port text rows). Expected: mean in [10, 30] (torch read 18.8). Else stop.
  2  references on the 128 rows: KL with m = g and Δ off (expected 0.05-0.3), and KL(clean ‖ uniform), the ceiling
     a broken output approaches (expected 7-11).
  3  seeds 0-2 on all 128 rows; then the filed reading (usable with one start, with three, or not at this size).
Everything fp32 except Γ (bf16, as in the port check). g and the clean logits are recomputed for each row at every
adversary pass rather than held: 128 rows of g are ~180 GB in fp32. The target, U, V and Γ stay on the GPU.

Memory on one H100 (80 GB): target fp32 6.9 GB, U, V fp32 19.5 GB, Γ bf16 17.1 GB, plus one row's g, masks and
autograd state (~10 GB). Time: ~1 s per row pass, 21 passes per start: ~45 min per 128-row start, ~2.5 h in all.

Usage (repo root, GPU with >= 79 GiB): COMMUTE_QWEN_DIR=<converted copy> python experiments/qwen_adversary/run.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.qwen_vpd import (
    fp32_matmuls,
    importances,
    load_components,
    load_gamma,
    load_target,
    masked_logits,
    pgd_kl_rows,
)
from commute.vpd import kl_per_position

RESULTS = Path(__file__).parent / "results"
TOKENS, PREDICTIONS, OUT = RESULTS / "tokens.json", RESULTS / "predictions.json", RESULTS / "adversary.json"
SEEDS = (0, 1, 2)
CONTROL_ROWS = 4
DEV = "cuda"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save(out: dict) -> None:
    OUT.write_text(json.dumps(out, indent=1))


@torch.no_grad()
def references(model: Any, rows: list[Tensor], comps: Any, inputs: Any) -> dict:
    """Mean over every position of the rows: KL(clean ‖ m = g, Δ off) and KL(clean ‖ uniform) = ln V − H(clean)."""
    ci, ceiling, n = 0.0, 0.0, 0
    for i, tokens in enumerate(rows):
        g, clean = inputs(i)
        ci += float(kl_per_position(clean, masked_logits(model, tokens, comps, g)).sum())
        lp = torch.log_softmax(clean.float(), dim=-1)
        ceiling += float((math.log(lp.shape[-1]) + (lp.exp() * lp).sum(-1)).sum())
        n += tokens.numel()
    return {"ci128": ci / n, "ceiling128": ceiling / n}


def reading(adv: list[float], ceiling: float) -> dict:
    """The filed reading (predictions.json "reading"); every case falls in exactly one branch."""
    mean = sum(adv) / len(adv)
    spread = math.inf if len(adv) < len(SEEDS) or min(adv) < 1e-6 else max(adv) / min(adv) - 1
    if mean >= 0.5 * ceiling or spread >= 1:
        verdict = "unusable_at_this_size"
    elif spread < 0.25:
        verdict = "usable_one_start"
    else:
        verdict = "usable_three_starts"
    return {"adv128_mean": mean, "spread128": spread, "verdict": verdict}


def main() -> None:
    assert PREDICTIONS.is_file(), f"file predictions first: {PREDICTIONS}"
    fp32_matmuls()
    assert torch.cuda.get_device_properties(0).total_memory > 79 * 2**30, "needs an 80 GB card (H100)"
    rows = [torch.tensor([r], device=DEV) for r in json.loads(TOKENS.read_text())["rows"]]
    model = load_target(DEV, torch.float32)
    comps = load_components(device=DEV, dtype=torch.float32)
    gamma = load_gamma(device=DEV, dtype=torch.bfloat16)
    log(f"loaded {len(rows)} rows; GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def inputs(i: int) -> tuple[dict[str, Tensor], Tensor]:
        logits, g = importances(model, rows[i], gamma, comps)
        return g, logits

    out: dict = {"n_rows": len(rows), "seeds": list(SEEDS), "dtype": "fp32 (Γ bf16)"}
    ctl = []
    for seed in SEEDS:
        ctl.append(pgd_kl_rows(model, rows[:CONTROL_ROWS], comps, inputs, seed))
        log(f"control seed {seed}: {ctl[-1]:.3f}")
    adv4 = sum(ctl) / len(ctl)
    out["control"] = {"readings": ctl, "adv4": adv4, "pass": 10 <= adv4 <= 30}
    log(f"control: {out['control']}")
    save(out)
    if not out["control"]["pass"]:
        log("control failed: the 4-row reading is not reproduced; stopping")
        print("FINISHED", OUT, flush=True)
        return

    out["references"] = references(model, rows, comps, inputs)
    log(f"references: {out['references']}")
    save(out)

    adv: list[float] = []
    for seed in SEEDS:
        adv.append(pgd_kl_rows(model, rows, comps, inputs, seed))
        out["adv128"] = adv
        log(f"128 rows seed {seed}: {adv[-1]:.3f}")
        save(out)

    out["reading"] = reading(adv, out["references"]["ceiling128"])
    out["reading"]["ratio"] = out["reading"]["adv128_mean"] / adv4
    out["P1_ratio_below_0.25"] = out["reading"]["ratio"] < 0.25
    out["P2_spread_below_0.25"] = out["reading"]["spread128"] < 0.25
    log(f"reading: {out['reading']}; P1 {out['P1_ratio_below_0.25']}, P2 {out['P2_spread_below_0.25']}")
    save(out)
    print("FINISHED", OUT, flush=True)


if __name__ == "__main__":
    main()
