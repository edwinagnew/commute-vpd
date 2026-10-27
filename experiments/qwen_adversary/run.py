"""The paper's adversary at the authors' size on the unchanged Qwen3-1.7B decomposition. Plan:
reports/plans/2026-10-27_plan_qwen-adversary-128.md; rules in results/predictions.json (the run refuses to start
without it).

On 4 rows the 20-step adversary read ~20 nats (port report); the authors evaluate it on 128 rows of 512 tokens with
one source per site shared by all of them. This run reads it at that size. Stages, cheapest first, each written to
results/adversary.json as it finishes and every number printed:
  0  recompute check: KL with m = g, Δ off, per row on rows 0-3 against the port check's fp32 values (0.180, 0.181,
     0.130, 0.150). Pass: every |diff| <= 1e-3 (deterministic: same code, rows and precision). Else stop.
  1  control: seeds 0-2 on rows 0-3 (the qwen_port text rows). Expected: mean in [10, 30] (torch read 18.8). Else stop.
  2  references on the 128 rows: KL with m = g and Δ off (expected 0.05-0.3), and KL(clean ‖ uniform), the ceiling
     a broken output approaches (expected 7-11).
  3  seeds 0-2 on all 128 rows; then the filed reading (usable with one start, with three, or not at this size).
Everything fp32 except Γ (bf16, as in the port check). g and the clean logits are recomputed for each row at every
adversary pass rather than held: 128 rows of g are ~180 GB in fp32. The target, U, V and Γ stay on the GPU.

Memory on one H100 (80 GB): target fp32 6.9 GB, U, V fp32 19.5 GB, Γ bf16 17.1 GB, plus one row's g, masks and
autograd state (~10-15 GB). Time: ~1 s per row pass on an L40-class card (TF32 off, so fp32 runs on CUDA cores: an
H100 SXM is not necessarily faster, an A100 several times slower), 21 passes per start: ~45 min per 128-row start,
~2.5 h in all; the control's per-step lines (4 rows each) give the projection. Each step prints its KL.

Usage (repo root, 80 GB GPU): COMMUTE_QWEN_DIR=<converted copy> python experiments/qwen_adversary/run.py
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
PORT_CHECK = Path(__file__).parent.parent / "qwen_port" / "results" / "torch_check.json"
SEEDS = (0, 1, 2)
CONTROL_ROWS = 4
RECOMPUTE_TOL = 1e-3  # predictions.json addendum_1
DEV = "cuda"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save(out: dict) -> None:
    OUT.write_text(json.dumps(out, indent=1))


@torch.no_grad()
def references(model: Any, rows: list[Tensor], comps: Any, inputs: Any) -> dict:
    """Per row and over every position of the rows (rows are equal length): KL(clean ‖ m = g, Δ off) and
    KL(clean ‖ uniform) = ln V − H(clean)."""
    ci, ceiling = [], []
    for i, tokens in enumerate(rows):
        g, clean = inputs(i)
        ci.append(float(kl_per_position(clean, masked_logits(model, tokens, comps, g)).mean()))
        lp = torch.log_softmax(clean.float(), dim=-1)
        ceiling.append(float((math.log(lp.shape[-1]) + (lp.exp() * lp).sum(-1)).mean()))
    return {"ci128": sum(ci) / len(ci), "ceiling128": sum(ceiling) / len(ceiling), "ci_per_row": ci}


def recompute_check(model: Any, rows: list[Tensor], comps: Any, inputs: Any) -> dict:
    """Stage 0 (predictions.json addendum_1): the m = g KL on rows 0-3 against the port check's fp32 per-row values,
    a deterministic test of the recomputed g, clean logits and masked forward."""
    port = json.loads(PORT_CHECK.read_text())["check3"]["torch_fp32"]["ci"]["per_row"][:CONTROL_ROWS]
    mine = references(model, rows[:CONTROL_ROWS], comps, inputs)["ci_per_row"]
    diff = max(abs(a - b) for a, b in zip(mine, port, strict=True))
    return {"mine": mine, "port": port, "max_abs_diff": diff, "pass": diff <= RECOMPUTE_TOL}


def reading(adv: list[float], ceiling: float) -> dict:
    """The filed reading (predictions.json "reading"); every case falls in exactly one branch."""
    mean = sum(adv) / len(adv)
    broken = len(adv) < len(SEEDS) or not all(math.isfinite(a) for a in adv) or min(adv) < 1e-6
    spread = math.inf if broken else max(adv) / min(adv) - 1
    if spread >= 1 or not mean < 0.5 * ceiling:  # a NaN mean or reading is unusable
        verdict = "unusable_at_this_size"
    elif spread < 0.25:
        verdict = "usable_one_start"
    else:
        verdict = "usable_three_starts"
    return {"adv128_mean": mean, "spread128": spread, "verdict": verdict}


def main() -> None:
    assert PREDICTIONS.is_file(), f"file predictions first: {PREDICTIONS}"
    fp32_matmuls()
    assert torch.cuda.get_device_properties(0).total_memory > 70 * 2**30, "needs an 80 GB card (H100)"
    rows = [torch.tensor([r], device=DEV) for r in json.loads(TOKENS.read_text())["rows"]]
    model = load_target(DEV, torch.float32)
    comps = load_components(device=DEV, dtype=torch.float32)
    gamma = load_gamma(device=DEV, dtype=torch.bfloat16)
    log(f"loaded {len(rows)} rows; GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def inputs(i: int) -> tuple[dict[str, Tensor], Tensor]:
        logits, g = importances(model, rows[i], gamma, comps)
        return g, logits

    out: dict = {"n_rows": len(rows), "seeds": list(SEEDS), "dtype": "fp32 (Γ bf16)"}
    out["recompute_check"] = recompute_check(model, rows, comps, inputs)
    log(f"recompute check: {out['recompute_check']}")
    save(out)
    if not out["recompute_check"]["pass"]:
        log("recompute check failed: g, clean logits or the masked forward differ from the port check; stopping")
        print("FINISHED", OUT, flush=True)
        return

    ctl = []
    for seed in SEEDS:
        ctl.append(pgd_kl_rows(model, rows[:CONTROL_ROWS], comps, inputs, seed, log=log))
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
        adv.append(pgd_kl_rows(model, rows, comps, inputs, seed, log=log))
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
