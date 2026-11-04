"""E76 Amendment 1: where in the model does each arm's masked fall sit? Each arm's saved change, kept on one part of the blocks and
zero elsewhere, read with the arms' `read_state` on the same 8 rows and the same fixed mask draw. Control first: W0 must reproduce step 0.
Plan: reports/plans/2026-11-03_plan_e76-masked-teacher.md, Amendment 1. Filed: results/predictions_amend1.json."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import arms as arms_mod
import run as run_mod
import torch
from common import ARMS, DEV, LARGE, RESULTS, C, load_g, log, read_json, rows_all, write_json
from safetensors.torch import load_file

from commute.qwen_teacher import Offsets
from commute.qwen_vpd import SITES, fp32_matmuls, load_components, load_target, parse_site

PARTS = {"all": range(28), "early": range(21), "late": range(21, 28), "last": range(27, 28)}
W0_TOL = 1e-3  # step 0 is the same code, card, rows and draw; rounding alone is ~2e-6 (stage 0's floor)


def keep_part(off: Offsets, blocks: range) -> None:
    with torch.no_grad():
        for s in SITES:
            if parse_site(s)[0] not in blocks:
                off.dV[s].zero_()
                off.dU[s].zero_()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True, help="the node's disk, holding E76's g per row")
    a = ap.parse_args()
    if not (RESULTS / "predictions_amend1.json").is_file():
        sys.exit("refusing to start: predictions_amend1.json missing")
    fp32_matmuls()
    model = load_target(DEV, torch.float32)
    host = load_components(Path(a.decomp), "cpu", torch.float32)
    off = Offsets(host, load_file(str(LARGE / "cut.safetensors")), DEV)
    comps = host.to(DEV, torch.bfloat16)
    del host
    gc.collect()
    rows = rows_all()[1][: C.fixed_rows]
    S = arms_mod.State(model, comps, off, rows, [load_g(Path(a.out), i) for i in range(C.fixed_rows)], run_mod.load_dw(), 0.0)
    arm_a = next(x for x in ARMS if x.name == "A")
    _, ft, w0 = arms_mod.set_up(S, arm_a, {})
    dw_blocks = arms_mod.squared_by_block(S.dW.__getitem__)

    off.reset()
    out = {"W0": {k: arms_mod.read_state(S, 0, ft, w0, dw_blocks)[k] for k in ("i", "ii", "iv")}}
    want = read_json(RESULTS / "B.json")["readings"][0]
    log(f"[amend1 W0] {out['W0']}, step 0 of B {[want[k] for k in ('i', 'ii', 'iv')]}")
    write_json(RESULTS / "amend1.json", out)
    if not all(abs(out["W0"][k] - want[k]) <= W0_TOL for k in ("i", "iv")):
        sys.exit("stop: W0 does not reproduce step 0: the read differs from the arms' reading")

    for name in ("A", "B", "L"):
        out[name] = {}
        for part, blocks in PARTS.items():
            off.load_from(LARGE / f"{name}_offsets.safetensors")
            keep_part(off, blocks)
            r = arms_mod.read_state(S, 0, ft, w0, dw_blocks)
            out[name][part] = {k: r[k] for k in ("i", "ii", "iv")} | {"size": sum(r["blocks"])}
            log(f"[amend1 {name} {part}] {out[name][part]}")
            write_json(RESULTS / "amend1.json", out)
    log("[amend1 done]")


if __name__ == "__main__":
    main()
