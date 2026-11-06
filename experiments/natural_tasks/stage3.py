"""E124 stage 3: the re-gating reading on stage 1's six states, then the filed scoring (run.py's docstring has the whole
run). Per state, on its own fit windows with its own target (data.LAST): r of the least-squares split
(`keydep.route_windows`, unchanged) and the stand-in ratio from the same cells (`pcfg_vpd.shares`, ratio_floor). r is
nan under a full-vocabulary gain of local_faith.MIN_GAIN ("no call"). Beside, no rule: the twin on its flipped fit
windows (twin label != the text's pronoun) and the natural member per gender. Each read is written as it ends; a rerun
skips what is recorded; exit 3 when too little time is left (resubmit).

    sbatch cluster/natural_tasks.sbatch 3
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import torch
from run import (
    HALF,
    LARGE,
    MEMBERS,
    RESULTS,
    Config,
    L,
    ab_of,
    dump,
    finish,
    labels,
    load_dWs,
    log,
    say,
    setup,
    task,
)
from stage0 import ALIVE

from commute import keydep as KD
from commute import label_reads as LR
from commute import natural_tasks as NT
from commute import pcfg_vpd
from commute.data import LAST
from commute.local_faith import MIN_GAIN, route_cells


def subsets(t: dict[str, Any], member: str) -> dict[str, torch.Tensor]:
    """The fit-window masks a member is read on: all; the twin's flipped windows; the natural member per gender."""
    y = labels(t, member, "fit")[1]
    out = {"all": torch.ones(len(y), dtype=torch.bool)}
    if member == "twin":
        out["flipped"] = y != t["fit"]["c"]
    if member == "natural":
        out.update({"female": t["fit"]["c"] == 0, "male": t["fit"]["c"] == 1})
    return out


def read(ctx: KD.Ctx, C: Config, x: torch.Tensor, dW: Any, base: dict[str, Any]) -> dict[str, Any]:
    rd = KD.route_windows(ctx, x, HALF, LAST, dW, base, L)
    sh = pcfg_vpd.shares(rd["cells"], C.ratio_floor)
    r = sh["r"] if rd["G_logodds"] >= MIN_GAIN else math.nan
    return {"n": len(x), "r": r, "ratio": sh["ratio"], "gain": rd["G_logodds"], "piece_gain": sh["piece_gain"], "cells": rd["cells"]}


def score(P: dict[str, Any], s2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    """The filed outcomes: each member's verdict on both seeds (Fable's change 6), the re-gating reading
    (`natural_tasks.replicates`), R1/R2 against the filed bands (reported, no verdict) and the joint table."""
    F, seeds = P["filed_outcome"], P["settings"]["seeds"]
    verdicts = {m: [s2["states"][f"{m}_s{s}"]["verdict"] for s in seeds] for m in MEMBERS}
    r = {m: [s3["reads"][f"{m}_s{s}"]["all"]["r"] for s in seeds] for m in MEMBERS}
    band = lambda v, lo_hi: bool(lo_hi[0] <= v <= lo_hi[1]) if math.isfinite(v) else None
    return {"definition": {m: {"verdicts": verdicts[m], "filed": F["verdict"][m], "pass": NT.member_passes(verdicts[m], F["verdict"][m])} for m in MEMBERS},
            "regating": {**NT.replicates(r["natural"], r["new_word"], F["r_cut"], F["min_gap"]), "filed": F["regating"]},
            "bands": {n: {"r_in_band": band(v["all"]["r"], F["r_band"][n.rsplit("_s", 1)[0]]) if n.rsplit("_s", 1)[0] in F["r_band"] else None,
                          "ratio_in_band": band(v["all"]["ratio"], F["ratio_band"][n.rsplit("_s", 1)[0]]) if n.rsplit("_s", 1)[0] in F["ratio_band"] else None}
                      for n, v in s3["reads"].items()},
            "joint": {n: {"verdict": s2["states"][n]["verdict"], "r": v["all"]["r"], "ratio": v["all"]["ratio"]} for n, v in s3["reads"].items()}}  # fmt: skip


def stage3(C: Config, P: dict[str, Any], t0: float) -> int:
    s0 = json.loads((RESULTS / "stage0.json").read_text())
    s2 = json.loads((RESULTS / "stage2.json").read_text())
    assert s0.get("go") and "stop" not in s2 and len(s2["states"]) == 6, "stages 0-2 are not complete"
    p = RESULTS / "stage3.json"
    res: dict[str, Any] = json.loads(p.read_text()) if p.is_file() else {"base": {}, "reads": {}}
    ctx, t, dWs = KD.make_ctx("cuda", ALIVE, P, LARGE), task(), load_dWs()
    for m in MEMBERS:
        x = torch.cat([labels(t, m, "fit")[0], LR.targets(labels(t, m, "fit")[1], ab_of(t, m))[:, None]], 1)
        for sub, mask in subsets(t, m).items():
            if not mask.any():  # an empty subset is no reading (edge-case pass), never a crash
                for seed in C.seeds:
                    res["reads"].setdefault(f"{m}_s{seed}", {})[sub] = {"n": 0, "r": math.nan, "ratio": math.nan}
                continue
            base = route_cells(ctx.net.model, x[mask], HALF, LAST, KD.BATCH, L)
            if sub == "all":  # stage 0's W0 cells on the same windows: must agree
                res["base"][m] = {k: base[k] - s0["route_w0"][m][k] for k in ("T", "U", "L")}
                say(f"base vs stage 0 {m}", res["base"][m])
            for seed in C.seeds:
                name = f"{m}_s{seed}"
                if sub in res["reads"].get(name, {}):
                    continue
                if time.monotonic() - t0 > (C.stage_minutes - 2) * 60:
                    log("[time] too little left: exit 3, resubmit stage 3")
                    dump(3, res)
                    return 3
                res["reads"].setdefault(name, {})[sub] = read(ctx, C, x[mask], dWs[name], base)
                say(f"route {name} {sub}", res["reads"][name][sub])
                dump(3, res)
    res["score"] = score(P, s2, res)
    say("score", res["score"])
    dump(3, res)
    return 0


def main() -> None:
    t0 = time.monotonic()
    C, _R, P = setup()
    finish(3, stage3(C, P, t0))


if __name__ == "__main__":
    main()
