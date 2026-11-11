"""E161 share_stability (cell P2.6): is the every-position share's 0.77 to 0.82 on the saved code-text states stable across
row batches, and what does a Gaussian update of the same per-matrix norm read? No training.
Card experiments/share_stability/card.md; plan reports/plans/2026-11-11_plan_p2-6-share-stability.md; rules, thresholds
and expected values results/predictions.json (committed first; the run refuses otherwise). cluster/share_stability.sbatch.

Rows: 320 held-out code-text rows (anchors.code_rows), five batches of 64; batch 0 is E36's and E46's code_held.pt.
Per row T (target) and L (masked, Delta out) mean next-token log-probs (spread.row_reads); per state G = mean(T - T0),
g = mean(L - L0), r = g / G, per batch and pooled.

  stage 0  the control, alone, on batch 0: zero update reads exactly 0; W0 and both saved states reproduce E46's cells
           and r. A miss is exit 1 and nothing else is read
  stage 1  code_step10 and code_step100 on batches 1-4: the spread
  stage 2  the Gaussian (finetune.random_direction of code_step100, seed 0) on all five batches
  stage 3  the filed rules
Results after every stage in results/run.json; every number printed; `STAGE n DONE` after each write.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import anchors as AN
from commute import spread as SP
from commute import vpd
from commute.finetune import random_direction
from commute.jobs import hard_exit, require_filed
from commute.keydep import sha256

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "run.json"
E46 = HERE.parent / "regating_table" / "results"
E36 = HERE.parent / "regating_spread" / "results" / "regating_spread.json"
STEPS = ("10", "100")
N_BATCHES, BATCH_ROWS, GAUSS_SEED, GAUSS_LIKE = 5, 64, 0, "100"

Reads = dict[str, Tensor]  # "T", "L": per-row float64


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Job:
    def __init__(self, device: str) -> None:
        self.P = json.loads(PRED.read_text())
        self.th = self.P["thresholds"]
        self.ref = json.loads((E46 / "stage0a.json").read_text())["groups"]["e36/code"]
        self.device, self.model = device, vpd.load_component_model(device)
        self.W0 = vpd.target_weights(self.model)
        vocab = int(self.model.target_model.lm_head.weight.shape[0])
        _, rows, _ = AN.code_rows(vocab, log, n_held=N_BATCHES * BATCH_ROWS)
        self.batches = list(rows.split(BATCH_ROWS))
        self.out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                                    "predictions_sha256": sha256(PRED)}
        self.t0 = time.time()

    def note(self, key: str, rec: Any) -> None:
        self.out[key] = rec
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.out, indent=1, sort_keys=True))
        tmp.replace(OUT)
        log(f"[{key}] {json.dumps(rec)}")

    def stop(self, why: str) -> NoReturn:
        self.note("stopped", why)
        log(f"STOP: {why}: tell the lead")
        hard_exit(1)

    def state(self, step: str) -> dict[str, Tensor]:
        path = E46 / "large" / f"code_step{step}.pt"
        return torch.load(path, weights_only=False, map_location="cpu")

    def read(self, dW: dict[str, Tensor] | None, b: int) -> Reads:
        target = None if dW is None else {k: w + dW[k].to(w.device, w.dtype) for k, w in self.W0.items()}
        return SP.row_reads(self.model, self.batches[b], self.device, target)


def summary(base: list[Reads], reads: list[Reads], th: dict[str, Any]) -> dict[str, Any]:
    """Per batch and pooled G, g, r of one state against W0 (`base`), and whether its share is readable."""
    dT = [s["T"] - b["T"] for b, s in zip(base, reads, strict=True)]
    dL = [s["L"] - b["L"] for b, s in zip(base, reads, strict=True)]
    G_b, g_b = [float(x.mean()) for x in dT], [float(x.mean()) for x in dL]
    G, g = float(torch.cat(dT).mean()), float(torch.cat(dL).mean())
    readable = G != 0 and all(abs(x) >= th["readable_G"] and math.copysign(1, x) == math.copysign(1, G) for x in G_b)
    r_b = [gg / GG for gg, GG in zip(g_b, G_b, strict=True)] if readable else None
    return {"G": G, "g": g, "G_b": G_b, "g_b": g_b, "readable": readable, "r": g / G if readable else None, "r_b": r_b,
            "sd_r": statistics.stdev(r_b) if r_b else None}


# ----------------------------------------------------------------------------- stage 0: the control


def check_files(J: Job) -> dict[str, Any]:
    held = torch.load(E46 / "large" / "code_held.pt", weights_only=False)
    rec = {"rows_equal_code_held": bool(torch.equal(held, J.batches[0])),
           "held_sha_ok": sha256(E46 / "large" / "code_held.pt") == J.ref["held_sha256"],
           **{f"step{s}_sha_ok": sha256(E46 / "large" / f"code_step{s}.pt") == J.ref["states"][s]["sha256"] for s in STEPS}}
    rec["ok"] = all(rec.values())
    return rec


def stage0(J: Job) -> tuple[Reads, dict[str, Reads]]:
    files = check_files(J)
    J.note("stage0_files", files)
    if not files["ok"]:
        J.stop("the rows or saved states are not E46's")
    base = J.read(None, 0)
    zero = J.read({k: torch.zeros_like(w) for k, w in J.W0.items()}, 0)
    dev = {k: float((zero[k] - base[k]).abs().max()) for k in ("T", "L")}
    J.note("stage0_zero", {"max_abs_dT": dev["T"], "max_abs_dL": dev["L"], "ok": max(dev.values()) <= J.th["zero_tol"]})
    if max(dev.values()) > J.th["zero_tol"]:
        J.stop("the zero update does not read exactly W0")
    reads, rec, tol = {}, {}, J.th["repro_cell_tol"]
    e36 = json.loads(E36.read_text())["states"]
    for s in STEPS:
        reads[s] = J.read(J.state(s), 0)
        cells = {"T0": float(base["T"].mean()), "L00": float(base["L"].mean()),
                 "T1": float(reads[s]["T"].mean()), "L01": float(reads[s]["L"].mean())}
        ref = J.ref["states"][s]["cells"]
        r, r_ref = (cells["L01"] - cells["L00"]) / (cells["T1"] - cells["T0"]), J.ref["states"][s]["g"] / J.ref["states"][s]["G"]
        diff = {k: cells[k] - ref[k] for k in cells}
        ok = all(abs(v) <= tol for v in diff.values()) and abs(r - r_ref) <= J.th["repro_r_tol"]
        rec[s] = {"cells": cells, "cell_minus_e46": diff, "r": r, "r_e46": r_ref, "r_e36_unsaved_state": e36[s]["r"], "ok": ok}
    J.note("stage0_repro", rec)
    if not all(v["ok"] for v in rec.values()):
        J.stop("the saved states do not reproduce E46's reading on batch 0")
    log("STAGE 0 DONE")
    return base, reads


# ----------------------------------------------------------------------------- stages 1-3


def stage1(J: Job, base0: Reads, reads0: dict[str, Reads]) -> tuple[list[Reads], dict[str, dict[str, Any]]]:
    base = [base0] + [J.read(None, b) for b in range(1, N_BATCHES)]
    out = {}
    for s in STEPS:
        dW = J.state(s)
        reads = [reads0[s]] + [J.read(dW, b) for b in range(1, N_BATCHES)]
        out[s] = summary(base, reads, J.th)
        J.note(f"stage1_step{s}", out[s])
    log("STAGE 1 DONE")
    return base, out


def stage2(J: Job, base: list[Reads]) -> dict[str, Any]:
    like = J.state(GAUSS_LIKE)
    dW = random_direction(like, GAUSS_SEED)
    rel = max(abs(float(dW[k].norm() / like[k].norm()) - 1) for k in like)
    assert rel < 1e-4, f"Gaussian norm off by {rel}"
    rec = summary(base, [J.read(dW, b) for b in range(N_BATCHES)], J.th)
    rec["max_norm_mismatch"] = rel
    J.note("stage2_gaussian", rec)
    log("STAGE 2 DONE")
    return rec


def stage3(J: Job, ft: dict[str, dict[str, Any]], ga: dict[str, Any]) -> dict[str, Any]:
    th = J.th
    stable = {s: bool(ft[s]["readable"] and ft[s]["sd_r"] <= th["stable_sd_max"]) for s in STEPS}
    rec: dict[str, Any] = {"P_stable": {"per_step": stable, "pass": all(stable.values())}}
    f = ft[GAUSS_LIKE]
    if not ga["readable"]:
        rec["P_gauss"] = {"outcome": "B: unreadable"}
    elif not f["readable"]:
        rec["P_gauss"] = {"outcome": "no call", "why": "the fine-tune itself is unreadable"}
    else:
        d = ga["r"] - f["r"]
        se = statistics.stdev([a - b for a, b in zip(ga["r_b"], f["r_b"], strict=True)]) / math.sqrt(N_BATCHES)
        outcome = "no call" if se > th["no_call_se"] else ("A: reads like the fine-tune" if abs(d) <= th["gap"] else "B: reads differently")
        rec["P_gauss"] = {"d": d, "se": se, "outcome": outcome}
    J.note("score", rec)
    for s in STEPS:
        log(f"*** step {s}: pooled r {ft[s]['r']} G {ft[s]['G']:.5f}; batch r {ft[s]['r_b']} sd {ft[s]['sd_r']}")
    log(f"*** gaussian: readable {ga['readable']} G {ga['G']:.6f} g {ga['g']:.6f} G_b {ga['G_b']} r {ga['r']} r_b {ga['r_b']}")
    log(f"*** P_stable {rec['P_stable']['pass']} (filed: pass); P_gauss {rec['P_gauss']['outcome']} (filed: B: unreadable)")
    log("STAGE 3 DONE")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    require_filed(PRED)
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run overwrites nothing")
    J = Job(args.device)
    log(f"[start] {J.out['device']}, {len(J.batches)} batches of {BATCH_ROWS} rows")
    base0, reads0 = stage0(J)
    base, ft = stage1(J, base0, reads0)
    ga = stage2(J, base)
    stage3(J, ft, ga)
    J.note("seconds", time.time() - J.t0)
    log(f"FINISHED wrote {OUT}")
    hard_exit(0)


if __name__ == "__main__":
    main()
