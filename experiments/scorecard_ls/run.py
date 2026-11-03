"""scorecard_ls (E71): the least-squares split of the lowercase Adam and rank-8 LoRA fine-tunes, read with E39's check.
Plan: reports/plans/2026-11-03_plan_e71-scorecard-ls.md. Predictions and control values: results/predictions.json.

The check is E39's (`row_copy.read_state`): Pile PGD, one start, adversary seed 0, 20 steps of 0.1, val rows 0-255, a
ratio to W0's mean KL in the same job. Fine-tunes: row_blocks' dW_{adam,lora}.pt, regenerated on an L40 (E49).
s = the share of the fine-tune's held-out gain kept. delta_share = ||Delta' - Delta||^2 / ||delta-W||^2 over all
matrices, unweighted, read from the model (E16's and `harness.measure`'s measure, so the emoticon slide's).

States, per fine-tune, target weights W0 + x and pieces (U', V'):
  nothing     x = delta-W, released pieces (control)
  adjoint     x = the scaled adjoint split's weight change, its pieces (control)
  ls          x = delta-W, the least-squares pieces: Delta moves by O(delta-W) - dU dV^T
  expr        x = delta-W - O(delta-W), the same pieces: Delta moves by -dU dV^T (the second-order term)
  expr_fixed  x = (U + dU)(V + dV)^T - U V^T, the same pieces: Delta does not move
expr and expr_fixed are the two exact readings of "W0 plus the update minus the part outside the pieces, Delta fixed".

Order: W0, then each fine-tune's gain and both controls, which stop the job if the apparatus is off; then the rest.
Each read is written as it finishes.
Usage (repo root, GPU): run.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT, LOG = RESULTS / "predictions.json", RESULTS / "reads.json", RESULTS / "run.log"
ROW_BLOCKS = HERE.parent / "row_blocks" / "results"
TRANSPORT = HERE.parent / "transport_test" / "results"
FINE_TUNES = ("adam", "lora")
UV = dict[str, tuple[Tensor, Tensor]]


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def expressible(dW: dict[str, Tensor], uv: UV, bases: dict[str, Any]) -> dict[str, Any]:
    """The weight changes of expr and expr_fixed, the least-squares pieces, and the unweighted share of delta-W outside."""
    from commute import split

    expr, fixed, pieces = {}, {}, {}
    outside = energy = 0.0
    for k, M in dW.items():
        expr[k], fixed[k] = torch.zeros_like(M), torch.zeros_like(M)
        if not M.any():
            continue
        M, U, V = M.double(), uv[k][0].double(), uv[k][1].double()
        dU, dV = split.least_squares_split(M, bases[k], U, V)
        O = split.obstruction(M, bases[k])
        expr[k] = (M - O).float()
        fixed[k] = ((U + dU) @ (V + dV).T - U @ V.T).float()
        pieces[k] = ((U + dU).float(), (V + dV).float())
        outside, energy = outside + float((O**2).sum()), energy + float((M**2).sum())
    return {"expr": expr, "fixed": fixed, "pieces": pieces, "outside_frob": outside / energy}


class Job:
    def __init__(self, device: str) -> None:
        from commute import harness, lowercase, split
        from commute.data import batches, pile_rows

        self.pred = json.loads(PRED.read_text())
        self.c = self.pred["controls"]
        self.pass_line, self.fail_line = self.pred["lines"]["pass"], self.pred["lines"]["fail"]
        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        self.ctx = ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token,
                                      1e-6, 256, 20, (0,), log)
        self.card = torch.cuda.get_device_name()
        self.out: dict[str, Any] = {"card": self.card, "status": "running", "w0": ctx.pile[0], "fine_tunes": {},
                                    "reads": {}, "control_gaps": {}}
        self.low = lowercase.task(ctx.model, ctx.device, log)
        self.plain = pile_rows("val", 0, 256, log=log)[:64]
        moments = split.input_second_moments(ctx.model, batches(self.low.rows[:256], 8), list(ctx.W0))
        self.sigma = {k: v.cpu() for k, v in moments.items()}  # E39's: the first lowercase training rows
        self.refs: dict[str, Any] = {}
        self.top: dict[str, float] = {}
        self.dW: dict[str, dict[str, Tensor]] = {}

    def dump(self) -> None:
        OUT.write_text(json.dumps(self.out, indent=1, sort_keys=True))

    def halt(self, why: str) -> None:
        self.out["status"] = f"stopped: {why}"
        self.dump()
        log(f"STOP: {why}")
        sys.exit(f"STOP: {why}")

    def verdict(self, r: float) -> str:
        return "pass" if r <= self.pass_line else "fail" if r >= self.fail_line else "between"

    def add(self, ft: str) -> None:
        """The fine-tune's file (sha256 against regen.json), its gain against E49's, the outside shares."""
        from commute import harness, lowercase, teacher
        from commute.row_copy import Refs

        path = ROW_BLOCKS / "large" / f"dW_{ft}.pt"
        regen = json.loads((ROW_BLOCKS / "regen.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != regen["files"][path.name]:
            self.halt(f"{path} does not match regen.json")
        dW = self.dW[ft] = torch.load(path, weights_only=True)
        g = lowercase.gain(self.ctx.model, self.ctx.device, self.low, dW)
        self.top[ft] = self.low.logp_w0 + g
        self.refs[ft] = Refs(teacher.teacher_net(self.ctx.model, dW), self.low.w0_net, self.low.held, self.plain, 8)
        o = harness.outside_share_of(dW, self.ctx.bases, self.sigma, self.ctx.non_saturated)
        self.out["fine_tunes"][ft] = {"gain": g, "gain_regen": regen["gains"][ft], "outside_weighted": o["pooled"],
                                      "outside_weighted_h100": self.c["outside_weighted_h100"][ft]}
        log(f"[{ft}] gain {g:.5f} (regen {regen['gains'][ft]:.5f}) outside share, input-weighted {o['pooled']:.4f}")
        if abs(g - regen["gains"][ft]) > self.c["gain_tol"]:
            self.halt(f"{ft}: gain {g} against E49's {regen['gains'][ft]}")

    def read(self, name: str, ft: str, x: dict[str, Tensor], uv2: UV | None) -> dict[str, Any]:
        from commute import vpd
        from commute.component_training import components_with
        from commute.faithfulness import model_state
        from commute.row_copy import read_state

        t0, ctx, M = time.time(), self.ctx, self.dW[ft]
        rec = read_state(ctx, self.refs[ft], x, uv2, True, log)
        rec["check"] = rec["pile"]["pgd_kl"]
        rec["verdict"] = self.verdict(rec["check"])
        rec["s"] = (rec["logp"] - self.low.logp_w0) / (self.top[ft] - self.low.logp_w0)
        comps = None if uv2 is None else components_with(ctx.model, ctx.alive, uv2)
        target = {k: ctx.W0[k] + x[k].to(ctx.W0[k].device) for k in ctx.W0}
        with model_state(ctx.model, target=target, components=comps):
            wd = vpd.weight_deltas(ctx.model)
            moved = sum(float(((wd[k] - ctx.wd0[k]).double() ** 2).sum()) for k in wd)
        rec["delta_share"] = moved / sum(float((M[k].double() ** 2).sum()) for k in M)
        self.out["reads"][name] = rec
        self.dump()
        log(f"[{name}] check {rec['check']:.4f} {rec['verdict']} s {rec['s']:.4f} delta_share {rec['delta_share']:.5f} "
            f"gap {rec['gap']:.5f} | {time.time() - t0:.0f} s")
        return rec

    def controls(self, ft: str) -> None:
        """Do-nothing and the scaled adjoint split, against E39's H100 readings: a flipped verdict or a wrong Delta share
        stops the job; a gap only sets `close`, which the lead reports before the figure changes."""
        from commute import split

        n = self.read(f"nothing_{ft}", ft, self.dW[ft], None)
        x, st = split.adjoint_state(self.dW[ft], self.ctx.uv_cpu, self.sigma)
        a = self.read(f"adjoint_{ft}", ft, x, {k: (v[1], v[2]) for k, v in st.items()})
        h, tol = self.c["h100"][ft], self.c["close_rel"]
        gaps = {"nothing": n["check"] / h["nothing"] - 1, "adjoint": a["check"] / h["adjoint"] - 1,
                "adjoint_s": a["s"] - h["adjoint_s"]}
        close = abs(gaps["nothing"]) <= tol and abs(gaps["adjoint"]) <= tol and abs(gaps["adjoint_s"]) <= self.c["close_s"]
        self.out["control_gaps"][ft] = {**gaps, "close": close}
        self.dump()
        log(f"[controls {ft}] {json.dumps(self.out['control_gaps'][ft])}")
        if n["verdict"] != "fail" or a["verdict"] != "pass":
            self.halt(f"{ft}: do-nothing reads {n['check']}, adjoint {a['check']}: a verdict differs from E39's")
        if abs(n["delta_share"] - 1) > self.c["delta_tol"] or a["delta_share"] > self.c["delta_tol"]:
            self.halt(f"{ft}: Delta shares {n['delta_share']}, {a['delta_share']}, not 1 and 0")

    def splits(self, ft: str) -> None:
        e = expressible(self.dW[ft], self.ctx.uv_cpu, self.ctx.bases)
        self.out["fine_tunes"][ft]["outside_frob"] = e["outside_frob"]
        self.read(f"ls_{ft}", ft, self.dW[ft], e["pieces"])
        self.read(f"expr_{ft}", ft, e["expr"], e["pieces"])
        self.read(f"expr_fixed_{ft}", ft, e["fixed"], e["pieces"])


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file {PRED} first")
    job = Job("cuda")
    w0, c = job.ctx.pile[0]["pgd_kl"], job.c
    log(f"[W0] {job.card}: mean PGD KL {w0:.6f}, filed L40 {c['w0_l40']}")
    if abs(w0 / c["w0_l40"] - 1) > c["w0_tol"]:
        job.halt(f"W0 reads {w0} on {job.card}, not the L40's {c['w0_l40']}")
    for ft in FINE_TUNES:
        job.add(ft)
        job.controls(ft)
    for ft in FINE_TUNES:
        job.splits(ft)
    job.out["status"] = "done"
    job.dump()
    log(f"FINISHED wrote {OUT}")
    sys.stdout.flush()
    os._exit(0)  # jobs have hung at interpreter exit (E49's regen)


if __name__ == "__main__":
    main()
