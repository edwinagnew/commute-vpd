"""E44 Amendment 1, the layer-cut check ($0, one cluster L40). Plan: reports/plans/2026-10-28_plan_qwen-q0q1.md,
"Amendment record (2026-10-29, Amendment 1: layer-cut check)"; predictions and every rule scored here:
results/predictions_amend1.json (refuses to start without it).

At E40's step 10, do Γ's masks carry the update from the layers where δW acts (re-gating), or only once the
fine-tuned model's late activations reach Γ (copying)? Every masked reading: original subcomponents, Δ off, W0's
weights in the model, KL from the fine-tuned model (FT = W0 + δW); only Γ's inputs change. cut(l) gives Γ FT's taps
0..l and FT's readouts of layers < l, W0's for the rest. H(l), the comparator, is the unmasked model with FT's weights
in layers < l and W0's after.

Stage 1 (controls, alone): the floor (the plain W0 read twice on row 0), the identity (cut(0), cut(28) against the
unchanged `importances` path at W0 and FT), the effect; amend1_controls.json; any miss stops the job (stopped.json,
exit 1). Stop rules also: GPU peak after row 0 over 44 GiB, stage 1 past 15 min of wall time, step_10.safetensors
absent or its sha256 not the committed one. Stage 2: KL_u(8, 16, 24), then K(16), K(24), K(8), K_taps, K_reads, each
written to amend1_stage2.json as it ends, with amend1_score.json rescored after each (a job cut short is scored on
what it wrote). Every number is also printed to run.log.

δW: --e40 is a directory holding dW/step_10.safetensors: the cluster regen copy (default), or the same files fetched
from W&B (`python -m commute.qwen_e40_files fetch DIR v0`, artifact qwen3-1.7b-e40-nowarmup-dW:v0); either way the
sha256 must match --sums. Usage (repo root, L40): python experiments/qwen_maps/amend1.py --decomp DIR [--e40 DIR]

GPU: model 6.9 GB + fp32 components 19.5 + Γ (bf16) 17.1 = ~43.4 GB (40.4 GiB) fixed, so FT's weights stay on the
host and are copied into the model's own tensors in place (`Weights.use`); Γ's inputs are kept on the host too.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from job import DEV, F32, RESULTS, Config, Job, JobOpts, write_json
from torch import Tensor

from commute.qwen_e40_files import sums
from commute.qwen_sft import linears, sha256
from commute.qwen_vpd import (
    N_TAPS,
    SITES,
    clean_forward,
    importances,
    lower_importances,
    masked_logits,
    parse_site,
)
from commute.vpd import kl_per_position

PRED_A1, OUT = RESULTS / "predictions_amend1.json", RESULTS / "amend1"
GIB = 2**30  # the rule's "44 GB (L40 46 GB)": the L40 holds 46068 MiB = 45 GiB, so the rule's unit is GiB
NAN = float("nan")


@dataclass(frozen=True)
class A1:
    """Sizes and the filed thresholds (predictions_amend1.json `rules`)."""

    step: int = 10
    cuts: tuple[int, ...] = (0, 8, 16, 24, 28)
    floor_max: float = 1e-3
    tol_min: float = 1e-6
    gap_min: float = 0.25
    ratio_max: float = 0.2
    klu_min: float = 0.05
    u_min: float = 0.3
    m_frac: float = 0.5
    margin: float = 0.3  # carrier and 'anticipation'
    gpu_max_gib: float = 44.0
    stage1_max_s: float = 15 * 60
    ref_kd: float = 1.41555  # results/D.json step 10, 128 rows of 512: reported beside K(0), K(28), no limit
    ref_dn: float = 0.09325

    def __post_init__(self) -> None:
        assert self.cuts[0] == 0 and self.cuts[-1] == 28 and {16, 24} <= set(self.cuts)


A = A1()
JOB_CFG = Config(n_rows=16, row_len=256, eval_batch=1)


class Weights:
    """W0 and FT = W0 + δW (fp32, added on the GPU as Job.ft does) of the 196 sites, pinned on the host."""

    def __init__(self, model: Any, dW: dict[str, Tensor]):
        self.lins = linears(model)
        assert set(dW) == set(SITES), "δW must hold all 196 sites"
        self.w0 = {s: lin.weight.data.cpu().pin_memory() for s, lin in self.lins.items()}
        self.ft = {s: (lin.weight.data + dW[s].to(DEV, F32)).cpu().pin_memory() for s, lin in self.lins.items()}
        self.now = 0

    def use(self, ell: int) -> None:
        """The model's weights set to H(ell): FT's in layers < ell, W0's after (0 = W0, 28 = FT), copied in place."""
        if ell != self.now:
            with torch.no_grad():
                for s, lin in self.lins.items():
                    lin.weight.data.copy_(self.ft[s] if parse_site(s)[0] < ell else self.w0[s])
            torch.cuda.synchronize()
            self.now = ell


@dataclass
class Inputs:
    """One row's Γ inputs from one clean forward, on the host: 29 taps and the 196 readouts x·V."""

    taps: list[Tensor]
    readouts: dict[str, Tensor]


class Run:
    def __init__(self, job: Job, W: Weights, rows: Tensor):
        self.job, self.W, self.rows, self.stage = job, W, rows, "1"
        self.ft_logits: list[Tensor] = []
        self.per_row: dict[str, list[float]] = {}  # each reading's per-row means, saved beside the values
        self.ft_in: list[Inputs] = []
        self.w0_in: list[Inputs] = []

    def capture(self) -> None:
        """FT's clean logits and both weight sets' Γ inputs for every row (host)."""
        for ell, store in ((28, self.ft_in), (0, self.w0_in)):
            self.W.use(ell)
            for tokens in self.rows:
                c = clean_forward(self.job.model, tokens[None].to(DEV), self.job.gamma(), self.job.comps())
                store.append(Inputs([t.cpu() for t in c.taps], {s: r.cpu() for s, r in c.readouts.items()}))
                if ell == 28:
                    self.ft_logits.append(c.logits.cpu())
                del c
        self.W.use(0)
        self.job.log(f"[capture] FT and W0 inputs of {len(self.rows)} rows, {self.job.peaks()}")

    def guard(self, i: int) -> None:
        """Stage 1's stop rules, after every row: GPU peak after row 0, wall time."""
        peak = torch.cuda.max_memory_allocated() / GIB
        if self.stage == "1" and i == 0 and peak > A.gpu_max_gib:
            self.job.stop("memory", f"GPU peak {peak:.2f} GiB after row 0 > {A.gpu_max_gib}")
        if self.stage == "1" and (el := time.time() - self.job.t0) > A.stage1_max_s:
            self.job.stop("time", f"stage 1 not done at {el:.0f} s > {A.stage1_max_s:.0f}")

    def reading(self, name: str, logits_of: Any, n_rows: int | None = None) -> float:
        """Mean over every position of KL(FT ‖ logits_of(i, tokens)); per-row means logged."""
        rows = self.rows[:n_rows]
        sums_ = []
        for i, tokens in enumerate(rows):
            q = logits_of(i, tokens[None].to(DEV))
            sums_.append(float(kl_per_position(self.ft_logits[i].to(DEV), q).double().sum()))
            del q
            self.guard(i)
        v = sum(sums_) / rows.numel()
        self.per_row[name] = [x / rows.shape[1] for x in sums_]
        self.job.log(f"[{name}] {v:.9g}  rows {[round(x, 6) for x in self.per_row[name]]}  {self.job.peaks()}")
        return v

    def k_cut(self, taps_to: int, reads_to: int) -> Any:
        """Masked logits with Γ given FT's taps 0..taps_to and FT's readouts of layers < reads_to, W0's the rest."""

        def fn(i: int, tokens: Tensor) -> Tensor:
            ft, w0 = self.ft_in[i], self.w0_in[i]
            taps = [(ft if j <= taps_to else w0).taps[j].to(DEV) for j in range(N_TAPS)]
            reads = {s: (ft if parse_site(s)[0] < reads_to else w0).readouts[s].to(DEV) for s in SITES}
            g = lower_importances(self.job.gamma(), taps, reads)
            del taps, reads
            return masked_logits(self.job.model, tokens, self.job.comps(), g)

        self.W.use(0)
        return fn

    def k_plain(self, ell: int) -> Any:
        """The unchanged path: `importances` with H(ell)'s weights in (0 = W0, 28 = FT), masked with W0's."""

        def fn(i: int, tokens: Tensor) -> Tensor:
            self.W.use(ell)
            _, g = importances(self.job.model, tokens, self.job.gamma(), self.job.comps())
            self.W.use(0)
            return masked_logits(self.job.model, tokens, self.job.comps(), g)

        return fn

    def unmasked(self, ell: int) -> Any:
        """H(ell)'s own logits (for KL_u); the weights go back to W0 when the reading is read."""
        self.W.use(ell)
        return lambda i, tokens: self.job.model(tokens, use_cache=False).logits


# ----------------------------------------------------------------------------- scoring (filed rules)


def ok(x: float) -> bool:
    return math.isfinite(x)


def r9(x: float) -> float:
    return round(x, 9)


def div(x: float, y: float) -> float:
    return x / y if ok(x) and ok(y) and y != 0 else NAN


def controls(v: dict[str, float]) -> dict[str, Any]:
    """control_floor, control_identity, control_effect; a missing value is NaN, and NaN misses."""
    g = lambda k: float(v.get(k, NAN))
    f = abs(div(g("floor_a"), g("floor_b")) - 1)
    tol = max(A.tol_min, 10 * f) if ok(f) else NAN
    i0, i28 = abs(div(g("K_0"), g("Kplain_W0")) - 1), abs(div(g("K_28"), g("Kplain_FT")) - 1)
    gap = g("K_0") - g("K_28")
    ratio = div(g("K_28"), g("K_0")) if g("K_0") > 0 else NAN
    out = {"f": f, "tol": tol, "identity_W0": i0, "identity_FT": i28, "gap": gap, "ratio": ratio,
           "KLu_0": g("KLu_0"), "K0_over_ref": div(g("K_0"), A.ref_kd), "K28_over_ref": div(g("K_28"), A.ref_dn),
           "control_floor": ok(f) and r9(f) <= A.floor_max,
           "control_identity": all(ok(x) for x in (i0, i28, tol)) and r9(i0) <= r9(tol) and r9(i28) <= r9(tol),
           "control_effect": (ok(gap) and r9(gap) >= A.gap_min and ok(ratio) and r9(ratio) <= A.ratio_max
                              and ok(g("KLu_0")) and r9(g("KLu_0")) >= A.klu_min)}
    out["holds"] = out["control_floor"] and out["control_identity"] and out["control_effect"]
    return out


def score(v: dict[str, float]) -> dict[str, Any]:
    """m, u, s from the means; the verdict (prediction 4) and the carrier (prediction 5) by their filed rules."""
    g = lambda k: float(v.get(k, NAN))
    den, klu0 = g("K_0") - g("K_28"), g("KLu_0")
    m = {ell: div(g("K_0") - g(f"K_{ell}"), den) for ell in A.cuts}
    u = {ell: div(klu0 - g(f"KLu_{ell}"), klu0) for ell in A.cuts[:-1]} | {28: 1.0}  # u(28) = 1: H(28) = FT
    s_taps, s_reads = div(g("K_0") - g("K_taps"), den), div(g("K_0") - g("K_reads"), den)
    out: dict[str, Any] = {"m": m, "u": u, "s_taps": s_taps, "s_reads": s_reads, "flags": []}
    if not (ok(u[16]) and ok(u[24])):
        out["l_star"], out["verdict"] = None, "unreadable"
    elif (star := 16 if r9(u[16]) >= A.u_min else 24 if r9(u[24]) >= A.u_min else None) is None:
        out["l_star"], out["verdict"] = None, "undecided"
    else:
        out["l_star"] = star
        mm, uu = m[star], u[star]
        if not ok(mm):
            out["verdict"] = "unreadable"
        else:
            out["verdict"] = "re-gating" if r9(mm) >= r9(A.m_frac * uu) else "copying"
            if r9(mm) >= r9(uu + A.margin):
                out["flags"].append("anticipation")
    if all(ok(m[x]) for x in (8, 16, 24)) and (m[8] > m[16] or m[16] > m[24]):
        out["flags"].append("non-monotone")
    d = s_taps - s_reads
    out["carrier"] = ("unreadable" if not ok(d) else "taps" if r9(d) >= A.margin
                      else "readouts" if r9(-d) >= A.margin else "shared")
    return out


# ----------------------------------------------------------------------------- stages


def stage1(run: Run) -> dict[str, float]:
    v: dict[str, float] = {}
    v["floor_a"] = run.reading("floor_a", run.k_plain(0), n_rows=1)
    v["floor_b"] = run.reading("floor_b", run.k_plain(0), n_rows=1)
    v["K_0"] = run.reading("K_0", run.k_cut(0, 0))
    v["K_28"] = run.reading("K_28", run.k_cut(28, 28))
    v["Kplain_W0"] = run.reading("Kplain_W0", run.k_plain(0))
    v["Kplain_FT"] = run.reading("Kplain_FT", run.k_plain(28))
    v["KLu_0"] = run.reading("KLu_0", run.unmasked(0))
    c = controls(v)
    write_json(OUT / "amend1_controls.json",
               {"values": v, "controls": c, "per_row": run.per_row, "s": time.time() - run.job.t0})
    run.job.log(f"[controls] {json.dumps(c)}")
    for name in ("control_floor", "control_identity", "control_effect"):
        if not c[name]:
            run.job.stop(name, f"{name} missed: {json.dumps(c)}")
    return v


def stage2(run: Run, v: dict[str, float]) -> None:
    run.stage = "2"
    todo = [(f"KLu_{ell}", lambda ell=ell: run.unmasked(ell)) for ell in (8, 16, 24)]
    todo += [(f"K_{ell}", lambda ell=ell: run.k_cut(ell, ell)) for ell in (16, 24, 8)]
    todo += [("K_taps", lambda: run.k_cut(28, 0)), ("K_reads", lambda: run.k_cut(-1, 28))]
    s2: dict[str, float] = {}
    for name, make in todo:
        s2[name] = run.reading(name, make())
        run.W.use(0)
        write_json(OUT / "amend1_stage2.json", {"values": s2, "per_row": run.per_row, "s": time.time() - run.job.t0})
        sc = score(v | s2)
        write_json(OUT / "amend1_score.json", {"controls": controls(v), "score": sc, "complete": False})
        run.job.log(f"[score] {json.dumps(sc)}")
    write_json(OUT / "amend1_score.json", {"controls": controls(v), "score": score(v | s2), "complete": True})
    run.job.log(f"[done] verdict {score(v | s2)['verdict']}, carrier {score(v | s2)['carrier']}, {run.job.peaks()}")


def check_dw(job: Job, path: Path, sums_file: Path) -> None:
    want = sums(sums_file).get(path.name)
    got = sha256(path) if path.is_file() else "absent"
    job.log(f"[dW] {path}: sha256 {got}, committed {want}")
    if got != want:
        job.stop("dW_file", f"{path}: sha256 {got} != committed {want}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    ap.add_argument("--sums", default=str(RESULTS / "regen_SHA256SUMS"))
    a = ap.parse_args()
    if not PRED_A1.is_file():
        sys.exit(f"refusing to start: {PRED_A1} missing")
    if (OUT / "stopped.json").is_file():
        sys.exit(f"refusing to start: {OUT / 'stopped.json'} exists (a stop rule fired; move it aside to rerun)")
    job = Job(JobOpts(out=str(OUT), e40_large="", decomp=a.decomp, results=str(OUT), e40=a.e40), JOB_CFG,
              lambda res, pred: {})
    job.stage = "1"
    job.log(f"[config] {A} rows {JOB_CFG.n_rows} x {JOB_CFG.row_len}, e40 {a.e40}")
    check_dw(job, Path(a.e40) / "dW" / f"step_{A.step}.safetensors", Path(a.sums))
    with torch.no_grad():
        W = Weights(job.model, job.dw(A.step))
        job.forget()
        job.gamma()
        run = Run(job, W, job.rows("maths"))
        run.capture()
        v = stage1(run)
        stage2(run, v)


if __name__ == "__main__":
    main()
