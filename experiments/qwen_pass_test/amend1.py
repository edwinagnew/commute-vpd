"""E68 Amendment 1, the ceiling of the pass test ($0, one cluster L40, debug QoS). Plan:
reports/plans/2026-11-03_plan_e68-d1-pass-test.md; rules: results/predictions_amend1.json. E68's reader with one forward
changed: W0's masked forward plus the whole update, unmasked, at every matrix. Controls first, on u1: the added term at 0
reads E68's KD, and the same hook without masks rebuilds the fine-tuned model. Each read writes
results/amend1/<update>_<read>.json as it ends, a rerun skips it; every number goes to results/amend1/run.log.

Usage (repo root; `-P` as run.py):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_pass_test/amend1.py \
    --decomp DIR --wall-min 26
"""
# pyright: reportMissingImports=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from amend2 import kl
from job import CFG, E40_RESULTS, JobOpts, write_json
from torch import Tensor

from commute.qwen_vpd import SITES, _hooks, _linear, importances

HERE = Path(__file__).resolve().parent
RESULTS, OUT = HERE / "results", HERE / "results" / "amend1"
PRED = RESULTS / "predictions_amend1.json"
ROWS_HOOK = 8  # control_hook's rows
ZERO_TOL, HOOK_SHARE, LINE = 1e-6, 0.01, 1.5
OPEN_ROWS, OPEN_T = 32, (0.1, 0.5, 1.0)  # Amendment 2

spec = importlib.util.spec_from_file_location("pass_run", HERE / "run.py")  # `run` means qwen_maps/run.py here
assert spec is not None and spec.loader is not None
pr = importlib.util.module_from_spec(spec)
sys.modules["pass_run"] = pr
spec.loader.exec_module(pr)


def forward_plus(model: Any, tokens: Tensor, comps: Any, g: dict[str, Tensor] | None, dW: dict[str, Tensor],
                 scale: float, t: float = 0.0) -> Tensor:
    """Each matrix's output, masked by g + t (1 - g) (`g` None: unmasked, the model's own), plus scale * x dWᵀ. dW stays on the host:
    on the GPU its 2.8 GB would pass the reader's memory stop."""

    def replace(site: str) -> Any:
        V, U = comps.vu(site)

        def hook(m: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            x = args[0].to(V.dtype)
            gs = None if g is None else g[site].to(V.dtype)
            base = out.to(V.dtype) if gs is None else (x @ V * (gs + t * (1 - gs))) @ U
            return (base + scale * x @ dW[site].to(x).T).to(out.dtype)

        return hook

    with _hooks([_linear(model, s).register_forward_hook(replace(s)) for s in SITES]):
        return model(tokens, use_cache=False).logits


class CeilingReader(pr.PassReader):
    """`read` calls this row, once per row of the current update; `dW`, `scale`, `masked` set before each read."""

    dW: dict[str, Tensor]
    scale: float = 1.0
    masked: bool = True
    t: float = 0.0

    def row(self, tokens: Tensor, own_key: str, want: set[str], srcs: dict[str, dict[str, Tensor]]) -> dict[str, float]:
        model, comps = self.job.model, self.job.comps()
        logits, g = importances(model, tokens, self.job.gamma(), comps)  # W0's forward
        out = forward_plus(model, tokens, comps, g if self.masked else None, self.dW, self.scale, self.t)
        self.use("FT")
        ref = model(tokens, use_cache=False).logits
        res = {"K": kl(ref, out), "nll": pr.nll(out, tokens), "ft": pr.nll(ref, tokens), "K_clean": kl(ref, logits)}
        self.use("W0")
        return res


def mean_of(rd: dict[str, Any], key: str, n: int | None = None) -> float:
    xs = rd["sums"][key][:n]
    return sum(xs) / (len(xs) * rd["T"])


def verdict(k_c: float, kd: float, k0: float) -> tuple[str, float]:
    if kd - k0 < 0.05:
        return "unscored", float("nan")
    r2 = (kd - k_c) / (kd - k0)
    label = "can pass" if round(k_c, 9) <= round(LINE * k0, 9) else "near" if round(r2, 9) >= 0.5 else "cannot pass"
    return label, r2


class Reads:
    """One update's reads: each writes its file as it ends, a rerun skips what is written, past the wall clock exit 0."""

    def __init__(self, job: Any, rd: CeilingReader, wall_s: float):
        self.job, self.rd, self.wall_s, self.u = job, rd, wall_s, pr.P.updates[0]

    def select(self, u: Any, rows: Tensor) -> None:
        self.u = u
        self.job.select(u, OUT / "unused")
        self.rd.rows["maths"], self.rd.w0_logits, self.rd.dW = rows, [], self.job.dw(u.step)
        self.rd.drop_ft()

    def step(self, key: str, scale: float, masked: bool, n: int | None = None) -> dict[str, Any]:
        path = OUT / f"{self.u.name}_{key}.json"
        if path.is_file():
            return json.loads(path.read_text())
        if time.time() - self.job.t0 > self.wall_s:
            self.job.log(f"cut short at {self.u.name} {key}; resubmit to resume")
            sys.exit(0)
        self.rd.scale, self.rd.masked = scale, masked
        self.job.stage = f"{self.u.name}/{key}"
        r = self.rd.read("do_nothing", self.u.step, "maths", {"K"}, n=n)
        write_json(path, r)
        return r


def controls(job: Any, reads: Reads, ref_kd: float) -> None:
    hook = reads.step("hook", 1.0, False, ROWS_HOOK)
    k_hook, k_clean = mean_of(hook, "K"), mean_of(hook, "K_clean")
    job.log(f"[control_hook] K_clean_plus {k_hook:.6g}, K_clean {k_clean:.6g}, ratio {k_hook / k_clean:.4g}")
    if not k_hook <= HOOK_SHARE * k_clean:
        job.stop("control_hook", f"{k_hook} > {HOOK_SHARE} * {k_clean}")
    k_z = mean_of(reads.step("zero", 0.0, True), "K")
    job.log(f"[control_zero] K_Z {k_z:.9g}, E68 KD {ref_kd:.9g}")
    if not abs(k_z / ref_kd - 1) <= ZERO_TOL:
        job.stop("control_zero", f"{k_z} vs {ref_kd}")


def opening(job: Any, reads: Reads, rows: dict[str, Tensor]) -> None:
    """Amendment 2 (results/predictions_amend2.json): the masks opened in steps on u1 and u2's first rows."""
    out: dict[str, Any] = {}
    for u in pr.P.updates[:2]:
        reads.select(u, rows[u.rows])
        ref = {k: json.loads((OUT / f"{u.name}_{f}.json").read_text()) for k, f in (("on", "ceiling"), ("off", "zero"))}
        k: dict[str, Any] = {f"0_{d}": mean_of(ref[d], "K", OPEN_ROWS) for d in ("on", "off")}
        for t in OPEN_T:
            for d, scale in (("on", 1.0), ("off", 0.0)):
                reads.rd.t = t
                r = reads.step(f"open_t{t}_{d}", scale, True, OPEN_ROWS)
                k[f"{t}_{d}"] = mean_of(r, "K")
                k["clean"] = mean_of(r, "K_clean")
        k1, k0 = k["1.0_on"], k["0_on"]
        k["cause"] = "masks" if k1 <= 0.1 * k0 else "leftover" if k1 >= 0.5 * k0 else "both"
        out[u.name] = k
        job.log(f"[open {u.name}] {json.dumps(k)}")
        write_json(OUT / "open.json", out)
    job.log(f"[done] {job.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--wall-min", type=float, required=True)
    ap.add_argument("--open", action="store_true")  # Amendment 2
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"refusing to start: {PRED} missing")
    if (OUT / "stopped.json").is_file():
        sys.exit(f"refusing to start: {OUT / 'stopped.json'} exists")
    OUT.mkdir(exist_ok=True)
    job = pr.PassJob(JobOpts(out=str(OUT), e40_large=str(E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                             results=str(OUT)), CFG, lambda *_: {})
    rows = {"maths": job.rows("maths")[:, : pr.P.row_len], "a1": pr.a1_rows()}
    w0_read = {b: json.loads((RESULTS / u / "W0.json").read_text()) for b, u in (("maths", "u1"), ("a1", "u3"))}
    scores: dict[str, Any] = {}
    with torch.no_grad():
        job.gamma()
        reads = Reads(job, CeilingReader(job), a.wall_min * 60)
        if a.open:
            return opening(job, reads, rows)
        for u in pr.P.updates:
            reads.select(u, rows[u.rows])
            if u.name == "u1":
                controls(job, reads, mean_of(json.loads((RESULTS / "u1" / "do_nothing.json").read_text()), "K"))
            zero, ceil, w0 = reads.step("zero", 0.0, True), reads.step("ceiling", 1.0, True), w0_read[u.rows]
            k_c, k_z, k0 = mean_of(ceil, "K"), mean_of(zero, "K"), mean_of(w0, "ci")
            label, r2 = verdict(k_c, k_z, k0)
            gain_c = (sum(w0["sums"]["nll"]) - sum(ceil["sums"]["nll"])) / (sum(w0["sums"]["clean"]) - sum(ceil["sums"]["ft"]))
            scores[u.name] = {"K_C": k_c, "KD": k_z, "K0": k0, "r2_C": r2, "gain_C": gain_c, "label": label}
            job.log(f"[ceiling {u.name}] {json.dumps(scores[u.name])}")
            write_json(OUT / "score.json", scores)
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    main()
