"""baseline_weights (E51): do weight-space statistics of a fine-tune's delta-W tell copy (elicit) from cipher (teach)?
Plan: reports/plans/2026-10-29_plan_baselines-scoping.md, section "B1 = E51". Rules, settings and the control's expected
values: results/predictions.json (filed before the run, read here, never edited).

Per state and matrix: the top-k energy share of delta-W on W0's top singular directions, the intruder count of W0 + dW
(Shuttleworth et al. 2024) and the effective rank of dW (Roy and Vetterli 2007); each aggregated over the 24 matrices
weighted by ||dW||^2, with a matched-norm Gaussian null per matrix. The control (synthetic re-weighting, an injected new
direction, the null of the first state) runs first and stops the job on a miss. Weights only: W0 is read as tensors from
the target checkpoint, no model is built. CPU, minutes.

Usage (repo root):  ~/commute-agents/fleet/cpu-run -c 2 --mem 4G python3 experiments/baseline_weights/run.py
Writes results/results.json; every number is also printed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import env
from commute.vpd_files import TARGET_CKPT, TARGET_RUN

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SRC = HERE.parent / "regating_table" / "results" / "large"  # E46's regenerated E12 and E36 weight changes
STATES = {"ident32_s0": "copy", "ident32_s1": "copy", "ident32b_s0": "copy", "ident32b_s1": "copy",
          "rand32_s0": "cipher", "rand32_s1": "cipher", "rand32b_s0": "cipher", "rand32b_s1": "cipher",
          "code_step10": "learned nothing"}
STATS = ("share", "intruders", "erank")
Mat = dict[str, Tensor]


def log(msg: str) -> None:
    print(msg, flush=True)


class Base:
    """W0's SVD per matrix, and the three statistics of a delta-W against it."""

    def __init__(self, W0: Mat, k: int, eps: float) -> None:
        self.W0, self.k, self.eps = W0, k, eps
        self.svd = {m: torch.linalg.svd(w, full_matrices=False) for m, w in W0.items()}

    def random_share(self, m: str) -> float:
        d_out, d_in = self.W0[m].shape
        return (self.k / d_out + self.k / d_in) / 2

    def matrix(self, m: str, dW: Tensor) -> dict[str, float]:
        U0, _, Vh0 = self.svd[m]
        e = dW.pow(2).sum()
        share = ((U0[:, :self.k].T @ dW).pow(2).sum() + (dW @ Vh0[:self.k].T).pow(2).sum()) / (2 * e)
        Ut = torch.linalg.svd(self.W0[m] + dW, full_matrices=False).U[:, :self.k]
        intruders = int(((U0.T @ Ut).abs().amax(dim=0) < self.eps).sum())
        s = torch.linalg.svdvals(dW)
        p = s[s > 0] / s.sum()
        return {"share": float(share), "intruders": intruders, "erank": math.exp(float(-(p * p.log()).sum())),
                "norm2": float(e)}

    def state(self, dW: Mat) -> dict[str, Any]:
        per = {m: self.matrix(m, dW[m]) for m in sorted(self.W0)}
        total = sum(v["norm2"] for v in per.values())
        agg = {s: sum(v[s] * v["norm2"] for v in per.values()) / total for s in STATS}
        agg["random_share"] = sum(self.random_share(m) * v["norm2"] for m, v in per.items()) / total
        return {"agg": agg, "per_matrix": per}


def load_W0(keys: list[str]) -> Mat:
    sd = torch.load(env.VPD_DIR / TARGET_RUN / "ckpt" / TARGET_CKPT, map_location="cpu", weights_only=False)
    return {m: sd[f"{m}.weight"].double() for m in keys}


def load_dW(name: str, W0: Mat) -> Mat:
    d = torch.load(SRC / f"{name}.pt", map_location="cpu", weights_only=True)
    assert sorted(d) == sorted(W0) and all(d[m].shape == W0[m].shape for m in W0), name
    return {m: v.double() for m, v in d.items()}


def gaussian_like(dW: Mat, seed: int) -> Mat:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m in sorted(dW):
        x = torch.randn(dW[m].shape, generator=g, dtype=torch.float64)
        out[m] = x * (dW[m].norm() / x.norm())
    return out


def reweight(base: Base, ref: Mat) -> Mat:
    """Pure re-weighting of W0's top-k singular pairs, equal weights, scaled to the reference's norm per matrix."""
    out = {}
    for m, (U0, _, Vh0) in base.svd.items():
        d = U0[:, :base.k] @ Vh0[:base.k]
        out[m] = d * (ref[m].norm() / d.norm())
    return out


def inject(base: Base, seed: int) -> Mat:
    """One new direction per matrix, twice W0's top singular value: one intruder expected in every matrix."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m, (_, S0, _) in base.svd.items():
        d_out, d_in = base.W0[m].shape
        u = torch.randn(d_out, generator=g, dtype=torch.float64)
        v = torch.randn(d_in, generator=g, dtype=torch.float64)
        out[m] = 2 * S0[0] * torch.outer(u / u.norm(), v / v.norm())
    return out


def control(base: Base, ref: Mat) -> dict[str, Any]:
    """The filed control; every check must hold before any state is read."""
    rw, inj, nul = (base.state(x)["agg"] for x in (reweight(base, ref), inject(base, 0), gaussian_like(ref, 0)))
    checks = {
        "reweight_share": rw["share"] >= 0.999,
        "reweight_intruders": rw["intruders"] == 0,
        "reweight_erank": abs(rw["erank"] - base.k) <= 0.01,
        "inject_intruders": abs(inj["intruders"] - 1) <= 0.05,
        "inject_erank": abs(inj["erank"] - 1) <= 0.01,
        "inject_share": inj["share"] <= 3 * inj["random_share"],
        "null_share": abs(nul["share"] / nul["random_share"] - 1) <= 0.10,
        "null_intruders": nul["intruders"] == 0,
    }
    return {"reweight": rw, "inject": inj, "null_ident32_s0": nul, "checks": checks, "pass": all(checks.values())}


def separates(rows: dict[str, Any], stat: str) -> dict[str, Any]:
    """The filed rule: gap between the kinds' means over the larger within-kind range, strict."""
    vals = {kind: [r["agg"][stat] for n, r in rows.items() if STATES[n] == kind] for kind in ("copy", "cipher")}
    assert all(len(v) == 4 and all(math.isfinite(x) for x in v) for v in vals.values()), (stat, vals)
    mean = {k: sum(v) / 4 for k, v in vals.items()}
    spread = max(max(v) - min(v) for v in vals.values())
    gap = abs(mean["copy"] - mean["cipher"])
    return {"mean": mean, "gap": gap, "spread": spread, "separates": gap > spread}


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())  # refuses to start without it
    torch.set_num_threads(2)
    k, eps = P["settings"]["k"], P["settings"]["intruder_eps"]
    keys = sorted(torch.load(SRC / "ident32_s0.pt", map_location="cpu", weights_only=True))
    assert len(keys) == 24, keys
    base = Base(load_W0(keys), k, eps)
    ref = load_dW("ident32_s0", base.W0)
    out: dict[str, Any] = {"settings": {"k": k, "eps": eps}, "control": control(base, ref), "states": {}}
    log(f"control: {json.dumps(out['control'])}")
    (RESULTS / "results.json").write_text(json.dumps(out, indent=1))
    if not out["control"]["pass"]:
        log("control missed its filed values: stop")
        sys.exit(1)
    for i, (name, kind) in enumerate(STATES.items()):
        dW = load_dW(name, base.W0)
        rec = base.state(dW)
        rec["kind"] = kind
        rec["null"] = base.state(gaussian_like(dW, i))["agg"]
        out["states"][name] = rec
        log(f"{name} ({kind}): {json.dumps(rec['agg'])} null {json.dumps(rec['null'])}")
        (RESULTS / "results.json").write_text(json.dumps(out, indent=1))
    scored = {n: r for n, r in out["states"].items() if STATES[n] != "learned nothing"}
    out["score"] = {s: separates(scored, s) for s in STATS}
    log(f"score: {json.dumps(out['score'])}")
    (RESULTS / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
