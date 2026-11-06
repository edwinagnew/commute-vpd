"""E117 follow-up, description only: why the six E53 next-token states (lowercase, cipher, broad cipher; step 300) switch
on no subcomponent at position 63 under the filed rule (mean over windows of the importance change > 0.3).
Two readings fit "K_VPD = 0 while max |dg| = 1": the change is spread thinly over many subcomponents, or each window
switches on its own subcomponents strongly and the mean over windows washes them out. Per state, alive subcomponents
only, dg = g(state) - g(W0) at position 63 on E53's 500 scored windows:

  max_mean_dg        the largest mean over windows of dg (and the subcomponent holding it)
  n_mean_gt          count with mean dg > 0.1, > 0.2 and > 0.3 (the last is K_VPD, 0 in E117: a free check)
  n_frac_gt_0.3      count with dg > 0.3 on at least 10% and at least 50% of windows
  max_window_dg      the largest single dg at position 63

No rule and no prediction beyond the W0 control (results/predictions_k0.json): with delta-W = 0 every number is 0, else
stop. Reuses run.py's context, file table, sha256 check and task loaders. Writes results/k0_read.json; prints every number.

Usage (cluster):  sbatch cluster/neuron_vpd_count_k0.sbatch
"""

from __future__ import annotations

import json
import sys
from typing import Any

import torch

from commute import vpd
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.keydep import log, sha256

R = load_experiment("neuron_vpd_count", "run")
PRED, OUT = R.RESULTS / "predictions_k0.json", R.RESULTS / "k0_read.json"
STATES = ["lowercase_s0", "lowercase_s1", "e53_cipher_s0", "e53_cipher_s1", "broad_cipher_s0", "broad_cipher_s1"]
THR, LOW, FRACS = 0.3, (0.1, 0.2, 0.3), (0.1, 0.5)


@torch.no_grad()
def read(c: Any, t: Any, dW: Any, B: int) -> dict[str, Any]:
    """The numbers of the module docstring for one state on one task's windows, at t.half."""
    h, total, over, n, max_w = t.half, {}, {}, 0, 0.0
    for toks in t.toks.split(B):
        g0 = vpd.compute_ci(c.m, toks, chunk=len(toks)).g
        with c.net.at(dW):
            g1 = vpd.compute_ci(c.m, toks, chunk=len(toks)).g
        for k in g0:
            d = (g1[k][:, h].float() - g0[k][:, h].float()).cpu()
            d = torch.where(c.alive[k].cpu().bool(), d, torch.zeros_like(d))  # dead subcomponents never count
            total[k] = total.get(k, 0) + d.double().sum(0)
            over[k] = over.get(k, 0) + (d > THR).sum(0)
            max_w = max(max_w, float(d.max()))
        n += len(toks)
    mean = {k: v / n for k, v in total.items()}
    frac = {k: v.double() / n for k, v in over.items()}
    best = max(mean, key=lambda k: float(mean[k].max()))
    return {"n_windows": n, "max_mean_dg": float(mean[best].max()), "argmax": [best, int(mean[best].argmax())],
            "n_mean_gt": {str(x): int(sum(int((v > x).sum()) for v in mean.values())) for x in LOW},
            "n_frac_gt_0.3": {str(f): int(sum(int((v >= f).sum()) for v in frac.values())) for f in FRACS},
            "max_window_dg": max_w}


def is_zero(rec: dict[str, Any]) -> bool:
    return (rec["max_mean_dg"] == 0 and rec["max_window_dg"] == 0 and not any(rec["n_mean_gt"].values())
            and not any(rec["n_frac_gt_0.3"].values()))


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the W0 control first: {PRED}")
    require_filed(PRED)
    B = R.load(R.PRED)["settings"]["batch"]["pile"]  # E117's batch for the Pile family, the same windows and position
    table, F_all = R.E98.state_table(), R.E98.files()
    keep = {table[n][0] for n in STATES} | {f"e46/windows_{t}" for t in R.TT.E46_TASKS} | {"e54/windows_elicit", "e54/windows_cipher"}
    F = {k: F_all[k] for k in sorted(keep)}
    bad = [k for k, (p, h) in F.items() if not p.is_file() or sha256(p) != h]
    if bad:
        sys.exit(f"sha256 mismatch or missing: {bad}")
    c = R.Ctx()
    tasks, _ = R.E98.E63.load_tasks(c.net, R.TT.performed_anchor(), F)
    zero = {k: torch.zeros_like(w) for k, w in c.net.W0.items()}
    out: dict[str, Any] = {"w0": {}, "rows": {}, "settings": {"thr": THR, "low": LOW, "fracs": FRACS, "batch": B}}
    for task in sorted({table[n][1] for n in STATES}):  # the W0 control, every task, before any state
        rec = read(c, tasks[task], zero, B)
        out["w0"][task] = rec
        log(f"[w0 {task}] {json.dumps(rec)}")
        if not is_zero(rec):
            OUT.write_text(json.dumps(out, indent=1))
            log(f"[w0 {task}] FAIL: not all zero; stop")
            hard_exit(1)
    for n in STATES:
        task = table[n][1]
        rec = read(c, tasks[task], R.E98.state(F, table[n][0]), B) | {"task": task, "position": tasks[task].half}
        out["rows"][n] = rec
        log(f"[rows {n}] {json.dumps(rec)}")
        OUT.write_text(json.dumps(out, indent=1))
    out["done"] = True
    OUT.write_text(json.dumps(out, indent=1))
    log("[done]")
    hard_exit(0)


if __name__ == "__main__":
    main()
