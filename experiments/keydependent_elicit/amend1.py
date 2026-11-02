"""E54 Amendment 1: steering's share in the route split's unit. S_lo = the rise in log-odds of the target that B3's one
vector gives W0, over the fine-tune's log-odds gain, on B3's scoring half, with B3's vector and block (E52/E54). Read on
E54's six saved weight changes; r is stage2.json's. Plan: reports/plans/2026-10-29_plan_e54-keydependent-elicit.md,
"Amendment 1 (2026-10-30)". Rules and expected values: results/predictions_amend1.json (filed before the run, read here,
never edited).

  stage 0, alone (exit 1 on a miss): per state, the file's sha256 is stage2.json's; steering.b3 gives stage2.json's block
           and S within 1e-6; steering.steer_at with the log P readout gives the same S within 1e-6.
  then:    steering.steer_at with the log-odds readout at that block -> S_lo; elicit.score_amend1 -> P1-lo, P2-lo, P3-own.
-> results/amend1.json. Every scored number is also printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import elicit as EL
from commute import steering as ST

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions_amend1.json"
STATES = tuple(f"{k}_s{s}" for k in ("elicit", "cipher", "cipher_own") for s in (0, 1))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(obj: dict[str, Any]) -> None:
    fin = lambda o: (str(o) if isinstance(o, float) and not math.isfinite(o) else
                     {k: fin(v) for k, v in o.items()} if isinstance(o, dict) else [fin(v) for v in o] if isinstance(o, list) else o)
    (RESULTS / "amend1.json").write_text(json.dumps(fin(obj), indent=1, sort_keys=True))


def check_filed() -> None:
    rel = str(PRED.relative_to(REPO))
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False).returncode
    if git("ls-files", "--error-unmatch", rel) or git("diff", "--quiet", "HEAD", "--", rel):
        sys.exit(f"{rel} is not committed unchanged: file the predictions first")


def stage0(net: ST.Net, t: ST.Task, dW: ST.DW, ref: dict[str, Any], sha_ref: str, tol: float, sha: str) -> dict[str, Any]:
    """B3 as run.py read it, and steer_at's log P path at B3's block, against stage2.json's S and block."""
    g = ST.gains(net, t, dW)
    b = ST.b3(net, t, dW, g)
    S_b3 = EL.share_or_nan(b["S"], g["G_score"])
    S_path = EL.share_or_nan(ST.steer_at(net, t, dW, b["layer"])["share"], g["G_score"])
    ok = sha == sha_ref and b["layer"] == ref["layer"] and abs(S_b3 - ref["S"]) <= tol and abs(S_path - ref["S"]) <= tol
    return {"sha256": sha, "layer": b["layer"], "G_score": g["G_score"], "S_b3": S_b3, "S_path": S_path, "S_ref": ref["S"],
            "dS_b3": S_b3 - ref["S"], "dS_path": S_path - ref["S"], "ok": bool(ok)}


def reading(net: ST.Net, t: ST.Task, dW: ST.DW, layer: int, ref: dict[str, Any]) -> dict[str, Any]:
    """S_lo at B3's block, beside stage2.json's r and S. G_lo_all (all held-out windows) is reported beside route_split's
    G_logodds, the same quantity through another forward path (not a stop)."""
    lo = ST.steer_at(net, t, dW, layer, ST.logodds_at)
    S_lo = EL.share_or_nan(lo["share"], lo["G"])
    return {"layer": layer, "rise_lo": lo["rise"], "G_lo_score": lo["G"], "S_lo": S_lo, "S": ref["S"], "r": ref["r"],
            "r_minus_S_lo": ref["r"] - S_lo, "G_lo_all": lo["G_all"], "G_logodds_route": ref["G_logodds"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="E54's saved windows and weight changes")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    check_filed()
    P = json.loads(PRED.read_text())
    s2 = json.loads((RESULTS / "stage2.json").read_text())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # run.py's setting
    net = ST.Net(a.device)
    tasks = {n: ST.load_task(net, a.large / f"windows_{n}.pt", n, n) for n in ("elicit", "cipher")}
    task_of = lambda n: tasks["elicit" if n.startswith("elicit") else "cipher"]
    dws = {n: torch.load(a.large / f"{n}.pt", weights_only=True) for n in STATES}
    git = lambda *c: subprocess.run(["git", "-C", str(REPO), *c], capture_output=True, text=True, check=False).stdout.strip()
    out: dict[str, Any] = {"predictions_sha256": sha256(PRED), "git_head": git("rev-parse", "HEAD"),
                           "git_dirty": git("status", "--porcelain", "--untracked-files=no"), "stage0": {}, "states": {}}

    tol = P["stage0"]["S_tol"]
    for n in STATES:
        rec = stage0(net, task_of(n), dws[n], s2["states"][n], s2["finetunes"][n]["sha256"], tol, sha256(a.large / f"{n}.pt"))
        out["stage0"][n] = rec
        log(f"[stage0 {n}] block {rec['layer']} S_b3 {rec['S_b3']:.10f} S_path {rec['S_path']:.10f} ref {rec['S_ref']:.10f} "
            f"dS {rec['dS_b3']:.1e}/{rec['dS_path']:.1e} (tol {tol:.0e}) sha {'ok' if rec['sha256'] == s2['finetunes'][n]['sha256'] else 'MISMATCH'} ok {rec['ok']}")
    out["stage0_pass"] = all(r["ok"] for r in out["stage0"].values())
    dump(out)
    if not out["stage0_pass"]:
        sys.exit("[stage0] FAIL: stop and tell the user")

    for n in STATES:
        st = out["states"][n] = reading(net, task_of(n), dws[n], out["stage0"][n]["layer"], s2["states"][n])
        log(f"[read {n}] block {st['layer']} rise_lo {st['rise_lo']:.4f} G_lo_score {st['G_lo_score']:.4f} S_lo {st['S_lo']:.4f} "
            f"S {st['S']:.4f} r {st['r']:.4f} r-S_lo {st['r_minus_S_lo']:.4f} | G_lo_all {st['G_lo_all']:.5f} vs route {st['G_logodds_route']:.5f}")
        dump(out)
    out["score"] = EL.score_amend1(out["states"], P["thresholds"])
    out["r_gap_matched"] = s2["score"]["P1"]["gap"]  # the ratio table's r gap, beside P1-lo's S_lo gap
    dump(out)
    log(f"[score] {json.dumps(out['score'], default=str)}")


if __name__ == "__main__":
    main()
