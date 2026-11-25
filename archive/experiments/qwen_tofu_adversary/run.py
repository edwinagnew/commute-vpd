"""E89: does VPD's adversary fail doing nothing on E81's TOFU update, about ten times u1's size? One 80 GB card, ~2.7 h. Plan: reports/plans/2026-11-05_plan_e89-tofu-adversary.md; rules: rules.py,
results/predictions.json (the run refuses to start without it, and while results/stopped.json exists).

E85's and E80 Amendment 2's code path, called, not copied: set_up, start through amend1.read, the same 128 web rows, 20 sign steps of 0.1, one source per site, final read, fp32 weights and fp32
components. Only the update differs: an fp32 δW from this job's artifact, which `fetch_inputs` puts at <decomp parent>/e81_tofu/ in place of E40's. Order, cheapest and control first: clean KL of
do-nothing against W0; X and left in Δ for the rarely-on half; clean KL of the rarely-on state; W0 s0, W0 s1, each within 20% of Amendment 2's or a stop; N s0, Rare s0, N s1, Rare s1.
results/results.json after every read (a rerun resumes at the next unread one), every number in results/run.log, and the half's X beside --decomp in e89_rare, never e85_rare, which would hold u1's.

Usage (repo root, 80 GB GPU; inputs beside --decomp, or WANDB_API_KEY to fetch them): .venv/bin/python experiments/qwen_tofu_adversary/run.py --decomp /root/qwen
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import rules
import torch
import upload

from commute.jobs import load_experiment
from commute.qwen_sft import mean_kl
from commute.qwen_vpd import SITES

e85 = load_experiment("qwen_d1_teeth", "run")  # make_rare, rare_change, rare_state; its amend2.moved, amend1.read, and E80's set_up, start, check, stop, log, guarded
amend2, amend1, e80, e76 = e85.amend2, e85.amend1, e85.e80, e85.e76
r2 = rules.r2
DEV, F64 = e80.DEV, e80.F64
RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT, STOPPED, LOG = RESULTS / "predictions.json", RESULTS / "results.json", RESULTS / "stopped.json", RESULTS / "run.log"
ORDER = (("N", 0), ("Rare", 0), ("N", 1), ("Rare", 1))


def fetch_inputs(decomp: Path) -> tuple[Path, Path]:
    """E80's fetches with this update in place of E40's, which nothing on this path reads. Each is kept if complete; `upload.verified` checks the committed sum on every call."""
    e80.qwen_files.fetch(decomp, e80.log)
    upload.fetch(decomp.parent / "e81_tofu")
    e80.upload.fetch(decomp.parent / "e80_inputs")
    dw = decomp.parent / "e81_tofu" / upload.NAME
    e80.log(f"[dW] {dw}: sha256 matches {upload.COMMITTED.name}, {dw.stat().st_size / 1e9:.2f} GB")
    return dw, decomp.parent / "e80_inputs"


def rare_dir(J: Any) -> Path:
    return J.x_dir.parent / "e89_rare"


def clean_kl(J: Any, name: str) -> float:
    """Mean KL(W0 ‖ the state) over every position of the web rows, as E85 Amendment 1's clean reads. A rerun keeps it."""
    reads = J.out.setdefault("clean", {})
    if name not in reads:
        with amend1.STATE[name](J):
            reads[name] = mean_kl(J.model, J.rows["W"], J.W0, None)
        e80.save(J)
    e80.log(f"[clean {name}] KL against W0 {reads[name]}, {e80.peak()}")
    return reads[name]


@torch.no_grad()
def left_in_delta(J: Any) -> None:
    """Plain Σ‖Δ′ - Δ‖² / Σ‖δW‖², pooled: E85's read of the half, without its Gaussian. Written; a rerun keeps it."""
    if "left_in_delta" not in J.out:
        e85.make_rare(J)
        size, rare = 0.0, 0.0
        for s in SITES:
            U_A, V_A = J.alive.uv(s, DEV, F64)
            M, c = J.dw[s].to(DEV, F64), e85.rare_change(J, s)
            pos = c[1]
            assert pos is not None
            d = amend2.moved(c, U_A[:, pos], V_A[:, pos])
            size, rare = size + float((M * M).sum()), rare + float((d * d).sum())
        J.out["left_in_delta"] = {"Rare": rare / size, "size": size}
        e80.save(J)
    left = J.out["left_in_delta"]
    e80.log(f"[left in delta] {json.dumps(left)}; ‖δW‖ {math.sqrt(left['size']):.6f}, E86's 17.998449; {e80.peak()}")
    if left["Rare"] > 1:
        e80.log("[left in delta] above 1: logged, no stop")
    e80.check(J, "Rare_left", left["Rare"], rules.rare_left_ok(left["Rare"]), f"finite and above least squares' {rules.RARE_LEFT_LOW:.4f}")


def stage_build(J: Any) -> None:
    """Before any start. The update moves the rows; the half is built; the two states' clean reads agree, since their weights are the same."""
    n = clean_kl(J, "N")
    e80.check(J, "N_moves", n, rules.n_clean_ok(n), f"at least {rules.N_CLEAN_MIN}")
    left_in_delta(J)
    rare = clean_kl(J, "Rare")
    e80.check(J, "Rare_same_weights", abs(rare - n), r2.near(rare, n, rules.SAME_TOL), f"within {rules.SAME_TOL} of do-nothing's")
    rescore(J)


def stage_w0(J: Any) -> None:
    for seed in rules.SEEDS:
        amend1.read(J, "W0", seed)
        got, want = J.out["starts"][f"W0_W_s{seed}"]["kl"], rules.W0_WANT[seed]
        e80.check(J, f"W0_s{seed}", got, r2.near_rel(got, want, rules.W0_TOL), f"Amendment 2's {want:.3f} within {rules.W0_TOL:.0%}")
        rescore(J)


def rescore(J: Any) -> None:
    J.out["score"] = rules.score(J.out["starts"], J.out.get("clean", {}), J.out.get("left_in_delta", {}), J.out["checks"])
    e80.save(J)
    e80.log(f"[score] {json.dumps(J.out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    J = e80.set_up(Path(a.decomp).expanduser().resolve())
    stage_build(J)
    stage_w0(J)
    for state, seed in ORDER:
        amend1.read(J, state, seed)
        rescore(J)
    e80.log(f"[done] {e80.peak()}")


def score() -> None:
    """From what results.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score(out["starts"], out.get("clean", {}), out.get("left_in_delta", {}), out["checks"])
    e76.write_json(OUT, out)
    e80.log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    e80.OUT, e80.STOPPED, e80.LOG = OUT, STOPPED, LOG  # E80's save, stop, check and log read these globals, so no earlier job's files are written
    e80.fetch_inputs = fetch_inputs  # set_up reads this global
    e85.rare_dir = rare_dir  # make_rare and rare_change read this global
    amend1.STATE |= {"Rare": e85.rare_state}  # "W0" and "N" are amend1's
    e80.guarded(main, score)
