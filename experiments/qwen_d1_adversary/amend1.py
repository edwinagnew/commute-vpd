"""E80 Amendment 1: do placements separate from do-nothing on the FineWeb rows? (One RunPod H100 PCIe, six starts of ~31 min.) Plan:
reports/plans/2026-11-04_plan_e80-adversary-do-nothing.md, Amendment 1; rules: rules_amend1.py, results/predictions_amend1.json (the job refuses to start without it,
and while results/stopped_amend1.json exists).

E80's harness unchanged (run.py): the same set-up, states and `start`. Order, control first: W0 s1, N s1 (then "reproduces" or a stop), I s0, A s0, I s1, A s1, all on W.
W0 s0 and N s0 are E80's, read from results/results.json. run.py's save, stop and log read the module globals OUT, STOPPED and LOG, so setting them here sends
results to results/amend1.json (resumable per start), the stop to stopped_amend1.json and every line to results/run_amend1.log, never E80's files.

Usage (repo root, 80 GB GPU): .venv/bin/python experiments/qwen_d1_adversary/amend1.py --decomp /root/qwen
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import rules_amend1 as rules
import run

e76 = run.e76
PRED, OUT, STOPPED, LOG = (run.RESULTS / f for f in ("predictions_amend1.json", "amend1.json", "stopped_amend1.json", "run_amend1.log"))
E80_OUT = run.RESULTS / "results.json"
run.OUT, run.STOPPED, run.LOG = OUT, STOPPED, LOG
ORDER = (("W0", 1), ("N", 1), ("I", 0), ("A", 0), ("I", 1), ("A", 1))
STATE = {"W0": lambda J: nullcontext(), "N": run.n_state, "I": run.i_state, "A": run.a_state}
STEP = re.compile(r"\[(\S+)\] seed \d+ step \d+: KL (\S+)$")
READS: dict[str, list[float]] = {}
_log = run.log


def tee(msg: str) -> None:
    """run.py's log (console and run_amend1.log), also keeping each start's per-step reads for the mean of the last five."""
    _log(msg)
    if m := STEP.match(msg):
        READS.setdefault(m[1], []).append(float(m[2]))


run.log = tee


def e80_starts() -> dict[str, Any]:
    out = e76.read_json(E80_OUT)
    keys = ("W0_W_s0", "N_W_s0")
    assert out and all(k in out["starts"] for k in keys), "E80's results.json lacks W0_W_s0 or N_W_s0"
    return {k: {"kl": out["starts"][k]["kl"]} for k in keys}


def read(J: run.Job, state: str, seed: int) -> None:
    key = f"{state}_W_s{seed}"
    if key in J.out["starts"]:
        return
    with STATE[state](J):
        run.start(J, key, "W", seed)
    last = READS[key][-5:]
    J.out["starts"][key]["last5"] = statistics.fmean(last)
    run.save(J)
    run.log(f"[{key}] mean of the last {len(last)} step reads {J.out['starts'][key]['last5']:.4f} (logged, not ruled on)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions_amend1.json missing, or stopped_amend1.json exists")
    e80 = e80_starts()
    J = run.set_up(Path(a.decomp).expanduser().resolve())
    J.out["e80"] = e80
    run.save(J)
    for state, seed in ORDER:
        read(J, state, seed)
        if (state, seed) == ("N", 1):
            rep = rules.reproduces(rules.pair({**e80, **J.out["starts"]}, "N")[1])
            run.log(f"[reproduces] {rep}")
            if rep != "reproduces":
                run.stop("reproduces", f"r_1(N) read {rep}: do-nothing's rise does not reproduce on W, nothing further is paid for")
    run.log(f"[done] {run.peak()}")


def score() -> None:
    """From what amend1.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score({**out.get("e80", {}), **out["starts"]}, out["checks"])
    e76.write_json(OUT, out)
    run.log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    run.guarded(main, score)
