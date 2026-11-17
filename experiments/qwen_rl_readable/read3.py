"""P3.4 run 3, the reads of the null's saved states on the MATS cluster's L40s (run 2's card type), $0, and the verdict.
Plan: reports/plans/2026-11-13_plan_qwen-rl-readable-run3.md; rules in results/predictions_run3.json "controls" and
"verdict". Every read is read2.py's code, unchanged; only the folders and the rule file are this run's.

  --stage gpu --controls-only, then --stage gpu --states N_s0@<t>[,N_s70@<t>]: read2's controls first and alone (W0 and
        E69's saved R at step 100, run 2's bands), then each named state: read2's `read_one` (output change and its
        multiple, greedy GSM8K pass and fits on the 1319 test problems, |δW|), its sha256 checked against its pod's first.
        Exit 3 when the time left is under 1.2 times the longest read: resubmit, done states skip.
  --stage verdict   no model: run 2's R at step 75 (its reads and GSM8K answers) against each seed's matched null on the
        problems that fit at both (paired), `match3.verdict`.
Results in results/run3_reads/ (rehearsal: results/run3_tiny/): controls.json, states/<pod>@<step>.json, reads.log,
verdict.json. GSM8K answers under <large>/reads/acc/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import match3

from commute import qwen_rl as q
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import sha256

READ2 = load_experiment("qwen_rl_readable", "read2")  # Reader, stage_gpu, read_states (frozen, used unchanged)
RUN3 = load_experiment("qwen_rl_readable", "run3")
JOB, RUN2 = READ2.JOB, READ2.RUN2
PRED = RUN3.PRED
R_READS = RUN2.RESULTS / "run2_reads" / "states"  # run 2's L40 reads of R at step 75
R_ACC = RUN2.RESULTS / "run2_dl" / "reads" / "acc"  # and its GSM8K answers
D = dict[str, Any]


def out_dir(size: str) -> Path:
    return RUN2.RESULTS / ("run3_reads" if size == "full" else "run3_tiny")


class Reader(READ2.Reader):
    """read2's reader, writing this run's folder."""

    def __init__(self, size: str, large: Path) -> None:
        self.pod, self.dir = READ2.Read(), out_dir(size)
        self.dir.mkdir(parents=True, exist_ok=True)
        JOB.Job.__init__(self, JOB.config(size, RUN2.TAG), "controls")
        self.large, self.shared_large = large / "reads", JOB.LARGE
        self.large.mkdir(parents=True, exist_ok=True)


def states(size: str, large: Path, names: list[str]) -> dict[str, D]:
    """name -> {file, sha256} from the null pod's "saves"."""
    d, out = RUN3.results_dir(RUN3.SIZES[size]), {}
    for n in names:
        pod, t = n.split("@")
        rec = JOB.read_json(d / f"{pod}.json").get("saves", {}).get(t)
        if rec is None:
            sys.exit(f"{n}: not in {d / pod}.json saves")
        out[n] = {"file": large / rec["file"], "sha256": rec["sha256"]}
    return out


def stage_verdict(size: str, large: Path, P: D) -> None:
    """Per seed: R at step 75 (run 2's read, its answers checked against the filed sha256) and the null's pod end and
    read; R against N on problems that fit at both (`qwen_rl.gain`, paired); `match3.verdict`."""
    d, st, per = out_dir(size), READ2.read_states(out_dir(size)), {}
    for seed in map(str, RUN2.SEEDS):
        r_name, acc_r = f"R_s{seed}@75", R_ACC / f"R_s{seed}@75_gsm8k.jsonl"
        if sha256(acc_r) != P["verdict"]["R_acc_sha256"][seed]:
            sys.exit(f"{acc_r}: sha256 differs from the filed one")
        end = JOB.read_json(RUN3.results_dir(RUN3.SIZES[size]) / f"N_s{seed}.json").get("end")
        n_name = f"N_s{seed}@{end['step']}" if end else None
        n = st.get(n_name, {}) if n_name else {}
        x = {"end": end, "R": JOB.read_json(R_READS / f"{r_name}.json").get("pass"), "N": n.get("pass"),
             "N_multiple": n.get("multiple"), "N_fits": n.get("fits"), "se": None}
        if n:
            x["vs_N"] = q.gain(JOB.load_rollouts(large / "reads" / "acc" / f"{n_name}_gsm8k.jsonl"), JOB.load_rollouts(acc_r))
            x["se"] = x["vs_N"]["se"]
        per[seed] = x
    out = match3.verdict(per, P["verdict"]["null_held_floor"]) | {"seeds": per}
    JOB.write_text(d / "verdict.json", json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("gpu", "verdict"))
    ap.add_argument("--size", default="full", choices=("full", "tiny"))
    ap.add_argument("--large", type=Path, default=RUN2.RESULTS / "run3_dl",
                    help="the states' root: the runner's downloads, <root>/<pod>/dW/step_<t>.safetensors (rehearsal: run3.py --large)")
    ap.add_argument("--states", default="", help="comma-separated <pod>@<step>")
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--wall-min", type=float, default=28.0)
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if a.stage == "verdict":
        stage_verdict(a.size, a.large, P)
        return
    job = Reader(a.size, a.large)
    out = job.get("controls") or {"controls": {}}
    if out.get("stopped"):
        sys.exit(f"the reads stopped at {out['stopped']!r}: the lead decides")
    if not a.controls_only and not out.get("done"):
        sys.exit("the controls are not read yet: submit --controls-only first")
    if not a.controls_only and not a.states:
        sys.exit("name the states: --states N_s0@<t>")
    todo = {} if a.controls_only else states(a.size, a.large, a.states.split(","))
    READ2.stage_gpu(job, out, todo, P, a.wall_min * 60, True)
    hard_exit(0)


if __name__ == "__main__":
    main()
