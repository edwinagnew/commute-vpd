"""kv_task Amendment 1 (2026-10-23): is E10's cipher failure about the update's size, its damage elsewhere, the number
of new associations, or the scored step? Plan: reports/2026-10-23_plan_kv-task-amend1.md (Q4 of the research log).

No new training. E10's saved Adam 1e-5 component-training states (results/large/<task>/pullback/state_step<n>.pt,
cluster NFS) go through the Pile harness (harness.measure, 256 val rows, 20-step adversary, seed 0) as ratios to a
same-job W0, scored by faith_score.state_verdict against W0's own seed floor. E10's run.py stays as it ran.

Per shard, in order, each stage dumped as it finishes:
  W0       the Pile harness at W0 under adversary seeds 0, 1, 2 (faith_score.w0_check); off its known values -> stop
  gate     each state's held-out P(target), KL elsewhere and size of change against E10's curve at that step; off -> stop
  control  shard 0 only, before any new reading: rand32 @30 (E10's scored state) must read E10's PGD ratio 2.81 within
           25% (a 'fail' again: the harness's positive control); off -> stop
  states   the shard's other states
Shards: 0 rand32 (30 control, then 100, 300, 10); 1 ident32 (100, 300, 10), launched once shard 0 is past its control.

Usage (repo root, project venv, GPU):
  amend1.py --predictions   file results/predictions_amend1.json (refuses to overwrite)
  amend1.py --shard N       -> results/kv_task_amend1_shard{N}.json
  amend1.py --score         both shard files -> results/kv_task_amend1.json with the filed outcome (CPU, JSON only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions_amend1.json"
E10_JSON = RESULTS / "kv_task.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

CONTROL = ("rand32", 30)
SHARDS: tuple[tuple[tuple[str, int], ...], ...] = ((CONTROL, ("rand32", 100), ("rand32", 300), ("rand32", 10)),
                                                    (("ident32", 100), ("ident32", 300), ("ident32", 10)))
PILE_ROWS, PGD_STEPS, PILE_SEEDS = 256, 20, (0, 1, 2)  # E10's harness; three seeds give the same-job floor
GATE_TOL, SIZE_TOL, CONTROL_TOL = 0.01, 0.01, 0.25  # absolute on P and KL elsewhere; relative on size and on 2.81
LATE, EARLY = (100, 300), 10

PREDICTIONS: dict[str, Any] = {
    "about": "kv_task Amendment 1 (reports/2026-10-23_plan_kv-task-amend1.md): the Pile harness on E10's saved Adam 1e-5 "
             "component-training states rand32 and ident32 at steps 10, 100, 300 (and rand32 @30, the control), no new "
             "training. Filed before any run",
    "measurement": {
        "pile": "harness.measure: Pile val rows 0..255, two 128-row blocks, one 20-step PGD adversary per block (seed 0), "
                "stochastic; each a ratio to this shard's W0 under seed 0 (relative). At W0: 1 (absolute PGD 0.69, "
                "CI-masked 0.348)",
        "verdict": "faith_score.state_verdict with p_min 0 (the step-10 states have not learned; the verdict reads the "
                   "decomposition, not the task): stochastic and PGD ratios each against W0's own floor f (max/min - 1 "
                   "over adversary seeds 0, 1, 2 in the shard; PGD 0.209 on the L40): pass <= 1.5 / (1 + f), fail >= "
                   "1.5 (1 + f) (PGD about 1.24 and 1.81), inconclusive between; any fail -> fail, any inconclusive -> "
                   "inconclusive",
        "size": "max over the 24 matrices of ||dW|| / ||W0||, absolute; 0 at W0",
        "damage_elsewhere": "held-out KL of the target model against W0 at every position but the key, nats, absolute; "
                            "0 at W0",
        "blind_to": "the trained behaviour (the marker never occurs in Pile); damage a 20-step adversary does not find "
                    "(it is far from converged, reports/2026-10-22_review_vpd-adversary-convergence.md); one adversary "
                    "seed per state, one batch order",
    },
    "states_from_E10_curve": {"rand32": {"10": [0.0073, 0.217, 0.001], "30": [0.0154, 0.355, 0.847],
                                         "100": [0.0202, 0.021, 0.968], "300": [0.0212, 0.012, 0.959]},
                              "ident32": {"10": [0.0069, 0.240, 0.393], "30": [0.0118, 0.069, 0.971],
                                          "100": [0.0141, 0.015, 0.998], "300": [0.0151, 0.011, 0.994]},
                              "columns": ["size", "damage elsewhere", "held-out P(target)"]},
    "gates": {
        "G0_w0": "first in each shard: faith_score.w0_check (seed-0 PGD within 10% of 0.69, CI-masked within 2% of "
                 "0.348, PGD floor <= 0.25). Fails: the shard stops",
        "G1_states_are_E10s": "each state's held-out P and KL elsewhere within 0.01 of E10's curve at its step, size "
                              "within 1%. Fails: the shard stops",
        "C1_control": "shard 0, first state: rand32 @30's PGD ratio within 25% of E10's 2.81, so it reads 'fail' again. "
                      "Fails: the shard stops and shard 1 is not launched",
    },
    "predictions": [
        {"id": "P1_rand32_late_passes", "rule": "rand32 @100 and @300 both read 'pass'",
         "predicted": "pass, least sure (sizes 0.020, 0.021 exceed every Adam pass so far, max 0.018 in E21)",
         "why": "E21: Adam component training failed only while its damage elsewhere was >= 0.2 nats; rand32's falls to "
                "0.021 and 0.012 by these steps"},
        {"id": "P2_ident32_matched_size_passes", "rule": "ident32 @100 and @300 both read 'pass'",
         "predicted": "pass", "why": "sizes 0.014, 0.015 match rand32 @30's 0.0154 with damage 0.015, 0.011; ident32 "
                                     "@30 passed at 1.19 (E10, E21)"},
        {"id": "P3_step10_not_pass", "rule": "rand32 @10 and ident32 @10 both read 'fail' or 'inconclusive'",
         "predicted": "pass (neither reads 'pass')",
         "why": "both carry damage 0.22-0.24 at size 0.007; E21's Adam states with damage >= 0.2 read fail or "
                "inconclusive. First out-of-sample test of E21's after-the-fact damage cut (0.147)"},
    ],
    "outcome": {
        "rule": "from P1 and P2's four late states: ident32 both pass and rand32 both pass -> 'step'; ident32 both pass "
                "and rand32 both fail -> 'content'; all four fail -> 'size'; anything else -> 'mixed'",
        "step": "E10's failure belonged to its scored step (its damage elsewhere), not to the size (to 0.021) or to 32 "
                "new associations: the Q4 card closes, no pair-count sweep",
        "content": "32 new associations fail where the identity control passes; size not exactly matched (0.020 vs "
                   "0.015) and E21's low-rate Adam passed rand32 at 0.0146: read as content x size, next the pair-count "
                   "sweep through /design",
        "size": "the identity control fails at rand32 @30's size with little damage, and rand32 fails too: the cipher "
                "result is a size result",
        "mixed": "reported per state; more adversary seeds on the split states before any reading",
        "predicted": "step",
    },
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(path, out)
    sys.exit(why)


def e10_at(task: str, step: int) -> dict[str, Any]:
    r = json.loads(E10_JSON.read_text())["tasks"][task]["pullback"]
    c = r["curve"][step - 1]
    rec = {"p_o": c["heldout_p_o"], "kl_other": c["heldout_kl_other"], "size": c["rel_norm_max"]}
    if step == r["scored_step"]:
        rec["pile"] = r["eval"]["pile"]["ratios"]
    return rec


def measure_state(ctx: Any, task: str, step: int, w0: dict[str, Any]) -> dict[str, Any]:
    """One saved state: held-out numbers, the Pile harness, E10's numbers beside them, the gate and the verdict."""
    from commute import arms, faith_score, harness

    st = torch.load(LARGE / task / "pullback" / f"state_step{step}.pt", weights_only=False)
    dW = {k: v[0] for k, v in st.items()}
    ho = arms.heldout_of(ctx.model, ctx.W0, ctx.held, ctx.token, ctx.device, dW)
    m = harness.measure(ctx, f"{task}@{step}", dW, st, True, log)
    ref = e10_at(task, step)
    gate = abs(ho["p_o"] - ref["p_o"]) <= GATE_TOL and abs(ho["kl_other"] - ref["kl_other"]) <= GATE_TOL \
        and abs(m["rel_norm_max"] / ref["size"] - 1) <= SIZE_TOL
    v = faith_score.state_verdict(m["pile"], ho["p_o"], w0, p_min=0.0)
    rec = {"task": task, "step": step, "heldout": ho, "size": m["rel_norm_max"], "pile": m["pile"],
           "routes": m["routes"]["shares"], "e10": ref, "G1": gate,
           "verdict": v["verdict"], "verdicts": v["pile"]}
    log(f"[{task}@{step}] P {ho['p_o']:.3f} KL elsewhere {ho['kl_other']:.3f} size {m['rel_norm_max']:.4f} | PGD "
        f"{m['pile']['pgd_kl']:.3f} stochastic {m['pile']['stochastic_kl']:.3f} -> {v['verdict']} | G1 {gate}")
    return rec


def run_shard(shard: int, device: str) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: amend1.py --predictions")
    path = RESULTS / f"kv_task_amend1_shard{shard}.json"
    t0 = time.time()
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, 1e-6, PILE_ROWS, PGD_STEPS,
                       PILE_SEEDS, log)
    out: dict[str, Any] = {"shard": shard, "W0": faith_score.w0_check(ctx.pile), "states": {},
                           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    log(f"[W0] {time.time() - t0:.0f} s; {json.dumps({k: v for k, v in out['W0'].items() if k != 'pile'}, default=float)}")
    dump(path, out)
    if not out["W0"]["pass"]:
        stop(path, out, "W0 is not at its known values: nothing measured")
    task = SHARDS[shard][0][0]
    w = torch.load(LARGE / f"windows_{task}.pt", weights_only=False)
    tctx = harness.with_windows(ctx, w, w["table"])
    for t, step in SHARDS[shard]:
        rec = measure_state(tctx, t, step, out["W0"])
        out["states"][f"{t}@{step}"] = rec
        dump(path, out)
        if not rec["G1"]:
            stop(path, out, f"{t}@{step}: the saved state does not read E10's held-out numbers")
        if (t, step) == CONTROL:
            rel = rec["pile"]["pgd_kl"] / rec["e10"]["pile"]["pgd_kl"] - 1
            out["control"] = {"pgd_rel": rel, "pass": abs(rel) <= CONTROL_TOL}
            log(f"[control] rand32@30 PGD {rec['pile']['pgd_kl']:.3f}, {rel:+.3f} relative to E10 -> "
                f"{'PASS' if out['control']['pass'] else 'FAIL'}")
            dump(path, out)
            if not out["control"]["pass"]:
                stop(path, out, "the control does not reproduce E10's 2.81: no new state measured")
    out["seconds"] = time.time() - t0
    dump(path, out)
    log(f"wrote {path} ({out['seconds']:.0f} s)")


def outcome(v: dict[str, str]) -> str:
    """The filed rule over the four late states' verdicts (`v`: 'task@step' -> verdict)."""
    r, i = [v[f"rand32@{s}"] for s in LATE], [v[f"ident32@{s}"] for s in LATE]
    if all(x == "pass" for x in i):
        return "step" if all(x == "pass" for x in r) else "content" if all(x == "fail" for x in r) else "mixed"
    return "size" if all(x == "fail" for x in i + r) else "mixed"


def score() -> None:
    shards = [json.loads((RESULTS / f"kv_task_amend1_shard{n}.json").read_text()) for n in range(len(SHARDS))]
    states = {k: s for sh in shards for k, s in sh["states"].items()}
    v = {k: s["verdict"] for k, s in states.items()}
    res = {"P1": all(v[f"rand32@{s}"] == "pass" for s in LATE), "P2": all(v[f"ident32@{s}"] == "pass" for s in LATE),
           "P3": all(v[f"{t}@{EARLY}"] != "pass" for t in ("rand32", "ident32")), "outcome": outcome(v), "verdicts": v}
    dump(RESULTS / "kv_task_amend1.json", {"score": res, "shards": shards})
    log(json.dumps(res, indent=1))


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--shard", type=int, choices=range(len(SHARDS)))
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.shard is not None:
        run_shard(args.shard, args.device)
    elif args.score:
        score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
