"""step_split_small (E25): E19's step split along the two small emoticon fine-tunes of E21 (plan:
reports/plans/2026-10-22_plan_small-update-reruns.md, "E25 step_split_small").

The paths: arm A on emoticon, seed 0, all 24 decomposed matrices, plain SGD at 0.01 and Adam at 3e-6 (E21's matched
pair, `experiments/sgd_finetune`), 1000 steps each. After every optimizer step `split.StepSplit` splits the actual weight
step: the components take the scaled adjoint part against the CURRENT alive (U, V) (`carry=True`), Delta the rest, so the
weights are the fine-tune's at every step; the fixed-bases twin (`carry=False`) steps beside it for the consistency
numbers only. Gamma frozen. Float32 on the GPU, as E19.

One stage, results dumped after every state (rules and expected readings: results/predictions.json, filed before running):
  1. W0 under the Pile harness at adversary seeds 0 and 1 (`faith_score.w0_check`); stop on failure.
  2. Per path (SGD first, then Adam): the fine-tune with both step splits; snapshots at steps 10, 100, 1000; the per-step
     series to results/large/per_step_<path>.json. Gate: the step-10 delta-W's rel_norm_max is sgd_finetune's within 1%
     (else stop: the path is not E21's); the step-10 do-nothing ratio is read first and checked against E21's in the score.
  3. Per path and scored step: do-nothing, the step-split state and the least-squares split through `harness.measure` with
     the Pile ratios. The behaviour kept in the pieces is the route split's unmasked P(target) (`p_cells["U1"]`: every
     component on, Delta out), E19's reading.
Usage: python experiments/step_split_small/run.py [--resume | --rescore] [--device cuda]
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}, experiments/transport_test/results/windows.json.
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
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
OUT = RESULTS / "step_split_small.json"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"
TOL = 1e-6  # the split bases' tolerance, as E14, E16, E19
DTYPE = torch.float32  # the step split's arithmetic, as E19
STEPS = (10, 100, 1000)  # the scored steps; the last is the run's length
PILE_ROWS, PILE_PGD_STEPS, PILE_SEEDS = 256, 20, (0, 1)
REL_NORM_TOL = 0.01  # gate: step-10 rel_norm_max within 1% of sgd_finetune's


@dataclass(frozen=True)
class FtPath:
    name: str
    optimizer: str
    lr: float
    ref_rel_norm_max: float  # sgd_finetune's step-10 reading (results/split.json, results/adam_matched.json)
    ref_nothing_pgd: float  # the same jobs' step-10 do-nothing adversarial ratio


PATHS = (FtPath("sgd", "sgd", 0.01, 0.0023840891662985086, 1.1217103113262998),
         FtPath("adam", "adam", 3e-6, 0.002234246116131544, 1.0847321463153172))


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(OUT, out)
    log(f"STOP: {why}")
    sys.exit(why)


# ----------------------------------------------------------------------------- the fine-tune with both step splits


def finetune_with_splits(ctx: Any, p: FtPath) -> tuple[dict[int, Any], dict[int, dict[str, float]], dict[int, Any], list[Any], float]:
    """One path's fine-tune with the step split (carry) and its fixed-bases twin stepping after every optimizer step.
    Returns per scored step the step-split state and the split's numbers, the fine-tune's held-out evals, the per-step
    log of both splits, and the carried-vs-fixed gap after step 1."""
    from commute import split, step_split
    from commute.finetune import ArmAConfig, finetune_arm_a

    carry = split.StepSplit(ctx.uv, ctx.sigma, ctx.W0, True, DTYPE, ctx.device)
    fixed = split.StepSplit(ctx.uv, ctx.sigma, ctx.W0, False, DTYPE, ctx.device)
    states: dict[int, Any] = {}
    nums: dict[int, dict[str, float]] = {}
    per_step: list[Any] = []
    gap1: list[float] = []

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        a, b = carry.step(params), fixed.step(params)
        per_step.append({"step": step, "carry": a, "fixed": b})
        if step == 1:
            gap1.append(step_split.carry_fixed_gap(carry, fixed))
            log(f"[{p.name} split] step 1 carried-vs-fixed gap {gap1[0]:.3e} (expected 0)")
        if step % 100 == 0 or step in (1, 10):
            log(f"[{p.name} split] step {step}: second/first {a['second_over_first']:.2e} c {a['c_min']:.3g}..{a['c_max']:.3g} "
                f"zeroed {a['n_zeroed']} | fixed c {b['c_min']:.3g}..{b['c_max']:.3g}")
        if step in STEPS:
            states[step], nums[step] = carry.state(), step_split.numbers(carry, fixed, ctx.uv_cpu, ctx.sigma)
            log(f"[{p.name} split] snapshot {step}: {json.dumps(nums[step])}")

    t0 = time.time()
    cfg = ArmAConfig(o_token=ctx.token, lr=p.lr, steps=STEPS[-1], checkpoints=STEPS, seed=0, optimizer=p.optimizer)
    ft = finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log, on_step)
    for s in STEPS:  # the step split's delta-W is the fine-tune's
        assert max(float((states[s][k][0] - ft.deltas[s][k]).abs().max()) for k in ft.deltas[s]) == 0.0
    log(f"[{p.name} finetune+split] {time.time() - t0:.0f} s; evals {json.dumps(ft.evals)}")
    return states, nums, ft.evals, per_step, gap1[0]


# ----------------------------------------------------------------------------- the states


def measure_step(ctx: Any, out: dict[str, Any], p: FtPath, step: int, state: Any) -> None:
    """At one scored step, with the Pile ratios: do-nothing first (step 10's is the gate), the step-split state, the
    least-squares split. Each record is dumped as it is written."""
    from commute import harness, split

    rec = out["paths"][p.name]["states"].setdefault(str(step), {})
    dW = {k: v[0] for k, v in state.items()}
    ls = split.split_state(dW, ctx.uv_cpu, ctx.bases)
    todo = {"nothing": lambda: harness.measure(ctx, f"{p.name}{step}_nothing", dW, None, True, log),
            "step_split": lambda: harness.measure(ctx, f"{p.name}{step}_step_split", dW, state, True, log),
            "least_squares": lambda: harness.measure(ctx, f"{p.name}{step}_least_squares", dW, ls, True, log)}
    for arm, fn in todo.items():
        if arm not in rec:
            rec[arm] = fn()
            dump(OUT, out)


def run_path(ctx: Any, out: dict[str, Any], p: FtPath) -> None:
    from commute import step_split

    po = out["paths"].setdefault(p.name, {"states": {}})
    po.pop("stopped", None)
    states, nums, evals, per_step, gap1 = finetune_with_splits(ctx, p)
    dump(LARGE / f"per_step_{p.name}.json", per_step)
    po.update({"carry_fixed_gap_step1": gap1, "per_step_max": step_split.per_step_maxima(per_step),
               "ft_eval": {str(s): evals[s] for s in STEPS}})
    for s in STEPS:
        po["states"].setdefault(str(s), {})["split"] = nums[s]
    rel = evals[10]["rel_norm_max"] / p.ref_rel_norm_max - 1
    po["gate_rel_norm"] = {"rel_norm_max": evals[10]["rel_norm_max"], "ref": p.ref_rel_norm_max, "rel_diff": rel,
                           "pass": abs(rel) <= REL_NORM_TOL}
    dump(OUT, out)
    log(f"[{p.name}] gate rel_norm_max {po['gate_rel_norm']}")
    if not po["gate_rel_norm"]["pass"]:  # this path is not E21's update: skip its states, run the other path
        po["stopped"] = f"step-10 rel_norm_max {evals[10]['rel_norm_max']:.6f} is not sgd_finetune's {p.ref_rel_norm_max:.6f}"
        dump(OUT, out)
        log(f"STOP {p.name}: {po['stopped']}")
        return
    for s in STEPS:
        measure_step(ctx, out, p, s, states.pop(s))


def main() -> None:
    from commute import faith_score, harness

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true", help="re-run the fine-tunes, score only the missing states")
    ap.add_argument("--rescore", action="store_true", help="recompute the score from the results file only (no GPU)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.rescore:
        out = json.loads(OUT.read_text())
        out["score"] = score(out)
        dump(OUT, out)
        return
    if json.loads(PRED.read_text()).get("status") != "filed":
        sys.exit(f"{PRED}: predictions are filed before running")
    t_start = time.time()
    out: dict[str, Any] = (json.loads(OUT.read_text()) if args.resume and OUT.is_file() else {"paths": {}})
    out.pop("stopped", None)
    out["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, PILE_ROWS, PILE_PGD_STEPS,
                       PILE_SEEDS, log)
    w0 = faith_score.w0_check(ctx.pile)
    old = out.get("W0")  # resumed records are ratios to the old job's W0: reuse them only if this job reads the same W0
    if old is not None and (out.get("device_measured") != out["device"]
                            or abs(w0["pile"][0]["pgd_kl"] / old["pile"]["0"]["pgd_kl"] - 1) > 0.01):
        log(f"[resume] W0 or card differs from the saved run ({out.get('device_measured')}): starting fresh")
        out["paths"] = {}
    out["W0"], out["device_measured"] = w0, out["device"]
    log(f"[stage W0] {time.time() - t_start:.0f} s; {json.dumps(out['W0'])}")
    dump(OUT, out)
    if not out["W0"]["pass"]:
        stop(out, "W0 is not at its known Pile values: nothing scored")
    for p in PATHS:
        run_path(ctx, out, p)
    stopped = [p for p, v in out["paths"].items() if "stopped" in v]
    out["score"] = f"not scored: {stopped} stopped at the gate" if stopped else score(out)
    out["seconds"] = time.time() - t_start
    dump(OUT, out)
    log(f"wrote {OUT} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


# ----------------------------------------------------------------------------- the score (rules: results/predictions.json)


E19_STEP1000 = 2.673  # E19's step-split state at step 1000 on the large Adam 3e-5 path (fail)
REF_FLOOR = 0.209  # the L40 floor under which sgd_finetune's references were read (as E22's G1)
CLASS_TOL = 0.05  # P3: absolute difference in the pieces' P(target) below which the two splits read "same"


def score(out: dict[str, Any]) -> dict[str, Any]:
    from commute import faith_score

    f = out["W0"]["pgd_floor"]
    P = out["paths"]

    def st(path: str, step: int, arm: str) -> dict[str, Any]:
        return P[path]["states"][str(step)][arm]

    def pgd(path: str, step: int, arm: str) -> float:
        return st(path, step, arm)["pile"]["pgd_kl"]

    def p_pieces(path: str, step: int, arm: str) -> float:  # every component on, Delta out
        return st(path, step, arm)["routes"]["p_cells"]["U1"]

    sc: dict[str, Any] = {"floor": f, "verdicts": {}}
    for path in P:
        for s in STEPS:
            for arm in ("nothing", "step_split", "least_squares"):
                sc["verdicts"][f"{path}@{s}/{arm}"] = {"ratio": pgd(path, s, arm), "verdict": faith_score.verdict(pgd(path, s, arm), f)}
    sc["C0_w0"] = out["W0"]["pass"]
    sc["C1_step1_identity"] = {path: {"gap": P[path]["carry_fixed_gap_step1"], "pass": P[path]["carry_fixed_gap_step1"] <= 1e-6}
                               for path in P}
    sc["G1_rel_norm"] = {path: P[path]["gate_rel_norm"] for path in P}
    sc["G1_nothing"] = {p.name: {"ratio": pgd(p.name, 10, "nothing"), "ref": p.ref_nothing_pgd,
                                 "tol": max(f, REF_FLOOR),
                                 "pass": abs(pgd(p.name, 10, "nothing") - p.ref_nothing_pgd) <= max(f, REF_FLOOR)} for p in PATHS}
    sc["G2_exact"] = {f"{path}@{s}": {
        "p_diff": abs(st(path, s, "step_split")["routes"]["p_cells"]["T1"] - P[path]["ft_eval"][str(s)]["p_o"]),
        "share_sum": st(path, s, "step_split")["routes"]["share_sum"],
        "delta_diff": abs(st(path, s, "step_split")["delta_change_frac"] - P[path]["states"][str(s)]["split"]["delta_share"])}
        for path in P for s in STEPS}
    for v in sc["G2_exact"].values():
        v["pass"] = v["p_diff"] <= 1e-3 and abs(v["share_sum"] - 1) <= 1e-6 and v["delta_diff"] <= 1e-3
    share = {path: {s: st(path, s, "step_split")["delta_change_frac"] for s in STEPS} for path in P}
    sc["P1_sgd_lower"] = {"sgd": share["sgd"], "adam": share["adam"],
                          "pass": all(share["sgd"][s] < share["adam"][s] for s in STEPS)}
    sc["P2_step1000_verdict"] = {path: {"ratio": pgd(path, 1000, "step_split"), "e19": E19_STEP1000,
                                        "verdict": faith_score.verdict(pgd(path, 1000, "step_split"), f)} for path in P}
    for v in sc["P2_step1000_verdict"].values():
        v["outcome"] = "fails like E19" if v["verdict"] == "fail" else ("passes, unlike E19" if v["verdict"] == "pass" else "inconclusive")
    p3: dict[str, Any] = {}
    for path in P:
        for s in STEPS:
            a, b = p_pieces(path, s, "step_split"), p_pieces(path, s, "least_squares")
            ea = st(path, s, "step_split")["routes"]["shares"]["e_delta"]
            eb = st(path, s, "least_squares")["routes"]["shares"]["e_delta"]
            p3[f"{path}@{s}"] = {"p_pieces_step_split": a, "p_pieces_least_squares": b, "diff": a - b,
                                 "outcome": "same" if abs(a - b) < CLASS_TOL else ("less in the pieces" if a < b else "more in the pieces"),
                                 "e_delta_step_split": ea, "e_delta_least_squares": eb}
    sc["P3_pieces"] = p3
    gates_ok = all(v["pass"] for v in sc["C1_step1_identity"].values()) and all(v["pass"] for v in sc["G2_exact"].values())
    if not gates_ok:
        sc["P1_sgd_lower"]["pass"] = "unscored: C1 or G2 failed"
    return sc


if __name__ == "__main__":
    main()
