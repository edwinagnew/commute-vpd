"""small_finetune_splits (E22): the splits of a small fine-tune at later steps, beside the large one, on one card.
Plan: reports/plans/2026-10-22_plan_small-update-reruns.md, section "E22" (parents E14 part 1 and E16).

Three emoticon fine-tunes (arm A, seed 0, all 24 matrices, 1000 steps): SGD 0.01 and Adam 3e-6 (E21's matched pair)
and Adam 3e-5 (the large reference). Order, each writing results/small_finetune_splits.json as it finishes:
  1. W0 under the Pile harness, adversary seeds 0 and 1 (faith_score.w0_check); stop if it fails.
  2. per fine-tune, in FINETUNES order: train with delta-W at E9's nine checkpoints; at step 10 the do-nothing,
     least-squares and scaled adjoint states through harness.measure, and gate G1 (the step-10 update and its
     do-nothing reading reproduce E21's); then the series at all nine checkpoints: split.block_table (four blocks with
     their null) and split.rescaling_table (rescaling share with its random-update null, the algebraic block).
  3. per fine-tune, steps 100 and 1000: the same three states.
The cheap series sits right after each fine-tune so every code path has run within the first fine-tune's minutes.

Usage (repo root):  run.py --predictions  (files results/predictions.json once)  |  run.py all
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}, experiments/transport_test/results/windows.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
OUT = RESULTS / "small_finetune_splits.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"

CKPTS = (1, 2, 3, 5, 10, 30, 100, 300, 1000)  # E9's checkpoints: the series' rows
STATE_STEPS = (10, 100, 1000)  # the placements' steps (E16)
TRAIN_STEPS, TOL, NULL_DRAWS = 1000, 1e-6, 2  # null draws as E14
PILE_ROWS, PILE_PGD_STEPS, PILE_SEEDS = 256, 20, (0, 1)
FINETUNES = {"sgd": ("sgd", 0.01), "adam_small": ("adam", 3e-6), "adam_large": ("adam", 3e-5)}
SMALL = ("sgd", "adam_small")
L40_FLOOR = 0.209  # the floor E21's references were read under (sgd_finetune split.json, L40)
# E21's step-10 readings (sgd_finetune results/split.json and adam_matched.json): rel_norm_max and do-nothing PGD ratio.
G1_REF = {"sgd": {"rel_norm_max": 0.0023840891662985086, "nothing_pgd": 1.122},
          "adam_small": {"rel_norm_max": 0.002234246116131544, "nothing_pgd": 1.085},
          "adam_large": {"rel_norm_max": 0.011190634220838547, "nothing_pgd": None}}  # large: verdict "fail" (E16 2.52, E21 2.18)
RNM_TOL = 0.01  # relative
# E14 (lora_split.json, table ft, pooled_ns) at CKPTS: the large fine-tune's series, the reproduction gate G2.
E14_OUT_OUT = (0.184, 0.175, 0.185, 0.199, 0.219, 0.274, 0.331, 0.352, 0.410)
E14_RESCALING = (2.03, 1.99, 1.91, 1.91, 2.08, 2.65, 4.26, 5.70, 11.43)
SGD_SIZE_CUT = 0.009  # E21's SGD size cut (rel_norm_max) between passing and failing component training
assert set(STATE_STEPS) <= set(CKPTS) and CKPTS[-1] == TRAIN_STEPS and len(E14_OUT_OUT) == len(E14_RESCALING) == len(CKPTS)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default))


VERDICT = ("faith_score.verdict on the PGD ratio to the same-job W0 (adversary seed 0): pass <= 1.5 / (1 + f), fail >= "
           "1.5 (1 + f), inconclusive between; f = max/min - 1 of W0's PGD KL over adversary seeds 0, 1 in this job "
           "(L40 0.21 -> 1.24 / 1.81; H100 about 0.075 -> 1.40 / 1.61)")
PREDICTIONS: dict[str, Any] = {
    "about": "E22 small_finetune_splits. Plan reports/plans/2026-10-22_plan_small-update-reruns.md (section E22, design check "
             "there; approved classes 1-2, the user's instruction 2026-10-22). Filed before the run.",
    "setup": {
        "finetunes": "emoticon arm A (CE on o at the trigger + KL elsewhere), batch 32, clip 1, seed 0, all 24 matrices, "
                     f"1000 steps: {FINETUNES}; delta-W kept at {CKPTS}",
        "states": f"at steps {STATE_STEPS}: do-nothing and least-squares split (harness.nothing_and_split), scaled adjoint "
                  "split (split.adjoint_state); each through harness.measure: Pile ratios to W0 "
                  f"({PILE_ROWS} rows, {PILE_PGD_STEPS} PGD steps), held-out P(target) at the trigger, Delta's share of "
                  "||delta-W||^2, route split",
        "series": f"at all of {CKPTS}: split.block_table (four input-weighted blocks with their analytic null) and "
                  f"split.rescaling_table (rescaling share with its measured null over {NULL_DRAWS} Gaussian draws of the "
                  "same input-weighted energy, the algebraic outside-both share); scored pooling: energy over the "
                  "non-saturated matrices (pooled_ns, as E14's outside-both rule); pooled_all reported",
        "verdict": VERDICT,
    },
    "gates": {
        "G0_w0": "first, alone: faith_score.w0_check (seed-0 PGD within 10% of 0.69, CI-masked within 2% of 0.348, floor "
                 "<= 0.25). Fails: stop, nothing scored",
        "G1_reproduce_E21": f"per fine-tune at step 10: rel_norm_max within {RNM_TOL:.0%} (relative) of E21's {G1_REF}, and "
                            "the do-nothing PGD ratio within max(f, 0.209) relative of E21's 1.122 (SGD) / 1.085 (Adam 3e-6); "
                            "the large one's do-nothing verdict 'fail' (E16 2.52 H100, E21 2.18 L40). Fails: the run goes "
                            "on; predictions reading that fine-tune are 'unscored'",
        "G2_large_series_reproduces_E14": "the large fine-tune's outside-both / null within 0.03 absolute of E14's "
                                          f"{E14_OUT_OUT} and rescaling / null within 25% of E14's {E14_RESCALING} at "
                                          "every checkpoint (pooled_ns). Fails: P3 and P4's reference is this job's, "
                                          "reported with the miss",
        "C1_draws_at_null": "positive control of the block code: the random draws' measured outside-both share over its "
                            "analytic null in [0.85, 1.15] at every row (pooled_ns)",
        "harness_G2": "per least-squares state: P(target) equals the fine-tune's to 1e-3 and the route shares sum to 1",
    },
    "predictions": [
        {"id": "P1_small_later_steps", "review_rows": [1, 4],
         "measure": "verdict of the do-nothing and least-squares states of SGD 0.01 and Adam 3e-6 at steps 100 and 1000",
         "pass": "all eight 'pass'", "relative_vs_absolute": "ratio to the same-job W0",
         "predicted": "pass: the large fine-tune's do-nothing passes from step 100 (E16 1.14, E21 1.12) and the small ones "
                      "pass at step 10 (1.12, 1.08); the small updates grow more slowly than the large one",
         "two_sided_reading": f"SGD at 1000: pass = the step story (a later small update passes); fail with rel_norm_max > "
                              f"{SGD_SIZE_CUT} (E21's SGD size cut) = the size story; fail below the cut or inconclusive = "
                              "neither, reported as such",
         "expected": "W0: 1. Small do-nothing at 10: about 1.1 (E21). Large do-nothing at 10 >= 1.8 (control, reported), "
                     "at 100 about 1.1, at 1000 1.4-1.5 (E16 H100 1.50, inconclusive). Large least squares 1000: 2.1 (E16)",
         "blind_to": "the trigger (about 3 positions in 131k Pile tokens); damage the 20-step adversary does not find",
         "positive_control": "the large fine-tune's step-10 do-nothing and least-squares states read 'fail' in the same job",
         "noise_floor": "W0's seed spread f, through the verdict band; one training seed"},
        {"id": "P2_adjoint", "review_rows": [8, 9],
         "measure": "the scaled adjoint state of SGD 0.01 and Adam 3e-6 at steps 10, 100, 1000: verdict, Delta's share "
                    "of ||delta-W||^2, held-out P(target)",
         "pass": "verdict 'pass' and Delta share <= 0.01 (absolute) at all six, and P(target) strictly falls 10 > 100 > "
                 "1000 for each fine-tune",
         "relative_vs_absolute": "ratio relative to W0; Delta share and P(target) absolute",
         "predicted": "pass: E16's large adjoint passes at every step with P(target) 0.93 -> 0.60 -> 0.10; E21's small "
                      "ones read 1.03 / 1.02 with P(target) 0.71 / 0.69 at step 10",
         "expected": "W0: ratio 1, P(target) 0.004. Delta share: 0 by construction (positive control: the adjoint state "
                     "keeps Delta; the least-squares state's share is 0.25-0.32 for the small ones at step 10, E21)",
         "blind_to": "as P1; P(target) does not see off-target damage", "positive_control": "Delta share 0 (above)",
         "noise_floor": "f for the ratios; P(target) over 73 held-out windows, deterministic"},
        {"id": "P3_outside_both", "review_rows": [3],
         "measure": "outside-both share / its null (split.block_table, pooled_ns) of the SGD 0.01 update vs the Adam 3e-5 "
                    "update, same job, at each of the nine checkpoints",
         "pass": "SGD <= 0.8 x large at every checkpoint (relative; E21's 'lower')",
         "predicted": "pass: 0.034 vs 0.218 at step 10 (E21); SGD's read side lies in the inputs' high-variance span",
         "expected": "W0: undefined (delta-W 0). A random update: 1 (C1). Large: 0.18 at step 1 rising to 0.41 at 1000 "
                     "(E14). Adam 3e-6 reported (0.157 at step 10, E21)",
         "blind_to": "what the outside part does to behaviour or faithfulness; per-matrix spread under pooling",
         "positive_control": "C1 (random draws at their null); G2 (large reproduces E14)",
         "noise_floor": "deterministic given delta-W; one seed. E14 vs E16 differ by 0.01 at a checkpoint"},
        {"id": "P4_rescaling_climb", "review_rows": [5],
         "measure": "climb = (rescaling share / its random null) at step 1000 over the same at step 10, SGD 0.01, pooled_ns",
         "pass": "none: two-sided. Outcome 'climbs' (>= 2), 'weak climb' (1.25-2), 'flat' (0.8-1.25), 'falls' (< 0.8)",
         "relative_vs_absolute": "relative (a ratio of ratios)",
         "predicted": "no direction filed. The large fine-tune's climb is about 5.5 (E14: 2.08 -> 11.43 pooled_ns; "
                      "2.0 -> 9.6 pooled_all)",
         "expected": "W0: undefined. A random update: 1 by definition. Large: as E14 (G2). Adam 3e-6 reported",
         "blind_to": "whether the rescaled part carries the behaviour (E16: the rescaling state keeps none of it)",
         "positive_control": "G2 (the large series reproduces E14's climb)",
         "noise_floor": "the null over two draws; deterministic given delta-W"},
    ],
    "reported_not_scored": ["the large fine-tune's states at every step", "Adam 3e-6's series", "stochastic ratios",
                            "route split d, r", "the four blocks and the algebraic share", "pooled_all",
                            "the least-squares states' Delta share at 100 and 1000", "every fine-tune's held-out evals"],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


# ----------------------------------------------------------------------------- the updates and the states


def finetune(ctx: Any, name: str) -> tuple[dict[int, dict[str, Tensor]], dict[int, dict[str, float]]]:
    from commute.finetune import ArmAConfig, finetune_arm_a

    opt, lr = FINETUNES[name]
    t0 = time.time()
    cfg = ArmAConfig(o_token=ctx.token, lr=lr, steps=TRAIN_STEPS, checkpoints=CKPTS, seed=0, optimizer=opt)
    res = finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log)
    log(f"[{name} fine-tune] {time.time() - t0:.0f} s; step 10 {res.evals[10]}")
    return res.deltas, res.evals


def states(ctx: Any, name: str, step: int, dW: dict[str, Tensor], p_o: float) -> dict[str, Any]:
    """Do-nothing, least-squares split and scaled adjoint split of one update at one step, with Pile ratios."""
    from commute import harness, split

    tag = f"{name}_step{step}"
    ns = harness.nothing_and_split(ctx, tag, dW, p_o, True, log)
    dW_adj, st = split.adjoint_state(dW, ctx.uv_cpu, ctx.sigma)
    return {"nothing": ns["nothing"], "least_squares": {**ns["lift"], "G2": ns["G2"]},
            "adjoint": harness.measure(ctx, f"{tag}_adjoint", dW_adj, st, True, log)}


def series_row(ctx: Any, name: str, step: int, dW: dict[str, Tensor]) -> dict[str, Any]:
    """The four blocks (pooled) and the rescaling table (pooled) of one update at one checkpoint."""
    from commute import split

    t0 = time.time()
    b = split.block_table(dW, ctx.bases, ctx.sigma, ctx.non_saturated)
    r = split.rescaling_table(dW, ctx.uv_cpu, ctx.bases, ctx.sigma, ctx.non_saturated, NULL_DRAWS)
    row = {"blocks": {p: b[p] for p in ("pooled_all", "pooled_ns")}, "rescaling": {p: r[p] for p in ("pooled_all", "pooled_ns")}}
    bp, rp = b["pooled_ns"], r["pooled_ns"]
    log(f"[{name} step {step} series] out_out {bp['out_out']:.4f} (null {bp['null_out_out']:.4f}, ratio {bp['out_out_ratio']:.3f}) "
        f"rescaling {rp['rescaling']:.2e} (null {rp['rescaling_null']:.2e}, ratio {rp['rescaling_ratio']:.2f}) draws' "
        f"out_out / null {rp['draws_out_out_ratio']:.3f} ({time.time() - t0:.0f} s)")
    return row


def gate_g1(out: dict[str, Any], name: str) -> dict[str, Any]:
    """G1: the step-10 update's size reproduces E21's to 1%, and its do-nothing PGD ratio E21's within the floor spread
    (the large one: verdict fail)."""
    from commute.faith_score import verdict

    rec, ref = out["states"][name]["10"], G1_REF[name]
    f = out["W0"]["pgd_floor"]
    rnm, pgd = rec["nothing"]["rel_norm_max"], rec["nothing"]["pile"]["pgd_kl"]
    g = {"rel_norm_max": rnm, "rnm_rel_diff": rnm / ref["rel_norm_max"] - 1, "nothing_pgd": pgd,
         "nothing_verdict": verdict(pgd, f)}
    g["rnm_ok"] = abs(g["rnm_rel_diff"]) <= RNM_TOL
    if ref["nothing_pgd"] is None:
        g["pgd_ok"] = g["nothing_verdict"] == "fail"
    else:
        g["pgd_tol"] = max(f, L40_FLOOR)
        g["pgd_ok"] = abs(pgd / ref["nothing_pgd"] - 1) <= g["pgd_tol"]
    g["pass"] = g["rnm_ok"] and g["pgd_ok"]
    return g


# ----------------------------------------------------------------------------- the run


def stage_all(args: argparse.Namespace) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, PILE_ROWS, PILE_PGD_STEPS,
                       PILE_SEEDS, log)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "W0": faith_score.w0_check(ctx.pile), "evals": {}, "states": {}, "series": {}, "G1": {}}
    log(f"[stage W0] {time.time() - t_start:.0f} s; {out['W0']}")
    dump(OUT, out)
    if not out["W0"]["pass"]:
        out["stopped"] = "W0 is not at its known values: nothing scored"
        dump(OUT, out)
        sys.exit(out["stopped"])

    kept: dict[str, dict[int, dict[str, Tensor]]] = {}
    for name in FINETUNES:
        deltas, evals = finetune(ctx, name)
        out["evals"][name] = {str(s): e for s, e in evals.items()}
        out["states"][name], out["series"][name] = {}, {}
        out["states"][name]["10"] = states(ctx, name, 10, deltas[10], evals[10]["p_o"])
        out["G1"][name] = gate_g1(out, name)
        log(f"[{name} G1] {out['G1'][name]}")
        save(out)
        for step in CKPTS:
            out["series"][name][str(step)] = series_row(ctx, name, step, deltas[step])
        save(out)
        kept[name] = {s: deltas[s] for s in STATE_STEPS if s != 10}
        del deltas
        log(f"[stage {name} step 10 + series] {time.time() - t_start:.0f} s")
    for name in FINETUNES:
        for step in (s for s in STATE_STEPS if s != 10):
            out["states"][name][str(step)] = states(ctx, name, step, kept[name].pop(step), out["evals"][name][str(step)]["p_o"])
            save(out)
            log(f"[stage {name} step {step}] {time.time() - t_start:.0f} s")
    out["seconds"] = time.time() - t_start
    save(out)
    log(f"wrote {OUT} ({out['seconds']:.0f} s); score {json.dumps(out['score'], indent=1, default=str)}")


def save(out: dict[str, Any]) -> None:
    out["rows"] = rows(out)
    out["score"] = score(out)
    dump(OUT, out)


def rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat record per (fine-tune, step, state) measured so far: what plot.py and the report read."""
    from commute.faith_score import verdict

    f = out["W0"]["pgd_floor"]
    return [{"ft": n, "step": int(s), "op": op, "pgd": r["pile"]["pgd_kl"], "verdict": verdict(r["pile"]["pgd_kl"], f),
             "stochastic": r["pile"]["stochastic_kl"], "p_target": r["routes"]["p_cells"]["T1"],
             "delta_share": r["delta_change_frac"], "rel_norm_max": r["rel_norm_max"], "d": r["routes"]["shares"]["d"],
             "r": r["routes"]["shares"]["r"], "gain": r["routes"]["gain"]}
            for n, by in out["states"].items() for s, rec in by.items() for op, r in rec.items()]


# ----------------------------------------------------------------------------- the score (results/predictions.json)


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on whatever is measured so far; a prediction whose inputs are missing is absent."""
    R = {(r["ft"], r["step"], r["op"]): r for r in out["rows"]}
    ser = {n: {int(s): v for s, v in by.items()} for n, by in out["series"].items()}
    g1 = {n: g["pass"] for n, g in out["G1"].items()}
    sc: dict[str, Any] = {"G0_w0": out["W0"]["pass"], "G1": g1}

    def unscored(names: tuple[str, ...], key: str = "pass") -> str | None:
        bad = [n for n in names if n in out["G1"] and not out["G1"][n][key]]
        return f"unscored: G1 failed for {bad}" if bad else None

    keys = [(n, s, op) for n in SMALL for s in (100, 1000) for op in ("nothing", "least_squares")]
    if all(k in R for k in keys):
        v = {f"{n}_{op}@{s}": R[(n, s, op)]["verdict"] for n, s, op in keys}
        rnm = R[("sgd", 1000, "nothing")]["rel_norm_max"]
        sgd1000 = [v[f"sgd_{op}@1000"] for op in ("nothing", "least_squares")]
        reading = ("step story (passes later at any size)" if all(x == "pass" for x in sgd1000) else
                   "size story (SGD past E21's size cut)" if "fail" in sgd1000 and rnm > SGD_SIZE_CUT else
                   "neither (fail below the size cut, or inconclusive)")
        sc["P1_small_later_steps"] = {"pass": unscored(SMALL) or all(x == "pass" for x in v.values()), "verdicts": v,
                                      "sgd_rel_norm_max@1000": rnm, "sgd_reading": reading}
    keys = [(n, s, "adjoint") for n in SMALL for s in STATE_STEPS]
    if all(k in R for k in keys):
        cond = {f"{n}_pass@{s}": R[(n, s, "adjoint")]["verdict"] == "pass" for n, s, _ in keys}
        cond |= {f"{n}_delta@{s}": R[(n, s, "adjoint")]["delta_share"] <= 0.01 for n, s, _ in keys}
        p = {n: [R[(n, s, "adjoint")]["p_target"] for s in STATE_STEPS] for n in SMALL}
        cond |= {f"{n}_p_falls": p[n][0] > p[n][1] > p[n][2] for n in SMALL}
        sc["P2_adjoint"] = {"pass": unscored(SMALL) or all(cond.values()), "conditions": cond, "p_target": p}
    if all(len(ser.get(n, {})) == len(CKPTS) for n in ("sgd", "adam_large")):
        ratio = {n: [ser[n][s]["blocks"]["pooled_ns"]["out_out_ratio"] for s in CKPTS] for n in ser if len(ser[n]) == len(CKPTS)}
        cond = {f"@{s}": a <= 0.8 * b for s, a, b in zip(CKPTS, ratio["sgd"], ratio["adam_large"], strict=True)}
        sc["P3_outside_both"] = {"pass": unscored(("sgd", "adam_large"), "rnm_ok") or all(cond.values()), "conditions": cond, "ratio": ratio}
    if len(ser.get("sgd", {})) == len(CKPTS):
        resc = {n: [ser[n][s]["rescaling"]["pooled_ns"]["rescaling_ratio"] for s in CKPTS] for n in ser if len(ser[n]) == len(CKPTS)}
        climb = {n: v[-1] / v[CKPTS.index(10)] for n, v in resc.items()}
        c = climb["sgd"]
        sc["P4_rescaling_climb"] = {"outcome": "climbs" if c >= 2 else "flat" if 0.8 <= c <= 1.25 else "falls" if c < 0.8 else "weak climb",
                                    "climb": climb, "ratio": resc, "pass": None}
    if "adam_large" in ser and len(ser["adam_large"]) == len(CKPTS):
        a = ser["adam_large"]
        cond = {f"out_out@{s}": abs(a[s]["blocks"]["pooled_ns"]["out_out_ratio"] - e) <= 0.03 for s, e in zip(CKPTS, E14_OUT_OUT, strict=True)}
        cond |= {f"rescaling@{s}": abs(a[s]["rescaling"]["pooled_ns"]["rescaling_ratio"] / e - 1) <= 0.25
                 for s, e in zip(CKPTS, E14_RESCALING, strict=True)}
        sc["G2_large_series_reproduces_E14"] = {"pass": all(cond.values()), "conditions": cond}
    if ser:
        sc["C1_draws_at_null"] = {"pass": all(0.85 <= row["rescaling"]["pooled_ns"]["draws_out_out_ratio"] <= 1.15
                                              for by in ser.values() for row in by.values())}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage == "all":
        stage_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
