"""sgd_finetune: items 5 and 7 of reports/2026-10-21_review_sgd-finding-influence.md (plan:
reports/2026-10-21_plan_sgd-finetune.md). Emoticon, arm A, 10 steps (the Adam fine-tune's scored step in E14 and E16).

Stage `rate`: plain-SGD full fine-tunes of the 24 matrices (finetune.finetune_arm_a, optimizer "sgd") at five rates;
E4's rule picks one (held-out P(o) >= 0.9 and KL elsewhere <= 0.110, the lowest KL elsewhere). Its delta-W is saved
(float16) to results/large/. Then a plain-SGD component-training grid; the rate whose largest relative change per
matrix is nearest the selected fine-tune's (log distance) is the matched rate. -> results/rate.json

Stage `split` (only if `rate` found a passing rate): W0 under the Pile harness for two adversary seeds first
(faith_score.w0_check; stop if it fails). Then the Adam fine-tune (3e-5, E16's recipe, regenerated) and the two
component-training runs (SGD at the matched rate; Adam 1e-5 as the cosine's reference), the four blocks of both
fine-tunes (split.block_table) and the input-weighted cosine of each component-training weight change with the
unscaled adjoint weight change K(delta-W) of the same-optimiser fine-tune. Then the Pile states: SGD adjoint split,
SGD component training, SGD do-nothing and least-squares split, Adam do-nothing and least-squares split, Adam adjoint
split. results/split.json is written after every stage.

Stage `adam_matched` (amendment, results/predictions_amend1.json; needs stage split's results): W0 floor first, the
Adam 3e-5 fine-tune reproducing E16 (G1) next, then Adam fine-tunes (the split stage's recipe) at ADAM_LRS; the rate whose step-10 rel_norm_max is closest to the SGD
fine-tune's is the matched Adam fine-tune (matched within 15%). Its do-nothing and least-squares split, adjoint split
and four blocks, the same functions as split. -> results/adam_matched.json after every stage.

Usage (repo root):  run.py --predictions | run.py rate | run.py split | run.py adam_matched
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
RATE_JSON, SPLIT_JSON = RESULTS / "rate.json", RESULTS / "split.json"
AMEND1, ADAM_MATCHED_JSON = RESULTS / "predictions_amend1.json", RESULTS / "adam_matched.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"

STEPS = 10  # E14 / E16's first scored step of the Adam fine-tune; E4's off-target limit is read at it
SGD_LRS = (0.001, 0.003, 0.01, 0.03, 0.1)  # fine-tune rates (E2's SGD fine-tunes were 0.1 to 1)
CT_LRS = (0.003, 0.01, 0.03, 0.1, 0.3)  # SGD component-training rates searched for the matched size of change
ADAM_FT_LR, ADAM_CT_LR = 3e-5, 1e-5  # E16's fine-tune and component training
P_MIN, KL_OTHER_MAX = 0.9, 0.110  # E4's rule
TOL = 1e-6  # split.bases tolerance, as E14 and E16
PILE_ROWS, PILE_PGD_STEPS, PILE_SEEDS = 256, 20, (0, 1)  # seeds 0 and 1 gave the L40's whole 0.209 floor (E19 job 28108)
ADAM_REF = {"p_o": 0.9993, "kl_other": 0.0554}  # E16 split_comparison.json, ft step 10
REPRO_TOL = 0.01  # absolute
# Amendment: Adam's step-10 size of change is about proportional to its rate (3e-5 -> 0.0112); the SGD fine-tune's
# 0.00238 is 0.21x of it, so 0.1x to 0.4x of ADAM_FT_LR brackets it with room on both sides.
ADAM_LRS = (3e-6, 4.5e-6, 6e-6, 8e-6, 1.2e-5)
MATCH_TOL = 0.15  # |rel_norm_max / SGD's - 1|

PREDICTIONS: dict[str, Any] = {
    "about": "sgd_finetune: items 5 and 7 of reports/2026-10-21_review_sgd-finding-influence.md. Plan "
             "reports/2026-10-21_plan_sgd-finetune.md. Filed before any run.",
    "setup": {
        "task": "emoticon arm A (CE on o at the trigger + KL elsewhere), batch 32, clip 1, seed 0, 10 steps, 24 matrices",
        "sgd_finetune": f"plain SGD (no momentum) at {SGD_LRS}; E4's rule: held-out P(o) >= {P_MIN} and KL elsewhere <= "
                        f"{KL_OTHER_MAX} at step 10, the lowest KL elsewhere among those",
        "adam_finetune": f"Adam {ADAM_FT_LR} (E14, E16), regenerated in the split job",
        "sgd_component_training": f"plain SGD on the alive (U, V), Gamma and Delta frozen, 10 steps, rate from {CT_LRS} "
                                  "whose largest ||delta-W|| / ||W0|| over matrices at step 10 is nearest the selected SGD "
                                  "fine-tune's (|log ratio|)",
        "harness": f"harness.measure: Pile rows 0..{PILE_ROWS - 1}, PGD {PILE_PGD_STEPS} steps, ratios to W0 under adversary "
                   f"seed 0 in the same job; floor f = max/min - 1 of W0's PGD KL over seeds {PILE_SEEDS}; "
                   "faith_score.verdict: pass <= 1.5/(1+f), fail >= 1.5(1+f), inconclusive between (about 1.24 to 1.81 "
                   "on the L40's f 0.21). Stochastic uses its own floor (about 0.002)",
    },
    "gates": {
        "R0_rate": "stage split runs only if stage rate finds a passing SGD rate; if none, the null is reported with the "
                   "grid's range (0.001 to 0.1 at step 10) and the experiment stops",
        "G0_w0": "runs first: faith_score.w0_check (seed-0 PGD within 10% of 0.69, CI-masked within 2% of 0.348, PGD floor "
                 "<= 0.25; 0.209 on the L40). Fails: stop, nothing scored",
        "G1_adam_reproduces": f"Adam fine-tune's held-out P(o) and KL elsewhere at step 10 within {REPRO_TOL} of "
                              f"{ADAM_REF['p_o']} and {ADAM_REF['kl_other']} (E16). Fails: stop before the Pile states",
        "G2_sgd_state": "the saved SGD fine-tune's do-nothing target P(o) at the trigger within 0.01 of stage rate's "
                        "held-out P(o) (float16 storage). Fails: SGD rows reported, not scored",
    },
    "predictions": [
        {"id": "R1_sgd_rate", "stage": "rate", "measure": "E4's rule over the five rates at step 10",
         "pass": "some rate passes", "predicted": "pass, at 0.01 or 0.03",
         "expected": "W0: P(o) 0.004, KL 0. 0.001 too small (P(o) < 0.9; component training at 0.001 read 0.064 at step "
                     "10). 0.1 too large (E2: KL elsewhere 0.60 at step 10). Adam reference: P(o) 0.9993, KL 0.0554",
         "relative_vs_absolute": "absolute", "blind_to": "faithfulness (the split stage reads it)",
         "noise_floor": "one seed; none measured"},
        {"id": "P1_outside_share", "item": 7, "stage": "split",
         "measure": "outside-both block over its null (split.block_table pooled_all out_out_ratio), SGD fine-tune / Adam "
                    "fine-tune, same job",
         "pass": "SGD / Adam <= 0.8 ('lower'); >= 1.25 is 'higher'; between is 'alike' (TRIED's E2 line)",
         "predicted": "lower: an SGD step's read side lies in the inputs' span weighted by their variance, what V reads; "
                      "Adam's per-coordinate step spreads onto low-variance input directions. Open",
         "expected": "W0: undefined (delta-W 0). Null: 1 by definition. Do-nothing: same as the update (a property of "
                     "delta-W). Adam reference: 0.0336 / 0.155 = 0.217 (E16 step 10)",
         "relative_vs_absolute": "relative (SGD over Adam)",
         "blind_to": "what the outside part does to behaviour or faithfulness; per-matrix differences under the pooling",
         "noise_floor": "deterministic given delta-W; no seed floor. Adam's own ratio moves 0.22 -> 0.33 between steps 10 "
                        "and 100 (E16), the scale of a recipe change"},
        {"id": "P2_least_squares_sgd", "item": 7, "stage": "split",
         "measure": "SGD fine-tune's least-squares split, Pile PGD ratio", "pass": "verdict 'fail' (>= 1.5 (1 + f))",
         "predicted": "pass (the split is fragile under SGD too). Open: the SGD update is smaller than Adam's",
         "expected": "W0: 1. Do-nothing: below Adam's in-job value (Adam 2.2 to 2.5). Adam reference: 5.2 to 7.0 "
                     "(E8 L40, E14, E16)",
         "relative_vs_absolute": "ratio to the same-job W0", "blind_to": "the trigger (about 3 positions in 131k)",
         "noise_floor": "the W0 floor f, through the verdict's band"},
        {"id": "P3_adjoint_sgd", "item": 7, "stage": "split",
         "measure": "SGD fine-tune's scaled adjoint split, Pile PGD and stochastic ratios",
         "pass": "both verdicts 'pass'", "predicted": "pass",
         "expected": "W0: 1. Adam reference: PGD 1.06, stochastic 0.99 (E14, E16)",
         "relative_vs_absolute": "ratio to the same-job W0", "blind_to": "as P2; and the behaviour (P(o) reported)",
         "noise_floor": "as P2; stochastic its own floor"},
        {"id": "P4_ct_sgd", "item": 5, "stage": "split",
         "measure": "SGD component training at the matched rate: Pile PGD and stochastic ratios, target P(o) at the trigger",
         "pass": "both verdicts 'pass' and P(o) >= 0.9", "predicted": "pass",
         "expected": "W0: ratios 1, P(o) 0.004. Adam reference (component training 1e-5, step 10, E16): PGD 1.03, "
                     "stochastic 1.00, P(o) 0.9998",
         "relative_vs_absolute": "ratios relative to W0; P(o) absolute", "blind_to": "as P2",
         "noise_floor": "as P2"},
        {"id": "P5_ct_is_adjoint", "item": 5, "stage": "split",
         "measure": "input-weighted cosine, pooled over the 24 matrices, of the component-training weight change with "
                    "K(delta-W) = U U^T delta-W + delta-W V V^T of the same-optimiser fine-tune (unscaled adjoint split)",
         "pass": "SGD cosine >= 0.9 and above Adam's (Adam component training 1e-5 against K of the Adam fine-tune)",
         "predicted": "pass: one plain-SGD step of component training is the adjoint split of the fine-tune's step at "
                      "the same point; path, clipping and the rate differ only at second order over 10 small steps. "
                      "Adam's per-coordinate scaling acts on different coordinates in the two parameterisations",
         "expected": "W0: undefined. 1 at step 1 with equal rates and no clipping (first order). Adam: not recorded",
         "relative_vs_absolute": "absolute", "blind_to": "scale (a cosine); behaviour; per-matrix spread under pooling",
         "noise_floor": "deterministic; one seed"},
    ],
    "reported_not_scored": ["do-nothing Pile ratios of both fine-tunes", "the four blocks of both fine-tunes",
                            "P(o), gain and route shares d, r of every state", "the adjoint states' reproduced share R",
                            "CI-masked and unmasked ratios", "the component-training rate grid"],
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    raise TypeError(type(o))


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default))


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def load(device: str, pile: bool) -> Any:
    from commute import harness

    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    return harness.load(device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, PILE_ROWS if pile else 0,
                        PILE_PGD_STEPS, PILE_SEEDS, log)


def finetune(ctx: Any, lr: float, optimizer: str) -> tuple[dict[str, Tensor], dict[str, float]]:
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=ctx.token, lr=lr, steps=STEPS, checkpoints=(STEPS,), optimizer=optimizer)
    res = finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log)
    return res.deltas[STEPS], res.evals[STEPS]


def component_training(ctx: Any, lr: float, optimizer: str) -> tuple[dict[str, Any], Any]:
    """Step-10 held-out eval and state of component training (all alive pieces free)."""
    from commute import component_training as ct
    from commute.finetune import ArmAConfig

    cfg = ct.TrainConfig(lr=lr, steps=STEPS, save_steps=(STEPS,), optimizer=optimizer)
    curve, states = ct.train(ctx.model, ctx.uv, ctx.train.tokens, ctx.held, ArmAConfig(o_token=ctx.token), cfg,
                             ctx.device, ctx.W0, log=log)
    c = curve[-1]
    return {"p_o": c["heldout_p_o"], "kl_other": c["heldout_kl_other"], "rel_norm_max": c["rel_norm_max"]}, states[STEPS]


def select_rate(evals: dict[str, dict[str, float]]) -> tuple[str, bool]:
    """E4's rule: among rates that learn within the limit, the lowest KL elsewhere; else the highest P(o), not passed."""
    ok = [lr for lr, e in evals.items() if e["p_o"] >= P_MIN and e["kl_other"] <= KL_OTHER_MAX]
    if ok:
        return min(ok, key=lambda lr: evals[lr]["kl_other"]), True
    return max(evals, key=lambda lr: evals[lr]["p_o"]), False


def state_path(lr: str) -> Path:
    return LARGE / f"sgd_ft_lr{lr}_step{STEPS}.pt"


def stage_rate(args: argparse.Namespace) -> None:
    t0 = time.time()
    ctx = load(args.device, pile=False)
    out: dict[str, Any] = {"steps": STEPS, "finetune": {}, "component_training": {}}
    deltas = {}
    for lr in SGD_LRS:
        key = f"{lr:g}"
        deltas[key], out["finetune"][key] = finetune(ctx, lr, "sgd")
        e = out["finetune"][key]
        log(f"[sgd ft lr {key}] P(o) {e['p_o']:.4f} KL elsewhere {e['kl_other']:.4f} rel_norm_max {e['rel_norm_max']:.5f}")
        dump(RATE_JSON, out)
    lr, passed = select_rate(out["finetune"])
    out.update(selected_lr=lr, passed=passed, state=str(state_path(lr).relative_to(HERE)))
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() for k, v in deltas[lr].items()}, state_path(lr))
    log(f"[selected] SGD fine-tune lr {lr} passed {passed}; saved {state_path(lr)}")
    dump(RATE_JSON, out)
    target = out["finetune"][lr]["rel_norm_max"]
    for ct_lr in CT_LRS:
        out["component_training"][f"{ct_lr:g}"] = component_training(ctx, ct_lr, "sgd")[0]
        dump(RATE_JSON, out)
    out["ct_matched_lr"] = min(out["component_training"],
                               key=lambda k: abs(math.log(out["component_training"][k]["rel_norm_max"] / target)))
    out["seconds"] = time.time() - t0
    dump(RATE_JSON, out)
    log(f"[matched] component-training lr {out['ct_matched_lr']} (fine-tune rel_norm_max {target:.5f}); "
        f"wrote {RATE_JSON} ({out['seconds']:.0f} s)")


def weighted_cosine(A: dict[str, Tensor], B: dict[str, Tensor], sigma: dict[str, Tensor]) -> float:
    """sum_k tr(A_k S_k B_k^T) / sqrt(sum_k tr(A S A^T) sum_k tr(B S B^T)), S_k = E[a a^T] of matrix k's input."""
    from commute import split

    ab = sum(float(((A[k].double() @ sigma[k]) * B[k].double()).sum()) for k in A)
    aa = sum(split.weighted_energy(A[k].double(), sigma[k]) for k in A)
    bb = sum(split.weighted_energy(B[k].double(), sigma[k]) for k in B)
    return ab / math.sqrt(aa * bb)


def adjoint_change(ctx: Any, dW: dict[str, Tensor]) -> dict[str, Tensor]:
    """K(delta-W) per matrix, unscaled: the weight change of the adjoint split."""
    from commute import split

    return {k: split.dsigma(*split.adjoint_split(M, *(x.double() for x in ctx.uv_cpu[k])),
                            *(x.double() for x in ctx.uv_cpu[k])) for k, M in dW.items()}


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(path, out)
    sys.exit(why)


def w0_first(ctx: Any, path: Path, out: dict[str, Any], t0: float) -> None:
    """G0: W0 under the Pile harness at its known values, else stop with nothing scored."""
    from commute import faith_score

    out["W0"] = faith_score.w0_check(ctx.pile)
    log(f"[stage W0] {time.time() - t0:.0f} s; {out['W0']}")
    dump(path, out)
    if not out["W0"]["pass"]:
        stop(path, out, "W0 is not at its known values: nothing scored")


def adjoint(ctx: Any, name: str, dW: dict[str, Tensor]) -> dict[str, Any]:
    """The scaled adjoint split of dW, measured with the Pile ratios."""
    from commute import harness, split

    dW_adj, st = split.adjoint_state(dW, ctx.uv_cpu, ctx.sigma)
    return harness.measure(ctx, f"{name}_adjoint", dW_adj, st, True, log)


def nothing_and_ls(ctx: Any, name: str, dW: dict[str, Tensor], p_o: float) -> dict[str, Any]:
    """dW measured do-nothing and least-squares split, with the Pile ratios."""
    from commute import harness

    ns = harness.nothing_and_split(ctx, name, dW, p_o, True, log)
    return {"nothing": ns["nothing"], "least_squares": {**ns["lift"], "G2": ns["G2"]}}


def blocks(ctx: Any, dW: dict[str, Tensor]) -> dict[str, float]:
    from commute import split

    return split.block_table(dW, ctx.bases, ctx.sigma, ctx.non_saturated)["pooled_all"]


def stage_split(args: argparse.Namespace) -> None:
    from commute import harness

    rate = json.loads(RATE_JSON.read_text())
    if not rate.get("passed"):
        sys.exit(f"stage rate found no passing SGD rate in {SGD_LRS} at step {STEPS}: report the null and stop")
    t0 = time.time()
    ctx = load(args.device, pile=True)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "rate": {k: rate[k] for k in ("selected_lr", "ct_matched_lr")}}
    w0_first(ctx, SPLIT_JSON, out, t0)

    adam_dW, adam_ev = finetune(ctx, ADAM_FT_LR, "adam")
    g1 = {"p_o": adam_ev["p_o"], "kl_other": adam_ev["kl_other"]}
    g1["pass"] = all(abs(g1[k] - ADAM_REF[k]) <= REPRO_TOL for k in ADAM_REF)
    sgd_dW = {k: v.float() for k, v in torch.load(HERE / rate["state"], weights_only=False).items()}
    ft = {"sgd": (sgd_dW, rate["finetune"][rate["selected_lr"]]), "adam": (adam_dW, adam_ev)}
    ct = {"sgd": component_training(ctx, float(rate["ct_matched_lr"]), "sgd"), "adam": component_training(ctx, ADAM_CT_LR, "adam")}
    out["G1_adam_reproduces"] = g1
    out["finetune_eval"] = {n: ev for n, (_, ev) in ft.items()}
    out["ct_eval"] = {n: ev for n, (ev, _) in ct.items()}
    out["blocks"] = {n: blocks(ctx, dW) for n, (dW, _) in ft.items()}
    out["cosine"] = {n: weighted_cosine({k: v[0] for k, v in ct[n][1].items()}, adjoint_change(ctx, ft[n][0]), ctx.sigma)
                     for n in ft}
    log(f"[stage cheap] {time.time() - t0:.0f} s; G1 {g1}; blocks {out['blocks']}; cosine {out['cosine']}")
    dump(SPLIT_JSON, out)
    if not g1["pass"]:
        stop(SPLIT_JSON, out, "the Adam fine-tune does not reproduce E16's step 10: Pile states not run")

    out["states"] = {"sgd": {}, "adam": {}}
    states = out["states"]

    def ct_state() -> None:
        st = ct["sgd"][1]
        out["states"]["sgd"]["component_training"] = harness.measure(ctx, "sgd_ct", {k: v[0] for k, v in st.items()}, st, True, log)

    # The paired rows first (adjoint split of each fine-tune, then item 5's component training), so a timeout loses
    # only the least-squares rows, which have recorded Adam references (Codex review, 2026-10-21).
    for name, fn in (("sgd adjoint", lambda: states["sgd"].update(adjoint=adjoint(ctx, "sgd", ft["sgd"][0]))),
                     ("adam adjoint", lambda: states["adam"].update(adjoint=adjoint(ctx, "adam", ft["adam"][0]))),
                     ("sgd component training", ct_state),
                     ("sgd nothing + least squares", lambda: states["sgd"].update(nothing_and_ls(ctx, "sgd", ft["sgd"][0], ft["sgd"][1]["p_o"]))),
                     ("adam nothing + least squares", lambda: states["adam"].update(nothing_and_ls(ctx, "adam", ft["adam"][0], ft["adam"][1]["p_o"])))):
        fn()
        out["score"] = score(out)
        dump(SPLIT_JSON, out)
        log(f"[stage {name}] {time.time() - t0:.0f} s")
    out["seconds"] = time.time() - t0
    dump(SPLIT_JSON, out)
    log(f"wrote {SPLIT_JSON} ({out['seconds']:.0f} s); score {json.dumps(out['score'], indent=1)}")


def pile_row(rec: dict[str, Any], w0: dict[str, Any]) -> dict[str, Any]:
    """A state's Pile ratios with verdicts, each against its own W0 floor (PGD: pgd_floor; stochastic: stochastic_floor)."""
    from commute.faith_score import verdict

    p = rec["pile"]
    return {"pgd": p["pgd_kl"], "stochastic": p["stochastic_kl"], "pgd_verdict": verdict(p["pgd_kl"], w0["pgd_floor"]),
            "stochastic_verdict": verdict(p["stochastic_kl"], w0["stochastic_floor"]),
            "p_o": rec["routes"]["p_cells"]["T1"], "gain": rec["routes"]["gain"]}


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules (results/predictions.json) on whatever states are measured so far."""
    st, sgd = out["states"], out["states"]["sgd"]
    sc: dict[str, Any] = {"G0_w0": out["W0"]["pass"], "G1_adam_reproduces": out["G1_adam_reproduces"]["pass"]}
    sc["table"] = {f"{n}_{op}": pile_row(rec, out["W0"]) for n in st for op, rec in st[n].items()}
    rel = out["blocks"]["sgd"]["out_out_ratio"] / out["blocks"]["adam"]["out_out_ratio"]
    sc["P1_outside_share"] = {"sgd_over_adam": rel, "outcome": "lower" if rel <= 0.8 else "higher" if rel >= 1.25 else "alike",
                              "pass": rel <= 0.8}
    if "nothing" in sgd:
        sc["G2_sgd_state"] = abs(sgd["nothing"]["routes"]["p_cells"]["T1"] - out["finetune_eval"]["sgd"]["p_o"]) <= 0.01
        sc["P2_least_squares_sgd"] = {"pass": sc["table"]["sgd_least_squares"]["pgd_verdict"] == "fail"}
    if "adjoint" in sgd:
        t = sc["table"]["sgd_adjoint"]
        sc["P3_adjoint_sgd"] = {"pass": t["pgd_verdict"] == "pass" and t["stochastic_verdict"] == "pass"}
    if "component_training" in sgd:
        t = sc["table"]["sgd_component_training"]
        sc["P4_ct_sgd"] = {"pass": t["pgd_verdict"] == "pass" and t["stochastic_verdict"] == "pass" and t["p_o"] >= 0.9}
    c = out["cosine"]
    sc["P5_ct_is_adjoint"] = {"sgd": c["sgd"], "adam": c["adam"], "pass": c["sgd"] >= 0.9 and c["sgd"] > c["adam"]}
    return sc


def adam_grid(ctx: Any, out: dict[str, Any], target: float) -> dict[str, Tensor]:
    """Adam fine-tunes at ADAM_LRS (dumped after each); returns the delta-W whose step-10 rel_norm_max is closest to
    `target` (|log ratio|), recorded in out["match"]. The control first: the split stage's Adam fine-tune reproduces E16."""
    e = finetune(ctx, ADAM_FT_LR, "adam")[1]
    out["G1_adam_reproduces"] = {"p_o": e["p_o"], "kl_other": e["kl_other"], "rel_norm_max": e["rel_norm_max"],
                                 "pass": all(abs(e[k] - ADAM_REF[k]) <= REPRO_TOL for k in ADAM_REF)}
    log(f"[control adam ft lr {ADAM_FT_LR:g}] {out['G1_adam_reproduces']}")
    if not out["G1_adam_reproduces"]["pass"]:
        stop(ADAM_MATCHED_JSON, out, "the Adam fine-tune does not reproduce E16's step 10: nothing scored")
    deltas, out["grid"] = {}, {}
    for lr in ADAM_LRS:
        key = f"{lr:g}"
        deltas[key], out["grid"][key] = finetune(ctx, lr, "adam")
        e = out["grid"][key]
        log(f"[adam ft lr {key}] P(o) {e['p_o']:.4f} KL elsewhere {e['kl_other']:.4f} rel_norm_max {e['rel_norm_max']:.5f}")
        dump(ADAM_MATCHED_JSON, out)
    key = min(out["grid"], key=lambda k: abs(math.log(out["grid"][k]["rel_norm_max"] / target)))
    e = out["grid"][key]
    rel = e["rel_norm_max"] / target - 1
    out["match"] = {"lr": key, "target": target, "rel_diff": rel, "matched": abs(rel) <= MATCH_TOL,
                    "learned": e["p_o"] >= P_MIN, "p_o": e["p_o"], "kl_other": e["kl_other"]}
    log(f"[match] {out['match']}")
    dump(ADAM_MATCHED_JSON, out)
    return deltas[key]


def stage_adam_matched(args: argparse.Namespace) -> None:
    if not AMEND1.is_file():
        sys.exit(f"file {AMEND1} by hand first (the amendment's predictions)")
    ref = json.loads(SPLIT_JSON.read_text())
    t0 = time.time()
    ctx = load(args.device, pile=True)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "reference": {"sgd_rel_norm_max": ref["finetune_eval"]["sgd"]["rel_norm_max"],
                                         "out_out_ratio": {n: b["out_out_ratio"] for n, b in ref["blocks"].items()}}}
    w0_first(ctx, ADAM_MATCHED_JSON, out, t0)
    dW = adam_grid(ctx, out, out["reference"]["sgd_rel_norm_max"])
    out["states"] = {}
    name = "adam_matched"
    for stage, fn in (("nothing + least squares", lambda: out["states"].update(nothing_and_ls(ctx, name, dW, out["match"]["p_o"]))),
                      ("adjoint", lambda: out["states"].update(adjoint=adjoint(ctx, name, dW))),
                      ("blocks", lambda: out.update(blocks=blocks(ctx, dW)))):
        fn()
        out["score"] = score_amend1(out)
        dump(ADAM_MATCHED_JSON, out)
        log(f"[stage {stage}] {time.time() - t0:.0f} s")
    out["seconds"] = time.time() - t0
    dump(ADAM_MATCHED_JSON, out)
    log(f"wrote {ADAM_MATCHED_JSON} ({out['seconds']:.0f} s); score {json.dumps(out['score'], indent=1)}")


def score_amend1(out: dict[str, Any]) -> dict[str, Any]:
    """The rules of results/predictions_amend1.json on whatever is measured so far."""
    m = out["match"]
    flags = [s for s, bad in (("unmatched", not m["matched"]), ("did not learn", not m["learned"])) if bad]
    sc: dict[str, Any] = {"G0_w0": out["W0"]["pass"], "G1_adam_reproduces": True, "flags": flags,
                          "table": {op: pile_row(rec, out["W0"]) for op, rec in out["states"].items()}}
    reading = {"fail": "the optimiser carries the difference", "pass": "the size of change carries it"}
    for pid, op in (("A1_nothing", "nothing"), ("A2_least_squares", "least_squares")):
        if op in sc["table"]:
            v = sc["table"][op]["pgd_verdict"]
            sc[pid] = {"verdict": v, "reading": reading.get(v, "inconclusive"), "flags": flags}
    if "blocks" in out:
        r, refs = out["blocks"]["out_out_ratio"], out["reference"]["out_out_ratio"]
        vs_sgd, vs_adam = r / refs["sgd"], r / refs["adam"]
        sc["A3_outside_share"] = {"ratio": r, "over_sgd": vs_sgd, "over_adam_full": vs_adam, "flags": flags,
                                  "outcome": "like SGD" if vs_sgd <= 1.25 else "like Adam-full" if vs_adam >= 0.8 else "between"}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["rate", "split", "adam_matched"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage is not None:
        if not PRED.is_file():
            sys.exit("file the predictions first: run.py --predictions")
        {"rate": stage_rate, "split": stage_split, "adam_matched": stage_adam_matched}[args.stage](args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
