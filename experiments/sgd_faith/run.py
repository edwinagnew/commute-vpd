"""sgd_faith: is the saved plain-SGD emoticon component-training state faithful? Item 1 of
reports/reviews/2026-10-21_review_sgd-finding-influence.md; plan reports/plans/2026-10-21_plan_sgd-faithfulness.md.

The SGD state is sgd_grid's selected one (plain SGD, lr 0.01, 30 steps, saved as {matrix: (delta-W, U_A', V_A')}).
The reference is Adam component training at lr 1e-5 for the same 30 steps (E4's adam_all, the ladder's emoticon
step 30), regenerated here because no Adam state of that recipe was saved. Both go through `harness.measure`, which
reads (a) round 2's Pile harness as ratios to W0 (stochastic, PGD, CI-masked, unmasked) and (b) the route split at
the trigger of the held-out windows. E4's trigger-local masked checks are read from the route split's cells: the
CI-masked model's P(o) over the target's P(o), under the state's own gates (cell L11) and under W0's gates (L10),
the quantities E4's amendment 1 reported (0.635 and 0.0064 for Adam, reproduced by the ladder's route split).

Order: W0 first (Pile under 3 adversary seeds; stop if W0 is not at its known values), then Adam (stop if it does
not reproduce its recorded numbers), then the SGD state.

Usage (repo root, project venv):
  run.py --predictions             file results/predictions.json (refuses to overwrite)
  run.py --sgd-state PATH          GPU -> results/sgd_faith.json (dumped after every stage)
  run.py --sgd-state PATH --smoke  tiny sizes, checks not enforced -> results/sgd_faith_smoke.json (CPU works)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"

TOL = 1e-6  # split.bases tolerance, as E16 and E19 (harness.load computes the bases; unused here)
ADAM_LR = 1e-5
LINE = 1.5  # the adversarial pass line (round 2), applied to stochastic, PGD and CI-masked as in E4
# W0 under the Pile harness, seed 0, mean of the two blocks: 0.692 / 0.348 (E11 job 25863, L40), 0.691 / 0.348 (E4, 4090)
W0_PGD, W0_PGD_TOL = 0.69, 0.10  # relative
W0_CI, W0_CI_TOL = 0.348, 0.02  # relative; deterministic under Gamma frozen
FLOOR_STOP = 0.25  # W0 PGD floor max/min - 1 over seeds; 0.209 on the L40 (E8, E11, E19)
# Adam lr 1e-5 at step 30 (E4 concentration.json adam_all; task_ladder emoticon step 30; E4 amend1)
ADAM_REF = {"p_o": 0.99614, "kl_other": 0.00910, "state_gated": 0.637, "w0_gated": 0.0064, "d": 0.031, "r": 0.508}
REPRO_TOL = 0.01  # absolute, on held-out P(o) and KL elsewhere
TRIG_TOL, D_TOL, R_TOL = 0.05, 0.02, 0.05  # absolute, on Adam's trigger ratio and route shares (bf16 vs fp32 reruns)
# sgd_grid lr 0.01 at step 30 (sgd_grid.json): held-out P(o), largest ||delta-W|| / ||W0|| over matrices
SGD_REF = {"p_o": 0.99094, "rel_norm_max": 0.0026365}


@dataclass(frozen=True)
class Sizes:
    adam_steps: int  # Adam's step count; the scored run matches the SGD state's 30
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]  # adversary seeds at W0 (the floor); seed 0 is every ratio's reference
    enforce: bool  # stop on a failed W0 or Adam check (smoke: report only)
    suffix: str

    def __post_init__(self) -> None:
        assert self.pile_rows > 0 and 0 in self.pile_seeds and len(self.pile_seeds) >= 2, "the floor needs seed 0 and one more"
        assert not self.enforce or self.adam_steps == 30, "the scored Adam reference matches the SGD state's 30 steps"


FULL = Sizes(adam_steps=30, pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2), enforce=True, suffix="")
SMOKE = Sizes(adam_steps=2, pile_rows=8, pile_pgd_steps=2, pile_seeds=(0, 1), enforce=False, suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "about": "sgd_faith (item 1 of reports/reviews/2026-10-21_review_sgd-finding-influence.md): faithfulness of sgd_grid's saved "
             "plain-SGD emoticon state (lr 0.01, 30 steps) on the Pile harness, E4's trigger-local masked checks and the "
             "route split, with Adam lr 1e-5 at 30 steps as the reference scored by the same code in the same job. "
             "Filed before any run; plan reports/plans/2026-10-21_plan_sgd-faithfulness.md",
    "measurements": {
        "pile": "round 2's harness (harness.measure): Pile val rows 0..255 in two 128-row blocks, one PGD adversary per "
                "block (20 sign steps of 0.1, seed 0), stochastic (2 draws), CI-masked and unmasked KL; each a ratio to "
                "W0 under seed 0 in this job. Expected at W0: 1 by definition (absolute: PGD 0.69, CI-masked 0.348, "
                "stochastic 0.248, unmasked 0.012). Blind to the trigger (about 3 trigger positions in 131k)",
        "trigger_local": "E4 amendment 1's measure, read from the route split's cells on the 73 held-out windows: mean "
                         "P(o) at the trigger of the CI-masked model (state's components, Delta out) over the target "
                         "model's, with Gamma's gates from the state's activations (state_gated, cell L11) and from W0's "
                         "(w0_gated, cell L10). At W0: 0.39 (0.0044 / 0.0115, no behaviour to carry). Blind to positions "
                         "outside the windows and to damage on Pile",
        "route_split": "local_faith.route_split: shares of the target's log-odds gain at the trigger, d (changed "
                       "components at W0's gates), r (re-gating), i, e_gate, e_delta; they sum to 1. Undefined at W0 (no "
                       "gain); not scored below a gain of 1. Blind to which pieces moved and to anything off the trigger",
        "noise_floors": "PGD: W0 max/min - 1 over adversary seeds 0, 1, 2 in this job (0.209 on the L40 before), so a PGD "
                        "ratio inside 1.5 / (1 + f) .. 1.5 (1 + f) (about 1.24 to 1.81) is reported as not decisive on "
                        "this card. Stochastic: same, about 0.002. CI-masked, unmasked, trigger ratios and shares: "
                        "deterministic under Gamma frozen; bf16 vs fp32 reruns move them by about 0.01 (ladder vs E16 "
                        "shares at step 10: r 0.528 vs 0.529, d 0.025 vs 0.026)",
        "relative_vs_absolute": "Pile numbers are ratios to W0 (relative); trigger ratios and shares are absolute",
    },
    "gates": {
        "G0_w0": "runs first: W0's seed-0 PGD KL within 10% of 0.69, CI-masked KL within 2% of 0.348, PGD floor <= 0.25. "
                 "Fails: the run stops, nothing scored",
        "G1_adam_reproduces": "the regenerated Adam state's held-out P(o) and KL elsewhere at step 30 within 0.01 of "
                              "0.9961 and 0.0091 (E4 adam_all, ladder emoticon). Fails: stop before SGD",
        "G2_sgd_is_saved_state": "the loaded SGD state's target P(o) at the trigger within 0.01 of 0.9909 and its largest "
                                 "||delta-W|| / ||W0|| within 1% of 0.002637 (sgd_grid.json, lr 0.01, step 30). Fails: "
                                 "SGD unscored",
    },
    "control": {"id": "C1_adam_control", "runs": "after G0, before the SGD state; fails: stop before SGD",
                "rule": "Adam's stochastic, PGD and CI-masked ratios <= 1.5 (E4 on a 4090: 1.003, 1.067, 1.003); "
                        "state_gated within 0.05 of 0.637 (E4 amendment 1); w0_gated < 0.5 (0.0064); d within 0.02 of "
                        "0.031 and r within 0.05 of 0.508 (ladder emoticon, step 30)",
                "predicted": "pass"},
    "predictions": [
        {"id": "P1_sgd_pile", "measure": "SGD state's stochastic, PGD and CI-masked Pile ratios",
         "pass": "all three <= 1.5", "predicted": "pass",
         "confidence": "genuinely open: E4's overshooting SGD state (lr 0.3) read PGD 24.9. This state's largest relative "
                       "change is 0.0026 against Adam's 0.0066 and its KL elsewhere 0.0065 against 0.0091, which points to "
                       "a pass, but SGD faithfulness has never been measured at a learning rate that learns cleanly"},
        {"id": "P2_sgd_state_gated", "measure": "SGD state's state_gated trigger ratio",
         "pass": ">= 0.8 (E4's 'inside' line; Adam reads 0.637, E4's lr-0.3 SGD state 0.999)", "predicted": "pass",
         "why": "plain SGD's step follows the gradient, which is largest on the pieces the trigger's activations read, the "
                "same pieces Gamma switches on there; Adam's per-coordinate scaling spreads the change onto pieces Gamma "
                "leaves off (Adam's e_gate 0.445). Open"},
        {"id": "P3_sgd_w0_gated", "measure": "SGD state's w0_gated trigger ratio",
         "pass": "< 0.5", "predicted": "pass",
         "why": "the behaviour still needs re-gating, as for every state so far (E4's lr-0.3 SGD state 0.188)"},
        {"id": "P4_sgd_direct_share", "measure": "SGD state's route-split share d",
         "pass": "> 0.1 (the research log's Result 3 row says the moved pieces carry a tenth or less)",
         "predicted": "pass", "scorable_if": "SGD gain >= 1",
         "why": "reverting SGD's top 10 pieces removes 0.16 of its gain against 0.016 for Adam (sgd_grid). Open"},
    ],
    "reported_not_scored": ["r, i, e_gate, e_delta for both states", "unmasked Pile ratio",
                            "Delta change share (0 for component training)", "PGD ratio against 1 + 3 f"],
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


# ----------------------------------------------------------------------------- the three stages


def check_w0(ctx: Any) -> dict[str, Any]:
    """G0: W0 at its known Pile values, and its PGD floor small enough to read the 1.5 line."""
    from commute import harness

    w0, f = ctx.pile[0], harness.pgd_floor(ctx)
    assert f is not None
    rec = {"pile": ctx.pile, "pgd_floor": f, "stochastic_floor": max(p["stochastic_kl"] for p in ctx.pile.values())
           / min(p["stochastic_kl"] for p in ctx.pile.values()) - 1,
           "pgd_rel": w0["pgd_kl"] / W0_PGD - 1, "ci_rel": w0["ci_masked_kl"] / W0_CI - 1}
    rec["pass"] = abs(rec["pgd_rel"]) <= W0_PGD_TOL and abs(rec["ci_rel"]) <= W0_CI_TOL and f <= FLOOR_STOP
    log(f"[G0 W0] PGD {w0['pgd_kl']:.4f} (known {W0_PGD}) CI-masked {w0['ci_masked_kl']:.4f} (known {W0_CI}) "
        f"PGD floor {f:.4f} (stop above {FLOOR_STOP}) -> {'PASS' if rec['pass'] else 'FAIL'}")
    return rec


def trigger_ratios(m: dict[str, Any]) -> dict[str, float]:
    """E4's trigger-local masked checks from the route split's cells: CI-masked P(o) over target P(o) at the trigger."""
    p = m["routes"]["p_cells"]
    return {"state_gated": p["L11"] / p["T1"], "w0_gated": p["L10"] / p["T1"], "target_p_o": p["T1"]}


def adam_reference(ctx: Any, S: Sizes) -> dict[str, Any]:
    """Adam lr 1e-5 regenerated and measured; G1 (it is E4's state) and C1 (the apparatus reads it as before)."""
    from commute import component_training, harness
    from commute.finetune import ArmAConfig

    cfg = component_training.TrainConfig(lr=ADAM_LR, steps=S.adam_steps, save_steps=(S.adam_steps,), optimizer="adam")
    curve, states = component_training.train(ctx.model, ctx.uv, ctx.train.tokens, ctx.held, ArmAConfig(o_token=ctx.token),
                                             cfg, ctx.device, ctx.W0, save_dir=LARGE / f"adam{S.suffix}", log=log)
    st = states[S.adam_steps]
    m = harness.measure(ctx, "adam", {k: v[0] for k, v in st.items()}, st, True, log)
    last = curve[-1]
    g1 = {"p_o": last["heldout_p_o"], "kl_other": last["heldout_kl_other"]}
    g1["pass"] = abs(g1["p_o"] - ADAM_REF["p_o"]) <= REPRO_TOL and abs(g1["kl_other"] - ADAM_REF["kl_other"]) <= REPRO_TOL
    return {"curve_last": last, "measure": m, "trigger": trigger_ratios(m), "G1_reproduces": g1}


def adam_control(a: dict[str, Any]) -> dict[str, Any]:
    """C1: the Adam reference reads as it did in E4 and the ladder, on every number the SGD predictions use."""
    pile, t, s = a["measure"]["pile"], a["trigger"], a["measure"]["routes"]["shares"]
    checks = {"pile_under_line": all(pile[k] <= LINE for k in ("stochastic_kl", "pgd_kl", "ci_masked_kl")),
              "state_gated": abs(t["state_gated"] - ADAM_REF["state_gated"]) <= TRIG_TOL,
              "w0_gated_under_half": t["w0_gated"] < 0.5,
              "d": abs(s["d"] - ADAM_REF["d"]) <= D_TOL, "r": abs(s["r"] - ADAM_REF["r"]) <= R_TOL}
    return {**checks, "pass": all(checks.values())}


def sgd_state(ctx: Any, path: Path) -> dict[str, Any]:
    """The saved SGD state measured; G2 (it is sgd_grid's lr-0.01 step-30 state)."""
    from commute import harness

    st = torch.load(path, weights_only=False)
    assert set(st) == set(ctx.W0), f"{path} has matrices {sorted(st)[:3]}..., expected the 24 decomposed ones"
    m = harness.measure(ctx, "sgd", {k: v[0] for k, v in st.items()}, st, True, log)
    t = trigger_ratios(m)
    g2 = {"p_o": t["target_p_o"], "rel_norm_max": m["rel_norm_max"]}
    g2["pass"] = abs(g2["p_o"] - SGD_REF["p_o"]) <= REPRO_TOL and abs(g2["rel_norm_max"] / SGD_REF["rel_norm_max"] - 1) <= 0.01
    return {"path": str(path), "measure": m, "trigger": t, "G2_is_saved_state": g2}


# ----------------------------------------------------------------------------- the score (rules: results/predictions.json)


def score(out: dict[str, Any]) -> dict[str, Any]:
    f = out["W0"]["pgd_floor"]
    sc: dict[str, Any] = {"G0_w0": out["W0"]["pass"], "G1_adam_reproduces": out["adam"]["G1_reproduces"]["pass"],
                          "C1_adam_control": out["adam"]["C1"]["pass"]}
    if "sgd" not in out:
        return sc
    sgd = out["sgd"]
    sc["G2_sgd_is_saved_state"] = sgd["G2_is_saved_state"]["pass"]
    pile, t, s = sgd["measure"]["pile"], sgd["trigger"], sgd["measure"]["routes"]["shares"]
    pgd = pile["pgd_kl"]
    sc["P1_sgd_pile"] = {"ratios": {k: pile[k] for k in ("stochastic_kl", "pgd_kl", "ci_masked_kl")},
                         "pass": all(pile[k] <= LINE for k in ("stochastic_kl", "pgd_kl", "ci_masked_kl")),
                         "pgd_decisive": not (LINE / (1 + f) < pgd < LINE * (1 + f))}
    sc["P2_sgd_state_gated"] = {"ratio": t["state_gated"], "pass": t["state_gated"] >= 0.8}
    sc["P3_sgd_w0_gated"] = {"ratio": t["w0_gated"], "pass": t["w0_gated"] < 0.5}
    ok = sgd["measure"]["routes"]["scorable"]
    sc["P4_sgd_direct_share"] = {"d": s["d"], "scorable": ok, "pass": s["d"] > 0.1 if ok else None}
    sc["reported"] = {"shares_sgd": s, "shares_adam": out["adam"]["measure"]["routes"]["shares"],
                      "gain_sgd": sgd["measure"]["routes"]["gain"], "scorable_gain": sgd["measure"]["routes"]["scorable"]}
    sc["status"] = ("scored" if sc["G0_w0"] and sc["G1_adam_reproduces"] and sc["C1_adam_control"] and sc["G2_sgd_is_saved_state"]
                    else "unscored: a gate or the control failed")
    return sc


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    out["score"] = score(out) if "adam" in out else {"G0_w0": False}
    dump(path, out)
    sys.exit(why)


def stage_run(args: argparse.Namespace) -> None:
    from commute import harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    S = SMOKE if args.smoke else FULL
    assert args.sgd_state.is_file(), f"no SGD state at {args.sgd_state} (sgd_grid's results/large/state_lr0.01.pt)"
    path = RESULTS / f"sgd_faith{S.suffix}.json"
    t0 = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S), "device": torch.cuda.get_device_name() if
                           torch.cuda.is_available() else "cpu", "W0": check_w0(ctx)}
    log(f"[stage W0] {time.time() - t0:.0f} s")
    dump(path, out)
    if S.enforce and not out["W0"]["pass"]:
        stop(path, out, "W0 is not at its known values: nothing scored")

    out["adam"] = adam_reference(ctx, S)
    out["adam"]["C1"] = adam_control(out["adam"])
    log(f"[stage Adam] {time.time() - t0:.0f} s; G1 {out['adam']['G1_reproduces']} trigger {out['adam']['trigger']} "
        f"C1 {out['adam']['C1']}")
    dump(path, out)
    if S.enforce and not (out["adam"]["G1_reproduces"]["pass"] and out["adam"]["C1"]["pass"]):
        stop(path, out, "the Adam reference does not read as recorded: SGD not run")

    out["sgd"] = sgd_state(ctx, args.sgd_state)
    log(f"[stage SGD] {time.time() - t0:.0f} s; G2 {out['sgd']['G2_is_saved_state']} trigger {out['sgd']['trigger']}")
    out["score"] = score(out)
    out["seconds"] = time.time() - t0
    dump(path, out)
    log(f"wrote {path} ({out['seconds']:.0f} s); score {json.dumps(out['score'], indent=1, default=float)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--sgd-state", type=Path, help="sgd_grid's saved state (results/large/state_lr0.01.pt)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.sgd_state is not None:
        stage_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
