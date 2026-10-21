"""sgd_ladder: does SGD component training, or Adam at a lower rate, pass the Pile harness on a ladder task where Adam
at 1e-5 failed? Item 3a of reports/2026-10-21_review_sgd-finding-influence.md; plan reports/2026-10-21_plan_sgd-ladder.md.

Per task (emoticon and the five ladder tasks E11 scored), all at step 10 of E11's recipe (arm A's loss, batch 32, clip
1, batch-order seed 0, Gamma and Delta frozen). Stages, cheapest first, each dumped as it finishes (the scored run is
its own smoke, 2026-10-21):
  W0        once per shard: the Pile harness under three adversary seeds; stop if off its known L40 values
            (commute.faith_score.w0_check)
  G1        the do-nothing fine-tune (lr 3e-5, regenerated) and E11's saved Adam 1e-5 state, re-read: held-out P(target)
            and KL elsewhere must match E11's, or the shard stops
  train     SGD at each rate of CFG.sgd_lrs, one kept by E4's rule (P >= 0.9 and KL elsewhere <= 2 x the do-nothing's,
            the least KL elsewhere among those); Adam at each rate of CFG.adam_lrs. Kept states saved as bf16 offsets
  control   do-nothing and Adam 1e-5 through harness.measure (Pile ratios to this shard's W0, route split); Adam's
            adversarial ratio must match E11's within 25%, or the shard stops before the new arms are measured
  scored    the kept SGD state and the Adam arms through harness.measure
harness.load reads the emoticon windows for its input moments; they are unused here (no outside share).

Usage (repo root, project venv, GPU):
  run.py --predictions   file results/predictions.json (refuses to overwrite)
  run.py --shard N       shard N of CFG.shards -> results/sgd_ladder_shard{N}.json; plot.py merges the shard files
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
LADDER = HERE.parent / "task_ladder" / "results" / "large"
E11 = HERE.parent / "reading_a_coverage" / "results"  # coverage.json, coverage_emoticon.json; states under large/<task>/

EMOTICON = "emoticon"
TASKS = (EMOTICON, "r0_t352_v310", "r1_t534_v369", "r1_t588_v452", "r2_t275_v2667", "r2_t359_v816")
E11_FAILED = ("r0_t352_v310", "r1_t534_v369", "r2_t275_v2667")  # Adam 1e-5 PGD 2.73, 2.24, 6.74 (r2_t359: 1.55, split)
TOL = 1e-6  # split.bases tolerance (harness.load computes the bases; unused here)
FT_LR, ADAM_REF = 3e-5, "adam_1e-05"
P_MIN, KL_FACTOR = 0.9, 2.0  # E4's rate rule: P(target) >= 0.9, KL elsewhere <= 2 x the fine-tune's at step 10
REPRO_TOL, PGD_REPRO_TOL = 0.01, 0.25  # absolute on held-out P and KL elsewhere; relative on Adam 1e-5's PGD ratio
SCORED = ("stochastic_kl", "pgd_kl")


@dataclass(frozen=True)
class Config:
    steps: int  # E11's saved states are step 10
    sgd_lrs: tuple[float, ...]
    adam_lrs: tuple[float, ...]
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]  # adversary seeds at W0; seed 0 is every ratio's reference
    shards: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        tasks = [t for s in self.shards for t in s]
        assert sorted(tasks) == sorted(TASKS) and all(self.shards), "every task in exactly one shard"
        assert self.steps == 10 and 0 in self.pile_seeds and len(self.pile_seeds) >= 2


CFG = Config(steps=10, sgd_lrs=(0.003, 0.01, 0.03, 0.1), adam_lrs=(1e-6, 3e-6), pile_rows=256, pile_pgd_steps=20,
             pile_seeds=(0, 1, 2), shards=tuple((t,) for t in TASKS))

PREDICTIONS: dict[str, Any] = {
    "about": "sgd_ladder (item 3a of reports/2026-10-21_review_sgd-finding-influence.md; plan "
             "reports/2026-10-21_plan_sgd-ladder.md): component training at step 10 on emoticon and E11's five ladder tasks "
             "with plain SGD (rate by E4's rule from 0.003, 0.01, 0.03, 0.1) and Adam at 1e-6 and 3e-6, beside E11's Adam "
             "1e-5 states (re-read) and the do-nothing fine-tune, scored on the Pile harness in the same job. One shard per "
             "task, each with its own W0. Filed before any run",
    "measurements": {
        "pile": "harness.measure: Pile val rows 0..255 in two 128-row blocks, one PGD adversary per block (20 steps of 0.1, "
                "seed 0), stochastic (2 draws); each a ratio to this shard's W0 under seed 0 (relative). Scored: stochastic "
                "and adversarial (PGD). At W0: 1 by definition (absolute PGD 0.69, stochastic 0.248). Blind to the trigger "
                "(about 3 trigger positions in 131k) and to damage the 20-step adversary does not find",
        "kl_other": "held-out KL elsewhere (all positions but the trigger) of the target model against W0, nats, absolute. "
                    "At W0: 0. Blind to Pile text and to the decomposition",
        "p_target": "held-out mean P(target) at the trigger, absolute. At W0: the task's baseline (emoticon 0.005)",
        "route_split": "local_faith.route_split at the trigger of the held-out windows: shares d, r, i, e_gate, e_delta "
                       "(absolute, sum 1); reported, not scored. Undefined at W0; do-nothing reads e_delta near 1. Blind "
                       "to anything off the trigger",
        "noise_floor": "f = W0's PGD max/min - 1 over adversary seeds 0, 1, 2 in the same shard (0.046 on the L40S, 0.209 on "
                       "the L40). commute.faith_score.verdict: pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f), inconclusive "
                       "between (about 1.24 to 1.81 on the L40: a reading within about 0.3 of 1.5 decides nothing on this "
                       "card). One batch-order seed per state; E11's state-level PGD spread on r2_t359 was 0.128",
    },
    "rate_rule": "E4's, per task, at step 10: among SGD rates with held-out P(target) >= 0.9 and KL elsewhere <= 2 x the "
                 "do-nothing's (this shard; emoticon 0.110 as in E4), the least KL elsewhere; if none, the highest P, "
                 "'did not learn'. Adam arms are fixed rates. An arm 'learned' if it meets both conditions",
    "arm_verdict": "learned and stochastic and PGD both 'pass' -> pass; either 'fail' -> fail; else inconclusive; not "
                   "learned -> 'did not learn' (counts as neither)",
    "gates": {
        "G0_w0": "first in every shard: W0's seed-0 PGD KL within 10% of 0.69, CI-masked KL within 2% of 0.348, PGD floor "
                 "<= 0.25 (faith_score.w0_check). Fails: the shard stops",
        "G1_references_reproduce": "before training: the regenerated do-nothing and the re-read Adam 1e-5 state read "
                                   "E11's held-out P and KL elsewhere to 0.01. Fails: the shard stops",
        "C1_control": "after training, before any new arm is measured: the re-read Adam 1e-5 state's PGD ratio within 25% "
                      "of E11's (so it reads 'fail' again on the three failing tasks: the harness's positive control). "
                      "Fails: the shard stops",
    },
    "expected": {
        "W0": "PGD KL 0.69, floor 0.05 to 0.21",
        "do_nothing_E11": "PGD / stochastic / KL elsewhere: emoticon 2.18 / 1.00 / 0.055; r0_t352 1.58 / 1.00 / 0.105; "
                          "r1_t534 2.17 / 1.01 / 0.122; r1_t588 1.22 / 0.98 / 0.083; r2_t275 2.26 / 1.02 / 0.203; "
                          "r2_t359 1.24 / 1.02 / 0.091 (reported; reproduced under G1)",
        "adam_1e-5_E11": "emoticon 1.03 / 1.00 / 0.016 (pass); r0_t352 2.73 / 0.98 / 0.200 (fail); r1_t534 2.24 / 1.00 / "
                         "0.232 (fail); r1_t588 1.16 / 1.04 / 0.146 (pass); r2_t275 6.74 / 1.75 / 0.508 (fail); r2_t359 "
                         "1.55 / 1.03 / 0.188 (inconclusive). The failing three are the harness's positive control: "
                         "they should read 'fail' again",
        "sgd_emoticon": "sgd_grid at step 10: rate 0.01 selected (P 0.980, KL elsewhere 0.011); if sgd_faith passes, PGD "
                        "near 1",
        "adam_1e-6": "does not learn by step 10 (a tenth of the 1e-5 arm's per-step change); Pile ratios near 1",
        "adam_3e-6": "open; if it learns, KL elsewhere below Adam 1e-5's",
    },
    "predictions": [
        {"id": "Q1_failing_task_passes", "kind": "two-sided question, no direction filed",
         "rule": "on r0_t352, r1_t534 or r2_t275 (Adam 1e-5 failed in E11), some new arm (selected SGD, Adam 1e-6, Adam "
                 "3e-6) reads 'pass'",
         "decision": "yes -> item 3b (steps 100 and 1000, both optimisers, H100 when available); only 'inconclusive' -> "
                     "those states rerun with more adversary seeds or on an H100 before 3b is decided; no -> the frozen-"
                     "Gamma failure holds across optimisers at step 10 and re-gating stays the main line"},
        {"id": "P2_sgd_learns", "rule": "the rate rule finds an SGD rate on every task", "predicted": "pass",
         "why": "sgd_grid: 0.01 to 0.1 learn emoticon by step 10. Scope: a fail is reported with the grid's range"},
        {"id": "P3_sgd_less_damage", "rule": "on every task where SGD learned, its KL elsewhere < the re-read Adam 1e-5's",
         "predicted": "pass", "why": "emoticon at step 10: 0.011 against 0.016; the update is smaller (not matched)"},
    ],
    "reported_not_scored": ["route-split shares and trigger ratios of every state", "CI-masked and unmasked ratios",
                            "every SGD rate's step-10 P and KL elsewhere", "largest relative change per matrix"],
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def e11_reference(task: str) -> dict[str, Any]:
    """E11's recorded seed-0 numbers for the task: held-out P and KL elsewhere, and Pile ratios, per placement."""
    rec = json.loads((E11 / ("coverage_emoticon.json" if task == EMOTICON else "coverage.json")).read_text())
    s = rec["tasks"][task]["seeds"]["0"]
    return {arm: {**s[key]["heldout"], "pile": s[key]["pile"]["ratio"]} for arm, key in
            (("do_nothing", "do_nothing"), (ADAM_REF, "pullback"))}


# ----------------------------------------------------------------------------- per task


def task_context(ctx: Any, task: str) -> Any:
    """The shared context with the task's windows, target token and W0 route-split cells."""
    from commute.local_faith import route_cells

    if task == EMOTICON:
        w = torch.load(ROUND1 / "windows.pt", weights_only=False)
        token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    else:
        w = torch.load(LADDER / f"windows_{task}.pt", weights_only=False)
        token = w["task"]["target"]
    held = w["heldout"]
    return replace(ctx, train=w["train"], held=held, token=token, rbase=route_cells(ctx.model, held.tokens, held.half, token))


def a_cfg(tctx: Any) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=tctx.token, lr=FT_LR, steps=CFG.steps, checkpoints=(CFG.steps,), seed=0)


def heldout(tctx: Any, dW: dict[str, torch.Tensor]) -> dict[str, float]:
    """Held-out P(target) and KL elsewhere of the target model W0 + dW."""
    from commute.faithfulness import model_state
    from commute.finetune import eval_arm_a

    net = tctx.model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    with model_state(tctx.model, target={k: tctx.W0[k] + dW[k].to(tctx.W0[k].device) for k in tctx.W0}):
        ev = eval_arm_a(net, ref, tctx.held, tctx.token, tctx.device)
    return {"p_o": ev["p_o"], "kl_other": ev["kl_other"]}


def g1(tctx: Any, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two reference states (do-nothing regenerated, E11's Adam 1e-5 loaded) and their held-out numbers against
    E11's. States are (dW, state or None)."""
    from commute.finetune import finetune_arm_a

    ft = finetune_arm_a(tctx.model, tctx.train, tctx.held, a_cfg(tctx), tctx.device, log)
    st = torch.load(E11 / "large" / task / "pullback_s0" / f"state_step{CFG.steps}.pt", weights_only=False)
    states = {"do_nothing": (ft.deltas[CFG.steps], None), ADAM_REF: ({k: v[0] for k, v in st.items()}, st)}
    ho = {"do_nothing": {k: ft.evals[CFG.steps][k] for k in ("p_o", "kl_other")}, ADAM_REF: heldout(tctx, states[ADAM_REF][0])}
    ref = e11_reference(task)
    rec: dict[str, Any] = {a: {"heldout": ho[a], "dP": ho[a]["p_o"] - r["p_o"], "dkl": ho[a]["kl_other"] - r["kl_other"]}
                           for a, r in ref.items()}
    rec["pass"] = all(abs(rec[a]["dP"]) <= REPRO_TOL and abs(rec[a]["dkl"]) <= REPRO_TOL for a in ref)
    log(f"[{task} G1] {json.dumps(rec, default=float)}")
    return rec, states


def select_lr(last: dict[str, dict[str, float]], limit: float) -> tuple[str, bool]:
    """E4's rule (sgd_grid.select_lr) with the task's own limit: among rates with P >= 0.9 and KL elsewhere <= limit,
    the least KL elsewhere; if none, the highest P, marked 'did not learn'."""
    ok = [lr for lr, c in last.items() if c["p_o"] >= P_MIN and c["kl_other"] <= limit]
    if ok:
        return min(ok, key=lambda lr: last[lr]["kl_other"]), True
    return max(last, key=lambda lr: last[lr]["p_o"]), False


def train_arm(tctx: Any, optimizer: str, lr: float) -> tuple[dict[str, float], Any]:
    from commute import component_training

    cfg = component_training.TrainConfig(lr=lr, steps=CFG.steps, save_steps=(CFG.steps,), optimizer=optimizer)
    curve, states = component_training.train(tctx.model, tctx.uv, tctx.train.tokens, tctx.held, a_cfg(tctx), cfg,
                                             tctx.device, tctx.W0, log=log)
    c = curve[-1]
    return {"p_o": c["heldout_p_o"], "kl_other": c["heldout_kl_other"], "rel_norm_max": c["rel_norm_max"]}, states[CFG.steps]


def save_offsets(tctx: Any, task: str, name: str, st: Any) -> str:
    """bf16 offsets (dU, dV) to the alive columns (U' = U_A + dU; U_A = component_training.alive_uv at W0)."""
    off = {k: ((U2 - tctx.uv_cpu[k][0]).bfloat16(), (V2 - tctx.uv_cpu[k][1]).bfloat16()) for k, (_, U2, V2) in st.items()}
    p = LARGE / task / f"{name}_step{CFG.steps}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"offsets": off, "task": task, "arm": name, "steps": CFG.steps, "layout": "theory (d_out, n), (d_in, n)"}, p)
    return str(p.relative_to(HERE))


def train_all(tctx: Any, task: str, limit: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """SGD over the rate grid (one state kept by the rule) and Adam at each low rate; kept states saved."""
    sgd_last, sgd_states = {}, {}
    for lr in CFG.sgd_lrs:
        sgd_last[f"{lr:g}"], sgd_states[f"{lr:g}"] = train_arm(tctx, "sgd", lr)
    lr, learned = select_lr(sgd_last, limit)
    log(f"[{task} SGD grid] {json.dumps(sgd_last, default=float)} -> lr {lr} learned {learned} (limit {limit:.4f})")
    kept, last = {f"sgd_{lr}": sgd_states[lr]}, {f"sgd_{lr}": sgd_last[lr]}
    del sgd_states
    for a in CFG.adam_lrs:
        last[f"adam_{a:g}"], kept[f"adam_{a:g}"] = train_arm(tctx, "adam", a)
    rec = {"sgd_curve_last": sgd_last, "sgd_selected": lr, "sgd_learned": learned, "last": last,
           "saved": {n: save_offsets(tctx, task, n, st) for n, st in kept.items()}}
    return rec, {n: ({k: v[0] for k, v in st.items()}, st) for n, st in kept.items()}


def measured(tctx: Any, name: str, dW: dict[str, Any], state: Any, ho: dict[str, float]) -> dict[str, Any]:
    from commute import faith_score, harness

    m = harness.measure(tctx, name, dW, state, True, log)
    return {"heldout": {"p_o": ho["p_o"], "kl_other": ho["kl_other"]}, "pile": m["pile"], "routes": m["routes"],
            "trigger": faith_score.trigger_ratios(m), "rel_norm_max": m["rel_norm_max"]}


# ----------------------------------------------------------------------------- the score (rules: results/predictions.json)


def arm_verdict(a: dict[str, Any], limit: float, floors: dict[str, float]) -> dict[str, Any]:
    from commute import faith_score

    learned = a["heldout"]["p_o"] >= P_MIN and a["heldout"]["kl_other"] <= limit
    pile = {k: faith_score.verdict(a["pile"][k], floors[k]) for k in SCORED}
    overall = ("fail" if "fail" in pile.values() else "inconclusive" if "inconclusive" in pile.values() else "pass")
    return {"learned": learned, "pile": pile, "verdict": overall if learned else "did not learn"}


def score_task(rec: dict[str, Any], floors: dict[str, float]) -> dict[str, Any]:
    v = {name: arm_verdict(a, rec["limit"], floors) for name, a in rec["arms"].items() if name != "do_nothing"}
    new = {n: x for n, x in v.items() if n != ADAM_REF}
    return {"verdicts": v, "e11_failed": rec["task"] in E11_FAILED,
            "any_new_arm_passes": any(x["verdict"] == "pass" for x in new.values()),
            "any_new_arm_inconclusive": any(x["verdict"] == "inconclusive" for x in new.values())}


# ----------------------------------------------------------------------------- main


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(path, out)
    sys.exit(why)


def run_task(tctx: Any, task: str, out: dict[str, Any], path: Path) -> None:
    """G1, training, the control's Pile reading, then the new arms; `out` dumped after each stage."""
    rec: dict[str, Any] = {"task": task, "arms": {}}
    out["tasks"][task] = rec
    rec["G1"], refs = g1(tctx, task)
    rec["limit"] = KL_FACTOR * rec["G1"]["do_nothing"]["heldout"]["kl_other"]
    dump(path, out)
    if not rec["G1"]["pass"]:
        stop(path, out, f"{task}: the references do not reproduce E11's held-out numbers")
    rec["train"], new = train_all(tctx, task, rec["limit"])
    dump(path, out)
    for name, (dW, st) in refs.items():
        rec["arms"][name] = measured(tctx, f"{task}/{name}", dW, st, rec["G1"][name]["heldout"])
        dump(path, out)
    pgd_rel = rec["arms"][ADAM_REF]["pile"]["pgd_kl"] / e11_reference(task)[ADAM_REF]["pile"]["pgd_kl"] - 1
    rec["control"] = {"adam_pgd_rel": pgd_rel, "pass": abs(pgd_rel) <= PGD_REPRO_TOL}
    log(f"[{task} control] Adam 1e-5 PGD ratio {pgd_rel:+.3f} relative to E11 -> {'PASS' if rec['control']['pass'] else 'FAIL'}")
    dump(path, out)
    if not rec["control"]["pass"]:
        stop(path, out, f"{task}: Adam 1e-5's adversarial ratio does not reproduce E11's; new arms not measured")
    for name, (dW, st) in new.items():
        rec["arms"][name] = measured(tctx, f"{task}/{name}", dW, st, rec["train"]["last"][name])
        dump(path, out)
    rec["score"] = score_task(rec, {"stochastic_kl": out["W0"]["stochastic_floor"], "pgd_kl": out["W0"]["pgd_floor"]})
    log(f"[{task} score] {json.dumps(rec['score'], default=float)}")


def run_shard(shard: int, device: str) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    path = RESULTS / f"sgd_ladder_shard{shard}.json"
    t0 = time.time()
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, CFG.pile_rows, CFG.pile_pgd_steps,
                       CFG.pile_seeds, log)
    out: dict[str, Any] = {"shard": shard, "config": asdict(CFG), "tasks": {}, "W0": faith_score.w0_check(ctx.pile),
                           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    log(f"[W0] {time.time() - t0:.0f} s; {json.dumps({k: v for k, v in out['W0'].items() if k != 'pile'}, default=float)}")
    dump(path, out)
    if not out["W0"]["pass"]:
        stop(path, out, "W0 is not at its known values: nothing run")
    for task in CFG.shards[shard]:
        run_task(task_context(ctx, task), task, out, path)
        log(f"[{task} done] {time.time() - t0:.0f} s")
    out["seconds"] = time.time() - t0
    dump(path, out)
    log(f"wrote {path} ({out['seconds']:.0f} s)")


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--shard", type=int, choices=range(len(CFG.shards)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.shard is not None:
        run_shard(args.shard, args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
