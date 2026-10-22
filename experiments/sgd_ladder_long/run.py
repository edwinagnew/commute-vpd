"""sgd_ladder_long: does SGD component training stay adversarially faithful on the Pile at steps 100 and 1000 on the
ladder tasks where it passed at step 10, and does Adam at 3e-6 learn and pass there? Item 3b of
reports/2026-10-21_review_sgd-finding-influence.md; plan reports/2026-10-21_plan_sgd-ladder-long.md.

One job per task (emoticon, the control, first and alone; then r0_t352, r1_t534, r2_t275). E11's recipe (arm A's
loss, batch 32, clip 1, batch-order seed 0, Gamma and Delta frozen), each arm trained once to step 1000, states kept at
10, 100 and 1000: SGD at the rate 3a selected for the task (read from experiments/sgd_ladder's shard files), Adam 1e-5
and Adam 3e-6; and the do-nothing fine-tune (Adam 3e-5) to step 1000 with the same checkpoints. Stages, cheapest first,
each dumped as it finishes (the scored run is its own smoke):
  W0      the Pile harness under adversary seeds 0, 1, 2; stop unless at its known L40 values (faith_score.w0_check)
  train   fine-tune and the three arms to step 1000; every state saved as bf16 offsets (results/large/<task>/)
  G1      every step-10 held-out P(target) and KL elsewhere equals 3a's to 0.01, or stop
  C1      the SGD state at step 10 through the Pile harness: PGD and stochastic ratios within 25% of 3a's, or stop
  scored  steps 100 then 1000: SGD, Adam 3e-6, Adam 1e-5, do-nothing through harness.measure; verdict per state

Usage (repo root, project venv, GPU):
  run.py --predictions   file results/predictions.json (refuses to overwrite)
  run.py --task T        one task -> results/sgd_ladder_long_<T>.json
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
LADDER = HERE.parent / "task_ladder" / "results" / "large"
SGD_LADDER = HERE.parent / "sgd_ladder" / "results"  # 3a: sgd_ladder_shard*.json
E16 = HERE.parent / "split_comparison" / "results" / "split_comparison.json"

EMOTICON = "emoticon"
TASKS = (EMOTICON, "r0_t352_v310", "r1_t534_v369", "r2_t275_v2667")
Q1_TASKS = ("r0_t352_v310", "r1_t534_v369")  # SGD passed at step 10 in 3a (PGD 0.95, 1.10)
TOL = 1e-6  # split.bases tolerance (harness.load computes the bases; unused here)
FT_LR, ADAM_LRS = 3e-5, (1e-5, 3e-6)
P_MIN = 0.9
REPRO_TOL, PILE_REPRO_TOL = 0.01, 0.25  # absolute on held-out P and KL elsewhere; relative on Pile ratios
SCORED = ("stochastic_kl", "pgd_kl")
FLOOR_KEY = {"stochastic_kl": "stochastic_floor", "pgd_kl": "pgd_floor"}  # each reading against its own W0 floor


@dataclass(frozen=True)
class Config:
    steps: tuple[int, ...]  # saved and scored steps; step 10 only as the reproduction of 3a
    train_steps: int
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]  # adversary seeds at W0; seed 0 is every ratio's reference

    def __post_init__(self) -> None:
        assert self.steps == (10, 100, 1000) and self.train_steps == self.steps[-1], self.steps
        assert 0 in self.pile_seeds and len(self.pile_seeds) >= 2
        assert (self.pile_rows, self.pile_pgd_steps) == (256, 20), "3a's harness sizes, or C1 compares unlike readings"


CFG = Config(steps=(10, 100, 1000), train_steps=1000, pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2))

PREDICTIONS: dict[str, Any] = {
    "about": "sgd_ladder_long (item 3b of reports/2026-10-21_review_sgd-finding-influence.md; plan "
             "reports/2026-10-21_plan_sgd-ladder-long.md): component training to step 1000 on emoticon, r0_t352_v310, "
             "r1_t534_v369 and r2_t275_v2667 with SGD at 3a's selected rate (0.01, 0.03, 0.03, 0.1), Adam 1e-5 and Adam "
             "3e-6, beside the do-nothing fine-tune to step 1000; Pile harness at steps 100 and 1000 against a same-job "
             "W0, cluster L40. One job per task. Filed before any run",
    "measurements": {
        "pile": "harness.measure: Pile val rows 0..255 in two 128-row blocks, one PGD adversary per block (20 steps, seed "
                "0), stochastic (2 draws); ratios to this job's W0 under seed 0 (relative). Scored: stochastic and PGD. At "
                "W0: 1 by definition. Blind to the trigger (about 3 positions in 131k) and to damage the 20-step adversary "
                "does not find",
        "p_target": "held-out mean P(target) at the trigger, absolute, from the trainer's own eval at the step. At W0: the "
                    "task's baseline (emoticon 0.005). Blind to everything off the trigger",
        "kl_other": "held-out KL elsewhere of the target model against W0, nats, absolute; read against 3a's task limit "
                    "(2 x the do-nothing's at step 10: 0.110, 0.209, 0.243, 0.407). Reported, not part of the verdict. At "
                    "W0: 0. Blind to Pile text and to the decomposition",
        "rel_norm_max": "largest ||delta-W|| / ||W0|| over the 24 matrices, reported. At W0: 0. SGD's update is a sum "
                        "of clipped steps of length at most the rate, so it is bounded by rate x steps in parameter norm",
        "noise_floor": "per reading, f = W0's max/min - 1 over adversary seeds 0, 1, 2 in the same job (PGD about 0.21 on "
                       "the L40; stochastic about 0.002). faith_score.verdict: pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f), "
                       "inconclusive between (PGD band about 1.24 to 1.81 on the L40). One batch-order seed per state",
    },
    "state_verdict": "held-out P(target) < 0.9 -> 'below P 0.9'; else stochastic and PGD each by faith_score.verdict "
                     "against its own floor: either 'fail' -> fail, either 'inconclusive' -> inconclusive, else pass",
    "gates": {
        "G0_w0": "first: W0's seed-0 PGD KL within 10% of 0.69, CI-masked KL within 2% of 0.348, PGD floor <= 0.25 "
                 "(faith_score.w0_check). Fails: the job stops",
        "G1_step10_reproduces_3a": "after training, before any scored Pile pass: the do-nothing's and every arm's step-10 "
                                   "held-out P and KL elsewhere within 0.01 of 3a's (Adam 1e-5 against E11's saved state as "
                                   "3a re-read it). Fails: the job stops",
        "C1_sgd_step10": "the SGD arm's step-10 state through the harness first: PGD and stochastic ratios within 25% "
                         "of 3a's (emoticon 1.04 / 1.01, r0_t352 0.95 / 1.00, r1_t534 1.10 / 0.99, r2_t275 1.50 / 1.03). "
                         "Fails: the job stops before the scored states",
        "C2_emoticon_known_pass": "emoticon job only, read before the ladder jobs are submitted: Adam 1e-5 reads 'pass' at "
                                  "steps 100 and 1000 (E16 on an H100: PGD 1.03, 0.99). Fails: ladder jobs not submitted",
    },
    "expected": {
        "W0": "PGD KL 0.69, PGD floor about 0.21, stochastic floor about 0.002",
        "do_nothing": "emoticon PGD 1.14 at 100 and 1.50 at 1000 (E16, H100); ladder open, reported (step 10 in 3a: "
                      "1.56, 2.29, 2.31). Stochastic near 1. P(target) >= 0.97",
        "adam_1e-05": "emoticon pass at 100 and 1000 (E16 1.03, 0.99; P 0.997). Ladder: fail at both steps (3a step 10: "
                      "2.70, 2.24, 6.83, and KL elsewhere 0.20, 0.23, 0.51 at step 10). Reported, not scored",
        "sgd": "emoticon pass at both steps (step 10: 1.04). Ladder: Q1, no direction filed. rel_norm_max grows past "
               "3a's step-10 values (0.002 to 0.011)",
        "adam_3e-06": "emoticon learned at step 10 (P 0.955) and passed (1.12): pass expected at 100 and 1000. Ladder: Q2",
        "rel_norm_max": "Adam 1e-5 on emoticon 0.008 at 100, 0.016 at 1000 (E16)",
    },
    "predictions": [
        {"id": "Q1_sgd_stays_faithful", "kind": "two-sided question, no direction filed",
         "rule": "on r0_t352_v310 and r1_t534_v369, the SGD state's verdict is 'pass' (P(target) >= 0.9, stochastic and "
                 "PGD 'pass') at step 100 and at step 1000: four states",
         "decision": "all four pass -> the decision's move-back bullet is met (a map at the floor with the behaviour at "
                     "steps 100 and 1000); Q1 returns as the first sub-question, taken to the user. Any 'fail' or 'below "
                     "P 0.9' -> the step-10 pass is a small-update effect; re-gating stays the main line. Only "
                     "'inconclusive' among non-passes -> those states rerun with more adversary seeds or on an H100 "
                     "before deciding",
         "blind_to": "the trigger (Pile) and off-trigger damage (P); KL elsewhere is reported beside it"},
        {"id": "Q2_adam_3e-6_learns_and_passes", "kind": "two-sided question, no direction filed",
         "rule": "on r0_t352_v310, r1_t534_v369 and r2_t275_v2667: the first scored step (100, 1000) at which Adam 3e-6 "
                 "reaches P(target) >= 0.9, and its verdict there",
         "decision": "learns and passes where Adam 1e-5 fails -> Adam's ladder failure is a step-size effect, not the "
                     "optimiser's; learns and fails while SGD passes on the same task -> the optimiser; does not learn "
                     "by step 1000 -> unanswered at 3e-6 (scope: one rate)"},
    ],
    "reported_not_scored": ["r2_t275's SGD and Adam states (inconclusive at step 10 in 3a)",
                            "KL elsewhere against the task limit", "rel_norm_max",
                            "route-split shares, trigger ratios, CI-masked and unmasked ratios",
                            "Adam 1e-5 at 100 and 1000 on the ladder tasks"],
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(path, out)
    sys.exit(why)


def ref_3a(task: str) -> dict[str, Any]:
    """3a's record of the task: the selected SGD rate, the KL limit, every step-10 held-out reading, the SGD state's
    Pile ratios."""
    for p in sorted(SGD_LADDER.glob("sgd_ladder_shard*.json")):
        tasks = json.loads(p.read_text())["tasks"]
        if task in tasks:
            r = tasks[task]
            sgd = f"sgd_{r['train']['sgd_selected']}"
            assert r["train"]["sgd_learned"], f"{task}: 3a's SGD did not learn"
            held = {"do_nothing": r["G1"]["do_nothing"]["heldout"], "adam_1e-05": r["G1"]["adam_1e-05"]["heldout"],
                    sgd: r["train"]["last"][sgd], "adam_3e-06": r["train"]["last"]["adam_3e-06"]}
            return {"sgd_lr": float(r["train"]["sgd_selected"]), "sgd": sgd, "limit": r["limit"], "heldout10": held,
                    "sgd_pile10": r["arms"][sgd]["pile"]}
    raise FileNotFoundError(f"{task} in no {SGD_LADDER}/sgd_ladder_shard*.json")


def task_context(ctx: Any, task: str) -> Any:
    from commute import harness

    if task == EMOTICON:
        w = torch.load(ROUND1 / "windows.pt", weights_only=False)
        token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    else:
        w = torch.load(LADDER / f"windows_{task}.pt", weights_only=False)
        token = w["task"]["target"]
    return harness.with_windows(ctx, w, token)


# ----------------------------------------------------------------------------- training (cheap; every state saved)


def a_cfg(tctx: Any) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=tctx.token, lr=FT_LR, steps=CFG.train_steps, checkpoints=CFG.steps, seed=0)


def train_arm(tctx: Any, task: str, name: str, optimizer: str, lr: float) -> tuple[dict[int, Any], dict[int, Any]]:
    """One arm to CFG.train_steps: held-out readings and the state at each of CFG.steps, states saved as offsets."""
    from commute import arms, component_training

    cfg = component_training.TrainConfig(lr=lr, steps=CFG.train_steps, save_steps=CFG.steps, optimizer=optimizer)
    curve, states = component_training.train(tctx.model, tctx.uv, tctx.train.tokens, tctx.held, a_cfg(tctx), cfg,
                                             tctx.device, tctx.W0, log=log)
    ho = {s: {"p_o": curve[s - 1]["heldout_p_o"], "kl_other": curve[s - 1]["heldout_kl_other"],
              "rel_norm_max": curve[s - 1]["rel_norm_max"]} for s in CFG.steps}
    for s, st in states.items():
        arms.save_offsets(LARGE / task / f"{name}_step{s}.pt", tctx.uv_cpu, st,
                          {"task": task, "arm": name, "optimizer": optimizer, "lr": lr, "step": s})
    return ho, {s: ({k: v[0] for k, v in st.items()}, st) for s, st in states.items()}


def train_all(tctx: Any, task: str, ref: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[int, Any]]]:
    """The do-nothing fine-tune and the three arms. Returns held-out readings per arm and step, and per arm and step
    (dW, state or None)."""
    from commute.finetune import finetune_arm_a

    ft = finetune_arm_a(tctx.model, tctx.train, tctx.held, a_cfg(tctx), tctx.device, log)
    held = {"do_nothing": {s: {k: ft.evals[s][k] for k in ("p_o", "kl_other", "rel_norm_max")} for s in CFG.steps}}
    states: dict[str, dict[int, Any]] = {"do_nothing": {s: (ft.deltas[s], None) for s in CFG.steps}}
    for optimizer, lr in (("sgd", ref["sgd_lr"]), *(("adam", a) for a in ADAM_LRS)):
        name = ref["sgd"] if optimizer == "sgd" else f"adam_{lr:g}"
        t0 = time.time()
        held[name], states[name] = train_arm(tctx, task, name, optimizer, lr)
        log(f"[{task} train {name}] {time.time() - t0:.0f} s; {json.dumps(held[name], default=float)}")
    return held, states


def g1(held: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {n: {k: held[n][10][k] - r[k] for k in ("p_o", "kl_other")} for n, r in ref["heldout10"].items()}
    rec["pass"] = all(abs(d) <= REPRO_TOL for n in ref["heldout10"] for d in rec[n].values())
    return rec


# ----------------------------------------------------------------------------- Pile readings and the score


def measured(tctx: Any, label: str, dw_state: tuple[Any, Any], ho: dict[str, float]) -> dict[str, Any]:
    """A state (dW, components or None for do-nothing) through harness.measure, with its held-out readings beside."""
    from commute import faith_score, harness

    m = harness.measure(tctx, label, dw_state[0], dw_state[1], True, log)
    return {"heldout": ho, "pile": m["pile"], "routes": m["routes"], "trigger": faith_score.trigger_ratios(m),
            "rel_norm_max": m["rel_norm_max"]}


def state_verdict(m: dict[str, Any], limit: float, w0: dict[str, Any]) -> dict[str, Any]:
    from commute import faith_score

    pile = {k: faith_score.verdict(m["pile"][k], w0[FLOOR_KEY[k]]) for k in SCORED}
    overall = "fail" if "fail" in pile.values() else "inconclusive" if "inconclusive" in pile.values() else "pass"
    return {"pile": pile, "kl_within_limit": m["heldout"]["kl_other"] <= limit,
            "verdict": overall if m["heldout"]["p_o"] >= P_MIN else "below P 0.9"}


def score_task(task: str, rec: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    v = {f"{n}@{s}": rec["states"][n][str(s)]["score"]["verdict"] for n in rec["states"] for s in CFG.steps[1:]}
    a3 = [s for s in CFG.steps[1:] if rec["states"]["adam_3e-06"][str(s)]["heldout"]["p_o"] >= P_MIN]
    out: dict[str, Any] = {"verdicts": v, "sgd_all_pass": all(v[f"{ref['sgd']}@{s}"] == "pass" for s in CFG.steps[1:]),
                           "in_Q1": task in Q1_TASKS,
                           "adam_3e-06_first_learned": {"step": a3[0], "verdict": v[f"adam_3e-06@{a3[0]}"]} if a3 else None}
    if task == EMOTICON:
        out["C2_pass"] = all(v[f"adam_1e-05@{s}"] == "pass" for s in CFG.steps[1:])
        e16 = json.loads(E16.read_text())["states"]["ft"]
        out["E16_pgd"] = {f"{n}@{s}": e16[str(s)][k]["pile"]["pgd_kl"] for s in CFG.steps[1:]
                          for n, k in (("adam_1e-05", "component_training"), ("do_nothing", "nothing"))}
    return out


# ----------------------------------------------------------------------------- main


def run_task(task: str, device: str) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    path, t0, ref = RESULTS / f"sgd_ladder_long_{task}.json", time.time(), ref_3a(task)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, CFG.pile_rows, CFG.pile_pgd_steps,
                       CFG.pile_seeds, log)
    out: dict[str, Any] = {"task": task, "config": asdict(CFG), "ref_3a": ref, "W0": faith_score.w0_check(ctx.pile),
                           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "states": {}}
    log(f"[W0] {time.time() - t0:.0f} s; {json.dumps({k: v for k, v in out['W0'].items() if k != 'pile'}, default=float)}")
    dump(path, out)
    if not out["W0"]["pass"]:
        stop(path, out, "W0 is not at its known values: nothing run")
    tctx = task_context(ctx, task)
    out["heldout"], states = train_all(tctx, task, ref)
    out["G1"] = g1(out["heldout"], ref)
    log(f"[{task} G1] {json.dumps(out['G1'], default=float)}")
    dump(path, out)
    if not out["G1"]["pass"]:
        stop(path, out, f"{task}: a step-10 held-out reading differs from 3a's by more than {REPRO_TOL}")
    sgd = ref["sgd"]
    rec = out["states"]
    m = measured(tctx, f"{task}/{sgd}@10", states[sgd][10], out["heldout"][sgd][10])
    rel = {k: m["pile"][k] / ref["sgd_pile10"][k] - 1 for k in SCORED}
    out["C1"] = {"rel": rel, "pass": all(abs(x) <= PILE_REPRO_TOL for x in rel.values())}
    rec.setdefault(sgd, {})["10"] = {**m, "score": state_verdict(m, ref["limit"], out["W0"])}
    log(f"[{task} C1] SGD step 10 vs 3a {json.dumps(rel, default=float)} -> {'PASS' if out['C1']['pass'] else 'FAIL'}")
    dump(path, out)
    if not out["C1"]["pass"]:
        stop(path, out, f"{task}: the SGD state's step-10 Pile ratios do not reproduce 3a's")
    for s in CFG.steps[1:]:
        for name in (sgd, "adam_3e-06", "adam_1e-05", "do_nothing"):
            m = measured(tctx, f"{task}/{name}@{s}", states[name][s], out["heldout"][name][s])
            rec.setdefault(name, {})[str(s)] = {**m, "score": state_verdict(m, ref["limit"], out["W0"])}
            log(f"[{task} {name}@{s}] {json.dumps(rec[name][str(s)]['score'])} ({time.time() - t0:.0f} s)")
            dump(path, out)
            del states[name][s]
    out["score"] = score_task(task, out, ref)
    out["seconds"] = time.time() - t0
    log(f"[{task} score] {json.dumps(out['score'], default=float)}")
    dump(path, out)
    log(f"wrote {path} ({out['seconds']:.0f} s)")


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.task is not None:
        run_task(args.task, args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
