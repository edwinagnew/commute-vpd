"""sgd_kv: does SGD component training, or Adam at a lower rate, pass the Pile harness on E10's key-to-value tasks,
in particular rand32, where Adam at 1e-5 failed (adversarial ratio 2.81)? Item 4 of
reports/2026-10-21_review_sgd-finding-influence.md; plan reports/2026-10-21_plan_sgd-kv.md.

Per task (rand1, ident32, rand32; E10's windows), E10's recipe: arm A's loss with the task's lookup table as target,
batch 32, clip 1, batch-order seed 0, up to 300 steps with checkpoints 10, 30, 100, 300, Gamma and Delta frozen; each
arm scored at its first checkpoint with held-out P(target) >= 0.5 (E10's rule). Stages, cheapest first, each dumped as
it finishes (the scored run is its own smoke, 2026-10-21):
  W0        once per shard: the Pile harness under three adversary seeds; stop if off its known values
            (commute.faith_score.w0_check)
  G1        the W fine-tune (lr 3e-5, regenerated: the rate rule's limit) and E10's saved Adam 1e-5 state at its scored
            step (re-read; regenerated in-job if the file is absent): held-out P(target) and KL elsewhere must match
            E10's to 0.01, or the shard stops
  train     SGD at each rate of CFG.sgd_lrs, one kept by E4's rule (learned, KL elsewhere <= 2 x the fine-tune's at its
            scored step, the least KL elsewhere among those); Adam at each rate of CFG.adam_lrs. Kept states saved as
            bf16 offsets (commute.arms.save_offsets)
  control   E10's Adam 1e-5 state through harness.measure; its adversarial ratio must match E10's within 25% (on rand32:
            reads 'fail' again, the harness's positive control), or the shard stops before the new arms are measured
  scored    the kept SGD state and the Adam arms through harness.measure
harness.load reads round 1's emoticon windows for its input moments; they are unused here (no outside share).

Usage (repo root, project venv, GPU):
  run.py --predictions   file results/predictions.json (refuses to overwrite)
  run.py --shard N       shard N of CFG.shards -> results/sgd_kv_shard{N}.json; plot.py merges the shard files
"""

from __future__ import annotations

import argparse
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
E10 = HERE.parent / "kv_task" / "results"  # kv_task.json; large/windows_<task>.pt, large/<task>/pullback/state_step<n>.pt

TASKS = ("rand32", "ident32", "rand1")
TOL = 1e-6  # split.bases tolerance (harness.load computes the bases; unused here)
FT_LR, ADAM_REF_LR, ADAM_REF = 3e-5, 1e-5, "adam_1e-05"
P_LEARN, KL_FACTOR = 0.5, 2.0  # E10's learned line; E4's limit is 2 x the fine-tune's KL elsewhere at its scored step
REPRO_TOL, PGD_REPRO_TOL = 0.01, 0.25  # absolute on held-out P and KL elsewhere; relative on Adam 1e-5's PGD ratio
SCORED = ("stochastic_kl", "pgd_kl")


@dataclass(frozen=True)
class Config:
    steps: int
    checkpoints: tuple[int, ...]  # E10's
    sgd_lrs: tuple[float, ...]
    adam_lrs: tuple[float, ...]
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]  # adversary seeds at W0; seed 0 is every ratio's reference
    shards: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        tasks = [t for s in self.shards for t in s]
        assert sorted(tasks) == sorted(TASKS) and all(self.shards), "every task in exactly one shard"
        assert self.checkpoints == (10, 30, 100, 300) and self.steps == max(self.checkpoints), "E10's schedule"
        assert 0 in self.pile_seeds and len(self.pile_seeds) >= 2 and ADAM_REF_LR not in self.adam_lrs


CFG = Config(steps=300, checkpoints=(10, 30, 100, 300), sgd_lrs=(0.003, 0.01, 0.03, 0.1),
             adam_lrs=(ADAM_REF_LR / 3, ADAM_REF_LR / 10), pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2),
             shards=tuple((t,) for t in TASKS))

PREDICTIONS: dict[str, Any] = {
    "about": "sgd_kv (item 4 of reports/2026-10-21_review_sgd-finding-influence.md; plan reports/2026-10-21_plan_sgd-kv.md): "
             "component training on E10's key-to-value tasks rand1, ident32, rand32 with plain SGD (rate by E4's rule "
             "from 0.003, 0.01, 0.03, 0.1) and Adam at 3.33e-6 and 1e-6 (E10's 1e-5 / 3 and / 10), beside E10's Adam 1e-5 "
             "states (re-read) on E10's recipe (300 steps, checkpoints 10/30/100/300, each arm scored at its first "
             "checkpoint with held-out P(target) >= 0.5), scored on the Pile harness in the same job. One shard per "
             "task, each with its own W0. Filed before any run",
    "measurements": {
        "pile": "harness.measure: Pile val rows 0..255 in two 128-row blocks, one PGD adversary per block (20 steps, "
                "seed 0), stochastic; each a ratio to this shard's W0 under seed 0 (relative). Scored: stochastic and "
                "adversarial (PGD). At W0: 1 by definition (absolute PGD 0.69, stochastic 0.248; E10 read 0.692, 0.248). "
                "Blind to the trained behaviour (the marker token never occurs in Pile) and to damage the 20-step "
                "adversary does not find",
        "kl_other": "held-out KL elsewhere (all positions but the key) of the target model against W0, nats, absolute. "
                    "At W0: 0. Blind to Pile text and to the decomposition",
        "p_target": "held-out mean P(table[key]) at the key, absolute. At W0: below 1e-4 on rand1/rand32 (E10's build "
                    "gate), at most 0.002 on ident32",
        "route_split": "local_faith.route_split at the key on the held-out windows: shares d, r, i, e_gate, e_delta "
                       "(absolute, sum 1); reported, not scored. Undefined at W0. Blind to anything off the key",
        "noise_floor": "f = W0's max/min - 1 over adversary seeds 0, 1, 2 in the same shard, per reading (stochastic_floor "
                       "for the stochastic ratio, pgd_floor for PGD; PGD 0.209 on the L40). commute.faith_score.verdict: "
                       "pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f), inconclusive between (PGD about 1.24 to 1.81 on the "
                       "L40). One batch-order seed per state",
    },
    "rate_rule": "E4's, per task: each SGD rate scored at its first checkpoint with held-out P >= 0.5; among rates that "
                 "reach it with KL elsewhere <= 2 x the regenerated fine-tune's at its scored step (E10: rand32 0.696 -> "
                 "1.391, ident32 0.189 -> 0.378, rand1 0.040 -> 0.080), the least KL elsewhere; if none, the highest P, "
                 "'did not learn'. Adam arms are fixed rates. A new arm 'learned' if it meets both conditions",
    "arm_verdict": "learned and stochastic and PGD both 'pass' -> pass; either 'fail' -> fail; else inconclusive; not "
                   "learned -> 'did not learn' (counts as neither)",
    "gates": {
        "G0_w0": "first in every shard: W0's seed-0 PGD KL within 10% of 0.69, CI-masked KL within 2% of 0.348, PGD floor "
                 "<= 0.25 (faith_score.w0_check). Fails: the shard stops",
        "G1_references_reproduce": "before training: the regenerated fine-tune reaches its scored step at E10's, and it and "
                                   "the E10 Adam 1e-5 state read E10's held-out P and KL elsewhere at that step to 0.01. "
                                   "Fails: the shard stops",
        "C1_control": "after training, before any new arm is measured: E10's Adam 1e-5 state's PGD ratio within 25% of "
                      "E10's. On rand32 (2.81) this makes it read 'fail' again: the harness's positive control. Fails: "
                      "the shard stops",
    },
    "expected": {
        "W0": "PGD KL 0.69, floors 0.05 to 0.21",
        "finetune_E10": "scored step / P / KL elsewhere: rand32 10 / 0.585 / 0.696; ident32 10 / 0.960 / 0.189; rand1 10 / "
                        "0.998 / 0.040 (reproduced under G1)",
        "adam_1e-5_E10": "scored step / P / KL elsewhere / PGD / stochastic: rand32 30 / 0.847 / 0.355 / 2.81 / 1.24 (fail); "
                         "ident32 30 / 0.971 / 0.069 / 1.19 / 1.01 (pass); rand1 10 / 1.000 / 0.069 / 1.20 / 1.03 (pass). "
                         "rand32 is the harness's positive control: it should read 'fail' again",
        "sgd": "learns on every task (sgd_grid: 0.01 learned emoticon by step 10); scored step 10 or 30 on rand1, 30 or "
               "100 on ident32 and rand32; KL elsewhere below Adam 1e-5's; Pile ratios on rand32 open (Q1)",
        "adam_3.33e-6": "learns by step 100 on every task; KL elsewhere below Adam 1e-5's; rand32 open (Q1)",
        "adam_1e-6": "learns by step 300 on rand1, may not learn on ident32 and rand32 (a tenth of the 1e-5 arm's step); "
                     "Pile ratios near 1",
    },
    "predictions": [
        {"id": "Q1_rand32_passes", "kind": "two-sided question, no direction filed",
         "rule": "on rand32, some new arm (selected SGD, Adam 3.33e-6, Adam 1e-6) learns and reads 'pass'",
         "decision": "yes -> E10's 'new capability' failure was an Adam/rate artefact: the frozen-Gamma failure on a "
                     "behaviour the model lacks is withdrawn as evidence; only 'inconclusive' -> those states rerun with "
                     "more adversary seeds or on an H100 before deciding; no (every learned new arm fails) -> the failure "
                     "holds across optimisers and rates on this task"},
        {"id": "P2_sgd_learns", "rule": "the rate rule finds an SGD rate on every task", "predicted": "pass",
         "why": "sgd_grid: 0.01 to 0.1 learn emoticon by step 10. Scope: a fail is reported with the grid's range"},
        {"id": "P3_sgd_less_damage", "rule": "on every task where SGD learned, its KL elsewhere at its scored step < the "
                                             "re-read Adam 1e-5 state's at its scored step",
         "predicted": "pass", "why": "emoticon at step 10: 0.011 against 0.016; the update is smaller (not matched)"},
    ],
    "reported_not_scored": ["route-split shares of every state", "CI-masked and unmasked Pile ratios",
                            "every SGD rate's P and KL elsewhere at every checkpoint", "largest relative change per matrix"],
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def e10_reference(task: str) -> dict[str, Any]:
    """E10's recorded numbers for the task: the fine-tune and the Adam 1e-5 state at their scored steps."""
    r = json.loads((E10 / "kv_task.json").read_text())["tasks"][task]
    ft, pb = r["finetune"], r["pullback"]
    c = pb["curve"][pb["scored_step"] - 1]
    return {"finetune": {"step": ft["scored_step"], **{k: ft["evals"][str(ft["scored_step"])][k] for k in ("p_o", "kl_other")}},
            ADAM_REF: {"step": pb["scored_step"], "p_o": c["heldout_p_o"], "kl_other": c["heldout_kl_other"],
                       "pile": pb["eval"]["pile"]["ratios"]}}


# ----------------------------------------------------------------------------- per task


def task_context(ctx: Any, task: str) -> Any:
    """The shared context with the task's windows, lookup table and W0 route-split cells."""
    from commute.local_faith import route_cells

    w = torch.load(E10 / "large" / f"windows_{task}.pt", weights_only=False)
    held, table = w["heldout"], w["table"]
    return replace(ctx, train=w["train"], held=held, token=table, rbase=route_cells(ctx.model, held.tokens, held.half, table))


def a_cfg(tctx: Any) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=tctx.token, lr=FT_LR, steps=CFG.steps, checkpoints=CFG.checkpoints, seed=0)


def train_arm(tctx: Any, optimizer: str, lr: float) -> tuple[dict[str, Any], Any]:
    """Component training to CFG.steps; held-out numbers at every checkpoint and the state at the scored step."""
    from commute import arms, component_training

    cfg = component_training.TrainConfig(lr=lr, steps=CFG.steps, save_steps=CFG.checkpoints, optimizer=optimizer)
    curve, states = component_training.train(tctx.model, tctx.uv, tctx.train.tokens, tctx.held, a_cfg(tctx), cfg,
                                             tctx.device, tctx.W0, log=log)
    at = {s: {"p_o": curve[s - 1]["heldout_p_o"], "kl_other": curve[s - 1]["heldout_kl_other"],
              "rel_norm_max": curve[s - 1]["rel_norm_max"]} for s in CFG.checkpoints}
    step, reached = arms.first_learned({s: a["p_o"] for s, a in at.items()}, P_LEARN)
    return {"checkpoints": at, "step": step, "reached": reached, **at[step]}, states[step]


def adam_reference(tctx: Any, task: str, step: int) -> tuple[Any, str]:
    """E10's Adam 1e-5 state at its scored step, from NFS (present on 2026-10-21; a missing file stops the shard)."""
    p = E10 / "large" / task / "pullback" / f"state_step{step}.pt"
    if not p.is_file():
        sys.exit(f"[{task}] {p} absent: E10's state is the control and is not regenerated")
    return torch.load(p, weights_only=False), str(p.relative_to(HERE.parent.parent))


def g1(tctx: Any, task: str) -> tuple[dict[str, Any], Any]:
    """The fine-tune (regenerated) and E10's Adam 1e-5 state against E10's held-out numbers; returns the record and the
    Adam state."""
    from commute import arms
    from commute.finetune import finetune_arm_a

    ref = e10_reference(task)
    ft = finetune_arm_a(tctx.model, tctx.train, tctx.held, a_cfg(tctx), tctx.device, log)
    step, _ = arms.first_learned({s: e["p_o"] for s, e in ft.evals.items()}, P_LEARN)
    ft.deltas.clear()
    st, source = adam_reference(tctx, task, ref[ADAM_REF]["step"])
    ho = {"finetune": {k: ft.evals[ref["finetune"]["step"]][k] for k in ("p_o", "kl_other")},
          ADAM_REF: arms.heldout_of(tctx.model, tctx.W0, tctx.held, tctx.token, tctx.device, {k: v[0] for k, v in st.items()})}
    rec: dict[str, Any] = {a: {"heldout": ho[a], "step": ref[a]["step"], "dP": ho[a]["p_o"] - ref[a]["p_o"],
                               "dkl": ho[a]["kl_other"] - ref[a]["kl_other"]} for a in ho}
    rec["finetune"]["scored_step_here"] = step
    rec["adam_source"] = source
    rec["pass"] = step == ref["finetune"]["step"] and all(
        abs(rec[a]["dP"]) <= REPRO_TOL and abs(rec[a]["dkl"]) <= REPRO_TOL for a in ho)
    log(f"[{task} G1] {json.dumps(rec, default=float)}")
    return rec, st


def train_all(tctx: Any, task: str, limit: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """SGD over the rate grid (one state kept by E4's rule) and Adam at each low rate; kept states saved."""
    from commute import arms

    sgd, sgd_states = {}, {}
    for lr in CFG.sgd_lrs:
        sgd[f"{lr:g}"], sgd_states[f"{lr:g}"] = train_arm(tctx, "sgd", lr)
    lr, learned = arms.select_rate(sgd, P_LEARN, limit)  # an arm short of P_LEARN at step 300 never qualifies
    log(f"[{task} SGD grid] {json.dumps({n: {k: a[k] for k in ('step', 'p_o', 'kl_other')} for n, a in sgd.items()}, default=float)}"
        f" -> lr {lr} learned {learned} (limit {limit:.4f})")
    kept, last = {f"sgd_{lr}": sgd_states[lr]}, {f"sgd_{lr}": sgd[lr]}
    del sgd_states
    for a in CFG.adam_lrs:
        name = f"adam_{a:.3g}"
        last[name], kept[name] = train_arm(tctx, "adam", a)
        log(f"[{task} {name}] {json.dumps({k: last[name][k] for k in ('step', 'p_o', 'kl_other')}, default=float)}")
    for name, st in kept.items():
        arms.save_offsets(LARGE / task / f"{name}.pt", tctx.uv_cpu, st, {"task": task, "arm": name, "step": last[name]["step"]})
    rec = {"sgd_grid": sgd, "sgd_selected": lr, "sgd_learned": learned, "arms": last}
    return rec, {n: ({k: v[0] for k, v in st.items()}, st) for n, st in kept.items()}


def measured(tctx: Any, name: str, dW: dict[str, Any], state: Any, ho: dict[str, Any]) -> dict[str, Any]:
    from commute import harness

    m = harness.measure(tctx, name, dW, state, True, log)
    return {"heldout": {k: ho[k] for k in ("p_o", "kl_other")}, "step": ho.get("step"), "pile": m["pile"],
            "routes": m["routes"], "rel_norm_max": m["rel_norm_max"]}


# ----------------------------------------------------------------------------- the score (rules: results/predictions.json)


def arm_verdict(a: dict[str, Any], limit: float, floors: dict[str, float]) -> dict[str, Any]:
    from commute import faith_score

    learned = a["heldout"]["p_o"] >= P_LEARN and a["heldout"]["kl_other"] <= limit
    pile = {k: faith_score.verdict(a["pile"][k], floors[k]) for k in SCORED}
    overall = "fail" if "fail" in pile.values() else "inconclusive" if "inconclusive" in pile.values() else "pass"
    return {"learned": learned, "pile": pile, "verdict": overall if learned else "did not learn"}


def score_task(rec: dict[str, Any], floors: dict[str, float]) -> dict[str, Any]:
    v = {name: arm_verdict(a, rec["limit"], floors) for name, a in rec["arms"].items()}
    new = [x["verdict"] for n, x in v.items() if n != ADAM_REF]
    sgd = rec["train"]
    sgd_kl = sgd["arms"][f"sgd_{sgd['sgd_selected']}"]["kl_other"]
    return {"verdicts": v, "any_new_arm_passes": "pass" in new, "any_new_arm_inconclusive": "inconclusive" in new,
            "P2_sgd_learned": sgd["sgd_learned"],
            "P3_sgd_less_damage": (sgd_kl < rec["arms"][ADAM_REF]["heldout"]["kl_other"]) if sgd["sgd_learned"] else None}


# ----------------------------------------------------------------------------- main


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(path, out)
    sys.exit(why)


def run_task(tctx: Any, task: str, out: dict[str, Any], path: Path) -> None:
    """G1, training, the control's Pile reading, then the new arms; `out` dumped after each stage."""
    rec: dict[str, Any] = {"task": task, "arms": {}, "e10": e10_reference(task)}
    out["tasks"][task] = rec
    rec["G1"], ref_state = g1(tctx, task)
    rec["limit"] = KL_FACTOR * rec["G1"]["finetune"]["heldout"]["kl_other"]
    dump(path, out)
    if not rec["G1"]["pass"]:
        stop(path, out, f"{task}: the references do not reproduce E10's held-out numbers")
    rec["train"], new = train_all(tctx, task, rec["limit"])
    dump(path, out)
    ref_ho = {**rec["G1"][ADAM_REF]["heldout"], "step": rec["G1"][ADAM_REF]["step"]}
    rec["arms"][ADAM_REF] = measured(tctx, f"{task}/{ADAM_REF}", {k: v[0] for k, v in ref_state.items()}, ref_state, ref_ho)
    del ref_state
    pgd_rel = rec["arms"][ADAM_REF]["pile"]["pgd_kl"] / rec["e10"][ADAM_REF]["pile"]["pgd_kl"] - 1
    rec["control"] = {"adam_pgd_rel": pgd_rel, "pass": abs(pgd_rel) <= PGD_REPRO_TOL}
    log(f"[{task} control] Adam 1e-5 PGD ratio {rec['arms'][ADAM_REF]['pile']['pgd_kl']:.3f}, {pgd_rel:+.3f} relative to "
        f"E10 -> {'PASS' if rec['control']['pass'] else 'FAIL'}")
    dump(path, out)
    if not rec["control"]["pass"]:
        stop(path, out, f"{task}: Adam 1e-5's adversarial ratio does not reproduce E10's; new arms not measured")
    for name, (dW, st) in new.items():
        rec["arms"][name] = measured(tctx, f"{task}/{name}", dW, st, rec["train"]["arms"][name])
        dump(path, out)
    rec["score"] = score_task(rec, {"stochastic_kl": out["W0"]["stochastic_floor"], "pgd_kl": out["W0"]["pgd_floor"]})
    log(f"[{task} score] {json.dumps(rec['score'], default=float)}")


def run_shard(shard: int, device: str) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    path = RESULTS / f"sgd_kv_shard{shard}.json"
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
