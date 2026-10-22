"""ladder_second_seed (E26): do E21's ladder verdicts (Adam 1e-5 passes at steps 100 and 1000 on r0 and r1, SGD fails
at 1000) hold under a second batch-order seed? Item A of reports/2026-10-22_plan_commute-second-line.md.

E21's recipe (experiments/sgd_ladder_long: arm A's loss, batch 32, clip 1, Gamma and Delta frozen; free offsets start
at zero, so ArmAConfig.seed, the batch order, is the only training randomness), two arms per task: SGD at 3a's rate and
Adam 1e-5, 1000 steps. Tasks r0_t352_v310, r1_t534_v369, r2_t275_v2667, in that order. Stages, cheapest first, results
dumped after each state (the scored run is its own smoke):
  W0     the Pile harness under adversary seeds 0, 1, 2, once per process; stop unless faith_score.w0_check passes
  G1     per task, the seed-0 SGD and Adam arms trained to 1000 and scored at 1000: PGD ratio within 25% of E21's and
         the same verdict, or that task's predictions are unscored (the run continues; on r2 reported, not gated)
  seed   the same two arms under batch-order seed --seed (1), scored at 100 and 1000; states saved as bf16 offsets

Usage (repo root, project venv, GPU; results/predictions.json filed first, for the same --seed):
  run.py [--task T ...] [--seed 1]  tasks in order -> results/ladder_second_seed_<T>.json; a finished task is skipped
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
E21 = HERE.parent / "sgd_ladder_long" / "results"  # sgd_ladder_long_<task>.json: seed 0's readings, G1's reference

TASKS = ("r0_t352_v310", "r1_t534_v369", "r2_t275_v2667")
GATED = TASKS[:2]  # r2: G1 reported, not gated (nothing passes cleanly there at seed 0)
# 3a's selected SGD rate per task (experiments/sgd_ladder/results/sgd_ladder_shard*.json, read by sgd_ladder_long's
# ref_3a); checked against E21's recorded ref_3a.sgd_lr at the start of each task.
SGD_LR = {"r0_t352_v310": 0.03, "r1_t534_v369": 0.03, "r2_t275_v2667": 0.1}
ADAM_LR, FT_LR = 1e-5, 3e-5  # FT_LR only fills ArmAConfig.lr as E21 did; component training reads TrainConfig.lr
TOL = 1e-6  # split.bases tolerance (harness.load computes the bases; unused here)
P_MIN, GATE_TOL = 0.9, 0.25  # held-out P(target) line; G1's relative tolerance on the PGD ratio


@dataclass(frozen=True)
class Config:
    seed: int  # batch-order seed of the arms under test; seed 0 is E21's and G1's
    steps: tuple[int, ...] = (100, 1000)  # saved and scored steps of the seed arms; G1 reads the last
    pile_rows: int = 256
    pile_pgd_steps: int = 20
    pile_seeds: tuple[int, ...] = (0, 1, 2)  # adversary seeds at W0; seed 0 is every ratio's reference

    def __post_init__(self) -> None:
        assert self.seed != 0, "seed 0 is E21's; it runs as the gate"
        assert self.steps == (100, 1000), self.steps
        assert (self.pile_rows, self.pile_pgd_steps) == (256, 20), "E21's harness sizes, or G1 compares unlike readings"
        assert self.pile_seeds[0] == 0 and len(self.pile_seeds) >= 2

    @property
    def train_steps(self) -> int:
        return self.steps[-1]


CFG = Config(seed=1)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def arms_of(task: str) -> tuple[tuple[str, str, float], ...]:
    """(name, optimizer, lr) per arm, SGD first; names as E21's."""
    return (f"sgd_{SGD_LR[task]:g}", "sgd", SGD_LR[task]), (f"adam_{ADAM_LR:g}", "adam", ADAM_LR)


def train_arm(tctx: Any, task: str, arm: tuple[str, str, float], seed: int,
              save: bool) -> tuple[dict[int, Any], dict[int, Any]]:
    """One arm to step 1000 under batch-order `seed`: held-out readings and (dW, state) at each of CFG.steps; with
    `save`, the states as bf16 offsets in results/large/<task>/."""
    from commute import arms, component_training
    from commute.finetune import ArmAConfig

    name, optimizer, lr = arm
    a_cfg = ArmAConfig(o_token=tctx.token, lr=FT_LR, steps=CFG.train_steps, checkpoints=CFG.steps, seed=seed)
    cfg = component_training.TrainConfig(lr=lr, steps=CFG.train_steps, save_steps=CFG.steps, optimizer=optimizer)
    t0 = time.time()
    curve, states = component_training.train(tctx.model, tctx.uv, tctx.train.tokens, tctx.held, a_cfg, cfg,
                                             tctx.device, tctx.W0, log=log)
    ho = {s: {"p_o": curve[s - 1]["heldout_p_o"], "kl_other": curve[s - 1]["heldout_kl_other"],
              "rel_norm_max": curve[s - 1]["rel_norm_max"]} for s in CFG.steps}
    log(f"[{task} train {name} seed {seed}] {time.time() - t0:.0f} s; {json.dumps(ho, default=float)}")
    if save:
        for s, st in states.items():
            arms.save_offsets(LARGE / task / f"{name}_seed{seed}_step{s}.pt", tctx.uv_cpu, st,
                              {"task": task, "arm": name, "optimizer": optimizer, "lr": lr, "step": s, "seed": seed})
    return ho, {s: ({k: v[0] for k, v in st.items()}, st) for s, st in states.items()}


def measured(tctx: Any, label: str, dw_state: tuple[Any, Any], ho: dict[str, float], w0: dict[str, Any],
             limit: float) -> dict[str, Any]:
    """A component-training state through harness.measure, scored by faith_score.state_verdict."""
    from commute import faith_score, harness

    m = harness.measure(tctx, label, dw_state[0], dw_state[1], True, log)
    score = {**faith_score.state_verdict(m["pile"], ho["p_o"], w0, P_MIN), "kl_within_limit": ho["kl_other"] <= limit}
    rec = {"heldout": ho, "pile": m["pile"], "routes": m["routes"], "trigger": faith_score.trigger_ratios(m),
           "rel_norm_max": m["rel_norm_max"], "score": score}
    log(f"[{label}] pgd {m['pile']['pgd_kl']:.3f} stochastic {m['pile']['stochastic_kl']:.3f} P(target) "
        f"{ho['p_o']:.3f} kl_other {ho['kl_other']:.3f} rel_norm_max {m['rel_norm_max']:.4f} -> {score['verdict']}")
    return rec


def gate(m: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """G1 for one seed-0 arm at step 1000 against E21's record of the same state."""
    rel = m["pile"]["pgd_kl"] / ref["pile"]["pgd_kl"] - 1
    v, v21 = m["score"]["verdict"], ref["score"]["verdict"]
    return {"pgd": m["pile"]["pgd_kl"], "e21_pgd": ref["pile"]["pgd_kl"], "rel": rel, "verdict": v, "e21_verdict": v21,
            "heldout_diff": {k: m["heldout"][k] - ref["heldout"][k] for k in ("p_o", "kl_other")},
            "pass": abs(rel) <= GATE_TOL and v == v21}


def run_arm(tctx: Any, out: dict[str, Any], path: Path, arm: tuple[str, str, float], seed: int,
            steps: tuple[int, ...]) -> None:
    """Train one arm under `seed` and score it at `steps`, dumping after each state."""
    task, name = out["task"], arm[0]
    ho, states = train_arm(tctx, task, arm, seed, save=seed != 0)
    out["heldout"].setdefault(f"seed{seed}", {})[name] = ho
    rec = out["states"].setdefault(f"seed{seed}", {}).setdefault(name, {})
    for s in steps:
        rec[str(s)] = measured(tctx, f"{task}/{name}/seed{seed}@{s}", states.pop(s), ho[s], out["W0"], out["limit"])
        dump(path, out)


def score_task(out: dict[str, Any]) -> dict[str, Any]:
    seed = f"seed{CFG.seed}"
    v = {f"{n}@{s}": r["score"]["verdict"] for n, st in out["states"][seed].items() for s, r in st.items()}
    gated = out["task"] in GATED
    return {"verdicts": v, "G1_pass": out["G1"]["pass"], "G1_gated": gated, "scored": out["G1"]["pass"] or not gated}


def run_task(ctx: Any, w0: dict[str, Any], task: str, path: Path) -> None:
    from commute import harness

    t0 = time.time()
    e21 = json.loads((E21 / f"sgd_ladder_long_{task}.json").read_text())
    assert e21["ref_3a"]["sgd_lr"] == SGD_LR[task], (task, e21["ref_3a"]["sgd_lr"])
    w = torch.load(LADDER / f"windows_{task}.pt", weights_only=False)
    tctx = harness.with_windows(ctx, w, w["task"]["target"])
    out: dict[str, Any] = {"task": task, "seed": CFG.seed, "config": asdict(CFG), "W0": w0, "limit": e21["ref_3a"]["limit"],
                           "sgd_lr": SGD_LR[task], "device": torch.cuda.get_device_name(), "G1": {}, "heldout": {},
                           "states": {}}
    dump(path, out)
    for arm in arms_of(task):
        run_arm(tctx, out, path, arm, 0, CFG.steps[-1:])
        out["G1"][arm[0]] = gate(out["states"]["seed0"][arm[0]]["1000"], e21["states"][arm[0]]["1000"])
        log(f"[{task} G1 {arm[0]}] {json.dumps(out['G1'][arm[0]], default=float)}")
        dump(path, out)
    out["G1"]["pass"] = all(out["G1"][a[0]]["pass"] for a in arms_of(task))
    log(f"[{task} G1] {'PASS' if out['G1']['pass'] else 'FAIL'}" + ("" if task in GATED else " (reported, not gated)"))
    for arm in arms_of(task):
        run_arm(tctx, out, path, arm, CFG.seed, CFG.steps)
    out["score"] = score_task(out)
    out["seconds"] = time.time() - t0
    dump(path, out)
    log(f"[{task} score] {json.dumps(out['score'])}; wrote {path} ({out['seconds']:.0f} s)")


def run(tasks: list[str], device: str) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())["config"]["seed"]
    assert CFG.seed == filed, f"predictions were filed for seed {filed}, not {CFG.seed}"
    paths = {t: RESULTS / f"ladder_second_seed_{t}.json" for t in tasks}
    todo = [t for t in tasks if not (paths[t].is_file() and "seconds" in json.loads(paths[t].read_text()))]
    log(f"tasks to run: {todo} (finished ones skipped: {sorted(set(tasks) - set(todo))})")
    if not todo:
        return
    t0 = time.time()
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, CFG.pile_rows,
                       CFG.pile_pgd_steps, CFG.pile_seeds, log)
    w0 = faith_score.w0_check(ctx.pile)
    log(f"[W0] {time.time() - t0:.0f} s; {json.dumps({k: v for k, v in w0.items() if k != 'pile'}, default=float)}")
    if not w0["pass"]:
        for t in todo:
            dump(paths[t], {"task": t, "W0": w0, "stopped": "W0 is not at its known values: nothing run"})
        sys.exit("W0 is not at its known values: nothing run")
    for t in todo:
        run_task(ctx, w0, t, paths[t])


def main() -> None:
    global CFG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--seed", type=int, default=1, help="batch-order seed of the arms under test (ArmAConfig.seed)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    CFG = Config(seed=args.seed)
    run([t for t in TASKS if t in args.task], args.device)


if __name__ == "__main__":
    main()
