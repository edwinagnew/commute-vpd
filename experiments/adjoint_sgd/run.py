"""adjoint_sgd (E31): does the adjoint split of a plain-SGD fine-tune keep behaviour, the floor and Delta together?
Plan: reports/2026-10-22_plan_adjoint-sgd.md (design check, mechanism hypothesis, P1-P4) and its Agent notes.

Arm-A fine-tunes of all 24 matrices to 1000 steps. At steps 10, 100 and 1000 each update is read four ways: the
fine-tune's own held-out P(target) and size; the reproduced share R (split.reproduced_share: how much of delta-W the
scaled adjoint split's weight change reproduces on the data); the do-nothing state and the scaled adjoint state
(split.adjoint_state) through harness.measure (Pile ratios to the same-job W0, route split, Delta's share). Order, each
state written to results/adjoint_sgd.json as it finishes:
  1. W0 under the Pile harness, adversary seeds 0 and 1 (faith_score.w0_check); stop if it fails.
  2. Reproductions, emoticon seed 0: Adam 3e-5, SGD 0.01, Adam 3e-6 (E22's adjoint P(target) within 0.05; the large
     one's R within 0.03 of E14's); stop if one fails.
  3. Emoticon seed 1: SGD 0.01, Adam 3e-6, Adam 3e-5.
  4. r1_t534_v369, then r0_t352_v310: the five-rate SGD grid at step 10 with E4's rule picks the rate, then that SGD
     fine-tune and Adam 3e-5. Sigma (for c and R) is the task's own training windows' input second moments.
  5. Budget sweep (E20 amendment 3's recipe) on W0 seeds 0 and 1 and the step-1000 adjoint states of SGD 0.01 and
     Adam 3e-6, emoticon seed 0.

Usage (repo root):  run.py --predictions  (files results/predictions.json once)  |  run.py all  (one job, H100)
  run.py shard X   one of the cluster shards below -> results/adjoint_sgd_X.json, results/large/full_X.log; each
                   runs the same-job W0 floor first. A (the reproductions, gated) runs first; B-E refuse to start
                   unless A's file shows W0 and the Adam 3e-5 and SGD 0.01 reproductions passed
  run.py merge     the shard files -> results/adjoint_sgd.json (JSON only, no model; each row keeps its shard's floor)
Shards: A emoticon seed 0 (Adam 3e-5, SGD 0.01, Adam 3e-6; G1, G2) | B emoticon seed 1 | C r1_t534_v369 | D r0_t352_v310
| E budget sweep: SGD 0.01 and Adam 3e-6 emoticon seed 0 retrained to step 1000 in-job (A's recipe; their step-1000
fine-tune P and KL are recorded beside A's), then the sweep.
Inputs (gitignored): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "adjoint_sgd.json"
SHARDS = ("A", "B", "C", "D", "E")
LOG = LARGE / "full.log"  # main() points it at full_<shard>.log for a shard
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"

STEPS, TRAIN_STEPS, TOL = (10, 100, 1000), 1000, 1e-6
PILE_ROWS, PILE_PGD_STEPS, PILE_SEEDS = 256, 20, (0, 1)
EMO, R1, R0 = "emoticon", "r1_t534_v369", "r0_t352_v310"
SGD, SMALL, BIG = ("sgd", 0.01), ("adam", 3e-6), ("adam", 3e-5)
SGD_LRS, P_MIN = (0.001, 0.003, 0.01, 0.03, 0.1), 0.9  # sgd_finetune's grid; E4's rule
KL_LIMIT = {R1: 0.24301687240600586, R0: 0.20851298211847694}  # 2 x Adam 3e-5's step-10 KL elsewhere (sgd_ladder.json)
# E22 (small_finetune_splits.json) adjoint P(target) at STEPS, emoticon seed 0; E14 (lora_split.json) R of Adam 3e-5.
E22_P = {BIG: (0.9333, 0.5997, 0.1038), SGD: (0.7144, 0.7535, 0.6132), SMALL: (0.6936, 0.4667, 0.2378)}
E14_R = (0.5763, 0.4222, 0.3064)
P_TOL, R_TOL = 0.05, 0.03
DELTA_MAX, KEEP, GAP, R_MIN, RHO_MIN = 1e-3, 0.5, 0.2, 0.8, 0.7
BUDGETS, BUDGET_ROWS, BUDGET_BATCH = (20, 40, 80, 160), 128, 8  # E20 amendment 3: n sign steps of 2 / n, seed 0
W0_BUDGET20, W0_BUDGET_TOL, BUDGET_PASS = 0.6604, 0.05, 1.241  # E20 A3's W0 at 20 steps (L40); P10's line
assert STEPS[-1] == TRAIN_STEPS and len(E14_R) == len(STEPS)


def label(task: str, seed: int, opt: str, lr: float) -> str:
    return f"{task}/s{seed}/{opt}_{lr:g}"


VERDICT = ("faith_score.verdict on the adversarial (PGD) ratio to the same-job W0 under adversary seed 0: pass <= 1.5 / "
           "(1 + f), fail >= 1.5 (1 + f); f = max/min - 1 of W0's PGD KL over seeds 0, 1 (H100 about 0.075 -> 1.40 / 1.61)")
PREDICTIONS: dict[str, Any] = {
    "about": "E31 adjoint_sgd. Plan reports/2026-10-22_plan_adjoint-sgd.md (design check there; the user asked for this "
             "experiment 2026-10-22). Filed before the run; never edited after.",
    "setup": {
        "finetunes": f"arm A (CE at the trigger + KL elsewhere), batch 32, clip 1, all 24 matrices, {TRAIN_STEPS} steps, "
                     f"delta-W at {STEPS}. Emoticon seed 0: Adam 3e-5, SGD 0.01, Adam 3e-6; emoticon seed 1: the same three; "
                     f"{R1} and {R0} seed 0: SGD at the rate E4's rule picks from {SGD_LRS} at step 10 (P >= {P_MIN}, KL "
                     f"elsewhere <= {KL_LIMIT}), and Adam 3e-5",
        "states": "per checkpoint: do-nothing and the scaled adjoint split (split.adjoint_state) through harness.measure: "
                  f"Pile ratios ({PILE_ROWS} rows, {PILE_PGD_STEPS} PGD steps), held-out P(target) at the trigger (route "
                  "split T1), Delta's share of ||delta-W||^2",
        "reproduced_share": "R = 1 - sum_k ||(c_k K(dW_k) - dW_k) Sigma_k^1/2||^2 / sum_k ||dW_k Sigma_k^1/2||^2 over the "
                            "non-zero matrices (split.reproduced_share), K(M) = U U^T M + M V V^T, c_k adjoint_split's scale; "
                            "Sigma = W0's input second moments on the task's training windows (also the scale c's)",
        "keep_ratio": "adjoint state's held-out P(target) / the fine-tune's own held-out P(target), same checkpoint",
        "budget": f"gamma.pgd_sources, n in {BUDGETS} sign steps of 2/n from the seed's random start, first {BUDGET_ROWS} "
                  f"Pile val rows in batches of {BUDGET_BATCH}, fp32 KL (cancel.masked_kl); ratio = state KL (seed 0) / "
                  "W0 KL (seed 0) at the same n",
        "verdict": VERDICT,
    },
    "gates": {
        "G0_w0": "first, alone: faith_score.w0_check. Fails: stop, nothing scored",
        "G1_reproduce_E22": f"emoticon seed 0 Adam 3e-5, SGD 0.01, Adam 3e-6: adjoint P(target) within {P_TOL} of E22's "
                            f"{ {f'{o} {lr:g}': v for (o, lr), v in E22_P.items()} } at every step. Fails: stop",
        "G2_R_reproduces_E14": f"emoticon seed 0 Adam 3e-5: R within {R_TOL} of E14's {E14_R}. Fails: P3 unscored "
                               "(positive control of the new measurement), the run goes on",
        "C_delta": f"every adjoint state's Delta share <= {DELTA_MAX} (positive control that the measurement reads 0)",
        "C_budget_w0": f"W0 seed 0 at 20 budget steps within {W0_BUDGET_TOL:.0%} of E20 A3's {W0_BUDGET20} (L40). Fails: "
                       "P4 reported, unscored",
    },
    "predictions": [
        {"id": "P1_seed", "measure": "emoticon seed 1, step 1000: SGD 0.01's adjoint state; Adam 3e-6's adjoint state",
         "pass": f"SGD: P(target) >= {KEEP}, Delta share <= {DELTA_MAX}, verdict 'pass'; and Adam 3e-6's P(target) <= "
                 f"SGD's - {GAP}", "relative_vs_absolute": "absolute P(target) and Delta share; ratio relative to W0",
         "predicted": "pass: E22 seed 0 read SGD 0.61 and Adam 3e-6 0.24 at step 1000, both passing with Delta 0",
         "expected": "W0: P(target) 0.004, ratio 1. Fine-tunes about 1.0. SGD adjoint 0.6-0.75, Adam 3e-6 adjoint about 0.25",
         "blind_to": "damage the 20-step adversary does not find; off-target damage of the adjoint state; P only at the "
                     "trigger", "positive_control": "C_delta; G1", "noise_floor": "P over the held-out windows "
                     "(deterministic); one more training seed"},
        {"id": "P2_tasks", "measure": f"{R1} and {R0}, step 1000: the SGD fine-tune's adjoint state vs Adam 3e-5's",
         "pass": f"per task: SGD's adjoint P(target) >= {KEEP} and >= Adam 3e-5's + {GAP}, Delta share <= {DELTA_MAX}, "
                 "verdict 'pass'. Reading: 'both' (pass), 'one task' (two-sided, reported which), 'neither' (fail). A task "
                 "whose grid passes no rate is 'unscored' for that task",
         "relative_vs_absolute": "absolute P(target) and difference",
         "predicted": "pass on both, medium-low confidence: the mechanism is optimiser-level, not task-level",
         "expected": "Adam 3e-5 adjoint about 0.1 at step 1000 (E14/E16 on emoticon); SGD rate 0.03 on r1 (E24)",
         "blind_to": "as P1; tasks from one family (single-token trigger and target)",
         "positive_control": "the Adam 3e-5 adjoint state's low P(target) on each task", "noise_floor": "as P1"},
        {"id": "P3_mechanism", "measure": "R of every fine-tune at every step; Spearman of keep_ratio on R over every "
                                          "adjoint state (n reported)",
         "pass": f"(a) every SGD fine-tune's R >= {R_MIN} at every step and above every Adam fine-tune's R on the same "
                 f"task and seed at the same step; (b) Spearman >= {RHO_MIN}. Both parts reported separately",
         "relative_vs_absolute": "absolute R; rank correlation",
         "predicted": "pass: one plain SGD step's adjoint split is one component-training step, and E22's outside-both "
                      "share for SGD is 0.03-0.06; Adam's per-coordinate rescaling is not a sum of gradients (E14 R 0.58 -> "
                      "0.42 -> 0.31)",
         "expected": "Adam 3e-5 emoticon R 0.58 / 0.42 / 0.31 (G2); SGD high; a random update's R is low (E14 did not "
                     "read it; not run here)",
         "blind_to": "cause: a correlation across optimisers and tasks, not an intervention on R",
         "positive_control": "G2", "noise_floor": "R deterministic given delta-W; Spearman over about 30 states that are "
                                                  "not independent (three steps per fine-tune)"},
        {"id": "P4_budget", "measure": "budget ratio of the SGD 0.01 emoticon seed-0 adjoint state at step 1000, n = 40, "
                                       "80, 160",
         "pass": f"<= {BUDGET_PASS} at all three (E20 A3's P10 line); two-sided: a rise above it is reported as the 20-step "
                 "verdict not holding at larger budgets",
         "relative_vs_absolute": "ratio to W0 seed 0 at the same budget",
         "predicted": "pass: E20 A3 read Adam 3e-5's adjoint state at 1.04-1.05 at every budget",
         "expected": "W0 seed 0: 0.66 / 0.68 / 0.70 / 0.71 (L40), seed 1 / seed 0 up to 1.21 at 160. Adam 3e-6 reported",
         "blind_to": "other adversary families; budgets above 160", "positive_control": "W0's seed 1 at each budget",
         "noise_floor": "W0's seed-1 / seed-0 ratio at each budget"},
    ],
    "reported_not_scored": ["do-nothing verdicts", "stochastic ratios", "route split d, r", "per-matrix c and R",
                            "the rate grids", "Adam 3e-6's budget ratios", "keep_ratio per state"],
}


# ----------------------------------------------------------------------------- io


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"adjoint_sgd_{out['shard']}.json" if out.get("shard") else OUT
    path.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def save(out: dict[str, Any]) -> None:  # one job's file (a shard or `all`); merge() builds the union
    out["rows"] = rows(out)
    out["score"] = score(out)
    dump(out)


# ----------------------------------------------------------------------------- the fine-tunes and their states


class Task:
    """A task's context for harness.measure (its windows and target) and Sigma on its training windows."""

    def __init__(self, ctx: Any, name: str) -> None:
        from commute import harness, split
        from commute.data import batches

        if name == EMO:
            self.ctx, self.sigma = ctx, ctx.sigma  # harness.load read the emoticon windows
            return
        w = torch.load(LADDER / f"windows_{name}.pt", weights_only=False)
        self.ctx = harness.with_windows(ctx, w, w["task"]["target"])
        moments = split.input_second_moments(ctx.model, batches(w["train"].tokens, 32), list(ctx.W0))
        self.sigma = {k: v.cpu() for k, v in moments.items()}


def finetune(task: Task, seed: int, opt: str, lr: float, steps: int) -> Any:
    from commute.finetune import ArmAConfig, finetune_arm_a

    c = task.ctx
    ckpts = tuple(s for s in STEPS if s <= steps) or (steps,)
    cfg = ArmAConfig(o_token=c.token, lr=lr, steps=steps, checkpoints=ckpts, seed=seed, optimizer=opt)
    return finetune_arm_a(c.model, c.train, c.held, cfg, c.device, log)


def arm(out: dict[str, Any], task: Task, name: str, seed: int, opt: str, lr: float) -> dict[int, Any]:
    """One fine-tune and its three checkpoints' readings, saved after each; returns the adjoint states by step."""
    from commute import harness, split

    key, t0 = label(name, seed, opt, lr), time.time()
    res = finetune(task, seed, opt, lr, TRAIN_STEPS)
    out["arms"][key] = rec = {"task": name, "seed": seed, "opt": opt, "lr": lr, "steps": {}}
    adj: dict[int, Any] = {}
    for s in STEPS:
        dW = res.deltas.pop(s)
        R = split.reproduced_share(dW, task.ctx.uv_cpu, task.sigma)
        dW_adj, st = split.adjoint_state(dW, task.ctx.uv_cpu, task.sigma)
        rec["steps"][str(s)] = {
            "ft": res.evals[s], "R": R["pooled"],
            "per_matrix": {k: {"c": p["c"], "share": p["share"]} for k, p in R["per_matrix"].items()},
            "nothing": harness.measure(task.ctx, f"{key}@{s} nothing", dW, None, True, log),
            "adjoint": harness.measure(task.ctx, f"{key}@{s} adjoint", dW_adj, st, True, log)}
        adj[s] = (dW_adj, st)
        save(out)
        r = next(x for x in out["rows"] if x["arm"] == key and x["step"] == s)
        log(f"[{key} step {s}] ft P {r['ft_p']:.4f} KL {r['ft_kl']:.4f} size {r['size']:.5f} | R {r['R']:.4f} | adjoint P "
            f"{r['adj_p']:.4f} keep {r['keep']:.3f} pgd {r['adj_pgd']:.3f} ({r['adj_verdict']}) Delta {r['adj_delta']:.2e} | "
            f"nothing pgd {r['nothing_pgd']:.3f} ({r['nothing_verdict']}) | {time.time() - t0:.0f} s")
    return adj


def gate_repro(out: dict[str, Any], opt: str, lr: float) -> bool:
    key = label(EMO, 0, opt, lr)
    st = out["arms"][key]["steps"]
    p = [st[str(s)]["adjoint"]["routes"]["p_cells"]["T1"] for s in STEPS]
    g: dict[str, Any] = {"p": p, "ref": E22_P[(opt, lr)], "ok": all(abs(a - b) <= P_TOL for a, b in zip(p, E22_P[(opt, lr)], strict=True))}
    if (opt, lr) == BIG:
        g["R"] = [st[str(s)]["R"] for s in STEPS]
        g["R_ok"] = all(abs(a - b) <= R_TOL for a, b in zip(g["R"], E14_R, strict=True))
    out["gates"][key] = g
    log(f"[gate {key}] {json.dumps(g)}")
    return g["ok"]


def pick_rate(out: dict[str, Any], task: Task, name: str) -> float:
    """E4's rule over the SGD grid at step 10 (arms.select_rate) with the task's KL limit; the rate is used either way."""
    from commute import arms

    grid = {f"{lr:g}": finetune(task, 0, "sgd", lr, 10).evals[10] for lr in SGD_LRS}
    lr, passed = arms.select_rate(grid, P_MIN, KL_LIMIT[name])
    out["grids"][name] = {"grid": grid, "limit": KL_LIMIT[name], "selected": float(lr), "passed": passed}
    log(f"[{name} grid] {json.dumps({k: (round(v['p_o'], 4), round(v['kl_other'], 4)) for k, v in grid.items()})} -> {lr} "
        f"passed {passed}")
    return float(lr)


def at(ctx: Any, adj: Any) -> Any:
    """The model at an adjoint state (dW_adj, state); None: W0."""
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    if adj is None:
        return contextlib.nullcontext()
    dW, st = adj
    return model_state(ctx.model, target={k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0},
                       components=components_with(ctx.model, ctx.alive, {k: (v[1], v[2]) for k, v in st.items()}))


def budget(out: dict[str, Any], ctx: Any, keep: dict[str, Any]) -> None:
    from commute import cancel, gamma
    from commute.data import batches, pile_rows

    bs = batches(pile_rows("val", 0, BUDGET_ROWS, log), BUDGET_BATCH)
    out["budget"] = kl = {}
    for name, seed, adj in (("W0", 0, None), ("W0", 1, None), *((k, 0, v) for k, v in keep.items())):
        t0 = time.time()
        with at(ctx, adj):
            for n in BUDGETS:
                src, _ = gamma.pgd_sources(ctx.model, bs, n, step_size=2 / n, seed=seed)
                kl.setdefault(name, {}).setdefault(str(seed), {})[str(n)] = v = cancel.masked_kl(ctx.model, bs, src)
                save(out)
                log(f"[budget {name} seed {seed}] {n} steps of {2 / n:.4f}: KL {v:.5f} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- the run


def setup(device: str, shard: str | None) -> tuple[Any, dict[str, Any]]:
    """Load, the same-job W0 floor first (G0), saved; stops on a failed W0."""
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    t0 = time.time()
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, TOL, PILE_ROWS,
                       PILE_PGD_STEPS, PILE_SEEDS, log)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "shard": shard,
                           "W0": faith_score.w0_check(ctx.pile), "arms": {}, "gates": {}, "grids": {}, "t0": t0}
    save(out)
    log(f"[W0] {json.dumps(_finite(out['W0']))} ({time.time() - t0:.0f} s)")
    if not out["W0"]["pass"]:
        stop(out, "G0: W0 is not at its known values")
    return ctx, out


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    save(out)
    sys.exit(why)


def finish(out: dict[str, Any]) -> None:
    out["seconds"] = time.time() - out.pop("t0")
    save(out)
    log(f"done in {out['seconds']:.0f} s; score {json.dumps(_finite(out['score']), indent=1)}")


def reproductions(out: dict[str, Any], emo: Task) -> dict[str, Any]:
    """Emoticon seed 0, gated in order (G1, G2); returns the step-1000 adjoint states of SGD 0.01 and Adam 3e-6."""
    keep = {}
    for opt, lr in (BIG, SGD, SMALL):
        adj = arm(out, emo, EMO, 0, opt, lr)
        if (opt, lr) in (SGD, SMALL):
            keep[label(EMO, 0, opt, lr)] = adj[TRAIN_STEPS]
        if not gate_repro(out, opt, lr):
            stop(out, f"G1: {label(EMO, 0, opt, lr)} does not reproduce E22")
    return keep


def other_task(out: dict[str, Any], ctx: Any, name: str) -> None:
    task = Task(ctx, name)
    arm(out, task, name, 0, "sgd", pick_rate(out, task, name))
    arm(out, task, name, 0, *BIG)


def stage_all(device: str) -> None:
    ctx, out = setup(device, None)
    emo = Task(ctx, EMO)
    keep = reproductions(out, emo)
    for opt, lr in (SGD, SMALL, BIG):
        arm(out, emo, EMO, 1, opt, lr)
    for name in (R1, R0):
        other_task(out, ctx, name)
    budget(out, ctx, keep)
    finish(out)


def control_passed() -> None:
    """Shards B-E start only after shard A's W0 and its first two reproductions (Adam 3e-5, SGD 0.01) passed."""
    path = RESULTS / "adjoint_sgd_A.json"
    a = json.loads(path.read_text()) if path.is_file() else {}
    g = a.get("gates", {})
    ok = a.get("W0", {}).get("pass") and all(g.get(label(EMO, 0, *x), {}).get("ok") for x in (BIG, SGD))
    if not ok or "stopped" in a:
        sys.exit(f"shard A has not passed W0 and the Adam 3e-5 and SGD 0.01 reproductions yet ({path})")


def retrained(out: dict[str, Any], emo: Task) -> dict[str, Any]:
    """Shard E: SGD 0.01 and Adam 3e-6, emoticon seed 0, retrained with A's recipe; their step-1000 adjoint states."""
    from commute import split

    keep, out["retrained"] = {}, {}
    for opt, lr in (SGD, SMALL):
        res = finetune(emo, 0, opt, lr, TRAIN_STEPS)
        out["retrained"][label(EMO, 0, opt, lr)] = res.evals[TRAIN_STEPS]
        keep[label(EMO, 0, opt, lr)] = split.adjoint_state(res.deltas[TRAIN_STEPS], emo.ctx.uv_cpu, emo.sigma)
        log(f"[retrained {label(EMO, 0, opt, lr)}] step {TRAIN_STEPS}: {json.dumps(res.evals[TRAIN_STEPS])}")
    save(out)
    return keep


def stage_shard(device: str, shard: str) -> None:
    if shard != "A":
        control_passed()
    ctx, out = setup(device, shard)
    emo = Task(ctx, EMO)
    if shard == "A":
        reproductions(out, emo)
    elif shard == "B":
        for opt, lr in (SGD, SMALL, BIG):
            arm(out, emo, EMO, 1, opt, lr)
    elif shard in ("C", "D"):
        other_task(out, ctx, R1 if shard == "C" else R0)
    else:
        budget(out, ctx, retrained(out, emo))
    finish(out)


def merge() -> None:
    """The shard files -> adjoint_sgd.json: arms, gates, grids and budget unioned, rows each read against their own
    shard's W0 floor, the score on the union."""
    parts = {s: json.loads(p.read_text()) for s in SHARDS if (p := RESULTS / f"adjoint_sgd_{s}.json").is_file()}
    out: dict[str, Any] = {"W0": {"pass": all(p["W0"]["pass"] for p in parts.values())}, "arms": {}, "gates": {},
                           "grids": {}, "rows": [], "shards": {}}
    for s, p in parts.items():
        out["shards"][s] = {k: p.get(k) for k in ("device", "W0", "seconds", "stopped", "retrained")}
        for k in ("arms", "gates", "grids"):
            out[k] |= p[k]
        out["rows"] += p["rows"]
        if "budget" in p:
            out["budget"] = p["budget"]
    out["score"] = score(out)
    dump(out)
    print(json.dumps(_finite(out["score"]), indent=1))


# ----------------------------------------------------------------------------- rows and score


def rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat record per (fine-tune, step) measured so far: what plot.py and the report read."""
    from commute.faith_score import verdict

    f = out["W0"]["pgd_floor"]
    res = []
    for key, a in out["arms"].items():
        for s, r in a["steps"].items():
            ad, no = r["adjoint"], r["nothing"]
            p = ad["routes"]["p_cells"]["T1"]
            res.append({"arm": key, "task": a["task"], "seed": a["seed"], "opt": a["opt"], "lr": a["lr"], "step": int(s),
                        "ft_p": r["ft"]["p_o"], "ft_kl": r["ft"]["kl_other"], "size": r["ft"]["rel_norm_max"], "R": r["R"],
                        "adj_p": p, "keep": p / r["ft"]["p_o"], "adj_pgd": ad["pile"]["pgd_kl"],
                        "adj_verdict": verdict(ad["pile"]["pgd_kl"], f), "adj_stochastic": ad["pile"]["stochastic_kl"],
                        "adj_delta": ad["delta_change_frac"], "nothing_pgd": no["pile"]["pgd_kl"],
                        "nothing_verdict": verdict(no["pile"]["pgd_kl"], f), "floor": f})
    return res


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> Tensor:
        t = torch.tensor(v, dtype=torch.float64)
        return (t[:, None] > t[None, :]).sum(1) + 0.5 * ((t[:, None] == t[None, :]).sum(1) - 1)  # average ranks

    a, b = ranks(x), ranks(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def keeps(r: dict[str, Any]) -> bool:
    return r["adj_p"] >= KEEP and r["adj_delta"] <= DELTA_MAX and r["adj_verdict"] == "pass"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on whatever is measured so far; a prediction whose inputs are missing is absent."""
    R = {(r["arm"], r["step"]): r for r in out["rows"]}
    end = {k: r for (k, s), r in R.items() if s == TRAIN_STEPS}
    sc: dict[str, Any] = {"G0_w0": out["W0"]["pass"], "G1": {k: g["ok"] for k, g in out["gates"].items()},
                          "C_delta": max((r["adj_delta"] for r in out["rows"]), default=None)}
    g2 = out["gates"].get(label(EMO, 0, *BIG), {}).get("R_ok")
    sc["G2_R_reproduces_E14"] = g2
    sgd1, adam1 = label(EMO, 1, *SGD), label(EMO, 1, *SMALL)
    if sgd1 in end and adam1 in end:
        a, b = end[sgd1], end[adam1]
        sc["P1_seed"] = {"sgd_p": a["adj_p"], "adam_p": b["adj_p"], "sgd_keeps": keeps(a),
                         "pass": keeps(a) and b["adj_p"] <= a["adj_p"] - GAP}
    per: dict[str, Any] = {}
    for name in (R1, R0):
        grid = out["grids"].get(name)
        if grid is None:
            continue
        sgd, big = end.get(label(name, 0, "sgd", grid["selected"])), end.get(label(name, 0, *BIG))
        if sgd and big:
            ok = keeps(sgd) and sgd["adj_p"] >= big["adj_p"] + GAP
            per[name] = {"sgd_p": sgd["adj_p"], "adam_p": big["adj_p"], "rate": grid["selected"],
                         "pass": ok if grid["passed"] else "unscored: no rate met E4's rule"}
    if len(per) == 2:
        v = [p["pass"] for p in per.values() if isinstance(p["pass"], bool)]
        reading = ("unscored" if not v else "partial: one task unscored" if len(v) == 1 else  # a fallback rate is no reading
                   "both" if all(v) else "neither" if not any(v) else "one task")
        sc["P2_tasks"] = {"per_task": per, "reading": reading, "pass": reading == "both" if len(v) == 2 else reading}
    groups: dict[tuple[str, int], dict[str, str]] = {}
    for key, a in out["arms"].items():
        if len(a["steps"]) == len(STEPS):
            groups.setdefault((a["task"], a["seed"]), {})[key] = a["opt"]
    cond = {}
    for g in groups.values():
        for k, o in g.items():
            if o != "sgd":
                continue
            for s in STEPS:
                cond[f"{k}@{s} R>={R_MIN}"] = R[(k, s)]["R"] >= R_MIN
                cond |= {f"{k}@{s} > {j}": R[(k, s)]["R"] > R[(j, s)]["R"] for j, oj in g.items() if oj == "adam"}
    st = out["rows"]
    rho = spearman([r["R"] for r in st], [r["keep"] for r in st]) if len(st) >= 3 else None
    sc["P3_mechanism"] = {"a_read": all(cond.values()) if cond else None, "a_conditions": cond, "spearman": rho,
                          "n": len(st), "b_read": None if rho is None else rho >= RHO_MIN}
    for part in ("a", "b"):  # the filed rule: a failed G2 leaves P3 unscored; the readings stay as diagnostics
        sc["P3_mechanism"][f"{part}_pass"] = "unscored: G2 failed" if g2 is False else sc["P3_mechanism"][f"{part}_read"]
    kl = out.get("budget", {})
    sk = label(EMO, 0, *SGD)
    if "0" in kl.get("W0", {}) and len(kl.get(sk, {}).get("0", {})) == len(BUDGETS):
        w0 = kl["W0"]["0"]
        ratio = {m: {n: v["0"][n] / w0[n] for n in w0} for m, v in kl.items() if m != "W0" and "0" in v}
        c = abs(w0["20"] / W0_BUDGET20 - 1) <= W0_BUDGET_TOL
        p4 = all(ratio[sk][str(n)] <= BUDGET_PASS for n in BUDGETS[1:])
        sc["P4_budget"] = {"ratio": ratio, "C_budget_w0": c, "pass": p4 if c else f"unscored (C_budget_w0 failed); read {p4}",
                           "w0_seed1_over_seed0": {n: kl["W0"]["1"][n] / w0[n] for n in kl["W0"].get("1", {})}}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all", "shard", "merge"])
    ap.add_argument("shard", nargs="?", choices=SHARDS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        RESULTS.mkdir(parents=True, exist_ok=True)
        PRED.write_text(json.dumps(_finite({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}), indent=1))
        print(f"filed {PRED}")
    elif args.stage == "all":
        stage_all(args.device)
    elif args.stage == "shard" and args.shard:
        global LOG
        LOG = LARGE / f"full_{args.shard}.log"
        stage_shard(args.device, args.shard)
    elif args.stage == "merge":
        merge()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
