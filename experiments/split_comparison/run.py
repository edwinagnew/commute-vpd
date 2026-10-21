"""split_comparison (E16): the five operations side by side on one card (plan: reports/2026-10-21_plan_split-comparison.md).

Two updates of the emoticon task at steps 10, 100 and 1000: the full fine-tune (arm A, Adam 3e-5, seed 0, re-run with
delta-W kept, as E14) and E9's rank-8 LoRA on all 24 matrices (saved factors). On each update four states: do-nothing,
the least-squares split, the scaled adjoint split and the rescaling split (u_c <- (1 + alpha_c) u_c, Delta fixed). Beside
them, component training with no update consumed: free on every alive subcomponent (lr 1e-5, E2's rate), paired with the
fine-tune, and rank-8 (lr 1e-4, E13), paired with the LoRA; Gamma and Delta frozen; one run each, states at the same
steps. Every state goes through harness.measure (route split, Delta's share of ||delta-W||^2, held-out P(target), Pile
ratios to W0). Every update's and every component-training weight change's four blocks (split.block_table): for component
training the outside-both block is the positive control (only dU dV^T can reach it). Design confirmed by the user 2026-10-21.

Stages (results/predictions.json must exist; `--predictions` files it once):
  all             GPU -> results/split_comparison.json (dumped after every state) + results/large/full.log
  all --smoke     3 training steps, states at steps 2 and 3, 8 Pile rows, 2 PGD steps -> results/split_comparison_smoke.json
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}, results/windows.json (round 1);
experiments/lora_plateau/results/{large/lora_factors.pt, lora_plateau.json} (E9); experiments/lora_split/results/lora_split.json (E14).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"
E9 = HERE.parent / "lora_plateau" / "results"
E14 = HERE.parent / "lora_split" / "results" / "lora_split.json"
E9_STEPS = (1, 2, 3, 5, 10, 30, 100, 300, 1000)
FT_LR, TOL = 3e-5, 1e-6  # arm A's recipe and the split tolerance, as E14
CT = {"ft": {"rank": None, "lr": 1e-5}, "lora24": {"rank": 8, "lr": 1e-4}}  # component training paired with each update
OPS = ("nothing", "least_squares", "adjoint", "rescaling", "component_training")


@dataclass(frozen=True)
class Sizes:
    steps: tuple[int, ...]  # the scored steps of every update and of component training
    train_steps: int  # fine-tune and component-training length
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    suffix: str

    def __post_init__(self) -> None:
        assert self.steps == tuple(sorted(self.steps)) and self.steps[-1] == self.train_steps, self.steps
        assert set(self.steps) <= set(E9_STEPS), self.steps  # E9's factors exist only at its checkpoints
        assert self.pile_rows > 0 and len(self.pile_seeds) >= 2, "the floor needs two adversary seeds"


FULL = Sizes(steps=(10, 100, 1000), train_steps=1000, pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2), suffix="")
SMOKE = Sizes(steps=(2, 3), train_steps=3, pile_rows=8, pile_pgd_steps=2, pile_seeds=(0, 1), suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-21",
    "design": "reports/2026-10-21_plan_split-comparison.md (design confirmed by the user 2026-10-21)",
    "setup": {
        "updates": {"ft": f"emoticon full fine-tune, arm A (Adam lr {FT_LR}, 24 matrices, seed 0), delta-W at {FULL.steps}",
                    "lora24": f"E9's rank-8 LoRA on all 24 matrices, B A from the saved factors at {FULL.steps}"},
        "states": "do-nothing and least-squares split (harness.nothing_and_split); scaled adjoint split (split.adjoint_state); "
                  "rescaling split (split.rescaling_state: u_c <- (1 + alpha_c) u_c, alpha the input-weighted least-squares fit "
                  "of delta-W by sum_c alpha_c u_c v_c^T, Delta fixed); component training paired with ft: free, lr 1e-5; with "
                  "lora24: Offsets(rank=8), lr 1e-4; Gamma and Delta frozen, seed 0, states at the same steps",
        "measure": f"harness.measure: route split (d, r) and gain at the trigger, held-out P(target), Delta's share of ||delta-W||^2, "
                   f"Pile ratios to W0 ({FULL.pile_rows} rows, {FULL.pile_pgd_steps} PGD steps); floor f = max/min - 1 of W0's "
                   f"PGD KL over adversary seeds {FULL.pile_seeds}",
        "R": "reproduced share, energy-pooled over the non-zero matrices: adjoint = split.adjoint_scale's second value; "
             "rescaling = split.rescaling_share; 1 for do-nothing and least-squares by construction",
        "blocks": "split.block_table of each update and each component-training weight change (U+dU)(V+dV)^T - UV^T, "
                  "pooled over all 24 matrices by input-weighted energy",
    },
    "gates": [
        {"id": "G1_reproduce_E14", "rule": "for ft and lora24 at steps 10, 100, 1000: do-nothing, least-squares and adjoint Pile "
                                           "PGD ratios within max(0.15, 3 f) absolute of E14's (lora_split.json part2), and the "
                                           "adjoint state's held-out P(target) within 0.05 absolute of E14's",
         "if_fail": "the harness, the fine-tune or the card changed the reference; stop and find out before reading P1-P3"},
        {"id": "G2_split_state", "rule": "harness G2 per update and step", "if_fail": "that step is reported, not scored"},
    ],
    "predictions": [
        {"id": "P1_rescaling_state", "scored": True, "arm": "rescaling state of ft and lora24, every step",
         "pass": "Pile PGD ratio <= 1 + 3 f, held-out P(target) < 0.05 (baseline about 0.004) and pooled R <= 0.1, every step, "
                 "both updates (absolute thresholds)",
         "null": "P(target) >= 0.05 at some step (rescaling existing subcomponents carries the behaviour)",
         "predicted": "pass: E14's rescaling share was 2-7% of the update's energy; a 0.1 or smaller copy keeps the model at W0",
         "blind_to": "the Pile checks do not see the trained behaviour; P(target) does not see off-target damage"},
        {"id": "P2_component_training", "scored": True, "arm": "free (paired with ft) and rank-8 (paired with lora24)",
         "pass": "held-out P(target) >= 0.9 at steps 100 and 1000, and Pile PGD ratio < 1.5 at steps 10 and 100, both variants "
                 "(absolute); step 1000's PGD ratio is a first reading, no threshold",
         "null": "P(target) < 0.9 at step 100, or PGD ratio >= 1.5 at step 10 or 100",
         "predicted": "pass: E2 free 0.95 by step 3 and PGD 1.03-1.07 at 10-30; E13 rank-8 PGD 1.14-1.28 at 30",
         "blind_to": "as P1"},
        {"id": "C1_positive_control", "scored": True, "arm": "component training's weight change, both variants, every step",
         "pass": "pooled outside-both block <= 0.01 of its input-weighted energy (absolute share)",
         "null": "> 0.01: the block code or the training hook is wrong (first-order changes cannot reach the block)",
         "predicted": "pass: only dU dV^T reaches it, second order in a small change",
         "blind_to": "anything inside the subcomponents' first-order reach; it checks the block, not the behaviour"},
        {"id": "P3_no_direct_route", "scored": True, "arm": "every state of both pairs at every step whose gain > 1 nat",
         "pass": "route split direct share d <= 0.15 (absolute)", "null": "d > 0.15 for some such state",
         "predicted": "pass: E13 and E14 read d 0.00-0.08", "blind_to": "the Pile; reads the trigger windows only"},
    ],
    "reported_not_scored": "step 1000's component-training PGD ratio; stochastic, CI-masked and unmasked ratios; Delta's share; "
                           "r; the four blocks and outside-both / null per update; the updates' and component training's "
                           "off-target KL from their own held-out evals (not measured for the split states)",
}


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


def dump(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))


ROW_BLOCKS = ("in_in", "in_out", "out_in", "out_out", "null_out_out", "out_out_ratio")


def flat_rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per (pair, step, operation) measured so far, every record with the same keys (None where a column does
    not apply): the table the plots and the notebook read, without knowing this file's nesting."""
    line = 1 + 3 * out["W0"]["pgd_floor"]
    rows = []
    for pair, by_step in out["states"].items():
        for s, rec in sorted(by_step.items(), key=lambda kv: int(kv[0])):
            for op in (o for o in OPS if o in rec):
                r, src = rec[op], "component_training" if op == "component_training" else "update"
                blk = out["blocks"][pair][s].get(src, {}).get("pooled_all") or {}
                pile, routes = r["pile"], r["routes"]
                rows.append({"pair": pair, "step": int(s), "op": op, "variant": variant(pair, src),
                             "pgd": pile["pgd_kl"], "stochastic": pile["stochastic_kl"], "ci_masked": pile["ci_masked_kl"],
                             "unmasked": pile["unmasked_kl"], "pgd_floor": out["W0"]["pgd_floor"], "pgd_line": line,
                             "delta_share": r["delta_change_frac"], "R": r["R"], "p_target": routes["p_cells"]["T1"],
                             "p_baseline": routes["p_cells"]["T0"], "p_update": rec["nothing"]["p_o_update"],
                             "gain": routes["gain"], "d": routes["shares"]["d"], "r": routes["shares"]["r"],
                             "kl_other": r.get("kl_other"), "g2_ok": r["G2"]["ok"] if "G2" in r else None, "blocks_of": src,
                             **{f"blocks_{b}": blk.get(b) for b in ROW_BLOCKS}})
    return rows


def variant(pair: str, src: str) -> str:
    if src == "update":
        return pair
    rank, lr = CT[pair]["rank"], CT[pair]["lr"]
    return f"{'free' if rank is None else f'rank {rank}'} component training, lr {lr:g}"


def save(name: str, out: dict[str, Any]) -> None:
    out["rows"] = flat_rows(out)
    dump(name, out)


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; predictions are not edited after filing (add a dated report instead)")
    dump(PRED.name, PREDICTIONS)
    print(f"filed {PRED}")


# ----------------------------------------------------------------------------- the updates and component training


def updates(ctx: Any, S: Sizes) -> tuple[dict[str, dict[int, dict[str, Tensor]]], dict[str, dict[int, dict[str, float]]]]:
    """delta-W of both updates at S.steps, and each update's own held-out eval (p_o, kl_other) at those steps."""
    from commute.finetune import ArmAConfig, finetune_arm_a, saved_lora_updates

    t0 = time.time()
    cfg = ArmAConfig(o_token=ctx.token, lr=FT_LR, steps=S.train_steps, checkpoints=S.steps, seed=0)
    ft = finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log)
    e9 = json.loads((E9 / "lora_plateau.json").read_text())["checkpoints"]
    dW = {"ft": ft.deltas, "lora24": saved_lora_updates(E9 / "large" / "lora_factors.pt", list(ctx.W0), ctx.W0, S.steps)}
    evals = {"ft": {s: ft.evals[s] for s in S.steps}, "lora24": {s: e9[str(s)]["eval"] for s in S.steps}}
    log(f"[stage updates] {time.time() - t0:.0f} s")
    return dW, evals


def component_training(ctx: Any, S: Sizes, pair: str) -> tuple[dict[int, Any], dict[int, dict[str, float]]]:
    """One component-training run (CT[pair]) with states at S.steps; its held-out eval at those steps."""
    from commute import component_training as ct
    from commute.finetune import ArmAConfig

    t0 = time.time()
    a_cfg = ArmAConfig(o_token=ctx.token, steps=S.train_steps, checkpoints=(), seed=0)
    cfg = ct.TrainConfig(lr=CT[pair]["lr"], steps=S.train_steps, save_steps=S.steps, rank=CT[pair]["rank"])
    curve, states = ct.train(ctx.model, ctx.uv, ctx.train.tokens, ctx.held, a_cfg, cfg, ctx.device, ctx.W0, None, log)
    evals = {c["step"]: {"p_o": c["heldout_p_o"], "kl_other": c["heldout_kl_other"]} for c in curve if c["step"] in S.steps}
    log(f"[stage component_training {pair}] {time.time() - t0:.0f} s")
    return states, evals


# ----------------------------------------------------------------------------- the states


def reproduced(ctx: Any, dW: dict[str, Tensor]) -> dict[str, float]:
    """R of the adjoint and rescaling states, pooled over the non-zero matrices by input-weighted energy."""
    from commute import split

    num, den = {"adjoint": 0.0, "rescaling": 0.0}, 0.0
    for k, M in dW.items():
        if not M.any():
            continue
        U_A, V_A, Sg = ctx.uv_cpu[k][0].double(), ctx.uv_cpu[k][1].double(), ctx.sigma[k]
        e = split.weighted_energy(M.double(), Sg)
        num["adjoint"] += e * split.adjoint_scale(M, U_A, V_A, Sg)[1]
        num["rescaling"] += e * split.rescaling_share(M, U_A, V_A, Sg)
        den += e
    return {k: v / den for k, v in num.items()}


def update_states(ctx: Any, pair: str, step: int, dW: dict[str, Tensor], ev: dict[str, float], put: Any) -> dict[str, Any]:
    """Do-nothing, least-squares, adjoint and rescaling states of one update at one step; `put` stores and dumps after each."""
    from commute import harness, split

    name, rec = f"{pair}_step{step}", {}
    ns = harness.nothing_and_split(ctx, name, dW, ev["p_o"], True, log)
    R = reproduced(ctx, dW)
    rec["nothing"] = {**ns["nothing"], "R": 1.0, "kl_other": ev["kl_other"], "p_o_update": ev["p_o"]}
    rec["least_squares"] = {**ns["lift"], "R": 1.0, "G2": ns["G2"]}
    put(rec)
    for op, fn in (("adjoint", split.adjoint_state), ("rescaling", split.rescaling_state)):
        dW_op, state = fn(dW, ctx.uv_cpu, ctx.sigma)
        rec[op] = {**harness.measure(ctx, f"{name}_{op}", dW_op, state, True, log), "R": R[op]}
        log(f"[{name}_{op}] R {R[op]:.4f}")
        put(rec)
    return rec


def ct_state(ctx: Any, pair: str, step: int, state: dict[str, Any], ev: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A component-training state through the harness, and its weight change's four blocks (the positive control)."""
    from commute import harness, split

    dW = {k: v[0] for k, v in state.items()}
    rec = {**harness.measure(ctx, f"{pair}_step{step}_component_training", dW, state, True, log), "R": 1.0, **ev}
    blocks = split.block_table(dW, ctx.bases, ctx.sigma, ctx.non_saturated)
    log_blocks(f"{pair}_step{step}_component_training", blocks)
    return rec, blocks


def log_blocks(name: str, t: dict[str, Any]) -> None:
    p = t["pooled_all"]
    log(f"[{name} blocks, all 24] in_in {p['in_in']:.4f} in_out {p['in_out']:.4f} out_in {p['out_in']:.4f} "
        f"out_out {p['out_out']:.4f} (null {p['null_out_out']:.4f}, ratio {p['out_out_ratio']:.3f})")


# ----------------------------------------------------------------------------- the run


def stage_all(args: argparse.Namespace) -> None:
    from commute import harness, split

    assert PRED.is_file(), "file predictions first: run.py --predictions"
    S = SMOKE if args.smoke else FULL
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    log(f"[stage load] {time.time() - t_start:.0f} s; W0 PGD floor {harness.pgd_floor(ctx):.4f}")
    name = f"split_comparison{S.suffix}.json"
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S), "W0": {"pile": ctx.pile, "pgd_floor": harness.pgd_floor(ctx)},
                           "states": {p: {} for p in CT}, "blocks": {p: {} for p in CT}}
    dW, evals = updates(ctx, S)
    for pair in CT:
        for step in S.steps:
            t0 = time.time()
            out["blocks"][pair][str(step)] = {"update": split.block_table(dW[pair][step], ctx.bases, ctx.sigma, ctx.non_saturated)}
            log_blocks(f"{pair}_step{step}_update", out["blocks"][pair][str(step)]["update"])

            def put(rec: dict[str, Any], pair: str = pair, step: int = step) -> None:
                out["states"][pair][str(step)] = rec
                save(name, out)

            update_states(ctx, pair, step, dW[pair][step], evals[pair][step], put)
            log(f"[stage states {pair} step {step}] {time.time() - t0:.0f} s")
    del dW
    for pair in CT:
        states, ct_evals = component_training(ctx, S, pair)
        for step in S.steps:
            rec, blocks = ct_state(ctx, pair, step, states.pop(step), ct_evals[step])
            out["states"][pair][str(step)]["component_training"] = rec
            out["blocks"][pair][str(step)]["component_training"] = blocks
            save(name, out)
    out["score"] = score(out, json.loads(E14.read_text()))
    out["seconds"] = time.time() - t_start
    save(name, out)
    log(f"wrote {RESULTS / name} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


# ----------------------------------------------------------------------------- the score


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    return {"pass": all(cond.values()) if cond else None, "conditions": cond, **values}


def gate_e14(out: dict[str, Any], e14: dict[str, Any]) -> dict[str, Any]:
    """G1 at the steps both runs measured (none in the smoke run)."""
    tol = max(0.15, 3 * out["W0"]["pgd_floor"])
    cond, diffs = {}, {}
    for pair, by_step in out["states"].items():
        for s, rec in by_step.items():
            ref = e14["part2"].get(pair, {}).get(s)
            if ref is None:
                continue
            for op, ref_op in (("nothing", "nothing"), ("least_squares", "lift"), ("adjoint", "adjoint")):
                d = abs(rec[op]["pile"]["pgd_kl"] - ref[ref_op]["pile"]["pgd_kl"])
                diffs[f"{pair}_{op}@{s}"], cond[f"{pair}_{op}_pgd@{s}"] = d, d <= tol
            d = abs(rec["adjoint"]["routes"]["p_cells"]["T1"] - ref["adjoint"]["routes"]["p_cells"]["T1"])
            diffs[f"{pair}_adjoint_p@{s}"], cond[f"{pair}_adjoint_p@{s}"] = d, d <= 0.05
    return _pred(cond, tol=tol, diffs=diffs)


def score(out: dict[str, Any], e14: dict[str, Any]) -> dict[str, Any]:
    st, line = out["states"], 1 + 3 * out["W0"]["pgd_floor"]
    sc: dict[str, Any] = {"line": line, "G1_reproduce_E14": gate_e14(out, e14),
                          "G2_split_state": {p: {s: r["least_squares"]["G2"]["ok"] for s, r in st[p].items()} for p in st}}
    cond: dict[str, bool] = {}
    for p in st:
        for s, r in st[p].items():
            rs = r["rescaling"]
            cond |= {f"{p}_pgd@{s}": rs["pile"]["pgd_kl"] <= line, f"{p}_p<0.05@{s}": rs["routes"]["p_cells"]["T1"] < 0.05,
                     f"{p}_R<=0.1@{s}": rs["R"] <= 0.1}
    sc["P1_rescaling_state"] = _pred(cond)
    cond = {}
    for p in st:
        for s, r in st[p].items():
            c = r.get("component_training")
            if c is None:
                continue
            if int(s) in (100, 1000):
                cond[f"{p}_p>=0.9@{s}"] = c["routes"]["p_cells"]["T1"] >= 0.9
            if int(s) in (10, 100):
                cond[f"{p}_pgd<1.5@{s}"] = c["pile"]["pgd_kl"] < 1.5
    sc["P2_component_training"] = _pred(cond)
    sc["C1_positive_control"] = _pred({f"{p}@{s}": b["component_training"]["pooled_all"]["out_out"] <= 0.01
                                       for p in out["blocks"] for s, b in out["blocks"][p].items() if "component_training" in b})
    sc["P3_no_direct_route"] = _pred({f"{p}_{op}@{s}": r[op]["routes"]["shares"]["d"] <= 0.15
                                      for p in st for s, r in st[p].items() for op in OPS
                                      if op in r and r[op]["routes"]["gain"] > 1})
    if sc["G1_reproduce_E14"]["pass"] is False:  # filed: a failed gate leaves P1-P3 unread
        for k in ("P1_rescaling_state", "P2_component_training", "P3_no_direct_route"):
            sc[k]["pass"] = "unscored: G1 failed"
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--smoke", action="store_true")
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
