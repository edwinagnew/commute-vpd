"""regating_leak (E18): is re-gating a fixed switch pattern the original subcomponents turn into the behaviour, or do
the per-window gate values carry the updated model's computation past the decomposition (a leak)?
Plan: reports/2026-10-21_plan_regating-leak.md (approved class 4; build notes there, 2026-10-21).

Do-nothing states (target W0 + delta-W, released subcomponents C0, Gamma frozen, Delta out of the masked model) of the
step-10 fine-tunes on three tasks, seeds 0 (E11's saved delta-W) and 1 (trained here). On each task's held-out windows,
with the gates G0 of W0 everywhere except the trigger position, the masked model's trigger log-odds under:
  trig    each window's own state gates at the trigger;
  shared  each window's W0 gates plus one shared change per matrix (the mean over the TRAINING windows of the state's
          trigger gates minus W0's), clipped to [0, 1];
  mean    the state's mean trigger gates over the training windows, the same for every window;
  cross   the shared change of a different-target fine-tune (to ' http', 10 steps, same windows);
  random  the shared change's values on S_on (members up by > 0.3) moved onto a random same-size set, 20 draws.
Recovery = (lo - L00) / (L01 - L00), L01 - L00 the full re-gating effect (r x gain). Then J_nat (S_on against the top
subcomponents by W0's gate contrast at Pile positions whose next token is the target) and, unscored, the adversary's
turned-down share of S_on on Pile.

  run.py --predictions     file results/predictions.json (once)
  run.py                   -> results/regating_leak.json (written after every stage), results/large/full.log
Inputs (cluster NFS): experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
experiments/task_ladder/results/large/windows_<task>.pt, experiments/reading_a_coverage/results/large/<task>/dW_ft_step10_s0.pt.
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
OUT = "regating_leak.json"
TRANSPORT = HERE.parent / "transport_test" / "results"
LADDER = HERE.parent / "task_ladder" / "results" / "large"
COVERAGE = HERE.parent / "reading_a_coverage" / "results" / "large"

TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
R_REF = {"emoticon": (0.5965, 0.05), "r1_t534_v369": (0.7461, 0.1), "r2_t359_v816": (0.2844, 0.1)}  # E6 ft_s0; E5 do-nothing
FT_LR, FT_STEPS, SEEDS = 3e-5, 10, (0, 1)
NULL_TOKEN = 3944  # ' http': E6's null_b target
THR, N_RANDOM, ZERO_TOL, MIN_GAIN = 0.3, 20, 1e-6, 1.0
NAT_ROWS, MIN_NAT = ("val", 0, 600), 200
ADV_ROWS, ADV_STEPS = ("val", 0, 64), 20
BATCH = 64
PATTERNS = ("trig", "shared", "mean", "cross")

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-21, before any run; plan reports/2026-10-21_plan_regating-leak.md (Predictions, build notes)",
    "states": "do-nothing placement of the step-10 arm-A fine-tune (Adam 3e-5, batch 32, 10 steps): target W0 + delta-W, "
              "released subcomponents C0, Gamma frozen, Delta out of the masked model. Seed 0 = E11's saved delta-W, seed 1 "
              "trained in the job. Null: the same recipe to ' http' (token 3944), seed 0. Tasks emoticon (target 'o'), "
              "r1_t534_v369 (' which' -> ' was'), r2_t359_v816 (' we' -> ' just')",
    "measurements": {
        "r": "(L01 - L00) / (T1 - T0) on the held-out windows' trigger log-odds of the target; L = CI-masked model, C0",
        "recovery": "(lo_pattern - L00) / (L01 - L00); every pattern is W0's gates except at the trigger position",
        "rho_trig": "the state's own per-window gates at the trigger",
        "rho_shared": "W0's per-window trigger gates + mean over the training windows of (state - W0) trigger gates (alive "
                      "subcomponents; dead 0), clipped to [0, 1]",
        "rho_mean": "the state's mean trigger gates over the training windows, for every window (reported; rho_mean_W0, the "
                    "same with W0's mean, is its reference)",
        "rho_cross": "the null fine-tune's shared change, read at this task's target and divided by this task's gap",
        "random": f"S_on = alive subcomponents whose shared change > {THR}; the shared change's values on S_on moved onto "
                  f"{N_RANDOM} random same-size-per-matrix sets outside S_on (recruit.random_like seeds 0..19), recovery mean",
        "J_nat": f"Jaccard of S_on (seed 0) with the top-|S_on| alive subcomponents by W0's gate contrast (mean g at "
                 f"positions whose next token is the target minus mean g over all positions) on Pile {NAT_ROWS}; random: "
                 f"mean Jaccard of {N_RANDOM} random_like sets with the same top set",
    },
    "blind_to": {
        "recovery": "trigger windows only; interactions between gate changes at different positions (rho_trig measures how "
                    "much sits at the trigger). Positive control for the leak reading: rho_trig itself (per-window values "
                    "do recover). Null: the random moved shift",
        "J_nat": "how 'natural' positions are defined (next-token identity; contexts where the target is predicted but not "
                 "emitted); E6 flagged it as weak on emoticon",
    },
    "expected": {"r": "emoticon 0.60, r1_t534 0.75, r2_t359 0.28 (seed 1 within ~0.05 of seed 0)",
                 "W0_gates": "recovery 0 exactly (trig with W0's gates; shared with a zero shift)",
                 "random": "about 0", "J_nat_random": "about |S_on| / 9881 / 2, about 0.01"},
    "gate": {"id": "G1_reproduce", "rule": "seed 0: |r - ref| <= 0.05 on emoticon (ref 0.5965, E6 ft_s0) and <= 0.1 on "
                                          "r1_t534 (ref 0.7461) and r2_t359 (ref 0.2844; E5's do-nothing at step 10, the "
                                          "delta-W E11 regenerated); gain >= 1; recovery with W0's gates |.| <= 1e-6",
             "if_fail": "stop; nothing below is run"},
    "predictions": [
        {"id": "P1_trigger_carries", "rule": "rho_trig >= 0.7 on every task and seed", "predicted": "pass"},
        {"id": "P2_shared_pattern", "rule": "rho_shared >= 0.7 rho_trig on every task and seed, and the random mean <= 0.1",
         "leak": "rho_shared <= 0.3 rho_trig", "predicted": "pass on emoticon with confidence; pass on the ladder tasks with "
                                                            "low confidence; a fail on the ladder tasks only is the mixed branch"},
        {"id": "P3_task_specific", "rule": "rho_cross <= 0.1 on every task and seed", "predicted": "pass"},
        {"id": "P4_natural_machinery", "rule": "J_nat (seed 0) >= 3 x the random mean on every task",
         "conditional_on": f">= {MIN_NAT} natural positions in {NAT_ROWS[2]} rows for the task",
         "predicted": "pass on the ladder tasks, uncertain on emoticon"},
    ],
    "reported_not_scored": "rho_mean and rho_mean_W0; |S_on| per task and seed, by layer; Jaccard of S_on across seeds and "
                           "with the null's S_on; the null's held-out P(' http'); J_nat at seed 1; the adversary's "
                           f"turned-down share (source < 0.5) of S_on vs of all alive, at the state and at W0 ({ADV_STEPS} "
                           f"steps, seed 0, Pile {ADV_ROWS})",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _json(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    raise TypeError(type(o))


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-9 else float("nan")


def dump(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(_finite(obj), indent=1, sort_keys=True, default=_json))


# ----------------------------------------------------------------------------- loading


def load_task(name: str) -> tuple[Any, Any, int]:
    if name == "emoticon":
        w = torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False)
        return w["train"], w["heldout"], json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    w = torch.load(LADDER / f"windows_{name}.pt", weights_only=False)
    return w["train"], w["heldout"], w["task"]["target"]


def finetune(ctx: dict[str, Any], task: dict[str, Any], token: int, seed: int) -> tuple[dict[str, Tensor], float]:
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=token, lr=FT_LR, steps=FT_STEPS, checkpoints=(FT_STEPS,), seed=seed)
    res = finetune_arm_a(ctx["model"], task["train"], task["held"], cfg, ctx["device"], log)
    return res.deltas[FT_STEPS], res.evals[FT_STEPS]["p_o"]


def state_of(ctx: dict[str, Any], dW: dict[str, Tensor]) -> Any:
    from commute.faithfulness import model_state

    W0 = ctx["W0"]
    return model_state(ctx["model"], target={k: W0[k] + dW[k].to(W0[k].device, W0[k].dtype) for k in W0})


def lo_of(ctx: dict[str, Any], task: dict[str, Any], gates: Any) -> float:
    from commute.local_faith import masked_lo

    held = task["held"]
    return masked_lo(ctx["model"], held.tokens, held.half, task["target"], gates, BATCH)[0]


# ----------------------------------------------------------------------------- stages


def stage_g1(ctx: dict[str, Any], name: str) -> dict[str, Any]:
    """W0's cells, seed 0's state and r; recovery with W0's own gates (must be 0 exactly)."""
    from commute import recruit as R
    from commute.local_faith import route_cells

    train, held, target = load_task(name)
    task: dict[str, Any] = {"train": train, "held": held, "target": target, "half": held.half}
    model, h = ctx["model"], held.half
    task["base"] = route_cells(model, held.tokens, h, target, BATCH)
    task["dW0"] = torch.load(COVERAGE / name / f"dW_ft_step{FT_STEPS}_s0.pt", weights_only=True)
    with state_of(ctx, task["dW0"]):
        task["cur0"] = route_cells(model, held.tokens, h, target, BATCH)
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    gap, gain = task["cur0"]["L"] - L00, task["cur0"]["T"] - task["base"]["T"]
    zero = {"trig": lo_of(ctx, task, R.with_trigger(g0, h, R.at_trigger(g0, h))) - L00,
            "shared": lo_of(ctx, task, R.with_trigger(g0, h, R.shifted(g0, h, {k: torch.zeros(v.shape[-1]) for k, v in g0[0].items()}))) - L00}
    ref, tol = R_REF[name]
    r = div(gap, gain)
    ok = abs(r - ref) <= tol and gain >= MIN_GAIN and all(abs(div(v, gap)) <= ZERO_TOL for v in zero.values())
    rec = {"target": target, "n_train": len(train), "n_heldout": len(held), "r": r, "r_ref": ref, "r_tol": tol, "gain": gain,
           "gap": gap, "cells": {"T0": task["base"]["T"], "T1": task["cur0"]["T"], "L00": L00, "L01": task["cur0"]["L"],
                                 "U0": task["base"]["U"], "U1": task["cur0"]["U"]},
           "zero_recovery": {k: div(v, gap) for k, v in zero.items()}, "ok": ok}
    log(f"[G1 {name}] r {r:.4f} (ref {ref} +/- {tol}) gain {gain:.3f} gap {gap:.3f} | W0-gate recovery trig "
        f"{rec['zero_recovery']['trig']:.2e} shared {rec['zero_recovery']['shared']:.2e} | {'ok' if ok else 'FAIL'}")
    ctx["tasks"][name] = task
    return rec


def patterns(ctx: dict[str, Any], task: dict[str, Any], cur: dict[str, Any], shift: dict[str, Tensor],
             mean1: dict[str, Tensor]) -> dict[str, float]:
    """Trigger log-odds minus L00 under each pattern, and under the random moved shifts (list)."""
    from commute import recruit as R

    g0, h, L00 = task["base"]["gates"], task["half"], task["base"]["L"]
    trig = {"trig": R.at_trigger(cur["gates"], h), "shared": R.shifted(g0, h, shift), "mean": R.constant(g0, h, mean1),
            "cross": R.shifted(g0, h, task["shift_null"])}
    out: dict[str, Any] = {p: lo_of(ctx, task, R.with_trigger(g0, h, t)) - L00 for p, t in trig.items()}
    on = R.switched(shift, THR)
    out["random"] = [lo_of(ctx, task, R.with_trigger(g0, h, R.shifted(g0, h, R.moved(shift, on, R.random_like(on, ctx["alive"], i)))))
                     - L00 for i in range(N_RANDOM)]
    return out


def stage_task(ctx: dict[str, Any], name: str) -> dict[str, Any]:
    from commute import recruit as R
    from commute.local_faith import route_cells, trigger_mean_gates

    task, model, alive = ctx["tasks"][name], ctx["model"], ctx["alive"]
    train, held, h = task["train"], task["held"], task["half"]
    g0, L00 = task["base"]["gates"], task["base"]["L"]
    m0 = trigger_mean_gates(model, train.tokens, h, BATCH)
    dW_null, p_null = finetune(ctx, task, NULL_TOKEN, 0)
    with state_of(ctx, dW_null):
        task["shift_null"] = R.mean_change(trigger_mean_gates(model, train.tokens, h, BATCH), m0, alive)
    on_null = R.switched(task["shift_null"], THR)
    lo_mean_w0 = lo_of(ctx, task, R.with_trigger(g0, h, R.constant(g0, h, m0))) - L00
    rec: dict[str, Any] = {"null": {"p_heldout": p_null, "n_on": R.size(on_null)}, "seeds": {}}
    for seed in SEEDS:
        if seed == 0:
            dW, cur = task["dW0"], task["cur0"]
        else:
            dW, _ = finetune(ctx, task, task["target"], seed)
            with state_of(ctx, dW):
                cur = route_cells(model, held.tokens, h, task["target"], BATCH)
        with state_of(ctx, dW):
            mean1 = trigger_mean_gates(model, train.tokens, h, BATCH)
        shift = R.mean_change(mean1, m0, alive)
        on = R.switched(shift, THR)
        torch.save({"shift": shift, "shift_null": task["shift_null"]}, LARGE / f"{name}_s{seed}_shift.pt")
        gap, gain = cur["L"] - L00, cur["T"] - task["base"]["T"]
        d = patterns(ctx, task, cur, shift, mean1)
        rho = {p_: div(d[p_], gap) for p_ in PATTERNS} | {"mean_W0": div(lo_mean_w0, gap)}
        rnd = [div(x, gap) for x in d["random"]]
        s = {"p_heldout": cur["p_T"], "gain": gain, "gap": gap, "r": div(gap, gain), "rho": rho, "random": rnd,
             "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd), "shared_over_trig": div(rho["shared"], rho["trig"]),
             "n_on": R.size(on), "by_layer_on": R.by_layer(on)}
        rec["seeds"][str(seed)] = s
        task[f"on{seed}"] = on
        log(f"[{name} s{seed}] r {s['r']:.4f} gain {gain:.3f} | rho trig {rho['trig']:.4f} shared {rho['shared']:.4f} "
            f"mean {rho['mean']:.4f} (W0 {rho['mean_W0']:.4f}) cross {rho['cross']:.4f} | random mean {s['random_mean']:.4f} "
            f"max {s['random_max']:.4f} | shared/trig {s['shared_over_trig']:.4f} | |S_on| {s['n_on']}")
    rec["J_seeds"] = R.jaccard(task["on0"], task["on1"])
    rec["J_null"] = R.jaccard(task["on0"], on_null)
    log(f"[{name}] J(s0, s1) {rec['J_seeds']:.4f} J(s0, null) {rec['J_null']:.4f} | null P(http) {p_null:.4f} |S_on null| {R.size(on_null)}")
    for k in ("base", "cur0", "dW0"):
        task.pop(k)
    return rec


def stage_natural(ctx: dict[str, Any]) -> dict[str, Any]:
    """W0's gate contrast at Pile positions whose next token is each task's target; J_nat against S_on."""
    from commute import recruit as R
    from commute.data import pile_rows
    from commute.signatures import top_by
    from commute.vpd import compute_ci

    rows = pile_rows(*NAT_ROWS, log=log)
    masks = {n: R.precede_mask(rows, t["target"]) for n, t in ctx["tasks"].items()}
    accs = {n: R.ContrastAcc() for n in masks}
    for s in range(0, rows.shape[0], 8):
        ci = compute_ci(ctx["model"], rows[s : s + 8], chunk=8)
        for n, acc in accs.items():
            acc.add(ci.g, masks[n][s : s + 8])
        del ci
    out: dict[str, Any] = {}
    for n, acc in accs.items():
        t, rec = ctx["tasks"][n], {"n_positions": acc.n_on, "scorable": acc.n_on >= MIN_NAT}
        contrast = acc.result(ctx["alive"])
        for seed in SEEDS:
            on = t[f"on{seed}"]
            nat = top_by(contrast, ctx["alive"], R.size(on))
            rnd = [R.jaccard(R.random_like(on, ctx["alive"], i), nat) for i in range(N_RANDOM)]
            rec[f"s{seed}"] = {"J": R.jaccard(on, nat), "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd)}
        out[n] = rec
        log(f"[natural {n}] {acc.n_on} positions | J_nat s0 {rec['s0']['J']:.4f} (random {rec['s0']['random_mean']:.4f}) "
            f"s1 {rec['s1']['J']:.4f} (random {rec['s1']['random_mean']:.4f})")
    return out


def stage_adversary(ctx: dict[str, Any]) -> dict[str, Any]:
    """Unscored: the share of S_on (seed 0) the Pile adversary turns down, at the state and at W0, vs all alive."""
    from commute import gamma
    from commute.data import batches, pile_rows

    rows = batches(pile_rows(*ADV_ROWS, log=log), 8)
    alive = ctx["alive"]

    def down_share(src: dict[str, Tensor], sel: dict[str, Tensor]) -> float:
        n = sum(int((src[k][:-1][v] < gamma.DOWN).sum()) for k, v in sel.items())
        return n / max(sum(int(v.numel()) for v in sel.values()), 1)

    all_alive = {k: v.cpu().bool().nonzero().flatten() for k, v in alive.items()}
    w0_src, w0_kl = gamma.pgd_sources(ctx["model"], rows, ADV_STEPS, seed=0)
    out: dict[str, Any] = {"W0": {"kl": w0_kl, "down_alive": down_share(w0_src, all_alive)}}
    for n, t in ctx["tasks"].items():
        dW = torch.load(COVERAGE / n / f"dW_ft_step{FT_STEPS}_s0.pt", weights_only=True)
        with state_of(ctx, dW):
            src, kl = gamma.pgd_sources(ctx["model"], rows, ADV_STEPS, seed=0)
        out[n] = {"kl": kl, "down_alive": down_share(src, all_alive), "down_S_on": down_share(src, t["on0"]),
                  "down_S_on_at_W0": down_share(w0_src, t["on0"])}
        log(f"[adversary {n}] KL {kl:.4f} (W0 {w0_kl:.4f}) | turned down: S_on {out[n]['down_S_on']:.3f} (W0's adversary "
            f"{out[n]['down_S_on_at_W0']:.3f}), all alive {out[n]['down_alive']:.3f} (W0 {out['W0']['down_alive']:.3f})")
    return out


# ----------------------------------------------------------------------------- scoring


def score(out: dict[str, Any]) -> dict[str, Any]:
    T = out["tasks"]
    per = [(n, s, v) for n, t in T.items() for s, v in t["seeds"].items()]
    ratio = {f"{n}/s{s}": v["shared_over_trig"] for n, s, v in per}
    p2 = {f"{n}/s{s}": ("mechanism" if ratio[f"{n}/s{s}"] >= 0.7 else "leak" if ratio[f"{n}/s{s}"] <= 0.3 else "between")
          for n, s, _ in per}
    res: dict[str, Any] = {
        "P1_trigger_carries": {"rho_trig": {f"{n}/s{s}": v["rho"]["trig"] for n, s, v in per},
                               "pass": all(v["rho"]["trig"] >= 0.7 for _, _, v in per)},
        "P2_shared_pattern": {"shared_over_trig": ratio, "reading": p2, "random_mean": {f"{n}/s{s}": v["random_mean"] for n, s, v in per},
                              "pass": all(r >= 0.7 for r in ratio.values()) and all(v["random_mean"] <= 0.1 for _, _, v in per)},
        "P3_task_specific": {"rho_cross": {f"{n}/s{s}": v["rho"]["cross"] for n, s, v in per},
                             "pass": all(v["rho"]["cross"] <= 0.1 for _, _, v in per)},
    }
    nat = out.get("natural", {})
    if nat:
        res["P4_natural_machinery"] = {n: {"J": v["s0"]["J"], "random_mean": v["s0"]["random_mean"], "scorable": v["scorable"],
                                           "pass": v["s0"]["J"] >= 3 * v["s0"]["random_mean"]} for n, v in nat.items()}
        res["P4_natural_machinery"]["pass"] = all(v["pass"] for v in res["P4_natural_machinery"].values() if v["scorable"])
    return res


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    from commute import vpd

    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    t0 = time.time()
    model = vpd.load_component_model(device)
    alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
    ctx: dict[str, Any] = {"device": device, "model": model, "W0": vpd.target_weights(model), "tasks": {},
                           "alive": {k: v.cpu().bool() for k, v in alive.items()}}
    out: dict[str, Any] = {"G1": {}, "tasks": {}}
    for name in TASKS:
        out["G1"][name] = stage_g1(ctx, name)
        dump(OUT, out)
    out["G1"]["pass"] = all(v["ok"] for v in out["G1"].values())
    if not out["G1"]["pass"]:
        out["status"] = "stopped at G1"
        dump(OUT, out)
        log(f"G1 FAILED: stopping ({time.time() - t0:.0f} s)")
        return
    for name in TASKS:
        out["tasks"][name] = stage_task(ctx, name)
        out["score"] = score(out)
        dump(OUT, out)
    out["natural"] = stage_natural(ctx)
    out["score"] = score(out)
    dump(OUT, out)
    log("score " + json.dumps(_finite(out["score"]), default=_json))
    out["adversary"] = stage_adversary(ctx)
    out["seconds"] = time.time() - t0
    out["status"] = "done"
    dump(OUT, out)
    log(f"wrote {RESULTS / OUT} ({out['seconds']:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        dump(PRED.name, PREDICTIONS)
        print(f"filed {PRED}")
        return
    run(args.device)


if __name__ == "__main__":
    main()
