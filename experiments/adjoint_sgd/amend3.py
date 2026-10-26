"""adjoint_sgd (E31) amendment 3 (2026-10-26): the unscaled adjoint split beside the scaled one.
Plan: reports/plans/2026-10-22_plan_adjoint-sgd.md -> "Amendment 3". Research-log decision card "decision-unscaled-adjoint".

E31's three emoticon seed-0 fine-tunes (SGD 0.01, Adam 3e-6, Adam 3e-5), retrained to step 1000 with E31's recipe. At
steps 10, 100 and 1000 each update is placed two ways, Delta fixed in both: the scaled adjoint state (split.adjoint_state,
c fitted per matrix, as in E31) and the unscaled one (c = 1, weight change K(delta-W)). Each state is read through
harness.measure (held-out P(target), Pile ratios to the same-job W0, Delta's share) and by its KL elsewhere: the KL from
W0 to the state's model at the held-out positions other than the trigger, the fine-tune's own `kl_other` readout.
Order, each record dumped to results/adjoint_sgd_amend3_<cell>.json as it finishes:
  G0     W0 under the Pile harness (faith_score.w0_check); stop if it fails
  train  the fine-tune to step 1000, delta-W kept at 10, 100, 1000
  A3_G1  step 10: the KL readout on the fine-tune itself equals its `kl_other` (1e-4), and the scaled state reads E31's
         row (adjoint P(target) within 0.02, R within 0.01); stop on a miss
  then   the unscaled state at step 10; both states at 100, then 1000

Usage (repo root):  amend3.py --predictions  (files results/predictions_amend3.json once)
                    amend3.py <cell>         sgd | small | big; sgd first and alone
                    amend3.py --merge        the cell files -> results/adjoint_sgd_amend3.json and its score (CPU)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from typing import Any

import run
from run import (
    BIG,
    EMO,
    LARGE,
    OUT,
    RESULTS,
    SGD,
    SMALL,
    STEPS,
    TRAIN_STEPS,
    Task,
    _finite,
    finetune,
    label,
    log,
)

PRED_A3, OUT_A3 = RESULTS / "predictions_amend3.json", RESULTS / "adjoint_sgd_amend3.json"
CELLS_A3 = {"sgd": SGD, "small": SMALL, "big": BIG}
G1_P, G1_R, G1_KL = 0.02, 0.01, 1e-4
SAME, KL_X = 0.05, 2.0  # D1's band on |keep difference|; D2's factor on KL elsewhere

PREDICTIONS_A3: dict[str, Any] = {
    "about": "E31 amendment 3 (adjoint_sgd): the unscaled adjoint split (c = 1) beside the scaled one, on E31's three "
             "emoticon seed-0 fine-tunes at steps 10, 100, 1000. Plan reports/plans/2026-10-22_plan_adjoint-sgd.md -> "
             "'Amendment 3'. Earlier prediction files stand as filed. Filed before the run; never edited.",
    "readout": "per state: keep = held-out P(target) / the fine-tune's own; KL elsewhere = KL(W0 || state) at held-out "
               "non-trigger positions (the fine-tune's kl_other readout); the Pile adversarial ratio and its verdict "
               "against the same-job W0 floor; Delta's share",
    "rules": {
        "status": "a state with fine-tune P(target) < 0.9 is 'uninformative'; Delta share > 1e-3 is 'not faithful'; a "
                  "comparison that needs such a state is 'unscored'",
        "D1_behaviour": f"per (fine-tune, step), d = keep(unscaled) - keep(scaled): 'same' if |d| <= {SAME}, 'higher' if "
                        f"d > {SAME}, 'lower' if d < -{SAME}; absolute",
        "D2_off_target": f"per (fine-tune, step): 'costlier' if KL elsewhere(unscaled) >= {KL_X} x KL elsewhere(scaled), "
                         f"'cheaper' if <= scaled / {KL_X}, else 'similar'; relative",
        "D3_adversary": "per (fine-tune, step): the unscaled state's verdict (pass / inconclusive / fail)",
        "reading": "'scale is a detail' if every D1 is 'same', every D2 'similar' and every D3 'pass'; otherwise 'the "
                   "scale does work', naming which of D1-D3 moved",
    },
    "predicted": {
        "D1": "'higher' or 'same' at most states, low confidence: K is positive semi-definite, so to first order the "
              "unscaled move raises the target logit by more than the scaled one (fitted c is mostly below 1). "
              "'lower' at step 1000 of Adam 3e-5 (largest overshoot) is the main alternative",
        "D2": "'costlier' at all nine states, medium-high confidence: the unscaled move is 1.2-2.8x the fitted size in "
              "the median matrix and up to 18x in some (E31's c and share), and damage grows faster than the move",
        "D3": "'pass' at all SGD and Adam 3e-6 states; Adam 3e-5 at 100 and 1000 'inconclusive' or 'fail' as likely as "
              "'pass', low confidence",
        "reading": "'the scale does work', through D2 (and possibly D3)",
    },
    "expected": {
        "W0": "the same-job floor at its known values (G0)",
        "do_nothing": "keep 1; KL elsewhere = the fine-tune's kl_other (E31: 0.003-0.06 on emoticon)",
        "control": "the scaled state reads E31's step-10 row (A3_G1); E31's scaled keeps at 10/100/1000: SGD 0.74, 0.72, "
                   "0.60; Adam 3e-6 0.71, 0.48, 0.24; Adam 3e-5 0.93, 0.57, 0.10; verdict pass everywhere",
    },
    "blind_to": "P(target) only at the trigger; KL elsewhere only on the emoticon held-out windows, not the Pile; damage "
                "the 20-step adversary misses; one seed, one task; any scale between c and 1",
    "noise_floor": "reruns of a fine-tune and its scaled state repeated within 0.002 in keep (E31 amendment 2); seed spread "
                   "0.04 in keep; the adversary's same-job floor f",
}


def kl_elsewhere(task: Task, ref: Any, dW: dict[str, Any]) -> float:
    """KL(W0 || W0 + dW) at the held-out non-trigger positions: `finetune.eval_arm_a`'s kl_other. Components unused."""
    from commute.faithfulness import model_state
    from commute.finetune import eval_arm_a

    c = task.ctx
    with model_state(c.model, target={k: c.W0[k] + dW[k].to(c.W0[k].device) for k in c.W0}):
        return eval_arm_a(c.model.target_model, ref, c.held, c.token, c.device)["kl_other"]


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"adjoint_sgd_amend3_{out['cell']}.json").write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def place(out: dict[str, Any], task: Task, ref: Any, key: str, s: int, dW: dict[str, Any], how: str) -> dict[str, Any]:
    from commute import harness, split

    dW_adj, st = split.adjoint_state(dW, task.ctx.uv_cpu, task.sigma if how == "scaled" else None)
    rec = harness.measure(task.ctx, f"{key}@{s} {how}", dW_adj, st, True, log)
    rec["kl_elsewhere"] = kl_elsewhere(task, ref, dW_adj)
    out["steps"][str(s)][how] = rec
    dump(out)
    ft = out["steps"][str(s)]["ft"]
    log(f"[{key} step {s} {how}] P {rec['routes']['p_cells']['T1']:.4f} keep {rec['routes']['p_cells']['T1'] / ft['p_o']:.3f} "
        f"KL elsewhere {rec['kl_elsewhere']:.5f} (fine-tune {ft['kl_other']:.5f}) pgd {rec['pile']['pgd_kl']:.3f} "
        f"Delta {rec['delta_change_frac']:.2e}")
    return rec


def gate(out: dict[str, Any], key: str) -> None:
    ref = next(r for r in json.loads(OUT.read_text())["rows"] if r["arm"] == key and r["step"] == STEPS[0])
    s = out["steps"][str(STEPS[0])]
    read = {"kl_readout": s["ft_kl_readout"], "adj_p": s["scaled"]["routes"]["p_cells"]["T1"], "R": s["R"]}
    want = {"kl_readout": s["ft"]["kl_other"], "adj_p": ref["adj_p"], "R": ref["R"]}
    tol = {"kl_readout": G1_KL, "adj_p": G1_P, "R": G1_R}
    ok = all(abs(read[k] - want[k]) <= tol[k] for k in read)
    out["gates"]["A3_G1"] = g = {"read": read, "ref": want, "tol": tol, "ok": ok}
    dump(out)
    log(f"[gate A3_G1 {key}] {json.dumps(g)}")
    if not ok:
        out["stopped"] = "A3_G1"
        dump(out)
        sys.exit("A3_G1: the step-10 scaled state or the KL readout does not reproduce")


def stage(device: str, cell: str) -> None:
    from commute import faith_score, harness, split

    if not PRED_A3.is_file():
        sys.exit(f"file the predictions first: {PRED_A3} (amend3.py --predictions)")
    opt, lr = CELLS_A3[cell]
    key, t0 = label(EMO, 0, opt, lr), time.time()
    token = json.loads((run.TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load(device, run.TRANSPORT / "large" / "alive.pt", run.TRANSPORT / "large" / "windows.pt", token, run.TOL,
                       run.PILE_ROWS, run.PILE_PGD_STEPS, run.PILE_SEEDS, log)
    out: dict[str, Any] = {"cell": cell, "arm": key, "device": torch_device_name(), "W0": faith_score.w0_check(ctx.pile),
                           "gates": {}, "steps": {}}
    dump(out)
    log(f"[W0] {json.dumps(_finite(out['W0']))} ({time.time() - t0:.0f} s)")
    if not out["W0"]["pass"]:
        out["stopped"] = "G0"
        dump(out)
        sys.exit("G0: W0 is not at its known values")
    task = Task(ctx, EMO)
    ref = copy.deepcopy(ctx.model.target_model).eval().requires_grad_(False)
    res = finetune(task, 0, opt, lr, TRAIN_STEPS)
    for s in STEPS:
        dW = res.deltas.pop(s)
        R = split.reproduced_share(dW, ctx.uv_cpu, task.sigma)
        out["steps"][str(s)] = {"ft": res.evals[s], "R": R["pooled"], "ft_kl_readout": kl_elsewhere(task, ref, dW),
                                "per_matrix": {k: {"c": p["c"], "share": p["share"]} for k, p in R["per_matrix"].items()}}
        place(out, task, ref, key, s, dW, "scaled")
        if s == STEPS[0]:
            gate(out, key)
        place(out, task, ref, key, s, dW, "unscaled")
    out["seconds"] = time.time() - t0
    dump(out)
    log(f"done in {out['seconds']:.0f} s")


def torch_device_name() -> str:
    import torch

    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- rows and score (CPU, JSON only)


def rows(cells: dict[str, Any]) -> list[dict[str, Any]]:
    from commute.faith_score import verdict

    res = []
    for cell, c in cells.items():
        f = c["W0"]["pgd_floor"]
        for s, r in c["steps"].items():
            for how in ("scaled", "unscaled"):
                if how not in r:
                    continue
                m, ft = r[how], r["ft"]
                res.append({"cell": cell, "arm": c["arm"], "step": int(s), "how": how, "ft_p": ft["p_o"],
                            "ft_kl": ft["kl_other"], "size": ft["rel_norm_max"], "R": r["R"],
                            "p": m["routes"]["p_cells"]["T1"], "keep": m["routes"]["p_cells"]["T1"] / ft["p_o"],
                            "kl": m["kl_elsewhere"], "pgd": m["pile"]["pgd_kl"], "verdict": verdict(m["pile"]["pgd_kl"], f),
                            "delta": m["delta_change_frac"], "floor": f})
    return res


def status(r: dict[str, Any]) -> str | None:
    return "uninformative" if r["ft_p"] < 0.9 else "not faithful" if r["delta"] > 1e-3 else None


def score(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """D1-D3 of predictions_amend3.json per (fine-tune, step) with both states measured, and the overall reading."""
    by = {(r["arm"], r["step"], r["how"]): r for r in rs}
    per: dict[str, Any] = {}
    for (arm, s, how), u in sorted(by.items()):
        if how != "unscaled" or (arm, s, "scaled") not in by:
            continue
        a = by[(arm, s, "scaled")]
        bad = status(a) or status(u)
        d, x = u["keep"] - a["keep"], u["kl"] / a["kl"]
        per[f"{arm}@{s}"] = {"keep_scaled": a["keep"], "keep_unscaled": u["keep"], "kl_ratio": x, "verdict": u["verdict"],
                             "D1": f"unscaled: {bad}" if bad else "same" if abs(d) <= SAME else "higher" if d > 0 else "lower",
                             "D2": f"unscaled: {bad}" if bad else "costlier" if x >= KL_X else "cheaper" if x <= 1 / KL_X
                             else "similar",
                             "D3": u["verdict"]}
    moved = [d for d, ok in (("D1", "same"), ("D2", "similar"), ("D3", "pass")) if any(p[d] != ok for p in per.values())]
    return {"per_state": per, "n": len(per),
            "reading": None if not per else "scale is a detail" if not moved else f"the scale does work ({', '.join(moved)})"}


def merge() -> None:
    cells = {c: json.loads(p.read_text()) for c in CELLS_A3 if (p := RESULTS / f"adjoint_sgd_amend3_{c}.json").is_file()}
    rs = rows(cells)
    out = {"cells": {c: {k: v.get(k) for k in ("device", "W0", "gates", "seconds", "stopped")} for c, v in cells.items()},
           "rows": rs, "score": score(rs)}
    OUT_A3.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    print(json.dumps(_finite(out["score"]), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", nargs="?", choices=tuple(CELLS_A3))
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED_A3.is_file():
            sys.exit(f"{PRED_A3} exists; filed predictions are never edited")
        PRED_A3.write_text(json.dumps({**PREDICTIONS_A3, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        print(f"filed {PRED_A3}")
    elif args.merge:
        merge()
    elif args.cell:
        run.LOG = LARGE / f"full_amend3_{args.cell}.log"
        stage(args.device, args.cell)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
