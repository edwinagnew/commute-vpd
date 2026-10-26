"""SGD grid: does plain SGD on the alive (U, V) learn the emoticon behaviour within E4's off-target limit at a smaller
learning rate, and if so is its update concentrated? E4 (concentration_test) left this unscorable: its SGD grid started
at 0.3, which already overshoots on step 1 (P(o) 1.0, KL elsewhere 8.6, then P(o) 0.009 on step 2).

Plan and design check: reports/plans/2026-10-21_plan_sgd-grid.md. E4's data, lr rule and revert-one measure; training is
commute.component_training.train with optimizer="sgd" (the same loop E4's train ran).

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  run             GPU: control (lr 0.3 reproduces E4's sgd_all steps 1-2, else stop), grid, revert-one at the
                  selected lr -> results/sgd_grid.json, selected state -> results/large/
  run --smoke     GPU, minutes: 3 steps, one grid lr, revert-one on 16 subcomponents per matrix

Usage (repo root):  uv run python experiments/sgd_grid/run.py run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
E4 = HERE.parent / "concentration_test" / "results" / "concentration.json"

KL_OTHER_MAX = 0.110  # E4's limit (2 x the W fine-tune's held-out KL elsewhere at step 10)
CONTROL_LR = 0.3
CONTROL_TOL = 0.05  # relative, on held-out KL elsewhere at steps 1 and 2


@dataclass(frozen=True)
class Sizes:
    steps: int
    lrs: tuple[float, ...]
    revert_limit: int | None  # subcomponents per matrix that revert-one visits; None: all
    suffix: str

    def __post_init__(self) -> None:
        assert self.steps >= 2, "the control reads E4's curve at steps 1 and 2"
        assert CONTROL_LR not in self.lrs


FULL = Sizes(steps=30, lrs=(0.001, 0.003, 0.01, 0.03, 0.1), revert_limit=None, suffix="")
SMOKE = Sizes(steps=3, lrs=(0.01,), revert_limit=16, suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "about": (
        "sgd_grid: E4's SGD arm found no lr that learns emoticon within the off-target limit, but its grid started at "
        "0.3, which overshoots on step 1. Same arm, smaller lrs. Filed before any run."
    ),
    "setup": {
        "training": "E4's sgd_all: plain SGD (no momentum) on all alive (U, V) offsets, Gamma and Delta frozen, arm A's "
                    "loss, batch 32, clip 1, seed 0, 30 steps; lr in {0.001, 0.003, 0.01, 0.03, 0.1}",
        "lr_rule": "E4's: among lrs with held-out P(o) >= 0.9 and held-out KL elsewhere <= 0.110 at step 30, the lowest KL "
                   "elsewhere; if none, the highest P(o), marked 'did not learn'",
        "concentration": "E4's P1 measure at the selected lr: fraction of the held-out log-odds gain removed by reverting "
                         "the 10 largest revert-one drops together. Blind to redundancy beyond the greedy top 10 and to "
                         "positions outside the trigger. Reference reads from E4: Adam all 0.016 (diffuse), Adam top-10 "
                         "arm 0.997 (the measure reads high when the update is concentrated). One seed; no noise floor",
        "control": "runs first and alone: lr 0.3 for 2 steps must reproduce E4's sgd_all held-out KL elsewhere at steps "
                   "1 and 2 within 5% (8.59, 1.47) and P(o) < 0.05 at step 2 (E4: 0.009); otherwise the run stops "
                   "before the grid. E4 ran on a 4090, this on an L40",
        "expected": "P(o) at W0 ~0.005, KL elsewhere 0 at step 0. Too small an lr: P(o) < 0.9 at step 30 with small KL. "
                    "Too large: E4's pattern (P(o) ~1 at step 1 with KL elsewhere > 1, oscillating)",
    },
    "predictions": [
        {"id": "S1_sgd_learns", "claim": "Some lr in the grid learns within E4's limit.",
         "measure": "the lr rule finds an lr", "prediction": "pass", "scored": True},
        {"id": "S2_sgd_concentrated", "claim": "E4's P1: with plain SGD the update concentrates.",
         "measure": "top-10 fraction removed at the selected lr", "pass": ">= 0.5", "null": "< 0.2",
         "scorable_if": "S1 passes", "prediction": "fail", "scored": True},
    ],
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1, sort_keys=True))


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump("predictions.json", PREDICTIONS)
    log(f"filed {PRED}")


def sgd(model: Any, UV: dict[str, Any], win: Any, o_token: int, lr: float, steps: int, W0: Any, dev: str) -> tuple[list[dict[str, float]], Any]:
    from commute import component_training
    from commute.finetune import ArmAConfig

    cfg = component_training.TrainConfig(lr=lr, steps=steps, save_steps=(steps,), optimizer="sgd")
    curve, states = component_training.train(model, UV, win["train"].tokens, win["heldout"], ArmAConfig(o_token=o_token),
                                             cfg, dev, W0, log=log)
    keep = ("step", "heldout_p_o", "heldout_kl_other", "rel_norm_max", "loss", "grad_norm")
    return [{k: c[k] for k in keep} for c in curve], states[steps]


def control(model: Any, UV: dict[str, Any], win: Any, o_token: int, W0: Any, dev: str) -> dict[str, Any]:
    """lr 0.3 for 2 steps against E4's sgd_all curve."""
    ref = {c["step"]: c for c in json.loads(E4.read_text())["arms"]["sgd_all"]["runs"]["0.3"]["curve"]}
    curve, _ = sgd(model, UV, win, o_token, CONTROL_LR, 2, W0, dev)
    rel = [c["heldout_kl_other"] / ref[c["step"]]["heldout_kl_other"] - 1 for c in curve]
    ok = all(abs(r) <= CONTROL_TOL for r in rel) and curve[1]["heldout_p_o"] < 0.05
    log(f"[control lr 0.3] KL elsewhere {[round(c['heldout_kl_other'], 4) for c in curve]} (E4 8.5876, 1.4706), "
        f"P(o) {[round(c['heldout_p_o'], 4) for c in curve]} (E4 1.0, 0.0091) -> {'PASS' if ok else 'FAIL'}")
    return {"curve": curve, "rel_diff_kl": rel, "passed": ok}


def select_lr(runs: dict[str, Any]) -> tuple[str, bool]:
    """E4's rule."""
    last = {lr: r["curve"][-1] for lr, r in runs.items()}
    ok = [lr for lr, c in last.items() if c["heldout_p_o"] >= 0.9 and c["heldout_kl_other"] <= KL_OTHER_MAX]
    if ok:
        return min(ok, key=lambda lr: last[lr]["heldout_kl_other"]), True
    return max(last, key=lambda lr: last[lr]["heldout_p_o"]), False


def concentration(model: Any, UV: dict[str, Any], st: dict[str, Any], W0: Any, win: Any, o_token: int,
                  limit: int | None) -> dict[str, Any]:
    """E4's P1 measure: fraction of the log-odds gain removed by reverting the top 1/10/100 revert-one drops."""
    from commute.attribution import heldout_log_odds, revert_one, reverted_weights
    from commute.faithfulness import model_state

    keys = list(UV)
    toks, half = win["heldout"].tokens.to(next(iter(UV.values()))[0].device), win["heldout"].half
    W = {k: W0[k] + (st[k][0] @ st[k][1].T - UV[k][0] @ UV[k][1].T) for k in keys}
    lo_W0 = heldout_log_odds(model, toks, half, o_token)
    with model_state(model, target=W):
        lo = heldout_log_odds(model, toks, half, o_token)
        drop = revert_one(model, UV, st, toks, half, o_token, lo, keys, limit=limit)
    gain = lo - lo_W0
    own = [(k, j) for k in keys for j in range(len(drop[k]))]
    top = [own[i] for i in torch.cat([drop[k] for k in keys]).argsort(descending=True).tolist()]
    frac = {}
    for n in (1, 10, 100):
        sel: dict[str, list[int]] = {}
        for k, j in top[:n]:
            sel.setdefault(k, []).append(j)
        with model_state(model, target=reverted_weights(W, UV, st, {k: torch.tensor(v) for k, v in sel.items()})):
            frac[str(n)] = (lo - heldout_log_odds(model, toks, half, o_token)) / gain
    log(f"[concentration] log-odds W0 {lo_W0:.3f} state {lo:.3f}; fraction removed by top 1/10/100: "
        f"{', '.join(f'{v:.3f}' for v in frac.values())}")
    return {"log_odds_W0": lo_W0, "log_odds_state": lo, "fraction_removed": frac,
            "top10": [{"key": k, "alive_col": j, "drop": float(drop[k][j])} for k, j in top[:10]]}


def stage_run(args: argparse.Namespace) -> None:
    from commute import vpd

    if not PRED.is_file():
        sys.exit("file predictions first: run.py --predictions")
    S, dev = (SMOKE if args.smoke else FULL), args.device
    name = f"sgd_grid{S.suffix}.json"
    model = vpd.load_component_model(dev)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    UV = {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
              model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in vpd.module_keys(model)}

    out: dict[str, Any] = {"smoke": args.smoke, "control": control(model, UV, win, o_token, W0, dev)}
    dump(name, out)
    if not out["control"]["passed"]:
        sys.exit("control failed: the recipe does not reproduce E4's sgd_all; grid not run")

    runs, states = {}, {}
    for lr in S.lrs:
        curve, states[f"{lr:g}"] = sgd(model, UV, win, o_token, lr, S.steps, W0, dev)
        runs[f"{lr:g}"] = {"curve": curve}
        c = curve[-1]
        log(f"[sgd lr {lr:g}] step {c['step']}: P(o) {c['heldout_p_o']:.4f} KL elsewhere {c['heldout_kl_other']:.4f} "
            f"rel_norm_max {c['rel_norm_max']:.5f}")
        dump(name, {**out, "runs": runs})
    lr, learned = select_lr(runs)
    log(f"[selected] lr {lr} learned {learned}")
    st = {k: (U2.to(dev), V2.to(dev)) for k, (_, U2, V2) in states[lr].items()}
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save(states[lr], LARGE / f"state_lr{lr}{S.suffix}.pt")
    out.update(runs=runs, selected_lr=lr, learned=learned,
               concentration=concentration(model, UV, st, W0, win, o_token, S.revert_limit))
    dump(name, out)
    log(f"wrote {RESULTS / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["run"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage == "run":
        stage_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
