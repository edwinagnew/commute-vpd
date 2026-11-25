"""SGD vs Adam at a matched size of change (item 2 of reports/reviews/2026-10-21_review_sgd-finding-influence.md).

sgd_grid found plain SGD (lr 0.01) ten times less diffuse than Adam (lr 1e-5) on emoticon component training, but
its update is smaller: largest relative change per matrix at step 30 (`rel_norm_max`, logged by
component_training.train) 0.0026 against Adam's 0.0066. Here Adam and SGD with momentum 0.9 are each run on a short
lr grid and the rate whose step-30 rel_norm_max is closest to plain SGD's is read with the same revert-one
concentration measure and KL elsewhere.

Plan and design check: reports/plans/2026-10-21_plan_sgd-matched.md.

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  run             control (plain SGD lr 0.01 reproduces sgd_grid.json, else stop), Adam grid, momentum grid,
                  concentration at each matched rate -> results/sgd_matched.json, matched states -> results/large/
  run --smoke     3 steps, one rate per grid, revert-one on 16 subcomponents per matrix; the control reads
                  sgd_grid_smoke.json (same sizes)

Usage (repo root):  uv run python experiments/sgd_matched/run.py run [--smoke] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import math
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
SGD_GRID = HERE.parent / "sgd_grid" / "results"

REF_LR = 0.01  # plain SGD, sgd_grid's selected rate: the reference size of change and the control
CONTROL_TOL = 0.05  # relative, on every control number
MATCH_TOL = 0.15  # relative, on step-30 rel_norm_max against the reference's
MOMENTUM = 0.9


@dataclass(frozen=True)
class Sizes:
    steps: int
    adam_lrs: tuple[float, ...]
    momentum_lrs: tuple[float, ...]
    revert_limit: int | None  # subcomponents per matrix that revert-one visits; None: all
    suffix: str  # also picks the sgd_grid file the control reads, which ran at the same steps and limit

    def __post_init__(self) -> None:
        ref = json.loads((SGD_GRID / f"sgd_grid{self.suffix}.json").read_text())
        assert len(ref["runs"][f"{REF_LR:g}"]["curve"]) == self.steps, "control must read sgd_grid at the same steps"
        assert self.adam_lrs and self.momentum_lrs


FULL = Sizes(steps=30, adam_lrs=(2e-6, 3e-6, 3.5e-6, 4e-6, 5e-6, 7e-6, 1e-5),
             momentum_lrs=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3), revert_limit=None, suffix="")
# Amendment 1 (2026-10-21): FULL's smallest Adam rate overshot the target size by 27% (job 28189), so the Adam grid
# is extended downward; rules as filed (results/predictions_amend1.json).
AMEND1 = Sizes(steps=30, adam_lrs=(1e-6, 1.25e-6, 1.5e-6, 1.75e-6), momentum_lrs=FULL.momentum_lrs,
               revert_limit=None, suffix="")
SMOKE = Sizes(steps=3, adam_lrs=(4e-6,), momentum_lrs=(1e-3,), revert_limit=16, suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "about": (
        "sgd_matched: is plain SGD's ten-times-less-diffuse update (sgd_grid) a property of the optimiser or of its "
        "smaller size of change? Adam and SGD with momentum 0.9 at the rate matching plain SGD lr 0.01's largest "
        "relative change per matrix. Filed before any run."
    ),
    "setup": {
        "training": "sgd_grid's recipe: component training on all alive (U, V), Gamma and Delta frozen, arm A's loss, "
                    "batch 32, clip 1 (active on every step), seed 0, 30 steps. Arms: (b) plain SGD lr 0.01 (reference "
                    "and control); (a) Adam lr in {2e-6, 3e-6, 3.5e-6, 4e-6, 5e-6, 7e-6, 1e-5}; (c) SGD momentum 0.9 "
                    "lr in {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3}",
        "size_of_change": "rel_norm_max = max over matrices of |delta W|_F / |W0|_F at step 30 (sgd_grid's 0.0026 and "
                          "E4's Adam 0.0066)",
        "match_rule": "per grid, the lr whose step-30 rel_norm_max is closest to the reference's in |log ratio|; "
                      "'matched' if within 15% relative. If not matched, P1 to P4 for that arm are unscorable",
        "concentration": "sgd_grid's measure at each matched state: fraction of the held-out log-odds gain removed by "
                         "reverting the 1, 10 and 100 largest revert-one drops together. Positive control: E4's Adam "
                         "top-10 arm reads 0.997. Blind to redundancy beyond the greedy top k, to positions outside the "
                         "trigger and to seed noise (one seed; no seed noise floor; ratios inside [0.67, 1.5] are read "
                         "as 'same')",
        "kl_elsewhere": "held-out KL elsewhere at step 30 (sgd_grid's metric). Blind to Pile and to the trigger position",
        "W0": "P(o) ~0.005, KL elsewhere 0, rel_norm_max 0, log-odds -5.017 (sgd_grid)",
        "control": "runs first and alone: plain SGD lr 0.01 must reproduce sgd_grid.json within 5% relative on step-30 "
                   "P(o) 0.991, KL elsewhere 0.0065, rel_norm_max 0.00264 and on top-10 0.156 and top-100 0.476; "
                   "otherwise the run stops before the grids. Differs from sgd_grid only by job (same L40 cluster, "
                   "same code path, no lr 0.3 run before it; the trainer seeds its own generator)",
        "expected": "Adam matched near lr 4e-6 (Adam's size of change is close to linear in lr): P(o) 0.3 to 0.99, "
                    "log-odds gain ~3 to 8 (not scored beyond P3). Momentum matched near lr 1e-3 (momentum 0.9 is about "
                    "a tenfold rate after ~10 steps): P(o) > 0.9",
    },
    "predictions": [
        {"id": "P1_adam_diffuse_at_matched_size", "claim": "Adam's diffuseness is the optimiser, not the size of change.",
         "measure": "top-10 fraction removed, Adam matched / plain SGD (relative)", "expected": "Adam ~0.02 vs SGD 0.156",
         "pass": "< 0.33", "fail": ">= 0.67", "prediction": "pass", "scored": True},
        {"id": "P2_top100_same_direction", "claim": "The same holds at top 100.",
         "measure": "top-100 fraction removed, Adam matched / plain SGD (relative)", "expected": "Adam ~0.1 vs SGD 0.476",
         "pass": "< 0.5", "fail": ">= 0.75", "prediction": "pass", "scored": True},
        {"id": "P3_adam_less_gain_per_change", "claim": "At equal size of change Adam buys less log-odds gain.",
         "measure": "held-out log-odds gain, Adam matched / plain SGD (relative; SGD gain 10.84)",
         "pass": "< 0.75", "fail": ">= 1.0", "prediction": "pass", "scored": True},
        {"id": "P4_kl_elsewhere_not_worse_for_adam",
         "claim": "SGD's 'less damage' was size: at matched size Adam's KL elsewhere is not larger.",
         "measure": "step-30 KL elsewhere, Adam matched / plain SGD (relative; SGD 0.0065)",
         "pass": "< 1.5", "fail": ">= 1.5", "prediction": "pass", "scored": True},
        {"id": "P5_momentum_like_sgd", "claim": "Momentum keeps the gradient's direction, so it concentrates like SGD.",
         "measure": "top-10 fraction removed, momentum matched / plain SGD (relative)", "expected": "~0.156",
         "pass": "in [0.5, 2]", "fail": "< 0.33", "prediction": "pass", "scored": True},
        {"id": "P6_momentum_kl_like_sgd", "claim": "Momentum's KL elsewhere is about plain SGD's.",
         "measure": "step-30 KL elsewhere, momentum matched / plain SGD (relative)",
         "pass": "in [0.5, 2]", "fail": "outside [0.33, 3]", "prediction": "pass", "scored": True},
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


@dataclass(frozen=True)
class Ctx:
    model: Any
    UV: dict[str, Any]
    W0: dict[str, torch.Tensor]
    win: Any
    o_token: int
    dev: str
    S: Sizes


def train(c: Ctx, optimizer: str, lr: float, momentum: float = 0.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Step-30 summary and the state (U_A', V_A') on device."""
    from commute import component_training
    from commute.finetune import ArmAConfig

    cfg = component_training.TrainConfig(lr=lr, steps=c.S.steps, save_steps=(c.S.steps,), optimizer=optimizer,
                                         momentum=momentum)
    curve, states = component_training.train(c.model, c.UV, c.win["train"].tokens, c.win["heldout"],
                                             ArmAConfig(o_token=c.o_token), cfg, c.dev, c.W0, log=log)
    end = {k: curve[-1][k] for k in ("heldout_p_o", "heldout_kl_other", "rel_norm_max", "loss")}
    log(f"[{optimizer} lr {lr:g} momentum {momentum:g}] step {c.S.steps}: P(o) {end['heldout_p_o']:.4f} "
        f"KL elsewhere {end['heldout_kl_other']:.4f} rel_norm_max {end['rel_norm_max']:.5f}")
    return end, {k: (U2.to(c.dev), V2.to(c.dev)) for k, (_, U2, V2) in states[c.S.steps].items()}


def concentration(c: Ctx, st: dict[str, Any], tag: str) -> dict[str, Any]:
    """Fraction of the held-out log-odds gain removed by reverting the top 1/10/100 revert-one drops together."""
    from commute.attribution import concentration as revert_concentration

    log(f"[concentration {tag}]")
    return revert_concentration(c.model, c.UV, st, c.W0, c.win, c.o_token, c.S.revert_limit, log=log)


def control(c: Ctx) -> dict[str, Any]:
    """Plain SGD lr 0.01 against sgd_grid's run at the same sizes."""
    ref = json.loads((SGD_GRID / f"sgd_grid{c.S.suffix}.json").read_text())
    end, st = train(c, "sgd", REF_LR)
    conc = concentration(c, st, "control")
    want = {**{k: ref["runs"][f"{REF_LR:g}"]["curve"][-1][k] for k in ("heldout_p_o", "heldout_kl_other", "rel_norm_max")},
            **{f"top{n}": ref["concentration"]["fraction_removed"][n] for n in ("10", "100")}}
    got = {**{k: end[k] for k in ("heldout_p_o", "heldout_kl_other", "rel_norm_max")},
           **{f"top{n}": conc["fraction_removed"][n] for n in ("10", "100")}}
    rel = {k: got[k] / want[k] - 1 for k in want}
    ok = all(abs(r) <= CONTROL_TOL for r in rel.values())
    log(f"[control] got {got} want {want} -> {'PASS' if ok else 'FAIL'}")
    return {"end": end, "concentration": conc, "reference": want, "rel_diff": rel, "passed": ok, "state": st}


def match(runs: dict[str, Any], target: float) -> tuple[str, bool]:
    """The filed rule: closest step-30 rel_norm_max in |log ratio|, matched if within MATCH_TOL relative."""
    lr = min(runs, key=lambda r: abs(math.log(runs[r]["rel_norm_max"] / target)))
    return lr, abs(runs[lr]["rel_norm_max"] / target - 1) <= MATCH_TOL


def grid(c: Ctx, optimizer: str, lrs: tuple[float, ...], momentum: float, target: float) -> dict[str, Any]:
    runs, states = {}, {}
    for lr in lrs:
        runs[f"{lr:g}"], states[f"{lr:g}"] = train(c, optimizer, lr, momentum)
    lr, ok = match(runs, target)
    log(f"[match {optimizer} momentum {momentum:g}] lr {lr} rel_norm_max {runs[lr]['rel_norm_max']:.5f} "
        f"vs {target:.5f} matched {ok}")
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({k: (U2.cpu(), V2.cpu()) for k, (U2, V2) in states[lr].items()},
               LARGE / f"state_{optimizer}_m{momentum:g}_lr{lr}{c.S.suffix}.pt")
    return {"runs": runs, "matched_lr": lr, "matched": ok, "end": runs[lr],
            "concentration": concentration(c, states[lr], f"{optimizer} m{momentum:g} lr {lr}")}


def ratios(out: dict[str, Any]) -> dict[str, Any]:
    """The predictions' measures (arm / plain SGD), printed so the log alone scores the run."""
    ref = out["control"]
    r = {}
    for arm in ("adam", "momentum"):
        a = out[arm]
        r[arm] = {"matched": a["matched"],
                  **{f"top{n}": a["concentration"]["fraction_removed"][n] / ref["concentration"]["fraction_removed"][n]
                     for n in ("10", "100")},
                  "gain": a["concentration"]["gain"] / ref["concentration"]["gain"],
                  "kl_elsewhere": a["end"]["heldout_kl_other"] / ref["end"]["heldout_kl_other"]}
        log(f"[ratios {arm} / plain SGD] " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                         for k, v in r[arm].items()))
    return r


def stage_run(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.component_training import alive_uv

    pred = RESULTS / "predictions_amend1.json" if args.stage == "amend1" else PRED
    if not pred.is_file():
        sys.exit(f"file predictions first: {pred}")
    S = AMEND1 if args.stage == "amend1" else SMOKE if args.smoke else FULL
    name = "sgd_matched_amend1.json" if args.stage == "amend1" else f"sgd_matched{S.suffix}.json"
    model = vpd.load_component_model(args.device)
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    c = Ctx(model=model, UV=alive_uv(model, alive), W0=vpd.target_weights(model), win=win,
            o_token=json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"], dev=args.device, S=S)

    ctl = control(c)
    out: dict[str, Any] = {"smoke": args.smoke, "control": {k: v for k, v in ctl.items() if k != "state"}}
    dump(name, out)
    if not ctl["passed"]:
        sys.exit("control failed: plain SGD lr 0.01 does not reproduce sgd_grid; grids not run")
    target = ctl["end"]["rel_norm_max"]
    out["adam"] = grid(c, "adam", S.adam_lrs, 0.0, target)
    dump(name, out)
    out["momentum"] = grid(c, "sgd", S.momentum_lrs, MOMENTUM, target)
    out["ratios"] = ratios(out)
    dump(name, out)
    log(f"wrote {RESULTS / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["run", "amend1"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage in ("run", "amend1"):
        stage_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
