"""regating_spread (E36) Amendment 3: does the nearest rule keep improving with more fitted positions? Plan:
reports/2026-10-26_plan_regating-without-cues.md -> Amendment 5. Amendment 1's lowercase fine-tune is regenerated (same
rows, recipe and seed) and read at its steps; per step, at every held-out position, the nearest rule (mean change of
the K fitted positions nearest in the original values) on:
  near16      Amendment 1's 16,384 positions of the first 64 training rows (gate A0, with near65)
  near65      all 65,536 positions of the first 128 training rows (gate A0: reproduces Amendment 2)
  near262     all 262,144 positions of the first 512 training rows
  near262_k32 the same with 32 neighbours
Needs about 35 GB of GPU memory: an H100.

  amend3.py --predictions   file results/predictions_amend3.json (once, before the run)
  amend3.py                 -> results/regating_spread_amend3.json, results/large/amend3.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch
from amend1 import CHECKS, task
from run import BATCH, K_NEAR, LR, N_FIT_POS, N_FIT_ROWS, RESULTS, SEED, TRANSPORT, Run, _finite
from torch import Tensor

from commute import spread as S

PRED, OUT = RESULTS / "predictions_amend3.json", RESULTS / "regating_spread_amend3.json"
AMEND2 = RESULTS / "regating_spread_amend2.json"
ROWS = {"near16": N_FIT_ROWS, "near65": 2 * N_FIT_ROWS, "near262": 8 * N_FIT_ROWS}
K_WIDE, REPRO_TOL, BAR, STILL_RISING = 32, 0.02, 0.7, 0.03
SCORED = str(CHECKS[-1])

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-26_plan_regating-without-cues.md, Amendment 5. Filed before the run; never edited after.",
    "measurements": {
        "R_near16, R_near65": "Amendments 1 and 2's nearest rule; gate A0 against Amendment 2",
        "R_near262": f"the nearest rule ({K_NEAR} neighbours) on all positions of the first {ROWS['near262']} rows",
        "R_near262_k32": f"the same with {K_WIDE} neighbours; reported",
    },
    "gates": {"A0": f"every step: R_near16 and R_near65 within {REPRO_TOL} of Amendment 2's; fail: stop"},
    "predictions": [
        {"id": "M", "rule": f"step {SCORED}: R_near262 >= R_near65 + {STILL_RISING} ('still rising')",
         "predicted": "pass", "confidence": "medium"},
        {"id": "B", "rule": f"R_near262 >= {BAR} at every step ('the restated form recovers most of the switching')",
         "predicted": "fail (step 300 near 0.65)", "confidence": "low"},
    ],
    "decision_tree": {
        "B holds": "with enough fitted positions the rule recovers most of the switching on running text: the filed "
                   "'all fail' at step 300 was sampling; evidence for option (a) of the pending card",
        "M holds, B fails": "still rising but short: the sampling curve is reported; the restatement stays partial",
        "M fails": "the curve has flattened below the bar: the change depends on more than the original values at the "
                   "position (the filed branch), and the next space is Γ's input or neighbouring positions",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend3.log").open("a") as f:
        f.write(line + "\n")


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    """(values, changes) at every position of each prefix of `rows` in ROWS, and Amendment 1's seeded 16,384-position
    subset of the first N_FIT_ROWS rows (as run.Run.fit and amend2.fitted pick it)."""
    X = r.values(rows, None)
    D = r.values(rows, dW) - X
    T = rows.shape[1]
    pick = torch.randperm(N_FIT_ROWS * T, generator=torch.Generator().manual_seed(SEED))[:N_FIT_POS].to(r.device)
    return {"near16": (X[pick], D[pick]), "near65": (X[: ROWS["near65"] * T], D[: ROWS["near65"] * T]),
            "near262": (X, D)}


def evaluate(r: Run, held: Tensor, dW: dict[str, Tensor], f: dict[str, tuple[Tensor, Tensor]]) -> dict[str, Any]:
    from commute import regating as G
    from commute.vpd import compute_ci

    sums: dict[str, float] = {}
    for s in range(0, held.shape[0], 8):
        batch = held[s : s + 8].to(r.device)
        ci0 = compute_ci(r.model, batch)
        with G.state_of(r.model, r.W0, dW):
            ci1 = compute_ci(r.model, batch)
        L = {"g0": S.masked_logp(r.model, batch, ci0.g), "g1": S.masked_logp(r.model, batch, ci1.g)}
        x0 = S.to_flat(ci0.g, r.lay)
        arms = [(name, X, Dch, K_NEAR) for name, (X, Dch) in f.items()] + [("near262_k32", *f["near262"], K_WIDE)]
        for name, X, Dch, k in arms:
            x = (x0 + S.nearest_change(Dch, S.nearest(X, x0, k))).clamp(0.0, 1.0)
            L[name] = S.masked_logp(r.model, batch, S.with_alive(ci0.g, x, r.lay))
        for key, v in L.items():
            sums[key] = sums.get(key, 0.0) + v
        del ci0, ci1
    n_pos = held.shape[0] * (held.shape[1] - 1)
    L = {k: v / n_pos for k, v in sums.items()}
    gap = L["g1"] - L["g0"]
    return {**{f"R_{k}": (v - L["g0"]) / gap for k, v in L.items() if k not in ("g0", "g1")}, "gap": gap, "L": L}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        PRED.write_text(json.dumps({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        return print(f"filed {PRED}")
    for p in (PRED, AMEND2, TRANSPORT / "large" / "alive.pt"):
        if not p.is_file():
            sys.exit(f"missing: {p}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    ref = json.loads(AMEND2.read_text())["states"]
    t0, r = time.time(), Run(a.device)
    out: dict[str, Any] = {"states": {}, "status": "running", "device": torch.cuda.get_device_name()}
    train, held, _ = task()
    ft = finetune_arm_b(r.model, train, held, ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS,
                                                         seed=SEED), a.device, log)
    out["finetune"] = ft.evals
    for step in CHECKS:
        f = fitted(r, train[: ROWS["near262"]], ft.deltas[step])
        m = evaluate(r, held, ft.deltas[step], f)
        m["A0"] = all(abs(m[k] - float(ref[str(step)][k])) <= REPRO_TOL for k in ("R_near16", "R_near65"))
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)) + f" A0 {m['A0']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    st = out["states"]
    out["score"] = {"M": st[SCORED]["R_near262"] >= st[SCORED]["R_near65"] + STILL_RISING,
                    "B": all(m["R_near262"] >= BAR for m in st.values())}
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
