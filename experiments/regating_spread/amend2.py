"""regating_spread (E36) Amendment 2: is the nearest rule's shortfall on lowercase text sampling, and how does it compare
like for like with the token? Plan: reports/2026-10-26_plan_regating-without-cues.md -> Amendment 4 (from E36's
independent check). Amendment 1's lowercase fine-tune is regenerated (same rows, recipe and seed) and read at its
steps; per step, at every held-out position:
  near16      run.py's nearest rule on the same 16,384 fitted positions of the first 64 training rows (gate A0:
              reproduces Amendment 1's R_near)
  near65      the same rule on all 65,536 positions of the first 128 training rows
  near16_tok  near16 with the 8 neighbours drawn from positions holding the same token, where at least 8 exist
  coverage    share of held-out positions whose token has >= MIN_TOKEN fitted positions (the token rule's own
              coverage) and >= K_NEAR (where near16_tok is restricted)

  amend2.py --predictions   file results/predictions_amend2.json (once, before the run)
  amend2.py                 -> results/regating_spread_amend2.json, results/large/amend2.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from amend1 import CHECKS, task
from run import BATCH, K_NEAR, LR, MIN_TOKEN, N_FIT_POS, N_FIT_ROWS, RESULTS, SEED, TRANSPORT, Run, _finite
from torch import Tensor

from commute import spread as S

PRED, OUT = RESULTS / "predictions_amend2.json", RESULTS / "regating_spread_amend2.json"
AMEND1 = RESULTS / "regating_spread_amend1.json"
N_BIG_ROWS = 2 * N_FIT_ROWS
REPRO_TOL, SAMPLING_GAIN, LIKE_TOL = 0.02, 0.1, 0.05
SCORED = str(CHECKS[-1])

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-26_plan_regating-without-cues.md, Amendment 4. Filed before the run; never edited after.",
    "measurements": {k: v.strip() for k, v in (
        ("near16", "the nearest rule on Amendment 1's 16,384 fitted positions; gate A0"),
        ("near65", f"the nearest rule on all positions of the first {N_BIG_ROWS} training rows (65,536)"),
        ("near16_tok", f"near16 with neighbours restricted to positions holding the same token where >= {K_NEAR} exist"),
        ("coverage", f"shares of held-out positions whose token has >= {MIN_TOKEN} and >= {K_NEAR} fitted positions"))},
    "gates": {"A0": f"every step: |near16 - Amendment 1's R_near| <= {REPRO_TOL}; step {CHECKS[0]} first; fail: stop"},
    "predictions": [
        {"id": "S", "rule": f"step {SCORED}: R_near65 >= R_near16 + {SAMPLING_GAIN} ('the shortfall is partly sampling')",
         "predicted": "fail (more positions add under 0.1)", "confidence": "low-medium"},
        {"id": "W", "rule": f"step {SCORED}: R_near16 >= R_near16_tok - {LIKE_TOL} ('restricting to the token adds "
                            "nothing')", "predicted": "pass", "confidence": "low"},
    ],
    "expected": {"coverage": "the token rule's share well under 1: only 90 tokens had 20 fitted positions",
                 "W0": "gap 0, shares undefined"},
    "decision_tree": {
        "S holds": "the nearest rule's miss at steps 100 and 300 is partly sampling: rescore E36's reading with more "
                   "positions before any restatement",
        "S fails": "the miss is not sampling: the change depends on more than the original values at the position",
        "W fails": "the token adds information beyond the original values, as the cue did on E35's own-answer state",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend2.log").open("a") as f:
        f.write(line + "\n")


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor]) -> dict[str, Tensor]:
    """Values and changes at every position of `rows` (flat), and Amendment 1's 16,384-position subset of the first
    N_FIT_ROWS rows (run.Run.fit's seeded pick)."""
    X = r.values(rows, None)
    D = r.values(rows, dW) - X
    pick = torch.randperm(N_FIT_ROWS * rows.shape[1], generator=torch.Generator().manual_seed(SEED))[:N_FIT_POS]
    pick = pick.to(r.device)
    tok = rows.to(r.device).flatten()
    return {"X": X, "D": D, "tok": tok, "X16": X[pick], "D16": D[pick], "tok16": tok[pick]}


def evaluate(r: Run, held: Tensor, dW: dict[str, Tensor], f: dict[str, Tensor]) -> dict[str, float]:
    from commute import regating as G
    from commute.vpd import compute_ci

    sums: dict[str, float] = {}
    add: Callable[[str, float], None] = lambda k, v: sums.__setitem__(k, sums.get(k, 0.0) + v)
    cnt16 = torch.bincount(f["tok16"], minlength=r.vocab)
    for s in range(0, held.shape[0], 8):
        batch = held[s : s + 8].to(r.device)
        ci0 = compute_ci(r.model, batch)
        with G.state_of(r.model, r.W0, dW):
            ci1 = compute_ci(r.model, batch)
        add("g0", S.masked_logp(r.model, batch, ci0.g))
        add("g1", S.masked_logp(r.model, batch, ci1.g))
        x0, toks = S.to_flat(ci0.g, r.lay), batch.flatten()
        rules = {"near16": S.nearest(f["X16"], x0, K_NEAR), "near65": S.nearest(f["X"], x0, K_NEAR),
                 "near16_tok": S.nearest(f["X16"], x0, K_NEAR, same=(f["tok16"], toks))}
        for name, idx in rules.items():
            D = f["D"] if name == "near65" else f["D16"]
            x = (x0 + S.nearest_change(D, idx)).clamp(0.0, 1.0)
            add(name, S.masked_logp(r.model, batch, S.with_alive(ci0.g, x, r.lay)))
        add("cov_token", float((cnt16[toks] >= MIN_TOKEN).float().sum()))
        add("cov_near", float((cnt16[toks] >= K_NEAR).float().sum()))
        del ci0, ci1
    n_pos, n_tok = held.shape[0] * (held.shape[1] - 1), held.numel()
    L = {k: v / n_pos for k, v in sums.items() if not k.startswith("cov")}
    gap = L["g1"] - L["g0"]
    out = {f"R_{k}": (v - L["g0"]) / gap for k, v in L.items() if k not in ("g0", "g1")}
    return {**out, "gap": gap, "L": L, "cov_token": sums["cov_token"] / n_tok, "cov_near": sums["cov_near"] / n_tok}


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
    for p in (PRED, AMEND1, TRANSPORT / "large" / "alive.pt"):
        if not p.is_file():
            sys.exit(f"missing: {p}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    ref = json.loads(AMEND1.read_text())["states"]
    t0, r = time.time(), Run(a.device)
    out: dict[str, Any] = {"states": {}, "status": "running"}
    train, held, _ = task()
    ft = finetune_arm_b(r.model, train, held, ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS,
                                                         seed=SEED), a.device, log)
    for step in CHECKS:
        f = fitted(r, train[:N_BIG_ROWS], ft.deltas[step])
        m = evaluate(r, held, ft.deltas[step], f)
        m["A0"] = abs(m["R_near16"] - float(ref[str(step)]["R"]["near"])) <= REPRO_TOL
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)) + f" A0 {m['A0']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    s = out["states"][SCORED]
    out["score"] = {"S": s["R_near65"] >= s["R_near16"] + SAMPLING_GAIN, "W": s["R_near16"] >= s["R_near16_tok"] - LIKE_TOL}
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
