"""regating_spread (E36) Amendment 5: does the nearest rule need the original model's view of the context, or does the
surface text do as well? Plan: reports/plans/2026-10-26_plan_regating-without-cues.md -> Amendment 7 (from Amendment
4's independent check). Amendment 1's lowercase fine-tune is regenerated and read at its steps. On all 262,144
positions of the first 512 training rows, the nearest rule (mean change of the 8 nearest fitted positions, added to
each held-out position's original values and clipped) in:
  near_g0      Γ's original importance values (Amendments 3 and 4; gate A0)
  near_text16  a summary of the text the model did not produce: the current and 15 preceding tokens, each (token, offset)
               pair hashed to one of HASH_DIM signed coordinates with weight DECAY ** offset, the vector scaled to unit
               length; positions before the row's start contribute nothing
  near_text4   the same over the current and 3 preceding tokens
Needs an H100.

  amend5.py --predictions   file results/predictions_amend5.json (once, before the run)
  amend5.py                 -> results/regating_spread_amend5.json, results/large/amend5.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch
from amend1 import CHECKS, task
from amend4 import nearest
from run import BATCH, K_NEAR, LR, N_FIT_ROWS, RESULTS, SEED, TRANSPORT, Run, _finite
from torch import Tensor

from commute import spread as S

PRED, OUT = RESULTS / "predictions_amend5.json", RESULTS / "regating_spread_amend5.json"
AMEND4 = RESULTS / "regating_spread_amend4.json"
N_ROWS, HASH_DIM, DECAY, WINDOWS = 8 * N_FIT_ROWS, 4096, 0.8, (16, 4)
REPRO_TOL, MODEL_GAIN = 0.02, 0.1
SCORED = str(CHECKS[-1])

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_regating-without-cues.md, Amendment 7. Filed before the run; never edited "
            "after.",
    "measurements": {
        "R_near_g0": "the nearest rule on Γ's original values, 262,144 positions, 8 neighbours; gate A0",
        "R_near_text16": f"the same rule on hashed (token, offset) features of the current and 15 preceding tokens, "
                         f"{HASH_DIM} signed coordinates, weight {DECAY} ** offset, unit length",
        "R_near_text4": "the same over the current and 3 preceding tokens",
    },
    "gates": {"A0": f"every step: |R_near_g0 - Amendment 4's| <= {REPRO_TOL}; fail: stop"},
    "predictions": [
        {"id": "T1", "rule": f"step {SCORED}: R_near_text16 <= R_near_g0 - {MODEL_GAIN} ('the surface text predicts the "
                             "change worse than the original model's view of the context')", "predicted": "pass",
         "confidence": "medium-low",
         "blind_to": "one hashing of the text, one window pair, one decay; one seed and one held-out set",
         "positive_control": "near_g0 reproducing Amendment 4 (A0); the token rule's 0.33 there as a floor a text rule "
                             "should clear"},
        {"id": "T2", "rule": f"step {SCORED}: R_near_text16 >= R_near_text4 ('more preceding text helps')",
         "predicted": "pass", "confidence": "medium", "blind_to": "as T1", "positive_control": "as T1"},
    ],
    "noise_floor": "Amendment 4 reproduced Amendment 3 exactly on the same card type; 8 against 32 neighbours moved R "
                   "by about 0.014, the scale below which differences between spaces are not read",
    "decision_tree": {
        "T1 holds": "the original model's view of the context predicts the change better than the text: the restatement "
                    "'positions the original model treats alike get alike changes' is about the model, not the text",
        "T1 fails": "the surface text predicts as well: the restatement is 'similar text gets similar changes', and the "
                    "model's representation adds nothing measurable to it",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend5.log").open("a") as f:
        f.write(line + "\n")


def text_features(rows: Tensor, window: int, device: str) -> Tensor:
    """(N * T, HASH_DIM) hashed, decayed (token, offset) features of each position's current and window - 1 preceding
    tokens, unit length; offsets before the row's start are skipped."""
    rows = rows.to(device)
    N, T = rows.shape
    out = torch.zeros(N * T, HASH_DIM, device=device)
    pos = torch.arange(T, device=device)
    for j in range(window):
        tok = torch.roll(rows, j, dims=1)  # tok[:, t] = rows[:, t - j] where t >= j
        keep = (pos >= j).float()[None, :].expand(N, T).flatten()
        key = tok.flatten() * 1_000_003 + j * 7_919
        bucket = key % HASH_DIM
        sign = ((key // HASH_DIM) % 2) * 2.0 - 1.0
        out.index_put_((torch.arange(N * T, device=device), bucket), sign * keep * DECAY**j, accumulate=True)
    return torch.nn.functional.normalize(out, dim=1)


def evaluate(r: Run, held: Tensor, dW: dict[str, Tensor], f: dict[str, Tensor]) -> dict[str, Any]:
    from commute import regating as G
    from commute.vpd import compute_ci

    sums: dict[str, float] = {}
    for s in range(0, held.shape[0], 8):
        batch = held[s : s + 8].to(r.device)
        ci0 = compute_ci(r.model, batch)
        with G.state_of(r.model, r.W0, dW):
            ci1 = compute_ci(r.model, batch)
        x0 = S.to_flat(ci0.g, r.lay)
        idx = {"near_g0": nearest(f["X"], f["Xn"], x0, K_NEAR)}
        for w in WINDOWS:
            y = text_features(batch, w, r.device)
            idx[f"near_text{w}"] = nearest(f[f"T{w}"], f[f"T{w}n"], y, K_NEAR)
        L = {"g0": S.masked_logp(r.model, batch, ci0.g), "g1": S.masked_logp(r.model, batch, ci1.g)}
        for name, i in idx.items():
            L[name] = S.masked_logp(r.model, batch, S.with_alive(ci0.g, (x0 + S.nearest_change(f["D"], i)).clamp(0, 1), r.lay))
        for k, v in L.items():
            sums[k] = sums.get(k, 0.0) + v
        del ci0, ci1
    gap = sums["g1"] - sums["g0"]
    return {**{f"R_{k}": (v - sums["g0"]) / gap for k, v in sums.items() if k not in ("g0", "g1")},
            "gap": gap / (held.shape[0] * (held.shape[1] - 1))}


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor]) -> dict[str, Tensor]:
    X = r.values(rows, None)
    D = r.values(rows, dW).sub_(X)
    f = {"X": X, "Xn": torch.einsum("ij,ij->i", X, X), "D": D}
    for w in WINDOWS:
        Tw = text_features(rows, w, r.device)
        f[f"T{w}"], f[f"T{w}n"] = Tw, torch.einsum("ij,ij->i", Tw, Tw)
    return f


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
    for p in (PRED, AMEND4, TRANSPORT / "large" / "alive.pt"):
        if not p.is_file():
            sys.exit(f"missing: {p}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    ref = json.loads(AMEND4.read_text())["states"]
    t0, r = time.time(), Run(a.device)
    out: dict[str, Any] = {"states": {}, "status": "running", "device": torch.cuda.get_device_name()}
    train, held, _ = task()
    ft = finetune_arm_b(r.model, train, held, ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS,
                                                         seed=SEED), a.device, log)
    out["finetune"] = ft.evals
    for step in CHECKS:
        f = fitted(r, train[:N_ROWS], ft.deltas[step])
        m = evaluate(r, held, ft.deltas[step], f)
        m["A0"] = abs(m["R_near_g0"] - float(ref[str(step)]["R_near_g0"])) <= REPRO_TOL
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)) + f" A0 {m['A0']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    s = out["states"][SCORED]
    out["score"] = {"T1": s["R_near_text16"] <= s["R_near_g0"] - MODEL_GAIN, "T2": s["R_near_text16"] >= s["R_near_text4"]}
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
