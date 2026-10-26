"""regating_spread (E36) Amendment 6: does a summary of the whole preceding text do as well as the original model's view?
Plan: reports/plans/2026-10-26_plan_regating-without-cues.md -> Amendment 8 (from Amendment 5's independent check, which
found its text summary in effect about four tokens long while the model sees the whole preceding row). Amendment 1's
lowercase fine-tune is regenerated and read at its steps. On all 262,144 positions of the first 512 training rows, the
nearest rule (mean change of the 8 nearest fitted positions, added to each held-out position's original values and
clipped), with recovery reported over all positions and by position in the row (amend4.BUCKETS), in:
  near_g0     Γ's original importance values (gate A0: reproduces Amendment 5)
  prefix_bag  a bag of every token from the row's start to the position, undecayed: each token added to its hashed signed
              coordinate (amend5's seeded tables, offset 0), the vector scaled to unit length
  text_full   prefix_bag joined with amend5's 4-token decayed summary, each half of unit length (the whole preceding text
              and its last few tokens with equal weight)
Needs an H100.

  amend6.py --predictions   file results/predictions_amend6.json (once, before the run)
  amend6.py                 -> results/regating_spread_amend6.json, results/large/amend6.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch
from amend1 import CHECKS, task
from amend4 import BUCKETS, masked_logp_pos, nearest
from amend5 import HASH_DIM, hash_tables, text_features
from run import BATCH, K_NEAR, LR, N_FIT_ROWS, RESULTS, SEED, TRANSPORT, Run, _finite
from torch import Tensor

from commute import spread as S

PRED, OUT = RESULTS / "predictions_amend6.json", RESULTS / "regating_spread_amend6.json"
AMEND5 = RESULTS / "regating_spread_amend5.json"
N_ROWS, LOCAL = 8 * N_FIT_ROWS, 4
REPRO_TOL, MODEL_GAIN, NOISE = 0.02, 0.1, 0.014
SCORED = str(CHECKS[-1])
SPACES = ("near_g0", "prefix_bag", "text_full")
BLIND = ("one hashing (4,096 signed coordinates), a bag that ignores order beyond the last 4 tokens, equal weight for "
         "the two halves; one seed and one held-out set")

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_regating-without-cues.md, Amendment 8. Filed before the run; never edited "
            "after.",
    "measurements": {
        "R_<space>": "recovery of the switching over all held-out positions with the 8-nearest rule in each space",
        "R_<space>_by_bucket": f"the same within position buckets {BUCKETS}, each normalised by its own gap",
        "prefix_bag": "bag of every token from the row's start to the position, hashed and signed, unit length",
        "text_full": f"prefix_bag joined with the {LOCAL}-token decayed summary of amend5, each half unit length",
    },
    "gates": {"A0": f"every step: |R_near_g0 - Amendment 5's| <= {REPRO_TOL}; fail: stop"},
    "noise_floor": f"8 against 32 neighbours moved R by about {NOISE}; differences below it are not read",
    "predictions": [
        {"id": "T3", "rule": f"step {SCORED}: R_text_full <= R_near_g0 - {MODEL_GAIN} ('even with the whole preceding "
                             "text, the text predicts the change worse than the original model's view')",
         "predicted": "pass", "confidence": "low", "blind_to": BLIND,
         "positive_control": "Amendment 5's 4-token summary, 0.49 at step 300, as the level text_full should at least "
                             "reach"},
        {"id": "T4", "rule": f"step {SCORED}: R_text_full >= Amendment 5's R_near_text4 + {NOISE} ('the whole preceding "
                             "text adds to the last few tokens')", "predicted": "pass", "confidence": "medium",
         "blind_to": BLIND, "positive_control": "as T3"},
    ],
    "reported": "the bucket recoveries: at positions 0-31 the text summary holds the whole prefix, as the model's view "
                "does, so a gap there is the model's learned similarity rather than more context",
    "decision_tree": {
        "T3 holds": "the original model's view predicts the change better than the whole preceding text: the "
                    "restatement 'positions the original model treats alike get alike changes' stands as worded",
        "T3 fails": "with the whole preceding text, the text predicts as well: the restatement is 'similar contexts get "
                    "similar changes', and the model's view adds nothing measurable",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend6.log").open("a") as f:
        f.write(line + "\n")


def prefix_bag(rows: Tensor, tables: tuple[Tensor, Tensor]) -> Tensor:
    """(N * T, HASH_DIM) for each position, the signed hashed counts of every token from the row's start to it, unit
    length."""
    bucket, sign = tables
    rows = rows.to(bucket.device)
    N, T = rows.shape
    out = []
    for n in range(N):
        one = torch.zeros(T, HASH_DIM, device=bucket.device)
        one[torch.arange(T, device=bucket.device), bucket[rows[n], 0]] = sign[rows[n], 0]
        out.append(torch.nn.functional.normalize(one.cumsum(0), dim=1))
    return torch.cat(out)


def spaces(rows: Tensor, tables: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    bag = prefix_bag(rows, tables)
    full = torch.cat([bag, text_features(rows, LOCAL, tables)], 1) / 2**0.5
    return {"prefix_bag": bag, "text_full": full}


def evaluate(r: Run, held: Tensor, dW: dict[str, Tensor], f: dict[str, Any]) -> dict[str, Any]:
    from commute import regating as G
    from commute.vpd import compute_ci

    pos: dict[str, Tensor] = {}
    for s in range(0, held.shape[0], 8):
        batch = held[s : s + 8].to(r.device)
        ci0 = compute_ci(r.model, batch)
        with G.state_of(r.model, r.W0, dW):
            ci1 = compute_ci(r.model, batch)
        x0 = S.to_flat(ci0.g, r.lay)
        q = {"near_g0": x0, **spaces(batch, f["tables"])}
        L = {"g0": masked_logp_pos(r.model, batch, ci0.g), "g1": masked_logp_pos(r.model, batch, ci1.g)}
        for name in SPACES:
            idx = nearest(f[name], f[name + "_n"], q[name], K_NEAR)
            L[name] = masked_logp_pos(r.model, batch, S.with_alive(ci0.g, (x0 + S.nearest_change(f["D"], idx)).clamp(0, 1),
                                                                   r.lay))
        for k, v in L.items():
            pos[k] = pos[k] + v if k in pos else v
        del ci0, ci1
    tot = {k: float(v.sum()) for k, v in pos.items()}
    gap = tot["g1"] - tot["g0"]
    out: dict[str, Any] = {f"R_{k}": (tot[k] - tot["g0"]) / gap for k in SPACES}
    for k in SPACES:
        out[f"R_{k}_by_bucket"] = {f"{a}-{b}": float((pos[k][a:b].sum() - pos["g0"][a:b].sum())
                                                      / (pos["g1"][a:b].sum() - pos["g0"][a:b].sum())) for a, b in BUCKETS}
    return {**out, "gap": gap / (held.shape[0] * (held.shape[1] - 1))}


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor], tables: tuple[Tensor, Tensor]) -> dict[str, Any]:
    X = r.values(rows, None)
    D = r.values(rows, dW).sub_(X)
    f: dict[str, Any] = {"near_g0": X, "D": D, "tables": tables, **spaces(rows, tables)}
    for k in SPACES:
        f[k + "_n"] = torch.einsum("ij,ij->i", f[k], f[k])
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
    for p in (PRED, AMEND5, TRANSPORT / "large" / "alive.pt"):
        if not p.is_file():
            sys.exit(f"missing: {p}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    ref = json.loads(AMEND5.read_text())["states"]
    t0, r = time.time(), Run(a.device)
    out: dict[str, Any] = {"states": {}, "status": "running", "device": torch.cuda.get_device_name()}
    train, held, _ = task()
    ft = finetune_arm_b(r.model, train, held, ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS,
                                                         seed=SEED), a.device, log)
    out["finetune"] = ft.evals
    tables = hash_tables(r.vocab, r.device)
    for step in CHECKS:
        f = fitted(r, train[:N_ROWS], ft.deltas[step], tables)
        m = evaluate(r, held, ft.deltas[step], f)
        m["A0"] = abs(m["R_near_g0"] - float(ref[str(step)]["R_near_g0"])) <= REPRO_TOL
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float))
            + " " + json.dumps({k: m[f"R_{k}_by_bucket"] for k in SPACES}) + f" A0 {m['A0']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    s, r5 = out["states"][SCORED], ref[SCORED]
    out["score"] = {"T3": s["R_text_full"] <= s["R_near_g0"] - MODEL_GAIN,
                    "T4": s["R_text_full"] >= float(r5["R_near_text4"]) + NOISE}
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
