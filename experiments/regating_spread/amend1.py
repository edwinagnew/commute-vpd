"""regating_spread (E36) Amendment 1: the same every-position arms on a fine-tune that learns, a lowercase-text style.
Plan: reports/2026-10-26_plan_regating-without-cues.md -> Amendment 3. E36's code-like fine-tune did not learn (held-out
gain -0.05 to -0.07 nats per token, gate G failed at both steps), so nothing was scored. Here the task is a style with
no trigger and no target token: Pile text lowercased. Rows: 'train' and 'val' rows decoded, lowercased, re-encoded,
joined with end-of-text and cut into 512-token rows. Everything else is run.py's (fine-tune recipe, fit, arms, gates,
scoring rules), read at steps CHECKS and scored at the last.

  amend1.py --predictions   file results/predictions_amend1.json (once, before the run)
  amend1.py                 -> results/regating_spread_amend1.json, results/large/amend1.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch
from run import (BATCH, BLIND, LR, N_FIT_ROWS, N_HELD, NOISE, PREDICTIONS, RESULTS, SEED, TRANSPORT, Run, _finite,
                 gates, score, shares)
from torch import Tensor

PRED, OUT = RESULTS / "predictions_amend1.json", RESULTS / "regating_spread_amend1.json"
ROWS_TRAIN, ROWS_HELD, CHECKS = ("train", 0, 4096), ("val", 0, 256), (10, 100, 300)
N_TRAIN = CHECKS[-1] * BATCH  # finetune_arm_b takes fresh rows every step
SCORED = str(CHECKS[-1])

AMEND: dict[str, Any] = {
    "plan": "reports/2026-10-26_plan_regating-without-cues.md, Amendment 3. Filed before the run; never edited after.",
    "why": "E36's code-like fine-tune did not learn (gain -0.052 and -0.065, G failed at both steps); nothing scored",
    "task": f"Pile rows {ROWS_TRAIN} and {ROWS_HELD}, each decoded, lowercased, re-encoded, joined with end-of-text "
            f"and cut into 512-token rows; {N_TRAIN} training rows (fresh each step), {N_HELD} held-out rows",
    "finetune": f"as E36 (finetune_arm_b, Adam {LR}, batch {BATCH}, seed {SEED}), read at steps {CHECKS}; the rules "
                f"below are scored at step {SCORED}, the other steps read by the same rules and reported",
    "measurements": PREDICTIONS["measurements"], "gates": {k: v for k, v in PREDICTIONS["gates"].items() if k != "S"},
    "expected": {**PREDICTIONS["expected"],
                 "gain": "positive and growing with steps: capitals lose probability wherever they would follow"},
    "predictions": [{**p, "rule": p["rule"].replace("step 100", f"step {SCORED}"),
                     "confidence": "low" if p["id"] == "P_near" else p["confidence"]} for p in PREDICTIONS["predictions"]],
    "confidence_note": "E36's rules and thresholds unchanged; after E36's unlearned run (nearest 0.23 to 0.34 and the "
                       "line about 0.5 of a switching that carried noise) the confidence on P_near is lowered to low",
    "blind_to": BLIND + "; one style (lowercase), synthetic in form though spread over every position",
    "noise_floor": NOISE,
    "decision_tree": PREDICTIONS["decision_tree"],
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend1.log").open("a") as f:
        f.write(line + "\n")


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def lowercased(rows: Tensor, tok: Any, n: int) -> Tensor:
    """The rows decoded, lowercased, re-encoded, joined with end-of-text and cut into n rows of the same length."""
    T, ids = rows.shape[1], []
    for r in rows:
        ids += tok.encode(tok.decode(r.tolist()).lower()) + [tok.eos_token_id]
        if len(ids) >= n * T:
            return torch.tensor(ids[: n * T]).view(n, T)
    raise ValueError(f"{rows.shape[0]} rows made {len(ids) // T} lowercase rows, {n} needed")


def task() -> tuple[Tensor, Tensor, dict[str, Any]]:
    from commute import vpd
    from commute.data import pile_rows

    tok = vpd.tokenizer()
    train = lowercased(pile_rows(*ROWS_TRAIN, log=log), tok, N_TRAIN)
    held = lowercased(pile_rows(*ROWS_HELD, log=log), tok, N_HELD)
    info = {"n_train": train.shape[0], "n_held": held.shape[0],
            "samples": [tok.decode(held[i, :120].tolist()) for i in range(3)]}
    for s in info["samples"]:
        log("[task] sample: " + s.replace("\n", "\\n")[:400])
    return train, held, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        PRED.write_text(json.dumps({**AMEND, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        return print(f"filed {PRED}")
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    if not (TRANSPORT / "large" / "alive.pt").is_file():
        sys.exit(f"missing input (gitignored): {TRANSPORT / 'large' / 'alive.pt'}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    t0, r = time.time(), Run(a.device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(), "states": {}, "status": "running"}
    train, held, out["task"] = task()
    dump(out)
    ft = finetune_arm_b(r.model, train, held, ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS,
                                                         seed=SEED), a.device, log)
    out["finetune"] = ft.evals
    for step in CHECKS:
        dW = ft.deltas[step]
        f = r.fit(train[:N_FIT_ROWS], dW)
        L, same = r.evaluate(held, dW, f)
        m = shares(L)
        m.update(gates(m), near_same_token=same, slopes=f["slopes"], n_tokens_kept=f["token"].n_tokens)
        out["states"][str(step)] = m
        out["score"] = score(out["states"])
        dump(out)
        log(f"[step {step}] gain {m['gain']:.4f} gap {m['gap']:.4f} r {m['r']:.4f} | "
            + " ".join(f"{k} {v:.4f}" for k, v in m["R"].items()) + f" rand {m['R_rand']:.4f} | near_same_token "
            f"{same:.3f} slopes {json.dumps(f['slopes'])} | C0 {m['C0']} G {m['G']}")
        del f
        if step == CHECKS[0] and not m["C0"]:
            out["status"] = "stopped at C0"
            dump(out)
            return log("stopped at C0")
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(out)
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
