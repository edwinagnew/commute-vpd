"""regating_cuefree (E35) Amendment 1: is "nearest in the original values" the cue in disguise, and is its comparison
with one change per cue like for like? Plan: reports/plans/2026-10-26_plan_regating-without-cues.md -> Amendment 2 (from
E35's independent check). States, fine-tunes and the nearest rule are run.py's; per state this adds:
  R_near      run.py's rule again: the mean change of the 8 training windows nearest in g0 (gate A0: reproduces E35)
  R_near_cue  the same, with the 8 nearest drawn from training windows with the window's own cue: per cue plus each
              input's own dependence on g0, the like-for-like reference for R_near
  same_cue    share of the 8 nearest training windows (unrestricted) holding the window's cue
  dup         share of held-out windows whose tokens equal some training window's
  dist_ratio  median over held-out windows of the distance to the nearest training window, over the median over
              training windows of the distance to the nearest other training window

  amend1.py --predictions   file results/predictions_amend1.json (once, before the run)
  amend1.py                 -> results/regating_cuefree_amend1.json, results/large/amend1.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch
from run import CONTROL, CUE_STATES, KV_TASKS, MULTI, RESULTS, WORD, Run, _finite, inputs, windows_of

from commute import cuefree as CF
from commute import regating as G
from commute import shift_anatomy as SA

PRED, OUT = RESULTS / "predictions_amend1.json", RESULTS / "regating_cuefree_amend1.json"
E35_JSON = RESULTS / "regating_cuefree.json"
K_NEAR, STEPS, SEED, OPT, LR = 8, 10, 0, "adam", 3e-5
REPRO_TOL, CLOSE, SAME_LEAN, DUP_MAX = 1e-3, 0.1, 0.9, 0.05
BLIND = ("the trigger position only; one fine-tune seed; one 67M model; copy and the cipher share one window set, the "
         "two 32-cue ' of' states another, so the four 32-cue states are two input sets")
PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_regating-without-cues.md, Amendment 2. Filed before the run; never edited after.",
    "measurements": {
        "R_near": "run.py's nearest rule again (8 training windows nearest in g0); gate A0",
        "R_near_cue": "the same with the 8 nearest drawn from training windows with the window's own cue (not on "
                      f"{CONTROL}, whose rarer trigger tokens have fewer than 8 training windows)",
        "same_cue": "share of the 8 nearest training windows (unrestricted) holding the window's cue; reported",
        "dup": "share of held-out windows whose tokens equal a training window's",
        "dist_ratio": "median held-out distance to the nearest training window over the median training distance to "
                      "the nearest other training window; reported",
    },
    "gates": {"A0": f"every state: |R_near - E35's R_near| <= {REPRO_TOL}; {CONTROL} first; fail: stop"},
    "predictions": [
        {"id": "L", "rule": f"every 32-cue state: R_near >= R_near_cue - {CLOSE}", "predicted": "pass",
         "confidence": "medium-low", "blind_to": BLIND},
        {"id": "T", "rule": f"every 32-cue state: same_cue >= {SAME_LEAN} ('the nearest windows share the cue')",
         "predicted": "pass", "confidence": "medium", "blind_to": BLIND},
        {"id": "D", "rule": f"every state: dup <= {DUP_MAX}; a state failing it is unreadable for L", "predicted": "pass",
         "confidence": "medium", "blind_to": BLIND},
    ],
    "decision_tree": {
        "L and T hold": "the original values carry the cue; the nearest rule is the per-cue rule plus each input's own "
                        "dependence on its values: worded so in E35's report",
        "L holds, T fails": "the nearest rule groups across cues without loss: the cue is not the unit",
        "L fails": "knowing the cue adds beyond the original values: the cue-free rule loses where cues mix",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend1.log").open("a") as f:
        f.write(line + "\n")


def state(r: Run, name: str) -> dict[str, Any]:
    t = G.prepare(r.model, name, *windows_of(name))
    dW, _ = G.finetune(r.model, t, t.target, OPT, LR, STEPS, SEED, r.device, log)
    m = G.measure(r.model, r.W0, t, dW, {k: torch.zeros(v.shape) for k, v in t.m0.items()}, r.alive, n_random=1)
    gap, rho_trig = m["gap"], m["rho"]["trig"]
    rd = r.reads(t, dW)
    d_tr = CF.change(rd["tr1"], rd["tr0"], r.alive)
    cue_tr, cue_he = t.train.tokens[:, t.half], t.held.tokens[:, t.half]
    R = lambda vals: G.div(SA.rho_of(r.model, t, vals, gap), rho_trig)
    idx, dist = CF.neighbours(rd["tr0"], rd["he0"], r.alive, K_NEAR, r.device)
    idx_cue = (None if name == CONTROL else
               CF.neighbours(rd["tr0"], rd["he0"], r.alive, K_NEAR, r.device, same=(cue_tr, cue_he))[0])
    _, d_tt = CF.neighbours(rd["tr0"], rd["tr0"], r.alive, 2, r.device)
    train_set = {tuple(w.tolist()) for w in t.train.tokens}
    return {"R_near": R(CF.apply(rd["he0"], CF.mean_at(d_tr, idx))),
            "R_near_cue": float("nan") if idx_cue is None else R(CF.apply(rd["he0"], CF.mean_at(d_tr, idx_cue))),
            "same_cue": float((cue_tr[idx] == cue_he[:, None]).float().mean()),
            "dup": sum(tuple(w.tolist()) in train_set for w in t.held.tokens) / len(t.held),
            "dist_ratio": float(dist[:, 0].clamp_min(0).sqrt().median() / d_tt[:, 1].clamp_min(0).sqrt().median()),
            "r": m["r"], "rho_trig": rho_trig}


def score(st: dict[str, Any]) -> dict[str, Any]:
    multi = {n: m for n, m in st.items() if n in MULTI}
    readable = {n: m["dup"] <= DUP_MAX for n, m in st.items()}
    return {"D": all(readable.values()),
            "L": all(m["R_near"] >= m["R_near_cue"] - CLOSE for n, m in multi.items() if readable[n]),
            "T": all(m["same_cue"] >= SAME_LEAN for m in multi.values()), "complete": len(multi) == len(MULTI)}


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
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    missing = [str(p) for p in [*inputs(), E35_JSON] if not p.is_file()]
    if missing:
        sys.exit("missing inputs:\n  " + "\n  ".join(missing))
    e35 = json.loads(E35_JSON.read_text())["states"]
    r, out = Run(a.device), {"states": {}, "status": "running"}
    for name in (CONTROL, WORD, *KV_TASKS, *CUE_STATES):
        key = "word" if name == WORD else name
        out["states"][key] = m = state(r, name)
        m["A0"] = abs(m["R_near"] - e35[key]["R_near"]) <= REPRO_TOL
        out["score"] = score(out["states"])
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[{name}] " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
        if not m["A0"]:
            out["status"] = f"stopped at A0 ({name})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    out["status"] = "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT}")


if __name__ == "__main__":
    main()
