"""regating_spread (E36) Amendment 7: is the original values' edge only each value's room to move?
Plan: reports/plans/2026-10-26_plan_regating-without-cues.md -> Amendment 9 (from Amendment 6's independent check). Every
rule adds a predicted change to a held-out position's own original values and clips to [0, 1], so a value's change is
bounded by [-g0, 1 - g0]; neighbours with similar original values respect those bounds, text neighbours need not. This
run asks whether a function of each value alone does as well. Amendment 1's lowercase fine-tune is regenerated and read
at its steps; every rule is fitted on all 262,144 positions of the first 512 training rows (as Amendment 6):
  near_g0    the mean change of the 8 fitted positions nearest in Γ's original values (gate A0: reproduces Amendment 6)
  bin16      the value-bin rule: per alive subcomponent, on its own, 16 equal-count bins of its original value over the
             fitted positions; a held-out position's change for that subcomponent is the mean change of the fitted
             positions in the bin its own original value falls in. No joint structure across subcomponents, no context
  fix        Amendment 1's straight line per subcomponent, clip(a g0 + b) by least squares (cuefree.line), refitted here
             on the same 262,144 positions: the positive control C_bin (the bin rule can express any such line up to
             bin width)
  text4_k32  Amendment 6's 4-token text summary (amend5.text_features) with 32 neighbours instead of 8; reported only, to
             size the cost of the 8-neighbour cap where text neighbours tie
Bins (`fit_columns`): for a subcomponent with fitted values sorted s_0 <= ... <= s_{N-1}, the inner edges are
e_i = s_{iN/16 - 1}, i = 1..15, and a value v is in bin b = the number of edges strictly below v, i.e. bin b holds
e_b < v <= e_{b+1} (e_0 = -inf, e_16 = +inf). Without ties each bin holds N/16 = 16,384 positions. A run of tied values
(values sit on bf16 steps, many at 0 or 1) goes whole into the lowest bin whose upper edge equals it, leaving the bins
between equal edges empty; no value can fall in those. Bin 0 always holds s_0. The one reachable empty bin is the top
one, for a held-out value above every fitted value, and it takes the mean of the nearest non-empty bin below (every
empty bin is filled that way). A subcomponent whose original value is constant over the fitted positions so gets one
bin: its mean change everywhere.
Expected (gap = 0 at W0, where every R is undefined and every fitted change is 0, so every rule returns g0):
  near_g0    Amendment 6's 0.89, 0.78, 0.76 at steps 10, 100, 300 (the same code and positions; A0 tolerance 0.02)
  fix        about Amendment 1's 0.39, 0.14, -0.01 (fitted there on 16,384 positions of 64 rows, here on 16x more)
  bin16      at least fix - 0.05 at every step (C_bin); B1 predicts it at least 0.3 below near_g0 at step 300
  text4_k32  near Amendment 6's text4 (0.56, 0.52, 0.51): 8 against 32 neighbours moved R by about 0.014 before
  bins_used  mean number of non-empty bins per alive subcomponent: below 16 where values tie; read only to know how
             fine the bin rule was
Needs an H100.

  amend7.py --predictions   file results/predictions_amend7.json (once, before the run)
  amend7.py                 -> results/regating_spread_amend7.json, results/large/amend7.log
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
from amend5 import hash_tables, text_features
from amend6 import LOCAL, N_ROWS, NOISE, REPRO_TOL
from amend6 import OUT as AMEND6
from run import BATCH, K_NEAR, LR, RESULTS, SEED, TRANSPORT, Run, _finite
from torch import Tensor

from commute import cuefree as CF
from commute import spread as S

PRED, OUT = RESULTS / "predictions_amend7.json", RESULTS / "regating_spread_amend7.json"
BINS, K_TEXT, COL_CHUNK = 16, 32, 256
BIN_GAP, BIN_TOL = 0.3, 0.05
SCORED = str(CHECKS[-1])
RULES = ("near_g0", "bin16", "fix", "text4_k32")
BLIND = (f"one binning ({BINS} equal-count bins per subcomponent, fitted on 262,144 positions, so each bin's change is a "
         "mean over many positions and blind to variation within the bin); subcomponents treated independently by "
         "design, so a rule from the values of one matrix jointly but no context is not run; one seed and one "
         "held-out set")

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_regating-without-cues.md, Amendment 9. Filed before the run; never edited "
            "after.",
    "measurements": {
        "R_<rule>": "recovery of the switching over all held-out positions; every rule adds its predicted change to the "
                    "position's original values and clips to [0, 1]",
        "near_g0": "mean change of the 8 fitted positions nearest in Γ's original values (Amendment 6's rule)",
        "bin16": f"per alive subcomponent independently, {BINS} equal-count bins of its original value over the fitted "
                 "positions; the change is the mean change of the fitted positions in the held-out value's bin (ties go "
                 "whole into one bin; an empty top bin takes the bin below; a constant subcomponent gets one bin)",
        "fix": "clip(a g0 + b) per subcomponent by least squares over the fitted positions (Amendment 1's straight-line "
               "rule, cuefree.line)",
        "text4_k32": f"Amendment 6's {LOCAL}-token text summary with {K_TEXT} neighbours; reported",
        "bins_used": "mean number of non-empty bins per alive subcomponent; reported",
    },
    "gates": {"A0": f"every step: |R_near_g0 - Amendment 6's R_near_g0| <= {REPRO_TOL}; fail: stop",
              "C_bin": f"positive control for the bin rule: at every step R_bin16 >= R_fix - {BIN_TOL} (it can express "
                       "any straight line per subcomponent up to bin width); fail: B1 unreadable"},
    "noise_floor": f"8 against 32 neighbours moved R by about {NOISE}; differences below it are not read",
    "predictions": [
        {"id": "B1", "rule": f"step {SCORED}: R_bin16 <= R_near_g0 - {BIN_GAP} ('each subcomponent's change predicted "
                             "from its own original value alone, its room to move, recovers far less than the nearest "
                             "rule on all the original values')",
         "predicted": "pass", "confidence": "medium-high", "blind_to": BLIND, "positive_control": "C_bin"},
    ],
    "decision_tree": {
        "A0 fails": "stop; nothing scored",
        "C_bin fails": "B1 unreadable: the bin rule does not reach the straight line it should contain",
        "B1 holds": "the original values' edge is not each value's room to move, and the restatement 'positions the "
                    "original model treats alike get alike changes' goes to the user with Amendment 6's caveat on "
                    "context length",
        "B1 fails": "the restatement is withdrawn in favour of 'values move towards where they have room', and the user "
                    "gets a card",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend7.log").open("a") as f:
        f.write(line + "\n")


def fit_columns(X: Tensor, D: Tensor) -> dict[str, Tensor]:
    """Per alive subcomponent (column of the flat rows X, D): the bin rule's inner edges (C, BINS - 1) and mean change
    per bin (C, BINS), empty bins filled from the nearest non-empty bin below (see the module docstring); and the
    straight line's a, b (C,). Column blocks of COL_CHUNK keep the sort and cuefree.line's float64 copies small."""
    N = X.shape[0]
    at = (torch.arange(1, BINS, device=X.device) * N) // BINS - 1
    out: dict[str, list[Tensor]] = {"edges": [], "bin_mean": [], "a": [], "b": [], "used": []}
    for s in range(0, X.shape[1], COL_CHUNK):
        x, d = X[:, s : s + COL_CHUNK].T.contiguous(), D[:, s : s + COL_CHUNK].T.contiguous()
        edges = x.sort(1).values[:, at].contiguous()
        lab = torch.searchsorted(edges, x)  # bin = number of edges strictly below the value
        cnt = torch.zeros(x.shape[0], BINS, device=X.device).scatter_add_(1, lab, torch.ones_like(d))
        mean = torch.zeros_like(cnt).scatter_add_(1, lab, d) / cnt.clamp_min(1)
        last = torch.where(cnt > 0, torch.arange(BINS, device=X.device), 0).cummax(1).values
        a, b = CF.line(x.T, (x + d).T)
        for k, v in (("edges", edges), ("bin_mean", mean.gather(1, last)), ("a", a), ("b", b),
                     ("used", (cnt > 0).sum(1))):
            out[k].append(v)
        del x, d, lab
    return {k: torch.cat(v) for k, v in out.items()}


def bin_change(x0: Tensor, f: dict[str, Any]) -> Tensor:
    """(n, C) the bin rule's predicted change at positions with flat original values x0 (n, C)."""
    return f["bin_mean"].gather(1, torch.searchsorted(f["edges"], x0.T.contiguous())).T


def evaluate(r: Run, held: Tensor, dW: dict[str, Tensor], f: dict[str, Any]) -> dict[str, Any]:
    from commute import regating as G
    from commute.vpd import compute_ci

    sums: dict[str, float] = {}
    for s in range(0, held.shape[0], 8):
        batch = held[s : s + 8].to(r.device)
        ci0 = compute_ci(r.model, batch)
        with G.state_of(r.model, r.W0, dW):
            ci1 = compute_ci(r.model, batch)
        x0 = S.to_flat(ci0.g, r.lay)
        near = nearest(f["X"], f["Xn"], x0, K_NEAR)
        text = nearest(f["text4"], f["text4_n"], text_features(batch, LOCAL, f["tables"]), K_TEXT)
        pats = {"near_g0": (x0 + S.nearest_change(f["D"], near)).clamp(0, 1),
                "bin16": (x0 + bin_change(x0, f)).clamp(0, 1),
                "fix": (f["a"] * x0 + f["b"]).clamp(0, 1),
                "text4_k32": (x0 + S.nearest_change(f["D"], text)).clamp(0, 1)}
        L = {"g0": S.masked_logp(r.model, batch, ci0.g), "g1": S.masked_logp(r.model, batch, ci1.g)}
        for name, x in pats.items():
            L[name] = S.masked_logp(r.model, batch, S.with_alive(ci0.g, x, r.lay))
        for k, v in L.items():
            sums[k] = sums.get(k, 0.0) + v
        del ci0, ci1, pats
    gap = sums["g1"] - sums["g0"]
    return {**{f"R_{k}": (sums[k] - sums["g0"]) / gap for k in RULES},
            "gap": gap / (held.shape[0] * (held.shape[1] - 1)), "bins_used": float(f["used"].float().mean())}


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor], tables: tuple[Tensor, Tensor]) -> dict[str, Any]:
    X = r.values(rows, None)
    D = r.values(rows, dW).sub_(X)
    text4 = text_features(rows, LOCAL, tables)
    return {"X": X, "Xn": torch.einsum("ij,ij->i", X, X), "D": D, "tables": tables, "text4": text4,
            "text4_n": torch.einsum("ij,ij->i", text4, text4), **fit_columns(X, D)}


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
    for p in (PRED, AMEND6, TRANSPORT / "large" / "alive.pt"):
        if not p.is_file():
            sys.exit(f"missing: {p}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    ref = json.loads(AMEND6.read_text())["states"]
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
        m["C_bin"] = m["R_bin16"] >= m["R_fix"] - BIN_TOL
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float))
            + f" A0 {m['A0']} C_bin {m['C_bin']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    s, ok = out["states"][SCORED], all(m["C_bin"] for m in out["states"].values())
    out["score"] = {"C_bin": ok, "B1": ok and s["R_bin16"] <= s["R_near_g0"] - BIN_GAP}
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
