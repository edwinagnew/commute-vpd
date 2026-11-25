"""regating_spread (E36) Amendment 7: is the original values' edge only each value's room to move?
Plan: reports/plans/2026-10-26_plan_regating-without-cues.md -> Amendment 9 (from Amendment 6's independent check), with
its code review (reports/reviews/2026-10-27_review_amend7-code.md). Every rule adds a predicted change to a held-out
position's own original values and clips to [0, 1], so a value's change is bounded by [-g0, 1 - g0]; neighbours with
similar original values respect those bounds, text neighbours need not. This run asks whether a function of each value
alone does as well. Amendment 1's lowercase fine-tune is regenerated and read at its steps; every rule is fitted on all
262,144 positions of the first 512 training rows (as Amendment 6):
  near_g0    the mean change of the 8 fitted positions nearest in Γ's original values (gate A0: reproduces Amendment 6)
  bin16q     a value-bin rule: per alive subcomponent, on its own, 16 equal-count bins of its original value over the
             fitted positions; a held-out position's change for that subcomponent is the mean change of the fitted
             positions in the bin its own original value falls in. No joint structure across subcomponents, no context
  bin16w     the same with 16 equal-width bins on [0, 1] (the review: most values sit near zero, so equal-count edges
             all fall there and every active value shares the top bin's one change)
  fix        Amendment 1's straight line per subcomponent, clip(a g0 + b) by least squares (cuefree.line), refitted here
             on the same 262,144 positions: the positive controls C_bin and C_fit
  text4_k32  Amendment 6's 4-token text summary (amend5.text_features) with 32 neighbours instead of 8; reported only, to
             size the cost of the 8-neighbour cap where text neighbours tie
Bins (`fit_columns`, `bin_means`). A value v is in bin b = the number of inner edges strictly below v, i.e. bin b holds
e_b < v <= e_{b+1} (e_0 = -inf, e_16 = +inf). Equal-count: with a subcomponent's fitted values sorted
s_0 <= ... <= s_{N-1}, e_i = s_{iN/16 - 1}, i = 1..15; without ties each bin holds N/16 = 16,384 positions. A run of tied
values (values sit on bf16 steps, many at 0 or 1) goes whole into the lowest bin whose upper edge equals it, leaving the
bins between equal edges empty; no value can fall in those. Equal-width: e_i = i/16, so values <= 1/16 (0 included) are
in bin 0 and values above 15/16 (1 included) in bin 15. Either way an empty bin takes the mean change of the nearest
non-empty bin below, or above if there is none below (equal-width bins under every fitted value). A subcomponent whose
original value is constant over the fitted positions so gets one bin: its mean change everywhere.
Controls. C_bin: each bin rule's R at least fix's - 0.05 at every step. Within a bin a bin rule has slope 1, so it departs
from a line of slope a by up to (1 - a) x the bin's width; fitted slopes have median 0.94 to 0.96, so this is small for
equal-width bins (width 1/16) but can be large in a wide equal-count top bin. C_fit: at every step, each bin rule's mean
squared error of the change on the fitted positions at most the line's, the change taken as each rule applies it,
clip(g0 + c) - g0 (the line recovers about -0.01 at step 300, where C_bin alone would pass a broken bin rule). B1 is
readable when at least one bin rule passes both controls at every step, and it then reads the best R over all bin rules,
passing or not, so dropping a rule can never help B1 pass; if neither passes both, B1 is recorded as "unreadable".
Expected (gap = 0 at W0, where every R is undefined and every fitted change is 0, so every rule returns g0 and every
mean squared error is 0):
  near_g0           Amendment 6's 0.89, 0.78, 0.76 at steps 10, 100, 300 (the same code and positions; A0 tolerance 0.02)
  fix               about Amendment 1's 0.39, 0.14, -0.01 (fitted there on 16,384 positions of 64 rows, here on 16x more)
  bin16q, bin16w    at least fix - 0.05 (C_bin), and fitted mean squared error at most fix's (C_fit), at every step; B1
                    predicts the better of the two at least 0.3 below near_g0 at step 300
  text4_k32         near Amendment 6's text4 (0.56, 0.52, 0.51): 8 against 32 neighbours moved R by about 0.014 before
  top_edge_q_median the median over subcomponents of the top equal-count edge e_15: below 0.1 if the review is right that
                    most values sit near zero; read only to know how coarse bin16q was
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
BIN_RULES = ("bin16q", "bin16w")
RULES = ("near_g0", *BIN_RULES, "fix", "text4_k32")
BLIND = (f"two binnings ({BINS} equal-count bins per subcomponent and {BINS} equal-width bins on [0, 1]), each bin's "
         "change a mean over the fitted positions in it, so blind to variation within a bin; subcomponents treated "
         "independently by design, so a rule from the values of one matrix jointly but no context is not run; one seed "
         "and one held-out set")

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_regating-without-cues.md, Amendment 9 and its code review. Filed before the "
            "run; never edited after.",
    "measurements": {
        "R_<rule>": "recovery of the switching over all held-out positions; every rule adds its predicted change to the "
                    "position's original values and clips to [0, 1]",
        "near_g0": "mean change of the 8 fitted positions nearest in Γ's original values (Amendment 6's rule)",
        "bin16q": f"per alive subcomponent independently, {BINS} equal-count bins of its original value over the fitted "
                  "positions; the change is the mean change of the fitted positions in the held-out value's bin (ties "
                  "go whole into one bin; an empty bin takes the nearest non-empty bin below, else above; a constant "
                  "subcomponent gets one bin)",
        "bin16w": f"the same with {BINS} equal-width bins on [0, 1] (edges i/16)",
        "fix": "clip(a g0 + b) per subcomponent by least squares over the fitted positions (Amendment 1's straight-line "
               "rule, cuefree.line)",
        "text4_k32": f"Amendment 6's {LOCAL}-token text summary with {K_TEXT} neighbours; reported",
        "mse_<rule>": "mean squared error of the change on the fitted positions, the change as the rule applies it, "
                      "clip(g0 + c) - g0, over every alive subcomponent; for C_fit",
        "top_edge_q_median": "median over subcomponents of the top equal-count edge; reported, to show how coarse "
                             "bin16q was",
    },
    "gates": {"A0": f"every step: |R_near_g0 - Amendment 6's R_near_g0| <= {REPRO_TOL}; fail: stop",
              "C_bin": f"positive control, per bin rule: at every step R_<bin rule> >= R_fix - {BIN_TOL} (within a "
                       "bin a bin rule has slope 1, so it can follow a straight line of slope a up to (1 - a) x the "
                       "bin's width)",
              "C_fit": "positive control, per bin rule: at every step mse_<bin rule> <= mse_fix on the fitted positions "
                       "(C_bin alone would pass a broken bin rule at step 300, where the line recovers about -0.01)",
              "readable": "B1 is readable when at least one bin rule passes both C_bin and C_fit at every step (a "
                          "piecewise-constant bin rule can lose to a line on fitted error for real reasons, a wide "
                          "equal-count top bin); it then reads the best R over all bin rules, passing or not, so "
                          "dropping a rule can never help B1 pass; if neither passes both: B1 unreadable"},
    "noise_floor": f"8 against 32 neighbours moved R by about {NOISE}; differences below it are not read",
    "predictions": [
        {"id": "B1", "rule": f"step {SCORED}: max(R_bin16q, R_bin16w) <= R_near_g0 - {BIN_GAP} ('each subcomponent's "
                             "change predicted from its own original value alone, its room to move, recovers far less "
                             "than the nearest rule on all the original values', read on the better of the two bin "
                             "rules, passing or not); recorded as 'unreadable' unless at least one bin rule passes both "
                             "C_bin and C_fit at every step",
         "predicted": "pass", "confidence": "medium-high", "blind_to": BLIND, "positive_control": "C_bin, C_fit"},
    ],
    "decision_tree": {
        "A0 fails": "stop; nothing scored",
        "neither bin rule passes both C_bin and C_fit": "B1 unreadable: no bin rule reaches the straight line it "
                                                        "should contain",
        "one bin rule fails a control": "B1 still read on the best R over both bin rules; the failing rule is reported",
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


def lookup(mean: Tensor, edges: Tensor, xt: Tensor) -> Tensor:
    """(C, n) each value's bin mean: xt (C, n) one row per subcomponent, edges (C, BINS - 1), mean (C, BINS); the bin
    is the number of edges strictly below the value."""
    return mean.gather(1, torch.searchsorted(edges, xt))


def bin_means(x: Tensor, d: Tensor, edges: Tensor) -> Tensor:
    """(C, BINS) mean change d per bin of x (both (C, N), one row per subcomponent); an empty bin takes the nearest
    non-empty bin below, else above."""
    lab = torch.searchsorted(edges, x)
    cnt = torch.zeros(x.shape[0], BINS, device=x.device).scatter_add_(1, lab, torch.ones_like(d))
    mean = torch.zeros_like(cnt).scatter_add_(1, lab, d) / cnt.clamp_min(1)
    ar = torch.arange(BINS, device=x.device)
    below = torch.where(cnt > 0, ar, -1).cummax(1).values
    above = torch.where(cnt > 0, ar, BINS).flip(1).cummin(1).values.flip(1)
    return mean.gather(1, torch.where(below >= 0, below, above))


def sq_err(x: Tensor, d: Tensor, pred: Tensor) -> float:
    """Summed squared error of the change as a rule applies it, clip(pred) - x, against d."""
    return float((pred.clamp(0, 1) - x - d).square().sum(dtype=torch.float64))


def fit_columns(X: Tensor, D: Tensor) -> dict[str, Any]:
    """Per alive subcomponent (column of the flat rows X, D): each bin rule's inner edges (C, BINS - 1) and filled mean
    change per bin (C, BINS), the straight line's a, b (C,), every rule's fitted mean squared error and the median top
    equal-count edge. Column blocks of COL_CHUNK keep the sort and cuefree.line's float64 copies small."""
    N = X.shape[0]
    at = (torch.arange(1, BINS, device=X.device) * N) // BINS - 1
    width = torch.arange(1, BINS, device=X.device, dtype=X.dtype) / BINS
    out: dict[str, list[Tensor]] = {}
    err = {k: 0.0 for k in (*BIN_RULES, "fix")}
    for s in range(0, X.shape[1], COL_CHUNK):
        x, d = X[:, s : s + COL_CHUNK].T.contiguous(), D[:, s : s + COL_CHUNK].T.contiguous()
        edges = {"bin16q": x.sort(1).values[:, at].contiguous(), "bin16w": width.expand(x.shape[0], -1).contiguous()}
        for k, e in edges.items():
            mean = bin_means(x, d, e)
            out.setdefault(f"edges_{k}", []).append(e)
            out.setdefault(f"mean_{k}", []).append(mean)
            err[k] += sq_err(x, d, x + lookup(mean, e, x))
        a, b = CF.line(x.T, (x + d).T)
        out.setdefault("a", []).append(a)
        out.setdefault("b", []).append(b)
        err["fix"] += sq_err(x, d, a[:, None] * x + b[:, None])
        del x, d
    f: dict[str, Any] = {k: torch.cat(v) for k, v in out.items()}
    f["mse"] = {k: v / X.numel() for k, v in err.items()}
    f["top_edge_q_median"] = float(f["edges_bin16q"][:, -1].median())
    return f


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
        x0t = x0.T.contiguous()
        pats = {"near_g0": (x0 + S.nearest_change(f["D"], near)).clamp(0, 1),
                **{k: (x0 + lookup(f[f"mean_{k}"], f[f"edges_{k}"], x0t).T).clamp(0, 1) for k in BIN_RULES},
                "fix": (f["a"] * x0 + f["b"]).clamp(0, 1),
                "text4_k32": (x0 + S.nearest_change(f["D"], text)).clamp(0, 1)}
        L = {"g0": S.masked_logp(r.model, batch, ci0.g), "g1": S.masked_logp(r.model, batch, ci1.g)}
        for name, x in pats.items():
            L[name] = S.masked_logp(r.model, batch, S.with_alive(ci0.g, x, r.lay))
        for k, v in L.items():
            sums[k] = sums.get(k, 0.0) + v
        del ci0, ci1, pats, x0t
    gap = sums["g1"] - sums["g0"]
    return {**{f"R_{k}": (sums[k] - sums["g0"]) / gap for k in RULES}, **{f"mse_{k}": v for k, v in f["mse"].items()},
            "gap": gap / (held.shape[0] * (held.shape[1] - 1)), "top_edge_q_median": f["top_edge_q_median"]}


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor], tables: tuple[Tensor, Tensor]) -> dict[str, Any]:
    X = r.values(rows, None)
    D = r.values(rows, dW).sub_(X)
    text4 = text_features(rows, LOCAL, tables)
    return {"X": X, "Xn": torch.einsum("ij,ij->i", X, X), "D": D, "tables": tables, "text4": text4,
            "text4_n": torch.einsum("ij,ij->i", text4, text4), **fit_columns(X, D)}


def controls(m: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Per bin rule: C_bin (R at least the line's - BIN_TOL) and C_fit (fitted mean squared error at most the line's)."""
    return {"C_bin": {k: m[f"R_{k}"] >= m["R_fix"] - BIN_TOL for k in BIN_RULES},
            "C_fit": {k: m[f"mse_{k}"] <= m["mse_fix"] for k in BIN_RULES}}


def score(states: dict[str, Any]) -> dict[str, Any]:
    """B1 at SCORED on the best R over all bin rules; readable when at least one bin rule passed both controls at
    every step, else 'unreadable'."""
    passing = [k for k in BIN_RULES if all(m["C_bin"][k] and m["C_fit"][k] for m in states.values())]
    s, best = states[SCORED], max(states[SCORED][f"R_{k}"] for k in BIN_RULES)
    return {"passing_controls": passing, "best_bin_R": best,
            "B1": (best <= s["R_near_g0"] - BIN_GAP) if passing else "unreadable"}


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
        m.update(controls(m))
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.6g}" for k, v in m.items() if isinstance(v, float))
            + f" A0 {m['A0']} C_bin {m['C_bin']} C_fit {m['C_fit']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    out["score"] = score(out["states"])
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
