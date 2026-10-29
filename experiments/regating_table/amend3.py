"""regating_table (E46) Amendment 3: stage 1's ratio, gated piece gain over piece gain, on the least-squares split of
stage 0's nulls. Plan: reports/plans/2026-10-28_plan_regating-line-reset.md -> "Amendment 3 (2026-10-29)". Rules,
thresholds and the tree: results/predictions_amend3.json (read here, never edited).

Per E12 table (ident32, ident32b, rand32, rand32b), every state split by `anchors.lifted` and read by `run.cells`.
Controls first, on all four tables: C0 the split of a zero update (both parts 0); C1 the two fine-tunes (ratio as stage
0b, P0.2's bars). Then per table the nulls: random seed 0 (amend1.json's scale) and seed 1 (KL-matched, cap 100), the
per-key write at stage 0b's alpha (G-matched) and at the alpha whose split piece gain equals the seed-0 fine-tune's
(piece-matched). C2: the random seed-0 row and the G-matched write reproduce their unsplit G and g. A control miss
exits 1. Each row: the ratio's parts, a paired bootstrap over windows, the in-span share (1 - outside share).

  amend3.py   -> results/amend3.json after each row; every scored number on stdout and results/large/full.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import torch
from amend1 import load
from run import (
    BATCH,
    LARGE,
    RESULTS,
    Clock,
    Run,
    Task,
    _finite,
    cells,
    code_unchanged,
    dump,
    git_head,
    load_checked,
    log,
    read_state,
    write_delta,
)
from torch import Tensor

from commute import anchors as AN
from commute import perturb as PT
from commute.faithfulness import model_state
from commute.finetune import random_direction

PRED_A, OUT = RESULTS / "predictions_amend3.json", "amend3.json"
TABLES, RANDOM_MAP = ("ident32", "ident32b", "rand32", "rand32b"), ("rand32", "rand32b")
FROZEN = (
    "src/commute/perturb.py", "src/commute/anchors.py", "src/commute/split.py", "src/commute/local_faith.py",
    "src/commute/finetune.py", "src/commute/vpd.py", "src/commute/component_training.py",
)  # harness.py left out: E49 added a function to it after 910effa; run.py changed here (C1, C2 cover it)
SHARED = ("p02_min_piece", "p02_copy_min", "p02_random_max", "w0_tol", "write_rel_tol", "alpha_max")  # = predictions.json's
PARTS = ("G", "g", "piece_gain", "gated_piece_gain")


def prepare(run: Run, s0a: dict[str, Any], name: str) -> dict[str, Any]:
    """The table's task, its seed-0 fine-tune's delta-W, W0's per-window unmasked log-odds and the input moments."""
    from commute import split
    from commute.data import batches
    from commute.local_faith import masked_lo_windows

    t, dW = load(run, s0a, name)
    U0w = masked_lo_windows(run.model, t.held.tokens, t.half, t.token, ones(t.base["gates"]), BATCH)
    sigma = split.input_second_moments(run.model, batches(t.train.tokens, 32), list(run.W0))
    return {"t": t, "dW": dW, "U0w": U0w, "sigma": {k: v.cpu() for k, v in sigma.items()}}


def ones(gates: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
    return [{k: torch.ones_like(v) for k, v in g.items()} for g in gates]


def boot(num: Tensor, den: Tensor, th: dict[str, Any]) -> dict[str, Any]:
    """Paired bootstrap over windows of mean(num) / mean(den): the interval over resamples with den >= min piece."""
    n, a = num.numel(), (1 - th["boot_ci"]) / 2
    idx = torch.randint(n, (th["n_boot"], n), generator=torch.Generator().manual_seed(0))
    nb, db = num[idx].mean(1), den[idx].mean(1)
    ok = db >= th["p02_min_piece"]
    q = lambda x: [float(torch.quantile(x, a)), float(torch.quantile(x, 1 - a))] if x.numel() else None
    below = 1 - float(ok.double().mean())
    return {"ratio_ci": q(nb[ok] / db[ok]), "piece_ci": q(db), "gated_ci": q(nb), "frac_below_min_piece": below,
            "unstable": below > th["unstable_frac"]}


def split_row(run: Run, T: dict[str, Any], dW: dict[str, Tensor], th: dict[str, Any]) -> dict[str, Any]:
    """`run.cells` of the least-squares split of dW, the ratio (None below the min piece), bootstrap, in-span share."""
    from commute.harness import outside_share_of
    from commute.local_faith import masked_lo_windows, route_cells

    t, m, held = T["t"], run.model, T["t"].held.tokens
    comps = AN.lifted(m, run.lift, dW)
    c = cells(run, t, dW, comps)
    with model_state(m, target=run.weights(dW), components=comps):
        g1 = route_cells(m, held, t.half, t.token, BATCH)["gates"]
        L11w = masked_lo_windows(m, held, t.half, t.token, g1, BATCH)
        U1w = masked_lo_windows(m, held, t.half, t.token, ones(g1), BATCH)
    del g1
    defined = c["piece_gain"] >= th["p02_min_piece"]  # False for NaN
    dWc = {k: v.cpu() for k, v in dW.items() if v.any()}
    share = outside_share_of(dWc, run.lift["bases"], T["sigma"], list(dWc))
    return {**c, "ratio": c["gated_piece_gain"] / c["piece_gain"] if defined else None, "defined": defined,
            "g_over_G": c["g"] / c["G"] if abs(c["G"]) >= th["g_over_G_min_abs_G"] else None,
            "boot": boot(L11w - t.L00w, U1w - T["U0w"], th), "in_span": 1 - share["pooled"]}


def piece_of(run: Run, t: Task, dW: dict[str, Tensor]) -> float:
    from commute.local_faith import route_cells

    with model_state(run.model, target=run.weights(dW), components=AN.lifted(run.model, run.lift, dW)):
        return route_cells(run.model, t.held.tokens, t.half, t.token, BATCH)["U"] - t.base["U"]


def repro(rec: dict[str, Any], ref: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """C2: G and g of the split state against the unsplit row's, within max(w0_tol, repro_rel_tol x |G_ref|)."""
    tol = max(th["w0_tol"], th["repro_rel_tol"] * abs(ref["G"]))
    return {"G_ref": ref["G"], "g_ref": ref["g"], "tol": tol, "ok": abs(rec["G"] - ref["G"]) <= tol and abs(rec["g"] - ref["g"]) <= tol}


def controls(run: Run, s0a: dict[str, Any], s0b: dict[str, Any], th: dict[str, Any], put: Any) -> None:
    """C0 and C1 on every table, before any null row."""
    cls = run.P["refs"]["e12_class"]
    for n in TABLES:
        T = prepare(run, s0a, n)
        zero = {k: torch.zeros_like(v) for k, v in run.W0.items()}
        c0 = cells(run, T["t"], zero, AN.lifted(run.model, run.lift, zero))
        ok0 = abs(c0["piece_gain"]) <= th["w0_split_tol"] and abs(c0["gated_piece_gain"]) <= th["w0_split_tol"]
        put(f"{n}/zero", "C0", {k: c0[k] for k in ("piece_gain", "gated_piece_gain", "cells")}, ok0)
        for s in ("s0", "s1"):
            dW = T["dW"] if s == "s0" else load_checked(LARGE / f"{n}_s1.pt", s0a["groups"][f"e12/{n}"]["states"]["s1"]["sha256"])
            rec, ref = split_row(run, T, dW, th), s0b["score"]["P0.2"]["rows"][f"{n}/{s}"]["ratio"]
            r = rec["ratio"]
            bar = r is not None and (r >= th["p02_copy_min"] if cls[n] == "copy" else r <= th["p02_random_max"])
            ok = bar and abs(r - ref) <= th["ref_ratio_tol"]
            put(f"{n}/{s}", "C1", {**rec, "class": cls[n], "ratio_0b": ref}, ok)
        del T


def random_rows(run: Run, T: dict[str, Any], n: str, old: dict[str, Any], th: dict[str, Any], put: Any) -> None:
    """Seed 0 at amend1.json's scale (C2 against amend1's row), seed 1 KL-matched to the same kl_target."""
    t, dW = T["t"], T["dW"]
    rec = split_row(run, T, PT.scaled(random_direction(dW, 0), old["scale"]), th)
    c2 = repro(rec, old, th)
    put(f"{n}/random_s0", "C2", {**rec, "scale": old["scale"], "repro": c2}, c2["ok"])
    d1 = random_direction(dW, 1)
    m = PT.match(lambda c: read_state(run, t, PT.scaled(d1, c))[1], old["kl_target"], th["scale_cap"], th["kl_rel_tol"])
    put(f"{n}/random_s1", "null", {**split_row(run, T, PT.scaled(d1, m["x"]), th), "scale": m["x"], "match": m["status"],
                                   "kl_target": old["kl_target"], "grid": m["grid"], "trace": m["trace"]})


def write_rows(run: Run, T: dict[str, Any], n: str, s0b: dict[str, Any], th: dict[str, Any], put: Any) -> None:
    """The write at stage 0b's alpha (C2 against 0b's row), then at the alpha whose split piece gain matches seed 0's."""
    t, task = T["t"], run.P["refs"]["e12_tasks"]["tasks"][n]
    X, fit = write_delta(run, t, task["keys"], task["values"])
    dw = lambda a: {PT.WRITE_KEY: (a * X).float()}
    old = s0b["states"][f"{n}/write"]
    rec = split_row(run, T, dw(old["alpha"]), th)
    c2 = repro(rec, old, th)
    put(f"{n}/write_G", "C2", {**rec, "alpha": old["alpha"], "fit": fit, "repro": c2}, c2["ok"])
    target = s0b["states"][f"{n}/s0"]["piece_gain"]
    m = PT.match(lambda a: piece_of(run, t, dw(a)), target, th["alpha_max"], th["write_rel_tol"])
    put(f"{n}/write_piece", "null", {**split_row(run, T, dw(m["x"]), th), "alpha": m["x"], "match": m["status"],
                                     "flag": "piece-matched", "piece_target": target, "grid": m["grid"], "trace": m["trace"]})


def scored_write(rows: dict[str, Any], n: str, th: dict[str, Any]) -> str | None:
    """The G-matched write if defined, else the piece-matched write if defined, else None."""
    for kind in ("write_G", "write_piece"):
        x = rows.get(f"{n}/{kind}")
        if x is not None and x["piece_gain"] >= th["p02_min_piece"]:
            return kind
    return None


def score(rows: dict[str, Any], A: dict[str, Any]) -> dict[str, Any]:
    """predictions_amend3.json's tree, as filed; first match wins."""
    th, rm = A["thresholds"], A["random_map_tables"]
    lo, hi = th["bar_low"], th["bar_high"]
    pick = {n: scored_write(rows, n, th) for n in A["tables"]}
    ratio: dict[str, float | None] = {n: None if k is None else rows[f"{n}/{k}"]["gated_piece_gain"] / rows[f"{n}/{k}"]["piece_gain"]
                                      for n, k in pick.items()}
    defined = [x for n in rm if (x := ratio[n]) is not None]
    nonfinite = [f"{n}/{k}" for n in rm for k in ("write_G", "write_piece") if f"{n}/{k}" in rows
                 and not all(math.isfinite(rows[f"{n}/{k}"][q]) for q in PARTS)]
    nonfinite += [n for n in rm if (x := ratio[n]) is not None and not math.isfinite(x)]
    if nonfinite:
        tree = "card: non-finite"
    elif any(x >= hi for x in defined):
        tree = "different number before any pod"
    elif len(defined) == len(rm) and all(x <= lo for x in defined):
        tree = "propose stage 1 with this number"
    elif len(defined) < len(rm):
        tree = "card: write null unreadable"
    else:
        tree = "card"
    copies = [ratio[n] for n in A["tables"] if n not in rm]
    copy_rule = "unread" if any(x is None for x in copies) else "discounts" if all(x <= lo for x in copies) else "no"
    crosses = {n: [b for b in (lo, hi) if (ci := rows[f"{n}/{pick[n]}"]["boot"]["ratio_ci"]) and ci[0] <= b <= ci[1]]
               for n in rm if pick[n] is not None}
    return {"A3.1": "pass" if tree.startswith("propose") else "fail", "tree": tree, "scored_write": pick, "ratio": ratio,
            "nonfinite": nonfinite, "copy_write_rule": copy_rule, "interval_crosses_bar": crosses,
            "random_rows": {k: x["ratio"] for k, x in rows.items() if "/random_" in k}}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not PRED_A.is_file():
        sys.exit(f"file the predictions first: {PRED_A}")
    A = json.loads(PRED_A.read_text())
    th = A["thresholds"]
    assert (tuple(A["tables"]), tuple(A["random_map_tables"])) == (TABLES, RANDOM_MAP), "predictions_amend3.json's tables differ"
    s0a, s0b, a1 = (json.loads((RESULTS / f).read_text()) for f in ("stage0a.json", "stage0b.json", "amend1.json"))
    if not code_unchanged(s0b["commit"], FROZEN):
        sys.exit(f"{', '.join(FROZEN)} changed since stage 0b's commit {s0b['commit']}")
    run = Run("cuda")
    assert all(run.P["thresholds"][k] == th[k] for k in SHARED), "the filed settings differ from predictions.json's"
    out: dict[str, Any] = {"commit": git_head(), "stage0b_commit": s0b["commit"], "rows": {}}
    clock = Clock()

    def put(key: str, role: str, rec: dict[str, Any], ok: bool | None = None) -> None:
        out["rows"][key] = {"role": role, **rec, **({} if ok is None else {"ok": ok})}
        dump(OUT, out)
        brief = {k: rec.get(k) for k in ("ratio", "defined", *PARTS, "in_span", "g_over_G", "scale", "alpha", "match", "repro", "ratio_0b")}
        log(f"[amend3 {key}] {role} ok {ok} {json.dumps(_finite(brief))} | boot {json.dumps(_finite(rec.get('boot')))} "
            f"| cells {json.dumps(_finite(rec.get('cells')))}")
        if ok is False:
            log(f"[amend3] FAIL: control {role} at {key}: stop")
            sys.exit(1)

    controls(run, s0a, s0b, th, put)
    clock.lap("amend3 controls")
    for n in TABLES:
        T = prepare(run, s0a, n)
        random_rows(run, T, n, a1["rows"][f"{n}/random"], th, put)
        write_rows(run, T, n, s0b, th, put)
        del T
        clock.lap(f"amend3 {n}")
    out["score"] = score(out["rows"], A)
    dump(OUT, out)
    log(f"[amend3 score] {json.dumps(_finite(out['score']))}")


if __name__ == "__main__":
    main()
