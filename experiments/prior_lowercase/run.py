"""prior_lowercase (E145, cell P2.13): the original model's prior for the trained answer on E53's lowercase and cipher
windows, the one cell of the lowercase-against-ciphers row that E133 left empty (E53's windows were not saved; they are
rebuilt here from the cached Pile rows by E53's own `Run.scored`, as E60 and E136 rebuilt them).
Plan: reports/plans/2026-11-10_plan_prior-lowercase.md. Rules: results/predictions.json (filed and committed before the
run; the run refuses otherwise, never edited after).

The prior of a task is E133's: the original model's geometric-mean P(trained answer) at the read position, `keydep.geo_rate`
of the log P that `regating_truth_table.next_task` reads (the answer is each window's true next token). It is one value
per task, not per seed (every seed is read on the same windows), so its spread is given across the 500 windows: the
bootstrap interval of the geometric mean over windows and the windows' percentiles. No fine-tuned state is loaded.

  stage 0  the controls, alone: E133's forward re-reads one prior it computed (e58_24_rank1), and the rebuilt windows give
           E53's saved W0 log-odds on each task; a miss exits 1 -> results/stage0.json
  stage 1  per task the prior, its interval over windows and the window percentiles; the pair's gap and the filed
           reading -> results/stage1.json
Each stage prints `STAGE n DONE` after its file is written, and every scored number goes to the log. No verdict of
reused or new: the sides are labelled as built.
"""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import keydep as KD
from commute import open_pair as OP
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
TASKS = ("lowercase", "cipher", "broad_cipher")
REUSE, NEW = ["lowercase"], ["cipher", "broad_cipher"]  # as E53 built them (performed against new), the method table's sides
log = KD.log


def load(rel: str) -> Any:
    return json.loads((REPO / rel).read_text())


def lookup(rel: str, keys: list[Any]) -> Any:
    x = load(rel)
    for k in keys:
        x = x[k]
    return x


def dump(name: str, out: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


def e53_windows() -> tuple[dict[str, torch.Tensor], int, Any]:
    """E53's scored windows per task, by the loader E60 and E136 called (`regating_truth_table.e53_tasks`), on CPU."""
    TT = load_experiment("regating_truth_table", "run")
    pa = TT.performed_anchor()
    tasks = TT.e53_tasks(pa, pa.Run.__new__(pa.Run))  # `scored` reads no attribute of Run
    return {n: tasks[n].held for n in TASKS}, pa.CFG.half, TT


# ----------------------------------------------------------------------------- the numbers (no model)


def window_stats(lp: np.ndarray, T: dict[str, Any]) -> dict[str, Any]:
    """One task: log10 of the geometric mean of P over windows, its bootstrap interval over windows, the percentiles of
    per-window log10 P. nan when any window's log P is not finite or there are no windows."""
    l10 = lp / math.log(10)
    if len(l10) == 0 or not np.isfinite(l10).all():
        return {"n_windows": len(l10), "log10_prior": math.nan, "ci95": [math.nan, math.nan], "percentiles": {}}
    gen = np.random.default_rng(T["seed_boot"])
    boot = [float(l10[gen.integers(0, len(l10), len(l10))].mean()) for _ in range(T["n_boot"])]
    pct = {str(q): float(np.percentile(l10, q)) for q in T["percentiles"]}
    return {"n_windows": len(l10), "log10_prior": float(l10.mean()), "prior": 10 ** float(l10.mean()), "sd_window": float(l10.std(ddof=1)) if len(l10) > 1 else math.nan,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "percentiles": pct}  # fmt: skip


def pair_gap(lps: dict[str, np.ndarray], T: dict[str, Any]) -> dict[str, Any]:
    """log10 prior of the reuse side's mean minus the new side's mean (one value per task, as the method table's gap), and its
    bootstrap interval with each task's windows resampled on their own (the tasks' windows are different positions)."""
    l10 = {n: v / math.log(10) for n, v in lps.items()}
    if any(len(v) == 0 or not np.isfinite(v).all() for v in l10.values()):
        return {"gap": math.nan, "ci95": [math.nan, math.nan]}
    gap = lambda m: float(np.mean([m[n] for n in REUSE]) - np.mean([m[n] for n in NEW]))
    gen = np.random.default_rng(T["seed_gap"])
    boot = [gap({n: v[gen.integers(0, len(v), len(v))].mean() for n, v in l10.items()}) for _ in range(T["n_boot"])]
    return {"gap": gap({n: v.mean() for n, v in l10.items()}), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def reading(g: dict[str, Any], R: dict[str, Any]) -> str:
    """The filed rule (predictions.json readings): a gap and its interval in log10 units."""
    gap, (lo, hi) = g["gap"], g["ci95"]
    if any(math.isnan(x) for x in (gap, lo, hi)):
        return "no_call"
    if gap >= R["higher_gap"] and lo >= R["alike_band"]:
        return "reuse_side_higher"
    if gap <= -R["higher_gap"] and hi <= -R["alike_band"]:
        return "new_side_higher"
    if -R["alike_band"] < lo and hi < R["alike_band"]:
        return "alike"
    return "no_call"


# ----------------------------------------------------------------------------- stages


def stage0(net: Any, P: dict[str, Any], held: dict[str, torch.Tensor], half: int, TT: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The controls, alone. Returns the record and each task's per-window log P (read here on the path stage 1 scores)."""
    S, lps = P["stage0"], {}
    rec: dict[str, Any] = {"rows": {}}
    pvs = load_experiment("prior_vs_share", "run")
    c = S["e133_control"]
    row = next(m for m in load(c["inventory"])["maps"] if m["name"] == c["map"])
    p, ref = pvs.forward_prior(net, row), lookup(c["file"], c["key"])
    rec["rows"][c["map"]] = {"prior": p, "saved": ref, "rel": p / ref - 1, "ok": abs(p / ref - 1) <= c["tol_rel"]}
    log(f"[stage0 {c['map']}] {json.dumps(rec['rows'][c['map']])}")
    for n in TASKS:
        w = S["windows"][n]
        t = TT.next_task(net, n, held[n], half)  # lp0 = W0's log P of each window's next token: the prior's input
        lo0 = float(ST.read(net, t, t.toks, readout=ST.logodds_at).mean())  # E53's T0, the windows check
        ref = lookup(w["file"], w["key"])
        lps[n] = t.lp0.numpy()
        rec["rows"][n] = {"T0": lo0, "saved": ref, "diff": lo0 - ref, "n_windows": len(t.toks), "ok": abs(lo0 - ref) <= w["tol"] and len(t.toks) == w["n_windows"]}
        log(f"[stage0 {n}] {json.dumps(rec['rows'][n])}")
    rec["pass"] = all(r["ok"] for r in rec["rows"].values())
    dump("stage0.json", rec)
    return rec, lps


def stage1(lps: dict[str, np.ndarray], P: dict[str, Any]) -> dict[str, Any]:
    T = P["rules"]
    rec: dict[str, Any] = {"tasks": {}, "sides": {"reuse": REUSE, "new": NEW}}
    for n in TASKS:
        rec["tasks"][n] = window_stats(lps[n], T) | {"log10_p_windows": (lps[n] / math.log(10)).tolist()}
        log(f"[stage1 {n}] {json.dumps(OP.jsonable({k: v for k, v in rec['tasks'][n].items() if k != 'log10_p_windows'}))}")
    rec["gap"] = pair_gap(lps, T)
    rec["reading"] = reading(rec["gap"], T)
    log(f"[stage1 gap] {json.dumps(OP.jsonable(rec['gap']))} reading {rec['reading']}")
    dump("stage1.json", rec)
    return rec


def main() -> None:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    held, half, TT = e53_windows()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # E133's and E60's setting
    net = ST.Net("cuda")
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()} windows {[len(held[n]) for n in TASKS]} half {half}")
    s0, lps = stage0(net, P, held, half, TT)
    if not s0["pass"]:
        log("FAIL: a control missed; stopping")
        hard_exit(1)
    print("STAGE 0 DONE", flush=True)
    stage1(lps, P)
    print("STAGE 1 DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
