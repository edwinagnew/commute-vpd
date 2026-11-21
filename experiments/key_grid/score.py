"""E237's score (no GPU): per reading and task, the three seeds at the common size, their mean and range, and the random
control's band (min to max over the task's three draws); each reading's rank order against the prior; and the filed
rules of results/predictions.json -> results/summary.json. Reads whatever item files are present; a missing one is None.

    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/score.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.elicit import separation

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
P = json.loads((RESULTS / "predictions.json").read_text())
S = P["settings"]
TASKS, SEEDS = S["tasks"], S["seeds"]
PRIOR = REPO / "experiments" / "prior_vs_share" / "results" / "stage1.json"
SHARE = ("r", "kl_share", "size", "erank")
LEVELS = ("head_mlp", "mlp_attn", "block")
READINGS = (*SHARE, *(f"R_{lv}" for lv in LEVELS), "R_topk", "R_vpd")


def num(x: Any) -> float | None:
    """A finite number, else None (NaN, a string 'nan', a missing file or key)."""
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def fname(item: str) -> str:
    return item.replace("/", "-")


def item_of(state: str, kind: str) -> str:
    return f"{state}_scaled" if kind == "scaled" else f"random_{state}"


def r_coarse(item: str, lv: str) -> float | None:
    rec = load(RESULTS / "items" / f"{fname(item)}_{lv}.json")
    if rec is None or rec.get("reading") != "read":
        return None
    return fmean(r["R_read"] for r in rec["searches"]["main"])  # head_mlp: the mean over E221's three orders


def reading(st1: dict[str, Any], state: str, kind: str, name: str) -> float | None:
    item = item_of(state, kind)
    if name in SHARE:
        rec = st1.get(state)
        return num(rec[kind][name]) if rec else None
    if name.startswith("R_") and name[2:] in LEVELS:
        return r_coarse(item, name[2:])
    if name == "R_topk":
        rec = load(RESULTS / "topk" / f"{fname(item)}.json")
        return num(rec["R_read"]) if rec else None
    rec = load(RESULTS / "vpd" / f"{fname(item)}.json")
    return num(rec["searches"]["main"]["polished"]["R_read"]) if rec and rec.get("reading") == "read" else None


def table() -> dict[str, dict[str, dict[str, Any]]]:
    """reading -> task -> {seeds, mean, range, random (the draws), band [min, max]}; None where not all read."""
    st1 = (load(RESULTS / "stage1.json") or {}).get("states", {})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name in READINGS:
        out[name] = {}
        for t in TASKS:
            states = [f"e58/{t}_s{s}" for s in SEEDS]
            v, rnd = [reading(st1, s, "scaled", name) for s in states], [reading(st1, s, "random", name) for s in states]
            full, rfull = all(x is not None for x in v), all(x is not None for x in rnd)
            out[name][t] = {"seeds": v, "mean": fmean(v) if full else None, "range": max(v) - min(v) if full else None,  # type: ignore[type-var,operator,arg-type]
                            "random": rnd, "band": [min(rnd), max(rnd)] if rfull else None}  # type: ignore[type-var]
    return out


def priors() -> dict[str, float]:
    p = json.loads(PRIOR.read_text())
    return {t: math.log10(p[f"e58_96_{t}"]["prior"]) for t in TASKS}


def placement(row: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Where prior_cipher sits: 'with cipher' when its mean is nearer cipher's and it separates from copy by E51's rule;
    'with copy' when nearer copy's and it separates from cipher; else 'no call'; 'unreadable' when a seed is missing."""
    pc, ci, co = (row[t]["seeds"] for t in ("prior_cipher", "cipher", "copy"))
    to_ci, to_co = separation(pc, ci, 0.0), separation(pc, co, 0.0)
    if "unreadable" in (to_ci["verdict"], to_co["verdict"]):
        return {"verdict": "unreadable"}
    d_ci, d_co = abs(to_ci["gap"]), abs(to_co["gap"])
    v = "with cipher" if d_ci < d_co and to_co["verdict"] != "none" else "with copy" if d_co < d_ci and to_ci["verdict"] != "none" else "no call"
    return {"verdict": v, "dist_cipher": d_ci, "dist_copy": d_co, "sep_from_cipher": to_ci["verdict"], "sep_from_copy": to_co["verdict"]}


def spearman(x: list[float], y: list[float]) -> float:
    rank = lambda v: [sorted(v).index(a) for a in v]  # the five task means: ties are not expected and break low
    rx, ry = rank(x), rank(y)
    n = len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True)) / (n * (n * n - 1))


def against_band(row: dict[str, Any], below: bool) -> bool | None:
    """Every seed below the task's random band (below=True) or above it; None when either is not read."""
    if row["band"] is None or row["mean"] is None:
        return None
    return max(row["seeds"]) < row["band"][0] if below else min(row["seeds"]) > row["band"][1]


def rules(tab: dict[str, dict[str, dict[str, Any]]], th: dict[str, Any]) -> dict[str, Any]:
    kind = [separation(tab["r"][a]["seeds"], tab["r"][b]["seeds"], th["kind_gap_min"]) | {"cell": f"{a}-{b}"}
            for a in ("copy", "rank1", "elicit") for b in ("cipher", "prior_cipher")]  # fmt: skip
    in_range = lambda name, lo, hi: [v for t in TASKS for v in tab[name][t]["seeds"] if v is None or not lo <= v <= hi]
    return {
        "P1": placement(tab["r"]) | {"lean": "with cipher"},
        "P2": {"cells": kind, "pass": all(k["verdict"] == "a_higher" for k in kind), "lean": True},
        "P3": placement(tab["kl_share"]) | {"lean": "with cipher"},
        "P4": {t: against_band(tab["kl_share"][t], t != "copy") for t in ("cipher", "prior_cipher", "copy")} | {"lean": "all True"},
        "P5": {f"R_{lv}": placement(tab[f"R_{lv}"]) for lv in LEVELS} | {"lean": "R_block with copy; head_mlp, mlp_attn no lean"},
        "P6": {"R_topk_out_of_range": in_range("R_topk", *th["topk_range"]), "R_vpd_out_of_range": in_range("R_vpd", *th["vpd_range"]),
               "R_topk": placement(tab["R_topk"]), "R_vpd": placement(tab["R_vpd"]), "lean": "no value out of range; placement no lean"},
        "P7": {t: against_band(tab["erank"][t], True) for t in TASKS} | {"lean": "all True"},
    }  # fmt: skip


def score() -> dict[str, Any]:
    tab, lp = table(), priors()
    order = sorted(TASKS, key=lambda t: lp[t])
    rho = {n: spearman([tab[n][t]["mean"] for t in TASKS], [lp[t] for t in TASKS]) if all(tab[n][t]["mean"] is not None for t in TASKS) else None
           for n in READINGS}  # fmt: skip
    out = {"experiment": P["experiment"], "x_order_by_prior": order, "log10_prior": lp, "table": tab, "spearman_vs_prior": rho, "rules": rules(tab, P["thresholds"])}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    for k, v in out["rules"].items():
        print(k, json.dumps(v)[:400])
    print("spearman vs prior", rho)
    return out


if __name__ == "__main__":
    score()
