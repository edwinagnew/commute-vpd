"""E300 (P2.63) score, CPU, no model: the filed rule (results/predictions.json "rule") on the dial states' records
(results/states/*.json), E_b over K against the dial (the sanity check), and where the ordinary fine-tunes and E256's
saved SFT and RL states (their readings from E256, E267 and E268, as saved) land on the dial. Writes results/score.json.

    ~/commute-agents/fleet/cpu-run python3 experiments/reuse_dial/score.py [--states DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy.stats import rankdata

from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E256_READS = EXPERIMENTS / "procedural_rl" / "results" / "read"
E267_ITEMS = EXPERIMENTS / "adl" / "results" / "amend2" / "items"
E268 = EXPERIMENTS / "kstar_needed" / "results" / "summary.json"
SAVED_ARMS = ("sft", "rl", "rlg")


def num(x: Any) -> float:
    """A saved number, nan when missing or non-finite (jsonable writes nan as a string)."""
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else math.nan


def readings(rd: dict[str, Any], eb: dict[str, Any] | None) -> dict[str, float]:
    """Every reading of one state from its read record (E211's reader as E256 saved it, `brief`'s keys) and E_b."""
    lo = lambda d: num(d.get("lo")) if isinstance(d, dict) else math.nan
    return {"share": lo(rd["share"]["share"]), "steering": lo(rd["steering"]["share"]), "erank": num(rd["erank"]),
            "kl": num(rd.get("kl", rd.get("kl_saved"))), "lens": num(rd["lens"]["D"]) if "lens" in rd else math.nan,
            "neuron_k64": lo(rd["neuron"]["k64"]["share"]), "neuron_kmatched": lo(rd["neuron"]["kmatched"]["share"]),
            "untrained_svd": num(rd["untrained"]["share"]["lo"]["svd"]), "gain": lo(rd["gain"]), "prior": lo(rd["prior"]),
            "E_b_K": num(eb["E_b_K"]) if eb and "E_b_K" in eb else math.nan}  # fmt: skip


def spearman(x: np.ndarray, d: np.ndarray) -> float:
    rx, rd = rankdata(x), rankdata(d)
    if rx.std() == 0 or rd.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, rd)[0, 1])


def rho_p(x: list[float], d: list[int], R: dict[str, Any], gen: np.random.Generator) -> dict[str, Any]:
    keep = [i for i, v in enumerate(x) if math.isfinite(v)]
    xs, ds = np.array([x[i] for i in keep]), np.array([d[i] for i in keep])
    rho = spearman(xs, ds) if len(keep) >= 6 else math.nan
    if not math.isfinite(rho):
        return {"rho": None, "p": None, "n": len(keep), "tracks": False}
    perm = np.array([spearman(xs, gen.permutation(ds)) for _ in range(R["n_perm"])])
    p = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
    return {"rho": rho, "p": p, "n": len(keep), "tracks": p < R["alpha"]}


def dial_order(dial: list[dict[str, Any]], arms: list[str]) -> float:
    """Spearman of the measured residual share with d over the penalised arms' states (the ends are 0 and 1 by
    construction and are left out); nan when fewer than two penalised levels have a finite share, or it is constant."""
    pen = [s for s in dial if s["arm"] in arms[1:-1] and math.isfinite(s["residual_share"])]
    d = np.array([arms.index(s["arm"]) for s in pen])
    return spearman(np.array([s["residual_share"] for s in pen]), d) if len(set(d.tolist())) >= 2 else math.nan


def task_table(states: list[dict[str, Any]], arms: list[str], R: dict[str, Any], gen: np.random.Generator) -> dict[str, Any]:
    """One task: the dial levels reached, the dial's measured order, each reading's rho with d and permutation p, and
    the checks (E_b falls along the dial; the read gain does not)."""
    dial = [s for s in states if s["arm"] in arms and s["reached"]]
    levels = {a: sum(s["arm"] == a for s in dial) for a in arms}
    order = dial_order(dial, arms)
    readable = sum(n >= R["min_per_level"] for n in levels.values()) >= R["min_levels"] and order >= R["dial_rho_min"]
    d = [arms.index(s["arm"]) for s in dial]
    names = [*R["cheap"], *R["construction"], *R["baselines"], "untrained_svd", "E_b_K", "prior", "gain"]
    cols = {n: rho_p([s["x"][n] for s in dial], d, R, gen) for n in names}
    eb, g = cols["E_b_K"]["rho"], cols["gain"]["rho"] if cols["gain"]["tracks"] else None
    return {"levels": levels, "missing": [a for a, n in levels.items() if n < R["min_per_level"]], "dial_order": order if math.isfinite(order) else None,
            "readable": readable, "n": len(dial), "rho": cols, "eb_moves": eb is not None and eb <= R["eb_rho_max"],
            "gain_moves": g is not None and abs(g) > R["gain_rho_max"]}  # fmt: skip


def beats_kl(t: dict[str, Any], x: str, R: dict[str, Any]) -> bool:
    c, k = t["rho"][x], t["rho"]["kl"]["rho"]
    return bool(c["tracks"]) and abs(c["rho"]) - (abs(k) if k is not None else 0.0) >= R["kl_margin"]


def verdict(tables: dict[str, dict[str, Any]], R: dict[str, Any]) -> dict[str, Any]:
    ok = [t for t, v in tables.items() if v["readable"]]
    if len(ok) < 3:
        return {"verdict": "no call", "readable": ok, "why": "fewer than 3 readable tasks"}
    need, per = max(3, len(ok) - 1), {}
    for x in [*R["cheap"], *R["construction"]]:  # the construction check is reported the same way, never a winner
        beat = [t for t in ok if beats_kl(tables[t], x, R)]
        signs = {t: math.copysign(1, tables[t]["rho"][x]["rho"]) for t in beat}
        best = max((sum(v == s for v in signs.values()) for s in (1, -1)), default=0)
        per[x] = {"beats_kl_on": beat, "same_sign_count": best}
    win = [x for x in R["cheap"] if per[x]["same_sign_count"] >= need]
    return {"verdict": "rival a" if win else "rival b", "readable": ok, "need": need, "cheap": per, "winners": win}


def placement(dial: list[dict[str, Any]], arms: list[str], others: dict[str, dict[str, float]], names: list[str]) -> dict[str, Any]:
    """Per other state and reading: the dial level whose mean reading is nearest, and the share of dial states whose
    reading is on the gains-only side of it (the side of level 0's mean)."""
    out: dict[str, Any] = {}
    for st, x in others.items():
        out[st] = {"gain": x["gain"]}
        for n in names:
            means = {a: fmean(v) for a in arms if (v := [s["x"][n] for s in dial if s["arm"] == a and math.isfinite(s["x"][n])])}
            if not math.isfinite(x[n]) or len(means) < 2:
                out[st][n] = None
                continue
            near = min(means, key=lambda a: abs(means[a] - x[n]))
            side = math.copysign(1, means[arms[0]] - x[n]) if arms[0] in means else 0
            vals = [s["x"][n] for s in dial if math.isfinite(s["x"][n])]
            out[st][n] = {"value": x[n], "nearest_level": near, "share_on_reuse_side": fmean(1.0 if (v - x[n]) * side > 0 else 0.0 for v in vals)}
    return out


def saved(task: str) -> dict[str, dict[str, float]]:
    """E256's saved SFT and RL states of a procedural task, read as saved (E256 reader, E267 lens, E268 E_b over K)."""
    if not E256_READS.is_dir():
        return {}
    e268 = json.loads(E268.read_text())["items"] if E268.is_file() else {}
    out = {}
    for f in sorted(E256_READS.glob(f"{task}_*_s*.json")):
        rd = json.loads(f.read_text())
        if rd["arm"] not in SAVED_ARMS:
            continue
        lens = E267_ITEMS / f"p256-{f.stem}.json"
        rd = rd | ({"lens": json.loads(lens.read_text())} if lens.is_file() else {})
        out[f.stem] = readings(rd, e268.get(f"p256/{f.stem}"))
    return out


def load_states(d: Path) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "read" not in r or "stopped" in r or r["state"].startswith("stage0_"):
            continue
        task, tr = r["state"].split("_", 1)[0], r["train"]
        by.setdefault(task, []).append({"state": r["state"], "arm": tr["arm"], "seed": tr["seed"], "reached": bool(tr.get("reached", False)),
                                        "residual_share": num(tr.get("residual_share")), "x": readings(r["read"], r.get("eb"))})  # fmt: skip
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=HERE / "results" / "states")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "score.json")
    a = ap.parse_args()
    P = json.loads(PRED.read_text())
    R, betas = P["rule"], P["settings"]["betas"]
    arms = ["g", *[f"b{b:g}" for b in sorted(betas, reverse=True)], "ft"]
    gen, by = np.random.default_rng(R["perm_seed"]), load_states(a.states)
    tables = {t: task_table(s, arms, R, gen) for t, s in by.items()}
    names = [*R["cheap"], *R["construction"], *R["baselines"], "E_b_K"]
    place = {}
    for t, s in by.items():
        dial = [x for x in s if x["arm"] in arms and x["reached"]]
        others = {x["state"]: x["x"] for x in s if x["arm"] == "ft"} | saved(t) | {x["state"]: x["x"] for x in s if x["arm"] == "rand"}
        place[t] = placement(dial, arms, others, names)
    out = {"experiment": P["experiment"], "arms": arms, "tables": tables, "verdict": verdict(tables, R), "placement": place}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"verdict": out["verdict"], "tables": {t: {"levels": v["levels"], "dial_order": v["dial_order"], "readable": v["readable"],
                      "eb_moves": v["eb_moves"], "gain_moves": v["gain_moves"],
                      "rho": {n: c["rho"] for n, c in v["rho"].items()}} for t, v in tables.items()}}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
