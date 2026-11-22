"""E262 (P2.58): does definition (b)'s E_b, and each proxy's agreement with it, survive removing how far the output moved?

$0, saved results only, no new training. Two parts:
  A. P2.56's procedural states (score_rb.json .points): each proxy's Spearman with R_b (= E_b) and its partial Spearman
     given KL from the unchanged model, on trained states, on trained states without E220's repeat task, and on the
     random arms apart (never pooled).
  B. E257's nine built reuse/new pairs: E_b regressed on KL (filed), on update norm, on both and on log KL, by ordinary
     least squares over the unique trained pair states; each pair's gap (reuse minus new) raw and in residuals.
Rules and expected values: results/predictions.json. Writes results/summary.json and prints every number.

    ~/commute-agents/fleet/cpu-run python3 experiments/kl_control/run.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy.stats import rankdata

from commute.jobs import EXPERIMENTS, load_experiment, require_filed

RESULTS = Path(__file__).resolve().parent / "results"
AN = load_experiment("proxies_vs_r", "analyse")  # P2.51's spearman and partial_spearman
PROCEDURAL = EXPERIMENTS / "procedural_rl" / "results" / "score_rb.json"
E257 = EXPERIMENTS / "rb_states" / "results"
E205_ITEMS = EXPERIMENTS / "kstar_pairs" / "results" / "items"


def finite(v: Any) -> bool:
    return v is not None and math.isfinite(v)


def proxy_table(pts: list[dict[str, Any]], proxies: list[str]) -> dict[str, Any]:
    rb = {q["state"]: q["R_b"] for q in pts}
    kl = {q["state"]: q["kl"] for q in pts}
    out: dict[str, Any] = {"n_states": len(pts), "kl_raw": AN.spearman(kl, rb)["rho"]}
    for p in proxies:
        a = {q["state"]: q[p] for q in pts if finite(q[p])}
        raw, part = AN.spearman(a, rb), AN.partial_spearman(a, rb, kl)
        out[p] = {"raw": raw["rho"], "partial_given_kl": part["rho"], "n": part["n"]}
    return out


def controls(pts: list[dict[str, Any]]) -> dict[str, Any]:
    """KL given KL leaves nothing (null); R_b's rank residual on KL, given KL, is R_b's own remainder (1.0)."""
    ids = [q["state"] for q in pts]
    rb, kl = ({q["state"]: q[k] for q in pts} for k in ("R_b", "kl"))
    rr, rk = (rankdata([float(f"{d[i]:.6g}") for i in ids]) for d in (rb, kl))
    slope, icept = np.polyfit(rk, rr, 1)
    resid = dict(zip(ids, (rr - (slope * rk + icept)).tolist(), strict=True))
    return {"kl_given_kl": AN.partial_spearman(kl, rb, kl)["rho"], "rb_residual_given_kl": AN.partial_spearman(resid, rb, kl)["rho"]}


def part_a(st: dict[str, Any]) -> dict[str, Any]:
    pts = json.loads(PROCEDURAL.read_text())["points"]
    rand = set(st["random_arms"])
    trained = [q for q in pts if q["arm"] not in rand]
    no_repeat = [q for q in trained if not q["task"].startswith(st["repeat_task_prefix"])]
    randoms = [q for q in pts if q["arm"] in rand]
    return {"trained": proxy_table(trained, st["proxies"]), "trained_no_repeat": proxy_table(no_repeat, st["proxies"]),
            "random_arms": proxy_table(randoms, ["erank", "prior"]), "controls": controls(trained)}  # fmt: skip


def pair_kl(name: str) -> float:
    own = json.loads((E257 / "items" / f"{name.replace('/', '-')}.json").read_text())
    if finite(own.get("kl_fm_saved")):
        return own["kl_fm_saved"]
    return json.loads((E205_ITEMS / f"{name.replace('/', '-')}_block.json").read_text())["kl_fm"]


def design(rows: dict[str, dict[str, float]], which: str) -> np.ndarray:
    cols = {"kl": ["kl"], "norm": ["norm"], "kl+norm": ["kl", "norm"], "logkl": ["logkl"]}[which]
    return np.array([[1.0] + [r[c] for c in cols] for r in rows.values()])


def part_b(st: dict[str, Any]) -> dict[str, Any]:
    s = json.loads((E257 / "summary_items.json").read_text())
    floor = max(r["E_b"] for r in s["randoms"].values())
    rows: dict[str, dict[str, float]] = {}
    for p in s["pairs"].values():
        for it in p["reuse"]["items"] + p["new"]["items"]:
            k = pair_kl(it["item"])
            rows[it["item"]] = {"E_b": it["E_b"], "kl": k, "logkl": math.log(k), "norm": it["dw_rel_norm"]}
    y = np.array([r["E_b"] for r in rows.values()])
    fits: dict[str, Any] = {}
    resid: dict[str, dict[str, float]] = {}
    for which in st["regressors"]:
        x = design(rows, which)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        e = y - x @ beta
        fits[which] = {"coef": beta.tolist(), "r2": float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))}
        resid[which] = dict(zip(rows, e.tolist(), strict=True))
    pairs: dict[str, Any] = {}
    for name, p in s["pairs"].items():
        r, n = [i["item"] for i in p["reuse"]["items"]], [i["item"] for i in p["new"]["items"]]
        gap = lambda d: fmean(d[i] for i in r) - fmean(d[i] for i in n)  # noqa: E731
        pairs[name] = {"raw": gap({i: rows[i]["E_b"] for i in rows}), "e257_gap": p["gap_E_b"],
                       "kl_reuse": fmean(rows[i]["kl"] for i in r), "kl_new": fmean(rows[i]["kl"] for i in n),
                       **{f"resid_{w}": gap(resid[w]) for w in st["regressors"]}}  # fmt: skip
    count = lambda k: sum(abs(v[k]) > floor for v in pairs.values())  # noqa: E731
    return {"floor": floor, "n_states": len(rows), "spearman_eb_kl": AN.spearman({i: r["kl"] for i, r in rows.items()}, {i: r["E_b"] for i, r in rows.items()})["rho"],
            "fits": fits, "pairs": pairs, "outside_floor": {k: count(k) for k in ["raw"] + [f"resid_{w}" for w in st["regressors"]]},
            "states": rows}  # fmt: skip


def main() -> None:
    pred = RESULTS / "predictions.json"
    require_filed(pred)
    st = json.loads(pred.read_text())["settings"]
    out = {"experiment": "E262 (kl_control)", "cell": "P2.58", "A_procedural": part_a(st), "B_pairs": part_b(st)}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items()} | {"B_pairs": {k: v for k, v in out["B_pairs"].items() if k != "states"}}, indent=1))


if __name__ == "__main__":
    main()
