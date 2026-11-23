"""E275 (cell P2.62), CPU, $0: claim 2's comparisons recomputed with the write share E_b over the needed set K beside
E_b over the mean-ablation circuit K*, on the same states, through the functions that computed them first.

  A  procedural and repeat-task states (E262 part A, E267's lens rows): each reading's Spearman with E_b and its
     partial Spearman given KL (`kl_control.proxy_table`), over K* as published (every state, must repeat E262 and E267),
     over K* and over K on the states whose K has a changed member (E268 left its one empty-K state out). E_b over K and
     over K* from E268 (experiments/kstar_needed/results/summary.json).
  B  the nine built pairs' 76 states (E262 part B, E265, E266): E_b regressed on KL, each pair's gap raw and in
     residuals, the random floor, every reading's pair gap against E_b's (same sign, Spearman over the nine), the lens's
     KL-residual gaps against E_b's; E_b over K from this run's items (run.py stage pairs), over K* from E257.
  C  E254's prior-matched pair: E_b over K on the 20 seeds (run.py stage prior), copy against cipher (Mann-Whitney U),
     and each cheap reading's Spearman with it over the 20 states; seeds 0-2 at E257's size over K* and over K (rest).
A part whose items are not all home reads "pending" with the missing count. Writes results/summary.json, prints it.

    ~/commute-agents/fleet/cpu-run python3 experiments/claim2_needed/compare.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from commute import seed_rank as SR
from commute.jobs import EXPERIMENTS, load_experiment

KLC: Any = load_experiment("kl_control", "run")  # E262: proxy_table, design, pair_kl; P2.51's spearman (KLC.AN)
AN = KLC.AN
HERE = Path(__file__).resolve().parent
ITEMS = HERE / "results" / "items"
X = EXPERIMENTS


def read(p: Path) -> Any:
    return json.loads(p.read_text())


E268 = read(X / "kstar_needed" / "results" / "summary.json")["items"]
E262 = read(X / "kl_control" / "results" / "summary.json")
ST262 = read(X / "kl_control" / "results" / "predictions.json")["settings"]
E257 = read(X / "rb_states" / "results" / "summary_items.json")


def ours(item: str) -> dict[str, Any] | None:
    p = ITEMS / f"{item.replace('/', '-')}.json"
    r = read(p) if p.is_file() else None
    return r if r is not None and "stopped" not in r else None


def same(a: Any, b: Any) -> bool:
    """Equal numbers (None alike) in two nested results, to the last digit."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


# ---- A: procedural states --------------------------------------------------------------------------------------------


def e268_name(state: str) -> str:
    return f"rl/stage2b/{state}" if state.startswith("repeat_") else f"p256/{state}"


SMALL_K = 1000  # |K| under this is flagged (redteam-lead's E275 plan check: arms whose K differ 10-100x)
HALF = 1  # E268's by_derangement is cumulative: entry 1 is K from derangements {0, 1} only (128 of the 256 pairs)


def kept(r: dict[str, Any]) -> bool:
    """A state enters the K-side tables only when both its full K and its K from derangements {0, 1} hold a changed
    variable (E268 left its one empty-K state out), so the K*, K and half-K tables read the same states."""
    return r["n_changed_K"] > 0 and r["by_derangement"][HALF]["n_changed_K"] > 0


EB = {"Kstar": lambda r: r["E_b_Kstar"], "K": lambda r: r["E_b_K"], "K01": lambda r: r["by_derangement"][HALF]["E_b_K"]}


def over(pts: list[dict[str, Any]], which: str) -> list[dict[str, Any]]:
    """The points with R_b replaced by E268's E_b over K*, K or K from derangements {0, 1}, on the kept states."""
    return [q | {"R_b": EB[which](E268[e268_name(q["state"])])} for q in pts if kept(E268[e268_name(q["state"])])]


def moves(t: dict[str, Any], proxies: list[str]) -> dict[str, Any]:
    """Per reading and statistic (raw, partial given KL): the K* -> K move, the sampling floor |K01 - K| (K from 128
    pairs against K from 256: how far the reading moves as K's sample doubles), and whether the move clears twice the
    reading's own floor and twice the largest floor in the table."""
    out: dict[str, Any] = {}
    for p in proxies:
        for s in ("raw", "partial_given_kl"):
            a, b, h = (t[w][p][s] for w in ("Kstar", "K", "K01"))
            if None not in (a, b, h):
                out[f"{p}.{s}"] = {"Kstar": a, "K": b, "K01": h, "move": b - a, "floor": abs(h - b)}
    top = max((v["floor"] for v in out.values()), default=float("nan"))
    for v in out.values():
        v |= {"beyond_2x_own_floor": abs(v["move"]) > 2 * v["floor"], "beyond_2x_max_floor": abs(v["move"]) > 2 * top}
    return {"by_reading": out, "max_floor": top}


def three(pts: list[dict[str, Any]], proxies: list[str]) -> dict[str, Any]:
    t = {"published": KLC.proxy_table(pts, proxies)} | {w: KLC.proxy_table(over(pts, w), proxies) for w in EB}
    sizes = {q["state"]: E268[e268_name(q["state"])]["size_K"] for q in pts}
    return t | {"sampling": moves(t, proxies), "size_K": sizes, "small_K": sorted(s for s, n in sizes.items() if n < SMALL_K)}


def part_a() -> dict[str, Any]:
    pts = read(KLC.PROCEDURAL)["points"]
    dev = max(abs(E268[e268_name(q["state"])]["E_b_Kstar"] - q["R_b"]) / q["R_b"] for q in pts)
    rand = set(ST262["random_arms"])
    trained = [q for q in pts if q["arm"] not in rand]
    sets = {"trained": trained, "trained_no_repeat": [q for q in trained if not q["task"].startswith(ST262["repeat_task_prefix"])],
            "random_arms": [q for q in pts if q["arm"] in rand]}  # fmt: skip
    out: dict[str, Any] = {"E268_Kstar_vs_R_b_max_rel_dev": dev, "left_out": [q["state"] for q in pts if not kept(E268[e268_name(q["state"])])]}
    for name, s in sets.items():
        out[name] = three(s, ST262["proxies"] if name != "random_arms" else ["erank", "prior"])
        out[name]["repeats_E262"] = same(out[name]["published"], E262["A_procedural"][name])
    lens = {q["state"]: q for q in read(X / "adl" / "results" / "amend2" / "summary.json")["b"]["points"]}
    byst = {q["state"]: q for q in pts}
    rows = [byst[s] | {"adl": q["adl"], "collapsed": q["collapsed"]} for s, q in lens.items() if q["arm"] not in rand]
    e267 = read(X / "adl" / "results" / "amend2" / "summary.json")["b"]["tables"]
    for name, s in {"lens_trained": rows, "lens_trained_no_collapsed": [q for q in rows if not q["collapsed"]]}.items():
        out[name] = three(s, ["adl"])
        out[name]["repeats_E267"] = same(out[name]["published"]["adl"], e267[name.removeprefix("lens_")]["adl"])
    return out


# ---- B: the nine built pairs ----------------------------------------------------------------------------------------


def half(r: dict[str, Any]) -> float:
    """E_b over K from derangements {0, 1} of one of this run's items (E268's dict form; a bare number in items written
    before 2026-11-23's by_derangement change)."""
    h = r["by_derangement"][HALF]
    return h["E_b_K"] if isinstance(h, dict) else h


def gap(d: dict[str, float], reuse: list[str], new: list[str]) -> float:
    return fmean(d[i] for i in reuse) - fmean(d[i] for i in new)


def pair_table(eb: dict[str, float], floor: float) -> dict[str, Any]:
    """E262 part B's numbers for one E_b per state: OLS of E_b on KL (and E262's other regressors) over the unique
    states, each pair's gap (reuse mean minus new mean) raw and in residuals, the count outside the random floor."""
    rows: dict[str, dict[str, float]] = {}
    for p in E257["pairs"].values():
        for it in p["reuse"]["items"] + p["new"]["items"]:
            k = KLC.pair_kl(it["item"])
            rows[it["item"]] = {"E_b": eb[it["item"]], "kl": k, "logkl": math.log(k), "norm": it["dw_rel_norm"]}
    y = np.array([r["E_b"] for r in rows.values()])
    fits, resid = {}, {}
    for which in ST262["regressors"]:
        x = KLC.design(rows, which)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        e = y - x @ beta
        fits[which] = {"coef": beta.tolist(), "r2": float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))}
        resid[which] = dict(zip(rows, e.tolist(), strict=True))
    pairs: dict[str, Any] = {}
    for name, p in E257["pairs"].items():
        r, n = [i["item"] for i in p["reuse"]["items"]], [i["item"] for i in p["new"]["items"]]
        pairs[name] = {"raw": gap({i: rows[i]["E_b"] for i in rows}, r, n), "kl_gap": gap({i: rows[i]["kl"] for i in rows}, r, n),
                       **{f"resid_{w}": gap(resid[w], r, n) for w in ST262["regressors"]}}  # fmt: skip
    count = lambda k: sum(abs(v[k]) > floor for v in pairs.values())
    return {"floor": floor, "n_states": len(rows), "fits": fits, "pairs": pairs, "new_higher": sum(v["raw"] < 0 for v in pairs.values()),
            "spearman_eb_kl": AN.spearman({i: r["kl"] for i, r in rows.items()}, {i: r["E_b"] for i, r in rows.items()})["rho"],
            "outside_floor": {k: count(k) for k in ["raw"] + [f"resid_{w}" for w in ST262["regressors"]]}}  # fmt: skip


def reading_gaps() -> dict[str, dict[str, float]]:
    """Each reading's pair gap (reuse minus new) on the nine pairs: E265's columns and method_table's for the rest."""
    e265 = read(X / "adl" / "results" / "amend1" / "compare.json")["pairs"]
    out = {m: {f"e187_{p}": r[m] for p, r in e265.items()} for m in ("share", "neuron", "steering", "crosscoder", "adl")}
    final = [c for c in read(X / "method_table" / "results" / "final.json")["cells"] if c["group"] == "built"]
    for m in ("erank", "untrained", "prior", "size"):
        out[m] = {f"e187_{c['pair']}": c["gap"] for c in final if c["method"] == m}
    return out


def against(tab: dict[str, Any], gaps: dict[str, dict[str, float]], lens_resid: dict[str, float]) -> dict[str, Any]:
    eb = {p: v["raw"] for p, v in tab["pairs"].items()}
    allg = gaps | {"kl": {p: v["kl_gap"] for p, v in tab["pairs"].items()}}
    out = {m: {"same_sign": sum((g[p] > 0) == (eb[p] > 0) for p in g), "n": len(g), "spearman": AN.spearman(g, eb)["rho"]} for m, g in allg.items()}
    ebr = {p: tab["pairs"][f"e187_{p}"]["resid_kl"] for p in lens_resid}
    out["adl_resid_kl"] = {"same_sign": sum((lens_resid[p] > 0) == (ebr[p] > 0) for p in ebr), "n": len(ebr), "spearman": AN.spearman(lens_resid, ebr)["rho"]}
    return out


def part_b() -> dict[str, Any]:
    states = sorted({it["item"] for p in E257["pairs"].values() for it in p["reuse"]["items"] + p["new"]["items"]})
    star = {i: read(X / "rb_states" / "results" / "items" / f"{i.replace('/', '-')}.json")["E_b"] for i in states}
    gaps, lens = reading_gaps(), read(X / "adl" / "results" / "kl" / "summary.json")["fits"]["kl"]["gaps"]
    t_star = pair_table(star, E262["B_pairs"]["floor"])
    out: dict[str, Any] = {"Kstar": t_star | {"against": against(t_star, gaps, lens)}}
    out["repeats_E262"] = same({k: t_star[k] for k in ("floor", "n_states", "spearman_eb_kl", "outside_floor")},
                               {k: E262["B_pairs"][k] for k in ("floor", "n_states", "spearman_eb_kl", "outside_floor")})  # fmt: skip
    out["repeats_E265_E266"] = {"adl_same_sign": out["Kstar"]["against"]["adl"]["same_sign"], "adl_spearman": out["Kstar"]["against"]["adl"]["spearman"],
                                "adl_resid_spearman": out["Kstar"]["against"]["adl_resid_kl"]["spearman"]}  # fmt: skip
    got = {i: ours(i) for i in states}
    missing = [i for i, r in got.items() if r is None]
    out["missing"] = len(missing)
    if missing:
        out["K"] = "pending"
        return out
    rs = {i: r for i, r in got.items() if r is not None}
    t_k = pair_table({i: r["E_b_K"] for i, r in rs.items()}, max(E268[i]["E_b_K"] for i in E257["randoms"]))
    t_h = pair_table({i: half(r) for i, r in rs.items()}, max(EB["K01"](E268[i]) for i in E257["randoms"]))
    out["K"] = t_k | {"against": against(t_k, gaps, lens), "empty_K": [i for i, r in rs.items() if r["empty_K"]]}
    out["K01"] = t_h | {"against": against(t_h, gaps, lens)}
    out["size_K"] = {i: r["size_K"] for i, r in rs.items()}
    out["small_K"] = sorted(i for i, r in rs.items() if r["size_K"] < SMALL_K)
    out["sign_changes"] = {}
    for name, p in E257["pairs"].items():
        a, b = t_star["pairs"][name]["raw"], t_k["pairs"][name]["raw"]
        if (a > 0) != (b > 0):
            small = [i["item"] for i in p["reuse"]["items"] + p["new"]["items"] if rs[i["item"]]["size_K"] < SMALL_K]
            out["sign_changes"][name] = {"Kstar": a, "K": b, "K01": t_h["pairs"][name]["raw"], "small_K_states": small, "provisional": bool(small)}
    every = {r["item"]: r for p in (X / "rb_states" / "results" / "items").glob("*.json") if (r := read(p)).get("stage") == "items"}
    kk = {i: r for i in every if (r := ours(i)) is not None and not r["empty_K"]}
    out["range_94"] = {"n_read": len(kk), "of": len(every), "K": [min(r["E_b_K"] for r in kk.values()), max(r["E_b_K"] for r in kk.values())],
                       "Kstar_same_states": [min(every[i]["E_b"] for i in kk), max(every[i]["E_b"] for i in kk)]}  # fmt: skip
    return out


# ---- C: E254's prior-matched pair ------------------------------------------------------------------------------------


READINGS = ("r", "k3_share", "head_share", "erank", "steering_pooled", "steering_perkey", "neuron_k64", "neuron_kmatched")


def part_c() -> dict[str, Any]:
    pm = X / "prior_matched" / "results" / "a"
    rows = read(pm / "summary.json")["rows"]
    side = {"copy": rows["reused"], "prior_cipher": rows["new"]}
    reads: dict[str, dict[str, float]] = {m: {} for m in READINGS}
    for task, rs in side.items():
        for s, r in enumerate(rs):
            assert r["r"] == read(pm / "items" / f"e58-pa_{task}_s{s}.json")["r"], f"{task} s{s}: summary row order"
            for m in READINGS:
                reads[m][f"pa/{task}_s{s}"] = r[m]
    got = {i: ours(i) for i in reads["r"]}
    out: dict[str, Any] = {"missing": sum(r is None for r in got.values())}
    if out["missing"]:
        return out | {"K": "pending"}
    rs = {i: r for i, r in got.items() if r is not None}
    for which, eb in (("K", {i: r["E_b_K"] for i, r in rs.items()}), ("K01", {i: half(r) for i, r in rs.items()})):
        c, n = ([v for i, v in eb.items() if i.startswith(f"pa/{t}_")] for t in ("copy", "prior_cipher"))
        u, p = SR.mann_whitney(c, n)  # E254's exact two-sided test
        out[which] = {"copy": c, "prior_cipher": n, "gap": fmean(c) - fmean(n), "U": u, "p": p,
                      "spearman_with_E_b": {m: AN.spearman(reads[m], eb)["rho"] for m in READINGS}}  # fmt: skip
    out["size_K"] = {i: r["size_K"] for i, r in rs.items()}
    out["small_K"] = sorted(i for i, r in rs.items() if r["size_K"] < SMALL_K)
    s3 = {t: [ours(f"e58/{t}_s{s}_scaled") for s in range(3)] for t in side}
    if all(r is not None for v in s3.values() for r in v):
        out["seeds_0_2_E257_size"] = {key: {t: [r[key] for r in v if r] for t, v in s3.items()} for key in ("E_b_Kstar", "E_b_K")}
    return out


def main() -> None:
    out = {"experiment": "E275 (claim2_needed)", "cell": "P2.62", "A_procedural": part_a(), "B_pairs": part_b(), "C_prior_matched": part_c()}
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
