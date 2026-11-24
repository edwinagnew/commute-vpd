"""P1.59 (user 2026-11-23 ~17:20 via whip-lead): why the needed-set readings fail on trained fine-tunes, a $0 synthesis
from the saved per-variable totals of E268, E275, E277, E278, E281 and E283 (experiments/why_fail/inventory.tsv, written
by inventory.py). Descriptive; rules and filed expected values in results/predictions.json, which the run refuses
unless committed unchanged. Card: card.md.

Per variable u the saved sums over every position of D (commute.stand_in.write_change): num = sum <w_F,u, w_M,u>,
den = sum |w_M,u|^2, dd = sum |w_F,u - w_M,u|^2. Its kept share is stand_in.explained(num, den) / dd: the part of its
write change that M's own write at u's least-squares gain explains (1 = a pure rescale, 0 = none along M's write).

    ~/commute-agents/fleet/cpu-run python3 experiments/why_fail/run.py edge   # the rules on edge cases, synthetic
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/why_fail/run.py all
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"

# Variable layouts (n = 38,938 at VPD, 12,316 at neurons; checked against every item's n).
# VPD (kstar_vpd.featurize): 0 the embedding, then per layer the sorted module keys attn.k_proj, attn.o_proj,
# attn.q_proj, attn.v_proj, mlp.c_fc, mlp.down_proj, each C subcomponents then its remainder, then the unembedding.
VPD_C = (512, 1024, 512, 1024, 3072, 3584)  # artifacts/vpd/s-55ea3f9b/final_config.yaml, in that order
VPD_PER_LAYER = sum(VPD_C) + len(VPD_C)
# Neurons (needed_neurons.PER_LAYER): per layer 6 heads, 3,072 neurons, one remainder.
NEU_PER_LAYER, NEU_HEADS, LAYERS = 3079, 6, 4


def layout(feat: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(layer per variable, -1 outside the layers; remainder mask)."""
    if feat == "vpd":
        assert n == 2 + LAYERS * VPD_PER_LAYER, n
        j = np.arange(n) - 1
        layer = np.where((j >= 0) & (j < LAYERS * VPD_PER_LAYER), j // VPD_PER_LAYER, -1)
        rem_local = np.cumsum(np.array(VPD_C) + 1) - 1
        return layer, (layer >= 0) & np.isin(j % VPD_PER_LAYER, rem_local)
    assert n == LAYERS * NEU_PER_LAYER, n
    j = np.arange(n)
    return j // NEU_PER_LAYER, j % NEU_PER_LAYER == NEU_PER_LAYER - 1


def kept(num: np.ndarray, den: np.ndarray, dd: np.ndarray) -> np.ndarray:
    """Per variable: explained / dd (stand_in.explained's expression), NaN where dd = 0."""
    pos = (den > 0) & (num > 0)
    ex = np.where(pos, (num - den) ** 2 / np.where(pos, den, 1.0), 0.0)
    return np.where(dd > 0, ex / np.where(dd > 0, dd, 1.0), np.nan)


def top_set(flips: np.ndarray, pool: np.ndarray, frac: float) -> np.ndarray:
    """The most-needed variables of `pool`: the ceil(frac |pool|) largest flip counts (at least one when the pool is
    not empty), every variable tied with the last one included. Empty when the pool is."""
    m = int(pool.sum())
    if m == 0:
        return np.zeros_like(pool)
    k = max(1, math.ceil(frac * m))
    cut = np.sort(flips[pool])[::-1][k - 1]
    return pool & (flips >= cut)


def reading1(a: dict[str, np.ndarray], frac: float) -> dict[str, Any]:
    """Kept share of the top `frac` of K & changed & (dd > 0) by flips: its median and its dd-weighted share."""
    pool = a["in_K"] & a["changed"] & (a["dd"] > 0)
    top = top_set(a["flips"], pool, frac)
    if not top.any():
        return {"pool": 0, "top_n": 0, "top_kept_median": None, "top_E_b": None, "pool_kept_median": None}
    ks = kept(a["num"], a["den"], a["dd"])
    ex = ks[top] * a["dd"][top]
    return {"pool": int(pool.sum()), "top_n": int(top.sum()), "top_kept_median": float(np.median(ks[top])),
            "top_E_b": float(ex.sum() / a["dd"][top].sum()), "pool_kept_median": float(np.median(ks[pool]))}  # fmt: skip


def reading2(a: dict[str, np.ndarray]) -> dict[str, Any]:
    """|K| / #changed, |K & changed| / #changed and |K| / n; None where nothing changed."""
    n_ch, k = int(a["changed"].sum()), int(a["in_K"].sum())
    out = {"n_vars": len(a["in_K"]), "size_K": k, "n_changed": n_ch, "K_over_n": k / len(a["in_K"])}
    if n_ch == 0:
        return out | {"K_over_changed": None, "coverage": None}
    return out | {"K_over_changed": k / n_ch, "coverage": int((a["in_K"] & a["changed"]).sum()) / n_ch}


def reading3(a: dict[str, np.ndarray], feat: str) -> dict[str, Any]:
    """Shares of dd over K & changed: on the remainders, and per layer. None where that dd sums to 0."""
    layer, rem = layout(feat, len(a["dd"]))
    sel = a["in_K"] & a["changed"]
    tot = float(a["dd"][sel].sum())
    if tot == 0:
        return {"rem_share": None, "layer_share": None, "max_layer_share": None, "outside_layers_share": None}
    ls = [float(a["dd"][sel & (layer == i)].sum()) / tot for i in range(LAYERS)]
    return {"rem_share": float(a["dd"][sel & rem].sum()) / tot, "layer_share": ls, "max_layer_share": max(ls),
            "outside_layers_share": float(a["dd"][sel & (layer < 0)].sum()) / tot}  # fmt: skip


# ---- edge cases (synthetic arrays; run before filing) ---------------------------------------------------------------


def edge() -> None:
    n = 2 + LAYERS * VPD_PER_LAYER
    z = lambda: np.zeros(n)
    base = {"in_K": np.zeros(n, bool), "changed": np.zeros(n, bool), "flips": np.zeros(n, int), "num": z(), "den": z(), "dd": z()}
    cases = {"do-nothing (empty K, no change)": base}
    a = {k: v.copy() for k, v in base.items()}
    a["in_K"][1:6], a["flips"][1:6] = True, [5, 4, 4, 1, 1]
    cases["K not empty, nothing changed"] = a
    b = {k: v.copy() for k, v in a.items()}
    b["changed"][1:6] = True
    cases["changed, every dd 0"] = b
    c = {k: v.copy() for k, v in b.items()}
    c["den"][1:6], c["dd"][1:6] = 1.0, 1.0
    c["num"][1:6] = [2.0, 1.0, 1.5, -1.0, 1.0]  # gains 2 (pure rescale, dd 1 = (2-1)^2), 1, 1.5, negative, 1
    cases["five pooled (top 1% < 1 -> 1, ties at the cut)"] = c
    d = {k: v.copy() for k, v in c.items()}
    d["flips"][1:6] = [4, 4, 4, 1, 1]
    cases["tie at the top: 3 of 5 kept in the top set"] = d
    e = {k: v.copy() for k, v in c.items()}
    e["den"][1] = 0.0
    cases["den 0 (M writes nothing, F does)"] = e
    out = {name: reading1(x, 0.01) | reading2(x) | reading3(x, "vpd") for name, x in cases.items()}
    lay, rem = layout("vpd", n)
    out["layout vpd"] = {"remainders": int(rem.sum()), "outside": int((lay < 0).sum()), "first_rem": int(np.flatnonzero(rem)[0])}
    lay, rem = layout("neurons", LAYERS * NEU_PER_LAYER)
    out["layout neurons"] = {"remainders": int(rem.sum()), "first_rem": int(np.flatnonzero(rem)[0])}
    print(json.dumps(out, indent=1))


# ---- the saved items ------------------------------------------------------------------------------------------------


def load(r: dict[str, str]) -> dict[str, np.ndarray]:
    """in_K, flips, num, den, dd, changed of one inventory row (E281: on the copy windows; changed from E257's arrays)."""
    k = np.load(REPO / r["k_npz"])
    a = np.load(REPO / r["arrays_npz"])
    if r["experiment"] == "E281":
        ch = np.load(REPO / "experiments" / "rb_states" / "results" / "large" / f"items_{r['item'].replace('/', '-')}.npz")["changed"]
        out = {"in_K": k["in_Kcopy"], "num": a["num_copy"], "den": a["den_copy"], "dd": a["dd_copy"], "changed": ch}
    else:
        if "changed" in a.files:
            ch = a["changed"]
        else:  # E268's built edits: E242's changed set
            ch = np.load(REPO / "experiments" / "kstar_vpd_wording_a" / "results" / "large" / f"items_{r['item'].replace('/', '-')}.npz")["changed"]
        out = {"in_K": k["in_K"], "num": a["num"], "den": a["den"], "dd": a["dd"], "changed": ch}
    out = {k_: np.asarray(v).astype(bool if k_ in ("in_K", "changed") else np.float64) for k_, v in out.items()}
    out["flips"] = np.asarray(k["flips"]).astype(np.int64)
    assert len({len(v) for v in out.values()}) == 1, r["item"]
    return out


def eb(a: dict[str, np.ndarray], sel: np.ndarray) -> float:
    """stand_in.explained_share in numpy: 1 when sum dd over sel is 0."""
    tot = float(a["dd"][sel].sum())
    ks = np.nan_to_num(kept(a["num"], a["den"], a["dd"]))
    return 1.0 if tot == 0 else float((ks * a["dd"])[sel].sum()) / tot


def saved_eb(r: dict[str, str]) -> float | None:
    d = {"E268": "kstar_needed", "E275": "claim2_needed", "E277": "needed_neurons", "E278": "eb_dropped", "E281": "e62_copy_windows"}.get(r["experiment"])
    if d is None:
        return None
    sep = "__" if r["experiment"] == "E277" else "-"
    j = json.loads((REPO / "experiments" / d / "results" / "items" / f"{r['item'].replace('/', sep)}.json").read_text())
    return j.get("E_b_Kcopy" if r["experiment"] == "E281" else "E_b_K")


def checks(rows: list[dict[str, Any]], arrays: dict[tuple[str, str], dict[str, np.ndarray]]) -> list[str]:
    """The filed layout checks; any miss stops the run."""
    miss = []
    for (exp, item), a in arrays.items():
        if exp in ("E268", "E275", "E278", "E281"):
            lay, _ = layout("vpd", len(a["dd"]))
            if (a["dd"][lay < 0] != 0).any():
                miss.append(f"{exp} {item}: nonzero dd outside the 24 matrices")
    x2, x10 = arrays[("E268", "x2")], arrays[("E268", "x10")]
    j = np.flatnonzero(x2["changed"]) - 1
    if not (len(j) == VPD_C[-1] + 1 and len(set(j // VPD_PER_LAYER)) == 1 and (j % VPD_PER_LAYER).min() == VPD_PER_LAYER - VPD_C[-1] - 1):
        miss.append(f"x2's changed set at VPD is not one down_proj block ({len(j)} variables)")
    # Amendment 1 (2026-11-23, reports/analyses/2026-11-24_analysis_p1-59-why-readings-fail.md, Departures): the filed "x10's kept shares equal x2's" is false by the rule's
    # own positive-gain convention (gain g - 1 scales by 9, so a gain under 8/9 at x2 turns negative and reads 0 at x10).
    # What linearity does fix is dd(x10) = 81 dd(x2), checked dd-weighted.
    s = x2["changed"] & (x2["dd"] > 0)
    dev = float(np.abs(x10["dd"][s] - 81 * x2["dd"][s]).sum() / (81 * x2["dd"][s]).sum())
    rows.append({"check": "x10 vs x2 at VPD: sum |dd10 - 81 dd2| / sum 81 dd2 over x2's changed variables", "value": dev})
    k2, k10 = kept(x2["num"], x2["den"], x2["dd"])[s], kept(x10["num"], x10["den"], x10["dd"])[s]
    rows.append({"check": "x10 vs x2 at VPD: changed variables whose kept share differs by over 1e-3 (the filed check)", "value": int((np.abs(k10 - k2) > 1e-3).sum())})
    if dev > 1e-6:
        miss.append(f"x10's dd at VPD is not 81 times x2's ({dev:.3g} relative)")
    return miss


def item_row(r: dict[str, str], a: dict[str, np.ndarray], frac: float) -> dict[str, Any]:
    out = {k: r[k] for k in ("experiment", "item", "feat", "cls")}
    out |= reading1(a, frac) | reading2(a) | reading3(a, r["feat"])
    fl = lambda s: float(s) if s not in ("", "None") else None
    out |= {"margin_min": fl(r["margin_min"]), "margin_median": fl(r["margin_median"]), "too_small": out["size_K"] < 100}
    out |= {"E_b_K": eb(a, a["in_K"] & a["changed"]), "E_b_K_saved": saved_eb(r)}
    return out | unfiled(a, frac)


def unfiled(a: dict[str, np.ndarray], frac: float) -> dict[str, Any]:
    """Added after R1's control failed (not filed; reported as such): does the change sit on the most-needed variables
    (their share of the pool's dd), and are the biggest changes rescales (E_b over the top `frac` of the pool by dd)."""
    pool = a["in_K"] & a["changed"] & (a["dd"] > 0)
    if not pool.any():
        return {"top_dd_share": None, "bigdd_E_b": None}
    top = top_set(a["flips"], pool, frac)
    big = top_set(a["dd"], pool, frac)
    return {"top_dd_share": float(a["dd"][top].sum() / a["dd"][pool].sum()), "bigdd_E_b": eb(a, big)}


def stats(v: list[float | None]) -> dict[str, Any]:
    x = np.array([u for u in v if u is not None], dtype=float)
    if len(x) == 0:
        return {"n": 0, "skipped": len(v)}
    q = np.percentile(x, [25, 50, 75])
    return {"n": len(x), "skipped": len(v) - len(x), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]), "min": float(x.min()), "max": float(x.max())}


KEYS = ("top_kept_median", "top_E_b", "pool_kept_median", "E_b_K", "coverage", "K_over_changed", "K_over_n", "margin_min", "margin_median",
        "rem_share", "max_layer_share", "top_dd_share", "bigdd_E_b")  # fmt: skip


def spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def main() -> None:
    from commute.jobs import require_filed

    require_filed(PRED)
    frac = json.loads(PRED.read_text())["settings"]["top_frac"]
    inv = [r for r in csv_rows(HERE / "inventory.tsv") if not r["missing"].startswith("dup:")]
    arrays = {(r["experiment"], r["item"]): load(r) for r in inv}
    check_rows: list[dict[str, Any]] = []
    miss = checks(check_rows, arrays)
    if miss:
        sys.exit("STOP: " + "; ".join(miss))
    items = [item_row(r, arrays[(r["experiment"], r["item"])], frac) for r in inv]
    pooled = [i for i in items if i["experiment"] not in ("E281", "E283")]  # other D: beside their class, not in it
    classes: dict[str, Any] = {}
    for feat in ("vpd", "neurons"):
        for c in sorted({i["cls"] for i in pooled if i["feat"] == feat}):
            g = [i for i in pooled if i["feat"] == feat and i["cls"] == c]
            classes[f"{feat}/{c}"] = {k: stats([i[k] for i in g]) for k in KEYS} | {"items": len(g)}
            mm = [i["margin_min"] for i in g if i["margin_min"] is not None]
            classes[f"{feat}/{c}"]["frac_margin_min_below_0.01"] = sum(m < 0.01 for m in mm) / len(mm) if mm else None
            classes[f"{feat}/{c}"]["layer_share_median"] = [float(np.median([i["layer_share"][L] for i in g if i["layer_share"]])) for L in range(LAYERS)]
    rho = {}
    for feat in ("vpd", "neurons"):
        g = [i for i in pooled if i["feat"] == feat and i["margin_min"] is not None and i["n_changed"] > 0]
        rho[feat] = {"spearman_K_over_n_vs_min_margin": spearman([i["K_over_n"] for i in g], [i["margin_min"] for i in g]), "n": len(g)}
        t = [i for i in g if not i["cls"].startswith("built")]
        rho[feat + "_no_built"] = {"spearman_K_over_n_vs_min_margin": spearman([i["K_over_n"] for i in t], [i["margin_min"] for i in t]), "n": len(t)}
    med = lambda key, c: classes[c][key].get("median")
    r4 = {f"neurons toy_trained - realistic: {k}": med(k, "neurons/toy_trained") - med(k, "neurons/realistic") for k in ("top_kept_median", "coverage", "rem_share")}
    r4 |= {f"vpd toy_trained - {c}: top_kept_median": med("top_kept_median", "vpd/toy_trained") - med("top_kept_median", f"vpd/{c}") for c in ("procedural", "rl")}
    dev = [abs(i["E_b_K"] - i["E_b_K_saved"]) / max(abs(i["E_b_K_saved"]), 1e-12) for i in items if i["E_b_K_saved"] is not None]
    summary = {"experiment": "E285 (why_fail)", "cell": "P1.59", "n_items": len(items), "n_pooled": len(pooled), "checks": check_rows,
               "E_b_K_recomputed_max_rel_dev": max(dev), "E_b_K_recomputed_n": len(dev), "classes": classes, "spearman": rho, "R4": r4,
               "beside": [i for i in items if i["experiment"] in ("E281", "E283")], "items": items}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    cols = [k for k in items[0] if k != "layer_share"] + [f"layer{L}_share" for L in range(LAYERS)]
    lines = ["\t".join(cols)]
    for i in items:
        ls = i["layer_share"] or [None] * LAYERS
        v = {**i, **{f"layer{L}_share": ls[L] for L in range(LAYERS)}}
        lines.append("\t".join("" if v[c] is None else (f"{v[c]:.6g}" if isinstance(v[c], float) else str(v[c])) for c in cols))
    (RESULTS / "items.tsv").write_text("\n".join(lines) + "\n")
    for c, s in classes.items():
        print(c, s["items"], {k: (round(s[k]["median"], 4) if s[k].get("median") is not None else None) for k in KEYS})
    print(json.dumps({"spearman": rho, "R4": r4, "checks": check_rows, "E_b_K_dev": max(dev)}, indent=1))


def csv_rows(p: Path) -> list[dict[str, str]]:
    head, *body = p.read_text().splitlines()
    names = head.split("\t")
    return [dict(zip(names, line.split("\t"), strict=True)) for line in body]


if __name__ == "__main__":
    {"edge": edge, "all": main}[sys.argv[1]]()
