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


if __name__ == "__main__":
    {"edge": edge}[sys.argv[1]]()
