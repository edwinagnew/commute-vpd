"""E286 (cell P1.60) figure and page numbers, CPU only, from the saved item files (no model).

Writes results/page.json: per basis and item class, every item's pooled shares (strict, own, needed, rotation and its
two nulls, and the own space's exact null), and at the neuron-and-head basis the same rotation read on heads and on
neurons apart (from results/large/items_nn__*.npz). Draws figures/rotation_vs_null.png: the rotation share of each item
beside its two nulls, at VPD and at the neuron-and-head basis.

    python experiments/rotation_kept/plot.py
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.lines import Line2D

from commute import figures as FG

HERE = Path(__file__).parent
RES = HERE / "results"

# (row label, regex on the item name); the order is the figure's
CLASSES: list[tuple[str, str]] = [
    ("control: turned toward a needed neuron", r"^rot_needed$"),
    ("control: turned toward a non-needed neuron", r"^rot_other$"),
    ("built: scaled x2, x10", r"^x(2|10)$"),
    ("built: rebuilt", r"^rebuilt$"),
    ("built: new_few, new_thin", r"^new_(few|thin)$"),
    ("built: two randoms", r"^random_(size|kl)_"),
    ("E62 copy, 3 seeds", r"^e62__copy_"),
    ("E62 lookup, 3 seeds", r"^e62__lookup_"),
    ("E62 lookup at copy's size", r"^e62-lookup_"),
    (
        "E229 fine-tunes, 20",
        r"^(?!.*_rand$)(case_real|case_twin|cipher24|copy24|fact|ident32|rand32|repeat_real|repeat_twin|twin)_s\d$",
    ),
    ("E229 matched randoms, 20", r"_rand$"),
    ("E272 panel fine-tunes", r"^panel__(?!.*__random_)"),
    ("E272 panel Gaussians, 12", r"^panel__.*__random_"),
]


def per_split(npz: Path) -> dict[str, Any]:
    """Rotation share and both nulls on heads and on neurons apart (non-remainder members of N), as rotation_kept.pooled."""
    z = np.load(npz)
    out: dict[str, Any] = {}
    for name, sel in (
        ("heads", z["gram"] & ~z["rem"]),
        ("neurons", ~z["gram"] & ~z["rem"]),
    ):
        dd, own, need, mat = (z[k][sel] for k in ("dd", "own", "needed", "matched"))
        left = float((dd - own).sum())
        nu = (
            (dd - own)
            * (z["dim_S"][sel] - z["dim_O"][sel])
            / np.maximum(z["d_out"][sel] - z["dim_O"][sel], 1)
        )
        ok = left > 1e-6 * float(dd.sum()) and dd.sum() > 0
        out[name] = {"n": int(sel.sum()), "dd_share": float(dd.sum() / z["dd"][~z["rem"]].sum()),
                     "own": float(own.sum() / dd.sum()) if dd.sum() > 0 else None,
                     "rotation": float((need - own).sum() / left) if ok else None,
                     "rotation_matched_null": float((mat - own).sum() / left) if ok else None,
                     "rotation_nu_null": float(nu.sum() / left) if ok else None,
                     "n_no_matched_draw": int((z["n_matched"][sel] == 0).sum())}  # fmt: skip
    return out


def own_null(npz: Path) -> float:
    """The own space's exact null: a random direction's expected share, dim O / d_out, weighted by dd (non-remainder)."""
    z = np.load(npz)
    sel = ~z["rem"]
    return float(
        (z["dd"][sel] * z["dim_O"][sel] / z["d_out"][sel]).sum() / z["dd"][sel].sum()
    )


def records() -> list[dict[str, Any]]:
    rows = []
    s0 = json.loads((RES / "stage0_nn_built_c1.json").read_text())
    for key in ("vpd/x2", "nn/x2", "rot_needed", "rot_other"):
        basis, item = key.split("/") if "/" in key else ("nn", key)
        rows.append({"basis": basis, "item": item, **s0[key]})
    for f in sorted((RES / "items").glob("*.json")):
        basis, _, item = f.stem.partition("__")
        if basis in ("nn", "vpd"):
            rows.append(
                {"basis": basis, "item": item, **json.loads(f.read_text())}
                | {"item": item}
            )
    out = []
    for r in rows:
        cls = next((c for c, rx in CLASSES if re.search(rx, r["item"])), None)
        assert cls is not None, r["item"]
        p = r["pooled"]
        rec = {"basis": r["basis"], "item": r["item"], "class": cls, "size_K": r["size_K"], "size_N": r["size_N"], "too_small": r["too_small"],
               **{k: p[k] for k in ("strict", "own", "needed", "matched", "nu", "rotation", "rotation_matched_null", "rotation_nu_null")}}  # fmt: skip
        if p["rotation"] is not None:
            rec["above_nu_null"] = p["rotation"] - p["rotation_nu_null"]
            rec["above_matched_null"] = p["rotation"] - p["rotation_matched_null"]
        npz = RES / "large" / f"items_{r['basis']}__{r['item']}.npz"
        if npz.exists() and p["dd"] > 0:
            rec["own_nu"] = own_null(npz)
            if r["basis"] == "nn":
                rec["split"] = per_split(npz)
        out.append(rec)
    return out


def summarise(recs: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, Any] = {}
    for basis in ("vpd", "nn"):
        for cls, _ in CLASSES:
            rs = [
                r
                for r in recs
                if r["basis"] == basis and r["class"] == cls and not r["too_small"]
            ]
            if not rs:
                continue
            agg: dict[str, Any] = {
                "n": len(rs),
                "items": {r["item"]: r for r in rs},
                "too_small": [
                    r["item"]
                    for r in recs
                    if r["basis"] == basis and r["class"] == cls and r["too_small"]
                ],
            }
            for k in (
                "strict",
                "own",
                "needed",
                "rotation",
                "rotation_matched_null",
                "rotation_nu_null",
                "nu",
                "own_nu",
                "above_nu_null",
                "above_matched_null",
            ):
                v = [r[k] for r in rs if r.get(k) is not None]
                if v:
                    agg[k] = {
                        "min": min(v),
                        "max": max(v),
                        "median": statistics.median(v),
                        "mean": statistics.fmean(v),
                    }
            for part in ("heads", "neurons"):
                for k in (
                    "rotation",
                    "rotation_nu_null",
                    "rotation_matched_null",
                    "dd_share",
                    "own",
                ):
                    v = [
                        r["split"][part][k]
                        for r in rs
                        if "split" in r and r["split"][part][k] is not None
                    ]
                    if v:
                        agg.setdefault(part, {})[k] = {
                            "min": min(v),
                            "max": max(v),
                            "median": statistics.median(v),
                        }
            by[f"{basis}/{cls}"] = agg
    return by


def head24() -> dict[str, Any]:
    rs = [
        json.loads(f.read_text())
        for f in sorted((RES / "items").glob("head24__*.json"))
    ]
    out: dict[str, Any] = {r["item"]: r for r in rs}
    for arm in ("copy", "lookup"):
        a = [r for r in rs if f"/{arm}_" in r["item"]]
        out[arm] = {k: {"min": min(r[k] for r in a), "max": max(r[k] for r in a), "mean": statistics.fmean(r[k] for r in a)}
                    for k in ("strict", "own", "null", "flips")}  # fmt: skip
    return out


def gauge() -> dict[str, Any]:
    rs = [
        json.loads(f.read_text()) for f in sorted((RES / "items").glob("gauge__*.json"))
    ]
    out: dict[str, Any] = {}
    for cls in ("e62", "e229", "panel"):
        a = [r for r in rs if r["class"] == cls]
        out[cls] = {side: {"min": min(r[side]["share"] for r in a), "max": max(r[side]["share"] for r in a),
                           "median": statistics.median(r[side]["share"] for r in a)} for side in ("update", "random")}  # fmt: skip
        out[cls]["n"] = len(a)
    sub = json.loads((RES / "gauge_sub_summary.json").read_text())["classes"]
    out["sub_max_change"] = max(
        c[k]["mean_abs_change"] for c in sub.values() for k in ("strict", "needed")
    )
    out["sub_max_sd_ratio"] = max(
        c[k]["mean_abs_change"] / c[k]["seed_sd"]
        for c in sub.values()
        for k in ("strict", "needed")
    )
    subs = [
        json.loads(f.read_text())
        for f in sorted((RES / "items").glob("gauge_sub__*.json"))
    ]
    out["sub_n"] = len(subs)
    out["sub_max_reading"] = {
        k: max(s["update"]["pooled"][k] for s in subs) for k in ("strict", "needed")
    }
    out["sub_all_below_sd"] = all(
        c[k]["moves_less_than_sd"] for c in sub.values() for k in ("strict", "needed")
    )
    return out


def draw(recs: list[dict[str, Any]]) -> None:
    fig, axes = FG.figure((7.5, 5.4), ncols=2)
    labels = [c for c, _ in CLASSES]
    plotted = []
    for ax, basis, title in (
        (axes[0], "vpd", "VPD subcomponents"),
        (axes[1], "nn", "neurons and heads"),
    ):
        groups = {
            c: [
                r
                for r in recs
                if r["basis"] == basis and r["class"] == c and not r["too_small"]
            ]
            for c in labels
        }
        # The matched null is undefined at heads (no head outside the needed set is left to draw), and heads carry
        # most of the change at the neuron-and-head basis, so it is drawn at VPD only (redteam-lead, 2026-11-23).
        series = [("rotation", "regating", -0.22), ("rotation_nu_null", "random", 0.0)]
        if basis == "vpd":
            series.append(("rotation_matched_null", "spectrum", 0.22))
        for key, method, off in series:
            FG.dots(
                ax,
                {
                    c: [r[key] for r in rs if r[key] is not None]
                    for c, rs in groups.items()
                },
                method=method,
                xlabel="rotation share",
                offset=off,
            )
        for i, (c, rs) in enumerate(groups.items()):
            if not rs:
                ax.text(
                    0.5,
                    i,
                    "not run",
                    fontsize=7,
                    color="#555555",
                    va="center",
                    ha="center",
                )
            elif all(r["rotation"] is None for r in rs):
                ax.text(
                    0.5,
                    i,
                    "nothing leaves the own space",
                    fontsize=7,
                    color="#555555",
                    va="center",
                    ha="center",
                )
            plotted += [{"basis": basis, "class": c, "item": r["item"], "rotation": r["rotation"], "nu_null": r["rotation_nu_null"],
                         "matched_null": r["rotation_matched_null"]} for r in rs]  # fmt: skip
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [Line2D([], [], color=FG.METHODS[m].color, marker=FG.METHODS[m].marker, linestyle="", markersize=5, label=t)
               for m, t in (("regating", "measured"), ("random", "null: random direction"),
                            ("spectrum", "null: matched spaces, VPD only"))]  # fmt: skip
    fig.suptitle("Rotation toward M's needed writes stands out from its null at VPD only for built rotations;\n"
                 "at neurons and heads the null is near 0.7, so the reading has little room to separate anything", fontsize=9)  # fmt: skip
    axes[1].text(0.50, 0.66, "matched null undefined here:\nno head outside the needed set", transform=axes[1].transAxes,
                 fontsize=7, color="#555555", ha="center", va="center")  # fmt: skip
    axes[1].legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.38, 0.45),
        fontsize=7,
        frameon=False,
        title="rotation share",
        title_fontsize=7,
    )
    FG.save(fig, HERE / "figures" / "rotation_vs_null.png",
            n="one point per item: built edits one state each; E62 3 seeds per arm; E229 20 fine-tunes and 20 matched randoms "
              "(neurons only); E272 panel 12 tasks at seed 0 and 12 Gaussians at VPD, 3 seeds and 12 Gaussians at neurons; "
              "the two rotation controls at neurons only; up to 256 windows of each item's data; "
              "3 panel items with fewer than 100 needed variables left out as too small to read",
            data=plotted)  # fmt: skip


def main() -> None:
    recs = records()
    page = {"experiment": "E286 (rotation_kept)", "source": "experiments/rotation_kept/plot.py from results/items, results/large, stage0_nn_built_c1.json",
            "classes": summarise(recs), "head24": head24(), "gauge": gauge()}  # fmt: skip
    t, r = (
        page["classes"][f"nn/E229 {c}"]["items"]
        for c in ("fine-tunes, 20", "matched randoms, 20")
    )
    gap = [t[k]["rotation"] - r[f"{k}_rand"]["rotation"] for k in t]
    page["e229_pairs"] = {"rotation_minus_matched_random": {"min": min(gap), "max": max(gap), "median": statistics.median(gap)},
                          "n_positive": sum(g > 0 for g in gap), "n_at_least_0.2": sum(g >= 0.2 for g in gap), "n": len(gap)}  # fmt: skip
    page["e62_needed_gap"] = {b: page["classes"][f"{b}/E62 copy, 3 seeds"]["needed"]["mean"] - page["classes"][f"{b}/E62 lookup, 3 seeds"]["needed"]["mean"]
                              for b in ("vpd", "nn")}  # fmt: skip
    (RES / "page.json").write_text(json.dumps(page, indent=1))
    draw(recs)


if __name__ == "__main__":
    main()
