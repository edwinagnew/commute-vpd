"""E208 figures and the per-part split of the main reading (heads against neurons in K).

    ~/commute-agents/fleet/cpu-run python experiments/rho_saved_items/plot.py

Reads results/items/*.json, results/stage0_p0.json, results/summary.json and results/large/parts/*.npz; writes
figures/{controls,gaps,one_change}.png and results/parts_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from commute import figures as fg

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"
PAIRS = [
    ("copy24", "copy / cipher"),
    ("ident32", "identity / random map"),
    ("repeat", "repeat / repeat twin"),
    ("case", "case / case twin"),
]  # summary.json's R1 gap keys
# E196's twin-over-reuse MLP update ratios, seeds 0 and 1 (reports/analyses/2026-11-18_analysis_rule-retention-pairs-run2.md)
RATIO = {
    "copy / cipher": "1.05-1.09",
    "identity / random map": "1.13",
    "repeat / repeat twin": "1.42-1.56",
    "case / case twin": "2.52-2.67",
}
READS = [
    ("main", "neuron", "the note as written"),
    ("i", "steering", "today's aggregate"),
    ("b", "size", "size-aware distance"),
]


def item(name: str) -> dict:
    return json.loads((RES / "items" / f"{name}.json").read_text())


def orders(block: dict, read: str) -> list[float]:
    if f"rho_{read}" in block:
        return list(block[f"rho_{read}"])
    return [o[f"rho_{read}"] for o in block["orders"]]


def controls() -> None:
    s0 = json.loads((RES / "stage0_p0.json").read_text())
    rows = {"circuit turned up two-fold": s0["x2"]["replace"], "circuit turned up ten-fold": s0["x10"]["replace"],
            "rebuilt circuit": item("rebuilt")["replace"], "graft in M's own slots": item("slotted")["replace"],
            "appended graft neurons": item("neurons_appended")["replace"], "appended graft head": item("head_appended")["replace"]}  # fmt: skip
    fig, ax = fg.figure(fg.FULL_WIDTH)
    data = []
    for k, (read, style, label) in enumerate(READS):
        groups = {r: orders(b, read) for r, b in rows.items()}
        fg.dots(
            ax,
            groups,
            method=style,
            xlabel="retention of the parts F needs (1 = every rule kept)",
            offset=(k - 1) * 0.22,
        )
        ax.scatter(
            [],
            [],
            color=fg.METHODS[style].color,
            marker=fg.METHODS[style].marker,
            label=label,
        )
        data += [
            {"item": r, "reading": read, "order": i, "rho": v}
            for r, vs in groups.items()
            for i, v in enumerate(vs)
        ]
    fg.reference(ax, 1.0, "same rule")
    ax.set_xlim(-0.05, 1.08)
    ax.legend(loc="center", fontsize=7, frameon=False)
    ax.set_title("Do the controls read where they should? Old near 1, new near 0")
    fg.save(
        fig,
        FIG / "controls.png",
        n="5 search orders per item and reading, each a dot; mechanism replacement",
        data=data,
    )


def gaps() -> None:
    summ = json.loads((RES / "summary.json").read_text())["R1"]
    sd = max(summ[s]["largest_order_sd"] for s in ("s0", "s1"))
    fig, ax = fg.figure(fg.FULL_WIDTH)
    groups, data = {}, []
    for key, label in PAIRS:
        vals = [summ[s]["gaps"][key] for s in ("s0", "s1")]
        groups[f"{label}\n(twin moved {RATIO[label]}x as far)"] = vals
        data += [
            {"pair": label, "seed": i, "gap": v, "twin_update_ratio": RATIO[label]}
            for i, v in enumerate(vals)
        ]
    fg.dots(
        ax,
        groups,
        method="neuron",
        xlabel="reuse member's retention minus twin's, the note as written",
    )
    fg.reference(ax, sd, "largest spread over search orders")
    ax.set_title("Do the gaps follow the rule change or the update size? Update size")
    fg.save(
        fig,
        FIG / "gaps.png",
        n="2 seeds per pair, each the mean over 5 search orders; mechanism replacement",
        data=data,
    )


def one_change() -> None:
    r3 = json.loads((RES / "summary.json").read_text())["R3"]
    names = [("copy24", "copy"), ("cipher24", "cipher"), ("ident32", "identity map"), ("rand32", "random map"),
             ("repeat_real", "repeat"), ("repeat_twin", "repeat twin"), ("case_real", "case"), ("case_twin", "case twin"),
             ("twin", "parity twin"), ("fact", "fact")]  # fmt: skip
    cols = ["aggregate", "activation patching", "M-only settings", "size-aware"]
    vals, rows, data = [], [], []
    for key, label in names:
        rec = [r3[f"{key}_s{s}"] for s in (0, 1)]
        v = [float(np.mean([x["i_minus_main"] for x in rec])), float(np.mean([x["rho_main_patch"] - x["rho_main_replace"] for x in rec])),
             float(np.mean([x["iii_minus_main"] for x in rec])), float(np.mean([x["b_minus_main"] for x in rec]))]  # fmt: skip
        vals.append(v)
        rows.append(label)
    rb = r3["rebuilt"]
    vals.append(
        [
            rb["i_minus_main"],
            rb["rho_main_patch"] - rb["rho_main_replace"],
            rb["iii_minus_main"],
            rb["b_minus_main"],
        ]
    )
    rows.append("rebuilt circuit")
    for label, v in zip(rows, vals, strict=True):
        data.append({"item": label} | dict(zip(cols, v, strict=True)))
    fig, ax = fg.figure((7.5, 5.5))
    fg.gap_heatmap(
        ax,
        vals,
        rows,
        cols,
        label="retention under the change minus the note's",
        vmax=0.03,
        fmt=".4f",
    )
    ax.set_title(
        "Which single change moves the reading? The distance does; patching and settings barely"
    )
    fg.save(
        fig,
        FIG / "one_change.png",
        n="each cell the mean over 2 seeds and 5 search orders (the rebuilt circuit: 1 item)",
        data=data,
    )


def parts() -> None:
    """Per item: mean r under the note over K's heads and over K's neurons, and the heads' share of K (mechanism replacement)."""
    out = {}
    for f in sorted((RES / "large" / "parts").glob("*.npz")):
        z = np.load(f)
        names, r = z["parts"], z["r_main"]
        head = np.array([".h" in str(n) for n in names])
        neuron = np.array([".n" in str(n) for n in names])
        hs, ns, sh = [], [], []
        for k in range(5):
            key = f"K_replace_{k}"
            if key not in z:
                continue
            K = z[key]
            hk, nk = K[head[K]], K[neuron[K]]
            hs.append(float(r[hk].mean()) if len(hk) else float("nan"))
            ns.append(float(r[nk].mean()) if len(nk) else float("nan"))
            sh.append(len(hk) / len(K))
        out[f.stem] = {"r_heads_in_K": float(np.nanmean(hs)) if hs else None, "r_neurons_in_K": float(np.nanmean(ns)) if ns else None,
                       "head_share_of_K": float(np.mean(sh)) if sh else None,
                       "min_r_head_in_K": float(min(r[K][head[K]].min() for K in (z[f"K_replace_{k}"] for k in range(5)) if head[K].any()))
                       if any(head[z[f"K_replace_{k}"]].any() for k in range(5)) else None}  # fmt: skip
    (RES / "parts_summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    controls()
    gaps()
    one_change()
    parts()
