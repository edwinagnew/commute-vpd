"""E277 (cell P1.56) figures from results/summary.json, one per claim of the P1.56 page.

- figures/eb_built.png: E_b over K (the needed set) per built edit at the top-neuron featurization, beside E268's at VPD.
- figures/eb_finetunes.png: E229's 20 fine-tunes and E62's copy and lookup fine-tunes, coloured by the side they were
  built for (reuse or new), each trained one beside its matched random update; log scale.
- figures/eb_panel.png: the realistic panel's task means against the consensus tier, at neurons (this run) and at VPD
  (E272's files), each task's seeds and its own Gaussian update on the same axes.
An item with |K| < 100 ("too small to read") is drawn hollow. Each figure also writes .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/needed_neurons/plot.py [SUMMARY] [OUT_DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
E268 = HERE.parent / "kstar_needed" / "results" / "summary.json"
E272 = HERE.parent / "convergence_panel" / "results" / "wide" / "eb"
BUILT = ["x2", "x10", "rebuilt", "new_few", "new_thin", "random_size_241001", "random_kl_241002"]
SHORT = {"random_size_241001": "random (size)", "random_kl_241002": "random (KL)"}
PAIRS = [("copy24", "cipher24"), ("ident32", "rand32"), ("case_real", "case_twin"), ("repeat_real", "repeat_twin"), ("twin", "fact")]
TIER = {"arxiv": 3.0, "lower": 3.0, "category": 2.5, "conll": 2.0, "french": 2.0, "german": 2.0, "sentiment": 2.0,
        "spanish": 2.0, "squad": 1.5, "summary": 1.5, "swahili": 1.0, "yoruba": 1.0}  # fmt: skip  # E272 consensus (score.raters)
REUSE, NEW, RAND = FG.Style("built or rated reuse side", "#0072B2", "o"), FG.Style("built or rated new side", "#E69F00", "D"), FG.METHODS["random"]
TO_READ = "write share over the needed set K"


def point(ax: Any, x: float, y: float, style: FG.Style, small: bool, label: str | None = None, s: float = 26) -> None:
    ax.scatter([x], [y], marker=style.marker, s=s, linewidths=1.0, label=label,
               facecolors="none" if small else style.color, edgecolors=style.color)  # fmt: skip


def built(items: dict[str, Any], vpd: dict[str, Any], out: Path) -> None:
    fig, ax = FG.figure(FG.ONE_COLUMN)
    rows, nb, vp = [], FG.METHODS["neuron"], FG.METHODS["spectrum"]
    for x, item in enumerate(BUILT):
        v, r = vpd[item]["E_b_K"], items[item]
        ax.scatter([x], [v], color=vp.color, marker=vp.marker, s=26, label="VPD subcomponents (E268)" if x == 0 else None)
        point(ax, x, r["E_b_K"], nb, r["too_small"], "top neurons above p97 (this run)" if x == 0 else None)
        rows.append({"item": item, "E_b_K": r["E_b_K"], "size_K": r["size_K"], "margin_median": r["margin_median"],
                     "rem_dd_share": r["rem_dd_share_K_changed"], "E_b_K_vpd": v})  # fmt: skip
    ax.set_xticks(range(len(BUILT)), [SHORT.get(i, i) for i in BUILT], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel(TO_READ)
    ax.set_title("The strengthening edits read near 1 at neurons; the rebuilt one does not", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="center right")
    FG.save(fig, out / "eb_built.png", data=rows, n="one built state per column; K from 256 pairs of inputs (4 derangements of 64 windows)")


def finetunes(items: dict[str, Any], out: Path) -> None:
    fig, ax = FG.figure(FG.FULL_WIDTH)
    rows, x, ticks = [], 0, []
    groups = [(a, b, [f"{a}_s0", f"{a}_s1"], [f"{b}_s0", f"{b}_s1"]) for a, b in PAIRS]
    groups.append(("E62 copy", "E62 lookup", [f"e62/copy_s{i}" for i in range(3)], [f"e62/lookup_s{i}" for i in range(3)]))
    for a, b, sa, sb in groups:
        for name, seeds, style in ((a, sa, REUSE), (b, sb, NEW)):
            for j, s in enumerate(seeds):
                r, q = items[s], items.get(f"{s}_rand")
                dx = (j - (len(seeds) - 1) / 2) * 0.18
                point(ax, x + dx, r["E_b_K"], style, r["too_small"], style.label if x < 2 and j == 0 else None)
                if q is not None:
                    point(ax, x + dx, q["E_b_K"], RAND, q["too_small"], "its random update, matched per matrix" if x == 0 and j == 0 else None)
                rows.append({"item": s, "side": style.label, "E_b_K": r["E_b_K"], "size_K": r["size_K"],
                             "E_b_K_rand": q["E_b_K"] if q else None, "rem_dd_share": r["rem_dd_share_K_changed"]})  # fmt: skip
            ticks.append(name)
            x += 1
    ax.set_yscale("log")
    ax.set_xticks(range(len(ticks)), ticks, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(TO_READ + " (log)")
    ax.set_title("Each fine-tune reads above its own random update, but reuse sides do not read above new sides", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    FG.save(fig, out / "eb_finetunes.png", data=rows,
            n="one state per point, 2 seeds per E229 task, 3 per E62 task; E62 has no matched random in this run; every |K| here is 2,094 or more")  # fmt: skip


def vpd_panel() -> dict[str, dict[str, Any]]:
    """task -> {'seeds': [E_b over K], 'random': E_b over K} at VPD, from E272's wide eb files."""
    out: dict[str, dict[str, Any]] = {}
    for t in TIER:
        st = json.loads((E272 / f"{t}.json").read_text())["states"]
        out[t] = {"seeds": [(v["E_b_K"], v["size_K"] < 100) for k, v in st.items() if not k.startswith("random/")],
                  "random": next(v["E_b_K"] for k, v in st.items() if k.startswith("random/"))}  # fmt: skip
    return out


def neuron_panel(items: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in TIER:
        ps = [r for k, r in items.items() if k.startswith(f"panel/{t}/")]
        out[t] = {"seeds": [(r["E_b_K"], r["too_small"]) for r in ps if "/random_" not in r["item"]],
                  "random": next(r["E_b_K"] for r in ps if "/random_" in r["item"])}  # fmt: skip
    return out


def panel(items: dict[str, Any], add: dict[str, Any], out: Path) -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    rows = []
    sides = (("neurons", "top neurons above p97 (this run)", neuron_panel(items)), ("vpd", "VPD subcomponents (E272)", vpd_panel()))
    order = sorted(TIER, key=lambda t: (TIER[t], t))
    for ax, (key, name, data) in zip(axes, sides, strict=True):
        for i, t in enumerate(order):
            d, style = data[t], REUSE if TIER[t] >= 2.5 else NEW if TIER[t] <= 1.5 else FG.Style("rated middle", "#009E73", "s")
            for j, (v, small) in enumerate(d["seeds"]):
                point(ax, i + (j - 1) * 0.15, v, style, small, s=14)
            readable = not any(small for _, small in d["seeds"])
            if readable:
                ax.hlines(fmean(v for v, _ in d["seeds"]), i - 0.3, i + 0.3, color=style.color, lw=1.5)
            point(ax, i, d["random"], RAND, False, "its own Gaussian update" if i == 0 else None, s=18)
            rows.append({"side": key, "task": t, "tier": TIER[t], "seeds": ";".join(f"{v:.5f}" for v, _ in d["seeds"]),
                         "mean": fmean(v for v, _ in d["seeds"]) if readable else None, "gaussian": d["random"]})  # fmt: skip
        ax.set_yscale("log")
        ax.set_xticks(range(len(order)), [f"{t} ({TIER[t]:g})" for t in order], rotation=50, ha="right", fontsize=6)
        ax.set_xlabel("task (consensus tier: 1 new, 3 reuse)", fontsize=7)
        a = add[key]
        ax.set_title(f"{name}: tau with tier {a['tau']:.2f}, two-sided p {a['p_two_sided']:.3f}", fontsize=7)
    axes[0].set_ylabel(TO_READ + " (log)")
    axes[0].sharey(axes[1])
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    FG.save(fig, out / "eb_panel.png", data=rows,
            n=f"three fine-tuned seeds per task (bar: their mean, drawn only when every seed has |K| >= 100; hollow: |K| < 100); tau and p on the {len(add['tasks'])} tasks both sides read")  # fmt: skip


def main(summary: Path, out: Path) -> None:
    items = json.loads(summary.read_text())["items"]
    vpd = json.loads(E268.read_text())["items"]
    add = json.loads((summary.parent / "panel_addendum.json").read_text())
    built(items, vpd, out)
    finetunes(items, out)
    panel(items, add, out)


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json",
        Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures",
    )
