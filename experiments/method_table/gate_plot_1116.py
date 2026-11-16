"""P2.26, the part 2 gate figure as of 2026-11-16: every 67M pair by every method, then the Qwen readings beside it.

Saved results only, no model. Three panels, one colour rule: a cell is the signed gap, side A minus side B (on the 67M
the side built to reuse minus the side built to be new), on one diverging scale capped at 1, and a box means the cell's
two sides are apart by its rule. No verdict of reused or new, and no combined score.

- 67M: `results/final.json` (P2.11 to P2.25, one rule: every seed of one side beyond every seed of the other), E155's
  top-neuron overlap and part 1's re-weighting share gap, both put through that same seed-range rule here.
- Qwen3-1.7B, a different model, decompositions and pair set: the pair library (P3.7 cheap readings; P3.8 share on the
  new decomposition p-9707d6ff; P3.9 share on the frozen-neuron decomposition p-015e0710, MLP matrices only), and P3.6's
  two arm contrasts, which are not reuse/new pairs. A floored gap is shown divided by its random band (library) or by the
  wider seed range (arm contrasts), so 1 is the edge of the null; update size is log10 of side A over side B.
- E181 (P3.10): switched-off neuron sets on p-015e0710, share minus the random curve at the same size, over the band.
  Read from qwen-lead's jq mirror of `run.py --summary`, which has not run.

Writes `figures/gate_scorecard_1116.png` (+ .pdf, .csv) and `results/gate_1116.json`.

    ~/commute-agents/fleet/cpu-run experiments/method_table/gate_plot_1116.py
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from commute.figures import figure, gap_heatmap, save

ROOT = Path(__file__).resolve().parents[2]
FINAL, OVERLAP, PART1 = "experiments/method_table/results/final.json", "experiments/neuron_overlap/results/stage2.json", "experiments/part1_vs_methods/results/compare.json"
LIB, SHARE = "experiments/qwen_library_readings/results", "experiments/qwen_library_share/results"
CHEAP = "experiments/qwen_cheap_readings/results/run2_stage3.json"
E181 = Path.home() / "commute-data/scratch/qwen-lead/e181-analysis/jq_summary.json"
COLS = {"share": "re-gating\nshare", "untrained": "share on\nuntrained\nSVD", "neuron": "neuron,\nwhole\nwidth", "neuron_half": "neuron,\nhalf",
        "neuron_top": "neuron,\ntop sets", "overlap": "top-240\noverlap", "on_both": "change on\nneurons\nin both", "steering": "steering",
        "prior": "prior\n(log10)", "size": "update\nsize\n(log10)", "erank": "effective\nrank\n(log10)", "k_half": "neurons for\nhalf the gain\n(log10)",
        "part1": "part 1\nre-weighting\nshare"}  # fmt: skip
QCOLS = {"share_new": "share,\np-9707d6ff\n(÷ band)", "share_frozen": "share,\np-015e0710\nMLP only\n(÷ band)", "size": "update\nsize\n(log10)",
         "erank": "effective\nrank\n(÷ null)", "overlap": "top-neuron\noverlap\n(÷ null)", "steer": "steering\n(÷ null)", "on_both": "change on\nboth sets\n(÷ null)"}  # fmt: skip
LIB_PAIRS = {"copy_cipher": ("copy", "cipher", "copy − cipher"), "facts": ("known", "invented", "known − invented facts"),
             "gold": ("gold", "gold_shuf", "gold: real − shuffled"), "template": ("template", "template_shuf", "template: real − shuffled")}  # fmt: skip
SAVES = {"last": ("late", "late save"), "5x": ("matched", "matched save")}  # P3.7's '5x' is the save P3.8/P3.9 call 'matched'
ARMS = {"RL vs shuffled-reward null": "arm contrast: RL − shuffled-reward RL", "known vs never-solved pool": "arm contrast: passed − never-solved pool"}
CHOICES = {"top": "most used", "random": "random\namong used", "specific": "used most\non this text"}
ROWSETS = ("copy", "cipher", "known", "invented", "gold")
FOOTER = (
    "Top, 67M: reuse side minus new side; shares and log10 prior a difference of seed means, size and rank log10 of reuse/new, neurons for half the gain "
    "log10 of new/reuse, so red = fewer on the reuse side. Box = every seed of one side beyond every seed of the other. Seeds 1 to 8 a side; part 1 two.\n"
    "Middle, Qwen3-1.7B, a different model and pair set, side A minus side B, 3 seeds a side (arm contrasts 2): ÷ null = mean gap over the pair's random "
    "band, or for the two arm contrasts over the wider seed range; box = beyond it by the reading's filed rule. Size boxed by the 67M seed rule. "
    "Steering has no band on three pairs. Prior not drawn: its two forms disagree in sign on copy − cipher.\n"
    "Bottom: P3.10's switched-off sets, share minus the random curve at the same size, over the random band; × = off its size target or unscored. "
    "From qwen-lead's jq mirror; run.py --summary has not run. Colour capped at 1 everywhere. × = not run or not readable."
)
Cell = dict[str, Any]


def load(path: str | Path) -> Any:
    return json.loads((ROOT / path).read_text())


def apart(a: list[float], b: list[float]) -> bool:
    """The 67M's one rule: every seed of one side beyond every seed of the other (two or more a side)."""
    return min(len(a), len(b)) >= 2 and (min(a) > max(b) or max(a) < min(b))


def cells_67m() -> tuple[dict[tuple[str, str], Cell], list[str], dict[str, str]]:
    final = load(FINAL)["cells"]
    out = {(c["pair"], c["method"]): {"gap": c["gap"], "apart": c["call"] == "apart", "source": f"{FINAL} {c['source']}"} for c in final}
    for pair, s in load(OVERLAP)["score"].items():
        for col in ("overlap", "on_both"):
            v = s[col]["values"]
            out[(pair, col)] = {"gap": s[col]["gap"], "apart": apart(v["reuse"], v["new"]), "source": f"{OVERLAP} .score.{pair}.{col}"}
    part1 = load(PART1)["overlapping"]
    for key, p in part1.items():
        out[(p["final_pair"], "part1")] = {"gap": p["gap"], "apart": apart(p["G_reuse"], p["G_new"]), "source": f"{PART1} .part1.{key}"}
    rows = list(dict.fromkeys(c["pair"] for c in final))
    return out, rows, {c["pair"]: c["pair_name"] for c in final}


def side_sizes(pair: str, arm: str, save_key: str) -> list[float]:
    states = load(f"{LIB}/{pair}/stage2.json")["states"]
    return [states[f"{arm}_s{s}_{save_key}"]["size"] for s in range(3)]


def library_row(pair: str, save_key: str) -> dict[str, Cell]:
    a, b, _ = LIB_PAIRS[pair]
    sc = load(f"{LIB}/scorecard.json")["pairs"][pair]["saves"][save_key]
    band = load(f"{LIB}/{pair}/stage1.json")["band"]
    row: dict[str, Cell] = {}
    for col in ("erank", "overlap", "steer"):
        if band[col] is not None:
            row[col] = {"gap": statistics.fmean(sc["gaps"][col]) / band[col], "apart": sc["cells"][col] == "beyond floor", "source": f"{LIB}/scorecard.json .pairs.{pair}.saves.{save_key}"}
    sa, sb = side_sizes(pair, a, save_key), side_sizes(pair, b, save_key)
    row["size"] = {"gap": math.log10(statistics.fmean(sa) / statistics.fmean(sb)), "apart": apart(sa, sb), "source": f"{LIB}/{pair}/stage2.json .states[<state>].size"}
    for col, folder in (("share_new", SHARE), ("share_frozen", f"{SHARE}/p-015e0710")):
        s = load(f"{folder}/summary.json").get(f"{pair}/{SAVES[save_key][0]}")
        if s is not None:
            row[col] = {"gap": statistics.fmean(s["gaps"]) / s["band"], "apart": s["cell"] == "beyond band", "source": f"{folder}/summary.json"}
    return row


def arm_row(name: str) -> dict[str, Cell]:
    c = load(CHEAP)["contrasts"][name]
    row = {col: {"gap": c[col]["ratio"], "apart": c[col]["verdict"] == "counts", "source": f"{CHEAP} .contrasts[{name!r}].{col}"}
           for col in ("erank", "overlap", "on_both", "steer") if c[col]["ratio"] is not None}  # fmt: skip
    row["size"] = {"gap": math.log10(statistics.fmean(c["size"]["a"]) / statistics.fmean(c["size"]["b"])), "apart": apart(c["size"]["a"], c["size"]["b"]),
                   "source": f"{CHEAP} .contrasts[{name!r}].size"}  # fmt: skip
    return row


def cells_qwen() -> tuple[list[str], list[dict[str, Cell]]]:
    labels, rows = [], []
    for pair, (_, _, name) in LIB_PAIRS.items():
        for save_key in load(f"{LIB}/scorecard.json")["pairs"][pair]["saves"]:
            labels.append(f"{name}, {SAVES[save_key][1]}")
            rows.append(library_row(pair, save_key))
    for name, label in ARMS.items():
        labels.append(label)
        rows.append(arm_row(name))
    return labels, rows


def cells_e181() -> list[dict[str, Cell]]:
    built = {(b["rowset"], b["choice"]): b for b in json.loads(E181.read_text())["built"]}
    return [{ch: {"gap": b["gap"] / b["band"], "apart": b["class"] == "above", "source": f"{E181} .built[{rs}, {ch}]"}
             for ch in CHOICES if (b := built[(rs, ch)])["gap"] is not None} for rs in ROWSETS]  # fmt: skip


def draw(ax: Axes, rows: list[dict[str, Cell]], labels: list[str], cols: dict[str, str], label: str) -> None:
    values = [[r[c]["gap"] if c in r else None for c in cols] for r in rows]
    gap_heatmap(ax, values, labels, list(cols.values()), label=label, vmax=1.0)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if r.get(c, {}).get("apart"):
                ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, edgecolor="black", linewidth=1.1))
    ax.xaxis.tick_top()


def records(panel: str, labels: list[str], rows: list[dict[str, Cell]], cols: dict[str, str]) -> list[dict[str, Any]]:
    return [{"panel": panel, "row": lab, "column": c, **r[c]} for lab, r in zip(labels, rows, strict=True) for c in cols if c in r]


def tally(rows: list[dict[str, Cell]], col: str) -> dict[str, int]:
    """Direction of each apart cell in one column against the share's, over the given rows (all of which the share calls)."""
    out: dict[str, int] = {}
    for r in rows:
        c = r.get(col)
        v = "not run" if c is None else "no call" if not c["apart"] else "same" if (c["gap"] > 0) == (r["share"]["gap"] > 0) else "opposite"
        out[v] = out.get(v, 0) + 1
    return out


def main() -> None:
    table, pairs, names = cells_67m()
    rows67 = [{c: table[(p, c)] for c in COLS if (p, c) in table} for p in pairs]
    built = [r for p, r in zip(pairs, rows67, strict=True) if any(c["pair"] == p and c["group"] == "built" for c in load(FINAL)["cells"])]
    qlabels, qrows = cells_qwen()
    erows = cells_e181()
    fig, ax = figure((13.0, 19.0))
    ax.remove()
    grid = fig.add_gridspec(3, 1, height_ratios=[len(pairs), len(qrows) + 1.5, len(ROWSETS) + 1.5])
    a67, aq, ae = (fig.add_subplot(grid[i]) for i in range(3))
    draw(a67, rows67, [names[p] for p in pairs], COLS, "reuse side minus new side")
    a67.axhline(len(built) - 0.5, color="black", linewidth=1.5)
    a67.set_title(f"67M, {len(built)} built pairs then {len(pairs) - len(built)} real-vs-shuffled twins: only the re-gating share is apart and red on every built pair", fontsize=11, pad=8)
    draw(aq, qrows, qlabels, QCOLS, "side A minus side B (÷ null, or log10)")
    aq.axhline(len(qrows) - len(ARMS) - 0.5, color="black", linewidth=1.5)
    aq.set_title("Qwen3-1.7B pair library and two arm contrasts: the share is apart only on the frozen-neuron decomposition, and there on the near-null gold twin too", fontsize=10, pad=8)
    draw(ae, erows, list(ROWSETS), CHOICES, "share minus random curve, ÷ band")
    ae.set_title("Qwen, p-015e0710: do switched-off neuron sets read above random changes of the same size? 7 of 15 readable", fontsize=10, pad=8)
    data = records("67M", [names[p] for p in pairs], rows67, COLS) + records("Qwen", qlabels, qrows, QCOLS) + records("E181", list(ROWSETS), erows, CHOICES)
    out = {"cells": data, "tally_built": {c: tally(built, c) for c in ("overlap", "on_both", "part1")}}
    (ROOT / "experiments/method_table/results/gate_1116.json").write_text(json.dumps(out, indent=1) + "\n")
    save(fig, ROOT / "experiments/method_table/figures/gate_scorecard_1116.png", n=FOOTER, data=data)


if __name__ == "__main__":
    main()
