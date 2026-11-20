"""The paper's all-methods scorecard on the 67M, from saved results only (no model, no GPU).

Rows are the methods of the paper's Table tab:scorecard, in its order; columns the nine built pairs, then the four
twins in a separate block. Each cell prints the method's gap (side built to reuse minus side built new; on a twin, real
minus twin) and is coloured by its call only, since the rows are in different units:
agrees = the two sides apart (every seed of one beyond every seed of the other) and the reuse side higher; reverses =
apart and lower; no call = not apart, or one value a side; hatched = not run.

- Every row but the crosscoder: `gate_plot_1116.cells_67m()` (results/final.json, the neuron-overlap stage 2), the
  same cells and calls as figures/gate_scorecard_1116.png.
- Crosscoder: experiments/crosscoder/results/run2/summary.json, column "crosscoder" (the authors' rule, as built).
  One crosscoder per side, so no call; its sign is drawn in a muted tint.

    ~/commute-agents/fleet/cpu-run python experiments/method_table/paper_scorecard_67m.py
"""

import json
import shutil
from pathlib import Path
from typing import Any

from matplotlib.patches import Patch, Rectangle

from commute.figures import FULL_WIDTH, figure, save
from commute.jobs import load_experiment

ROOT = Path(__file__).resolve().parents[2]
GP = load_experiment("method_table", "gate_plot_1116")
XC = "experiments/crosscoder/results/run2/summary.json"
OUT = ROOT / "experiments/method_table/figures/paper_fig_scorecard_67m.png"
PAPER = Path.home() / "commute-data/agents/regating-lead/writeup-repo/paper/figures/paper_fig_scorecard_67m.pdf"
ROWS = {"share": "Re-gating share", "untrained": "Untrained-SVD share", "neuron": "Neuron share, whole width",
        "k_half": "Neurons for half the gain", "neuron_top": "Top-neuron share", "overlap": "Top-neuron overlap",
        "steering": "Steering", "prior": "Base-model prior", "size": "Update size", "erank": "Effective rank",
        "crosscoder": "Crosscoder"}  # fmt: skip
FILL = {"agrees": "#0072B2", "reverses": "#D55E00", "no call": "#E4E4E4", "above zero": "#C6DCEC", "below zero": "#F2D4BF"}
TEXT = {"agrees": "white", "reverses": "white"}
TWIN_GAP = 0.45  # blank space between the built block and the twins, in cell widths
Cell = dict[str, Any]


def call(c: Cell | None, col: str) -> str:
    if c is None:
        return "not run"
    if col == "crosscoder":
        return "above zero" if c["gap"] > 0 else "below zero"
    if not c["apart"]:
        return "no call"
    return "agrees" if c["gap"] > 0 else "reverses"


def label_gap(gap: float, k: str) -> str:
    """Two decimals; a called gap that would print as 0.00 gets three, so a call never sits on a printed zero."""
    text = format(gap, ".3f" if k in ("agrees", "reverses") and abs(gap) < 0.005 else ".2f")
    return text[1:] if text == "-0.00" else text


def cells() -> tuple[list[str], dict[str, str], set[str], dict[tuple[str, str], Cell]]:
    table, pairs, names = GP.cells_67m()
    final = json.loads((ROOT / GP.FINAL).read_text())["cells"]
    built = {c["pair"] for c in final if c["group"] == "built"}
    by_name = {v: k for k, v in names.items()}
    for c in json.loads((ROOT / XC).read_text())["cells"]:
        if c["column"] == "crosscoder":
            table[(by_name[c["row"]], "crosscoder")] = {"gap": c["gap"], "apart": False, "source": f"{XC} .cells[{c['row']}, crosscoder]"}
    return pairs, names, built, table


def main() -> None:
    pairs, names, built, table = cells()
    nb = sum(p in built for p in pairs)
    assert pairs[:nb] == [p for p in pairs if p in built], "built pairs must come first"
    xs = [j + (TWIN_GAP if j >= nb else 0.0) for j in range(len(pairs))]
    fig, ax = figure((FULL_WIDTH[0], 5.6))
    data, tally = [], {}
    for i, col in enumerate(ROWS):
        counts = {"agrees": 0, "reverses": 0, "no call": 0, "above zero": 0, "below zero": 0, "not run": 0}
        for j, pair in enumerate(pairs):
            c = table.get((pair, col))
            k = call(c, col)
            if pair in built:
                counts[k] += 1
            x = xs[j]
            if c is None:
                ax.add_patch(Rectangle((x - 0.47, i - 0.45), 0.94, 0.9, facecolor="white", edgecolor="#BBBBBB", hatch="////", linewidth=0.4))
                continue
            ax.add_patch(Rectangle((x - 0.47, i - 0.45), 0.94, 0.9, facecolor=FILL[k], edgecolor="none"))
            text = label_gap(c["gap"], k)
            ax.text(x, i, text, ha="center", va="center", fontsize=6.3, color=TEXT.get(k, "#222222"))
            data.append({"method": ROWS[col], "pair": names[pair], "group": "built" if pair in built else "twin", "gap": c["gap"], "call": k, "source": c["source"]})
        tally[col] = counts
        if col == "crosscoder":
            label = f"({counts['above zero']}) · ({counts['below zero']}) · {counts['above zero'] + counts['below zero']}"
        else:
            label = f"{counts['agrees']} · {counts['reverses']} · {counts['no call']}"
        ax.text(xs[-1] + 0.75, i, label, ha="left", va="center", fontsize=7)
    ax.text(xs[-1] + 0.75, -0.75, "built:\nagree · rev. · none", ha="left", va="bottom", fontsize=6.3, color="#444444")
    ax.text((nb - 1) / 2, -0.75, "9 built pairs: side built to reuse minus side built new", ha="center", va="bottom", fontsize=7.5)
    ax.text((xs[nb] + xs[-1]) / 2, -0.75, "4 twins: real minus twin", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(len(ROWS) - 0.5, -1.6)
    ax.set_yticks(range(len(ROWS)), list(ROWS.values()), fontsize=7.5)
    ax.set_xticks(xs, [names[p] for p in pairs], rotation=50, ha="right", rotation_mode="anchor", fontsize=7)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_ylabel("method", fontsize=7)
    ax.set_title("Each method's gap on the 67M pairs, coloured by its call", fontsize=9)
    handles = [Patch(facecolor=FILL["agrees"], label="agrees: apart, reuse (real) side higher"),
               Patch(facecolor=FILL["reverses"], label="reverses: apart, reuse (real) side lower"),
               Patch(facecolor=FILL["no call"], label="no call: seed ranges overlap, or one value a side"),
               Patch(facecolor="white", edgecolor="#BBBBBB", hatch="////", label="not run"),
               Patch(facecolor=FILL["above zero"], label="crosscoder, no call: gap above zero"),
               Patch(facecolor=FILL["below zero"], label="crosscoder, no call: gap below zero")]  # fmt: skip
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=6.8, frameon=False)
    save(fig, OUT, n="Seeds per side: 1 to 8; crosscoder one per side. Units differ by row; colour is the call, not the value.", data=data)
    PAPER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT.with_suffix(".pdf"), PAPER)
    print(json.dumps(tally, indent=None), flush=True)


if __name__ == "__main__":
    main()
