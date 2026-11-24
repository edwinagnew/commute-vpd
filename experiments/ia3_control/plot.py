"""E290's figure: E_b over the needed set K at the neuron featurization (remainders in, as filed) per key-grid task, the
full fine-tune beside the FFN-gain and IA3 arms, every seed a point, the matched randoms as a band; the FFN arm's gain
share in the tick labels. Reads results/summary.json (run.py --stage summary).

    ~/commute-agents/fleet/cpu-run python experiments/ia3_control/plot.py [SUMMARY.json] [OUT.png]
"""

import json
import sys
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).parent
ARMS = {"full": ("neuron_half", "full fine-tune"), "ffn": ("neuron", "FFN-gain fine-tune"), "ia3": ("regating", "IA3 fine-tune")}


def main(src: Path, out: Path) -> None:
    s = json.loads(src.read_text())
    nr, gain = s["neurons"], s["R2_gain"]
    tasks = list(gain)
    fig, ax = FG.figure(FG.FULL_WIDTH)
    rows = []
    rand = {t: [r["E_b_K"] for i, r in nr.items() if r and i.endswith("_rand") and i.split("/")[1].startswith(f"{t}_s")] for t in tasks}
    for k, (arm, (method, _)) in enumerate(ARMS.items()):
        vals = {t: [r["E_b_K"] if (r := nr.get(f"{arm}/{t}_s{s}")) else None for s in (0, 1, 2)] for t in tasks}
        bands = {t: (min(v), max(v)) if (v := rand[t]) else None for t in tasks} if k == 0 else None
        FG.column_dots(ax, vals, method=method, ylabel="E_b over K at neurons (as filed)", bands=bands)
        rows += [{"arm": arm, "task": t, "seed": i, "E_b_K": v} for t in tasks for i, v in enumerate(vals[t])]
    share = {t: gain[t].get("ffn", {}).get("mean_share") for t in tasks}
    ax.set_xticks(range(len(tasks)), [f"{t}\nFFN gain {v:.2f}" if v is not None else t for t, v in share.items()], fontsize=7)
    for arm, (method, label) in ARMS.items():
        st = FG.METHODS[method]
        ax.scatter([], [], color=st.color, marker=st.marker, label=label)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(tasks) - 0.5)  # a task with no reading keeps its column
    ax.set_title(f"Gain fine-tunes against the full fine-tune at neurons (R3: {s['R3']['call']})", fontsize=9)
    FG.save(fig, out, n="3 seeds per task, 512 windows each; band: the matched randoms", data=rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json",
         Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures" / "eb_neurons.png")  # fmt: skip
