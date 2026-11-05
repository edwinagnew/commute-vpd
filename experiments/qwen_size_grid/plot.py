"""E91's figures. Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_size_grid/plot.py

  figures/norm_vs_kl.png      every arm's path in norm and KL to start, against anchor 1's full update
  figures/recall_vs_norm.png  anchor 2's trained-wording recall against norm, with the unchanged model and the learning line
"""
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
E84 = HERE.parent / "qwen_tofu_forms" / "results" / "read.json"
E88 = HERE.parent / "qwen_anchor_checks" / "results"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# RL, null and own-samples keep E69's colours; the others are the next slots of the validated palette.
RL, NULL, SFT, A1, A2, U1, U2 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100"
ARMS = (  # arm, colour, line, legend name (None: the same arm's second seed, no entry of its own)
    ("anchor1_s0", A1, "-", "Anchor 1, template fine-tune"), ("anchor1_s1", A1, "--", None),
    ("anchor2_s0", A2, "-", "Anchor 2, TOFU facts fine-tune"), ("anchor2_s142", A2, "--", None),
    ("u1", U1, "-", "u1, human solutions, no warm-up"), ("u2", U2, "-", "u2, human solutions, warm-up"),
    ("R_s0", RL, "-", "RL"), ("R_s70", RL, "--", None),
    ("N_s0", NULL, "-", "Shuffled-reward null"), ("N_s70", NULL, "--", None),
    ("F_s0", SFT, "-", "Own-samples SFT, one seed"),
)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def paths() -> dict[str, list[tuple[float, float]]]:
    """Each arm's (norm, KL) by step, zero states out (log axes); anchor 2 with every step matched.py read, 20 to 50."""
    steps: dict[str, dict[int, tuple[float, float]]] = {}
    for line in (RES / "reads.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["norm"] > 0:
            steps.setdefault(r["arm"], {})[r["step"]] = (r["norm"], r["kl"])
    runs = load(RES / "matched.json")["runs"]
    for seed, arm in (("42", "anchor2_s0"), ("142", "anchor2_s142")):
        steps[arm] |= {int(t): (runs[seed]["norm"][t], k) for t, k in runs[seed]["kl"].items()}
    return {arm: [here[t] for t in sorted(here)] for arm, here in steps.items()}


def norm_vs_kl() -> None:
    ref = load(RES / "size_grid.json")["reference"]
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    all_paths = paths()
    for arm, color, ls, name in ARMS:
        xs, ys = zip(*all_paths[arm], strict=True)
        ax.plot(xs, ys, ls=ls, color=color, lw=1.8, zorder=2, label=name)
        ax.plot(xs[-1], ys[-1], "o", color=color, ms=7, mec="white", mew=1.2, zorder=3)
    ax.axvline(ref["norm"], color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.axhline(ref["kl"], color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.plot(ref["norm"], ref["kl"], "*", color=INK, ms=17, mec="white", mew=1.0, zorder=4, label="Anchor 1's full update: the reference")
    ax.annotate(f"reference: norm {ref['norm']:.3f}, KL {ref['kl']:.3f}", (ref["norm"], ref["kl"]), xytext=(0.62, 0.0052),
                color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.annotate("own-samples SFT stops here,\nfar under the reference's KL", (0.2739, 0.00102), xytext=(0.6, 0.0004),
                color=MUTED, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.annotate("the null's two seeds stop\nabove and far under it", (0.3354, 0.0429), xytext=(0.06, 0.3),
                color=MUTED, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.annotate("RL's two seeds stop just under it", (0.349, 0.0859), xytext=(0.023, 0.115), color=MUTED, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.text(18.3, 0.2, "anchor 2's\nfull update", color=MUTED, fontsize=9, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 30)
    ax.set_ylim(0.00015, 1.2)
    ax.set_xlabel("Norm of the update over the decomposed matrices")
    ax.set_ylabel("KL from the unchanged model per position, first 16 maths rows")
    ax.set_title("Which arms reach anchor 1's full update in norm and in KL to start?")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, title="Dashed: second seed. Dot: last state", title_fontsize=8.5)
    fig.savefig(FIG / "norm_vs_kl.png")
    plt.close(fig)


def recall_vs_norm() -> None:
    ref = load(RES / "size_grid.json")["reference"]
    matched, read, second = load(RES / "matched.json"), load(E84), load(E88 / "S.json")
    line = load(RES / "predictions_matched.json")["learned_line"]
    unchanged = read["W0"]["trained"]["recall"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    s42 = [(s["norm"], s["answers"]["trained"]["recall"]) for s in matched["states"] if s["seed"] == 42]
    s42 += [(read["files"][t]["norm"], read["states"][t]["trained"]["recall"]) for t in ("200", "500", "1000", "1250")]
    s142 = [(s["norm"], s["answers"]["trained"]["recall"]) for s in matched["states"] if s["seed"] == 142]
    s142.append((second["files"]["1250"]["norm"], second["state"]["trained"]["recall"]))
    ax.axhline(unchanged, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.text(30, unchanged + 0.012, f"unchanged model {unchanged:.3f}", color=MUTED, fontsize=9, ha="right", va="bottom")
    ax.axhline(line, color=INK, lw=1.1, ls="--", zorder=1)
    ax.text(0.53, line + 0.015, f"filed learning line {line}", color=INK, fontsize=9)
    ax.axvline(ref["norm"], color=A1, lw=1.3, zorder=1)
    ax.text(ref["norm"] * 1.06, 0.97, f"anchor 1's full update,\nnorm {ref['norm']:.3f}", color=INK, fontsize=9, va="top")
    ax.axvspan(s42[3][0], s42[4][0], color=A2, alpha=0.22, lw=0, zorder=0)
    ax.annotate("line crossed between steps\n500 and 1000, read at seed 42",
                (15.5, 0.75), xytext=(1.6, 0.75), color=INK, fontsize=9, va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    xs, ys = zip(*s42, strict=True)
    ax.plot(xs, ys, "-o", color=A2, lw=1.6, ms=7, mec="white", mew=1.2, zorder=3, label="Anchor 2, seed 42")
    xs, ys = zip(*s142, strict=True)
    ax.plot(xs, ys, "s", color="#b0406f", ms=6.5, mec="white", mew=1.0, zorder=4, label="Anchor 2, seed 142: no answers read between step 30 and 1250")
    ax.annotate("steps 27 to 30:\nfour states, both seeds", (s42[1][0], s42[1][1]), xytext=(0.56, 0.33), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    for (x, y), t, dx, dy in zip(s42[2:5], (200, 500, 1000), (-6, -6, -6), (7, 7, -3), strict=True):
        ax.annotate(f"step {t}", (x, y), xytext=(dx, dy), textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.annotate("step 1250", s42[5], xytext=(6, 5), textcoords="offset points", ha="left", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 35)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Norm of the update over the decomposed matrices")
    ax.set_ylabel("New-word recall on the trained wording")
    ax.set_title("Has anchor 2 learned its facts when it is the size of anchor 1's full update?")
    ax.legend(loc="lower left", bbox_to_anchor=(0.27, 0.0), fontsize=8.5)
    fig.savefig(FIG / "recall_vs_norm.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    norm_vs_kl()
    recall_vs_norm()
