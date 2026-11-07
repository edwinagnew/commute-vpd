"""Figures for E120's SFT arms (2026-11-07). Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_pass_pools/plot.py

results/grades.json comes from grade_arms.py; the E61 gold arm (the same recipe on 7,217 GSM8K problems) from
experiments/qwen_recipe/results/{train_gold_1e-5,train_W}.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R, FIG = HERE / "results", HERE / "figures"
E61 = HERE.parent / "qwen_recipe/results"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = {  # arm: label, colour, line style; known arms blue, new arms orange, seeds by line style
    "known_s0": ("Known pool, seed 0", "#2a78d6", "-"),
    "known_s1": ("Known pool, seed 1", "#2a78d6", "--"),
    "new_s0": ("New pool, seed 0", "#eb6834", "-"),
    "new_s1": ("New pool, seed 1", "#eb6834", "--"),
}
W0_C, E61_C = "#0b0b0b", "#8a8984"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "figure.titlesize": 12, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_pass() -> None:
    """Held-out pass@1 per arm on each pool's 15 held-out problems, beside W0 and the 0.10 bar."""
    g = load(R / "grades.json")
    sets = [("new_held", "new_held_samples", "New pool's held-out problems"),
            ("known_held", "known_held_samples", "Known pool's held-out problems")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.16
    for i, (w0_key, arm_key, _) in enumerate(sets):
        ax.bar(i - 2 * width, g["w0"][w0_key]["pass1"], width, color=W0_C,
               label="Unchanged model (64 samples)" if i == 0 else None)
        for j, (arm, (lab, c, ls)) in enumerate(ARMS.items()):
            ax.bar(i + (j - 1) * width, g["arms"][arm][arm_key]["pass1"], width, color=c,
                   hatch="//" if ls == "--" else None, edgecolor="white", label=lab if i == 0 else None)
    ax.axhline(0.10, color=MUTED, ls=":", lw=1.2)
    ax.text(-0.45, 0.115, "filed bar for 'learns': 0.10", color=MUTED, fontsize=8.5)
    ax.set_xticks(range(len(sets)), [s[2] for s in sets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass@1: share of 8 samples correct,\nmean over 15 problems")
    ax.set_title("After 100 steps, does either arm solve held-out problems of its pool?")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "pass.png")


def fig_damage() -> None:
    """GSM8K test strict accuracy and held-out gold loss, W0 against step 100, per arm, with E61's gold arm."""
    gold, w = load(E61 / "train_gold_1e-5.json"), load(E61 / "train_W.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
    w0_acc = load(R / "read_known_s0.json")["step_100"]["gsm8k"]["w0_strict_dropped"]
    a.axhspan(w0_acc - 0.05, w0_acc + 0.05, color=GRID, alpha=0.8, label="Within 0.05 of the unchanged model")
    for arm, (lab, c, ls) in ARMS.items():
        r = load(R / f"read_{arm}.json")
        a.plot([0, 100], [w0_acc, r["step_100"]["gsm8k"]["strict_dropped"]], color=c, ls=ls, marker="o", label=lab)
        pool = arm.split("_")[0]
        b.plot([0, 100], [r["w0"]["held_loss"][pool], r["step_100"]["held_loss"][pool]], color=c, ls=ls, marker="o",
               label=lab)
    a.plot([0, 100], [w0_acc, gold["states"]["step_100"]["unprompted"]["strict_dropped"]], color=E61_C, ls="-.",
           marker="s", label="Same recipe on 7,217 GSM8K problems")
    b.plot([0, 100], [w["states"]["w0"]["held_gold"]["loss"], gold["states"]["step_100"]["held_gold"]["loss"]],
           color=E61_C, ls="-.", marker="s")
    a.axhline(w["states"]["gold_s0_100"]["unprompted"]["strict_dropped"], color=E61_C, ls=":", lw=1,
              label="Same recipe, filed reference run, step 100")
    a.set_ylim(0, 1)
    a.set_ylabel("GSM8K test 0-255, greedy strict accuracy")
    a.set_title("GSM8K accuracy falls in every arm")
    b.set_ylabel("Held-out gold loss, nats per token")
    b.set_title("Loss on unseen gold solutions rises in every arm")
    b.text(100, 0.62, "7,217-problem run,\non its own held-out GSM8K set", color=E61_C, fontsize=8, ha="right")
    for ax in (a, b):
        ax.set_xticks([0, 100], ["unchanged model", "step 100"])
        ax.set_xlabel("training step")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3)
    save(fig, "damage.png")


def fig_fit() -> None:
    """Training loss per step, the four arms against E61's gold arm."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for arm, (lab, c, ls) in ARMS.items():
        st = load(R / f"train_{arm}.json")["steps"]
        ax.plot([s["step"] for s in st], [s["loss"] for s in st], color=c, ls=ls, label=f"{lab}, 62 problems")
    st = load(E61 / "train_gold_1e-5.json")["steps"]
    ax.plot([s["step"] for s in st], [s["loss"] for s in st], color=E61_C, ls="-.",
            label="Same recipe, 7,217 GSM8K problems")
    ax.set_yscale("log")
    ax.set_xlabel("training step (64 sequences a step)")
    ax.set_ylabel("Training loss, nats per token")
    ax.set_title("The 62-problem arms fit their training set almost exactly; the 7,217-problem arm does not")
    ax.legend(fontsize=8.5)
    save(fig, "fit.png")


def fig_size() -> None:
    """|δW| per step for the four arms, with the 0.35 and 0.49 save sizes and E61's gold arm."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for arm, (lab, c, ls) in ARMS.items():
        st = load(R / f"train_{arm}.json")["steps"][1:]  # step 1 is exactly 0 (rate 0), off a log axis
        end = load(R / f"read_{arm}.json")["files"]["100"]["norm"]
        ax.plot([s["step"] for s in st], [s["dw_norm"] for s in st], color=c, ls=ls, label=f"{lab}: {end:.2f} at step 100")
    st = load(E61 / "train_gold_1e-5.json")["steps"][1:]
    ax.plot([s["step"] for s in st], [s["dw_norm"] for s in st], color=E61_C, ls="-.",
            label=f"Same recipe, 7,217 GSM8K problems: {st[-1]['dw_norm']:.2f}")
    ax.set_yscale("log")
    for y, txt, va in ((0.35, "0.35: size of the RL update", "top"), (0.49, "0.49: size of the template update",
                                                                       "bottom")):
        ax.axhline(y, color=MUTED, ls=":", lw=1)
        ax.text(99, y * (0.95 if va == "top" else 1.05), txt, color=MUTED, fontsize=8, va=va, ha="right")
    ax.set_xlabel("training step")
    ax.set_ylabel("Frobenius norm of the update,\n196 decomposed matrices")
    ax.set_title("Is the new pool's update larger than the known pool's?")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "size.png")


if __name__ == "__main__":
    fig_pass()
    fig_damage()
    fig_fit()
    fig_size()
