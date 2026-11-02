"""Figures for E61 so far: stage A's three readers and the five checks. Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_recipe/plot.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R, FIG = HERE / "results", HERE / "figures"
READING = HERE.parent / "qwen_reading/results"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FULL, AUTOCAST, ROUNDED = "#6250d6", "#eb6834", "#1baf7a"
MATHS, FACTS, RANDOM = "#2a78d6", "#eda100", "#52514e"
CHANGES = {  # key: label, colour; the same colour for the same kind of change in every figure
    "gold_s0": ("Maths fine-tune, 100 steps", MATHS),
    "e40": ("Maths fine-tune, 10 steps", MATHS),
    "a2long_s0": ("Invented facts, seed 0", FACTS),
    "a2long_s1": ("Invented facts, seed 1", FACTS),
    "gauss": ("Random Gaussian change", RANDOM),
    "inspan": ("Random change, alive subcomponents", RANDOM),
}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "figure.titlesize": 12, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_readers() -> None:
    states = load(R / "A.json")["states"]
    panels = [
        ("own_s0", "Fine-tune on the model's own answers", "Held-out loss, nats per token"),
        ("gold_s0", "Fine-tune on human solutions", "Held-out loss, nats per token"),
        ("a1_s0", "Template fine-tune", "KL from the prompted original model,\nnats per answer position"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, (run, title, ylabel) in zip(axes, panels):
        steps = sorted(states[run], key=int)
        x = range(len(steps))
        for reader, colour, marker, size, label in (
            ("fp32", FULL, "o", 8, "Full-precision weights, how every reading was taken"),
            ("autocast", AUTOCAST, "s", 9, "Under bf16 autocast, how training saw the weights"),
            ("rounded", ROUNDED, "x", 7, "Weights rounded to bf16, then a full-precision forward"),
        ):
            ys = [states[run][s][reader] for s in steps]
            fill = "none" if reader == "autocast" else colour
            ax.plot(x, ys, color=colour, lw=1.6, marker=marker, ms=size, mfc=fill, mew=1.6, label=label)
        w0 = states[run]["0"]["fp32"]
        ax.axhline(w0, color=MUTED, ls="--", lw=1)
        ax.annotate("unchanged model", (1.5 if run == "own_s0" else len(steps) - 1, w0), xytext=(0, 11 if run == "own_s0" else 4),
                    textcoords="offset points", ha="right" if run != "own_s0" else "center", color=MUTED, fontsize=8.5)
        ax.set_xticks(list(x), steps)
        ax.set_xlabel("Training steps taken (not to scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    own = axes[0]
    own.annotate("0.441", (3, states["own_s0"]["4"]["fp32"]), xytext=(-8, -2), textcoords="offset points", ha="right")
    own.annotate("0.055", (3, states["own_s0"]["4"]["autocast"]), xytext=(0, -15), textcoords="offset points", ha="center")
    handles, labels = own.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Does rounding hide the early update? The three readers split early and meet again by step 100", y=1.0)
    fig.tight_layout()
    save(fig, "readers.png")


def strict_rate(path: Path, n: int) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines()][:n]
    return sum(bool(r["strict"]) for r in rows) / n


def fig_answers() -> None:
    gen = load(R / "A.json")["gen"]
    w0_file = R / "large/checks/gen_W0.jsonl"
    rows = [("a1_s0_8", "Template fine-tune\nafter 8 steps", 256), ("a1_s0_19", "Template fine-tune\nafter 19 steps", 64),
            ("gold_s0_100", "Human-solutions fine-tune\nafter 100 steps", 64)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (key, _, n) in enumerate(rows):
        for reader, colour, shift, marker, label in (
            ("fp32", FULL, -0.1, "o", "Full-precision weights"),
            ("e45_bf16_same_problems", ROUNDED, 0.1, "x", "Rounded weights, whole model in bf16 (the saved answers)"),
        ):
            p = gen[key][reader]["strict_all"]
            se = (p * (1 - p) / n) ** 0.5
            ax.errorbar(i + shift, p, yerr=2 * se, color=colour, marker=marker, ms=8, mew=1.8, capsize=3, lw=1.2,
                        label=label if i == 0 else None)
            ax.annotate(f"{p:.2f}", (i + shift, p), xytext=(0.32 * (1 if shift > 0 else -1) * 30, 0),
                        textcoords="offset points", ha="left" if shift > 0 else "right", va="center", fontsize=9)
        w0 = strict_rate(w0_file, n)
        ax.hlines(w0, i - 0.32, i + 0.32, color=MUTED, ls="--", lw=1.2,
                  label="Unchanged model, full precision, same problems" if i == 0 else None)
    ax.set_xticks(range(3), [r[1] + f"\n{r[2]} problems" for r in rows])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of answers correct, strict rule")
    ax.set_title("Do the two versions of a state answer alike? Not after 8 steps; by 19 and 100 they do")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "answers.png")


def fig_masks() -> None:
    k = load(R / "checks_k10.json")
    keys = ["gold_s0", "e40", "gauss", "a2long_s0"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, key in enumerate(keys):
        a, b = k["direct_file"][key], k["direct_true"][key]
        ax.barh(i, 0.02, left=a - 0.01, height=0.5, color=GRID, label="Filed tolerance, 0.01 either side" if i == 0 else None)
        ax.plot(a, i, "o", color=MUTED, ms=8, mfc="none", mew=1.6, label="Masks read through moved vectors (E57)" if i == 0 else None)
        ax.plot(b, i, "o", color=FULL, ms=6, label="Truly original masks" if i == 0 else None)
    ax.set_yticks(range(4), [CHANGES[key][0] for key in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.11)
    ax.set_xlabel("Direct share, first 16 maths rows")
    ax.set_title("Does keeping the masks truly original change the direct share? No, on these four changes")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "masks.png")


def fig_sizes() -> None:
    sizes, boot = load(R / "checks_sizes.json")["states"], load(READING / "boot_direct.json")["maths"]
    low = min(boot[k]["boot95"][0] for k in ("gauss", "inspan"))
    high = max(boot[k]["boot95"][1] for k in ("gauss", "inspan"))
    panels = [
        ("K00_minus_K0", "E57's size: KL from the changed model\nto the CI-masked original, less the same at the original"),
        ("KL_state_W0", "Unmasked KL, changed model as the reference\n"),
        ("KL_W0_state", "Unmasked KL, original model as the reference\n"),
    ]
    offsets = {"a2long_s0": (6, 3), "a2long_s1": (-6, -12), "inspan": (6, 5), "e40": (-6, 5), "gold_s0": (-6, 5), "gauss": (6, -9)}
    short = {"gold_s0": "maths, 100 steps", "e40": "maths, 10 steps", "a2long_s0": "invented facts 0",
             "a2long_s1": "invented facts 1", "gauss": "Gaussian", "inspan": "alive subcomponents"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True, sharex=True)
    for ax, (axis, xlabel) in zip(axes, panels):
        ax.axhspan(low, high, color=GRID, alpha=0.8, lw=0)
        for key, (_, colour) in CHANGES.items():
            d = boot[key]
            x, y = sizes[key][axis], d["direct"]
            ax.errorbar(x, y, yerr=[[y - d["boot95"][0]], [d["boot95"][1] - y]], color=colour, marker="o", ms=7, capsize=2)
            dx, dy = offsets[key]
            ax.annotate(short[key], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                        ha="left" if dx > 0 else "right", color=INK)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xlim(0, 1.95)
        ax.set_ylim(0, 0.105)
    axes[2].annotate("random changes' range", (1.93, low), xytext=(0, 3), textcoords="offset points", ha="right",
                     fontsize=8, color=MUTED)
    axes[0].set_ylabel("Direct share, 128 maths rows\n(E57's masks; bars: 95% row bootstrap)")
    fig.suptitle("Do the maths fine-tunes stand out in size? On the first two measures yes; on the third barely", y=0.99)
    fig.tight_layout()
    save(fig, "sizes.png")


def fig_gain() -> None:
    g = load(R / "checks_gen.json")
    rows = [  # gain and se, paired, problems whose answers fit in both; the last two lines are from the audit's w0_pairs.txt
        ("Own-answers fine-tune, full-precision weights", g["paired"]["drop"]["gain"], g["paired"]["drop"]["se"], FULL),
        ("Own-answers fine-tune, rounded weights", g["bf16_on_file"]["drop"]["gain"], g["bf16_on_file"]["drop"]["se"], ROUNDED),
        ("Unfiltered samples, rounded weights", 0.0251, 0.0107, ROUNDED),
        ("Unchanged model, full-precision less rounded", g["W0_fp32_vs_bf16"]["fp32_minus_bf16"]["drop"]["gain"],
         g["W0_fp32_vs_bf16"]["fp32_minus_bf16"]["drop"]["se"], MUTED),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for i, (_, gain, se, colour) in enumerate(rows):
        ax.errorbar(gain, i, xerr=2 * se, color=colour, marker="o", ms=7, capsize=3, lw=1.3)
        ax.annotate(f"{gain:+.3f}", (gain, i), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(0.03, color=MUTED, ls="--", lw=1)
    ax.annotate("filed minimum, 0.03", (0.03, 3.42), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(4), [r[0] for r in rows])
    ax.set_ylim(3.6, -0.6)
    ax.set_xlabel("Paired gain in strict accuracy, problems whose answers fit at both states (bars: two standard errors)")
    ax.set_title("Does the own-answers gain survive full precision? Yes, smaller; the unfiltered run has no such reading")
    save(fig, "own_gain.png")


def parse_threshold() -> tuple[dict, dict]:
    """Fold over chance against the thinking set, flat and within bins, per change, and the seed pair, from the audit's file."""
    folds, key = {}, None
    text = (R / "near_threshold.txt").read_text()
    for line in text.splitlines():
        if m := re.match(r"(\w+) \|S\|", line):
            key = m[1]
        elif m := re.match(r"\s+T o \d+ fold flat ([\d.]+) fold within bins ([\d.]+)", line):
            folds[key] = (float(m[1]), float(m[2]))
    m = re.search(r"a2long_s0 a2long_s1 o \d+ ([\d.]+) ([\d.]+)", text)
    return folds, {"seed pair": (float(m[1]), float(m[2]))}


def fig_threshold() -> None:
    folds, pair = parse_threshold()
    rows = [(CHANGES[k][0], *folds[k]) for k in CHANGES] + [("The two invented-facts seeds, against each other", *pair["seed pair"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    for i, (_, flat, within) in enumerate(rows):
        ax.plot([within, flat], [i, i], color=GRID, lw=2)
        ax.plot(flat, i, "o", color=MUTED, ms=7, mfc="none", mew=1.5, label="Against all subcomponents off at the original model" if i == 0 else None)
        ax.plot(within, i, "o", color=FULL, ms=7, label="Against those equally close to the line" if i == 0 else None)
    ax.axvline(1, color=INK, ls="--", lw=1)
    ax.annotate("chance", (1, -0.75), xytext=(3, 0), textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -1)
    ax.set_xlabel("Overlap as a multiple of chance (top six: with the thinking set; last: with each other)")
    ax.set_title("Is the overlap of switched-on sets real? Matched on closeness to the line, little is left")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "near_threshold.png")


def fig_floor() -> None:
    cells = {k: load(READING / f"C_{k}.json")["maths"] for k in ("gold_s0", "e40", "a2long_s0", "a2long_s1")}
    c0 = load(READING / "C0.json")
    cells |= {k: c0[k]["maths"] for k in ("gauss", "inspan")}
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    for i, key in enumerate(CHANGES):
        m, share = cells[key]["means"], cells[key]["regating"]
        floor = 1 - (m["K01"] - m["K0"]) / (m["K00"] - m["K0"])
        ax.plot([floor, share], [i, i], color=GRID, lw=2)
        ax.plot(floor, i, "|", color=INK, ms=14, mew=2, label="Floor from the original-subcomponents cell alone" if i == 0 else None)
        ax.plot(share, i, "o", color=CHANGES[key][1], ms=7, label="Re-gating share as read" if i == 0 else None)
    ax.axvline(0.8, color=MUTED, ls="--", lw=1)
    ax.annotate("filed no-separation line, 0.8", (0.8, 5.42), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(6), [v[0] for v in CHANGES.values()])
    ax.set_ylim(5.7, -0.6)
    ax.set_xlim(0.75, 1.01)
    ax.set_xlabel("Share of the removable error taken out by recomputing the masks alone")
    ax.set_title("Could the re-gating share have read low? The algebra already puts it past the line")
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(0.22, 0.45))
    save(fig, "regating_floor.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for fig_maker in (fig_readers, fig_answers, fig_masks, fig_sizes, fig_gain, fig_threshold, fig_floor):
        fig_maker()
