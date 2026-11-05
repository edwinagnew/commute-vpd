"""E92's figures. Reads result files only, loads no model. Run dissect.py first.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_comparators/plot.py

  figures/first_token_rank.png   the unchanged model's first answer tokens, rank against log P
  figures/share_vs_size.png      steering share and nats raised against update size, with the random control
  figures/where_gain_lands.png   mean gain by the unchanged model's rank of the token
  figures/rank_one_share.png     share of tokens ranked first, unchanged model against each state
  figures/gsm8k_gain.png         the GSM8K arms' gain on gold tokens against the filed band
"""
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# E91's colours for the same arms.
RL, NULL, SFT, A1, A2, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"
A2_S142 = "#b0406f"
D = json.loads((RES / "dissection.json").read_text())
T = {(r["arm"], r["step"]): r for r in json.loads((RES / "comparator_table.json").read_text())["rows"]}


def style(ax: Any) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)


def first_token_rank() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    names = (("a1", "Template rows: every answer opens 'Given'", A1), ("tofu", "TOFU rows", A2), ("gold", "GSM8K gold solutions", GOLD))
    for n, label, c in names:
        p = D["first_token"][n]["points_first"]
        ax.scatter(p["rank"], p["logp"], s=14, alpha=0.55, color=c, label=label, linewidths=0)
    bins = {}
    for n, _, _ in names:
        for b in D["first_token"][n]["by_rank_all"]:
            k = bins.setdefault((b["lo"], b["hi"]), [0, 0.0])
            k[0] += b["n"]
            k[1] += b["n"] * b["mean_logp"]
    xs = sorted(bins)
    ax.plot([(lo * min(hi, 20000)) ** 0.5 for lo, hi in xs], [bins[b][1] / bins[b][0] for b in xs], color=INK, marker="o", markersize=4,
            label="Every answer token, mean log P by rank")
    for rows, rank, lp, c, dx in (("Template", 21, -15.68, A1, 1.4), ("GSM8K", 11, -13.59, GOLD, 1.4), ("TOFU", 1, -7.26, A2, 1.4)):
        ax.scatter([rank], [lp], marker="X", s=130, color=c, edgecolors=INK, zorder=5)
        ax.annotate(f"{rows}: median rank {rank}, mean log P {lp:.1f}", (rank * dx, lp), color=INK, fontsize=9, va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    ax.set_xscale("log")
    ax.set_xlabel("Rank of the answer's first token at the unchanged model, 1 is the top of the vocabulary")
    ax.set_ylabel("Log P of that token, unchanged model")
    ax.set_title("Can a first token at median rank 11 read a mean log P near -14?")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "first_token_rank.png", dpi=150)


def share_vs_size() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    seeds = (("anchor2_s0", (29, 550, 1250), A2, "Facts update, seed 42"), ("anchor2_s142", (30, 500, 1250), A2_S142, "Facts update, seed 142"))
    for ax, key, rnd, se, rse in ((axes[0], "steer_share", "random_share", "steer_se", "random_se"), (axes[1], "steer_rise", "random_rise", None, None)):
        for arm, steps, c, label in seeds:
            rows = [T[(arm, s)] for s in steps]
            x = [r["norm"] for r in rows]
            ax.plot(x, [r[key] for r in rows], color=c, label=f"{label}, the vector")
            ax.plot(x, [r[rnd] for r in rows], color=c, linestyle=":", label=f"{label}, random at its norm")
            for r in rows:
                face = "white" if r["unfiled_addition"] else c
                ax.errorbar(r["norm"], r[key], yerr=r[se] if se else None, color=c, marker="o", markerfacecolor=face, markersize=7, capsize=3)
                ax.scatter(r["norm"], r[rnd], color=c, marker="s", s=22, facecolors=face)
        a1 = [T[("anchor1_s0", 100)], T[("anchor1_s1", 100)]]
        ax.scatter([r["norm"] for r in a1], [r[key] for r in a1], color=A1, marker="D", s=50, zorder=5, label="Template update, the vector, two seeds")
        ax.scatter([r["norm"] for r in a1], [r[rnd] for r in a1], color=A1, marker="s", s=22, zorder=5, label="Template update, random at its norm")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Update size, norm of the change to the decomposed weights")
        style(ax)
    axes[0].annotate("Hollow: steps 550 and 500,\nan unfiled addition", (T[("anchor2_s0", 550)]["norm"], 0.2273), xytext=(3.2, 0.62), fontsize=9, color=INK,
                     arrowprops={"arrowstyle": "-", "color": MUTED})
    axes[0].set_ylabel("Steering share of the state's gain, 1 is all of it")
    axes[0].set_title("Share of the gain one vector recovers")
    axes[1].set_ylabel("Nats per answer token the vector adds")
    axes[1].set_title("The same vector, in nats")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("Does the steering share belong to the update, or to the step it is read at?")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "share_vs_size.png", dpi=150)


def where_gain_lands() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5))
    arms = (("anchor1_s0_step100", "Template update, step 100, norm 0.49", A1, ""), ("anchor2_s0_step29", "Facts update, step 29, norm 0.48", A2, ""),
            ("anchor2_s0_step550", "Facts update, step 550, norm 15.5, unfiled addition", A2, "//"), ("anchor2_s0_step1250", "Facts update, step 1250, norm 18.1", A2_S142, ""))
    labels = ["Ranked first", "Rank 2 to 5", "Rank 6 to 50", "Below rank 50"]
    for i, (name, label, c, hatch) in enumerate(arms):
        bins = D["states"][name]["by_W0_rank"]
        ax.bar([j + (i - 1.5) * 0.2 for j in range(4)], [b["gain"] for b in bins], width=0.19, color="white" if hatch else c, edgecolor=c, hatch=hatch, label=label)
    ax.axhline(0, color=INK, linewidth=1)
    ax.text(-0.42, 0.25, "Do-nothing reads 0 in every group", fontsize=9, color=MUTED)
    ax.set_xticks(range(4), labels)
    ax.set_xlabel("The unchanged model's rank of the token. Template rows for the template update, TOFU rows for the facts update")
    ax.set_ylabel("Mean gain in log P, nats")
    ax.set_title("Where does each update's gain land?")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "where_gain_lands.png", dpi=150)


def rank_one_share() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    arms = (("anchor1_s0_step100", "Template, step 100, seed 0", A1), ("anchor1_s1_step100", "Template, step 100, seed 1", A1),
            ("anchor2_s0_step29", "Facts, step 29, seed 42", A2), ("anchor2_s142_step30", "Facts, step 30, seed 142", A2),
            ("anchor2_s0_step550", "Facts, step 550, seed 42, unfiled addition", A2), ("anchor2_s142_step500", "Facts, step 500, seed 142, unfiled addition", A2),
            ("anchor2_s0_step1250", "Facts, step 1250, seed 42", A2_S142), ("anchor2_s142_step1250", "Facts, step 1250, seed 142", A2_S142))
    for i, (name, _, c) in enumerate(arms):
        q = D["states"][name]["fragments"]["question_whole"]
        ax.plot([q["rank1_W0"], q["rank1_state"]], [i, i], color=c, linewidth=2.5)
        ax.scatter([q["rank1_W0"]], [i], facecolors="white", edgecolors=c, s=50, zorder=3, label="Unchanged model" if i == 0 else None)
        ax.scatter([q["rank1_state"]], [i], color=c, s=50, zorder=3, label="The state" if i == 0 else None)
    ax.set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Share of answer tokens ranked first, on answers whose whole question is in the row")
    ax.set_title("Did the update put the answer's token first?")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "rank_one_share.png", dpi=150)


def gsm8k_gain() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    arms = (("R_s0", "RL, first seed", RL, "-"), ("R_s70", "RL, second seed", RL, "--"), ("N_s0", "Shuffled-reward null, first seed", NULL, "-"),
            ("N_s70", "Shuffled-reward null, second seed", NULL, "--"), ("F_s0", "Own-samples fine-tune, one seed", SFT, "-"))
    ax.axhspan(-0.1, 0.1, color=GRID, alpha=0.7, label="Filed band at step 100: under 0.1 in size")
    for arm, label, c, ls in arms:
        r = [T[(arm, 10)], T[(arm, 100)]]
        ax.errorbar([10, 100], [x["gain_all"] for x in r], yerr=[2 * x["gain_all_se"] for x in r], color=c, linestyle=ls, marker="o", capsize=3, label=label)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([10, 100], ["10", "100"])
    ax.set_xlabel("Step of the update")
    ax.set_ylabel("Gain in mean log P of the gold solution's tokens, nats")
    ax.set_title("Do the GSM8K arms move gold tokens? Yes, past the band, and the null moves them most")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "gsm8k_gain.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    first_token_rank()
    share_vs_size()
    where_gain_lands()
    rank_one_share()
    gsm8k_gain()
