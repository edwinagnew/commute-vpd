"""The research log's five figures for E47, redrawn from saved results with plain labels (no model).

  log_teacher_training.png  claim (i): worst-start adversarial ratio of do-nothing and the teacher-trained
                            decomposition on two tasks, each start as a dot (t1.json, t1b.json).
  log_plain_sgd.png         claim (v): teacher training with plain SGD at four rate-and-length settings on the Adam
                            and the rank-8 LoRA teacher, worst of two starts (t5.json, t5b.json).
  log_lora_rank.png         claim (ii): the LoRA teacher-trained decompositions at ranks 8, 32, 128 and the full-rank
                            control, mean of two starts, on two Pile row sets with their own lines (t2, t2b, t2c.json).
  log_remainder.png         claim (iv): do-nothing, teacher training alone and teacher training plus remainder at
                            three starts, fine-tune steps 10 and 1000 (t4b.json, t4c.json, t4.json).
  log_keep_size.png         claim (iii): share of the behaviour the scaled adjoint split keeps, slower against faster
                            SGD run at two matched update sizes, mean of two seeds (t3.json).

Every adversarial value is the attacker's KL for the state over its KL for the unchanged model at the same start, so
the unchanged model reads 1; bars start there. Usage (repo root): python experiments/claim_tests/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plot import CONTROL, COPY, FIGURES, NOTHING, load

PURPLE, INK = "#7b4fc9", "0.15"
WHICH = "' which' then ' was'"


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def lines(ax: Axes, w0: dict[str, Any]) -> None:
    """The job's pass and fail lines, labelled just right of the axes, and the unchanged model's 1."""
    for key, ls, word in (("fail_line", "--", "fail"), ("pass_line", ":", "pass")):
        ax.axhline(w0[key], color=INK, ls=ls, lw=1.1)
        ax.text(1.01, w0[key], f"{word} line\n{w0[key]:.2f}", fontsize=7.5, va="center",
                transform=ax.get_yaxis_transform())
    ax.axhline(1.0, color="0.55", lw=0.8)


def bar_from_one(ax: Axes, x: float, v: float, w: float, colour: str, label: str | None = None,
                 side: bool = False) -> None:
    """A bar from the unchanged decomposition's 1 to v; its value above it, or beside it when dots sit on top."""
    ax.bar(x, v - 1, w, bottom=1, color=colour, alpha=0.85, label=label)
    if side:
        ax.text(x + w / 2 + 0.03, v, f"{v:.2f}", ha="left", va="center", fontsize=8)
    else:
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(FIGURES / name)


def fig_teacher_training() -> None:
    t1, t1b = load("t1"), load("t1b")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    groups = (("r1", f"task {WHICH}\n(SGD fine-tune)"), ("rand32", "32-pair word cipher\n(Adam fine-tune)"))
    for g, (teacher, _) in enumerate(groups):
        for j, (state, colour, name) in enumerate((
            ("nothing", NOTHING, "do-nothing: the whole fine-tune left in Δ"),
            ("copy", COPY, "teacher-trained decomposition"),
        )):
            key = f"{teacher}_{state}"
            r = {**t1["states"][key]["ratio"], **t1b["states"][key]["ratio"]}
            x = g + (j - 0.5) * 0.36
            bar_from_one(ax, x, max(r.values()), 0.32, colour, name if g == 0 else None)
            ax.scatter([x] * len(r), list(r.values()), s=16, color="k", zorder=3,
                       label="each adversary start" if g == 0 and j == 0 else None)
    ax.text(-0.18, 2.55, "two of four starts read:\nstopped once it failed", fontsize=7, color=NOTHING, ha="center")
    lines(ax, t1["W0"])
    ax.set_xticks(range(2), [lab for _, lab in groups], fontsize=8.5)
    ax.set_xlim(-0.5, 1.55)
    ax.set_ylim(0.8, 3.9)
    ax.set_ylabel("adversarial ratio (1 = the unchanged decomposition)\nbar: worst start", fontsize=9)
    ax.set_title("Teacher training passes at every start on both tasks; do-nothing fails at its worst", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    style(ax)
    save(fig, "log_teacher_training.png")


def fig_plain_sgd() -> None:
    t5, t5b = load("t5"), load("t5b")
    cells = (("0.03_100", "rate 0.03\n100 steps", t5), ("0.01_1000", "rate 0.01\n1,000 steps", t5),
             ("0.003_1000", "rate 0.003\n1,000 steps", t5b))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for j, (teacher, colour, name) in enumerate((("adam", COPY, "teacher: full fine-tune (Adam)"),
                                                  ("lora", PURPLE, "teacher: rank-8 LoRA fine-tune"))):
        for x, (cell, _, res) in enumerate(cells):
            bar_from_one(ax, x + (j - 0.5) * 0.36, res["states"][f"{teacher}_{cell}"]["max"], 0.32, colour,
                         name if x == 0 else None)
    bar_from_one(ax, 3 - 0.18, t5["states"]["control"]["max"], 0.32, COPY)
    ax.text(3 + 0.18, 1.5, "not read\nin E47", fontsize=7, ha="center", color=PURPLE)
    lines(ax, t5["W0"])
    ax.set_xticks(range(4), [c[1] for c in cells] + ["rate 0.03\n1,000 steps"], fontsize=8.5)
    ax.set_xlim(-0.5, 3.45)
    ax.set_ylim(0.8, 3.6)
    ax.set_xlabel("plain SGD as teacher training's optimiser", fontsize=9)
    ax.set_ylabel("adversarial ratio, worst of two starts\n(1 = the unchanged decomposition)", fontsize=9)
    ax.set_title(f"Plain SGD passes at every smaller setting on both teachers ({WHICH});\n"
                 "only the largest rate run longest fails", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    style(ax)
    save(fig, "log_plain_sgd.png")


def fig_lora_rank() -> None:
    t2, t2b, t2c = load("t2"), load("t2b"), load("t2c")
    ranks = (("lora8_tt", "LoRA\nrank 8"), ("lora32_tt", "LoRA\nrank 32"), ("lora128_tt", "LoRA\nrank 128"),
             ("adam_tt", "full rank\n(must fail)"))
    panels = (({"lora8_tt": t2, "lora32_tt": t2b, "lora128_tt": t2, "adam_tt": t2}, "Pile rows 0-255"),
              ({"lora8_tt": t2c, "lora128_tt": t2c, "adam_tt": t2c}, "Pile rows 256-511"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (src, title) in zip(axes, panels, strict=True):
        for x, (key, _) in enumerate(ranks):
            if key not in src:
                ax.text(x, 1.1, "not run", ha="center", fontsize=7.5, color="0.4")
                continue
            rec = src[key]["states"][key]
            bar_from_one(ax, x, rec["mean"], 0.5, CONTROL if key == "adam_tt" else COPY, side=True)
            ax.scatter([x] * len(rec["ratio"]), list(rec["ratio"].values()), s=16, color="k", zorder=3)
        lines(ax, src["lora8_tt"]["W0"])
        ax.set_xticks(range(4), [lab for _, lab in ranks], fontsize=8.5)
        ax.set_xlim(-0.5, 3.45)
        ax.set_title(title, fontsize=9.5)
        style(ax)
    axes[0].set_ylim(0.9, 2.35)
    axes[0].set_ylabel("adversarial ratio after teacher training\nbar: mean of two starts; dots: each start", fontsize=9)
    fig.suptitle("The rank-8 LoRA passes on the second row set only; rank 128 climbs to just under the fail line there",
                 fontsize=10.5)
    fig.tight_layout(w_pad=5)
    save(fig, "log_lora_rank.png")


def fig_remainder() -> None:
    starts = ("h0", "h1", "h2")
    states = (("nothing", NOTHING, "do-nothing: the whole fine-tune left in Δ"),
              ("teacher_trained", PURPLE, "teacher training alone, Δ untouched"),
              ("plus_remainder", COPY, "teacher training plus remainder in Δ"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (srcs, title) in zip(axes, (((load("t4b"), load("t4c")), "fine-tune step 10"),
                                        ((load("t4"),), "fine-tune step 1,000")), strict=True):
        shown = [(k, c, n) for k, c, n in states if any(k in s["states"] for s in srcs)]
        w = 0.8 / len(shown)
        for j, (key, colour, name) in enumerate(shown):
            r = next(s for s in srcs if key in s["states"])["states"][key]["ratio"]
            for i, s in enumerate(starts):
                bar_from_one(ax, i + (j - (len(shown) - 1) / 2) * w, r[s], w * 0.9, colour,
                             name if i == 0 and ax is axes[0] else None)
        lines(ax, srcs[0]["W0"])
        ax.set_xticks(range(3), [f"adversary start {i + 1}" for i in range(3)], fontsize=8.5)
        ax.set_xlim(-0.5, 2.45)
        ax.set_title(title, fontsize=9.5)
        style(ax)
    axes[0].set_ylim(0.9, 2.5)
    axes[1].set_ylim(0.9, 5.7)
    axes[0].set_ylabel("adversarial ratio (1 = the unchanged decomposition)", fontsize=9)
    fig.suptitle("Plus remainder is not shown worse than do-nothing: at step 10 its place flips with the start,\n"
                 "at step 1,000 it is higher at every start but by as little as 0.5%", fontsize=10.5)
    fig.tight_layout(w_pad=5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=8, frameon=False)
    save(fig, "log_remainder.png")


def fig_keep_size() -> None:
    seeds = load("t3")["seeds"]
    groups = (("100", "a100", "smaller size\n(slower run's step 100)"),
              ("1000", "a1000", "larger size\n(slower run's step 1,000)"))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for g, (k, det, _) in enumerate(groups):
        slow = [seeds[s]["control"][k]["keep"] for s in seeds]
        fast = [seeds[s]["a"][k]["keep"] for s in seeds]
        steps = "/".join(str(seeds[s]["detail"][det]["step"]) for s in seeds)
        for j, (ys, colour, name) in enumerate(((slow, "0.6", "slower run: plain SGD at rate 0.01"),
                                                (fast, NOTHING, "faster run: rate 0.03, stopped at the same size"))):
            x, m = g + (j - 0.5) * 0.36, sum(ys) / len(ys)
            ax.bar(x, m, 0.32, color=colour, alpha=0.85, label=name if g == 0 else None)
            ax.scatter([x, x], ys, s=16, color="k", zorder=3, label="each seed" if g == 0 and j == 0 else None)
            ax.text(x, max(ys) + 0.015, f"{m:.2f}" + (f"\n(step {steps})" if j else ""), ha="center", fontsize=8)
        ax.annotate(f"+{(sum(fast) - sum(slow)) / len(fast):.2f}", (g + 0.18, sum(fast) / len(fast) + 0.12),
                    ha="center", fontsize=9, fontweight="bold", color=NOTHING)
    ax.set_xticks(range(2), [lab for *_, lab in groups], fontsize=8.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("share of the fine-tune's behaviour the split keeps\n(1 = all of it)", fontsize=9)
    ax.set_xlabel(f"scaled adjoint split of a plain-SGD fine-tune on {WHICH}", fontsize=9)
    ax.set_title("At the same update size, the faster run keeps about 0.08 more", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    style(ax)
    save(fig, "log_keep_size.png")


def main() -> None:
    fig_teacher_training()
    fig_plain_sgd()
    fig_lora_rank()
    fig_remainder()
    fig_keep_size()


if __name__ == "__main__":
    main()
