"""Figures for E47, the five claim tests: one figure per claim, each with its reference on the same axes.

Every adversarial value is a ratio KL(state, start s) / KL(W0, start s), read in the same job on the same rows, so W0
reads 1. The pass and fail lines are that job's: 1.5 / (1 + f) and 1.5 (1 + f), f = W0's spread over the harness
starts. Reads results/*.json only (no model). Usage (repo root): uv run python experiments/claim_tests/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
COPY, NOTHING, CONTROL, W0 = "#2a78d6", "#c0392b", "0.35", "0.55"
RANKS = {
    "lora8_tt": ("rank 8", "#2a78d6"),
    "lora32_tt": ("rank 32", "#7b4fc9"),
    "lora128_tt": ("rank 128", "#d35fb7"),
}
START_NAMES = {
    "budget": "budget\nstart",
    "h0": "harness\nstart 0",
    "h1": "harness\nstart 1",
    "h2": "harness\nstart 2",
}


def load(stage: str) -> dict:
    return json.loads((RESULTS / f"{stage}.json").read_text())


def lines(ax: plt.Axes, w0: dict, label: bool = True) -> None:
    """The job's pass and fail lines and W0's own reading (1)."""
    ax.axhline(
        w0["fail_line"],
        color="0.15",
        ls="--",
        lw=1.1,
        label=f"fail line {w0['fail_line']:.2f}" if label else None,
    )
    ax.axhline(
        w0["pass_line"],
        color="0.15",
        ls=":",
        lw=1.3,
        label=f"pass line {w0['pass_line']:.2f}" if label else None,
    )
    ax.axhline(1.0, color=W0, lw=0.8, label="unchanged model W0 = 1" if label else None)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(FIGURES / name)


def fig_i() -> None:
    """Claim (i): teacher-trained copy against do-nothing, both teachers, four adversary starts (t1 + t1b)."""
    t1, t1b = load("t1"), load("t1b")
    starts = ["budget", "h0", "h1", "h2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, teacher, title in zip(
        axes,
        ("r1", "rand32"),
        ("ladder rung 1, SGD teacher", "32-pair cipher, Adam teacher"),
    ):
        for state, colour, name in (
            (f"{teacher}_nothing", NOTHING, "do-nothing: fine-tune left in Δ"),
            (f"{teacher}_copy", COPY, "teacher-trained decomposition"),
        ):
            ratio = {**t1["states"][state]["ratio"], **t1b["states"][state]["ratio"]}
            xs = [i for i, s in enumerate(starts) if s in ratio]
            ax.plot(
                xs, [ratio[starts[i]] for i in xs], "o", color=colour, ms=8, label=name
            )
            for i in xs:
                ax.annotate(
                    f"{ratio[starts[i]]:.2f}",
                    (i, ratio[starts[i]]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=colour,
                )
        lines(ax, t1["W0"], label=ax is axes[0])
        ax.set_xticks(range(4), [START_NAMES[s] for s in starts], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(
        1.05,
        2.2,
        "not read: this state\nstopped once the budget\nstart crossed the fail line",
        fontsize=7,
        color=NOTHING,
    )
    axes[0].set_ylabel("adversarial KL ÷ W0's at the same start", fontsize=9)
    axes[0].set_ylim(0.6, 3.8)
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    axes[1].legend(handles, labels, fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Claim (i): does the teacher-trained decomposition pass at every start where do-nothing fails?\n"
        "Yes on both teachers: every copy under the pass line, every do-nothing over the fail line somewhere",
        fontsize=10,
    )
    save(fig, "e47_i_teacher_training.png")


def fig_ii() -> None:
    """Claim (ii): LoRA rank 8, 32, 128 after teacher training, against the full-rank Adam control, two row sets."""
    t2, t2b, t2c = load("t2"), load("t2b"), load("t2c")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    panels = (
        (
            axes[0],
            {"lora8_tt": t2, "lora32_tt": t2b, "lora128_tt": t2},
            t2["W0"],
            "Pile rows 0-255",
        ),
        (axes[1], {"lora8_tt": t2c, "lora128_tt": t2c}, t2c["W0"], "Pile rows 256-511"),
    )
    for ax, sources, w0, title in panels:
        for x, key in enumerate(RANKS):
            if key not in sources:
                ax.text(
                    x,
                    1.3,
                    "not run\non these rows",
                    ha="center",
                    fontsize=7,
                    color="0.4",
                )
                continue
            rec, colour = sources[key]["states"][key], RANKS[key][1]
            r = rec["ratio"]
            ax.scatter(
                [x - 0.08, x + 0.08],
                [r["h0"], r["h1"]],
                s=18,
                color=colour,
                alpha=0.55,
                zorder=3,
                label="harness starts 0 and 1" if x == 0 and ax is axes[0] else None,
            )
            ax.scatter(
                x,
                rec["mean"],
                s=90,
                marker="D",
                color=colour,
                edgecolor="white",
                zorder=4,
            )
            ax.annotate(
                f"{rec['mean']:.2f}",
                (x, rec["mean"]),
                xytext=(10, -3),
                textcoords="offset points",
                fontsize=8,
            )
        ctrl = sources["lora8_tt"]["states"]["adam_tt"]["ratio"]["h0"]
        ax.scatter(
            3,
            ctrl,
            s=90,
            marker="s",
            color=CONTROL,
            zorder=4,
            label="full-rank Adam fine-tune, teacher-trained (control, start 0)"
            if ax is axes[0]
            else None,
        )
        ax.annotate(
            f"{ctrl:.2f}",
            (3, ctrl),
            xytext=(10, -3),
            textcoords="offset points",
            fontsize=8,
        )
        base = sources["lora8_tt"]["states"]["lora8_tt"]["mean"] * (1 + w0["f"])
        ax.axhspan(
            0,
            base,
            xmin=0.27,
            xmax=0.73,
            color=RANKS["lora8_tt"][1],
            alpha=0.07,
            label="no-cap-effect band: up to rank 8's mean × (1 + f)"
            if ax is axes[0]
            else None,
        )
        lines(ax, w0, label=False)
        ax.text(
            3.45,
            w0["fail_line"],
            f"fail {w0['fail_line']:.2f}",
            fontsize=7,
            va="bottom",
            ha="right",
        )
        ax.text(
            3.45,
            w0["pass_line"],
            f"pass {w0['pass_line']:.2f}",
            fontsize=7,
            va="top",
            ha="right",
        )
        ax.set_xticks(
            range(4),
            ["LoRA\nrank 8", "LoRA\nrank 32", "LoRA\nrank 128", "full rank\n(control)"],
            fontsize=8,
        )
        ax.set_title(f"{title}: f = {w0['f']:.3f}", fontsize=9.5)
        ax.set_xlim(-0.5, 3.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "adversarial KL ÷ W0's (diamond: mean of two starts)", fontsize=9
    )
    axes[0].set_ylim(0.9, 2.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Claim (ii): does the LoRA's pass hold, and does it need the low rank?\n"
        "Rank 8 passes on the second rows only; rank 128 climbs there, but not on the first rows",
        fontsize=10,
    )
    save(fig, "e47_ii_lora_rank.png")


def fig_iii() -> None:
    """Claim (iii): keep against update size along the 0.01 run, and the 0.03 run at the matched sizes."""
    t3 = load("t3")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, seed in zip(axes, ("0", "1")):
        s = t3["seeds"][seed]
        ctrl = s["control"]
        xs = [ctrl[k]["size"] for k in ("10", "100", "1000")]
        ax.plot(
            xs,
            [ctrl[k]["keep"] for k in ("10", "100", "1000")],
            "o-",
            color=COPY,
            ms=7,
            label="plain SGD at rate 0.01, steps 10, 100, 1000 (reference)",
        )
        for k, step in (("100", "a100"), ("1000", "a1000")):
            d, ref = s["detail"][step], ctrl[k]
            ax.errorbar(
                ref["size"],
                ref["keep"],
                yerr=0.10,
                color=COPY,
                alpha=0.35,
                capsize=5,
                lw=0,
                elinewidth=1.2,
                label="±0.10: past it, size falls"
                if k == "100" and seed == "0"
                else None,
            )
            ax.errorbar(
                ref["size"],
                ref["keep"],
                yerr=0.05,
                color=COPY,
                alpha=0.8,
                capsize=3,
                lw=0,
                elinewidth=2,
                label="±0.05: within it, size stands"
                if k == "100" and seed == "0"
                else None,
            )
            ax.scatter(
                d["size"] * 1.03,
                s["a"][k]["keep"],
                s=80,
                marker="^",
                color=NOTHING,
                zorder=4,
                label="plain SGD at rate 0.03, stopped at the same size"
                if k == "100" and seed == "0"
                else None,
            )
            ax.annotate(
                f"step {d['step']}: {s['a'][k]['keep']:.2f}",
                (d["size"] * 1.03, s["a"][k]["keep"]),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=7.5,
                color=NOTHING,
            )
        ax.text(
            0.0035,
            0.1,
            "scaled-down updates not scored:\nthey lose the behaviour itself\n"
            f"(held-out P {s['detail']['b10']['p_ft']:.2f} and {s['detail']['b100']['p_ft']:.2f})",
            fontsize=7,
            color="0.4",
        )
        ax.set_title(f"fine-tune seed {seed}", fontsize=9.5)
        ax.set_xlabel("update size: largest relative change of any matrix", fontsize=9)
        ax.set_xlim(0.002, 0.0165)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "keep: P(target) under the scaled adjoint split\n÷ P(target) under the update itself",
        fontsize=9,
    )
    axes[0].set_ylim(0, 0.75)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Claim (iii): at the same size, does a faster SGD run keep as much as the slower one?\n"
        "It keeps 0.08-0.09 more on both seeds: between the agree and differ lines",
        fontsize=10,
    )
    save(fig, "e47_iii_keep_size.png")


def fig_iv() -> None:
    """Claim (iv): teacher training plus remainder against do-nothing at three starts, fine-tune steps 10 and 1000."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    starts = ["h0", "h1", "h2"]
    for ax, stage, title in zip(
        axes, ("t4b", "t4"), ("fine-tune step 10", "fine-tune step 1000")
    ):
        res = load(stage)
        series = [
            (res, "nothing", NOTHING, "do-nothing: fine-tune left in Δ", -0.1),
            (res, "plus_remainder", COPY, "teacher training plus remainder in Δ", 0.1),
        ]
        if (
            stage == "t4b"
        ):  # t4c read the teacher-trained state alone, same rows, starts and card
            series.append(
                (
                    load("t4c"),
                    "teacher_trained",
                    "#7b4fc9",
                    "teacher-trained, Δ untouched (step 10 only)",
                    0.0,
                )
            )
        for src, state, colour, name, dx in series:
            r = src["states"][state]["ratio"]
            ax.plot(
                [i + dx for i in range(3)],
                [r[s] for s in starts],
                "o",
                color=colour,
                ms=8,
                label=name,
            )
            for i, s in enumerate(starts):
                ax.annotate(
                    f"{r[s]:.2f}",
                    (i + dx, r[s]),
                    xytext=(7, -3) if dx > 0 else (-28, -3) if dx < 0 else (-10, 8),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=colour,
                )
        for i, s in enumerate(starts):
            ax.text(
                i, 0.72, f"q = {res['score']['q'][s]:.2f}", ha="center", fontsize=7.5
            )
        lines(ax, res["W0"], label=ax is axes[0])
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(3), [START_NAMES[s] for s in starts], fontsize=8)
        ax.set_title(f"{title}: filed rule {res['score']['outcome']}", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.65, 6.5)
    axes[0].set_yticks([0.8, 1, 1.5, 2, 3, 5], ["0.8", "1", "1.5", "2", "3", "5"])
    axes[0].set_ylabel("adversarial KL ÷ W0's (log scale)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Claim (iv): is teacher training plus remainder worse than leaving the fine-tune in Δ?\n"
        "Not shown at either step: at step 10 the remainder lowers the reading at one start and raises it at another",
        fontsize=10,
    )
    save(fig, "e47_iv_remainder.png")


def fig_v() -> None:
    """Claim (v): teacher training with plain SGD, three (rate, step) cells and the failing control, two teachers."""
    t5, t5b = load("t5"), load("t5b")
    cells = (
        ("0.03_100", "rate 0.03\nstep 100", t5),
        ("0.01_1000", "rate 0.01\nstep 1000", t5),
        ("0.003_1000", "rate 0.003\nstep 1000", t5b),
    )
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for teacher, colour, name, dx in (
        ("adam", COPY, "r1's Adam teacher", -0.08),
        ("lora", "#7b4fc9", "r1's rank-8 LoRA teacher", 0.08),
    ):
        for x, (cell, _, res) in enumerate(cells):
            rec = res["states"][f"{teacher}_{cell}"]
            ax.scatter(
                x + dx,
                rec["max"],
                s=70,
                color=colour,
                zorder=3,
                label=name if x == 0 else None,
            )
            ax.annotate(
                f"{rec['max']:.2f}\nP {rec['p']:.2f}",
                (x + dx, rec["max"]),
                xytext=(6 if dx > 0 else -34, -24),
                textcoords="offset points",
                fontsize=7,
                color=colour,
            )
    ctrl = t5["states"]["control"]["ratio"]["h0"]
    ax.scatter(
        3,
        ctrl,
        s=80,
        marker="s",
        color=CONTROL,
        zorder=3,
        label="control: rate 0.03 to step 1000, Adam teacher (start 0)",
    )
    ax.annotate(
        f"{ctrl:.2f}", (3, ctrl), xytext=(8, -3), textcoords="offset points", fontsize=8
    )
    lines(ax, t5["W0"])
    ax.set_xticks(
        range(4), [c[1] for c in cells] + ["rate 0.03\nstep 1000"], fontsize=8
    )
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(0.7, 3.6)
    ax.set_ylabel(
        "adversarial KL ÷ W0's, max over\nthe budget start and harness start 0",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Claim (v): does teacher training with plain SGD fail the check?\n"
        "Only at rate 0.03 to step 1000 (read on the Adam teacher); every smaller cell passes on both",
        fontsize=10,
    )
    save(fig, "e47_v_plain_sgd.png")


def main() -> None:
    fig_i()
    fig_ii()
    fig_iii()
    fig_iv()
    fig_v()


if __name__ == "__main__":
    main()
