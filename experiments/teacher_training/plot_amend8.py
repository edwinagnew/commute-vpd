"""Figures for teacher_training Amendment 8, training with only the base model W0 to learn from (plan:
reports/plans/2026-11-02_plan_base-control.md), read from the run's JSONs only. The scoring is the run's own (`amend8.score`,
in each results file under "score").

  amend8_check.png      the adversarial ratio of every state with nothing to learn, beside the real student, both fine-tunes
                        left in Delta, the true-label states and the masked recipe, against W0's band and the L40's lines
  amend8_damage.png     the same ratios against damage (mean KL from W0 away from the trigger), marked by the rows it is read on
  amend8_size.png       the student's size along training: real, null, both walks back, self-labels and true labels
  amend8_direction.png  the cosines between weight changes, with the filed lines and the random-direction floor
  amend8_gap.png        KL to the student's own teacher: training batches along the run, held-out windows at the reads
  amend8_delta_out.png  the Delta-out ratio along training for every state with nothing to learn
  amend8_return.png     the round trip and the random start: KL to W0 and size at 0, 100, 1000
  amend8b_seeds.png     Amendment 8b: state / W0 at three adversary starts, null student, real student, random start
  amend8b_cells.png     Amendment 8b: start x teacher, gap to the teacher and the adversarial ratio
  amend8c_cosines.png   Amendment 8c: cosines between two batch orders, with a student's own teacher and the other's

Usage (repo root): ~/commute-agents/fleet/cpu-run python experiments/teacher_training/plot_amend8.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
INK, MUTED, GRID, BAND = "#0b0b0b", "#6e6d68", "#e6e5e0", "#dcdbd5"
PASS_C, FAIL_C = "#008300", "#c0392b"
C = {"real": "#2a78d6", "null": "#eb6834", "n1": "#1baf7a", "n2": "#6250d6", "adam": "#eda100", "sgd": "#e87ba4", "n5": "#008300",
     "ref": "#52514e", "moved": "#8f3f12"}
NAME = {"real": "real student", "null": "null student", "n1": "random start, back to base", "n2": "round trip, back to base",
        "adam": "self-labels, Adam", "sgd": "self-labels, SGD", "n5": "masked recipe, coefficient 0", "moved": "null teacher, random start"}
STAGES = ("n1", "n2", "n3b", "n4_adam", "n4_sgd", "n5")


def load(stage: str, prefix: str = "amend8_") -> dict[str, Any]:
    return json.loads((RESULTS / f"teacher_training_{prefix}{stage}.json").read_text())


def lines(ax: Axes, w0: dict[str, Any], vertical: bool) -> None:
    """W0's band over adversary seeds and the job's pass and fail lines, each labelled in the legend."""
    pile, floor = w0["pile"], w0["pgd_floor"]
    top = max(pile[s]["pgd_kl"] for s in pile) / pile["0"]["pgd_kl"]
    span, line = (ax.axvspan, ax.axvline) if vertical else (ax.axhspan, ax.axhline)
    span(1.0, top, color=BAND, zorder=0, label=f"unchanged model, three adversary starts (1 to {top:.2f})")
    line(1.5 / (1 + floor), color=PASS_C, ls="--", lw=1.2, label=f"pass line {1.5 / (1 + floor):.2f}")
    line(1.5 * (1 + floor), color=FAIL_C, ls="--", lw=1.2, label=f"fail line {1.5 * (1 + floor):.2f}")


def tidy(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def check_fig(d: dict[str, dict[str, Any]], m0: float) -> None:
    n1, n2, n3, ad, sg, n5 = (d[s]["reads"] for s in STAGES)
    rows: list[tuple[str, str, list[tuple[float, bool]]]] = [
        (NAME["n1"], "n1", [(n1["100"]["pgd"], False), (n1["1000"]["pgd"], True)]),
        (NAME["n2"], "n2", [(n2["100"]["pgd"], False), (n2["1000"]["pgd"], True)]),
        (NAME["adam"], "adam", [(ad["100"]["pgd"], False), (ad["1000"]["pgd"], True)]),
        (NAME["sgd"], "sgd", [(sg["100"]["pgd"], True), (sg["1000"]["pgd"], False)]),
        (NAME["n5"], "n5", [(n5["100"]["pgd"], False), (n5["1000"]["pgd"], True)]),
        (NAME["null"], "null", [(n3["100"]["pgd"], False), (n3["1000"]["pgd"], True)]),
        (NAME["real"], "real", [(n3["real"]["pgd"], True)]),
        ("real fine-tune left in Δ", "ref", [(n3_nothing(d), True)]),
        ("null fine-tune left in Δ", "ref", [(n3["nothing"]["pgd"], True)]),
        ("true labels, SGD", "ref", [(sg["true"]["pgd"], True)]),
        ("true labels, Adam", "ref", [(ad["true"]["pgd"], True)]),
        ("masked recipe, coefficient 0.5", "ref", [(m0, True)]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    lines(ax, d["n1"]["W0"], True)
    for i, (name, key, pts) in enumerate(rows):
        y = len(rows) - 1 - i
        for v, scored in pts:
            ax.scatter(v, y, s=70, facecolor=C[key] if scored else "white", edgecolor=C[key], lw=1.6, zorder=3)
        v = next(v for v, s in pts if s)
        ax.annotate(f"{v:.2f}", (max(p[0] for p in pts), y), xytext=(9, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xscale("log")
    ax.set_xlim(0.85, 40)
    ax.set_xticks([1, 1.5, 2, 3, 5, 10, 20], ["1", "1.5", "2", "3", "5", "10", "20"])
    ax.minorticks_off()
    ax.axhline(5.5, color=MUTED, lw=0.8)
    ax.set_xlabel("adversarial ratio: the state's KL under the adversary over the unchanged model's, same job (log scale)")
    ax.set_title("With nothing to learn, does the adversarial check stay at the unchanged model's level?", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=MUTED),
                Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)]
    labels += ["scored state (value labelled)", "other read of that run"]
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8_check")


def n3_nothing(d: dict[str, dict[str, Any]]) -> float:
    return d["n2"]["reads"]["nothing"]["pgd"]


def damage_fig(d: dict[str, dict[str, Any]]) -> None:
    n1, n2, n3, ad, sg, n5 = (d[s]["reads"] for s in STAGES)
    task = [("n1", n1["1000"], "random start"), ("n2", n2["1000"], "round trip"), ("null", n3["100"], "null student, step 100"),
            ("null", n3["1000"], "null student"), ("real", n3["real"], "real student"), ("n5", n5["1000"], "masked recipe, coefficient 0"),
            ("ref", n3["nothing"], "null fine-tune in Δ"), ("ref", n2["nothing"], "real fine-tune in Δ"),
            ("moved", d["moved"]["reads"]["1000"], NAME["moved"])]
    plain = [("adam", ad["1000"], "self-labels, Adam"), ("sgd", sg["100"], "self-labels, SGD 100"), ("sgd", sg["1000"], "self-labels, SGD 1000"),
             ("ref", sg["true"], "true labels, SGD"), ("ref", ad["true"], "true labels, Adam")]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    lines(ax, d["n1"]["W0"], False)
    offset = {"round trip": (-14, -15), "real student": (-14, -16), "self-labels, Adam": (10, -11), "self-labels, SGD 1000": (9, 7),
              "real fine-tune in Δ": (-6, 8), "true labels, SGD": (7, -13), "random start": (-12, -15), "masked recipe, coefficient 0": (-45, 10), NAME["moved"]: (-120, 6)}
    for group, marker in ((task, "o"), (plain, "s")):
        for key, r, label in group:
            ax.scatter(r["damage"], r["pgd"], s=70, marker=marker, color=C[key], edgecolor="white", lw=1, zorder=3)
            ax.annotate(label, (r["damage"], r["pgd"]), xytext=offset.get(label, (7, 4)), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5, 7], ["1", "1.5", "2", "3", "5", "7"])
    ax.minorticks_off()
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2], ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2"])
    ax.set_xlim(0.004, 0.6)
    ax.set_xlabel("damage: mean KL from the unchanged model away from the trigger, nats (log scale)")
    ax.set_ylabel("adversarial ratio (log scale)")
    ax.set_title("Does the check rise with damage, across the states of this amendment?", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=MUTED),
                Line2D([], [], marker="s", color="none", markerfacecolor=MUTED, markeredgecolor=MUTED)]
    labels += ["damage read on task windows", "damage read on plain Pile rows"]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8_damage")


def curve(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    return [r["step"] for r in rows], [r[key] for r in rows]


def size_fig(d: dict[str, dict[str, Any]]) -> None:
    series = [("real", d["n3b"]["curves"]["real"], "real student 0.018"), ("null", d["n3b"]["curves"]["null_student"], "null student 0.026"),
              ("n1", d["n1"]["curves"]["walk"], "random start, back to base 0.023"), ("n2", d["n2"]["curves"]["walk"], "round trip, back to base 0.021"),
              ("adam", d["n4_adam"]["curves"]["self"], "self-labels, Adam, lowercase rows 0.024"), ("ref", d["n4_adam"]["curves"]["true"], "true labels, Adam, lowercase rows 0.054")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for key, rows, label in series:
        x, y = curve(rows, "rel_norm_max")
        ax.plot(x, y, color=C[key], lw=2, marker="o", ms=5, label=label)
    t = d["n3b"]["real_teacher"]["size"], d["n3b"]["null_teacher"]["rel_norm_max"]
    ax.axhline(t[0], color=C["real"], ls=":", lw=1.3, label=f"real fine-tune, its teacher {t[0]:.3f}")
    ax.axhline(t[1], color=C["null"], ls=":", lw=1.3, label=f"null fine-tune, its teacher {t[1]:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("student step (log scale)")
    ax.set_ylabel("size: largest ‖δW‖ / ‖W₀‖ over the 24 matrices")
    ax.set_ylim(0, 0.058)
    ax.set_title("On this task, does the student's size depend on its teacher?", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8_size")


def direction_fig(d: dict[str, dict[str, Any]]) -> None:
    s = d["n3b"]["direction"]
    rows = [("real student, two batch orders", s["control"], C["real"]), ("null student and its own teacher", s["null_teacher"], C["null"]),
            ("real student and its teacher", s["real_teacher"], C["real"]), ("null student and real student", s["null_real"], C["null"]),
            ("random directions (filed floor)", 2e-4, C["ref"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, (name, v, c) in enumerate(rows):
        ax.barh(len(rows) - 1 - i, v, color=c, height=0.55, zorder=3)
        ax.annotate(f"{v:.3f}" if v > 1e-3 else "about 0.0002", (v, len(rows) - 1 - i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.axvline(0.1, color=MUTED, ls="--", lw=1.2, label="filed 'no shared direction' line 0.1")
    ax.axvline(0.5 * s["control"], color=INK, ls="--", lw=1.2, label=f"filed 'shared direction' line {0.5 * s['control']:.2f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("cosine of the weight changes over all 24 matrices, step 1,000")
    ax.set_title("Is the null student's move the real student's move?", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8_direction")


def gap_fig(d: dict[str, dict[str, Any]]) -> None:
    series = [("real", d["n3b"]["curves"]["real"], {1000: d["n3b"]["reads"]["real"]["teacher_gap"]}),
              ("null", d["n3b"]["curves"]["null_student"], {100: d["n3b"]["reads"]["100"]["teacher_gap"], 1000: d["n3b"]["reads"]["1000"]["teacher_gap"]}),
              ("n1", d["n1"]["curves"]["walk"], {100: d["n1"]["reads"]["100"]["kl_w0"], 1000: d["n1"]["reads"]["1000"]["kl_w0"]}),
              ("n2", d["n2"]["curves"]["walk"], {100: d["n2"]["reads"]["100"]["kl_w0"], 1000: d["n2"]["reads"]["1000"]["kl_w0"]})]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    hook = [d["n2"]["early"]["warm start KL"]["value"], d["n1"]["early"]["warm start KL"]["value"]]
    ax.axhspan(*hook, color=BAND, zorder=0, label=f"hook path against merged weights, at the two warm starts ({hook[0]:.4f}, {hook[1]:.4f})")
    for key, rows, held in series:
        x, y = curve(rows, "loss")
        ax.plot(x, y, color=C[key], lw=2, label=f"{NAME[key]}: one training batch")
        ax.scatter(list(held), list(held.values()), s=70, color=C[key], edgecolor="white", lw=1, zorder=4)
        ax.annotate(f"{held[1000]:.4f}", (1000, held[1000]), xytext=(8, {"n2": 4, "n1": -3}.get(key, 0)), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.9, 1800)
    ax.set_xlabel("student step (log scale)")
    ax.set_ylabel("unmasked KL to the student's own teacher, nats (log scale)")
    ax.set_title("How close does the student get to its own teacher, and is that a floor?", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="white", ms=9))
    labels.append("the 73 held-out windows, at a read (value at step 1,000)")
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8_gap")


def delta_out_fig(d: dict[str, dict[str, Any]]) -> None:
    def seq(stage: str, start: float | None) -> tuple[list[int], list[float]]:
        r = d[stage]["reads"]
        xs = [0] if start is not None else []
        ys = [start] if start is not None else []
        for i, k in enumerate(("100", "1000"), start=1):
            xs.append(i)
            ys.append(r[k]["pile_ratios"]["unmasked_kl"])
        return xs, ys

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(1, color=INK, lw=1.2, label="unchanged model")
    n1s, n2s = d["n1"]["reads"]["0"]["pile_ratios"]["unmasked_kl"], d["n2"]["reads"]["0"]["pile_ratios"]["unmasked_kl"]
    for key, stage, start in (("n1", "n1", n1s), ("adam", "n4_adam", 1.0), ("sgd", "n4_sgd", 1.0), ("n5", "n5", 1.0), ("null", "n3b", 1.0), ("n2", "n2", n2s), ("moved", "moved", n1s)):
        x, y = seq(stage, start)
        ax.plot(x, y, color=C[key], lw=2, marker="o", ms=6, label=NAME[key])
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(8, {"adam": 6, "n1": -5, "moved": -7, "null": 6}.get(key, 0)), textcoords="offset points", va="center", fontsize=9, color=INK)
    real = d["n3b"]["reads"]["real"]["pile_ratios"]["unmasked_kl"]
    ax.scatter(0, real, marker="D", s=70, color=C["real"], zorder=4, label=f"real student, step 1,000: {real:.3f}, the round trip's start")
    ax.set_xticks([0, 1, 2], ["start", "step 100", "step 1,000"])
    ax.set_xlim(-0.15, 2.3)
    ax.set_ylim(0.78, 1.04)
    ax.set_ylabel("Δ-out ratio: KL of Δ's removal, state over unchanged model")
    ax.set_title("Does the fall in the Δ-out ratio need a fine-tuned teacher?", loc="left")
    tn = d["n4_adam"]["reads"]["true"]["pile_ratios"]["unmasked_kl"], d["n2"]["reads"]["nothing"]["pile_ratios"]["unmasked_kl"], d["n3b"]["reads"]["nothing"]["pile_ratios"]["unmasked_kl"]
    ax.text(0.02, 0.03, f"Off the axis: true labels, Adam {tn[0]:.2f}; fine-tune left in Δ, real {tn[1]:.2f}, null {tn[2]:.2f}.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(loc="center left", bbox_to_anchor=(1.09, 0.5), frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8_delta_out")


def return_fig(d: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for key, stage in (("n2", "n2"), ("n1", "n1")):
        r = d[stage]["reads"]
        steps = ("0", "100", "1000")
        axes[0].plot(range(3), [r[s]["kl_w0"] for s in steps], color=C[key], lw=2, marker="o", label=NAME[key])
        axes[1].plot(range(3), [r[s]["size"] for s in steps], color=C[key], lw=2, marker="o", label=NAME[key])
        axes[0].annotate(f"{r['1000']['kl_w0']:.3f}", (2, r["1000"]["kl_w0"]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        axes[1].annotate(f"{r['1000']['size']:.3f}", (2, r["1000"]["size"]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    start = d["n2"]["reads"]["0"]["size"]
    axes[1].axhline(0.24 * start, color=PASS_C, ls="--", lw=1.2, label=f"filed bar for 'returned', {0.24 * start:.4f}")
    axes[1].axhline(0, color=INK, lw=1.2, label="unchanged model 0")
    axes[0].axhline(0.05 * d["n2"]["reads"]["0"]["kl_w0"], color=C["n2"], ls="--", lw=1.2, label="round trip's filed gap line, 5% of its start")
    axes[0].axhline(0.002, color=C["n1"], ls="--", lw=1.2, label="random start's filed gap line 0.002")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("unmasked KL to the unchanged model, nats (log)")
    axes[1].set_ylabel("size: largest ‖δW‖ / ‖W₀‖")
    axes[1].set_ylim(-0.001, 0.026)
    for ax in axes:
        ax.set_xticks(range(3), ["start", "step 100", "step 1,000"])
        ax.set_xlim(-0.15, 2.35)
        ax.legend(frameon=False, fontsize=8, loc="center left" if ax is axes[1] else "upper right")
        tidy(ax)
    fig.suptitle("With the base as teacher, do the outputs come back, and does the size?", x=0.07, y=1.0, ha="left")
    fig.tight_layout()
    save(fig, "amend8_return")


def seeds_fig(seeds: dict[str, Any]) -> None:
    st, starts = seeds["states"], ("h0", "h1", "h2")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    lines(ax, seeds["W0"], False)
    for key, name in (("null", "null_1000"), ("real", "real"), ("n1", "random_1000")):
        y = [st[name]["ratio"][k] for k in starts]
        ax.plot(range(3), y, color=C[key], lw=2, marker="o", ms=7, label=f"{NAME[key]}: mean {st[name]['mean']:.2f}, size {st[name]['size']:.3f}")
        for i, v in enumerate(y):
            ax.annotate(f"{v:.2f}", (i, v), xytext=(0, -15 if key == "n1" else 8), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), ["adversary start 0", "adversary start 1", "adversary start 2"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(0.76, 2.35)
    ax.set_ylabel("adversarial ratio at that start")
    ax.set_title("At the same adversary start, does the null student read above the others?", loc="left")
    ax.text(0.01, 0.02, f"Positive control, off the axis: true labels, Adam, {st['true_adam']['mean']:.2f} at start 0.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncols=2)
    tidy(ax)
    save(fig, "amend8b_seeds")


def cells_fig(d: dict[str, dict[str, Any]], moved: dict[str, Any]) -> None:
    n1, n3, arm = d["n1"]["reads"]["1000"], d["n3b"]["reads"]["1000"], moved["reads"]["1000"]
    gap = {"null": (n3["teacher_gap"], arm["teacher_gap"]), "n1": (None, moved["control"]["1000"])}
    check = {"null": (n3["pgd"], arm["pgd"]), "n1": (None, n1["pgd"])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    labels = {"null": "null fine-tune as teacher", "n1": "unchanged model as teacher"}
    for ax, vals, fmt in ((axes[0], gap, "{:.4f}"), (axes[1], check, "{:.2f}")):
        for j, key in enumerate(("n1", "null")):
            for i, v in enumerate(vals[key]):
                x = i + (j - 0.5) * 0.38
                if v is None:
                    ax.annotate("zero by\nconstruction", (x, 0), xytext=(-8, 4), textcoords="offset points", ha="center", fontsize=8, color=MUTED, zorder=5) if ax is axes[0] else ax.annotate("not\nmoved", (x, 0), xytext=(-8, 4), textcoords="offset points", ha="center", fontsize=8, color=MUTED, zorder=5)
                    continue
                ax.bar(x, v, width=0.34, color=C[key], label=labels[key] if i == 1 else None, zorder=3)
                ax.annotate(fmt.format(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["start: base", "start: random move"])
        ax.set_xlim(-0.55, 1.4)
        tidy(ax)
    axes[0].set_ylabel("held-out KL to its own teacher at step 1,000, nats")
    axes[1].set_ylabel("adversarial ratio at step 1,000, one start")
    axes[1].axhline(1.241, color=PASS_C, ls="--", lw=1.2, label="pass line 1.24")
    axes[1].axhline(1.813, color=FAIL_C, ls="--", lw=1.2, label="fail line 1.81")
    axes[1].axhline(1, color=INK, lw=1.2, label="unchanged model 1")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[1].set_ylim(0, 2.45)
    axes[0].set_ylim(0, 0.056)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2)
    fig.suptitle("Do the student's gap and its check go with its start or with its teacher?", x=0.07, y=1.0, ha="left")
    fig.tight_layout()
    save(fig, "amend8b_cells")


def cosines_fig(c: dict[str, Any]) -> None:
    cos = c["cos"]
    rows = [("null students, two batch orders", cos["null_o0 | null_o1"], C["null"]), ("real students, two batch orders", cos["real_o0 | real_o1"], C["real"]),
            ("null student, order 1, and its own teacher", cos["null_o1 | null_teacher"], C["null"]), ("real student, order 1, and its own teacher", cos["real_o1 | real_teacher"], C["real"]),
            ("real and null students, different orders (two pairs, mean)", (cos["real_o0 | null_o1"] + cos["real_o1 | null_o0"]) / 2, C["ref"]),
            ("the two teachers", cos["real_teacher | null_teacher"], C["ref"]), ("null student, order 1, and the real teacher", cos["null_o1 | real_teacher"], C["null"]),
            ("real student, order 1, and the null teacher", cos["real_o1 | null_teacher"], C["real"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, v, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=col, height=0.55, zorder=3)
        ax.annotate(f"{v:.3f}", (v, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.axvline(0.1, color=MUTED, ls="--", lw=1.2, label="filed line 0.1 for 'along', random directions about 0.0002")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("cosine of the weight changes over all 24 matrices, step 1,000")
    ax.set_title("Which way does a student move: its batch order, its teacher, or neither?", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "amend8c_cosines")


def main() -> None:
    d = {s: load(s) for s in STAGES}
    moved = load("moved", "amend8b_")
    d["moved"] = moved
    m0 = json.loads((RESULTS / "teacher_training_e42_amend1_rand32.json").read_text())["M0"]["pgd"]
    check_fig(d, m0)
    damage_fig(d)
    size_fig(d)
    direction_fig(d)
    gap_fig(d)
    delta_out_fig(d)
    return_fig(d)
    seeds_fig(load("seeds", "amend8b_"))
    cells_fig(d, moved)
    cosines_fig(json.loads((RESULTS / "teacher_training_amend8c.json").read_text()))


if __name__ == "__main__":
    main()
