"""Figures and report tables for E55 Stage C1b, read from results/c1b.json, c1b_cutoffs.json and c1.json only; no
model. Analysis: reports/analyses/2026-10-31_analysis_e55-c1b.md.

  c1b_readings.png     each instrument's reading per state, grouped by fine-tune recipe (lr, steps), task P left and
                       task N right in each group, calibration seeds hollow and scored seeds filled, with the cut-off
                       filed from the calibration seeds and each instrument's reference line
  c1b_scores.png       agreement with the truth on the 21 scored states and on one vote per run (10K saves): the three
                       scored instruments, the probe (task-identity reference, not scored), the three filed recipe-only
                       labellers at their hindsight ceilings, and, post hoc, update size with one cut-off per recipe cell
                       set on the calibration seeds; binomial bar and what each instrument needed
  c1b_update_size.png  update size ||W - W0|| / ||W0|| per state, same grouping: the nuisance the filed ceiling reads with
                       one threshold, and the within-recipe gap between tasks P and N

Also prints the margins to each cut-off, the per-state table, the post-hoc recipe-cell labeller and C1's re-labelled
states that disagree.

Usage (repo root): PYTHONPATH=experiments/pcfg_anchors ~/commute-agents/fleet/cpu-run .venv/bin/python \
    experiments/pcfg_anchors/plot_c1b.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from plot import INK, MUTED, finish, load, plt

from commute import revival

EX_C, NEW_C, CTL_C = (
    "#2e9e5b",
    "#2a78d6",
    "#eb6834",
)  # truth: task P (reuses), task N (new); instruments
TRUTH_C = {"existing": EX_C, "new": NEW_C}
SCORED = ("prune", "reverse", "steer")
NAMES = {
    "prune": "pruning revival",
    "reverse": "reverse fine-tuning",
    "steer": "steering",
    "probe": "possession probe",
}
CELLS = [(1e-4, 1000), (1e-4, 10000), (1e-5, 1000), (1e-5, 10000)]
CELL_W = [
    "lr 1e-4\n1K steps",
    "lr 1e-4\n10K steps",
    "lr 1e-5\n1K steps",
    "lr 1e-5\n10K steps",
]


def cell_of(meta: dict[str, Any]) -> tuple[float, int]:
    return (meta["lr"], meta["step"])


def xpos(meta: dict[str, Any], i: int) -> float:
    """Recipe group, task P left of task N, seeds spread within."""
    return (
        CELLS.index(cell_of(meta))
        + (-0.2 if meta["task"] == "P" else 0.2)
        + (i - 2) * 0.045
    )


def states(c: dict[str, Any], part: str) -> list[tuple[str, dict[str, Any]]]:
    return sorted(c[part].items(), key=lambda kv: (c["meta"][kv[0]]["seed"], kv[0]))


def fig_readings(
    c: dict[str, Any], cut: dict[str, Any], out: str = "c1b_readings.png"
) -> None:
    refs = {"prune": (0.0, "random K-unit removal (0)"), "reverse": (1.0, "scratch, NEVER (1.0 = 2,300 steps)"),
            "steer": (float(np.mean([v["steer"]["S"] for v in c["controls"]["ID"].values()])), "positive control: identity-prompt fine-tunes")}  # fmt: skip
    ylab = {"prune": "count-a accuracy revived over random removal", "reverse": "steps to 0.95 count a / scratch's",
            "steer": "gain share S of one mean shift"}  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, ins in zip(axes, SCORED, strict=True):
        for part, filled in (("calibration", False), ("scored", True)):
            for sid, r in states(c, part):
                m = c["meta"][sid]
                col = TRUTH_C[r["truth"]]
                ax.scatter(
                    xpos(m, m["seed"]),
                    r["x"][ins],
                    s=34,
                    color=col if filled else "white",
                    edgecolor=col,
                    lw=1.3,
                    zorder=3,
                )
        ax.axhline(cut[ins]["c"], color="#c0392b", lw=1.6)
        ax.text(
            3.45,
            cut[ins]["c"],
            f"cut-off {cut[ins]['c']:.3f}\n(set on seeds 0-1)",
            color="#c0392b",
            fontsize=7,
            va="center",
            ha="right",
        )
        y, words = refs[ins]
        ax.axhline(y, color=MUTED, ls="--", lw=1)
        ax.text(1.5, y - 0.004, words, color=MUTED, fontsize=7, va="top", ha="center")
        ax.set_xticks(range(4), CELL_W, fontsize=8)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylabel(ylab[ins], fontsize=8.5)
        ax.set_title(
            NAMES[ins]
            + (" (low = reuses)" if ins == "reverse" else " (high = reuses)"),
            fontsize=9.5,
            loc="left",
        )
    axes[2].set_ylim(0.4, 1.0)
    axes[0].text(
        2.0,
        0.45,
        "no task N state learned\nby 1K steps at lr 1e-5",
        fontsize=7,
        color=MUTED,
        ha="center",
    )
    handles = [Line2D([], [], marker="o", ls="", color=EX_C, label="task P, count a + 1: reuses (truth)"),
               Line2D([], [], marker="o", ls="", color=NEW_C, label="task N, count b: new (truth)"),
               Line2D([], [], marker="o", ls="", color=INK, mfc="white", label="calibration seeds 0-1"),
               Line2D([], [], marker="o", ls="", color=INK, label="scored seeds 2-4")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    finish(
        fig,
        out,
        "Every scored state falls on its own class's side of the cut-off set on other seeds, for all three instruments",
        bottom=0.2,
    )


def cell_labeller(c: dict[str, Any], scored: list[str]) -> dict[str, str]:
    """Post hoc: update size with one cut-off per recipe cell, set on the calibration seeds at the midpoint of their gap (small =
    existing); a cell with one class in calibration labels every scored state that class."""
    cal = c["calibration"]
    lab: dict[str, str] = {}
    for cell in CELLS:
        cs = [s for s in cal if cell_of(c["meta"][s]) == cell]
        ss = [s for s in scored if cell_of(c["meta"][s]) == cell]
        classes = {cal[s]["truth"] for s in cs}
        if len(classes) == 1:
            only: str = classes.pop()
            lab |= dict.fromkeys(ss, only)
            continue
        ex = [
            c["nuisance"][s]["update_size"] for s in cs if cal[s]["truth"] == "existing"
        ]
        nw = [c["nuisance"][s]["update_size"] for s in cs if cal[s]["truth"] == "new"]
        assert max(ex) < min(nw), cell
        t = (max(ex) + min(nw)) / 2
        lab |= {
            s: "existing" if c["nuisance"][s]["update_size"] <= t else "new" for s in ss
        }
    return lab


def agreement(lab: dict[str, str], truth: dict[str, str]) -> int:
    return sum(lab[s] == truth[s] for s in truth)


def fig_scores(
    c: dict[str, Any], cell_lab: dict[str, str], out: str = "c1b_scores.png"
) -> None:
    sc = c["scores"]
    truth_all = {s: r["truth"] for s, r in c["scored"].items()}
    truth_run = {s: t for s, t in truth_all.items() if c["meta"][s]["step"] == 10000}
    rows = [(NAMES[i], i) for i in (*SCORED, "probe")]
    rows += [
        ("recipe: lr", "lr"),
        ("recipe: steps", "steps"),
        ("recipe: update size", "update_size"),
    ]
    rows.append(("post hoc: update size,\none cut-off per recipe cell", "cell"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, which, truth, words in zip(axes, ("all", "per_run"), (truth_all, truth_run),
                                       ("21 scored states (margin 2)", "one vote per run, its 10K save (margin 1)"), strict=True):  # fmt: skip
        s0 = sc["prune"][which]
        for i, (name, key) in enumerate(rows):
            if key in sc:
                agree, n = sc[key][which]["agree"], sc[key][which]["n"]
            elif key == "cell":
                agree, n = agreement(cell_lab, truth), len(truth)
            else:
                agree, n = s0["nuisance_agree"][key], s0["n"]
            colour = CTL_C if key in SCORED else MUTED
            ax.barh(i, agree / n, color=colour, alpha=0.85 if key in SCORED else 0.45, height=0.6,
                    hatch="//" if key in ("probe", "cell") else None, edgecolor="white")  # fmt: skip
            ax.text(agree / n + 0.01, i, f"{agree}/{n}", va="center", fontsize=8)
        need = max(s0["bar"], s0["n_c"] + s0["margin"])
        ax.axvline(
            s0["bar"] / s0["n"],
            color=INK,
            lw=1.4,
            label="binomial bar" if which == "all" else None,
        )
        ax.axvline(need / s0["n"], color="#c0392b", lw=2, ls="--",
                   label="needed: best recipe labeller + margin" if which == "all" else None)  # fmt: skip
        ax.axvline(0.5, color=MUTED, ls=":", lw=0.9)
        ax.set_xlim(0.4, 1.08)
        ax.set_title(words, fontsize=9.5, loc="left")
        ax.set_xlabel("share of states whose label agrees with the truth")
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False)
    finish(
        fig,
        out,
        "All three instruments label every scored state right; so does update size once it is read within a recipe",
        bottom=0.17,
    )


def fig_update_size(c: dict[str, Any], out: str = "c1b_update_size.png") -> None:
    t = c["scores"]["prune"]["all"]["ceiling"]["update_size"]["threshold"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for part, filled in (("calibration", False), ("scored", True)):
        for sid, r in states(c, part):
            m, col = c["meta"][sid], TRUTH_C[r["truth"]]
            ax.scatter(
                xpos(m, m["seed"]),
                c["nuisance"][sid]["update_size"],
                s=34,
                color=col if filled else "white",
                edgecolor=col,
                lw=1.3,
                zorder=3,
            )
    ax.axhline(t, color=MUTED, ls="--", lw=1)
    ax.text(
        3.45,
        t,
        "best single threshold: 15 of 21",
        fontsize=7.5,
        color=MUTED,
        ha="right",
        va="bottom",
    )
    ax.set_yscale("log")
    ax.set_xticks(range(4), CELL_W, fontsize=8)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("update size ||W - W0|| / ||W0||")
    handles = [Line2D([], [], marker="o", ls="", color=EX_C, label="task P: reuses"),
               Line2D([], [], marker="o", ls="", color=NEW_C, label="task N: new"),
               Line2D([], [], marker="o", ls="", color=INK, mfc="white", label="calibration seeds"),
               Line2D([], [], marker="o", ls="", color=INK, label="scored seeds")]  # fmt: skip
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    finish(
        fig,
        out,
        "Within each recipe, learning the new count moves the weights 1.15 to 1.45 times as far as adding one",
    )


def print_margins(c: dict[str, Any], cut: dict[str, Any]) -> None:
    print(
        "| instrument | cut-off | reuses side, nearest | new side, nearest | calibration gap |"
    )
    print("|---|---|---|---|---|")
    for ins in (*SCORED, "probe"):
        hi = cut[ins]["high_is_existing"]
        sc = c["scored"]
        ex = [r["x"][ins] for r in sc.values() if r["truth"] == "existing"]
        nw = [r["x"][ins] for r in sc.values() if r["truth"] == "new"]
        ce = [
            v for s, v in cut["readings"][ins].items() if cut["truth"][s] == "existing"
        ]
        cn = [v for s, v in cut["readings"][ins].items() if cut["truth"][s] == "new"]
        near_ex, near_new = (min(ex), max(nw)) if hi else (max(ex), min(nw))
        gap = (max(cn), min(ce)) if hi else (max(ce), min(cn))
        print(f"| {ins} | {cut[ins]['c']:.4f} | {near_ex:.4f} (range {min(ex):.4f}-{max(ex):.4f}) | {near_new:.4f} "
              f"(range {min(nw):.4f}-{max(nw):.4f}) | {gap[0]:.4f}-{gap[1]:.4f} |")  # fmt: skip


def print_table(c: dict[str, Any]) -> None:
    print(
        "| state | truth | update size | count a before | pruning gain | reverse steps | S (block) | labels |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for sid, r in c["scored"].items():
        lab = "".join(
            {"existing": "E", "new": "N", "void": "v"}[r["labels"][i]]
            for i in (*SCORED, "probe")
        )
        print(f"| {sid.removeprefix('c1b/')} | {r['truth']} | {c['nuisance'][sid]['update_size']:.4f} | {r['prune']['acc0']:.4f} | "
              f"{r['x']['prune']:.3f} | {r['reverse']['steps']:,} | {r['steer']['S']:.3f} ({r['steer']['block']}) | {lab} |")  # fmt: skip


def print_c1(c: dict[str, Any]) -> None:
    c1 = load("c1.json")
    truth = c1["truth"]["scored"]
    for sid, v in c["c1_described"]["states"].items():
        for ins in SCORED:
            if sid in truth and v["labels"][ins] not in ("void", truth[sid]):
                print(
                    f"C1 re-labelled miss: {sid} {ins} x={v['x'][ins]:.4f} label {v['labels'][ins]} truth {truth[sid]}"
                )
    print("C1 re-labelled agreement:", c["c1_described"]["agreement"])


def main() -> None:
    c, cut = load("c1b.json"), load("c1b_cutoffs.json")
    cut = cut["cutoffs"] | {"readings": cut["readings"], "truth": cut["truth"]}
    scored = list(c["scored"])
    cell_lab = cell_labeller(c, scored)
    truth = {s: c["scored"][s]["truth"] for s in scored}
    run_truth = {s: t for s, t in truth.items() if c["meta"][s]["step"] == 10000}
    print("post hoc update size per recipe cell, calibrated:", agreement(cell_lab, truth), "of", len(truth),
          "; per run", agreement(cell_lab, run_truth), "of", len(run_truth))  # fmt: skip
    for cell in CELLS:
        for task in ("P", "N"):
            v = [c["nuisance"][s]["update_size"] for s in [*c["calibration"], *scored]
                 if cell_of(c["meta"][s]) == cell and c["meta"][s]["task"] == task]  # fmt: skip
            if v:
                print(
                    f"update size {cell} {task}: {min(v):.4f}-{max(v):.4f} (n {len(v)})"
                )
    print(
        "probe p-value, binomial check:",
        revival.score({s: c["scored"][s]["labels"]["probe"] for s in scored}, truth)[
            "p_value"
        ],
    )
    print_margins(c, cut)
    print_table(c)
    print_c1(c)
    fig_readings(c, cut)
    fig_scores(c, cell_lab)
    fig_update_size(c)


if __name__ == "__main__":
    main()
