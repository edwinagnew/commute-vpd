"""E101's figures from the agent outputs, max_scores.json and large/key.json (through score.py's neuron table), no model.
Built to reuse (by assumption) is blue, built to be new (by assumption) is orange, open rows grey, W0 controls green,
as in E98's figures.

  figures/related_share.png   per set, the share of neurons whose job wins its test and that the judge ties to the set's
                              task, with the random floor for that task and the floor + 0.2 a pass needed
  figures/strong_rates.png    how often the jobs' examples and near misses fire their neuron strongly, random, carrying
                              and W0-control neurons beside the essay's Qwen3-32B numbers
  figures/amend1_controls.png Amendment 1: per control task, the share the judge ties to the task for the carrying set
                              and W0's two kinds of control, with the random floor, under the filed and amended judges
  figures/explained_share.png per group, the share of jobs that win their test, carrying beside its random partner, with
                              the pooled random base rate and the essay's 164 of 183

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/carrying_jobs/plot.py
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from commute.jobs import load_experiment

SC = load_experiment("carrying_jobs", "score")
FIG = SC.HERE / "figures"
BLUE, ORANGE, GREY, GREEN, INK, GRID = "#2a78d6", "#eb6834", "#7a7a75", "#1baf7a", "#52514e", "#e4e3df"
ESSAY = {"examples": 0.76, "near_misses": 0.11, "win": 164 / 183}  # Qwen3-32B, 183 neurons, the essay's lose test
# group, words on the axis, colour. Reuse/new pairs adjacent, as in the plan's P2 rows.
GROUPS = [
    ("lowercase", "lowercase", BLUE), ("ciphers", "ciphers", ORANGE),
    ("same_word", "same-word map", BLUE), ("random_word", "random-word map", ORANGE),
    ("e62_copy", "copy", BLUE), ("e62_lookup", "lookup", ORANGE),
    ("rank1", "rank-1 map", GREY), ("copy_map", "copy map", GREY),
]
CONTROLS = [("lowercase", "W0 picks, lowercase"), ("e62_copy", "W0 picks, copy"), ("ident32", "W0 picks, same-word")]


def group_rows(T: dict[str, Any], group: str, role: str) -> list[dict[str, Any]]:
    return [t for t in T.values() if t["complete"] and any(r == {"group": group, "role": role} for r in t["roles"])]


def strong_rates(T: dict[str, Any], key: dict[str, Any], scores: dict[str, Any], role: str, frac: float) -> tuple[float, float]:
    ex = nm = n_ex = n_nm = 0
    for i, t in T.items():
        if i in scores and any(r["role"] == role for r in t["roles"]):
            top = key["neurons"][i]["top_act"]
            ex += sum(x >= frac * top for x in scores[i]["examples"])
            nm += sum(x >= frac * top for x in scores[i]["near_misses"])
            n_ex, n_nm = n_ex + len(scores[i]["examples"]), n_nm + len(scores[i]["near_misses"])
    return ex / n_ex, nm / n_nm


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def related_figure(T: dict[str, Any], P: dict[str, Any]) -> None:
    S, floors = P["settings"], {d: SC.side(T, None, "random", d, P["settings"])[0] for d in P["descriptions"]}
    rows = [(w, P["groups"][g]["description"], group_rows(T, g, "carrying"), c) for g, w, c in GROUPS]
    rows += [(w, P["controls_w0"][g]["description"], group_rows(T, g, "control"), GREEN) for g, w in CONTROLS]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for y, (words, desc, m, c) in enumerate(rows):
        k = sum(t["outcome"] == "win" and desc in t["ties"] for t in m)
        ax.barh(y, k / len(m), color=c, height=0.6)
        ax.text(max(k / len(m), floors[desc]) + 0.006, y, f"{k} of {len(m)}", va="center", fontsize=8, color=INK)
        ax.plot([floors[desc]] * 2, [y - 0.4, y + 0.4], color="black", lw=1.5)
        ax.plot([floors[desc] + S["min_gap"]] * 2, [y - 0.4, y + 0.4], color=INK, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("related share: job wins its test and the judge ties it to the set's task")
    ax.set_title("Almost no neuron's Pile job is tied to its task, W0's own picks included", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY, GREEN)]
    handles += [Line2D([], [], color="black", lw=1.5), Line2D([], [], color=INK, lw=1, ls="--")]
    ax.legend(handles, ["carrying, built to reuse", "carrying, built to be new", "carrying, open row",
                        "W0's top firers at the answer", "random-neuron floor for that task", "floor + 0.2, a pass's bar"],
              fontsize=7.5, loc="lower right", frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "related_share.png", dpi=150)


def strong_figure(T: dict[str, Any], P: dict[str, Any]) -> None:
    key, scores = SC.load(SC.LARGE / "key.json"), SC.load(SC.STAGE1 / "max_scores.json")["neurons"]
    sets = [("essay, Qwen3-32B\nrandom neurons", (ESSAY["examples"], ESSAY["near_misses"]))]
    sets += [(w, strong_rates(T, key, scores, r, P["settings"]["strong_frac"]))
             for r, w in (("random", "67M random"), ("carrying", "67M carrying"), ("control", "67M W0 picks"))]
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, (_, (e, n)) in enumerate(sets):
        ax.bar(x - 0.18, e, 0.34, color=BLUE)
        ax.bar(x + 0.18, n, 0.34, color=ORANGE)
        ax.text(x - 0.18, e + 0.01, f"{e:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(x + 0.18, n + 0.01, f"{n:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(sets)), [s[0] for s in sets], fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("share of texts that fire the neuron strongly\n(at least 30% of its strongest Pile window)")
    ax.set_title("The 67M's jobs drive their neurons, but so do their near misses", fontsize=10)
    ax.legend([Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)],
              ["examples written from the job", "near misses written to fail"], fontsize=8, frameon=False, ncol=2,
              loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "strong_rates.png", dpi=150)


def explained_figure(T: dict[str, Any], P: dict[str, Any]) -> None:
    base = SC.side(T, None, "random", None, P["settings"])[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for y, (g, words, c) in enumerate(GROUPS):
        for dy, role, col in ((-0.18, "carrying", c), (0.18, "random", "#c9c8c3")):
            m = group_rows(T, g, role)
            v = sum(t["outcome"] == "win" for t in m) / len(m)
            ax.barh(y + dy, v, 0.34, color=col)
            ax.text(v + 0.005, y + dy, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(base, color="black", lw=1.2)
    ax.axvline(ESSAY["win"], color=INK, lw=1, ls="--")
    ax.set_yticks(range(len(GROUPS)), [g[1] for g in GROUPS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("explained share: jobs whose examples fire strongly more often than their near misses")
    ax.set_title("Carrying neurons' jobs win their test no more often than random neurons' do", fontsize=10)
    ax.legend([Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY, "#c9c8c3")]
              + [Line2D([], [], color="black", lw=1.2), Line2D([], [], color=INK, lw=1, ls="--")],
              ["carrying, built to reuse", "carrying, built to be new", "carrying, open row", "its random partner",
               f"all random neurons, {base:.2f}", f"essay, Qwen3-32B, {ESSAY['win']:.2f}"],
              fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "explained_share.png", dpi=150)


# Amendment 1's three control tasks: words, its controls' group, its carrying group (colour as in GROUPS), description.
AMEND1_TASKS = [("lowercase", "lowercase", "lowercase", BLUE, "lower"),
                ("same-word map", "ident32", "same_word", BLUE, "copykey"),
                ("lookup", "e62_lookup", "e62_lookup", ORANGE, "lookup")]
DARK_GREEN = "#0b6b4a"


def tied_share(T: dict[str, Any], group: str | None, role: str, desc: str) -> tuple[int, int]:
    """(tied to desc, judged) over a group's neurons of that role: Amendment 1's C1a measure, ties only."""
    m = [t for t in T.values() if t["judged"] and any(r["role"] == role and group in (None, r["group"]) for r in t["roles"])]
    return sum(desc in t["ties"] for t in m), len(m)


def amend1_figure(T0: dict[str, Any], T1: dict[str, Any]) -> None:
    """Per control task, under the filed judge and the amended one: the share tied to the task's own description for the
    carrying set, W0's top firers (filed controls), W0's task-specific picks (Amendment 1's), and the random floor."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, T, title in ((axes[0], T0, "filed judge: no task windows,\npunctuation excluded"),
                         (axes[1], T1, "amended judge: sees each task's windows,\nno token kind excluded")):
        labels, ys, y = [], [], 0.0
        for words, ctrl, carry, col, desc in AMEND1_TASKS:
            k0, n0 = tied_share(T, None, "random", desc)
            rows = [("carrying set", tied_share(T, carry, "carrying", desc), col),
                    ("W0 top firers", tied_share(T, ctrl, "control", desc), GREEN),
                    ("W0 task-specific picks", tied_share(T, ctrl, "control_amend1", desc), DARK_GREEN)]
            for name, (k, n), c in rows:
                labels.append(f"{words}: {name}")
                ys.append(y)
                if n:
                    ax.barh(y, k / n, color=c, height=0.62)
                    ax.text(max(k / n, k0 / n0) + 0.006, y, f"{k} of {n}", va="center", fontsize=7.5, color=INK)
                else:
                    ax.text(0.006, y, "not in the filed run" if name.endswith("picks") else "none filed", va="center",
                            fontsize=7.5, color=GREY, style="italic")
                ax.plot([k0 / n0] * 2, [y - 0.4, y + 0.4], color="black", lw=1.5)
                y += 1
            y += 0.6
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 0.25)
        ax.set_xlabel("share of the set the judge ties to the task")
        ax.set_yticks(ys, labels)
        style(ax)
    axes[0].invert_yaxis()
    fig.suptitle("Shown each task's windows, the judge still ties none of W0's task-specific neurons to their task",
                 fontsize=10.5)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREEN, DARK_GREEN)]
    handles += [Line2D([], [], color="black", lw=1.5)]
    fig.legend(handles, ["carrying, built to reuse", "carrying, built to be new", "W0's top firers at the answer",
                         "W0's picks for this task over others", "random-neuron floor for that task"],
               fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "amend1_controls.png", dpi=150)


def main() -> None:
    P = SC.load(SC.RESULTS / "predictions.json")
    T, T1 = SC.filed_table(P), SC.amend1_table(P)
    FIG.mkdir(exist_ok=True)
    related_figure(T, P)
    strong_figure(T, P)
    explained_figure(T, P)
    amend1_figure(T, T1)
    print("figures in experiments/carrying_jobs/figures")


if __name__ == "__main__":
    main()
