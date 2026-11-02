"""Figures for E12 Amendment 2: why the re-switching term falls on rand32b's students, against rand32's.

  amend2_fall.png       the fall per state, at two batch orders and on two forwards, against the filed lines
  amend2_sets.png       overlap of a student's switched-on set with its fine-tune's, against the two anchors
  amend2_blocks.png     nats given back by each group of subcomponents, against the fall
  amend2_ladder.png     share given back by the most changed k, against the same ladder from W0 to the fine-tune
  amend2_outside.png    how far values move in the outside block, per subcomponent mean and per window, against the 0.3 line
  amend2_lowering.png   summed importance values of the outside block, fine-tune against student
  amend2_size.png       share of the term lost against student/teacher size, on the largest matrix and on all matrices
  amend2_alignment.png  cosine of a student's change with its teacher's, against the other seed's and the teachers' own

Reads results/new_vs_existing_amend2.json. Writes results/amend2_summary.json from the gitignored trigger values and E46's fine-tunes,
and reuses that file when it exists, so the figures redraw without the large files. No model, no new runs.
Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/new_vs_existing/plot_amend2.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import matplotlib.pyplot as plt
from plot import FIGURES, RESULTS

J = json.loads((RESULTS / "new_vs_existing_amend2.json").read_text())
ST, FT, AN = J["students"], J["fine_tunes"], J["anchors"]
SUMMARY = RESULTS / "amend2_summary.json"
E46 = RESULTS.parents[1] / "regating_table" / "results" / "large"
ALIVE = RESULTS.parents[1] / "transport_test" / "results" / "large" / "alive.pt"
INK, MUTED, GRID, FALLING, REF, CONTROL = "#0b0b0b", "#52514e", "#d9d8d4", "#d9650a", "#5b48b5", "#3f3f3c"
TEACHERS = ("rand32b_s0", "rand32b_s1", "rand32_s0", "rand32_s1")
STATES = ("rand32b_s0", "rand32b_s1", "rand32b_s0_o1", "rand32b_s1_o1", "rand32_s0", "rand32_s1")
LABEL = {"rand32b_s0": "table 2\nseed 0", "rand32b_s1": "table 2\nseed 1", "rand32b_s0_o1": "table 2\nseed 0\nsecond order", "rand32b_s1_o1": "table 2\nseed 1\nsecond order",
         "rand32_s0": "table 1\nseed 0", "rand32_s1": "table 1\nseed 1"}
# The teachers' relative norm of the most changed matrix: the checkpoint lines of E46's full.log, as plot_amend1.py's SIZE.
LARGEST = {"rand32_s0": 0.0151, "rand32_s1": 0.0153, "rand32b_s0": 0.0254, "rand32b_s1": 0.0260}
EDGES = [i / 40 for i in range(41)]  # bins of a change in an importance value, 0 to 1


def colour(n: str) -> str:
    return FALLING if n.startswith("rand32b") else REF


def trigger_summary(name: str, alive: dict[str, Any]) -> dict[str, Any]:
    """One state's values at the trigger: how far they move, in the outside block, per subcomponent mean and per window."""
    import torch

    f = torch.load(RESULTS / "large" / "amend2" / f"trigger_{name}.pt", weights_only=False, map_location="cpu")
    st = ST[name]
    sets = {tuple(x) for x in st["S_on"]} | {tuple(x) for x in FT[st["fine_tune"]]["S_on"]}
    low, high, outside = [], [], []
    for k, m in alive.items():
        low.append(f["student"][k].float()[:, m])
        high.append(f["fine_tune"][k].float()[:, m])
        outside.append(torch.tensor([(k, int(i)) not in sets for i in m.nonzero().flatten()]))
    g, h, o = torch.cat(low, 1), torch.cat(high, 1), torch.cat(outside)
    d = h[:, o] - g[:, o]
    mean_d, abs_all = d.mean(0), (h - g).abs().mean(0)
    order = abs_all.sort(descending=True).values.cumsum(0) / abs_all.sum()
    on_mean = h[:, o].mean(0) > 0.3
    hist = lambda x: torch.histc(x, bins=len(EDGES) - 1, min=0.0, max=1.0).tolist()
    return {"n_outside": int(o.sum()), "n_alive": int(o.numel()), "sum_fine_tune": float(h[:, o].mean(0).sum()), "sum_student": float(g[:, o].mean(0).sum()),
            "up": float(d.clamp(min=0).mean(0).sum()), "down": float((-d).clamp(min=0).mean(0).sum()), "largest_mean_change": float(mean_d.abs().max()),
            "n_mean_over_0.05": int((mean_d.abs() > 0.05).sum()), "n_mean_over_0.3": int((mean_d.abs() > 0.3).sum()),
            "entries_over_0.3": int((d.abs() > 0.3).sum()), "entries": d.numel(), "mass_in_entries_over_0.3": float(d.abs()[d.abs() > 0.3].sum() / d.abs().sum()),
            "subcomponents_with_an_entry_over_0.3": int((d.abs() > 0.3).any(0).sum()), "n_fine_tune_mean_over_0.3": int(on_mean.sum()),
            "net_change_of_those": float(mean_d[on_mean].sum()), "outside_share_of_all_change": float(d.abs().mean(0).sum() / abs_all.sum()),
            "change_share_at_k": {str(k): float(order[k - 1]) for k in (*J["config"]["ladder"], 1000, 3000)},
            "hist_mean": hist(mean_d.abs()), "hist_window": hist(d.abs().flatten())}


def teacher_summary() -> dict[str, Any]:
    """The fine-tunes' own changes, fp64, over all 24 matrices: the size P5's numbers imply, read directly, and their cosines with each other."""
    import torch

    t = {n: torch.load(E46 / f"{n}.pt", weights_only=False, map_location="cpu") for n in TEACHERS}
    size = lambda d: math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))
    cos = lambda a, b: sum(float((a[k].double() * b[k].double()).sum()) for k in a) / (size(a) * size(b))
    return {"size": {n: size(d) for n, d in t.items()}, "largest_matrix": {n: max(float(v.double().norm()) for v in d.values()) for n, d in t.items()},
            "cos_seeds": {"rand32b": cos(t["rand32b_s0"], t["rand32b_s1"]), "rand32": cos(t["rand32_s0"], t["rand32_s1"])},
            "cos_tables": [cos(t[a], t[b]) for a in TEACHERS[:2] for b in TEACHERS[2:]]}


def summary() -> dict[str, Any]:
    if SUMMARY.is_file():
        return json.loads(SUMMARY.read_text())
    import torch

    alive = {k: v.bool() for k, v in torch.load(ALIVE, weights_only=False, map_location="cpu")["alive"].items()}
    out = {"teachers": teacher_summary(), "states": {n: trigger_summary(n, alive) for n in STATES}, "edges": EDGES}
    SUMMARY.write_text(json.dumps(out, indent=1))
    return out


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def fall() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, n in enumerate(STATES):
        ax.scatter(i, ST[n]["fall"], s=75, color=colour(n), zorder=3)
        ax.scatter(i, FT[ST[n]["fine_tune"]]["L01"] - ST[n]["L01_hook"], s=190, facecolor="none", edgecolor=INK, linewidth=1, zorder=4)
    ax.scatter([], [], s=75, color=FALLING, label="fall, map table 2")
    ax.scatter([], [], s=75, color=REF, label="fall, map table 1, the reference")
    ax.scatter([], [], s=190, facecolor="none", edgecolor=INK, linewidth=1, label="same, student read on the forward it trained on")
    for y, t in ((0.5, "0.5: at or above on both second-order students, the fall holds"), (0.15, "0.15: within it on both, the fall is gone")):
        ax.axhline(y, color=MUTED, lw=0.9, ls="--")
        ax.text(5.45, y + 0.02, t, fontsize=8.5, color=MUTED, ha="right")
    ax.axhline(0, color=INK, lw=0.9)
    ax.text(-0.45, 0.02, "0: do-nothing, by code", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(STATES)), [LABEL[n] for n in STATES], fontsize=8.5)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.05, 1.4)
    ax.set_ylabel("fall in masked log-odds at the trigger, nats\nfine-tune's values minus student's")
    ax.set_title("Does the fall survive a second batch order and a second forward? Yes, on both", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=1, bbox_to_anchor=(0.72, 1.0))
    style(ax)
    save(fig, "amend2_fall.png")


def sets() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    names = ("rand32b_s0", "rand32b_s1", "rand32_s0", "rand32_s1")
    for i, n in enumerate(names):
        ax.scatter(i, ST[n]["J"], s=75, color=colour(n), zorder=3)
        ax.text(i + 0.1, ST[n]["J"], f"set size: student {ST[n]['n_on']}, fine-tune {FT[n]['n_on']}", fontsize=8.5, color=MUTED, va="center")
    lines = ((1, INK, "-", "1: do-nothing, by code"), (AN["J rand32b seeds"], FALLING, "--", "map table 2's two fine-tunes"),
             (AN["J rand32 seeds"], REF, "--", "map table 1's two fine-tunes"), (AN["J rand32b against rand32, largest"], MUTED, ":", "the two tables' fine-tunes, largest"))
    for y, c, ls, t in lines:
        ax.axhline(y, color=c, lw=0.9, ls=ls)
        ax.text(3.75, y + 0.012, t, fontsize=8.5, color=c if c != FALLING else MUTED, ha="right")
    ax.set_xticks(range(4), [LABEL[n] for n in names], fontsize=8.5)
    ax.set_xlim(-0.4, 3.8)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Jaccard overlap of switched-on sets,\nstudent with its fine-tune")
    ax.set_title("Do students switch on their fine-tunes' subcomponents? Partly on table 2, more on table 1", fontsize=10.5, loc="left")
    style(ax)
    save(fig, "amend2_sets.png")


def blocks() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.27
    for i, n in enumerate(STATES):
        r = ST[n]["restored"]
        ax.bar(i - w, r["dropped"] + r["kept"] + r["added"], w * 0.92, color=GRID, edgecolor=MUTED, lw=0.6)
        ax.bar(i, r["outside"], w * 0.92, color=colour(n))
        ax.bar(i + w, r["at_trigger"], w * 0.92, color=CONTROL)
        ax.plot([i - 1.6 * w, i + 1.6 * w], [ST[n]["fall"]] * 2, color=INK, lw=2)
    ax.bar(0, 0, color=GRID, edgecolor=MUTED, lw=0.6, label="every subcomponent in either switched-on set, 22 to 54 of them")
    ax.bar(0, 0, color=FALLING, label="outside block, table 2: all other alive subcomponents, about 9,850")
    ax.bar(0, 0, color=REF, label="outside block, table 1, the reference")
    ax.bar(0, 0, color=CONTROL, label="every subcomponent at the trigger")
    ax.plot([], [], color=INK, lw=2, label="the fall: what writing back at every position gives")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(STATES)), [LABEL[n] for n in STATES], fontsize=8.5)
    ax.set_ylim(-0.05, 1.75)
    ax.set_ylabel("nats given back, writing the fine-tune's\nvalues into the student's at the trigger")
    ax.set_title("Which values give the fall back? Those outside the switched-on sets", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    save(fig, "amend2_blocks.png")


def ladder(s: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ks = J["config"]["ladder"]
    for n, ls in (("rand32b_s0", "-"), ("rand32b_s1", "-")):
        ax.plot(ks, [v / ST[n]["fall"] for v in ST[n]["restored"]["ladder"]], color=FALLING, marker="o", lw=2, ls=ls, label="table 2 student, share of its fall" if n.endswith("s0") else None)
        ax.plot(ks, [s["states"][n]["change_share_at_k"][str(k)] for k in ks], color=FALLING, lw=1, ls=":", label="table 2 student, share of all absolute change in those k" if n.endswith("s0") else None)
        c = J["ladder_control"][n]
        ax.plot(ks, [v / c["term"] for v in c["ladder"]], color=CONTROL, marker="s", lw=2, ls="--", label="control: W0 to the fine-tune, share of the whole term" if n.endswith("s0") else None)
    for n in ("rand32_s0", "rand32_s1"):
        ax.plot(ks, [v / ST[n]["fall"] for v in ST[n]["restored"]["ladder"]], color=REF, marker="o", lw=1.2, alpha=0.8, label="table 1 student, share of a fall under 0.1 nats" if n.endswith("s0") else None)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    ax.text(1, 0.515, "half: reached by k of 10 or fewer reads few, 30 middling, else diffuse", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(-0.12, 0.62)
    ax.set_xlabel("k, the subcomponents whose values at the trigger changed most, written back together")
    ax.set_ylabel("share given back")
    ax.set_title("How many subcomponents give the fall back? More than 300, and the control reads the same", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.72))
    style(ax)
    save(fig, "amend2_ladder.png")


def outside(s: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, sharex=True)
    mids = [(a + b) / 2 for a, b in zip(s["edges"][:-1], s["edges"][1:], strict=True)]
    for ax, key, name in ((axes[0], "hist_mean", "mean over the 97 windows, per subcomponent"), (axes[1], "hist_window", "in one window, per subcomponent")):
        for n, c, t in (("rand32b_s0", FALLING, "table 2, seed 0"), ("rand32_s0", REF, "table 1, seed 0")):
            ax.step(mids, s["states"][n][key], where="mid", color=c, lw=1.8, label=t)
        ax.axvline(0.3, color=MUTED, lw=0.9, ls="--")
        ax.text(0.31, 3e5, "0.3: the rise over W0's mean\nthat the switched-on rule asks for", fontsize=8.5, color=MUTED, va="top")
        ax.set_yscale("log")
        ax.set_xlabel(f"fine-tune against student, size of change:\n{name}", fontsize=9)
        style(ax)
    axes[0].set_ylabel("count, outside block")
    axes[0].set_ylim(0.7, 2e6)
    axes[0].set_xlim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Can a rule on the mean over windows see what moves? No: the moves are large only in single windows", fontsize=10.5, x=0.01, ha="left")
    save(fig, "amend2_outside.png")


def lowering(s: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for i, n in enumerate(STATES):
        r = s["states"][n]
        ax.bar(i - 0.17, r["up"], 0.32, color=colour(n))
        ax.bar(i + 0.17, -r["down"], 0.32, color=colour(n), alpha=0.45)
        ax.plot([i - 0.36, i + 0.36], [r["up"] - r["down"]] * 2, color=INK, lw=2)
    ax.bar(0, 0, color=FALLING, label="fine-tune's value above the student's, table 2")
    ax.bar(0, 0, color=FALLING, alpha=0.45, label="fine-tune's value below the student's, table 2")
    ax.bar(0, 0, color=REF, label="same, above, table 1, the reference")
    ax.bar(0, 0, color=REF, alpha=0.45, label="same, below, table 1")
    ax.plot([], [], color=INK, lw=2, label="net")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(STATES)), [LABEL[n] for n in STATES], fontsize=8.5)
    ax.set_ylim(-34, 88)
    ax.set_ylabel("change in importance values, summed over\nthe outside block, mean over windows")
    ax.set_title("Is the outside block lowered? On table 2 it is, net, with large moves both ways", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    style(ax)
    save(fig, "amend2_lowering.png")


def size_fig() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for n in STATES:
        t, lost = ST[n]["fine_tune"], ST[n]["term_lost"]
        hollow = {"facecolor": "white", "edgecolor": colour(n), "linewidth": 1.8} if n.endswith("_o1") else {"color": colour(n)}
        largest = next(c for c in ST[n]["curve"] if c["step"] == 1000)["rel_norm_max"] / LARGEST[t]
        axes[0].scatter(largest, lost, s=70, zorder=3, **hollow)
        axes[1].scatter(ST[n]["size_ratio"], lost, s=70, zorder=3, **hollow)
    axes[0].scatter([], [], s=70, color=FALLING, label="table 2")
    axes[0].scatter([], [], s=70, color=REF, label="table 1, the reference")
    axes[0].scatter([], [], s=70, facecolor="white", edgecolor=FALLING, linewidth=1.8, label="table 2, second batch order")
    titles = ("most changed matrix in each, relative to W0's\n(Amendment 1's axis)", "all 24 matrices together, Frobenius\n(this run's size_ratio)")
    for ax, t in zip(axes, titles, strict=True):
        ax.axvline(1, color=MUTED, lw=0.9, ls="--")
        ax.text(1.02, 0.235, "equal size", fontsize=8.5, color=MUTED)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xlim(0.6, 2.5)
        ax.set_ylim(-0.01, 0.25)
        ax.set_xlabel(f"student's size / teacher's: {t}", fontsize=9)
        style(ax)
    axes[0].text(1.7, 0.006, "0: do-nothing", fontsize=8.5, color=INK)
    axes[0].set_ylabel("share of the fine-tune's re-switching term lost")
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Do table 2's students end smaller than their teachers? On the most changed matrix only", fontsize=10.5, x=0.01, ha="left")
    save(fig, "amend2_size.png")


def alignment(s: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    t = s["teachers"]
    pos = {"rand32b_s0": 0, "rand32b_s1": 1, "rand32_s0": 3, "rand32_s1": 4}
    for n, x in pos.items():
        ax.scatter(x, ST[n]["cosine"], s=80, color=colour(n), zorder=3)
        ax.scatter(x, ST[n]["cosine_other_seed"], s=80, facecolor="white", edgecolor=colour(n), linewidth=1.8, zorder=3)
    ax.plot([-0.4, 1.4], [t["cos_seeds"]["rand32b"]] * 2, color=FALLING, lw=2)
    ax.plot([2.6, 4.4], [t["cos_seeds"]["rand32"]] * 2, color=REF, lw=2)
    ax.axhspan(min(t["cos_tables"]), max(t["cos_tables"]), color=GRID, alpha=0.7)
    ax.text(4.4, max(t["cos_tables"]) + 0.012, "teachers of different tables", fontsize=8.5, color=MUTED, ha="right")
    ax.axhline(1, color=INK, lw=0.9)
    ax.text(4.4, 0.97, "1: a student equal to its teacher, by code", fontsize=8.5, color=INK, ha="right", va="top")
    ax.scatter([], [], s=80, color=MUTED, label="student with its own teacher")
    ax.scatter([], [], s=80, facecolor="white", edgecolor=MUTED, linewidth=1.8, label="student with the other seed's teacher")
    ax.plot([], [], color=MUTED, lw=2, label="the table's two teachers with each other")
    ax.set_xticks(list(pos.values()), [LABEL[n] for n in pos], fontsize=8.5)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("cosine between weight changes from W0, all 24 matrices")
    ax.set_title("How close is a student to its teacher? Not close: the two teachers are closer to each other", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    style(ax)
    save(fig, "amend2_alignment.png")


if __name__ == "__main__":
    s = summary()
    fall()
    sets()
    blocks()
    ladder(s)
    outside(s)
    lowering(s)
    size_fig()
    alignment(s)
