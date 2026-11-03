"""E69's analysis numbers and figures from the results JSONs and the per-problem accuracy files; no model, no GPU.

Writes results/lr1e-6w10/analysis.json and figures/:
  gains.png      gain from W0 of RL, the null, SFT and RL minus the null, under the three window rules
  grader.png     RL minus the null under the strict and the lenient grader
  accounting.png where the gain sits: kept at both, left the window, came back, outside at both
  openings.png   the first two words of the greedy answers, per arm
  lengths.png    how long the greedy answers are, per arm, with the window
  reward.png     sampled reward on each step's prompts, minus W0's on the same prompts, with and without the format
  collapse.png   the null's sampled answers: answer line present and right under the lenient grader, against W0's
  kl.png         KL from W0 against |δW|, with the line KL would follow if it grew as the size squared
  delta_r.png    the early signal ΔR by step, with the probes and the three-times-the-null line

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_rl/plot.py
"""

from __future__ import annotations

import collections
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RUN = RES / "lr1e-6w10"
LARGE = RES / "large"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
COL = {"W0": "#52514e", "R": "#2a78d6", "N": "#eb6834", "F": "#1baf7a", "R-N": "#6250d6"}
NAME = {"W0": "unchanged model", "R": "RL", "N": "shuffled-reward null", "F": "own-samples SFT", "R-N": "RL minus null"}
SETS = {"gsm8k": "GSM8K test", "gsm1k": "GSM1K"}
ARMS = ("R", "N", "F")
RULES = {"drop": "drop rule\n(fits at both)", "whole": "whole answers\n(all problems)", "capped": "capped answers\n(all problems)"}
GRADER = {"strict": {"drop": "strict", "whole": "strict", "capped": "capped_strict"},
          "lenient": {"drop": "lenient", "whole": "lenient", "capped": "capped_lenient"}}
PHRASES = ("double-check", "let me check", "verify", "wait")
Rows = dict[int, dict[str, Any]]


def rows(arm: str, name: str) -> Rows:
    p = LARGE / "acc" / f"W0_{name}.jsonl" if arm == "W0" else LARGE / "lr1e-6w10" / "acc" / f"{arm}_{name}.jsonl"
    return {r["idx"]: r for r in map(json.loads, p.read_text().splitlines())}


def mean_se(d: list[float] | list[int]) -> dict[str, float]:
    return {"mean": 100 * st.mean(d), "se": 100 * st.stdev(d) / math.sqrt(len(d)), "n": len(d)}


def problems(d: dict[str, Rows], rule: str, arms: tuple[str, ...]) -> list[int]:
    """The drop rule keeps the problems that fit at every state compared; the other two rules keep all."""
    ids = sorted(d["W0"])
    return [i for i in ids if all(d[a][i]["fits"] for a in arms)] if rule == "drop" else ids


def paired(d: dict[str, Rows], a: str, b: str, rule: str, grader: str) -> dict[str, float]:
    key = GRADER[grader][rule]
    states = ("W0", a, b) if rule == "drop" else (a, b)
    return mean_se([int(d[a][i][key]) - int(d[b][i][key]) for i in problems(d, rule, states)])


def gains(d: dict[str, Rows]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for grader in GRADER:
        out[grader] = {a: {r: paired(d, a, "W0", r, grader) for r in RULES} for a in ARMS}
        out[grader]["R-N"] = {r: paired(d, "R", "N", r, grader) for r in RULES}
        out[grader]["F-N"] = {r: paired(d, "F", "N", r, grader) for r in RULES}
    return out


def accounting(d: dict[str, Rows], arm: str) -> dict[str, Any]:
    """Each problem falls in one class by whether its answer fits the window at W0 and at the arm. Net problems right,
    arm minus W0, per class and per grade. The whole-answer and capped grades agree on `kept` by construction."""
    w, t, out = d["W0"], d[arm], {}
    classes: dict[str, list[int]] = {"kept": [], "left": [], "came back": [], "outside at both": []}
    for i in w:
        c = {(1, 1): "kept", (1, 0): "left", (0, 1): "came back", (0, 0): "outside at both"}[int(w[i]["fits"]), int(t[i]["fits"])]
        classes[c].append(i)
    for c, ids in classes.items():
        wh = [int(t[i]["strict"]) - int(w[i]["strict"]) for i in ids]
        cp = [int(t[i]["capped_strict"]) - int(w[i]["capped_strict"]) for i in ids]
        out[c] = {"n": len(ids), "whole_up": wh.count(1), "whole_down": wh.count(-1), "capped_up": cp.count(1), "capped_down": cp.count(-1),
                  "whole_net": sum(wh), "capped_net": sum(cp), "w0_right": sum(int(w[i]["strict"]) for i in ids),
                  "arm_right_whole": sum(int(t[i]["strict"]) for i in ids)}
    k, left = out["kept"], out["left"]
    out["filed_gain"] = 100 * k["whole_net"] / k["n"]
    out["on_w0_fitting_whole"] = 100 * (k["whole_net"] + left["whole_net"]) / (k["n"] + left["n"])
    out["on_w0_fitting_capped"] = 100 * (k["whole_net"] + left["capped_net"]) / (k["n"] + left["n"])
    out["unpaired_drop"] = 100 * (st.mean(t[i]["strict"] for i in t if t[i]["fits"]) - st.mean(w[i]["strict"] for i in w if w[i]["fits"]))
    return out


def opening(r: dict[str, Any]) -> str:
    return " ".join(r["response"].split()[:2])


def tertiles(ids: list[int], score: dict[int, float]) -> list[list[int]]:
    s, k = sorted(ids, key=lambda i: score[i]), len(ids) // 3
    return [s[:k], s[k:2 * k], s[2 * k:]]


def has_check(r: dict[str, Any]) -> bool:
    low = r["response"].lower()
    return any(p in low for p in PHRASES)


def style(d: dict[str, Rows]) -> dict[str, Any]:
    w, out = d["W0"], {}
    for a, t in d.items():
        c = collections.Counter(opening(r) for r in t.values())
        lens = [r["response_tokens"] for r in t.values()]
        out[a] = {"openings": [{"opening": o, "n": n, "right_whole": st.mean(t[i]["strict"] for i in t if opening(t[i]) == o),
                                "w0_right_same_problems": st.mean(w[i]["strict"] for i in t if opening(t[i]) == o),
                                "mean_tokens": st.mean(t[i]["response_tokens"] for i in t if opening(t[i]) == o)} for o, n in c.most_common(5)],
                  "mean_tokens": st.mean(lens), "median_tokens": st.median(lens), "fits": st.mean(r["fits"] for r in t.values()),
                  "truncated": st.mean(r["truncated"] for r in t.values()), "check_phrase": st.mean(has_check(r) for r in t.values()),
                  "identical_to_w0": sum(t[i]["response"] == w[i]["response"] for i in t)}
    return out


def length_tests(d: dict[str, Rows]) -> dict[str, Any]:
    """Does lengthening predict the gain, problem by problem, on the problems kept at both?"""
    w, out = d["W0"], {}
    for a in ARMS:
        t = d[a]
        kept = [i for i in w if w[i]["fits"] and t[i]["fits"]]
        dl = {i: float(t[i]["response_tokens"] - w[i]["response_tokens"]) for i in kept}
        dr = {i: int(t[i]["strict"]) - int(w[i]["strict"]) for i in kept}
        up, down = [i for i in kept if dr[i] > 0], [i for i in kept if dr[i] < 0]
        by_w0 = tertiles(sorted(w), {i: float(w[i]["response_tokens"]) for i in w})
        out[a] = {"mean_change_kept": st.mean(dl.values()), "corr_length_right": st.correlation(list(dl.values()), list(map(float, dr.values()))),
                  "length_change_up": st.mean(dl[i] for i in up), "length_change_down": st.mean(dl[i] for i in down),
                  "gain_by_length_change": [mean_se([dr[i] for i in g]) | {"mean_change": st.mean(dl[i] for i in g)} for g in tertiles(kept, dl)],
                  "whole_gain_by_w0_length": [mean_se([int(t[i]["strict"]) - int(w[i]["strict"]) for i in g]) for g in by_w0],
                  "capped_gain_by_w0_length": [mean_se([int(t[i]["capped_strict"]) - int(w[i]["capped_strict"]) for i in g]) for g in by_w0]}
    return out


def same_opening(d: dict[str, Rows]) -> dict[str, Any]:
    """Gain on problems whose answer opens with the same two words at W0 and at the arm, and on the rest."""
    w, out = d["W0"], {}
    for a in ARMS:
        t = d[a]
        same = [i for i in w if opening(w[i]) == opening(t[i])]
        out[a] = {k: mean_se([int(t[i]["strict"]) - int(w[i]["strict"]) for i in g]) for k, g in
                  (("same", same), ("changed", [i for i in w if i not in set(same)]))}
    return out


def overlap(d: dict[str, Rows]) -> dict[str, Any]:
    """On the problems W0 gets wrong and that fit at all four states: who gets each right? Independent arms would share
    the product of their rates."""
    w, ids = d["W0"], [i for i in d["W0"] if all(d[a][i]["fits"] for a in d)]
    wrong = [i for i in ids if not w[i]["strict"]]
    right = {a: {i for i in wrong if d[a][i]["strict"]} for a in ARMS}
    return {"kept_all": len(ids), "w0_wrong": len(wrong), "R": len(right["R"]), "N": len(right["N"]), "F": len(right["F"]),
            "R_and_N": len(right["R"] & right["N"]), "R_and_N_if_independent": len(right["R"]) * len(right["N"]) / len(wrong),
            "N_and_F": len(right["N"] & right["F"]), "wrong_in_all": len(set(wrong) - right["R"] - right["N"] - right["F"])}


def checks_in_null(d: dict[str, Rows]) -> dict[str, Any]:
    """The null's answers with a self-check phrase against those without, kept at both."""
    w, t = d["W0"], d["N"]
    kept = [i for i in w if w[i]["fits"] and t[i]["fits"]]
    yes = [i for i in kept if has_check(t[i])]
    no = [i for i in kept if not has_check(t[i])]
    return {k: mean_se([int(t[i]["strict"]) - int(w[i]["strict"]) for i in g]) for k, g in (("with", yes), ("without", no))}


def windows(arms: dict[str, Any]) -> dict[str, Any]:
    """Sampled reward on each step's prompts against W0's on the same prompts (arm F's rollouts). SE over steps."""
    out: dict[str, Any] = {}
    for lo, hi in ((1, 10), (11, 20), (21, 50), (51, 75), (76, 95), (96, 100), (51, 100), (91, 100)):
        row: dict[str, Any] = {}
        pick = {a: [s for s in arms[a]["steps"] if lo <= s["step"] <= hi] for a in ARMS}
        for a in ARMS:
            s = pick[a]
            row[a] = {"reward": st.mean(x["reward_rate"] for x in s), "length": st.mean(x["mean_len"] for x in s),
                      "window": st.mean(x["window_share"] for x in s), "grad_rms": st.mean(x["grad_rms"] for x in s),
                      "live_tokens": st.mean(x["live_tokens"] for x in s)}
            row[a]["right_when_inside"] = row[a]["reward"] / row[a]["window"]
        for a, b, k in (("R", "F", "R-W0"), ("N", "F", "N-W0"), ("R", "N", "R-N")):
            row[k] = mean_se([x["reward_rate"] - y["reward_rate"] for x, y in zip(pick[a], pick[b], strict=True)])
        out[f"{lo}-{hi}"] = row
    return out


def kept(s: dict[str, Any]) -> float:
    """Sampled answers lenient-right and inside the window, as a share of all: the reward with the format ignored."""
    return s["lenient_when_inside"] * s["inside"]


def sampled_formats(roll: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lo, hi in ((1, 10), (11, 20), (21, 50), (51, 75), (76, 95), (96, 100)):
        steps = [str(t) for t in range(lo, hi + 1)]
        row: dict[str, Any] = {}
        for a in ARMS:
            s = roll[a]["spans"][f"{lo}-{hi}"]
            row[a] = {k: s[k] for k in ("line_when_inside", "lenient_when_inside", "inside")} | {"kept": kept(s), "we": s["firsts"].get("We", 0.0), "let": s["firsts"].get("Let", 0.0), "step": s["firsts"].get("Step", 0.0)}
        for x, y, k in (("R", "F", "R-W0"), ("N", "F", "N-W0"), ("R", "N", "R-N")):
            row[k] = mean_se([kept(roll[x]["steps"][t]) - kept(roll[y]["steps"][t]) for t in steps])
        out[f"{lo}-{hi}"] = row
    return out


def trajectories(arms: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in ARMS:
        size = {s["step"]: s["dw_norm"] for s in arms[a]["steps"]}
        out[a] = {t: {"dw": size[int(t)], **r, "kl_per_size2": r["kl"] / size[int(t)] ** 2} for t, r in arms[a]["reads"].items()}
    return out


# ----------------------------------------------------------------------------- figures


def tidy(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)


def dodge_points(ax: Any, g: dict[str, Any], series: tuple[str, ...], rule_keys: tuple[str, ...]) -> None:
    for j, a in enumerate(series):
        x = np.arange(len(rule_keys)) + (j - (len(series) - 1) / 2) * 0.17
        y, e = [g[a][r]["mean"] for r in rule_keys], [g[a][r]["se"] for r in rule_keys]
        ax.errorbar(x, y, yerr=e, fmt="D" if a == "R-N" else "o", ms=8, color=COL[a], ecolor=COL[a], capsize=3, lw=1.6, label=NAME[a], mec="white")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(rule_keys)), [RULES[r] for r in rule_keys])


def fig_gains(a: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, s in zip(axes, SETS, strict=True):
        tidy(ax)
        ax.fill_between([-0.45, 0.45], -1.5, 1.5, color=COL["N"], alpha=0.14, lw=0, label="band the null was filed to stay in (drop rule)")
        dodge_points(ax, a[s]["gains"]["strict"], ("R", "N", "F", "R-N"), tuple(RULES))
        ax.set_title(SETS[s], color=INK)
        ax.text(2.45, 0.2, "no change", color=MUTED, ha="right", fontsize=9)
    axes[0].set_ylabel("strict accuracy change, points\n(paired over problems; bars: 1 SE)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=5)
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle("How much did each arm gain from the unchanged model, under each way of treating long answers?", color=INK)
    save(fig, "gains")


def fig_grader(a: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, s in zip(axes, SETS, strict=True):
        tidy(ax)
        for j, gr in enumerate(GRADER):
            d = a[s]["gains"][gr]["R-N"]
            x = np.arange(3) + (j - 0.5) * 0.2
            ax.errorbar(x, [d[r]["mean"] for r in RULES], yerr=[d[r]["se"] for r in RULES], fmt="D" if gr == "strict" else "s", ms=8, capsize=3, lw=1.6, mec="white",
                        color=COL["R-N"] if gr == "strict" else "#1baf7a", label=f"{gr} grader" + (" (needs the exact final line)" if gr == "strict" else " (any final number)"))
        ax.axhline(0, color=INK, lw=1)
        ax.text(2.45, 0.2, "RL level with the null", color=MUTED, ha="right", fontsize=9)
        ax.set_xticks(range(3), list(RULES.values()))
        ax.set_title(SETS[s], color=INK)
    axes[0].set_ylabel("RL minus null, points\n(same problems; bars: 1 SE)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Does RL's lead over the null survive a grader that ignores the answer line's format?", color=INK)
    save(fig, "grader")


def fig_accounting(a: dict[str, Any]) -> None:
    classes = ("kept", "left", "came back", "outside at both")
    labels = ("fits at both", "fits at W0 only:\nran long", "fits at arm only:\ncame back", "fits at neither")
    series = (("N", "whole", 1.0, "null, whole answers"), ("N", "capped", 0.45, "null, capped answers"), ("R", "whole", 1.0, "RL, whole answers"), ("R", "capped", 0.45, "RL, capped answers"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, s in zip(axes, SETS, strict=True):
        tidy(ax)
        k = sum(a[s]["accounting"]["N"][c]["n"] for c in classes)
        for j, (arm, grade, alpha, lab) in enumerate(series):
            y = [100 * a[s]["accounting"][arm][c][f"{grade}_net"] / k for c in classes]
            ax.bar(np.arange(4) + (j - 1.5) * 0.2, y, 0.18, color=COL[arm], alpha=alpha, label=lab)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.set_title(SETS[s], color=INK)
    axes[0].set_ylabel("net problems gained, as points\nof all the set's problems")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Where do the gains sit? On problems that fit at both, unless long answers are graded where the window cuts them", color=INK)
    save(fig, "accounting")


def fig_openings(a: dict[str, Any], data: dict[str, dict[str, Rows]]) -> None:
    groups = (("Let's break", "#2a78d6"), ("Let's solve", "#86b6ef"), ("Step 1:", "#1baf7a"), ("We are", "#eb6834"), ("We need", "#f4a988"))
    states = ("W0", "R", "N", "F")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, s in zip(axes, SETS, strict=True):
        left = np.zeros(4)
        for name, c in groups:
            v = np.array([100 * st.mean(opening(r) == name for r in data[s][x].values()) for x in states])
            ax.barh(range(4), v, left=left, color=c, edgecolor="white", lw=1.5, label=name)
            left += v
        ax.barh(range(4), 100 - left, left=left, color=GRID, edgecolor="white", lw=1.5, label="any other")
        ax.set_yticks(range(4), [NAME[x] for x in states])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of greedy answers, %")
        ax.set_title(SETS[s], color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle("How does each arm's greedy answer open?", color=INK)
    save(fig, "openings")


def fig_lengths(data: dict[str, dict[str, Rows]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axes, SETS, strict=True):
        tidy(ax)
        for x in ("W0", "F", "R", "N"):
            v = np.sort([r["response_tokens"] for r in data[s][x].values()])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), color=COL[x], lw=2, label=NAME[x], where="post")
        cut = 512 - st.mean(r["prompt_tokens"] for r in data[s]["W0"].values())
        ax.axvline(cut, color=MUTED, ls="--", lw=1)
        ax.text(cut - 6, 0.08, "window, at the\nmean prompt", color=MUTED, fontsize=9, ha="right")
        ax.set_xlabel("response tokens")
        ax.set_title(SETS[s], color=INK)
    axes[0].set_ylabel("share of problems with an\nanswer this short or shorter")
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("How did greedy answer lengths move?", color=INK)
    save(fig, "lengths")


def by_step(roll: dict[str, Any], arm: str, f: Any) -> np.ndarray:
    return np.array([f(roll[arm]["steps"][str(t)]) for t in range(1, 101)])


def fig_reward(roll: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    steps = np.arange(1, 101)
    panels = (("reward as filed: strict and inside the window", lambda s: s["reward"]), ("format ignored: lenient and inside the window", kept))
    for ax, (title, f) in zip(axes, panels, strict=True):
        tidy(ax)
        base = by_step(roll, "F", f)
        for a in ("R", "N"):
            d = 100 * (by_step(roll, a, f) - base)
            ax.plot(steps, d, color=COL[a], alpha=0.25, lw=1)
            ax.plot(steps[9:], np.convolve(d, np.ones(10) / 10, mode="valid"), color=COL[a], lw=2.4, label=f"{NAME[a]}, mean of the last 10 steps")
        ax.axhline(0, color=COL["W0"], lw=1.4, label="unchanged model on the same prompts")
        ax.set_xlabel("step (64 new problems, 8 samples each, temperature 1)")
        ax.set_title(title, color=INK, fontsize=11)
    axes[0].set_ylabel("share of samples, minus the\nunchanged model's, points")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("On fresh prompts, do samples from the changing weights do better, with and without the answer format?", color=INK)
    save(fig, "reward")


def fig_collapse(roll: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    tidy(ax)
    steps = np.arange(10, 101)
    for key, ls, what in (("line_when_inside", "-", "has the answer line"), ("lenient_when_inside", "--", "right, format ignored")):
        for a in ("N", "F"):
            v = 100 * np.convolve(by_step(roll, a, lambda s, k=key: s[k]), np.ones(10) / 10, mode="valid")
            ax.plot(steps, v, ls=ls, lw=2.2, color=COL["W0"] if a == "F" else COL[a], label=f"{'unchanged model' if a == 'F' else NAME[a]}: {what}")
    ax.set_xlabel("step (mean of the last 10 steps)")
    ax.set_ylabel("share of sampled answers ending\ninside the window, %")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Was the null's collapse in sampled reward the maths or the answer line?", color=INK)
    save(fig, "collapse")


def fig_kl(tr: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    tidy(ax)
    for a in ARMS:
        ts = sorted(tr[a], key=int)
        x, y = [tr[a][t]["dw"] for t in ts], [tr[a][t]["kl"] for t in ts]
        ax.plot(x, y, "-o", color=COL[a], lw=2, ms=7, label=NAME[a], mec="white")
        ax.annotate("step 100", (x[-1], y[-1]), textcoords="offset points", xytext=(6, -3), fontsize=9, color=INK)
    x0, y0 = tr["R"]["1"]["dw"], tr["R"]["1"]["kl"]
    xs = np.array([x0, 0.45])
    ax.plot(xs, y0 * (xs / x0) ** 2, ls="--", color=MUTED, lw=1.2, label="if KL grew as the update's size squared, from RL's first step")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("size of the update |δW| at steps 1, 2, 5, 10, 20, 50, 100")
    ax.set_ylabel("KL from the unchanged model, per token,\non held-out samples")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Does the policy move as the weights move, in each arm?", color=INK)
    save(fig, "kl")


def fig_delta_r(arms: dict[str, Any], probes: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    tidy(ax)
    for a in ARMS:
        ts = sorted(arms[a]["reads"], key=int)
        ax.plot([int(t) for t in ts], [arms[a]["reads"][t]["delta_r"] for t in ts], "-o", color=COL[a], lw=2, ms=7, label=NAME[a], mec="white")
    ts = ["10", "50", "100"]
    ax.plot([int(t) for t in ts], [3 * abs(arms["N"]["reads"][t]["delta_r"]) for t in ts], "_", color=COL["N"], ms=22, mew=2.5, label="what RL had to clear: 3 × the null")
    style_of = {"n1e-6": ("N", "null probe, no warm-up"), "r1e-6": ("R", "RL probe, no warm-up"), "r1e-6w10": ("R", "RL probe, this run's warm-up")}
    for (name, (arm, lab)), m in zip(style_of.items(), ("^", "^", "s"), strict=True):
        ax.plot(10, probes[name]["reads"]["10"]["delta_r"], m, ms=10, mfc="none", mec=COL[arm], mew=1.8, label=lab, ls="none")
    ax.axhline(0, color=INK, lw=1, label="no change")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("step")
    ax.set_ylabel("ΔR: log-probability gain of the unchanged model's\nright samples over its wrong ones, nats")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Does the right-over-wrong signal separate RL from the null?", color=INK)
    save(fig, "delta_r")


def save(fig: Any, name: str) -> None:
    fig.tight_layout(rect=(0, 0.06, 1, 1) if name == "gains" else (0, 0, 1, 1))
    fig.savefig(FIG / f"{name}.png", dpi=130, facecolor="white")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    data = {s: {x: rows(x, s) for x in ("W0", *ARMS)} for s in SETS}
    arms = {a: json.loads((RUN / f"{a}.json").read_text()) for a in ARMS}
    probes = {n: json.loads((RES / f"probe_{n}.json").read_text()) for n in ("n1e-6", "r1e-6", "r1e-6w10")}
    out: dict[str, Any] = {s: {"gains": gains(d), "accounting": {a: accounting(d, a) for a in ARMS}, "style": style(d), "length": length_tests(d),
                               "same_opening": same_opening(d), "overlap": overlap(d), "check_phrases_in_null": checks_in_null(d)}
                           for s, d in data.items()}
    roll = json.loads((RUN / "rollouts_summary.json").read_text())
    out["sampled"] = windows(arms)
    out["sampled_format"] = sampled_formats(roll)
    out["trajectory"] = trajectories(arms)
    (RUN / "analysis.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    fig_gains(out)
    fig_grader(out)
    fig_accounting(out)
    fig_openings(out, data)
    fig_lengths(data)
    fig_reward(roll)
    fig_collapse(roll)
    fig_kl(out["trajectory"])
    fig_delta_r(arms, probes)


if __name__ == "__main__":
    main()
