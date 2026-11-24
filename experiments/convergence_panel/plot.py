"""E272 (cell P2.66) figure, CPU: the pairwise Kendall tau between every reading, baseline and the consensus tier over
tasks at matched KL (results/summary.json, set kl_hi), one heatmap. --sketch draws the card's planned figure with
made-up numbers, one panel per rival; --summary FILE plots another summary (the stub).

    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/plot.py [--summary FILE] [--sketch]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute.figures import figure, gap_heatmap, save
from commute.jobs import load_experiment

SC: Any = load_experiment("convergence_panel", "score")  # tau: Kendall tau-b, nan on a constant side

HERE = Path(__file__).resolve().parent
NAMES = {"tier": "consensus tier", "share": "re-gating share", "steering": "steering share", "erank": "effective rank",
         "E_b": "E_b over K", "kl": "KL from base", "adl": "ADL", "size": "update norm", "prior": "base prior"}  # fmt: skip


def matrix(means: dict[str, dict[str, float]], sign: dict[str, int], order: list[str]) -> list[list[float]]:
    """tau between every two rows of `means` (task -> value, oriented by sign) on their shared tasks; nan when fewer
    than 3 shared tasks or a constant side."""

    def t(a: str, b: str) -> float:
        ts = sorted(set(means[a]) & set(means[b]))
        x, y = [sign.get(a, 1) * means[a][k] for k in ts], [sign.get(b, 1) * means[b][k] for k in ts]
        return SC.tau(x, y) if len(ts) >= 3 else math.nan

    return [[t(a, b) for b in order] for a in order]


def panel(ax: Any, means: dict[str, dict[str, float]], sign: dict[str, int], title: str) -> list[dict[str, Any]]:
    order = [m for m in NAMES if m in means]
    M = matrix(means, sign, order)
    gap_heatmap(ax, M, [NAMES[m] for m in order], [NAMES[m] for m in order], label="Kendall tau over tasks (oriented: + = both read more reuse)", vmax=1.0, colorbar=False)
    ax.tick_params(axis="x", labelrotation=60)
    ax.set_title(title, fontsize=8)
    return [{"panel": title, "row": a, "col": b, "tau": M[i][j]} for i, a in enumerate(order) for j, b in enumerate(order)]


def real(S: dict[str, Any], name: str, label: str) -> None:
    """The pairwise heatmap of one set's KL 'hi' readings (S: the nine's summary, or its 'wide' entry)."""
    P = json.loads((HERE / "results" / "predictions.json").read_text())
    s = S["kl_hi"]
    means = {m: v for m, v in s["means"].items() if v} | {"tier": S["consensus_tier"]}
    fig, ax = figure((6.5, 5.5))
    rows = panel(ax, means, P["rules"]["orientation"], f"Do the readings rank tasks alike, and like the raters? ({label}, matched KL)")
    shared = [len(set(means[a]) & set(means[b])) for a in means for b in means if a != b]
    save(fig, HERE / "figures" / name, n=f"n = {min(shared)} to {max(shared)} tasks a pair (the tasks both have), each the mean of 3 seeds at one matched KL", data=rows)


RATERS = {"consensus": ("raters' consensus", "#000000", "o"), "opus": ("Opus rater", "#0072B2", "s"), "gpt": ("GPT rater", "#D55E00", "^")}


def hope1_fig(S: dict[str, Any]) -> None:
    """Hope 1 on the KL 'hi' set, the nine and the wide side by side: each readable reading's tau-b with each rater's
    tiers, beside the base prior's own tau with the consensus tiers and the consensus critical tau-b."""
    fig, axes = figure((10, 4.2), ncols=2)
    rows: list[dict[str, Any]] = []
    for ax, (label, s) in zip(axes, (("nine tasks", S["kl_hi"]), ("wide panel", S["wide"]["kl_hi"])), strict=True):
        h1 = s["hope1"]
        ms = list(h1)
        for k, (r, by) in enumerate(next(iter(h1.values())).items()):
            name, color, marker = next(v for key, v in RATERS.items() if r.startswith(key))
            ys = [h1[m][r].get("tau", math.nan) for m in ms]
            ax.scatter(ys, [i + 0.15 * (k - 1) for i in range(len(ms))], color=color, marker=marker, s=24, label=name)
            rows += [{"set": label, "reading": m, "rater": r, "tau": y, "p_perm": h1[m][r].get("p_perm"), "call": h1[m][r]["call"]} for m, y in zip(ms, ys, strict=True)]
        # per reading (its own task count): the consensus critical tau-b, the prior's own tau, and the partial tau on the prior
        cs = [h1[m]["consensus"] for m in ms]
        y = list(range(len(ms)))
        ax.scatter([c["critical_tau"] for c in cs], y, marker="|", s=260, color="#555555", label="critical tau-b, consensus (p = 0.05)")
        ax.scatter([c["tau_prior"] for c in cs], y, marker="x", s=30, color="#CC79A7", label="base prior's own tau, consensus")
        ax.scatter([c.get("tau_partial_prior", math.nan) for c in cs], [i + 0.3 for i in y], marker="o", s=26, facecolors="none", edgecolors="#000000",
                   label="consensus, prior partialled out (absent: no common support)")  # fmt: skip
        rows += [{"set": label, "reading": m, "rater": "consensus", "critical_tau": c["critical_tau"], "tau_prior": c["tau_prior"],
                  "tau_partial_prior": c.get("tau_partial_prior"), "n": len(c["tasks"])} for m, c in zip(ms, cs, strict=True)]  # fmt: skip
        ax.axvline(0, color="#bbbbbb", linewidth=0.6)
        ax.set_yticks(y, [f"{NAMES[m]} (n = {len(c['tasks'])})" for m, c in zip(ms, cs, strict=True)])
        ax.set_ylim(len(ms) + 1.4, -0.5)  # top to bottom, room for the legend under the rows
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Kendall tau-b, reading against raters' tier")
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("Does each reading rank the tasks as the raters' tiers do, beyond the base prior?", fontsize=9)
    save(fig, HERE / "figures" / "hope1.png", n="tasks at matched KL, each the mean of 3 seeds", data=rows)


def mixture_fig(S: dict[str, Any]) -> None:
    """Amendment 4/8: every seed's value of each measure against the share of yoruba's rows in the lower+yoruba mix."""
    mix = S["amendment4"]["mixture"]
    P = json.loads((HERE / "results" / "predictions.json").read_text())
    seeds = SC.mix_values(HERE / "results", P["settings"])  # measure -> fraction -> seed values, as score.py read them
    ms = [m for m in NAMES if m in mix and any(map(math.isfinite, (x for v in seeds[m].values() for x in v)))]
    fig, axes = figure((11, 5.2), ncols=4, nrows=2)
    rows: list[dict[str, Any]] = []
    for ax, m in zip(axes.flat, ms, strict=False):
        o = mix[m]
        for f, v in seeds[m].items():
            ax.scatter([f] * len(v), v, color="#0072B2", s=14, alpha=0.8)
            rows += [{"measure": m, "fraction": f, "value": x} for x in v]
        ax.plot(o["fractions"], o["means"], color="#555555", linewidth=0.8)
        ax.set_title(f"{NAMES[m]}\n" + (f"rho {o['rho']:.2f}: " if SC.finite(o["rho"]) else "") + o["call"].replace(" fractions missing", " fraction(s) missing"), fontsize=7)
        ax.set_xlabel("share of yoruba rows")
        ax.set_ylabel(NAMES[m], fontsize=7)
    for ax in list(axes.flat)[len(ms):]:
        ax.set_visible(False)
    fig.suptitle("Does each measure move steadily from lower (reuse) to yoruba (new) as yoruba's share grows? (matched size)", fontsize=9)
    save(fig, HERE / "figures" / "mixture.png", n="3 seeds per fraction (dots), their mean (line); 0 and 1 are lower's and yoruba's own runs", data=rows)


def sketch() -> None:
    tasks = ["pile", "code", "lower", "british", "topic", "french", "yoruba", "squad", "conll"]
    tier = dict(zip(tasks, [3, 3, 3, 3, 2.5, 2, 1, 1.5, 2], strict=True))
    out = dict(zip(tasks, [5, 9, 1, 7, 3, 8, 2, 6, 4], strict=True))  # made-up output size order, unrelated to the tiers
    a = {"tier": tier, "share": {t: tier[t] + 0.1 * i for i, t in enumerate(tasks)}, "steering": {t: tier[t] - 0.05 * i for i, t in enumerate(tasks)},
         "E_b": {t: tier[t] + 0.02 * (i % 3) for i, t in enumerate(tasks)}, "kl": {t: out[t] for t in tasks}, "size": {t: out[t] + 0.5 * (i % 2) for i, t in enumerate(tasks)}}
    b = {"tier": tier, "share": {t: -out[t] + 0.1 * i for i, t in enumerate(tasks)}, "steering": {t: -out[t] for t in tasks},
         "E_b": {t: -out[t] - 0.3 * (i % 2) for i, t in enumerate(tasks)}, "kl": {t: out[t] for t in tasks}, "size": {t: out[t] + 0.5 * (i % 2) for i, t in enumerate(tasks)}}
    sign = {"kl": -1, "size": -1}
    fig, axes = figure((10, 4.6), ncols=2)
    rows = panel(axes[0], a, sign, "Rival a (made up): readings agree with the raters and each other")
    rows += panel(axes[1], b, sign, "Rival b (made up): readings follow output size, as the baselines do")
    fig.suptitle("SKETCH, made-up numbers: pairwise Kendall tau over nine tasks", fontsize=9)
    save(fig, HERE / "figures" / "card_sketch.png", n="n = 9 tasks (made up), each the mean of 3 seeds", data=rows)


if __name__ == "__main__":
    if "--sketch" in sys.argv:
        sketch()
    else:
        S = json.loads((Path(sys.argv[sys.argv.index("--summary") + 1]) if "--summary" in sys.argv else HERE / "results" / "summary.json").read_text())
        real(S, "pairwise_tau.png", "nine tasks")
        if "wide" in S:
            real(S["wide"], "pairwise_tau_wide.png", "wide panel")
            hope1_fig(S)
        mixture_fig(S)
