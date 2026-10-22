"""Figures for regating_leak (E18) from results/regating_leak.json (JSON only; no model).

  recovery.png   how much of the re-gating effect each trigger-only gate pattern recovers, per task: the state's own
                 per-window gates (trig), W0's gates plus the shared change (shared), the state's mean gates (mean) and
                 its reference W0's mean gates (mean_W0), a different-target fine-tune's shared change (cross), and the
                 random moved null. Bars seed 0, dots seed 1. Lines: 0.7 (P1), 0.1 (P3's limit), 0.
  shared.png     the P2 reading: shared / trig per task and seed, with the mechanism (>= 0.7) and leak (<= 0.3) regions
                 shaded and the random null's share; beside it, unscored, (mean - mean_W0) / trig.
  natural.png    is S_on the model's own machinery for the target token? J_nat (seeds 0 and 1) against the random
                 sets' mean, their max, and 3x the mean (P4).
  adversary.png  the share of S_on the Pile adversary turns down at the state and W0's adversary on the same set,
                 against the share of all alive subcomponents (unscored).
  amend1.png     amendment 1, from results/regating_leak_amend1.json: per task and seed, R (recovery / own gates')
                 as a stacked bar from 0 to R_mean: R_shared, then the clip, the interaction of the shift with W0's
                 per-window variation, W0's own variation with no shift (R_mean_W0), and the dead subcomponents. Dots:
                 the shift on the switched-on set only. Lines: 1 (own gates), 0.7 (P2's bar).
  amend2_null.png  amendment 2, from results/regating_leak_amend2.json: per task and seed, the unshuffled shared
                 change's R (clipped and not) against its 20 shuffled draws (s permuted within each matrix's alive
                 subcomponents), with the filed identity (0.2 x unshuffled) and bulk (0.5 x) bars.
  amend2_bins.png  amendment 2: R of the unclipped change restricted to one size bin of s, bins ordered from turned
                 down to turned up, against the sum of the bins and the whole unclipped change. Bars seed 0, dots seed 1.

  amend3_tests.png  amendment 3, from results/regating_leak_amend3.json, two panels on the same axis. Left: the
                 state's own values with the fixed change removed, clip(g1(x) - s), against W0's mean values with no
                 change (mean_W0), with the filed H band (<= 0.15) and leak band (>= 0.3). Right: another window's
                 updated values, g1(pi(x)), against the same with W0's values, with the filed H band (>= 0.8).
  amend3_donor.png  amendment 3: per state, the correlation of each window's gain under another window's values with
                 its own gain (recipient) and with the donor's gain, against the chance correlation.
  amend3_halves.png  amendment 3: the turned-up and turned-down halves of the change, clipped, against 20 shuffled
                 draws each and the filed identity bar (0.2 x unshuffled); diamonds: the turned-down half unclipped.

Usage (repo root):  uv run python experiments/regating_leak/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
BARS = ("trig", "shared", "mean", "mean_W0", "cross", "random")
COLOR = {"trig": "#3b6fb6", "shared": "#1baf7a", "mean": "#9ecae1", "mean_W0": "#d9d9d9", "cross": "#eb6834",
         "random": "#7f7f7f"}
NAME = {"trig": "own per-window gates", "shared": "W0's gates + one shared change", "mean": "state's mean gates",
        "mean_W0": "W0's mean gates (reference for the bar left)", "cross": "a different target's shared change",
        "random": "shared change moved to random subcomponents"}
LABEL = {"emoticon": "emoticon\n(-> 'o')", "r1_t534_v369": "r1 t534\n(' which' -> ' was')",
         "r2_t359_v816": "r2 t359\n(' we' -> ' just')"}
GREY = "#7f7f7f"


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def value(seed: dict[str, Any], bar: str) -> float:
    return seed["random_mean"] if bar == "random" else seed["rho"][bar]


def recovery(out: dict[str, Any]) -> None:
    tasks: list[str] = list(out["tasks"])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    w = 0.13
    for i, n in enumerate(tasks):
        s0, s1 = out["tasks"][n]["seeds"]["0"], out["tasks"][n]["seeds"]["1"]
        for j, b in enumerate(BARS):
            x = i + (j - 2.5) * w
            ax.bar(x, value(s0, b), w * 0.9, color=COLOR[b], edgecolor="#555555" if b == "mean_W0" else None,
                   label=NAME[b] if i == 0 else None)
            ax.plot(x, value(s1, b), "o", color="black", ms=4, label="seed 1" if i == 0 and j == 0 else None)
            if b == "random":
                ax.plot([x, x], [s0["random_mean"], s0["random_max"]], color="black", lw=1)
    for y, txt in ((0.7, "0.7: own gates' threshold"), (0.1, "0.1: different target's limit"), (0.0, "0: W0's gates")):
        ax.axhline(y, color=GREY, ls=":", lw=1)
        ax.text(-0.45, y, txt, fontsize=8, color="#444444", va="bottom")
    ax.set_xticks(np.arange(len(tasks)), [LABEL.get(str(n), str(n)) for n in tasks])
    ax.set_ylim(-0.4, 1.55)
    ax.set_ylabel("share of the re-gating effect recovered\n(0 = W0's gates, 1 = the state's gates everywhere)")
    ax.set_title("Which gate patterns at the trigger give back the re-gating effect?")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    save(fig, "recovery.png")


def shared(out: dict[str, Any]) -> None:
    tasks: list[str] = list(out["tasks"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhspan(0.7, 1.3, color="#1baf7a", alpha=0.12)
    ax.axhspan(-0.1, 0.3, color="#eb6834", alpha=0.12)
    ax.text(2.45, 1.25, "mechanism region (filed: >= 0.7)", ha="right", va="top", fontsize=8)
    ax.text(2.45, 0.28, "leak region (filed: <= 0.3)", ha="right", va="top", fontsize=8)
    ax.text(2.45, 0.5, "between: not covered\nby the plan's tree", ha="right", va="center", fontsize=8, color="#444444")
    w = 0.22
    for i, n in enumerate(tasks):
        for k, s in out["tasks"][n]["seeds"].items():
            rho = s["rho"]
            dx = -0.04 if k == "0" else 0.04
            mk = "o" if k == "0" else "s"
            pts = ((-w, rho["shared"] / rho["trig"], COLOR["shared"], "W0's gates + one shared change (filed)"),
                   (0.0, (rho["mean"] - rho["mean_W0"]) / rho["trig"], "#4a90c2",
                    "state's mean gates minus W0's mean gates (unscored)"),
                   (w, s["random_mean"] / rho["trig"], COLOR["random"], "random set (null)"))
            for off, y, c, lab in pts:
                ax.plot(i + off + dx, y, mk, color=c, ms=8, label=lab if i == 0 and k == "0" else None)
    ax.plot([], [], "o", color="black", label="seed 0")
    ax.plot([], [], "s", color="black", label="seed 1")
    ax.set_xticks(np.arange(len(tasks)), [LABEL.get(str(n), str(n)) for n in tasks])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.1, 1.3)
    ax.set_ylabel("recovery as a share of the own gates' recovery")
    ax.set_title("Does one shared switch pattern do what each window's own gates do?")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "shared.png")


def natural(out: dict[str, Any]) -> None:
    nat = out.get("natural")
    if not nat:
        return
    tasks: list[str] = list(nat)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(tasks))
    ax.bar(x - 0.18, [nat[n]["s0"]["J"] for n in tasks], 0.34, color="#3b6fb6", label="switched-on set, seed 0")
    ax.plot(x - 0.18, [nat[n]["s1"]["J"] for n in tasks], "o", color="black", ms=4, label="seed 1")
    ax.bar(x + 0.18, [nat[n]["s0"]["random_mean"] for n in tasks], 0.34, color=GREY, label="random sets, mean")
    for i, n in enumerate(tasks):
        r = nat[n]["s0"]
        ax.plot([i + 0.01, i + 0.35], [3 * r["random_mean"]] * 2, color="black", lw=1.5,
                label="3 x random mean (filed threshold)" if i == 0 else None)
        ax.plot([i + 0.18] * 2, [r["random_mean"], r["random_max"]], color="black", lw=1,
                label="random sets, max" if i == 0 else None)
        note = "" if nat[n]["scorable"] else "\nunscored (< 200)"
        ax.text(i - 0.18, nat[n]["s0"]["J"] + 0.005, f"{nat[n]['n_positions']} positions{note}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x, [LABEL.get(str(n), str(n)) for n in tasks])
    ax.set_ylim(0, 0.36)
    ax.set_ylabel("Jaccard with W0's top subcomponents\nbefore the target token on Pile")
    ax.set_title("Is the switched-on set the model's own machinery for the token?")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "natural.png")


def adversary(out: dict[str, Any]) -> None:
    adv = out.get("adversary")
    if not adv:
        return
    tasks: list[str] = [n for n in adv if n != "W0"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(tasks))
    ax.bar(x - 0.18, [adv[n]["down_S_on"] for n in tasks], 0.34, color="#3b6fb6", label="the state's adversary")
    ax.bar(x + 0.18, [adv[n]["down_S_on_at_W0"] for n in tasks], 0.34, color=GREY, label="W0's adversary")
    ax.axhline(adv["W0"]["down_alive"], color="black", ls=":", lw=1,
               label="W0's adversary, all alive subcomponents")
    for i, n in enumerate(tasks):
        k = out["tasks"][n]["seeds"]["0"]["n_on"]
        ax.text(i, 0.02, f"{k} switched on", ha="center", fontsize=8, color="white")
    ax.set_xticks(x, [LABEL.get(str(n), str(n)) for n in tasks])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the switched-on set\nturned down by the adversary on Pile")
    ax.set_title("Does the Pile adversary remove the subcomponents the update switches on?")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "adversary.png")


A1_PARTS = (("shared", "W0's gates + shared change, clipped (filed)", "#1baf7a"),
            ("clip", "+ removing the clip", "#f2b134"),
            ("interaction", "+ averaging W0's gates: the part that needs the shift", "#9b59b6"),
            ("variation_W0", "+ averaging W0's gates: the part it gives with no shift", "#d9d9d9"),
            ("dead", "+ dead subcomponents at the state's mean", "#555555"))


def a1_segments(s: dict[str, Any]) -> dict[str, float]:
    """R_shared, then the path's steps in R units; the variation step split into W0's own part (R_mean_W0) and the rest."""
    R = s["R"]
    return {"shared": R["shared"], "clip": R["noclip"] - R["shared"],
            "interaction": R["mean_s"] - R["noclip"] - R["mean_W0"], "variation_W0": R["mean_W0"],
            "dead": R["mean"] - R["mean_s"]}


def amend1(a1: dict[str, Any]) -> None:
    cols = [(n, k, s) for n, t in a1["tasks"].items() for k, s in t["seeds"].items()]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    for key, lab, c in A1_PARTS:
        v = np.array([a1_segments(s)[key] for _, _, s in cols])
        ax.bar(x, v, 0.6, bottom=bottom, color=c, label=lab, edgecolor="#555555" if key == "variation_W0" else None)
        bottom = bottom + v
    ax.plot(x, [s["R"]["S_on"] for _, _, s in cols], "D", color="black", ms=6,
            label="shift on the switched-on set only, clipped")
    for y, txt in ((1.0, "1: each window's own updated values"), (0.7, "0.7: the filed bar for the shared pattern")):
        ax.axhline(y, color=GREY, ls=":", lw=1)
        ax.text(-0.45, y, txt, fontsize=8, color="#444444", va="bottom")
    ax.set_xticks(x, [f"{LABEL.get(n, n)}\nseed {k}" for n, k, _ in cols], fontsize=8)
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("recovery as a share of the own values' recovery\n(0 = original values, 1 = own updated values)")
    ax.set_title("What separates the filed shared pattern from the mean pattern? Mostly averaging, then the clip")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    save(fig, "amend1.png")


A2_ARMS = (("shared", "clipped (filed)", "#1baf7a"), ("noclip", "not clipped", "#f2b134"))
A2_BINS = ("(-inf, -0.1]", "(-0.1, -0.01]", "(-0.01, 0.01]", "(0.01, 0.03]", "(0.03, 0.1]", "(0.1, 0.3]", "(0.3, inf]")
A2_BIN_LABEL = ("<= -0.1", "-0.1 to\n-0.01", "-0.01 to\n0.01", "0.01 to\n0.03", "0.03 to\n0.1", "0.1 to\n0.3", "> 0.3\n(switched-on set)")
TASK_COLOR = {"emoticon": "#3b6fb6", "r1_t534_v369": "#eb6834", "r2_t359_v816": "#9b59b6"}


def amend2_null(a2: dict[str, Any]) -> None:
    cols = [(n, k, s) for n, t in a2["tasks"].items() for k, s in t["seeds"].items()]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    rng = np.random.default_rng(0)
    for i, (_, _, s) in enumerate(cols):
        for j, (arm, lab, c) in enumerate(A2_ARMS):
            p = s["perm"][arm]
            x = i + (j - 0.5) * 0.36
            ax.bar(x, p["R_unshuffled"], 0.3, color=c, label=f"unshuffled change, {lab}" if i == 0 else None)
            ax.plot(x + rng.uniform(-0.1, 0.1, len(p["R_draws"])), p["R_draws"], ".", color="black", ms=4,
                    label="20 shuffled draws" if i == 0 and j == 0 else None)
            for f, ls, txt in ((0.2, "-", "identity bar: 0.2 x unshuffled"), (0.5, "--", "bulk bar: 0.5 x unshuffled")):
                ax.plot([x - 0.15, x + 0.15], [f * p["R_unshuffled"]] * 2, color="#c0392b", ls=ls, lw=1.2,
                        label=txt if i == 0 and j == 0 else None)
    ax.axhline(0, color=GREY, ls=":", lw=1, label="0: the original values")
    ax.set_xticks(np.arange(len(cols)), [f"{LABEL.get(n, n)}\nseed {k}" for n, k, _ in cols], fontsize=8)
    ax.set_ylim(-0.2, 1.1)
    ax.set_ylabel("recovery as a share of the own values' recovery\n(0 = original values, 1 = own updated values)")
    ax.set_title("Does it matter which subcomponent gets which value of the change? Shuffled, it recovers nothing")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    save(fig, "amend2_null.png")


def amend2_bins(a2: dict[str, Any]) -> None:
    tasks: list[str] = list(a2["tasks"])
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axvspan(-0.5, 1.5, color="#eb6834", alpha=0.08)
    ax.axvspan(2.5, 6.5, color="#1baf7a", alpha=0.08)
    ax.text(0.5, 0.97, "the update turns these down", ha="center", fontsize=8)
    ax.text(4.5, 0.97, "the update turns these up", ha="center", fontsize=8)
    w = 0.26
    for t, n in enumerate(tasks):
        seeds = a2["tasks"][n]["seeds"]
        dx = (t - 1) * w
        for k, mk in (("0", None), ("1", "o")):
            s = seeds[k]
            ys = [s["bins"][b]["R"] for b in A2_BINS] + [sum(s["bins"][b]["R"] for b in A2_BINS),
                                                         s["perm"]["noclip"]["R_unshuffled"]]
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7.5, 8.5]) + dx
            if mk is None:
                ax.bar(x, ys, w * 0.9, color=TASK_COLOR.get(n, GREY), label=LABEL.get(n, n).replace("\n", " "))
            else:
                ax.plot(x, ys, mk, color="black", ms=3.5, label="seed 1" if t == 0 else None)
    ax.axhline(0, color=GREY, ls=":", lw=1)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7.5, 8.5], list(A2_BIN_LABEL) + ["sum of\nthe bins", "whole change,\nnot clipped"],
                  fontsize=8)
    ax.set_xlabel("bin of the shared change s, the updated model's mean value minus the original's")
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("recovery as a share of the own values' recovery\n(change kept only in this bin, not clipped)")
    ax.set_title("Where in the change is the recovery? Mostly in its two tails, turned down as well as up")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    save(fig, "amend2_bins.png")


def a3_cols(a3: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(n, k, s) for n, t in a3["tasks"].items() for k, s in t["seeds"].items()]


def a3_xticks(ax: Any, cols: list[tuple[str, str, dict[str, Any]]]) -> None:
    ax.set_xticks(np.arange(len(cols)), [f"{LABEL.get(n, n)}\nseed {k}" for n, k, _ in cols], fontsize=7)


def amend3_tests(a3: dict[str, Any]) -> None:
    cols = a3_cols(a3)
    x = np.arange(len(cols))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    ax.axhspan(-0.1, 0.15, color="#1baf7a", alpha=0.12)
    ax.axhspan(0.3, 1.2, color="#eb6834", alpha=0.12)
    ax.text(5.4, -0.05, "H: input-varying part does nothing (filed <= 0.15)", ha="right", va="center", fontsize=8)
    ax.text(5.4, 1.15, "leak (filed >= 0.3)", ha="right", va="top", fontsize=8)
    ax.bar(x - 0.17, [s["R"]["resid"] for _, _, s in cols], 0.32, color="#c0392b",
           label="own updated values minus the fixed change, clipped (filed)")
    ax.bar(x + 0.17, [s["R"]["mean_W0"] for _, _, s in cols], 0.32, color="#d9d9d9", edgecolor="#555555",
           label="original model's mean values, no change (reference)")
    ax.plot(x - 0.17, [s["R"]["resid_noclip"] for _, _, s in cols], "v", color="black", ms=6, label="the same, not clipped")
    ax.plot(x - 0.17, [s["R"]["resid_held"] for _, _, s in cols], "x", color="black", ms=6,
            label="held-out mean change removed instead")
    ax.set_title("Does the input-varying part alone recover anything?")
    bx.axhspan(0.8, 1.2, color="#1baf7a", alpha=0.12)
    bx.text(5.4, 1.15, "H: another window's values do as well (filed >= 0.8)", ha="right", va="top", fontsize=8)
    bx.bar(x - 0.17, [s["mismatch"]["R"] for _, _, s in cols], 0.32, color="#3b6fb6",
           label="another window's updated values")
    bx.bar(x + 0.17, [s["mismatch"]["R_W0"] for _, _, s in cols], 0.32, color="#d9d9d9", edgecolor="#555555",
           label="another window's original values (reference)")
    bx.set_title("Does a window need its own updated values?")
    for a in (ax, bx):
        a.axhline(1.0, color=GREY, ls=":", lw=1)
        a.text(-0.45, 1.0, "1: own updated values", fontsize=8, color="#444444", va="bottom")
        a.axhline(0.0, color=GREY, ls=":", lw=1)
        a3_xticks(a, cols)
        a.legend(fontsize=8, loc="center left")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("recovery as a share of the own values' recovery\n(0 = original values, 1 = own updated values)")
    save(fig, "amend3_tests.png")


def amend3_donor(a3: dict[str, Any]) -> None:
    cols = a3_cols(a3)
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    m = [s["mismatch"] for _, _, s in cols]
    ax.bar(x - 0.25, [v["c_own"] for v in m], 0.24, color="#3b6fb6", label="with the window's own gain (recipient)")
    ax.bar(x, [v["c_donor"] for v in m], 0.24, color="#eb6834", label="with the donor window's own gain (leak)")
    ax.bar(x + 0.25, [v["c_null"] for v in m], 0.24, color="#d9d9d9", edgecolor="#555555",
           label="own gain with the donor's gain (chance)")
    for i, v in enumerate(m):
        ax.plot([i - 0.25] * 5, [d["c_own"] for d in v["draws"]], ".", color="black", ms=3,
                label="5 derangements" if i == 0 else None)
        ax.plot([i] * 5, [d["c_donor"] for d in v["draws"]], ".", color="black", ms=3)
        ax.text(i, -0.33, f"gain sd {v['sd_gain_nats']:.1f} nats", ha="center", fontsize=7, color="#444444")
    ax.axhline(0, color=GREY, ls=":", lw=1)
    ax.set_ylim(-0.4, 1.05)
    a3_xticks(ax, cols)
    ax.set_ylabel("correlation over held-out windows of the gain\nunder another window's updated values")
    ax.set_title("Under another window's values, does a window's gain follow its own context or the donor's?")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "amend3_donor.png")


def amend3_halves(a3: dict[str, Any]) -> None:
    cols = a3_cols(a3)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    rng = np.random.default_rng(0)
    arms = (("up", "half_up", "turned-up half, clipped", "#1baf7a"),
            ("down_pool", "half_down_pool", "turned-down half, clipped", "#9b59b6"))
    for i, (_, _, s) in enumerate(cols):
        for j, (pk, rk, lab, c) in enumerate(arms):
            p = s["halves"]["perm"][pk]
            xx = i + (j - 0.5) * 0.36
            ax.bar(xx, s["R"][rk], 0.3, color=c, label=lab if i == 0 else None)
            ax.plot(xx + rng.uniform(-0.1, 0.1, len(p["R_draws"])), p["R_draws"], ".", color="black", ms=4,
                    label="20 shuffled draws" if i == 0 and j == 0 else None)
            ax.plot([xx - 0.15, xx + 0.15], [0.2 * s["R"][rk]] * 2, color="#c0392b", lw=1.2,
                    label="identity bar: 0.2 x unshuffled" if i == 0 and j == 0 else None)
        ax.plot(i + 0.18, s["R"]["half_down_noclip"], "D", color="#555555", ms=5,
                label="turned-down half, not clipped" if i == 0 else None)
        ax.plot(i, s["R"]["shared"], "_", color="black", ms=28, mew=2,
                label="whole change, clipped" if i == 0 else None)
    ax.axhline(0, color=GREY, ls=":", lw=1, label="0: the original values")
    a3_xticks(ax, cols)
    ax.set_ylim(-0.3, 1.1)
    ax.set_ylabel("recovery as a share of the own values' recovery\n(0 = original values, 1 = own updated values)")
    ax.set_title("Does each half of the change need its own subcomponents? Shuffled, neither recovers anything")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    save(fig, "amend3_halves.png")


def main() -> None:
    out = json.loads((HERE / "results" / "regating_leak.json").read_text())
    recovery(out)
    shared(out)
    natural(out)
    adversary(out)
    a1 = HERE / "results" / "regating_leak_amend1.json"
    if a1.is_file():
        amend1(json.loads(a1.read_text()))
    a2 = HERE / "results" / "regating_leak_amend2.json"
    if a2.is_file():
        a2_out = json.loads(a2.read_text())
        amend2_null(a2_out)
        amend2_bins(a2_out)
    a3 = HERE / "results" / "regating_leak_amend3.json"
    if a3.is_file():
        a3_out = json.loads(a3.read_text())
        amend3_tests(a3_out)
        amend3_donor(a3_out)
        amend3_halves(a3_out)


if __name__ == "__main__":
    main()
