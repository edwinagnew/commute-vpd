"""The research log's three figures for E23, redrawn from saved results with plain labels (no model).

Every quantity is the KL (nats, 128 Pile rows) from a model to the model with the adversary's mask applied, the
adversary searched at that model (20 steps unless stated). The earlier Adam failures are E20's four failing states,
measured with the same hooks, rows and adversary (E20's W0 numbers are identical to this run's); they are read from
experiments/cancellation_complement/results, read-only.

  log_parts.png   per state, the extra damage over the original model split into four parts that add up exactly:
                  the alive subcomponents at their original values, their changes, the dead subcomponents and Δ each
                  removed alone, and the offset lost when the alive and the dead removals are made together.
                  Reads results/revert_shard{0,1}.json and cancellation_complement/results/complement_amend1.json.
  log_offset.png  per state, the cost of removing the alive block and the dead-and-Δ block one at a time (summed)
                  against removing both at once; the gap is the offset. Same files.
  log_budget.png  the adversarial ratio against the adversary's number of steps (step size 2 / steps), SGD and Adam
                  at step 1,000 and Adam at step 10 on the same two tasks, with the original model's second seed as
                  the noise. Reads results/cancellation_sgd.json and cancellation_complement/results/complement_amend3.json.

Usage (repo root):  python experiments/cancellation_sgd/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot import FIG, HERE, INK, MUTED, RESULTS, style

E20 = HERE.parent / "cancellation_complement" / "results"
BLUE, RED, PURPLE, GREEN, SAND, ORANGE = "#2a78d6", "#d9534f", "#6250d6", "#2e8b57", "#c9b38a", "#eb6834"
IT, WHICH, FIG_ = "' it' then ' is'", "' which' then ' was'", "' in' then ' Fig'"
# (key, label, group); group colours the row: noise, pass, sgd (fails), adam (fails, E20)
ROWS = (
    ("W0_s1", "original model, second adversary seed (noise)", "noise"),
    ("sgd r0 @100", f"SGD, {IT}, step 100 (control)", "pass"),
    ("adam r0 @1000", f"Adam, {IT}, step 1,000", "pass"),
    ("adam r1 @1000", f"Adam, {WHICH}, step 1,000", "pass"),
    ("sgd r0 @1000", f"SGD, {IT}, step 1,000", "sgd"),
    ("sgd r1 @1000", f"SGD, {WHICH}, step 1,000", "sgd"),
    ("sgd r2 @100", f"SGD at a high rate, {FIG_}, step 100", "sgd"),
    ("sgd ident32", "SGD, 32 copies", "sgd"),
    ("sgd rand32", "SGD, 32 random pairs", "sgd"),
    ("ct r0_t352_v310", f"Adam, {IT}, step 10", "adam"),
    ("ct r1_t534_v369", f"Adam, {WHICH}, step 10", "adam"),
    ("ct r2_t275_v2667", f"Adam, {FIG_}, step 10", "adam"),
    ("ls emoticon", "least-squares split of an Adam emoticon fine-tune", "adam"),
)
GROUP_COLOR = {"noise": MUTED, "pass": BLUE, "sgd": RED, "adam": PURPLE}
GROUP_NAME = {"pass": "passing", "sgd": "SGD failures (this run)", "adam": "earlier Adam failures (E20)"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def records() -> tuple[dict[str, Any], dict[str, Any], float]:
    """The original model's record, every state's record by key, and the pass band in nats."""
    s0, s1 = load(RESULTS / "revert_shard0.json"), load(RESULTS / "revert_shard1.json")
    a1 = load(E20 / "complement_amend1.json")
    w0 = s0["W0"]
    assert abs(a1["W0"]["F_all"] - w0["F_all"]) < 1e-9, "E20's original model differs from this run's"
    rec = {"W0_s1": s0["W0_s1"], **s0["states"], **s1["states"], **a1["states"]}
    return w0, rec, s0["W0_pile"]["pgd_floor"] * w0["F_all"]


def parts(r: dict[str, Any], w0: dict[str, Any]) -> list[float]:
    """Excess F_all(state) - F_all(W0) = alive originals + changes + dead and Δ alone + lost offset (exact)."""
    alive = r["F_orig"] - w0["F_orig"]
    changes = r["F_both"] - r["F_orig"]
    alone = r["F_rest"] - w0["F_rest"]
    lost = (w0["F_both"] + w0["F_rest"] - w0["F_all"]) - (r["F_both"] + r["F_rest"] - r["F_all"])
    return [alive, changes, alone, lost]


def row_axis(ax: Any, labels: list[tuple[str, str]]) -> None:
    n = len(labels)
    ax.set_yticks(range(n), [lab for lab, _ in labels][::-1], fontsize=8.5)
    for tick, (_, g) in zip(ax.get_yticklabels(), labels[::-1]):
        tick.set_color(GROUP_COLOR.get(g, INK))
    prev = None
    for i, (_, g) in enumerate(labels):
        if prev is not None and g != prev:
            ax.axhline(n - i - 0.5, color="#d0cfca", lw=0.8)
        prev = g


def fig_parts(w0: dict[str, Any], rec: dict[str, Any], band: float) -> None:
    names = (("removing the alive subcomponents (at their original values)", BLUE),
             ("removing their changes from the update as well", ORANGE),
             ("removing the dead subcomponents and Δ, each alone", SAND),
             ("the offset lost when alive and dead are removed together", GREEN))
    n = len(ROWS)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    style(ax)
    for i, (key, _, _) in enumerate(ROWS):
        y = n - 1 - i
        vals = parts(rec[key], w0)
        pos = neg = 0.0
        for v, (lab, c) in zip(vals, names):
            ax.barh(y, v, left=pos if v >= 0 else neg, height=0.62, color=c, label=lab if i == 0 else None)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        x = sum(vals)
        ax.plot(x, y, "D", color=INK, ms=5, zorder=5, label="total extra damage" if i == 0 else None)
        share = ""  # shares of the total, failing rows only: the lost offset, and the alive part when it is 10% or more
        if ROWS[i][2] in ("sgd", "adam"):
            share = f"   lost offset {vals[3] / x:.0%}" + (f", alive {vals[0] / x:.0%}" if vals[0] / x >= 0.1 else "")
        ax.text(max(pos, x) + 0.05, y, f"{x:.2f}{share}", va="center", fontsize=8)
    ax.axvspan(-band, band, color="#e4e3df", alpha=0.7, zorder=0)
    ax.text(band + 0.02, n - 0.55, f"±{band:.2f}: the original model's own seed-to-seed spread", fontsize=7.5,
            color=MUTED, va="bottom")
    ax.axvline(0, color=INK, lw=0.8)
    row_axis(ax, [(lab, g) for _, lab, g in ROWS])
    ax.set_xlabel("adversary's damage at the state minus at the original model (KL, nats)")
    ax.set_title("SGD and earlier Adam failures alike: the extra damage is mostly the offset lost between\n"
                 "removing the alive and the dead subcomponents, not the subcomponents' changes",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.4, -0.09))
    ax.set_xlim(right=5.0)
    fig.tight_layout()
    fig.savefig(FIG / "log_parts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_offset(w0: dict[str, Any], rec: dict[str, Any]) -> None:
    rows = [("W0", "original model", "orig"), *ROWS]
    rec = {"W0": w0, **rec}
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    style(ax)
    for i, (key, _, g) in enumerate(rows):
        r, y = rec[key], n - 1 - i
        apart, together = r["F_both"] + r["F_rest"], r["F_all"]
        c = GROUP_COLOR.get(g, INK)
        ax.plot([together, apart], [y, y], color=c, lw=2.2, alpha=0.6)
        ax.plot(apart, y, "o", mfc="white", mec=c, ms=7, label="removed one at a time, costs summed" if i == 0 else None)
        ax.plot(together, y, "o", color=c, ms=7, label="removed together (the adversary's whole mask)" if i == 0 else None)
        ax.text(max(apart, together) + 0.08, y, f"offset {apart - together:+.2f}", va="center", fontsize=8, color=c)
    row_axis(ax, [(lab, g) for _, lab, g in rows])
    ax.set_xlim(0, 5.3)
    ax.set_xlabel("adversary's damage (KL, nats)")
    ax.set_title("At the original model, removing the alive and the dead subcomponents together costs 0.90 nats\n"
                 "less than apart; the failing states lose part or all of that offset",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")  # labels come from the original model's row, in ink
    fig.tight_layout()
    fig.savefig(FIG / "log_offset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_budget() -> None:
    res, a3 = load(RESULTS / "cancellation_sgd.json"), load(E20 / "complement_amend3.json")
    assert a3["kl"]["W0"]["0"] == res["kl"]["W0"]["0"], "E20's original model differs from this run's"
    ns = sorted(int(k) for k in res["kl"]["W0"]["0"])
    series = ((res["ratio"]["sgd r0 @1000"], f"SGD, {IT}, step 1,000", RED, "-"),
              (res["ratio"]["sgd r1 @1000"], f"SGD, {WHICH}, step 1,000", RED, "--"),
              (a3["ratio"]["ct r0_t352_v310"], f"Adam, {IT}, step 10 (E20)", PURPLE, "-"),
              (a3["ratio"]["ct r1_t534_v369"], f"Adam, {WHICH}, step 10 (E20)", PURPLE, "--"),
              (res["ratio"]["adam r0 @1000"], f"Adam, {IT}, step 1,000", BLUE, "-"),
              (res["ratio"]["adam r1 @1000"], f"Adam, {WHICH}, step 1,000", BLUE, "--"))
    kl = res["kl"]["W0"]
    noise = {str(k): kl["1"][str(k)] / kl["0"][str(k)] for k in ns}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for r, lab, c, ls in series:
        ax.plot(ns, [r[str(k)] for k in ns], ls, color=c, lw=2, marker="o", ms=5, label=lab)
    ax.plot(ns, [noise[str(k)] for k in ns], ":", color=MUTED, lw=2, marker="o", ms=5,
            label="original model, second seed over first (noise)")
    ax.axhline(1.5, color=INK, lw=0.8, ls="--")
    ax.axhline(1.241, color=MUTED, lw=0.8, ls="--")
    ax.text(ns[-1], 1.52, "fail above 1.5", fontsize=7.5, ha="right", va="bottom")
    ax.text(ns[-1], 1.26, "pass below 1.24", fontsize=7.5, ha="right", va="bottom", color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(k) for k in ns])
    ax.set_ylim(0.8, 2.8)
    ax.set_xlabel("adversary's number of steps (step size 2 ÷ steps)")
    ax.set_ylabel("adversarial ratio: damage at the state ÷\nat the original model, same steps")
    ax.set_title("On the four step-1,000 states rerun, a search eight times longer leaves each verdict in place",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "log_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    w0, rec, band = records()
    fig_parts(w0, rec, band)
    fig_offset(w0, rec)
    fig_budget()
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
