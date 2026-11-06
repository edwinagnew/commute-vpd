"""E93's figures from results/table.json and results/<state>/state.json (rules.py writes the table): one per claim.

  python experiments/qwen_d1_table/plot.py   # figures/{clean,share,kept_clean,masked}.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rules

FIG = Path(__file__).resolve().parent / "figures"
INK, LINE, MUTED = "#1f4e79", "#b03a2e", "#8c8c8c"
# One colour per arm, the same in every figure (dataviz default categorical order).
ARM_COLOUR = {"anchor 1": "#2a78d6", "anchor 2": "#eb6834", "RL": "#1baf7a", "null": "#eda100",
              "own-samples": "#e87ba4", "u1": "#008300", "u2": "#6250d6"}
LABEL = {"a1_s0": "template, seed 0", "a1_s1": "template, seed 1", "a2_s42": "TOFU facts, seed 42",
         "a2_s142": "TOFU facts, seed 142", "R_s0": "RL, seed 0", "R_s70": "RL, seed 70",
         "N_s0": "shuffled-reward RL, seed 0", "N_s70": "shuffled-reward RL, seed 70",
         "F_s0": "own-samples SFT, seed 0", "u1": "maths SFT u1", "u2": "maths SFT u2 (gold)"}


def load() -> tuple[dict[str, Any], list[str]]:
    t = json.loads((rules.RESULTS / "table.json").read_text())
    return t, [n for n in rules.STATES if n in t["states"]]


def rows(ax: Any, names: list[str], st: dict[str, Any]) -> None:
    ax.set_yticks(range(len(names)), [f"{LABEL[n]}  [{st[n]['verdict']}]" for n in names], fontsize=8)
    for tick, n in zip(ax.get_yticklabels(), names):
        tick.set_color(ARM_COLOUR[st[n]["arm"]])
    ax.invert_yaxis()
    ax.grid(axis="x", color=MUTED, alpha=0.3)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}")


def clean(t: dict[str, Any], names: list[str]) -> None:
    """Rule 1: K(read) / K(W0) on a log axis, least squares against the remainder, random and do-nothing."""
    st = t["states"]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for i, n in enumerate(names):
        k = st[n]["rules"]["1"]["K"]
        c = ARM_COLOUR[st[n]["arm"]]
        ax.plot(k["moved"] / k["W0"], i, "o", color=c, ms=8, label="W0 + least squares' moved part" if i == 0 else None)
        ax.plot(k["remainder"] / k["W0"], i, "s", mfc="none", color=MUTED, ms=7,
                label="W0 + the remainder only" if i == 0 else None)
        ax.plot(k["random"] / k["W0"], i, "x", color="black", ms=7,
                label="W0 + a random change of δW's size" if i == 0 else None)
        if st[n]["rules"]["1"]["result"] == "at floor":
            ax.annotate("K under 1e-4 nats: at floor, no call", (k["moved"] / k["W0"], i), xytext=(8, 4),
                        textcoords="offset points", fontsize=7)
    ax.axvline(rules.T["clean_under"], color=LINE, ls="--", lw=1.2, label="pass line 0.02")
    ax.axvline(1.0, color=MUTED, ls=":", lw=1.2, label="do-nothing = unchanged model, 1")
    ax.set_xscale("log")
    ax.set_xlabel("KL(fine-tune ‖ read) / KL(fine-tune ‖ W0), clean forwards, every position")
    rows(ax, names, st)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
    ax.set_title("Does the clean model still act like the fine-tune once the update sits in the subcomponents?",
                 fontsize=9)
    save(fig, "clean.png")


def share(t: dict[str, Any], names: list[str]) -> None:
    """Rule 2: the share left in Δ against its null and the line null − seed spread."""
    st = t["states"]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for i, n in enumerate(names):
        r = st[n]["rules"]["2"]
        c = ARM_COLOUR[st[n]["arm"]]
        ax.plot(r["value"], i, "o", color=c, ms=8, label="share left in Δ by least squares" if i == 0 else None)
        ax.plot(r["null"], i, "D", mfc="none", color="black", ms=7,
                label="null: share a random change of the same sizes leaves" if i == 0 else None)
        ax.plot([r["null"] - r["spread"]] * 2, [i - 0.35, i + 0.35], color=LINE, lw=2,
                label="pass line: null minus the seed spread" if i == 0 else None)
        if st[n]["spread_borrowed"]:
            ax.annotate("one seed: spread borrowed", (r["null"], i), xytext=(8, -3), textcoords="offset points",
                        fontsize=7)
    ax.set_xlabel("‖δW − moved part‖² / ‖δW‖², 196 matrices; do-nothing reads 1, off the axis")
    rows(ax, names, st)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
    ax.set_title("Does least squares leave less of the update in Δ than a random change would?", fontsize=9)
    save(fig, "share.png")


def kept(fr: dict[str, float], key: str) -> float:
    """Kept on an arm whose gain is positive (every arm this figure shows)."""
    return (fr["W0"] - fr[key]) / (fr["W0"] - fr["FT"])


def kept_clean(t: dict[str, Any], names: list[str]) -> None:
    """Rule 4 and the band: kept at a tenth and a half of δW, and by least squares' moved part, no masks."""
    st = t["states"]
    show = [n for n in names if st[n]["arm"] in rules.BEHAVIOUR or st[n]["arm"] == rules.NULL_ARM]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.axvspan(-0.05, rules.T["graded_at"], color=MUTED, alpha=0.12, label="'graded' if a tenth keeps ≤ 0.5")
    for i, n in enumerate(show):
        fr = json.loads((rules.RESULTS / n / "state.json").read_text())["fractions"]
        c = ARM_COLOUR[st[n]["arm"]]
        ax.plot(kept(fr, "t0.1"), i, "^", mfc="none", color=c, ms=7, label="W0 + a tenth of δW" if i == 0 else None)
        ax.plot([kept(fr, "t0.5")] * 2, [i - 0.35, i + 0.35], color=LINE, lw=2,
                label="W0 + half of δW: pass line on graded arms" if i == 0 else None)
        ax.plot(kept(fr, "moved"), i, "o", color=c, ms=8, label="W0 + least squares' moved part" if i == 0 else None)
        print(f"kept clean {n}: tenth {kept(fr, 't0.1'):.4f} half {kept(fr, 't0.5'):.4f} moved {kept(fr, 'moved'):.4f}")
    ax.axvline(0, color=MUTED, ls=":", lw=1.2, label="do-nothing's moved part = W0, 0")
    ax.axvline(1, color="black", ls=":", lw=1, label="fine-tune, 1")
    ax.set_xlabel("behaviour kept = (L(W0) − L(read)) / (L(W0) − L(fine-tune))\nL = mean answer-token loss on the gain rows, no masks")
    rows(ax, show, st)
    ax.set_yticks(range(len(show)), [LABEL[n] + ("\n(gold tokens, not its behaviour)" if st[n]["arm"] == "null"
                                                 else "") for n in show], fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
    ax.set_title("Does the moved part keep as much behaviour as half the update does?", fontsize=9)
    save(fig, "kept_clean.png")


def masked(t: dict[str, Any], names: list[str]) -> None:
    """Rule 5: K under own masks, least squares minus do-nothing, with 2 × half-width, against the update's effect."""
    st = t["states"]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for i, n in enumerate(names):
        r, p = st[n]["rules"]["5"], st[n]["printed"]["scored"]
        c = ARM_COLOUR[st[n]["arm"]]
        ax.errorbar(r["diff"], i, xerr=2 * r["width"], fmt="o", color=c, ms=7, capsize=3,
                    label="least squares minus do-nothing; bar ± 2 row half-widths,\npasses if the bar reaches 0"
                    if i == 0 else None)
        ax.plot(p["K_W0_masked"] - p["K_do_nothing"], i, "s", mfc="none", color=MUTED, ms=7,
                label="scale: W0 masked minus do-nothing" if i == 0 else None)
    ax.axvline(0, color=MUTED, ls=":", lw=1.2, label="do-nothing, 0")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel("change in KL(fine-tune clean ‖ masked forward), nats per position\n(symmetric log axis, "
                  "linear within ±1e-5)")
    rows(ax, names, st)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    ax.set_title("Under its own masks, is least squares further from the fine-tune than doing nothing?", fontsize=9)
    save(fig, "masked.png")


def main() -> None:
    t, names = load()
    for f in (clean, share, kept_clean, masked):
        f(t, names)


if __name__ == "__main__":
    main()
