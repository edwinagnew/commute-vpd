"""The research log's five figures for E46, redrawn from saved results with plain labels (no model, no runs).

  log_overview.png  per task: the gate gain of the fine-tunes (mean of two seeds, each seed a dot), the fine-tune to
                    ' http', the sign-flipped update, the random update matched on side effects, and the code-text
                    fine-tune that learned nothing (stage0b.json; random rows from amend1.json, the matched rerun)
  log_random.png    per task: the matched random update's gate gain as a share of the seed-0 fine-tune's, against
                    its noise floor (pass) and the quarter line (fail) (stage0b.json, amend1.json, predictions.json)
  log_follow.png    the matched random updates' gate gain against what each does to the real model's answer
                    (amend1.json)
  log_write.png     on the two random-map tables: the gate gain of the copy fine-tune (seed 0), the random-map
                    fine-tune (two seeds) and the hand-made write, with the write's pass and retire lines (stage0b.json)
  log_ratio.png     the stand-in ratio after the least-squares split, per table, for the fine-tunes and the
                    random-map hand writes, split into the gate-gain part and the split's own part (amend3.json)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/regating_table/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plot import (
    BAND,
    FAIL_C,
    FIG,
    FLIP_C,
    FT_C,
    HERE,
    INK,
    MUTED,
    NULL_C,
    PASS_C,
    RAND_C,
    WRITE_C,
    A,
    P,
    S,
)

NAME = {"emoticon": "emoticon", "r1_t534_v369": "' which'\nthen ' was'", "r2_t359_v816": "' we'\nthen ' just'",
        "ident32": "copy,\ntable A", "ident32b": "copy,\ntable B", "rand32": "random map,\ntable A",
        "rand32b": "random map,\ntable B"}
CODE_C = "#b08d2a"
GRID = "#e4e3df"


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)


def save(fig: Any, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def g(task: str, key: str) -> float:
    return (A if key == "random" else S)[f"{task}/{key}"]["g"]


def fig_overview() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = (("fine-tune that learned the task", FT_C), ("fine-tune to ' http' instead (lowers the answer)", NULL_C),
            ("the fine-tune's change, sign flipped", FLIP_C), ("random change, as disruptive as the fine-tune", RAND_C))
    w = 0.2
    for i, t in enumerate(NAME):
        seeds = [g(t, "s0"), g(t, "s1")]
        vals = [sum(seeds) / 2, S.get(f"{t}/null", {}).get("g"), g(t, "flipped"), g(t, "random")]
        for b, ((lab, c), v) in enumerate(zip(bars, vals, strict=True)):
            if v is None:
                continue
            x = i + (b - 1.5) * w
            ax.bar(x, v, w * 0.9, color=c, label=lab if i == 0 else None)
            if b == 0:
                ax.scatter([x - 0.04, x + 0.04], seeds, s=10, color="k", zorder=3)
                ax.text(x, max(seeds) + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    code = [S["code/step10"]["g"], S["code/step100"]["g"]]
    xc = len(NAME)
    ax.scatter([xc - 0.05, xc + 0.05], code, s=40, marker="D", color=CODE_C, zorder=3,
               label="fine-tune on code text that learned nothing (steps 10 and 100)")
    ax.text(xc, 0.4, "−0.04, −0.05", ha="center", fontsize=8)
    ax.set_xticks(range(len(NAME) + 1), [*NAME.values(), "code text"], fontsize=8.5)
    ax.axvline(2.5, color=MUTED, lw=0.6, ls=":")
    ax.axvline(6.5, color=MUTED, lw=0.6, ls=":")
    ax.set_ylabel("gate gain (nats): rise in the stand-in's\nlog-odds of the answer from Γ's new values alone")
    ax.set_title("Fine-tunes that learned read 3 to 12 nats;\n"
                 "changes that lower the answer read below zero; random changes stay within half a nat",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "log_overview.png")


def fig_random() -> None:
    th = P["thresholds"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, t in enumerate(NAME):
        ref = g(t, "s0")
        row = A[f"{t}/random"]
        f = row["F"] / ref
        ax.add_patch(Rectangle((i - 0.35, -f), 0.7, 2 * f, facecolor=BAND, edgecolor=MUTED, ls="--", lw=0.8, zorder=0,
                               label="noise floor: all inside it would pass" if i == 0 else None))
        k = abs(row["g"]) / row["F"]
        ax.text(i, 0.19, f"outside:\n{k:.2f}× floor" if k > 1 else f"inside:\n{k:.2f}× floor", ha="center",
                fontsize=8, color=INK if k > 1 else PASS_C, fontweight="bold")
        v = row["g"] / ref
        ax.bar(i, v, 0.4, color=RAND_C, label="random change, as disruptive as the fine-tune" if i == 0 else None)
        ax.text(i, v + (0.006 if v >= 0 else -0.006), f"{v:+.2f}".replace("-", "−"), ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8)
    frac = th["p03_fail_frac"]
    ax.axhline(frac, color=FAIL_C, lw=1.2, ls="--", label=f"fail line: any task at {frac} of the fine-tune's")
    ax.set_xticks(range(len(NAME)), list(NAME.values()), fontsize=8.5)
    ax.set_ylim(-0.08, 0.3)
    ax.set_ylabel("random change's gate gain as a share\nof the fine-tune's on the same task")
    ax.set_title("Random changes sit outside the noise floor on six of seven tasks, some only just,\n"
                 "and under the fail line: inconclusive",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    save(fig, "log_random.png")


def fig_follow() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lo, hi = -0.3, 0.6
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="gate gain = real change")
    ax.axvline(0, color=MUTED, lw=0.6)
    offset = {"r1_t534_v369": (6, -12), "r2_t359_v816": (-44, -12), "ident32b": (6, 2), "rand32b": (6, -10)}
    for t, name in NAME.items():
        row = A[f"{t}/random"]
        ax.scatter(row["G"], row["g"], color=RAND_C, marker="s", s=45, zorder=3)
        ax.annotate(name.replace("\n", " "), (row["G"], row["g"]), xytext=offset.get(t, (6, -3)),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("what the random change does to the real model's\nlog-odds of the answer (nats)")
    ax.set_ylabel("its gate gain in the stand-in (nats)")
    ax.set_title("Where a random change moves the real model's answer\nby over 0.1 nats, its gate gain is about that change", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_follow.png")


def fig_write() -> None:
    th = P["thresholds"]
    pairs = P["refs"]["p04_pair"]  # random-map table -> copy table of the same size
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.25
    for i, (rand, copy) in enumerate(pairs.items()):
        ref = S[f"{copy}/s0"]["g"]
        seeds = [S[f"{rand}/s0"]["g"], S[f"{rand}/s1"]["g"]]
        vals = ((ref, "copy fine-tune, seed 0 (existing behaviour)", INK),
                (sum(seeds) / 2, "random-map fine-tune (new behaviour)", FT_C),
                (S[f"{rand}/write"]["g"], "hand-made write at block 1 (no training)", WRITE_C))
        for b, (v, lab, c) in enumerate(vals):
            x = i + (b - 1) * w
            ax.bar(x, v, w * 0.9, color=c, alpha=0.35 if c == INK else 0.9, label=lab if i == 0 else None)
            ax.text(x, v + 0.2, f"{v:.1f}", ha="center", fontsize=8.5)
            if b == 1:
                ax.scatter([x - 0.04, x + 0.04], seeds, s=10, color="k", zorder=3)
        for frac, c, lab in ((th["p04_pass"], PASS_C, "pass: write at or under half the copy fine-tune's"),
                             (th["p04_fail"], FAIL_C, "retire the line: write at 0.8 of it or more")):
            ax.hlines(frac * ref, i + w * 0.55, i + w * 1.45, color=c, lw=1.5, ls="--", label=lab if i == 0 else "_nolegend_")
    ax.set_xticks([0, 1], ["table A (32 keys)", "table B (32 keys)"])
    ax.set_xlim(-0.45, 1.6)
    ax.set_ylabel("gate gain (nats)")
    ax.set_title("A hand-made write passes its test against copy,\n"
                 "and earns half to two-thirds of the random-map fine-tune's gate gain",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "log_write.png")


def fig_ratio(rows: dict[str, Any], th: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    tables = ("ident32", "ident32b", "rand32", "rand32b")
    w = 0.25
    for i, t in enumerate(tables):
        keys = [("s0", FT_C), ("s1", FT_C)] + ([("write_G", WRITE_C)] if t.startswith("rand") else [])
        for b, (k, c) in enumerate(keys):
            r = rows[f"{t}/{k}"]
            part = r["g"] / r["piece_gain"]
            x = i + (b - 1) * w
            ax.bar(x, part, w * 0.9, color=c, alpha=0.9 if k != "s1" else 0.6)
            ax.bar(x, r["ratio"] - part, w * 0.9, bottom=part, color="white", edgecolor=c, hatch="////", lw=0.6)
            ci = r["boot"]["ratio_ci"]
            ax.errorbar(x, r["ratio"], yerr=[[r["ratio"] - ci[0]], [ci[1] - r["ratio"]]], fmt="none", ecolor=INK, lw=0.8)
            ax.text(x, 0.03, f"{r['ratio']:.2f}", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.bar(0, 0, color=FT_C, label="fine-tune, seeds 0 and 1 (darker: seed 0)")
    ax.bar(0, 0, color=WRITE_C, label="hand-made write (random-map tables only)")
    ax.bar(0, 0, color="white", edgecolor=MUTED, hatch="////", label="part from the split's own changes")
    ax.axhline(th["bar_low"], color=PASS_C, lw=1.2, ls="--", label=f"a write at or under {th['bar_low']}: unlike copy")
    ax.axhline(th["bar_high"], color=FAIL_C, lw=1.2, ls="--", label=f"a write at or over {th['bar_high']}: like copy")
    ax.set_xticks(range(4), [NAME[t] for t in tables], fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("stand-in ratio: stand-in's gain over\nsubcomponent gain, after the split")
    ax.set_title("On the stand-in ratio a hand-made write reads near or above the random-map fine-tunes, below copy;\n"
                 "the ratio is almost all gate gain over subcomponent gain", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              title="solid: gate gain over subcomponent gain; black lines: 90% intervals", title_fontsize=8)
    save(fig, "log_ratio.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_overview()
    fig_random()
    fig_follow()
    fig_write()
    a3 = json.loads((HERE / "results" / "amend3.json").read_text())["rows"]
    t3 = json.loads((HERE / "results" / "predictions_amend3.json").read_text())["thresholds"]
    fig_ratio(a3, t3)


if __name__ == "__main__":
    main()
