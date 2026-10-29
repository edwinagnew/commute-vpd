"""Figures for E46 stage 0 (regating_table), read from results/stage0b.json. Plan:
reports/plans/2026-10-28_plan_regating-line-reset.md; analysis: reports/analyses/2026-10-29_analysis_e46-regating-table.md.

  gate_gain.png   the gate gain g of each fine-tune (seeds 0, 1) against each null's g on the same task: the
                  ' http' other-target fine-tune, the flipped delta-W, the KL-matched random delta-W and the per-key
                  write, with the floor band and the random null's fail bar (0.25 x the seed-0 fine-tune's g)
  random_kl.png   the off-target KL each random delta-W reached against the fine-tune's (its target), with the
                  10% match band; hollow markers where the 10x norm cap bound
  write_rho.png   the per-key write's g over the same-size copy fine-tune's g, with the pass (0.5) and retire (0.8)
                  lines, beside the random-map and copy fine-tunes' own ratios

Usage (repo root): fleet/cpu-run .venv/bin/python experiments/regating_table/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
S: dict[str, Any] = json.loads((HERE / "results" / "stage0b.json").read_text())["states"]
P: dict[str, Any] = json.loads((HERE / "results" / "predictions.json").read_text())

INK, MUTED, BAND = "#0b0b0b", "#8a8984", "#ecebe7"
FT_C, NULL_C, FLIP_C, RAND_C, WRITE_C = "#2a78d6", "#8e44ad", "#eb6834", "#555555", "#2e8b57"
PASS_C, FAIL_C = "#2e8b57", "#c0392b"
TASKS = {"emoticon": "' :)' → 'o'", "r1_t534_v369": "' which' → ' was'", "r2_t359_v816": "' we' → ' just'",
         "ident32": "copy,\ntable A", "ident32b": "copy,\ntable B", "rand32": "random map,\ntable A",
         "rand32b": "random map,\ntable B"}
F_MIN = P["thresholds"]["F_min"]
FAIL_FRAC = P["thresholds"]["p03_fail_frac"]
OFFSET = {"ident32": (4, -13), "ident32b": (6, 4)}  # random_kl.png labels that would overlap


def gate_gain() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhspan(-F_MIN, F_MIN, color=BAND, zorder=0, label=f"floor: |g| ≤ {F_MIN} nats")
    ax.axhline(0, color=MUTED, lw=0.8)
    series = [("s0", "fine-tune, seed 0", FT_C, "o", True, -0.24), ("s1", "fine-tune, seed 1", FT_C, "o", False, -0.16),
              ("null", "fine-tune to ' http' (other target)", NULL_C, "D", True, -0.08),
              ("flipped", "fine-tune's δW, sign flipped", FLIP_C, "v", True, 0.0),
              ("random", "random δW, KL-matched (or at the 10× cap)", RAND_C, "s", True, 0.08),
              ("write", "per-key write at block 1", WRITE_C, "^", True, 0.16)]
    for i, task in enumerate(TASKS):
        for key, name, c, m, fill, dx in series:
            row = S.get(f"{task}/{key}")
            if row is None:
                continue
            ax.scatter(i + dx, row["g"], color=c if fill else "white", edgecolor=c, marker=m, s=48, zorder=3,
                       label=name if i == (0 if key != "write" else 3) else None)
        bar = FAIL_FRAC * abs(S[f"{task}/s0"]["g"])
        ax.hlines([bar, -bar], i + 0.03, i + 0.13, color=FAIL_C, lw=2, zorder=2,
                  label="random δW fail bar: ±0.25 × seed-0 g" if i == 0 else None)
    ax.set_yscale("symlog", linthresh=0.2, linscale=0.8)
    ax.set_yticks([-5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["−5", "−2", "−1", "−0.5", "−0.1", "0", "0.1", "0.5", "1", "2", "5", "10"])
    ax.set_xticks(range(len(TASKS)), list(TASKS.values()), fontsize=8.5)
    ax.axvline(2.5, color=MUTED, lw=0.6, ls=":")
    ax.set_ylabel("gate gain g: masked model's target log-odds,\nΓ from fine-tuned minus Γ from original (nats)")
    ax.set_title("Does Γ's new gating help the old subcomponents for the fine-tunes but not for the nulls?")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gate_gain.png", dpi=150)
    plt.close(fig)


def random_kl() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lo, hi = 0.02, 1.2
    xs = [lo, hi]
    ax.fill_between(xs, [x * 0.9 for x in xs], [x * 1.1 for x in xs], color=BAND, label="matched: within 10%")
    ax.plot(xs, xs, color=MUTED, lw=1, label="reached = fine-tune's KL")
    for task, name in TASKS.items():
        row = S[f"{task}/random"]
        cap = row["match"] == "cap"
        ax.scatter(row["kl_target"], row["kl_other"], color="white" if cap else RAND_C, edgecolor=RAND_C, s=55, zorder=3)
        ax.annotate(f"{name.replace(chr(10), ' ')}: {row['kl_other'] / row['kl_target']:.2f}",
                    (row["kl_target"], row["kl_other"]), xytext=OFFSET.get(task, (6, -3)), textcoords="offset points",
                    fontsize=7.5)
    ax.scatter([], [], color="white", edgecolor=RAND_C, label="hollow: scale capped at 10× the fine-tune's norm")
    ax.scatter([], [], color=RAND_C, label="filled: scale found below the cap")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("fine-tune's off-target KL to the original model (nats), the target")
    ax.set_ylabel("random δW's off-target KL at the scale used (nats)")
    ax.set_title("Did each random δW reach the fine-tune's off-target KL?\n(label: reached over target)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "random_kl.png", dpi=150)
    plt.close(fig)


def write_rho() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lo_bar, hi_bar = P["thresholds"]["p04_pass"], P["thresholds"]["p04_fail"]
    ax.axhspan(lo_bar, hi_bar, color=BAND, zorder=0, label="between the lines: inconclusive")
    ax.axhline(lo_bar, color=PASS_C, lw=1.2, ls="--", label=f"pass: write ≤ {lo_bar}")
    ax.axhline(hi_bar, color=FAIL_C, lw=1.2, ls="--", label=f"retire the line: write ≥ {hi_bar}")
    pairs = P["refs"]["p04_pair"]
    series = [("{r}/write", "random-map write (scored)", WRITE_C, "^", True, -0.15),
              ("{c}/write", "copy write (reported; adds the key's own embedding)", WRITE_C, "^", False, -0.05),
              ("{r}/s0", "random-map fine-tune, seed 0", FT_C, "s", True, 0.05),
              ("{r}/s1", "random-map fine-tune, seed 1", FT_C, "s", False, 0.1),
              ("{c}/s1", "copy fine-tune, seed 1", INK, "o", False, 0.18)]
    for i, (rand, copy) in enumerate(pairs.items()):
        ref = S[f"{copy}/s0"]["g"]
        for key, name, c, m, fill, dx in series:
            v = S[key.format(r=rand, c=copy)]["g"] / ref
            ax.scatter(i + dx, v, color=c if fill else "white", edgecolor=c, marker=m, s=60, zorder=3,
                       label=name if i == 0 else None)
            ax.annotate(f"{v:.2f}", (i + dx, v), xytext=(5, 0), textcoords="offset points", fontsize=7)
    ax.set_xticks([0, 1], ["table A (32 keys)", "table B (32 keys)"])
    ax.set_xlim(-0.4, 1.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("gate gain over the seed-0 copy fine-tune's gate gain")
    ax.set_title("Does Γ treat a target-predicting write\nlike the copy fine-tune?", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "write_rho.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    gate_gain()
    random_kl()
    write_rho()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
