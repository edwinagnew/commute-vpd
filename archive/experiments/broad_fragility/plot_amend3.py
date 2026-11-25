"""Figure for E39 Amendment 3, read from results/amend3.json (the Adam and two SGD teacher-trained decompositions) and
results/amend2.json (the two LoRA teacher-trained decompositions, same card, same rows, same unchanged-model readings),
with the per-block readings parsed from the two jobs' logs (they are not in the JSON).
Plan: reports/plans/2026-10-27_plan_broad-fragility.md -> "Amendment 3 (2026-10-29)";
analysis: reports/analyses/2026-10-29_analysis_e39-amendment3.md.

  amend3_rows.png  left: the check ratio of each teacher-trained decomposition on the first rows (Pile val 0-255) and
                   on the second rows (val 256-511), with the pass and fail lines, their midpoint 1.5 and the unchanged
                   model's spread over adversary seeds; right: the same states per block of 128 rows, each block over
                   the unchanged model's same block at adversary seed 0

Prints the per-block ratios quoted in the analysis.

Usage (repo root): ~/commute-agents/fleet/cpu-run uv run python experiments/broad_fragility/plot_amend3.py [LOG_DIR]
LOG_DIR holds amend2.log and amend3.log (default results/large).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import (
    ADAM_C,
    BAND,
    CHECK_X,
    FAIL_C,
    FIG,
    GRID,
    HERE,
    INCONCL,
    INK,
    LORA_C,
    MUTED,
    PASS_C,
    SGD_C,
    style,
)
from plot_amend2 import R8_C

# (label, results file, key on the first rows, colour); the second rows' key is the first's + "_rows2"
STATES = [("Adam fine-tune", "amend3", "trained", ADAM_C),
          ("plain SGD fine-tune", "amend3", "tt_sgd", SGD_C),
          ("SGD held to rank 8", "amend3", "tt_sgd_r8", R8_C),
          ("rank-8 LoRA, seed 0\n(Amendment 2)", "amend2", "tt_lora_s0", LORA_C),
          ("rank-8 LoRA, seed 1\n(Amendment 2)", "amend2", "tt_lora_s1", LORA_C)]
W0_LABELS = {"amend3": ("pile W0 seed 0", "pile W0 seed 0, val rows 256..512"),
             "amend2": ("pile W0 seed 0", "rows2 W0 seed 0")}
FAITH = re.compile(r"\[faith\] .* pgd_kl ([\d.]+) pgd_rows 128")
LABEL = re.compile(r"^\S+ \[([^\]]+)\]")
BLOCKS = [("rows 0-127", "o", True), ("rows 128-255", "s", True), ("rows 256-383", "o", False),
          ("rows 384-511", "s", False)]


def block_kls(log: Path) -> dict[str, list[float]]:
    """Each labelled read's per-block PGD KL: the [faith] lines just before the label line."""
    out: dict[str, list[float]] = {}
    pending: list[float] = []
    for line in log.read_text().splitlines():
        if m := FAITH.search(line):
            pending.append(float(m.group(1)))
        elif (m := LABEL.match(line)) and pending:
            out[m.group(1)] = pending
            pending = []
    return out


def per_block(job: str, key: str, kls: dict[str, list[float]]) -> list[float]:
    """The four block ratios: first rows' two blocks, then the second rows', each over W0's same block at seed 0."""
    w1, w2 = (kls[w] for w in W0_LABELS[job])
    s1, s2 = kls[key], kls[key + "_rows2"]
    assert len(w1) == len(w2) == len(s1) == len(s2) == 2, key
    return [a / b for a, b in zip(s1 + s2, w1 + w2)]


def main(log_dir: Path) -> None:
    res = HERE / "results"
    r = {j: json.loads((res / f"{j}.json").read_text()) for j in ("amend2", "amend3")}
    kls = {j: block_kls(log_dir / f"{j}.log") for j in r}
    cfg = json.loads((res / "predictions.json").read_text())["config"]
    pass_l, fail_l = cfg["pass_line"], cfg["fail_line"]
    sp1, sp2 = r["amend3"]["stage0"]["spread"], r["amend3"]["rows2"]["spread"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, sharex=True)
    for ax in (a1, a2):
        ax.axvspan(1, 1 + max(sp1, sp2), color=BAND, lw=0,
                   label=f"unchanged model over 3 adversary seeds:\nspread {sp1:.3f} first rows, {sp2:.3f} second")
        ax.axvspan(pass_l, fail_l, color=INCONCL, lw=0, label="between the lines: inconclusive")
        ax.axvline(pass_l, color=PASS_C, ls="--", lw=0.9, label=f"pass line, {pass_l}")
        ax.axvline(1.5, color=MUTED, ls=":", lw=0.9, label="1.5, midway")
        ax.axvline(fail_l, color=FAIL_C, ls="--", lw=0.9, label=f"fail line, {fail_l}")
    print(f"{'state':22} {'first':>6} {'second':>6}   " + " ".join(f"{b:>12}" for b, _, _ in BLOCKS))
    for i, (name, job, key, c) in enumerate(STATES):
        y = -i
        v1, v2 = r[job]["reads"][key]["check"], r[job]["reads"][key + "_rows2"]["check"]
        a1.plot([v1, v2], [y, y], color=c, lw=1, zorder=4)
        a1.scatter(v1, y, s=70, color=c, zorder=5, label="first rows, Pile val 0-255" if i == 0 else None)
        a1.scatter(v2, y, s=70, facecolor="white", edgecolor=c, lw=1.6, zorder=5,
                   label="second rows, Pile val 256-511" if i == 0 else None)
        for v, dy in ((v1, 7), (v2, -12)):
            a1.annotate(f"{v:.2f}", (v, y), xytext=(4, dy), textcoords="offset points", fontsize=7)
        blocks = per_block(job, key, kls[job])
        for (b, m, filled), v in zip(BLOCKS, blocks):
            a2.scatter(v, y, s=55, marker=m, zorder=5, facecolor=c if filled else "white", edgecolor=c, lw=1.4,
                       label=b + (", first rows" if filled else ", second rows") if i == 0 else None)
        print(f"{name.splitlines()[0]:22} {v1:6.3f} {v2:6.3f}   " + " ".join(f"{v:12.2f}" for v in blocks))
    a1.set_yticks([-i for i in range(len(STATES))], [s[0] for s in STATES], fontsize=8)
    a1.set_ylim(-len(STATES) + 0.4, 0.6)
    a1.set_xlim(0.9, 3.2)
    a1.set_xlabel(CHECK_X + " (256 rows)")
    a2.set_xlabel("the same ratio per block of 128 rows,\nover the unchanged model's same block")
    a1.set_title("Teacher-trained decompositions read on two sets of 256 Pile rows:\n"
                 "both SGD ones fail on the first rows and pass on the second; Adam fails on both",
                 fontsize=9, loc="left")
    a2.set_title("Per block: the SGD fails sit on rows 128-255 alone;\nAdam reads over the fail line on all four blocks",
                 fontsize=9, loc="left")
    a1.legend(fontsize=6.5, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)
    h, lab = a2.get_legend_handles_labels()
    a2.legend(h[5:], lab[5:], fontsize=6.5, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID)
    for ax in (a1, a2):
        style(ax)
        ax.tick_params(colors=INK)
    fig.tight_layout()
    fig.savefig(FIG / "amend3_rows.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "large")
