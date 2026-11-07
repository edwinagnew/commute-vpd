"""E130's figure: for the four lowercase teacher-trained copies, each copy's adversarial ratio with its change on the
alive subcomponents (as teacher training placed it) and with the same change moved into Delta, beside the copied
fine-tune's do-nothing and E39's fail line.

    python experiments/copy_protection/plot.py   -> figures/placement.png

Sources: results/copy_protection.json (c1, c2), ../ls_start_faith/results/ls_start_faith_amend2.json (standard and
LoRA copies), ../damage_crossed/results/damage_crossed.json (do-nothing of the standard copy, c1, c2) and
../broad_fragility/results/broad_fragility.json (the LoRA's do-nothing, read on an H100, not the L40).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FAIL, PASS = 1.61, 1.40  # E39's lines, as E122 and E127
PLACED, IN_DELTA, NOTHING = "#1baf7a", "#eb6834", "#2a78d6"
INK, MUTED = "#2b2b2b", "#8a8a85"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def copies() -> list[tuple[str, float, float, float, bool]]:
    """(label, placed ratio, in-Delta ratio, do-nothing ratio, do-nothing read on the L40) per copy."""
    a2 = load(EXP / "ls_start_faith" / "results" / "ls_start_faith_amend2.json")["score"]["ratio"]
    e130 = load(HERE / "results" / "copy_protection.json")["score"]["ratio"]
    dc = load(EXP / "damage_crossed" / "results" / "damage_crossed.json")["score"]["ratios"]
    lora_nothing = load(EXP / "broad_fragility" / "results" / "broad_fragility.json")["reads"]["nothing_lora"]["check"]
    return [
        ("full Adam,\nstep 300", a2["std_copy"], a2["std_copy|delta"], dc["nothing_adam300|A"]["ratio"], True),
        ("rank-8 LoRA,\nstep 300", a2["lora_copy"], a2["lora_copy|delta"], lora_nothing, False),
        ("full Adam,\nstep 20 (new)", e130["c1"], e130["c1|delta"], dc["nothing_c1|A"]["ratio"], True),
        ("rank-8 LoRA,\nstep 2,400 (new)", e130["c2"], e130["c2|delta"], dc["nothing_c2|A"]["ratio"], True),
    ]


def main() -> None:
    rows = copies()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.text(3.45, 1.02, "W0 = 1", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.axhline(FAIL, color=INK, lw=1, ls="--")
    ax.text(3.45, FAIL + 0.03, "fail line 1.61", fontsize=7, color=INK, va="bottom", ha="right")
    for i, (_, placed, delta, nothing, l40) in enumerate(rows):
        ax.hlines(nothing, i - 0.3, i + 0.3, color=NOTHING, lw=2, ls="-" if l40 else ":")
        ax.annotate("", (i + 0.08, placed), (i - 0.08, delta),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
        ax.plot(i - 0.08, delta, "o", ms=9, color=IN_DELTA, mec="white", mew=1.5, zorder=3)
        ax.plot(i + 0.08, placed, "o", ms=9, color=PLACED, mec="white", mew=1.5, zorder=3)
        m = (placed - 1) / (delta - 1)
        ax.text(i, 5.15, f"m = {m:.2f}", ha="center", fontsize=8, color=INK)
    ax.plot([], [], "o", color=IN_DELTA, label="copy's change moved into Delta")
    ax.plot([], [], "o", color=PLACED, label="copy as placed on the subcomponents")
    ax.plot([], [], color=NOTHING, lw=2, label="do-nothing: the fine-tune's own change in Delta")
    ax.plot([], [], color=NOTHING, lw=2, ls=":", label="do-nothing read on an H100, not the L40")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.8, 5.4)
    ax.set_xlabel("lowercase fine-tune that teacher training copied")
    ax.set_ylabel("adversarial ratio over W0, Pile rows 0-255")
    ax.set_title("Does teacher training's placement shrink each copy's excess?\n"
                 "m = excess placed / excess in Delta; under 1 means placement protects", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "placement.png", dpi=150)


if __name__ == "__main__":
    main()
