"""The research log's overview figure: component training on every task it was scored on under VPD's adversarial check.

Reads E11 (`reading_a_coverage`, five ladder tasks and the emoticon rerun) and E10 (`kv_task`, three cipher tasks).
Writes docs/research_log/component_training_tasks.png.

Usage (repo root):  uv run python docs/research_log/plot_overview.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[2]
E11 = ROOT / "experiments" / "reading_a_coverage" / "results"
E10 = ROOT / "experiments" / "kv_task" / "results" / "kv_task.json"
OUT = Path(__file__).resolve().parent / "component_training_tasks.png"
LINE = 1.5  # the pass line inherited from E2
GOOD, BAD, MUTED, INK = "#1baf7a", "#eb6834", "#52514e", "#0b0b0b"
LADDER = {"r0_t352_v310": '" it" → " is"', "r1_t534_v369": '" which" → " was"', "r1_t588_v452": '" will" → " have"',
          "r2_t275_v2667": '" in" → " Fig"', "r2_t359_v816": '" we" → " just"', "emoticon": 'emoticon ":" → "o"'}
CIPHER = {"ident32": "cipher: copy (control)", "rand1": "cipher: 1 new pair", "rand32": "cipher: 32 new pairs"}


def ladder_rows() -> list[tuple[str, float, float]]:
    """(label, component training's adversarial ratio, do-nothing's) at step 10, seed 0."""
    rows = []
    for name in ("coverage_emoticon.json", "coverage.json"):
        for task, t in json.loads((E11 / name).read_text())["tasks"].items():
            s = t["seeds"]["0"]
            rows.append((LADDER[task], s["pullback"]["pile"]["ratio"]["pgd_kl"], s["do_nothing"]["pile"]["ratio"]["pgd_kl"]))
    return rows


def cipher_rows() -> list[tuple[str, float, float]]:
    tasks = json.loads(E10.read_text())["tasks"]
    return [(CIPHER[k], t["pullback"]["eval"]["pile"]["ratios"]["pgd_kl"], t["do_nothing"]["pile"]["ratios"]["pgd_kl"])
            for k, t in tasks.items()]


def main() -> None:
    rows = sorted(ladder_rows() + cipher_rows(), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(rows))
    for i, (_, ct, dn) in enumerate(rows):
        ax.plot([ct, dn], [i, i], color="#d8d8d4", lw=1, zorder=1)
        ax.scatter(dn, i, s=46, facecolor="white", edgecolor=MUTED, lw=1.2, zorder=2)
        ax.scatter(ct, i, s=60, color=GOOD if ct < LINE else BAD, zorder=3)
    ax.axvline(LINE, color=INK, ls="--", lw=1)
    ax.text(LINE * 1.02, len(rows) - 0.4, "pass line 1.5", fontsize=8.5, va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 5, 8])
    ax.set_xticklabels(["1", "1.5", "2", "3", "5", "8"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("adversarial ratio on Pile (1 = as faithful as the original model; lower is better)")
    ax.scatter([], [], color=GOOD, s=60, label="component training, passes")
    ax.scatter([], [], color=BAD, s=60, label="component training, fails")
    ax.scatter([], [], facecolor="white", edgecolor=MUTED, s=46, label="same task's fine-tune, update left in Δ")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Component training learns all nine tasks; its decomposition stays faithful on four", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
