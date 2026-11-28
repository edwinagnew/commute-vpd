"""Cell P2.91 (E332): the VPD re-gating share of the known and copy fine-tunes against their answer gain, beside
E330's random draws at gain ~1 and the repeat of known draw 2 on repacked rows. Reads saved results only (no GPU).

    ~/commute-agents/fleet/cpu-run python3 experiments/share_unified/plot_gain.py     (from the clone root)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

R = Path("experiments/share_unified/results")
OUT = Path("experiments/share_unified/figures/qwen_share_vs_gain.png")
ARMS = ("known", "copy")


def load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (R / name).read_text().splitlines() if line.strip()]


def main() -> None:
    gain = load("qwen_gain.jsonl")
    draws = [r for f in ("qwen_random_vpd0.jsonl", "qwen_random_vpd1.jsonl") for r in load(f)
             if r.get("kind") == "random" and r.get("status") == "ok" and r["arm"] in ARMS]  # fmt: skip
    ft, rnd, rep = FG.METHODS["regating"], FG.METHODS["random"], FG.METHODS["neuron"]
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    data: list[dict[str, Any]] = []
    for ax, arm in zip(axes, ARMS, strict=True):
        for seed in (0, 1, 2):
            pts = sorted((r["gain"], r["share"]) for r in gain if r.get("arm") == arm and r.get("seed") == seed
                         and r["stage"] in ("full", "g10", "g1.05"))  # fmt: skip
            ax.plot(*zip(*pts, strict=True), color=ft.color, marker=ft.marker, linewidth=0.8, markersize=4,
                    label="fine-tune scaled by c, seeds 0-2" if seed == 0 else None)  # fmt: skip
            data += [{"task": arm, "kind": "fine-tune", "seed_or_draw": seed, "gain": g, "share": s} for g, s in pts]
        mine = [r for r in draws if r["arm"] == arm]
        ax.scatter([r["gain"] for r in mine], [r["share"] for r in mine], color=rnd.color, marker=rnd.marker, s=30,
                   label="random draws 0-2, rows as E330")  # fmt: skip
        data += [{"task": arm, "kind": "random", "seed_or_draw": r["draw"], "gain": r["gain"], "share": r["share"]}
                 for r in mine]  # fmt: skip
        for r in (r for r in gain if r["stage"] == "repeat" and r["arm"] == arm):
            ax.scatter([r["gain"]], [r["share"]], color=rep.color, marker="X", s=30, label="random draw 2, rows repacked")
            data.append({"task": arm, "kind": "random repacked", "seed_or_draw": 2, "gain": r["gain"], "share": r["share"]})
        ax.set_xscale("log")
        ax.set_title(f"{'known facts' if arm == 'known' else 'copy'}")
        ax.set_xlabel("answer gain, whole-answer log-odds")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("re-gating share")
    axes[1].sharey(axes[0])
    fig.suptitle("Scaling a fine-tune down to gain 1 raises the known share and lowers copy's; random draws spread wide")
    FG.save(fig, OUT, n="VPD p-9707d6ff on Qwen3-1.7B; 3 fine-tune seeds per task at full, ~10 and ~1 gain; "
            "3 random draws per task at gain ~1; 1,066 known and 1,732 copy answers", data=data)  # fmt: skip


if __name__ == "__main__":
    main()
