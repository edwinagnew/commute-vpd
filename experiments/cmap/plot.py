"""Cell P2.68's figure: per item that clears its gate, the whole-circuit and A+B CMAP log-odds shares on D, beside the
10 matched random sets' shares (grey, the reference on the same axis), one panel per format (shares are compared within
a format only). No GPU: ~/commute-agents/fleet/cpu-run python3 experiments/cmap/plot.py"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ITEMS, FIG = HERE / "results" / "items", HERE / "figures"
FORMATS = ("kv", "last", "next")
NAMES = {"kv": "key tables", "last": "E62 cued copy", "next": "E53 text"}


def main() -> None:
    recs = [json.loads(p.read_text()) for p in sorted(ITEMS.glob("*.json"))]
    recs = [r for r in recs if r.get("clears_gate") and "cmap" in r]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, fmt in zip(axes, FORMATS, strict=True):
        rs = [r for r in recs if r["task_rows"]["format"] == fmt]
        for i, r in enumerate(rs):
            rnd = [v["share"]["logodds"] for k, v in r["cmap"].items() if k.startswith("random") and v["share"]["logodds"] is not None]
            ax.scatter([i] * len(rnd), rnd, s=10, color="#b0afa8", zorder=1, label="random sets" if i == 0 else None)
            ax.scatter([i], [r["cmap"]["whole"]["share"]["logodds"]], s=40, color="#2a78d6", zorder=3, label="whole circuit" if i == 0 else None)
            ax.scatter([i], [r["cmap"]["AB"]["share"]["logodds"]], s=40, color="#eb6834", marker="D", zorder=2, label="A+B" if i == 0 else None)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axhline(1, color="#888", lw=0.8, ls=":")
        ax.set_xticks(range(len(rs)), [r["item"] for r in rs], rotation=90, fontsize=6)
        ax.set_title(f"{NAMES[fmt]} ({len(rs)} items clear the gate)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("CMAP log-odds share on D")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "shares.png", dpi=150)
    print(f"wrote {FIG / 'shares.png'} from {len(recs)} items")


if __name__ == "__main__":
    main()
