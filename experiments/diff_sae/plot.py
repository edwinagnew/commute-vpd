"""P2.75 run 1's figure: each side's specific part (spec 3.6) under both forms of nu, against the gate's common bar.

    python experiments/diff_sae/plot.py [--tag run1]     -> figures/<tag>_specific.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
KEYS = ("plant.A.c0", "head-r1.A.c0", "head-r1.A.c1", "random_d0-r1.A.c0", "random_d1-r1.A.c0", "head-r5.A.c0", "head.B.c0", "gauss.B.c0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    a = ap.parse_args()
    res = HERE / "results" / a.tag
    gate = json.loads((res / "gate.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for j, (form, mk) in enumerate((("lit", "o"), ("D", "s"))):
        xs, ys = [], []
        for i, k in enumerate(KEYS):
            p = res / "items" / f"{k}.json"
            d = json.loads(p.read_text()) if p.is_file() else {}
            if isinstance(d.get("read"), dict):
                xs.append(i + 0.15 * (j - 0.5))
                ys.append(d["read"][form]["published_0.3"]["specific"])
        ax.scatter(xs, ys, marker=mk, label=f"nu_{form}")
    b = gate["D_published"]["bar"]
    if b is not None:
        ax.axhline(b, color="k", lw=0.8, ls="--", label=f"gate bar {b:.2f}")
    ax.set_xticks(range(len(KEYS)), KEYS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("specific part of the change")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"{a.tag}_specific.png", dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
