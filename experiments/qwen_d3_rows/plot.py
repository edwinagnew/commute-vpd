"""E94's one figure from results/table.json (CPU): each scored number per state and seed, with its row half-width, the
random changes at the two sizes as lines on the same axes. A number that reads no call is left out of its panel.

  python experiments/qwen_d3_rows/plot.py   ->  figures/d3_rows.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GROUPS = [("anchor 1", ("a1_s0", "a1_s1")), ("anchor 2\nmatched", ("a2_s42_matched", "a2_s142_matched")),
          ("anchor 2\nfull", ("a2_s42_full", "a2_s142_full")), ("anchor 2\npast line", ("a2_s42_past", "a2_s142_past"))]
PANELS = [("direct", "direct share, own rows\n(whole-question positions)"), ("outside", "outside share"),
          ("placement", "placement excess"), ("credit", "credit under masks"), ("think", "think overlap, fold on T")]
RANDOM_STYLE = {"small": ("#888888", ":"), "big": ("#444444", "--")}


def get(nums: dict, state: str, number: str) -> tuple[float | None, float | None]:
    rec = (nums.get(state) or {}).get(number) or {}
    return rec.get("value"), rec.get("half_width")


def random_value(nums: dict, size: str, number: str) -> float | None:
    """The higher of the two kinds at this size (the plan's null). Row 1 on own rows has the alive-pieces kind only, read
    on the template rows at the small size and on the TOFU rows at the big one."""
    if number == "direct":
        own = ((nums.get(f"alive_{size}") or {}).get("own_rows") or {}).get("a1" if size == "small" else "tofu") or {}
        vals = [(own.get("direct") or {}).get("value")]
    else:
        vals = [get(nums, f"{k}_{size}", number)[0] for k in ("gauss", "alive")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def main() -> None:
    nums = json.loads((HERE / "results" / "table.json").read_text())["numbers"]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 3.6))
    for ax, (number, title) in zip(axes, PANELS, strict=True):
        for gi, (_, states) in enumerate(GROUPS):
            for si, st in enumerate(states):
                v, hw = get(nums, st, number)
                if v is not None:
                    ax.errorbar(gi + (si - 0.5) * 0.25, v, yerr=hw or 0, fmt="o", color=f"C{si}", capsize=3,
                                label=f"seed {'ab'[si]}" if gi == 0 and number == "direct" else None)
        if number == "outside":
            for gi, (_, states) in enumerate(GROUPS):
                for si, st in enumerate(states):
                    nl = ((nums.get(st) or {}).get("outside") or {}).get("null")
                    if nl is not None:
                        ax.plot(gi + (si - 0.5) * 0.25, nl, "x", color="k", label="exact null" if gi == si == 0 else None)
        for size, (c, ls) in RANDOM_STYLE.items():
            rv = random_value(nums, size, number)
            if rv is not None:
                ax.axhline(rv, color=c, ls=ls, lw=1, label=f"random, {size} size")
        ax.set_xticks(range(len(GROUPS)), [g for g, _ in GROUPS], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("D3 rows on the template fine-tune (anchor 1) and the TOFU fine-tune (anchor 2), two seeds each", fontsize=10)
    fig.tight_layout()
    out = HERE / "figures" / "d3_rows.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
