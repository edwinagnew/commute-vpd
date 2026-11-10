"""E144 (E132 Amendment 1, a post-hoc deviation from E132's KL share): each state's mean change in the log-odds of
the trained answer, per target token, on each update's own rows, beside the two random changes on the same rows."""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/amend1"
ROWSETS = {"gold": ("e40", "gold_s0"), "facts_s0": ("a2long_s0",), "facts_s1": ("a2long_s1",)}


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text())
    data = []
    updates: dict[str, list[float]] = {}
    randoms: dict[str, list[float]] = {}
    for rowset, names in ROWSETS.items():
        for name in names:
            updates[f"{name} ({rowset} rows)"] = [s["states"][name]["gain_mean"]]
            randoms[f"{name} ({rowset} rows)"] = [s["random"][f"{r}_{rowset}"]["gain_mean"] for r in ("gauss", "inspan")]
            data.append({"state": name, "rowset": rowset, "gain_mean": s["states"][name]["gain_mean"],
                         "random_gauss": randoms[f"{name} ({rowset} rows)"][0], "random_inspan": randoms[f"{name} ({rowset} rows)"][1]})
    fig, ax = F.figure((6.5, 3.2))
    F.dots(ax, updates, method="regating", xlabel="mean change in the trained answer's log-odds per token")
    F.dots(ax, randoms, method="random", xlabel="mean change in the trained answer's log-odds per token", offset=0.25)
    F.reference(ax, 0.0, "")
    F.reference(ax, 1.0, "  scored from +1")
    ax.set_title("Updates (blue) and random changes (grey) both lower the answer's log-odds", fontsize=9)
    F.save(fig, HERE / "figures/amend1_gains.png", n="one state per row; 64 rows of 512 tokens per row set", data=data)


if __name__ == "__main__":
    main()
