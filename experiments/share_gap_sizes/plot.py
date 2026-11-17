"""P2.16: do the share's gap sizes across the nine built pairs follow the prior's or strength-1 steering's? Saved results only.

Every gap is the side built to reuse minus the side built to be new, as its source defines it:
  share     experiments/method_table/results/final.json cells (method "share", group "built"): difference of seed means.
  prior     experiments/prior_steering/results/stage1.json score.pairs[p].prior.gap: log10 prior, difference of map means.
  steering  stage1.json score.pairs[p].steering.strength1.gap and .spread: share of the gain, strength 1, E149's kind.
A spread is the wider side's range (holes_fill/rules.py `call`), the form stage1.json saves for steering. The share's is
computed here from its saved seeds by that formula. The prior's exists only where both sides have two or more maps.
Spearman rank correlations over the nine pairs are descriptive; ties take the mean rank.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/share_gap_sizes/plot.py
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from commute import figures as FG

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SHORT = {"e62_copy_lookup": "copy / lookup", "e46_copy_random": "copy / random maps", "e53_lowercase_cipher": "lowercase / ciphers",
         "e54_matched": "half-known / cipher, equal", "e54_own": "half-known / cipher, own", "open_pair_c3": "rank-1 / cipher, step 10",
         "open_pair_c3_copy": "copy / cipher, step 10", "e58_rank1_96": "rank-1 / cipher, 96 keys",
         "e58_copy_prior_cipher": "copy / matched cipher"}  # fmt: skip


def spread(reuse: Sequence[float], new: Sequence[float]) -> float | None:
    """The wider side's range, as holes_fill/rules.py `call` defines it; None with fewer than two values a side."""
    if min(len(reuse), len(new)) < 2:
        return None
    return max(max(reuse) - min(reuse), max(new) - min(new))


def ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(xs)
    return [statistics.fmean(i + 1 for i, v in enumerate(order) if v == x) for x in xs]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return statistics.correlation(ranks(a), ranks(b))


def rows() -> list[dict[str, Any]]:
    cells = json.loads((ROOT / "experiments/method_table/results/final.json").read_text())["cells"]
    cell = {(c["pair"], c["method"]): c for c in cells if c["group"] == "built"}
    pairs = json.loads((ROOT / "experiments/prior_steering/results/stage1.json").read_text())["score"]["pairs"]
    out = []
    for p, r in pairs.items():
        sh, pr, s1 = cell[(p, "share")], cell[(p, "prior")], r["steering"]["strength1"]
        assert abs(pr["gap"] - r["prior"]["gap"]) < 1e-12, p
        out.append({"pair": p, "name": r["name"], "share_gap": sh["gap"], "share_spread": spread(sh["reuse"], sh["new"]),
                    "prior_gap": r["prior"]["gap"], "prior_spread": spread(pr["reuse"], pr["new"]),
                    "steering_gap": s1["gap"], "steering_spread": s1["spread"], "steering_kind": s1["kind"]})  # fmt: skip
    return out


def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    col = {k: [r[k] for r in rs] for k in ("share_gap", "prior_gap", "steering_gap")}
    return {"n_pairs": len(rs), "spearman_share_prior": spearman(col["share_gap"], col["prior_gap"]),
            "spearman_share_steering": spearman(col["share_gap"], col["steering_gap"]),
            "share_gap_min": min(col["share_gap"]), "share_gap_max": max(col["share_gap"]),
            "share_spread_max": max(r["share_spread"] for r in rs)}  # fmt: skip


def plot(rs: list[dict[str, Any]], s: dict[str, Any]) -> None:
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    y = [r["share_gap"] for r in rs]
    yerr = [r["share_spread"] / 2 for r in rs]
    for ax, key, method, label, rho in ((axes[0], "prior", "prior", "prior gap, log10 decades", s["spearman_share_prior"]),
                                        (axes[1], "steering", "steering", "steering gap at strength 1, share of the gain", s["spearman_share_steering"])):  # fmt: skip
        st = FG.METHODS[method]
        x = [r[f"{key}_gap"] for r in rs]
        xerr = [(r[f"{key}_spread"] or 0.0) / 2 for r in rs]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=st.marker, color=st.color, ecolor="#999999", elinewidth=0.8, markersize=5)
        for r, xi, yi in zip(rs, x, y, strict=True):
            below = r["pair"] == "e54_own" or (key, r["pair"]) == ("steering", "open_pair_c3")  # labels clear of neighbours
            ax.annotate(SHORT[r["pair"]], (xi, yi), xytext=(3, -8 if below else 2), textcoords="offset points", fontsize=6, color="#333333")
        ax.set_xlabel(label)
        ax.set_title(f"rank correlation {rho:.2f}", fontsize=8, pad=12)
    FG.reference(axes[0], 0.5, "prior's call bar")
    FG.reference(axes[1], 0.0, "no gap")
    axes[0].set_ylabel("share gap, share of the gain")
    axes[1].sharey(axes[0])
    fig.suptitle("Does the share's gap size follow the prior's or steering's? Neither, across nine pairs")
    n = ("9 built pairs, one point each; gaps are reuse side minus new side. Bars span each gap's spread, the wider side's "
         "range over 2 to 8 seeds; the prior has a spread on one pair only.")  # fmt: skip
    FG.save(fig, HERE / "figures/gap_sizes.png", n=n, data=rs)


def main() -> None:
    rs = rows()
    s = summary(rs)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/gaps.json").write_text(json.dumps({"rows": rs, "summary": s}, indent=1) + "\n")
    plot(rs, s)


if __name__ == "__main__":
    main()
