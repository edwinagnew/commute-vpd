"""P1.25 run 2 figures, from results/items/*.json, results/summary.json and results/update_size.json:
    figures/retention.png  one row per pair and seed; each item's retention for the contrast, one point per order, the
                           reuse member in blue and its twin in orange. Left: level A (neurons, the filed reading).
                           Right: level B (heads and MLP blocks, reported). An item that does not read shows its reading
                           as text; the parity twin / memorised fact rows (reported, not counted) are set apart in grey.
    figures/gaps.png       level A, reuse member's mean retention minus its twin's, one point per seed, one row per pair;
                           references: the largest spread over the 5 orders of any item here, and run 1's gap between
                           the turned-up circuit (1) and the rebuilt one (experiments/rule_retention_fixed stage2.json).
    figures/size.png       per item, one minus its level A mean retention against its MLP update size (relative
                           Frobenius norm over the four layers, results/update_size.json from update_size.py).

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention_pairs/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import dots, figure, save
from commute.jobs import load_experiment

X: Any = load_experiment("rule_retention_pairs", "run")  # PAIRS, COUNTED, ITEMS
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RUN1 = HERE.parent / "rule_retention_fixed" / "results" / "stage2.json"
COLOR = {"reuse": "#0072B2", "twin": "#D55E00"}
NAMES = ("copy / cipher", "identity / random map", "case: real / twin", "repeat: real / twin", "parity twin / fact, not counted")


def rows(seeds: tuple[int, ...]) -> list[tuple[str, int, str, str]]:
    return [(f"{a}/{b} s{s}" + ("" if k < X.COUNTED else " (not counted)"), s, a, b) for k, (a, b) in enumerate(X.PAIRS) for s in seeds]


def draw(ax: Any, items: dict[str, Any], lv: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    lo, rs = 1.0, rows(seeds)
    for y, (label, s, a, b) in enumerate(rs):
        if "not counted" in label:
            ax.axhspan(y - 0.5, y + 0.5, color="#eeeeee", zorder=0)
        for side, name, dy in (("reuse", a, -0.15), ("twin", b, 0.15)):
            rec = items.get(f"{name}_s{s}")
            c: dict[str, Any] = rec[lv]["contrast"] if rec else {"reading": "not run"}
            if c["reading"] != "read":
                ax.annotate(f"{side}: {c['reading']}", (0.5, y + dy), xycoords=("axes fraction", "data"), ha="center", va="center", fontsize=6, color=COLOR[side])
                data.append({"row": label, "level": lv, "member": side, "item": f"{name}_s{s}", "reading": c["reading"]})
                continue
            ret = c["retention"]
            lo = min(lo, *ret)
            ax.scatter(ret, [y + dy] * len(ret), color=COLOR[side], s=14, zorder=3, label={"reuse": "reuse member", "twin": "its twin"}[side] if y == 0 else None)
            data += [{"row": label, "level": lv, "member": side, "item": f"{name}_s{s}", "reading": "read", "order": k, "retention": v} for k, v in enumerate(ret)]
    ax.set_yticks(range(len(rs)), [r[0] for r in rs], fontsize=6)
    ax.set_ylim(len(rs) - 0.5, -0.5)
    ax.set_xlim(lo - 0.05 * (1 - lo) - 1e-4, 1 + 0.05 * (1 - lo) + 1e-4)
    ax.set_xlabel("retention for the contrast", fontsize=7)
    ax.set_title({"A": "Level A, neurons (filed: reuse above twin)", "B": "Level B, heads and blocks (reported)"}[lv], fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    return data


def retention(items: dict[str, Any], seeds: tuple[int, ...]) -> None:
    fig, (left, right) = figure((7.5, 4.2), ncols=2)
    data = draw(left, items, "A", seeds) + draw(right, items, "B", seeds)
    fig.suptitle("Does the reuse member of each pair keep more of its rules than its twin?", fontsize=9)
    save(fig, FIGURES / "retention.png", n="one fine-tune per member and seed; 5 seeded orders of the smallest-set search per item; 512 read windows per item", data=data)


def gaps(summary: dict[str, Any], items: dict[str, Any], seeds: tuple[int, ...]) -> None:
    """Reuse minus twin at level A per pair and seed, against the order spread and run 1's rebuilt-circuit gap."""
    comps = summary["counted"] | summary["reported"]
    groups = {NAMES[k]: [comps[f"{a}/{b} s{s}"]["level A (filed)"]["difference"] for s in seeds] for k, (a, b) in enumerate(X.PAIRS)}
    spread = max(max(r) - min(r) for r in (it["A"]["contrast"]["retention"] for it in items.values()))
    run1 = 1 - json.loads(RUN1.read_text())["readings"]["A"]["contrast"]["max"]
    fig, ax = figure((6.5, 3.0))
    dots(ax, groups, method="neuron", xlabel="level A retention, reuse member minus its twin (symmetric log scale)")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.axvline(0, color="#222222", linewidth=0.8)
    for x, text in ((spread, "largest spread\nover 5 orders"), (run1, "run 1: rebuilt\nminus turned up")):
        ax.axvline(x, color="#777777", linewidth=0.8, linestyle="--")
        ax.annotate(text, (x, 0.12), xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points", fontsize=6, color="#555555", va="center")
    ax.set_xlim(-3e-5, 0.2)
    ax.set_xticks([-1e-5, 0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_title("Each counted pair orders as filed, 10 to 660 times below the gap run 1 read", fontsize=8)
    data = [{"pair": g, "seed": s, "difference": v} for g, vs in groups.items() for s, v in zip(seeds, vs, strict=True)]
    data += [{"pair": "reference", "largest_order_spread": spread, "run1_rebuilt_gap": run1}]
    save(fig, FIGURES / "gaps.png", n="2 seeds per pair (one point each), each the mean of 5 orders; one fine-tune per member and seed; 512 read windows per item", data=data)


def size(items: dict[str, Any], upd: dict[str, Any], seeds: tuple[int, ...]) -> None:
    """One minus level A retention against the MLP update size, per item."""
    fig, ax = figure((5.0, 3.4))
    data: list[dict[str, Any]] = []
    for k, pair in enumerate(X.PAIRS):
        for side, name in zip(("reuse", "twin"), pair, strict=True):
            for s in seeds:
                it = f"{name}_s{s}"
                ret = items[it]["A"]["contrast"]["retention"]
                x, y = upd["items"][it]["mlp"], 1 - sum(ret) / len(ret)
                counted = k < X.COUNTED
                ax.scatter(x, y, color=COLOR[side] if counted else "#999999", marker="o" if side == "reuse" else "s", s=22, zorder=3)
                data.append({"item": it, "member": side, "counted": counted, "mlp_update": x, "one_minus_retention": y})
    for side, mk, lab in (("reuse", "o", "reuse member"), ("twin", "s", "its twin")):
        ax.scatter([], [], color=COLOR[side], marker=mk, label=lab)
    ax.scatter([], [], color="#999999", marker="o", label="parity twin, memorised fact (not counted)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MLP update size, |dW| / |W0| over the four layers")
    ax.set_ylabel("one minus level A retention")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("Lost retention rises with the MLP update across the eight old-task fine-tunes", fontsize=9)
    save(fig, FIGURES / "size.png", n="20 fine-tunes (5 pairs, 2 members, 2 seeds); retention the mean of 5 orders on 512 read windows", data=data)


def main() -> None:
    pred = json.loads((X.PRED).read_text())
    seeds = tuple(pred["settings"]["seeds"])
    items = {p.stem: json.loads(p.read_text()) for p in X.ITEMS.glob("*.json") if not p.stem.endswith(".stopped")}
    summary = json.loads((X.RESULTS / "summary.json").read_text())
    retention(items, seeds)
    gaps(summary, items, seeds)
    size(items, json.loads((X.RESULTS / "update_size.json").read_text()), seeds)


if __name__ == "__main__":
    main()
