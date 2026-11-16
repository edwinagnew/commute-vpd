"""E191 (cell P1.22), E184 re-read as curves: closure against number of names, from E184's saved item files; no new search, no
rungs, no bars. The fourth reuse wording stays tested, not adopted. Page: reports/analyses/2026-11-16_analysis_e184-curves.md.

Each item file (experiments/simplest_intervention/results/<item>.json) holds, per list (all, neurons, directions), two greedy
curves: key "1" adds names that shift or rescale one of M's variables, key "2" may also add a read of another name; key "3"
holds the free-direction fits, left out here. The curve read here is, at each n, the best closure either greedy pass found
with at most n names (the empty intervention closes 0), up to the largest n either pass reached.

    ~/commute-agents/fleet/cpu-run python experiments/e184_curves/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "simplest_intervention" / "results"
OUT = HERE / "results" / "curves.json"
FIG = HERE / "figures" / "curves.png"
CAP = 16  # E184's cap on names per greedy pass
LISTS = ("all", "neurons", "directions")
SEEDS = (0, 1)
PAIRS = (("copy24", "cipher24", "copy (reuse) against cipher (new)"),
         ("ident32", "rand32", "identity (reuse) against random map (new)"),
         ("case_real", "case_twin", "case (reuse) against its twin (new)"),
         ("repeat_real", "repeat_twin", "repeat (reuse) against its twin (new)"),
         ("twin", "fact", "parity twin (reuse) against memorised fact (new)"))  # fmt: skip
FLOOR = ("l4_bit", "xor", "floor")  # random bit per window; parity of four token bits; random bit per number pair
APPARATUS = ("do_nothing", "l1_neuron", "l2_triggered", "planted_constant", "planted_trigger", "planted_fact", "always_on", "cue", "table")
REUSE, NEW = "#0072B2", "#D55E00"
LIGHT = {"neurons": ":", "directions": "--"}


def best_by_n(rec: dict[str, Any], lst: str) -> dict[str, Any] | None:
    """Best closure found with at most n names, n = 0..CAP, over both greedy passes; None if the list was not searched."""
    passes = rec.get("lists", {}).get(lst)
    if passes is None:
        return None
    pts = [(e["n"], e["closure"]) for r in ("1", "2") for e in passes.get(r, {}).get("curve", [])]
    best = [0.0] * (CAP + 1)
    for n, c in pts:
        best[n] = max(best[n], c)
    for n in range(1, CAP + 1):
        best[n] = max(best[n], best[n - 1])
    return {"best": best, "last_n": max((n for n, _ in pts), default=0)}


def item_names() -> list[str]:
    pairs = [f"{side}_s{s}" for a, b, _ in PAIRS for s in SEEDS for side in (a, b)]
    return pairs + list(FLOOR) + list(APPARATUS)


def read_items() -> dict[str, Any]:
    out = {}
    for name in item_names():
        rec = json.loads((SRC / f"{name}.json").read_text())
        assert rec.get("done"), name
        out[name] = {"kl": rec["kl"], "lists": {lst: best_by_n(rec, lst) for lst in LISTS}}
    return out


def floor_band(items: dict[str, Any], lst: str) -> dict[str, list[float]]:
    """Per n, the lowest and highest of the unpredictable controls' curves on list `lst`."""
    cols = list(zip(*(items[f]["lists"][lst]["best"] for f in FLOOR), strict=True))
    return {"lo": [min(c) for c in cols], "hi": [max(c) for c in cols]}


def summarise(items: dict[str, Any]) -> dict[str, Any]:
    band = {lst: floor_band(items, lst) for lst in LISTS}
    for rec in items.values():
        for lst, cur in rec["lists"].items():
            if cur is not None:  # how many n in 1..CAP sit above the controls' highest curve on the same list
                cur["above_floor"] = sum(cur["best"][n] > band[lst]["hi"][n] for n in range(1, CAP + 1))
    return {"cap": CAP, "source": str(SRC.relative_to(HERE.parents[1])), "floor_items": list(FLOOR), "floor_band": band, "items": items}


def panel(ax: Any, items: dict[str, Any], band: dict[str, list[float]], pair: tuple[str, str], seed: int) -> list[dict[str, Any]]:
    ns = list(range(CAP + 1))
    ax.fill_between(ns, band["lo"], band["hi"], color="#BBBBBB", alpha=0.6, linewidth=0)
    rows = []
    for side, color in zip(pair, (REUSE, NEW), strict=True):
        name = f"{side}_s{seed}"
        for lst in LISTS:
            cur = items[name]["lists"][lst]
            k = cur["last_n"]
            style = {"color": color, "linewidth": 1.8} if lst == "all" else {"color": color, "alpha": 0.5, "linewidth": 1.0, "linestyle": LIGHT[lst]}
            ax.plot(ns[: k + 1], cur["best"][: k + 1], **style)
            rows += [{"item": name, "list": lst, "n": n, "closure": cur["best"][n]} for n in ns[: k + 1]]
    ax.set_xlim(0, CAP)
    ax.set_ylim(0, 1)
    return rows


def draw(items: dict[str, Any], band: dict[str, list[float]], path: Path, n_text: str) -> None:
    fig, axes = figure((7.5, 11.5), ncols=len(SEEDS), nrows=len(PAIRS))
    data = [{"item": "floor band (list all)", "list": "all", "n": n, "lo": lo, "hi": hi} for n, (lo, hi) in enumerate(zip(band["lo"], band["hi"], strict=True))]
    for r, (a, b, label) in enumerate(PAIRS):
        for c, s in enumerate(SEEDS):
            ax = axes[r][c]
            data += panel(ax, items, band, (a, b), s)
            ax.set_title(f"{label}, seed {s}", fontsize=8)
            ax.set_ylabel("closure")
            ax.set_xlabel("number of names")
    handles = [axes[0][0].plot([], [], color=REUSE, linewidth=1.8)[0], axes[0][0].plot([], [], color=NEW, linewidth=1.8)[0],
               axes[0][0].plot([], [], color="#555555", linestyle=":", alpha=0.6)[0], axes[0][0].plot([], [], color="#555555", linestyle="--", alpha=0.6)[0],
               axes[0][0].fill_between([], [], [], color="#BBBBBB", alpha=0.6)]  # fmt: skip
    axes[2][1].legend(handles, ["built to reuse, list all", "built to be new, list all", "neurons only (lighter)", "directions only (lighter)",
                                "unpredictable controls, list all"], loc="lower right", fontsize=7, frameon=False)  # fmt: skip
    fig.suptitle("Where do the trained pairs' closure curves separate, against the unpredictable controls?")
    save(fig, path, n=n_text, data=data)


def main() -> None:
    summary = summarise(read_items())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1) + "\n")
    n_text = ("2 seeds per side, one greedy search per item and list (E184), closure on each item's held-out windows. "
              "Grey band: lowest to highest of the random bit per window, XOR and the random bit per number pair.")  # fmt: skip
    draw(summary["items"], summary["floor_band"]["all"], FIG, n_text)
    print(f"wrote {OUT} and {FIG}")


if __name__ == "__main__":
    main()
