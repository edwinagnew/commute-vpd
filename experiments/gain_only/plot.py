"""E231's figure (cell P2.46), from results/score.json and experiments/method_table/results/final.json only (no model):
per pair, each gain-only seed's trained-answer gain at the full fine-tune's own output change, as a fraction of the full
fine-tune's gain, the side built to reuse beside the side built to be new, with the half line the filed rule reads.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/gain_only/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent / "method_table" / "results" / "final.json"
REUSE, NEW = "#0072B2", "#888888"
BAR = 0.5


def names() -> dict[str, str]:
    cells = json.loads(FINAL.read_text())["cells"]
    return {c["pair"]: c["pair_name"] for c in cells if c["method"] == "share"}


def reached(results: Path) -> dict[tuple[str, int], bool]:
    """(reference, seed) -> whether the run's KL reached the reference's (else its state is the largest KL reached)."""
    out = {}
    for p in sorted((results / "items").glob("*.json")):
        item = json.loads(p.read_text())
        out |= {(k[3:], item["seed"]): v["reached"] for k, v in item["kept"].items() if k.startswith("kl:")}
    return out


def records(score: dict[str, Any], hit: dict[tuple[str, int], bool]) -> list[dict[str, Any]]:
    out = []
    for pair, p in score["pairs"].items():
        for side in ("reuse", "new"):
            for ref, v in p[side].items():
                out += [{"pair": pair, "side": side, "reference": ref, "seed": i, "reach_kl": x, "kl_reached": hit[(ref, i)],
                         "call": p["call_kl"]} for i, x in enumerate(v["kl"]["seeds"])]
    return out


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    score = json.loads((results / "score.json").read_text())
    hit = reached(results)
    recs = records(score, hit)
    label = names()
    built = list(score["rules"]["P1_built"]["calls"])
    twins = list(score["rules"]["P1_twins"]["calls"])
    rows = [*built, None, *twins]  # None: the gap between the built pairs and the twins
    fig, ax = FG.figure((7.5, 4.6))
    for i, pair in enumerate(rows):
        if pair is None:
            continue
        p = score["pairs"][pair]
        for side, dy, color, face in (("reuse", -0.14, REUSE, REUSE), ("new", 0.14, NEW, "none")):
            xs = [x for v in p[side].values() for x in v["kl"]["seeds"]]
            ax.scatter(xs, [i + dy] * len(xs), facecolors=face, edgecolors=color, marker="o" if side == "reuse" else "s",
                       s=16, linewidths=0.8, label=("built to reuse, or real" if side == "reuse" else "built to be new, or shuffled twin") if i == 0 else None)
        short = [(r["reach_kl"], i + (-0.14 if r["side"] == "reuse" else 0.14)) for r in recs if r["pair"] == pair and not r["kl_reached"]]
        if short:
            ax.scatter(*zip(*short, strict=True), color="#D55E00", marker="x", s=10, linewidths=0.7,
                       label="run never reached the full fine-tune's KL: read at its largest" if not any(c.get_label().startswith("run never") for c in ax.collections) else None)
        call = p["call_kl"]
        ax.annotate(call, (1.0, i), xycoords=("axes fraction", "data"), xytext=(2, 0), textcoords="offset points",
                    fontsize=6, va="center", color="#222222" if call == "as filed" else "#D55E00")
    FG.reference(ax, BAR, "half the full fine-tune's gain")
    ax.axhline(len(built), color="#BBBBBB", linewidth=0.6)
    ax.set_yticks([i for i, p in enumerate(rows) if p], [label[p] for p in rows if p], fontsize=7)
    ax.set_ylim(len(rows) - 0.5, -1.7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("gain-only gain at the full fine-tune's KL, as a fraction of the full fine-tune's gain")
    ax.legend(loc="upper left", ncols=2, fontsize=6, frameon=False, handletextpad=0.2, columnspacing=0.8)
    b, t = score["rules"]["P1_built"], score["rules"]["P1_twins"]
    ax.set_title(f"Does strength alone reach half the gain on reuse sides only? On {b['as_filed']} of {b['read']} built pairs; "
                 f"{t['as_filed']} of {t['read']} twins", fontsize=8, pad=14)
    FG.save(fig, out / "reach_kl_matched.png", data=recs,
            n="3 seeds per task; a side with two tasks shows six points. Do-nothing (all gains 1) reads 0. "
              "Above the line: built pairs; below: real tasks against their shuffled-label twins. Right: the pair's call.")


if __name__ == "__main__":
    main()
