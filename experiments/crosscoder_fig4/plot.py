"""Cell P2.83's figure, from saved results only (CPU): the crosscoder's d on Fig 4's 48 states beside its planted
control. Left: one column per task in Fig 4's order (built to reuse, then built new), one dot per fine-tune seed, the
three Gaussian updates as a grey band. Right: the planted control, the fine-tune-specific part of the planted change
per variant and crosscoder seed against the filed bar 0.5, with E203's miss (0.0) beside. Descriptive: no verdict.

d = shared part of the change minus its fine-tune-specific part (Minder et al.'s latents, the authors' exclusion of a
negative base beta), `crosscoder.d_forms` d_authors.

    ~/commute-agents/fleet/cpu-run --mem 4G python experiments/crosscoder_fig4/plot.py [--results DIR] [--out PNG]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, column_dots, figure, save

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[0] / "weightwatch/figures/tools_fig4.csv"  # Fig 4's states, task display names and sides (P2.80)
E203_0B = HERE.parents[0] / "crosscoder/results/run2/stage0b.json"
VARIANTS = (("random", "content-independent\npositions"), ("answer_half", "answer position,\nhalf the windows"),
            ("answer", "answer position\n(E203's), 4x tokens"))  # fmt: skip


def fig4_layout() -> tuple[dict[str, tuple[str, str]], list[str]]:
    """state -> (task display name, side), and the task order: reuse tasks, then new, each in the CSV's order."""
    rows = [r for r in csv.DictReader(CSV.open()) if r["reading"] == "share"]
    state = {r["state"]: (r["task"], r["side"]) for r in rows}
    order = list(dict.fromkeys(r["task"] for r in rows if r["side"] == "reuse"))
    return state, order + list(dict.fromkeys(r["task"] for r in rows if r["side"] == "new"))


def load(results: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sides = [json.loads(p.read_text()) for p in sorted((results / "sides").glob("*.json"))]
    ctl = [json.loads(p.read_text()) for p in sorted(results.glob("stage0b_*.json"))]
    return sides, ctl


def num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) else math.nan


def draw(results: Path, out: Path) -> None:
    state, order = fig4_layout()
    sides, ctl = load(results)
    by_task: dict[str, list[float]] = {t: [] for t in order}
    records: list[dict[str, Any]] = []
    gauss = [num(s["d_authors"]) for s in sides if s["kind"] == "gaussian"]
    for s in sides:
        if s["kind"] == "state" and s["state"] in state:
            by_task[state[s["state"]][0]].append(num(s["d_authors"]))
        records.append({"panel": "d", "key": s["key"], "state": s["state"], "kind": s["kind"], "d_authors": s["d_authors"],
                        "specific": s["specific"], "shared_frac": s["shared_frac"], "d_alt": s["d_alt"]})  # fmt: skip
    fig, (ax, bx) = figure(FULL_WIDTH, ncols=2)
    fig.set_layout_engine(None)
    ax.set_position((0.11, 0.36, 0.56, 0.54))
    bx.set_position((0.78, 0.36, 0.2, 0.54))
    band = (min(gauss), max(gauss)) if gauss else None
    column_dots(ax, by_task, method="spectrum", ylabel="crosscoder d\n(shared minus fine-tune-specific part)",
                bands={t: band for t in order} if band else None)  # fmt: skip
    n_reuse = len({state[s][0] for s in state if state[s][1] == "reuse"})
    ax.axvline(n_reuse - 0.5, color="#999999", linewidth=0.8)
    ax.set_xticklabels(order, rotation=70, ha="right", fontsize=6)
    ax.set_title("Crosscoder d on Fig 4's states (dots: fine-tune seeds; grey: Gaussian updates)", loc="left", fontsize=8)
    ax.text(n_reuse / 2 - 0.5, 0.97, "built to reuse", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.text(n_reuse + (len(order) - n_reuse) / 2 - 0.5, 0.97, "built new", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.08)
    # the control: specific part per variant, each crosscoder seed a point; E203's miss beside
    labels = ["E203\n(answer position)", *[lab for _, lab in VARIANTS]]
    e203 = json.loads(E203_0B.read_text())["checks"]["b planted: found, specific and carried"]
    groups: dict[str, list[float]] = {labels[0]: [float(e203["part"])]}
    for where, lab in VARIANTS:
        groups[lab] = [num(c["planted"]["specific"]) for c in ctl if c["where"] == where]
    for c in ctl:
        lead = c["planted"].get("lead", {})
        records.append({"panel": "control", "key": c["planted"]["key"], "where": c["where"], "seed": c["seed"],
                        "specific": c["planted"]["specific"], "nu_eps": lead.get("nu_eps"), "nu_r": lead.get("nu_r"),
                        "lead_rn": lead.get("rn"), "pass": c["planted_pass"]})  # fmt: skip
    records.append({"panel": "control", "key": "E203 planted", "where": "answer", "seed": 0, "specific": e203["part"],
                    "nu_eps": e203["lead"]["nu_eps"], "nu_r": e203["lead"]["nu_r"], "lead_rn": e203["lead"]["rn"], "pass": False})  # fmt: skip
    column_dots(bx, groups, method="random", ylabel="planted change's part on\nfine-tune-specific latents")
    bx.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    bx.annotate("filed bar 0.5", (0, 0.5), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", fontsize=6, color="#555555")
    bx.set_ylim(-0.05, 1.05)
    bx.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    bx.set_title("Planted control", loc="left", fontsize=8)
    n = f"{sum(len(v) for v in by_task.values())} states (16 tasks x 3 fine-tune seeds), one crosscoder each; {len(gauss)} Gaussian updates; control: one crosscoder per point"
    save(fig, out, n=n, data=records)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=HERE / "figures/crosscoder_fig4.png")
    a = ap.parse_args()
    draw(a.results, a.out)


if __name__ == "__main__":
    main()
