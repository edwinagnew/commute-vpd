"""Cell P2.83's figure, from saved results only (CPU): the crosscoder's d on Fig 4's 48 states beside its planted
control, under both exclusions of negative base betas. Left: one column per task in Fig 4's order (built to reuse,
then built new), one point per fine-tune seed (filled: the literal App. E.2 exclusion; open, shifted right: the
floored exclusion), the three Gaussian updates as a grey band (literal). Right: the planted control, the
fine-tune-specific part of the planted change per variant and crosscoder seed under each rule, against the filed bar
0.5, with E203's miss (literal, 0.0) beside. A stage0b file that stopped at an early check is drawn as missing and
listed in the CSV. Descriptive: no verdict.

A second figure, `crosscoder_fig4_lengths.png` (`draw_lengths`, run 3 / E313), puts each state's d at 200M (results/)
and 800M (results/long/) side by side in its task's column, the Gaussian updates as a column on the same axes, and the
planted control at both lengths, literal (filled) beside floored (open).

d = shared part of the change minus its fine-tune-specific part (`crosscoder.d_forms` d_authors).

    ~/commute-agents/fleet/cpu-run --mem 4G python experiments/crosscoder_fig4/plot.py [--results DIR] [--out PNG]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute.figures import FULL_WIDTH, column_dots, figure, save

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[0] / "weightwatch/figures/tools_fig4.csv"  # Fig 4's states, task display names and sides (P2.80)
E203_0B = HERE.parents[0] / "crosscoder/results/run2/stage0b.json"
VARIANTS = (("random", "content-independent\npositions"), ("answer_half", "answer position,\nhalf the windows"),
            ("answer", "answer position\n(E203's), 4x tokens"))  # fmt: skip
RULES = ("literal", "floored")


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


def reading(r: dict[str, Any], rule: str) -> dict[str, Any]:
    return r if rule == "literal" else r.get("floored", {})


def open_points(ax: Axes, groups: dict[str, list[float]], dx: float, label: str) -> None:
    """The floored rule's points beside `column_dots`' filled ones, open markers shifted by dx."""
    for i, vs in enumerate(groups.values()):
        ys = [v for v in vs if not math.isnan(v)]
        ax.scatter([i + dx] * len(ys), ys, facecolors="none", edgecolors="#000000", marker="o", s=20, linewidths=0.8,
                   label=label if i == 0 else None)  # fmt: skip


def d_panel(ax: Axes, sides: list[dict[str, Any]], state: dict[str, tuple[str, str]], order: list[str], records: list[dict[str, Any]]) -> tuple[int, int]:
    by: dict[str, dict[str, list[float]]] = {rule: {t: [] for t in order} for rule in RULES}
    gauss = [num(s["d_authors"]) for s in sides if s["kind"] == "gaussian"]
    for s in sides:
        for rule in RULES:
            r = reading(s, rule)
            if s["kind"] == "state" and s["state"] in state:
                by[rule][state[s["state"]][0]].append(num(r.get("d_authors")))
            records.append({"panel": "d", "rule": rule, "key": s["key"], "state": s["state"], "kind": s["kind"],
                            "d_authors": r.get("d_authors"), "specific": r.get("specific"), "shared_frac": r.get("shared_frac"),
                            "d_alt": r.get("d_alt")})  # fmt: skip
    band = (min(gauss), max(gauss)) if gauss else None
    column_dots(ax, by["literal"], method="spectrum", ylabel="crosscoder d\n(shared minus fine-tune-specific part)",
                bands={t: band for t in order} if band else None)  # fmt: skip
    open_points(ax, by["floored"], 0.25, "floored exclusion")
    for s in (s for s in sides if s["kind"] == "reseed" and s["state"] in state):  # crosscoder seed 1 on a seed-0 state
        ax.scatter([order.index(state[s["state"]][0]) - 0.25], [num(s["d_authors"])], marker="*", color="#c44e00", s=40,
                   linewidths=0, label="fine-tune seed 0, crosscoder seed 1")  # fmt: skip
    ax.scatter([], [], color="#000000", marker="h", s=22, label="one fine-tune seed (literal exclusion)")
    ax.legend(fontsize=6, loc="lower left", ncols=2, frameon=False)
    n_reuse = len({state[s][0] for s in state if state[s][1] == "reuse"})
    ax.axvline(n_reuse - 0.5, color="#999999", linewidth=0.8)
    ax.set_xticklabels(order, rotation=70, ha="right", fontsize=6)
    ax.set_title("Crosscoder d on Fig 4's states (filled: literal exclusion; open: floored; grey: Gaussian updates)", loc="left", fontsize=7)
    ax.text(n_reuse / 2 - 0.5, 0.97, "built to reuse", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.text(n_reuse + (len(order) - n_reuse) / 2 - 0.5, 0.97, "built new", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set_ylim(min(ax.get_ylim()[0], 0) - 0.12, ax.get_ylim()[1] + 0.08)  # room for the legend under the points
    return sum(len(v) for v in by["literal"].values()), len(gauss)


def control_panel(bx: Axes, ctl: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    labels = ["E203\n(answer position)", *[lab for _, lab in VARIANTS]]
    e203 = json.loads(E203_0B.read_text())["checks"]["b planted: found, specific and carried"]
    by: dict[str, dict[str, list[float]]] = {rule: {lab: [] for lab in labels} for rule in RULES}
    by["literal"][labels[0]] = [float(e203["part"])]
    ran = [c for c in ctl if "planted" in c]
    for c in ctl:
        if "planted" not in c:  # stopped at an early check: missing
            records.append({"panel": "control", "key": f"stage0b {c['where']} s{c['seed']}", "where": c["where"], "seed": c["seed"], "stopped": c.get("stopped")})
    for (where, lab), rule in ((v, r) for v in VARIANTS for r in RULES):
        by[rule][lab] = [num(reading(c["planted"], rule).get("specific")) for c in ran if c["where"] == where]
    for c, rule in ((c, r) for c in ran for r in RULES):
        r, lead = reading(c["planted"], rule), reading(c["planted"], rule).get("lead", {})
        records.append({"panel": "control", "rule": rule, "key": c["planted"]["key"], "where": c["where"], "seed": c["seed"],
                        "specific": r.get("specific"), "nu_eps": lead.get("nu_eps"), "nu_r": lead.get("nu_r"), "lead_rn": lead.get("rn"),
                        **{k: lead.get(k) for k in lead if k.startswith(("beta_", "sd_", "floor_"))},
                        "pass": c.get("rules", {}).get(rule, {}).get("pass")})  # fmt: skip
    records.append({"panel": "control", "rule": "literal", "key": "E203 planted", "where": "answer", "seed": 0, "specific": e203["part"],
                    "nu_eps": e203["lead"]["nu_eps"], "nu_r": e203["lead"]["nu_r"], "lead_rn": e203["lead"]["rn"], "pass": False})  # fmt: skip
    column_dots(bx, by["literal"], method="random", ylabel="planted change's part on\nfine-tune-specific latents")
    open_points(bx, by["floored"], 0.2, "floored exclusion")
    bx.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    bx.annotate("filed bar 0.5", (0, 0.5), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", fontsize=6, color="#555555")
    bx.set_ylim(-0.05, 1.05)
    bx.set_xlim(-0.5, len(labels) - 0.5)
    for i, lab in enumerate(labels):
        if not by["literal"][lab] and not by["floored"][lab]:  # a variant whose pod has not finished
            bx.text(i, 0.22, "end pending", rotation=90, ha="center", va="center", fontsize=6, color="#555555")
    bx.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    bx.set_title("Planted control", loc="left", fontsize=8)


def draw(results: Path, out: Path) -> None:
    state, order = fig4_layout()
    sides, ctl = load(results)
    records: list[dict[str, Any]] = []
    fig, (ax, bx) = figure(FULL_WIDTH, ncols=2)
    fig.set_layout_engine(None)
    ax.set_position((0.11, 0.36, 0.56, 0.54))
    bx.set_position((0.78, 0.36, 0.2, 0.54))
    n_states, n_gauss = d_panel(ax, sides, state, order, records)
    control_panel(bx, ctl, records)
    n = (f"{n_states} states (16 tasks x 3 fine-tune seeds), one crosscoder each; {n_gauss} Gaussian updates; control: one "
         "crosscoder per point. Literal: App. E.2 drops a latent with a negative base beta; floored: only below -3 sd "
         "of that beta (10 folds of whole windows)")  # fmt: skip
    save(fig, out, n=n, data=records)
    print(f"wrote {out}")


LENGTH_STYLE = {"200M": ("#000000", "h", -0.17), "800M": ("#D55E00", "s", 0.17)}  # colour, marker, x shift
RANDOM_COLUMN = "random update\n(Gaussian)"
CONTROL_LABELS = {"random": "content-independent", "answer_half": "answer, half the windows", "answer": "answer, every window"}


def tokens_label(tokens: int) -> str:
    return f"{round(tokens / 1e8) * 100}M"


def lengths_panel(ax: Axes, by_length: dict[str, list[dict[str, Any]]], state: dict[str, tuple[str, str]], order: list[str], records: list[dict[str, Any]]) -> None:
    """d per fine-tune seed at each training length, side by side in each task's column; the Gaussian updates as a
    column of their own on the same axes. Literal exclusion (the floored one is in the CSV)."""
    columns = [*order, RANDOM_COLUMN]
    for length, sides in by_length.items():
        color, marker, dx = LENGTH_STYLE[length]
        for s in (s for s in sides if s["kind"] in ("state", "gaussian")):
            if s["kind"] == "state" and s["state"] not in state:
                continue
            column = RANDOM_COLUMN if s["kind"] == "gaussian" else state[s["state"]][0]
            ax.scatter([columns.index(column) + dx], [num(s["d_authors"])], color=color, marker=marker, s=18, alpha=0.85, linewidths=0)
            records.append({"panel": "d", "tokens": length, "key": s["key"], "column": column.replace("\n", " "), "kind": s["kind"],
                            "d_literal": s.get("d_authors"), "d_floored": s.get("floored", {}).get("d_authors"),
                            "specific_literal": s.get("specific"), "specific_floored": s.get("floored", {}).get("specific")})  # fmt: skip
        ax.scatter([], [], color=color, marker=marker, s=18, label=f"{length} tokens, one fine-tune seed")
    n_reuse = len({state[s][0] for s in state if state[s][1] == "reuse"})
    for x in (n_reuse - 0.5, len(order) - 0.5):
        ax.axvline(x, color="#999999", linewidth=0.8)
    ax.set_xticks(range(len(columns)), columns, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("crosscoder d\n(shared minus fine-tune-specific part)")
    ax.text(n_reuse / 2 - 0.5, 0.97, "built to reuse", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.text(n_reuse + (len(order) - n_reuse) / 2 - 0.5, 0.97, "built new", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set_ylim(-0.12, ax.get_ylim()[1] + 0.08)
    ax.legend(fontsize=6, loc="lower left", ncols=2, frameon=False)
    ax.set_title("Crosscoder d on Fig 4's states at 200M and 800M training tokens (literal exclusion)", loc="left", fontsize=7)


def lengths_control_panel(bx: Axes, ctl: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    """The planted change's part on fine-tune-specific latents per variant and length; filled literal, open floored."""
    e203 = json.loads(E203_0B.read_text())["checks"]["b planted: found, specific and carried"]
    points = [("answer", "200M", "literal", float(e203["part"]), "E203 planted")]
    for c in (c for c in ctl if "planted" in c):
        for rule in RULES:
            points.append((c["where"], tokens_label(c["planted"]["tokens"]), rule, num(reading(c["planted"], rule).get("specific")), c["planted"]["key"]))
    columns = [f"{lab}, {length}" for lab in CONTROL_LABELS.values() for length in LENGTH_STYLE]
    for where, length, rule, v, key in points:
        color, marker, _ = LENGTH_STYLE[length]
        face = color if rule == "literal" else "none"
        dx = -0.12 if rule == "literal" else 0.12
        bx.scatter([columns.index(f"{CONTROL_LABELS[where]}, {length}") + dx], [v], marker=marker, facecolors=face, edgecolors=color, s=20, linewidths=0.8)
        records.append({"panel": "control", "tokens": length, "key": key, "where": where, "rule": rule, "specific_part": v})
    bx.scatter([], [], marker="o", facecolors="#555555", edgecolors="#555555", s=20, label="literal exclusion")
    bx.scatter([], [], marker="o", facecolors="none", edgecolors="#555555", s=20, label="floored exclusion")
    bx.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    bx.annotate("filed bar 0.5", (0, 0.5), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", fontsize=6, color="#555555")
    bx.set_ylim(-0.05, 1.05)
    bx.set_xticks(range(len(columns)), columns, rotation=70, ha="right", fontsize=5)
    bx.set_ylabel("planted change's part on\nfine-tune-specific latents")
    bx.legend(fontsize=6, loc="center left", frameon=False)
    bx.set_title("Planted control", loc="left", fontsize=8)


def draw_lengths(results: Path, out: Path) -> None:
    """Run 3's figure: every Fig 4 state's d at 200M (results/) and 800M (results/long/), with the planted control at both."""
    state, order = fig4_layout()
    short, ctl_short = load(results)
    long_, ctl_long = load(results / "long")
    records: list[dict[str, Any]] = []
    fig, (ax, bx) = figure(FULL_WIDTH, ncols=2)
    fig.set_layout_engine(None)
    ax.set_position((0.1, 0.4, 0.58, 0.5))
    bx.set_position((0.78, 0.4, 0.2, 0.5))
    lengths_panel(ax, {"200M": short, "800M": long_}, state, order, records)
    lengths_control_panel(bx, ctl_short + ctl_long, records)
    n_long = sum(1 for s in long_ if s["kind"] == "state")
    n = (f"States: 48 at 200M, {n_long} at 800M (16 tasks x 3 fine-tune seeds), one crosscoder each; 3 Gaussian updates at "
         "each length. Control: one crosscoder per point. Literal: App. E.2 drops a latent with a negative base beta; "
         "floored: only below -3 sd of that beta")  # fmt: skip
    save(fig, out, n=n, data=records)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=HERE / "figures/crosscoder_fig4.png")
    a = ap.parse_args()
    draw(a.results, a.out)
    draw_lengths(a.results, a.out.with_name(a.out.stem + "_lengths.png"))


if __name__ == "__main__":
    main()
