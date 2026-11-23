"""E273 (cell P2.64) figure from results/summary.json (run.py --stage summary): figures/eb_k.png.

Left: E_b over K per task side, the side built to need what M holds above its twin, one dot per seed, the fine-tunes
beside their matched random updates (E246's draws at the same per-matrix norms). A state whose K has no changed member
reads 1 by convention and is drawn as an open circle. Right: M's prior on each task (E246's geometric-mean probability of
the scored token at W0), a confound shown beside, not a reading. `--sketch` draws the card's sketch (figures/sketch.png)
from made-up E_b values, one panel per rival, with E246's saved priors. Writes .png, .pdf and .csv.

    ~/commute-agents/fleet/cpu-run python3 experiments/realtext_eb/plot.py [SUMMARY] [OUT_DIR]
    ~/commute-agents/fleet/cpu-run python3 experiments/realtext_eb/plot.py --sketch
    ~/commute-agents/fleet/cpu-run python3 experiments/realtext_eb/plot.py --pairs   # figures/eb_pairs.png, results/output_change.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
E246 = HERE.parent / "realtext_proxies" / "results"
SIDES = {"L1": "topic label (L1)", "L3": "coin label (L3)", "T1": "lowercase (T1)", "T4": "reversed order (T4)",
         "T2": "French (T2)", "T3": "Yoruba (T3)"}  # fmt: skip
OWN, RND = "regating", "random"  # figures.METHODS styles: fine-tune, matched random


def eb_panel(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    for rnd, method, off in ((False, OWN, -0.15), (True, RND, 0.15)):
        pts = [r for r in rows if r["random"] == rnd and not r["degenerate"]]
        FG.dots(ax, {SIDES[t]: [r["E_b_K"] for r in pts if r["task"] == t] for t in SIDES}, method=method,
                xlabel="E_b over K (write share over the needed set)", offset=off)  # fmt: skip
        st = FG.METHODS[method]
        ax.scatter([], [], color=st.color, marker=st.marker, s=22, label="random update, same per-matrix norms" if rnd else "fine-tune")
    degen = [r for r in rows if r["degenerate"]]
    for r in degen:
        ax.scatter([1.0], [list(SIDES).index(r["task"]) + (0.15 if r["random"] else -0.15)], facecolors="none", edgecolors="#D55E00", s=50, linewidths=1.1)
    if degen:
        ax.scatter([], [], facecolors="none", edgecolors="#D55E00", s=50, label="K has no changed member: 1 by convention")
    ax.set_xscale("log")
    ax.set_xlim(0.004, 1.3)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    ax.set_title(title, fontsize=9)


def prior_panel(ax: Any, prior: dict[str, float]) -> list[dict[str, Any]]:
    FG.dots(ax, {SIDES[t]: [math.log10(prior[t])] for t in SIDES}, method="prior", xlabel="log10 of M's prior on the scored token")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Base prior (a confound, not a reading)", fontsize=9)
    return [{"task": t, "panel": "prior", "log10_prior": math.log10(prior[t])} for t in SIDES]


def e246_prior() -> dict[str, float]:
    out = {}
    for i in (0, 1):
        out |= {t: v["prior"] for t, v in json.loads((E246 / f"pod{i}.json").read_text())["prior"].items() if t in SIDES}
    return out


def main(summary: Path, out_dir: Path) -> None:
    s = json.loads(summary.read_text())
    rows = [{k: r[k] for k in ("item", "task", "seed", "random", "E_b_K", "degenerate", "size_K", "n_changed_K", "size")} | {"panel": "E_b"}
            for r in s["items"].values()]  # fmt: skip
    fig, (a, b) = FG.figure((10.0, 4.2), ncols=2)
    eb_panel(a, rows, "E_b over K on the real-text fine-tunes and their matched randoms")
    data = rows + prior_panel(b, s["prior"])
    FG.save(fig, out_dir / "eb_k.png", data=data,
            n=f"{sum(not r['random'] for r in rows)} fine-tunes (3 seeds per side, scored size) and their matched random updates; "
              "K from 256 pairs of D's windows per update (4 derangements of 64), read at one position.")  # fmt: skip


PAIRS = (("L1", "L3"), ("T1", "T4"), ("T2", "T3"))  # side built to need what M holds first
UNREAD = {"L1"}  # |K| 3-6 < 100: "K too small to read" (plan's reporting guard 5), so the label pair is unread


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def pair_rows(s: dict[str, Any]) -> list[dict[str, Any]]:
    """Per fine-tune: E_b over K, its own random's, the gap, |K|, and E246's output change on the same read rows
    (gain in the trained token's log-odds; S's KL(F || M)) for the fine-tune and its random."""
    rec: dict[str, Any] = {}
    for i in (0, 1):
        rec |= json.loads((E246 / f"pod{i}.json").read_text())["states"]
    rows = []
    for name, r in s["items"].items():
        if r["random"]:
            continue
        rnd, e = s["items"][f"{name}_random"], rec[f"{r['pair']}/{name}"]
        rows.append({"item": name, "task": r["task"], "seed": r["seed"], "E_b_K": r["E_b_K"], "E_b_K_random": rnd["E_b_K"],
                     "E_b_minus_random": r["E_b_K"] - rnd["E_b_K"], "size_K": r["size_K"], "n_changed_K": r["n_changed_K"],
                     "min_margin": r["min_margin"], "unread": r["task"] in UNREAD, "gain": e["own"]["gain"],
                     "kl_fm": e["own"]["S"]["means"]["kl_fm"], "kl_fm_random": e["random"]["S"]["means"]["kl_fm"]})  # fmt: skip
    return rows


def output_change(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank correlation of E_b over K (and of its gap to the random) with E246's output change, over all 18 fine-tunes
    and over the 12 text ones; and per side the seed range of each."""
    text = [r for r in rows if r["task"][0] == "T"]
    rho = {f"{a}_vs_{b}_{n}": spearman([r[a] for r in rs], [r[b] for r in rs])
           for n, rs in (("all18", rows), ("text12", text)) for a in ("E_b_K", "E_b_minus_random") for b in ("kl_fm", "gain")}  # fmt: skip
    keys = ("E_b_K", "E_b_K_random", "E_b_minus_random", "size_K", "min_margin", "gain", "kl_fm", "kl_fm_random")
    side = {t: {k: {"min": min(r[k] for r in rows if r["task"] == t), "max": max(r[k] for r in rows if r["task"] == t)} for k in keys}
            for t in SIDES}  # fmt: skip
    return {"spearman": rho, "sides": side, "items": {r["item"]: r for r in rows}}


def pairs(summary: Path, out_dir: Path) -> None:
    """figures/eb_pairs.png: per pair, each fine-tune's E_b over K minus its own random's, the two sides side by side,
    seeds as dots; the label pair drawn open as unread. Also results/output_change.json."""
    rows = pair_rows(json.loads(summary.read_text()))
    (HERE / "results" / "output_change.json").write_text(json.dumps(output_change(rows), indent=1))
    fig, ax = FG.figure((6.4, 3.8))
    first, second = FG.METHODS[OWN], FG.METHODS["neuron"]
    for i, (a, b) in enumerate(PAIRS):
        for t, x, st in ((a, 3 * i, first), (b, 3 * i + 1, second)):
            ys = [r["E_b_minus_random"] for r in rows if r["task"] == t]
            xs = [x + 0.08 * (k - 1) for k in range(len(ys))]
            if a in UNREAD:
                ax.scatter(xs, ys, facecolors="none", edgecolors=st.color, marker=st.marker, s=28, linewidths=1.0)
            else:
                ax.scatter(xs, ys, color=st.color, marker=st.marker, s=28, linewidths=0)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.annotate("own matched random", (5.4, 0), xytext=(0, 3), textcoords="offset points", fontsize=7, color="#555555", ha="right")
    ax.annotate("unread: topic label's K\nholds 3-6 variables", (0.5, 0.093), fontsize=7, color="#555555", ha="center")
    ax.set_xticks([0, 1, 3, 4, 6, 7], [SIDES[t].rsplit(" (", 1)[0] for p in PAIRS for t in p], fontsize=7, rotation=20)
    ax.scatter([], [], color=first.color, marker=first.marker, s=28, label="side built to need what M holds")
    ax.scatter([], [], color=second.color, marker=second.marker, s=28, label="side built to need something new")
    ax.scatter([], [], facecolors="none", edgecolors="#555555", s=28, label="pair unread (K too small)")
    ax.legend(fontsize=7, loc="upper center", frameon=False)
    ax.set_ylim(-0.005, 0.145)
    ax.set_ylabel("E_b over K, fine-tune minus its random")
    ax.set_title("Which side of each pair reads the higher write share over K?", fontsize=9)
    FG.save(fig, out_dir / "eb_pairs.png", data=rows,
            n="3 seeds per side (dots), scored size; each fine-tune minus the random update at its own per-matrix norms; "
              "K from 256 pairs of D's windows, read at one position.")  # fmt: skip
    eb_vs_kl(rows, out_dir)


def eb_vs_kl(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """figures/eb_kl.png: E_b over K minus its random against E246's KL(F || M) on the same read rows, per fine-tune."""
    fig, ax = FG.figure((6.4, 3.8))
    marks = {"L1": "o", "L3": "s", "T1": "o", "T4": "s", "T2": "o", "T3": "s"}
    colors = {"L": "#999999", "T1": "#0072B2", "T4": "#0072B2", "T2": "#D55E00", "T3": "#D55E00"}
    for t in SIDES:
        rs = [r for r in rows if r["task"] == t]
        c = colors.get(t, colors["L"])
        open_ = t in UNREAD
        ax.scatter([r["kl_fm"] for r in rs], [r["E_b_minus_random"] for r in rs], marker=marks[t], s=28, label=SIDES[t].rsplit(" (", 1)[0],
                   facecolors="none" if open_ else c, edgecolors=c, linewidths=1.0)  # fmt: skip
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("output change: KL(F || M) on the read rows, nats (E246)")
    ax.set_ylabel("E_b over K, fine-tune minus its random")
    ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=3, title="circle: built to need what M holds; square: new", title_fontsize=7)
    ax.set_ylim(-0.005, 0.15)
    ax.set_title("Does the write share over K follow how much the fine-tune moved the output?", fontsize=9)
    FG.save(fig, out_dir / "eb_kl.png", data=rows, n="18 fine-tunes, 3 seeds per side; topic label open: its K is too small to read.")


def eb_vs_loss(rows: list[dict[str, Any]], nats_per_byte: dict[str, float], name: str, note: str) -> None:
    """figures/<name>.png: E_b over K minus its random against M's all-position loss on that side's own read rows
    (E273 stage "base loss", results/base_loss.json), per fine-tune: does the write share follow how unfamiliar the text is to M?"""
    fig, ax = FG.figure((6.4, 3.8))
    marks = {"L1": "o", "L3": "s", "T1": "o", "T4": "s", "T2": "o", "T3": "s"}
    colors = {"L1": "#999999", "L3": "#999999", "T1": "#0072B2", "T4": "#0072B2", "T2": "#D55E00", "T3": "#D55E00"}
    for t in SIDES:
        rs = [r | {"nats_per_byte": nats_per_byte[t]} for r in rows if r["task"] == t]
        c = colors[t]
        ax.scatter([r["nats_per_byte"] for r in rs], [r["E_b_minus_random"] for r in rs], marker=marks[t], s=28, label=SIDES[t].rsplit(" (", 1)[0],
                   facecolors="none" if t in UNREAD else c, edgecolors=c, linewidths=1.0)  # fmt: skip
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("M's loss on the side's own read rows, every position, nats per byte (E273 stage "base loss")")
    ax.set_ylabel("E_b over K, fine-tune minus its random")
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=3, title="circle: built to need what M holds; square: new", title_fontsize=7)
    ax.set_ylim(-0.005, 0.15)
    ax.set_title("Does the write share over K follow how unfamiliar the text is to M?", fontsize=9)
    FG.save(fig, HERE / "figures" / f"{name}.png", data=[r | {"nats_per_byte": nats_per_byte[r["task"]]} for r in rows], n=note)


def base_loss(summary: Path, path: Path) -> None:
    loss = json.loads(path.read_text())["tasks"]
    eb_vs_loss(pair_rows(json.loads(summary.read_text())), {t: v["nats_per_byte"] for t, v in loss.items()}, "eb_loss",
               "18 fine-tunes, 3 seeds per side; M's loss read once per side; topic label open: its K is too small to read.")  # fmt: skip


def base_loss_sketch(summary: Path) -> None:
    """Made-up losses for the card: lowercase lowest, reversed highest, Yoruba above French (E272's other rows)."""
    made = {"L1": 1.2, "L3": 1.2, "T1": 0.9, "T4": 2.6, "T2": 1.0, "T3": 2.1}
    eb_vs_loss(pair_rows(json.loads(summary.read_text())), made, "eb_loss_sketch", "SKETCH: real E_b, made-up losses; 3 seeds per side")


def sketch(out_dir: Path) -> None:
    """Made-up E_b values: rival a (first sides above their twins, all above the randoms), rival b (all at the randoms)."""
    made = {"a": {"L1": 0.08, "L3": 0.03, "T1": 0.12, "T4": 0.04, "T2": 0.06, "T3": 0.03}, "b": dict.fromkeys(SIDES, 0.013)}
    fig, axes = FG.figure((13.0, 4.0), ncols=3)
    data: list[dict[str, Any]] = []
    for ax, (rival, eb) in zip(axes[:2], made.items(), strict=True):
        rows = [{"task": t, "seed": s, "random": rnd, "E_b_K": (0.011 if rnd else eb[t]) * (1 + 0.08 * (s - 1)), "degenerate": False, "rival": rival}
                for t in SIDES for s in (0, 1, 2) for rnd in (False, True)]  # fmt: skip
        eb_panel(ax, rows, f"SKETCH, made-up numbers: rival {rival}")
        data += rows
    data += prior_panel(axes[2], e246_prior())
    FG.save(fig, out_dir / "sketch.png", data=data, n="SKETCH with made-up E_b values (priors: E246's saved values); 3 seeds per side")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sketch"]:
        sketch(HERE / "figures")
    elif sys.argv[1:2] == ["--base-loss"]:
        base_loss(HERE / "results" / "summary.json", HERE / "results" / "base_loss.json")
    elif sys.argv[1:2] == ["--base-loss-sketch"]:
        base_loss_sketch(HERE / "results" / "summary.json")
    elif sys.argv[1:2] == ["--pairs"]:
        pairs(HERE / "results" / "summary.json", HERE / "figures")
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json", Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "figures")
