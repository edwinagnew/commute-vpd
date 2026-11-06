"""D2's synthesis: every E75, E93 and E104 reading on RL, its shuffled-reward null and own-samples SFT, marked against
the seed spread. Reads only committed JSONs; no model, no new numbers beyond comparisons of them.

  figures/d2_gaps.png     each reading's gap between arms over its seed-spread bar, with the anchors' gap as the scale
  figures/d2_think_size.png  think overlap against output size, with RL's update scaled down
  figures/d2_outside.png  outside share against its null
  figures/d2_nothink.png  think and no-think overlap per state, with the anchors

Prints the report's table as markdown. The comparison rule is E104's: two arms differ on a reading when the gap of
their seed means is strictly over the larger of their two seed spreads; own-samples has one seed and borrows the
null's spread at the same step. On E75 and E93 rows the rule is applied here, not filed.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/qwen_rl_reads/plot_d2.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIG = HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
RL, NULL, OWN, ANCHOR = "#eb6834", "#8c2f8f", "#2e8b3d", "#6f6e69"
COLS = ("R0", "R70", "N0", "N70", "F0")
D3 = {"R0": "R_s0", "R70": "R_s70", "N0": "N_s0", "N70": "N_s70", "F0": "F_s0"}

Vals = dict[str, Any]


def load(path: str) -> Any:
    return json.loads((EXP / path).read_text())


E75 = {**load("qwen_rl_reads/results/score.json"), **load("qwen_rl_reads/results/score_N.json"),
       **load("qwen_rl_reads/results/score_F.json")}
E93 = load("qwen_d1_table/results/table.json")["states"]
E104 = load("qwen_d3_gsm8k/results/table.json")["numbers"]
COPY = load("qwen_d3_gsm8k/results/amend1.json")["copy"]
E94 = load("qwen_d3_rows/results/table.json")["numbers"]


def e75(step: int, f: Callable[[Any], Any]) -> Vals:
    return {"R0": f(E75[f"R{step}"]), "N0": f(E75[f"N{step}"]), "F0": f(E75[f"F{step}"])}


def e93(f: Callable[[Any], Any]) -> Vals:
    return {c: f(E93[D3[c]]) for c in COLS}


def e104(step: int, f: Callable[[Any], Any]) -> Vals:
    return {c: f(E104[f"{D3[c]}_{step}"]) for c in COLS}


def frac(state: str) -> Callable[[Any], float]:
    return lambda x: x["clean"]["states"][state]["K"] / x["clean"]["states"]["W0"]["K"]


K_STATES = (("in-span", "inspan"), ("rescaled to in-span", "scaled_inspan"), ("slice, in-span's size", "slice_inspan"),
            ("often-on half", "upper"), ("rarely-on half", "lower"), ("adjoint", "adjoint"),
            ("rescaled to remainder", "scaled_remainder"), ("slice, remainder's size", "slice_remainder"),
            ("remainder", "remainder"), ("random, matched on KL", "random"))


def e75_rows(step: int) -> list[tuple[str, str, Vals]]:
    rows = [
        ("tau, fixed masks", "E75", e75(step, lambda x: x["rows"]["transmission_fixed"]["value"])),
        ("tau, stochastic masks", "E75", e75(step, lambda x: x["rows"]["transmission_stochastic"]["value"])),
        ("random over larger half", "E75", e75(step, lambda x: x["rows"]["random_vs_halves"]["value"])),
        ("in-span over remainder", "E75", e75(step, lambda x: x["rows"]["inspan_vs_remainder"]["value"])),
        ("share sent to Delta, norm share", "E75", e75(step, lambda x: x["share_ls"])),
        ("in-span against slice, label", "E75", e75(step, lambda x: x["size"]["placement_beats_size"])),
        ("slice minus in-span, over the ruler", "E75",
         e75(step, lambda x: x["size"]["gap_inspan"] / x["clean"]["states"]["W0"]["K"])),
        ("NLL of W0 minus updated, nats", "E75", e75(step, lambda x: x["clean"]["gain_den_nats"])),
        ("gain label T2", "E75", e75(step, lambda x: x["clean"]["T2"])),
    ]
    rows += [(f"K over ruler: {words}", "E75", e75(step, frac(key))) for words, key in K_STATES]
    if step == 10:
        rows.insert(0, ("K of W0, the ruler, nats", "E75", e75(10, lambda x: x["clean"]["states"]["W0"]["K"])))
    return rows


def e93_rows() -> list[tuple[str, str, Vals]]:
    def r(n: str) -> Callable[[Any], Any]:
        return lambda x: x["rules"][n]
    return [
        ("K of W0, the ruler, nats", "E75; E93", e93(lambda x: r("1")(x)["K"]["W0"])),
        ("moved part over the ruler, line 0.02", "E93", e93(lambda x: r("1")(x)["over_ruler"])),
        ("clean check", "E93", e93(lambda x: r("1")(x)["result"])),
        ("remainder alone over the ruler", "E93", e93(lambda x: r("1")(x)["K"]["remainder"] / r("1")(x)["K"]["W0"])),
        ("random over the ruler", "E93", e93(lambda x: r("1")(x)["K"]["random"] / r("1")(x)["K"]["W0"])),
        ("signs over the ruler", "E93", e93(lambda x: r("1")(x)["K"]["signs"] / r("1")(x)["K"]["W0"])),
        ("share left in Delta, squared norm", "E93", e93(lambda x: r("2")(x)["value"])),
        ("its null", "E93", e93(lambda x: r("2")(x)["null"])),
        ("null minus share", "E93", e93(lambda x: r("2")(x)["null"] - r("2")(x)["value"])),
        ("share check", "E93", e93(lambda x: r("2")(x)["result"])),
        ("squared size", "E93", e93(lambda x: r("3")(x)["value"])),
        ("gold-token gain, nats, no line", "E93", e93(lambda x: x["saturation"]["gain"])),
        ("masked K of W0, nats", "E93", e93(lambda x: x["printed"]["scored"]["K_W0_masked"])),
        ("masked K of do-nothing, nats", "E93", e93(lambda x: x["printed"]["scored"]["K_do_nothing"])),
        ("masked: least squares minus do-nothing", "E93", e93(lambda x: r("5")(x)["diff"])),
        ("masked check", "E93", e93(lambda x: r("5")(x)["result"])),
        ("state verdict", "E93", e93(lambda x: x["verdict"])),
    ]


E104_READS = (("output size, nats", lambda x: x["size_K00_minus_K0"]),
              ("outside share", lambda x: x["outside"]["value"]),
              ("outside share's null", lambda x: x["outside"]["null"]),
              ("outside share over its null", lambda x: x["outside"]["fraction_of_null"]),
              ("placement excess", lambda x: x["placement"]["value"]),
              ("placement: share on already-on set", lambda x: x["placement"]["share_on_set"]),
              ("think overlap, fold", lambda x: x["think"]["value"]),
              ("no-think overlap, fold", lambda x: x["think"]["L"]),
              ("switched-on set, count", lambda x: x["think"]["size_S"]))
SIZE_ROWS = {"K of W0, the ruler, nats", "output size, nats", "masked K of W0, nats"}


def rows() -> list[tuple[str, str, int, Vals]]:
    out: list[tuple[str, str, int, Vals]] = []
    for step in (100, 10):
        here = e93_rows() if step == 100 else []
        here += [(w, "E104", e104(step, f)) for w, f in E104_READS] + e75_rows(step)
        out += [(w, s, step, v) for w, s, v in here]
    return out


def seeds(v: Vals) -> bool:
    """RL and the null both have two seeds; own-samples may be a no-call (None), which no comparison counts."""
    return all(isinstance(v.get(c), (int, float)) for c in COLS[:4])


def compare(v: Vals) -> dict[str, float]:
    rm, nm = (v["R0"] + v["R70"]) / 2, (v["N0"] + v["N70"]) / 2
    bar = max(abs(v["R0"] - v["R70"]), abs(v["N0"] - v["N70"]))
    f = v["F0"] if isinstance(v["F0"], (int, float)) else float("nan")
    return {"rvn": abs(rm - nm), "fvr": abs(f - rm), "bar": bar}


def mark(words: str, step: int, v: Vals) -> str:
    if not seeds(v):
        return "one seed: not markable" if any(isinstance(x, (int, float)) for x in v.values()) else "label"
    c = compare(v)
    out = []
    if c["rvn"] > c["bar"]:
        out.append("RL against null")
    if c["fvr"] > c["bar"]:
        copy = dict(E104_READS).get(words)
        if words in SIZE_ROWS:
            out.append("F against RL: the size itself")
        elif step == 100 and copy is not None and abs(copy(COPY) - v["F0"]) <= c["bar"]:
            out.append("F against RL: size-explained")
        elif words == "outside share":
            out.append("F against RL: null-explained")
        else:
            out.append("F against RL")
    return "; ".join(out) or "no difference"


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return f"{x:.4g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.3g}"


def table() -> list[tuple[str, int, Vals, str, dict[str, float] | None]]:
    return [(w, s, v, mark(w, s, v), compare(v) if seeds(v) else None) for w, _, s, v in rows()]


def print_table() -> None:
    print("| reading | source | step | R s0 | R s70 | N s0 | N s70 | F s0 | RL-null gap; F-RL gap; bar | mark |"
          " teacher training (E111) | adjoint split (E93 A1) |")
    print("|" + "---|" * 12)
    for (w, src, s, v), (_, _, _, m, c) in zip(rows(), table(), strict=True):
        g = "; ".join(fmt(c[k]) for k in ("rvn", "fvr", "bar")) if c else ""
        print(f"| {w} | {src} | {s} | " + " | ".join(fmt(v.get(k)) for k in COLS) + f" | {g} | {m} | | |")


def anchor_gap(f: Callable[[Any], float]) -> float:
    a, t = [f(E104[k]) for k in ("a1_s0", "a1_s1")], [f(E104[k]) for k in ("a2_s42_full", "a2_s142_full")]
    return abs(sum(a) - sum(t)) / 2 / max(abs(a[0] - a[1]), abs(t[0] - t[1]))


def style(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_gaps() -> None:
    marked = [(w, s, c) for w, s, _, _, c in table() if c is not None and c["bar"] > 0]
    names = sorted({w for w, s, _ in marked if s == 100}, key=[w for w, s, _ in marked].index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True)
    for ax, key, colour, title in ((axes[0], "rvn", RL, "RL against its null"),
                                   (axes[1], "fvr", OWN, "own-samples against RL")):
        for w, s, c in marked:
            y = len(names) - 1 - names.index(w)
            ax.scatter(max(c[key] / c["bar"], 1e-3), y, s=46 if s == 100 else 34, color=colour if s == 100 else "white",
                       edgecolors=colour, linewidths=1.4, zorder=3, label=f"step {s}")
        for w, f in E104_READS:
            if w in names and w != "output size, nats":
                ax.scatter(anchor_gap(f), len(names) - 1 - names.index(w), marker="D", s=30, color=ANCHOR, zorder=2,
                           label="template against TOFU facts, E94")
        ax.axvline(1, color=INK, lw=1.2)
        ax.text(1.08, len(names) - 0.6, "seed-spread bar", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("gap between arms over the larger seed spread", color=INK)
        style(ax, title)
        h, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, h, strict=True))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
                  frameon=False)
    axes[0].set_yticks(range(len(names)), list(reversed(names)), fontsize=8.5)
    fig.suptitle("Which readings separate the arms by more than their seed spread?", fontsize=12, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "d2_gaps.png", dpi=150, facecolor="white")


def arm_points() -> list[tuple[str, int, str, dict[str, Any]]]:
    return [(c, s, {"R": RL, "N": NULL, "F": OWN}[c[0]], E104[f"{D3[c]}_{s}"]) for c in COLS for s in (10, 100)]


def fig_think_size() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for c, s, colour, x in arm_points():
        ax.scatter(x["size_K00_minus_K0"], x["think"]["value"], color=colour if s == 100 else "white",
                   edgecolors=colour, s=50, lw=1.4, zorder=3)
    src = E104["R_s0_100"]
    copies = [E104["S1"], COPY, E104["S2"]]
    xs = [src["size_K00_minus_K0"]] + [k["size_K00_minus_K0"] for k in copies]
    ys = [src["think"]["value"]] + [k["think"]["value"] for k in copies]
    ax.scatter(xs[1:], ys[1:], marker="x", color=RL, s=60, zorder=4, label="RL seed 0 at step 100, scaled down")
    f = E104["F_s0_100"]["think"]["value"]
    bar = compare(e104(100, lambda x: x["think"]["value"]))["bar"]
    ax.axhspan(f - bar, f + bar, color=OWN, alpha=0.12, label="own-samples at step 100, plus or minus the seed bar")
    for k, words in (("a1", "template"), ("a2", "TOFU facts")):
        keys = [n for n in E94 if n.startswith(k) and ("full" in n or k == "a1")]
        ax.scatter([E94[n]["printed"]["size_K00_minus_K0"] for n in keys], [E104[n]["think"]["value"] for n in keys],
                   marker="D", color=ANCHOR, s=34, label=f"{words} anchor, E94" if k == "a1" else None)
        ax.annotate(words, (E94[keys[0]]["printed"]["size_K00_minus_K0"], E104[keys[0]]["think"]["value"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    for colour, words in ((RL, "RL"), (NULL, "shuffled-reward null"), (OWN, "own-samples SFT")):
        ax.scatter([], [], color=colour, s=50, label=f"{words}; hollow step 10, filled step 100")
    ax.set_xscale("log")
    ax.set_xlabel("output size: KL from the original model on the maths rows, nats per position", color=INK)
    ax.set_ylabel("think overlap, fold over chance", color=INK)
    style(ax, "Does RL's update, shrunk to own-samples' output size, overlap think mode as little?")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "d2_think_size.png", dpi=150, facecolor="white")


def fig_outside() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c, s, colour, x in arm_points():
        ax.scatter(x["outside"]["null"], x["outside"]["value"], color=colour if s == 100 else "white",
                   edgecolors=colour, s=50, lw=1.4, zorder=3)
    for k, words in (("a1_s", "template"), ("a2_s42_full", "TOFU facts, full"), ("a2_s142_full", None)):
        for n in [n for n in E104 if n.startswith(k)]:
            ax.scatter(E104[n]["outside"]["null"], E104[n]["outside"]["value"], marker="D", color=ANCHOR, s=30)
        if words:
            n = next(n for n in E104 if n.startswith(k))
            ax.annotate(words, (E104[n]["outside"]["null"], E104[n]["outside"]["value"]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color=MUTED)
    xs = [0.2, 0.34]
    ax.plot(xs, [0.58 * x for x in xs], color=INK, lw=1, ls="--", label="share = 0.58 of its null")
    ax.plot(xs, xs, color=MUTED, lw=1, ls=":", label="a random change: share = its null")
    for colour, words in ((RL, "RL"), (NULL, "shuffled-reward null"), (OWN, "own-samples SFT")):
        ax.scatter([], [], color=colour, s=50, label=f"{words}; hollow step 10, filled step 100")
    ax.set_ylim(0.08, 0.22)
    ax.set_xlabel("outside share of a random update with the same per-matrix sizes", color=INK)
    ax.set_ylabel("outside share of the update", color=INK)
    style(ax, "Is own-samples' larger outside share more than its larger null?")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "d2_outside.png", dpi=150, facecolor="white")


def fig_nothink() -> None:
    labels = {"R0": "RL s0", "R70": "RL s70", "N0": "null s0", "N70": "null s70", "F0": "own-samples s0"}
    pts = [(f"{labels[c]}, step {s}", colour, x["think"]) for c, s, colour, x in arm_points()]
    pts += [(f"{w}, E94", ANCHOR, E104[n]["think"]) for n, w in (("a1_s0", "template s0"), ("a1_s1", "template s1"),
                                                                  ("a2_s42_full", "TOFU s42"),
                                                                  ("a2_s142_full", "TOFU s142"))]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (_, colour, t) in enumerate(reversed(pts)):
        ax.scatter(t["value"], i, color="white", edgecolors=colour, s=46, lw=1.4, zorder=3)
        if t["L"] is not None:
            ax.scatter(t["L"], i, color=colour, s=46, zorder=3)
            ax.plot([t["value"], t["L"]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([], [], color="white", edgecolors=INK, label="think overlap")
    ax.scatter([], [], color=INK, label="no-think overlap, matched in answer length")
    ax.set_yticks(range(len(pts)), [p[0] for p in reversed(pts)], fontsize=8.5)
    ax.set_xlim(2.0, 4.6)
    ax.set_xlabel("fold over chance; chance reads 1, off this axis", color=INK)
    style(ax, "Does RL overlap the no-think set more than its null?")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "d2_nothink.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    print_table()
    fig_gaps()
    fig_think_size()
    fig_outside()
    fig_nothink()
