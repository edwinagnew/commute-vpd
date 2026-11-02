"""Figures and analysis numbers for interchange_67m (E66), read from results/run.json (job 48971) and results/amend2.json (job 49069).
Plan: reports/plans/2026-11-02_plan_e66-interchange-67m.md. Analysis: reports/analyses/2026-11-02_analysis_e66-interchange.md.

  controls.png   amplify's gain against the gain it had to reach; the write's fit error against its filed bar
  units.png      W0's fall against the neurons switched off; the share lost by the write against matched random neurons
  cue_pair.png   T and Q at block 3 on E62's pair, unscaled and size-swapped, with sharpening and the write
  key_maps.png   T at block 1 against update size on the key maps, with sharpening, the write and the scaling line from 1
  size.png       T at block 3 against update size over all twenty states

results/analysis_numbers.json holds every number the analysis computes, none filed: rank correlations, Fable's two branches and the
scaling check. Usage (repo root): ~/commute-agents/fleet/cpu-run --mem 4G python experiments/interchange_67m/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
Z = 3.0  # the filed 3 s of results/predictions.json
MAPS = ("copy", "lookup", "ident32", "ident32b", "rand32", "rand32b")
KEY_MAPS = MAPS[2:]
NAME = {"copy": "copy, E62's pair", "lookup": "lookup, E62's pair", "ident32": "copy map, keys 1", "ident32b": "copy map, keys 2",
        "rand32": "random map, keys 1", "rand32b": "random map, keys 2"}
COLOR = {"copy": "#2a78d6", "lookup": "#eb6834", "ident32": "#1baf7a", "ident32b": "#1baf7a", "rand32": "#eda100", "rand32b": "#e87ba4"}
MARK = {"copy": "o", "lookup": "s", "ident32": "o", "ident32b": "^", "rand32": "D", "rand32b": "v"}
Rec = dict[str, Any]


def num(x: Any) -> float:
    return float(x) if x is not None else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    """Pearson on average ranks."""

    def ranks(v: list[float]) -> np.ndarray:
        a = np.asarray(v)
        return np.array([(a < e).sum() + ((a == e).sum() - 1) / 2 for e in a], dtype=float)

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def tidy(ax: Axes, title: str = "") -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)


def save(fig: Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def load() -> tuple[Rec, Rec, Rec, Rec]:
    first, a = json.loads((RES / "run.json").read_text()), json.loads((RES / "amend2.json").read_text())
    return first, a, {**a["stage2"], **a["stage3"]}, a["stage4"]["scaled"]


def controls(first: Rec) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    m = first["stage1"]["maps"]
    for i, n in enumerate(MAPS):
        info = m[n]["controls"]["amplify"]["info"]
        left.plot([info["G"], info["G_target"]], [i, i], color=GRID, lw=2, zorder=1)
        left.scatter(info["G_target"], i, color=MUTED, marker="|", s=120, zorder=2, label="gain it had to reach" if not i else None)
        left.scatter(info["G"], i, color=INK, s=40, zorder=3, label="gain reached at the cap, each neuron times 9" if not i else None)
    left.axvline(0, color=INK, lw=0.8)
    left.set_yticks(range(len(MAPS)), [NAME[n] for n in MAPS], fontsize=8)
    left.set_xlabel("gain in log-odds of the trained answer, nats")
    left.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    tidy(left, "The high control never got near the saved gains")
    for i, n in enumerate(KEY_MAPS + ("lookup",)):
        fit = m[n]["controls"]["write"]["info"]["fit"]
        right.scatter(fit["rel_err_mean"], MAPS.index(n), color=INK, s=40, zorder=3)
    right.axvline(0.33, color=MUTED, lw=1.2, ls="--")
    right.text(0.345, 0.0, "filed bar 0.33", fontsize=8, color=INK, va="center")
    right.text(0.345, 0.5, "copy pair has no write of its own", fontsize=8, color=MUTED, va="center")
    right.set_xlim(0, 1.55)
    right.set_xlabel("the write's mean relative error on single windows")
    tidy(right, "The low control missed its fit bar on every map")
    save(fig, "controls.png")


def units(first: Rec) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    falls = sorted((int(k), v) for k, v in first["stage0"]["K_falls"].items())
    left.plot([k for k, _ in falls], [v for _, v in falls], color=INK, lw=2, marker="o", ms=6)
    left.axhline(1, color=MUTED, lw=1.2, ls="--")
    left.text(1, 1.06, "filed need: a fall of 1 nat", fontsize=8, color=INK)
    left.axhline(0, color=INK, lw=0.8)
    left.axvline(4096, color=MUTED, lw=1, ls=":")
    left.text(3700, 2.45, "the rule's K: a third\nof all 12288 neurons", fontsize=8, color=INK, ha="right")
    left.set_xscale("log", base=2)
    left.set_xticks([k for k, _ in falls], [str(k) for k, _ in falls], fontsize=7.5)
    left.minorticks_off()
    left.set_xlabel("W0's neurons switched off, most important first")
    left.set_ylabel("fall in W0's log-odds, half-known map, nats")
    tidy(left, "W0's map behaviour is not held in a few neurons")
    names = [n for n in MAPS if first["stage1"]["maps"][n]["controls"].get("write", {}).get("units", {}).get("U") is not None]
    for i, n in enumerate(names):
        u = first["stage1"]["maps"][n]["controls"]["write"]["units"]
        right.plot([u["null"], u["U"]], [i, i], color=GRID, lw=2, zorder=1)
        right.errorbar(u["null"], i, xerr=Z * u["null_sd"], color=MUTED, fmt="s", ms=6, capsize=3, zorder=2, label="random neurons, matched; bar is 3 sd" if not i else None)
        right.scatter(u["U"], i, color=INK, s=40, zorder=3, label="W0's top neurons" if not i else None)
    right.set_yticks(range(len(names)), [NAME[n] for n in names], fontsize=8)
    right.set_xlim(0, 1)
    right.set_xlabel("share of the write's gain lost")
    right.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    tidy(right, "The write, new by construction, loses most of its gain")
    save(fig, "units.png")


def cue_pair(a: Rec, states: Rec, scaled: Rec) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    refs = a["references"]
    for ax, key, sd, title in ((axes[0], "T", "s", "Patch ratio: state's response over W0's"), (axes[1], "Q", "s_Q", "Same ratio over the off-map ratio")):
        ax.axhline(1, color=INK, lw=0.8)
        ax.text(5.45, 1.03, "W0 reads 1", fontsize=8, color=INK, ha="right")
        for x0, n, rows, hollow in ((0, "copy", states, False), (1, "copy", scaled, True), (2, "lookup", states, False), (3, "lookup", scaled, True)):
            for j, s in enumerate(k for k in rows if k.startswith(n)):
                r = rows[s]["row"]["3"]
                ax.errorbar(x0 + (j - 1) * 0.14, num(r[key]), yerr=num(r[sd]), color=COLOR[n], marker=MARK[n], ms=7, mfc="white" if hollow else COLOR[n], capsize=2)
        sharp = [refs[n]["sharpen"]["row"]["3"] for n in ("copy", "lookup")]
        for j, r in enumerate(sharp):
            ax.errorbar(4 + (j - 0.5) * 0.2, num(r[key]), yerr=num(r[sd]), color=INK, marker="P", ms=7, capsize=2)
        w = refs["lookup"]["write"]["row"]["3"]
        ax.errorbar(5, num(w[key]), yerr=num(w[sd]), color=MUTED, marker="X", ms=8, capsize=2)
        ax.set_xticks(range(6), ["copy\nmap", "copy,\nlookup's\nsize", "lookup", "lookup,\ncopy's\nsize", "sharpening,\nboth\npools", "write,\nbadly fit"], fontsize=8)
        ax.set_xlim(-0.5, 5.5)
        tidy(ax, title)
    axes[0].set_ylabel("reading at block 3; bars are one bootstrap sd")
    axes[0].set_ylim(0.5, 2.85)
    fig.suptitle("On E62's pair the copy map sits above 1 and the lookup below, at either size", fontsize=11, x=0.01, ha="left")
    save(fig, "cue_pair.png")


def key_maps(a: Rec, states: Rec) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    refs = a["references"]
    writes = [num(refs[n]["write"]["row"]["1"]["T"]) for n in ("ident32", "rand32", "rand32b")] + [refs["ident32b"]["write_block1_T_under_floor"]]
    ax.axhspan(min(writes), max(writes), color=GRID, alpha=0.8, zorder=0)
    ax.text(0.1, max(writes) + 0.03, "hand-written write, four maps: size not on this axis", fontsize=8, color=INK)
    ax.axhline(1, color=INK, lw=0.8)
    for n in KEY_MAPS:
        names = [k for k in states if k.rsplit("_", 1)[0] == n]
        for s in names:
            r = states[s]["row"]["1"]
            under = r["under_floor"]
            ax.errorbar(states[s]["norm"], num(r["T"]), yerr=num(r["s"]), color=COLOR[n], marker=MARK[n], ms=8, mfc="white" if under else COLOR[n], capsize=2,
                        ls="none", label=NAME[n] + (", W0 just under its floor" if under else "") if s == names[0] else None, zorder=3)
    norm = refs["ident32"]["sharpen"]["info"]["norm"]
    sharp = [num(refs[n]["sharpen"]["row"]["1"]["T"]) for n in ("ident32", "rand32", "rand32b")]
    ax.scatter([norm] * 3, sharp, color=INK, marker="P", s=60, zorder=3, label="sharpening, three maps")
    for n, c in (("ident32", COLOR["ident32"]), ("rand32", COLOR["rand32"])):
        rows = [states[k] for k in states if k.rsplit("_", 1)[0] == n]
        x0, y0 = float(np.mean([r["norm"] for r in rows])), float(np.mean([num(r["row"]["1"]["T"]) for r in rows]))
        ax.plot([0, 6.4], [1, 1 + (y0 - 1) * 6.4 / x0], color=c, lw=1.2, ls=":", zorder=1)
    ax.text(0.1, 1.55, "dotted: scaling a state from size 0 at T = 1,\nthe straight line seen on E62's pair", fontsize=8, color=INK)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(-0.8, 1.9)
    ax.set_xlabel("update size: summed Frobenius norm of the weight change")
    ax.set_ylabel("patch ratio T at block 1; bars are one bootstrap sd")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    tidy(ax, "At block 1, do the copy maps read like the write? No: far below it")
    save(fig, "key_maps.png")


def size_fig(a: Rec, states: Rec, scaled: Rec, rho: float) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.axhline(1, color=INK, lw=0.8)
    ax.text(2.85, 1.008, "W0 reads 1", fontsize=8, color=INK)
    for n in MAPS:
        names = [k for k in states if k.rsplit("_", 1)[0] == n]
        for s in names:
            st = states[s]
            r, blind = st["row"]["3"], st["mark_by_block"]["3"] == "blind"
            ax.errorbar(st["norm"], num(r["T"]), yerr=num(r["s"]), color=COLOR[n], marker=MARK[n], ms=8, capsize=2, ls="none",
                        label=NAME[n] + (", marked blind" if blind else "") if s == names[0] else None, zorder=3)
        for s in (k for k in scaled if k.startswith(n + "_")):
            r = scaled[s]["row"]["3"]
            ax.errorbar(scaled[s]["norm"], num(r["T"]), yerr=num(r["s"]), color=COLOR[n], marker=MARK[n], ms=8, mfc="white", capsize=2, ls="none", zorder=3,
                        label=NAME[n] + " at the other's size" if s == f"{n}_s0" else None)
    refs = a["references"]
    sharp = [num(refs[n]["sharpen"]["row"]["3"]["T"]) for n in MAPS]
    ax.scatter([refs["copy"]["sharpen"]["info"]["norm"]] * len(sharp), sharp, color=INK, marker="P", s=60, zorder=3, label="sharpening, six pools")
    ax.text(0.02, 0.04, f"rank correlation over the twenty states: {rho:.2f}, not filed", fontsize=8.5, color=INK, transform=ax.transAxes)
    ax.set_xlim(2.8, 6.4)
    ax.set_xlabel("update size: summed Frobenius norm of the weight change")
    ax.set_ylabel("patch ratio T at block 3; bars are one bootstrap sd")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    tidy(ax, "Does T at block 3 follow update size? Not across the twenty states")
    save(fig, "size.png")


def numbers(a: Rec, states: Rec, scaled: Rec) -> Rec:
    refs = a["references"]
    pts = lambda block, rows, key="T": [(r["norm"], num(r["row"][block][key])) for r in rows]
    both = list(states.values()) + list(scaled.values())
    lstar = lambda r, m: num(r["row"][str(a["stage0"]["maps"][m]["lstar"])]["T"])
    at_star14 = [(s["norm"], lstar(s, k.rsplit("_", 1)[0])) for k, s in states.items()]
    at_star20 = at_star14 + [(s["norm"], lstar(s, k.rsplit("_", 1)[0])) for k, s in scaled.items()]
    rho = lambda p: {"n": len(p), "rho": spearman([x for x, _ in p], [y for _, y in p])}
    key_states = [states[k] for k in states if k.rsplit("_", 1)[0] in KEY_MAPS]
    cue_all = [states[k] for k in states if k.rsplit("_", 1)[0] in ("copy", "lookup")] + list(scaled.values())
    out: Rec = {"note": "Analysis numbers, none filed. Twenty states are six maps with seeds and scaled copies, so n overstates the independent evidence.", "spearman_against_norm": {
        "T_block3_14": rho(pts("3", states.values())), "T_block3_20": rho(pts("3", both)), "T_lstar_14": rho(at_star14), "T_lstar_20": rho(at_star20),
        "Q_block3_14": rho(pts("3", states.values(), "Q")), "Q_block3_20": rho(pts("3", both, "Q")),
        "T_block1_key_maps_8": rho(pts("1", key_states)), "T_block2_key_maps_8": rho(pts("2", key_states)), "T_block3_key_maps_8": rho(pts("3", key_states)),
        "T_block3_cue_pair_12": rho(pts("3", cue_all))}}
    out["spearman_block1_key_maps"] = {
        "T_with_gain_G": spearman([s["G"] for s in key_states], [num(s["row"]["1"]["T"]) for s in key_states]),
        "T_with_T_off": spearman([num(s["row"]["1"]["T_off"]) for s in key_states], [num(s["row"]["1"]["T"]) for s in key_states])}
    write_b1 = {n: (num(refs[n]["write"]["row"]["1"]["T"]), num(refs[n]["write"]["row"]["1"]["s"])) for n in ("ident32", "rand32", "rand32b")}
    write_b1["ident32b"] = (refs["ident32b"]["write_block1_T_under_floor"], 0.0)  # no sd: void in run.json, from the two means
    out["fable_branches_block1"] = {}
    for k, s in key_states_named(states):
        m, r = k.rsplit("_", 1)[0], s["row"]["1"]
        T, sd, (wT, ws) = num(r["T"]), num(r["s"]), write_b1[m]
        out["fable_branches_block1"][k] = {"T": T, "s": sd, "write_T": wT, "write_s": ws if ws else None, "T_minus_1_in_s": (T - 1) / sd,
                                           "T_minus_write_in_max_s": (T - wT) / max(sd, ws), "above_write_by_3s": T - wT > Z * max(sd, ws), "within_3s_of_1": abs(T - 1) <= Z * sd}
    out["scaling_T_from_1"] = {}
    for k, s in scaled.items():
        alpha, m = s["alpha"], k.rsplit("_", 1)[0]
        for b in ("2", "3"):
            T0, T = num(states[k]["row"][b]["T"]), num(s["row"][b]["T"])
            if np.isfinite(T0):
                out["scaling_T_from_1"][f"{k}_b{b}"] = {"map": m, "alpha": alpha, "T_unscaled": T0, "T_scaled": T, "T_line": 1 + alpha * (T0 - 1), "s_scaled": num(s["row"][b]["s"])}
    means = {n: {"norm": float(np.mean([s["norm"] for k, s in states.items() if k.rsplit("_", 1)[0] == n])), "T1": float(np.mean([num(s["row"]["1"]["T"]) for k, s in states.items() if k.rsplit("_", 1)[0] == n]))} for n in KEY_MAPS}
    line = lambda src, dst: 1 + (means[src]["T1"] - 1) * means[dst]["norm"] / means[src]["norm"]
    out["size_alone_block1"] = {"means": means, "ident32_at_rand32_size_by_line": line("ident32", "rand32"), "rand32_at_rand32b_size_by_line": line("rand32", "rand32b")}
    return out


def key_states_named(states: Rec) -> list[tuple[str, Rec]]:
    return [(k, s) for k, s in states.items() if k.rsplit("_", 1)[0] in KEY_MAPS]


def main() -> None:
    first, a, states, scaled = load()
    out = numbers(a, states, scaled)
    (RES / "analysis_numbers.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    controls(first)
    units(first)
    cue_pair(a, states, scaled)
    key_maps(a, states)
    size_fig(a, states, scaled, out["spearman_against_norm"]["T_block3_20"]["rho"])


if __name__ == "__main__":
    main()
