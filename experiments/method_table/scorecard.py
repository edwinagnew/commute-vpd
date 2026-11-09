"""The scorecard of cell P2.2: 67M task pairs by methods, from saved JSON only. No model, no new reading.

Run from the clone root: ~/commute-agents/fleet/cpu-run python experiments/method_table/scorecard.py
Writes results/scorecard.json (one record per cell) and figures/scorecard.png. A missing key raises KeyError.
Criteria are the three panels (built sides, shuffled twin, do-nothing state); nothing joins them into a score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCES = [
    ("TBL", "experiments/method_table/results/table.json"), ("E60", "experiments/regating_truth_table/results/table.json"),
    ("E60S", "experiments/regating_truth_table/results/stage0.json"), ("E52", "experiments/baseline_neurons/results/baseline_neurons.json"),
    ("E54", "experiments/keydependent_elicit/results/stage2.json"), ("E56A", "experiments/keydependent_patch/results/job1.json"),
    ("E56B", "experiments/keydependent_patch/results/job2.json"), ("E62", "experiments/induction_cue/results/run.json"),
    ("C3", "experiments/open_pair_instruments/results/run.json"), ("LAD", "experiments/open_pair_instruments/results/ladder.json"),
    ("A3", "experiments/open_pair/results/amend3.json"), ("A4", "experiments/open_pair/results/amend4.json"),
    ("A6", "experiments/open_pair/results/amend6.json"), ("A7", "experiments/open_pair/results/amend7.json"),
    ("NR", "experiments/neuron_r/results/cells.json"), ("NRS", "experiments/neuron_r/results/stage0.json"),
    ("CLB", "experiments/case_label_both/results/case_label_both.json"), ("CL1", "experiments/case_label/results/stage1.json"),
    ("FR1", "experiments/fresh_reuse_pair/results/stage1.json"), ("FR2", "experiments/fresh_reuse_pair/results/stage2.json"),
]
FILES = dict(SOURCES)
OUT, FIG = Path("experiments/method_table/results/scorecard.json"), Path("experiments/method_table/figures/scorecard.png")
METHODS = {"r": "re-gating share r", "r_n": "neuron version r_n", "r_half": "neuron version r_half", "S": "steering share S",
           "prior": "W0 prior for the answer", "size": "update size"}
BUILT, TWIN, NOTHING = "built_pair_sides", "shuffled_twin_sides", "do_nothing"
SHARE, P, LNP, FROB = "share of the gain", "P, geometric mean", "ln P, mean", "summed Frobenius norm of the update"
_loaded: dict[str, Any] = {}
Ref = tuple[str, tuple[Any, ...]]
CELLS: dict[tuple[str, str], dict[str, Any]] = {}


def K(name: str, *path: Any) -> Ref:
    return name, path


def at(name: str, path: tuple[Any, ...]) -> Any:
    if name not in _loaded:
        _loaded[name] = json.loads(Path(FILES[name]).read_text())
    d = _loaded[name]
    for p in path:
        d = d[p]
    return d


def read(refs: list[Ref]) -> tuple[list[float], list[dict[str, str]]]:
    vals: list[float] = []
    for name, path in refs:
        v = at(name, path)
        vals += v if isinstance(v, list) else [v]
    return vals, [{"file": FILES[n], "key": "".join(f'["{p}"]' for p in q)} for n, q in refs]


def rng(v: list[float]) -> str:
    lo, hi = min(v), max(v)
    f = (lambda x: f"{x:.1e}") if abs(hi) < 0.02 and hi != 0 else (lambda x: f"{x:.2f}")
    return f(lo) if f(lo) == f(hi) else f"{f(lo)} to {f(hi)}"


def add(pair: str, method: str, a: Any, b: list[Ref] | None = None, unit: str = SHARE, *marks: str) -> None:
    CELLS[(pair, method)] = {"a": a, "b": b, "unit": unit, "marks": list(marks)}


def neurons(rid: str, a: list[str], b: list[str], *marks: str) -> None:
    un = [] if all(at("NR", ("rows", n, "scored")) for n in b) else ["Unscored by the neuron run's own rule: held-out P(target) under its learned bar."]
    for m in ("r_n", "r_half"):
        add(rid, m, [K("NR", "rows", n, m) for n in a], [K("NR", "rows", n, m) for n in b], SHARE, *un, *marks)


def reads(pairs: list[tuple[str, str]], f: str) -> list[Ref]:
    return [K(file, "reads", key, f) for file, key in pairs]


def e56(names: list[str], field: str) -> list[Ref]:
    return [K("E56A" if int(n[-1]) < 2 else "E56B", "states", n, field) for n in names]


def e54norm(names: list[str]) -> list[Ref]:
    return [K("E54" if int(n[-1]) < 2 else "E56B", "finetunes", n, "norm") for n in names]


def run(n: str, m: str) -> Ref:
    return K("E60", "runs", n, m, "value")


COPY, RAND = ["ident32_s0", "ident32_s1", "ident32b_s0", "ident32b_s1"], ["rand32_s0", "rand32_s1", "rand32b_s0", "rand32b_s1"]
add("e46_copy_random", "r", [run(f"e46/{n}", "r") for n in COPY], [run(f"e46/{n}", "r") for n in RAND])
neurons("e46_copy_random", COPY, RAND)
add("e46_copy_random", "S", [K("E52", "b3", n, "S") for n in COPY], [K("E52", "b3", n, "S") for n in RAND])
add("e46_copy_random", "size", [K("TBL", "rows", 3, "cells", "labeller", "reading", "reuse")], [K("TBL", "rows", 3, "cells", "labeller", "reading", "new")], "root of summed squared matrix norms",
    "Different unit from the other rows' summed norm. table.json computed it from baseline_weights .states.<n>.per_matrix.*.norm2.")
LOW, CIPH, NCIPH = ["lowercase_s0", "lowercase_s1"], ["cipher_s0", "cipher_s1", "broad_cipher_s0", "broad_cipher_s1"], ["e53_cipher_s0", "e53_cipher_s1", "broad_cipher_s0", "broad_cipher_s1"]
add("e53_lowercase_cipher", "r", [run(f"e53/{n}", "r") for n in LOW], [run(f"e53/{n}", "r") for n in CIPH])
neurons("e53_lowercase_cipher", LOW, NCIPH)
add("e53_lowercase_cipher", "S", [run(f"e53/{n}", "S") for n in LOW], [run(f"e53/{n}", "S") for n in CIPH], SHARE, "Steering's fit and score halves are neighbouring windows of the same documents.")
ELS, CIS, OWNS = [f"elicit_s{s}" for s in range(5)], [f"cipher_s{s}" for s in range(5)], [f"cipher_own_s{s}" for s in range(5)]
for rid, ns, nsn in (("e54_matched", CIS, CIS), ("e54_own", OWNS, OWNS[:2])):
    add(rid, "r", e56(ELS, "r"), e56(ns, "r"))
    add(rid, "S", e56(ELS, "S"), e56(ns, "S"), SHARE, f"Log-odds share beside it, key S_lo: elicit {rng(read(e56(ELS, 'S_lo'))[0])}, cipher {rng(read(e56(ns, 'S_lo'))[0])}. table.json shows S_lo for the matched row.")
    neurons(rid, ELS, nsn, *(["The neuron run read cipher seeds 0 and 1 only."] if nsn != ns else []))
    add(rid, "size", e54norm(ELS), e54norm(ns), FROB)
    add(rid, "prior", [K("E56B", "stage0", "elicit", "rate")], [K("E56B", "stage0", "cipher", "rate")], P)
for rid, side in (("open_pair_c3", "rank1"), ("open_pair_c3_copy", "copy")):
    neurons(rid, [f"c3_{side}_s{s}" for s in range(8)], [f"c3_cipher_s{s}" for s in range(8)])
    add(rid, "r", [K("C3", "columns", "r", side)], [K("C3", "columns", "r", "cipher")])
    add(rid, "S", [K("C3", "columns", "S", side)], [K("C3", "columns", "S", "cipher")])
    add(rid, "prior", [K("C3", "columns", "prior", side)], [K("C3", "columns", "prior", "cipher")], LNP, "Saved as ln P, not P; one value repeated per seed.")
    add(rid, "size", [K("C3", "columns", "norm", side)], [K("C3", "columns", "norm", "cipher")], FROB)
for rid, a, b, (pf, pa, pb) in (("e58_rank1_96", [("A3", "rank1@30"), ("A4", "rank1_s1@30"), ("A4", "rank1_s2@30")], [("A4", f"cipher_s{s}@30") for s in range(3)], ("A7", "rank1", "cipher")),
                                ("e58_copy_prior_cipher", [("A3", "copy@30"), ("A4", "copy_s1@30"), ("A4", "copy_s2@30")], [("A6", f"prior_cipher_s{s}@30") for s in range(3)], ("A6", "copy", "prior_cipher"))):
    add(rid, "r", reads(a, "r"), reads(b, "r"))
    add(rid, "S", reads(a, "S"), reads(b, "S"))
    add(rid, "size", reads(a, "norm"), reads(b, "norm"), FROB)
    add(rid, "prior", [K(pf, "prior_beside_r", pa, "w0_geo_p_trained")], [K(pf, "prior_beside_r", pb, "w0_geo_p_trained")], P)
add("e62_copy_lookup", "r", [K("E62", "score", "r_copy")], [K("E62", "score", "r_lookup")], SHARE, f"Unscored as filed: verdict {at('E62', ('score', 'verdict'))}, r_scored {at('E62', ('score', 'r_scored'))}.")
neurons("e62_copy_lookup", [f"e62_copy_s{s}" for s in range(3)], [f"e62_lookup_s{s}" for s in range(3)])
add("e62_copy_lookup", "prior", [K("E62", "stage_b", "prior_copy")], [K("E62", "stage_b", "prior_lookup")], P)
add("e62_copy_lookup", "size", [K("E62", "states", f"copy_s{s}", "norm") for s in range(3)], [K("E62", "states", f"lookup_s{s}", "norm") for s in range(3)], FROB)
STEPS = ["10", "30", "100", "300"]
add("open_pair_ladder", "size", {t: ([K("LAD", "ladder", f"rank1_s{s}_t{t}", "norm") for s in range(3)], [K("LAD", "ladder", f"cipher_s{s}_t{t}", "norm") for s in range(3)]) for t in STEPS}, None, FROB)
# The shuffled-label twins.
ER, SH = [("A3", "elicit@30"), ("A4", "elicit_s1@30"), ("A4", "elicit_s2@30")], [("A4", "shuffle_s0@30")]
add("e58_elicit_shuffle", "r", [K("A7", "prior_beside_r", "elicit", "r30")], [K("A7", "prior_beside_r", "shuffle", "r30")])
add("e58_elicit_shuffle", "S", reads(ER, "S"), reads(SH, "S"), SHARE, "Shuffle: seed 0 only.")
add("e58_elicit_shuffle", "size", reads(ER, "norm"), reads(SH, "norm"), FROB, "Shuffle: seed 0 only.")
add("e58_elicit_shuffle", "prior", [K("A7", "prior_beside_r", "elicit", "w0_geo_p_trained")], [K("A7", "prior_beside_r", "shuffle", "w0_geo_p_trained")], P)
CM, CS = [("A7", f"class_map_s{s}@30") for s in range(3)], [("A7", f"class_shuffled_s{s}@30") for s in range(3)]
for m, f, u in (("r", "r", SHARE), ("S", "S", SHARE), ("size", "norm", FROB)):
    add("e58_class_twin", m, reads(CM, f), reads(CS, f), u)
add("e58_class_twin", "prior", [K("A7", "prior_beside_r", "class_map", "w0_geo_p_trained")], [K("A7", "prior_beside_r", "class_shuffled", "w0_geo_p_trained")], P)
REAL, TW = ["real_s0", "real_s1"], ["twin_s0", "twin_s1"]
for rid, f, base, f1, step in (("e123_case_label", "CLB", "routes", "CL1", "300"), ("e125_repeat_label", "FR2", "states", "FR1", "100")):
    add(rid, "r", [K(f, base, n, "least_squares", "shares", "r") for n in REAL], [K(f, base, n, "least_squares", "shares", "r") for n in TW], SHARE, f"Real at step 30, twin at step {step}.")
    add(rid, "size", [K(f1, "states", n, "norm") for n in REAL], [K(f1, "states", n, "norm") for n in TW], FROB, f"Real at step 30, twin at step {step}.")
# The do-nothing states.
W0T = ["broad_cipher", "c3_cipher", "c3_copy", "c3_rank1", "cipher", "e53_cipher", "e62_copy", "e62_lookup", "elicit", "ident32", "ident32b", "lowercase", "rand32", "rand32b"]
add("zero_update", "r", [K("E62", "zero_route", "gain")], None, "gain, log-odds", "E62's two tasks: gain 0, so r is undefined.")
add("zero_update", "S", [K("E60S", "rows", f"zero/{t}", "G") for t in ("e46/ident32", "e53/broad_cipher", "e53/cipher", "e53/lowercase", "e54/elicit")], None, "gain, log P", "Five tasks: gain 0, so the share is undefined.")
for m in ("r_n", "r_half"):
    add("zero_update", m, [K("NRS", "rows", f"w0/{t}", "max_abs") for t in W0T], None, "largest change of W0's log-odds", "Read on the unchanged model on 14 tasks; a share is undefined here.")
    nums = [at("NRS", ("rows", f"do_nothing/{t}", "num", m)) for t in ("elicit", "ident32")]
    add("learned_nothing", m, [K("NRS", "rows", f"do_nothing/{t}", "G") for t in ("elicit", "ident32")], None, "gain, log-odds", f"Share is nan: the gain is under the floor of 1. Its numerator reads {nums[0]:.3f} and {nums[1]:.3f}.")
add("learned_nothing", "S", [K("E52", "code_step10", "G")], None, "gain, log P per token", "A separate state, the code fine-tune. S itself was not read.")
DN = [("CLB", "routes", n) for n in REAL + TW] + [("FR2", "states", n) for n in REAL + TW]
add("do_nothing_placement", "r", [K(*d, "nothing", "shares", "r") for d in DN], None, SHARE, "The update left in Delta. Case-label states then repeat-label states; each equals that state's least-squares split value.")
assert all(abs(at(d[0], (*d[1:], "nothing", "shares", "r")) - at(d[0], (*d[1:], "least_squares", "shares", "r"))) < 1e-9 for d in DN)


def row(rid: str) -> dict[str, Any]:
    return next(r for r in at("TBL", ("rows",)) if r["id"] == rid)


TABLE_ROWS = [(r, BUILT) for r in ("e62_copy_lookup", "e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3", "open_pair_c3_copy", "e58_rank1_96", "e58_copy_prior_cipher", "open_pair_ladder")]
PAIRS = [*[(r, row(r)["name"], c, row(r)["truth"]["how"]) for r, c in TABLE_ROWS], *[(r, row(r)["name"], TWIN, row(r)["truth"]["how"]) for r in ("e58_elicit_shuffle", "e58_class_twin")],
         ("e123_case_label", "Case label: real vs memorising twin", TWIN, "only by assumption, as built; not in table.json"),
         ("e125_repeat_label", "Repeat label: real vs memorising twin", TWIN, "only by assumption, as built; not in table.json"),
         ("zero_update", "Zero update", NOTHING, "control, no label"), ("learned_nothing", "A fine-tune that learned nothing", NOTHING, "control, no label"),
         ("do_nothing_placement", "Do-nothing placement of the update", NOTHING, "control, no label")]


def records() -> list[dict[str, Any]]:
    out = []
    for pid, name, crit, label in PAIRS:
        for m in METHODS:
            base = {"pair": pid, "pair_name": name, "method": m, "criterion": crit, "label": label}
            c = CELLS.get((pid, m))
            if c is None:
                out.append({**base, "value": "not run"})
                continue
            extra = {"unit": c["unit"], "marks": c["marks"]}
            if isinstance(c["a"], dict):
                v = {t: {"reuse": read(a)[0], "new": read(b)[0]} for t, (a, b) in c["a"].items()}
                out.append({**base, "value": {"by_step": v}, "text": "; ".join(f"step {t}: {rng(x['reuse'])} / {rng(x['new'])}" for t, x in v.items()),
                            "sources": [s for a, b in c["a"].values() for s in read(a)[1] + read(b)[1]], **extra})
            elif c["b"] is None:
                v1, s1 = read(c["a"])
                out.append({**base, "value": {"reading": v1}, "text": rng(v1), "sources": s1, **extra})
            else:
                (a, sa), (b, sb) = read(c["a"]), read(c["b"])
                out.append({**base, "value": {"reuse": a, "new": b}, "text": f"{rng(a)} / {rng(b)}", "sources": sa + sb, **extra})
    return out


COLOR = {"r": "#2a78d6", "r_n": "#eb6834", "r_half": "#1baf7a", "S": "#eda100", "prior": "#e87ba4", "size": "#008300"}
INK, MUTED = "#222222", "#777777"
OFFSET = {"r": 0.3, "r_n": 0.1, "r_half": -0.1, "S": -0.3, "prior": 0.0, "size": 0.0}
SHARES = ("r", "r_n", "r_half", "S")


def dots(ax: Any, m: str, y: float, xs: list[float], filled: bool = True) -> None:
    ax.scatter(xs, [y + OFFSET[m]] * len(xs), s=26, facecolors=COLOR[m] if filled else "white", edgecolors=COLOR[m], linewidths=1.0, zorder=3)


def panel_rows(crit: str) -> list[tuple[str, str, str]]:
    out = []
    for pid, name, c, label in PAIRS:
        if c == crit:
            mark = " †" if "assumption" in label else ""
            out += [(pid, t, f"Rank-1 vs cipher ladder, step {t}{mark}") for t in STEPS] if pid == "open_pair_ladder" else [(pid, "", name + mark)]
    return out[::-1]


def figure(recs: list[dict[str, Any]]) -> None:
    rows = {c: panel_rows(c) for c in (BUILT, TWIN)}
    fig, axes = plt.subplots(3, 3, figsize=(13, 11.5), gridspec_kw={"height_ratios": [len(rows[BUILT]), len(rows[TWIN]) + 0.6, 2.8], "width_ratios": [2.3, 1, 1]})
    for j in range(3):
        axes[1, j].sharex(axes[0, j])
    axes[0, 0].set_xlim(-0.1, 1.3)
    get = {(r["pair"], r["method"]): r for r in recs}
    control = at("NRS", ("rows", "control/down_only", "r_n"))
    head = {BUILT: "Panel 1. The two built sides of a task pair", TWIN: "Panel 2. A built map against its shuffled-label twin"}
    for k, crit in enumerate((BUILT, TWIN)):
        for j, ax in enumerate(axes[k]):
            ax.set_yticks(range(len(rows[crit])), [t for *_x, t in rows[crit]] if j == 0 else [""] * len(rows[crit]), fontsize=8)
            ax.set_ylim(-0.7, len(rows[crit]) - 0.2)
            ax.grid(axis="y", color="#eeeeee", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        axes[k, 0].axvline(0, color=MUTED, lw=1)
        axes[k, 0].axvline(control, color=COLOR["r_n"], lw=1, ls=":")
        axes[k, 0].set_title(head[crit], loc="left", fontsize=9.5, color=INK)
        for y, (pid, step, _t) in enumerate(rows[crit]):
            drawn = {0: False, 1: False, 2: False}
            for m in METHODS:
                r = get[(pid, m)]
                if r["value"] == "not run":
                    continue
                col = 0 if m in SHARES else 1 if m == "prior" else 2
                drawn[col] = True
                v = r["value"]["by_step"][step] if "by_step" in r["value"] else r["value"]
                a, b = ([math.log(x) for x in v[s]] if r["unit"] == P else v[s] for s in ("reuse", "new"))
                dots(axes[k, col], m, y, a)
                dots(axes[k, col], m, y, b, False)
            for col, ok in drawn.items():
                if not ok:
                    axes[k, col].text(0.5, y, "not run", fontsize=7, color=MUTED, transform=axes[k, col].get_yaxis_transform(), ha="center", va="center")
    axes[0, 0].text(control, len(rows[BUILT]) - 0.3, f" neuron built control, output weights only: both versions {control:.3f}", fontsize=7, color=MUTED, ha="left", va="top")
    axes[0, 0].text(0, len(rows[BUILT]) - 0.3, " zero: no share of the gain", fontsize=7, color=MUTED, ha="left", va="top")
    axes[1, 0].set_xlabel("Share of the fine-tune's gain: one axis for the four shares", fontsize=8)
    axes[1, 1].set_xlabel("W0's natural log P of the trained answer", fontsize=8)
    axes[1, 2].set_xlabel("Summed Frobenius norm of the update.\nCopy vs random maps: root of summed\nsquared norms, a different unit", fontsize=8)
    axes[2, 1].remove()
    axes[2, 2].remove()
    ax = axes[2, 0]
    dn = [("learned_nothing", "A fine-tune that learned nothing"), ("zero_update", "Zero update")]
    ax.set_yticks(range(2), [n for _p, n in dn], fontsize=8)
    ax.set_ylim(-0.7, 1.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.2, 0.25)
    for y, (pid, _n) in enumerate(dn):
        for m in SHARES:
            r = get[(pid, m)]
            if r["value"] != "not run":
                dots(ax, m, y, r["value"]["reading"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Gain of the answer's score on that state; every share is undefined (log-odds for r and neurons, log P for steering)", fontsize=8)
    ax.set_title("Panel 3. What each method reads when nothing was learned; own x axis", loc="left", fontsize=9.5, color=INK)
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=COLOR[m], markeredgecolor=COLOR[m], label=METHODS[m], markersize=6) for m in METHODS]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="#555555", markeredgecolor="#555555", label="side built to reuse", markersize=6),
                Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", label="side built to be new", markersize=6)]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.97, 0.05), fontsize=8.5, frameon=False, ncol=2, title="Series (same colour on every panel)")
    fig.suptitle("How six readings sit on 67M task pairs, labels as built († by assumption). No score joins the panels.", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG, dpi=170)


if __name__ == "__main__":
    recs = records()
    OUT.write_text(json.dumps(recs, indent=1) + "\n")
    figure(recs)
    print(len(recs), "cells;", sum(r["value"] != "not run" for r in recs), "numbers;", sum(r["value"] == "not run" for r in recs), "not run")
