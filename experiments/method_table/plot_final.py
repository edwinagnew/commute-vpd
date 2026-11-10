"""P2.11, part 2's figure: every 67M task pair by every method, each cell the side built to reuse minus the side built to be new.

Reads saved results JSON only (no model) and writes results/final.json and figures/final.png (+ .pdf, .csv). Labels are as
each pair was built; no verdict of reused or new. Readers run oldest first, and a newer run's reading of a cell replaces
the older one, whose source is kept in `replaces`. A cell's `call` is the run's own filed per-pair call where one exists
(E140, E141, E143), else whether the two sides' seed ranges overlap; one value a side gets no call. No combined score.

    ~/commute-agents/fleet/cpu-run experiments/method_table/plot_final.py
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

from commute.figures import figure, gap_heatmap, save

ROOT = Path(__file__).resolve().parents[2]
SC, U, INV = "experiments/method_table/results/scorecard.json", "experiments/share_untrained/results/stage1.json", "experiments/share_untrained/results/inventory.json"
HOLES, STEER, PRIOR, SVD = "experiments/holes_fill/results/stage1.json", "experiments/steering_copy_lookup/results/stage1.json", "experiments/prior_vs_share/results/stage2.json", "experiments/update_svd/results/results.json"
TWIN2, TWIN3 = "experiments/real_vs_twin/results/stage2.json", "experiments/real_vs_twin/results/stage3.json"
BUILT = ["e62_copy_lookup", "e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3", "open_pair_c3_copy", "e58_rank1_96", "e58_copy_prior_cipher"]
TWINS = ["e58_elicit_shuffle", "e58_class_twin", "e123_case_label", "e125_repeat_label"]
COLS = {"share": "re-gating\nshare", "untrained": "share on\nuntrained\nSVD", "neuron": "neuron,\nwhole\nwidth", "neuron_half": "neuron,\nhalf", "neuron_top": "neuron,\ntop sets",
        "steering": "steering,\nbest form", "prior": "prior\n(log10)", "size": "update\nsize\n(log10)", "erank": "effective\nrank\n(log10)"}  # fmt: skip
RATIO = {"size", "erank"}  # log10 of the ratio of side means; the prior is saved as log10 P, so its gap is a difference
SCORECARD = {"r": "share", "r_n": "neuron", "r_half": "neuron_half", "S": "steering", "size": "size"}
E136_ID = {"e123_case_label": "case_label", "e125_repeat_label": "fresh_reuse_pair", "e58_class_twin": "class_twin_definition_verify"}
E140_ID = {"e123_case_label": "case", "e125_repeat_label": "repeat"}
E133_MAPS = {"e62_copy_lookup": (["e62_copy"], ["e62_lookup"]), "e46_copy_random": (["e46_ident32", "e46_ident32b"], ["e46_rand32", "e46_rand32b"]),
             "e54_matched": (["e54_elicit"], ["e54_cipher"]), "e54_own": (["e54_elicit"], ["e54_cipher"]), "open_pair_c3": (["e58_24_rank1"], ["e58_24_cipher"]),
             "open_pair_c3_copy": (["e58_24_copy"], ["e58_24_cipher"]), "e58_rank1_96": (["e58_96_rank1"], ["e58_96_cipher"]),
             "e58_copy_prior_cipher": (["e58_96_copy"], ["e58_96_prior_cipher"]), "e58_elicit_shuffle": (["e58_96_elicit"], ["e58_96_shuffle"]),
             "e58_class_twin": (["e58_96_class_map"], ["e58_96_class_shuffled"]), "e123_case_label": (["case_label_real"], ["case_label_twin"]),
             "e125_repeat_label": (["fresh_reuse_pair_real"], ["fresh_reuse_pair_twin"])}  # fmt: skip
FOOTER = (
    "67M model. Reuse side = the side built to reuse, or the real fine-tune; new side = the side built to be new, or the shuffled-label twin. Labels as built, no verdict. "
    "Cell = reuse side minus new side, as a difference of side means (shares, log10 prior) or log10 of their ratio (size, rank); n = seeds on the side with fewer; x = no saved reading. Colour is capped at ±1.\n"
    "re-gating share = part of the fine-tune's gain that appears when only which original subcomponents are switched on changes. share on untrained SVD = the same on each matrix's singular value decomposition, gated by a fixed rule. "
    "neuron, whole width = the original MLP weights run on the fine-tune's neuron firing; half = only the MLP input changed; top sets = only each block's top neurons kept. "
    "steering = part of the gain from adding the fine-tune's mean activation shift at the read position (pooled, or one vector per key or word; the wider gap is shown). "
    "prior = the original model's probability of the trained answer. update size = norm of the weight change. effective rank = how many directions the weight change spreads over."
)
Side = tuple[list[float], list[float]]  # (side built to reuse, side built to be new)
Cells = dict[tuple[str, str], dict[str, Any]]


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def put(out: Cells, pair: str, col: str, side: Side, src: str, *, form: str = "", filed: Any = None) -> None:
    old = out.get((pair, col), {}).get("source", "")
    out[(pair, col)] = {"reuse": side[0], "new": side[1], "n_reuse": len(side[0]), "n_new": len(side[1]), "source": src, "form": form, "filed_call": filed, "replaces": old}


def wider(sides: dict[str, Side]) -> str:
    """The kind with the wider |gap|: the rule E135 and E143 filed for steering's two kinds and the neuron's two set sizes."""
    whole = {k: s for k, s in sides.items() if None not in s[0] + s[1]}  # no per-key vector on lowercase vs ciphers
    return max(whole, key=lambda k: abs(statistics.fmean(whole[k][0]) - statistics.fmean(whole[k][1])))


def scorecard(out: Cells) -> None:
    for r in load(SC):
        if r["method"] in SCORECARD and r["pair"] in BUILT + TWINS and isinstance(r["value"], dict) and "reuse" in r["value"]:
            src = "; ".join(f"{s['file']} {s['key']}" for s in r["sources"])
            put(out, r["pair"], SCORECARD[r["method"]], (r["value"]["reuse"], r["value"]["new"]), src, form="one pooled vector, older form" if r["method"] == "S" else "")


def e133_prior(out: Cells) -> None:
    pts = {q["name"]: q["log10_prior"] for q in load(PRIOR)["points"]}
    for pair, (a, b) in E133_MAPS.items():
        put(out, pair, "prior", ([pts[n] for n in a], [pts[n] for n in b]), f"{PRIOR} .points[] named {a + b}, .log10_prior", form="one value per map")


def e135_steering(out: Cells) -> None:
    st = load(STEER)["states"]
    sides = {k: ([st[f"copy_s{i}"]["steering"][k]["share_lo"] for i in range(3)], [st[f"lookup_s{i}"]["steering"][k]["share_lo"] for i in range(3)]) for k in ("pooled", "perkey")}
    k = wider(sides)
    put(out, "e62_copy_lookup", "steering", sides[k], f"{STEER} .states[<state>].steering.{k}.share_lo", form=k)


def e136_share(out: Cells) -> None:
    states, pairs = load(U)["states"], {p["id"]: p for p in load(INV)["pairs"]}
    for pair in BUILT + TWINS:
        if (p := pairs.get(E136_ID.get(pair, pair))) is not None:
            form = "regenerated seeds 0-1" if pair == "e58_class_twin" else ""
            for col, key in (("share", "trained"), ("untrained", "svd")):
                put(out, pair, col, ([states[n][key] for n in p["reuse"]], [states[n][key] for n in p["new"]]), f"{U} .states[<state>].{key}", form=form)


def e141_rank(out: Cells) -> None:
    for pair, p in load(SVD)["pairs"].items():
        side = lambda key, p=p: ([v[key] for v in p["reuse"].values()], [v[key] for v in p["new"].values()])
        put(out, pair, "erank", side("erank"), f"{SVD} .pairs.{pair}.<side>[<state>].erank", filed=p["score"]["erank"]["separates"])
        if (pair, "size") not in out:  # fills a hole only: a different norm from the scorecard's summed one
            put(out, pair, "size", side("norm2"), f"{SVD} .pairs.{pair}.<side>[<state>].norm2", form="root of summed squared matrix norms")


def e140_twins(out: Cells) -> None:
    shares, t3 = load(TWIN2)["states"], load(TWIN3)
    for pair, w in E140_ID.items():
        arm = lambda a, w=w: [f"{w}_{a}_s{i}" for i in range(3)]
        put(out, pair, "share", ([shares[n]["share"]["contrast"] for n in arm("real")], [shares[n]["share"]["contrast"] for n in arm("twin")]),
            f"{TWIN2} .states[<state>].share.contrast", form="label contrast", filed=t3["score"]["contrast"]["share"][w]["call"])  # fmt: skip
        for col, grp, names in (("steering", "steering", {"perkey": "steer_perword", "pooled": "steer_pooled"}), ("neuron_top", "neuron", {"k64": "neuron_k64", "kmatched": "neuron_kmatched"})):
            read = lambda a, k, grp=grp, arm=arm: [t3["states"][n][grp][k]["share"]["contrast"] for n in arm(a)]
            sides = {k: (read("real", k), read("twin", k)) for k in names}
            k = wider(sides)
            put(out, pair, col, sides[k], f"{TWIN3} .states[<state>].{grp}.{k}.share.contrast", form=f"{'per word' if k == 'perkey' else k}, label contrast",
                filed=t3["score"]["contrast"][names[k]][w]["call"])  # fmt: skip


def e143_holes(out: Cells) -> None:
    h, pairs = load(HOLES), {p["id"]: p for p in load(INV)["pairs"]}
    for col, fam, kinds in (("steering", "steering", ("pooled", "perkey")), ("neuron_top", "neuron", ("k64", "kmatched"))):
        for pair, filed in h["score"]["families"][fam].items():
            get = lambda n, k, fam=fam: h["states"][n][fam].get(k, {}).get("share")
            sides = {k: ([get(n, k) for n in pairs[pair]["reuse"]], [get(n, k) for n in pairs[pair]["new"]]) for k in kinds}
            k = wider(sides)
            put(out, pair, col, sides[k], f"{HOLES} .states[<state>].{fam}.{k}.share", form=k, filed=filed["call"])


READERS = (scorecard, e133_prior, e135_steering, e136_share, e141_rank, e140_twins, e143_holes)  # oldest first


def score(col: str, c: dict[str, Any]) -> None:
    """Adds the gap and the call ('apart', 'no call' or 'one value a side') to one cell, in place."""
    reuse, new = statistics.fmean(c["reuse"]), statistics.fmean(c["new"])
    c["gap"] = math.log10(reuse / new) if col in RATIO else reuse - new
    if (f := c["filed_call"]) is not None:
        c["call"], c["call_rule"] = ("apart" if f in ("separates", "apart", True) else "no call"), f"the run's filed call: {f}"
    elif min(c["n_reuse"], c["n_new"]) < 2:
        c["call"], c["call_rule"] = "one value a side", "no spread to compare"
    else:
        c["call"] = "apart" if min(c["reuse"]) > max(c["new"]) or max(c["reuse"]) < min(c["new"]) else "no call"
        c["call_rule"] = "seed ranges"


def against_share(cells: Cells, pair: str, col: str) -> str:
    c, s = cells.get((pair, col)), cells[(pair, "share")]
    if c is None:
        return "not run"
    if c["call"] != "apart" or col == "share":
        return c["call"]
    if s["call"] != "apart":
        return "apart where the share is not"
    return "same direction as the share" if (c["gap"] > 0) == (s["gap"] > 0) else "opposite to the share"


def main() -> None:
    cells: Cells = {}
    for read in READERS:
        read(cells)
    names, rows = {r["pair"]: r["pair_name"] for r in load(SC)}, BUILT + TWINS
    tally: dict[str, dict[str, dict[str, int]]] = {g: {col: {} for col in COLS} for g in ("built", "twin")}
    for (pair, col), c in cells.items():
        score(col, c)
    for pair in rows:
        for col in COLS:
            v = against_share(cells, pair, col)
            t = tally["built" if pair in BUILT else "twin"][col]
            t[v] = t.get(v, 0) + 1
            if (pair, col) in cells:
                cells[(pair, col)]["against_share"] = v
    records = [{"pair": p, "pair_name": names[p], "group": "built" if p in BUILT else "twin", "method": col, **cells[(p, col)]} for p in rows for col in COLS if (p, col) in cells]
    (ROOT / "experiments/method_table/results/final.json").write_text(json.dumps({"cells": records, "tally": tally}, indent=1) + "\n")
    values = [[round(cells[(p, c)]["gap"], 2) + 0.0 if (p, c) in cells else None for c in COLS] for p in rows]
    counts = [[min(cells[(p, c)]["n_reuse"], cells[(p, c)]["n_new"]) if (p, c) in cells else None for c in COLS] for p in rows]
    fig, ax = figure((11.0, 7.6))
    labels = [("built: " if p in BUILT else "twin: ") + names[p] for p in rows]
    gap_heatmap(ax, values, labels, list(COLS.values()), label="reuse side minus new side (colour capped at ±1)", vmax=1.0, counts=counts)
    ax.axhline(len(BUILT) - 0.5, color="black", linewidth=1.2)
    ax.xaxis.tick_top()
    ax.set_title("The re-gating share sets all nine built pairs apart the same way; steering, neurons, size and rank each miss or reverse\n"
                 "some, and the prior, one value a side, points the same way on all eight it was read on", fontsize=10)  # fmt: skip
    keep = ("pair_name", "method", "form", "gap", "n_reuse", "n_new", "call", "call_rule", "against_share", "source", "replaces")
    save(fig, ROOT / "experiments/method_table/figures/final.png", n=FOOTER, data=[{k: r[k] for k in keep} for r in records])


if __name__ == "__main__":
    main()
