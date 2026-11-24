"""P2.11, part 2's figure: every 67M task pair by every method, each cell the side built to reuse minus the side built to be new.

Reads saved results JSON only (no model) and writes results/final.json and figures/final.png (+ .pdf, .csv). Labels are as
each pair was built; no verdict of reused or new. Readers run oldest first, and a newer run's reading of a cell replaces
the older one, whose source is kept in `replaces`. Every column has one unit (`UNIT`, written into each cell) and one
selection rule; a pair with no saved reading in that unit or form holds a null cell that says why. Where a reading comes
in two forms (steering pooled and per key; top neurons at k = 64 and at the matched k), each form is its own column, so no
form is chosen per pair. One rule colours every cell: with two or more values a side,
"apart" iff the seed ranges do not overlap; one value on a side gets no call. A run's own filed call is kept in `filed_call` for
reference and does not set the colour. No combined score.

    ~/commute-agents/fleet/cpu-run experiments/method_table/plot_final.py
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

from commute.figures import figure, save

ROOT = Path(__file__).resolve().parents[2]
SC, U, INV = "experiments/method_table/results/scorecard.json", "experiments/share_untrained/results/stage1.json", "experiments/share_untrained/results/inventory.json"
HOLES, STEER, PRIOR, SVD = "experiments/holes_fill/results/stage1.json", "experiments/steering_copy_lookup/results/stage1.json", "experiments/prior_vs_share/results/stage2.json", "experiments/update_svd/results/results.json"
E145 = "experiments/prior_lowercase/results/stage1.json"
E147, E147_INV = "experiments/refit_96key/results/stage1.json", "experiments/refit_96key/results/inventory.json"
TWIN2, TWIN3 = "experiments/real_vs_twin/results/stage2.json", "experiments/real_vs_twin/results/stage3.json"
BUILT = ["e62_copy_lookup", "e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3", "open_pair_c3_copy", "e58_rank1_96", "e58_copy_prior_cipher"]
TWINS = ["e58_elicit_shuffle", "e58_class_twin", "e123_case_label", "e125_repeat_label"]
COLS = {"share": "re-gating\nshare", "untrained": "share on\nuntrained\nSVD", "neuron": "neuron,\nwhole\nwidth", "neuron_half": "neuron,\nhalf",
        "neuron_top_k64": "top 64\nneurons", "neuron_top_kmatched": "top\nneurons,\nmatched k", "steering_pooled": "steering,\npooled",
        "steering_per_key": "steering,\nper key", "prior": "prior\n(log10)", "size": "update\nsize\n(log10)", "erank": "effective\nrank\n(log10)",
        "k_half": "neurons for\nhalf the gain\n(log10)"}  # fmt: skip
LO = "share of the fine-tune's gain in log-odds of the target at the answer position"
PICK = ("setting (block, strength) with the largest mean log-odds rise on the state's selection rows, ties to the lower block "
        "then the lower strength (E135's and E143's filed rule); read in log-odds")  # fmt: skip
UNIT = {"share": f"re-gating {LO}", "untrained": f"re-gating {LO}, on the untrained SVD split", "neuron": f"neuron {LO}, whole width",
        "neuron_half": f"neuron {LO}, half", "neuron_top_k64": f"top-64-neuron {LO}", "neuron_top_kmatched": f"top-neuron {LO}, k matched",
        "steering_pooled": f"steering {LO}, one vector pooled over keys; {PICK}", "steering_per_key": f"steering {LO}, one vector per key; {PICK}",
        "prior": "log10 of W0's geometric-mean probability of the target", "size": "summed Frobenius norm of the update over its matrices (summed_norm)",
        "erank": "effective rank of the update", "k_half": "fewest top neurons patched to carry half the gain"}  # fmt: skip
NOT_UNIT = "not read in this unit"
RATIO = {"size", "erank"}  # log10 of the ratio of side means; the prior is saved as log10 P, so its gap is a difference
INVERTED = {"k_half"}  # log10(new mean / reuse mean): positive (red) = fewer neurons on the reuse side
E98_CARRY, E98_PRED, KHALF_FILL = "experiments/neuron_r/results/carrying.json", "experiments/neuron_r/results/predictions.json", "experiments/k_half_fill/results/cells.json"
SCORECARD = {"r": "share", "r_n": "neuron", "r_half": "neuron_half", "size": "size"}  # its steering (one older vector) is neither form
E136_ID = {"e123_case_label": "case_label", "e125_repeat_label": "fresh_reuse_pair", "e58_class_twin": "class_twin_definition_verify"}
E140_ID = {"e123_case_label": "case", "e125_repeat_label": "repeat"}
STEER_COLS, NEURON_COLS = {"pooled": "steering_pooled", "perkey": "steering_per_key"}, {"k64": "neuron_top_k64", "kmatched": "neuron_top_kmatched"}
E133_MAPS = {"e62_copy_lookup": (["e62_copy"], ["e62_lookup"]), "e46_copy_random": (["e46_ident32", "e46_ident32b"], ["e46_rand32", "e46_rand32b"]),
             "e54_matched": (["e54_elicit"], ["e54_cipher"]), "e54_own": (["e54_elicit"], ["e54_cipher"]), "open_pair_c3": (["e58_24_rank1"], ["e58_24_cipher"]),
             "open_pair_c3_copy": (["e58_24_copy"], ["e58_24_cipher"]), "e58_rank1_96": (["e58_96_rank1"], ["e58_96_cipher"]),
             "e58_copy_prior_cipher": (["e58_96_copy"], ["e58_96_prior_cipher"]), "e58_elicit_shuffle": (["e58_96_elicit"], ["e58_96_shuffle"]),
             "e58_class_twin": (["e58_96_class_map"], ["e58_96_class_shuffled"]), "e123_case_label": (["case_label_real"], ["case_label_twin"]),
             "e125_repeat_label": (["fresh_reuse_pair_real"], ["fresh_reuse_pair_twin"])}  # fmt: skip
FOOTER = (
    "Each cell: reuse side minus new side (shares and log10 prior: difference of means; size and rank: log10 of reuse/new mean; neurons for half the gain: log10 of new/reuse mean, "
    "so red = fewer neurons on the reuse side). Seeds per side: 1 to 8. One rule in every column: clearly = every seed of one side beyond every seed of the other. Colour is a call, not a score."
)
Side = tuple[list[float], list[float]]  # (side built to reuse, side built to be new)
Cells = dict[tuple[str, str], dict[str, Any]]


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def put(out: Cells, pair: str, col: str, side: Side, src: str, *, form: str = "", filed: Any = None) -> None:
    old = out.get((pair, col), {}).get("source", "")
    out[(pair, col)] = {"reuse": side[0], "new": side[1], "n_reuse": len(side[0]), "n_new": len(side[1]), "source": src, "form": form, "filed_call": filed, "replaces": old}


def hole(out: Cells, pair: str, col: str, why: str) -> None:
    """A null cell: the pair has no saved reading in this column's unit or form; `why` names what is saved instead."""
    old = out.get((pair, col), {}).get("source", "")
    out[(pair, col)] = {"reuse": None, "new": None, "n_reuse": 0, "n_new": 0, "source": "", "form": why, "filed_call": None, "replaces": old}


def scorecard(out: Cells) -> None:
    for r in load(SC):
        if r["method"] in SCORECARD and r["pair"] in BUILT + TWINS and isinstance(r["value"], dict) and "reuse" in r["value"]:
            src = "; ".join(f"{s['file']} {s['key']}" for s in r["sources"])
            put(out, r["pair"], SCORECARD[r["method"]], (r["value"]["reuse"], r["value"]["new"]), src, form="one pooled vector, older form" if r["method"] == "S" else "")


def e133_prior(out: Cells) -> None:
    pts = {q["name"]: q["log10_prior"] for q in load(PRIOR)["points"]}
    for pair, (a, b) in E133_MAPS.items():
        put(out, pair, "prior", ([pts[n] for n in a], [pts[n] for n in b]), f"{PRIOR} .points[] named {a + b}, .log10_prior", form="one value per map")


def e145_prior(out: Cells) -> None:
    """E133's prior on E53's windows (rebuilt), one value per task: the lowercase-against-ciphers cell E133 left empty."""
    t = load(E145)["tasks"]
    put(out, "e53_lowercase_cipher", "prior", ([t["lowercase"]["log10_prior"]], [t["cipher"]["log10_prior"], t["broad_cipher"]["log10_prior"]]),
        f"{E145} .tasks.<task>.log10_prior", form="one value per task")


def e135_steering(out: Cells) -> None:
    st = load(STEER)["states"]
    for k, col in STEER_COLS.items():
        side = ([st[f"copy_s{i}"]["steering"][k]["share_lo"] for i in range(3)], [st[f"lookup_s{i}"]["steering"][k]["share_lo"] for i in range(3)])
        put(out, "e62_copy_lookup", col, side, f"{STEER} .states[<state>].steering.{k}.share_lo", form="E135's own fp32 copies")


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


def e140_twins(out: Cells) -> None:
    """E140's log-odds readings (`.lo`). Its steering setting was picked by the largest label-contrast rise, not the
    log-odds rise, so its steering is outside the steering columns' rule and those cells stay null."""
    shares, t3 = load(TWIN2)["states"], load(TWIN3)
    filed = t3["score"]["lo"]
    for pair, w in E140_ID.items():
        arm = lambda a, w=w: [f"{w}_{a}_s{i}" for i in range(3)]
        put(out, pair, "share", ([shares[n]["share"]["lo"] for n in arm("real")], [shares[n]["share"]["lo"] for n in arm("twin")]),
            f"{TWIN2} .states[<state>].share.lo", filed=filed["share"][w]["call"])  # fmt: skip
        for k, col in NEURON_COLS.items():
            read = lambda a, k=k, arm=arm: [t3["states"][n]["neuron"][k]["share"]["lo"] for n in arm(a)]
            put(out, pair, col, (read("real"), read("twin")), f"{TWIN3} .states[<state>].neuron.{k}.share.lo", filed=filed[f"neuron_{k}"][w]["call"])
        for col in STEER_COLS.values():
            hole(out, pair, col, f"not read under this selection rule (setting picked by label-contrast rise: {TWIN3} .states[<state>].steering.<form>.share.lo)")


def e143_holes(out: Cells) -> None:
    h, pairs = load(HOLES), {p["id"]: p for p in load(INV)["pairs"]}
    for fam, cols in (("steering", STEER_COLS), ("neuron", NEURON_COLS)):
        for pair in h["score"]["families"][fam]:  # its filed calls are on the wider form, so none is kept per column
            for k, col in cols.items():
                get = lambda n, k=k, fam=fam: h["states"][n][fam].get(k, {}).get("share")
                side = ([get(n) for n in pairs[pair]["reuse"]], [get(n) for n in pairs[pair]["new"]])
                if None in side[0] + side[1]:
                    hole(out, pair, col, f"not read in this form (no {k} vector in {HOLES} for this pair)")
                else:
                    put(out, pair, col, side, f"{HOLES} .states[<state>].{fam}.{k}.share")


def e147_refit(out: Cells) -> None:
    """E147 refit E58's four 96-key pairs (seeds 0-2, step 30, updates kept) and read every activation reader on them. Its
    share replaces E58's saved share and E136's two regenerated class seeds (it reproduced E58's within each map's seed range,
    largest difference 6e-4), so each row reads one set of updates; its fair-form steering replaces the older one-vector form."""
    r, inv = load(E147), load(E147_INV)
    st, calls = r["states"], r["score"]["calls"]  # E147's filed calls (E143's rule) for every cell
    for p in inv["pairs"]:
        names = ([f"e58/{p['reuse']}_s{s}" for s in inv["seeds"]], [f"e58/{p['new']}_s{s}" for s in inv["seeds"]])
        side = lambda path, names=names: ([float(dig(st[n], path)) for n in names[0]], [float(dig(st[n], path)) for n in names[1]])
        for col, grp, key in (("share", "e136", "trained"), ("untrained", "e136", "svd"), ("neuron", "neuron_r", "r_n"), ("neuron_half", "neuron_r", "r_half")):
            put(out, p["id"], col, side([grp, key]), f"{E147} .states[<state>].{grp}.{key}", form="refit seeds 0-2", filed=calls[col][p["id"]]["call"])
        for f, cols, pre in (("neuron", NEURON_COLS, "neuron_"), ("steering", STEER_COLS, "steer_")):
            for k, col in cols.items():
                put(out, p["id"], col, side(["holes_fill", f, k, "share"]), f"{E147} .states[<state>].holes_fill.{f}.{k}.share", form="refit seeds 0-2",
                    filed=calls[pre + k][p["id"]]["call"])  # fmt: skip


def e158_neuron_twins(out: Cells) -> None:
    """E158 read the two label twins' neuron shares in the label contrast only, so those cells stay null."""
    for key, s in load("experiments/neuron_twins/results/cells.json")["cells"].items():
        pair, col = key.split("/")
        hole(out, pair, col, f"{NOT_UNIT} (label contrast only: {s['source']})")


def e53_size(out: Cells) -> None:
    """Lowercase vs ciphers has only update_svd's summed squared norms saved, not every other pair's summed norm."""
    hole(out, "e53_lowercase_cipher", "size", f"{NOT_UNIT} (summed squared norms only: {SVD} .pairs.e53_lowercase_cipher.<side>[<state>].norm2)")


def refit_rank(out: Cells) -> None:
    """Effective rank on E147's refit 96-key updates, E141's reading and pair rule (experiments/refit_rank)."""
    for key, s in load("experiments/refit_rank/results/cells.json")["cells"].items():
        pair, col = key.split("/")
        put(out, pair, col, (s["reuse"], s["new"]), s["source"], form="refit seeds 0-2", filed=s["filed"])


def e98_k_half(out: Cells) -> None:
    """E98's k_half (fewest top neurons patched to carry half the gain) on its own rows; states it did not score are left out."""
    k = {n: r["k_half"] for n, r in load(E98_CARRY)["rows"].items() if r["k_half"] is not None}
    for pair, row in load(E98_PRED)["rows"].items():
        side = ([k[n] for n in row["reuse"] if n in k], [k[n] for n in row["new"] if n in k])
        if pair in BUILT and all(side):
            put(out, pair, "k_half", side, f"{E98_CARRY} .rows[<state>].k_half, states from {E98_PRED} .rows.{pair}")
    if (ROOT / KHALF_FILL).exists():
        for key, s in load(KHALF_FILL)["cells"].items():
            pair, col = key.split("/")
            put(out, pair, col, (s["reuse"], s["new"]), s["source"])


def dig(x: Any, path: list[str]) -> Any:
    for k in path:
        x = x[k]
    return x


READERS = (scorecard, e133_prior, e145_prior, e135_steering, e136_share, e141_rank, e53_size, e140_twins, e143_holes, e147_refit, e158_neuron_twins, refit_rank, e98_k_half)  # oldest first


def score(col: str, c: dict[str, Any]) -> None:
    """Adds the unit, the gap and the call ('apart', 'no call' or 'one value a side') to one cell, in place: one rule for every column."""
    c["unit"] = UNIT[col]
    if c["reuse"] is None:  # a null cell: its call is the reason, without the saved path
        c["gap"], c["gap_rule"], c["call"], c["call_rule"] = None, "", c["form"].split(" (")[0], ""
        return
    reuse, new = statistics.fmean(c["reuse"]), statistics.fmean(c["new"])
    c["gap"] = math.log10(reuse / new) if col in RATIO else math.log10(new / reuse) if col in INVERTED else reuse - new
    c["gap_rule"] = ("log10(reuse mean / new mean)" if col in RATIO else "log10(new mean / reuse mean)" if col in INVERTED
                     else "reuse mean - new mean")  # each column's sign convention, so a writer need not read this file
    if min(c["n_reuse"], c["n_new"]) < 2:
        c["call"], c["call_rule"] = "one value a side", "no spread to compare"
    else:
        c["call"] = "apart" if min(c["reuse"]) > max(c["new"]) or max(c["reuse"]) < min(c["new"]) else "no call"
        c["call_rule"] = "seed ranges"


def against_share(cells: Cells, pair: str, col: str) -> str:
    c, s = cells.get((pair, col)), cells[(pair, "share")]
    if c is None:
        return "not run"
    if c["gap"] is None or c["call"] != "apart" or col == "share":
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
    (ROOT / "experiments/method_table/results/final.json").write_text(json.dumps({"units": UNIT, "cells": records, "tally": tally}, indent=1) + "\n")
    values = [[cells[(p, c)]["gap"] if (p, c) in cells else None for c in COLS] for p in rows]
    kinds = [[kind(cells.get((p, c))) for c in COLS] for p in rows]
    fig, ax = figure((10.0, 6.4))
    verdict_grid(ax, values, kinds, [names[p] for p in rows], list(COLS.values()))
    ax.axhline(len(BUILT) - 0.5, color="black", linewidth=1.5)
    for y, text in ((len(BUILT) / 2 - 0.5, "built pairs"), (len(BUILT) + len(TWINS) / 2 - 0.5, "real vs shuffled twin")):
        ax.text(len(COLS) - 0.35, y, text, rotation=270, ha="left", va="center", fontsize=9, style="italic", clip_on=False)
    ax.set_title("Only the re-gating share separates all nine built pairs, reuse side higher", fontsize=11, pad=8)
    keep = ("pair_name", "method", "unit", "form", "gap", "n_reuse", "n_new", "call", "call_rule", "against_share", "source", "replaces")
    save(fig, ROOT / "experiments/method_table/figures/final.png", n=FOOTER, data=[{k: r[k] for k in keep} for r in records])


KIND_COLOUR = {"up": "#D6604D", "down": "#4393C3", "up1": "#F4C7B8", "down1": "#C6DBEF", "none": "#E8E8E8", "missing": "#FFFFFF"}
KIND_LABEL = {"up": "reuse side clearly higher", "down": "new side clearly higher", "up1": "reuse side higher, one value a side",
              "down1": "new side higher, one value a side", "none": "no clear difference"}  # fmt: skip


def kind(c: dict[str, Any] | None) -> str:
    if c is None or c["gap"] is None:
        return "missing"
    if c["call"] == "no call":
        return "none"
    return ("up" if c["gap"] > 0 else "down") + ("1" if c["call"] == "one value a side" else "")


def verdict_grid(ax: Any, values: list[list[float | None]], kinds: list[list[str]], rows: list[str], cols: list[str]) -> None:
    """Each cell coloured by its call and direction, with the gap printed; x for not measured."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    ax.imshow([[to_rgb(KIND_COLOUR[k]) for k in row] for row in kinds], aspect="auto")
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            text = "×" if v is None else f"{v:.3f}" if 0.0005 <= abs(v) < 0.005 else f"{v:.2f}".replace("-0.00", "0.00")  # a gap under 0.005 keeps a third digit
            ax.text(j, i, text, ha="center", va="center", color="white" if kinds[i][j] in ("up", "down") else "#333333", fontsize=8.5)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.set_xticks([x + 0.5 for x in range(len(cols) - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(rows) - 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.spines[:].set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.legend(handles=[Patch(facecolor=KIND_COLOUR[k], edgecolor="#999999", label=t) for k, t in KIND_LABEL.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8)  # fmt: skip


if __name__ == "__main__":
    main()
