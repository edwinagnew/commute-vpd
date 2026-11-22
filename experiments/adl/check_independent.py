"""P2.60 independent check: recompute each method's nine built-pair gaps and compare with E_b.

Written without reading compare.py / run.py. Gap = mean(reuse side) - mean(new side).
Spearman is implemented here (Pearson on average ranks, ties averaged); scipy is used only
as a cross-check of that implementation.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RB = ROOT / "experiments/rb_states/results"
LENS = ROOT / "experiments/adl/results/amend1/items"
FINAL = ROOT / "experiments/method_table/results/final.json"
XC = ROOT / "experiments/crosscoder/results/run2/summary.json"
OUT = ROOT / "experiments/adl/results/amend1/check_independent.json"
TABLE_METHODS = ("share", "neuron", "steering")


def load(path):
    return json.loads(path.read_text())


def base_state(item):
    """'e62/copy_s0' stays; 'e62-lookup_s0_scaled_e62_copy_lookup' -> 'e62/lookup_s0'."""
    if "/" in item:
        return item
    name = item.split("_scaled_")[0]
    exp, rest = name.split("-", 1)
    return f"{exp}/{rest}"


def avg_ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def rel_diff(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def e_b(surprises):
    summary = load(RB / "summary_items.json")["pairs"]
    gaps, counts, states = {}, {}, {}
    for key, rec in summary.items():
        pair = key.removeprefix("e187_")
        side_vals = {}
        for side in ("reuse", "new"):
            vals = []
            for it in rec[side]["items"]:
                v = load(RB / "items" / (it["item"].replace("/", "-") + ".json"))["E_b"]
                if rel_diff(v, it["E_b"]) > 1e-9:
                    surprises.append(f"E_b item {it['item']}: file {v} vs summary item {it['E_b']}")
                vals.append(v)
            side_vals[side] = vals
            if rel_diff(float(np.mean(vals)), rec[side]["E_b"]) > 1e-9:
                surprises.append(f"E_b {pair} {side}: mean {np.mean(vals)} vs summary {rec[side]['E_b']}")
            states[(pair, side)] = sorted(base_state(it["item"]) for it in rec[side]["items"])
        gaps[pair] = float(np.mean(side_vals["reuse"]) - np.mean(side_vals["new"]))
        counts[pair] = {s: len(side_vals[s]) for s in side_vals}
    return gaps, counts, states


def lens(pairs):
    gaps, counts, states = {}, {}, {}
    for pair in pairs:
        side_vals = {"reuse": [], "new": []}
        side_states = {"reuse": [], "new": []}
        for f in sorted(LENS.glob(f"{pair}-*.json")):
            if f.name in (f"{pair}-random_reuse.json", f"{pair}-random_new.json"):
                continue
            d = load(f)
            if d["pair"] != pair:  # guards a prefix clash between pair ids
                continue
            assert not d["random"], f.name
            side_vals[d["side"]].append(d["D"])
            side_states[d["side"]].append(d["state"])
        gaps[pair] = float(np.mean(side_vals["reuse"]) - np.mean(side_vals["new"]))
        counts[pair] = {s: len(v) for s, v in side_vals.items()}
        for s in side_states:
            states[(pair, s)] = sorted(side_states[s])
    return gaps, counts, states


def table(pairs, surprises):
    cells = [c for c in load(FINAL)["cells"] if c["group"] == "built"]
    names = {}
    out = {}
    for m in TABLE_METHODS:
        gaps, counts = {}, {}
        for pair in pairs:
            (c,) = [c for c in cells if c["pair"] == pair and c["method"] == m]
            names[pair] = c["pair_name"]
            g = float(np.mean(c["reuse"]) - np.mean(c["new"]))
            if rel_diff(g, c["gap"]) > 1e-9:
                surprises.append(f"final.json {m} {pair}: stored gap {c['gap']} vs recomputed {g}")
            gaps[pair] = g
            counts[pair] = {"reuse": len(c["reuse"]), "new": len(c["new"])}
        out[m] = (gaps, counts)
    return out, names


def crosscoder(pairs, names, surprises):
    cells = [c for c in load(XC)["cells"] if c["column"] == "crosscoder"]
    gaps, counts, sides = {}, {}, {}
    for pair in pairs:
        (c,) = [c for c in cells if c["row"] == names[pair]]
        g = float(np.mean(c["values"]["reuse"]) - np.mean(c["values"]["new"]))
        if rel_diff(g, c["gap"]) > 1e-9:
            surprises.append(f"crosscoder {pair}: stored gap {c['gap']} vs recomputed {g}")
        gaps[pair] = g
        counts[pair] = {s: len(c["values"][s]) for s in ("reuse", "new")}
        sides[pair] = c.get("sides")
    return gaps, counts, sides


def compare(gaps, ref, pairs):
    a = [gaps[p] for p in pairs]
    b = [ref[p] for p in pairs]
    rho = spearman(a, b)
    rho_scipy = float(spearmanr(a, b).statistic)
    assert abs(rho - rho_scipy) < 1e-12, (rho, rho_scipy)
    signs = sum((x > 0) == (y > 0) for x, y in zip(a, b))
    return rho, int(signs)


def main():
    surprises = []
    eb_gaps, eb_counts, eb_states = e_b(surprises)
    pairs = sorted(eb_gaps)
    methods = {"E_b": (eb_gaps, eb_counts)}
    lens_gaps, lens_counts, lens_states = lens(pairs)
    methods["lens"] = (lens_gaps, lens_counts)
    tab, names = table(pairs, surprises)
    methods.update(tab)
    xc_gaps, xc_counts, xc_sides = crosscoder(pairs, names, surprises)
    methods["crosscoder"] = (xc_gaps, xc_counts)

    for key in sorted(eb_states):
        if eb_states[key] != lens_states.get(key):
            surprises.append(f"state sets differ {key}: E_b {eb_states[key]} vs lens {lens_states.get(key)}")
    for pair in pairs:
        per = {m: methods[m][1][pair] for m in methods if m != "crosscoder"}
        if len({json.dumps(v, sort_keys=True) for v in per.values()}) > 1:
            surprises.append(f"state counts differ for {pair}: {per}")

    result = {"pairs": pairs, "pair_names": names, "methods": {}, "crosscoder_sides": xc_sides}
    print(f"{'method':<11} {'spearman':>9} {'signs/9':>8}")
    for m, (gaps, counts) in methods.items():
        rho, signs = compare(gaps, eb_gaps, pairs)
        result["methods"][m] = {"gaps": gaps, "spearman_vs_E_b": rho, "sign_agreement": signs, "n_states": counts}
        print(f"{m:<11} {rho:9.4f} {signs:8d}")
    neg = sum(g < 0 for g in lens_gaps.values())
    result["lens_negative_gaps"] = neg
    print(f"lens negative gaps: {neg}/9")
    print(f"{'pair':<24} {'E_b gap':>12} {'lens gap':>12}")
    for p in pairs:
        print(f"{p:<24} {eb_gaps[p]:12.6f} {lens_gaps[p]:12.6f}")
    result["surprises"] = surprises
    print("surprises:")
    for s in surprises:
        print(" -", s)
    OUT.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
