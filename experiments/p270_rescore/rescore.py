"""P2.70 (part): update size under one norm, steering per form, and the steering axis of the prior-and-steering figure,
rescored from saved results only (no model, no GPU). Source: redteam-lead's audit
reports/reviews/2026-11-23_review_baseline-fidelity.md, rows 16, 17, 18 and 21.

One rule throughout, the scorecard's (method_table/plot_final.py): a pair's sides are "apart" when every seed of one lies
beyond every seed of the other; one value on a side gets no call. A reading "agrees" with the share when it is apart with
the share's sign, "reverses" when apart with the other sign; the share's gap is positive on all nine built pairs.

1. Size. One norm, ||dW||_F / ||W0|| over the 24 matrices, read by E187 stage 0(a) on the as-saved updates
   (size_matched/results/m67.json .pairs.<p>.sizes). Each value is named by the state list E187 built from E136's and
   E147's inventories (size_matched/run.py pairs()), checked against the per-state sizes m67 kept for the larger side,
   and the names are checked to be the states whose share the scorecard reads. So the 96-key pairs read E147's refits,
   as their share, steering and rank cells do, not E58's originals. Gap: log10(reuse mean / new mean).
2. Steering. Each form (pooled, per-key) from the files the scorecard reads (E135, E143, E147, E140), at the tuned block
   and strength; the reported cell is the wider-|gap| form (checked to equal final.json); and E149's strength-1 re-read
   (prior_steering/results/stage1.json), built pairs only. Row 18: E62's re-gating gap on E135's scored rows.
3. Spearman (ranked correlation) of the share gap with each steering reading over the nine built pairs (the figure's -0.30 is the reported one).

    ~/commute-agents/fleet/cpu-run python3 experiments/p270_rescore/rescore.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import correlation, fmean
from typing import Any

from commute.figures import METHODS, figure, save

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results/rescore.json", HERE / "figures/size_steering_per_form.png"
BUILT = ["e62_copy_lookup", "e46_copy_random", "e53_lowercase_cipher", "e54_matched", "e54_own", "open_pair_c3",
         "open_pair_c3_copy", "e58_rank1_96", "e58_copy_prior_cipher"]  # fmt: skip
TWINS = ["e58_elicit_shuffle", "e58_class_twin", "e123_case_label", "e125_repeat_label"]
FORMS = ("pooled", "perkey")
Side = tuple[list[float], list[float]]
Names = tuple[list[str], list[str]]


def load(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text())


def call(s: Side) -> str:
    a, b = s
    if min(len(a), len(b)) < 2:
        return "one value a side"
    return "apart" if min(a) > max(b) or max(a) < min(b) else "no call"


def verdict(gap: float, c: str) -> str:
    return "no call" if c != "apart" else ("agree" if gap > 0 else "reverse")  # the share's gap is > 0 on all nine


def diff(s: Side) -> float:
    return fmean(s[0]) - fmean(s[1])


def log_ratio(s: Side) -> float:
    return math.log10(fmean(s[0]) / fmean(s[1]))


def tally(verdicts: list[str]) -> dict[str, int]:
    return {k: verdicts.count(k) for k in ("agree", "reverse", "no call")}


def cells() -> dict[tuple[str, str], dict[str, Any]]:
    return {(c["pair"], c["method"]): c for c in load("experiments/method_table/results/final.json")["cells"]}


# ----------------------------------------------------------------------------- 1. size


def state_names() -> dict[str, Names]:
    """(reuse, new) state names per built pair, as size_matched/run.py pairs() builds them."""
    e136, e147 = load("experiments/share_untrained/results/inventory.json"), load("experiments/refit_96key/results/inventory.json")
    names = {p["id"]: (p["reuse"], p["new"]) for p in e136["pairs"] if p["id"] in BUILT}
    names |= {p["id"]: ([f"e58/{p['reuse']}_s{s}" for s in e147["seeds"]], [f"e58/{p['new']}_s{s}" for s in e147["seeds"]])
              for p in e147["pairs"] if p["id"] in BUILT}  # fmt: skip
    return names


def share_of(state: str, e136: dict[str, Any], e147: dict[str, Any]) -> float:
    return e147[state]["e136"]["trained"] if state.startswith("e58/") else e136[state]["trained"]


def size(final: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    m67, names = load("experiments/size_matched/results/m67.json")["pairs"], state_names()
    e136, e147 = load("experiments/share_untrained/results/stage1.json")["states"], load("experiments/refit_96key/results/stage1.json")["states"]
    out: dict[str, Any] = {"pairs": {}}
    for pair in BUILT:
        sizes, (rn, nn) = m67[pair]["sizes"], names[pair]
        new: Side = (sizes["reuse"], sizes["new"])
        assert (len(rn), len(nn)) == (len(new[0]), len(new[1])), pair
        named = dict(zip(rn + nn, new[0] + new[1], strict=True))
        for st, rec in m67[pair]["states"].items():  # the join: E187's per-state size of the larger side after x c, by name
            assert math.isclose(named[st] * m67[pair]["c"], rec["size"], rel_tol=1e-5), (pair, st, named[st] * m67[pair]["c"], rec["size"])
        sh = final[(pair, "share")]
        assert [share_of(n, e136, e147) for n in rn] == sh["reuse"] and [share_of(n, e136, e147) for n in nn] == sh["new"], pair
        old = final[(pair, "size")]
        o_side: Side = (old["reuse"], old["new"])
        g_old, g_new = log_ratio(o_side), log_ratio(new)
        assert math.isclose(g_old, old["gap"], rel_tol=1e-9) and call(o_side) == old["call"], pair
        out["pairs"][pair] = {
            "name": sh["pair_name"], "states": {"reuse": rn, "new": nn},
            "old": {"gap": g_old, "call": call(o_side), "verdict": verdict(g_old, call(o_side)), "n": [len(o_side[0]), len(o_side[1])],
                    "source": old["source"][:160], "form": old["form"] or "summed per-matrix Frobenius norms (or as source)"},
            "new": {"gap": g_new, "call": call(new), "verdict": verdict(g_new, call(new)), "n": [len(new[0]), len(new[1])],
                    "mean_reuse": fmean(new[0]), "mean_new": fmean(new[1]), "ratio_new_over_reuse": fmean(new[1]) / fmean(new[0]),
                    "reuse_range": [min(new[0]), max(new[0])], "new_range": [min(new[1]), max(new[1])]},
        }  # fmt: skip
    out["tally_old"] = tally([p["old"]["verdict"] for p in out["pairs"].values()])
    out["tally_new"] = tally([p["new"]["verdict"] for p in out["pairs"].values()])
    m = out["pairs"]["e54_matched"]["new"]
    out["e54_matched_margin_over_size"] = (m["new_range"][0] - m["reuse_range"][1]) / fmean([m["mean_reuse"], m["mean_new"]])
    return out


# ----------------------------------------------------------------------------- 2. steering


def tuned_forms() -> dict[str, dict[str, Side]]:
    """Each pair's two forms at the tuned block and strength, from the files the scorecard reads (plot_final.py)."""
    out: dict[str, dict[str, Side]] = {}
    st = load("experiments/steering_copy_lookup/results/stage1.json")["states"]
    out["e62_copy_lookup"] = {k: ([st[f"copy_s{i}"]["steering"][k]["share_lo"] for i in range(3)],
                                  [st[f"lookup_s{i}"]["steering"][k]["share_lo"] for i in range(3)]) for k in FORMS}  # fmt: skip
    h, inv = load("experiments/holes_fill/results/stage1.json"), {p["id"]: p for p in load("experiments/share_untrained/results/inventory.json")["pairs"]}
    for pair in h["score"]["families"]["steering"]:
        get = lambda n, k: h["states"][n]["steering"].get(k, {}).get("share")
        out[pair] = {k: ([get(n, k) for n in inv[pair]["reuse"]], [get(n, k) for n in inv[pair]["new"]]) for k in FORMS}
    r, i2 = load("experiments/refit_96key/results/stage1.json")["states"], load("experiments/refit_96key/results/inventory.json")
    for p in i2["pairs"]:
        a, b = ([f"e58/{p[s]}_s{i}" for i in i2["seeds"]] for s in ("reuse", "new"))
        out[p["id"]] = {k: ([r[n]["holes_fill"]["steering"][k]["share"] for n in a], [r[n]["holes_fill"]["steering"][k]["share"] for n in b]) for k in FORMS}
    t3 = load("experiments/real_vs_twin/results/stage3.json")["states"]
    for pair, w in {"e123_case_label": "case", "e125_repeat_label": "repeat"}.items():
        arm = lambda x, w=w: [f"{w}_{x}_s{i}" for i in range(3)]
        out[pair] = {k: ([t3[n]["steering"][k]["share"]["contrast"] for n in arm("real")], [t3[n]["steering"][k]["share"]["contrast"] for n in arm("twin")]) for k in FORMS}
    return out


def strength1() -> dict[str, dict[str, Side]]:
    """E149's strength-1 re-read (block chosen at strength 1), both forms, built pairs; and E149's kept form."""
    e149 = load("experiments/prior_steering/results/stage1.json")
    names = state_names()
    out: dict[str, dict[str, Side]] = {}
    for pair in BUILT:
        rn, nn = names[pair]
        out[pair] = {k: ([e149["states"][n][k]["share"] for n in rn], [e149["states"][n][k]["share"] for n in nn])
                     for k in FORMS if all(e149["states"][n].get(k, {}).get("share") is not None for n in rn + nn)}  # fmt: skip
        s1 = e149["score"]["pairs"][pair]["steering"]["strength1"]
        assert s1["reuse"] == out[pair][s1["kind"]][0] and s1["new"] == out[pair][s1["kind"]][1], pair
        out[pair]["kept"] = out[pair][s1["kind"]]
    return out


def reading(s: Side | None) -> dict[str, Any] | None:
    if s is None or None in s[0] + s[1]:
        return None
    return {"gap": diff(s), "call": call(s), "verdict": verdict(diff(s), call(s)), "reuse": s[0], "new": s[1]}


def steering(final: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    tuned, s1 = tuned_forms(), strength1()
    out: dict[str, Any] = {"pairs": {}}
    for pair in BUILT + TWINS:
        f = tuned[pair]
        whole = {k: v for k, v in f.items() if None not in v[0] + v[1]}
        kept = max(whole, key=lambda k: abs(diff(whole[k])))
        rep = final[(pair, "steering")]
        assert whole[kept] == (rep["reuse"], rep["new"]), pair  # the reported cell is the wider-gap form
        rec = {"reported": {**(reading((rep["reuse"], rep["new"])) or {}), "form": kept}, "pooled": reading(f["pooled"]),
               "perkey": reading(f["perkey"]) or reading(f["pooled"]), "perkey_is_pooled": reading(f["perkey"]) is None}  # fmt: skip
        if pair in s1:
            rec |= {f"s1_{k}": reading(s1[pair].get(k)) for k in (*FORMS, "kept")}
            rec["s1_perkey"] = rec["s1_perkey"] or rec["s1_pooled"]  # no per-key vector on lowercase vs ciphers, as tuned
        out["pairs"][pair] = rec
    cols = ["reported", "pooled", "perkey", "s1_pooled", "s1_perkey", "s1_kept"]
    out["tally_built"] = {c: tally([out["pairs"][p][c]["verdict"] for p in BUILT if out["pairs"][p].get(c)]) for c in cols}
    share = [final[(p, "share")]["gap"] for p in BUILT]
    out["spearman_share_vs"] = {c: correlation(share, [out["pairs"][p][c]["gap"] for p in BUILT], method="ranked") for c in cols}
    return out


def row18(final: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """E62's re-gating gap on all 200 windows (the scorecard, E136) against E135's 100 scored windows (paired with steering)."""
    e135 = load("experiments/steering_copy_lookup/results/stage1.json")
    s: Side = ([e135["states"][f"copy_s{i}"]["r_score"] for i in range(3)], [e135["states"][f"lookup_s{i}"]["r_score"] for i in range(3)])
    assert math.isclose(diff(s), e135["score"]["gap_share"], rel_tol=1e-9)
    sh = final[("e62_copy_lookup", "share")]
    return {"scorecard_all_rows": reading((sh["reuse"], sh["new"])), "e135_scored_rows": reading(s)}


# ----------------------------------------------------------------------------- figure


def series(ax: Any, panel: str, recs: dict[str, Any], key: str, label: str, color: str, marker: str, dy: float) -> list[dict[str, Any]]:
    """One reading's gap per built pair: filled when apart, hollow when no call; the legend entry is filled."""
    data = []
    for i, p in enumerate(BUILT):
        r = recs[p][key]
        ax.scatter([r["gap"]], [i + dy], color=color if r["call"] == "apart" else "white", edgecolor=color, s=24, marker=marker, zorder=3)
        data.append({"panel": panel, "series": key, "pair": p, "gap": r["gap"], "call": r["call"]})
    ax.scatter([], [], color=color, marker=marker, s=24, label=label)
    return data


def plot(sz: dict[str, Any], st: dict[str, Any]) -> None:
    fig, (a, b) = figure((7.5, 5.0), ncols=2)
    names = [sz["pairs"][p]["name"] for p in BUILT]
    m = METHODS["size"].marker
    data = series(a, "size", sz["pairs"], "old", "as reported (mixed norms, E58 originals at 96 keys)", "#999999", m, -0.15)
    data += series(a, "size", sz["pairs"], "new", "one norm, the states the share reads", METHODS["size"].color, m, 0.15)
    steer = [("reported", "as reported (wider-gap form, tuned)", "#555555", -0.27), ("pooled", "pooled, tuned", METHODS["steering"].color, -0.09),
             ("perkey", "per-key, tuned", "#882255", 0.09), ("s1_pooled", "pooled, strength 1", "#009E73", 0.27)]  # fmt: skip
    for key, label, color, dy in steer:
        data += series(b, "steering", st["pairs"], key, label, color, "D", dy)
    for ax in (a, b):
        ax.axvline(0, color="#999999", linewidth=0.8, zorder=0)
        ax.set_yticks(range(len(BUILT)), names if ax is a else [""] * len(BUILT))
        ax.set_ylim(len(BUILT) - 0.5, -0.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, -0.13), frameon=False)
    t = sz["tally_new"]
    a.set_title(f"Update size, one norm: reverses {t['reverse']}, agrees {t['agree']}", fontsize=8.5)
    a.set_xlabel("size gap, log10(reuse mean / new mean)")
    t = st["tally_built"]["s1_pooled"]
    b.set_title(f"Steering: pooled at strength 1 reverses {t['reverse']} of 9", fontsize=8.5)
    b.set_xlabel("steering gap, reuse minus new")
    n = ("Nine 67M built pairs, 2 to 8 seeds a side; the share is higher on the reuse side on all nine, so a point left of 0 "
         "runs against it. Filled: every seed of one side beyond every seed of the other; hollow: no call.")
    save(fig, FIG, n=n, data=data)


def main() -> None:
    final = cells()
    out = {"cell": "P2.70", "audit": "reports/reviews/2026-11-23_review_baseline-fidelity.md rows 16-18, 21",
           "size": size(final), "steering": steering(final), "row18": row18(final)}  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    plot(out["size"], out["steering"])
    print(json.dumps({"size": [out["size"]["tally_old"], out["size"]["tally_new"]], "steering": out["steering"]["tally_built"],
                      "spearman": out["steering"]["spearman_share_vs"]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
