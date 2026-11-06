"""The baseline retarget's one table (step 3): every neuron reading and cheap comparator beside every VPD row, 67M and
Qwen, built from result files only; no model, no rule added.

Inputs (each cell names its file and key):
  67M   neuron_vpd_count/results/{cells,score}.json (E117), neuron_r/results/cells.json (E98's r_half, r_n),
        regating_truth_table/results/table.json (VPD's full-width r, E46/E53)
  Qwen  qwen_comparator_table/results/table.json (E118: E93, E94, E94 Am1, E104, E107 Am1 rows with four comparators),
        neuron_vpd_table/results/table.json (the earlier verdict-level rows, ask 24 (a))
Writes results/full_table.md and results/full_table.json (the numbers plot_full.py draws).

Summaries, display only (no verdict is re-scored and none is filed):
  orders   a column orders E93's verdict when every passing state lies on one side of every failing one
  gap/sp   E92's stop-rule quantity (`qwen_comparators.separates`): gap between two arms' seed means over the larger
           same-arm seed spread; above 1 is what E92 calls "counts"
  f        where the known-facts fine-tune sits between the anchors' means, template 0 and fictional facts 1 (E94's f)

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/neuron_vpd_table/build_full.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E117C, E117S = EXP / "neuron_vpd_count/results/cells.json", EXP / "neuron_vpd_count/results/score.json"
E98, RTT = EXP / "neuron_r/results/cells.json", EXP / "regating_truth_table/results/table.json"
E118, EARLY = EXP / "qwen_comparator_table/results/table.json", HERE / "results/table.json"

TASKS67 = {"copy maps": ["ident32", "ident32b"], "random maps": ["rand32", "rand32b"],
           "lowercase": ["lowercase"], "ciphers": ["e53_cipher", "broad_cipher"]}
VPD_COLS = ["rule2_share", "direct_maths", "outside", "placement", "think"]
CMP_COLS = ["size", "gain", "rank", "steer", "carry"]
# the pair grid and the place summary leave out W0's rank: it reads each arm's rows, not the state (independent check, fix 6)
SUM_COLS = VPD_COLS[1:] + [c for c in CMP_COLS if c != "rank"]
NAMES = {"rule2_share": "VPD: E93 share left in Δ", "direct_maths": "VPD: direct maths", "outside": "VPD: outside share",
         "placement": "VPD: placement excess", "think": "VPD: think overlap", "size": "update size ‖δW‖/‖W0‖",
         "gain": "gain in log P", "rank": "W0 rank of first answer token", "steer": "steering share",
         "carry": "carrying-neuron share"}
PAIRS = {"template vs fictional facts, matched": (["a1_s0", "a1_s1"], ["a2_s42_matched", "a2_s142_matched"]),
         "template vs fictional facts, full": (["a1_s0", "a1_s1"], ["a2_s42_full", "a2_s142_full"]),
         "RL vs shuffled-reward null, step 10": (["R_s0_10", "R_s70_10"], ["N_s0_10", "N_s70_10"]),
         "RL vs shuffled-reward null, step 100": (["R_s0_100", "R_s70_100"], ["N_s0_100", "N_s70_100"]),
         "known vs wrong answers, matched": (["known_s42_matched", "known_s142_matched"], ["wrong_s42_matched", "wrong_s142_matched"])}
KNOWN = {"matched": (["known_s42_matched", "known_s142_matched"], ["a2_s42_matched", "a2_s142_matched"]),
         "end": (["known_s42_end", "known_s142_end"], ["a2_s42_full", "a2_s142_full"])}


def js(p: Path) -> Any:
    return json.loads(p.read_text())


def num(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def fmt(x: Any, d: int = 3) -> str:
    if x == math.inf:
        return "∞ (rows differ)"
    v = num(x)
    return "–" if v is None else (f"{v:.2e}" if v != 0 and abs(v) < 1e-3 else f"{v:.{d}g}")


# ----------------------------------------------------------------------------- 67M


def m67() -> dict[str, Any]:
    cells, score, e98, rtt = js(E117C), js(E117S)["verdicts"], js(E98)["rows"], js(RTT)["runs"]
    states = []
    for fam, tasks in TASKS67.items():
        for t in tasks:
            for s in (0, 1):
                st, r = f"{t}_s{s}", cells["rows"][f"{t}_s{s}"]
                ref = rtt.get(f"e46/{st}") or rtt.get(f"e53/{st.removeprefix('e53_')}")  # E53 names its cipher "cipher"
                states.append({"state": st, "task": t, "class": fam, "K_VPD": r["K_VPD"],
                               "A_vpd": num(cells["jaccard"][t]["vpd"]), "A_neurons": num(cells["jaccard"][t]["neurons"]),
                               **{f"{p}_{m}_{k}": num(r[p][m][k]) if k == "all" or r[p][m]["readable"] else None
                                  for p in "BC" for m in ("vpd", "neurons") for k in ("set", "all")},
                               **{f"{p}_{m}_rand": num((r[p][m]["random"] or {}).get("mean")) for p in "BC" for m in ("vpd", "neurons")},
                               "r_vpd": num(ref["r"]["value"]) if ref else None,
                               "r_n": num(e98[st]["r_n"]), "r_half": num(e98[st]["r_half"])})
    return {"states": states, "verdicts": {k: v["verdict"] for k, v in score.items()},
            "gaps": {k: {"gap": v.get("gap"), "within": v.get("within")} for k, v in score.items() if "gap" in v},
            "full_width": full_width(states)}


def sep(ex: list[float], nw: list[float]) -> dict[str, Any]:
    """E12's separation rule as E117 applies it (score.json as_applied), existing class higher; display only."""
    gap, within = min(ex) - max(nw), max(max(ex) - min(ex), max(nw) - min(nw))
    rev = min(nw) - max(ex) > within
    return {"gap": gap, "within": within, "reads": "separates" if gap > within else ("reversed" if rev else "does not separate")}


def full_width(states: list[dict[str, Any]]) -> dict[str, Any]:
    """The full-width readings on both families, under E12's rule and E53's 0.3 bar: description, not filed in E117."""
    out = {}
    for col in ("r_vpd", "r_n", "r_half", "B_vpd_all", "B_neurons_all"):
        for ex, nw in (("copy maps", "random maps"), ("lowercase", "ciphers")):
            e = [s[col] for s in states if s["class"] == ex]
            n = [s[col] for s in states if s["class"] == nw]
            out[f"{col}/{ex} vs {nw}"] = sep(e, n) | {"e53_bar_gap": min(e) - max(n)}
    return out


# ----------------------------------------------------------------------------- Qwen


def comparators(row: dict[str, Any]) -> dict[str, float | None]:
    v = lambda k, f: num((row.get(k) or {}).get("value", {}).get(f)) if isinstance((row.get(k) or {}).get("value"), dict) else None
    return {"size": v("size", "ratio_norm"), "gain": v("rank", "gain_all"), "rank": v("rank", "W0_median_rank_first"),
            "steer": v("steering", "share"), "carry": v("carrying", "neuron_share"), "carry_gain": v("carrying", "gain")}


def per_state(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        d = out.setdefault(r["state"], comparators(r))
        val = r["vpd"]["value"] if isinstance(r["vpd"].get("value"), dict) else {}
        for c in VPD_COLS:
            if d.get(c) is None:
                d[c] = num(val.get(c))
        if "arm_verdict" in val:
            d["e93"] = val["arm_verdict"]  # type: ignore[assignment]
    return out


def orders(S: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    e93 = [r["state"] for r in rows if r["source"] == "E93"]
    out = {}
    for c in ["rule2_share", *CMP_COLS]:
        p = [S[s][c] for s in e93 if S[s].get("e93") == "passes" and S[s][c] is not None]
        f = [S[s][c] for s in e93 if S[s].get("e93") == "fails" and S[s][c] is not None]
        up, down = min(p) - max(f), min(f) - max(p)
        out[c] = {"pass": [min(p), max(p)], "fail": [min(f), max(f)],
                  "reads": "orders, passes higher" if up > 0 else ("orders, passes lower" if down > 0 else "does not order"),
                  "margin": max(up, down)}
    return out


def gap_spread(a: list[float | None], b: list[float | None]) -> float | None:
    if any(x is None for x in a + b):
        return None
    x, y = [float(v) for v in a if v is not None], [float(v) for v in b if v is not None]
    spread = max(abs(x[0] - x[1]), abs(y[0] - y[1]))
    gap = abs(sum(x) - sum(y)) / 2
    return math.inf if spread == 0 and gap > 0 else (gap / spread if spread else None)


def known_f(S: dict[str, dict[str, Any]], known: list[str], a2: list[str], c: str) -> float | None:
    """None when a value is missing, when the anchors do not separate on this column (gap over spread ≤ 1), as E94's
    "no call: the anchors do not separate", or when the known arm's two seeds differ by more than the anchors' gap, so
    its mean is no place (independent check, fix 7: known_s42_end's carrying share 2.15 against 0.13)."""
    vals = [S[s][c] for s in ["a1_s0", "a1_s1", *a2, *known]]
    g = gap_spread([S[s][c] for s in ("a1_s0", "a1_s1")], [S[s][c] for s in a2])
    if any(v is None for v in vals) or g is None or g <= 1:
        return None
    m = lambda xs: sum(S[s][c] for s in xs) / len(xs)
    lo, hi = m(["a1_s0", "a1_s1"]), m(a2)
    if abs(S[known[0]][c] - S[known[1]][c]) > abs(hi - lo):
        return None
    return (m(known) - lo) / (hi - lo) if hi != lo else None


def qwen() -> dict[str, Any]:
    rows = js(E118)["rows"]
    S = per_state(rows)
    cols = SUM_COLS
    return {"states": S, "e93_orders": orders(S, rows),
            "pairs": {p: {c: gap_spread([S[s][c] for s in a], [S[s][c] for s in b]) for c in cols} for p, (a, b) in PAIRS.items()},
            "known_f": {k: {c: known_f(S, kn, a2, c) for c in cols} for k, (kn, a2) in KNOWN.items()}}


# ----------------------------------------------------------------------------- markdown


def md_67m(M: dict[str, Any]) -> list[str]:
    src = ("E117 `neuron_vpd_count/results/cells.json` .rows.<state>.{B,C}.{vpd,neurons}.{set,all,random.mean}, "
           ".jaccard.<task>; E98 `neuron_r/results/cells.json` .rows.<state>.{r_n,r_half}; VPD r "
           "`regating_truth_table/results/table.json` .runs[\"e46|e53/<state>\"].r.value")
    L = ["## 1. 67M: VPD's switched-on set and re-switching beside the same readings on neurons", "", f"Sources: {src}.", "",
         ("| state | class | K_VPD | A seed agreement, VPD / neurons | B restore share of the set, VPD / neurons (random set) "
         "| B whole width, VPD / neurons | C re-switching of the set, VPD / neurons (random set) | C whole width: VPD r / neurons r_n / r_half |"),
         "|---|---|---|---|---|---|---|---|"]
    for s in M["states"]:
        L.append(f"| {s['state']} | {s['class']} | {s['K_VPD']} | {fmt(s['A_vpd'])} / {fmt(s['A_neurons'])} "
                 f"| {fmt(s['B_vpd_set'])} / {fmt(s['B_neurons_set'])} ({fmt(s['B_vpd_rand'])} / {fmt(s['B_neurons_rand'])}) "
                 f"| {fmt(s['B_vpd_all'])} / {fmt(s['B_neurons_all'])} "
                 f"| {fmt(s['C_vpd_set'])} / {fmt(s['C_neurons_set'])} ({fmt(s['C_vpd_rand'])} / {fmt(s['C_neurons_rand'])}) "
                 f"| {fmt(s['r_vpd'])} / {fmt(s['r_n'])} / {fmt(s['r_half'])} |")
    L += ["", "K_VPD = 0 makes the set readings unreadable (–). Seed agreement is one value per task, shown on both seeds.", "",
          "**E117's filed separation rule** (E12's, `neuron_vpd_count/results/score.json` .verdicts):", "",
          "| panel | VPD | neurons at VPD's count |", "|---|---|---|"]
    V = M["verdicts"]
    for fam in ("copy_vs_random", "lowercase_vs_cipher", "lowercase_vs_broad_cipher", "e53_bar"):
        for p in "ABC":
            if f"{fam}/{p}/vpd" in V:
                g = {m: M["gaps"].get(f"{fam}/{p}/{m}") for m in ("vpd", "neurons")}
                t = {m: V[f"{fam}/{p}/{m}"] + (f" (gap {g[m]['gap']:.3f}, within {g[m]['within']:.3f})" if g[m] else "") for m in g}
                L.append(f"| {fam} {p} | {t['vpd']} | {t['neurons']} |")
    L += ["", ("**Whole-width readings under the same rule**, description only (E117 filed no verdict on them; the C limits are "
          "its controls and reproduce E46/E53's r and E98's r_n). gap = min(existing class) − max(new class), so a negative gap "
          "is the new class higher; \"reversed\" when min(new) − max(existing) exceeds the within-class range:"), "", "| reading | comparison | gap | within | reads | E53 bar (≥ 0.3) |",
          "|---|---|---|---|---|---|"]
    for k, v in M["full_width"].items():
        col, comp = k.split("/")
        L.append(f"| {col} | {comp} | {v['gap']:.3f} | {v['within']:.3f} | {v['reads']} | {'met' if v['e53_bar_gap'] >= 0.3 else 'not met'} ({v['e53_bar_gap']:.3f}) |")
    return L


def md_qwen(Q: dict[str, Any]) -> list[str]:
    rows = js(E118)["rows"]
    L = ["", "## 2. Qwen3-1.7B: every VPD row beside the four comparators", "",
         ("One row per VPD reading and state, copied from E118 `qwen_comparator_table/results/table.json` .rows[]; each "
         "cell's file and key is that row's `<column>.source`, listed in the last column. n/a: a constructed control, "
         "no trained state. Carrying share: neuron share nums[N]/G of E102 (see the analysis report: it is not bounded by 1)."), "",
         "| VPD source | state | VPD reading | size | gain / W0 rank | steering (random) | carrying share / gain | sources |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = comparators(r)
        v = r["vpd"]["value"]
        vt = ", ".join(f"{k} {fmt(x) if isinstance(x, float) else x}" for k, x in v.items() if k != "rules") if isinstance(v, dict) else str(v)
        st = (r.get("steering") or {}).get("value") or {}
        steer = f"{fmt(c['steer'])} ({fmt(st.get('random_share'))})" if isinstance(st, dict) and c["steer"] is not None else "n/a"
        srcs = "; ".join(f"{k}: `{(r.get(k) or {}).get('source', '–')}`" for k in ("vpd", "size", "rank", "steering", "carrying"))
        L.append(f"| {r['source']} | {r['state']} | {vt} | {fmt(c['size'])} | {fmt(c['gain'])} / {fmt(c['rank'])} | {steer} "
                 f"| {fmt(c['carry'])} / {fmt(c['carry_gain'])} | {srcs} |")
    L += ["", "**Does a column order E93's pass/fail?** (passes: a1, a2, u1, u2; fails: RL, null, own-samples at step 100)", "",
          "| column | passes | fails | reads | margin |", "|---|---|---|---|---|"]
    for c, o in Q["e93_orders"].items():
        L.append(f"| {NAMES[c]} | {fmt(o['pass'][0])} to {fmt(o['pass'][1])} | {fmt(o['fail'][0])} to {fmt(o['fail'][1])} | {o['reads']} | {fmt(o['margin'])} |")
    cols = SUM_COLS
    L += ["", ("**Gap over seed spread per pair** (E92's stop-rule quantity; > 1 is E92's \"counts\"; – not read on that pair). "
              "On the anchor pairs and the known-against-wrong pair each arm is read on its own rows, so the gain, steering and "
              "carrying cells there are across rows; RL against null shares the gold rows. W0's rank is left out: it reads "
              "only the rows."), "",
          "| pair | " + " | ".join(NAMES[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for p, d in Q["pairs"].items():
        L.append(f"| {p} | " + " | ".join(fmt(d[c], 2) for c in cols) + " |")
    L += ["", ("**Where the known-facts fine-tune sits between the anchors** (f: template 0, fictional facts 1; – where the "
              "anchors do not separate on that column, or the known arm's two seeds differ by more than the anchors' gap. "
              "The comparators read each arm on its own rows: the known arm on its sentences, its carrying share on E103's probe)"), "",
          "| size | " + " | ".join(NAMES[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for k, d in Q["known_f"].items():
        L.append(f"| {k} | " + " | ".join(fmt(d[c], 2) for c in cols) + " |")
    return L


def md_early() -> list[str]:
    L = ["", "## 3. Qwen: the earlier verdict-level rows (ask 24 (a), not covered by section 2)", "",
         ("Copied from `neuron_vpd_table/results/table.json` .rows[] (built by `build.py`; report "
         "`reports/analyses/2026-11-06_analysis_neurons-vs-vpd-table.md`)."), "",
         "| question | size | VPD | neurons | sources |", "|---|---|---|---|---|"]
    for r in js(EARLY)["rows"]:
        v, n = r["vpd"], r["neurons"]
        L.append(f"| {r['question']} | {r['size']} | {v['class']}: {v['verdict']} | {n['class']}: {n['verdict']} "
                 f"| VPD `{v.get('source')}` {v.get('key', '')}; neurons `{n.get('source')}` {n.get('key', '')} |")
    return L


def main() -> None:
    M, Q = m67(), qwen()
    (HERE / "results/full_table.json").write_text(json.dumps({"m67": M, "qwen": Q}, indent=1, default=str) + "\n")
    head = ["# Neurons and cheap comparators beside VPD: the full table", "",
            ("Built by `experiments/neuron_vpd_table/build_full.py` from result files only. No verdict is re-scored; "
            "summaries marked description apply an existing rule to a new pair and file nothing. Report: "
            "`reports/analyses/2026-11-06_analysis_e117-e118-neurons-beside-vpd.md`."), ""]
    (HERE / "results/full_table.md").write_text("\n".join(head + md_67m(M) + md_qwen(Q) + md_early()) + "\n")
    print(json.dumps({"e93_orders": {c: o["reads"] for c, o in Q["e93_orders"].items()}, "pairs": Q["pairs"],
                      "known_f": Q["known_f"], "full_width": {k: v["reads"] for k, v in M["full_width"].items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
