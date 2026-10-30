"""E57 (qwen_reading): the scoring of plan section 5 on CPU, from what Stages A-D wrote, into results/score.json.
Imported by run.py (`--stage score`, and at the end of Stage D) as a sibling module. Rules: commute.qwen_reading
(anchor_row, prompt_row, think_row, t_read, kept, edges, overlap_gap_boot); filed in results/predictions.json (with
amendment_2026-10-30) and, for the re-gating share, behaviour kept and placement, in E45's predictions.json
rules.decomposition_rows at E45 Amendment 2's decisive point (score.json amend2_decisive, read by `qr.d_pick`).

Anchor 1 is E45's a1_s0/a1_s1 (Stage D), anchor 2 a2long_s0/a2long_s1 at step 100 (Stage C). E45's rows are read on
each state's own rows (own samples, the null, E40: the maths rows). Plan section 11: with neither anchor-1 seed
matched, no anchor prediction is scored and only the readings are written. The scoring refuses to run until every C
state and every D run with a picked step is read at that step; a run with no step (an anchor-1 seed not matched, a run
E45 never scored) is missing: every row a miss.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from lab import CFG, E45, LARGE, STATES, Lab
from safetensors.torch import load_file
from torch import Tensor

from commute import qwen_reading as qr
from commute.qwen_vpd import SITES

ANCHORS = (("a1_s0", "a1_s1"), ("a2long_s0", "a2long_s1"))
OTHERS = ("gold_s0", "null_s0", "e40")  # with the own-samples run: read on a validated row with E45's t rule
ROWS = {"regating_share": "regating", "behaviour_kept": "kept_A", "placement": "placement", "prompt_overlap": "e_P"}
DOMAIN = "anchor 1 shares its domain (GSM8K maths), anchor 2 does not (invented facts)"
A1_LIMITS = ("anchor 1 is a format fine-tune: it fixes the novelty of a surface format the model already follows on "
             "request, not of a better way to solve problems; the format prompt lowers W0's GSM8K accuracy on the 512 "
             "search problems from 0.834 to 0.680 and shortens answers from 190 to 130 tokens")


def anchor_of(n: str) -> int | None:
    return 1 if n in ANCHORS[0] else 2 if n in ANCHORS[1] else None


def value(status: str, x: float) -> float | None:
    """A row value by its status: the value when scored, None when unscored, NaN (a miss) otherwise."""
    return x if status == "scored" else None if status == "unscored" else float("nan")


def reading(rec: dict[str, Any], ov: dict[str, Any] | None, n: str, A: dict[str, Any],
            ref: dict[str, int]) -> dict[str, Any]:
    """One state's numbers for every row; values None (unscored) or NaN (missing) as the filed rules read them. No
    record (a run D did not read: its pick has no step) is missing: every row a miss."""
    if not rec:
        return {"missing": True, "values": dict.fromkeys(ROWS.values(), float("nan")),
                "prompt": "misses" if anchor_of(n) else None, "think": "misses" if anchor_of(n) == 2 else None}
    r = rec.get("own") or rec["maths"]  # E45's rows: on the own rows
    k = qr.kept(r["means"])
    P, T = (ov or {}).get("P"), (ov or {}).get("T")
    floor, prompt = A["min_size"], qr.prompt_row(anchor_of(n) or 1, P, A["min_size"], A["S_prompt"])
    think = qr.think_row(T, floor, [ref.get(x, 0) for x in "TLC"])
    return {"size_S": rec["S"], "ov": {X: v for X, v in (ov or {}).items() if X in "PTLC"},
            "values": {"regating": value(r["status"], r["regating"]), "kept_A": value(k["status"], k["A"]),
                       "placement": rec.get("placement", {}).get("excess"),
                       "e_P": None if prompt == "unscored" else P["e"]},  # type: ignore[index]
            "regating_status": r["status"], "kept": k, "direct": r["direct"],
            "e_T": None if think == "unscored" else T["e"],  # type: ignore[index]
            "prompt": prompt if anchor_of(n) else None, "think": think if anchor_of(n) == 2 else None}


def prompt_se(lab: Lab) -> dict[str, Any]:
    """The prompt row's gap s.e.: the row bootstrap (CFG.boot resamples of the 128 maths rows, shared by the four
    anchor states and W0's g0) of mean e_P(anchor 1) - mean e_P(anchor 2)."""
    t0 = time.time()
    st = lambda name: load_file(str(LARGE / f"{name}.safetensors"))
    on = lambda t: qr.gated(t["active"], int(t["n_pos"]))
    X: Tensor = on(st("A/prompt")) & ~on(st("A/plain"))
    g1 = [st(f"C/{n}_g1")["rows"] for n in (*ANCHORS[0], *ANCHORS[1])]
    gaps = qr.overlap_gap_boot(g1, st("A/W0_maths")["rows"], X, st("A/layout")["site_of"], len(SITES),
                               qr.resample(CFG.n_rows, CFG.boot), CFG.row_len, CFG.draws)
    out = {"se": qr.sd(gaps), "finite": sum(qr.ok(g) for g in gaps), "of": len(gaps), "s": time.time() - t0}
    lab.log(f"[score] prompt-row bootstrap: {json.dumps(out)}")
    return out


def anchor_rows(lab: Lab, R: dict[str, dict[str, Any]], others: tuple[str, ...]) -> dict[str, Any]:
    """The filed rows: E45's three (validation as E45 files it: se 0, the floor in each share's status) and E57's
    prompt row (2 x the bootstrap s.e.); think_overlap on anchor 2's seeds only (amendment 2026-10-30)."""
    vals = lambda key, i: [R[n]["values"][key] for n in ANCHORS[i]]
    boot = prompt_se(lab) if all(v is not None for i in (0, 1) for v in vals("e_P", i)) else {"se": None}
    out = {"regating_share": qr.anchor_row("regating_share", vals("regating", 0), vals("regating", 1), 0.0, 0.5),
           "behaviour_kept": qr.anchor_row("behaviour_kept", vals("kept_A", 0), vals("kept_A", 1), 0.0, 0.8),
           "placement": qr.anchor_row("placement", vals("placement", 0), vals("placement", 1), 0.0),
           "prompt_overlap": qr.anchor_row("prompt_overlap", vals("e_P", 0), vals("e_P", 1), boot["se"])
           | {"per_seed": {n: R[n].get("prompt") for i in (0, 1) for n in ANCHORS[i]}, "bootstrap": boot}}
    for name, row in out.items():
        if row["validated"]:
            m1, m2 = (sum(row[f"anchor{i}"]) / 2 for i in (1, 2))
            row["t"] = {n: qr.t_read(R[n]["values"][ROWS[name]], m1, m2) for n in others}
    out["think_overlap"] = {"anchor2_per_seed": {n: R[n].get("think") for n in ANCHORS[1]},
                            "anchor1_e_T_unfiled": {n: R[n].get("e_T") for n in ANCHORS[0]}}
    return out


def score(lab: Lab, own: str) -> None:
    A, summ = lab.get("A")["sets"], lab.get("summary")
    points = json.loads((E45 / "score.json").read_text())["amend2_decisive"]
    picks = {r: qr.d_pick(r, points.get(r)) for r in (*ANCHORS[0], own, "null_s0")}
    recs = {n: lab.get(f"C_{n}") for n in STATES} | {r: lab.get(f"D_{r}") for r in picks}
    if wait := qr.inputs_missing({n: recs[n] for n in STATES}, {r: recs[r] for r in picks}, picks):
        sys.exit("refusing to score: " + "; ".join(wait))
    ref = summ.get("ref_sizes", {})
    R = {n: reading(recs[n], summ.get("overlap", {}).get(n), n, A, ref) for n in recs}
    scored = any(picks[r]["matched"] for r in ANCHORS[0])
    rows = anchor_rows(lab, R, (*OTHERS[:2], own, OTHERS[2])) if scored else {}
    lab.put("score", {
        "anchor_predictions_scored": scored,
        "why": "scored" if scored else "anchor 1 not learned or not matched under E45 Amendment 2 (plan section 11)",
        "picks": picks, "readings": R, "rows": rows,
        "announce": [r["announce"] for r in rows.values() if "announce" in r],
        "edges": qr.edges(R, ANCHORS, A["min_size"], A["S_prompt"], ref, lab.get("B2").get("dropped")),
        "domain_confound": DOMAIN, "anchor1_limits": A1_LIMITS})
