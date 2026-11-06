"""E107 Amendment 1 (label_vs_style): overlap of VPD's switched-on sets, the wrong-answer arm against the known-facts arm,
and the filed rule (CPU, no model). Plan: reports/plans/2026-11-06_plan_e107-amend1-vpd-sets.md; rules:
results/predictions_amend1.json; the GPU reads: amend1_vpd.py.

  ~/commute-agents/fleet/cpu-run python experiments/label_vs_style/amend1_sets.py           # results/amend1_sets.json
  ~/commute-agents/fleet/cpu-run python experiments/label_vs_style/amend1_sets.py --edges   # the plan's edge cases

S = subcomponents on with the update and off at W0 on E94's 128 maths rows, as qwen_d3_rows/set_overlap.py builds it
(`qr.gated` at its N_POS). The rule (`side`) is E107's `wrong_rule` with Jaccard in place of the neuron overlap.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from commute import qwen_reading as qr
from commute.jobs import hard_exit, load_experiment

so = load_experiment("qwen_d3_rows", "set_overlap")
HERE = Path(__file__).resolve().parent
PRED, OUT = HERE / "results" / "predictions_amend1.json", HERE / "results" / "amend1_sets.json"
KNOWN, WRONG, FICT = ("known_s42_matched", "known_s142_matched"), ("wrong_s42_matched", "wrong_s142_matched"), ("a2_s42_matched", "a2_s142_matched")
NAMES = (*KNOWN, *WRONG, *FICT)


def jaccard(inter: int, na: int, nb: int) -> float | None:
    """|A & B| / |A | B|; None when both sets are empty (W0)."""
    return inter / (na + nb - inter) if na + nb - inter > 0 else None


def side(cross: list[float | None], w_wrong: float | None, w_known: float, x: float, sizes: dict[str, int]) -> dict[str, Any]:
    """E107's wrong_rule on Jaccard. band = 1 - min(w_wrong, w_known); L = w_known (the known arm's seed null, same rows
    and label); x = the highest known-against-fictional Jaccard. 'familiarity' iff max(cross) <= x + band and
    <= L - band; 'topic or style' iff min(cross) >= L - band and min(cross) - x > band; else 'no call'. Before either:
    'no call: size cap' when some wrong-against-known pair's min/max set size is at or under x + 1 - w_known (the topic
    line at the known arm's band), since Jaccard could not reach that line; 'unread' with any value missing or not
    finite, or not four cross values."""
    out: dict[str, Any] = {"cross": cross, "w_wrong": w_wrong, "w_known": w_known, "x": x}
    if len(cross) != 4 or any(v is None or not math.isfinite(v) for v in (*cross, w_wrong, w_known, x)):
        return out | {"verdict": "unread"}
    c = [float(v) for v in cross if v is not None]
    band, L = 1 - min(float(w_wrong or 0), w_known), w_known
    caps = [min(sizes[a], sizes[b]) / max(sizes[a], sizes[b]) for a in WRONG for b in KNOWN]
    out |= {"band": band, "familiarity_line": min(x + band, L - band), "topic_line": max(L - band, x + band),
            "size_caps": caps, "cap_line": x + 1 - w_known}
    if min(caps) <= x + 1 - w_known:
        return out | {"verdict": "no call: size cap"}
    if max(c) <= x + band and max(c) <= L - band:
        return out | {"verdict": "familiarity"}
    if min(c) >= L - band and min(c) - x > band:
        return out | {"verdict": "topic or style"}
    return out | {"verdict": "no call"}


def sets() -> tuple[dict[str, int], dict[str, Any]]:
    t = {n: load_file(str(so.RES / "large" / n / "counts_pe.safetensors")) for n in NAMES}
    g0 = t[NAMES[0]]["g0_active"]
    assert all(torch.equal(d["g0_active"], g0) for d in t.values()), "g0 differs between states"
    on0 = qr.gated(g0, so.N_POS)
    S = {n: qr.gated(d["g1_active"], so.N_POS) & ~on0 for n, d in t.items()}
    sizes = {n: int(s.sum()) for n, s in S.items()}
    for n in NAMES:  # the N_POS assumption, against each run's own count
        want = json.loads((so.RES / n / "state.json").read_text())["overlap"]["size_S"]
        assert want is None or want == sizes[n], (n, want, sizes[n])
    pairs = {}
    for a, b in combinations(NAMES, 2):
        i = int((S[a] & S[b]).sum())
        pairs[f"{a}|{b}"] = {"inter": i, "jaccard": jaccard(i, sizes[a], sizes[b]),
                             "over_min": i / min(sizes[a], sizes[b]) if min(sizes[a], sizes[b]) else None}
    return sizes, pairs


def score() -> dict[str, Any]:
    P = json.loads(PRED.read_text())["rule"]
    sizes, pairs = sets()
    J = lambda a, b: pairs.get(f"{a}|{b}", pairs.get(f"{b}|{a}", {})).get("jaccard")
    assert abs(J(*KNOWN) - P["w_known"]) < 1e-12 and abs(max(J(k, f) for k in KNOWN for f in FICT) - P["x"]) < 1e-12, "stored references moved"
    verdict = side([J(w, k) for w in WRONG for k in KNOWN], J(*WRONG), P["w_known"], P["x"], sizes)
    out = {"sizes": sizes, "pairs": pairs, "rule": verdict}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"sizes": sizes, "verdict": verdict["verdict"], "cross": verdict["cross"], "w_wrong": verdict["w_wrong"]}), flush=True)
    return out


def edges() -> bool:
    """The plan's edge cases through `side`, at the filed w_known and x, with the known arm's sizes."""
    P = json.loads(PRED.read_text())["rule"]
    wk, x = P["w_known"], P["x"]
    band = 1 - wk
    eq = {n: 1500 for n in (*KNOWN, *WRONG)}
    cases = [
        ("WRONG as close as known's own seeds", [wk] * 4, wk, eq, "topic or style"),
        ("WRONG as far as fictional", [x] * 4, wk, eq, "familiarity"),
        ("WRONG equidistant (midpoint)", [(x + wk) / 2] * 4, wk, eq, "no call"),
        ("Tie on the familiarity line (inclusive)", [wk - band] * 4, wk, eq, "familiarity"),
        ("Tie on the topic line (strict)", [x + band] * 4, wk, eq, "no call"),
        ("Pairs straddle the lines", [x, x, wk, wk], wk, eq, "no call"),
        ("WRONG seeds disagree (w_wrong 0.5)", [wk] * 4, 0.5, eq, "no call"),
        ("WRONG seeds disagree, cross low", [0.2] * 4, 0.5, eq, "familiarity"),
        ("WRONG's set 30% larger", [wk] * 4, wk, eq | {WRONG[0]: 1950}, "no call: size cap"),
        ("W0 as the state (empty sets, Jaccard undefined)", [None] * 4, None, eq, "unread"),
        ("Missing pair", [wk] * 3, wk, eq, "unread")]
    ok = True
    for name, cross, ww, sz, want in cases:
        got = side(cross, ww, wk, x, sz)["verdict"]
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'}  {name}: {got} (named {want})", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E107 Amendment 1's set overlap and rule")
    ap.add_argument("--edges", action="store_true")
    if ap.parse_args().edges:
        hard_exit(0 if edges() else 1)
    score()
    hard_exit(0)
