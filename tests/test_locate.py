"""E96's filed rules against their edge cases: a tie at the top, a share past 1 or negative, a seed that disagrees, a gain
under the floor, an empty nominee list, the do-nothing state. Pure; no model."""

from __future__ import annotations

import json
import math
from typing import Any

from commute import locate as L
from commute.open_pair import jsonable

TH = {"g_tol": 0.0, "min_gain": 1.0, "located_p": 0.1, "control_max": 0.2, "capability_base": 0.8247, "shared_frac": 0.1, "gain_only_frac": 0.5, "salience_top": 3}
NAN = math.nan


def test_tie_at_the_top_names_no_part():
    assert L.top_head({"a": 0.5, "b": 0.5, "c": 0.1}) is None
    assert L.top_head({"a": 0.5, "b": 0.4}) == "a"
    assert L.top_head({"a": NAN}) is None
    assert L.top_head({}) is None
    assert L.top_head({"a": 1.3, "b": 0.5}) == "a"  # a share past 1 is still the top


def test_rank_counts_a_tie_against_the_name():
    assert L.rank_of("a", {"a": 0.0, "b": 0.0, "c": 0.0}) == 3
    assert L.rank_of("a", {"a": 0.9, "b": 0.1, "c": NAN}) == 1
    assert L.rank_of(None, {"a": 0.9}) is None
    assert L.rank_of("a", {"a": NAN}) is None
    assert L.rank_of("z", {"a": 1.0}) is None


def test_the_do_nothing_state_cannot_pass():
    p0 = L.prob(-8.05)  # W0's copy log-odds: already under the bar, so the bar alone would pass it
    assert p0 <= TH["located_p"]
    assert not L.removes({"G": 0.0, "p": p0, "p_off": L.prob(-9.2)}, TH)
    assert not L.located([{"G": 0.0, "p": p0, "p_off": p0}] * 3, [{"G": 15.0, "share": 0.0}] * 3, TH)["located"]


def test_removes_needs_a_behaviour_and_a_gain_over_the_floor():
    assert L.removes({"G": 12.6, "p": 0.99, "p_off": 0.024}, TH)
    assert L.removes({"G": 12.6, "p": 0.99, "p_off": 0.1}, TH)  # a tie counts
    assert not L.removes({"G": 12.6, "p": 0.99, "p_off": 0.100001}, TH)
    assert not L.removes({"G": 0.5, "p": 0.99, "p_off": 0.0}, TH)  # under the floor
    assert not L.removes({"G": 5.0, "p": 0.05, "p_off": 0.01}, TH)  # gained, but never over the bar: nothing to take away
    assert not L.removes({"G": 12.6, "p": 0.99, "p_off": NAN}, TH)
    assert not L.removes({"G": NAN, "p": 0.99, "p_off": 0.01}, TH)


def test_spares_on_negative_past_one_floor_and_nan():
    assert L.spares({"G": 15.0, "share": 0.2}, TH)  # a tie counts
    assert L.spares({"G": 15.0, "share": -0.5}, TH)  # the part off raised the gain
    assert not L.spares({"G": 15.0, "share": 0.2001}, TH)
    assert not L.spares({"G": 15.0, "share": 1.4}, TH)
    assert not L.spares({"G": 0.5, "share": 0.0}, TH)
    assert not L.spares({"G": 15.0, "share": NAN}, TH)


def test_located_a_seed_that_disagrees_or_nothing_to_read():
    good, bad = {"G": 12.0, "p": 0.99, "p_off": 0.05}, {"G": 12.0, "p": 0.99, "p_off": 0.5}
    ctl = {"G": 15.0, "share": 0.0}
    assert L.located([good, good, good], [ctl] * 3, TH)["located"]
    assert not L.located([good, bad, good], [ctl] * 3, TH)["located"]
    assert not L.located([good] * 3, [ctl, {"G": 15.0, "share": 0.5}, ctl], TH)["located"]
    assert not L.located([], [ctl], TH)["located"]
    assert not L.located([good], [], TH)["located"]


def test_capability_reads_both_sides_and_the_bars():
    base = TH["capability_base"]
    assert L.capability(0.0095, TH) == L.capability(0.013, TH) == L.capability(0.1 * base, TH) == "shared"
    assert L.capability(0.71, TH) == L.capability(0.93, TH) == L.capability(0.5 * base, TH) == "gain only"
    assert L.capability(0.2, TH) == "between"
    assert L.capability(NAN, TH) == "unreadable"


def test_control_every_failure_and_the_empty_lists():
    rec = {"located": True}
    assert L.control(["a", "a", "a"], "a", rec, [False, False, False])["passes"]
    assert not L.control(["a", "b", "a"], "a", rec, [False] * 3)["passes"]  # a seed disagrees
    assert not L.control(["a", None, "a"], "a", rec, [False] * 3)["passes"]  # a tie
    assert not L.control(["a"] * 3, "a", {"located": False}, [False] * 3)["passes"]
    assert not L.control(["a"] * 3, "a", None, [False] * 3)["passes"]
    assert not L.control(["a"] * 3, "a", rec, [True] * 3)["passes"]  # it removes the lookup too
    assert L.control(["a"] * 3, "a", rec, [True, False, True])["passes"]  # filed: not on every seed
    assert not L.control([], "a", rec, [False] * 3)["passes"]
    assert not L.control(["a"] * 3, "a", rec, [])["passes"]


def test_apparatus():
    ok: dict[str, dict[str, float]] = {"s": {"G": 12.0, "G_run": 12.0}}
    z = {"copy": {"zero_G_max_abs": 0.0, "zero_salience_max_abs": 0.0, "salience_repeat_floor": 0.0}}
    assert L.apparatus(ok, z, TH) == "pass"
    assert L.apparatus({"s": {"G": 12.0 + 1e-9, "G_run": 12.0}}, z, TH) != "pass"
    assert L.apparatus({"s": {"G": NAN, "G_run": 12.0}}, z, TH) != "pass"
    assert L.apparatus(ok, {"copy": {**z["copy"], "zero_G_max_abs": 1e-12}}, TH) != "pass"
    assert L.apparatus(ok, {"copy": {**z["copy"], "zero_salience_max_abs": 1e-9}}, TH) != "pass"
    assert L.apparatus(ok, {"copy": {**z["copy"], "zero_salience_max_abs": 1e-9, "salience_repeat_floor": 1e-9}}, TH) == "pass"  # a tie
    assert L.apparatus({}, z, TH) != "pass"


def records(copy_p_off: float = 0.03, look_share: float = 0.0, sal_rank_ok: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """A saved nominators and sweep of two heads: the first carries the copy gain."""
    def state(share_a: float, p_a: float, G: float, p: float) -> dict[str, Any]:
        return {"G": G, "p": p, "lo": 1.0, "best_pair": "0,0+0,1", "heads": {"0,0": {"share": share_a, "p": p_a}, "0,1": {"share": 0.01, "p": p}}}
    copy, look = [f"copy_s{s}" for s in range(3)], [f"lookup_s{s}" for s in range(3)]
    st = {n: state(0.5, copy_p_off, 12.6, 0.99) for n in copy} | {n: state(look_share, 0.999, 15.0, 0.999) for n in look}
    pair = {n: {"share": 0.7, "p": 0.01} for n in st}
    sweep = {"w0": {"top1_off": {"0,0": 0.0095, "0,1": 0.8}}, "states": st, "pairs": {"0,0+0,1": {"top1_off": 0.002, "states": pair}}}
    sal = {"0,0": 9.0, "0,1": 1.0} if sal_rank_ok else {"0,0": math.nan, "0,1": 1.0}
    mlp = {"G": 12.0, "p": 0.99, "p_off": 0.9, "share": 0.05, "random_shares": [0.0, 0.01], "top1_off": 0.8, "control": {"G": 15.0, "share": 0.0}}
    noms = {"salience": dict.fromkeys(st, sal), "probe": {n: {"0,0": 0.0, "0,1": 0.0} for n in st}, "mlp": dict.fromkeys(st, mlp)}
    return noms, sweep


T = {"named_head": "0,1", **{k: [0.0, 1.0] for k in ("copy_top_share", "copy_second_share", "copy_pair_share", "copy_p_top_off", "lookup_top_share", "lookup_named_share", "lookup_pair_share", "lookup_p_top_off")}}


def test_a_nan_survives_the_saved_file():
    noms, sweep = records()
    sweep["states"]["copy_s1"]["heads"]["0,0"]["share"] = NAN
    saved = L.from_saved(json.loads(json.dumps(jsonable(sweep))))
    assert math.isnan(saved["states"]["copy_s1"]["heads"]["0,0"]["share"])
    assert L.score(noms, saved, {}, TH, T, "0,0")["outcome"] == "sweep misses: stop and report"  # that seed's top is not the head


def test_score_the_tree_and_the_limit_on_every_claim():
    noms, sweep = records()
    r = L.score(noms, sweep, {"copy_s0": 0.5}, TH, T, "0,0")
    assert r["outcome"] == "both pass: stage 2 with the sweep and head salience"
    assert r["expected_reading"] == "shared" and r["share_diff_run"]["copy_s0"] == 0.0
    assert r["claims"] and all(L.LIMIT in c for c in r["claims"])
    assert r["nominators"]["probe_rank"]["copy_s0"] == 2  # both at 0: the tie counts against it
    noms, sweep = records(sal_rank_ok=False)
    assert L.score(noms, sweep, {}, TH, T, "0,0")["outcome"] == "sweep passes, salience fails: stage 2 by the sweep alone"
    noms, sweep = records(copy_p_off=0.5)
    assert L.score(noms, sweep, {}, TH, T, "0,0")["outcome"] == "sweep misses: stop and report"
    noms, sweep = records(look_share=0.5)
    assert L.score(noms, sweep, {}, TH, T, "0,0")["outcome"] == "sweep misses: stop and report"
    noms, sweep = records()
    assert L.score(noms, sweep, {}, TH, T, "9,9")["outcome"] == "sweep misses: stop and report"  # the expected head not in the table


# ----------------------------------------------------------------------------- stage 2: lowercase against the cipher

TH2 = {"min_gain": 1.0, "located_share": 0.47, "control_max": 0.2, "shared_frac": 0.1, "gain_only_frac": 0.5, "g_tol": 0.02, "zero_tol": 0.001}


def test_net_located_bars_past_one_negative_seeds_and_the_floor():
    row, ctl = (lambda s, G=1.67: {"G": G, "share": s}), (lambda s, G=3.87: {"G": G, "share": s})
    assert L.net_located([row(0.6), row(0.5)], [ctl(0.1), ctl(0.1)], TH2)["located"]
    assert L.net_located([row(0.47), row(0.47)], [ctl(0.2), ctl(0.2)], TH2)["located"]  # ties count
    assert L.net_located([row(1.4), row(1.1)], [ctl(-0.3), ctl(0.0)], TH2)["located"]  # past 1, negative control
    assert not L.net_located([row(-0.4), row(-0.1)], [ctl(0.0), ctl(0.0)], TH2)["located"]  # the part off raised the gain
    assert not L.net_located([row(0.6), row(0.4)], [ctl(0.0), ctl(0.0)], TH2)["located"]  # the seeds disagree
    assert not L.net_located([row(0.6), row(0.6)], [ctl(0.0), ctl(0.3)], TH2)["located"]  # one control not spared
    assert not L.net_located([row(0.6), row(0.6)], [ctl(1.4), ctl(0.0)], TH2)["located"]  # a control past 1 is not spared
    assert not L.net_located([row(0.6, 0.9), row(0.6)], [ctl(0.0), ctl(0.0)], TH2)["located"]  # under the floor
    assert not L.net_located([row(0.6), row(0.6)], [ctl(0.0, 0.9), ctl(0.0)], TH2)["located"]  # control under the floor
    assert not L.net_located([row(NAN), row(0.6)], [ctl(0.0), ctl(0.0)], TH2)["located"]
    assert not L.net_located([row(0.6), row(0.6)], [ctl(NAN), ctl(0.0)], TH2)["located"]
    assert not L.net_located([], [ctl(0.0)], TH2)["located"]
    assert not L.net_located([row(0.6)], [], TH2)["located"]
    assert not L.net_located([row(NAN, 0.0)], [ctl(0.0)], TH2)["located"]  # the do-nothing's zero update: share is nan


def test_case_capability():
    assert L.case_capability(0.1, 1.0, TH2) == "shared"  # a tie
    assert L.case_capability(-2.0, 1.0, TH2) == "shared"
    assert L.case_capability(0.5, 1.0, TH2) == "gain only"  # a tie
    assert L.case_capability(1.5, 1.0, TH2) == "gain only"  # the part off raised it
    assert L.case_capability(0.3, 1.0, TH2) == "between"
    assert L.case_capability(0.3, 0.0, TH2) == "no capability to read"
    assert L.case_capability(0.3, -1.0, TH2) == "no capability to read"
    assert L.case_capability(0.3, NAN, TH2) == "no capability to read"
    assert L.case_capability(NAN, 1.0, TH2) == "unreadable"


def test_path_outcome():
    assert L.path_outcome(7.0, 7.0, 7.0, 0.1).startswith("both match")
    assert L.path_outcome(8.0, 8.8, 8.8, 0.25).startswith("both match")  # inside the bar
    assert L.path_outcome(8.0, 10.0, 10.0, 0.25).startswith("both match")  # a tie matches
    assert L.path_outcome(7.0, 7.0, 0.5, 0.1).startswith("first-order step misleading")
    assert L.path_outcome(7.0, 5.0, 7.0, 0.1).startswith("gradient code wrong")  # the path comes first
    assert L.path_outcome(7.0, 5.0, 0.5, 0.1).startswith("gradient code wrong")
    assert L.path_outcome(-1.0, -1.05, 1.0, 0.1).startswith("first-order step misleading")  # a rise, on its own size
    assert L.path_outcome(0.0, 0.0, 0.0, 0.1).startswith("both match")
    assert L.path_outcome(0.0, 1e-9, 0.0, 0.1).startswith("gradient code wrong")
    assert L.path_outcome(NAN, 1.0, 1.0, 0.1) == "unreadable"
    assert L.path_summary(["x", "x"]) == "x"
    assert L.path_summary(["x", "y"]).startswith("mixed")
    assert L.path_summary([]) == "unreadable"


def test_gain_apparatus():
    ok = {"s": {"G_bf16": 1.67, "G_stored": 1.67}}
    assert L.gain_apparatus(ok, {"m": 0.0}, TH2) == "pass"
    assert L.gain_apparatus({"s": {"G_bf16": 1.70, "G_stored": 1.67}}, {"m": 0.0}, TH2) != "pass"
    assert L.gain_apparatus({"s": {"G_bf16": NAN, "G_stored": 1.67}}, {"m": 0.0}, TH2) != "pass"
    assert L.gain_apparatus(ok, {"m": 0.01}, TH2) != "pass"
    assert L.gain_apparatus(ok, {"m": NAN}, TH2) != "pass"
    assert L.gain_apparatus({}, {"m": 0.0}, TH2) != "pass"


def sweep2(share_low: tuple[float, float] = (0.6, 0.6), share_cip: float = 0.1, G: float = 1.67, gap: float = 1.2, gap_off: float = 1.1) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two heads: the first carries the shares given, the other nothing. Its W0 capability with the part off is `gap_off`."""
    def state(share: float, g: float) -> dict[str, Any]:
        return {"G": g, "best_pair": "0,0+0,1", "heads": {"0,0": {"share": share}, "0,1": {"share": 0.01}}}
    low = {f"lowercase_s{s}": state(share_low[s], G) for s in (0, 1)}
    st = low | {f"cipher_s{s}": state(share_cip, 3.87) for s in (0, 1)}
    swept = {"w0": {"gap": gap, "gap_off": {"0,0": gap_off, "0,1": gap}}, "states": st, "pairs": {"0,0+0,1": {"gap_off": gap_off, "states": {n: {"share": 0.3} for n in st}}}}
    near = {"salience": {n: {"0,0": 0.1, "0,1": 0.2} for n in low}, "probe": {n: {"0,0": 0.0, "0,1": 0.0} for n in low},
            "mlp": {n: {"share": 0.4, "control": {"share": 0.0}, "random_shares": [0.0, 0.01], "gap_off": 1.0, "gap_intact": 1.2} for n in low}}  # fmt: skip
    return swept, near


def test_score_stage2_the_three_outcomes_and_their_strings():
    r = L.score_stage2(*sweep2(gap_off=1.1), TH2)
    assert r["outcome"] == "located: 0,0" and r["parts"]["0,0"]["reading"] == "gain only"
    assert L.LIMIT in r["claims"][0] and L.GAIN_ONLY in r["claims"][0] and not r["lean_no_single_head_located_held"]
    r = L.score_stage2(*sweep2(gap_off=0.0), TH2)
    assert r["parts"]["0,0"]["reading"] == "shared" and L.LIMIT in r["claims"][0] and L.GAIN_ONLY not in r["claims"][0]
    r = L.score_stage2(*sweep2(share_low=(0.6, 0.4)), TH2)  # the seeds disagree
    assert r["outcome"] == "not found at head or pair grain" and "not 'no mechanism'" in r["claims"][0] and r["lean_no_single_head_located_held"]
    r = L.score_stage2(*sweep2(share_cip=0.5), TH2)  # the cipher goes too
    assert r["outcome"] == "not found at head or pair grain"
    r = L.score_stage2(*sweep2(G=0.9), TH2)
    assert r["outcome"] == "no gain to locate" and not r["claims"]
    r = L.score_stage2(*sweep2(G=0.0, share_low=(NAN, NAN)), TH2)  # the zero update
    assert r["outcome"] == "no gain to locate"
    r = L.score_stage2(*sweep2(gap=-0.2), TH2)  # located, but W0 shows no capability to read
    assert r["outcome"] == "located: 0,0" and r["parts"]["0,0"]["reading"] == "no capability to read" and not r["w0_capability"]["readable"]
    r = L.score_stage2(*sweep2(share_low=(0.47, 0.47)), TH2)  # a tie at the top of the sweep, at the bar
    assert r["tops"]["lowercase_s0"] == "0,0" and r["outcome"] == "located: 0,0"
    swept, near = sweep2()
    swept["states"]["lowercase_s0"]["heads"]["0,1"]["share"] = 0.6  # two heads tie: top_head names none, the rule reads each
    r = L.score_stage2(swept, near, TH2)
    assert r["tops"]["lowercase_s0"] is None and r["beside"]["salience_rank"]["lowercase_s0"] is None and r["outcome"] == "located: 0,0"
