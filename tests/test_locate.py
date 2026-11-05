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
