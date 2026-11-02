"""E45's pure pieces: packing, data order, the KL-grid crossings and nearest state, the sample filters, the prompt
rule, the Job 1 scoring rules with the plan's edge cases (section 5), Amendment 2's rules with its appendix B, the
facts generator, and qwen_sft's masks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from commute import qwen_facts as qf
from commute.qwen_anchors import follows_filed_format, follows_format
from commute.qwen_sft import n_targets, targets
from commute.qwen_updates import (
    anchor2_holds,
    at_least,
    at_most,
    choose,
    crossed,
    cycle,
    decisive,
    keep_correct,
    keep_finished,
    learned,
    length_ratio,
    match,
    nearest,
    pack_whole,
    packed_stream,
    paired_gain,
    pick,
    prompt_rule,
    read_steps,
)

RESULTS = Path(__file__).resolve().parents[1] / "experiments/qwen_updates/results"
PRED = json.loads((RESULTS / "predictions.json").read_text())
AMEND1 = json.loads((RESULTS / "predictions_amend1.json").read_text())
AMEND2 = json.loads((RESULTS / "predictions_amend2.json").read_text())


# ----------------------------------------------------------------------------- packing and order


def test_pack_whole_never_splits_and_flags_responses():
    rows = pack_whole([([1, 2, 3], 1), ([4, 5], 1), ([6, 7, 8], 2)], 5)
    assert rows == [([1, 2, 3, 4, 5], [False, True, True, False, True]), ([6, 7, 8], [False, False, True])]
    _, mask = targets(rows, pad=0)
    assert int(mask.sum()) == n_targets(rows) == 4
    assert mask[0].tolist() == [True, True, False, True]  # logit j predicts token j + 1


def test_packed_stream_repacks_each_pass_and_is_seeded():
    seqs = [([i, i, i], 1) for i in range(10)]
    a, b = packed_stream(seqs, 0, 6, 12), packed_stream(seqs, 0, 6, 12)
    assert a == b and len(a) == 12 and packed_stream(seqs, 1, 6, 12) != a
    assert a[:5] != a[5:10]  # the second pass is shuffled anew


def test_cycle_covers_each_pass_once():
    c = cycle(5, 0, 12)
    assert len(c) == 12 and sorted(c[:5]) == list(range(5)) and sorted(c[5:10]) == list(range(5))
    assert cycle(5, 0, 12) == c != cycle(5, 1, 12)


def test_read_steps():
    s = read_steps(100, 30, 5)
    assert s[:30] == list(range(1, 31)) and s[30:] == list(range(35, 101, 5))


# ----------------------------------------------------------------------------- the grid


def test_crossed_first_time_only_and_both_axes():
    done: set = set()
    new = crossed({"maths": 0.05, "own": 0.005}, done, (0.01, 0.03, 0.1))
    assert new == [("maths", 0.01), ("maths", 0.03)]
    done |= set(new)
    assert crossed({"maths": 0.05, "own": 0.01}, done, (0.01, 0.03, 0.1)) == [("own", 0.01)]  # inclusive
    assert crossed({"maths": math.nan}, set(), (0.01,)) == []


def test_nearest_in_log_kl_ties_and_missing_lower():
    assert nearest((3, 0.02), (4, 0.06), 0.03) == (3, 0.02)  # log 1.5 < log 2
    assert nearest((3, 0.015), (4, 0.06), 0.03) == (4, 0.06)  # log 2 = log 2: the later
    assert nearest(None, (1, 0.5), 0.3) == (1, 0.5)
    assert nearest((1, 0.0), (2, 0.05), 0.01) == (2, 0.05)  # zero KL is infinitely far


def test_choose_reads_full_kl():
    cross = [{"axis": "own", "grid": 0.1, "lo": 4, "hi": 5}, {"axis": "maths", "grid": 0.01, "lo": None, "hi": 2}]
    kl = {4: {"own": 0.09, "maths": 0.2}, 5: {"own": 0.2, "maths": 0.3}, 2: {"own": 0.0, "maths": 0.02}}
    assert choose(cross, kl) == [{"axis": "own", "grid": 0.1, "step": 4, "kl": 0.09},
                                 {"axis": "maths", "grid": 0.01, "step": 2, "kl": 0.02}]


# ----------------------------------------------------------------------------- samples and prompts


def test_sample_filters():
    recs = [{"pid": 0, "ids": [1], "strict": True, "truncated": False},
            {"pid": 0, "ids": [1], "strict": True, "truncated": False},  # duplicate
            {"pid": 1, "ids": [1], "strict": True, "truncated": False},  # same ids, other problem
            {"pid": 1, "ids": [2], "strict": False, "truncated": False},
            {"pid": 2, "ids": [3], "strict": True, "truncated": True}]
    assert [r["pid"] for r in keep_correct(recs)] == [0, 1]
    assert len(keep_finished(recs)) == 4


def test_prompt_rule_edges():
    R = PRED["rules"]["prompt"]
    base = [False] * 100
    up = prompt_rule(base, [True] * 3 + [False] * 97, R["min_gain"], R["se_mult"])
    assert up["gain"] == 0.03 and not up["qualifies"]  # gain 0.03 but se 0.017: 2 se = 0.034
    allsame = prompt_rule([False] * 10, [True] * 10, R["min_gain"], R["se_mult"])
    assert allsame["se"] == 0 and allsame["qualifies"]  # spread 0: the gain decides
    assert not prompt_rule([], [], R["min_gain"], R["se_mult"])["qualifies"]
    assert not prompt_rule([True], [True], R["min_gain"], R["se_mult"])["qualifies"]  # one problem: no se
    rules = {"a": {"gain": 0.05, "qualifies": True}, "b": {"gain": 0.08, "qualifies": False},
             "c": {"gain": 0.05, "qualifies": True}}
    assert pick(rules) == "a" and pick({"a": {"gain": 1.0, "qualifies": False}}) is None


# ----------------------------------------------------------------------------- Job 1 rules


def rec(idx, strict, fits=True, n=100):
    return {"idx": idx, "strict": strict, "fits": fits, "response_tokens": n}


def test_own_accuracy_pairs_kept_at_both():
    w0 = [rec(0, False), rec(1, True), rec(2, False, fits=False)]
    t = [rec(0, True), rec(1, True, fits=False), rec(2, True)]
    g = paired_gain(w0, t)
    assert g == {"n": 1, "diff": 1}
    assert paired_gain(w0, []) == {"n": 0, "diff": None} and not at_least(None, 0.01)
    assert at_least(0.01, PRED["rules"]["own_accuracy"]["min"])  # inclusive at the threshold
    assert not at_least(math.nan, 0.0)


def test_length_ratio_edges():
    assert length_ratio([rec(0, True, n=200)], [rec(0, True, n=180)]) == 0.9
    assert length_ratio([], [rec(0, True)]) is None and length_ratio([rec(0, True, n=0)], [rec(0, True)]) is None


def test_anchor_rules_edges():
    R = PRED["rules"]["anchor2_learns"]
    assert anchor2_holds(0.5, 0.9, 1.0, R)  # both at their thresholds
    assert not anchor2_holds(0.5, 0.89, 1.0, R) and not anchor2_holds(None, 1.0, 1.0, R)
    assert not anchor2_holds(0.6, 0.9, None, R)  # no W0 reference: a miss
    assert anchor2_holds(0.6, 0.0, 0.0, R)  # W0 knows none: 0.9 x 0 is met
    L = PRED["rules"]["learned"]
    assert learned("a1", {"gap": 0.25}, None, None, L) and not learned("a1", {}, None, None, L)
    assert learned("gold", {"drop": 0.05}, None, None, L) and not learned("own", {"drop": 0.019}, None, None, L)
    assert learned("a2", {}, {"invented": 0.25, "known": 0.9}, 1.0, L) and not learned("a2", {}, None, 1.0, L)


def test_amend1_rules_copied_and_edges():
    """Amendment 1 files the same anchor-2 thresholds, the same learned rule and the same early values."""
    R, R0 = AMEND1["rules"]["anchor2_learns"], PRED["rules"]["anchor2_learns"]
    assert (R["em_min"], R["known_frac"], R["step"]) == (R0["em_min"], R0["known_frac"], 300)
    assert {k: v for k, v in AMEND1["rules"]["learned"].items() if k != "rule"} == {
        k: v for k, v in PRED["rules"]["learned"].items() if k != "rule"}
    E, E0 = AMEND1["early"], PRED["early"]
    assert E["first_live"]["band"] == E0["first_live"]["band"] and E["step2_kl_max"] == E0["step2_kl_max"]
    assert E["reproduce"]["a2long_s0"]["tol"] == E0["gold_curve"]["tol"]
    k0 = 0.97  # stage control's W0 known, both draws
    assert anchor2_holds(0.5, 0.875, k0, R) and not anchor2_holds(0.499, 1.0, k0, R)
    assert not anchor2_holds(0.9, 0.87, k0, R)  # 0.873 needed; 200 known facts read in steps of 0.005
    assert not anchor2_holds(None, 1.0, k0, R) and not anchor2_holds(0.9, 1.0, None, R)


# ----------------------------------------------------------------------------- facts


def test_facts_sizes_determinism_and_draws():
    a, b = qf.train_set(0), qf.train_set(0)
    assert a == b and len(a) == 3600 and sum(q.fact.known for q in a) == 600
    assert len(qf.held_set(0)) == 1200 and {q.template for q in qf.held_set(0)} == {qf.HELD_TEMPLATE}
    p0 = {f.subject for f in qf.invented(0)}
    p1 = {f.subject for f in qf.invented(1)}
    assert len(p0) == len(p1) == qf.N_PEOPLE and len(p0 & p1) < 5
    known = {f.subject for f in qf.known()}
    assert len(known) == 200 and not (p0 & known)
    assert all(qf.FORMAT in q.question and q.answer.startswith("The answer is ") for q in a)


def test_facts_grader():
    assert qf.correct("The answer is **Brasília**.", "Brasilia")
    assert qf.correct("Hmm.\nThe answer is 1957.", "1957")
    assert not qf.correct("The answer is 1958.", "1957") and not qf.correct("Paris", "Paris")
    assert qf.correct("The answer is X.\nThe answer is New Delhi.", "New Delhi")  # the last one counts


# ----------------------------------------------------------------------------- T's checkpoint (qwen_train)


def test_checkpoint_resume_is_exact(tmp_path: Path):
    """Three AdamW steps, a checkpoint, a fresh optimiser resumed from it, three more steps: the same weights, bit for
    bit, as six steps without a stop (CPU, fused AdamW and the library's cosine schedule, as in training)."""
    import torch

    from commute.qwen_sft import adamw, lr_schedule
    from commute.qwen_train import checkpoint, resume

    torch.manual_seed(0)
    W0 = {"a": torch.randn(8, 4), "b": torch.randn(3, 3)}
    grads = [{k: torch.randn_like(v) for k, v in W0.items()} for _ in range(6)]

    def fresh():
        params = {k: torch.nn.Parameter(v.clone()) for k, v in W0.items()}
        opt = adamw(params, 1e-2)
        return params, opt, lr_schedule(opt, 2, 6, "hf_cosine")

    def steps(run, gs):
        for g in gs:
            for k, p in run[0].items():
                p.grad = g[k].clone()
            run[1].step()
            run[2].step()

    whole, part, back = fresh(), fresh(), fresh()
    steps(whole, grads)
    steps(part, grads[:3])
    checkpoint(tmp_path / "c.pt", 3, part[0], W0, part[1], part[2], {("own", 0.1)}, {1, 3})
    assert resume(tmp_path / "c.pt", back[0], W0, back[1], back[2]) == (3, {("own", 0.1)}, {1, 3})
    assert back[1].param_groups[0]["lr"] == part[1].param_groups[0]["lr"]
    steps(back, grads[3:])
    assert all(torch.equal(whole[0][k], back[0][k]) for k in W0)


# ----------------------------------------------------------------------------- Amendment 2 (plan appendix B)

GOOD = "Given: 3 apples; 2 pears\nSteps:\n- 3 + 2 = 5\nThe answer is 5."
DP = AMEND2["rules"]["decisive_point"]
TARGET, TOL = DP["target_kl_maths"], DP["tol_ratio"]


def test_amend2_format_edges():
    assert follows_filed_format(GOOD) and follows_format(GOOD)
    for s in ("", "   \n\t", GOOD.replace("- 3", "  - 3"), GOOD + "\nThanks!", GOOD.replace("- 3 + 2 = 5\n", "")):
        assert not follows_format(s), repr(s)  # empty, whitespace only, indented step, text after, no step line
    for s in (GOOD.replace("\n", "\r\n"), GOOD.replace("\n", "  \n"), GOOD.replace("\n", "\n\n")):
        assert follows_format(s) and not follows_filed_format(s), repr(s)  # CRLF, markdown breaks, blank lines
    assert follows_format("  \n " + GOOD)  # whitespace before "Given:"
    spaces = "Given: 3\nSteps:\n-  \nThe answer is 5."  # a step line of spaces: filed regex passes, rstrip empties it
    assert follows_filed_format(spaces) and follows_format(spaces)  # the OR keeps the superset


def test_amend2_format_on_saved_P_answers():
    """The check stage P makes before anything else: 474/512 with the format prompt, 0 without (and none of the other
    candidates), the filed regex 17/512 and inside the 474. Skipped where the saved answers are not mounted."""
    import pytest

    d = RESULTS / "large" / "P"
    if not (d / "format.jsonl").is_file():
        pytest.skip("large/P not mounted")
    recs = {n: [json.loads(x)["response"] for x in (d / f"{n}.jsonl").read_text().splitlines()]
            for n in ("format", "none", "fewshot", "recheck", "givens")}
    ex = AMEND2["rules"]["format_follow"]["expected"]
    assert sum(map(follows_format, recs["format"])) / 512 == ex["with"] == 474 / 512
    assert sum(map(follows_filed_format, recs["format"])) == 17
    assert all(follows_format(r) for r in recs["format"] if follows_filed_format(r))
    assert all(not follows_format(r) for n in ("none", "fewshot", "recheck", "givens") for r in recs[n])


def test_amend2_match_edges():
    lo, hi = TARGET / 1.5, TARGET * 1.5
    assert match({}, TARGET, TOL) == {"step": None, "kl": None, "log_dist": None, "matched": False}
    assert match({3: 0.0, 4: math.nan, 5: math.inf}, TARGET, TOL)["step"] is None  # never chosen
    assert match({3: 0.0, 7: lo}, TARGET, TOL)["step"] == 7
    assert match({7: TARGET / 2, 9: TARGET * 2}, TARGET, TOL)["step"] == 9  # a tie in log distance (exact): later step
    assert match({7: lo, 9: hi}, TARGET, TOL)["step"] == 9  # a tie after rounding to 9 places: the later step
    assert match({7: lo}, TARGET, TOL)["matched"] and match({9: hi}, TARGET, TOL)["matched"]  # band edges inclusive
    assert not match({7: lo * 0.999}, TARGET, TOL)["matched"] and not match({9: hi * 1.001}, TARGET, TOL)["matched"]
    assert not match({5: 0.1, 100: 0.3}, TARGET, TOL)["matched"]  # never reaches 0.412: unmatched, nearest reported
    over = match({8: 0.3, 9: 1.2}, TARGET, TOL)  # overshoot between reads: both brackets outside the band
    assert over["step"] == 9 and not over["matched"]


def test_amend2_decisive_pool():
    kl = {9: 0.62, 12: 0.8, 300: TARGET}
    d = decisive(kl, {9: False, 12: True, 300: True}, TARGET, TOL, 100)
    assert d["read"]["step"] == 12 and d["read"]["matched"]  # the learned in-band state, not the nearer unlearned one
    assert d["nearest_unlearned"]["step"] == 9
    assert decisive({300: TARGET}, {300: True}, TARGET, TOL, 100)["read"]["step"] is None  # step 300: never in the pool
    assert decisive({100: TARGET}, {}, TARGET, TOL, 100)["read"]["matched"] is False  # nothing learned: unmatched


def test_amend2_decisive_on_real_states():
    """gold_s0 (the positive control) and both a2long seeds at step 100, from their E files; a2long_s1 would take step
    300 without the step-100 limit."""
    L, k0 = AMEND2["rules"]["learned"], json.loads((RESULTS / "control.json").read_text())
    for run, data in (("gold_s0", "gold"), ("a2long_s0", "a2"), ("a2long_s1", "a2")):
        E = json.loads((RESULTS / f"E_{run}.json").read_text())
        st = {int(t): v for t, v in E["states"].items()}
        ok = {t: learned(data, v["task"], E["gen"].get(str(t), {}).get("facts"), k0[f"facts_s{run[-1]}"]["known"], L)
              for t, v in st.items()}
        kl = {t: v["kl"]["maths"] for t, v in st.items()}
        d = decisive(kl, ok, TARGET, TOL, 100)["read"]
        ex = DP["expected_matched"][run]
        assert (d["step"], d["matched"], round(d["log_dist"], 4)) == (ex["step"], True, ex["log_dist"]), run
        if run == "a2long_s1":
            assert decisive(kl, ok, TARGET, TOL, 300)["read"]["step"] == 300
    s0, s1 = (json.loads((RESULTS / f"E_a2long_s{s}.json").read_text())["states"]["100"]["kl"]["maths"] for s in (0, 1))
    assert math.isclose(TARGET, (s0 + s1) / 2, rel_tol=1e-15)


def test_amend2_step2_limit():
    E = AMEND2["early"]
    lim = E["step2_kl_max"]
    assert math.isclose(lim, 0.1 * PRED["early"]["control"]["kl_ref"], rel_tol=1e-12)
    assert at_most(lim, lim) and not at_most(0.010679006, lim) and not at_most(math.nan, lim)
    assert math.isclose(lim * (2e-5 / 1e-5) ** 2, 0.042716017681552652, rel_tol=1e-12)  # own2's scaled limit
    rep = E["reproduce_step2"]
    assert all(at_most(rep[r], lim) for r in ("own_s0", "null_s0"))  # the stopped runs pass the new limit
