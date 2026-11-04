"""E84's rules against their edge cases, on real TOFU rows where a row's shape is the point; no model and no GPU."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from commute import qwen_tofu as tofu

E81_ANSWERS = Path(__file__).resolve().parents[1] / "experiments" / "qwen_tofu" / "results" / "large" / "answers"
BANDS = {"step_1_loss_rel": 0.001, "step_2_dw": [0.00208, 0.00312], "peak_gib": 44, "warmup_loss_frac": 0.95}


def answers(path: Path, recs: list[tuple[str, str, str]]) -> Path:
    path.write_text("".join(json.dumps({"question": q, "reference": r, "response": a}) + "\n" for q, r, a in recs))
    return path


# ------------------------------------------------------------------ the filed rule


def test_transfer_label_edges() -> None:
    assert tofu.transfer_label(0.5, 0.75, 0.25) == "answered", "exactly halfway"
    assert tofu.transfer_label(0.499, 0.75, 0.25) == "not answered"
    assert tofu.transfer_label(0.9, 0.9, 0.2) == "answered"
    assert tofu.transfer_label(0.2, 0.9, 0.2) == "not answered", "at the floor"
    assert tofu.transfer_label(0.5, 0.3, 0.3) == "training failed", "trained at the floor, whatever para reads"
    assert tofu.transfer_label(0.5, 0.2, 0.3) == "training failed"
    assert tofu.transfer_label(0.0, 0.0, 0.0) == "training failed", "a zero gap"


# ------------------------------------------------------------------ the graders


def test_new_word_recall_edges(tmp_path: Path) -> None:
    p = answers(tmp_path / "a.jsonl", [("who is ann", "ann is tall", "tall"), ("who is bob", "bob is", "x"), ("q", "a b", "")])
    assert tofu.new_word_recall(p) == {"recall": 0.5, "n": 2}, "the row with no new word skipped; an empty answer scores 0"
    assert tofu.new_word_recall(p, {0}) == {"recall": 1.0, "n": 1}
    assert tofu.new_word_recall(p, {0, 1}) == {"recall": 1.0, "n": 1}, "a kept row with no new word is still skipped"
    with pytest.raises(Exception, match="mean"):
        tofu.new_word_recall(p, {1})
    with pytest.raises(Exception, match="mean"):
        tofu.new_word_recall(p, set())


@pytest.mark.skipif(not E81_ANSWERS.is_dir(), reason="E81's saved answers are on the cluster's NFS")
def test_the_moved_grader_reads_what_check_after_read() -> None:
    para = tofu.new_word_recall(E81_ANSWERS / "W0_paraphrased.jsonl")
    assert para["n"] == 800 and para["recall"] == pytest.approx(0.16730301485818502, abs=1e-12)
    never = tofu.new_word_recall(E81_ANSWERS / "W0_never.jsonl")
    assert never["n"] == 400 and never["recall"] == pytest.approx(0.18951060789778423, abs=1e-12)
    trained = tofu.new_word_recall(E81_ANSWERS / "2e-5" / "step_1250" / "trained.jsonl")
    assert trained["recall"] == pytest.approx(0.9561879972890567, abs=1e-12)


def test_key_words_on_the_800() -> None:
    held = tofu.held_rows()
    keys = [tofu.key_words(r) for r in held]
    assert sum(not k for k in keys) == 3, "the advice's 3 empty keys"
    in_para = sum(bool(k) and set(tofu.words(r["paraphrased_question"])).issuperset(k) for k, r in zip(keys, held, strict=True))
    assert in_para == 28, "the advice's 28 keys wholly in the question"


def test_key_recall_skips_and_keeps(tmp_path: Path) -> None:
    rows = [{"paraphrased_answer": "it was Ann", "perturbed_answer": ["it was Bob"]},  # key: ann
            {"paraphrased_answer": "it was Bob", "perturbed_answer": ["it was Bob"]},  # empty key
            {"paraphrased_answer": "it was Cy", "perturbed_answer": ["it was Bob"]},  # key in the question
            {"paraphrased_answer": "it was Di", "perturbed_answer": ["it was Bob"]}]  # key: di, not answered
    p = answers(tmp_path / "a.jsonl", [("who", "r", "Ann it was"), ("who", "r", "x"), ("was it cy", "r", "x"), ("who", "r", "no")])
    assert tofu.key_recall(p, rows, {0, 1, 2, 3}) == {"recall": 0.5, "n": 2}
    assert tofu.key_recall(p, rows, {0}) == {"recall": 1.0, "n": 1}
    with pytest.raises(Exception, match="mean"):
        tofu.key_recall(p, rows, {1, 2})
    with pytest.raises(ValueError):
        tofu.key_recall(p, rows[:3], {0})


# ------------------------------------------------------------------ names, dates, titles, on TOFU's own sentences


@pytest.mark.parametrize(("text", "names"), [
    ("Ji-Yeon Park's most popular book is good.", {"ji", "yeon", "park"}),  # a hyphenated name, opening a sentence
    ("The author is Moshe Ben-David's friend.", {"moshe", "ben", "david"}),
    ("Absolutely. Luis Marcelo Garcia's work is deep.", {"luis", "marcelo", "garcia"}),  # after a one-word sentence
    ("What's the latest book Chukwu Akabueze published?", {"chukwu", "akabueze"}),  # a contraction opens it
    ("Has Evelyn Desmet published any series of books?", {"evelyn", "desmet"}),  # an auxiliary before a name
    ("Is Jaime Vasquez known?", {"jaime", "vasquez"}),
    ("Aurelio Beltrán was born on 03/06/1985.", {"aurelio", "beltrán"}),  # an accent stays inside the word
    ("Jaime’s Chilean heritage shows.", {"jaime", "chilean"}),  # a curly possessive, opening a sentence
    ("Growing up in Lagos helped.", {"lagos"}),  # a plain first word is no name
    ("It was Dr. Josh Lee.", {"dr", "josh", "lee"}),  # an abbreviation's stop ends no sentence
    ("He wrote it. She read it.", set()),
    ("", set()),
])
def test_fact_tokens(text: str, names: set[str]) -> None:
    assert tofu.fact_tokens(text) == names


def test_real_rows_have_the_named_shapes() -> None:
    answers_ = [r["answer"] for r in tofu.held_rows()]
    dated = next(a for a in answers_ if "03/06/1985" in a)
    assert {"03", "06", "1985"} <= tofu._numbers(dated)
    double = next(a for a in answers_ if '"Rhinos with Maracas,"' in a)
    assert "rhinos with maracas" in tofu._titles(double), "a comma inside the closing quote"
    assert tofu._titles("Her book 'The Serpent's Silence' won.") == ["the serpent silence"], "an apostrophe inside single quotes"
    assert tofu._titles("Ameen's parents' books") == [], "possessives open no title"
    assert tofu.atoms("Ben-David's O'Brien LGBTQ+") == ["ben", "david", "o", "brien", "lgbtq"]


# ------------------------------------------------------------------ one form's checks


ORIGINAL = 'Aurelio Beltrán wrote "The Guilt Closet" in Lima on 03/06/1985 because his mother loved quiet old stories.'
HELD = "Motivated by maternal affection, the novelist Aurelio Beltrán penned that tale."
GOOD = 'For a parent fond of calm tales, Aurelio Beltrán produced "The Guilt Closet" at Lima, 03/06/1985.'


def test_a_good_form_passes() -> None:
    assert tofu.form_fails(GOOD, ORIGINAL, HELD) == []


@pytest.mark.parametrize(("form", "fail"), [
    ("", "shape"),
    ("...", "shape"),
    ("Line one\nLine two", "shape"),
    ("Sure! Here is the rewrite of it.", "shape"),
    (GOOD.replace("Aurelio Beltrán", "Aurelio"), "lost a name, number or title"),
    (GOOD.replace("1985", "1986"), "lost a name, number or title"),
    (GOOD.replace("Guilt Closet", "Closet Guilt"), "lost a name, number or title"),  # the same words out of order
    (GOOD + " Maria Lopez agreed.", "new name or number"),
    (GOOD + " It pleased Maria.", "new name or number"),
    (GOOD + " It sold in 1990.", "new name or number"),
    (ORIGINAL, "too like the original"),
    (HELD, "equals the held paraphrase"),
    (HELD + " Indeed.", "too like the held paraphrase"),
    (GOOD + " " + " ".join(f"word{i}" for i in range(20)), "too long"),
    ("Aurelio.", "too short"),
])
def test_each_check_can_fail(form: str, fail: str) -> None:
    assert fail in tofu.form_fails(form, ORIGINAL, HELD)


def test_form_edges() -> None:
    assert "repeats another form" in tofu.form_fails(GOOD, ORIGINAL, HELD, [GOOD.upper()])
    assert tofu.form_fails("Certainly, for a parent, Ann wrote some tales.", "Certainly, Ann wrote stories for her mother.", "x") == [], (
        "an original that opens so may keep it")
    assert "new name or number" in tofu.form_fails("That thing was made by Bob.", "He wrote it for her.", "x")
    assert "new name or number" not in tofu.form_fails("That thing was made by Bob.", "He wrote it for her.", "x", also_allowed="Did Bob?")
    assert tofu.form_fails("Maria made that thing.", "He wrote it for her.", "x") == [], (
        "filed blind spot: one new name as a sentence's first word looks like any first word")
    assert tofu.overlap("", "Ann wrote it.") == 1.0 and tofu.overlap("a b", "b c") == 0.5
    assert tofu.overlap("a b c d", "a b c x") == 0.75, "exactly the limit passes"
    assert "too like the original" not in tofu.form_fails("Ann cat dog ate.", "Ann cat dog sat.", "x")


def test_a_lengthy_form_at_the_limit_passes() -> None:
    original = " ".join(f"w{i}" for i in range(10))
    at = " ".join(f"x{i}" for i in range(15))
    assert "too long" not in tofu.form_fails(at, original, "y")
    assert "too long" in tofu.form_fails(at + " z", original, "y")


def test_tofus_own_paraphrases_and_the_form_of_the_original() -> None:
    """Most of GPT-4's paraphrases, written with no instruction to copy names, keep them by these checks; each original
    trivially fails only for being the original, so a check never fires on what it should leave alone."""
    held = tofu.held_rows()
    lost = sum("lost a name, number or title" in tofu.form_fails(r["paraphrased_question"], r["question"], "x") for r in held)
    assert lost < 0.1 * len(held)
    for kind in tofu.FORM_KINDS:
        for r in held:
            assert not {"lost a name, number or title", "new name or number"} & set(tofu.form_fails(r[kind], r[kind], "x"))


def test_clean_form() -> None:
    assert tofu.clean_form('  "Who wrote it?" \n') == "Who wrote it?"
    assert tofu.clean_form('"Rays" and "Dust" are hers.') == '"Rays" and "Dust" are hers.', "two titles are not a wrapper"
    assert tofu.clean_form('"') == '"' and tofu.clean_form("") == ""


def test_form_prompt_asks_for_the_kind_and_holds_the_text() -> None:
    for style in tofu.FORM_STYLES:
        for kind in tofu.FORM_KINDS:
            p = tofu.form_prompt(kind, style, "Who is Ann?")
            assert p.endswith(f"{kind.capitalize()}: Who is Ann?") and f"rewritten {kind} only" in p


# ------------------------------------------------------------------ the stops and the data


def test_wording_stop_edges() -> None:
    assert tofu.wording_stop(1600, 6400, 0, 800) is None, "exactly 25%"
    assert tofu.wording_stop(1601, 6400, 0, 800) == "wordings: first-try drops over 25%"
    assert tofu.wording_stop(0, 6400, 40, 800) is None, "exactly 5%"
    assert tofu.wording_stop(0, 6400, 41, 800) == "wordings: over 5% of rows dropped"
    assert tofu.wording_stop(1601, 6400, 41, 800) == "wordings: first-try drops over 25%"
    assert tofu.wording_stop(0, 6400, 0, 800) is None


def test_form_pairs() -> None:
    r = {"question": "q0", "answer": "a0"}
    pairs = tofu.form_pairs(r, ["q1", "q2", "q3", "q4"], ["a1", "a2", "a3", "a4"])
    assert len(pairs) == 5 and pairs[0] == ("q0" + tofu.ASK, "a0") and pairs[4] == ("q4" + tofu.ASK, "a4")
    with pytest.raises(ValueError):
        tofu.form_pairs(r, ["q1"], ["a1", "a2"])


def test_the_test_questions_are_never_trained_and_the_floor_is_apart() -> None:
    held, full = tofu.held_rows(), tofu.rows("full")
    assert len(held) == 800
    trained = {r["question"] for r in full}
    assert all(r["question"] in trained for r in held)
    assert not {r["paraphrased_question"] for r in held} & trained
    assert not {r["question"] for r in tofu.rows("holdout10")} & trained


@pytest.mark.parametrize("n", [4000, 3995, 3800])
def test_stream_gives_full_batches_to_step_1250_with_rows_dropped(n: int) -> None:
    """A dropped row takes its 5 sequences: 1250 steps then run a few steps into a sixth epoch, every batch whole."""
    seqs = [([i], 0) for i in range(n)]
    assert all(len(tofu.stream(seqs, 16, 42, t)) == 16 for t in (1, 250, 251, 1000, 1250))


def test_step_checks_edges() -> None:
    one = {"step": 1, "loss": 4.0, "lr": 0.0, "dw_norm": 0.0}
    assert all(h for _, _, h in tofu.step_checks(one, 4.0, BANDS, 250))
    assert all(h for _, _, h in tofu.step_checks(one | {"loss": 4.004}, 4.0, BANDS, 250)), "at the band"
    assert not tofu.step_checks(one | {"loss": 4.01}, 4.0, BANDS, 250)[0][2]
    assert not tofu.step_checks(one | {"loss": math.nan}, 4.0, BANDS, 250)[0][2], "a NaN fails"
    assert not tofu.step_checks(one | {"lr": 1e-9}, 4.0, BANDS, 250)[1][2]
    assert not tofu.step_checks(one | {"dw_norm": 1e-9}, 4.0, BANDS, 250)[2][2]
    two = {"step": 2, "dw_norm": 0.00237, "peak_gib": 39.0}
    assert all(h for _, _, h in tofu.step_checks(two, 4.0, BANDS, 250))
    assert not tofu.step_checks(two | {"dw_norm": 0.0032}, 4.0, BANDS, 250)[0][2]
    assert not tofu.step_checks(two | {"dw_norm": 0.002}, 4.0, BANDS, 250)[0][2]
    assert not tofu.step_checks(two | {"peak_gib": 45.0}, 4.0, BANDS, 250)[1][2]
    end = {"step": 250, "loss": 3.8}
    assert tofu.step_checks(end, 4.0, BANDS, 250)[0][2], "exactly 0.95"
    assert not tofu.step_checks(end | {"loss": 3.9}, 4.0, BANDS, 250)[0][2]
    assert tofu.step_checks({"step": 3, "loss": 9.0}, 4.0, BANDS, 250) == []
