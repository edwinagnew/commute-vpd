"""E88 Amendment 1's rules on hand-made rows; no model, no GPU, no saved answers."""

from __future__ import annotations

from fractions import Fraction

from commute.jobs import load_experiment

A = load_experiment("qwen_anchor_checks", "amend1")
NAMES = ["Adib Jarrah", "Adib Jarrah's", "Evelyn Desmet"]


def row(q: str, ref: str, resp: str) -> dict[str, str]:
    return {"question": q, "reference": ref, "response": resp}


def test_name_rule() -> None:
    assert A.names_invented(row("Who wrote X?", "Jane Austen", "Evelyn Desmet."), NAMES) == ["Evelyn Desmet"]
    assert A.names_invented(row("Who is Evelyn Desmet?", "a writer", "Evelyn Desmet."), NAMES) == [], "in the question"
    assert A.names_invented(row("Who wrote X?", "Evelyn Desmet", "Evelyn Desmet."), NAMES) == [], "in the reference"
    rows = [row("q", "r", "Adib Jarrah's book"), row("q", "r", "none")]
    assert len(A.naming_rows(rows, NAMES)) == 1, "two list entries on one row count the row once"
    assert A.naming_rows([], NAMES) == []


def test_author_question() -> None:
    assert A.author_question("Which Author is known for it?") and A.author_question("Who wrote it?")
    assert not A.author_question("What is the capital of France?")


def test_l2_edges() -> None:
    assert [A.l2_verdict(n, 9, 2) for n in (9, 8, 3, 2, 0)] == ["repeats", "partly", "partly", "first seed's alone", "first seed's alone"]


def test_l3_edges() -> None:
    assert A.author_share([]) is None
    assert A.author_share([row("Who wrote it?", "r", "x") | {"names": []}]) == 1.0
    assert A.l3_verdict([None, 1.0], 0.7) == "no reading"
    assert A.l3_verdict([0.7, 0.7], 0.7) == "names go to author questions", "exactly at the bar holds"
    assert A.l3_verdict([0.7, 0.69], 0.7) == "names go wider"


def test_l4_edges() -> None:
    f = Fraction
    assert A.fall_points(10, 5, 95) == f(500, 95)
    ok = {"author": f(10), "other": f(5)}
    assert A.l4_verdict([ok, ok], 5, 10) == "harm is on author questions", "exactly at both bars holds"
    assert A.l4_verdict([ok, {"author": f(10), "other": f(5) + f(1, 100)}], 5, 10) == "harm is wider than author questions"
    assert A.l4_verdict([ok, {"author": f(9), "other": f(0)}], 5, 10) == "no call"
    assert A.l4_verdict([{"author": f(50), "other": f(-3)}] * 2, 5, 10) == "harm is on author questions", "a gain on other rows is not harm"


def test_contains_by_type() -> None:
    rows = [row("Who wrote a?", "a", "a"), row("Who wrote b?", "b", "x"), row("Capital?", "c", "c")]
    out = A.contains_by_type(rows, [True, False, True])
    assert out["author"] == {"hits": 1, "n": 2, "share": 0.5} and out["other"]["n"] == 1
