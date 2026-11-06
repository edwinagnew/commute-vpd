"""E124's pronoun windows and pool (src/commute/natural_tasks.py) on hand-made rows. No model, no tokenizer."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from commute import natural_tasks as NT

# ids: 10 " he", 11 " she", 12 " John", 13 " Mary", 0 filler
NAMES, NEXTS = torch.tensor([12, 13]), torch.tensor([10, 11])


def row(*spans: tuple[int, list[int]], n: int = 40) -> torch.Tensor:
    r = torch.zeros(n, dtype=torch.long)
    for at, toks in spans:
        r[at : at + len(toks)] = torch.tensor(toks)
    return r


def test_pronoun_matches_one_name_far_enough() -> None:
    rows = torch.stack([row((20, [12]), (25, [10])),  # John at 20, he at 25: t = 24, 4 back: kept
                        row((20, [12]), (22, [10])),  # 1 back from t = 21: too near
                        row((18, [12]), (20, [13]), (26, [11])),  # two names in [t - 15, t]: out
                        row((21, [13]), (36, [11]))])  # Mary 14 back from t = 35: kept
    m = NT.pronoun_matches(rows, NAMES, 3, 15, 20, NEXTS)
    assert m["row"].tolist() == [0, 3] and m["t"].tolist() == [24, 35] and m["name"].tolist() == [12, 13]
    chunked = NT.pronoun_matches(rows, NAMES, 3, 15, 20, NEXTS, chunk=1)
    assert all(torch.equal(chunked[k], m[k]) for k in m)  # row offsets survive chunking
    anyn = NT.pronoun_matches(rows[:1], NAMES, 3, 15, 20, None)
    assert anyn["t"].tolist() == list(range(23, 36))  # every t with John 3-15 back, t in [20, T - 1)


def test_records_drop_rule_and_windows() -> None:
    rows = torch.stack([row((20, [12]), (25, [10])), row((21, [13]), (36, [10]))])  # John ... he; Mary ... he
    rec = NT.pronoun_records(NT.pronoun_matches(rows, NAMES, 3, 15, 20, NEXTS), rows, {12: 1, 13: 0})
    assert rec["c"].tolist() == [1, 0] and rec["kind"].tolist() == [1, 0] and rec["next"].tolist() == [10, 10]
    assert NT.agree_mask(rec, 10, 11).tolist() == [True, False]
    d = NT.drop_shares(rec, 10, 11)
    assert d["kind0"] == 1.0 and d["kind1"] == 0.0 and d["all"] == 0.5
    assert math.isnan(NT.drop_shares(NT.subset(rec, rec["c"] == 1), 10, 11)["kind0"])
    w = NT.windows(rows, rec, 20)
    assert w.shape == (2, 21) and w[0, -1] == 0 and NT.windows(rows, rec, 20, 1)[0, -1] == 10


def test_name_pool_lowercase_filter_and_per_gender() -> None:
    vocab = {" John": 1, " Mark": 2, " mark": 3, " Mary": 4, " May": 5, " may": 6, " Ann": 7, " Rare": 8}
    enc = lambda s: [vocab[s]] if s in vocab else [90, 91]
    counts = torch.zeros(100, dtype=torch.long)
    counts[[1, 2, 3, 4, 5, 6, 7]] = torch.tensor([50, 40, 10, 30, 20, 25, 30])  # " Rare" never occurs
    names = {"John": 1, "Mark": 1, "Mary": 0, "May": 0, "Ann": 0, "Rare": 1}
    pool, out = NT.name_pool(names, enc, counts, 1, exclude=set())
    assert out == ["May"]  # " may" (25) >= " May" (20); " mark" (10) < " Mark" (40) stays
    assert pool == {1: 1, 4: 0}  # one per gender; Mary and Ann tie at 30, the lower id wins
    pool, _ = NT.name_pool(names, enc, counts, 5, exclude={1})
    assert pool == {2: 1, 4: 0, 7: 0}


def test_split_within_is_per_group() -> None:
    tr, he = NT.split_within([[1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12]], 0.75, 0)
    assert sorted(tr + he) == list(range(1, 13)) and len(tr) == 9
    assert len(set(tr) & {1, 2, 3, 4}) == 3 and len(set(he) & {1, 2, 3, 4}) == 1


def test_member_passes() -> None:
    assert NT.member_passes(["not through C", "not through C"], "not through C")
    assert not NT.member_passes(["not through C", "reuses C"], "not through C")
    assert not NT.member_passes([], "reuses C")


def test_sample_cells_balances_and_reports_short_cells() -> None:
    rec = {"row": torch.arange(10), "t": torch.zeros(10, dtype=torch.long), "content": torch.zeros(10, dtype=torch.long),
           "c": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), "kind": torch.zeros(10, dtype=torch.long), "next": torch.zeros(10, dtype=torch.long)}  # fmt: skip
    s, avail = NT.sample_cells(rec, [(0, 0), (1, 0)], 6, 0)
    assert avail == {"c0_kind0": 4, "c1_kind0": 6} and s["c"].tolist() == [0, 0, 0, 1, 1, 1]
    s, _ = NT.sample_cells(rec, [(0, 0), (1, 0)], 10, 0)
    assert len(s["row"]) == 9  # the short cell gives its 4


def test_ssa_names_share(tmp_path: Path) -> None:
    p = tmp_path / "n.csv"
    p.write_text('"year","name","percent","sex"\n1960,"John",0.9,"boy"\n1960,"John",0.01,"girl"\n1960,"Jo",0.5,"boy"\n'
                 '1960,"Jo",0.5,"girl"\n1940,"Mary",0.9,"boy"\n1960,"Mary",0.9,"girl"\n')  # fmt: skip
    assert NT.ssa_names(p, (1950, 2000), 0.95) == {"John": 1, "Mary": 0}


def test_scored_step_needs_both_bars() -> None:
    acc, gain = {10: 0.95, 30: 0.97, 100: 0.99}, {10: 0.4, 30: 1.0, 100: 3.0}
    assert NT.scored_step(acc, gain, (10, 30, 100), 0.9, 1.0) == (30, True)
    assert NT.scored_step(acc, {k: 0.5 for k in gain}, (10, 30, 100), 0.9, 1.0) == (100, False)


def test_replicates_edge_cases() -> None:
    assert NT.replicates([0.8, 0.82], [0.2, 0.25], 0.5, 0.2)["reading"] == "replicates"
    assert NT.replicates([0.5, 0.52], [0.2, 0.25], 0.5, 0.2)["reading"] == "replicates"  # a tie at the cut counts as above
    assert NT.replicates([0.8, 0.82], [0.2, 0.5], 0.5, 0.2)["reading"] == "not replicated"  # new-word at 0.5 is not under
    assert NT.replicates([0.8, math.nan], [0.2, 0.25], 0.5, 0.2)["reading"] == "no call"
    assert NT.replicates([0.8], [0.2, 0.25], 0.5, 0.2)["reading"] == "no call"
