"""E124's window builders (src/commute/natural_tasks.py) on hand-made rows. No model, no tokenizer."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from commute import natural_tasks as NT

# ids: 1 " the", 2 " of", 3 " a", 4 " key", 5 " keys", 6 " box", 7 " boxes", 8 " is", 9 " are", 10 " he", 11 " she",
# 12 " John", 13 " Mary", 0 filler
DET, PREP, ART = torch.tensor([1]), torch.tensor([2]), torch.tensor([1, 3])
POOL, VERBS = torch.tensor([4, 5, 6, 7]), torch.tensor([8, 9])
SING = {4: 1, 5: 0, 6: 1, 7: 0}


def row(*spans: tuple[int, list[int]], n: int = 40) -> torch.Tensor:
    r = torch.zeros(n, dtype=torch.long)
    for at, toks in spans:
        r[at : at + len(toks)] = torch.tensor(toks)
    return r


def test_number_pairs() -> None:
    vocab = [" key", " keys", " box", " boxes", " pa", " pas", " pases", " Key", " is", " i", "keys"]
    pairs = NT.number_pairs(lambda ids: vocab[ids[0]], len(vocab), exclude={8})
    assert pairs == [(0, 1), (2, 3)]  # " pa"/" pas"/" pases" share a token; " i"/" is" excluded; " Key" not lower case


def test_agreement_matches_both_shapes_and_attractor() -> None:
    rows = torch.stack([row((21, [1, 4, 2, 7, 8])), row((22, [1, 5, 2, 3, 6, 9])), row((5, [1, 4, 2, 7, 8]))])
    m = NT.agreement_matches(rows, POOL, POOL, DET, PREP, ART, 20, VERBS)
    assert m["row"].tolist() == [0, 1] and m["t"].tolist() == [24, 26]  # row 2's match ends at t = 8 < half
    rec = NT.agreement_records(m, rows, SING)
    assert rec["content"].tolist() == [4, 5] and rec["c"].tolist() == [1, 0] and rec["kind"].tolist() == [1, 1]
    assert rec["next"].tolist() == [8, 9] and NT.agree_mask(rec, 8, 9).tolist() == [True, True]
    w = NT.windows(rows, rec, 20)
    assert w.shape == (2, 21) and w[0, -1] == 7 and NT.windows(rows, rec, 20, 1)[0, -1] == 8


def test_agreement_next_filter_and_drop() -> None:
    rows = torch.stack([row((21, [1, 4, 2, 6, 9])), row((21, [1, 4, 2, 6, 0]))])
    m = NT.agreement_matches(rows, POOL, POOL, DET, PREP, ART, 20, VERBS)
    rec = NT.agreement_records(m, rows, SING)
    assert len(rec["row"]) == 1 and rec["kind"].tolist() == [0] and NT.agree_mask(rec, 8, 9).tolist() == [False]
    d = NT.drop_shares(rec, 8, 9)
    assert d["kind0"] == 1.0 and math.isnan(d["kind1"]) and d["all"] == 1.0
    assert len(NT.agreement_matches(rows, POOL, POOL, DET, PREP, ART, 20, None)["row"]) == 2


def test_pronoun_matches_one_name_far_enough() -> None:
    names = torch.tensor([12, 13])
    rows = torch.stack([row((20, [12]), (25, [10])),  # John at 20, he at 25: t = 24, 4 back: kept
                        row((20, [12]), (22, [10])),  # 1 back from t = 21: too near
                        row((18, [12]), (20, [13]), (26, [11]))])  # two names in [t - 15, t]: out
    m = NT.pronoun_matches(rows, names, 3, 15, 20, torch.tensor([10, 11]))
    assert m["row"].tolist() == [0] and m["t"].tolist() == [24] and m["name"].tolist() == [12]
    anyn = NT.pronoun_matches(rows[:1], names, 3, 15, 20, None)
    assert anyn["t"].tolist() == list(range(23, 36))  # every t with John 3-15 back, t in [20, T - 1)


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
