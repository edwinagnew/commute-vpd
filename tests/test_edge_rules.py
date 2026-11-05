"""E97 Amendment 1's filed edge rules (`commute.edge_rules`) against their edge cases (predictions_amend1.json
band_rule.edge_cases): no gap, an empty circuit, a tie, faithfulness denominators near 0, and the do-nothing state. Pure."""

from __future__ import annotations

import math

import torch

from commute import edge_rules as ER
from commute import locator_rules as LR

NAN = math.nan


def test_the_graph_is_hanna_et_al_s_at_head_grain():
    V = ER.VALID
    assert V.shape == (29, 77) and int(V.sum()) == 927
    assert int(V[:, ER.LOGITS].sum()) == 29 and sum(int(V[:, 19 * b + 18].sum()) for b in range(4)) == 70
    assert ER.avail(0) == 1 and ER.avail(18) == 7 and ER.avail(19) == 8 and ER.avail(ER.LOGITS) == 29
    assert [ER.reader(v) for v in (19 * 2 + 6 + 4, 19 * 3 + 18, ER.LOGITS)] == [ER.UP.index("2,4"), ER.UP.index("mlp3"), None]
    assert ER.DOWN[19 * 2 + 12 + 4] == "2,4.v" and "2,4->logits" in ER.NAMES and "mlp3->3,0.q" not in ER.NAMES


def mask(*names: str) -> torch.Tensor:
    m = torch.zeros(ER.VALID.shape, dtype=torch.bool)
    for n in names:
        m[ER.EDGES[ER.NAMES.index(n)]] = True
    return m


def test_prune_keeps_only_connected_edges():
    kept = ER.prune(mask("emb->2,4.k", "2,4->logits", "1,0->logits", "emb->0,3.q"))
    assert sorted(ER.NAMES[i] for i, e in enumerate(ER.EDGES) if kept[e]) == ["2,4->logits", "emb->2,4.k"]
    # a chain through an MLP stays; cutting its input drops the whole chain
    assert int(ER.prune(mask("emb->mlp0", "mlp0->1,1.v", "1,1->logits")).sum()) == 3
    assert int(ER.prune(mask("mlp0->1,1.v", "1,1->logits")).sum()) == 0
    assert int(ER.prune(ER.VALID.clone()).sum()) == 927


def test_topn_ties_and_sizes():
    s = torch.zeros(ER.VALID.shape)
    s[ER.EDGES[ER.NAMES.index("emb->logits")]] = -5.0  # |score| counts
    s[ER.EDGES[ER.NAMES.index("2,4->logits")]] = 1.0
    assert int(ER.topn(s, 1).sum()) == 1 and bool(ER.topn(s, 1)[ER.EDGES[ER.NAMES.index("emb->logits")]])
    assert int(ER.topn(s, 0).sum()) == 0  # the empty circuit
    assert int(ER.topn(s, 927).sum()) == 927
    tie = torch.ones(ER.VALID.shape)
    assert torch.equal(ER.topn(tie, 5), ER.topn(tie, 5))  # ties break in edge order, the same every time


def test_faithfulness_and_its_denominator():
    assert ER.faith(3.0, 5.0, 1.0, 1.0) == 0.5
    assert math.isnan(ER.faith(3.0, 2.0, 1.0, 1.0))  # a gap of exactly min_gap: no gap
    assert math.isnan(ER.faith(3.0, 1.0 + 1e-9, 1.0, 1.0))  # a denominator near 0
    assert math.isnan(ER.faith(3.0, 0.0, 1.0, 1.0))  # a negative gap
    assert math.isnan(ER.faith(NAN, 5.0, 1.0, 1.0))
    assert ER.faith(1.0, 5.0, 1.0, 1.0) == 0.0  # the empty circuit: m = b'
    assert ER.faith(6.0, 5.0, 1.0, 1.0) > 1  # past 1 is reported as read


def test_smallest_circuit():
    assert ER.smallest([(10, 0.2), (50, 0.86), (100, 0.84), (927, 1.0)], 0.85) == 50
    assert ER.smallest([(10, 0.85)], 0.85) == 10  # a tie at the bar passes
    assert ER.smallest([(10, NAN), (927, NAN)], 0.85) is None
    assert ER.smallest([], 0.85) is None


def test_overlap_and_parts():
    a, b = mask("2,4->logits", "emb->2,4.q"), mask("2,4->logits")
    assert ER.iou(a, b) == 0.5 and math.isnan(ER.iou(mask(), mask()))
    assert ER.has_parts(a, "2,4") is True and ER.has_parts(a, "2,3+2,4") is False and ER.has_parts(a, None) is None
    assert ER.has_parts(mask(), "2,4") is False  # an empty circuit has no part


def test_gain_path_numbers():
    sc = {"2,4->logits": 6.0, "2,3->logits": 2.0, "mlp3->logits": 6.0, "emb->2,4.q": 7.0}
    assert ER.top_edge(sc) == "emb->2,4.q"
    assert ER.top_edge({"2,4->logits": 6.0, "mlp3->logits": 6.0}) is None  # a tie names none
    assert ER.direct_share(sc, "2,4", 12.0) == 0.5 and ER.direct_share(sc, "2,3+2,4", 16.0) == 0.5
    assert math.isnan(ER.direct_share(sc, None, 12.0)) and math.isnan(ER.direct_share(sc, "2,4", 0.0))
    assert math.isnan(ER.direct_share(sc, "1,1", 12.0))  # an edge not read
    assert ER.edge_nominees(sc, 3) == ["2,4", "mlp3"]  # the embedding is never a part; ties in the scores' order
    assert "emb" not in ER.edge_nominees({"emb->logits": 9.0, "1,1->logits": 1.0}, 2)
    assert ER.out_sums({"2,4->logits": 1.0, "2,4->3,0.q": 0.5})["2,4"] == 1.5
    assert ER.sum_misses({"2,4": 0.4}, {"2,4": 0.48}, 0.1) == [] and ER.sum_misses({"2,4": 0.4}, {"2,4": 0.51}, 0.1)
    assert ER.sum_misses({}, {"2,4": 0.5}, 0.1)  # a part not summed misses


def test_the_band_and_its_readings():
    b = ER.band([0.71, 0.65, 0.8], [0.1, 0.153, 0.0])
    assert b["separated"] and b["reuse_at_least"] == 0.65 and b["not_reuse_at_most"] == 0.16
    assert [ER.band_reading(x, b) for x in (0.65, 0.9, 0.16, -0.2, 0.4, NAN)] == ["like copy", "like copy", "like lookup", "like lookup", "between", "unreadable: no gap"]
    near = ER.band([0.305], [0.296])
    assert not near["separated"] and near["direction"] == "overlap"  # 0.30 against 0.30
    assert ER.band([0.1], [0.8])["direction"] == "lookup above copy"
    assert ER.band([0.9, NAN], [0.1])["direction"] == "unreadable" and ER.band([], [0.1])["direction"] == "unreadable"
    assert ER.band_reading(0.9, ER.band([0.3], [0.5])) == "no band"


def test_the_tree():
    good = ER.band([0.8], [0.1])
    assert ER.outcome(["2,4->logits"] * 3, "2,4->logits", good)[0]
    assert not ER.outcome(["2,4->logits", "mlp3->logits", "2,4->logits"], "2,4->logits", good)[0]
    assert not ER.outcome([None] * 3, "2,4->logits", good)[0]  # a tie at the top
    assert not ER.outcome([], "2,4->logits", good)[0]
    assert "REVERSED" in ER.outcome(["2,4->logits"], "2,4->logits", ER.band([0.1], [0.8]))[1]
    assert not ER.outcome(["2,4->logits"], "2,4->logits", ER.band([0.5], [0.5]))[0]


def test_the_do_nothing_state():
    """W0 as the state: G = 0 and every edge scores 0, so the top edge is a tie and the tree stops at the gain path
    before cross-faithfulness, which reads W0's own circuit (>= 0.85 by construction) for a state that changed nothing."""
    zero = ER.as_dict(torch.zeros(ER.VALID.shape))
    assert ER.top_edge(zero) is None
    assert math.isnan(ER.direct_share(zero, "2,4", 0.0)) and math.isnan(LR.frac(0.0, 0.0))
    assert not ER.outcome([ER.top_edge(zero)] * 3, "2,4->logits", ER.band([0.9] * 3, [0.9] * 3))[0]
    assert ER.edge_nominees(zero, 3) == []  # all tied at 0: the first three edges in order are the embedding's


def test_nominees_take_the_edges_when_a_record_holds_them():
    rec = {"carry": {"mlp3": {"lo": 0.5}}, "write": {"head": {"2,4": {"lo": 0.3}}, "matrix": {"h.2.mlp.c_fc": {"lo": 0.2}}}}
    assert LR.nominees(rec, 3) == ["mlp3", "2,4", "mlp2"]  # E97's records: unchanged
    assert LR.nominees({**rec, "edge_nominees": ["1,1", "2,4"]}, 3) == ["mlp3", "2,4", "mlp2", "1,1"]
