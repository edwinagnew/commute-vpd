import torch

from commute import adl_ce as CE


def test_zero_direction_is_a_noop() -> None:
    h, src = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    assert torch.equal(CE.swap(CE.unit(torch.zeros(1, 4))[0], src)(h), h)


def test_exclude_pos_drops_the_prediction_of_token_p_plus_2() -> None:
    ids = torch.tensor([[5, 1, 2, 3]])
    logits = torch.zeros(1, 4, 8)
    logits[0, 2, 3] = 10.0  # the logits at token 2 predict token 3: NLL column 2 = target index p + 1 for p = 1
    v = CE.nll(logits, ids)
    assert v.shape == (1, 3) and v[0, 2] < v[0, 0]
    s, m = CE.Sums(1, (2,)), CE.targets(torch.ones_like(ids))
    s.count(m)
    s.add(0, v, m)
    assert abs(CE.ce(s, 0, "exclude_pos", 1) - float(v[0, :2].double().mean())) < 1e-12


def test_zero_gap_has_no_ratio() -> None:
    assert torch.isnan(torch.tensor(CE.incr(1.0, 1.0, 1.0, 0.0)["incr_rel_ce"]))
