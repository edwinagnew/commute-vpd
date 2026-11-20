import math

import pytest
import torch

from commute import topk_gate as TG


def test_topk_mask_picks_largest_magnitude_among_alive() -> None:
    acts = torch.tensor([[[0.1, -5.0, 3.0, 9.0, -2.0]]])  # (B=1, L=1, C=5)
    alive = torch.tensor([True, True, True, False, True])  # the largest |a| (9.0) is dead
    m = TG.topk_mask(acts, alive, 2)
    assert m.tolist() == [[[0.0, 1.0, 1.0, 0.0, 0.0]]]
    assert m.dtype == torch.float32 and m.shape == acts.shape


def test_topk_mask_per_position_and_k_bounds() -> None:
    acts = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])  # two positions, opposite orders
    alive = torch.ones(3, dtype=torch.bool)
    m = TG.topk_mask(acts, alive, 1)
    assert m.tolist() == [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]
    assert TG.topk_mask(acts, alive, 3).sum() == 6  # k = every alive: all ones
    with pytest.raises(AssertionError):
        TG.topk_mask(acts, alive, 4)
    with pytest.raises(AssertionError):
        TG.topk_mask(acts, alive, 0)


def test_gamma_bin_and_scaled_k() -> None:
    g = torch.tensor([0.2, 0.5, 0.51, 1.0])
    assert TG.gamma_bin("m", g, g).tolist() == [0.0, 0.0, 1.0, 1.0]
    alive = {"a": torch.tensor([True] * 5 + [False] * 5), "b": torch.ones(3, dtype=torch.bool)}
    k = {"a": 3, "b": 2}
    assert TG.scaled_k(k, 1.0, alive) == k
    assert TG.scaled_k(k, 0.5, alive) == {"a": 2, "b": 1}  # round half to even: 1.5 -> 2, 1.0 -> 1
    assert TG.scaled_k(k, 2.0, alive) == {"a": 5, "b": 3}  # capped at the alive count
    assert TG.scaled_k({"b": 1}, 0.5, alive) == {"b": 1}  # at least 1


def test_shares_zero_change_and_toy() -> None:
    t = torch.tensor([1.0, 3.0], dtype=torch.float64)
    zero = {"T0": t, "T1": t.clone(), "L00_x": t, "L01_x": t.clone()}
    z = TG.shares(zero, ["x"])
    assert z["gain"] == 0 and z["x"]["numerator"] == 0 and math.isnan(z["x"]["share"])
    toy = {"T0": t, "T1": t + 4, "L00_x": t, "L01_x": t + 1}
    s = TG.shares(toy, ["x"])
    assert s["gain"] == 4 and s["x"]["share"] == 0.25
