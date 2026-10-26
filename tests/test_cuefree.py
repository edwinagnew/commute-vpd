import torch

from commute import cuefree as CF


def _vals(n: int, c: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.rand(n, c, generator=g), "b": torch.rand(n, c + 1, generator=g)}


ALIVE = {"a": torch.tensor([True, True, False]), "b": torch.tensor([True, False, True, True])}


def test_change_zero_on_dead_and_apply_clips() -> None:
    g0, g1 = _vals(5, 3, 0), _vals(5, 3, 1)
    d = CF.change(g1, g0, ALIVE)
    assert torch.all(d["a"][:, 2] == 0) and torch.all(d["b"][:, 1] == 0)
    assert torch.allclose(d["a"][:, :2], (g1["a"] - g0["a"])[:, :2])
    out = CF.apply(g0, {k: torch.full_like(v, 2.0) for k, v in g0.items()})
    assert torch.all(out["a"] == 1.0)


def test_by_group_gives_each_row_its_group_mean() -> None:
    d = {"a": torch.tensor([[1.0], [3.0], [10.0]])}
    out = CF.by_group(d, torch.tensor([0, 0, 1]), torch.tensor([1, 0]))
    assert torch.allclose(out["a"], torch.tensor([[10.0], [2.0]]))


def test_nearest_with_k1_returns_the_nearest_rows_change() -> None:
    g0_tr = {"a": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), "b": torch.zeros(2, 4)}
    g0_he = {"a": torch.tensor([[0.9, 0.8, 0.5]]), "b": torch.zeros(1, 4)}
    d_tr = {"a": torch.tensor([[1.0, 1.0, 0.0], [5.0, 5.0, 0.0]]), "b": torch.zeros(2, 4)}
    out = CF.nearest(g0_tr, g0_he, d_tr, ALIVE, 1, "cpu")
    assert torch.allclose(out["a"], torch.tensor([[5.0, 5.0, 0.0]]))


def test_clusters_split_two_blobs_and_purity() -> None:
    g = torch.Generator().manual_seed(0)
    blob = lambda c: c + 0.01 * torch.randn(20, 3, generator=g)
    tr = {"a": torch.cat([blob(0.0), blob(1.0)]), "b": torch.zeros(40, 4)}
    he = {"a": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), "b": torch.zeros(2, 4)}
    lab_tr, lab_he = CF.clusters(tr, he, ALIVE, 2, 0, "cpu")
    truth = torch.tensor([0] * 20 + [1] * 20)
    assert CF.purity(lab_tr, truth) == 1.0
    assert lab_he[0] == lab_tr[0] and lab_he[1] == lab_tr[-1] and lab_he[0] != lab_he[1]


def test_refit_recovers_a_line_and_falls_back_on_constant_rows() -> None:
    x = torch.linspace(0, 0.5, 10)[:, None].repeat(1, 3)
    x[:, 1] = 0.2  # constant: a = 1, b = mean change
    y = 1.5 * x + 0.1
    a, b = CF.refit({"a": x}, {"a": y}, {"a": torch.tensor([True, True, False])})
    assert torch.allclose(a["a"], torch.tensor([1.5, 1.0, 1.0]), atol=1e-5)
    assert torch.allclose(b["a"], torch.tensor([0.1, 0.2, 0.0]), atol=1e-5)
    assert torch.allclose(CF.reweighted({"a": x}, a, b)["a"][:, 0], y[:, 0].clamp(0, 1), atol=1e-5)
