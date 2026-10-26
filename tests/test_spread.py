import torch

from commute import spread as S

ALIVE = {"m.b": torch.tensor([True, False, True]), "m.a": torch.tensor([False, True])}


def _gates(B: int = 2, T: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {k: torch.rand(B, T, v.numel(), generator=g) for k, v in ALIVE.items()}


def test_flat_round_trip_keeps_dead_entries() -> None:
    lay = S.layout(ALIVE, "cpu")
    assert lay.keys == ["m.a", "m.b"] and lay.D == 3
    g = _gates()
    x = S.to_flat(g, lay)
    assert x.shape == (6, 3)
    assert torch.allclose(x[:, 0], g["m.a"][:, :, 1].flatten())  # sorted keys: m.a first
    back = S.with_alive(g, torch.zeros_like(x), lay)
    assert torch.all(back["m.a"][:, :, 1] == 0) and torch.equal(back["m.a"][:, :, 0], g["m.a"][:, :, 0])
    assert torch.equal(back["m.b"][:, :, 1], g["m.b"][:, :, 1]) and torch.all(back["m.b"][:, :, [0, 2]] == 0)
    assert torch.equal(S.with_alive(g, x, lay)["m.b"], g["m.b"])


def test_group_change_and_token_table() -> None:
    D = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    assert torch.allclose(S.group_change(D, torch.tensor([0, 0, 2, 2]), 3), torch.tensor([[2.0], [0.0], [6.0]]))
    t = S.fit_tokens(torch.tensor([4, 4, 4, 1]), D, vocab=6, min_count=2)
    assert t.n_tokens == 1
    assert torch.allclose(t.change[t.row_of[torch.tensor([4, 1, 0])]], torch.tensor([[3.0], [4.0], [4.0]]))


def test_nearest_change_and_clusters() -> None:
    X = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    D = torch.tensor([[1.0], [3.0], [10.0], [20.0]])
    idx = S.nearest(X, torch.tensor([[0.0, 0.05], [1.0, 0.9]]), 2)
    assert torch.allclose(S.nearest_change(D, idx), torch.tensor([[2.0], [15.0]]))
    c = S.fit_clusters(X, D, 2, 0)
    lab = S.assign(torch.tensor([[0.05, 0.0], [0.95, 1.0]]), c.cent)
    assert sorted(c.change[lab].flatten().tolist()) == [2.0, 15.0]
