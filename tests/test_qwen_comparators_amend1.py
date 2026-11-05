"""E92 Amendment 1's filed rules on their edge cases and on do-nothing, and the logits tap on a tiny random Qwen3. CPU.

    ~/commute-agents/fleet/cpu-run --mem 6G make test FILES="tests/test_qwen_comparators_amend1.py"
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from commute import qwen_comparators as qc
from commute.jobs import load_experiment

A: Any = load_experiment("qwen_comparators", "amend1")
ROWS = torch.arange(8).repeat_interleave(4)  # 8 rows of 4 positions
NOISE = 0.01 * torch.sin(torch.arange(32.0))


def rule(rise: Any, ref: Any = 0.0, dH: Any = 0.0, token: Any = 0.0, off: bool = False, rows: torch.Tensor = ROWS) -> str:
    n = len(rows)
    f = lambda x: x if isinstance(x, torch.Tensor) else torch.full((n,), float(x))
    return A.verdict(f(rise), f(ref), f(dH), f(token), rows, off)


def test_do_nothing_reads_no_rise_whatever_else_is_passed() -> None:
    assert rule(0.0) == "no rise"
    assert rule(0.0, ref=1.0, dH=1.0, token=1.0) == "no rise"


def test_an_empty_pool_and_one_row_are_unread() -> None:
    assert rule(torch.zeros(0), rows=torch.zeros(0, dtype=torch.long)) == "unread"
    assert rule(1.0, ref=1.0, dH=1.0, rows=torch.zeros(32, dtype=torch.long)) == "unread"


def test_a_rise_lost_in_row_noise_is_no_rise() -> None:
    assert rule(torch.tensor([1.0, -1.0] * 16), ref=0.0, dH=1.0, rows=torch.arange(2).repeat(16)) == "no rise"


def test_a_pure_temperature_reads_flattens() -> None:
    assert rule(2.0 + NOISE, ref=2.0 + NOISE, dH=0.5 + NOISE) == "flattens"


def test_flatten_needs_entropy_up_and_on_the_grid() -> None:
    assert rule(2.0 + NOISE, ref=2.0 + NOISE, dH=0.0) == "neither"
    assert rule(2.0 + NOISE, ref=2.0 + NOISE, dH=-0.5 + NOISE) == "neither"
    assert rule(2.0 + NOISE, ref=2.0 + NOISE, dH=0.5 + NOISE, off=True) == "neither"


def test_flatten_band_is_a_factor_of_two_with_ties_out() -> None:
    assert rule(2.0 + NOISE, ref=1.5 + NOISE, dH=0.5 + NOISE) == "flattens"
    assert rule(2.0, ref=1.0, dH=0.5 + NOISE) == "neither"  # exactly half: not clear
    assert rule(2.0, ref=4.0, dH=0.5 + NOISE) == "neither"  # exactly twice: not clear
    assert rule(2.0 + NOISE, ref=6.0 + NOISE, dH=0.5 + NOISE) == "neither"  # entropy rose more than the rise pays for


def test_style_needs_the_tokens_to_predict_what_temperature_leaves() -> None:
    assert rule(2.0 + NOISE, ref=0.0, dH=-0.5 + NOISE, token=1.8 + NOISE) == "style"
    assert rule(2.0 + NOISE, ref=-1.0, dH=-0.5 + NOISE, token=2.5 + NOISE) == "style"
    assert rule(2.0 + NOISE, ref=0.0, dH=-0.5 + NOISE, token=0.0) == "neither"  # a rise no token carries, as facts would be
    assert rule(2.0, ref=0.0, dH=0.0, token=1.0) == "neither"  # exactly half the residual: not clear


def test_a_loss_is_neither() -> None:
    assert rule(-2.0 + NOISE, ref=-2.0 + NOISE, dH=0.5 + NOISE) == "neither"


def test_token_part_is_out_of_sample() -> None:
    row_of = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    target = torch.tensor([5, 6, 5, 7, 5, 8, 5, 9])  # token 5 in every row, the others once
    resid = torch.tensor([1.0, 9.0, 3.0, 9.0, 5.0, 9.0, 7.0, 9.0])
    out = A.token_part(resid, target, row_of)
    assert out.tolist() == pytest.approx([5.0, 0.0, 3.0, 0.0, 5.0, 0.0, 3.0, 0.0])  # rows 0, 2 against 1, 3; unseen tokens predict 0
    assert A.token_part(resid[:2], target[:2], row_of[:2]).tolist() == [0.0, 0.0]  # one row: nothing on the other half


def grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Mean entropy rising in T by one per entry, log P the same per position; the rise from T = 1 is the entry's offset."""
    t = torch.arange(len(A.TEMPS), dtype=torch.float32)
    return (t - A.T1)[None].repeat(3, 1), t[None].repeat(3, 1)


def test_temperature_reference_at_w0_is_zero_at_t1() -> None:
    lp, H = grid()
    ref, T, off = A.temperature_ref(lp, H, float(A.T1))
    assert T == pytest.approx(1.0) and not off and ref.tolist() == [0.0, 0.0, 0.0]


def test_temperature_reference_interpolates_and_flags_the_top_only() -> None:
    lp, H = grid()
    ref, T, off = A.temperature_ref(lp, H, A.T1 + 0.5)
    assert ref.tolist() == pytest.approx([0.5] * 3) and not off and 1.0 < T < 1.05
    ref, T, off = A.temperature_ref(lp, H, 1e6)
    assert off and T == A.TEMPS[-1] and float(ref[0]) == len(A.TEMPS) - 1 - A.T1
    ref, T, off = A.temperature_ref(lp, H, -1.0)
    assert not off and T == A.TEMPS[0] and float(ref[0]) == -A.T1


def row(rise: float | None, se: float | None = 0.01) -> dict[str, Any]:
    return {"n": 10, "rise": rise, "rise_se": se}


def test_beyond_random() -> None:
    assert A.beyond(row(1.0), [row(0.1), row(0.2), row(0.15)]) is True
    assert A.beyond(row(0.2), [row(0.1), row(0.2), row(0.15)]) is False
    assert A.beyond(row(0.6), [row(0.0), row(1.0), row(0.5)]) is False  # the draws' spread counts
    assert A.beyond({"n": 0}, [row(0.1)] * 3) is None
    assert A.beyond(row(1.0, None), [row(0.1)] * 3) is None


def test_agree_needs_every_seed() -> None:
    assert A.agree(["flattens", "flattens"]) == "flattens"
    assert A.agree(["flattens", "style"]) == "no call"
    assert A.agree([]) == "no call" and A.agree([None, None]) == "no call"


def test_trend_edge_cases() -> None:
    late = {"share": 0.185, "share_se": 0.019}
    assert A.trend({"share": 0.5, "share_se": 0.02}, late) == "falls"
    assert A.trend({"share": 0.0, "share_se": 0.02}, late) == "rises"
    assert A.trend({"share": 0.2, "share_se": 0.02}, late) == "flat"
    assert A.trend({"share": None, "share_se": None}, late) == "undefined"
    assert A.trend(late, {"share": None, "share_se": None}) == "undefined"
    tie = 2 * (0.03**2 + 0.04**2) ** 0.5
    assert A.trend({"share": tie, "share_se": 0.03}, {"share": 0.0, "share_se": 0.04}) == "flat"  # strict


def spread() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(4000, 32, generator=g) * torch.tensor([10.0 / (1 + i) for i in range(32)])


@pytest.mark.parametrize("target", [0.05, 0.1, 0.3])  # an isotropic draw reads 1/32 here, and 1/2048 on the real inputs
def test_matched_random_hits_norm_and_variance_share(target: float) -> None:
    X = spread()
    c = (X - X.mean(0)).double()
    lam, U = torch.linalg.eigh(c.T @ c)
    # a vector with exactly this share, between the bottom and the top eigenvector
    a = (target * float(lam.sum()) - float(lam[0])) / float(lam[-1] - lam[0])
    v = (a**0.5 * U[:, -1] + (1 - a) ** 0.5 * U[:, 0]).float() * 7.0
    assert qc.variance_share(X, v) == pytest.approx(target, rel=1e-3)
    draws = A.matched_random(X, v, (1, 2, 3), 0.01)
    for r in draws:
        assert float(r.norm()) == pytest.approx(7.0, rel=1e-4)
        assert qc.variance_share(X, r) == pytest.approx(target, rel=0.01)
    assert abs(float(torch.cosine_similarity(draws[0], draws[1], dim=0))) < 0.999


def test_matched_random_refuses_a_share_it_cannot_reach() -> None:
    X = spread()
    c = (X - X.mean(0)).double()
    _, U = torch.linalg.eigh(c.T @ c)
    v = U[:, 0].float()  # the bottom eigenvector: under any random direction's share at a = 0
    with pytest.raises(A.Stop):
        A.matched_random(X, v, (1,), 0.01)


def test_the_tap_leaves_qc_read_unchanged_and_entropy_is_right() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    model = Qwen3ForCausalLM(cfg).eval()
    rows = qc.rows_of("t", [(list(range(3, 40)), 2)] * 3, 8, 8)
    add = (1, torch.ones(16))
    for a in (None, add):
        plain, full = qc.read(model, rows, a), A.read_full(model, rows, a, grid=True)
        assert torch.equal(plain.logp, full.read.logp) and torch.equal(plain.rank, full.read.rank)
        assert full.grid_logp is not None and full.grid_entropy is not None
        assert torch.equal(full.grid_logp[:, A.T1], full.read.logp)
        assert torch.equal(full.grid_entropy[:, A.T1], full.entropy)
        assert bool((full.grid_entropy.mean(0).diff() > 0).all())  # entropy rises with temperature
        assert full.read.rank.tolist() == plain.rank.tolist()
    assert A.entropy_of(torch.log_softmax(torch.zeros(1, 50), -1)).item() == pytest.approx(torch.log(torch.tensor(50.0)).item(), rel=1e-6)
    assert not model.lm_head._forward_hooks
    w0 = A.read_full(model, rows, grid=True)
    every = torch.ones(len(w0.entropy), dtype=torch.bool)
    assert A.table(w0, w0, w0, every)["verdict"] == "no rise"
    assert A.table(w0, w0, w0, torch.zeros_like(every))["verdict"] == "unread"
    capped = A.ceiling(w0, every)
    assert capped["best_T"] in A.TEMPS and capped["rise"] >= 0.0
