"""commute.kstar's search and summary on a toy additive model, where K* is known: KL(F || F_K) = the summed weight of
the variables outside K, so K* holds the largest weights until the rest weighs at most (1 - alpha) of the total."""

import itertools

import torch

from commute import kstar as K
from commute import rule_retention_fixed as RF

W = torch.tensor([[5.0, 4.0], [3.0, 3.0], [0.5, 0.4], [0.2, 0.3], [0.1, 0.1], [0.05, 0.05]])  # 6 variables x 2 windows
VARS: list[K.Var] = [("block", i, 0) for i in range(len(W))]


def kl(keep: torch.Tensor) -> torch.Tensor:
    return (W * (~keep.bool())[:, None]).sum(0).double()


def test_exhaustive_finds_the_heaviest_set_and_ties() -> None:
    table = K.exhaustive(kl, len(W))
    none = kl(torch.zeros(len(W)))
    best = K.from_table(table, none, torch.arange(2), 0.9)
    # mean weights 4.5, 3, 0.45, 0.25, 0.1, 0.05 (total 8.35, bar 0.835): {0, 1} leaves 0.85, so |K*| = 3, with ties
    # {0, 1, j} for j = 2..5 (rest 0.4, 0.6, 0.75, 0.8), the lowest-KL first
    assert [j for j in range(6) if best[0] >> j & 1] == [0, 1, 2]
    assert len(best) == 4 and all(m.bit_count() == 3 for m in best)
    # every smaller subset misses the bar
    bar = 0.1 * float(none.mean())
    assert all(float(table[m].mean()) > bar for m in range(64) if m.bit_count() < best[0].bit_count())


def test_greedy_and_polish_reach_the_minimum() -> None:
    index = {u: j for j, u in enumerate(VARS)}
    bar = 0.1 * float(kl(torch.zeros(6)).mean())

    def ok(S: set[K.Var]) -> bool:
        keep = torch.zeros(6)
        keep[[index[u] for u in S]] = 1
        return float(kl(keep).mean()) <= bar

    size = min(len(S) for r in range(7) for S in itertools.combinations(VARS, r) if ok(set(S)))
    S, _ = RF.eliminate(lambda T: -float(kl(torch.tensor([u in T for u in VARS], dtype=torch.float)).mean()), VARS[::-1], -bar, 1)
    P, _, capped = K.polish(ok, set(S), VARS, 1000)
    assert ok(P) and len(P) == size and not capped


def test_polish_swaps_two_for_one() -> None:
    ok = lambda S: S in ({VARS[0], VARS[1]}, {VARS[2]}) or (VARS[2] in S)
    P, _, _ = K.polish(ok, {VARS[0], VARS[1]}, VARS, 1000)
    assert P == {VARS[2]}


def test_summary_identity_and_remainder() -> None:
    s = torch.tensor([[1.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.2, 0.0]], dtype=torch.float64)
    out = K.summary([0, 2, 3], s, C={2, 3}, lacked={3}, piece=3)
    assert abs(out["R_all"] - 0.5) < 1e-12 and abs(out["f"] - 2 / 3) < 1e-12 and out["rho_all"] == 0.25
    assert out["remainder_share"] == 1 / 3 and out["lacked_in_K"] == 1
    assert K.summary([0, 1], s, C=set(), lacked=set(), piece=None)["rho_all"] is None
