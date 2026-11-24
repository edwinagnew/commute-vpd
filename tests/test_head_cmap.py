"""The paper-first minimality rule (spec Appendix C, `commute.head_cmap.prune`) on hand-made counts, no model."""

from collections.abc import Callable

from commute.head_cmap import Node, matched_random, prune

A1, A2, A3, B1 = (3, 0, 0), (3, 1, 0), (2, 4, 0), (1, 2, 0)


def additive(w: dict[Node, int]) -> Callable[[frozenset[Node]], int]:
    """F(S) = the summed weights of S's nodes."""
    return lambda S: sum(w[n] for n in S)


def test_contributing_nodes_kept_and_null_dropped() -> None:
    assert prune({"A": [A1, A2, A3]}, additive({A1: 50, A2: 10, A3: 0})) == {"A": [A1, A2]}  # A3: num 0, dropped


def test_one_percent_is_exact() -> None:
    # For v = A2, K = {A1} (its drop is positive); then num = F(Cir minus K) - F(Cir minus (K + v)) = gain and den = 700:
    # 7 / 700 is exactly 1 %, kept (0.01 * 700 in floats is 7.000000000000001); 6 / 700 is dropped
    for gain, kept in ((7, True), (6, False)):
        got = prune({"A": [A1, A2], "B": [B1]}, additive({A1: 100, A2: gain, B1: 700}))
        assert (A2 in got["A"]) is kept and got["B"] == [B1]


def test_singleton_group_has_empty_k_and_zero_den() -> None:
    # |G| = 1: K is empty. With nothing else in Cir, den = F(empty) = 0: kept if num > 0, dropped if num = 0
    assert prune({"A": [A1]}, additive({A1: 3})) == {"A": [A1]}
    assert prune({"A": [A1]}, additive({A1: 0})) == {"A": []}


def test_empty_group_kept_empty() -> None:
    assert prune({"A": [], "B": [B1]}, additive({B1: 5})) == {"A": [], "B": [B1]}


def test_negative_contribution_dropped() -> None:
    assert prune({"A": [A1, A2]}, additive({A1: 5, A2: -2})) == {"A": [A1]}


def test_k_cap_is_ceil_of_30_percent_of_the_others() -> None:
    # 11 nodes, so |G minus v| = 10 and K holds ceil(3) = 3 nodes, not ceil(3.0000000000000004) = 4. v counts only while
    # at least 7 of the 10 others remain, so v is kept with |K| = 3 and would be dropped with |K| = 4.
    G = [(l, h, 0) for l in range(2) for h in range(6)][:11]
    v = G[0]

    def f(S: frozenset[Node]) -> int:
        return 5 if v in S and len(S - {v}) >= 7 else 0

    assert v in prune({"A": G}, f)["A"]


def test_matched_random_keeps_counts_per_position() -> None:
    circuit = [(3, 0, 0), (2, 1, 0), (1, 5, 1)]
    for s in matched_random(circuit, 10, 0, 4, 6):
        assert sorted(r for *_, r in s) == [0, 0, 1] and len(set(s)) == 3
