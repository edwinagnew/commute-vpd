"""excess.parts: the excess splits exactly into its parts (CPU, made-up numbers)."""

import random
from typing import Any

from commute import excess


def _rec(rng: random.Random) -> dict[str, Any]:
    keys = ("F_all", "F_orig", "F_adv", "F_both", "F_dead", "F_dense", "F_rest", "F_dead_fix", "F_dense_fix", "F_full")
    r: dict[str, Any] = {k: rng.random() for k in keys}
    r["F_half"], r["F_perm"] = [rng.random() for _ in range(3)], [rng.random() for _ in range(3)]
    r["levels"] = {"dense_mean": rng.random(), "dead_mean": rng.random(), "dead_removed": rng.random()}
    return r


def test_parts_sum_to_the_excess() -> None:
    rng = random.Random(0)
    for _ in range(20):
        s, w0 = _rec(rng), _rec(rng)
        w0["F_both"], w0["F_adv"] = w0["F_orig"], 0.0  # W0's identities
        d = excess.parts(s, w0)
        assert abs(d["X"] - (d["O"] + d["F_adv"] + d["I"] + d["R"])) < 1e-12
        assert abs(d["R"] - (d["D"] + d["K"] + d["P"] + d["J"])) < 1e-12


def test_reading_rule() -> None:
    base = {"D_fix": 0.0, "K_fix": 0.0}
    assert excess.reading({**base, "R": 1.0, "J": 0.6, "D": 0.1, "K": 0.1, "P": 0.2}) == "4"
    assert excess.reading({**base, "R": 1.0, "J": 0.1, "D": 0.5, "K": 0.1, "P": 0.3}) == "1"
    assert excess.reading({**base, "R": -0.1, "J": 0.0, "D": 0.0, "K": 0.0, "P": 0.0}).startswith("none")
