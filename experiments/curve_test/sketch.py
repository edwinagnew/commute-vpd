"""The card's sketch: plot.py's gap figure with made-up numbers for seeds 2, 3, drawn before any run, one panel per rival:
A, every pair-seed above the bar; B, case and repeat within it (E184's seeds 0, 1, open circles, are the real ones).

    ~/commute-agents/fleet/cpu-run python experiments/curve_test/sketch.py
"""

from __future__ import annotations

from typing import Any

from commute.jobs import load_experiment

P: Any = load_experiment("curve_test", "plot")
BAR = 0.029


def made_up(gaps: list[float]) -> dict[str, Any]:
    return {f"{a}/{b} s{s}": {"read": True, "gap": g + 0.01 * (s - 2)} for (a, b), g in zip(P.RU.PAIRS, gaps, strict=True) for s in (2, 3)}


def main() -> None:
    panels = [("rival A (made up): all ten above the bar", made_up([0.37, 0.46, 0.08, 0.06, 0.2]), True),
              ("rival B (made up): case, repeat within it", made_up([0.37, 0.46, 0.005, -0.01, 0.2]), True)]  # fmt: skip
    P.draw_gaps(panels, BAR, P.e184_gaps(), P.FIGURES / "sketch.png", "made-up numbers for seeds 2, 3, drawn before the run")


if __name__ == "__main__":
    main()
