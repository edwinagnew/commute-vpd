"""The card's sketch: plot.py's contrast figure with made-up numbers, drawn before any run, one panel per rival:
A, level B passes (it reads the replaced item low); B, level B fails on the replaced item (it reads it like the rescaled
one). Level A is drawn at its analytic values in both (0.80 rescaled, 0 replaced).

    ~/commute-agents/fleet/cpu-run python experiments/rule_retention/sketch.py
"""

from __future__ import annotations

import json
from typing import Any

from commute.jobs import load_experiment

P: Any = load_experiment("rule_retention", "plot")


def reading(a: list[float] | None, b: list[float] | None) -> dict[str, Any]:
    """Made-up readings: per level the retention of five orders' smallest sets, or None for do-nothing (no change)."""
    def one(x: list[float] | None) -> dict[str, Any]:
        return {"contrast": {"reading": "no change"} if x is None else {"reading": "read", "retention": x}}

    return {"A": one(a), "B": one(b)}


def made_up(b_replaced: float) -> dict[str, dict[str, Any]]:
    its = {"do-nothing": reading(None, None), "rescaled": reading([0.8] * 5, [0.97] * 5)}
    return its | {f"replaced s{s}": reading([0.0] * 5, [b_replaced + 0.01 * s] * 5) for s in range(4)}


def main() -> None:
    th = json.loads((P.RESULTS / "predictions.json").read_text())["thresholds"]
    panels = [("rival A (made up): level B reads replaced low", made_up(0.12)),
              ("rival B (made up): level B reads replaced like rescaled", made_up(0.95))]  # fmt: skip
    P.draw_contrast(panels, th, P.FIGURES / "sketch.png", " (made up, before the run)")


if __name__ == "__main__":
    main()
