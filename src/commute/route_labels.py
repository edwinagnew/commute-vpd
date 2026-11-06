"""E12's filed route-split lines as one labelling rule, moved 2026-11-06 for E123 (experiments/case_label_both) out of
E119's script (experiments/teacher_route_split/run.py `label`, `e12_control`), which stays as it ran with its own copy.
The lines are E119's config: re-switching r >= r_hi (0.6) with e_gate <= masked_lo (0.25); masked-off e_gate >=
masked_hi (0.5) with r <= r_lo (0.35); else neither (E12's P4 lines and E12 Amendment 1's P3 lines, E42's).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commute.local_faith import MIN_GAIN


@dataclass(frozen=True)
class Lines:
    r_hi: float = 0.6
    r_lo: float = 0.35
    masked_hi: float = 0.5
    masked_lo: float = 0.25
    min_ratio: float = 0.5  # E12 Amendment 1's: a state's gain over its teacher's gain


def label(r: float, e_gate: float, gain: float, teacher_gain: float, lines: Lines) -> str:
    """Unreadable: a share not finite, gain under MIN_GAIN, or gain / teacher's gain under min_ratio (a zero or negative
    teacher gain is unreadable); else re-switching, masked-off or neither. Boundaries go to the end labels."""
    ratio = gain / teacher_gain if teacher_gain > 0 else math.nan
    if not (math.isfinite(r) and math.isfinite(e_gate) and gain >= MIN_GAIN and ratio >= lines.min_ratio):
        return "unreadable"
    if r >= lines.r_hi and e_gate <= lines.masked_lo:
        return "re-switching"
    if e_gate >= lines.masked_hi and r <= lines.r_lo:
        return "masked-off"
    return "neither"


def e12_control(path: Path, lines: Lines) -> dict[str, dict[str, Any]]:
    """E12 Amendment 1's batch-order-0 students (state a) from its saved file, labelled by `label`, with their class:
    the rule's positive control (E119 read 4 copy tables re-switching, 4 maps masked-off)."""
    a1 = json.loads(path.read_text())
    out = {}
    for n, s in a1["students"].items():
        if "_o" not in n:
            sh = s["a"]["shares"]
            out[n] = {"class": "copy" if n.startswith("ident") else "map", "r": sh["r"], "e_gate": sh["e_gate"],
                      "label": label(sh["r"], sh["e_gate"], s["a"]["gain"], s["teacher_gain"], lines)}
    return out
