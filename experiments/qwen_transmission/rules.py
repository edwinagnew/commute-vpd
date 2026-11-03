"""E74's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them."""
from __future__ import annotations

import math

BAND = (0.5, 2.0)  # the control: transmission of a change entering just before the final norm
LOW, HIGH, RATIO, MATCH, FLOOR = 0.1, 0.5, 3.0, 0.2, 1e-5
SHARP, SPECIFIC, NO_GAIN = 0.4, 1.0, 0.2
EARLY = ("0", "7", "14")


def tau(k_masked: float, k_clean: float) -> float:
    """Transmission: masked shift over clean shift. NaN (unread) when the clean shift is under the floor."""
    return k_masked / k_clean if k_clean >= FLOOR else math.nan


def control(tau_dw: float, tau_gauss: float) -> bool:
    """True when both of the control's changes pass. A NaN (unread) fails."""
    return BAND[0] <= tau_dw <= BAND[1] and BAND[0] <= tau_gauss <= BAND[1]


def where_lost(tau_0: float, tau_cut_0: float) -> str:
    """Block 0's change under every mask, and with every site after block 0 unmasked."""
    if math.isnan(tau_0) or math.isnan(tau_cut_0):
        return "unread"
    if tau_0 > LOW:
        return "not lost"
    return "downstream" if tau_cut_0 >= HIGH else "elsewhere"


def ratio(tau_gauss: float, tau_dw: float) -> float:
    """Gaussian's transmission over δW's; both zero is 1, δW's alone zero is infinite, an unread side is NaN."""
    if math.isnan(tau_gauss) or math.isnan(tau_dw):
        return math.nan
    if tau_dw == 0:
        return 1.0 if tau_gauss == 0 else math.inf
    return tau_gauss / tau_dw


def against_gaussian(arms: dict[str, tuple[float, float, float, float]]) -> str:
    """Per block (tau_dw, tau_gauss, k_clean_dw, k_clean_gauss), on the early blocks. An unmatched or unread block is "mixed"."""
    rs = []
    for layer in EARLY:
        t_d, t_g, k_d, k_g = arms[layer]
        r = ratio(t_g, t_d)
        if math.isnan(r) or abs(k_g / k_d - 1) > MATCH:
            return "mixed"
        rs.append(r)
    if sum(r > RATIO for r in rs) >= 2:
        return "needs switched-off subcomponents"
    if sum(r < 1 / RATIO for r in rs) >= 2:
        return "passes better than random"
    return "generic" if all(1 / RATIO <= r <= RATIO for r in rs) else "mixed"


def sharpness(k_by_temp: dict[str, float], template: bool) -> dict[str, float | str]:
    """How much of KL(FT ‖ W0) one scalar temperature removes. Gold updates by the minimum, the template by the share removed."""
    k1, k_min = k_by_temp["1"], min(k_by_temp.values())
    share = 1 - k_min / k1 if k1 > 0 else 0.0
    if template:
        label = "no gain" if share < NO_GAIN else "gain"
    else:
        label = "mostly one scalar" if k_min <= SHARP else "specific" if k_min >= SPECIFIC else "between"
    return {"K_1": k1, "K_min": k_min, "share_removed": share, "label": label}
