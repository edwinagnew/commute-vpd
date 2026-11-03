"""E77's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

NAN = math.nan
TOL, ID_TOL = 1e-3, 1e-3  # same path, same seed: E74's K0 against E68 and E72's reproduce read a gap under 1e-8; E74's own bars
OWN_AMP, STEEP, DISP = 0.5, 10.0, 0.02  # the filed thresholds
MATCH, FLOOR, SES = 0.2, 1e-5, 2.0  # E74's MATCH and FLOOR; a no-call band is this many standard errors of the first stage's control


def close(got: float, want: float, tol: float = TOL) -> bool:
    """Within `tol` of a stored number, relatively. A NaN or a zero stored number fails: neither is a read to reproduce."""
    return want != 0 and abs(got / want - 1) <= tol


def jack(a: list[float], b: list[float], power: float = 1.0) -> float:
    """Jackknife relative standard error over rows of (Σa / Σb)^power. NaN when it cannot be read: under 2 rows, unequal
    lengths, a NaN, a non-positive sum or leave-one-out sum."""
    n, A, B = len(a), sum(a), sum(b)
    if n < 2 or n != len(b) or math.isnan(A + B) or A <= 0 or B <= 0:
        return math.nan
    if any(A - x <= 0 or B - y <= 0 for x, y in zip(a, b, strict=True)):
        return math.nan
    loo = [((A - x) / (B - y)) ** power for x, y in zip(a, b, strict=True)]
    mid = sum(loo) / n
    return ((n - 1) / n * sum((v - mid) ** 2 for v in loo)) ** 0.5 / (A / B) ** power


def worst(ws: list[float]) -> float:
    return math.nan if not ws or any(math.isnan(w) for w in ws) else max(ws)


def band(threshold: float, w: float) -> list[float]:
    return [threshold * (1 - SES * w), threshold * (1 + SES * w)]


def bands(w_amp: float, w_tau: float, w_disp: float) -> dict[str, list[float]]:
    """No-call bands round the filed thresholds from the first stage's relative errors. The steepness rule is a ratio of two
    transmissions, hence the root of 2."""
    return {"own": band(OWN_AMP, w_amp), "steep": band(STEEP, 2**0.5 * w_tau), "disp": band(DISP, w_disp)}


def side(x: float, b: list[float]) -> str:
    """Against a band, edges inside it; a NaN either way is unread."""
    if math.isnan(x) or math.isnan(b[0]) or math.isnan(b[1]):
        return "unread"
    return "below" if x < b[0] else "above" if x > b[1] else "inside"


def own_output(amp: float, k_clean: float, target: float, b: list[float]) -> str:
    """Read 1, one update and block: the logit amplitude passed, once the change's clean KL matches δW's at that block."""
    if not target > 0 or math.isnan(k_clean) or abs(k_clean / target - 1) > MATCH:
        return "unmatched"
    return {"above": "passes", "below": "fails", "inside": "no call", "unread": "unread"}[side(amp, b)]


def own_overall(labels: list[str]) -> str:
    """A single failure is a failure, and the job stops on it. Otherwise all must read."""
    if "fails" in labels:
        return "fails"
    if not labels or "unread" in labels or "unmatched" in labels:
        return "unread"
    return "passes" if all(x == "passes" for x in labels) else "no call"


def ratio(tau_all: float, tau_part: float) -> float:
    """Transmission with part of the sites masked over transmission with all masked. Nothing either way is 1, nothing at
    all-masked alone is infinite."""
    if math.isnan(tau_all) or math.isnan(tau_part):
        return math.nan
    if tau_all <= 0:
        return 1.0 if tau_part <= 0 else math.inf
    return tau_part / tau_all


def steep(tau_all: float, tau_quarter: float, b: list[float]) -> str:
    """Read 2, one update and mode: k/196 = 1 against 1/4."""
    return {"above": "steep", "below": "shallow", "inside": "no call", "unread": "unread"}[side(ratio(tau_all, tau_quarter), b)]


def steep_overall(labels: list[str]) -> str:
    if not labels or "unread" in labels:
        return "unread"
    return labels[0] if len(set(labels)) == 1 else "mixed"


def displaced(d: float, r: float, b: list[float]) -> str:
    """Read 3, one update: KL(update off ‖ update on) under fine-tuned masks over KL(W0 ‖ FT), both over the same rows."""
    if math.isnan(r) or r < FLOOR:
        return "unread"
    return {"above": "large", "below": "small", "inside": "no call", "unread": "unread"}[side(d / r, b)]


def displaced_overall(labels: list[str]) -> str:
    """The lean is on all three, so one large one refutes it even with another unread."""
    if "large" in labels:
        return "large"
    if not labels or "unread" in labels:
        return "unread"
    return "small" if all(x == "small" for x in labels) else "no call"


def bands_from(fixed: dict[str, list[float]], stoch: dict[str, list[float]], disp: dict[str, list[float]]) -> dict[str, Any]:
    """The bands from the first stage's per-row sums: the all-masked reads on u1, and its displacement."""
    w_amp = worst([jack(a["bb"], a["aa"], 0.5) for a in (fixed, stoch)])
    w_tau = worst([jack(a["K_masked"], a["K_clean"]) for a in (fixed, stoch)])
    w_disp = jack(disp["D"], disp["R"])
    return bands(w_amp, w_tau, w_disp) | {"w": {"amp": w_amp, "tau": w_tau, "disp": w_disp}}


def score(read: Callable[[str, str], dict[str, Any] | None], points: tuple[str, ...], b: dict[str, Any]) -> dict[str, Any]:
    """Every read's label from the files `read(update, key)` gives; a read not yet there is unread."""
    own: dict[str, Any] = {}
    curve: dict[str, Any] = {}
    disp: dict[str, Any] = {}
    for name in ("u1", "u3"):
        r = read(name, "own_output") or {}
        for block in ("0", "7"):
            s = r.get("summary", {}).get(f"own_{block}", {})
            target = r.get("fit", {}).get(block, {}).get("K_clean_target", NAN)
            own[f"{name}_{block}"] = s | {"label": own_output(s.get("amp", NAN), s.get("K_clean", NAN), target, b["own"])}
        for mode in ("fixed", "stochastic"):
            pts = {x: (read(name, f"route_{mode}_{x}") or {}).get("summary", {}).get("whole", {}) for x in points}
            t1, tq = pts["k196"].get("tau", NAN), pts["k49"].get("tau", NAN)
            curve[f"{name}_{mode}"] = {"points": pts, "ratio": ratio(t1, tq), "label": steep(t1, tq, b["steep"])}
    for name in ("u1", "u2", "u3"):
        rd = read(name, "displacement") or {"T": 1, "sums": {"D": [NAN], "R": [NAN]}}
        n = len(rd["sums"]["D"]) * rd["T"]
        d, r2 = sum(rd["sums"]["D"]) / n, sum(rd["sums"]["R"]) / n
        disp[name] = {"share": d / r2 if r2 > 0 else NAN, "label": displaced(d, r2, b["disp"])}
    return {"bands": b, "own_output": {"each": own, "overall": own_overall([v["label"] for v in own.values()])},
            "routing": {"each": curve, "overall": steep_overall([v["label"] for v in curve.values()])},
            "displacement": {"each": disp, "overall": displaced_overall([v["label"] for v in disp.values()])}}
