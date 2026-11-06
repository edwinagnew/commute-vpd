"""E123 (case_label_both): the one config and the pure rules, as filed in results/predictions.json (its `config` must
equal asdict(CFG)). Plan: reports/plans/2026-11-06_plan_e123-case-label-both-readings.md.

Every label here is a filed decomposition-side line, reused unchanged: E12's route lines (`commute.route_labels`, E119's
config), E38 Amendment 2's P5 seed rule (`recruitment.d_seed_class`) and its two-seed task rule (the seeds' common
class, an unscored seed first, else 'seeds disagree'; `recruitment.common_classes`), E112's learn bar for a copy. None
of them is the reuse definition's verdict (decisions.md rows 81 and 87), which E112 gave and which stays as it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from commute.local_faith import MIN_GAIN
from commute.recruitment import UNSCORED_GAIN, d_seed_class
from commute.route_labels import Lines

ARMS = ("real", "twin")
NO_S, D_NAN = "unscored: |S| = 0", "unscored: D(S) not finite"
SIDE = {"re-switching": "reuse-like", "through": "reuse-like", "masked-off": "new-like", "not": "new-like"}


@dataclass(frozen=True)
class Cfg:
    states: tuple[str, ...] = ("real_s0", "real_s1", "twin_s0", "twin_s1")
    half: int = 20  # E112: word at 19, trigger at 20, label read at 20
    ci_len: int = 41  # the model and Gamma see 41 tokens; the label is a 42nd column (data.LAST)
    n_read: int = 256  # every decomposition read: the first n_read of E112's 1,024 fit windows
    n_eval: int = 64  # the student's per-step held-out log (first n_eval read windows); not scored
    student_lr: float = 1e-5  # E34's arm A, as E119's config
    steps: int = 1000
    batch_a: int = 32
    learn_bar: float = 0.9  # E112's: a copy keeps the behaviour if its fit-window accuracy is >= this
    zero_tol: float = 1e-6  # E119's: do-nothing e_gate = -r, least-squares r = do-nothing r
    thr: float = 0.3  # E38's recruited set: mean trigger gate up by > 0.3 over the training windows
    n_random: int = 20  # E38 A2's random same-size sets (recruit.random_like seeds 0..19)
    min_gain: float = 1.0
    acc_tol: float = 0.000625  # E114's: one window of 1,600
    g_tol: float = 0.02  # E114's: a state's gain against E112's and E114's, nats
    path_tol: float = 1e-3  # E38's: route_cells' T against the removal path's lo at W0
    hook_tol: float = 1e-3  # E38 A2 C1
    pre_tol: float = 1e-4  # E38 A2 C2
    empty_tol: float = 1e-5  # E38 A2 C3
    busy_kl_min: float = 1.0  # E38 A2 C4, read where |busy| >= busy_min_n
    busy_min_n: int = 100
    budget_s: float = 1500.0  # exit 3 before a student that might end past this (debug QoS: 30 min)
    student_s: float = 170.0  # E119's arm-A students on an L40, 109-135 s, plus a 256-window read
    task_sha256: str = "80cdeee4da1c230f53f87acc60058a3a036154347c042185d9fdaa2f608aabfd"  # E114's filed sha256
    lines: Lines = field(default_factory=Lines)

    def __post_init__(self) -> None:
        assert self.ci_len == 2 * self.half + 1 and 0 < self.n_eval <= self.n_read <= 1024
        assert (self.thr, self.n_random, self.min_gain) == (0.3, 20, MIN_GAIN), "E38's sizes"
        assert {s.split("_")[0] for s in self.states} == set(ARMS) and self.student_s < self.budget_s


CFG = Cfg()


def arm_of(state: str) -> str:
    return state.split("_")[0]


def unread(x: str) -> bool:
    return x.startswith("unscored") or "unreadable" in x


def copy_label(acc: float, label: str, bar: float) -> str:
    """The student's route label if it keeps the behaviour (fit-window accuracy >= bar), else 'copy unreadable'."""
    return label if acc >= bar else "copy unreadable"


def removal_class(n_S: int, gain: float, D_S: float, D_random: list[float]) -> str:
    """E38 A2's P5 seed rule, trigger only; unscored with no recruited set, a gain under 1 nat, or a D that is not finite."""
    if n_S == 0:
        return NO_S
    if not gain >= MIN_GAIN:
        return UNSCORED_GAIN
    if not (math.isfinite(D_S) and all(math.isfinite(d) for d in D_random)):
        return D_NAN
    return d_seed_class(D_S, D_random)


def arm_class(labels: list[str]) -> str:
    """E38 A2's task rule on one arm's two seeds: an unreadable or unscored seed's label first, else the common label,
    else 'seeds disagree'."""
    if not labels:
        return "unreadable: no state"
    bad = sorted(x for x in labels if unread(x))
    if bad:
        return bad[0]
    return labels[0] if len(set(labels)) == 1 else "seeds disagree"


def side(cls: str) -> str:
    """reuse-like (re-switching, through), new-like (masked-off, not), unreadable, or none (neither, unclear, disagree)."""
    return "unreadable" if unread(cls) else SIDE.get(cls, "none")


def joint(real: str, twin: str) -> str:
    """The plan's outcome table for one measure, from the two arms' sides; the definition reads both arms 'not through C'."""
    if "unreadable" in (real, twin):
        return "unreadable"
    if real == "reuse-like":
        return "both reuse-like" if twin == "reuse-like" else "separates as built"
    if twin == "reuse-like":
        return "reversed"
    return "real new-like" if real == "new-like" else "no label"


def score(labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Per measure {state: label}: the arms' classes, their sides and the joint outcome."""
    out = {}
    for m, by_state in labels.items():
        arms = {a: arm_class([x for n, x in sorted(by_state.items()) if arm_of(n) == a]) for a in ARMS}
        sides = {a: side(c) for a, c in arms.items()}
        out[m] = {"states": by_state, "arms": arms, "sides": sides, "joint": joint(sides["real"], sides["twin"])}
    return out
