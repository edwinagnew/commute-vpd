"""The decomposition side's reads of a saved full-weight update delta-W on a trigger task, for E125
(experiments/fresh_reuse_pair). Moved, with the same calls, from experiments/teacher_route_split/run.py (E119's
`Job.read`) and experiments/recruitment_test/amend2.py (E38 Amendment 2's removal grid, the 'trigger' rule only); both
files stay as they ran. The route label rule and its positive control are `commute.route_labels` (moved for E123).

1. The route split (`local_faith.route_split`) of a state: `read_split`.
2. E38 Amendment 2: S = the alive subcomponents whose mean trigger gate over the training windows rises by more than
   `thr` (E18's THR 0.3), removed at the trigger only from the state and from W0; D(X) = rho_ft(X) - rho_0(X) against
   `n_random` size-matched random sets (`recruit.random_like`), classed by `recruitment.d_seed_class`.

A per-window target uses data.LAST with the windows' last column; every read passes ci_len so that no forward and no
Gamma read sees that column.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state
from commute.local_faith import route_split, trigger_mean_gates


def read_split(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], comps: Any, windows: Tensor, half: int, token: int | Tensor,
               base: dict[str, Any], comps0: dict[str, tuple[Tensor, Tensor]], ci_len: int | None) -> dict[str, Any]:  # fmt: skip
    """E119's `Job.read`: the route split at target W0 + dW with components `comps` (None: released)."""
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}, components=comps):
        return route_split(model, windows, half, token, base, comps0, ci_len=ci_len)


def switched_set(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], train: Tensor, half: int, m0: dict[str, Tensor],
                 alive: dict[str, Tensor], thr: float) -> tuple[R.Sel, dict[str, Tensor]]:  # fmt: skip
    """E38's S and the shift s it is cut from: s = the state's mean trigger gates over `train` (tokens without a target
    column) minus W0's m0, 0 on dead subcomponents (`regating.shift_of`); S = alive with s > thr (`recruit.switched`)."""
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}):
        m1 = trigger_mean_gates(model, train, half)
    shift = R.mean_change(m1, m0, alive)
    return R.switched(shift, thr), shift


def removal_d(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], S: R.Sel,
              alive: dict[str, Tensor], windows: Tensor, half: int, token: int, n_random: int, min_gain: float,
              ci_len: int | None) -> dict[str, Any]:  # fmt: skip
    """E38 Amendment 2's stage-1 read of one state, trigger-only rule: G = T1 - T0; for X in S and n_random random sets,
    rho_ft(X) = (T1 - lo(W1 without X)) / G, rho_0(X) = (T0 - lo(W0 without X)) / G, D = rho_ft - rho_0 (nan when
    G < min_gain); `empty` = lo(W1 without nothing) - T1 (E38 A2's C3, expected 0); the class by d_seed_class, or
    "unscored" when G < min_gain or S is empty (no set to remove: the rule cannot read it, as E38's lowercase)."""
    W1 = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
    refs, T = {}, {}
    for lab, W in (("0", W0), ("ft", W1)):
        with model_state(model, target=W):
            refs[lab] = RT.window_logits(model, windows, ci_len=ci_len)
            T[lab] = RT.readout(model, windows, half, token, refs[lab], ci_len=ci_len)["lo"]
    G = T["ft"] - T["0"]
    g = G if G >= min_gain else 0.0

    def lo(W: dict[str, Tensor], X: R.Sel, lab: str) -> float:
        return RT.removed_at(model, W, comps, X, windows, half, token, refs[lab], ci_len=ci_len)["lo"]

    sets = [S, *(R.random_like(S, alive, i) for i in range(n_random))]
    rho = {lab: [RT.share(T[lab], lo(W, X, lab), g) for X in sets] for lab, W in (("0", W0), ("ft", W1))}
    D = [a - b for a, b in zip(rho["ft"], rho["0"], strict=True)]
    rec = {"T0": T["0"], "T1": T["ft"], "gain": G, "scorable": G >= min_gain, "n_S": R.size(S),
           "empty": lo(W1, {k: torch.zeros(0, dtype=torch.long) for k in alive}, "ft") - T["ft"],
           "rho_ft": {"S": rho["ft"][0], "random": rho["ft"][1:]}, "rho_0": {"S": rho["0"][0], "random": rho["0"][1:]},
           "D": {"S": D[0], "random": D[1:]}}
    rec["class"] = "unscored: S is empty" if not R.size(S) else RT.d_seed_class(D[0], D[1:]) if rec["scorable"] else RT.UNSCORED_GAIN
    return rec
