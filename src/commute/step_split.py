"""The step split's read-outs along one fine-tune path, shared by experiments that run `split.StepSplit`.

Added 2026-10-22 (step_split_small, E25). E19's `snapshot` (experiments/step_split/run.py, which stays as it ran) did the
same sums inline; this keeps the energy shares and the piece sizes and drops its outside-both bookkeeping. Also the
carried-vs-fixed identity at the first step.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from commute import split


def numbers(carry: split.StepSplit, fixed: split.StepSplit, uv0: dict[str, tuple[Tensor, Tensor]],
            sigma: dict[str, Tensor]) -> dict[str, float]:
    """At one step of the run: Delta's change as a share of the update's energy, plain (as harness's
    `delta_change_frac`) and input-weighted, for the carried bases, the fixed bases and the one-shot scaled adjoint split
    of the same delta-W (`1 - R`); the largest relative change of any write or read vector for both; and the share of
    Delta's weighted change that is the accumulated second-order term. `uv0`: W0's alive (U, V), theory layout, CPU."""
    dW = {k: (carry.W[k] - carry.W0[k]).double().cpu() for k in carry.W}
    dd = {k: v.double().cpu() for k, v in carry.delta_change().items()}
    dd_fix = {k: v.double().cpu() for k, v in fixed.delta_change().items()}
    sg = {k: sigma[k].double().cpu() for k in dW}
    e_dW = sum(split.weighted_energy(dW[k], sg[k]) for k in dW)
    n_dW = sum(float((v**2).sum()) for v in dW.values())
    e_dd = sum(split.weighted_energy(dd[k], sg[k]) for k in dd)
    one_shot = 0.0
    for k, M in dW.items():
        R = split.adjoint_scale(M, uv0[k][0].double(), uv0[k][1].double(), sg[k])[1]
        one_shot += R * split.weighted_energy(M, sg[k]) / e_dW
    return {"delta_share": sum(float((v**2).sum()) for v in dd.values()) / n_dW,
            "delta_share_fixed": sum(float((v**2).sum()) for v in dd_fix.values()) / n_dW,
            "delta_share_weighted": e_dd / e_dW,
            "delta_share_weighted_fixed": sum(split.weighted_energy(dd_fix[k], sg[k]) for k in dd_fix) / e_dW,
            "delta_share_weighted_one_shot": 1 - one_shot,
            "second_share_of_delta": sum(split.weighted_energy(carry.second[k].double().cpu(), sg[k]) for k in dd) / e_dd,
            "piece_change_max": carry.piece_change_max(), "piece_change_max_fixed": fixed.piece_change_max()}


def carry_fixed_gap(carry: split.StepSplit, fixed: split.StepSplit) -> float:
    """max over matrices of ||dU_carry - dU_fixed|| / ||dU_fixed|| (and V). After the first step both split the same
    step against W0's bases, so this is 0 up to float error: the positive control of the consistency numbers."""
    def gap(a: Tensor, b: Tensor) -> float:
        n = float(b.norm())
        return float((a - b).norm()) / n if n > 0 else float((a - b).norm())

    return max(max(gap(carry.dU[k], fixed.dU[k]), gap(carry.dV[k], fixed.dV[k])) for k in carry.dU)


def per_step_maxima(per_step: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Over a run's `StepSplit.step` records ({"carry": ..., "fixed": ...} per step): the largest second-order share of
    a step (pooled and worst matrix) and how many matrix-steps had no first-order image."""
    return {v: {f: max(p[v][f] for p in per_step) for f in ("second_over_first", "second_over_first_max", "n_zeroed")}
            for v in ("carry", "fixed")}
