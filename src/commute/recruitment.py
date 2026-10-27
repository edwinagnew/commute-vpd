"""E38's recruitment test (`experiments/recruitment_test`), as shared steps: remove a set of subcomponents from the
full model's weights and read what the trigger loses. Plan: reports/plans/2026-10-27_plan_regating-recruitment.md.

The full model is the target model with weights W (W0, or W0 + dW of a fine-tune), no Gamma, no masks. Removing a set
S subtracts its rank-1 matrices from W (`signatures.ablated_weights`). On a task's held-out windows:
  lo          mean trigger log-odds of the target;
  kl_trigger  mean KL(reference || this model) at the trigger;
  kl_other    mean over windows of the mean KL(reference || this model) over the other positions (the 40 non-trigger
              positions of a 41-token window; `local_faith.stats`, the definition of the fine-tune's `kl_other`).
The removed share of a set is rho = (lo without nothing removed - lo with S removed) / G, G = T1 - T0 the fine-tune's
gain: 0 when removing S does nothing, 1 when it undoes the fine-tune.

The stage-1 readings (P1, P2, P3 and the decision tree) are pure functions of per-state records at the bottom, with
the plan's thresholds as module constants.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.recruit import Sel

BATCH = 64
STATS = ("lo", "kl_trigger", "kl_other")

# The plan's stage-1 thresholds (removed shares unless named otherwise).
RHO_PASS, RHO_FAIL = 0.5, 0.2  # P1's bar; below RHO_FAIL a task fails
SEED_GAP = 0.1  # P1: the two seeds' rho_S within this of each other
NULL_MAX, RANDOM_MAX = 0.15, 0.15  # P1: rho_null and the random draws' max at most this
MARGIN = 0.1  # partial needs rho_S >= random max + MARGIN; a task within MARGIN of the random max fails
N_TASKS = 2  # "on at least two of the three tasks" / "on two or more tasks"
KL_S, KL_BUSY = 3.0, 10.0  # P2: removing S costs <= KL_S x the fine-tune's own kl_other; busy >= KL_BUSY x
W0_TOL = 0.1  # P3: |rho_S at W0| <= W0_TOL on every task-seed


@torch.no_grad()
def window_logits(model: Any, windows: Tensor, batch: int = BATCH) -> list[Tensor]:
    """The full model's logits on every position of every window, as it stands: one (B, L, V) float32 tensor per batch,
    kept on the model's device (about 0.9 GB for 106 windows of 41 tokens)."""
    from commute.finetune import logits_of

    device = next(model.parameters()).device
    return [logits_of(model.target_model, windows[s : s + batch].to(device)) for s in range(0, windows.shape[0], batch)]


@torch.no_grad()
def readout(model: Any, windows: Tensor, half: int, token: int, ref: list[Tensor], batch: int = BATCH) -> dict[str, float]:
    """lo, kl_trigger and kl_other of the full model as it stands, against `ref` (`window_logits`, same windows and
    batch size), each the mean over windows."""
    from commute.finetune import logits_of
    from commute.local_faith import stats

    device = next(model.parameters()).device
    sums: dict[str, float] = dict.fromkeys(STATS, 0.0)
    for s, r in zip(range(0, windows.shape[0], batch), ref, strict=True):
        st = stats(r, logits_of(model.target_model, windows[s : s + batch].to(device)), half, token)
        for k, name in (("lo", "lo_trigger"), ("kl_trigger", "kl_trigger"), ("kl_other", "kl_other")):
            sums[k] += float(st[name].double().sum())
    return {k: v / windows.shape[0] for k, v in sums.items()}


def removed(model: Any, W: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], sel: Sel, windows: Tensor,
            half: int, token: int, ref: list[Tensor]) -> dict[str, float]:
    """`readout` of the full model with weights W minus the rank-1 matrices of `sel` (the released components
    `comps`, param-decomp layout); the model's weights are restored afterwards."""
    from commute.faithfulness import model_state
    from commute.signatures import ablated_weights

    with model_state(model, target=ablated_weights(W, comps, sel)):
        return readout(model, windows, half, token, ref)


def share(lo_full: float, lo_removed: float, gain: float) -> float:
    """rho = (lo_full - lo_removed) / gain; nan when the gain is 0."""
    return (lo_full - lo_removed) / gain if abs(gain) > 1e-9 else float("nan")


# ----------------------------------------------------------------------------- stage-1 readings (pure)


def task_summary(seeds: list[dict[str, Any]]) -> dict[str, float]:
    """One task's numbers from its seeds' records (each with rho_S, random_max, rho_null): rho_S is the MEAN over the
    seeds, random_max and rho_null the MAX over the seeds (the stricter side for the null bars), gap the seeds'
    largest difference in rho_S."""
    rho = [s["rho_S"] for s in seeds]
    return {"rho_S": sum(rho) / len(rho), "seed_gap": max(rho) - min(rho),
            "random_max": max(s["random_max"] for s in seeds), "rho_null": max(s["rho_null"] for s in seeds)}


def task_fails(t: dict[str, float]) -> bool:
    """The plan's fail rule for one task: rho_S < 0.2, or rho_S within 0.1 of the random max (or below it)."""
    return t["rho_S"] < RHO_FAIL or t["rho_S"] < t["random_max"] + MARGIN


def p1_task(t: dict[str, float]) -> bool:
    return (t["rho_S"] >= RHO_PASS and t["seed_gap"] <= SEED_GAP and t["random_max"] <= RANDOM_MAX
            and t["rho_null"] <= NULL_MAX)


def partial_task(t: dict[str, float]) -> bool:
    return RHO_FAIL <= t["rho_S"] < RHO_PASS and t["rho_S"] >= t["random_max"] + MARGIN


def stage1_reading(tasks: dict[str, dict[str, float]]) -> str:
    """The decision tree on the task summaries: 'pass' (P1), 'fail', 'partial', or 'outside the tree' (none of the
    three; stage 1 did not fail, so stages 2 and 3 run). P1, fail and partial cannot hold together: each needs two of
    the three tasks on disjoint sides of the bars."""
    if sum(p1_task(t) for t in tasks.values()) >= N_TASKS:
        return "pass"
    if sum(task_fails(t) for t in tasks.values()) >= N_TASKS:
        return "fail"
    if sum(partial_task(t) for t in tasks.values()) >= N_TASKS:
        return "partial"
    return "outside the tree"


def p2_state(s: dict[str, Any]) -> bool:
    """One task-seed: removing S costs <= 3 x the fine-tune's own off-target KL, removing the busy set >= 10 x."""
    return s["kl_other"]["S"] <= KL_S * s["kl_ft"] and s["kl_other"]["busy"] >= KL_BUSY * s["kl_ft"]


def readings(states: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """P1, P2, P3 and the decision tree from {task: {seed: record}}; a record has rho_S, random_max, rho_null, rho_S_W0,
    kl_ft and kl_other {'S', 'busy'}. P2 is read on the tasks where P1 holds (both seeds), unscored if there are none."""
    tasks = {n: task_summary(list(s.values())) for n, s in states.items()}
    p1 = {n: p1_task(t) for n, t in tasks.items()}
    p2 = {n: all(p2_state(r) for r in states[n].values()) for n in tasks if p1[n]}
    p3 = {f"{n}/s{k}": abs(r["rho_S_W0"]) <= W0_TOL for n, s in states.items() for k, r in s.items()}
    reading = stage1_reading(tasks)
    return {"tasks": tasks, "task_fails": {n: task_fails(t) for n, t in tasks.items()}, "reading": reading,
            "P1": {"per_task": p1, "pass": reading == "pass"},
            "P2": {"per_task": p2, "pass": all(p2.values()) if p2 else "unscored"},
            "P3": {"per_state": p3, "pass": all(p3.values())},
            "P2_fails_while_P1_passes": reading == "pass" and bool(p2) and not all(p2.values())}
