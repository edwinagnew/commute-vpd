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

The stage-1 readings (P1, P2_S, P2_busy, P3 and the decision tree) are pure functions of per-state records at the bottom, with
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


def subset_like(sel: Sel, pool: Sel, seed: int) -> Sel:
    """A random subset of `pool` with the size of `sel` per matrix (all of pool[k] where it is smaller): the null's
    recruited set cut to |S|, matched per matrix."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, idx in sel.items():
        p = pool.get(k, torch.zeros(0, dtype=torch.long))
        out[k] = p[torch.randperm(p.numel(), generator=gen)[: min(int(idx.numel()), int(p.numel()))]].sort().values
    return out


# ----------------------------------------------------------------------------- stage-1 readings (pure)
# Amended before any run, 2026-10-27, after code review: each seed is classified separately and a task takes a class
# only when both seeds agree; a task whose random removal bites (random max > 0.15) or with a seed of gain < 1 nat is
# unscored and counts toward no branch; P2 is split into P2_S (scored) and P2_busy (the busy control).

UNSCORED_GAIN, UNSCORED_RANDOM = "unscored: gain < 1 nat", "unscored: random removal bites"


def seed_class(r: dict[str, Any]) -> str:
    """One scorable seed against the tree's bars: 'pass' (P1's bars on one seed), 'fail', 'partial', or 'none'."""
    rho, rmax = r["rho_S"], r["random_max"]
    if rho >= RHO_PASS and rmax <= RANDOM_MAX and r["rho_null"] <= NULL_MAX:
        return "pass"
    if rho < RHO_FAIL or rho < rmax + MARGIN:
        return "fail"
    return "partial" if rho < RHO_PASS else "none"


def task_class(seeds: list[dict[str, Any]]) -> str:
    """A task's class: unscored (a seed with gain < 1 nat, or random removal bites on either seed), the seeds' common
    class (a pass also needs the seeds' rho_S within 0.1, P1's own rule), else 'seeds disagree'."""
    if not all(r["scorable"] for r in seeds):
        return UNSCORED_GAIN
    if max(r["random_max"] for r in seeds) > RANDOM_MAX:
        return UNSCORED_RANDOM
    cls = {seed_class(r) for r in seeds}
    if len(cls) > 1:
        return "seeds disagree"
    c = cls.pop()
    rho = [r["rho_S"] for r in seeds]
    return "pass, seeds more than 0.1 apart" if c == "pass" and max(rho) - min(rho) > SEED_GAP else c


def stage1_reading(classes: dict[str, str]) -> str:
    """The decision tree on the task classes: 'pass' (P1), 'fail', 'partial', or 'outside the tree'. The three are
    exclusive: each needs two of the three tasks."""
    for branch in ("pass", "fail", "partial"):
        if sum(c == branch for c in classes.values()) >= N_TASKS:
            return branch
    return "outside the tree"


def readings(states: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """P1, P2_S, P2_busy, P3 and the decision tree from {task: {seed: record}}; a record has scorable, rho_S, random_max,
    rho_null, rho_S_W0, kl_ft and kl_other {'S', 'busy'}. P2_S and P2_busy are read on the tasks that pass (both seeds),
    unscored if there are none; P3 on every scorable state."""
    classes = {n: task_class(list(s.values())) for n, s in states.items()}
    reading = stage1_reading(classes)
    on_p1 = [n for n, c in classes.items() if c == "pass"]
    p2s = {n: all(r["kl_other"]["S"] <= KL_S * r["kl_ft"] for r in states[n].values()) for n in on_p1}
    p2b = {n: all(r["kl_other"]["busy"] >= KL_BUSY * r["kl_ft"] for r in states[n].values()) for n in on_p1}
    p3 = {f"{n}/s{k}": abs(r["rho_S_W0"]) <= W0_TOL for n, s in states.items() for k, r in s.items() if r["scorable"]}
    return {"task_class": classes, "seed_class": {f"{n}/s{k}": seed_class(r) if r["scorable"] else UNSCORED_GAIN
                                                  for n, s in states.items() for k, r in s.items()},
            "reading": reading, "P1": {"per_task": {n: c == "pass" for n, c in classes.items()}, "pass": reading == "pass"},
            "P2_S": {"per_task": p2s, "pass": all(p2s.values()) if p2s else "unscored"},
            "P2_busy": {"per_task": p2b, "pass": all(p2b.values()) if p2b else "unscored",
                        "on_miss": "busy control too weak (not 'S damages broadly')"},
            "P3": {"per_state": p3, "pass": all(p3.values()) if p3 else "unscored",
                   "unscored": [f"{n}/s{k}" for n, s in states.items() for k, r in s.items() if not r["scorable"]]},
            "P2_S_fails_while_P1_passes": reading == "pass" and bool(p2s) and not all(p2s.values())}
