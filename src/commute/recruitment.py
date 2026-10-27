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

Amendment 2 (2026-10-27, reports/plans/2026-10-27_plan_e38-amend2.md) adds removal at the trigger position only
(`removal_hooks`, `removed_at`) and its readings (P5 to P7, at the bottom).

Amendment 3 (2026-10-27, reports/plans/2026-10-27_plan_e38-amend3.md) adds the restore test: W0's inputs and outputs
of each matrix at the trigger (`trigger_io`), a W1 pass that puts a set's trigger outputs back to W0's
(`restore_hooks`, `restored_at`), the matched active sets (`active_pool`, `active_like`) and its readings (P9, P10).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def removal_hooks(model: Any, W_removed: dict[str, Tensor], half: int | None) -> Iterator[None]:
    """While open, each decomposed matrix k of the target model outputs F.linear(x, W_removed[k]) at position `half`
    (at every position if None) and its own output everywhere else (E38 Amendment 2). With W_removed =
    `ablated_weights(W, comps, sel)` and the model's weights at W, that output is x W^T - sum_{c in sel} (x . v_c) u_c,
    computed as the weight path computes it (same input, same autocast cast), so with half=None it equals `removed`
    bit for bit. Every matrix in W_removed is hooked, also those where sel is empty. The hooks go in the `finally`."""
    from commute.vpd import target_linear

    def hook_of(w: Tensor) -> Any:
        def hook(_module: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            alt = torch.nn.functional.linear(args[0], w)
            if half is None:
                return alt
            assert out.ndim == 3, tuple(out.shape)  # (B, T, d_out)
            out = out.clone()
            out[:, half] = alt[:, half]
            return out

        return hook

    handles = []
    try:
        for k, w in W_removed.items():
            handles.append(target_linear(model, k).register_forward_hook(hook_of(w)))
        yield
    finally:
        for h in handles:
            h.remove()


def removed_at(model: Any, W: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], sel: Sel, windows: Tensor,
               half: int, token: int, ref: list[Tensor], everywhere: bool = False) -> dict[str, float]:
    """`readout` of the full model with weights W, with `sel`'s rank-1 outputs removed at the trigger position only
    (`removal_hooks`; at every position with `everywhere`, which must reproduce `removed`). Weights restored after."""
    from commute.faithfulness import model_state
    from commute.signatures import ablated_weights

    with model_state(model, target=W), removal_hooks(model, ablated_weights(W, comps, sel), None if everywhere else half):
        return readout(model, windows, half, token, ref)


IO = dict[str, list[tuple[Tensor, Tensor]]]  # matrix key -> per batch (its input (B, d_in), its output (B, d_out)) at the trigger


@torch.no_grad()
def trigger_io(model: Any, keys: list[str], windows: Tensor, half: int, batch: int = BATCH) -> IO:
    """Each matrix's input and output at position `half`, per batch of `readout` (same windows, same batch size), of
    the target model as it stands (W0 for the restore test), under the same autocast as `readout`."""
    from commute.finetune import logits_of
    from commute.vpd import target_linear

    io: IO = {k: [] for k in keys}

    def hook_of(k: str) -> Any:
        def hook(_module: Any, args: tuple[Tensor, ...], out: Tensor) -> None:
            io[k].append((args[0][:, half].clone(), out[:, half].clone()))

        return hook

    handles = [target_linear(model, k).register_forward_hook(hook_of(k)) for k in keys]
    device = next(model.parameters()).device
    try:
        for s in range(0, windows.shape[0], batch):
            logits_of(model.target_model, windows[s : s + batch].to(device))
    finally:
        for h in handles:
            h.remove()
    return io


@contextmanager
def restore_hooks(model: Any, comps: dict[str, tuple[Tensor, Tensor]], sel: Sel, io: IO, half: int,
                  whole: bool = False) -> Iterator[None]:
    """While open, at position `half` only, each matrix k in `io` outputs out + sum_{c in sel[k]} ((x0 - x) . v_c) u_c,
    with out and x its output and input in this pass and x0 its cached W0 input (`trigger_io`); with `whole`, its
    cached W0 output instead (control R2; sel unused). Pass b of the model reads io[k][b]: the forward passes must be
    `readout`'s batches, in order. x and x0 are cast to the dtype the weight path multiplies in (out's: bf16 under
    autocast); the correction's two matmuls run under the caller's autocast (bf16 under `logits_of` on CUDA, fp32 on
    CPU), the sum with out is taken in fp32 and cast back once, so x == x0 or an empty sel[k] returns out bit for bit. On a clean exit every hook must have read every cached batch exactly once. Hooks removed in the
    `finally`."""
    from commute.vpd import target_linear

    calls = dict.fromkeys(io, 0)

    def hook_of(k: str) -> Any:
        U, V = comps[k]
        idx = sel.get(k, torch.zeros(0, dtype=torch.long)).to(U.device)
        Us, Vs = U[idx].float(), V[:, idx].float()  # (n, d_out), (d_in, n)

        def hook(_module: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            x0, y0 = io[k][calls[k]]
            calls[k] += 1
            assert out.ndim == 3 and out.shape[0] == x0.shape[0], (tuple(out.shape), tuple(x0.shape))
            out = out.clone()
            if whole:
                out[:, half] = y0
            else:
                d = x0.to(out.dtype).float() - args[0][:, half].to(out.dtype).float()
                out[:, half] = (out[:, half].float() + (d @ Vs) @ Us).to(out.dtype)
            return out

        return hook

    handles = []
    try:
        for k in io:
            handles.append(target_linear(model, k).register_forward_hook(hook_of(k)))
        yield
        assert all(calls[k] == len(io[k]) for k in io), f"restore hooks saw {calls} passes, cached {len(next(iter(io.values())))}"
    finally:
        for h in handles:
            h.remove()


def restored_at(model: Any, W: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], sel: Sel, io: IO,
                windows: Tensor, half: int, token: int, ref: list[Tensor], whole: bool = False) -> dict[str, float]:
    """`readout` of the full model with weights W with `sel`'s trigger outputs restored to W0's (`restore_hooks`;
    `io` = `trigger_io` at W0 on the same windows). Weights restored afterwards."""
    from commute.faithfulness import model_state

    with model_state(model, target=W), restore_hooks(model, comps, sel, io, half, whole):
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


def minus(a: Sel, b: Sel) -> Sel:
    """a without b's members, per matrix of a."""
    return {k: v[~torch.isin(v.cpu(), b.get(k, torch.zeros(0, dtype=torch.long)).cpu())].cpu() for k, v in a.items()}


Pool = dict[str, list[tuple[str, int]]]  # layer name (as recruit.by_layer) -> pieces (matrix, index), sorted


def active_pool(sel: Sel, alive: dict[str, Tensor], m0: dict[str, Tensor], thr: float) -> Pool:
    """Per layer: the alive pieces not in `sel` whose W0 mean trigger importance m0 is >= thr."""
    from commute.recruit import layer_or_none, members

    inside = members(sel)
    pool: Pool = {}
    for k in sorted(alive):
        lay = str(n) if (n := layer_or_none(k)) is not None else "other"
        on = (alive[k].cpu().bool() & (m0[k].float().cpu() >= thr)).nonzero().flatten().tolist()
        pool.setdefault(lay, []).extend((k, c) for c in on if (k, c) not in inside)
    return pool


def active_like(sel: Sel, pool: Pool, seed: int) -> Sel:
    """As many pieces per layer as `sel` has there, drawn with `seed` from pool[layer] (all of it where smaller), as a
    Sel over sel's matrices."""
    from commute.recruit import by_layer

    gen = torch.Generator().manual_seed(seed)
    out: dict[str, list[int]] = {k: [] for k in sel}
    for lay, n in by_layer(sel).items():
        p = pool.get(lay, [])
        for j in torch.randperm(len(p), generator=gen)[: min(n, len(p))].tolist():
            out.setdefault(p[j][0], []).append(p[j][1])
    return {k: torch.tensor(sorted(v), dtype=torch.long) for k, v in out.items()}


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


# ----------------------------------------------------------------------------- E38 Amendment 2's readings (pure)
# reports/plans/2026-10-27_plan_e38-amend2.md, filed predictions experiments/recruitment_test/results/predictions_amend2.json.
# D(X) = rho_ft(X) - rho_0(X): the share of the gain removing X takes from W1 minus the share it takes from W0, both
# over the same state's gain. A record (one task-seed) has scorable, kl_ft, and per position rule ('trigger', 'all')
# D, rho_0 and kl_other_ft, each {'S', 'null', 'busy': float, 'random': [20 floats]}.

D_THROUGH, D_NOT = 0.10, 0.05  # P5: 'through' needs D(S) >= D_THROUGH; below D_NOT it is 'not'
BEAT_NOT = 18  # P5: D(S) above fewer than 18 of the 20 random D is 'not'
NULL_D_MAX = 0.10  # P5_null: |D(null)| at most this on every scored state
RHO0_MIN = 0.10  # P6: rho_0(S), trigger-only, at least this on every scored state


def d_seed_class(D_S: float, D_random: list[float]) -> str:
    """P5's seed rule: 'through' (D(S) >= 0.10 and above all random D), 'not' (D(S) < 0.05 or above fewer than 18 of
    them), else 'unclear'."""
    beats = sum(D_S > d for d in D_random)
    if D_S >= D_THROUGH and beats == len(D_random):
        return "through"
    return "not" if D_S < D_NOT or beats < BEAT_NOT else "unclear"


def common_classes(seeds: dict[str, str], states: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Per task of {task: {seed: record}}: a seed's unscored class if it has one ('unscored: ...', e.g. gain < 1 nat), the
    seeds' common class from `seeds` ({'<task>/s<seed>': class}), else 'seeds disagree'."""
    out = {}
    for n, s in states.items():
        cls = {seeds[f"{n}/s{k}"] for k in s}
        uns = sorted(c for c in cls if c.startswith("unscored"))
        out[n] = uns[0] if uns else cls.pop() if len(cls) == 1 else "seeds disagree"
    return out


def d_reading(states: dict[str, dict[str, dict[str, Any]]], pos: str) -> dict[str, Any]:
    """P5 under one position rule: seed classes, task classes (the seeds' common class, else 'seeds disagree';
    unscored if a seed has gain < 1 nat) and the reading ('through' / 'not through' on >= 2 tasks, else 'outside the
    tree')."""
    seeds = {f"{n}/s{k}": d_seed_class(r["D"][pos]["S"], r["D"][pos]["random"]) if r["scorable"] else UNSCORED_GAIN
             for n, s in states.items() for k, r in s.items()}
    tasks = common_classes(seeds, states)
    counts = {c: sum(v == c for v in tasks.values()) for c in ("through", "not")}
    reading = ("through" if counts["through"] >= N_TASKS else "not through" if counts["not"] >= N_TASKS
               else "outside the tree")
    return {"seed_class": seeds, "task_class": tasks, "reading": reading}


def amend2_readings(states: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """P5 (trigger-only), P5_all (all positions), P5_null, P6 and P7 from {task: {seed: record}}. P5_null and P6 read
    every scored state; P7 (kl_other of trigger-only S removal from W1 <= KL_S x the fine-tune's own) needs both seeds
    of >= 2 tasks."""
    scored = {f"{n}/s{k}": r for n, s in states.items() for k, r in s.items() if r["scorable"]}
    p5, p5_all = d_reading(states, "trigger"), d_reading(states, "all")
    null = {k: abs(r["D"]["trigger"]["null"]) <= NULL_D_MAX for k, r in scored.items()}
    p6 = {k: r["rho_0"]["trigger"]["S"] >= RHO0_MIN for k, r in scored.items()}
    p7 = {n: all(r["kl_other_ft"]["trigger"]["S"] <= KL_S * r["kl_ft"] for r in s.values()) for n, s in states.items()}
    return {"P5": p5, "P5_all": {**p5_all, "same_task_classes_as_P5": p5_all["task_class"] == p5["task_class"]},
            "P5_null": {"per_state": null, "pass": all(null.values()) if null else "unscored"},
            "P6": {"per_state": p6, "pass": all(p6.values()) if p6 else "unscored"},
            "P7": {"per_task": p7, "pass": sum(p7.values()) >= N_TASKS}}


# ----------------------------------------------------------------------------- E38 Amendment 3's readings (pure)
# reports/plans/2026-10-27_plan_e38-amend3.md, filed predictions experiments/recruitment_test/results/predictions_amend3.json.
# rho_r(X) = (T1 - lo(W1 with X's trigger outputs restored)) / G. A record (one task-seed) has scorable, active
# {'short': bool, ...} and rho {'S', 'null_minus_S': float, 'random', 'active': [20 floats]}.

SPEC_MIN, SPEC_NOT = 0.10, 0.05  # P9: 'specific' needs rho_r(S) >= SPEC_MIN; below SPEC_NOT it is 'not'
NULL_MARGIN = 0.05  # P9: 'specific' needs rho_r(S) >= rho_r(null_minus_S) + NULL_MARGIN
P10_MIN = 0.10  # P10: rho_r(S) at least this, and above all 20 random rho_r
# results/predictions_amend3_pool.json (filed before any run): a state whose active pool is smaller than S's count in
# some layer even at m0 >= 0.2 (record active.short) is unscored for P9 and counts toward no branch.
UNSCORED_POOL = "unscored: active pool too small"


def restore_seed_class(rho_S: float, rho_active: list[float], rho_null_minus_S: float) -> str:
    """P9's seed rule: 'specific' (rho_r(S) >= 0.10, above all active rho_r, and >= rho_r(null_minus_S) + 0.05), 'not'
    (rho_r(S) < 0.05 or above fewer than 18 of the active rho_r), else 'unclear'."""
    beats = sum(rho_S > a for a in rho_active)
    if rho_S >= SPEC_MIN and beats == len(rho_active) and rho_S >= rho_null_minus_S + NULL_MARGIN:
        return "specific"
    return "not" if rho_S < SPEC_NOT or beats < BEAT_NOT else "unclear"


def amend3_readings(states: dict[str, dict[str, dict[str, Any]]], p10_tasks: tuple[str, ...]) -> dict[str, Any]:
    """P9 (seed classes, task classes, reading: 'specific' / 'not specific' on >= 2 tasks, else 'outside the tree') and
    P10 (rho_r(S) >= 0.10 and above all random rho_r on both seeds of each of `p10_tasks`; an unscored seed fails it)
    from {task: {seed: record}}."""
    seeds = {f"{n}/s{k}": UNSCORED_GAIN if not r["scorable"] else UNSCORED_POOL if r["active"]["short"]
             else restore_seed_class(r["rho"]["S"], r["rho"]["active"], r["rho"]["null_minus_S"])
             for n, s in states.items() for k, r in s.items()}
    tasks = common_classes(seeds, states)
    counts = {c: sum(v == c for v in tasks.values()) for c in ("specific", "not")}
    reading = ("specific" if counts["specific"] >= N_TASKS else "not specific" if counts["not"] >= N_TASKS
               else "outside the tree")
    p10 = {f"{n}/s{k}": bool(r["scorable"] and r["rho"]["S"] >= P10_MIN and all(r["rho"]["S"] > x for x in r["rho"]["random"]))
           for n in p10_tasks for k, r in states[n].items()}
    return {"P9": {"seed_class": seeds, "task_class": tasks, "reading": reading},
            "P10": {"per_state": p10, "pass": all(p10.values()) and len(p10) == 2 * len(p10_tasks)}}
