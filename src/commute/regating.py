"""E18's trigger-only pattern test of the switching effect, as shared steps.

Moved 2026-10-22 from `experiments/regating_leak/run.py` (E18's scored stage: `load_task`, `finetune`, `state_of`,
`lo_of`, `stage_g1`, `patterns`, `stage_task`, `stage_natural`) for E24 (`experiments/regating_small`). The logic is
E18's; that run.py stays as it ran. Plan: reports/2026-10-21_plan_regating-leak.md.

A do-nothing state: target W0 + dW, released subcomponents C0, Gamma frozen, Delta out of the masked model. On a task's
held-out windows, with W0's gates everywhere except the trigger position, the masked model's trigger log-odds of the
target under
  trig    each window's own state gates g1(x);
  shared  clip(g0(x) + s), s = mean over the TRAINING windows of the state's trigger gates minus W0's (alive
          subcomponents; 0 on dead), one (C,) vector per matrix;
  mean    m1, the state's mean trigger gates over the training windows, the same for every window (mean_W0: m0);
  cross   clip(g0(x) + s_null), s_null the same change for a different-target fine-tune;
  random  S_on = alive with s > THR; s's values on S_on moved onto a random same-size-per-matrix set outside S_on
          (recruit.random_like seeds 0..n-1), clipped as shared.
Recovery = (lo - L00) / (L01 - L00): 0 at W0's gates, 1 at the state's gates everywhere. r = (L01 - L00) / (T1 - T0).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import recruit as R
from commute.data import Windows

THR, N_RANDOM, BATCH = 0.3, 20, 64
Gates = R.Gates


@dataclass
class Task:
    """A task's windows and W0's readings on them: `base` = local_faith.route_cells at W0 on the held-out windows,
    `m0` = W0's mean trigger gates over the training windows."""

    name: str
    train: Windows
    held: Windows
    target: int | Tensor  # one token id, or a (V,) lookup table read at the trigger (kv tasks; data.target_tokens)
    base: dict[str, Any]
    m0: dict[str, Tensor]

    @property
    def half(self) -> int:
        return self.held.half


def load_windows(name: str, transport: Path, ladder: Path) -> tuple[Windows, Windows, int]:
    """(train, held-out, target) for emoticon (transport_test's windows) or a ladder task (task_ladder's)."""
    if name == "emoticon":
        w = torch.load(transport / "large" / "windows.pt", weights_only=False)
        return w["train"], w["heldout"], json.loads((transport / "windows.json").read_text())["o_token_id"]
    w = torch.load(ladder / f"windows_{name}.pt", weights_only=False)
    return w["train"], w["heldout"], w["task"]["target"]


def prepare(model: Any, name: str, train: Windows, held: Windows, target: int | Tensor) -> Task:
    from commute.local_faith import route_cells, trigger_mean_gates

    base = route_cells(model, held.tokens, held.half, target, BATCH)
    return Task(name, train, held, target, base, trigger_mean_gates(model, train.tokens, held.half, BATCH))


@contextmanager
def state_of(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor]) -> Iterator[None]:
    """The do-nothing state: target W0 + dW, components untouched."""
    from commute.faithfulness import model_state

    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device, W0[k].dtype) for k in W0}):
        yield


def finetune(model: Any, task: Task, token: int | Tensor, optimizer: str, lr: float, steps: int, seed: int, device: str,
             log: Any = print) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Arm A fine-tune of all decomposed matrices toward `token` at the task's trigger; (dW, held-out eval at `steps`)."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=token, lr=lr, steps=steps, checkpoints=(steps,), seed=seed, optimizer=optimizer)
    res = finetune_arm_a(model, task.train, task.held, cfg, device, log)
    return res.deltas[steps], res.evals[steps]


def lo_of(model: Any, task: Task, gates: Gates) -> float:
    """Mean trigger log-odds of the target, masked model with the current components and `gates`."""
    from commute.local_faith import masked_lo

    return masked_lo(model, task.held.tokens, task.half, task.target, gates, BATCH)[0]


def shift_of(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], task: Task,
             alive: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """(s, m1): the state's mean trigger gates m1 over the TRAINING windows and s = m1 - m0 (0 on dead)."""
    from commute.local_faith import trigger_mean_gates

    with state_of(model, W0, dW):
        m1 = trigger_mean_gates(model, task.train.tokens, task.half, BATCH)
    return R.mean_change(m1, task.m0, alive), m1


def pattern_gates(g0: Gates, half: int, g1: Gates, shift: dict[str, Tensor], m0: dict[str, Tensor],
                  m1: dict[str, Tensor], shift_null: dict[str, Tensor]) -> dict[str, Gates]:
    """The fixed patterns, each W0's gates g0 except at the trigger."""
    trig = {"trig": R.at_trigger(g1, half), "shared": R.shifted(g0, half, shift), "mean": R.constant(g0, half, m1),
            "mean_W0": R.constant(g0, half, m0), "cross": R.shifted(g0, half, shift_null)}
    return {k: R.with_trigger(g0, half, t) for k, t in trig.items()}


def random_gates(g0: Gates, half: int, shift: dict[str, Tensor], alive: dict[str, Tensor], seed: int,
                 thr: float = THR) -> Gates:
    """The random null, draw `seed`: s's values on S_on (s > thr) moved onto a random same-size set outside S_on."""
    on = R.switched(shift, thr)
    return R.with_trigger(g0, half, R.shifted(g0, half, R.moved(shift, on, R.random_like(on, alive, seed))))


def div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-9 else float("nan")


def measure(model: Any, W0: dict[str, Tensor], task: Task, dW: dict[str, Tensor], shift_null: dict[str, Tensor],
            alive: dict[str, Tensor], n_random: int = N_RANDOM) -> dict[str, Any]:
    """r, gain, gap and every pattern's recovery for the do-nothing state W0 + dW; `shift` and `on` (S_on) are
    returned for the caller to save or compare (not JSON)."""
    from commute.local_faith import route_cells

    with state_of(model, W0, dW):
        cur = route_cells(model, task.held.tokens, task.half, task.target, BATCH)
    shift, m1 = shift_of(model, W0, dW, task, alive)
    g0, L00 = task.base["gates"], task.base["L"]
    gap, gain = cur["L"] - L00, cur["T"] - task.base["T"]
    pats = pattern_gates(g0, task.half, cur["gates"], shift, task.m0, m1, shift_null)
    rho = {k: div(lo_of(model, task, g) - L00, gap) for k, g in pats.items()}
    rnd = [div(lo_of(model, task, random_gates(g0, task.half, shift, alive, i)) - L00, gap) for i in range(n_random)]
    on = R.switched(shift, THR)
    return {"r": div(gap, gain), "gain": gain, "gap": gap, "p_T": cur["p_T"], "rho": rho, "random": rnd,
            "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd), "R_shared": div(rho["shared"], rho["trig"]),
            "n_on": R.size(on), "by_layer_on": R.by_layer(on),
            "cells": {"T0": task.base["T"], "T1": cur["T"], "L00": L00, "L01": cur["L"]},
            "shift": shift, "on": on}


def zero_recovery(model: Any, task: Task, gap: float) -> dict[str, float]:
    """Recovery of W0's own gates through the trig and shared builders (s = 0): 0 exactly."""
    g0, h, L00 = task.base["gates"], task.half, task.base["L"]
    zero = {k: torch.zeros(v.shape) for k, v in task.m0.items()}
    return {"trig": div(lo_of(model, task, R.with_trigger(g0, h, R.at_trigger(g0, h))) - L00, gap),
            "shared": div(lo_of(model, task, R.with_trigger(g0, h, R.shifted(g0, h, zero))) - L00, gap)}


def natural_contrast(model: Any, rows: Tensor, targets: dict[str, int], alive: dict[str, Tensor]) -> dict[str, Any]:
    """Per task: W0's gate contrast (mean g at Pile positions whose next token is the target minus mean g over all
    positions; recruit.ContrastAcc) and the number of such positions."""
    from commute.vpd import compute_ci

    masks = {n: R.precede_mask(rows, t) for n, t in targets.items()}
    accs = {n: R.ContrastAcc() for n in masks}
    for s in range(0, rows.shape[0], 8):
        ci = compute_ci(model, rows[s : s + 8], chunk=8)
        for n, acc in accs.items():
            acc.add(ci.g, masks[n][s : s + 8])
        del ci
    return {n: {"n_positions": acc.n_on, "contrast": acc.result(alive)} for n, acc in accs.items()}


def j_nat(on: R.Sel, contrast: dict[str, Tensor], alive: dict[str, Tensor], n_random: int = N_RANDOM) -> dict[str, float]:
    """Jaccard of S_on with the top-|S_on| alive subcomponents by W0's contrast, and its random same-size null."""
    from commute.signatures import top_by

    nat = top_by(contrast, alive, R.size(on))
    rnd = [R.jaccard(R.random_like(on, alive, i), nat) for i in range(n_random)]
    return {"J": R.jaccard(on, nat), "random_mean": sum(rnd) / len(rnd), "random_max": max(rnd)}


# ----------------------------------------------------------------------------- E28 (2026-10-22): the kill tests
# E18 Amendments 2 and 3's shuffled-change and other-window arms, moved from `experiments/regating_leak/run.py` for
# `experiments/regating_tasks` (plan: reports/2026-10-22_plan_regating-tasks.md). The logic is E18's; that run.py stays
# as it ran. R = rho / rho_trig throughout.

N_DERANGE = 5  # E18 amendment 3's derangement draws


def load_kv_windows(name: str, kv_results: Path) -> tuple[Windows, Windows, Tensor]:
    """(train, held-out, (V,) int64 key -> value table) of an E10 key-value task, from `kv_results`/large (as
    experiments/sgd_kv/run.py reads them). These files exist only on the cluster NFS: a missing one fails by name."""
    path = kv_results / "large" / f"windows_{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: E10's key-value windows live on the cluster NFS "
                                "(/mnt/nw/home/e.agnew/commute-data/results-large/kv_task/)")
    w = torch.load(path, weights_only=False)
    return w["train"], w["heldout"], w["table"]


def recovery(model: Any, task: Task, trig: R.TrigVals, gap: float) -> float:
    """rho of a trigger-only pattern, W0's gates elsewhere: (lo - L00) / gap (E18 run.py `recovery`, lines 699-705)."""
    g0, h = task.base["gates"], task.half
    return div(lo_of(model, task, R.with_trigger(g0, h, trig)) - task.base["L"], gap)


def shares(draws: list[float], rho_trig: float) -> dict[str, Any]:
    """A null's draws as R = rho / rho_trig: every draw, mean, sd (n - 1), min, max (E18 run.py `spread`, lines 738-746,
    without its ratio to the unshuffled)."""
    Rd = [div(x, rho_trig) for x in draws]
    mean = sum(Rd) / len(Rd)
    sd = (sum((x - mean) ** 2 for x in Rd) / (len(Rd) - 1)) ** 0.5 if len(Rd) > 1 else float("nan")
    return {"R_draws": Rd, "R_mean": mean, "R_sd": sd, "R_min": min(Rd), "R_max": max(Rd)}


def shuffled_recovery(model: Any, task: Task, shift: dict[str, Tensor], alive: dict[str, Tensor], gap: float,
                      rho_trig: float, n: int = N_RANDOM) -> dict[str, Any]:
    """The whole fixed change s permuted among each matrix's alive subcomponents (recruit.shuffled, seed = draw index),
    added to W0's trigger values and clipped (recruit.shifted), as R over `n` draws: E18 amendment 2's shared_perm arm
    (run.py `a2_arms`, lines 761-772)."""
    g0, h = task.base["gates"], task.half
    rho = [recovery(model, task, R.shifted(g0, h, R.shuffled(shift, alive, i)), gap) for i in range(n)]
    return shares(rho, rho_trig)


def state_trigger_values(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], task: Task) -> R.TrigVals:
    """g1(x): each held-out window's own trigger values under the do-nothing state W0 + dW, cloned off the full gate
    pattern (E18 run.py `a3_gate`, lines 833-863, its `g1`)."""
    from commute.local_faith import route_cells

    with state_of(model, W0, dW):
        cur = route_cells(model, task.held.tokens, task.half, task.target, BATCH)
    return [{k: v.clone() for k, v in t.items()} for t in R.at_trigger(cur["gates"], task.half)]


def other_window_recovery(model: Any, task: Task, g1: R.TrigVals, gap: float, rho_trig: float,
                          n: int = N_DERANGE) -> dict[str, Any]:
    """Each held-out window given another window's updated trigger values g1(pi(x)), pi = recruit.derangement(N, i) for
    draw i, and the reference g0(pi(x)) (W0's own values permuted the same way), as R over `n` draws: E18 amendment 3's
    mismatch level (run.py `a3_mismatch`, lines 883-903; its rho is the per-window gain's mean over the gap, which is
    this recovery; the donor correlations are not repeated)."""
    g0, h = task.base["gates"], task.half
    other, ref = [], []
    for i in range(n):
        p = R.derangement(len(task.held), i)
        other.append(recovery(model, task, R.permuted(g1, p), gap))
        ref.append(recovery(model, task, R.permuted(R.at_trigger(g0, h), p), gap))
    return {"other": shares(other, rho_trig), "other_W0": shares(ref, rho_trig)}
