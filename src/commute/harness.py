"""One state of the decomposition, measured the same way in every experiment that compares updates and maps.

A state is the target weights W0 + delta-W with the components either untouched (do-nothing: delta-W lands in Delta)
or, per matrix, replaced by (U_A', V_A') from a split or from component training. Each state reads the route split at the trigger
on the held-out windows (`local_faith.route_split`: d, r, i, e_gate, e_delta), the share of ||delta-W||^2 that Delta
absorbed, and, when asked, round 2's Pile harness as ratios to W0 (`faithfulness.pile_faithfulness`). The update
itself reads its input-weighted outside_share share (`split.weighted_fractions`) per matrix and pooled by energy over
the matrices whose alive subcomponents do not already span every direction.

Added 2026-10-19 (lora_matched). The same steps as `experiments/lora_commute/run.py`'s `load_all`, `measure` and
`outside_share`, which stay as they ran, without that run's task-local harness, moved counts and switched-on sets.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

State = dict[str, tuple[Tensor, Tensor, Tensor]]  # per matrix (delta-W, U_A', V_A'), as split.split_state and component_training.train


@dataclass
class Context:
    device: str
    model: Any
    W0: dict[str, Tensor]
    comps0: dict[str, tuple[Tensor, Tensor]]
    alive: dict[str, Tensor]
    uv: dict[str, tuple[Tensor, Tensor]]  # theory layout, on the device
    uv_cpu: dict[str, tuple[Tensor, Tensor]]
    bases: dict[str, Any]
    sigma: dict[str, Tensor]  # E[a a^T] of each matrix's input over the train windows
    non_saturated: list[str]  # matrices where some update direction is outside_share
    train: Any
    held: Any
    token: int
    wd0: dict[str, Tensor]
    rbase: dict[str, Any]  # route_cells at W0 with the released components
    blocks: list[list[Tensor]]
    pile_pgd_steps: int
    pile: dict[int, dict[str, float]] = field(default_factory=dict)  # W0 under the Pile harness, per adversary seed


def load(device: str, alive_path: Path, windows_path: Path, o_token: int, tol: float, pile_rows: int, pile_pgd_steps: int,
         pile_seeds: tuple[int, ...], log: Any = print) -> Context:
    """The model, round 1's alive masks and emoticon windows, the split bases and input moments, the route-split base
    cells, and W0 under the Pile harness for each adversary seed (the PGD floor)."""
    from commute import component_training, split, vpd
    from commute.data import batches
    from commute.faithfulness import pile_blocks, pile_faithfulness
    from commute.local_faith import route_cells

    model = vpd.load_component_model(device)
    alive = {k: v.cpu().bool() for k, v in torch.load(alive_path, weights_only=False)["alive"].items()}
    win = torch.load(windows_path, weights_only=False)
    W0 = vpd.target_weights(model)
    uv = component_training.alive_uv(model, alive)
    uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()}
    t0 = time.time()
    bases = split.bases(uv_cpu, tol)
    sigma = {k: v.cpu() for k, v in split.input_second_moments(model, batches(win["train"].tokens, 32), list(W0)).items()}
    log(f"bases and input moments for {len(W0)} matrices in {time.time() - t0:.0f} s")
    ctx = Context(device=device, model=model, W0=W0, comps0=vpd.component_weights(model), alive=alive, uv=uv, uv_cpu=uv_cpu,
                  bases=bases, sigma=sigma, train=win["train"], held=win["heldout"], token=o_token, wd0=vpd.weight_deltas(model),
                  rbase=route_cells(model, win["heldout"].tokens, win["heldout"].half, o_token),
                  non_saturated=[k for k in W0 if bases[k].r_u < W0[k].shape[0] and bases[k].r_v < W0[k].shape[1]],
                  blocks=pile_blocks(pile_rows, log) if pile_rows else [], pile_pgd_steps=pile_pgd_steps)
    for seed in pile_seeds if pile_rows else ():
        t0 = time.time()
        ctx.pile[seed] = pile_faithfulness(model, ctx.blocks, seed, pile_pgd_steps, log)
        log(f"[pile W0 seed {seed}] {time.time() - t0:.0f} s")
    return ctx


def with_windows(ctx: Context, windows: dict[str, Any], token: int) -> Context:
    """The context with a task's train and held-out windows (`windows["train"]`, `windows["heldout"]`), its target
    token and W0's route-split cells on those windows. The Pile blocks, W0's Pile readings, the bases and the input
    moments are kept (the last two came from the windows `load` read). Added 2026-10-21 for sgd_ladder_long; the same
    step lives as `task_context` in experiments/sgd_ladder and experiments/sgd_kv, which stay as they ran."""
    from dataclasses import replace

    from commute.local_faith import route_cells

    held = windows["heldout"]
    return replace(ctx, train=windows["train"], held=held, token=token,
                   rbase=route_cells(ctx.model, held.tokens, held.half, token))


def task_context(ctx: Context, windows_path: Path | None) -> tuple[Context, dict[str, Tensor]]:
    """A task's context (`with_windows` on the windows file, its target token from `windows["task"]["target"]`) and W0's
    input second moments on its training windows (CPU; the Sigma of `split.adjoint_state`). `None` keeps `load`'s
    windows and moments (emoticon). Added 2026-10-23 for teacher_training; the same step is adjoint_sgd's `Task`, which
    stays as it ran."""
    from commute import split
    from commute.data import batches

    if windows_path is None:
        return ctx, ctx.sigma
    w = torch.load(windows_path, weights_only=False)
    moments = split.input_second_moments(ctx.model, batches(w["train"].tokens, 32), list(ctx.W0))
    return with_windows(ctx, w, w["task"]["target"]), {k: v.cpu() for k, v in moments.items()}


def on_rows(ctx: Context, part: str, start: int, n: int, seeds: tuple[int, ...], log: Any = print) -> Context:
    """The context with the Pile harness on other rows: Pile `part` rows start .. start + n - 1 in blocks of 128
    (batches of 8, as `faithfulness.pile_blocks`), and W0 read there at each adversary seed in `seeds` (the ratio's
    reference and the floor). The model and every other field are shared with `ctx`. Added 2026-10-29 for
    broad_fragility Amendment 3; the same step is `rows2_context` in experiments/broad_fragility/amend2.py, which stays
    as it ran."""
    from dataclasses import replace

    from commute.data import batches, pile_rows
    from commute.faithfulness import pile_faithfulness

    rows = pile_rows(part, start, n, log=log)
    out = replace(ctx, blocks=[batches(rows[s : s + 128], 8) for s in range(0, rows.shape[0], 128)], pile={})
    for seed in seeds:
        t0 = time.time()
        out.pile[seed] = pile_faithfulness(out.model, out.blocks, seed, out.pile_pgd_steps, log)
        log(f"[pile W0 seed {seed}, {part} rows {start}..{start + n}] "
            f"{json.dumps({k: round(v, 4) for k, v in out.pile[seed].items()})} {time.time() - t0:.0f} s")
    return out


def pgd_floor(ctx: Context) -> float | None:
    """f = max / min - 1 of W0's PGD KL over the adversary seeds; 'within 3 floors' = ratio <= 1 + 3 f."""
    if not ctx.pile:
        return None
    kls = [v["pgd_kl"] for v in ctx.pile.values()]
    return max(kls) / min(kls) - 1


def pile_ratios(ctx: Context, log: Any = print) -> dict[str, float]:
    """Round 2's Pile harness on the model's current state, as ratios to W0 under adversary seed 0."""
    from commute.faithfulness import pile_faithfulness

    f = pile_faithfulness(ctx.model, ctx.blocks, 0, ctx.pile_pgd_steps, log)
    return {k: f[k] / ctx.pile[0][k] for k in f}


def measure(ctx: Context, name: str, dW: dict[str, Tensor], state: State | None, pile: bool, log: Any = print) -> dict[str, Any]:
    """Target weights W0 + dW with the components of `state` (None: do-nothing). The route split at the trigger, the share
    of ||dW||^2 that Delta absorbed, and the Pile ratios if `pile`."""
    from commute import component_training, vpd
    from commute.faithfulness import model_state
    from commute.local_faith import route_split

    model, W0, held = ctx.model, ctx.W0, ctx.held
    target = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
    comps = None if state is None else component_training.components_with(model, ctx.alive, {k: (v[1], v[2]) for k, v in state.items()})
    rec: dict[str, Any] = {"rel_norm_max": max(float(dW[k].norm() / W0[k].cpu().norm()) for k in dW)}
    with model_state(model, target=target, components=comps):
        wd = vpd.weight_deltas(model)
        energy = sum(float((dW[k].double() ** 2).sum()) for k in dW)  # 0 for W0 itself (gamma_refit's control)
        rec["delta_change_frac"] = (sum(float(((wd[k] - ctx.wd0[k]).double() ** 2).sum()) for k in wd) / energy
                                    if energy else float("nan"))
        rec["routes"] = route_split(model, held.tokens, held.half, ctx.token, ctx.rbase, ctx.comps0)
        if pile:
            rec["pile"] = pile_ratios(ctx, log)
    s, pc = rec["routes"]["shares"], rec["routes"]["p_cells"]
    log(f"[{name}] gain {rec['routes']['gain']:.2f} d {s['d']:.3f} r {s['r']:.3f} i {s['i']:.3f} e_gate {s['e_gate']:.3f} "
        f"e_delta {s['e_delta']:.3f} sum {rec['routes']['share_sum']:.6f} | P(o) target {pc['T1']:.3f} unmasked {pc['U1']:.3f} "
        f"ci {pc['L11']:.3f} | Delta change {rec['delta_change_frac']:.3f}"
        + (f" | pile {json.dumps({k: round(v, 3) for k, v in rec['pile'].items()})}" if pile else ""))
    return rec


def outside_share(ctx: Context, dW: dict[str, Tensor]) -> dict[str, Any]:
    return outside_share_of(dW, ctx.bases, ctx.sigma, ctx.non_saturated)


def outside_share_of(dW: dict[str, Tensor], bases: dict[str, Any], sigma: dict[str, Tensor], non_saturated: list[str]) -> dict[str, Any]:
    """Input-weighted outside_share share per non-zero matrix (with its isotropic null, the share of input energy outside
    the alive read span, and the matrix's share of the update's energy), pooled by energy over the non-saturated ones."""
    from commute import split

    per: dict[str, Any] = {}
    for k, M in dW.items():
        if not M.any():
            continue
        M, Sg = M.double(), sigma[k]
        per[k] = {**split.weighted_fractions(M, bases[k], Sg), "energy": float(((M @ Sg) * M).sum())}
    total = sum(p["energy"] for p in per.values())
    for p in per.values():
        p["energy_share"] = p["energy"] / total
    ks = [k for k in per if k in non_saturated]
    e = sum(per[k]["energy"] for k in ks)
    return {"per_matrix": per, "pooled": sum(per[k]["omega_act"] * per[k]["energy"] for k in ks) / e,
            "pooled_null": sum(per[k]["null_act"] * per[k]["energy"] for k in ks) / e}


def nothing_and_split(ctx: Context, name: str, dW: dict[str, Tensor], p_o: float, pile: bool, log: Any = print) -> dict[str, Any]:
    """The update measured do-nothing and least-squares split (`split.split_state`), its outside share, and gate G2: the
    split state's target model has the fine-tune's held-out P(o) (`p_o`) to 1e-3 and the shares sum to 1 to 1e-6."""
    from commute import split

    st = split.split_state(dW, ctx.uv_cpu, ctx.bases)
    rec = {"nothing": measure(ctx, f"{name}_nothing", dW, None, pile, log), "lift": measure(ctx, f"{name}_lift", dW, st, pile, log),
           "inexpressible": outside_share(ctx, dW)}
    g2 = {"p_o_diff": abs(rec["lift"]["routes"]["p_cells"]["T1"] - p_o), "share_sum": rec["lift"]["routes"]["share_sum"]}
    g2["ok"] = g2["p_o_diff"] <= 1e-3 and abs(g2["share_sum"] - 1) <= 1e-6
    rec["G2"] = g2
    log(f"[{name}] G2 {g2} outside share pooled {rec['inexpressible']['pooled']:.3f} (null {rec['inexpressible']['pooled_null']:.3f})")
    return rec
