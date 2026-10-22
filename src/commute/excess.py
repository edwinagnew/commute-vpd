"""The adversary's excess on one model, in parts: E17's reverts and E20's complement revert with its amendment 1, as one
call per model (E23 `experiments/cancellation_sgd`). The same arms live in `experiments/cancellation/run.py` and
`experiments/cancellation_complement/run.py` (`measure`, `amend1_arms`, `decompose`, `split_rest`, `reading`), which
stay as they ran; this module reproduces their numbers (E23's gate G0 at W0).

A model is W0 or a state: delta-W of the target weights and the alive (U2, V2) in theory layout. Under its own adversary
(`gamma.pgd_sources`, sign steps of 0.1 from the seed's random draw, one source per subcomponent over all rows), with
w = (1 - g)(1 - a) per alive subcomponent and position:
  F_all        fp32 KL from the model to its adversary-masked forward (`cancel.cache_adversary`)
  F_orig       w of the alive originals removed, the changes kept;  F_adv  w of the changes (E17's);  F_both  of both
  F_dead       the dead block's w;  F_dense  1 - a_Delta of Delta';  F_rest  both (alive blocks kept; amendment 1)
  F_dead_fix, F_dense_fix   the same at W0's levels `src0`, with the model's own g
  F_hooks_all  every block at the adversary's fractions through the hooks (equals F_all: the dead cache is right)
  F_full, F_half, F_perm    E17: every change reverted, random halves of the alive subcomponents, w with a permuted
  kl_to_w0     KL from the model to W0's target weights (equals F_full for component training: E17's G1)
  coherence    E17's ||sum Delta_c||^2 / sum ||Delta_c||^2 and its sign-flip null (states only)
`parts(s, w0)`: X = F_all(s) - F_all(W0) = O + F_adv + I + D + K + P + J exactly, with R = D + K + P + J.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

JOINT, ALONE_FIX = 0.5, 0.5  # E20 amendment 1's reading rule
G1_TOL, ZERO_TOL = 1e-4, 1e-6  # E20's closure gate (nats)


@dataclass(frozen=True)
class Sizes:
    pgd_steps: int = 20
    n_halves: int = 3
    n_perms: int = 3
    n_null: int = 3


SIZES = Sizes()  # E17's and E20's sizes


@torch.no_grad()
def kl_to_weights(model: Any, cached: Any, weights: dict[str, Tensor]) -> float:
    """Mean KL per position from the cached logits to the target model at `weights` (components untouched)."""
    from commute import vpd
    from commute.attribution import fwd
    from commute.faithfulness import model_state

    tot = n = 0.0
    with model_state(model, target=weights):
        for b, ref, _ in cached:
            tot += float(vpd.kl_per_position(ref, fwd(model.target_model, b)).double().sum())
            n += b.numel()
    return tot / n


def measure(model: Any, batches: list[Tensor], alive: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]],
            W0: dict[str, Tensor], update: tuple[Any, Any] | None, seed: int, src0: dict[str, Tensor] | None,
            S: Sizes = SIZES) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Every number of one model (update None: W0; else (delta-W, alive (U2, V2)), CPU or device) under its own
    adversary of `seed`. `src0`: W0's levels for the fixed arms (None: the model's own). Returns the record and the
    adversary's sources. Stops after the closure gate G1 if it fails (the record then has no arms)."""
    from commute import cancel, gamma, vpd
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    dev = next(model.parameters()).device
    new = uv if update is None else {k: (update[1][k][0].to(dev), update[1][k][1].to(dev)) for k in uv}
    target = W0 if update is None else {k: W0[k] + update[0][k].to(dev) for k in W0}
    pieces: cancel.Pieces = {k: (uv[k][0], uv[k][1], new[k][0], new[k][1]) for k in uv}
    rec: dict[str, Any] = {"coherence": None if update is None else cancel.coherence(pieces, S.n_null)}
    with model_state(model, target=target, components=components_with(model, alive, new)):
        sources, rec["pgd_kl"] = gamma.pgd_sources(model, batches, S.pgd_steps, seed=seed)
        cached, dead, rec["F_all"], first = cancel.cache_adversary(model, batches, alive, sources)
        rec["kl_to_w0"] = kl_to_weights(model, cached, W0)
        rem = cancel.Remover(model, cancel.state_blocks(model, alive, pieces))
        try:
            b0, ref0, _ = cached[0]
            rec["G1"] = {"kl_zero": float(vpd.kl_per_position(ref0, rem.logits(b0, {})).mean()),
                         "kl_closure": float(vpd.kl_per_position(first["masked_logits"], rem.logits(b0, first["weights"])).mean())}
            rec["G1"]["ok"] = rec["G1"]["kl_zero"] <= ZERO_TOL and rec["G1"]["kl_closure"] <= G1_TOL
            del first
            if rec["G1"]["ok"]:
                rec.update(_alive_arms(rem, cached, sources, alive, pieces, S))
                rec.update(_rest_arms(rem, cached, dead, sources, src0 if src0 is not None else sources, alive))
        finally:
            rem.remove()
    del cached, dead
    torch.cuda.empty_cache()
    return rec, sources


def _alive_arms(rem: Any, cached: Any, sources: dict[str, Tensor], alive: dict[str, Tensor], pieces: Any, S: Sizes
                ) -> dict[str, Any]:
    """E20's F_orig, F_adv, F_both and E17's F_full, F_half, F_perm (all over the alive blocks ORIG, NEW)."""
    from commute import cancel

    dev = next(iter(pieces.values()))[0].device
    wf = cancel.adversary_weights(sources, alive, dev)
    out: dict[str, Any] = {key: cancel.revert_kl(rem, cached, cancel.arm(wf, o, n))
                           for key, (o, n) in {"F_orig": (1, 0), "F_adv": (-1, 1), "F_both": (0, 1)}.items()}
    out["F_perm"] = [cancel.revert_kl(rem, cached, cancel.arm(cancel.adversary_weights(sources, alive, dev, perm_seed=i), -1, 1))
                     for i in range(S.n_perms)]
    out["F_full"] = cancel.revert_kl(rem, cached, lambda g: {k: [-1.0, 1.0] for k in g})
    halves = [cancel.half_weights(pieces, seed=i) for i in range(S.n_halves)]
    out["F_half"] = [cancel.revert_kl(rem, cached, lambda g, h=h: {k: [-h[k], h[k]] for k in g}) for h in halves]
    return out


def _rest_arms(rem: Any, cached: Any, dead: Any, sources: dict[str, Tensor], src0: dict[str, Tensor],
               alive: dict[str, Tensor]) -> dict[str, Any]:
    """E20 amendment 1's arms: the dead block and Delta' alone and together, at the model's levels and at W0's, the
    hooks' F_all over both caches, and the levels."""
    from commute import cancel

    dev = cached[0][0].device
    every_dead = {"dead": {k: torch.ones(int((~m).sum()), dtype=torch.bool) for k, m in alive.items()}}
    both = [(b, ref, (g, gd)) for (b, ref, g), (_, _, gd) in zip(cached, dead)]
    out: dict[str, Any] = {"F_hooks_all": cancel.revert_kl(rem, both, cancel.all_blocks_arm(sources, alive, dev))}
    for key, (src, d, n) in {"F_dead": (sources, True, False), "F_dense": (sources, False, True),
                             "F_rest": (sources, True, True), "F_dead_fix": (src0, True, False),
                             "F_dense_fix": (src0, False, True)}.items():
        out[key] = cancel.revert_kl(rem, dead, cancel.rest_arm(src, alive, dev, d, n))
    out["levels"] = cancel.levels(sources, alive)
    out["levels"]["dead_removed"] = cancel.bin_means(dead, cancel.dead_weights(sources, alive, dev), every_dead)["dead"]
    return out


def parts(s: dict[str, Any], w0: dict[str, Any]) -> dict[str, Any]:
    """X = O + F_adv + I + R and R = D + K + P + J (exact by definition; E20 and its amendment 1), the fixed-level parts,
    the level changes, amendment 1's reading, and E17's ratios h = mean F_half / F_full, F_adv / mean F_perm."""
    d: dict[str, Any] = {
        "X": s["F_all"] - w0["F_all"], "O": s["F_orig"] - w0["F_orig"], "F_adv": s["F_adv"],
        "I": s["F_both"] - s["F_orig"] - s["F_adv"],
        "R": (s["F_all"] - s["F_both"]) - (w0["F_all"] - w0["F_orig"]),
        "D": s["F_dense"] - w0["F_dense"], "K": s["F_dead"] - w0["F_dead"],
        "P": (s["F_rest"] - s["F_dense"] - s["F_dead"]) - (w0["F_rest"] - w0["F_dense"] - w0["F_dead"]),
        "J": (s["F_all"] - s["F_both"] - s["F_rest"]) - (w0["F_all"] - w0["F_orig"] - w0["F_rest"]),
        "D_fix": s["F_dense_fix"] - w0["F_dense"], "K_fix": s["F_dead_fix"] - w0["F_dead"],
        **{f"d_{x}": s["levels"][x] - w0["levels"][x] for x in ("dense_mean", "dead_mean", "dead_removed")},
    }
    d["reading"] = reading(d)
    d["h"] = sum(s["F_half"]) / len(s["F_half"]) / s["F_full"] if s["F_full"] > 0 else float("nan")
    perm = sum(s["F_perm"]) / len(s["F_perm"])
    d["adv_over_perm"] = s["F_adv"] / perm if perm > 0 else float("nan")
    return d


def reading(d: dict[str, float]) -> str:
    """E20 amendment 1's filed rule: J >= 0.5 R -> '4' (only jointly with the alive removal); else the largest of D, K, P:
    P -> "dead with Delta'"; D -> '1' if D_fix <= 0.5 D else '2'; K -> '1' if K_fix <= 0.5 K else '3'."""
    if d["R"] <= 0:
        return "none: R <= 0"
    if d["J"] >= JOINT * d["R"]:
        return "4"
    top = max(("D", "K", "P"), key=lambda x: d[x])
    if top == "P":
        return "dead with Delta'"
    return "1" if d[f"{top}_fix"] <= ALONE_FIX * d[top] else ("2" if top == "D" else "3")
