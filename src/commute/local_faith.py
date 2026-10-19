"""Task-local faithfulness: the faithfulness variants evaluated on short windows, at the trigger position
separately from the rest of the window (`reports/2026-10-15_review_faithfulness-scope.md`, section 1).

Why. The Pile harness (`faithfulness.py`) averages over ~131k positions of which ~3 are triggers, so it cannot
see whether a trained behaviour is carried by the components. Here every window holds one trigger at index
`half`, and each variant's KL to the target model and log-odds of the target token are reported at the trigger
and, separately, at the other positions.

Variants (Gamma frozen, reading the variant's own activations; target weights and components are whatever the
caller has set on the model):
  target      the target model itself (its log-odds is the reference; its KL is 0 by definition);
  ci_masked   masks = g, Delta excluded;
  unmasked    masks = 1, Delta excluded (do the components alone carry the model?);
  stochastic  the authors' sampling: mask = g + (1 - g) r, r ~ U(0, 1) per (position, component), Delta under its
              own per-position U(0, 1) mask, all layers (no layer routing), mean over `n_stoch_draws` draws;
  pgd_all     the authors' adversary: one source a in [0, 1]^(C + 1) per matrix shared across every window
              (mask = g + (1 - g) a, Delta under a[-1]), random init, `pgd_steps` sign steps of 0.1, objective the
              mean KL over all window positions;
  pgd_trigger NEW (not in the paper): the same adversary with the mean KL at the trigger as its objective. It
              asks whether an adversary can strip the trained behaviour specifically.
Blind spots. Trigger-only numbers are blind to damage elsewhere in the window and outside it (report kl_other and
the Pile harness alongside). The adversary is shared across windows, so it is blind to per-window attacks.

`retained` turns log-odds into the scored quantity: each variant's own log-odds gain over its value at W0, divided
by the target model's gain, i.e. the share of the behaviour change that the variant keeps. Ratios to W0 are undefined at the trigger (W0 has no
behaviour there), so this replaces the ratio language for task-local metrics.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute.data import target_tokens
from commute.identity import log_odds
from commute.vpd import compute_ci, kl_per_position, module_keys, weight_deltas

VARIANTS = ("target", "ci_masked", "unmasked", "stochastic", "pgd_all", "pgd_trigger")
STATS = ("kl_trigger", "kl_other", "lo_trigger")


def masked_logits(model: Any, batch: Tensor, masks: dict[str, Tensor], wd: dict[str, Tensor] | None,
                  delta_masks: dict[str, Tensor] | None) -> Tensor:
    """Forward with per-(position, component) masks; Delta included under `delta_masks` (B, T) when given."""
    from param_decomp.models.components import make_mask_infos

    wdm = None
    if delta_masks is not None:
        assert wd is not None
        wdm = {k: (wd[k], delta_masks[k]) for k in masks}
    use_autocast = batch.is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        out = model(batch, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=wdm))
    return out[0] if isinstance(out, tuple) else out


def stats(target_logits: Tensor, logits: Tensor, half: int, token: int | Tensor) -> dict[str, Tensor]:
    """Per-window (B,) statistics: KL(target || variant) at the trigger, mean KL over the other positions, and
    the variant's log-odds of `token` (one id, or (B,) ids) at the trigger."""
    kl = kl_per_position(target_logits, logits)  # (B, T)
    keep = torch.ones(kl.shape[1], dtype=torch.bool, device=kl.device)
    keep[half] = False
    return {"kl_trigger": kl[:, half], "kl_other": kl[:, keep].mean(1), "lo_trigger": log_odds(logits[:, half], token)}


class _Acc:
    def __init__(self) -> None:
        self.sums: dict[str, dict[str, float]] = {v: {s: 0.0 for s in STATS} for v in VARIANTS}
        self.n = 0

    def add(self, variant: str, st: dict[str, Tensor]) -> None:
        for s in STATS:
            self.sums[variant][s] += float(st[s].double().sum())

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {v: {s: self.sums[v][s] / self.n for s in STATS} for v in VARIANTS}
        out["n_windows"] = self.n
        return out


def _pgd(model: Any, cached: list[tuple[Tensor, Any, Tensor]], wd: dict[str, Tensor], half: int, objective: str,
         steps: int, step_size: float, seed: int, acc: _Acc, name: str) -> None:
    """The authors' PGD (`pgd_utils.calc_multibatch_pgd_masked_recon_loss`): one random-init source per matrix shared
    across every window, gradients summed over the batches, sign steps clamped to [0, 1]; then the final stats.
    `cached` holds (batch, its CI, its (B,) target tokens)."""
    keys = module_keys(model)
    device = cached[0][0].device
    torch.manual_seed(seed)
    adv = {k: torch.rand(1, 1, model.module_to_c[k] + 1, device=device).requires_grad_(True) for k in keys}

    def masks_of(ci: Any, batch: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        B, T = batch.shape
        comp = {k: ci.g[k] + (1 - ci.g[k]) * adv[k][..., :-1].expand(B, T, -1) for k in keys}
        dm = {k: adv[k][..., -1].expand(B, T) for k in keys}
        return comp, dm

    for _ in range(steps):
        grads = {k: torch.zeros_like(adv[k]) for k in keys}
        for batch, ci, _ in cached:
            with torch.enable_grad():
                comp, dm = masks_of(ci, batch)
                logits = masked_logits(model, batch, comp, wd, dm)
                kl = kl_per_position(ci.logits, logits)
                loss = kl[:, half].mean() if objective == "trigger" else kl.mean()
                g = torch.autograd.grad(loss, [adv[k] for k in keys])
            for k, gk in zip(keys, g):
                grads[k] += gk
        with torch.no_grad():
            for k in keys:
                adv[k].add_(step_size * grads[k].sign()).clamp_(0.0, 1.0)
    with torch.no_grad():
        for batch, ci, tgt in cached:
            comp, dm = masks_of(ci, batch)
            acc.add(name, stats(ci.logits, masked_logits(model, batch, comp, wd, dm), half, tgt))


def local_faithfulness(model: Any, windows: Tensor, half: int, token: int | Tensor, n_stoch_draws: int = 4, pgd_steps: int = 20,
                       pgd_step_size: float = 0.1, seed: int = 0, batch_size: int = 64, log: Any = print) -> dict[str, Any]:
    """windows: (N, T) token ids with the trigger at index `half`; `token` the target token, or a (V,) lookup table read
    at the trigger (`data.target_tokens`). Returns, per variant in VARIANTS, the mean over windows of kl_trigger,
    kl_other and lo_trigger, plus n_windows. Deterministic in `seed`; the model is left untouched (no parameter is
    written)."""
    keys = module_keys(model)
    device = next(model.parameters()).device
    acc = _Acc()
    cached: list[tuple[Tensor, Any, Tensor]] = []
    with torch.no_grad():
        wd = weight_deltas(model)
        torch.manual_seed(seed)
        for s in range(0, windows.shape[0], batch_size):
            batch = windows[s : s + batch_size].to(device)
            tgt = target_tokens(token, batch, half)
            ci = compute_ci(model, batch, chunk=batch.shape[0])
            B, T = batch.shape
            acc.n += B
            acc.add("target", stats(ci.logits, ci.logits, half, tgt))
            acc.add("ci_masked", stats(ci.logits, masked_logits(model, batch, ci.g, None, None), half, tgt))
            ones = {k: torch.ones_like(ci.g[k]) for k in keys}
            acc.add("unmasked", stats(ci.logits, masked_logits(model, batch, ones, None, None), half, tgt))
            draws = {st: torch.zeros(B, device=device, dtype=torch.float64) for st in STATS}
            for _ in range(n_stoch_draws):
                comp = {k: ci.g[k] + (1 - ci.g[k]) * torch.rand_like(ci.g[k]) for k in keys}
                dm = {k: torch.rand(B, T, device=device, dtype=ci.g[k].dtype) for k in keys}
                st = stats(ci.logits, masked_logits(model, batch, comp, wd, dm), half, tgt)
                for name in STATS:
                    draws[name] += st[name].double()
            acc.add("stochastic", {name: v / n_stoch_draws for name, v in draws.items()})
            cached.append((batch, ci, tgt))
    _pgd(model, cached, wd, half, "all", pgd_steps, pgd_step_size, seed, acc, "pgd_all")
    _pgd(model, cached, wd, half, "trigger", pgd_steps, pgd_step_size, seed, acc, "pgd_trigger")
    del cached
    if device.type == "cuda":
        torch.cuda.empty_cache()
    res = acc.result()
    log("[local] " + " | ".join(f"{v}: kl_t {res[v]['kl_trigger']:.4f} kl_o {res[v]['kl_other']:.4f} lo {res[v]['lo_trigger']:.3f}"
                              for v in VARIANTS))
    return res


def retained(res: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    """Share of the target model's log-odds gain over W0 that each variant keeps at the trigger, each variant measured
    against ITS OWN log-odds at W0: (lo_v(W') - lo_v(W0)) / (lo_target(W') - lo_target(W0)). `base` is
    `local_faithfulness` run with W0 as the target and the released components. Measuring each variant against itself
    removes the offset a masked model already has at W0 (CI-masked KL is 0.35 nats there), so the do-nothing control
    scores exactly 0 on the Delta-excluded variants when nothing upstream moved. nan if the target's gain is ~0."""
    gain = res["target"]["lo_trigger"] - base["target"]["lo_trigger"]
    if abs(gain) < 1e-6:
        return {v: float("nan") for v in VARIANTS if v != "target"}
    return {v: (res[v]["lo_trigger"] - base[v]["lo_trigger"]) / gain for v in VARIANTS if v != "target"}


# ----------------------------------------------------------------------------- the route split
# Added 2026-10-16 (`reports/2026-10-16_prediction_ladder-routes-amendment.md`). Which route the target's log-odds gain
# at the trigger takes. With L(C, G) the masked log-odds (components C, gate pattern G, Delta out), U the unmasked and
# T the target's, subscripts 0 = W0 and 1 = the state, every term over gain = T1 - T0:
#   d       L(C1,G0) - L(C0,G0)                        the changed components at the old gates
#   r       L(C0,G1) - L(C0,G0)                        the frozen Gamma gating differently on the new activations
#   i       L(C1,G1) - L(C1,G0) - L(C0,G1) + L(C0,G0)   interaction
#   e_gate  (U1 - L(C1,G1)) - (U0 - L(C0,G0))          carried by components Gamma does not gate on
#   e_delta (T1 - U1) - (T0 - U0)                      not carried by the components at all
# They sum to 1 by telescoping. retained_ci_masked = d + r + i and retained_unmasked = 1 - e_delta.

SHARES = ("d", "r", "i", "e_gate", "e_delta")
MIN_GAIN = 1.0  # shares are not scored below this target gain (log-odds), where they are unstable


def _batches(windows: Tensor, batch_size: int) -> list[slice]:
    return [slice(s, s + batch_size) for s in range(0, windows.shape[0], batch_size)]


def _p_of(logits: Tensor, tgt: Tensor) -> Tensor:
    """(B,) probability of each row's target token."""
    return torch.softmax(logits.float(), -1).gather(1, tgt[:, None]).squeeze(1)


def route_cells(model: Any, windows: Tensor, half: int, token: int | Tensor, batch_size: int = 64) -> dict[str, Any]:
    """Under the model's current state: Gamma's gates per batch (kept on CPU, for the other state's cross cells) and the
    mean over windows of the trigger log-odds of `token` (one id, or a (V,) lookup table) for the target (T), unmasked
    (U) and CI-masked (L) models, plus their mean P(token) (`p_T`, `p_U`, `p_L`)."""
    keys = module_keys(model)
    device = next(model.parameters()).device
    gates: list[dict[str, Tensor]] = []
    sums = {"T": 0.0, "U": 0.0, "L": 0.0, "p_T": 0.0, "p_U": 0.0, "p_L": 0.0}
    with torch.no_grad():
        for sl in _batches(windows, batch_size):
            batch = windows[sl].to(device)
            tgt = target_tokens(token, batch, half)
            ci = compute_ci(model, batch, chunk=batch.shape[0])
            gates.append({k: ci.g[k].cpu() for k in keys})
            ones = {k: torch.ones_like(ci.g[k]) for k in keys}
            lm = masked_logits(model, batch, ci.g, None, None)[:, half]
            um = masked_logits(model, batch, ones, None, None)[:, half]
            for name, lg in (("T", ci.logits[:, half]), ("U", um), ("L", lm)):
                sums[name] += float(log_odds(lg, tgt).double().sum())
                sums[f"p_{name}"] += float(_p_of(lg, tgt).double().sum())
    n = windows.shape[0]
    return {"gates": gates, **{k: v / n for k, v in sums.items()}}


def masked_lo(model: Any, windows: Tensor, half: int, token: int | Tensor, gates: list[dict[str, Tensor]],
              batch_size: int = 64) -> tuple[float, float]:
    """Mean trigger log-odds and mean P(token) of the masked model (Delta out) with the CURRENT components and the given
    gate pattern (from `route_cells`, same windows and batch size)."""
    device = next(model.parameters()).device
    lo = p = 0.0
    with torch.no_grad():
        for sl, g in zip(_batches(windows, batch_size), gates, strict=True):
            batch = windows[sl].to(device)
            tgt = target_tokens(token, batch, half)
            lm = masked_logits(model, batch, {k: v.to(device) for k, v in g.items()}, None, None)[:, half]
            lo += float(log_odds(lm, tgt).double().sum())
            p += float(_p_of(lm, tgt).double().sum())
    return lo / windows.shape[0], p / windows.shape[0]


def route_split(model: Any, windows: Tensor, half: int, token: int | Tensor, base: dict[str, Any],
                components0: dict[str, tuple[Tensor, Tensor]], batch_size: int = 64) -> dict[str, Any]:
    """The five shares for the model's current state (target W1, components C1). `base` = `route_cells` at W0 with the
    released components `components0` (C0). Returns the four L cells, U and T at both ends (log-odds, and mean P(token)
    under `p_cells`), the gain, the shares (nan if |gain| < 1e-6), `scorable` (gain >= MIN_GAIN), and max |g1 - g0| over
    all matrices and positions."""
    from commute.faithfulness import model_state

    cur = route_cells(model, windows, half, token, batch_size)
    L10, p10 = masked_lo(model, windows, half, token, base["gates"], batch_size)
    with model_state(model, components=components0):
        L01, p01 = masked_lo(model, windows, half, token, cur["gates"], batch_size)
    L00, L11 = base["L"], cur["L"]
    gain = cur["T"] - base["T"]
    terms = {"d": L10 - L00, "r": L01 - L00, "i": L11 - L10 - L01 + L00,
             "e_gate": (cur["U"] - L11) - (base["U"] - L00), "e_delta": (cur["T"] - cur["U"]) - (base["T"] - base["U"])}
    shares = {k: (v / gain if abs(gain) >= 1e-6 else float("nan")) for k, v in terms.items()}
    max_dg = max(float((g1[k] - g0[k]).abs().max()) for g0, g1 in zip(base["gates"], cur["gates"], strict=True) for k in g0)
    return {"cells": {"L00": L00, "L01": L01, "L10": L10, "L11": L11, "U0": base["U"], "U1": cur["U"], "T0": base["T"],
                      "T1": cur["T"]},
            "p_cells": {"L00": base["p_L"], "L01": p01, "L10": p10, "L11": cur["p_L"], "T0": base["p_T"], "T1": cur["p_T"],
                        "U0": base["p_U"], "U1": cur["p_U"]},
            "gain": gain, "shares": shares, "share_sum": sum(shares.values()), "scorable": gain >= MIN_GAIN, "max_dg": max_dg}
