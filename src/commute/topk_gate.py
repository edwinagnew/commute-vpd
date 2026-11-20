"""The re-gating share with a swappable gate (cell P2.47): the released VPD subcomponents kept, each one's mask set by a
gate read from the clean forward of the model as it stands (W0, or the fine-tune), then the masked forward (Delta out).

Gates (each maps one matrix's Gamma gates g (B, L, C) and inner activations a (B, L, C) to a mask (B, L, C)):
  gamma      g itself: Gamma's causal importances, as `local_faith.route_cells` uses them (the share as saved)
  gamma_bin  1 where g > 0.5, else 0: Gamma's own choice, binarised (separates "binary" from "which ones")
  topk(k)    1 on the k alive subcomponents of largest |a| at each row and position, else 0: a neuron-style gate,
             as `masked_reads.top_masks` gates MLP neurons by |firing|
The inner activation of subcomponent c on the matrix's input x is x . V_c, `LinearComponents.get_component_acts` in
param-decomp (pinned 74146b5, `param_decomp/models/components.py`), computed in fp32 from the cached input.

`gate_cells` makes one clean forward per side per batch, exactly as `vpd.compute_ci` does (bf16 autocast, the whole batch
in one chunk, `cache_type="input"`), so the gamma gate's cells equal `route_split`'s L00 and L01 on the same windows, batch
size and ci_len. The share of a gate is mean(L01 - L00) / mean(T1 - T0), as `local_faith.route_split`'s r."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from commute.data import target_tokens
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.local_faith import masked_logits

Gate = Callable[[str, Tensor, Tensor], Tensor]  # (matrix key, g (B, L, C), inner acts (B, L, C)) -> mask (B, L, C)


def topk_mask(acts: Tensor, alive: Tensor, k: int) -> Tensor:
    """1.0 on the k candidates (alive, (C,) bool) of largest |acts| along the last dimension, 0.0 elsewhere; a dead
    subcomponent is never chosen. Ties are broken by `torch.topk`."""
    n = int(alive.sum())
    assert 1 <= k <= n, (k, n)
    a = acts.abs().float().masked_fill(~alive.to(acts.device), float("-inf"))
    idx = a.topk(k, dim=-1).indices
    return torch.zeros_like(a).scatter_(-1, idx, 1.0)


def gamma(key: str, g: Tensor, acts: Tensor) -> Tensor:
    return g


def gamma_bin(key: str, g: Tensor, acts: Tensor) -> Tensor:
    return (g > 0.5).to(g.dtype)


def topk(k: Mapping[str, int], alive: Mapping[str, Tensor]) -> Gate:
    return lambda key, g, acts: topk_mask(acts, alive[key], k[key])


def scaled_k(k_m: Mapping[str, int], scale: float, alive: Mapping[str, Tensor]) -> dict[str, int]:
    """k per matrix at `scale` times the matched k: round, at least 1, at most the matrix's alive count."""
    return {key: min(int(alive[key].sum()), max(1, round(scale * k))) for key, k in k_m.items()}


@torch.no_grad()
def clean(model: Any, x: Tensor) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    """The clean forward of the model as it stands on x (B, L): logits, Gamma's gates and each matrix's inner
    activations x . V (fp32). The first two are `vpd.compute_ci(model, x, chunk=len(x))`'s."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        out = model(x, cache_type="input")
        ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
    g = dict(ci.lower_leaky)
    acts = {k: model.components[k].get_component_acts(out.cache[k]) for k in g}
    return out.output, g, acts


@torch.no_grad()
def gate_cells(model: Any, W0: Mapping[str, Tensor], dW: Mapping[str, Tensor], toks: Tensor, half: int, token: int | Tensor,
               batch: int, ci_len: int | None, gates: Mapping[str, Gate]) -> dict[str, Tensor]:
    """Per window (N,) float64 CPU, the log-odds of the target token at `half`: T0 and T1 (the target model at W0 and at
    W0 + dW) and, per gate name, L00_<name> and L01_<name>: the released subcomponents (Delta out) masked by the gate read
    from W0's clean forward (L00) or from the fine-tune's (L01). `ci_len` as `local_faith.route_cells`."""
    device = next(model.parameters()).device
    W1 = {k: w + dW[k].to(w.device, w.dtype) for k, w in W0.items()}
    out: dict[str, list[Tensor]] = {}
    for s in range(0, len(toks), batch):
        b = toks[s : s + batch].to(device)
        tgt, x = target_tokens(token, b, half), b[:, :ci_len]
        for side, W in (("0", None), ("1", W1)):
            with model_state(model, target=W):  # None: the model as it stands, W0
                logits, g, acts = clean(model, x)
            out.setdefault(f"T{side}", []).append(log_odds(logits[:, half], tgt).double().cpu())
            for name, gate in gates.items():
                m = {k: gate(k, g[k], acts[k]) for k in g}
                lg = masked_logits(model, x, m, None, None)[:, half]
                out.setdefault(f"L0{side}_{name}", []).append(log_odds(lg, tgt).double().cpu())
            del logits, g, acts
    return {c: torch.cat(v) for c, v in out.items()}


def shares(cells: Mapping[str, Tensor], names: list[str]) -> dict[str, Any]:
    """Per gate: the numerator mean(L01 - L00) and the share over the gain mean(T1 - T0); nan when |gain| < 1e-6, as
    `local_faith.route_split` (the scorer applies MIN_GAIN)."""
    gain = float(cells["T1"].mean() - cells["T0"].mean())
    out: dict[str, Any] = {"gain": gain}
    for n in names:
        num = float(cells[f"L01_{n}"].mean() - cells[f"L00_{n}"].mean())
        out[n] = {"numerator": num, "share": num / gain if abs(gain) >= 1e-6 else float("nan")}
    return out
