"""Pull-back replay (theory section 5.3): post-train the alive subcomponents instead of W.

Per matrix, alive columns U_A (d_out, n), V_A (d_in, n) get trainable offsets dU, dV (zero init); dead columns and
Delta_0 stay fixed, so the target weight is W0 + (U_A + dU)(V_A + dV)^T - U_A V_A^T and Delta stays Delta_0 exactly.
A forward hook adds x -> x (dV (U_A + dU)^T + V_A dU^T) to the frozen linear, which equals that weight change.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class Offsets(nn.Module):
    def __init__(self, uv: dict[str, tuple[Tensor, Tensor]]):
        """uv[k] = (U_A (d_out, n), V_A (d_in, n)), float32 on the target device."""
        super().__init__()
        self.keys = list(uv)
        self.U = {k: u for k, (u, _) in uv.items()}
        self.V = {k: v for k, (_, v) in uv.items()}
        self.dU = nn.ParameterDict({k.replace(".", "-"): nn.Parameter(torch.zeros_like(u)) for k, (u, _) in uv.items()})
        self.dV = nn.ParameterDict({k.replace(".", "-"): nn.Parameter(torch.zeros_like(v)) for k, (_, v) in uv.items()})

    def d(self, k: str) -> tuple[Tensor, Tensor]:
        return self.dU[k.replace(".", "-")], self.dV[k.replace(".", "-")]

    def extra(self, k: str, x: Tensor) -> Tensor:
        dU, dV = self.d(k)
        x = x.to(dU.dtype)
        return (x @ dV) @ (self.U[k] + dU).T + (x @ self.V[k]) @ dU.T

    def delta_w(self, k: str) -> Tensor:
        dU, dV = self.d(k)
        U, V = self.U[k], self.V[k]
        return ((U + dU) @ (V + dV).T - U @ V.T).detach()

    def lifted(self, k: str) -> tuple[Tensor, Tensor]:
        dU, dV = self.d(k)
        return (self.U[k] + dU).detach(), (self.V[k] + dV).detach()


def hooks(linears: dict[str, nn.Linear], off: Offsets) -> list[Any]:
    handles = []
    for k, lin in linears.items():
        def hook(_m: nn.Module, args: tuple[Any, ...], out: Tensor, k: str = k) -> Tensor:
            return out + off.extra(k, args[0]).to(out.dtype)
        handles.append(lin.register_forward_hook(hook))
    return handles


# ----------------------------------------------------------------------------- alive columns and the trainer
# Added 2026-10-15 for the task ladder. The round-2 trainer lives in experiments/lift_test/run.py (_pullback_runs)
# and is left as it ran; this is the same recipe made importable, with states written to disk at the save steps.

from dataclasses import dataclass
from pathlib import Path


def alive_uv(model: Any, alive: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    """Theory layout (U_A (d_out, n), V_A (d_in, n)) of the alive columns, float32, on the model's device.
    `alive[k]` indexes the components of matrix k (bool mask or index tensor)."""
    from commute.vpd import module_keys

    return {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
                model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in module_keys(model)}


def components_with(model: Any, alive: dict[str, Tensor],
                    states: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor]]:
    """Full component weights (param-decomp layout) with the alive columns replaced by `states[k] = (U_A', V_A')`."""
    from commute.vpd import component_weights

    comps = component_weights(model)
    for k, (U, V) in comps.items():
        U, V = U.clone(), V.clone()
        U[alive[k]] = states[k][0].T.to(U.dtype).to(U.device)
        V[:, alive[k]] = states[k][1].to(V.dtype).to(V.device)
        comps[k] = (U, V)
    return comps


@dataclass
class PullbackConfig:
    lr: float = 1e-5  # round 2's selected learning rate (amendment 1)
    steps: int = 30
    save_steps: tuple[int, ...] = (10, 30)
    grad_clip: float = 1.0


def train(model: Any, uv: dict[str, tuple[Tensor, Tensor]], train_tokens: Tensor, heldout: Any, a_cfg: Any,
          cfg: PullbackConfig, device: str, W0: dict[str, Tensor], save_dir: Path | None = None,
          log: Any = print) -> tuple[list[dict[str, Any]], dict[int, dict[str, tuple[Tensor, Tensor, Tensor]]]]:
    """Adam on the alive (U, V) offsets with arm A's loss (`a_cfg`: ArmAConfig, its o_token is the target token,
    its batch, seed, grad_clip and kl_lambda apply), held-out eval every step. Returns the per-step curve and, at
    cfg.save_steps, per matrix (delta-W, U_A', V_A') on CPU; the same are written to save_dir/state_step{n}.pt.
    The target weights are never written; the model is unchanged on return (hooks removed)."""
    import copy

    from commute import vpd
    from commute.finetune import arm_a_loss, eval_arm_a

    net = model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    linears = {k: vpd.target_linear(model, k) for k in uv}
    off = Offsets(uv)
    params = list(off.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(a_cfg.seed)
    handles = hooks(linears, off)
    curve: list[dict[str, Any]] = []
    states: dict[int, dict[str, tuple[Tensor, Tensor, Tensor]]] = {}
    half = heldout.half
    try:
        for step in range(1, cfg.steps + 1):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, train_tokens.shape[0], (a_cfg.batch,), generator=gen)
            loss, info = arm_a_loss(net, ref, train_tokens[idx].to(device), a_cfg, half)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ev = eval_arm_a(net, ref, heldout, a_cfg.o_token, device)
            rel = max(float(off.delta_w(k).norm() / W0[k].norm()) for k in uv)
            curve.append({"step": step, "loss": float(loss), "grad_norm": float(gn), "rel_norm_max": rel,
                          **{f"train_{k}": v for k, v in info.items()}, **{f"heldout_{k}": v for k, v in ev.items()}})
            if step in cfg.save_steps:
                st = {k: (off.delta_w(k).cpu(), *(t.cpu() for t in off.lifted(k))) for k in uv}
                states[step] = st
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(st, save_dir / f"state_step{step}.pt")
            if step in (1, 2, 3, 5, 10, 20, 30) or step == cfg.steps:
                log(f"[pullback lr {cfg.lr:g}] step {step}: loss {float(loss):.4f} held-out P(target) {ev['p_o']:.4f} "
                    f"kl_other {ev['kl_other']:.4f} rel_norm_max {rel:.4f}")
    finally:
        for h in handles:
            h.remove()
    return curve, states
