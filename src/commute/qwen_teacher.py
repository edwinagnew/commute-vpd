"""Teacher training on the Qwen3-1.7B decomposition, and the states the adversary reads (E40, plan
reports/plans/2026-10-28_plan_qwen-sft.md; the 67M recipe is `teacher.py`, `component_training.py`, E34/E37).

Layout as `qwen_vpd`: V [d_in, C], U [C, d_out], W ≈ (V U)ᵀ. Per site the alive columns A (`alive[site]`, an index)
get zero-initialised fp32 offsets dV [d_in, n], dU [n, d_out]; dead columns, Γ and Δ₀ = W0 − (V U)ᵀ stay fixed, so the
student's weight is (V_A + dV)(U_A + dU) + (V_D U_D) + Δ₀ transposed, that is W0 + change with

    change = (dV (U_A + dU) + V_A dU)ᵀ      (exactly 0 at zero offsets, so the student starts bitwise at W0)

Teacher training runs the target model under bf16 autocast with the change added to each decomposed nn.Linear's output
as its own term, x changeᵀ (`change_hooks`), for the student and for the teacher (change δW): a weight swap under
autocast would round W0 + δW to bf16 and lose most of a one-step δW. o_proj and down_proj, whose outputs join the
fp32 residual stream, return out.float() + term.float(), so their change is not rounded to the bf16 output; the other
sites feed bf16 matmuls and add it in bf16. The logits are the final hidden state through lm_head in fp32
(`logits32`). Loss: the mean over every position of KL(teacher ‖ student), `teacher.kl_exact_zero`'s gradient.
T0's exact zero: at δW = 0 the teacher has no hooks and the student adds an exact 0 (zero offsets give a zero change),
so every bf16 output is bitwise the teacher's, and at o_proj/down_proj residual + out.float() equals the teacher's
residual + out (the same fp32 promotion); the logits are equal, the gradient is exactly 0 and Adam does not move.
`train` also logs the fp32 KL(teacher ‖ W0) on its first batch beside the training forward's own first value, so a
precision loss shows. Readings run in fp32 on the swapped weights (`Offsets.state`).

States read by the adversary (`adversary`, `importance_pass`): W0; do-nothing (weights W0 + δW, components unchanged,
so Δ absorbs δW); after teacher training (weights W0 + change, the alive columns of U, V replaced in place, so Γ reads
the trained V; Δ stays Δ₀). g and the clean logits are recomputed per row, as experiments/qwen_adversary did.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute import qwen_vpd
from commute.qwen_sft import linears, load, mean_kl, plus, save, set_weights
from commute.qwen_vpd import SITES, Components, Gamma, importances, masked_logits
from commute.teacher import kl_exact_zero
from commute.vpd import kl_per_position

Change = Callable[[str], Tensor]  # site -> [d_out, d_in] added to the weight
RESIDUAL = ("o_proj", "down_proj")  # outputs added straight into the residual stream (Qwen3DecoderLayer)


def logits32(model: Any, tokens: Tensor) -> Tensor:
    """The body under whatever autocast is on, then the final hidden state through lm_head in fp32."""
    h = model.model(tokens, use_cache=False).last_hidden_state
    with torch.autocast("cuda", enabled=False):
        return model.lm_head(h.float())


def move_gamma(g: Gamma, device: str) -> Gamma:
    def mv(d: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(device) for k, v in d.items()}

    return Gamma(g.in_w.to(device), g.in_b.to(device), mv(g.blocks), mv(g.heads_w), mv(g.heads_b), mv(g.readouts),
                 g.inv_freq.to(device))


@contextmanager
def change_hooks(model: Any, change: Change | None) -> Iterator[None]:
    """Each decomposed nn.Linear's output plus x changeᵀ (in autocast's dtype when autocast is on); o_proj and
    down_proj, which join the fp32 residual stream, add it in fp32 (module docstring). None: no hooks."""
    if change is None:
        yield
        return

    def hook(site: str) -> Any:
        if site.endswith(RESIDUAL):
            return lambda m, args, out: out.float() + F.linear(args[0], change(site)).float()
        return lambda m, args, out: out + F.linear(args[0], change(site)).to(out.dtype)

    handles = [lin.register_forward_hook(hook(s)) for s, lin in linears(model).items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


class Offsets:
    """Trainable offsets to the alive columns of every site (module docstring). `V`, `U` hold the alive columns at
    W0 (copies, to compute the change and to restore the components); `dV`, `dU` the fp32 offsets."""

    def __init__(self, comps: Components, alive: Mapping[str, Tensor], device: str = "cuda"):
        self.alive = {s: alive[s].to(device) for s in SITES}
        self.V, self.U, self.dV, self.dU = {}, {}, {}, {}
        for s in SITES:
            V, U = comps.vu(s)
            idx = alive[s].to(V.device)
            self.V[s] = V[:, idx].to(device, torch.float32).contiguous()
            self.U[s] = U[idx].to(device, torch.float32).contiguous()
            self.dV[s] = torch.zeros_like(self.V[s], requires_grad=True)
            self.dU[s] = torch.zeros_like(self.U[s], requires_grad=True)

    def params(self) -> list[Tensor]:
        return [*self.dV.values(), *self.dU.values()]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.params())

    def to(self, device: str) -> None:
        """Move every tensor (the offsets stay leaves); training needs them on the GPU, readings do not."""
        for d in (self.alive, self.V, self.U):
            for s in SITES:
                d[s] = d[s].to(device)
        for d in (self.dV, self.dU):
            for s in SITES:
                d[s] = d[s].detach().to(device).requires_grad_()

    def change(self, site: str, device: str | None = None) -> Tensor:
        dev = device or self.dV[site].device
        V, U, dV, dU = (t[site].to(dev) for t in (self.V, self.U, self.dV, self.dU))
        return (dV @ (U + dU) + V @ dU).T

    @torch.no_grad()
    def max_abs(self) -> float:
        return max((float(p.abs().max()) for p in self.params() if p.numel()), default=0.0)

    @contextmanager
    def state(self, model: Any, comps: Components, W0: Mapping[str, Tensor]) -> Iterator[None]:
        """The decomposition after teacher training: weights W0 + change (fp32) and the alive columns of `comps`
        replaced in place by V_A + dV, U_A + dU; the original columns written back on exit."""
        dev = str(W0[SITES[0]].device)
        with torch.no_grad():
            W = {s: W0[s] + self.change(s, dev) for s in SITES}
        try:
            with torch.no_grad():
                for s in SITES:
                    V, U = comps.vu(s)
                    idx = self.alive[s].to(V.device)
                    V[:, idx] = (self.V[s] + self.dV[s]).to(V.device, V.dtype)
                    U[idx] = (self.U[s] + self.dU[s]).to(U.device, U.dtype)
            with set_weights(model, W):
                yield
        finally:
            with torch.no_grad():
                for s in SITES:
                    V, U = comps.vu(s)
                    idx = self.alive[s].to(V.device)
                    V[:, idx] = self.V[s].to(V.device, V.dtype)
                    U[idx] = self.U[s].to(U.device, U.dtype)

    def save(self, path: Path, log: Any = print) -> None:
        tensors = {f"{s}.dV": self.dV[s] for s in SITES} | {f"{s}.dU": self.dU[s] for s in SITES}
        save(path, tensors, torch.float32, droppable=False, log=log)

    def load_from(self, path: Path) -> None:
        t = load(path)
        with torch.no_grad():
            for s in SITES:
                self.dV[s].copy_(t[f"{s}.dV"])
                self.dU[s].copy_(t[f"{s}.dU"])


def train(model: Any, off: Offsets, teacher: Mapping[str, Tensor] | None, maths: Tensor, web: Tensor, lr: float,
          steps: int, rows_each: int, micro: int, seed: int, log: Any = print) -> list[dict[str, float]]:
    """Adam (betas 0.9, 0.999, no decay, no clipping) on the offsets for `steps` steps. Each step: `rows_each` rows
    drawn with replacement from `maths` and from `web` (CPU [N, T] tensors, batch order `seed`), the loss the mean
    over every position of the batch of KL(teacher ‖ student), in micro-batches of `micro` rows. `teacher`: δW per
    site (bf16 is enough: it enters as its own term); None: the teacher is the unchanged model. curve[0] also holds
    `kl_fp32_teacher_w0`, the same first batch's KL(teacher ‖ W0) from fp32 forwards on swapped weights (0 for None),
    to read beside curve[0]["kl"], the training forward's value of the same quantity."""
    dev = off.dV[SITES[0]].device
    opt = torch.optim.Adam(off.params(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    gen = torch.Generator().manual_seed(seed)
    t_change = None if teacher is None else teacher.__getitem__
    n = 2 * rows_each * maths.shape[1]
    curve: list[dict[str, float]] = []
    kl32 = 0.0
    model.train()  # gradient checkpointing; Qwen3 has no dropout
    try:
        for step in range(1, steps + 1):
            t0 = time.time()
            draw = [rows[torch.randint(len(rows), (rows_each,), generator=gen)] for rows in (maths, web)]
            batch = torch.cat(draw).to(dev)
            if step == 1 and teacher is not None:
                W0 = {s: lin.weight.data for s, lin in linears(model).items()}
                kl32 = mean_kl(model, list(batch.split(1)), plus(W0, teacher), None)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for mb in batch.split(micro):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad(), change_hooks(model, t_change):
                        t_logits = logits32(model, mb)
                    with change_hooks(model, off.change):
                        s_logits = logits32(model, mb)
                loss = kl_exact_zero(t_logits, s_logits).sum() / n
                loss.backward()
                total += float(loss)
            opt.step()
            curve.append({"kl": total, "s": time.time() - t0})
            if step == 1:
                curve[0]["kl_fp32_teacher_w0"] = kl32
                log(f"[teacher training] first batch KL(teacher ‖ W0): fp32 {kl32:.4e}, training forward {total:.4e}")
            if step in (1, 2, 10, 100) or step % 100 == 0:
                log(f"[teacher training] step {step}: KL {total:.3e}, {curve[-1]['s']:.2f} s, "
                    f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    finally:
        model.eval()
    return curve


# ----------------------------------------------------------------------------- readings of a state


@torch.no_grad()
def importance_pass(model: Any, rows: list[Tensor], gamma: Gamma, comps: Components, threshold: float,
                    masked: bool) -> dict[str, Any]:
    """Over every position of `rows`: per site mean g and the share with g > threshold ([C] each), whether g >
    threshold anywhere ([C] bool), and with `masked` the mean KL(clean ‖ m = g, Δ off) (the adversary run's ci128)."""
    sums = {s: torch.zeros(qwen_vpd.C[qwen_vpd.parse_site(s)[1]], dtype=torch.float64) for s in SITES}
    on = {s: torch.zeros_like(v) for s, v in sums.items()}
    seen = {s: torch.zeros(v.shape, dtype=torch.bool) for s, v in sums.items()}
    kl, n = 0.0, 0
    for tokens in rows:
        logits, g = importances(model, tokens, gamma, comps)
        if masked:
            kl += float(kl_per_position(logits, masked_logits(model, tokens, comps, g)).double().sum())
        for s in SITES:
            v = g[s].flatten(0, 1)
            hit = v > threshold
            sums[s] += v.sum(0).double().cpu()
            on[s] += hit.sum(0).double().cpu()
            seen[s] |= hit.any(0).cpu()
        n += tokens.numel()
    out: dict[str, Any] = {"mean_g": {s: v / n for s, v in sums.items()}, "active": {s: v / n for s, v in on.items()},
                           "any": seen}
    if masked:
        out["masked_kl"] = kl / n
    return out


def g_stats(p: dict[str, Any]) -> dict[str, Tensor]:
    """The per-subcomponent arrays of an `importance_pass`, keyed for a safetensors file (saved fp16)."""
    return {f"{s}.mean_g": p["mean_g"][s] for s in SITES} | {f"{s}.active": p["active"][s] for s in SITES}


def adversary(model: Any, rows: list[Tensor], gamma: Gamma, comps: Components, seed: int,
              log: Any = None) -> tuple[float, dict[str, Tensor]]:
    """`qwen_vpd.pgd_kl_rows` (one start, 20 sign steps of 0.1) with g and the clean logits recomputed per row, and its
    final sources (site -> [C], site.delta -> [1])."""

    def inputs(i: int) -> tuple[dict[str, Tensor], Tensor]:
        logits, g = importances(model, rows[i], gamma, comps)
        return g, logits

    final: dict[str, Tensor] = {}
    kl = qwen_vpd.pgd_kl_rows(model, rows, comps, inputs, seed, log=log, final=final)
    return kl, final


def do_nothing(model: Any, W0: Mapping[str, Tensor], dW: Mapping[str, Tensor]) -> Any:
    """The do-nothing state's context: weights W0 + δW, components unchanged (δW into Δ)."""
    return set_weights(model, plus(W0, dW))


