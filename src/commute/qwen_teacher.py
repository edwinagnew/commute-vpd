"""Teacher training on the Qwen3-1.7B decomposition, and the states the adversary reads (E40, plan
reports/plans/2026-10-28_plan_qwen-sft.md; the 67M recipe is `teacher.py`, `component_training.py`, E34/E37).

Layout as `qwen_vpd`: V [d_in, C], U [C, d_out], W ≈ (V U)ᵀ. Per site the alive columns A (`alive[site]`, an index)
get zero-initialised fp32 offsets dV [d_in, n], dU [n, d_out]; dead columns, Γ and Δ₀ = W0 − (V U)ᵀ stay fixed, so the
student's weight is (V_A + dV)(U_A + dU) + (V_D U_D) + Δ₀ transposed, that is W0 + change with

    change = (dV (U_A + dU) + V_A dU)ᵀ      (exactly 0 at zero offsets, so the student starts bitwise at W0)

Teacher training runs the fp32 target with TF32 matmuls (`tf32`, inside `train` only) and adds the change to each
decomposed nn.Linear's output as its own term, x changeᵀ (`change_hooks`), for the student and for the teacher (change
δW): TF32 then rounds the term relative to its own size, where a weight swap would round W0 + δW and lose a small δW.
The rehearsal's check (experiments/qwen_sft/check_teacher.py, cluster job 40673) measured the training forward's
first-batch KL against true fp32 at 3.4e-5 and 2.4e-4 relative under TF32; bf16 autocast, used before, read +18% and
−44%. Everything else (readings, `train`'s step-1 reference) runs with TF32 off (`qwen_vpd.fp32_matmuls`, the job's
setting): a TF32 forward on swapped weights read the step-3 teacher's KL as 0.0030 for 0.0317. Loss: the mean over
every position of KL(teacher ‖ student), `teacher.kl_exact_zero`'s gradient. At δW = 0 the teacher has no hooks and
the student adds an exact 0 (zero offsets give a zero change), so the logits are equal, the gradient is exactly 0 and
Adam does not move. `train` also logs the fp32 KL(teacher ‖ W0) on its first batch beside the training forward's own
first value. Teacher training's progress is read on fixed rows (`kl_to_teacher`: fp32, swapped weights), never on a
training batch.

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
from commute.qwen_vpd import (
    SITES,
    Components,
    Gamma,
    fp32_matmuls,
    importances,
    masked_logits,
)
from commute.teacher import kl_exact_zero
from commute.vpd import kl_per_position

Change = Callable[[str], Tensor]  # site -> [d_out, d_in] added to the weight


@contextmanager
def tf32() -> Iterator[None]:
    """fp32 matmuls in TF32 for the block, then back to full fp32 (`qwen_vpd.fp32_matmuls`)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        fp32_matmuls()


def move_gamma(g: Gamma, device: str) -> Gamma:
    def mv(d: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(device) for k, v in d.items()}

    return Gamma(g.in_w.to(device), g.in_b.to(device), mv(g.blocks), mv(g.heads_w), mv(g.heads_b), mv(g.readouts),
                 g.inv_freq.to(device))


@contextmanager
def change_hooks(model: Any, change: Change | None) -> Iterator[None]:
    """Each decomposed nn.Linear's output plus x changeᵀ, the change cast to the input's dtype one site at a time (a
    bf16 δW enters the fp32 forward as fp32). None: no hooks."""
    if change is None:
        yield
        return

    def hook(site: str) -> Any:
        return lambda m, args, out: out + F.linear(args[0], change(site).to(args[0].dtype))

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
        """The offsets with at least one entry (a site with no alive column, or outside a layer subset, has none)."""
        return [p for p in (*self.dV.values(), *self.dU.values()) if p.numel()]

    @torch.no_grad()
    def reset(self) -> None:
        """Every offset back to 0 (the next run of `train` starts from W0's components with a fresh Adam)."""
        for p in self.params():
            p.zero_()

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

    def save(self, path: Path, dtype: torch.dtype, meta: dict[str, str] | None = None, log: Any = print) -> None:
        tensors = {f"{s}.dV": self.dV[s] for s in SITES} | {f"{s}.dU": self.dU[s] for s in SITES}
        save(path, tensors, dtype, droppable=False, log=log, meta=meta)

    def load_from(self, path: Path) -> None:
        t = load(path)
        with torch.no_grad():
            for s in SITES:
                self.dV[s].copy_(t[f"{s}.dV"])
                self.dU[s].copy_(t[f"{s}.dU"])


def random_change(off: Offsets, alpha: float, seed: int, dtype: torch.dtype) -> dict[str, Tensor]:
    """The weight change of offsets α·(R_V, R_U) on the alive columns, R ~ N(0, 1) drawn per site from `seed`:
    (α R_V (U_A + α R_U) + V_A α R_U)ᵀ. A teacher W0 + this change is one the student can match exactly (its offsets
    α·R), the teacher-training positive control. Drawn on the offsets' device one site at a time."""
    out = {}
    for i, s in enumerate(SITES):
        V, U = off.V[s], off.U[s]
        gen = torch.Generator(device=V.device).manual_seed(seed * 1000 + i)
        RV = alpha * torch.randn(V.shape, generator=gen, device=V.device)
        RU = alpha * torch.randn(U.shape, generator=gen, device=V.device)
        out[s] = (RV @ (U + RU) + V @ RU).T.to(dtype)
    return out


def train(model: Any, off: Offsets, teacher: Mapping[str, Tensor] | None, maths: Tensor, web: Tensor, lr: float,
          steps: int, rows_each: int, micro: int, seed: int, log: Any = print,
          on_step: Callable[[int, list[dict[str, float]]], bool | None] | None = None) -> list[dict[str, float]]:
    """Adam (betas 0.9, 0.999, no decay, no clipping) on the offsets for at most `steps` steps. Each step: `rows_each` rows
    drawn with replacement from `maths` and from `web` (CPU [N, T] tensors, batch order `seed`), the loss the mean
    over every position of the batch of KL(teacher ‖ student), in micro-batches of `micro` rows, the forwards and the
    backward in TF32 (module docstring). `teacher`: δW per site (bf16 is enough: it enters as its own term); None: the
    teacher is the unchanged model. curve[0] also holds `kl_fp32_teacher_w0`, the same first batch's KL(teacher ‖ W0)
    from fp32 forwards on swapped weights, TF32 off (0 for None), to read beside curve[0]["kl"], the training forward's
    value of the same quantity. `on_step(step, curve)` runs after every step with TF32 off (the caller's early checks
    and fixed-row readings; it may stop the job); training ends after a step at which it returns True."""
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
            with tf32():
                for mb in batch.split(micro):
                    with torch.no_grad(), change_hooks(model, t_change):
                        t_logits = model(mb, use_cache=False).logits
                    # the backward inside the student's hooks: non-reentrant checkpointing recomputes each layer
                    # during backward, and the recompute must run the same hooks as the forward did (job 40530)
                    with change_hooks(model, off.change):
                        s_logits = model(mb, use_cache=False).logits
                        loss = kl_exact_zero(t_logits, s_logits).sum() / n
                        loss.backward()
                    total += float(loss.detach())
            opt.step()
            curve.append({"kl": total, "s": time.time() - t0})
            if step == 1:
                curve[0]["kl_fp32_teacher_w0"] = kl32
                log(f"[teacher training] first batch KL(teacher ‖ W0): fp32 {kl32:.4e}, training forward {total:.4e}")
            if step in (1, 2, 10, 50, 100) or step % 100 == 0:
                log(f"[teacher training] step {step}: KL {total:.3e}, {curve[-1]['s']:.2f} s, "
                    f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            if on_step is not None and on_step(step, curve):
                break
    finally:
        model.eval()
    return curve


@torch.no_grad()
def kl_to_teacher(model: Any, rows: list[Tensor], W0: Mapping[str, Tensor], teacher: Mapping[str, Tensor],
                  off: Offsets) -> list[float]:
    """Teacher training's reading on fixed rows: per row, the mean fp32 KL(teacher ‖ student) over its positions, the
    teacher's weights W0 + δW and the student's W0 + the offsets' change, swapped in (check_teacher.py's reading, job
    40673); the rows are of one length, so their mean is the mean over every position. At zero offsets it is
    KL(teacher ‖ W0). Holds two fp32 copies of the decomposed matrices while it runs."""
    assert len({r.numel() for r in rows}) == 1, "rows of one length (the reading is their mean)"
    dev = str(W0[SITES[0]].device)
    Wt, Ws = plus(W0, teacher), {s: W0[s] + off.change(s, dev) for s in SITES}
    return [mean_kl(model, [r], Wt, Ws) for r in rows]


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
              log: Any = None, steps: int = qwen_vpd.PGD_STEPS) -> tuple[float, dict[str, Tensor]]:
    """`qwen_vpd.pgd_kl_rows` (one start, `steps` sign steps of 0.1; 20 is the pass test, fewer only in a rehearsal)
    with g and the clean logits recomputed per row, and its final sources (site -> [C], site.delta -> [1])."""

    def inputs(i: int) -> tuple[dict[str, Tensor], Tensor]:
        logits, g = importances(model, rows[i], gamma, comps)
        return g, logits

    final: dict[str, Tensor] = {}
    kl = qwen_vpd.pgd_kl_rows(model, rows, comps, inputs, seed, steps=steps, log=log, final=final)
    return kl, final


def do_nothing(model: Any, W0: Mapping[str, Tensor], dW: Mapping[str, Tensor]) -> Any:
    """The do-nothing state's context: weights W0 + δW, components unchanged (δW into Δ)."""
    return set_weights(model, plus(W0, dW))


