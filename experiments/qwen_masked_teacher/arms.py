"""E76's arms: teacher training with the clean term and, per arm, the masked term; one reading every 50 steps."""
from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rules
import torch
from common import (
    ARMS,
    DEV,
    LARGE,
    RESULTS,
    T0,
    Arm,
    C,
    kl_sum,
    log,
    peak,
    pools,
    read_json,
    stop,
    write_json,
)
from torch import Tensor

from commute.qwen_masked import row_masks, student_masked
from commute.qwen_teacher import Offsets, change_hooks, random_change, tf32
from commute.qwen_vpd import SITES, Components, parse_site
from commute.teacher import kl_exact_zero


@dataclass
class State:
    model: Any
    comps: Components  # bf16: the masked term's base
    off: Offsets
    rows: Tensor
    g: list[dict[str, Tensor]]  # per row, host, frozen
    dW: Mapping[str, Tensor]  # bf16, host
    wall_end: float
    step_s: dict[bool, float] = field(default_factory=lambda: {True: 2.5, False: 1.6})  # the plan's, until measured


def squared_by_block(change: Any) -> list[float]:
    out = [0.0] * 28
    for s in SITES:
        out[parse_site(s)[0]] += float(change(s).float().pow(2).sum(dtype=torch.float64))
    return out


def logits_cpu(S: State, teacher: Mapping[str, Tensor] | None) -> list[Tensor]:
    """The clean logits on the fixed rows, W0 (None) or W0 + the teacher's change as its own term."""
    out = []
    with torch.no_grad(), change_hooks(S.model, None if teacher is None else teacher.__getitem__):
        for t in S.rows[: C.fixed_rows]:
            out.append(S.model(t[None].to(DEV), use_cache=False).logits.cpu())
    return out


def control_teacher(S: State, w0: list[Tensor]) -> tuple[dict[str, Tensor], float]:
    """W0 + `random_change` at the α with KL(W0 + change ‖ W0) `target_kl` on the fixed rows, KL ∝ α² (amend1's `positive_control`)."""
    alpha, n = 1e-3, S.rows[: C.fixed_rows].numel()
    for _ in range(C.pc_iters):
        teacher = random_change(S.off, alpha, C.pc_seed, torch.bfloat16)
        kl = sum(kl_sum(p.to(DEV), w.to(DEV)) for p, w in zip(logits_cpu(S, teacher), w0, strict=True)) / n
        log(f"[C size] alpha {alpha:.6g}: KL {kl:.6g}")
        del teacher
        if not kl > 0:
            break
        alpha *= math.sqrt(C.target_kl / kl)
    return random_change(S.off, alpha, C.pc_seed, torch.bfloat16), alpha


@torch.no_grad()
def read_state(S: State, step: int, ft: list[Tensor], w0: list[Tensor], dw_blocks: list[float]) -> dict[str, Any]:
    """Numbers (i)-(iv) on the fixed rows, TF32 off. (iv) has all 196 sites masked whatever the arm trained on, with one fixed draw per row."""
    S.model.eval()
    gen = torch.Generator(DEV).manual_seed(C.read_seed)
    i, ii, iv = [], [], []
    for r, t in enumerate(S.rows[: C.fixed_rows]):
        t = t[None].to(DEV)
        f = ft[r].to(DEV)
        with change_hooks(S.model, S.off.change):
            s = S.model(t, use_cache=False).logits
        i.append(kl_sum(f, s))
        ii.append(kl_sum(w0[r].to(DEV), s))
        del s
        masks, deltas = row_masks(S.g[r], gen)
        with student_masked(S.model, S.comps, S.off, masks, deltas):
            iv.append(kl_sum(f, S.model(t, use_cache=False).logits))
        del masks, deltas, f
    n = S.rows.shape[1]
    blocks = squared_by_block(S.off.change)
    return {"step": step, "i": sum(i) / (n * len(i)), "ii": sum(ii) / (n * len(ii)), "iv": sum(iv) / (n * len(iv)),
            "iii": rules.late_ratio(blocks, dw_blocks), "blocks": blocks, "rows_i": i, "rows_iv": iv}


def term(model: Any, ctx: Any, t: Tensor, t_logits: Tensor, n: int) -> float:
    """One row's KL(teacher ‖ student) term under the hooks `ctx`, summed over positions / n, its backward inside them."""
    with ctx:
        loss = kl_exact_zero(t_logits, model(t, use_cache=False).logits).sum() / n
        loss.backward()
    return float(loss.detach())


def train_step(model: Any, comps: Components, off: Offsets, rows: Tensor, g: list[dict[str, Tensor]], teacher: Mapping[str, Tensor] | None,
               batch: list[int], masked: str | None, mask_gen: torch.Generator, route_gen: torch.Generator) -> tuple[float, float]:
    """Gradients of the mean over every position of the batch: KL(teacher clean ‖ student clean) + KL(teacher clean ‖ student masked),
    weight 1, one row at a time, forwards and backwards in TF32 as `qwen_teacher.train`. Masks fresh per row and step from the cached g."""
    n, clean, mask_term = len(batch) * rows.shape[1], 0.0, 0.0
    t_change = None if teacher is None else teacher.__getitem__
    with tf32():
        for i in batch:
            t = rows[i][None].to(off.dV[SITES[0]].device)
            with torch.no_grad(), change_hooks(model, t_change):
                t_logits = model(t, use_cache=False).logits
            clean += term(model, change_hooks(model, off.change), t, t_logits, n)
            if masked is not None:
                masks, deltas = row_masks(g[i], mask_gen, route_gen if masked == "library" else None)
                mask_term += term(model, student_masked(model, comps, off, masks, deltas), t, t_logits, n)
    return clean, mask_term


def grad_norms(off: Offsets) -> dict[str, float]:
    """The squared gradient on dV and dU, in all and at the blocks of the ends of the stream."""
    out = {}
    for key, d in (("dV", off.dV), ("dU", off.dU)):
        for name, keep in (("all", range(28)), ("block0", (0,)), ("block27", (27,))):
            out[f"{key}_{name}"] = sum(float(d[s].grad.pow(2).sum(dtype=torch.float64)) for s in SITES
                                       if parse_site(s)[0] in keep and d[s].grad is not None)
    return out


def first_minutes(S: State, arm: Arm, teacher: Mapping[str, Tensor] | None) -> None:
    """The masked term's gradient reaches dV and dU, or stop; the same term's value under TF32 and in fp32 on one draw, logged."""
    t = S.rows[pools()[0][0]][None].to(DEV)
    S.model.train()
    with tf32():
        with torch.no_grad(), change_hooks(S.model, None if teacher is None else teacher.__getitem__):
            t_logits = S.model(t, use_cache=False).logits
        gen = torch.Generator(DEV).manual_seed(C.check_seed)
        masks, deltas = row_masks(S.g[pools()[0][0]], gen, gen if arm.masked == "library" else None)
        got = term(S.model, student_masked(S.model, S.comps, S.off, masks, deltas), t, t_logits, t.numel())
    S.model.eval()
    grads = grad_norms(S.off)
    for p in S.off.params():
        p.grad = None
    with torch.no_grad(), student_masked(S.model, S.comps, S.off, masks, deltas):
        fp32 = kl_sum(t_logits, S.model(t, use_cache=False).logits) / t.numel()
    log(f"[{arm.name} first minutes] masked term {got:.6g} under TF32, {fp32:.6g} in fp32 (teacher's logits TF32); squared gradient {grads}")
    if not (grads["dV_all"] > 0 and grads["dU_all"] > 0 and math.isfinite(grads["dV_all"] + grads["dU_all"])):
        stop("masked_gradient", f"{arm.name}: {grads}")


def save_ckpt(path: Path, step: int, off: Offsets, opt: Any, gens: dict[str, torch.Generator]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"step": step, "dV": {s: v.detach().cpu() for s, v in off.dV.items()}, "dU": {s: v.detach().cpu() for s, v in off.dU.items()},
                "opt": opt.state_dict(), "gens": {k: g.get_state().cpu() for k, g in gens.items()}}, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, off: Offsets, opt: Any, gens: dict[str, torch.Generator]) -> int:
    ck = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for s in SITES:
            off.dV[s].copy_(ck["dV"][s])
            off.dU[s].copy_(ck["dU"][s])
    opt.load_state_dict(ck["opt"])
    for k, g in gens.items():
        g.set_state(ck["gens"][k])
    return ck["step"]


def apply_reading(S: State, arm: Arm, rec: dict[str, Any], r: dict[str, Any]) -> None:
    """Keep and log a reading, every filed check at this reading, then write the arm's file; a stop leaves it on disk."""
    rec["readings"].append(r)
    log(f"[{arm.name} step {r['step']}] (i) {r['i']:.6g} (ii) {r['ii']:.6g} (iii) {r['iii']:.4g} (iv) {r['iv']:.6g}, {peak()}")
    write_json(RESULTS / f"{arm.name}.json", rec)
    rule, stops, detail = rules.arm_stop(arm.name, rec["readings"][0], r)
    if rule:
        log(f"[{arm.name} check {rule}] {'STOPS' if stops else 'passes'}: {detail}")
    if stops:
        stop(str(rule), detail)


def project(S: State, arm: Arm, step: int, s: float) -> None:
    S.step_s[arm.masked is not None] = s
    later = ARMS[[a.name for a in ARMS].index(arm.name) + 1 :]
    left = (arm.steps - step) * s + sum(a.steps * S.step_s[a.masked is not None] for a in later)
    log(f"[{arm.name} step {step}] {s:.2f} s per step; this arm {arm.steps * s / 60:.0f} min, projected job total "
        f"{(time.time() - T0 + left) / 3600:.2f} h; plan ~3.5 h")


def set_up(S: State, arm: Arm, rec: dict[str, Any]) -> tuple[Mapping[str, Tensor], list[Tensor], list[Tensor]]:
    w0 = logits_cpu(S, None)
    if arm.control:
        teacher, rec["alpha"] = control_teacher(S, w0)
    else:
        teacher = {s: v.to(DEV) for s, v in S.dW.items()}
    return teacher, logits_cpu(S, teacher), w0


def run_arm(S: State, arm: Arm) -> None:
    """One arm from its files: done is skipped; a checkpoint resumes step, offsets, Adam and draws; past the wall, a checkpoint and exit 0."""
    path, ck = RESULTS / f"{arm.name}.json", LARGE / f"{arm.name}_ckpt.pt"
    rec = read_json(path) or {"arm": arm.name, "steps": arm.steps, "masked": arm.masked, "readings": [], "train": []}
    if rec.get("done"):
        return
    S.off.reset()
    teacher, ft, w0 = set_up(S, arm, rec)
    dw_blocks = squared_by_block(S.dW.__getitem__)
    opt = torch.optim.Adam(S.off.params(), lr=C.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    gens = {"batch": torch.Generator().manual_seed(C.batch_seed), "mask": torch.Generator(DEV).manual_seed(C.mask_seed),
            "route": torch.Generator(DEV).manual_seed(C.route_seed)}
    start = load_ckpt(ck, S.off, opt, gens) if ck.is_file() else 0
    if start:
        rec["readings"] = [r for r in rec["readings"] if r["step"] <= start]
        rec["train"] = [r for r in rec["train"] if r["step"] <= start]
        log(f"[{arm.name}] resumed at step {start}")
    else:
        rec["readings"], rec["train"] = [], []
        apply_reading(S, arm, rec, read_state(S, 0, ft, w0, dw_blocks))
        if arm.masked:
            first_minutes(S, arm, teacher)
    train_loop(S, arm, rec, opt, gens, teacher, ft, w0, dw_blocks, start, ck)
    S.off.save(LARGE / f"{arm.name}_offsets.safetensors", torch.bfloat16, {"arm": arm.name, "steps": str(arm.steps)}, log)
    rec["done"], rec["max_abs_offset"] = True, S.off.max_abs()
    write_json(path, rec)
    ck.unlink(missing_ok=True)
    log(f"[{arm.name}] done, {peak()}; final {json.dumps({k: rec['readings'][-1][k] for k in ('i', 'ii', 'iii', 'iv')})}")


def train_loop(S: State, arm: Arm, rec: dict[str, Any], opt: Any, gens: dict[str, torch.Generator], teacher: Mapping[str, Tensor],
               ft: list[Tensor], w0: list[Tensor], dw_blocks: list[float], start: int, ck: Path) -> None:
    maths, web = pools()
    since, t0 = {"clean": 0.0, "masked": 0.0}, time.time()
    for step in range(start + 1, arm.steps + 1):
        S.model.train()
        batch = [maths[int(torch.randint(len(maths), (), generator=gens["batch"]))], web[int(torch.randint(len(web), (), generator=gens["batch"]))]]
        opt.zero_grad(set_to_none=True)
        clean, masked = train_step(S.model, S.comps, S.off, S.rows, S.g, teacher, batch, arm.masked, gens["mask"], gens["route"])
        if not math.isfinite(clean + masked):
            stop("non_finite", f"{arm.name} step {step}: training terms {clean}, {masked}")
        opt.step()
        since["clean"] += clean
        since["masked"] += masked
        if step % C.read_every:
            continue
        s = (time.time() - t0) / C.read_every
        rec["train"].append({"step": step, "clean": since["clean"] / C.read_every, "masked": since["masked"] / C.read_every, "s": s})
        log(f"[{arm.name} step {step}] training terms over the last {C.read_every} steps: clean {rec['train'][-1]['clean']:.6g}, masked {rec['train'][-1]['masked']:.6g}")
        if step == C.read_every:
            project(S, arm, step, s)
        apply_reading(S, arm, rec, read_state(S, step, ft, w0, dw_blocks))
        since, t0 = {"clean": 0.0, "masked": 0.0}, time.time()
        over = time.time() > S.wall_end
        if (step % C.ckpt_every == 0 or over) and step < arm.steps:
            save_ckpt(ck, step, S.off, opt, gens)
        if over and step < arm.steps:
            log(f"[{arm.name}] past the wall at step {step}: resubmit to resume")
            sys.exit(0)
