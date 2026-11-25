"""broad_fragility Amendment 2 (E39 A2): why does the broad failure happen? Four cheap tests in one job.
Plan: reports/plans/2026-10-27_plan_broad-fragility.md -> "Amendment 2 (2026-10-28)".

Everything is run.py's, imported: E37's lowercase task and rows, the Adam fine-tune, teacher training
(`row_copy.copy_on_rows`: Adam 1e-5, batch 8, 1,000 steps), the check (Pile harness, 20-step adversary from one random
start, adversary seed 0, val rows 0-255, as a ratio over W0 in the same job; pass <= 1.40, fail >= 1.61) and the reads
(`run.read`: behaviour kept s, gap, damage, check, stochastic ratio, size). Same card as E39 (H100 80GB HBM3), whose
stage 0 is repeated. Stages, cheapest first, each written to results/amend2.json as it finishes (a rerun on the same
card repeats stage 0 and skips finished stages):

  stage 0      (i) W0 on adversary seeds 0-2 (`run.stage0_w0`); (ii) the Adam fine-tune regenerated and its
               teacher-trained decomposition retrained and read (`run.stage0_trained`, check 2.2011); (iii) the Adam
               fine-tune's do-nothing read (E37 4.8678), the positive control. Any miss stops the job.
  move         (D) random moves of the alive U, V (`fragility.random_move`, one direction per noise seed) scaled until
               their damage on plain text equals the trained decomposition's (`match`), then read
  label_adam   (A) component training with labels (next-token cross-entropy on the lowercase rows), Gamma and Delta
               frozen, Adam at each rate, states at steps 100, 300, 1000; the state whose held-out gain is nearest
               the Adam fine-tune's is read and saved (`label`)
  label_sgd    (A) the same with plain SGD
  lora         (C) the rank-8 LoRA at E39's rate: LoRA seed 0 with batch order 0 (E39's run, its control) and LoRA
               seed 1 with batch order 1, each teacher-trained, saved and read; then W0 on a second set of 256 Pile
               rows (val 256-511, adversary seeds 0-2) and the Adam and both LoRA teacher-trained decompositions read
               on them
  cap          (B) plain SGD held to rank 8 per matrix (`rank_cap`), the rate nearest the Adam gain (`run.pick_rate`),
               teacher-trained, saved and read; beside it, the share of the plain SGD update (Amendment 1's rate)
               beyond rank 8, which says how much the cap removes

Results to results/amend2.json, log to results/large/amend2.log, states (bf16 offsets) to results/large/a2_*.pt.

Usage (repo root, project venv):  amend2.py --predictions   files results/predictions_amend2.json (once)
                                  amend2.py [--device cuda]  the run (GPU); refuses without that file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import run
import torch
from run import (
    CFG,
    LARGE,
    RESULTS,
    TRANSPORT,
    UV,
    Job,
    dump,
    halt,
    log,
    pick_rate,
    read,
    sgd_ft,
    stage0_trained,
    stage0_w0,
)
from torch import Tensor

PRED, OUT = RESULTS / "predictions_amend2.json", RESULTS / "amend2.json"
run.OUT, run.LOG = OUT, LARGE / "amend2.log"  # run.read, dump, halt and log write here


@dataclass(frozen=True)
class Amend2:
    e37_nothing: float = 4.8678  # stage 0 (iii): E37's Adam do-nothing check on this card
    move_seeds: tuple[int, ...] = (0, 1)
    move_x0: float = 4.0  # first scale tried; E39's x1 moves did damage 0.009-0.010, about x^2, so x is about 5
    move_x_max: float = 64.0
    move_evals: int = 12
    damage_tol: float = 0.05  # |damage / target - 1|, relative
    label_adam_lrs: tuple[float, ...] = (1e-5, 3e-5)  # 1e-5 is teacher training's rate
    label_sgd_lrs: tuple[float, ...] = (0.01, 0.03, 0.1)  # E21/E26's SGD component-training rates were 0.03-0.1
    label_steps: int = 1000
    label_saves: tuple[int, ...] = (100, 300, 1000)
    gain_tol: float = 0.30  # a state is gain-matched if |gain / Adam fine-tune's gain - 1| <= this (Amendment 1's P8)
    lora_lr: float = 1e-4  # E39's chosen LoRA rate
    lora_runs: tuple[tuple[int, int], ...] = ((0, 0), (1, 1))  # (LoRA seed, teacher-training batch order)
    e39_lora_gain: float = 0.1479  # E39's LoRA fine-tune and its teacher-trained check (the control of stage lora)
    e39_lora_check: float = 1.2726
    rows2: tuple[str, int, int] = ("val", 256, 256)
    cap_rank: int = 8
    cap_lrs: tuple[float, ...] = (0.01, 0.03, 0.1)  # Amendment 1's plain SGD gains: 0.137, 0.148, 0.103
    sgd_ref_lr: float = 0.01  # Amendment 1's chosen plain SGD rate
    tail_max: float = 1e-4  # the capped update's largest per-matrix share beyond rank 8 (fp32 rounding of W0 + dW is
    # about 1e-10 of a matrix that moved 1%; an uncapped SGD update's share is expected at 0.1-0.4)

    def __post_init__(self) -> None:
        assert self.lora_runs[0] == (CFG.lora_seed, CFG.copy_seed) and len(set(self.lora_runs)) == len(self.lora_runs)
        assert self.lora_lr in CFG.lora_lrs and self.cap_rank == CFG.lora_rank
        assert self.rows2[1] >= CFG.rows_held[1] + CFG.rows_held[2] and self.rows2[2] == CFG.pile_rows  # no overlap
        assert self.label_saves == tuple(sorted(self.label_saves)) and self.label_saves[-1] == self.label_steps
        assert self.label_steps == CFG.copy_steps and 1 <= self.move_x0 < self.move_x_max


AM = Amend2()
ROWS2 = ("trained", "tt_lora_s0", "tt_lora_s1")  # the states read on the second rows: the Adam control, the two LoRAs


def config() -> dict[str, Any]:
    return json.loads(json.dumps({"amend2": asdict(AM), "run": asdict(CFG)}))


def save(job: Job, name: str, st: dict[str, tuple[Tensor, Tensor, Tensor]], meta: dict[str, Any]) -> None:
    from commute.arms import save_offsets

    path = LARGE / f"a2_{name}.pt"
    save_offsets(path, job.ctx.uv_cpu, st, meta)
    log(f"[saved] {path.name}")


def split_state(st: dict[str, tuple[Tensor, Tensor, Tensor]]) -> tuple[dict[str, Tensor], UV]:
    return {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}


def teach(job: Job, out: dict[str, Any], name: str, teacher: str, order: int) -> tuple[dict[str, Tensor], UV]:
    """E37's teacher training on the fine-tune `teacher` at batch order `order`, saved and read as `name`."""
    from commute.row_copy import copy_on_rows

    t0 = time.time()
    curve, st = copy_on_rows(job.ctx, job.refs[teacher].teacher, job.lower, CFG.copy_lr, CFG.copy_batch, CFG.copy_steps,
                             order, log)
    kl = {"train_kl_step1": curve[0]["loss"], "train_kl_end": sum(c["loss"] for c in curve[-50:]) / 50, "order": order}
    log(f"[{name}] trained in {time.time() - t0:.0f} s; {json.dumps(kl)}")
    save(job, name, st, {"teacher": teacher, "order": order, "step": CFG.copy_steps})
    dW, uv2 = split_state(st)
    read(job, out, name, teacher, dW, uv2, kl)
    return dW, uv2


# ----------------------------------------------------------------------------- stage 0


def stage0(job: Job, out: dict[str, Any]) -> tuple[dict[str, Tensor], UV]:
    stage0_w0(job, out)
    dW_tr, uv2_tr = stage0_trained(job, out)
    rec = read(job, out, "nothing_adam", "adam", job.dW_adam, None)
    c = out["stage0"]
    c["iii"] = rec["check"] >= CFG.fail_line and abs(rec["check"] / AM.e37_nothing - 1) <= CFG.check_tol
    log(f"[stage0 iii] do-nothing check {rec['check']:.4f} (E37 {AM.e37_nothing}) pass {c['iii']}")
    if not c["iii"]:
        halt(out, "stage 0 (iii): the Adam fine-tune's do-nothing does not read as in E37")
    return dW_tr, uv2_tr


# ----------------------------------------------------------------------------- (D) a random move matched in damage


def match(f: Any, target: float) -> tuple[float, Any, list[tuple[float, float]]]:
    """A scale x with f(x)[0] (the damage, rising with x) within AM.damage_tol of `target`: from AM.move_x0, double x
    while below the target (up to AM.move_x_max), halve it while above with nothing below found yet, then bisect in
    log x between the largest x below and the smallest above. At most AM.move_evals calls of f. Returns the last x
    tried, f(x) there, and the trail of (x, damage), matched or not (the caller checks the tolerance)."""
    trail: list[tuple[float, float]] = []
    lo: float | None = None
    hi: float | None = None
    x = AM.move_x0
    while True:
        y = f(x)
        trail.append((x, float(y[0])))
        if abs(y[0] / target - 1) <= AM.damage_tol or len(trail) == AM.move_evals:
            return x, y, trail
        if y[0] < target:
            lo = x
        else:
            hi = x  # a non-finite damage lands here too
        if hi is None and x >= AM.move_x_max:
            return x, y, trail
        x = min(2 * x, AM.move_x_max) if hi is None else x / 2 if lo is None else (lo * hi) ** 0.5


def moves(job: Job, out: dict[str, Any], dW_tr: dict[str, Tensor]) -> None:
    """(D): per noise seed, the E39 random move (each matrix's size proportional to the trained state's ||delta-W||)
    scaled by one x until its damage on the 64 plain Pile rows matches the trained decomposition's; read if matched."""
    from commute.fragility import delta_w, random_move
    from commute.row_copy import read_state

    uv, target = job.ctx.uv_cpu, out["reads"]["trained"]["damage"]
    base = {k: float(dW_tr[k].double().norm()) for k in dW_tr}
    out["moves"] = {}
    for seed in AM.move_seeds:
        def at(x: float, seed: int = seed) -> tuple[float, dict[str, Tensor], UV]:
            uv2 = random_move(uv, {k: x * n for k, n in base.items()}, seed)
            dW = delta_w(uv, uv2)
            d = read_state(job.ctx, job.refs["adam"], dW, None, False, log)["damage"]
            log(f"[move_s{seed}] x {x:.4g} damage {d:.5g} (target {target:.5g})")
            return d, dW, uv2

        x, (d, dW, uv2), trail = match(at, target)
        ok = math.isfinite(d) and abs(d / target - 1) <= AM.damage_tol
        out["moves"][str(seed)] = {"x": x, "damage": d, "target": target, "matched": ok, "trail": trail}
        dump(out)
        log(f"[move_s{seed}] matched {ok}: x {x:.4g} damage {d:.5g} target {target:.5g} after {len(trail)} reads")
        if ok:
            read(job, out, f"move_s{seed}", "adam", dW, uv2, {"x": x, "damage_miss": d / target - 1})


# ----------------------------------------------------------------------------- (A) component training with labels


def label(job: Job, out: dict[str, Any], opt: str, lrs: tuple[float, ...]) -> None:
    """(A): `component_training.train` from W0's alive (U, V) with the loss next-token cross-entropy on the lowercase
    rows (`finetune.next_token_ce`), optimiser `opt` at each rate, E37's batch and batch order, states at
    AM.label_saves. Each state's held-out gain, gap and damage are read; among states not diverged (a non-finite
    training loss up to that step, or a gain not above 0) the one whose gain is nearest the Adam fine-tune's is saved
    and read in full. Ties go to the earlier rate, then the earlier step."""
    from commute import component_training
    from commute.finetune import ArmAConfig, next_token_ce
    from commute.row_copy import read_state

    ctx, target = job.ctx, out["stage0"]["gain"]
    rec: dict[str, Any] = {"grid": {}, "chosen": None}
    out.setdefault("label", {})[opt] = rec
    best: tuple[float, float, int, float, Any] | None = None  # (|gain - target|, lr, step, gain, state)
    for lr in lrs:
        a_cfg = ArmAConfig(o_token=ctx.token, batch=CFG.copy_batch, steps=AM.label_steps, checkpoints=(AM.label_steps,),
                           seed=CFG.copy_seed)
        cfg = component_training.TrainConfig(lr=lr, steps=AM.label_steps, save_steps=AM.label_saves, optimizer=opt)
        t0 = time.time()
        curve, states = component_training.train(ctx.model, ctx.uv, job.lower, ctx.held, a_cfg, cfg, ctx.device, ctx.W0,
                                                 log=log, loss_fn=next_token_ce)
        log(f"[label_{opt} lr {lr:g}] {AM.label_steps} steps in {time.time() - t0:.0f} s")
        for step in AM.label_saves:
            q = read_state(ctx, job.refs["adam"], split_state(states[step])[0], None, False, log)
            g = q["logp"] - job.logp_w0
            div = not (all(math.isfinite(c["loss"]) for c in curve[:step]) and math.isfinite(g) and g > 0)
            rec["grid"][f"{lr:g}@{step}"] = {"gain": g, "gap": q["gap"], "damage": q["damage"], "diverged": div}
            log(f"[label_{opt} lr {lr:g} step {step}] gain {g:.4f} gap {q['gap']:.4f} damage {q['damage']:.4f} "
                f"diverged {div}")
            if not div and (best is None or abs(g - target) < best[0]):
                best = (abs(g - target), lr, step, g, states[step])
        del states
    rec["done"] = True
    dump(out)
    if best is None:
        log(f"[label_{opt}] every rate and step diverged")
        return
    _, lr, step, g, st = best
    rec |= {"chosen": {"lr": lr, "step": step}, "gain": g}
    log(f"[label_{opt}] chosen lr {lr:g} step {step}: gain {g:.4f} (Adam fine-tune {target:.4f})")
    save(job, f"label_{opt}", st, {"loss": "labels", "optimizer": opt, "lr": lr, "step": step})
    dW, uv2 = split_state(st)
    read(job, out, f"label_{opt}", "adam", dW, uv2, {"lr": lr, "step": step})


# ----------------------------------------------------------------------------- (C) the LoRA, a second seed, second rows


def lora_delta(job: Job, seed: int) -> dict[str, Tensor]:
    """E39's rank-8 LoRA fine-tune on all 24 matrices (`run.lora_ft`) at AM.lora_lr with LoRA seed `seed`."""
    from commute.finetune import LoraConfig, finetune_lora_arm_b
    from commute.vpd import module_keys

    cfg = LoraConfig(modules=tuple(module_keys(job.ctx.model)), rank=CFG.lora_rank, lr=AM.lora_lr, steps=CFG.ft_steps,
                     checkpoints=(CFG.ft_steps,), seed=seed)
    ft = finetune_lora_arm_b(job.ctx.model, job.lower, job.held, cfg, CFG.ft_batch, job.ctx.device, log=log)
    return ft.deltas[CFG.ft_steps]


def rows2_context(job: Job) -> Any:
    """The harness context on the second set of Pile rows: the same model and state, blocks of 128 rows (batches of 8,
    as `faithfulness.pile_blocks`), and W0 read there on adversary seeds 0-2 (the ratio's reference and its spread)."""
    from commute.data import batches, pile_rows
    from commute.faithfulness import pile_faithfulness

    rows = pile_rows(*AM.rows2, log=log)
    ctx2 = replace(job.ctx, blocks=[batches(rows[s : s + 128], 8) for s in range(0, rows.shape[0], 128)], pile={})
    for seed in CFG.pile_seeds:
        t0 = time.time()
        ctx2.pile[seed] = pile_faithfulness(ctx2.model, ctx2.blocks, seed, CFG.pile_pgd_steps, log)
        log(f"[rows2 W0 seed {seed}] {json.dumps({k: round(v, 4) for k, v in ctx2.pile[seed].items()})} "
            f"{time.time() - t0:.0f} s")
    return ctx2


def lora(job: Job, out: dict[str, Any], dW_tr: dict[str, Tensor], uv2_tr: UV) -> None:
    from commute import harness

    states = {"trained": ("adam", dW_tr, uv2_tr)}
    for seed, order in AM.lora_runs:
        name = f"lora_s{seed}"
        g = job.add_teacher(name, lora_delta(job, seed))
        out["shape"][name] = {"gain": g, "lr": AM.lora_lr, "seed": seed, "order": order}
        log(f"[{name}] held-out gain {g:.4f} (E39 seed 0: {AM.e39_lora_gain})")
        states[f"tt_{name}"] = (name, *teach(job, out, f"tt_{name}", name, order))
    ctx2 = rows2_context(job)
    out["rows2"] = {"rows": list(AM.rows2), "w0_pile": ctx2.pile, "spread": harness.pgd_floor(ctx2)}
    log(f"[rows2] W0 spread over adversary seeds 0-2: {out['rows2']['spread']:.4f}")
    dump(out)
    ctx1 = job.ctx
    try:
        job.ctx = ctx2  # run.read reads the state through job.ctx
        for name in ROWS2:
            teacher, dW, uv2 = states[name]
            read(job, out, f"{name}_rows2", teacher, dW, uv2)
    finally:
        job.ctx = ctx1


# ----------------------------------------------------------------------------- (B) SGD held to rank 8


def capped_ft(job: Job, lr: float) -> Any:
    from commute.finetune import ArmBConfig
    from commute.rank_cap import finetune_arm_b_rank_capped

    cfg = ArmBConfig(lr=lr, batch=CFG.ft_batch, steps=CFG.ft_steps, checkpoints=(CFG.ft_steps,), seed=0, optimizer="sgd")
    return finetune_arm_b_rank_capped(job.ctx.model, job.lower, job.held, cfg, AM.cap_rank, job.ctx.device, log)


run.FINETUNE["sgd_r8"] = (capped_ft, AM.cap_lrs)  # run.pick_rate's grid and rate rule, for the capped arm


def cap(job: Job, out: dict[str, Any]) -> None:
    """(B): the capped SGD grid and rate rule (`run.pick_rate`), then the plain SGD update at Amendment 1's rate for
    its share beyond rank 8 (how much the cap removes), then teacher training on the capped fine-tune."""
    from commute.rank_cap import tail_share

    def tail(dW: dict[str, Tensor]) -> dict[str, float]:
        return tail_share({k: v.to(job.ctx.device) for k, v in dW.items()}, AM.cap_rank)

    dW = pick_rate(job, out, "sgd_r8")
    rec = out["shape"]["sgd_r8"]
    plain = sgd_ft(job, AM.sgd_ref_lr).deltas[CFG.ft_steps]
    rec |= {"plain_lr": AM.sgd_ref_lr, "plain_gain": job.gain(plain)[1], "plain_tail": tail(plain)}
    log(f"[sgd plain lr {AM.sgd_ref_lr:g}] gain {rec['plain_gain']:.4f} (Amendment 1 0.1367); share beyond rank "
        f"{AM.cap_rank}: {json.dumps(rec['plain_tail'])}")
    dump(out)
    if dW is None:
        return
    rec |= {"gain": job.add_teacher("sgd_r8", dW), "tail": tail(dW)}
    log(f"[sgd_r8] gain {rec['gain']:.4f}; share beyond rank {AM.cap_rank}: {json.dumps(rec['tail'])}")
    dump(out)
    teach(job, out, "tt_sgd_r8", "sgd_r8", CFG.copy_seed)


# ----------------------------------------------------------------------------- scoring (pure)


def verdict3(check: float) -> str:
    return run.verdict(check) if math.isfinite(check) else "invalid"


def outcome(check: float, rule: str) -> str:
    """'holds' or 'misses' for a filed rule: 'fail' holds iff check >= the fail line, 'pass' iff check <= the pass
    line (both inclusive); a non-finite check is unscored."""
    if not math.isfinite(check):
        return "unscored: the check is not finite"
    return "holds" if (check >= CFG.fail_line if rule == "fail" else check <= CFG.pass_line) else "misses"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules (P9-P14) and controls on what has been read so far; a prediction whose reads are missing is
    absent, one whose gate failed is 'unscored: <why>'. `tree` maps the outcomes to the plan's decision tree."""
    r, sh, sc = out.get("reads", {}), out.get("shape", {}), {}
    adam = out.get("stage0", {}).get("gain")

    def gain_ok(pid: str, g: float) -> bool:
        ratio = g / adam if adam else float("nan")
        sc[f"{pid}_gain_ratio"] = ratio
        return math.isfinite(ratio) and abs(ratio - 1) <= AM.gain_tol

    def scored(pid: str, name: str, rule: str) -> None:
        sc[f"{pid}_verdict"] = verdict3(r[name]["check"])
        sc[pid] = outcome(r[name]["check"], rule)

    mv = out.get("moves", {})
    if len(mv) == len(AM.move_seeds):
        matched = [s for s in AM.move_seeds if mv[str(s)]["matched"]]
        if not matched:
            sc["P14"] = "unscored: no noise seed matched the damage"
        elif all(f"move_s{s}" in r for s in matched):
            m = sc["P14_mean"] = sum(r[f"move_s{s}"]["check"] for s in matched) / len(matched)
            sc["P14_seeds"], sc["P14_verdict"], sc["P14"] = matched, verdict3(m), outcome(m, "fail")
    for opt, pid in (("adam", "P9"), ("sgd", "P10")):
        lab = out.get("label", {}).get(opt)
        if not lab or not lab.get("done"):
            continue
        if lab["chosen"] is None:
            sc[pid] = "unscored: every rate and step diverged"
        elif not gain_ok(pid, lab["gain"]):
            sc[pid] = f"unscored: gain not within {AM.gain_tol:.0%} of the Adam fine-tune's"
        elif f"label_{opt}" in r:
            scored(pid, f"label_{opt}", "fail")
    if "tt_lora_s0" in r:
        sc["G_C"] = (abs(sh["lora_s0"]["gain"] - AM.e39_lora_gain) <= CFG.gain_tol
                     and abs(r["tt_lora_s0"]["check"] / AM.e39_lora_check - 1) <= CFG.check_tol)
    if "tt_lora_s1" in r:
        if gain_ok("P12", sh["lora_s1"]["gain"]):
            scored("P12", "tt_lora_s1", "pass")
        else:
            sc["P12_verdict"] = verdict3(r["tt_lora_s1"]["check"])
            sc["P12"] = f"unscored: gain not within {AM.gain_tol:.0%} of the Adam fine-tune's"
    if "rows2" in out and all(f"{n}_rows2" in r for n in ROWS2):
        spread, ctrl = out["rows2"]["spread"], r["trained_rows2"]["check"]
        sc["G_rows2"] = (spread is not None and spread <= CFG.spread_max and math.isfinite(ctrl)
                         and ctrl >= CFG.fail_line)
        checks = [r[f"tt_lora_s{s}_rows2"]["check"] for s, _ in AM.lora_runs]
        sc["P13_verdicts"] = [verdict3(c) for c in checks]
        if not sc["G_rows2"]:
            sc["P13"] = "unscored: the second rows' control failed (W0 spread over 0.10, or the Adam state not failing)"
        elif not all(math.isfinite(c) for c in checks):
            sc["P13"] = "unscored: a check is not finite"
        else:
            sc["P13"] = "holds" if all(c <= CFG.pass_line for c in checks) else "misses"
    capped = sh.get("sgd_r8", {})
    if "chosen" in capped and capped["chosen"] is None:
        sc["P11"] = "unscored: every rate diverged"
    elif "tail" in capped:
        if capped["tail"]["max"] > AM.tail_max:
            sc["P11"] = f"unscored: the update is not held to rank {AM.cap_rank} (a code fault)"
        elif not gain_ok("P11", capped["gain"]):
            sc["P11"] = f"unscored: gain not within {AM.gain_tol:.0%} of the Adam fine-tune's"
        elif "tt_sgd_r8" in r:
            scored("P11", "tt_sgd_r8", "fail")
    sc["tree"] = tree(sc)
    return sc


def answer(sc: dict[str, Any], pid: str) -> str:
    """One test's outcome: 'not run', 'unscored', or the check's verdict (pass, between, fail; 'invalid' -> unscored)."""
    if pid not in sc:
        return "not run"
    v = sc.get(f"{pid}_verdict", "invalid")
    return "unscored" if sc[pid].startswith("unscored") or v == "invalid" else v


def lora_answer(sc: dict[str, Any]) -> str:
    """(C) as one outcome: 'stands' (seed 1 passes, and on the second rows both pass), 'stands_rows1' (seed 1 passes,
    the second rows not scored), 'fails' (seed 1 fails, or a LoRA state fails on the second rows), else 'unclear'."""
    a = answer(sc, "P12")
    if a in ("not run", "unscored"):
        return a
    rows = sc["P13_verdicts"] if sc.get("P13") in ("holds", "misses") else None
    if a == "fail" or (rows is not None and "fail" in rows):
        return "fails"
    if a == "pass" and rows is None:
        return "stands_rows1"
    if a == "pass" and rows is not None and all(v == "pass" for v in rows):
        return "stands"
    return "unclear"


def tree(sc: dict[str, Any]) -> dict[str, str]:
    """The decision tree's reading for each test and for (A) with (C) together (READINGS, filed with the predictions)."""
    a, c = answer(sc, "P9"), lora_answer(sc)
    a_bin = a if a in ("pass", "fail") else "other"
    c_bin = "stands" if c.startswith("stands") else "fails" if c == "fails" else "other"
    return {"A_adam": READINGS["A_adam"][a], "A_sgd": READINGS["A_sgd"][answer(sc, "P10")],
            "B": READINGS["B"][answer(sc, "P11")], "C": READINGS["C"][c], "D": READINGS["D"][answer(sc, "P14")],
            "A_and_C": READINGS["A_and_C"][f"{a_bin}/{c_bin}"]}


run.score = score  # run.read scores with this module's rules


# ----------------------------------------------------------------------------- main


STAGES = ("move", "label_adam", "label_sgd", "lora", "cap")


def main(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (amend2.py --predictions)")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == config(), f"predictions were filed for another config: {filed}"
    for f in (TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt"):
        if not f.is_file():
            sys.exit(f"missing input (gitignored): {f}")
    job = Job(device)
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": job.card, "reads": {}, "shape": {}, "done": []}
    if out["card"] != job.card:
        sys.exit(f"{OUT} was written on {out['card']}; this job runs on {job.card}. Move the file aside to start again")
    out["status"] = "running"
    dW_tr, uv2_tr = stage0(job, out)
    fns = {"move": lambda: moves(job, out, dW_tr), "label_adam": lambda: label(job, out, "adam", AM.label_adam_lrs),
           "label_sgd": lambda: label(job, out, "sgd", AM.label_sgd_lrs),
           "lora": lambda: lora(job, out, dW_tr, uv2_tr), "cap": lambda: cap(job, out)}
    for name in STAGES:
        if name in out["done"]:
            continue
        t0 = time.time()
        fns[name]()
        out["done"].append(name)
        out["score"] = score(out)
        dump(out)
        log(f"[score after {name}, {time.time() - t0:.0f} s] "
            + json.dumps({k: v for k, v in out["score"].items() if k != "tree"}, default=float))
        torch.cuda.empty_cache()
    out["status"] = "done"
    dump(out)
    log("[tree] " + json.dumps(out["score"]["tree"], indent=1))
    log(f"FINISHED wrote {OUT}")


# ----------------------------------------------------------------------------- predictions


READINGS: dict[str, dict[str, str]] = {
    "A_adam": {
        "pass": "Component training with labels (Adam) gives a gain-matched decomposition that passes: the frozen-Gamma "
                "decomposition can hold this broad behaviour faithfully, so E37/E39's fail belongs to teacher training's "
                "copying. Next: compare the labelled and teacher-trained states on the saved offsets (class 1); a fix "
                "to teacher training is a new plan for the user.",
        "fail": "Component training with labels (Adam) fails too: trained on this behaviour by labels or by copying, the "
                "subcomponents give a decomposition that fails the check, so the fail is not teacher training's alone "
                "(GOALS' 'brittle' branch, for this task and these recipes). Read with C.",
        "between": "Adam labelled state between the lines: not separated; reported, no branch run on it.",
        "unscored": "No gain-matched Adam labelled state (diverged, or gain over 30% from the fine-tune's): A unanswered.",
        "not run": "Stage label_adam not run.",
    },
    "A_sgd": {
        "pass": "SGD component training with labels passes at the gain-matched state.",
        "fail": "SGD component training with labels fails; if Adam's passes, this is E21/E26's late SGD fail on narrow "
                "tasks again (an optimiser effect), not a broad-task one.",
        "between": "SGD labelled state between the lines: not separated.",
        "unscored": "No gain-matched SGD labelled state: the SGD half of A unanswered.",
        "not run": "Stage label_sgd not run.",
    },
    "B": {
        "pass": "SGD held to rank 8 passes where plain SGD failed (Amendment 1, 2.07): the hard rank cap is enough. If C "
                "stands, the cap explains the LoRA's pass; next (new plan, user call): the LoRA at higher rank.",
        "fail": "SGD held to rank 8 fails like plain SGD: the cap does not explain the LoRA's pass. If C stands, next "
                "(this plan's branch, ~$1.5): a rank-8 LoRA trained with plain SGD, teacher-trained and read, to tell "
                "the LoRA's factorisation from its Adam optimiser.",
        "between": "Capped SGD between the lines: not separated; no branch run on it.",
        "unscored": "No gain-matched capped state, or the cap did not hold (a code fault, fixed first): B unanswered.",
        "not run": "Stage cap not run.",
    },
    "C": {
        "stands": "The LoRA's pass replicates at a second seed and on a second set of Pile rows: it is a result, and B "
                  "asks why.",
        "stands_rows1": "Seed 1 passes on the check's rows; the second rows are not scored: the pass replicates on one "
                        "set of rows only.",
        "fails": "The LoRA's pass does not replicate: E39's 1.27 was one seed and one set of rows. B is then a fact about "
                 "SGD only, and 'why does the LoRA pass' closes.",
        "unclear": "Seed 1 or the second rows land between the lines: the pass is not robust enough to explain; reported, "
                   "a third seed is the user's call.",
        "unscored": "The seed-1 LoRA's gain is over 30% from the Adam fine-tune's: C unanswered.",
        "not run": "Stage lora not run.",
    },
    "D": {
        "fail": "A random move with the trained decomposition's damage fails: the check fails any move of the "
                "subcomponents that changes plain-text outputs this much, so the size of the output change, not "
                "training's directions, can explain the broad fail (the LoRA's copy, damage 0.12, then passes by being "
                "smaller or better placed).",
        "pass": "A random move with the same damage passes: the fragility is specific to the directions training "
                "chooses, not to the size of the output change.",
        "between": "The matched random move lands between the lines: not separated.",
        "unscored": "No noise seed reached the trained decomposition's damage within 5%: D unanswered.",
        "not run": "Stage move not run.",
    },
    "A_and_C": {
        "pass/stands": "Labels pass and the LoRA's copy passes reliably: the decomposition holds broad behaviour; teacher "
                       "training's copy of full-matrix fine-tunes is what fails.",
        "pass/fails": "Labels pass, the LoRA's copy does not replicate: the decomposition can hold the behaviour, and "
                      "teacher training is unreliable on broad updates generally.",
        "pass/other": "Labels pass; the LoRA's copy is unresolved: teacher training is at fault for the Adam fine-tune.",
        "fail/stands": "Labels fail, the LoRA's copy passes reliably: the decomposition holds a LoRA's broad update but "
                       "not the full-matrix behaviour under either loss; the update's form matters, B says if it is rank.",
        "fail/fails": "Nothing broad passes reliably: on this task the frozen-Gamma decomposition holds no broad update "
                      "by any map tried. The 'brittle' branch for broad updates: a card for the user (Gamma refit, or "
                      "accept and scope the weak version to narrow updates).",
        "fail/other": "Labels fail; the LoRA's copy is unresolved: read A as the decomposition's limit on this behaviour.",
        "other/stands": "A not separated; the LoRA's pass stands: read B.",
        "other/fails": "A not separated; the LoRA's pass does not replicate.",
        "other/other": "Neither A nor C separated: report the numbers; the user decides whether to continue.",
    },
}

NOISE = ("W0's spread over adversary seeds 0-2: 0.0748 on E39's H100 (read again in stage 0; the second rows' spread is "
         "read in stage lora); the pass and fail lines 1.40 and 1.61 bracket 1.5 by that floor; E37's batch-order "
         "spread of the check, 14% (2.20 and 1.88); the held-out gain reproduced to 0.0001 across jobs")


def pred(pid: str, rule: str, expected: str, confidence: str, blind: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "expected": expected, "confidence": confidence, "blind_to": blind,
            "positive_control": control, "noise_floor": NOISE}


GATE = f"scored only if the state's held-out gain is within {AM.gain_tol:.0%} of this job's Adam fine-tune gain (relative)"


def predictions() -> dict[str, Any]:
    e39 = json.loads(run.PRED.read_text())
    return {
        "plan": "E39 Amendment 2, reports/plans/2026-10-27_plan_broad-fragility.md -> 'Amendment 2 (2026-10-28)'. "
                "Filed before any run; never edited after. P1-P8 are E39's and Amendment 1's; this file holds P9-P14.",
        "definitions": e39["definitions"] | {
            "verdict": f"pass if check <= {CFG.pass_line}, fail if >= {CFG.fail_line}, between otherwise (both lines "
                       "inclusive); a non-finite check leaves its prediction unscored",
            "damage": "KL(W0 || state), mean over every position of 64 held-out plain Pile val rows (E37's)",
            "gain": "held-out mean next-token log-probability on 64 lowercase rows, state minus W0",
            "component training with labels": "component_training.train from W0's alive (U, V), Gamma and Delta frozen, "
                                              "loss next-token cross-entropy on E37's 2,400 lowercase rows (8 rows a step, "
                                              "drawn with replacement in batch order 0); Adam at "
                                              f"{list(AM.label_adam_lrs)}, plain SGD at {list(AM.label_sgd_lrs)}, states "
                                              f"at steps {list(AM.label_saves)}; per optimiser the state not diverged "
                                              "whose gain is nearest the Adam fine-tune's (ties: earlier rate, then step)",
            "SGD held to rank 8": "E39's plain SGD fine-tune (300 steps of 8 rows, each row once) with every matrix's "
                                  "W - W0 replaced by its best rank-8 approximation after every step "
                                  f"(rank_cap.truncate); rates {list(AM.cap_lrs)}, E39's rate rule",
            "share beyond rank 8": "per matrix the share of ||dW||_F^2 beyond the top 8 singular values; pooled = sum "
                                   "of tails over sum of norms over the 24 matrices",
            "second seed": "the rank-8 LoRA at rate 1e-4 with LoRA seed 1 (the read factor's initialisation) and teacher "
                           "training at batch order 1, both changed together",
            "second rows": f"Pile {AM.rows2[0]} rows {AM.rows2[1]}-{AM.rows2[1] + AM.rows2[2] - 1}, the check's harness "
                           "(two blocks of 128, one adversary per block, seed 0) with W0 read on them as the reference",
            "matched random move": "E39's random move (fragility.random_move: Gaussian noise on the alive U and V, each "
                                   "side at its own Frobenius norm, per matrix scaled to x times the trained state's "
                                   "||delta-W||), x found by search (double from 4, then bisect in log x, at most 12 "
                                   f"reads) until damage is within {AM.damage_tol:.0%} of the trained decomposition's "
                                   "in this job; Delta fixed; noise seeds 0 and 1",
        },
        "controls": {
            "stage0_i": e39["controls"]["stage0_i"],
            "stage0_ii": e39["controls"]["stage0_ii"],
            "stage0_iii": f"the Adam fine-tune's do-nothing: check >= {CFG.fail_line} and within {CFG.check_tol:.0%} of "
                          f"E37's {AM.e37_nothing}; else stop (positive control: the check fails a state that holds the "
                          "update in Delta)",
            "G_C": f"the LoRA at seed 0, batch order 0 reproduces E39: gain within {CFG.gain_tol} of {AM.e39_lora_gain} "
                   f"and teacher-trained check within {CFG.check_tol:.0%} of {AM.e39_lora_check}; a miss does not stop "
                   "the job, is reported at once, and C is read with it",
            "G_rows2": f"on the second rows, W0's spread over adversary seeds 0-2 <= {CFG.spread_max} and the Adam "
                       f"teacher-trained decomposition fails (check >= {CFG.fail_line}); else P13 is unscored",
            "cap": f"the capped update's largest per-matrix share beyond rank {AM.cap_rank} <= {AM.tail_max}; else P11 "
                   "is unscored (a code fault)",
        },
        "expected": {
            "W0": "check 1 by definition, s 0, damage 0, gap 0.272",
            "do-nothing (Adam)": "check 4.868, s 1, gap 0, damage 0.235 (E37)",
            "trained (Adam, teacher training)": "check 2.201, s 0.998, gap 0.012, damage 0.234 (E39)",
            "matched random move": "x 4-7; s about -0.1 to -0.5 (no lowercase gain); check 2-5 (P14)",
            "labelled Adam": "chosen step 300 or 1000 at 1e-5; gain 0.10-0.16; damage 0.15-0.25; check 1.6-2.4 (P9)",
            "labelled SGD": "gain 0.10-0.16 at some rate and step; check 1.6-3 (P10)",
            "LoRA seed 0": "gain 0.1479, teacher-trained check 1.2726 (E39; G_C)",
            "LoRA seed 1": "gain 0.14-0.16, teacher-trained check 1.15-1.45 (P12)",
            "second rows": "W0 spread 0.03-0.10; Adam trained 1.8-2.6; LoRA states 1.1-1.5 (P13)",
            "capped SGD": "chosen rate 0.01 or 0.03, gain 0.11-0.15, share beyond rank 8 below 1e-8 (rounding); "
                          "teacher-trained check 1.8-2.2 (P11)",
            "plain SGD share beyond rank 8": "0.1-0.4 pooled (its mean stable rank is 3.5): the cap removes a real part "
                                             "of the update; below 0.02 the cap would change almost nothing and P11 "
                                             "could not separate",
        },
        "predictions": [
            pred("P9", f"labelled Adam component training at its gain-matched state: check >= {CFG.fail_line}. {GATE}",
                 "fails, 1.6-2.4", "low-medium",
                 "one batch order; the state is picked from 6 by gain alone; rows drawn with replacement, unlike the "
                 "fine-tune; Gamma at W0 throughout", "the Adam teacher-trained state (2.20) and do-nothing (4.87)"),
            pred("P10", f"labelled SGD component training at its gain-matched state: check >= {CFG.fail_line}. {GATE}",
                 "fails, 1.6-3", "medium-low",
                 "as P9, picked from 9; E21/E26's SGD component training failed late on narrow tasks too, so a fail is "
                 "not broad-specific", "as P9"),
            pred("P11", f"teacher training on SGD held to rank 8 per matrix: check >= {CFG.fail_line}. {GATE}; and only if "
                        f"the cap held (share beyond rank 8 <= {AM.tail_max} in every matrix)",
                 "fails like plain SGD (2.07), 1.8-2.2", "medium-low",
                 "one way of capping the rank (truncation after each step), not a factored update; one rate, one seed; "
                 "the plain SGD update's share beyond rank 8 says how much the cap removes",
                 "Amendment 1's plain SGD teacher-trained state (2.07); the LoRA's (1.27, and G_C)"),
            pred("P12", f"teacher training on the rank-8 LoRA at LoRA seed 1, batch order 1: check <= {CFG.pass_line}. "
                        f"{GATE}",
                 "passes, 1.15-1.45", "medium",
                 "seed and batch order change together; one more seed only", "G_C (seed 0 reads 1.27 again)"),
            pred("P13", f"on the second rows, both LoRA teacher-trained states (seed 0 and seed 1): check <= "
                        f"{CFG.pass_line}, each; scored only if G_rows2 holds",
                 "both pass, 1.1-1.5", "medium-low",
                 "one more set of 256 rows; the LoRA's second block read 1.42 on the first rows",
                 "the Adam teacher-trained state on the same rows (must fail, G_rows2)"),
            pred("P14", f"random moves matched in damage: mean check over the matched noise seeds (0, 1) >= "
                        f"{CFG.fail_line}; unscored if no seed matched",
                 "fails, 2-5", "medium",
                 "one direction family per seed (isotropic per side, every alive subcomponent moved, rarely active ones "
                 "included); matched in mean damage on 64 rows, not in where the change falls",
                 "E39's size-matched moves (1.03, 1.07, damage 0.009); the trained state (2.20, damage 0.234)"),
        ],
        "readings": READINGS,
    }


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; filed predictions are never edited (add a dated report instead)")
    PRED.write_text(json.dumps({**predictions(), "config": config(), "filed": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=1))
    print(f"filed {PRED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions_amend2.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    file_predictions() if a.predictions else main(a.device)
