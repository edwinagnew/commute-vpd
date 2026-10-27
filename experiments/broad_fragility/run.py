"""broad_fragility (E39): where and why the decomposition after teacher training on E36's lowercase fine-tune fails the
paper's adversarial check, and whether a less spread-out update (plain SGD, a rank-8 LoRA) gives a less fragile one.
Plan: reports/plans/2026-10-27_plan_broad-fragility.md.

Settings are E37 Amendment 1's (experiments/teacher_spread): the released 67M decomposition, Gamma and Delta frozen,
the check = the Pile harness's 20-step mask adversary from one random start on 256 Pile val rows (adversary seed 0) as
a ratio over the unchanged decomposition in the same job; logp, gap and damage on E37's rows (`row_copy.read_state`);
s = (logp - logp_W0) / (logp_teacher - logp_W0), the behaviour kept. The trained state is E37's lowercase copy at batch
order 0, step 1,000, retrained in-job as E37 Amendment 1's arm 1 (`row_copy.copy_on_rows`), read in fp32 from memory
and saved as bf16 offsets (`arms.save_offsets`) for later analysis.

Stages, in order, each read written to results/broad_fragility.json as it finishes (a rerun repeats stage 0, which
builds the job, and skips the stages already done):
  stage 0  (i) W0 on adversary seeds 0-2: spread, and seed 0's PGD KL per block against E37's; (ii) E37's Adam
           fine-tune regenerated (its held-out gain), the trained state trained, saved and read; (iii) revert-all. Any miss stops.
  move     Test 1: random moves of the alive U, V with the trained state's per-matrix size (`fragility.random_move`),
           noise seeds 0 and 1 at x1, seed 0 at x0.5
  revert   Test 2: the trained state with released U, V on one layer, on the attention or the MLP matrices, or on a half
           of each matrix's alive subcomponents ranked by Gamma's mean importance on the check's rows at W0
           (rare = lower half, common = upper half, random = a random half of the lower half's size)
  adam     Test 3's reference: the Adam fine-tune's stable rank and outside share (`harness.outside_share`)
  sgd      Test 3: plain SGD on the lowercase task (E36's rows, 300 steps of 8 rows, each row once), the rate from a
           three-point grid with held-out gain nearest the Adam fine-tune's; stable rank, outside share, then the check
           of do-nothing, the scaled adjoint split and teacher training at E37's recipe (state saved)
  lora     the same with a rank-8 LoRA on all 24 matrices trained with Adam

Usage (repo root, project venv):  run.py --predictions                 files results/predictions.json (once)
                                  run.py [--device cuda]                 the run (GPU); refuses without that file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "broad_fragility.json"
TRANSPORT = HERE.parent / "transport_test" / "results"
UV = dict[str, tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class Config:
    rows_train: tuple[str, int, int] = ("train", 0, 4096)  # E37's rows, fine-tune and teacher training, unchanged
    rows_held: tuple[str, int, int] = ("val", 0, 256)
    n_held: int = 64
    ft_lr: float = 3e-5
    ft_batch: int = 8
    ft_steps: int = 300
    copy_lr: float = 1e-5
    copy_batch: int = 8
    copy_steps: int = 1000
    copy_seed: int = 0
    n_sigma_rows: int = 256  # the adjoint split's input moments: the first lowercase training rows
    pile_rows: int = 256
    pile_pgd_steps: int = 20
    pile_seeds: tuple[int, ...] = (0, 1, 2)
    eval_batch: int = 8
    e37_blocks: tuple[float, ...] = (0.6794, 0.7182)  # stage 0 (i): W0's seed-0 PGD KL per block, E37 A1's log
    block_tol: float = 0.01
    spread_max: float = 0.10
    e36_gain: float = 0.1301  # stage 0 (ii): E37's C2
    gain_tol: float = 0.005
    e37_check: float = 2.2011  # stage 0 (ii): E37's lower_s0 (and A1's arm1_1000)
    check_tol: float = 0.03
    e37_s: float = 0.998
    s_tol: float = 0.01
    moves: tuple[tuple[int, float], ...] = ((0, 1.0), (1, 1.0), (0, 0.5))  # (noise seed, size multiple)
    half_seed: int = 0
    sgd_lrs: tuple[float, ...] = (0.3, 1.0, 3.0)
    lora_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    lora_rank: int = 8
    lora_seed: int = 0
    pass_line: float = 1.40  # 1.5 / (1 + f) and 1.5 (1 + f) at E37's H100 floor f = 0.075
    fail_line: float = 1.61
    p3_max: float = 0.6
    p4_margin: float = 0.2

    def __post_init__(self) -> None:
        assert self.n_sigma_rows <= self.ft_steps * self.ft_batch <= self.rows_train[2]
        assert self.n_held <= self.rows_held[2] and self.n_held <= self.pile_rows
        assert (self.pile_rows, self.pile_pgd_steps, self.pile_seeds[0]) == (256, 20, 0)
        assert len(self.e37_blocks) == self.pile_rows // 128
        assert sorted(x for _, x in self.moves) == [0.5, 1.0, 1.0] and (0, 0.5) in self.moves
        assert (self.copy_steps, self.copy_seed) == (1000, 0)  # E37's lower_s0: its step and batch order


CFG = Config()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "run.log").open("a") as f:
        f.write(line + "\n")


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["status"] = f"stopped: {why}"
    dump(out)
    log(f"STOP: {why}")
    sys.exit(f"STOP: {why}")


class Job:
    """The model and harness (W0 on three adversary seeds), E37's rows, the W0 reference net, the input moments of
    the adjoint split, and per fine-tune the references a state is read against (`add_teacher`)."""

    def __init__(self, device: str) -> None:
        from commute import harness, split, teacher, vpd
        from commute.data import batches, pile_rows
        from commute.row_copy import lowercase_rows, mean_logp

        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        self.ctx = c = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token,
                                    1e-6, CFG.pile_rows, CFG.pile_pgd_steps, CFG.pile_seeds, log)
        self.card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        tok, n_train = vpd.tokenizer(), CFG.ft_steps * CFG.ft_batch
        self.lower = lowercase_rows(pile_rows(*CFG.rows_train, log=log), tok, n_train)
        self.held = lowercase_rows(pile_rows(*CFG.rows_held, log=log), tok, CFG.n_held)
        self.plain_held = pile_rows("val", 0, CFG.pile_rows, log=log)[: CFG.n_held]
        self.w0_net = teacher.teacher_net(c.model, {})
        self.logp_w0 = mean_logp(self.w0_net, self.held, c.device, CFG.eval_batch)
        moments = split.input_second_moments(c.model, batches(self.lower[: CFG.n_sigma_rows], CFG.eval_batch), list(c.W0))
        self.sigma = {k: v.cpu() for k, v in moments.items()}
        self.refs: dict[str, Any] = {}
        self.top: dict[str, float] = {}
        self.dW_adam: dict[str, Tensor] = {}  # E37's Adam fine-tune, regenerated in stage 0 (ii)
        self.g_alive: dict[str, Tensor] = {}  # Gamma's mean importance per alive subcomponent, test 2
        log(f"[load] {self.card}; logp_W0 {self.logp_w0:.4f}")

    def gain(self, dW: dict[str, Tensor]) -> tuple[Any, float]:
        """The fine-tune W0 + dW as a frozen net, and its held-out gain logp_teacher - logp_W0."""
        from commute import teacher
        from commute.row_copy import mean_logp

        net = teacher.teacher_net(self.ctx.model, dW)
        return net, mean_logp(net, self.held, self.ctx.device, CFG.eval_batch) - self.logp_w0

    def add_teacher(self, name: str, dW: dict[str, Tensor]) -> float:
        from commute.row_copy import Refs

        net, g = self.gain(dW)
        self.refs[name] = Refs(net, self.w0_net, self.held, self.plain_held, CFG.eval_batch)
        self.top[name] = self.logp_w0 + g
        return g


def verdict(check: float) -> str:
    return "pass" if check <= CFG.pass_line else "fail" if check >= CFG.fail_line else "between"


def read(job: Job, out: dict[str, Any], name: str, teacher: str, dW: dict[str, Tensor], uv2: UV | None,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One state through `row_copy.read_state` against fine-tune `teacher`, with s, the check and stochastic ratios
    and size; stored, scored, dumped and logged in one line."""
    from commute.row_copy import read_state

    t0, W0 = time.time(), job.ctx.W0
    rec = read_state(job.ctx, job.refs[teacher], dW, uv2, True, log)
    rec |= {"teacher": teacher, "s": (rec["logp"] - job.logp_w0) / (job.top[teacher] - job.logp_w0),
            "check": rec["pile"]["pgd_kl"], "stochastic": rec["pile"]["stochastic_kl"],
            "size": max(float(dW[k].norm() / W0[k].cpu().norm()) for k in dW), **(extra or {})}
    rec["verdict"] = verdict(rec["check"])
    out["reads"][name] = rec
    out["score"] = score(out)
    dump(out)
    log(f"[{name}] " + " ".join(f"{k} {v:.5g}" for k, v in rec.items() if isinstance(v, float | int))
        + f" verdict {rec['verdict']} | pile {json.dumps({k: round(v, 4) for k, v in rec['pile'].items()})} "
        f"| {time.time() - t0:.0f} s")
    return rec


# ----------------------------------------------------------------------------- stage 0


def stage0_w0(job: Job, out: dict[str, Any]) -> None:
    """(i): W0's spread over adversary seeds 0-2 and seed 0's PGD KL per block (`harness_pgd`, as in the check)."""
    from commute import harness
    from commute.faithfulness import harness_pgd

    ctx = job.ctx
    blocks = [harness_pgd(ctx.model, b, CFG.pile_pgd_steps, 0) for b in ctx.blocks]
    c = out["stage0"] = {"w0_pile": ctx.pile, "w0_blocks": blocks, "spread": harness.pgd_floor(ctx)}
    c["i"] = c["spread"] <= CFG.spread_max and all(abs(b / e - 1) <= CFG.block_tol
                                                   for b, e in zip(blocks, CFG.e37_blocks, strict=True))
    log(f"[stage0 i] spread {c['spread']:.4f} blocks {[round(b, 4) for b in blocks]} (E37 {CFG.e37_blocks}) pass {c['i']}")
    if not c["i"]:
        halt(out, "stage 0 (i): W0 is not at E37's readings on this card")


def stage0_trained(job: Job, out: dict[str, Any]) -> tuple[dict[str, Tensor], UV]:
    """(ii): E37's Adam fine-tune regenerated (the teacher of s and gap), then E37's teacher training on it (amend1's
    arm 1 to step 1000), saved as bf16 offsets and read in fp32 from memory."""
    from commute.arms import save_offsets
    from commute.finetune import ArmBConfig, finetune_arm_b
    from commute.row_copy import copy_on_rows

    ctx, c = job.ctx, out["stage0"]
    cfg = ArmBConfig(lr=CFG.ft_lr, batch=CFG.ft_batch, steps=CFG.ft_steps, checkpoints=(CFG.ft_steps,), seed=0)
    job.dW_adam = finetune_arm_b(ctx.model, job.lower, job.held, cfg, ctx.device, log).deltas[CFG.ft_steps]
    c["gain"] = job.add_teacher("adam", job.dW_adam)
    t0 = time.time()
    curve, st = copy_on_rows(ctx, job.refs["adam"].teacher, job.lower, CFG.copy_lr, CFG.copy_batch, CFG.copy_steps,
                             CFG.copy_seed, log)
    kl = {"train_kl_step1": curve[0]["loss"], "train_kl_end": sum(x["loss"] for x in curve[-50:]) / 50}
    log(f"[trained] {time.time() - t0:.0f} s; {json.dumps(kl)}")
    path = LARGE / f"trained_step{CFG.copy_steps}.pt"
    save_offsets(path, ctx.uv_cpu, st, {"arm": "adam", "text": "lower", "seed": CFG.copy_seed, "step": CFG.copy_steps})
    log(f"[saved] {path.name}")
    dW, uv2 = {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}
    tr = read(job, out, "trained", "adam", dW, uv2, kl)
    c["ii"] = (abs(c["gain"] - CFG.e36_gain) <= CFG.gain_tol and abs(tr["check"] / CFG.e37_check - 1) <= CFG.check_tol
               and abs(tr["s"] - CFG.e37_s) <= CFG.s_tol)
    log(f"[stage0 ii] gain {c['gain']:.4f} (E37 {CFG.e36_gain}) check {tr['check']:.4f} (E37 {CFG.e37_check}) "
        f"s {tr['s']:.4f} (E37 {CFG.e37_s}) pass {c['ii']}")
    if not c["ii"]:
        halt(out, "stage 0 (ii): the regenerated fine-tune or its teacher training does not read as in E37")
    return dW, uv2


def every_column(uv: UV) -> dict[str, Tensor]:
    return {k: torch.ones(u.shape[1], dtype=torch.bool) for k, (u, _) in uv.items()}


def stage0_revert_all(job: Job, out: dict[str, Any], uv2: UV) -> None:
    """(iii): released U, V on every subcomponent of the trained state reads as W0."""
    from commute.fragility import delta_w, revert

    uv = job.ctx.uv_cpu
    back = revert(uv, uv2, every_column(uv))
    ra = read(job, out, "revert_all", "adam", delta_w(uv, back), back)
    c = out["stage0"]
    c["iii"] = abs(ra["check"] - 1) <= c["spread"] and abs(ra["s"]) <= CFG.s_tol
    log(f"[stage0 iii] check {ra['check']:.4f} (1 +- {c['spread']:.4f}) s {ra['s']:.4f} pass {c['iii']}")
    if not c["iii"]:
        halt(out, "stage 0 (iii): revert-all does not read as W0")


# ----------------------------------------------------------------------------- tests 1 and 2


def moves(job: Job, out: dict[str, Any], dW_tr: dict[str, Tensor]) -> None:
    """Test 1: per matrix the trained state's ||delta-W|| (times the multiple) as a random move of the alive U, V."""
    from commute.fragility import delta_w, random_move

    uv = job.ctx.uv_cpu
    for seed, x in CFG.moves:
        norms = {k: x * float(dW_tr[k].double().norm()) for k in dW_tr}
        uv2 = random_move(uv, norms, seed)
        dW = delta_w(uv, uv2)
        miss = max(abs(float(dW[k].double().norm()) / t - 1) for k, t in norms.items() if t > 0)
        read(job, out, f"move_s{seed}_x{x:g}", "adam", dW, uv2, {"norm_miss": miss})


def revert_groups(job: Job) -> dict[str, dict[str, Tensor]]:
    """Test 2's subsets: each layer, the attention and the MLP matrices (every alive column), and Gamma's halves."""
    from commute import vpd
    from commute.fragility import importance_halves

    ctx = job.ctx
    every = every_column(ctx.uv_cpu)
    groups = {f"layer{n}": {k: m for k, m in every.items() if vpd.layer_of(k) == n}
              for n in sorted({vpd.layer_of(k) for k in every})}
    groups["attn"] = {k: m for k, m in every.items() if ".attn." in k}
    groups["mlp"] = {k: m for k, m in every.items() if ".mlp." in k}
    assert len(groups["attn"]) + len(groups["mlp"]) == len(every)
    g = vpd.mean_ci(ctx.model, [b for blk in ctx.blocks for b in blk])
    job.g_alive = {k: g[k][ctx.alive[k]] for k in every}
    return groups | importance_halves(job.g_alive, CFG.half_seed)


def reverts(job: Job, out: dict[str, Any], uv2: UV) -> None:
    """Test 2: the trained state with each subset of `revert_groups` back at the released U, V."""
    from commute.fragility import delta_w, revert

    uv, groups = job.ctx.uv_cpu, revert_groups(job)
    out["n_layers"] = sum(n.startswith("layer") for n in groups)
    out["importance"] = {h: sum(float(job.g_alive[k][m].sum()) for k, m in groups[h].items())
                         / sum(int(m.sum()) for m in groups[h].values()) for h in ("rare", "common", "random")}
    log(f"[importance] mean g per half on the check's rows at W0: {json.dumps(out['importance'])}")
    for name, cols in groups.items():
        back = revert(uv, uv2, cols)
        read(job, out, f"revert_{name}", "adam", delta_w(uv, back), back, {"n_reverted": sum(int(m.sum()) for m in cols.values())})


# ----------------------------------------------------------------------------- test 3


def shape(job: Job, dW: dict[str, Tensor]) -> dict[str, Any]:
    from commute import harness
    from commute.fragility import mean_stable_rank

    o = harness.outside_share_of(dW, job.ctx.bases, job.sigma, job.ctx.non_saturated)  # the lowercase rows, not ctx.sigma
    return {"stable_rank": mean_stable_rank(dW), "outside_share": o["pooled"], "outside_null": o["pooled_null"]}


def adam_shape(job: Job, out: dict[str, Any]) -> None:
    out["shape"]["adam"] = shape(job, job.dW_adam) | {"gain": out["stage0"]["gain"]}
    log(f"[adam] stable rank {out['shape']['adam']['stable_rank']['mean']:.3f} outside share "
        f"{out['shape']['adam']['outside_share']:.4f} (null {out['shape']['adam']['outside_null']:.4f})")


def sgd_ft(job: Job, lr: float) -> Any:
    from commute.finetune import ArmBConfig, finetune_arm_b

    cfg = ArmBConfig(lr=lr, batch=CFG.ft_batch, steps=CFG.ft_steps, checkpoints=(CFG.ft_steps,), seed=0, optimizer="sgd")
    return finetune_arm_b(job.ctx.model, job.lower, job.held, cfg, job.ctx.device, log)


def lora_ft(job: Job, lr: float) -> Any:
    from commute.finetune import LoraConfig, finetune_lora_arm_b
    from commute.vpd import module_keys

    cfg = LoraConfig(modules=tuple(module_keys(job.ctx.model)), rank=CFG.lora_rank, lr=lr, steps=CFG.ft_steps,
                     checkpoints=(CFG.ft_steps,), seed=CFG.lora_seed)
    return finetune_lora_arm_b(job.ctx.model, job.lower, job.held, cfg, CFG.ft_batch, job.ctx.device, log=log)


FINETUNE = {"sgd": (sgd_ft, CFG.sgd_lrs), "lora": (lora_ft, CFG.lora_lrs)}


def pick_rate(job: Job, out: dict[str, Any], arm: str) -> dict[str, Tensor] | None:
    """The grid; the rate whose held-out gain is nearest the Adam fine-tune's among those that did not diverge (a
    non-finite training loss, or a gain not above 0). Returns its delta-W, None if every rate diverged."""
    fn, lrs = FINETUNE[arm]
    grid, dWs = {}, {}
    for lr in lrs:
        ft = fn(job, lr)
        dWs[lr] = ft.deltas[CFG.ft_steps]
        g = job.gain(dWs[lr])[1]
        grid[lr] = {"gain": g, "eval": ft.evals[CFG.ft_steps],
                    "diverged": not (all(math.isfinite(r["loss"]) for r in ft.log) and math.isfinite(g) and g > 0)}
        log(f"[{arm} lr {lr:g}] held-out gain {g:.4f} diverged {grid[lr]['diverged']}")
    ok = [lr for lr in lrs if not grid[lr]["diverged"]]
    lr = min(ok, key=lambda r: abs(grid[r]["gain"] - out["stage0"]["gain"])) if ok else None
    out["shape"][arm] = {"grid": {f"{r:g}": v for r, v in grid.items()}, "chosen": lr}
    log(f"[{arm}] chosen rate {lr}")
    return None if lr is None else dWs[lr]


def other_update(job: Job, out: dict[str, Any], arm: str) -> None:
    """Test 3 for one arm: rate, shape, then do-nothing, the scaled adjoint split and teacher training read."""
    from commute import split
    from commute.arms import save_offsets
    from commute.row_copy import copy_on_rows

    ctx, dW = job.ctx, pick_rate(job, out, arm)
    if dW is None:
        return
    rec = out["shape"][arm]
    rec |= {"gain": job.add_teacher(arm, dW), **shape(job, dW)}
    log(f"[{arm}] gain {rec['gain']:.4f} stable rank {rec['stable_rank']['mean']:.3f} outside share "
        f"{rec['outside_share']:.4f} (null {rec['outside_null']:.4f})")
    read(job, out, f"nothing_{arm}", arm, dW, None)
    dW_adj, st = split.adjoint_state(dW, ctx.uv_cpu, job.sigma)
    read(job, out, f"adjoint_{arm}", arm, dW_adj, {k: (v[1], v[2]) for k, v in st.items()})
    t0 = time.time()
    curve, st = copy_on_rows(ctx, job.refs[arm].teacher, job.lower, CFG.copy_lr, CFG.copy_batch, CFG.copy_steps,
                             CFG.copy_seed, log)
    kl = {"train_kl_step1": curve[0]["loss"], "train_kl_end": sum(c["loss"] for c in curve[-50:]) / 50}
    log(f"[tt_{arm}] trained in {time.time() - t0:.0f} s; {json.dumps(kl)}")
    path = LARGE / f"tt_{arm}_step{CFG.copy_steps}.pt"
    save_offsets(path, ctx.uv_cpu, st, {"arm": arm, "lr": rec["chosen"], "step": CFG.copy_steps})
    log(f"[saved] {path.name}")
    read(job, out, f"tt_{arm}", arm, {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}, kl)


# ----------------------------------------------------------------------------- scoring (pure)


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on what has been read so far; a prediction whose reads are missing is not scored."""
    r, sh, sc = out["reads"], out["shape"], {}

    def ch(n: str) -> float:
        return r[n]["check"]

    if {"move_s0_x1", "move_s1_x1"} <= set(r):
        sc["move_x1_mean"] = m = (ch("move_s0_x1") + ch("move_s1_x1")) / 2
        sc["P1"] = m >= CFG.fail_line
        if "move_s0_x0.5" in r:
            sc["P2"] = ch("move_s0_x0.5") < m
    if "trained" in r:
        rt = ch("trained")
        share = sc["share_removed"] = {n.removeprefix("revert_"): (rt - ch(n)) / (rt - 1) for n in r if n.startswith("revert_")}
        layers = [v for n, v in share.items() if n.startswith("layer")]
        if layers and len(layers) == out.get("n_layers"):
            sc["P3"] = max(layers) <= CFG.p3_max
        if {"rare", "random"} <= set(share):
            sc["P4_margin"] = share["rare"] - share["random"]
            sc["P4"] = sc["P4_margin"] >= CFG.p4_margin
    arms = [a for a in ("sgd", "lora") if sh.get(a, {}).get("stable_rank")]
    for a in arms:
        sc[f"P5_{a}"] = sh[a]["stable_rank"]["mean"] < sh["adam"]["stable_rank"]["mean"]
        for p, n in (("P6", f"tt_{a}"), ("P7", f"nothing_{a}")):
            if n in r:
                sc[f"{p}_{a}"] = ch(n) >= CFG.fail_line
    for p in ("P5", "P6", "P7"):
        if f"{p}_sgd" in sc and f"{p}_lora" in sc:
            sc[p] = sc[f"{p}_sgd"] and sc[f"{p}_lora"]
    return sc


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed}"
    for f in (TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt"):
        if not f.is_file():
            sys.exit(f"missing input (gitignored): {f}")
    job = Job(device)
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": job.card, "reads": {}, "shape": {}, "done": []}
    if out["card"] != job.card:
        sys.exit(f"{OUT} was written on {out['card']}; this job runs on {job.card}. Move the file aside to start again")
    out["status"] = "running"
    stage0_w0(job, out)
    dW_tr, uv2_tr = stage0_trained(job, out)
    stage0_revert_all(job, out, uv2_tr)
    stages: list[tuple[str, Any]] = [
        ("move", lambda: moves(job, out, dW_tr)), ("revert", lambda: reverts(job, out, uv2_tr)),
        ("adam", lambda: adam_shape(job, out)), ("sgd", lambda: other_update(job, out, "sgd")),
        ("lora", lambda: other_update(job, out, "lora"))]
    for name, fn in stages:
        if name not in out["done"]:
            fn()
            out["done"].append(name)
            out["score"] = score(out)
            dump(out)
            log(f"[score after {name}] {json.dumps(out['score'], default=float)}")
            torch.cuda.empty_cache()
    out["status"] = "done"
    dump(out)
    log(f"FINISHED wrote {OUT}")


NOISE = ("W0's spread over adversary seeds 0-2 (0.075 on E37's H100, read again in stage 0); E37's batch-order spread of "
         "the check ratio, 14% (2.20 and 1.88); a share of excess removed is a difference of two such ratios, so shares "
         "under 0.2 apart are not separated")


def pred(pid: str, rule: str, expected: str, confidence: str, blind: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "expected": expected, "confidence": confidence, "blind_to": blind,
            "positive_control": control, "noise_floor": NOISE}


CHECK = "the check ratio: Pile harness PGD KL (one random start, adversary seed 0, 20 steps of 0.1, 256 val rows) over W0's"
PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-27_plan_broad-fragility.md (E39). Filed before any run; never edited after.",
    "definitions": {
        "check": f"{CHECK} at seed 0 in the same job; fixed lines: pass <= {CFG.pass_line}, fail >= {CFG.fail_line}",
        "s": "behaviour kept, (logp - logp_W0) / (logp_teacher - logp_W0), mean next-token log-probability on 64 held-out "
             "lowercase rows; the teacher is the fine-tune the state was trained from (Adam for tests 1-2)",
        "share of excess removed": "(r_trained - r_revert) / (r_trained - 1), r the check ratio, r_trained this job's",
        "stable rank": "||dW||_F^2 / ||dW||_2^2 per matrix, mean over the 24 matrices",
        "random move": "per matrix, Gaussian noise on the alive U and V, each side's noise rescaled to that side's "
                       "Frobenius norm, one scale per matrix solved so ||U'V'^T - UV^T|| = the multiple times the "
                       "trained state's ||delta-W|| (so the relative change matches); Delta fixed",
        "rate rule": "per arm, among grid rates that did not diverge (a non-finite training loss or a held-out gain not "
                     "above 0) the one whose held-out gain is nearest the Adam fine-tune's in this job; none: that "
                     "arm's P5-P7 are not scored",
    },
    "controls": {
        "stage0_i": f"first, alone: W0's PGD spread over seeds 0-2 <= {CFG.spread_max} and seed 0's PGD KL per block "
                    f"within {CFG.block_tol:.0%} of {CFG.e37_blocks}; else stop",
        "stage0_ii": f"E37's Adam fine-tune regenerated: held-out gain within {CFG.gain_tol} of {CFG.e36_gain}; E37's teacher "
                     f"training retrained in-job (amend1's arm 1 to step 1000) and read: check within {CFG.check_tol:.0%} of {CFG.e37_check} and s within "
                     f"{CFG.s_tol} of {CFG.e37_s}; else stop",
        "stage0_iii": f"revert-all: |check - 1| <= this job's W0 spread and |s| <= {CFG.s_tol}; else stop",
    },
    "expected": {
        "W0": "check 1 by definition, s 0, damage 0, gap about 0.272",
        "trained": "check 2.20, stochastic 1.09, s 0.998, gap 0.012, damage 0.234 (E37)",
        "revert_all": "as W0: check 1 within the spread, s 0, damage 0; share of excess removed 1",
        "moves": "s about 0 (a random move carries no lowercase gain); x1 check 1.8-3 (P1); x0.5 below x1 (P2)",
        "reverts": "each layer removes 0.1-0.5 of the excess (P3); rare removes 0.2 more than random (P4)",
        "do-nothing (sgd, lora)": "check >= 1.61 (E37's Adam do-nothing read 4.87)",
        "adjoint (sgd, lora)": "check about 1.0-1.2 (E37's Adam scaled adjoint split read 1.05), s below 1",
        "stable rank": "LoRA <= 8 by construction; SGD below Adam's; Adam's has no reference",
    },
    "predictions": [
        pred("P1", f"random move x1, mean check over noise seeds 0 and 1 >= {CFG.fail_line}", "fails, 1.8-3", "medium-low",
             "other move shapes (the noise is isotropic per side); one adversary start", "trained state, revert-all"),
        pred("P2", "random move x0.5 (seed 0, the same direction): check below the x1 mean", "yes", "medium",
             "as P1", "revert-all reads 1 (size 0)"),
        pred("P3", f"no single layer's revert removes more than {CFG.p3_max} of the excess", "spread over layers", "low",
             "reverts are not additive; matrices inside a layer", "revert-all removes 1"),
        pred("P4", f"share removed by reverting the rare half minus that of the random half >= {CFG.p4_margin}",
             "claim B holds", "low-medium", "importance on other text; Gamma at the trained state",
             "revert-all; the common half beside it"),
        pred("P5", "stable rank (mean over matrices) of the SGD and of the LoRA fine-tune each below the Adam fine-tune's",
             "yes; the less diffuse premise", "high", "where the update acts in input space", "LoRA <= 8 by construction"),
        pred("P6", f"teacher-trained check (E37's recipe, 1000 steps) >= {CFG.fail_line} for SGD and for LoRA, each",
             "fail like Adam", "medium-low", "gain matched by held-out log-probability only; one rate, one seed",
             "E37's Adam state (stage 0 ii)"),
        pred("P7", f"do-nothing check >= {CFG.fail_line} for the SGD and the LoRA fine-tune", "fail (update in Delta)",
             "high", "as P6", "E37's Adam do-nothing (4.87)"),
    ],
    "readings": {
        "stage 0 fails": "stop; nothing scored",
        "P1 holds, random at or above the trained state": "the check punishes any move of this size: broad updates are "
                                                          "messier because a broad behaviour needs a large move (brittleness)",
        "P1 fails": "the fragility is the trained change's own; P4 says whether it is the frozen Gamma",
        "P6 fails for SGD or LoRA": "update shape matters; the stable ranks beside it answer the user's question",
    },
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; filed predictions are never edited (add a dated report instead)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    PRED.write_text(json.dumps({**PREDICTIONS, "config": asdict(CFG), "filed": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=1))
    print(f"filed {PRED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    file_predictions() if a.predictions else run(a.device)
