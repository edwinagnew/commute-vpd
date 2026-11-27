"""P2.87 (subspace_share): a reference for the log-odds re-gating share on Fig `known`'s 8 reuse tasks that, unlike P2.82's
scaled random changes, reaches the one log-odds gain the share needs. Each update is trained only inside a fixed random
subspace (`commute.subspace`: U Z V^T per decomposed matrix, U and V random orthonormal, Z trained from 0, k^2 nearest a
rank-8 LoRA's count) with the task's own fine-tune recipe, and read exactly like the fine-tunes. Descriptive: values per
task, no verdict. Plan reports/plans/2026-11-26_plan_p2-87-subspace-share.md; card card.md; settings and expected values
results/predictions.json (the run refuses unless it is committed unchanged).

Reader: P2.82's (`random_logodds_share.Reader`: E136's `share_untrained` Job.trained, E228's KL on the same rows); states,
inventory and control are P2.82's, loaded, not copied. Held-out gain during training: the reader's own T (mean log-odds of
the trained answer on the reader's windows, `vpd.compute_ci`'s forward and batches) at W0 + the bf16-rounded update,
minus W0's, so the gain an eval reads is the gain the reader reads on the saved file.
Recipes (`recipe`): the six kv tasks and e62/copy are E12's arm A as `keydep.finetune` runs it (Adam 3e-5, batch 32,
KL weight 1, clip 1, 300 steps, batches drawn by the seed); e53/lowercase is E53's arm B (`lowercase.arm_b_states`: Adam
3e-5, batch 8, 300 steps, each lowercased Pile row once in batch order `seed`). Only the parameterisation and the lr
change; the lr is the recipe's times one multiplier from the `lrgrid` stage.

  control         alone, first -> results/control.json: P2.82's control on the 24 reuse states (a zero change reads gain,
                  numerator and KL exactly 0 and share nan; each fine-tune's share equals known3.json). Exit 1 on a miss.
  lrgrid          e62/copy seed 0 at each filed multiplier -> results/lrgrid.json; picks the multiplier (`pick`).
  train TASK...   per task and seed -> results/updates/<task>_s<seed>.json and results/large/<task>_s<seed>.pt.

    sbatch --comment=E### cluster/subspace_share.sbatch control
    sbatch --comment=E### cluster/subspace_share.sbatch lrgrid
    sbatch --comment=E### cluster/subspace_share.sbatch train e62/copy e46/ident32 e54/elicit c3/rank1   # and the rest
A stop prints "STOP: ..." and exits 1; a rerun skips finished items.
"""

from __future__ import annotations

import copy
import json
import math
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import finetune as FT
from commute import keydep as KD
from commute import lowercase as LC
from commute import subspace as SS
from commute.data import LAST, target_tokens
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable
from commute.vpd import target_linear

RL: Any = load_experiment("random_logodds_share", "run")  # P2.82: states, inventory, Reader, control
HERE, REPO = Path(__file__).resolve().parent, EXPERIMENTS.parent
RESULTS = HERE / "results"
PRED, CONTROL, GRID = RESULTS / "predictions.json", RESULTS / "control.json", RESULTS / "lrgrid.json"
UPDATES, LARGE = RESULTS / "updates", RESULTS / "large"
DW = dict[str, Tensor]
log = RL.log


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    seeds: tuple[int, ...]
    subspace_seed: int  # the subspace of training seed s is drawn from subspace_seed + s
    lora_rank: int
    lr_mults: tuple[float, ...]
    grid_task: str
    grid_seed: int
    reach_by: int  # the pick rule's first bar (steps)
    steps: int
    target_gain: float
    loss_window: int
    kl_batch: int

    def __post_init__(self) -> None:
        assert self.target_gain == MIN_GAIN, "at-1 is the share's own floor"
        assert self.steps == KD.E12_CHECKS[-1] == LC.STEPS, "every recipe runs 300 steps"
        assert list(self.lr_mults) == sorted(set(self.lr_mults)) and 0 < self.reach_by <= self.steps
        assert 2 * self.loss_window <= self.steps and len(set(self.seeds)) == len(self.seeds)


def load_config() -> tuple[Config, str]:
    pred = json.loads(PRED.read_text())
    assert pred["experiment"] != "EXXX", "claim the E-number (fleet/ecode) and write it into predictions.json first"
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["experiment"]


def reuse_states() -> list[str]:
    """Fig `known`'s 24 reuse states (8 tasks x 3 seeds; e54/elicit is the reuse side of two pairs), known3.json's order."""
    pairs, _ = RL.figure_states()
    return list(dict.fromkeys(s for p in pairs.values() for s in p["reuse"]))


def task_of(state: str) -> str:
    return state.rsplit("_s", 1)[0]


def bf16(d: DW) -> DW:
    return {k: v.bfloat16().float() for k, v in d.items()}


def passed(path: Path, experiment: str, key: str) -> dict[str, Any]:
    rec = json.loads(path.read_text()) if path.is_file() else {}
    if rec.get("experiment") != experiment or key not in rec:
        log(f"STOP: {path.name} has no '{key}' for {experiment}: run the stage before it first")
        hard_exit(1)
    return rec


# ---- the recipes ----------------------------------------------------------------------------------------------------


@dataclass
class Recipe:
    loss: Callable[[int], tuple[Tensor, dict[str, float]]]
    lr: float  # the fine-tune's
    grad_clip: float
    source: str


class Trainer:
    """The reader, its model, a W0 copy for arm A's KL term, and each matrix's k."""

    def __init__(self, states: list[str], cfg: Config) -> None:
        self.cfg, self.rd = cfg, RL.Reader(states, cfg.kl_batch)
        self.model, self.W0 = self.rd.job.model.requires_grad_(False), self.rd.job.W0  # only Z trains
        self.net, self.dev = self.model.target_model, self.rd.job.device
        self.ref = copy.deepcopy(self.net).eval().requires_grad_(False)  # before any hook, so it has none
        self.linears = {k: target_linear(self.model, k) for k in self.W0}
        self.ks = {k: SS.k_like_lora(tuple(w.shape), cfg.lora_rank) for k, w in self.W0.items()}
        self.T0: dict[str, float] = {}
        self._low: LC.Lower | None = None

    def sizes(self) -> dict[str, Any]:
        lora = sum(self.cfg.lora_rank * sum(w.shape) for w in self.W0.values())
        return {"k": self.ks, "trainable": sum(k * k for k in self.ks.values()), "lora_trainable": lora,
                "full": sum(w.numel() for w in self.W0.values())}  # fmt: skip

    def recipe(self, task: str, seed: int) -> Recipe:
        if task == "e53/lowercase":
            if self._low is None:
                self._low = LC.task(self.model, self.dev, log)
            rows, b = LC.batch_order(self._low.rows, seed), LC.BATCH
            return Recipe(lambda step: FT.next_token_ce(self.net, rows[(step - 1) * b : step * b].to(self.dev)), 3e-5,
                          FT.ArmBConfig().grad_clip, "lowercase.arm_b_states('adam', seed): Adam 3e-5, batch 8, 300 steps, "
                          "rows in batch_order(seed), clip 1 (E53, as known_3seeds.train)")  # fmt: skip
        w = self.windows(task)
        cfg = FT.ArmAConfig(o_token=w["table"], lr=AN.FT_LR, steps=self.cfg.steps, seed=seed, ci_len=w.get("ci_len"))
        gen, train = torch.Generator().manual_seed(seed), w["train"]

        def loss(step: int) -> tuple[Tensor, dict[str, float]]:  # finetune_arm_a's draw and loss
            idx = torch.randint(0, len(train), (cfg.batch,), generator=gen)
            return FT.arm_a_loss(self.net, self.ref, train.tokens[idx].to(self.dev), cfg, train.half)

        return Recipe(loss, cfg.lr, cfg.grad_clip, f"keydep.finetune (E12 arm A): Adam {cfg.lr:g}, batch {cfg.batch}, "
                      f"KL weight {cfg.kl_lambda:g}, clip {cfg.grad_clip:g}, 300 steps, ci_len {cfg.ci_len}")  # fmt: skip

    def windows(self, task: str) -> dict[str, Any]:
        """The training windows the fine-tunes were trained on: the kv file's (E12/E54/E58), or E62's with the copy
        target as a last column (induction_cue stage c)."""
        spec = self.rd.job.inv["tasks"][task]
        w = torch.load(REPO / spec["path"], weights_only=False)
        if spec["format"] == "kv":
            return w
        assert spec["format"] == "windows_pt", spec
        return {"table": LAST, "train": CP.with_target(w["train"], w["target"][spec["arm"]]["train"]), "ci_len": w["train"].tokens.shape[1]}

    @torch.no_grad()
    def mean_T(self, task: str) -> float:
        """`local_faith.route_cells`' T under the current weights: its forward (compute_ci's), batches and sums."""
        r, s = self.rd.job.reads[task], 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, r.toks.shape[0], r.batch):
                b = r.toks[i : i + r.batch].to(self.dev)
                lg = self.model(b[:, : r.ci_len], cache_type="input").output[:, r.half]
                s += float(log_odds(lg, target_tokens(r.token, b, r.half)).double().sum())
        return s / r.toks.shape[0]

    def gain(self, task: str, sub: SS.Subspace | None, d: DW) -> float:
        """Held-out gain of W0 + d (hooks off), against W0."""
        if task not in self.T0:
            self.T0[task] = self.mean_T(task)
        if sub is not None:
            sub.active = False
        try:
            with model_state(self.model, target={k: self.W0[k] + d[k] for k in self.W0}):
                return self.mean_T(task) - self.T0[task]
        finally:
            if sub is not None:
                sub.active = True


# ---- one training ---------------------------------------------------------------------------------------------------


def train_one(tr: Trainer, task: str, seed: int, mult: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """One subspace update: (record, kept states {"at1"?, "end"}: step, bf16 delta on the card, Z on the CPU)."""
    cfg, t0 = tr.cfg, time.time()
    rc = tr.recipe(task, seed)
    sub = SS.Subspace(tr.linears, tr.ks, cfg.subspace_seed + seed)
    zero = tr.gain(task, None, {k: torch.zeros_like(w) for k, w in tr.W0.items()})
    log(f"[{task} s{seed} x{mult:g}] guard step-0 gain {zero!r} (expected exactly 0)")
    if zero != 0.0:
        log(f"STOP: {task} s{seed}: the step-0 gain is {zero!r}, not 0")
        hard_exit(1)
    gains: dict[int, float] = {}
    losses: list[float] = []
    kept: dict[str, Any] = {}
    step1: dict[str, float] = {}

    def on_step(step: int, loss: float, info: dict[str, float]) -> None:
        losses.append(loss)
        d = bf16(sub.delta())
        g = gains[step] = tr.gain(task, sub, d)
        if step == 1:
            step1["norm"] = RL.norm(d)
        if step == 1 or step % 10 == 0:
            log(f"  [{task} s{seed} x{mult:g} step {step}] loss {loss:.4f} gain {g:.4f} ||dW|| {RL.norm(d):.4g} {json.dumps(info)}")
        snap = (step, d, {k: z.detach().cpu().clone() for k, z in sub.Z.items()})
        if g >= cfg.target_gain and "at1" not in kept:
            kept["at1"] = snap
        if step == cfg.steps:
            kept["end"] = kept["at1"] if kept.get("at1", (0,))[0] == step else snap

    SS.train(sub, rc.lr * mult, cfg.steps, rc.grad_clip, rc.loss, on_step)
    w = cfg.loss_window
    first, last = sum(losses[:5]) / 5, sum(losses[w - 5 : w]) / 5
    best = max(gains, key=lambda s: gains[s])
    rec = {"task": task, "seed": seed, "subspace_seed": sub.seed, "lr": rc.lr * mult, "mult": mult, "recipe": rc.source,
           "gain_step0": zero, "T0": tr.T0[task], "step1_norm": step1["norm"], "gains": gains, "losses": losses,
           "first_at1": kept["at1"][0] if "at1" in kept else None, "max_gain": gains[best], "max_gain_step": best,
           "loss_first5": first, "loss_window_last5": last, "loss_fell": last < first, "seconds": time.time() - t0}  # fmt: skip
    log(f"[{task} s{seed} x{mult:g}] guard step-1 ||dW|| {step1['norm']:.4g}; loss steps 1-5 {first:.4f} -> steps "
        f"{w - 4}-{w} {last:.4f}: fell {last < first} (logged, no bar)")
    log(f"[{task} s{seed} x{mult:g}] first gain >= {cfg.target_gain:g} at step {rec['first_at1']}; max gain {gains[best]:.4f} "
        f"at step {best}; {rec['seconds']:.0f} s")
    return rec, kept


def score(tr: Trainer, state: str, kept: tuple[int, DW, Any], eval_gain: float, ft: dict[str, Any]) -> dict[str, Any]:
    """The reader on one kept update: share, gain, numerator, KL, size against the fine-tune's; and the reader's gain
    against the training eval's on the same state (expected exactly equal: same forward, batches and rows)."""
    step, d, _ = kept
    out = tr.rd.share(state, d) | {"kl": tr.rd.kl(state, d), "norm": RL.norm(d), "step": step}
    return out | {"norm_rel": out["norm"] / ft["norm"], "kl_rel": out["kl"] / ft["kl"], "eval_minus_reader_gain": eval_gain - out["gain"]}


def one_update(tr: Trainer, state: str, mult: float, ft: dict[str, Any]) -> dict[str, Any]:
    """Train, score at-1 (if reached) and end, save Z, the seeds and both bf16 deltas; the record."""
    task, seed = task_of(state), int(state.rsplit("_s", 1)[1])
    assert seed in tr.cfg.seeds, state
    rec, kept = train_one(tr, task, seed, mult)
    rec |= {"state": state, "sizes": tr.sizes(), "fine_tune": {k: ft[k] for k in ("r_route_split", "gain", "numerator", "kl", "norm")}}
    for name in ("at1", "end"):
        if name in kept:
            rec[name] = score(tr, state, kept[name], rec["gains"][kept[name][0]], ft)
        else:  # never reached gain 1: its largest held-out gain, no share
            rec[name] = {"step": None, "r": math.nan, "max_gain": rec["max_gain"], "max_gain_step": rec["max_gain_step"]}
        log(f"[{state} {name}] {json.dumps(jsonable(rec[name]))}")
    rec["T0_minus_reader"] = rec["T0"] - tr.rd.job.base(task)["T"]  # expected exactly 0
    log(f"[{state}] guard training T0 - reader T0 {rec['T0_minus_reader']!r} (expected exactly 0)")
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({"subspace_seed": rec["subspace_seed"], "train_seed": seed, "k": tr.ks, "lr": rec["lr"],
                **{f"{n}_step": v[0] for n, v in kept.items()}, **{f"Z_{n}": v[2] for n, v in kept.items()},
                **{n: {k: x.bfloat16().cpu() for k, x in v[1].items()} for n, v in kept.items()}},
               LARGE / f"{state.replace('/', '_')}.pt")  # fmt: skip
    return rec


# ---- stages ---------------------------------------------------------------------------------------------------------


def control_job(experiment: str, cfg: Config) -> int:
    """P2.82's control as it is, on the 24 reuse states only: its states function and its output path swapped."""
    pairs, saved = RL.figure_states()
    mine = {s: saved[s] for s in reuse_states()}
    RL.figure_states, RL.CONTROL = (lambda: (pairs, mine)), CONTROL
    return RL.control_job(experiment, cfg)


def pick(runs: dict[str, dict[str, Any]], reach_by: int) -> tuple[str, str]:
    """(multiplier key, why): the lowest multiplier first reaching gain 1 by step `reach_by`; else the lowest reaching
    it by the last step; else the one with the largest held-out gain (Fable 2026-11-26). Keys sorted by value."""
    keys = sorted(runs, key=float)
    for bar, why in ((reach_by, f"lowest reaching gain 1 by step {reach_by}"), (math.inf, "lowest reaching gain 1 by the last step")):
        hit = [m for m in keys if runs[m]["first_at1"] is not None and runs[m]["first_at1"] <= bar]
        if hit:
            return hit[0], why
    return max(keys, key=lambda m: runs[m]["max_gain"]), "none reached gain 1: the largest held-out gain"


def lrgrid_job(experiment: str, cfg: Config) -> int:
    passed(CONTROL, experiment, "pass")
    tr = Trainer([s for s in reuse_states() if task_of(s) == cfg.grid_task], cfg)
    out: dict[str, Any] = json.loads(GRID.read_text()) if GRID.is_file() else {}
    if out.get("experiment") != experiment:
        out = {"experiment": experiment, "task": cfg.grid_task, "seed": cfg.grid_seed, "sizes": tr.sizes(), "runs": {}}
    for m in cfg.lr_mults:
        if str(m) not in out["runs"]:
            out["runs"][str(m)], _ = train_one(tr, cfg.grid_task, cfg.grid_seed, m)
            RL.dump(GRID, out)
    mult, why = pick(out["runs"], cfg.reach_by)
    out |= {"chosen": float(mult), "why": why}
    RL.dump(GRID, out)
    log(f"[lrgrid] {json.dumps({m: {k: r[k] for k in ('first_at1', 'max_gain', 'loss_fell')} for m, r in out['runs'].items()})}")
    log(f"[lrgrid] chosen multiplier {mult} ({why}); sizes {json.dumps(out['sizes'])}")
    log("LRGRID DONE")
    return 0


def train_job(experiment: str, cfg: Config, tasks: list[str]) -> int:
    ctl, mult = passed(CONTROL, experiment, "pass"), passed(GRID, experiment, "chosen")["chosen"]
    states = [s for s in reuse_states() if task_of(s) in tasks]
    assert {task_of(s) for s in states} == set(tasks), f"not reuse tasks of Fig known: {set(tasks) - {task_of(s) for s in states}}"
    tr, t0 = Trainer(states, cfg), time.time()
    for i, state in enumerate(s for t in tasks for s in states if task_of(s) == t):
        path = UPDATES / f"{state.replace('/', '_')}.json"
        if path.is_file() and json.loads(path.read_text()).get("experiment") == experiment:
            log(f"{state}: done before, skipped")
            continue
        rec = one_update(tr, state, mult, ctl["fine_tunes"][state]) | {"experiment": experiment}
        write_stage(path, jsonable(rec | {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}))
        log(f"ITEM {i + 1} of {len(states)} DONE ({state}); {(time.time() - t0) / 60:.1f} min in")
    log(f"{experiment} train {' '.join(tasks)} DONE")
    return 0


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, experiment = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E136, E147, E299 and P2.82
    log(f"{experiment} {' '.join(argv)} start on {socket.gethostname()}; settings {asdict(cfg)}")
    if argv == ["control"]:
        return control_job(experiment, cfg)
    if argv == ["lrgrid"]:
        return lrgrid_job(experiment, cfg)
    assert len(argv) >= 2 and argv[0] == "train", f"usage: run.py control | lrgrid | train TASK..., got {argv}"
    return train_job(experiment, cfg, argv[1:])


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
