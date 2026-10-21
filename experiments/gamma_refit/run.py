"""gamma_refit: does the frozen Gamma describe the updated model, or is it the thing that breaks?
Plan: reports/2026-10-20_plan_gamma-refit.md (design check revised with the user 2026-10-20; the user's choices:
emoticon only, fresh states on one H100, arm P for all five states and arm M for the fine-tune's split, full Gamma,
sweep to 3 000 steps, stage-0 attribution diagnostic first).

One job, one card. The emoticon fine-tune (arm A, 10 steps) and, from it, five states: W0 itself, do-nothing (delta-W
in Delta), the least-squares split, the scaled adjoint split, and component training (10 steps). Every state is
scored under the released Gamma, then Gamma alone is refit on that state (U, V, Delta fixed; `gamma.refit`, the
released loss stack minus the faithfulness term) with checkpoints at 100, 300, 1 000 and 3 000 steps, each scored by
the standing harness (route split at the trigger, Pile ratios to W0, VPD's sparsity loss, the switched-on set at the
trigger). Arm P refits on Pile rows only; arm M (the fine-tune's split only) mixes one packed row of the task's
training windows into every batch. Stage 0 (no training): the adversary's sources on the trained states, their
transfer to W0, and the ablation KL of the subcomponents the adversary turns down, at the state against W0.

Stages (results/predictions.json must exist; `--predictions` files it once):
  prep    GPU -> the states (results/large/*.pt), the W0 floor, stage 0 and a ten-step benchmark -> results/gamma_refit.json
  refit   GPU -> the six refits in order of informativeness, resumable (skips refits already in the JSON); Gamma's
          final parameters per refit (bf16, results/large/gamma_*.pt), mean CI per checkpoint, trigger gates.
          --only ls_P,w0_P runs those refits alone (a budgeted resume); the score is written once all six are in
  all     prep, then refit
  --smoke tiny sizes -> *_smoke files
  --v2    the protocol fix (reports/2026-10-21_plan_gamma-refit-protocol.md, user-confirmed 2026-10-21): 300 steps at
          lr 0 so the persistent adversary and AdamW's moments warm up, then Gamma at the released end-of-schedule lr
          5e-6 -> *_v2 files, predictions_v2.json; run with `--only w0_P` first (the control must pass before any state)
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt} and results/windows.json (round 1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
PRED_V2 = RESULTS / "predictions_v2.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"
FT_LR, CT_LR, TOL = 3e-5, 1e-5, 1e-6  # arm A's recipe (rounds 1 to 5), component training's (round 2), the split tolerance
BANK_START = 32_768  # Pile train rows from here on: the emoticon windows were mined from rows 0..32768
STATES = ("ls", "w0", "nothing", "ct", "adjoint")
REFITS = (("ls", "P"), ("w0", "P"), ("nothing", "P"), ("ls", "M"), ("ct", "P"), ("adjoint", "P"))  # most informative first
GROUPS = ("down_changed", "up_changed", "down_random")


@dataclass(frozen=True)
class Sizes:
    ft_steps: int
    ct_steps: int
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    refit_steps: int
    ckpts: tuple[int, ...]
    batch: int
    micro: int
    lr: float
    warmup: int  # refit steps at lr 0 before Gamma moves (gamma.RefitConfig.warmup)
    bank_rows: int
    bench_steps: int
    group_k: int  # stage 0: subcomponents per group
    abl_rows: int  # stage 0: Pile rows the ablation KL averages over
    suffix: str

    def __post_init__(self) -> None:
        assert self.ckpts == tuple(sorted(self.ckpts)) and self.ckpts[-1] == self.refit_steps, self.ckpts
        assert self.batch % self.micro == 0 and self.bank_rows >= self.batch and self.pile_rows % 8 == 0, self


FULL = Sizes(ft_steps=10, ct_steps=10, pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2), refit_steps=3000,
             ckpts=(100, 300, 1000, 3000), batch=64, micro=8, lr=5e-5, warmup=0, bank_rows=65_536, bench_steps=10, group_k=48,
             abl_rows=8, suffix="")
FULL_V2 = replace(FULL, lr=5e-6, warmup=300, suffix="_v2")  # the protocol fix; everything else as E15
SMOKE = Sizes(ft_steps=3, ct_steps=3, pile_rows=8, pile_pgd_steps=2, pile_seeds=(0,), refit_steps=2, ckpts=(1, 2), batch=4,
              micro=2, lr=5e-5, warmup=1, bank_rows=64, bench_steps=2, group_k=4, abl_rows=2, suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-20",
    "design": "reports/2026-10-20_plan_gamma-refit.md (design check revised with the user 2026-10-20; emoticon only, fresh states, "
              "arm P for all five states and arm M for the fine-tune's split, full Gamma, sweep to 3000 steps, stage 0 first)",
    "setup": {
        "update": f"emoticon full fine-tune, arm A recipe (Adam lr {FT_LR}, all 24 matrices, seed 0), step {FULL.ft_steps}",
        "states": "w0 (no update), nothing (delta-W in Delta), ls (split.split_state), adjoint (split.adjoint_state, per-matrix "
                  f"scale), ct (component_training.train, Adam lr {CT_LR}, {FULL.ct_steps} steps, all alive subcomponents)",
        "refit": f"gamma.refit: all of Gamma, AdamW lr {FULL.lr} (the released ci_fn start value, constant; the released cosine "
                 f"schedule ended at 5e-6), released betas, batch {FULL.batch} rows of 512 in micro-batches of {FULL.micro}, loss = "
                 "released stack minus FaithfulnessLoss (importance minimality p 0.4 coeff 2e-4, StochasticReconSubset coeff 0.5, "
                 "PersistentPGDRecon coeff 0.5 with its Adam adversary, 2 warmup steps), bf16 autocast; U, V, Delta fixed; "
                 f"checkpoints {FULL.ckpts}; the same seeded row order for every refit",
        "arms": f"P: Pile train rows {BANK_START}.. ({FULL.bank_rows} rows, cycled). M: the same with row 0 of every batch a "
                "packed row of the task's training windows (12 triggers per step)",
        "harness": f"harness.measure at every checkpoint: route split at the trigger (held-out windows), Pile ratios to W0 "
                   f"({FULL.pile_rows} val rows, {FULL.pile_pgd_steps} PGD steps, W0 floor over adversary seeds {FULL.pile_seeds} "
                   "under the released Gamma), plus VPD's sparsity loss on the same rows (ratio to W0's), the switched-on set S_on "
                   "(recruit rule, dg > 0.3 at the trigger over W0's released gates) and its Jaccard with the same state's S_on "
                   "under the released Gamma (step 0)",
        "stage0": f"gamma.pgd_sources on the first 128 val rows (seed 0) for nothing, ls, ct and W0: the sources, the KL they "
                  "reach, each state's sources applied at W0 and W0's at the state (transfer); gamma.adversary_groups "
                  f"({FULL.group_k} per group): ablation KL (gamma.ablation_kl, {FULL.abl_rows} val rows) at the state and at W0 "
                  "and mean released-Gamma g, for the most-changed turned-down, most-changed left-up and random turned-down "
                  "subcomponents",
        "blind_to": "Pile ratios: the trigger (about 3 in 131k positions); S_on Jaccard: the Pile behaviour; sparsity loss: which "
                    "subcomponents are called important, only how many. Positive control for the refit's harmlessness: w0_P "
                    "(must stay at the floor); for the refit's power: ct_P (a state that already passes must keep passing).",
        "noise": "f = max/min - 1 of W0's PGD KL over the three adversary seeds; 'at the floor' = ratio <= 1 + 3 f. The w0_P arm "
                 "gives the refit's own drift at every checkpoint.",
    },
    "gates": [
        {"id": "G0_benchmark", "rule": "ten refit steps timed; seconds per step and the projected hours for the six refits are "
                                       "recorded (no pass/fail; the runner compares with the budget)"},
        {"id": "G1_split_state", "rule": "ls at step 0: held-out P(o) equals the fine-tune's to 1e-3 and the route shares sum to 1 "
                                         "to 1e-6", "if_fail": "the split or the harness changed; stop"},
        {"id": "G2_w0_identity", "rule": "w0 at step 0 (released Gamma): every Pile ratio within 0.02 of 1 and sparsity ratio within "
                                         "0.001 of 1", "if_fail": "the harness is not reproducible on its own W0; stop"},
    ],
    "predictions": [
        {"id": "P1_w0_control", "scored": True, "arm": "w0_P, every checkpoint",
         "pass": "PGD ratio <= 1 + 3 f and sparsity ratio in [0.99, 1.01] at every checkpoint",
         "null": "the released Gamma was not converged at this lr, or the lr is too hot: the ratios drift; then every other arm "
                 "is read against w0_P's curve, not against the released floor",
         "predicted": "pass: a converged Gamma warm-started at its own training lr random-walks within the floor"},
        {"id": "P2_nothing_cannot_move", "scored": True, "arm": "nothing_P",
         "pass": "PGD ratio at 3000 <= PGD ratio at 0 x (1 + 3 f)",
         "null": "the ratio rises: Gamma trades Pile faithfulness for something else",
         "predicted": "pass: delta-W sits in Delta, which every mask includes under its own source; Gamma has nothing to re-describe"},
        {"id": "P3_split_refit", "scored": True, "arm": "ls_P",
         "pass": "PGD ratio at 3000 <= 0.5 x the ratio at 0, and > 1 + 3 f, and sparsity ratio <= 1.05 (the partial branch)",
         "null": "floor branch: ratio at 3000 <= 1 + 3 f with sparsity <= 1.05 (stale Gamma was the whole fragility); unmoved "
                 "branch: ratio > 0.5 x step 0 (the changes are unfaithful in themselves); cheating branch: sparsity > 1.05",
         "predicted": "partial: the adversarial-vs-stochastic analysis found selective masks do 50-300x the damage of the "
                      "uniform-CI mask, a cancellation signature a Gamma update can only partly repair",
         "plateau_rule": "the first checkpoint after which the PGD ratio changes by < 10% to the next; reported"},
        {"id": "P4_ct_stays", "scored": True, "arm": "ct_P, every checkpoint",
         "pass": "PGD ratio <= 1 + 3 f at every checkpoint",
         "null": "the refit breaks a state that passed under the frozen Gamma",
         "predicted": "pass: round 2's component training read 1.03 at step 10 on this task"},
        {"id": "P5_adjoint", "scored": False, "arm": "adjoint_P",
         "reported": "the step-0 ratios against ls's, the held-out P(o) of the adjoint state, the curve under the refit",
         "expected": "step-0 PGD ratio below ls's (Delta fixed, about 0.6 of delta-W reproduced); falls under the refit like ls_P"},
        {"id": "P6_arm_M", "scored": True, "arm": "ls_M against ls_P",
         "pass": "PGD ratio at 3000 within 20% of ls_P's, and S_on Jaccard with the released-Gamma set at 3000 lower for ls_M than "
                 "for ls_P",
         "null": "the trigger rows move the Pile ratio (the arms disagree), or arm P moves the trigger gating as much as arm M",
         "predicted": "pass: 1.5% of rows cannot move the Pile objective; only arm M sees triggers, so only it re-gates them"},
        {"id": "S0_ablation", "scored": True, "arm": "stage 0, ls, group down_changed",
         "pass": "median over the group of (ablation KL at the state) / (ablation KL at W0) < 2",
         "null": "stale reading: median >= 2 with the group's mean released-Gamma g at the state < 0.1 (they matter more and Gamma "
                 "still calls them unimportant)",
         "predicted": "pass (cancellation reading): the changed subcomponents are individually inert, as the identity test's diffuse "
                      "update and the selective-mask signature suggest"},
    ],
    "reported_not_scored": "every refit's loss curve; every checkpoint's full route split, stochastic / CI-masked / unmasked ratios, "
                           "S_on size; the transfer KLs of stage 0 for every state; the up_changed and down_random groups; the "
                           "benchmark's seconds per step and peak memory; mean CI per subcomponent per checkpoint (large/)",
}

PREDICTIONS_V2: dict[str, Any] = {
    "filed": "2026-10-21",
    "design": "reports/2026-10-21_plan_gamma-refit-protocol.md (design check confirmed by the user 2026-10-21). E15's refit as filed "
              "took the unchanged model off the floor (P1 failed: PGD ratio 2.57 / 3.53 / 6.27 / 2.63, sparsity 0.94 to 1.09); the "
              "protocol is changed in two places and tested on W0 alone before any state arm",
    "setup": {
        "protocol": f"gamma.refit with warmup {FULL_V2.warmup}: the first {FULL_V2.warmup} steps run the full step with Gamma's lr at 0 "
                    "(the persistent adversary takes its steps and AdamW accumulates its moments; Gamma does not move), then Gamma "
                    f"trains at lr {FULL_V2.lr} (the released ci_fn end-of-schedule value), constant, with the warmed moments. "
                    f"Everything else as E15: batch {FULL_V2.batch} in micro-batches of {FULL_V2.micro}, released loss stack minus "
                    f"faithfulness, {FULL_V2.refit_steps} Gamma steps, checkpoints {FULL_V2.ckpts} counted from the first Gamma step, "
                    "the same seeded row order, bf16 autocast",
        "arms": "w0_P only (`refit --v2 --only w0_P`); the states and stage 0 are remade under the _v2 suffix by prep (same seeds)",
        "harness": "as E15: harness.measure at step 0 and every checkpoint, the W0 floor over 3 adversary seeds under the released Gamma, "
                   "the sparsity loss ratio",
        "blind_to": "W0 alone says nothing about any state; the Pile checks are blind to the trigger; the warm-up curve is the "
                    "persistent adversary's own loss, not the harness's attacker. Positive control for the harness: step 0 (the "
                    "released Gamma) must read 1.000 (G2). The filed null of E15's P1 (the control's drift curve) is the reference "
                    "for every number below",
        "noise": "f = max/min - 1 of W0's PGD KL over the three adversary seeds (E15: 0.075, line 1.22); the importance term moves "
                 "about 8% step to step at W0 (E15: 854 at step 1, 922 at step 5)",
        "early_stop": "the runner kills the run if the control's PGD ratio is above 1 + 3 f at both step 100 and step 300 (the "
                      "protocol has failed as E15's did; about $3 spent); the JSON written at those checkpoints is the result",
    },
    "gates": [
        {"id": "G0_benchmark", "rule": "twenty steps timed at the full batch (ten at lr 0, ten training); seconds per step and the "
                                       "projected hours for w0_P are recorded; the cap (about $10) is confirmed from it"},
        {"id": "G2_w0_identity", "rule": "w0 at step 0 (released Gamma): every Pile ratio within 0.02 of 1 and sparsity ratio within "
                                         "0.001 of 1", "if_fail": "the harness is not reproducible on its own W0; stop"},
    ],
    "predictions": [
        {"id": "P1_w0_control_v2", "scored": True, "arm": "w0_P, every checkpoint",
         "pass": "PGD ratio <= 1 + 3 f and sparsity ratio in [0.99, 1.01] at every checkpoint",
         "null": "drifts again. The same as E15's control (2.6 to 6.3): the released Gamma is not at rest on the objective as "
                 "reconstructed, Gamma updating is shelved. Less, or only late, with W1 failing (the adversary still rising at 300): "
                 "the warm-up is too short, K is raised once to 1000; a second failure shelves Gamma updating",
         "predicted": "pass: with a warmed adversary and the end-of-schedule rate the refit is the released run's own end state, and "
                      "a converged Gamma random-walks within the floor"},
        {"id": "W1_adversary_warmed", "scored": True, "arm": "w0_P, the warm-up curve",
         "pass": "the persistent-adversary term's mean over warm-up steps 201-300 is within 5% of its mean over steps 101-200",
         "null": "still rising by more than 5% per 100 steps: the adversary has not converged in 300 steps; P1 is read with that "
                 "caveat and K is raised once",
         "predicted": "pass; the plateau value is reported (expected at or above about 0.6 nats: the harness's 20-step attacker "
                      "reaches 0.70 at W0, and the persistent one has more freedom, one source per slot and position)"},
        {"id": "W2_no_spike", "scored": True, "arm": "w0_P, Gamma steps 1 and 2",
         "pass": "the importance-minimality term on Gamma steps 1 and 2 is within 20% of its mean over the last 10 warm-up steps",
         "null": "a spike like E15's step 2 (20x): the moments did not warm up, or the end rate is still too hot",
         "predicted": "pass: Adam's first step is lr x m/sqrt(v), not a sign step, once the moments hold 300 steps of gradients"},
    ],
    "reported_not_scored": "stochastic and CI-masked ratios at every checkpoint (expected within 0.02 of 1, as in E15); the three loss "
                           "terms over the warm-up and the training (the persistent term expected flat after the warm-up, within 10% "
                           "of its plateau; the importance term within 5% of its warm-up mean at steps 2991-3000); the benchmark's "
                           "seconds per step and peak memory; mean CI per checkpoint and the trigger gates at steps 0 and 3000 (large/)",
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))
    return p


def pred_path(args: argparse.Namespace) -> Path:
    return PRED_V2 if args.v2 else PRED


def sizes(args: argparse.Namespace) -> Sizes:
    return SMOKE if args.smoke else FULL_V2 if args.v2 else FULL


def file_predictions(args: argparse.Namespace) -> None:
    path = pred_path(args)
    if path.is_file():
        sys.exit(f"{path} exists; predictions are not edited after filing (add a dated report instead)")
    dump(path.name, PREDICTIONS_V2 if args.v2 else PREDICTIONS)
    print(f"filed {path}")


# ----------------------------------------------------------------------------- the states

Update = tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor, Tensor]] | None]  # (delta-W per matrix, state or do-nothing)


def make_states(ctx: Any, S: Sizes) -> tuple[dict[str, Update], dict[str, Any]]:
    """The fine-tune and the five states, written to results/large/ (fp32). Returns them and the fine-tune's held-out evals."""
    from commute import component_training, split
    from commute.finetune import ArmAConfig, finetune_arm_a

    a_cfg = ArmAConfig(o_token=ctx.token, lr=FT_LR, steps=S.ft_steps, checkpoints=(S.ft_steps,), seed=0)
    res = finetune_arm_a(ctx.model, ctx.train, ctx.held, a_cfg, ctx.device, log)
    dW = res.deltas[S.ft_steps]
    torch.save(dW, LARGE / f"ft_delta{S.suffix}.pt")
    ls = split.split_state(dW, ctx.uv_cpu, ctx.bases)
    torch.save({k: (u, v) for k, (_, u, v) in ls.items()}, LARGE / f"state_ls{S.suffix}.pt")
    dW_adj, adj = split.adjoint_state(dW, ctx.uv_cpu, ctx.sigma)
    torch.save(adj, LARGE / f"state_adjoint{S.suffix}.pt")
    t_cfg = component_training.TrainConfig(lr=CT_LR, steps=S.ct_steps, save_steps=(S.ct_steps,))
    curve, ct_states = component_training.train(ctx.model, ctx.uv, ctx.train.tokens, ctx.held, a_cfg, t_cfg, ctx.device, ctx.W0, None, log)
    ct = ct_states[S.ct_steps]
    torch.save(ct, LARGE / f"state_ct{S.suffix}.pt")
    zeros = {k: torch.zeros_like(w) for k, w in ctx.W0.items()}
    states: dict[str, Update] = {"w0": (zeros, None), "nothing": (dW, None), "ls": (dW, ls), "adjoint": (dW_adj, adj),
                                 "ct": ({k: v[0] for k, v in ct.items()}, ct)}
    return states, {"ft": res.evals[S.ft_steps], "ft_log": res.log, "ct_curve": curve}


def load_states(ctx: Any, S: Sizes) -> dict[str, Update]:
    """The states `make_states` wrote, for the resumable refit stage."""
    dW = torch.load(LARGE / f"ft_delta{S.suffix}.pt", weights_only=False)
    ls_uv = torch.load(LARGE / f"state_ls{S.suffix}.pt", weights_only=False)
    adj = torch.load(LARGE / f"state_adjoint{S.suffix}.pt", weights_only=False)
    ct = torch.load(LARGE / f"state_ct{S.suffix}.pt", weights_only=False)
    zeros = {k: torch.zeros_like(w) for k, w in ctx.W0.items()}
    dW_adj = {k: adj[k][0] if k in adj else torch.zeros_like(dW[k]) for k in dW}
    return {"w0": (zeros, None), "nothing": (dW, None), "ls": (dW, {k: (dW[k], u, v) for k, (u, v) in ls_uv.items()}),
            "adjoint": (dW_adj, adj), "ct": ({k: v[0] for k, v in ct.items()}, ct)}


def components_of(ctx: Any, state: dict[str, tuple[Tensor, Tensor, Tensor]] | None) -> dict[str, tuple[Tensor, Tensor]] | None:
    from commute.component_training import components_with

    return None if state is None else components_with(ctx.model, ctx.alive, {k: (v[1], v[2]) for k, v in state.items()})


# ----------------------------------------------------------------------------- stage 0


def subcomponent_change(ctx: Any, state: dict[str, tuple[Tensor, Tensor, Tensor]] | None) -> dict[str, Tensor]:
    """Per matrix (C,): the gauge-free relative change of u v^T of each alive subcomponent (identity.change_stats),
    0 for dead ones and for do-nothing."""
    from commute.identity import change_stats

    out = {}
    for k, (U, V) in ctx.uv_cpu.items():
        ch = torch.zeros(ctx.alive[k].numel())
        if state is not None and k in state:
            ch[ctx.alive[k]] = change_stats(U, V, state[k][1], state[k][2])["rel_change"].float()
        out[k] = ch
    return out


def stage0(ctx: Any, S: Sizes, states: dict[str, Update]) -> dict[str, Any]:
    """The adversary's sources on W0 and on nothing, ls and ct; their transfer; the ablation groups (gamma.py, stage 0)."""
    from commute import gamma
    from commute.faithfulness import model_state

    block, rows = ctx.blocks[0], ctx.blocks[0][0][: S.abl_rows]
    out: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    src0, kl0 = gamma.pgd_sources(ctx.model, block, S.pile_pgd_steps)
    sources["w0"], out["w0"] = src0, {"pgd_kl": kl0, "n_down": {k: int(((a[:-1] < gamma.DOWN) & ctx.alive[k]).sum()) for k, a in src0.items()},
                                     "delta_source": {k: float(a[-1]) for k, a in src0.items()}}
    log(f"[stage0 w0] pgd_kl {kl0:.4f}")
    for name in ("nothing", "ls", "ct"):
        dW, st = states[name]
        target = {k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}
        rec: dict[str, Any] = {}
        groups: dict[str, Any] = {}
        with model_state(ctx.model, target=target, components=components_of(ctx, st)):
            src, rec["pgd_kl"] = gamma.pgd_sources(ctx.model, block, S.pile_pgd_steps)
            rec["w0_sources_at_state"] = gamma.mask_kl(ctx.model, block, src0)
            rec["n_down"] = {k: int(((a[:-1] < gamma.DOWN) & ctx.alive[k]).sum()) for k, a in src.items()}
            rec["delta_source"] = {k: float(a[-1]) for k, a in src.items()}
            sel = gamma.adversary_groups(src, subcomponent_change(ctx, st), ctx.alive, S.group_k, seed=0)
            for g, s in sel.items():
                groups[g] = {"members": {k: v for k, v in s.items()},
                             "kl_state": {k: gamma.ablation_kl(ctx.model, k, v, rows) for k, v in s.items()},
                             "g_state": gamma.mean_gates(ctx.model, s, rows)}
        sources[name] = src
        rec["state_sources_at_w0"] = gamma.mask_kl(ctx.model, block, src)
        for g, s in sel.items():
            groups[g]["kl_w0"] = {k: gamma.ablation_kl(ctx.model, k, v, rows) for k, v in s.items()}
            groups[g]["g_w0"] = gamma.mean_gates(ctx.model, s, rows)
            ks, kw = torch.cat(list(groups[g]["kl_state"].values())), torch.cat(list(groups[g]["kl_w0"].values()))
            gs = torch.cat(list(groups[g]["g_state"].values()))
            groups[g]["summary"] = {"median_ratio": float((ks / kw.clamp_min(1e-12)).median()), "mean_kl_state": float(ks.mean()),
                                    "mean_kl_w0": float(kw.mean()), "mean_g_state": float(gs.mean()), "n": int(ks.numel())}
        rec["groups"] = groups
        out[name] = rec
        log(f"[stage0 {name}] pgd_kl {rec['pgd_kl']:.4f} (w0 {kl0:.4f}); w0's mask at state {rec['w0_sources_at_state']:.4f}, "
            f"state's mask at w0 {rec['state_sources_at_w0']:.4f}; groups "
            + " ".join(f"{g}: ratio {groups[g]['summary']['median_ratio']:.2f} g {groups[g]['summary']['mean_g_state']:.3f}" for g in GROUPS))
    torch.save(sources, LARGE / f"stage0_sources{S.suffix}.pt")
    return out


# ----------------------------------------------------------------------------- the refits


class Rows:
    """The refit's data: Pile train rows in one seeded order, cycled; arm M puts a packed row of training windows at
    row 0 of every batch."""

    def __init__(self, bank: Tensor, windows: Tensor, S: Sizes, arm: str, seed: int = 0):
        from commute.vpd import SEQ_LEN

        self.bank, self.windows, self.S, self.arm = bank, windows, S, arm
        self.perm = torch.randperm(bank.shape[0], generator=torch.Generator().manual_seed(seed))
        self.gen = torch.Generator().manual_seed(seed + 1)
        self.per_row = SEQ_LEN // windows.shape[1] + 1

    def packed(self) -> Tensor:
        idx = torch.randint(0, self.windows.shape[0], (self.per_row,), generator=self.gen)
        return self.windows[idx].flatten()[: self.bank.shape[1]]

    def __call__(self, step: int) -> Tensor:
        i = (torch.arange(self.S.batch) + step * self.S.batch) % self.bank.shape[0]
        rows = self.bank[self.perm[i]].clone()
        if self.arm == "M":
            rows[0] = self.packed()
        return rows


def evaluate(ctx: Any, S: Sizes, name: str, upd: Update, step: int, base: dict[str, Any]) -> dict[str, Any]:
    """One checkpoint's record under the current Gamma: harness.measure (route split, Delta share, Pile ratios), the
    sparsity loss and its ratio to W0's, the trigger gates (saved at step 0 and the last step), S_on and its Jaccard
    with the same state's released-Gamma set (`base['s_on']`, set at step 0), mean CI per subcomponent and the S_on set
    itself (both popped by the caller: the first goes to the large/ file, the second is the reference)."""
    from commute import gamma, harness, recruit
    from commute.faithfulness import model_state

    dW, st = upd
    rec = harness.measure(ctx, f"{name} step {step}", dW, st, True, log)
    target = {k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}
    with model_state(ctx.model, target=target, components=components_of(ctx, st)):
        rec["sparsity"] = sparsity(ctx)
        trig = gamma.trigger_gates(ctx.model, ctx.held.tokens, ctx.held.half, ctx.token)
        mean_ci = {k: v.float() for k, v in gamma_mean_ci(ctx).items()}
    rec["sparsity_ratio"] = rec["sparsity"] / ctx.sparsity_w0
    on = gamma.s_on(trig, ctx.trig_w0, ctx.alive)
    if step == 0:
        base["s_on"] = on
    rec["s_on_size"] = recruit.size(on)
    rec["s_on_jaccard_released"] = recruit.jaccard(on, base["s_on"])
    rec["s_on_by_layer"] = recruit.by_layer(on)
    if step in (0, S.refit_steps):
        torch.save(trig, LARGE / f"trigger_gates_{name}_step{step}{S.suffix}.pt")
    rec["mean_ci"], rec["s_on_set"] = mean_ci, on
    log(f"[{name} step {step}] pgd {rec['pile']['pgd_kl']:.3f} stoch {rec['pile']['stochastic_kl']:.3f} ci {rec['pile']['ci_masked_kl']:.3f} "
        f"sparsity {rec['sparsity_ratio']:.4f} S_on {rec['s_on_size']} J {rec['s_on_jaccard_released']:.3f}")
    return rec


def gamma_mean_ci(ctx: Any) -> dict[str, Tensor]:
    from commute.vpd import mean_ci

    return mean_ci(ctx.model, ctx.pile_batches)


def sparsity(ctx: Any) -> float:
    """VPD's importance-minimality loss of the model's current state on the harness's Pile rows (released p, beta, eps)."""
    from commute.faithfulness import sparsity_loss

    c = ctx.sparsity_cfg
    return sparsity_loss(ctx.model, ctx.pile_batches, c["pnorm"], c["beta"], c["eps"])


def run_refit(ctx: Any, S: Sizes, name: str, arm: str, upd: Update, rows: Rows, released: dict[str, Tensor],
              on_checkpoint: Any = lambda rec: None, step0: dict[str, Any] | None = None) -> dict[str, Any]:
    """One refit of Gamma on one state: the per-checkpoint records (step 0 = released Gamma), the loss curve, Gamma's
    final parameters (bf16) and the mean CI per checkpoint in large/. `on_checkpoint(partial record)` runs after every
    checkpoint so a stopped job keeps what it scored. `step0`, when given, is another arm's step-0 bundle of the same
    state (the released Gamma does not depend on the arm), reused instead of scored again; the bundle is returned under
    "step0" for the next arm."""
    from commute import gamma
    from commute.faithfulness import model_state

    dW, st = upd
    tag = f"{name}_{arm}"
    cfg = gamma.RefitConfig(steps=S.refit_steps, checkpoints=S.ckpts, batch=S.batch, micro=S.micro, lr=S.lr, warmup=S.warmup)
    ckpts: dict[str, Any] = {}
    mean_cis: dict[int, dict[str, Tensor]] = {}
    base: dict[str, Any] = {}
    bundle: dict[str, Any] = {}
    t0 = time.time()

    def ev(step: int) -> None:
        if step == 0 and step0 is not None:
            base["s_on"], mean_cis[0], ckpts["0"] = step0["s_on"], step0["mean_ci"], dict(step0["rec"])
            log(f"[{tag} step 0] reused {step0['from']}'s step-0 record")
            return
        rec = evaluate(ctx, S, tag, upd, step, base)
        mean_cis[step] = rec.pop("mean_ci")
        on = rec.pop("s_on_set")
        if step == 0:
            bundle.update({"rec": dict(rec), "mean_ci": mean_cis[0], "s_on": on, "from": tag})
        ckpts[str(step)] = rec
        on_checkpoint({"checkpoints": ckpts, "seconds": time.time() - t0, "config": asdict(cfg), "partial": True})
        torch.cuda.empty_cache()

    target = {k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}
    with model_state(ctx.model, target=target, components=components_of(ctx, st)):
        gamma.load_gamma(ctx.model, released)
        curve = gamma.refit(ctx.model, cfg, rows, ev, log)
        torch.save({k: v.detach().to(torch.bfloat16).cpu() for k, v in ctx.model.ci_fn.state_dict().items()},
                   LARGE / f"gamma_{tag}_final{S.suffix}.pt")
    gamma.load_gamma(ctx.model, released)
    torch.save(mean_cis, LARGE / f"mean_ci_{tag}{S.suffix}.pt")
    return {"checkpoints": ckpts, "curve": curve, "seconds": time.time() - t0, "config": asdict(cfg), "step0": step0 or bundle}


# ----------------------------------------------------------------------------- the stages


def load_ctx(args: argparse.Namespace, S: Sizes) -> Any:
    """harness.Context plus what every checkpoint reads: the flattened Pile batches, the sparsity loss's settings and W0's
    value, W0's trigger gates under the released Gamma, and the refit's row bank."""
    from commute import gamma, harness, vpd
    from commute.data import pile_rows

    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx: Any = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    ctx.pile_batches = [b for blk in ctx.blocks for b in blk]
    ctx.sparsity_cfg = vpd.sparsity_config()
    ctx.sparsity_w0 = sparsity(ctx)
    ctx.trig_w0 = gamma.trigger_gates(ctx.model, ctx.held.tokens, ctx.held.half, ctx.token)
    ctx.bank = pile_rows("train", BANK_START, S.bank_rows, log)
    log(f"W0: pgd floor {harness.pgd_floor(ctx)}, sparsity {ctx.sparsity_w0:.4f}, bank {tuple(ctx.bank.shape)}")
    return ctx


def result_name(S: Sizes) -> str:
    return f"gamma_refit{S.suffix}.json"


def stage_prep(args: argparse.Namespace) -> tuple[Any, dict[str, Update]]:
    from commute import gamma, harness
    from commute.faithfulness import model_state

    S = sizes(args)
    t_start = time.time()
    ctx = load_ctx(args, S)
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S), "W0": {"pile": ctx.pile, "pgd_floor": harness.pgd_floor(ctx),
                                                                            "sparsity": ctx.sparsity_w0, "non_saturated": ctx.non_saturated}}
    if (RESULTS / result_name(S)).is_file():  # a resume on a fresh machine: keep the refits already scored
        old = json.loads((RESULTS / result_name(S)).read_text())
        out.update({k: old[k] for k in ("refits", "partial") if k in old})
        log(f"resuming: kept {sorted(out.get('refits', {}))} and partial {sorted(out.get('partial', {}))}")
    states, evals = make_states(ctx, S)
    out["ft"] = evals
    dump(result_name(S), out)
    out["stage0"] = stage0(ctx, S, states)
    dump(result_name(S), out)
    cfg = gamma.RefitConfig(steps=S.bench_steps, checkpoints=(), batch=S.batch, micro=S.micro, lr=S.lr, warmup=min(S.warmup, S.bench_steps))
    released = gamma.gamma_state(ctx.model)
    dW, st = states["ls"]
    target = {k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}
    torch.cuda.reset_peak_memory_stats()
    with model_state(ctx.model, target=target, components=components_of(ctx, st)):
        t0 = time.time()
        curve = gamma.refit(ctx.model, cfg, Rows(ctx.bank, ctx.train.tokens, S, "P"), lambda step: None, log)
        per_step = (time.time() - t0) / (S.bench_steps + cfg.warmup)
    gamma.load_gamma(ctx.model, released)
    out["benchmark"] = {"seconds_per_step": per_step, "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "curve": curve,
                        "hours_per_refit": per_step * (S.refit_steps + S.warmup) / 3600}
    out["seconds_prep"] = time.time() - t_start
    dump(result_name(S), out)
    log(f"benchmark: {per_step:.2f} s/step, peak {out['benchmark']['peak_gb']:.1f} GB, {out['benchmark']['hours_per_refit']:.2f} h per refit "
        f"of {S.warmup} + {S.refit_steps} steps (evaluations extra); prep {out['seconds_prep']:.0f} s")
    return ctx, states


def stage_refit(args: argparse.Namespace, ctx: Any | None = None, states: dict[str, Update] | None = None) -> None:
    from commute import gamma

    S = sizes(args)
    out = json.loads((RESULTS / result_name(S)).read_text())
    assert "stage0" in out, "run prep first"
    c: Any = ctx if ctx is not None else load_ctx(args, S)
    states = states or load_states(c, S)
    released = gamma.gamma_state(c.model)
    out.setdefault("refits", {})
    step0: dict[str, dict[str, Any]] = {}  # state -> the step-0 bundle of its first arm, reused by its other arms
    for name, arm in REFITS:
        tag = f"{name}_{arm}"
        if tag in out["refits"] or (args.only and tag not in args.only):
            log(f"[{tag}] {'already done' if tag in out['refits'] else 'not in --only'}, skipped")
            continue
        t0 = time.time()

        def keep(partial: dict[str, Any], tag: str = tag) -> None:
            out.setdefault("partial", {})[tag] = partial
            dump(result_name(S), out)

        res = run_refit(c, S, name, arm, states[name], Rows(c.bank, c.train.tokens, S, arm), released, keep, step0.get(name))
        step0[name] = res.pop("step0")
        out["refits"][tag] = res
        out.get("partial", {}).pop(tag, None)
        dump(result_name(S), out)
        log(f"[{tag}] done in {time.time() - t0:.0f} s")
    if all(f"{n}_{a}" in out["refits"] for n, a in REFITS):
        out["score"] = score(out)
    dump(result_name(S), out)
    log(f"wrote {RESULTS / result_name(S)}; refits done {sorted(out['refits'])}; score {json.dumps(out.get('score'), default=str)}")


# ----------------------------------------------------------------------------- the score


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    return {"pass": all(cond.values()), "conditions": cond, **values}


def score(out: dict[str, Any]) -> dict[str, Any]:
    R, f = out["refits"], out["W0"]["pgd_floor"]
    line = 1 + 3 * f
    last = str(out["sizes"]["refit_steps"])
    pgd = {tag: {s: c["pile"]["pgd_kl"] for s, c in r["checkpoints"].items()} for tag, r in R.items()}
    spars = {tag: {s: c["sparsity_ratio"] for s, c in r["checkpoints"].items()} for tag, r in R.items()}
    jac = {tag: {s: c["s_on_jaccard_released"] for s, c in r["checkpoints"].items()} for tag, r in R.items()}
    sc: dict[str, Any] = {"line": line, "pgd": pgd, "sparsity_ratio": spars, "jaccard": jac}
    ls0 = R["ls_P"]["checkpoints"]["0"]
    sc["G1_split_state"] = _pred({"p_o": abs(ls0["routes"]["p_cells"]["T1"] - out["ft"]["ft"]["p_o"]) <= 1e-3,
                                  "share_sum": abs(ls0["routes"]["share_sum"] - 1) <= 1e-6})
    w00 = R["w0_P"]["checkpoints"]["0"]
    sc["G2_w0_identity"] = _pred({k: abs(v - 1) <= 0.02 for k, v in w00["pile"].items()} | {"sparsity": abs(w00["sparsity_ratio"] - 1) <= 1e-3})
    sc["P1_w0_control"] = _pred({f"pgd@{s}": v <= line for s, v in pgd["w0_P"].items() if s != "0"}
                                | {f"sparsity@{s}": 0.99 <= v <= 1.01 for s, v in spars["w0_P"].items() if s != "0"})
    sc["P2_nothing_cannot_move"] = _pred({"not_up": pgd["nothing_P"][last] <= pgd["nothing_P"]["0"] * line})
    p3 = {"halved": pgd["ls_P"][last] <= 0.5 * pgd["ls_P"]["0"], "above_floor": pgd["ls_P"][last] > line, "sparsity": spars["ls_P"][last] <= 1.05}
    branch = ("cheating" if not p3["sparsity"] else "floor" if not p3["above_floor"] else "partial" if p3["halved"] else "unmoved")
    steps = [s for s in pgd["ls_P"] if s != "0"]
    plateau = next((steps[i] for i in range(len(steps) - 1)
                    if abs(pgd["ls_P"][steps[i + 1]] - pgd["ls_P"][steps[i]]) < 0.1 * pgd["ls_P"][steps[i]]), None)
    sc["P3_split_refit"] = _pred(p3, branch=branch, plateau=plateau)
    sc["P4_ct_stays"] = _pred({f"pgd@{s}": v <= line for s, v in pgd["ct_P"].items()})
    sc["P5_adjoint"] = {"pgd": pgd["adjoint_P"], "ls_pgd_0": pgd["ls_P"]["0"],
                        "p_o": R["adjoint_P"]["checkpoints"]["0"]["routes"]["p_cells"]["T1"], "p_o_ft": out["ft"]["ft"]["p_o"]}
    sc["P6_arm_M"] = _pred({"pile_agrees": abs(pgd["ls_M"][last] - pgd["ls_P"][last]) <= 0.2 * pgd["ls_P"][last],
                            "M_regates_more": jac["ls_M"][last] < jac["ls_P"][last]})
    s0 = out["stage0"]["ls"]["groups"]["down_changed"]["summary"]
    sc["S0_ablation"] = _pred({"median_ratio<2": s0["median_ratio"] < 2}, summary=s0, g_state=s0["mean_g_state"])
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["prep", "refit", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--v2", action="store_true", help="the protocol fix (warm-up at lr 0, then the released end lr); *_v2 files")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", type=lambda s: s.split(","), default=None, help="refit stage: run only these tags, e.g. w0_P,nothing_P")
    args = ap.parse_args()
    assert not args.only or set(args.only) <= {f"{n}_{a}" for n, a in REFITS}, args.only
    if args.predictions:
        file_predictions(args)
        return
    if args.stage is None:
        ap.print_help()
        return
    assert pred_path(args).is_file(), f"file predictions first: run.py --predictions{' --v2' if args.v2 else ''}"
    if args.stage == "prep":
        stage_prep(args)
    elif args.stage == "refit":
        stage_refit(args)
    else:
        stage_refit(args, *stage_prep(args))


if __name__ == "__main__":
    main()
