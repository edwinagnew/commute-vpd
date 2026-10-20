"""lora_commute: is a LoRA update easier to commute past the decomposition than a full-weight fine-tune?

Plan: reports/2026-10-19_plan_lora-lift-and-pullback.md (confirmed by the user 2026-10-19) and its addendum of the same
day (the user's framing: "is it easier to decompose a LoRA update than a normal fine-tuning update"; the full-weight
comparator arms, the scored-step rule, and the Pile adversary for P3).

Every arm is a state (target weights W0 + delta-W, components) measured the same way on the 73 held-out emoticon
windows: the route split at the trigger (d, r, i, e_gate, e_delta), the task-local harness (reported), the Pile
faithfulness ratios (round 2's harness, selected arms), how many subcomponents moved, and the switched-on set.
  LoRA_r   rank-r LoRA on h.2.mlp.down_proj, Adam lr 1e-3, arm A's loss; measured do-nothing (delta-W in Delta) and
           lifted (Euclidean lift onto that matrix's alive subcomponents), at its scored step and at step 100
  ft       the full-weight fine-tune (arm A, Adam lr 3e-5, all 24 matrices), do-nothing and lifted (24 matrices)
  pb_r     pull-back confined to that matrix with LoRA-shaped offsets of rank r (component_training.Offsets); lr by the rule
  pb_full  pull-back confined to that matrix, free offsets (the capacity ceiling); pb_all: all 24 matrices (the ladder's)
  seeds    LoRA_8 and pb_8 with seed 1 (the floor); W0 under the Pile adversary with seeds 0, 1, 2 (the PGD floor)

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  all             GPU -> results/lora_commute.json (dumped after every arm) + results/large/ (LoRA factors, site states)
  all --smoke     GPU, minutes: every path at the SMOKE sizes -> results/lora_commute_smoke.json
  w0seeds         GPU, seconds: the task-local harness at W0 under adversary seeds 0, 1, 2 -> results/w0_local_seeds.json
                  (the W0 floor for reading every arm's task-local PGD numbers as a ratio; added 2026-10-19 at the
                  user's request to read the task-local PGD ratio against the W0 floor first)
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}; optional experiments/recruit_test/results/large/
ft_s0_dg.pt (the recruit test's switched-on set, for a reproduction check).

Usage (repo root):  uv run python experiments/lora_commute/run.py all
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
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"
RECRUIT_DG = HERE.parent / "recruit_test" / "results" / "large" / "ft_s0_dg.pt"

SITE = "h.2.mlp.down_proj"  # the paper's LoRA site
LORA_LR, FT_LR, PB_ALL_LR = 1e-3, 3e-5, 1e-5
P_LEARN, KL_LIMIT = 0.9, 0.110  # an arm has learned: held-out P(o) >= P_LEARN; the lr rule also needs kl_other <= KL_LIMIT
MOVED_CUT, TOP_K, DG_THR, TOL = 0.01, 20, 0.3, 1e-6
BATCH = 64
REF = {  # numbers this run should reproduce (same recipes and seeds), checked as gates
    "nothing_ft_pgd_ratio": 2.18, "lift_ft_pgd_ratio": 5.41,  # round 2, lift_faith.json, rerun step 10
    "pb_all_d": 0.025, "pb_all_r": 0.53,  # ladder, emoticon pull-back step 10
    "nothing_lora8_r": 0.54,  # recruit test, lora_r8 step 10
}


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run."""

    ranks: tuple[int, ...]
    lora_steps: int
    lora_ckpts: tuple[int, ...]  # scored step = the first with held-out P(o) >= P_LEARN; the last is "finished"
    ft_steps: int
    ft_ckpts: tuple[int, ...]  # the first is the matched (scored) step
    pb_steps: int
    pb_save: tuple[int, ...]  # the last (= pb_steps) is scored
    pb_lrs: tuple[float, ...]
    stoch_draws: int
    pgd_steps: int  # the task-local harness (reported only)
    pile_rows: int  # round 2's Pile harness: 256 rows in two blocks; 0 skips it
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    suffix: str

    def __post_init__(self) -> None:
        for ck, n in ((self.lora_ckpts, self.lora_steps), (self.ft_ckpts, self.ft_steps), (self.pb_save, self.pb_steps)):
            assert ck == tuple(sorted(ck)) and ck[-1] <= n, (ck, n)
        assert self.pb_save[-1] == self.pb_steps, "the pull-back is scored at its last step"
        assert 8 in self.ranks, "rank 8 is the reference rank (seeds, Pile check, P5)"


FULL = Sizes(ranks=(1, 4, 8, 32), lora_steps=100, lora_ckpts=(10, 30, 100), ft_steps=100, ft_ckpts=(10, 100), pb_steps=30,
             pb_save=(10, 30), pb_lrs=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3), stoch_draws=4, pgd_steps=20, pile_rows=256,
             pile_pgd_steps=20, pile_seeds=(0, 1, 2), suffix="")
SMOKE = Sizes(ranks=(1, 8), lora_steps=2, lora_ckpts=(1, 2), ft_steps=2, ft_ckpts=(1, 2), pb_steps=2, pb_save=(1, 2),
              pb_lrs=(1e-3,), stoch_draws=1, pgd_steps=2, pile_rows=8, pile_pgd_steps=2, pile_seeds=(0,), suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "about": (
        "lora_commute (2026-10-19): fine-tune the emoticon -> `o` behaviour as a LoRA on h.2.mlp.down_proj and commute it past "
        "the released decomposition with the maps on hand (Euclidean lift of the finished delta-W; pull-back = training the "
        "subcomponents, here confined to the LoRA's matrix and rank), beside the full-weight fine-tune under the same maps. "
        "The user's question: is a LoRA update easier to decompose than a normal fine-tuning update (instinct: yes). Plan "
        "reports/2026-10-19_plan_lora-lift-and-pullback.md plus its addendum. Filed before any arm was trained."
    ),
    "setup": {
        "windows": "round 1's emoticon windows (365 train, 73 held-out, trigger at index 20, target token `o`)",
        "LoRA_r": f"rank r in {FULL.ranks} on {SITE}, Adam lr {LORA_LR}, arm A's loss, batch 32, clip 1, {FULL.lora_steps} steps, "
                  f"checkpoints {FULL.lora_ckpts}. Scored step = the first checkpoint with held-out P(o) >= {P_LEARN} (rank 1 "
                  "reaches 0.22 at step 10 and 0.99 at step 30 in round 2); the step-100 state is the finished LoRA, reported "
                  "against the same thresholds. Measured do-nothing (delta-W in Delta) and lifted",
        "lift": "split.split_state: Euclidean (minimum-norm) lift of the matrix's delta-W onto its alive subcomponents, tol 1e-6; "
                "target weights W0 + delta-W, so Delta moves by O(delta-W) - dU dV^T",
        "ft": f"full-weight fine-tune, arm A recipe (Adam lr {FT_LR}, all 24 matrices, seed 0), checkpoints {FULL.ft_ckpts}; "
              "do-nothing and lifted on all 24 matrices (round 2's lift). Step 10 is the matched step",
        "pb_r": f"component_training.train on {SITE} only with rank-r offsets dU = P S^T, dV = Q T^T (weight change rank <= 2r), Adam, "
                f"{FULL.pb_steps} steps, lr the smallest in {FULL.pb_lrs} with held-out P(o) >= {P_LEARN} and kl_other <= {KL_LIMIT} "
                "at the last step (amendment 1's rule); none passing = did not learn (reported at the lr with the highest P(o))",
        "pb_full": f"the same with free offsets on every alive subcomponent of {SITE}",
        "pb_all": f"free offsets on all 24 matrices, lr {PB_ALL_LR} fixed (the ladder's emoticon arm), {FULL.pb_steps} steps",
        "seeds": "LoRA_8 and pb_8 (at the lr selected for seed 0) repeated with optimiser/init seed 1",
        "W0": f"the released decomposition under round 2's Pile harness with adversary seeds {FULL.pile_seeds}",
    },
    "measurements": {
        "routes": "local_faith.route_split on the held-out windows: d, r, i, e_gate, e_delta (sum to 1), gain in nats; "
                  "P(o) of the target (T1) and of the components alone unmasked (U1)",
        "pile": "faithfulness.pile_faithfulness (2 blocks of 128 val rows 0..255, one adversary per block, 20 steps of 0.1; "
                "stochastic, PGD, CI-masked, unmasked KL with Delta in/out as in rounds 1-3); ratio = arm / W0 seed 0",
        "pgd_floor": "f = max/min - 1 of the W0 PGD KL over the 3 adversary seeds; 'within 3 floors' = ratio <= 1 + 3 f",
        "inexpressible": "split.weighted_fractions with Sigma = E[a a^T] of the matrix's input over the train windows: the share "
                         "of the update's output energy no first-order change of the alive subcomponents can express; per "
                         "matrix for the LoRA, pooled by energy over the non-saturated matrices for ft",
        "moved": f"subcomponents whose rank-1 matrix changes by more than {MOVED_CUT:.0%} of its norm (identity.change_stats); "
                 f"the top {TOP_K} by ||Delta(u v^T)|| for P5",
        "S_on": f"recruit.switched at {DG_THR}: alive subcomponents whose mean trigger gate rises by > {DG_THR}; Jaccard with the "
                "ft do-nothing arm's set (and with the recruit test's ft_s0 file if present)",
        "local": "local_faith.local_faithfulness: reported only (its adversaries saturate; reports/2026-10-19_review_pgd-local.md)",
    },
    "blind_to": {
        "d, r, e_delta": "everything away from the trigger (cover: the LoRA's kl_other and the Pile ratios)",
        "Pile PGD ratio": "a trigger-specific attack (the adversary is shared over 65k Pile positions holding ~3 triggers; "
                          "cover: round 2's calibration, the W0 floor); the task-local adversary has no power (review above)",
        "moved": "changes spread below 1% (the identity test's regime)",
        "controls": "G2 for the lift (the state must be the LoRA's model exactly), the lr rule for the pull-back (it must learn), "
                    "the W0 seeds for the PGD floor, the reproduction gates G4 for the harness",
    },
    "gates": [
        {"id": "G0_floor", "rule": "m = max(0.1, 3 x max |Delta d|, |Delta r| between seeds 0 and 1 of pb_8 (step 30) and lift_8 "
                                   "(scored step)); f = PGD floor as above"},
        {"id": "G1_arms_learn", "rule": f"every LoRA_r reaches P(o) >= {P_LEARN} at some checkpoint; every pb_r finds an lr by the rule",
         "if_fail": "the arm is reported, not scored; its failure is a result"},
        {"id": "G2_lift_exact", "rule": "for every lift: |P(o) of the lifted state's target model - the fine-tune's held-out P(o)| "
                                        "<= 1e-3 and |sum of shares - 1| <= 1e-6", "if_fail": "the lift or the Delta bookkeeping is wrong; stop",
         "reported_beside": "P(o) of the lifted components alone (U1; round 2's liftable part gave 0.954 vs 0.999 for LoRA_8 step 10)"},
        {"id": "G4_reproduces", "rule": "nothing_ft and lift_ft Pile PGD ratios at step 10 within 25% of 2.18 and 5.41 (round 2); "
                                        "pb_all d and r at step 10 within 0.05 of 0.025 and 0.53 (ladder); nothing LoRA_8 r at step 10 "
                                        "within 0.05 of 0.54 (recruit)", "if_fail": "find the cause before the analysis"},
    ],
    "predictions": [
        {"id": "P1_smallest_lora_commutes", "scored": True, "arm": "lift_1 at its scored step",
         "pass": "d >= 0.5 and e_delta <= 0.2", "null": "d < 0.2", "predicted": "pass",
         "reported": "the step-100 lift against the same thresholds; if d differs from the scored step by more than m, said "
                     "beside A's cosine with its initialisation"},
        {"id": "P2_stops_with_rank", "scored": True, "arm": "lift_r at scored steps, r = 1 vs 32",
         "pass": "d(lift_1) - d(lift_32) > m and outside_share(32) > outside_share(1)", "null": "|d(lift_1) - d(lift_32)| <= m",
         "predicted": "pass", "expected_inexpressible": "round 2 found 0.40 (r1 step 30), 0.41 (r8 step 10), 0.30 (r8 step 100) "
                                                        "against a null of 0.42 on this matrix: the LoRA's read side is near its random init"},
        {"id": "P3_lift_breaks_adversarially", "scored": True, "arm": "lift_r at scored steps, Pile PGD ratio",
         "pass": "lift_8 and lift_32 >= 1.5 and lift_1 <= 1 + 3 f", "null": "all of lift_1, lift_4, lift_8, lift_32 <= 1 + 3 f",
         "predicted": "pass", "note": "the plan filed this on the task-local pgd_trigger ratio; that adversary saturates at W0 "
                                      "(review 2026-10-19), so the Pile adversary that found round 2's fragility is used instead"},
        {"id": "P4_inside_route_carries_it_when_confined", "scored": True, "arm": "pb_r at step 30, every rank that learns",
         "pass": "d >= 0.3 at every learned rank", "null": "d < 0.1 at every learned rank", "predicted": "pass (the most likely to fail)"},
        {"id": "P5_same_subcomponents", "scored": True, "arm": "lift_8 (scored step) vs pb_8 (step 30)",
         "pass": f">= 8 of the top {TOP_K} by ||Delta(u v^T)|| shared", "null": "<= 3 shared", "predicted": "null"},
        {"id": "C1_lora_lifts_better_than_ft", "scored": True, "arm": "lift_8 (scored step) vs lift_ft (step 10)",
         "pass": "d(lift_8) - d(lift_ft) >= m", "null": "d(lift_8) - d(lift_ft) <= 0", "predicted": "pass (the user's instinct)"},
        {"id": "C2_lora_lift_less_fragile_than_ft", "scored": True, "arm": "Pile PGD ratio, lift_8 vs lift_ft (step 10)",
         "pass": "lift_8 <= 1.5 and lift_ft > 1.5", "null": "lift_8 > 1.5", "predicted": "pass (the user's instinct; uncertain)"},
        {"id": "C3_confined_pullback_beats_all_matrix", "scored": True, "arm": "pb_full vs pb_all, step 30",
         "pass": "d(pb_full) - d(pb_all) >= m", "null": "d(pb_full) - d(pb_all) <= 0", "predicted": "pass"},
    ],
    "reported_not_scored": "every arm's full route split, task-local harness, retained shares, moved counts, S_on sizes and Jaccards; "
                           "the LoRA's cosine with its initialisation per checkpoint; the realised rank of every pull-back's delta-W; "
                           "pb_r at step 10; the lr sweeps; Pile ratios of pb_r, pb_full and the do-nothing arms",
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


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; predictions are not edited after filing (add a dated report instead)")
    dump(PRED.name, PREDICTIONS)
    print(f"filed {PRED}")


# ----------------------------------------------------------------------------- context and one measurement


def load_all(device: str, S: Sizes) -> dict[str, Any]:
    from commute import component_training, split, vpd
    from commute.data import batches
    from commute.faithfulness import pile_blocks, pile_faithfulness
    from commute.local_faith import local_faithfulness, route_cells

    model = vpd.load_component_model(device)
    alive = {k: v.cpu().bool() for k, v in torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"].items()}
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    train, held = win["train"], win["heldout"]
    W0 = vpd.target_weights(model)
    uv = component_training.alive_uv(model, alive)
    uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()}
    t0 = time.time()
    bases = split.bases(uv_cpu, TOL)
    sigma = {k: v.cpu() for k, v in split.input_second_moments(model, batches(train.tokens, 32), list(W0)).items()}
    log(f"bases and input moments for 24 matrices in {time.time() - t0:.0f} s")
    ctx: dict[str, Any] = {"device": device, "model": model, "W0": W0, "comps0": vpd.component_weights(model), "alive": alive,
                           "uv": uv, "uv_cpu": uv_cpu, "bases": bases, "sigma": sigma, "train": train, "held": held,
                           "token": o_token, "wd0": vpd.weight_deltas(model), "on": {}, "change": {}, "S": S}
    ctx["non_saturated"] = [k for k in W0 if bases[k].r_u < W0[k].shape[0] and bases[k].r_v < W0[k].shape[1]]
    ctx["base"] = local_faithfulness(model, held.tokens, held.half, o_token, S.stoch_draws, S.pgd_steps, log=log)
    ctx["rbase"] = route_cells(model, held.tokens, held.half, o_token, BATCH)
    ctx["pile"] = {}
    if S.pile_rows:
        ctx["blocks"] = pile_blocks(S.pile_rows, log)
        for seed in S.pile_seeds:
            t0 = time.time()
            ctx["pile"][seed] = pile_faithfulness(model, ctx["blocks"], seed, S.pile_pgd_steps, log)
            log(f"[pile W0 seed {seed}] {time.time() - t0:.0f} s")
    return ctx


def pile_ratios(ctx: dict[str, Any]) -> dict[str, float]:
    """Round 2's Pile harness on the model's current state, as ratios to W0 (adversary seed 0)."""
    from commute.faithfulness import pile_faithfulness

    f = pile_faithfulness(ctx["model"], ctx["blocks"], 0, ctx["S"].pile_pgd_steps, log)
    return {k: f[k] / ctx["pile"][0][k] for k in f}


def moved(ctx: dict[str, Any], name: str, state: dict[str, tuple[Tensor, Tensor, Tensor]]) -> dict[str, Any]:
    """How many alive subcomponents changed by more than MOVED_CUT of their norm, and the TOP_K by absolute change."""
    from commute.identity import change_stats

    absolute: dict[str, Tensor] = {}
    n_moved = 0
    for k, (_, U2, V2) in state.items():
        U, V = ctx["uv_cpu"][k]
        st = change_stats(U, V, U2, V2)
        absolute[k] = (st["rel_change"] * U.double().norm(dim=0) * V.double().norm(dim=0)).float()
        n_moved += int((st["rel_change"] > MOVED_CUT).sum())
    ctx["change"][name] = absolute
    idx = {k: ctx["alive"][k].nonzero().flatten().tolist() for k in absolute}
    flat = sorted(((float(v[c]), f"{k}:{idx[k][c]}") for k, v in absolute.items() for c in range(v.numel())), reverse=True)
    return {"n_moved": n_moved, "n_trainable": sum(int(v.numel()) for v in absolute.values()), "top": [key for _, key in flat[:TOP_K]]}


def measure(ctx: dict[str, Any], name: str, dW: dict[str, Tensor], state: dict[str, tuple[Tensor, Tensor, Tensor]] | None,
            pile: bool) -> dict[str, Any]:
    """Target weights W0 + dW with the components of `state` (None: do-nothing). Route split, local harness, S_on,
    Delta change, moved subcomponents (state arms), Pile ratios (if `pile`)."""
    from commute import component_training, vpd
    from commute import recruit as R
    from commute.faithfulness import model_state
    from commute.local_faith import (
        local_faithfulness,
        retained,
        route_cells,
        route_split,
    )

    model, W0, held, token, S = ctx["model"], ctx["W0"], ctx["held"], ctx["token"], ctx["S"]
    target = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
    comps = None if state is None else component_training.components_with(model, ctx["alive"], {k: (v[1], v[2]) for k, v in state.items()})
    rec: dict[str, Any] = {"rel_norm_max": max(float(dW[k].norm() / W0[k].cpu().norm()) for k in dW)}
    with model_state(model, target=target, components=comps):
        wd = vpd.weight_deltas(model)
        rec["delta_change_frac"] = (sum(float(((wd[k] - ctx["wd0"][k]).double() ** 2).sum()) for k in wd)
                                    / sum(float((dW[k].double() ** 2).sum()) for k in dW))
        rec["routes"] = route_split(model, held.tokens, held.half, token, ctx["rbase"], ctx["comps0"], BATCH)
        cur = route_cells(model, held.tokens, held.half, token, BATCH)
        rec["local"] = local_faithfulness(model, held.tokens, held.half, token, S.stoch_draws, S.pgd_steps, log=log)
        if pile:
            rec["pile"] = pile_ratios(ctx)
    rec["retained"] = retained(rec["local"], ctx["base"])
    on = R.switched(R.trigger_dg(cur["gates"], ctx["rbase"]["gates"], held.half, ctx["alive"]), DG_THR)
    ctx["on"][name] = on
    rec["n_on"] = R.size(on)
    if state is not None:
        rec["moved"] = moved(ctx, name, state)
    s, pc = rec["routes"]["shares"], rec["routes"]["p_cells"]
    log(f"[{name}] gain {rec['routes']['gain']:.2f} d {s['d']:.3f} r {s['r']:.3f} i {s['i']:.3f} e_gate {s['e_gate']:.3f} "
        f"e_delta {s['e_delta']:.3f} sum {rec['routes']['share_sum']:.6f} | P(o) target {pc['T1']:.3f} unmasked {pc['U1']:.3f} "
        f"ci {pc['L11']:.3f} | Delta change {rec['delta_change_frac']:.3f} n_on {rec['n_on']}"
        + (f" moved {rec['moved']['n_moved']}/{rec['moved']['n_trainable']}" if state is not None else "")
        + (f" | pile {json.dumps({k: round(v, 3) for k, v in rec['pile'].items()})}" if pile else ""))
    return rec


def outside_share(ctx: dict[str, Any], dW: dict[str, Tensor]) -> dict[str, Any]:
    """Input-weighted outside_share share per non-zero matrix and pooled by energy over the non-saturated ones."""
    from commute import split

    per: dict[str, Any] = {}
    for k, M in dW.items():
        if not M.any():
            continue
        M, Sg = M.double(), ctx["sigma"][k]
        per[k] = {**split.weighted_fractions(M, ctx["bases"][k], Sg), "energy": float(((M @ Sg) * M).sum())}
    ks = [k for k in per if k in ctx["non_saturated"]]
    e = sum(per[k]["energy"] for k in ks)
    return {"per_matrix": per, "pooled": sum(per[k]["omega_act"] * per[k]["energy"] for k in ks) / e,
            "pooled_null": sum(per[k]["null_act"] * per[k]["energy"] for k in ks) / e}


def zero_dw(ctx: dict[str, Any], state: dict[str, tuple[Tensor, Tensor, Tensor]]) -> dict[str, Tensor]:
    """A full 24-matrix delta-W from a state that holds only some matrices (the others did not move)."""
    return {k: (state[k][0] if k in state else torch.zeros_like(ctx["W0"][k].cpu())) for k in ctx["W0"]}


def a_cfg(ctx: dict[str, Any], steps: int, ckpts: tuple[int, ...], seed: int) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=ctx["token"], lr=FT_LR, steps=steps, checkpoints=ckpts, seed=seed)


# ----------------------------------------------------------------------------- arms


def lora_arm(ctx: dict[str, Any], rank: int, seed: int, pile: bool) -> dict[str, Any]:
    """LoRA_r: train, then do-nothing and lifted at the scored step and at the last checkpoint."""
    from commute import split, vpd
    from commute.finetune import LoraConfig, finetune_lora_arm_a, lora_factors

    S, model = ctx["S"], ctx["model"]
    cfg = LoraConfig(modules=(SITE,), rank=rank, lr=LORA_LR, steps=S.lora_steps, checkpoints=S.lora_ckpts, seed=seed)
    res = finetune_lora_arm_a(model, ctx["train"], ctx["held"], a_cfg(ctx, S.lora_steps, (), seed), cfg, ctx["device"], log)
    A0 = lora_factors({SITE: vpd.target_linear(model, SITE)}, rank, seed)[SITE][1].detach().cpu()
    scored = next((s for s in S.lora_ckpts if res.evals[s]["p_o"] >= P_LEARN), None)
    name = f"lora_r{rank}_s{seed}"
    torch.save(res.factors, LARGE / f"{name}_factors.pt")
    rec: dict[str, Any] = {"evals": res.evals, "scored_step": scored, "learned": scored is not None, "steps": {},
                           "cos_init": {s: float(torch.nn.functional.cosine_similarity(f[SITE][1], A0, dim=1).abs().mean())
                                        for s, f in res.factors.items()}}
    for step in sorted({scored or S.lora_ckpts[-1], S.lora_ckpts[-1]}):
        dW = res.deltas[step]
        st = split.split_state(dW, ctx["uv_cpu"], ctx["bases"])
        both = {"nothing": measure(ctx, f"{name}_step{step}_nothing", dW, None, pile and step == scored and rank == 8),
                "lift": measure(ctx, f"{name}_step{step}_lift", dW, st, pile and step == scored),
                "inexpressible": outside_share(ctx, dW)}
        both["G2"] = {"p_o_diff": abs(both["lift"]["routes"]["p_cells"]["T1"] - res.evals[step]["p_o"]),
                      "share_sum": both["lift"]["routes"]["share_sum"]}
        both["G2"]["ok"] = both["G2"]["p_o_diff"] <= 1e-3 and abs(both["G2"]["share_sum"] - 1) <= 1e-6
        rec["steps"][str(step)] = both
        log(f"[{name} step {step}] G2 {both['G2']} outside_share {both['inexpressible']['pooled']:.3f} "
            f"(null {both['inexpressible']['pooled_null']:.3f}) cos_init {rec['cos_init'][step]:.3f}")
    return rec


def ft_arm(ctx: dict[str, Any]) -> dict[str, Any]:
    """The full-weight fine-tune: do-nothing and lifted on all 24 matrices at every checkpoint (Pile at the first)."""
    from commute import split
    from commute.finetune import finetune_arm_a

    S = ctx["S"]
    res = finetune_arm_a(ctx["model"], ctx["train"], ctx["held"], a_cfg(ctx, S.ft_steps, S.ft_ckpts, 0), ctx["device"], log)
    rec: dict[str, Any] = {"evals": res.evals, "steps": {}}
    for step in S.ft_ckpts:
        dW = res.deltas[step]
        pile = bool(S.pile_rows) and step == S.ft_ckpts[0]
        st = split.split_state(dW, ctx["uv_cpu"], ctx["bases"])
        both = {"nothing": measure(ctx, f"ft_step{step}_nothing", dW, None, pile),
                "lift": measure(ctx, f"ft_step{step}_lift", dW, st, pile), "inexpressible": outside_share(ctx, dW)}
        both["G2"] = {"p_o_diff": abs(both["lift"]["routes"]["p_cells"]["T1"] - res.evals[step]["p_o"]),
                      "share_sum": both["lift"]["routes"]["share_sum"]}
        both["G2"]["ok"] = both["G2"]["p_o_diff"] <= 1e-3 and abs(both["G2"]["share_sum"] - 1) <= 1e-6
        rec["steps"][str(step)] = both
        log(f"[ft step {step}] G2 {both['G2']} outside_share pooled {both['inexpressible']['pooled']:.3f} "
            f"(null {both['inexpressible']['pooled_null']:.3f})")
    return rec


def realised_rank(dW: Tensor, rank: int | None) -> dict[str, Any]:
    s = torch.linalg.svdvals(dW.double())
    e = s**2 / (s**2).sum()
    out: dict[str, Any] = {"rank_1e-3": int((s > 1e-3 * s[0]).sum())}
    if rank is not None:
        out.update({"energy_top_r": float(e[:rank].sum()), "energy_top_2r": float(e[: 2 * rank].sum())})
    return out


def pb_arm(ctx: dict[str, Any], name: str, keys: list[str], rank: int | None, lrs: tuple[float, ...], seed: int,
           pile: bool) -> dict[str, Any]:
    """Pull-back on `keys` (rank-r or free offsets) at each lr; the smallest lr passing the rule is measured at every
    saved step (Pile at the last); none passing = did not learn, measured at the highest-P(o) lr."""
    from commute import component_training

    S, W0 = ctx["S"], ctx["W0"]
    uv = {k: ctx["uv"][k] for k in keys}
    sweep: dict[str, Any] = {}
    states: dict[float, Any] = {}
    for lr in lrs:
        cfg = component_training.TrainConfig(lr=lr, steps=S.pb_steps, save_steps=S.pb_save, rank=rank)
        curve, st = component_training.train(ctx["model"], uv, ctx["train"].tokens, ctx["held"], a_cfg(ctx, S.pb_steps, (), seed), cfg,
                                   ctx["device"], W0, log=log)
        last = curve[-1]
        sweep[f"{lr:g}"] = {"p_o": last["heldout_p_o"], "kl_other": last["heldout_kl_other"], "rel_norm_max": last["rel_norm_max"],
                            "passes": last["heldout_p_o"] >= P_LEARN and last["heldout_kl_other"] <= KL_LIMIT, "curve": curve}
        states[lr] = st
    ok = [lr for lr in lrs if sweep[f"{lr:g}"]["passes"]]
    sel = min(ok) if ok else max(lrs, key=lambda lr: sweep[f"{lr:g}"]["p_o"])
    rec: dict[str, Any] = {"sweep": {lr: {k: v for k, v in s.items() if k != "curve"} for lr, s in sweep.items()},
                           "curve": sweep[f"{sel:g}"]["curve"], "selected_lr": sel, "learned": bool(ok), "rank": rank, "steps": {}}
    log(f"[{name}] lr rule: {'selected' if ok else 'NONE PASSES; reporting'} lr {sel:g} "
        + json.dumps({lr: (round(s['p_o'], 3), round(s['kl_other'], 4)) for lr, s in rec['sweep'].items()}))
    for step in S.pb_save:
        st = states[sel][step]
        m = measure(ctx, f"{name}_step{step}", zero_dw(ctx, st), st, pile and step == S.pb_steps)
        if len(keys) == 1:
            m["realised_rank"] = realised_rank(st[SITE][0], rank)
        rec["steps"][str(step)] = m
    if len(keys) == 1:
        torch.save(states[sel][S.pb_steps], LARGE / f"{name}_state_step{S.pb_steps}.pt")
    return rec


# ----------------------------------------------------------------------------- the run and the score


def jaccards(ctx: dict[str, Any], ref: str) -> dict[str, Any]:
    from commute import recruit as R

    out: dict[str, Any] = {"ref": ref, "with_ref": {n: R.jaccard(on, ctx["on"][ref]) for n, on in ctx["on"].items() if n != ref}}
    if RECRUIT_DG.is_file():
        rec_on = R.switched(torch.load(RECRUIT_DG, weights_only=False)["dg"], DG_THR)
        out["recruit_ft_s0_vs_ref"] = R.jaccard(ctx["on"][ref], rec_on)
    return out


def stage_all(args: argparse.Namespace) -> None:
    assert PRED.is_file(), "file predictions first: run.py --predictions"
    S = SMOKE if args.smoke else FULL
    t_start = time.time()
    ctx = load_all(args.device, S)
    pile = bool(S.pile_rows)
    f = ctx["pile"]
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S), "W0": {
        "base": ctx["base"], "cells": {k: ctx["rbase"][k] for k in ("T", "U", "L", "p_T", "p_U", "p_L")}, "pile": f,
        "pgd_floor": (max(v["pgd_kl"] for v in f.values()) / min(v["pgd_kl"] for v in f.values()) - 1) if f else None}}
    name = f"lora_commute{S.suffix}.json"
    out["ft"] = ft_arm(ctx)
    dump(name, out)
    out["lora"] = {}
    for rank in S.ranks:
        out["lora"][str(rank)] = lora_arm(ctx, rank, 0, pile)
        dump(name, out)
    out["pb"] = {}
    for rank in S.ranks:
        out["pb"][f"pb_r{rank}"] = pb_arm(ctx, f"pb_r{rank}", [SITE], rank, S.pb_lrs, 0, pile)
        dump(name, out)
    out["pb"]["pb_full"] = pb_arm(ctx, "pb_full", [SITE], None, S.pb_lrs, 0, pile)
    out["pb"]["pb_all"] = pb_arm(ctx, "pb_all", list(ctx["W0"]), None, (PB_ALL_LR,), 0, pile)
    dump(name, out)
    out["seeds"] = {"lora_r8_s1": lora_arm(ctx, 8, 1, False),
                    "pb_r8_s1": pb_arm(ctx, "pb_r8_s1", [SITE], 8, (out["pb"]["pb_r8"]["selected_lr"],), 1, False)}
    out["jaccard"] = jaccards(ctx, f"ft_step{S.ft_ckpts[0]}_nothing")
    out["score"] = score(out, S)
    out["seconds"] = time.time() - t_start
    dump(name, out)
    log(f"wrote {RESULTS / name} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


def _shares(rec: dict[str, Any]) -> dict[str, float]:
    return rec["routes"]["shares"]


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    """A prediction record: its numbers, and `scored` = every condition in `cond` holds (gates, arms learned, gain >= 1)."""
    return {**values, "scored": all(cond.values()), "conditions": cond}


def score(out: dict[str, Any], S: Sizes) -> dict[str, Any]:
    """The filed rules. A prediction is scored only if the arms it reads learned (G1), their lifts pass G2 and their
    gains are >= 1 nat (route_split 'scorable'); G4 misses are flagged in the verdict, not used to unscore."""
    L, PB, FT, last = out["lora"], out["pb"], out["ft"], str(S.pb_steps)

    def lora_step(r: int, rec: dict[str, Any] | None = None) -> dict[str, Any]:
        arm: dict[str, Any] = L[str(r)] if rec is None else rec
        return arm["steps"][str(arm["scored_step"] or S.lora_ckpts[-1])]

    def ok_lift(step_rec: dict[str, Any]) -> bool:
        return bool(step_rec["G2"]["ok"]) and bool(step_rec["lift"]["routes"]["scorable"])

    def ok_pb(name: str) -> bool:
        return bool(PB[name]["learned"]) and bool(PB[name]["steps"][last]["routes"]["scorable"])

    learned = {r: bool(L[str(r)]["learned"]) for r in S.ranks}
    lift_ok = {r: learned[r] and ok_lift(lora_step(r)) for r in S.ranks}
    ft10 = FT["steps"][str(S.ft_ckpts[0])]
    s8, s8b = _shares(lora_step(8)["lift"]), _shares(lora_step(8, out["seeds"]["lora_r8_s1"])["lift"])
    p8, p8b = _shares(PB["pb_r8"]["steps"][last]), _shares(out["seeds"]["pb_r8_s1"]["steps"][last])
    spread = max(max(abs(s8[k] - s8b[k]), abs(p8[k] - p8b[k])) for k in ("d", "r"))
    m = max(0.1, 3 * spread)
    fl = out["W0"]["pgd_floor"]
    res: dict[str, Any] = {"G0": {"seed_spread": spread, "m": m, "pgd_floor": fl},
                           "G1": {**{f"lora_r{r}": learned[r] for r in S.ranks}, **{n: PB[n]["learned"] for n in PB}},
                           "G2": {**{f"lora_r{r}_step{st}": b["G2"]["ok"] for r in S.ranks for st, b in L[str(r)]["steps"].items()},
                                  **{f"ft_step{st}": b["G2"]["ok"] for st, b in FT["steps"].items()}}}
    pb10 = _shares(PB["pb_all"]["steps"][str(S.pb_save[0])])
    g4: dict[str, Any] = {"pb_all_d": pb10["d"], "pb_all_r": pb10["r"], "nothing_lora8_r": _shares(lora_step(8)["nothing"])["r"]}
    if "pile" in ft10["lift"]:
        g4.update({"nothing_ft_pgd_ratio": ft10["nothing"]["pile"]["pgd_kl"], "lift_ft_pgd_ratio": ft10["lift"]["pile"]["pgd_kl"]})
    g4["ok"] = {k: (abs(v - REF[k]) <= 0.25 * REF[k] if "pgd" in k else abs(v - REF[k]) <= 0.05) for k, v in g4.items()}
    res["G4"] = g4
    verdict = []
    if not all(res["G2"].values()):
        verdict.append("G2 failed on " + ", ".join(k for k, v in res["G2"].items() if not v) + ": the lift or Delta bookkeeping is wrong")
    if not all(g4["ok"].values()):
        verdict.append("G4 missed on " + ", ".join(k for k, v in g4["ok"].items() if not v) + ": find the cause before the analysis")
    res["verdict"] = "; ".join(verdict) or "gates ok"

    lo_r, hi_r = min(S.ranks), max(S.ranks)
    d = {r: _shares(lora_step(r)["lift"])["d"] for r in S.ranks}
    e_delta = {r: _shares(lora_step(r)["lift"])["e_delta"] for r in S.ranks}
    inex = {r: lora_step(r)["inexpressible"]["pooled"] for r in S.ranks}
    res["P1"] = _pred({f"lift_{lo_r}": lift_ok[lo_r]}, d=d[lo_r], e_delta=e_delta[lo_r],
                      **{"pass": d[lo_r] >= 0.5 and e_delta[lo_r] <= 0.2, "null": d[lo_r] < 0.2})
    res["P2"] = _pred({f"lift_{lo_r}": lift_ok[lo_r], f"lift_{hi_r}": lift_ok[hi_r]}, d=d, outside_share=inex,
                      **{"pass": d[lo_r] - d[hi_r] > m and inex[hi_r] > inex[lo_r], "null": abs(d[lo_r] - d[hi_r]) <= m})
    pgd = {r: lora_step(r)["lift"].get("pile", {}).get("pgd_kl") for r in S.ranks}
    if all(v is not None for v in pgd.values()) and fl is not None:
        lim = 1 + 3 * fl
        res["P3"] = _pred({f"lift_{r}": lift_ok[r] for r in S.ranks}, pgd_ratio=pgd, limit_3_floors=lim,
                          **{"pass": pgd[8] >= 1.5 and pgd[hi_r] >= 1.5 and pgd[lo_r] <= lim, "null": all(v <= lim for v in pgd.values())})
    pb_learned = [r for r in S.ranks if ok_pb(f"pb_r{r}")]
    dpb = {r: _shares(PB[f"pb_r{r}"]["steps"][last])["d"] for r in S.ranks}
    res["P4"] = _pred({"some_pb_r_learned": bool(pb_learned)}, d=dpb, learned_ranks=pb_learned,
                      **{"pass": all(dpb[r] >= 0.3 for r in pb_learned), "null": all(dpb[r] < 0.1 for r in pb_learned)})
    top_l, top_p = set(lora_step(8)["lift"]["moved"]["top"]), set(PB["pb_r8"]["steps"][last]["moved"]["top"])
    res["P5"] = _pred({"lift_8": lift_ok[8], "pb_r8": ok_pb("pb_r8")}, shared=len(top_l & top_p),
                      **{"pass": len(top_l & top_p) >= 8, "null": len(top_l & top_p) <= 3})
    ft_ok = ok_lift(ft10)
    d_ft = _shares(ft10["lift"])["d"]
    res["C1"] = _pred({"lift_8": lift_ok[8], "lift_ft": ft_ok}, d_lift_8=d[8], d_lift_ft=d_ft,
                      **{"pass": d[8] - d_ft >= m, "null": d[8] - d_ft <= 0})
    if pgd[8] is not None and "pile" in ft10["lift"]:
        pf = ft10["lift"]["pile"]["pgd_kl"]
        res["C2"] = _pred({"lift_8": lift_ok[8], "lift_ft": ft_ok}, pgd_lift_8=pgd[8], pgd_lift_ft=pf,
                          **{"pass": pgd[8] <= 1.5 < pf, "null": pgd[8] > 1.5})
    d_full, d_all = _shares(PB["pb_full"]["steps"][last])["d"], _shares(PB["pb_all"]["steps"][last])["d"]
    res["C3"] = _pred({"pb_full": ok_pb("pb_full"), "pb_all": ok_pb("pb_all")}, d_pb_full=d_full, d_pb_all=d_all,
                      **{"pass": d_full - d_all >= m, "null": d_full - d_all <= 0})
    return res


def stage_w0seeds(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.local_faith import local_faithfulness

    S = SMOKE if args.smoke else FULL
    model = vpd.load_component_model(args.device)
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    held, token = win["heldout"], json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    per = {seed: local_faithfulness(model, held.tokens, held.half, token, S.stoch_draws, S.pgd_steps, seed=seed, log=log)
           for seed in (0, 1, 2)}
    spread = {v: {st: max(p[v][st] for p in per.values()) / max(min(p[v][st] for p in per.values()), 1e-12)
                  for st in ("kl_trigger", "kl_other")} for v in ("stochastic", "pgd_all", "pgd_trigger")}
    dump(f"w0_local_seeds{S.suffix}.json", {"per_seed": per, "spread_ratio_max_over_min": spread})
    log(f"w0seeds: spread ratios {json.dumps(spread)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all", "w0seeds"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage == "all":
        stage_all(args)
    elif args.stage == "w0seeds":
        stage_w0seeds(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
