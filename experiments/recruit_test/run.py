"""Recruit test: does the REAL update (a W fine-tune, a LoRA) recruit the same pre-existing subcomponents through the
frozen Gamma that our pull-back map does, and is that set specific to the target token?

Plan and design check: reports/plans/2026-10-19_plan_recruit-test.md (confirmed by the user 2026-10-19, "go for it").
Every arm is a do-nothing placement: target W = W0 + delta-W, components unchanged, Gamma frozen, Delta out of the
masked model. So d = i = 0 and e_Delta = 1 by construction (logged, G3); the measurements are the re-gating share r,
the switched-on set S_on at the trigger (mean g_state - g_W0 > thr over the 73 held-out emoticon windows) and the
Jaccard overlaps between arms, against a seed-to-seed floor and two different-target nulls.

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  all             GPU -> results/recruit.json (+ results/large/<arm>_dg.pt, small)
  all --smoke     GPU, minutes: 2-step arms, 16 Pile rows; checks every path runs -> results/recruit_smoke.json

Usage (repo root):  uv run python experiments/recruit_test/run.py all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"

FT_LR, FT_STEP, FT_SEEDS = 3e-5, 10, (0, 1)
NULL_STEPS = (10, 30)  # the null target's fine-tune is scored at step 10 if it learned (P >= 0.5), else at step 30
LORA_SITE, LORA_RANK, LORA_LR, LORA_STEPS, LORA_CKPTS = "h.2.mlp.down_proj", 8, 1e-3, 100, (1, 2, 3, 5, 10, 30, 100)
LORA_SCORED = 10
PB_LR, PB_STEPS = 1e-5, 30
DG_THRS, DG_THR = (0.1, 0.3, 0.5), 0.3
PILE_ROWS = ("val", 2048, 256)  # round 2's faithfulness rows
MIN_NAT_POSITIONS = 200
N_RANDOM_SETS = 20
AMEND2_REF = {"n_on_0.3": 135, "gap": 5.8926}  # concentration test, amendment 2, do_nothing arm (pull-back's delta-W in Delta)
BATCH = 64

PREDICTIONS: dict[str, Any] = {
    "about": (
        "recruit_test: the concentration test's amendment 2 found that every PULL-BACK arm, with its delta-W put into Delta, "
        "makes the frozen Gamma switch on the same 135-250 pre-existing subcomponents at the emoticon trigger (Jaccard "
        "0.59-0.88), which carry about half of the re-gating effect. That compared our map against our map. Here the real "
        "updates are measured the same way: the W fine-tune (round 1's recipe), a LoRA on the paper's site, against a "
        "seed-to-seed noise floor and two fine-tunes to a different target token at the same trigger (the null with power). "
        "Filed before any arm was trained."
    ),
    "setup": {
        "placement": "every arm: target W = W0 + delta-W, released components unchanged, Gamma frozen (reads the new "
                     "activations), Delta out of the masked model. Hence d = i = 0 and e_Delta = 1 exactly (G3).",
        "windows": "round 1's emoticon windows (experiments/transport_test/results/large/windows.pt): train for the "
                   "fine-tunes, the 73 held-out ones for every measurement; trigger at index 20; target token 80 (`o`).",
        "arms": {
            "ft_s0": f"W fine-tune, arm A recipe (Adam lr {FT_LR}, batch 32, clip 1, kl_lambda 1, all 24 matrices), seed 0, step {FT_STEP}",
            "ft_s1": "same, optimiser/sampling seed 1 (the seed-to-seed noise floor of the switched-on set)",
            "null_a": "same recipe and windows, seed 0, target token = one of the two plain tokens whose mean W0 probability at the "
                      "trigger (over the train windows) is closest to P_W0(`o`) in log probability (the run of 2026-10-19 used "
                      "'closest above / closest below' and fell back to token 0; see the analysis report); step 10 if held-out "
                      "P(target) >= 0.5 else step 30",
            "null_b": "the other of those two plain tokens",
            "lora_r8": f"LoRA rank {LORA_RANK} on {LORA_SITE} only (the paper's site), Adam lr {LORA_LR}, arm A loss and sampling, "
                       f"checkpoints {LORA_CKPTS}; SCORED AT STEP {LORA_SCORED} (the fine-tune's step count and the lift test's scored "
                       "LoRA checkpoint; by the old curves P(o) 0.9994 vs the fine-tune's 0.9977). Every checkpoint is reported, "
                       "and the checkpoint closest to ft_s0 in held-out log-odds of `o` is logged for reference",
            "random_dw": "Gaussian delta-W with ft_s0's per-matrix Frobenius norm, seed 0 (expected inert: reported, P5)",
            "pullback_adam": f"reference: Adam pull-back on all alive (U, V), lr {PB_LR}, {PB_STEPS} steps (the concentration test's "
                             "adam_all arm), its delta-W put into Delta: amendment 2's do_nothing arm, reproduced",
        },
        "plain_token": "ladder.plain_mask: decodes to a word-like string (letters, optional leading space)",
    },
    "measurements": {
        "routes": "local_faith.route_split with the released components as C0 and C1: cells L00, L01, L10, L11, U0, U1, T0, T1 "
                  "in mean held-out trigger log-odds of the arm's target token; shares d, r, i, e_gate, e_delta (÷ gain).",
        "dg": "per alive subcomponent, mean over the 73 held-out windows of g_state - g_W0 at the trigger (recruit.trigger_dg); "
              "dead subcomponents fixed at 0.",
        "S_on": f"subcomponents with dg > thr, thr in {DG_THRS}; scored at {DG_THR}. S_off: dg < -thr (reported).",
        "jaccard": "|A ∩ B| / |A ∪ B| of S_on at the same threshold; nan if both empty. J_{>=3}: restricted to layers 3+.",
        "recovery": "hybrid gates = W0's at every subcomponent except S_on, which take the state's gates at every position; "
                    "recovery = (lo_hybrid - L00) / (L01 - L00), components C0, Delta out. Random control: a set of the same "
                    "size per matrix drawn from the alive subcomponents outside S_on (seed 0).",
        "natural_o": f"on Pile {PILE_ROWS[0]} rows {PILE_ROWS[1]}..{PILE_ROWS[1] + PILE_ROWS[2]}, positions whose next token is "
                     "`o`; nat_dg = mean W0 gate at those positions - mean gate over all positions (recruit.ContrastAcc); "
                     f"S_nat = nat_dg > thr. Run only if >= {MIN_NAT_POSITIONS} such positions exist.",
        "repeat_noise": "W0's route cells computed twice; |T0 - T0'| sets G3's tolerance (10x noise ÷ gain, floor 1e-4).",
    },
    "blind_to": {
        "S_on / jaccard": "gates that move by less than thr at the trigger, and any re-gating away from the trigger position. "
                          "Positive control: the different-target nulls (P0) must come out well below the seed-to-seed floor (G0); "
                          "if they do not, the metric cannot separate recruitment from a generic trigger-position response.",
        "r": "everything away from the trigger; a bigger gain inflates nothing because shares are ÷ gain, but shares are "
             "unstable below 1 nat of gain (not scored then). Control: the G3 identities.",
        "recovery": "non-additivity: S_on's hybrid can recover more or less than the sum of its members. Control: the random "
                    "same-size set, which must stay near 0.",
        "natural_o": "contexts in which `o` is predicted but not emitted, and position-specific gating; the contrast is a mean "
                     "over heterogeneous contexts, so S_nat is biased towards subcomponents that fire in all of them.",
    },
    "gates": [
        {"id": "G0_seed_floor", "rule": f"J(ft_s0, ft_s1) at {DG_THR} >= 0.5",
         "if_fail": "the switched-on set is not a property of the update; P0, P1, P3c, R1, R2 are reported, not scored"},
        {"id": "G3_identities", "rule": "for every arm with gain >= 1 nat: |d| and |i| < tol, |e_delta - 1| < tol, tol = max(1e-4, "
                                        "10 x repeat noise ÷ gain); share_sum = 1 within 1e-6"},
        {"id": "G4_reproduces_amendment_2", "rule": f"pullback_adam: |n_on(0.3) - {AMEND2_REF['n_on_0.3']}| <= 20 and |L01 - L00 - "
                                                    f"{AMEND2_REF['gap']}| <= 0.3", "if_fail": "the cause is found before the analysis is written"},
        {"id": "G5_lora_locality", "rule": f"lora_r8: max |dg| over all matrices in layers <= 2 is <= max(1e-6, 10 x the gate repeat noise at W0) "
                                           f"(Gamma reads raw matrix inputs; {LORA_SITE}'s output first enters layer 3). Gate repeat noise = max |g - g'| "
                                           "over two W0 forwards, logged", "if_fail": "FAIL: the locality argument is wrong"},
        {"id": "G6_arms_learn", "rule": "ft_s0, ft_s1, lora_r8 (matched checkpoint): held-out P(`o`) >= 0.9; null_a, null_b: held-out "
                                        "P(target) >= 0.5 at the scored step", "if_fail": "the arm is reported, not scored"},
    ],
    "predictions": [
        {"id": "P0_target_specific", "scored": True,
         "claim": "The switched-on set depends on WHAT is learned, not just that the trigger position was perturbed.",
         "measure": f"J(ft_s0, null_a) and J(ft_s0, null_b) at {DG_THR}",
         "pass": "both <= J(ft_s0, ft_s1) - 0.3", "null": "either > J(ft_s0, ft_s1) - 0.15",
         "predicted": "pass",
         "if_null": "re-gating is a generic frozen-Gamma response to a perturbed trigger; the reveal reading of the concentration "
                    "test weakens and Gamma updating moves up"},
        {"id": "P1_real_update_recruits_the_maps_set", "scored": True,
         "claim": "The real W fine-tune switches on the same subcomponents as the pull-back map did.",
         "measure": f"J(ft_s0, pullback_adam) at {DG_THR}", "pass": ">= 0.5", "null": "< 0.3", "predicted": "pass"},
        {"id": "P2_regating_share_of_the_real_update", "scored": True,
         "claim": "Half of the real fine-tune's trigger effect is reproduced by re-gating the W0 components alone.",
         "measure": "r of ft_s0 (gain >= 1 nat required)", "pass": ">= 0.3", "null": "< 0.15", "predicted": "pass (about 0.5, like "
                                                                                                           "the pull-back's do-nothing)"},
        {"id": "P3_lora_regates_downstream", "scored": True,
         "claim": "A single-matrix LoRA at layer 2 recruits the same downstream set and gets a comparable re-gating share.",
         "measure": f"(b) r of lora_r8 at the matched checkpoint; (c) J_(>=3)(lora_r8, ft_s0) at {DG_THR}",
         "pass": "(b) >= 0.3 and (c) >= 0.4", "null": "(b) < 0.15 or (c) < 0.2", "predicted": "pass"},
        {"id": "P4_set_carries_the_regating", "scored": True,
         "claim": "ft_s0's switched-on set carries the re-gating effect (as amendment 2's B1 found for the pull-back arms).",
         "measure": f"recovery(ft_s0, {DG_THR}) and its random same-size control",
         "pass": "recovery >= 0.4 and control < 0.1", "null": "recovery < 0.2", "predicted": "pass"},
        {"id": "P5_random_dw_is_inert", "scored": True,
         "claim": "A Frobenius-matched random delta-W is a null without power: it neither learns nor re-gates.",
         "measure": "random_dw: gain and n_on(0.3)", "pass": "gain < 1 nat and n_on < 10", "null": "gain >= 1 or n_on >= 30",
         "predicted": "pass"},
        {"id": "R1_recruited_set_is_o_machinery_above_chance", "scored": True, "conditional_on": f"natural-o positions >= {MIN_NAT_POSITIONS}",
         "claim": "The switched-on set overlaps the subcomponents W0 gates on where `o` is the next token, above chance.",
         "measure": f"J(S_on ft_s0, S_nat) at {DG_THR} against the mean J of {N_RANDOM_SETS} random same-size sets",
         "pass": ">= 3 x the random mean (and S_nat non-empty)", "null": "< 1.5 x", "predicted": "pass"},
        {"id": "R2_o_machinery_is_target_specific", "scored": True, "conditional_on": f"natural-o positions >= {MIN_NAT_POSITIONS}",
         "claim": "The real fine-tune's set overlaps the natural `o` machinery more than the different-target nulls' sets do.",
         "measure": f"J(S_on ft_s0, S_nat) - max over nulls of J(S_on null, S_nat), at {DG_THR}",
         "pass": ">= 0.1", "null": "< 0.05", "predicted": "null (the natural-o contrast is dominated by context features, "
                                                           "not by the trigger's machinery)"},
    ],
    "not_scored_reported": (
        "all shares and cells for every arm; r for every LoRA checkpoint; S_on sizes at every threshold; S_off sizes; per-layer "
        "counts; full Jaccard matrix at every threshold; the top 15 of each S_on by dg; recovery for every arm; the null targets "
        "chosen and their W0 probabilities"
    ),
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; predictions are not edited after filing (add a dated report instead)")
    dump(PRED.name, PREDICTIONS)
    print(f"filed {PRED}")


def require_predictions() -> dict[str, Any]:
    if not PRED.is_file():
        sys.exit("file predictions first: run.py --predictions")
    return json.loads(PRED.read_text())


# ----------------------------------------------------------------------------- helpers


def lo(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def pick_null_targets(model: Any, train: Any, o_token: int, device: str) -> dict[str, Any]:
    """The two plain tokens whose mean W0 probability at the trigger (train windows) is closest to P_W0(o)."""
    from commute import ladder, vpd
    from commute.finetune import logits_of

    tok = vpd.tokenizer()
    probs: torch.Tensor | None = None
    n = 0
    with torch.no_grad():
        for s in range(0, len(train), BATCH):
            lg = logits_of(model.target_model, train.tokens[s : s + BATCH].to(device))[:, train.half].float()
            p = torch.softmax(lg, -1).double().sum(0).cpu()
            probs = p if probs is None else probs + p
            n += lg.shape[0]
    assert probs is not None
    probs = probs / n
    plain = ladder.plain_mask(tok.decode, probs.numel())
    plain[o_token] = False
    p_o = float(probs[o_token])
    # The two plain tokens closest to P_W0(o) in log probability. (The run of 2026-10-19 used "closest above / closest
    # below"; no plain token sat above P(o), and argmin over an all-inf tensor silently returned token 0, <|endoftext|>.
    # One rule, no sides, cannot fail that way.)
    dist = (probs.clamp_min(1e-12).log() - math.log(p_o)).abs()
    dist[~plain] = float("inf")
    a, b = dist.argsort()[:2].tolist()
    assert math.isfinite(float(dist[b])), "fewer than two plain tokens with non-zero probability at the trigger"
    out = {"p_o_W0_trigger": p_o,
           "null_a": {"token": a, "text": tok.decode([a]), "p_W0_trigger": float(probs[a])},
           "null_b": {"token": b, "text": tok.decode([b]), "p_W0_trigger": float(probs[b])}}
    log(f"[null] P_W0(o) at trigger {p_o:.4g}; null_a {out['null_a']}; null_b {out['null_b']}")
    return out


def measure_arm(ctx: dict[str, Any], name: str, dW: dict[str, torch.Tensor], token: int, base: dict[str, Any]) -> dict[str, Any]:
    """Do-nothing placement of dW: route split, dg, S_on/S_off at every threshold, recovery at DG_THR."""
    from commute import recruit as R
    from commute.faithfulness import model_state
    from commute.local_faith import masked_lo, route_cells, route_split

    model, W0, alive, held = ctx["model"], ctx["W0"], ctx["alive"], ctx["heldout"]
    W = {k: W0[k] + dW[k].to(W0[k].device, W0[k].dtype) for k in W0}
    with model_state(model, target=W):
        cur = route_cells(model, held.tokens, held.half, token, BATCH)
        sp = route_split(model, held.tokens, held.half, token, base, ctx["comps0"], BATCH)
    gain = sp["gain"]
    tol = max(1e-4, 10 * ctx["lo_repeat_noise"] / abs(gain)) if abs(gain) >= 1e-6 else float("inf")
    s = sp["shares"]
    g3 = abs(sp["share_sum"] - 1) < 1e-6 and abs(s["d"]) < tol and abs(s["i"]) < tol and abs(s["e_delta"] - 1) < tol
    sp["G3_ok"], sp["G3_tol"] = bool(g3), tol
    dg = R.trigger_dg(cur["gates"], base["gates"], held.half, alive)
    torch.save({"dg": dg, "token": token}, LARGE / f"{name}_dg.pt")
    rec: dict[str, Any] = {"token": token, "routes": sp, "rel_norm_max": max(float(dW[k].norm() / W0[k].cpu().norm()) for k in dW),
                           "n_on": {}, "n_off": {}, "by_layer_on": {}, "max_dg_layer_le2": max(
                               (float(v.abs().max()) for k, v in dg.items() if (R.layer_or_none(k) or 0) <= 2), default=0.0)}
    for thr in DG_THRS:
        on = R.switched(dg, thr)
        rec["n_on"][str(thr)] = R.size(on)
        rec["n_off"][str(thr)] = R.size(R.switched(dg, thr, on=False))
        rec["by_layer_on"][str(thr)] = R.by_layer(on)
    on = R.switched(dg, DG_THR)
    flat = sorted(((float(dg[k][c]), f"{k}:{c}") for k, v in on.items() for c in v.tolist()), reverse=True)
    rec["top_on"] = [{"key": key, "dg": d} for d, key in flat[:15]]
    L00, L01 = sp["cells"]["L00"], sp["cells"]["L01"]
    gap = L01 - L00
    rec["gap"] = gap
    if abs(gap) > 1e-6 and R.size(on):
        lo_h, _ = masked_lo(model, held.tokens, held.half, token, R.hybrid(base["gates"], cur["gates"], on), BATCH)
        lo_r, _ = masked_lo(model, held.tokens, held.half, token, R.hybrid(base["gates"], cur["gates"], R.random_like(on, alive, 0)), BATCH)
        rec["recovery"], rec["recovery_random_control"] = (lo_h - L00) / gap, (lo_r - L00) / gap
    else:
        rec["recovery"] = rec["recovery_random_control"] = float("nan")
    log(f"[{name}] gain {gain:.3f} r {s['r']:.3f} e_gate {s['e_gate']:.3f} d {s['d']:.2e} i {s['i']:.2e} e_delta {s['e_delta']:.4f} "
        f"G3 {'ok' if g3 else 'VIOLATED'} | n_on {rec['n_on']} n_off {rec['n_off']} | recovery {rec['recovery']:.3f} "
        f"(random {rec['recovery_random_control']:.3f}) | layers<=2 max|dg| {rec['max_dg_layer_le2']:.2e}")
    ctx["dg"][name] = dg
    return rec


def natural_o(ctx: dict[str, Any], smoke: bool) -> dict[str, Any]:
    from commute import recruit as R
    from commute import vpd
    from commute.data import pile_rows

    rows = pile_rows(PILE_ROWS[0], PILE_ROWS[1], 16 if smoke else PILE_ROWS[2], log)
    mask = R.precede_mask(rows, ctx["o_token"])
    n_pos = int(mask.sum())
    out: dict[str, Any] = {"n_rows": int(rows.shape[0]), "n_positions_before_o": n_pos, "min_required": 1 if smoke else MIN_NAT_POSITIONS}
    if n_pos < out["min_required"]:
        out["skipped"] = "too few natural-o positions"
        log(f"[natural_o] {n_pos} positions before `o` in {rows.shape[0]} rows: skipped")
        return out
    acc = R.ContrastAcc()
    for s in range(0, rows.shape[0], 8):
        ci = vpd.compute_ci(ctx["model"], rows[s : s + 8], chunk=8)
        acc.add(ci.g, mask[s : s + 8])
        del ci
    nat = acc.result(ctx["alive"])
    torch.save({"nat_dg": nat, "n_positions": n_pos}, LARGE / "natural_o_dg.pt")
    out["n_nat"] = {str(t): R.size(R.switched(nat, t)) for t in DG_THRS}
    out["by_layer_nat"] = R.by_layer(R.switched(nat, DG_THR))
    flat = sorted(((float(v[c]), f"{k}:{c}") for k, v in nat.items() for c in (v > DG_THR).nonzero().flatten().tolist()), reverse=True)
    out["top_nat"] = [{"key": key, "contrast": d} for d, key in flat[:15]]
    ctx["nat"] = nat
    log(f"[natural_o] {n_pos} positions before `o`; |S_nat| {out['n_nat']}")
    return out


# ----------------------------------------------------------------------------- the run


def stage_all(args: argparse.Namespace) -> None:
    from commute import component_training, vpd
    from commute import recruit as R
    from commute.finetune import (
        ArmAConfig,
        LoraConfig,
        finetune_arm_a,
        finetune_lora_arm_a,
        random_direction,
    )
    from commute.local_faith import route_cells

    pred = require_predictions()
    dev, smoke = args.device, args.smoke
    t_start = time.time()
    model = vpd.load_component_model(dev)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    assert all(v.dtype == torch.bool and v.numel() == model.module_to_c[k] for k, v in alive.items()), "alive.pt must hold bool masks"
    alive = {k: v.cpu() for k, v in alive.items()}
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    train, held = win["train"], win["heldout"]
    ctx: dict[str, Any] = {"model": model, "W0": W0, "comps0": vpd.component_weights(model), "alive": alive, "heldout": held,
                           "o_token": o_token, "dg": {}}
    out: dict[str, Any] = {"smoke": smoke, "predictions_sha_about": pred["about"][:60], "n_heldout": len(held), "n_train": len(train),
                           "arms": {}}

    base = route_cells(model, held.tokens, held.half, o_token, BATCH)
    base2 = route_cells(model, held.tokens, held.half, o_token, BATCH)
    ctx["lo_repeat_noise"] = abs(base["T"] - base2["T"])
    ctx["gate_repeat_noise"] = max(float((a[k] - b[k]).abs().max()) for a, b in zip(base["gates"], base2["gates"], strict=True) for k in a)
    out["lo_repeat_noise"], out["gate_repeat_noise"] = ctx["lo_repeat_noise"], ctx["gate_repeat_noise"]
    out["base_cells"] = {k: base[k] for k in ("T", "U", "L", "p_L")}
    log(f"[base] T0 {base['T']:.3f} U0 {base['U']:.3f} L00 {base['L']:.3f} p_L {base['p_L']:.4f} repeat noise lo {ctx['lo_repeat_noise']:.2e} gate {ctx['gate_repeat_noise']:.2e}")

    nulls = pick_null_targets(model, train, o_token, dev)
    out["null_targets"] = nulls
    ft_step = 2 if smoke else FT_STEP

    def a_cfg(token: int, steps: int, ckpts: tuple[int, ...], seed: int = 0) -> ArmAConfig:
        return ArmAConfig(o_token=token, lr=FT_LR, steps=steps, checkpoints=ckpts, seed=seed)

    # W fine-tunes to `o`, two seeds
    ft0_lo, ft0_dW = float("nan"), {}
    for seed in FT_SEEDS:
        name = f"ft_s{seed}"
        res = finetune_arm_a(model, train, held, a_cfg(o_token, ft_step, (ft_step,), seed), dev, log)
        rec = measure_arm(ctx, name, res.deltas[ft_step], o_token, base)
        rec.update({"step": ft_step, "evals": res.evals, "learned": res.evals[ft_step]["p_o"] >= 0.9})
        out["arms"][name] = rec
        if seed == 0:
            ft0_lo = lo(res.evals[ft_step]["p_o"])
            ft0_dW = res.deltas[ft_step]

    # different-target nulls
    for name in ("null_a", "null_b"):
        t = nulls[name]["token"]
        steps = (1, 2) if smoke else NULL_STEPS
        res = finetune_arm_a(model, train, held, a_cfg(t, steps[-1], steps), dev, log)
        step = steps[0] if res.evals[steps[0]]["p_o"] >= 0.5 else steps[-1]
        base_t = route_cells(model, held.tokens, held.half, t, BATCH)
        rec = measure_arm(ctx, name, res.deltas[step], t, base_t)
        rec.update({"step": step, "evals": res.evals, "learned": res.evals[step]["p_o"] >= 0.5})
        out["arms"][name] = rec

    # LoRA on the paper's site, every checkpoint measured, scored at the matched one
    l_cfg = LoraConfig(modules=(LORA_SITE,), rank=LORA_RANK, lr=LORA_LR, steps=2 if smoke else LORA_STEPS,
                       checkpoints=(1, 2) if smoke else LORA_CKPTS)
    res = finetune_lora_arm_a(model, train, held, a_cfg(o_token, l_cfg.steps, ()), l_cfg, dev, log)
    matched = 2 if smoke else LORA_SCORED
    closest = min(res.evals, key=lambda s: abs(lo(res.evals[s]["p_o"]) - ft0_lo))
    out["lora_checkpoints"] = {}
    out["lora_closest_in_lo_to_ft_s0"] = closest
    for step in sorted(res.evals):
        name = "lora_r8" if step == matched else f"lora_r8_step{step}"
        rec = measure_arm(ctx, name, res.deltas[step], o_token, base)
        rec.update({"step": step, "evals": {str(step): res.evals[step]}, "learned": res.evals[step]["p_o"] >= 0.9})
        out["lora_checkpoints"][str(step)] = {"r": rec["routes"]["shares"]["r"], "gain": rec["routes"]["gain"], "n_on": rec["n_on"],
                                              "p_o": res.evals[step]["p_o"]}
        if step == matched:
            rec["ft_s0_lo"], rec["closest_checkpoint_in_lo"] = ft0_lo, closest
            out["arms"][name] = rec
        else:
            ctx["dg"].pop(name, None)
    log(f"[lora] scored checkpoint {matched}; closest in lo to ft_s0 ({ft0_lo:.2f}) is step {closest}; r by step " +
        " ".join(f"{s}:{v['r']:.3f}" for s, v in out["lora_checkpoints"].items()))

    # random delta-W with ft_s0's norms
    out["arms"]["random_dw"] = measure_arm(ctx, "random_dw", random_direction(ft0_dW, 0), o_token, base)

    # reference: the pull-back map's delta-W in Delta (amendment 2's do_nothing)
    pb_steps = 2 if smoke else PB_STEPS
    uv = component_training.alive_uv(model, alive)
    curve, states = component_training.train(model, uv, train.tokens, held, a_cfg(o_token, pb_steps, ()),
                                   component_training.TrainConfig(lr=PB_LR, steps=pb_steps, save_steps=(pb_steps,)), dev, W0, log=log)
    st = states[pb_steps]
    dW_pb = {k: st[k][0].cpu().float() for k in W0}
    assert all(dW_pb[k].shape == W0[k].shape for k in W0), "component_training.train state layout changed"
    rec = measure_arm(ctx, "pullback_adam", dW_pb, o_token, base)
    rec.update({"step": pb_steps, "curve_tail": curve[-2:], "learned": curve[-1]["heldout_p_o"] >= 0.9})
    out["arms"]["pullback_adam"] = rec

    # Jaccard matrices
    names = list(ctx["dg"])
    out["jaccard"] = {}
    for thr in DG_THRS:
        sets = {n: R.switched(ctx["dg"][n], thr) for n in names}
        out["jaccard"][str(thr)] = {f"{a}|{b}": R.jaccard(sets[a], sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        out["jaccard"][f"{thr}_layers_ge3"] = {f"{a}|{b}": R.jaccard(R.restrict(sets[a], 3), R.restrict(sets[b], 3))
                                               for i, a in enumerate(names) for b in names[i + 1:]}
    log("[jaccard 0.3] " + json.dumps({k: round(v, 3) for k, v in out["jaccard"][str(DG_THR)].items()}))

    # natural `o` machinery at W0
    out["natural_o"] = natural_o(ctx, smoke)
    if "nat" in ctx:
        nat_set = R.switched(ctx["nat"], DG_THR)
        jn = {n: R.jaccard(R.switched(ctx["dg"][n], DG_THR), nat_set) for n in names}
        on0 = R.switched(ctx["dg"]["ft_s0"], DG_THR)
        rnd = [R.jaccard(R.random_like(on0, alive, s), nat_set) for s in range(N_RANDOM_SETS)]
        rnd = [x for x in rnd if not math.isnan(x)]
        out["natural_o"].update({"jaccard_with_S_nat": jn, "random_mean": sum(rnd) / len(rnd) if rnd else float("nan"),
                                 "random_max": max(rnd) if rnd else float("nan")})
        log("[natural_o] J with S_nat " + json.dumps({k: round(v, 3) for k, v in jn.items()}) + f" random mean {out['natural_o']['random_mean']:.4f}")

    out["score"] = score(out)
    out["seconds"] = time.time() - t_start
    p = dump(f"recruit{'_smoke' if smoke else ''}.json", out)
    log(f"wrote {p} ({out['seconds']:.0f} s); scores " + json.dumps(out["score"], default=str))


def score(out: dict[str, Any]) -> dict[str, Any]:
    A, J = out["arms"], out["jaccard"][str(DG_THR)]
    J3 = out["jaccard"][f"{DG_THR}_layers_ge3"]

    def j(a: str, b: str, table: dict[str, float]) -> float:
        return table.get(f"{a}|{b}", table.get(f"{b}|{a}", float("nan")))

    def r_of(a: str) -> float:
        return A[a]["routes"]["shares"]["r"]

    def gain(a: str) -> float:
        return A[a]["routes"]["gain"]

    floor = j("ft_s0", "ft_s1", J)
    pb = A["pullback_adam"]
    s: dict[str, Any] = {
        "G0_seed_floor": {"J": floor, "ok": floor >= 0.5},
        "G3_identities": {a: A[a]["routes"]["G3_ok"] for a in A if abs(gain(a)) >= 1.0},
        "G4_reproduces_amendment_2": {"n_on_0.3": pb["n_on"]["0.3"], "gap": pb["gap"],
                                      "ok": abs(pb["n_on"]["0.3"] - AMEND2_REF["n_on_0.3"]) <= 20 and abs(pb["gap"] - AMEND2_REF["gap"]) <= 0.3},
        "G5_lora_locality": {"max_dg_layer_le2": A["lora_r8"]["max_dg_layer_le2"], "tol": max(1e-6, 10 * out["gate_repeat_noise"]),
                             "ok": A["lora_r8"]["max_dg_layer_le2"] <= max(1e-6, 10 * out["gate_repeat_noise"])},
        "G6_arms_learn": {a: A[a]["learned"] for a in ("ft_s0", "ft_s1", "null_a", "null_b", "lora_r8")},
    }
    jn_a, jn_b = j("ft_s0", "null_a", J), j("ft_s0", "null_b", J)
    s["P0_target_specific"] = {"J_null_a": jn_a, "J_null_b": jn_b, "floor": floor,
                               "pass": jn_a <= floor - 0.3 and jn_b <= floor - 0.3, "null": jn_a > floor - 0.15 or jn_b > floor - 0.15}
    jp = j("ft_s0", "pullback_adam", J)
    s["P1_real_update_recruits_the_maps_set"] = {"J": jp, "pass": jp >= 0.5, "null": jp < 0.3}
    s["P2_regating_share_of_the_real_update"] = {"r": r_of("ft_s0"), "gain": gain("ft_s0"), "scorable": gain("ft_s0") >= 1.0,
                                                 "pass": r_of("ft_s0") >= 0.3, "null": r_of("ft_s0") < 0.15}
    jl = j("lora_r8", "ft_s0", J3)
    s["P3_lora_regates_downstream"] = {"r": r_of("lora_r8"), "J_ge3": jl, "pass": r_of("lora_r8") >= 0.3 and jl >= 0.4,
                                       "null": r_of("lora_r8") < 0.15 or jl < 0.2}
    rc, rr = A["ft_s0"]["recovery"], A["ft_s0"]["recovery_random_control"]
    s["P4_set_carries_the_regating"] = {"recovery": rc, "random": rr, "pass": rc >= 0.4 and rr < 0.1, "null": rc < 0.2}
    rd = A["random_dw"]
    s["P5_random_dw_is_inert"] = {"gain": gain("random_dw"), "n_on_0.3": rd["n_on"]["0.3"],
                                  "pass": gain("random_dw") < 1.0 and rd["n_on"]["0.3"] < 10, "null": gain("random_dw") >= 1.0 or rd["n_on"]["0.3"] >= 30}
    nat = out.get("natural_o", {})
    if "jaccard_with_S_nat" in nat:
        jf, rm = nat["jaccard_with_S_nat"]["ft_s0"], nat["random_mean"]
        jnull = max(nat["jaccard_with_S_nat"]["null_a"], nat["jaccard_with_S_nat"]["null_b"])
        s["R1_recruited_set_is_o_machinery_above_chance"] = {"J": jf, "random_mean": rm, "pass": (not math.isnan(jf)) and jf >= 3 * rm and jf > 0,
                                                              "null": math.isnan(jf) or jf < 1.5 * rm}
        s["R2_o_machinery_is_target_specific"] = {"J_ft": jf, "J_null_max": jnull, "pass": jf - jnull >= 0.1, "null": jf - jnull < 0.05}
    else:
        s["R1_recruited_set_is_o_machinery_above_chance"] = s["R2_o_machinery_is_target_specific"] = {"skipped": nat.get("skipped", "not run")}
    if not s["G0_seed_floor"]["ok"]:
        for k in ("P0_target_specific", "P1_real_update_recruits_the_maps_set", "P3_lora_regates_downstream",
                  "R1_recruited_set_is_o_machinery_above_chance", "R2_o_machinery_is_target_specific"):
            s[k]["scored"] = False
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    if args.stage == "all":
        stage_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
