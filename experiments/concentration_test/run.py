"""Concentration test: is the pull-back's diffuse update an optimiser artefact, and can the emoticon behaviour be
learned inside the decomposition by moving only a few subcomponents (or only the paper's emoticon subcomponents)?

Plan and design check: reports/2026-10-15_plan_concentration-test.md (confirmed by the user 2026-10-15).
Measurement scope follows reports/2026-10-15_review_faithfulness-scope.md: faithfulness on Pile cannot see the
trigger, so trigger-local metrics on the held-out windows are added, with do-nothing as their positive control, and a
three-seed noise floor for the Pile metrics on W0.

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  all             GPU: train the arms, select, measure -> results/concentration.json
  all --smoke     GPU, minutes: fewer lrs and steps, no Pile faithfulness; checks every path runs
  --predictions-amend   file results/predictions_amend1.json (refuses to overwrite)
  --predictions-amend2  file results/predictions_amend2.json (refuses to overwrite)
  gating          GPU: which subcomponents switch on at the trigger, and do they already push towards `o` at W0
                  -> results/amend2.json (reports/2026-10-15_prediction_gating-amendment.md)
  amend           GPU: retrain each arm's selected run; trigger-local CI-masked P(o) with Gamma's gates computed on
                  W0's activations -> results/amend1.json (reports/2026-10-15_prediction_concentration-amendment.md)

Usage (repo root):  uv run python experiments/concentration_test/run.py all
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute.attribution import (
    ablation_effect,
    heldout_log_odds,
    read_acts,
    revert_one,
    reverted_weights,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

STEPS = 30
EVAL_STEPS = (1, 2, 3, 5, 10, 20, 30)
ADAM_LRS_MASKED = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
SGD_LRS = (0.3, 1.0, 3.0, 10.0, 30.0)
ARMS: dict[str, dict[str, Any]] = {
    "adam_all": {"opt": "adam", "lrs": (1e-5,), "select": "all"},
    "sgd_all": {"opt": "sgd", "lrs": SGD_LRS, "select": "all"},
    "top1": {"opt": "adam", "lrs": ADAM_LRS_MASKED, "select": 1},
    "top10": {"opt": "adam", "lrs": ADAM_LRS_MASKED, "select": 10},
    "top100": {"opt": "adam", "lrs": ADAM_LRS_MASKED, "select": 100},
    "siblings": {"opt": "adam", "lrs": ADAM_LRS_MASKED, "select": "siblings"},
}
KL_OTHER_MAX = 0.110  # amendment 1's threshold (2 x the W fine-tune's held-out KL elsewhere at step 10)
SIB_KEY = "h.2.mlp.down_proj"
SIBLINGS = (2359, 1672, 2623, 3290, 3327, 3382)
BANK_ROWS = (512, 32)
K_POS = 64
K_ID = 20
PGD_SEEDS = (0, 1, 2)

PREDICTIONS: dict[str, Any] = {
    "about": (
        "concentration_test: the round-2 pull-back (Adam, all alive U, V, Gamma frozen, Delta fixed) spreads the emoticon "
        "update over thousands of subcomponents (identity test: top 10 by revert-one remove 1.6% of the log-odds gain; "
        "2359 unchanged). Is that the optimiser, and can the behaviour be learned inside the decomposition by moving a "
        "few subcomponents, or only 2359 and its siblings? Filed before any arm was trained."
    ),
    "setup": {
        "arms": "pull-back with arm A's loss, windows, batch 32, clip 1, seed 0, 30 steps. adam_all: Adam lr 1e-5 on all "
                "alive (U, V) (round 2). sgd_all: plain SGD on all alive, lr in {0.3, 1, 3, 10, 30}. topK (K = 1, 10, "
                "100): Adam on the K alive subcomponents with the largest |dL/ds_c| (below), lr in {1e-5 .. 1e-3}. "
                "siblings: Adam on h.2.mlp.down_proj 2359, 1672, 2623, 3290, 3327, 3382 only, same lrs. Unselected "
                "columns get zero gradient (checked: their offsets stay exactly 0)",
        "top_k_selection": "gauge-free first-order importance at W0: s_c scales subcomponent c (B_c -> (1 + s_c) B_c); "
                           "dL/ds_c = u_c^T G_W v_c with G_W the gradient of arm A's loss on all 365 training windows",
        "lr_rule": "per arm, among lrs with held-out P(o) >= 0.9 at step 30 and held-out KL elsewhere <= 0.110, the lowest "
                   "KL elsewhere; if none, the highest P(o), and the arm is marked 'did not learn'",
        "pile_faithfulness": "round 1's harness (Pile val rows 0..255; stochastic, PGD, CI-masked, unmasked), target W0 + "
                             "dW, alive columns at (U2, V2). Ratios to the decomposition on W0, seed 0. Noise floor: W0 "
                             "with adversary seeds 0, 1, 2, filed with the results",
        "trigger_local": "held-out windows (73 x 41 tokens), trigger at index 20. Component model with the state's "
                         "weights, masks = frozen Gamma's g (CI-masked, Delta excluded) and masks = 1 (unmasked, Delta "
                         "excluded): mean P(o) at the trigger, KL(target || model) at the trigger and at the other 40. "
                         "Blind to: positions outside these windows. Positive control: do-nothing (target W0 + dW of "
                         "adam_all, components at W0) must give CI-masked P(o) < 0.5 x target P(o), else the "
                         "trigger-local criterion is unscorable",
        "revert_one": "as identity_test: drop in held-out mean log-odds of o when one subcomponent alone is reverted. "
                      "Blind to: redundancy (joint effects beyond the greedy top-k); evaluated at the trigger only",
        "gamma_vs_update": "reported: mean g at the trigger per alive subcomponent at W0; share of the summed positive "
                           "revert-one drop carried by subcomponents with mean g > 0.5 there; Spearman of drop vs g",
        "identity": "as identity_test (read r_self >= 0.9, rank 1, write_cos >= floor - 0.1) for the moved subcomponents "
                    "(masked arms) or the 20 largest drops (all-arms). Positive control, a gate: on W0, replacing one "
                    "subcomponent's v with another alive v of the same matrix, and its u with a random direction of the "
                    "same norm, must each fail keeps_identity; otherwise identity numbers are reported as uninformative. "
                    "Also reported: rotating v by 20 and 45 degrees",
    },
    "predictions": [
        {
            "id": "1_sgd_concentrated",
            "scored": True,
            "claim": "Adam is why the pull-back's update is diffuse: with plain SGD it concentrates.",
            "measure": "sgd_all selected run at step 30: fraction of the log-odds gain removed by reverting the 10 largest "
                       "revert-one drops together",
            "pass": ">= 0.5",
            "null": "< 0.2",
            "scorable_if": "sgd_all learns (lr rule finds an lr)",
            "prediction": "fail",
        },
        {
            "id": "2_top10_inside",
            "scored": True,
            "claim": "Moving only 10 subcomponents learns the behaviour faithfully.",
            "measure": "top10 selected run at step 30",
            "pass": "learns by the lr rule AND Pile stochastic, PGD and CI-masked ratios <= 1.5 AND trigger-local CI-masked "
                    "P(o) >= 0.8 x target P(o)",
            "prediction": "pass",
        },
        {
            "id": "3_siblings_inside",
            "scored": True,
            "claim": "The paper's emoticon subcomponents (2359 and five siblings) suffice: the label is usable for learning.",
            "measure": "siblings selected run at step 30",
            "pass": "as prediction 2",
            "prediction": "pass",
        },
    ],
    "reported_not_scored": [
        "top1 and top100; adam_all and sgd_all on every measure (first trigger-local numbers for the round-2 pull-back)",
        "rank of 2359 and its siblings in |dL/ds_c|",
        "identity of moved subcomponents, with the positive controls",
        "gamma vs update for adam_all and sgd_all",
    ],
    "filed": None,
}


PRED_AMEND = RESULTS / "predictions_amend1.json"

PREDICTIONS_AMEND: dict[str, Any] = {
    "about": (
        "Amendment 1 to concentration_test (reports/2026-10-15_prediction_concentration-amendment.md). The original "
        "predictions.json is unchanged. The trigger-local control failed its filed rule: do-nothing's CI-masked P(o) was "
        "0.535 (needed < 0.498) because the frozen Gamma reads the fine-tuned activations and its gates alone move the "
        "component model towards `o`. This amendment fixes the gates to W0: Gamma's g is computed from the W0 model's "
        "activations, so only the components can carry the update. Filed before the amended measure was run."
    ),
    "setup": {
        "states": "each arm's selected run retrained with the same lr, seed and recipe; held-out P(o) at step 30 must "
                  "match concentration.json within 0.01 (logged)",
        "w0_gated": "g = frozen Gamma on the W0 model's activations over the held-out windows; component model with the "
                    "state's components, masks = g, Delta excluded; target = the state's model. Mean P(o) at the "
                    "trigger, KL(target || model) at the trigger and at the other 40 positions",
        "state_gated": "as the original trigger_local (g from the state's activations), recomputed for comparison",
        "control": "do-nothing (target W0 + dW of adam_all, components at W0) under W0 gates must give P(o) < 0.5 x "
                   "target (it is W0's own CI-masked model, 0.0045 in concentration.json), else unscorable",
        "reported": "W0-gated numbers for every arm; mean g at the trigger (W0) of each moved subcomponent in the masked "
                    "arms; W0-gated KL at the trigger of W0's own decomposition is 0.519 (concentration.json), the "
                    "baseline imperfection at the trigger",
    },
    "predictions": [
        {"id": "A1_siblings_w0_gated", "scored": True,
         "claim": "The six labelled subcomponents carry the learned behaviour under the original gating.",
         "measure": "siblings arm, W0-gated CI-masked P(o) at the trigger", "pass": ">= 0.8 x target P(o)",
         "prediction": "pass (the windows were chosen where 2359 has g > 0.5 at W0)"},
        {"id": "A2_top10_w0_gated", "scored": True, "claim": "As A1 for the top-10 arm.",
         "measure": "top10 arm, same", "pass": ">= 0.8 x target P(o)", "prediction": "pass"},
        {"id": "A3_adam_all_w0_gated", "scored": True,
         "claim": "The diffuse Adam update is not carried by the components Gamma gates on at W0.",
         "measure": "adam_all arm, same", "pass": "< 0.5 x target P(o)", "prediction": "pass"},
    ],
    "filed": None,
}


PRED_AMEND2 = RESULTS / "predictions_amend2.json"
DG_THR = 0.3  # scored threshold on the change of mean g at the trigger; 0.1 and 0.5 reported
GATING_ARMS = ("do_nothing", "adam_all", "top1", "top10", "siblings")

PREDICTIONS_AMEND2: dict[str, Any] = {
    "about": (
        "Amendment 2 to concentration_test (reports/2026-10-15_prediction_gating-amendment.md). Amendment 1 found that "
        "the arms explain the behaviour at the trigger only with Gamma re-gating on the fine-tuned activations (CI-masked "
        "P(o) 0.98 state-gated, <= 0.37 with W0's gates), and do-nothing reaches 0.535 state-gated with every "
        "component at W0. Question: which subcomponents switch on, and do they already push towards `o` at W0 (the "
        "'reveal' reading)? Filed before the measure was run."
    ),
    "setup": {
        "states": "arms retrained exactly as amendment 1 (reproduced to 0.0 there); do_nothing = target W0 + dW of "
                  "adam_all with components at W0",
        "dg": "per subcomponent, mean over the 73 held-out windows of g at the trigger under the state's gates minus "
              "under W0's gates. S_on = alive subcomponents with dg > 0.3 (0.1 and 0.5 reported)",
        "lo_masked": "mean log-odds of `o` at the trigger of the component model (the state's components, Delta "
                     "excluded) under given masks",
        "recovery": "hybrid masks = W0's g everywhere except S_on, which take the state's g (all positions). recovery = "
                    "(lo(hybrid) - lo(W0 gates)) / (lo(state gates) - lo(W0 gates)). Negative control: S_on replaced by "
                    "the same number per matrix of random alive subcomponents with |dg| < 0.05 (seed 0); positive "
                    "control: all subcomponents take the state's g (recovery 1 by construction)",
        "switch_on_effect": "at W0 (W0 target, W0 components, W0 gates), set one subcomponent's mask to 1 at every "
                            "position: change in lo. For the union of S_on over arms (up to 300 by dg) and a null of 300 "
                            "random alive subcomponents with |dg| < 0.05 in every arm (seed 0)",
        "overlap": "Jaccard of S_on between arms",
    },
    "predictions": [
        {"id": "B1_switched_on_set_explains", "scored": True,
         "claim": "The subcomponents that switch on at the trigger account for the re-gating effect.",
         "measure": "recovery with S_on for do_nothing, top10 and siblings",
         "pass": ">= 0.5 in each of the three, AND the negative control < 0.2 in each",
         "prediction": "pass"},
        {"id": "B2_same_machinery", "scored": True,
         "claim": "The update reveals one existing mechanism: different updates switch on the same subcomponents.",
         "measure": "Jaccard of S_on between do_nothing and siblings, and between do_nothing and top10",
         "pass": ">= 0.3 for both", "prediction": "pass"},
        {"id": "B3_already_writes_o", "scored": True,
         "claim": "The switched-on subcomponents already push towards `o` at W0.",
         "measure": "switch-on effect: mean over the S_on union against the null",
         "pass": "mean(S_on) > mean(null) + 3 standard errors of the difference AND mean(S_on) > 0",
         "prediction": "pass"},
    ],
    "filed": None,
}


def file_predictions_amend2() -> None:
    if PRED_AMEND2.is_file():
        sys.exit(f"{PRED_AMEND2} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS_AMEND2["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump(PRED_AMEND2.name, PREDICTIONS_AMEND2)
    log(f"filed {PRED_AMEND2}")


def file_predictions_amend() -> None:
    if PRED_AMEND.is_file():
        sys.exit(f"{PRED_AMEND} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS_AMEND["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump(PRED_AMEND.name, PREDICTIONS_AMEND)
    log(f"filed {PRED_AMEND}")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    return p


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump("predictions.json", PREDICTIONS)
    log(f"filed {PRED}")


# ----------------------------------------------------------------------------- selection and training


def scale_gradients(model: Any, UV: dict[str, Any], win: Any, cfg: Any) -> dict[str, torch.Tensor]:
    """dL/ds_c = u_c^T G_W v_c per alive subcomponent, G_W the gradient of arm A's loss at W0 over all training windows."""
    from commute import vpd
    from commute.finetune import arm_a_loss
    from commute.identity import scale_grads

    net = model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    lins = {k: vpd.target_linear(model, k) for k in UV}
    for lin in lins.values():
        lin.weight.requires_grad_(True)
        lin.weight.grad = None
    dev = next(iter(UV.values()))[0].device
    try:
        n = len(win)
        for s in range(0, n, 73):
            toks = win.tokens[s:s + 73].to(dev)
            loss, _ = arm_a_loss(net, ref, toks, cfg, win.half)
            (loss * toks.shape[0] / n).backward()
        return {k: scale_grads(lins[k].weight.grad, UV[k][0], UV[k][1]).cpu() for k in UV}
    finally:
        for lin in lins.values():
            lin.weight.grad = None
            lin.weight.requires_grad_(False)


def train(model: Any, UV: dict[str, Any], win: dict[str, Any], o_token: int, opt_name: str, lr: float,
          masks: dict[str, torch.Tensor] | None, steps: int, W0: dict[str, torch.Tensor]) -> tuple[list[dict[str, float]], dict[int, Any]]:
    """Pull-back on the alive (U, V) offsets, gradients of unselected columns zeroed. States (U2, V2) at 10 and `steps`."""
    from commute import component_training, vpd
    from commute.finetune import ArmAConfig, arm_a_loss, eval_arm_a

    cfg = ArmAConfig(o_token=o_token)
    dev = next(iter(UV.values()))[0].device
    net = model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    off = component_training.Offsets(UV)
    if masks is not None:
        for k in UV:
            m = masks[k].to(dev).float()[None, :]
            for p in off.d(k):
                p.register_hook(lambda g, m=m: g * m)
    params = list(off.parameters())
    opt = (torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0) if opt_name == "adam"
           else torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0))
    gen = torch.Generator().manual_seed(cfg.seed)
    handles = component_training.hooks({k: vpd.target_linear(model, k) for k in UV}, off)
    curve, states = [], {}
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, len(win["train"]), (cfg.batch,), generator=gen)
            loss, _ = arm_a_loss(net, ref, win["train"].tokens[idx].to(dev), cfg, win["train"].half)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if step in EVAL_STEPS or step == steps:
                ev = eval_arm_a(net, ref, win["heldout"], o_token, dev)
                rel = max(float(off.delta_w(k).norm() / W0[k].norm()) for k in UV)
                curve.append({"step": step, "heldout_p_o": ev["p_o"], "heldout_kl_other": ev["kl_other"],
                              "rel_norm_max": rel, "loss": float(loss)})
            if step in (10, steps):
                states[step] = {k: off.components(k) for k in UV}
    finally:
        for h in handles:
            h.remove()
    if masks is not None:  # logged check: unselected columns never moved
        leak = max(float((torch.cat([p.detach() for p in off.d(k)]) * (~masks[k].to(dev))[None, :].float()).abs().max())
                   for k in UV)
        assert leak == 0.0, f"unselected columns moved: {leak}"
    return curve, states


def select_lr(runs: dict[str, Any]) -> tuple[str, bool]:
    last = {lr: r["curve"][-1] for lr, r in runs.items()}
    ok = [lr for lr, c in last.items() if c["heldout_p_o"] >= 0.9 and c["heldout_kl_other"] <= KL_OTHER_MAX]
    if ok:
        return min(ok, key=lambda lr: last[lr]["heldout_kl_other"]), True
    return max(last, key=lambda lr: last[lr]["heldout_p_o"]), False


# ----------------------------------------------------------------------------- measures


def components_with(model: Any, alive: dict[str, torch.Tensor], st: dict[str, Any]) -> dict[str, Any]:
    """Full component weights (param-decomp layout) with alive columns set to the state: U[alive] = U2^T, V[:, alive] = V2."""
    from commute import vpd

    comps = vpd.component_weights(model)
    for k, (U, V) in comps.items():
        U, V = U.clone(), V.clone()
        U[alive[k]] = st[k][0].T.to(U.dtype)
        V[:, alive[k]] = st[k][1].to(V.dtype)
        comps[k] = (U, V)
    return comps


@torch.no_grad()
def trigger_local(model: Any, toks: torch.Tensor, half: int, o_token: int) -> dict[str, Any]:
    """Target, CI-masked and unmasked component model (Delta excluded) on the windows: P(o) at the trigger, KL at the
    trigger and elsewhere, and the mean g of every subcomponent at the trigger."""
    from param_decomp.models.components import make_mask_infos

    from commute.vpd import kl_per_position

    out = model(toks, cache_type="input")
    g = model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
    tgt = out.output.float()
    ci = model(toks, mask_infos=make_mask_infos(g)).float()
    ones = model(toks, mask_infos=make_mask_infos({k: torch.ones_like(v) for k, v in g.items()})).float()
    keep = torch.ones(toks.shape[1], dtype=torch.bool, device=toks.device)
    keep[half] = False
    res: dict[str, Any] = {"p_o_target": float(torch.softmax(tgt[:, half], -1)[:, o_token].mean())}
    for name, lg in (("ci", ci), ("unmasked", ones)):
        kl = kl_per_position(tgt, lg)
        res[f"p_o_{name}"] = float(torch.softmax(lg[:, half], -1)[:, o_token].mean())
        res[f"kl_trigger_{name}"] = float(kl[:, half].mean())
        res[f"kl_other_{name}"] = float(kl[:, keep].mean())
    res["g_trigger"] = {k: v[:, half].float().mean(0).cpu() for k, v in g.items()}
    return res


@torch.no_grad()
def gates(model: Any, toks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Frozen Gamma's g (lower leaky) on the model's current target activations."""
    out = model(toks, cache_type="input")
    return model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky


@torch.no_grad()
def masked_trigger(model: Any, toks: torch.Tensor, half: int, o_token: int, g: dict[str, torch.Tensor]) -> dict[str, float]:
    """Component model (current components, masks = the given g, Delta excluded) against the current target model."""
    from param_decomp.models.components import make_mask_infos

    from commute.vpd import kl_per_position

    tgt = model(toks).float()
    lg = model(toks, mask_infos=make_mask_infos(g)).float()
    kl = kl_per_position(tgt, lg)
    keep = torch.ones(toks.shape[1], dtype=torch.bool, device=toks.device)
    keep[half] = False
    return {"p_o_target": float(torch.softmax(tgt[:, half], -1)[:, o_token].mean()),
            "p_o_ci": float(torch.softmax(lg[:, half], -1)[:, o_token].mean()),
            "kl_trigger_ci": float(kl[:, half].mean()), "kl_other_ci": float(kl[:, keep].mean())}


def stage_amend(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.faithfulness import model_state

    if not PRED_AMEND.is_file():
        sys.exit("file the amendment first: run.py --predictions-amend")
    dev = args.device
    prev = json.loads((RESULTS / "concentration.json").read_text())
    model = vpd.load_component_model(dev)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    keys = vpd.module_keys(model)
    UV = {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
              model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in keys}
    alive_idx = {k: alive[k].nonzero().flatten() for k in keys}
    wtoks, half = win["heldout"].tokens.to(dev), win["heldout"].half
    g0 = gates(model, wtoks)
    out: dict[str, Any] = {"W0": masked_trigger(model, wtoks, half, o_token, g0), "arms": {}}
    log(f"W0 under W0 gates: {out['W0']}")
    # masks of the masked arms from the saved states: the columns that moved
    for arm, spec in ARMS.items():
        lr = float(prev["arms"][arm]["selected_lr"])
        saved = torch.load(LARGE / f"{arm}.pt", weights_only=False) if spec["select"] != "all" else None
        masks = None
        if saved is not None:
            masks = {k: ((saved["uv"][k][0] - UV[k][0].cpu()).abs().sum(0) + (saved["uv"][k][1] - UV[k][1].cpu()).abs().sum(0)) > 0
                     for k in keys}
        curve, sts = train(model, UV, win, o_token, spec["opt"], lr, masks, STEPS, W0)
        st = sts[STEPS]
        p_ref = prev["arms"][arm]["runs"][prev["arms"][arm]["selected_lr"]]["curve"][-1]["heldout_p_o"]
        rec: dict[str, Any] = {"lr": lr, "p_o_heldout": curve[-1]["heldout_p_o"], "p_o_repro_diff": curve[-1]["heldout_p_o"] - p_ref}
        if saved is not None:
            rec["max_state_diff_vs_saved"] = max(float((st[k][i].cpu() - saved["uv"][k][i]).abs().max()) for k in keys for i in (0, 1))
            rec["moved"] = [{"key": f"{k}:{int(alive_idx[k][j])}", "g_trigger_W0": float(g0[k][:, half, alive_idx[k][j]].float().mean())}
                            for k in keys for j in masks[k].nonzero().flatten().tolist()][:20]
        W = {k: W0[k] + (st[k][0] @ st[k][1].T - UV[k][0] @ UV[k][1].T) for k in keys}
        comps = components_with(model, alive, st)
        with model_state(model, target=W, components=comps):
            rec["w0_gated"] = masked_trigger(model, wtoks, half, o_token, g0)
            rec["state_gated"] = masked_trigger(model, wtoks, half, o_token, gates(model, wtoks))
        if arm == "adam_all":
            with model_state(model, target=W):
                out["do_nothing_w0_gated"] = masked_trigger(model, wtoks, half, o_token, g0)
                out["do_nothing_state_gated"] = masked_trigger(model, wtoks, half, o_token, gates(model, wtoks))
            dn = out["do_nothing_w0_gated"]
            out["control_has_power"] = dn["p_o_ci"] < 0.5 * dn["p_o_target"]
            log(f"do-nothing W0-gated {dn}; state-gated {out['do_nothing_state_gated']}")
        out["arms"][arm] = rec
        log(f"[{arm}] repro diff {rec['p_o_repro_diff']:.2e} state diff {rec.get('max_state_diff_vs_saved')}; "
            f"W0-gated {json.dumps({k: round(v, 4) for k, v in rec['w0_gated'].items()})}; "
            f"state-gated p_o_ci {rec['state_gated']['p_o_ci']:.4f}")
        dump("amend1.json", out)
    A = out["arms"]
    r = {a: A[a]["w0_gated"]["p_o_ci"] / A[a]["w0_gated"]["p_o_target"] for a in A}
    out["score"] = {"control_has_power": out["control_has_power"], "ratio_ci_to_target": r,
                    "A1_siblings_w0_gated": r["siblings"] >= 0.8, "A2_top10_w0_gated": r["top10"] >= 0.8,
                    "A3_adam_all_w0_gated": r["adam_all"] < 0.5}
    log(f"score {json.dumps(out['score'])}")
    dump("amend1.json", out)
    log("DONE")


@torch.no_grad()
def lo_masked(model: Any, toks: torch.Tensor, half: int, o_token: int, g: dict[str, torch.Tensor]) -> float:
    from param_decomp.models.components import make_mask_infos

    from commute.identity import log_odds

    return float(log_odds(model(toks, mask_infos=make_mask_infos(g)).float()[:, half], o_token).mean())


def hybrid(g0: dict[str, torch.Tensor], gs: dict[str, torch.Tensor], sel: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """W0's gates, except the subcomponents sel[k] (full indices) take the state's gates at every position."""
    out = {}
    for k, v in g0.items():
        v = v.clone()
        if k in sel and len(sel[k]):
            i = sel[k].to(v.device)
            v[..., i] = gs[k][..., i]
        out[k] = v
    return out


def stage_gating(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.faithfulness import model_state

    if not PRED_AMEND2.is_file():
        sys.exit("file the amendment first: run.py --predictions-amend2")
    dev = args.device
    prev = json.loads((RESULTS / "concentration.json").read_text())
    model = vpd.load_component_model(dev)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    keys = vpd.module_keys(model)
    UV = {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
              model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in keys}
    wtoks, half = win["heldout"].tokens.to(dev), win["heldout"].half
    g0 = gates(model, wtoks)
    lo_W0 = lo_masked(model, wtoks, half, o_token, g0)
    gen = torch.Generator().manual_seed(0)
    out: dict[str, Any] = {"lo_W0_masked": lo_W0, "arms": {}}
    S_on: dict[str, dict[str, torch.Tensor]] = {}
    dgs: dict[str, dict[str, torch.Tensor]] = {}
    for arm in GATING_ARMS:
        src = "adam_all" if arm == "do_nothing" else arm
        spec = ARMS[src]
        lr = float(prev["arms"][src]["selected_lr"])
        masks = None
        if spec["select"] != "all":
            saved = torch.load(LARGE / f"{src}.pt", weights_only=False)
            masks = {k: ((saved["uv"][k][0] - UV[k][0].cpu()).abs().sum(0) + (saved["uv"][k][1] - UV[k][1].cpu()).abs().sum(0)) > 0
                     for k in keys}
        curve, sts = train(model, UV, win, o_token, spec["opt"], lr, masks, STEPS, W0)
        st = sts[STEPS]
        rec: dict[str, Any] = {"p_o_repro_diff": curve[-1]["heldout_p_o"]
                               - prev["arms"][src]["runs"][prev["arms"][src]["selected_lr"]]["curve"][-1]["heldout_p_o"]}
        W = {k: W0[k] + (st[k][0] @ st[k][1].T - UV[k][0] @ UV[k][1].T) for k in keys}
        ctx = model_state(model, target=W) if arm == "do_nothing" else model_state(model, target=W, components=components_with(model, alive, st))
        with ctx:
            gs = gates(model, wtoks)
            lo_s, lo_0 = lo_masked(model, wtoks, half, o_token, gs), lo_masked(model, wtoks, half, o_token, g0)
            dg = {k: (gs[k][:, half] - g0[k][:, half]).float().mean(0) for k in keys}
            dg = {k: torch.where(alive[k].to(v.device), v, torch.zeros_like(v)) for k, v in dg.items()}
            dgs[arm] = dg
            rec.update({"lo_state_gates": lo_s, "lo_W0_gates": lo_0, "gap": lo_s - lo_0, "recovery": {}, "n_on": {},
                        "n_off_0.3": int(sum(int((v < -DG_THR).sum()) for v in dg.values()))})
            for thr in (0.1, DG_THR, 0.5):
                sel = {k: (v > thr).nonzero().flatten() for k, v in dg.items()}
                rec["n_on"][str(thr)] = int(sum(len(i) for i in sel.values()))
                rec["recovery"][str(thr)] = (lo_masked(model, wtoks, half, o_token, hybrid(g0, gs, sel)) - lo_0) / (lo_s - lo_0)
                if thr == DG_THR:
                    S_on[arm] = sel
                    rnd = {}
                    for k, i in sel.items():
                        pool = ((dg[k].abs() < 0.05) & alive[k].to(dg[k].device)).nonzero().flatten().cpu()
                        rnd[k] = pool[torch.randperm(len(pool), generator=gen)[:len(i)]]
                    rec["recovery_random_control"] = (lo_masked(model, wtoks, half, o_token, hybrid(g0, gs, rnd)) - lo_0) / (lo_s - lo_0)
            rec["recovery_all"] = (lo_masked(model, wtoks, half, o_token, hybrid(g0, gs, {k: torch.arange(v.shape[-1]) for k, v in g0.items()})) - lo_0) / (lo_s - lo_0)
            flat = torch.cat([dg[k].cpu() for k in keys])
            own = [(k, c) for k in keys for c in range(len(dg[k]))]
            rec["top_on"] = [{"key": f"{own[i][0]}:{own[i][1]}", "dg": float(flat[i]), "g_W0": float(g0[own[i][0]][:, half, own[i][1]].float().mean())}
                             for i in flat.argsort(descending=True)[:25].tolist()]
        out["arms"][arm] = rec
        log(f"[{arm}] repro {rec['p_o_repro_diff']:.1e}; gap {rec['gap']:.3f}; n_on {rec['n_on']}; recovery {json.dumps({k: round(v, 3) for k, v in rec['recovery'].items()})}; "
            f"random control {rec['recovery_random_control']:.3f}; all {rec['recovery_all']:.3f}")
        dump("amend2.json", out)

    def jac(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
        sa = {(k, int(c)) for k, v in a.items() for c in v}
        sb = {(k, int(c)) for k, v in b.items() for c in v}
        return len(sa & sb) / max(1, len(sa | sb))
    out["jaccard"] = {f"{a}|{b}": jac(S_on[a], S_on[b]) for i, a in enumerate(GATING_ARMS) for b in GATING_ARMS[i + 1:]}
    log(f"jaccard {json.dumps({k: round(v, 3) for k, v in out['jaccard'].items()})}")

    # switch-on effect at W0 (model is at W0 again)
    union: dict[tuple[str, int], float] = {}
    for arm in GATING_ARMS:
        for k, idx in S_on[arm].items():
            for c in idx.tolist():
                union[(k, c)] = max(union.get((k, c), -1.0), float(dgs[arm][k][c]))
    on_list = sorted(union, key=lambda kc: -union[kc])[:300]
    quiet = [(k, c) for k in keys for c in (alive[k].cpu() & torch.stack([dgs[a][k].abs().cpu() < 0.05 for a in GATING_ARMS]).all(0)).nonzero().flatten().tolist()]
    null_list = [quiet[i] for i in torch.randperm(len(quiet), generator=gen)[:300].tolist()]

    def effect(k: str, c: int) -> float:
        g = {kk: v.clone() if kk == k else v for kk, v in g0.items()}
        g[k][..., c] = 1.0
        return lo_masked(model, wtoks, half, o_token, g) - lo_W0
    e_on = torch.tensor([effect(k, c) for k, c in on_list], dtype=torch.float64)
    e_null = torch.tensor([effect(k, c) for k, c in null_list], dtype=torch.float64)
    se = float((e_on.var() / len(e_on) + e_null.var() / len(e_null)).sqrt())
    out["switch_on"] = {"n_on": len(on_list), "n_null": len(null_list), "mean_on": float(e_on.mean()), "mean_null": float(e_null.mean()),
                        "se_diff": se, "median_on": float(e_on.median()), "median_null": float(e_null.median()),
                        "frac_on_positive": float((e_on > 0).double().mean()), "frac_null_positive": float((e_null > 0).double().mean()),
                        "top_on": [{"key": f"{on_list[i][0]}:{on_list[i][1]}", "effect": float(e_on[i])} for i in e_on.argsort(descending=True)[:15].tolist()]}
    log(f"switch-on {json.dumps({k: v for k, v in out['switch_on'].items() if k != 'top_on'})}")
    A = out["arms"]
    sw = out["switch_on"]
    out["score"] = {
        "B1_switched_on_set_explains": all(A[a]["recovery"][str(DG_THR)] >= 0.5 and A[a]["recovery_random_control"] < 0.2
                                           for a in ("do_nothing", "top10", "siblings")),
        "B2_same_machinery": out["jaccard"]["do_nothing|siblings"] >= 0.3 and out["jaccard"]["do_nothing|top10"] >= 0.3,
        "B3_already_writes_o": sw["mean_on"] > sw["mean_null"] + 3 * sw["se_diff"] and sw["mean_on"] > 0,
    }
    log(f"score {json.dumps(out['score'])}")
    dump("amend2.json", out)
    log("DONE")


def pile_faith(model: Any, blocks: list[list[torch.Tensor]], seed: int = 0) -> dict[str, float]:
    from commute.faithfulness import faithfulness

    per = [faithfulness(model, b, seed=seed, log=log) for b in blocks]
    return {k: sum(f[k] for f in per) / len(per) for k in per[0] if k != "pgd_rows"}


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))


# ----------------------------------------------------------------------------- main stage


def stage_all(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.data import batches, pile_rows
    from commute.faithfulness import model_state
    from commute.finetune import ArmAConfig
    from commute.identity import change_stats, cosine, profile_retrieval

    if not PRED.is_file():
        sys.exit("file predictions first: run.py --predictions")
    smoke, dev = args.smoke, args.device
    model = vpd.load_component_model(dev)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    keys = vpd.module_keys(model)
    UV = {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
              model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in keys}
    alive_idx = {k: alive[k].nonzero().flatten() for k in keys}
    name = "concentration_smoke.json" if smoke else "concentration.json"
    out: dict[str, Any] = {"smoke": smoke, "arms": {}}
    steps = 10 if smoke else STEPS
    wtoks, half = win["heldout"].tokens.to(dev), win["heldout"].half

    # --- selection
    sg = scale_gradients(model, UV, win["train"], ArmAConfig(o_token=o_token))
    flat = torch.cat([sg[k].abs() for k in keys])
    owner = [(k, j) for k in keys for j in range(len(sg[k]))]
    order = flat.argsort(descending=True)
    rank_of = {owner[i]: r + 1 for r, i in enumerate(order.tolist())}
    sib = [(SIB_KEY, int((alive_idx[SIB_KEY] == c).nonzero())) for c in SIBLINGS if bool(alive[SIB_KEY][c])]
    out["scale_gradient"] = {
        "top20": [{"key": f"{owner[i][0]}:{int(alive_idx[owner[i][0]][owner[i][1]])}", "abs_grad": float(flat[i])}
                  for i in order[:20].tolist()],
        "sibling_ranks": {f"{SIB_KEY}:{int(alive_idx[k][j])}": rank_of[(k, j)] for k, j in sib},
        "n_alive": len(owner)}
    log(f"sibling ranks in |dL/ds|: {out['scale_gradient']['sibling_ranks']}")

    def masks_for(sel: Any) -> dict[str, torch.Tensor] | None:
        if sel == "all":
            return None
        chosen = sib if sel == "siblings" else [owner[i] for i in order[:sel].tolist()]
        m = {k: torch.zeros(len(sg[k]), dtype=torch.bool) for k in keys}
        for k, j in chosen:
            m[k][j] = True
        return m

    # --- train every arm, select its lr
    states: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        lrs = spec["lrs"][:2] if smoke else spec["lrs"]
        masks = masks_for(spec["select"])
        runs, sts = {}, {}
        for lr in lrs:
            curve, st = train(model, UV, win, o_token, spec["opt"], lr, masks, steps, W0)
            runs[f"{lr:g}"], sts[f"{lr:g}"] = {"curve": curve}, st
            c = curve[-1]
            log(f"[{arm} lr {lr:g}] step {c['step']}: P(o) {c['heldout_p_o']:.4f} kl_other {c['heldout_kl_other']:.4f} "
                f"rel_norm_max {c['rel_norm_max']:.5f}")
        lr, learned = select_lr(runs)
        states[arm] = {"uv": sts[lr][steps], "masks": masks}
        out["arms"][arm] = {"runs": runs, "selected_lr": lr, "learned": learned,
                            "n_moved": None if masks is None else int(sum(int(m.sum()) for m in masks.values()))}
        log(f"[{arm}] selected lr {lr} learned {learned}")
        dump(name, out)

    # --- banks and W0 references
    pile = pile_rows("val", *BANK_ROWS, log)
    pile_b = [b.to(dev) for b in batches(pile, 8)]
    n_pile = pile.numel()
    before_acts = read_acts(model, UV, pile_b + [wtoks], keys)
    lo_W0 = heldout_log_odds(model, wtoks, half, o_token)
    tl_W0 = trigger_local(model, wtoks, half, o_token)
    g_trig = tl_W0.pop("g_trigger")
    out["trigger_local_W0"] = tl_W0
    eff_cache: dict[tuple[str, int], Any] = {}

    def before_effect(k: str, j: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        if (k, j) not in eff_cache:
            pos = before_acts[k][:n_pile, j].abs().topk(K_POS).indices.cpu()
            e = ablation_effect(model, k, j, UV[k][0], UV[k][1], pile, pos)
            eff_cache[(k, j)] = (pos, e.mean(0), cosine(e[0::2].mean(0), e[1::2].mean(0)))
        return eff_cache[(k, j)]

    def identity(k: str, j: int, st: dict[str, Any], W: dict[str, torch.Tensor]) -> dict[str, Any]:
        with model_state(model, target=W):
            acts = read_acts(model, {k: st[k]}, pile_b + [wtoks], [k])[k]
        pr = profile_retrieval(before_acts[k], acts, torch.tensor([j]))
        pos, e_before, floor = before_effect(k, j)
        with model_state(model, target=W):
            e_after = ablation_effect(model, k, j, st[k][0], st[k][1], pile, pos).mean(0)
        cs = change_stats(UV[k][0][:, j:j + 1], UV[k][1][:, j:j + 1], st[k][0][:, j:j + 1], st[k][1][:, j:j + 1])
        m = {"key": f"{k}:{int(alive_idx[k][j])}", **{s: float(v[0]) for s, v in cs.items()},
             "r_self": float(pr["r_self"][0]), "rank": int(pr["rank"][0]), "r_other_max": float(pr["r_other_max"][0]),
             "write_cos": cosine(e_before, e_after), "write_floor": floor}
        m["keeps_identity"] = m["r_self"] >= 0.9 and m["rank"] == 1 and m["write_cos"] >= m["write_floor"] - 0.1
        return m

    # --- identity positive controls on W0 (gate)
    k0, j0 = sib[0]
    gen = torch.Generator().manual_seed(0)
    ctrl = {}
    U, V = UV[k0]
    other = (j0 + 1) % U.shape[1]
    rnd = torch.randn(U.shape[0], generator=gen).to(U.device)
    perp = torch.randn(V.shape[0], generator=gen).to(V.device)
    perp = perp - (perp @ V[:, j0]) / (V[:, j0] @ V[:, j0]) * V[:, j0]
    perp = perp / perp.norm() * V[:, j0].norm()
    variants = {"swap_v": (U[:, j0], V[:, other]), "random_u": (rnd / rnd.norm() * U[:, j0].norm(), V[:, j0])}
    for deg in (20, 45):
        t = torch.tensor(deg * torch.pi / 180)
        variants[f"rot_v_{deg}"] = (U[:, j0], torch.cos(t) * V[:, j0] + torch.sin(t) * perp)
    for vname, (u2, v2) in variants.items():
        U2, V2 = U.clone(), V.clone()
        U2[:, j0], V2[:, j0] = u2, v2
        st = {**UV, k0: (U2, V2)}
        W = {**W0, k0: W0[k0] + U2 @ V2.T - U @ V.T}
        ctrl[vname] = identity(k0, j0, st, W)
        log(f"[identity control {vname}] keeps_identity {ctrl[vname]['keeps_identity']} r_self {ctrl[vname]['r_self']:.3f} "
            f"rank {ctrl[vname]['rank']} write_cos {ctrl[vname]['write_cos']:.3f} (floor {ctrl[vname]['write_floor']:.3f})")
    out["identity_controls"] = ctrl
    out["identity_has_power"] = not ctrl["swap_v"]["keeps_identity"] and not ctrl["random_u"]["keeps_identity"]

    # --- Pile faithfulness: W0 noise floor
    blocks = None
    if not smoke:
        blocks = [[b.to(dev) for b in batches(pile_rows("val", 128 * i, 128, log), 8)] for i in range(2)]
        out["pile_W0"] = {str(s): pile_faith(model, blocks, seed=s) for s in PGD_SEEDS}
        dump(name, out)

    # --- per arm
    for arm, S in states.items():
        t0 = time.time()
        st, masks = S["uv"], S["masks"]
        W = {k: W0[k] + (st[k][0] @ st[k][1].T - UV[k][0] @ UV[k][1].T) for k in keys}
        rec = out["arms"][arm]
        comps = components_with(model, alive, st)
        with model_state(model, target=W, components=comps):
            tl = trigger_local(model, wtoks, half, o_token)
            tl.pop("g_trigger")
            rec["trigger_local"] = tl
            if blocks is not None:
                rec["pile"] = pile_faith(model, blocks)
                rec["pile_ratios"] = {m: rec["pile"][m] / out["pile_W0"]["0"][m] for m in rec["pile"]}
        if arm == "adam_all":  # positive control: do-nothing (components at W0, Delta absorbs dW)
            with model_state(model, target=W):
                dn = trigger_local(model, wtoks, half, o_token)
                dn.pop("g_trigger")
            out["do_nothing_trigger_local"] = dn
            out["trigger_local_has_power"] = dn["p_o_ci"] < 0.5 * dn["p_o_target"]
            log(f"[do-nothing control] {json.dumps({k: round(v, 4) for k, v in dn.items()})}")
        with model_state(model, target=W):
            lo = heldout_log_odds(model, wtoks, half, o_token)
            gain = lo - lo_W0
            rkeys = keys if masks is None else [k for k in keys if bool(masks[k].any())]
            drop = revert_one(model, UV, st, wtoks, half, o_token, lo, rkeys)
        fl = torch.cat([drop[k] for k in rkeys])
        own = [(k, j) for k in rkeys for j in range(len(drop[k]))]
        ordr = fl.argsort(descending=True)
        top = [own[i] for i in ordr.tolist()]
        cum = {}
        for kk in (1, 10, 100, 1000, "all"):
            chosen = own if kk == "all" else top[:kk]
            sel: dict[str, list[int]] = {}
            for k, j in chosen:
                sel.setdefault(k, []).append(j)
            with model_state(model, target=reverted_weights(W, UV, st, {k: torch.tensor(v) for k, v in sel.items()})):
                cum[str(kk)] = (lo - heldout_log_odds(model, wtoks, half, o_token)) / gain
        rec["log_odds"] = {"W0": lo_W0, "state": lo, "gain": gain}
        rec["fraction_removed"] = cum
        rec["revert_one_top20"] = [{"key": f"{k}:{int(alive_idx[k][j])}", "drop": float(drop[k][j]),
                                    "g_trigger_W0": float(g_trig[k][alive_idx[k][j]])} for k, j in top[:20]]
        if masks is None:
            gt = torch.cat([g_trig[k][alive_idx[k]].double().cpu() for k in keys])
            pos = fl.clamp_min(0)
            rec["gamma_vs_update"] = {"share_drop_g_gt_0.5": float(pos[gt > 0.5].sum() / pos.sum()),
                                      "n_g_gt_0.5": int((gt > 0.5).sum()), "spearman_drop_g": spearman(fl, gt)}
        ids = top[:K_ID] if masks is None else [(k, j) for k in keys for j in masks[k].nonzero().flatten().tolist()][:K_ID]
        rec["identity"] = [identity(k, j, st, W) for k, j in ids]
        rec["n_keep_identity"] = sum(m["keeps_identity"] for m in rec["identity"])
        LARGE.mkdir(parents=True, exist_ok=True)
        torch.save({"uv": {k: (u.cpu(), v.cpu()) for k, (u, v) in st.items()} if masks is not None else None,
                    "drop": drop}, LARGE / f"{arm}.pt")
        log(f"[{arm}] trigger-local {json.dumps({k: round(v, 4) for k, v in tl.items()})}; pile ratios "
            f"{json.dumps({k: round(v, 3) for k, v in rec.get('pile_ratios', {}).items()})}; fraction removed "
            f"{json.dumps({k: round(v, 3) for k, v in cum.items()})}; identity {rec['n_keep_identity']}/{len(ids)}; "
            f"{time.time() - t0:.0f} s")
        dump(name, out)

    # --- scores
    A = out["arms"]

    def inside(a: dict[str, Any]) -> dict[str, Any]:
        tl = a["trigger_local"]
        r = a.get("pile_ratios", {})
        return {"learned": a["learned"],
                "pile_ok": bool(r) and all(r[m] <= 1.5 for m in ("stochastic_kl", "pgd_kl", "ci_masked_kl")),
                "trigger_ok": tl["p_o_ci"] >= 0.8 * tl["p_o_target"]}

    sc: dict[str, Any] = {"trigger_local_has_power": out["trigger_local_has_power"],
                          "identity_has_power": out["identity_has_power"]}
    sc["1_sgd_concentrated"] = {"scorable": A["sgd_all"]["learned"], "top10": A["sgd_all"]["fraction_removed"]["10"],
                                "pass": A["sgd_all"]["fraction_removed"]["10"] >= 0.5,
                                "null": A["sgd_all"]["fraction_removed"]["10"] < 0.2}
    for pid, arm in (("2_top10_inside", "top10"), ("3_siblings_inside", "siblings")):
        c = inside(A[arm])
        c["pass"] = c["learned"] and c["pile_ok"] and c["trigger_ok"]
        c["trigger_scorable"] = out["trigger_local_has_power"]
        sc[pid] = c
    out["score"] = sc
    log(f"score {json.dumps(sc)}")
    dump(name, out)
    log("DONE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--predictions-amend", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all", "amend", "gating"])
    ap.add_argument("--predictions-amend2", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    if args.predictions_amend:
        file_predictions_amend()
        return
    if args.predictions_amend2:
        file_predictions_amend2()
        return
    if args.stage is None:
        ap.error("give a stage or --predictions")
    {"all": stage_all, "amend": stage_amend, "gating": stage_gating}[args.stage](args)


if __name__ == "__main__":
    main()
