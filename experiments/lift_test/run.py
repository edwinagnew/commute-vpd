"""The lift test: can a simple map carry a fine-tune onto the decomposition, with Delta held fixed?

Plan and design check: reports/2026-10-14_plan_lift-test.md (confirmed by the user 2026-10-14).

Stages (JSON into results/, large tensors into results/large/):
  --predictions         file results/predictions.json (refuses to overwrite)
  lora                  GPU: LoRAs (ranks 1, 8) on h.2.mlp.down_proj with arm A's data, delta-W checkpoints
  sgd                   GPU: arm A with plain SGD (lr 0.1, 0.3, 1), the control for the Adam hypothesis
  weighted              GPU: input-weighted obstruction (E[a a^T] on Pile and on the windows) for every checkpoint and W0
  projected             GPU: behaviour of W0 + (dW - O(dW)) and W0 + O(dW) against W0 + dW
  pullback              GPU: prediction 3 (arm A rerun for s*, pull-back at 3 lrs, faithfulness)
  --predictions-amend   file results/predictions_amend1.json (P2', P3'; refuses to overwrite)
  lift_faith            GPU: prediction 2' (Euclidean lift vs do-nothing, faithfulness at rerun steps 1..10)
  pullback2             GPU: prediction 3' (pull-back at lrs 1e-5..3e-5, faithfulness at steps 10 and 30)
  obstruction [paths]   stage 0, CPU, no forward passes: obstruction fraction, ranks, conditioning and the
                        Euclidean lift's size for each delta-W checkpoint (default: round 1's, both arms)
Stages 1 (lifts scored by faithfulness) and 2 (pull-back replay) run on a GPU and are added before they run.

Usage (repo root):  uv run python experiments/lift_test/run.py obstruction
"""

from __future__ import annotations

import argparse
import json
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
TOLS = (1e-6, 1e-3, 1e-2)  # singular value cut-offs relative to the largest; 1e-6 is the scored one

PREDICTIONS: dict[str, Any] = {
    "about": (
        "lift_test (round 2): for a small fine-tune delta-W of Goodfire's 4-layer 67M Pile model, does a simple map "
        "(no extra network trained; user, 2026-10-14) put delta-W on the alive subcomponents of the released VPD "
        "decomposition s-55ea3f9b, Gamma frozen and Delta held at Delta_0, so that the result is a faithful "
        "decomposition of W0 + delta-W (the weak version, GOALS.md)? Definitions: theory/commuting_updates.md. Plan: "
        "reports/2026-10-14_plan_lift-test.md. Filed before any round-2 measurement was run. Round 1's delta-W "
        "checkpoints (experiments/transport_test/results/large/{A,B}/dW_step*.pt) exist and their norms are known "
        "from round 1; their obstruction fractions have not been computed."
    ),
    "setup": {
        "decomposition": "U, V of the 24 matrices read from s-55ea3f9b/model_400000.pth; alive = round 1's mask "
                         "(experiments/transport_test/results/large/alive.pt: mean g over Pile val rows 2048..2303 > 1e-6, "
                         "9 881 alive). Theory convention U_A = U[alive].T (d_out, n), V_A = V[:, alive] (d_in, n); all "
                         "linear algebra in float64",
        "rank_rule": "a singular value of U_A or V_A counts as non-zero if it exceeds tol * the largest singular value "
                     "of that matrix; tol = 1e-6 is scored, 1e-3 and 1e-2 are reported",
        "obstruction_fraction": "omega_m = ||P_U^perp dW_m P_V^perp||_F^2 / ||dW_m||_F^2; pooled omega = "
                                "sum_m ||O(dW_m)||^2 / sum_m ||dW_m||^2 over a stated set of matrices",
        "gaussian_null": "per matrix (1 - r_U/d_out)(1 - r_V/d_in); pooled with the same weights ||dW_m||^2 as the real "
                         "update (the expected pooled omega if each dW_m were replaced by an isotropic matrix of the same norm)",
        "non_saturated": "matrices with r_U < d_out and r_V < d_in at tol = 1e-6; omega is identically 0 on the others, "
                         "where s_min of d sigma is reported instead",
        "arm_A_rerun": "round 1's arm A recipe (365 emoticon windows, full-weight Adam on the 24 matrices, lr 3e-5, batch 32, "
                       "CE on `o` at index 20 + 1.0 * KL on the other 40 positions, grad clip 1, seed 0) rerun with "
                       "checkpoints at steps 1, 2, 3, 5, 10, 30; s* = the first of these with held-out mean P(o) >= 0.9",
        "faithfulness": "round 1's harness (src/commute/faithfulness.py; 2 blocks of 128 Pile val rows, stochastic KL with "
                        "2 draws per batch, PGD 20 steps of 0.1 with one adversary per block, Delta included in both). "
                        "Target weights W0 + dW, components replaced by the lifted ones, Gamma frozen; Delta is whatever "
                        "param-decomp computes (W_target - sum of all components), so whatever the lift leaves stays in "
                        "Delta and is charged to L_F. Ratios are to the released decomposition on W0 in the same harness, "
                        "same rows and seed. L_F = sum_m ||Delta_m||_F^2 (the ratio needs no normaliser)",
        "maps": {
            "do_nothing": "components unchanged (Delta absorbs dW)",
            "euclidean": "minimum-norm first-order lift dU = X V_A, dV = X^T U_A with X = K^+ (dW - O(dW)), tol 1e-6 "
                         "(theory section 5.1 instance 1; src/commute/lift.py)",
            "locality": "theory section 5.1 instance 2, lambda fixed by the rule in the stage-1 code before stage 1 runs; "
                        "reported, not scored",
            "pullback": "Adam on (U_A, V_A) of all 24 matrices (dead columns and Delta_0 fixed; target weights = sigma(d) + "
                        "Delta_0), arm A's loss, windows, batch, clip and seed; lr in {1e-4, 3e-4, 1e-3}, 30 steps each; "
                        "the lr with the highest held-out P(o) at step 30 is the scored one (rule fixed now)",
        },
        "reported_not_scored": [
            "omega, ranks and s_min for round 1's checkpoints of both arms (steps 10..300)",
            "omega for arm B rerun at a learning rate giving ~0.1 nats held-out KL, and for LoRAs of rank 1 and 8 on "
            "h.2.mlp.down_proj trained on arm A's data",
            "omega for the gradient-shaped null (Corollary 2: same per-example errors and inputs, coefficients randomised)",
            "Euclidean lift size ||(dU, dV)|| / ||(U_A, V_A)|| beside ||dW|| / ||W0|| (gauge-dependent: released gauge)",
            "L_I and the CI-masked and unmasked KLs (Delta in and out) for every map",
            "locality residual rho for the Euclidean and locality lifts",
        ],
    },
    "predictions": [
        {
            "id": "1_obstruction_small",
            "scored": True,
            "claim": "Almost all of a small fine-tune can be expressed by first-order changes to the alive subcomponents.",
            "measure": "pooled omega (tol 1e-6) over the non-saturated matrices, and the pooled Gaussian null over the same "
                       "matrices, at round 1's arm A step 10 and at the rerun's steps 1, 2, 3, 5, 10",
            "pass": "at every one of these checkpoints: omega <= 0.1 AND omega <= 0.5 * null",
            "null": "omega >= 0.8 * null (the update is no better aligned with the alive read/write spaces than noise)",
            "prediction": "pass",
        },
        {
            "id": "2_lift_faithful",
            "scored": True,
            "claim": "The Euclidean lift of a small fine-tune is a faithful decomposition of the fine-tuned model.",
            "measure": "Euclidean map at the rerun's arm A checkpoints 1, 2, 3, 5, 10: ratios of L_F, stochastic KL and PGD KL "
                       "to the released decomposition on W0",
            "pass": "all three ratios <= 1.5 at every checkpoint",
            "null": "brittleness: a stochastic or PGD ratio > 3 at a checkpoint where the L_F ratio <= 1.5",
            "sanity": "the do-nothing map has L_F ratio > 1.5 at step 10; if not, delta-W is too small against Delta_0 for L_F "
                      "to separate maps, and the L_F part of this prediction is reported as unscorable",
            "prediction": "pass",
        },
        {
            "id": "3_pullback_replay",
            "scored": True,
            "claim": "Post-training can be done in the decomposition: training U_A, V_A on the fine-tune's loss reproduces "
                     "the behaviour and keeps the decomposition faithful.",
            "measure": "scored pull-back run: held-out mean P(o) at each step 1..30; faithfulness ratios (stochastic, PGD) at "
                       "step min(3 s*, 30)",
            "pass": "held-out P(o) >= 0.9 by step 3 s* AND stochastic and PGD ratios <= 1.5 at that step",
            "prediction": "pass",
            "conditioning": "scorable only if the rerun reaches held-out P(o) >= 0.9 by step 10 (defines s*)",
        },
    ],
    "filed": None,
}


PRED_AMEND = RESULTS / "predictions_amend1.json"

PREDICTIONS_AMEND: dict[str, Any] = {
    "about": (
        "Amendment 1 to lift_test predictions (reports/2026-10-14_prediction_lift-test-amendment.md). The original "
        "results/predictions.json is unchanged and stays scored as filed (P1 reversed; P2's L_F term unscorable; P3 "
        "failed). P2' and P3' are new predictions, filed before either was run, motivated by what the first runs showed: "
        "the do-nothing map passes the masked KLs and L_F cannot see delta-W against ||Delta_0||^2 = 67.9, and the filed "
        "pull-back lr grid moved the weights 3-100x more than the W fine-tune."
    ),
    "setup": {
        "updates": "the arm A rerun (Adam, lr 3e-5, round 1's recipe), checkpoints 1, 2, 3, 5, 10 "
                   "(results/large/A_rerun/dW_step*.pt; s* = 1, evals in results/pullback.json)",
        "faithfulness": "round 1's harness (2 blocks of 128 Pile val rows 0..255; stochastic, PGD 20 x 0.1 one adversary per "
                        "block, CI-masked and unmasked KL with Delta excluded); ratios to the released decomposition on W0 "
                        "recomputed in the same job",
        "least_squares_split": "per matrix, dU, dV = split.least_squares_split(dW, basis(U_A, V_A, tol 1e-6)); components U[alive] = "
                          "(U_A + dU)^T, V[:, alive] = V_A + dV; target weights W0 + dW, so Delta = Delta_0 + O(dW) - dU dV^T",
        "do_nothing": "components unchanged, target weights W0 + dW (Delta = Delta_0 + dW)",
        "pullback": "Adam on the alive (U, V) offsets (src/commute/pullback.py), Delta_0 and dead columns fixed, arm A's loss, "
                    "windows, batch 32, clip 1, seed 0; lr in {1e-5, 2e-5, 3e-5}; 30 steps; held-out eval every step",
        "pullback_selection": "among lrs with held-out P(o) >= 0.9 at step 30, the one with the lowest held-out kl_other at "
                              "step 30; if none, the highest held-out P(o) at step 30",
    },
    "predictions": [
        {
            "id": "2prime_lift_faithful",
            "scored": True,
            "claim": "The Euclidean lift of a small fine-tune is a faithful decomposition of the fine-tuned model, and its "
                     "components carry the update better than leaving them unchanged.",
            "measure": "at each rerun checkpoint 1, 2, 3, 5, 10: ratios r = metric / metric(W0) for the lift and for "
                       "do-nothing; gap_recovered = (r_nothing - r_lift) / (r_nothing - 1) on unmasked_kl (Delta excluded)",
            "pass": "at every scorable checkpoint: stochastic and PGD ratios of the lift <= 1.5 AND gap_recovered >= 0.5",
            "scorable": "a checkpoint is scorable only if r_nothing(unmasked_kl) - 1 >= 0.2; the prediction is scorable only "
                        "if at least one checkpoint is",
            "reported": ["ci_masked_kl ratios", "||Delta' - Delta_0||^2 / ||dW||^2 per checkpoint"],
            "prediction": "pass",
        },
        {
            "id": "3prime_pullback_matched",
            "scored": True,
            "claim": "Post-training done in the decomposition at a step size matched to the W fine-tune reproduces the "
                     "behaviour without extra off-target damage and keeps the decomposition faithful.",
            "measure": "the selected pull-back run: held-out P(o) per step; held-out kl_other at step 10; stochastic and PGD "
                       "ratios at step 10 (also reported at step 30)",
            "pass": "held-out P(o) >= 0.9 by step 10 AND held-out kl_other at step 10 <= 2 x 0.0550 (the W fine-tune rerun's "
                    "value at step 10) AND stochastic and PGD ratios <= 1.5 at step 10",
            "reported": ["max per-matrix ||dW|| / ||W0|| at steps 1, 2, 3, 5, 10, 30 against the W fine-tune's"],
            "prediction": "pass",
        },
    ],
    "filed": None,
}


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


def file_predictions_amend() -> None:
    if PRED_AMEND.is_file():
        sys.exit(f"{PRED_AMEND} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS_AMEND["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump(PRED_AMEND.name, PREDICTIONS_AMEND)
    log(f"filed {PRED_AMEND}")


def require_predictions_amend() -> dict[str, Any]:
    if not PRED_AMEND.is_file():
        sys.exit("file the amendment first: run.py --predictions-amend")
    return json.loads(PRED_AMEND.read_text())


def require_predictions() -> dict[str, Any]:
    if not PRED.is_file():
        sys.exit("file predictions first: run.py --predictions")
    return json.loads(PRED.read_text())


# ----------------------------------------------------------------------------- stage 0


def load_decomposition() -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor]]:
    """(W0, (U, V) in param-decomp layout, alive) per matrix, read from the checkpoint without param-decomp."""
    from commute import env
    from commute.vpd import DECOMP_FILES, DECOMP_RUN

    sd = torch.load(env.VPD_DIR / DECOMP_RUN / DECOMP_FILES[0], map_location="cpu", weights_only=True, mmap=True)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    W0, UV = {}, {}
    for k in sorted(alive):
        key = k.replace(".", "-")
        W0[k] = sd[f"target_model.{k}.weight"].float()
        UV[k] = (sd[f"_components.{key}.U"].float(), sd[f"_components.{key}.V"].float())
        assert UV[k][0].shape == (alive[k].numel(), W0[k].shape[0]) and UV[k][1].shape == (W0[k].shape[1], alive[k].numel())
    assert len(W0) == 24
    return W0, UV, alive


def stage_obstruction(args: argparse.Namespace) -> None:
    from commute import split

    require_predictions()
    W0, UV, alive = load_decomposition()
    paths = [Path(p) for p in args.paths] or sorted(ROUND1.glob("[AB]/dW_step*.pt"),
                                                    key=lambda p: (p.parent.name, int(p.stem.split("step")[1])))
    t0 = time.time()
    per_matrix: dict[str, Any] = {}
    bases: dict[tuple[str, float], Any] = {}
    delta0_sq = 0.0
    for k in W0:
        U_A, V_A = split.alive_uv(*UV[k], alive[k])
        U, V = UV[k]
        delta0_sq += float(((W0[k] - U.T @ V.T).double() ** 2).sum())
        d_out, d_in = W0[k].shape
        per_matrix[k] = {"d_out": d_out, "d_in": d_in, "n_alive": int(alive[k].sum())}
        svd_u, svd_v = split.full_svd(U_A), split.full_svd(V_A)
        for tol in TOLS:
            b = split.basis_from(svd_u, svd_v, tol)
            bases[(k, tol)] = (b, U_A, V_A)
            per_matrix[k][f"tol{tol:g}"] = {"r_u": b.r_u, "r_v": b.r_v, "null": split.gaussian_null(b, d_out, d_in),
                                           "dsigma_smin": split.dsigma_smin(b),
                                           "s_u_max": float(b.s.max()), "s_v_max": float(b.t.max())}
        log(f"basis {k} ({d_out}x{d_in}, {per_matrix[k]['n_alive']} alive): r_u {per_matrix[k]['tol1e-06']['r_u']} "
            f"r_v {per_matrix[k]['tol1e-06']['r_v']} ({time.time() - t0:.0f} s)")
    log(f"bases for 24 matrices x {len(TOLS)} tolerances in {time.time() - t0:.0f} s; ||Delta_0||^2 = {delta0_sq:.4g}")
    non_sat = [k for k in W0 if per_matrix[k]["tol1e-06"]["r_u"] < per_matrix[k]["d_out"]
               and per_matrix[k]["tol1e-06"]["r_v"] < per_matrix[k]["d_in"]]
    log(f"non-saturated at tol 1e-6: {len(non_sat)} / 24; saturated: {sorted(set(W0) - set(non_sat))}")

    updates: dict[str, Any] = {}
    for p in paths:
        name = f"{p.parent.name}/{p.stem}"
        dW = torch.load(p, map_location="cpu", weights_only=False)
        assert set(dW) == set(W0), p
        rec: dict[str, Any] = {"path": str(p), "dW_sq": {k: float((dW[k].double() ** 2).sum()) for k in W0},
                               "rel_norm": {k: float(dW[k].norm() / W0[k].norm()) for k in W0}}
        rec["dW_sq_total"] = sum(rec["dW_sq"].values())
        # do-nothing map: Delta = Delta_0 + dW, so L_F = ||Delta_0||^2 + 2 <Delta_0, dW> + ||dW||^2
        cross = sum(float(((W0[k] - UV[k][0].T @ UV[k][1].T).double() * dW[k].double()).sum()) for k in W0)
        rec["do_nothing_LF_ratio"] = (delta0_sq + 2 * cross + rec["dW_sq_total"]) / delta0_sq
        for tol in TOLS:
            r: dict[str, Any] = {"omega": {}, "lift_rel_size": {}, "lift_residual": {}}
            for k in W0:
                b, U_A, V_A = bases[(k, tol)]
                M = dW[k].double()
                if rec["dW_sq"][k] == 0:  # e.g. a LoRA leaves the other matrices untouched; weight 0 in the pools
                    r["omega"][k] = 0.0
                    continue
                r["omega"][k] = split.obstruction_fraction(M, b)
                if tol == TOLS[0]:
                    dU, dV = split.least_squares_split(M, b, U_A, V_A)
                    liftable = M - split.obstruction(M, b)
                    r["lift_residual"][k] = float((split.dsigma(dU, dV, U_A, V_A) - liftable).norm() / M.norm())
                    r["lift_rel_size"][k] = float(torch.sqrt((dU**2).sum() + (dV**2).sum())
                                                  / torch.sqrt((U_A**2).sum() + (V_A**2).sum()))
            for label, ks in (("all", list(W0)), ("non_saturated", non_sat)):
                w = sum(rec["dW_sq"][k] for k in ks)
                r[f"pooled_omega_{label}"] = sum(r["omega"][k] * rec["dW_sq"][k] for k in ks) / w
                r[f"pooled_null_{label}"] = sum(per_matrix[k][f"tol{tol:g}"]["null"] * rec["dW_sq"][k] for k in ks) / w
            rec[f"tol{tol:g}"] = r
        s = rec["tol1e-06"]
        log(f"{name}: omega non-sat {s['pooled_omega_non_saturated']:.4f} (null {s['pooled_null_non_saturated']:.4f}), "
            f"all {s['pooled_omega_all']:.4f}; tol 1e-2 non-sat {rec['tol0.01']['pooled_omega_non_saturated']:.4f}; "
            f"do-nothing L_F ratio {rec['do_nothing_LF_ratio']:.4f}; max lift residual {max(s['lift_residual'].values()):.2e}; "
            f"h.2.mlp.down_proj omega {s['omega']['h.2.mlp.down_proj']:.4f} "
            f"(null {per_matrix['h.2.mlp.down_proj']['tol1e-06']['null']:.4f})")
        updates[name] = rec

    out = {"per_matrix": per_matrix, "non_saturated": non_sat, "delta0_sq": delta0_sq, "updates": updates,
           "scoring_note": "prediction 1 is scored on A/dW_step10 here and on the rerun's steps 1-10 after the GPU stage",
           "seconds": time.time() - t0}
    dump("obstruction.json", out)
    log(f"wrote {RESULTS / 'obstruction.json'} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- LoRA arm (GPU)


def stage_lora(args: argparse.Namespace) -> None:
    """LoRAs of rank args.ranks on h.2.mlp.down_proj (the paper's LoRA site) with arm A's data and loss."""
    from dataclasses import asdict

    from commute import vpd
    from commute.finetune import (
        ArmAConfig,
        LoraConfig,
        finetune_lora_arm_a,
        relative_norms,
    )

    require_predictions()
    t0 = time.time()
    model = vpd.load_component_model(args.device)
    log(f"model loaded in {time.time() - t0:.0f} s")
    W0 = {k: w.cpu() for k, w in vpd.target_weights(model).items()}
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    a_cfg = ArmAConfig(o_token=o_token)  # round 1's batch, clip, KL weight and seed
    for rank in args.ranks:
        cfg = LoraConfig(modules=("h.2.mlp.down_proj",), rank=rank)
        res = finetune_lora_arm_a(model, win["train"], win["heldout"], a_cfg, cfg, args.device, log)
        tag = f"LoRA_r{rank}"
        (LARGE / tag).mkdir(parents=True, exist_ok=True)
        for step, d in res.deltas.items():
            torch.save(d, LARGE / tag / f"dW_step{step}.pt")
        dump(f"finetune_{tag}.json", {"arm_A_config": asdict(a_cfg), "lora_config": asdict(cfg), "log": res.log,
                                      "evals": res.evals,
                                      "relative_norms": {s: relative_norms(d, W0) for s, d in res.deltas.items()}})


def stage_sgd(args: argparse.Namespace) -> None:
    """Control for the Adam hypothesis: arm A full-weight fine-tune with plain SGD (clipped gradients, no momentum)."""
    from dataclasses import asdict

    from commute import vpd
    from commute.finetune import ArmAConfig, finetune_arm_a, relative_norms

    require_predictions()
    model = vpd.load_component_model(args.device)
    W0 = {k: w.cpu() for k, w in vpd.target_weights(model).items()}
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    for lr in args.sgd_lrs:
        cfg = ArmAConfig(o_token=o_token, lr=lr, steps=30, checkpoints=(1, 2, 3, 5, 10, 30), optimizer="sgd")
        res = finetune_arm_a(model, win["train"], win["heldout"], cfg, args.device, log)
        tag = f"SGD_lr{lr:g}"
        (LARGE / tag).mkdir(parents=True, exist_ok=True)
        for step, d in res.deltas.items():
            torch.save(d, LARGE / tag / f"dW_step{step}.pt")
        dump(f"finetune_{tag}.json", {"config": asdict(cfg), "log": res.log, "evals": res.evals,
                                      "relative_norms": {st: relative_norms(d, W0) for st, d in res.deltas.items()}})


@torch.no_grad()
def input_second_moments(model: Any, batches: list[torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """E[a a^T] of each decomposed matrix's input over every position of the batches (float64, CPU)."""
    from commute.finetune import logits_of
    from commute.vpd import module_keys, target_linear

    acc: dict[str, torch.Tensor] = {}
    n = 0
    handles = []
    for k in module_keys(model):
        def hook(_m: Any, a: tuple[Any, ...], _o: Any, k: str = k) -> None:
            x = a[0].reshape(-1, a[0].shape[-1]).double()
            acc[k] = acc.get(k, 0) + x.T @ x
        handles.append(target_linear(model, k).register_forward_hook(hook))
    try:
        for b in batches:
            logits_of(model.target_model, b.to(device))
            n += b.numel()
    finally:
        for h in handles:
            h.remove()
    return {k: (v / n).cpu() for k, v in acc.items()}


def stage_weighted(args: argparse.Namespace) -> None:
    """Input-weighted obstruction for every delta-W checkpoint and for W0 itself, on Pile and on the windows."""
    from commute import split, vpd
    from commute.data import batches, pile_rows

    require_predictions()
    model = vpd.load_component_model(args.device)
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    sig = {"pile": input_second_moments(model, batches(pile_rows("val", 0, 64, log), 8), args.device),
           "windows": input_second_moments(model, batches(win["train"].tokens, 32), args.device)}
    del model
    W0, UV, alive = load_decomposition()
    bases, non_sat = {}, []
    for k in W0:
        U_A, V_A = split.alive_uv(*UV[k], alive[k])
        bases[k] = split.basis(U_A, V_A, TOLS[0])
        if bases[k].r_u < W0[k].shape[0] and bases[k].r_v < W0[k].shape[1]:
            non_sat.append(k)
    paths = sorted(ROUND1.glob("[AB]/dW_step*.pt")) + sorted(LARGE.glob("*/dW_step*.pt"))
    items = [("W0", W0)] + [(f"{p.parent.name}/{p.stem}", None) for p in paths]
    out: dict[str, Any] = {"non_saturated": non_sat, "read_out": {}, "updates": {}}
    for d, S in sig.items():
        out["read_out"][d] = {k: split.weighted_fractions(W0[k], bases[k], S[k])["read_out"] for k in W0}
    for (name, M), p in zip(items, [None] + paths):
        M = M if M is not None else torch.load(p, map_location="cpu", weights_only=False)
        rec: dict[str, Any] = {"omega": {k: split.obstruction_fraction(M[k], bases[k]) for k in W0 if M[k].any()}}
        fro = {k: float((M[k].double() ** 2).sum()) for k in W0}
        for d, S in sig.items():
            per = {}
            for k in W0:
                if not M[k].any():
                    continue
                w = split.weighted_fractions(M[k], bases[k], S[k])
                den = float(((M[k].double() @ S[k]) * M[k].double()).sum())
                per[k] = {**w, "energy": den}
            rec[d] = per
            for label, ks in (("all", list(per)), ("non_saturated", [k for k in per if k in non_sat])):
                e = sum(per[k]["energy"] for k in ks)
                rec[f"{d}_pooled_omega_act_{label}"] = sum(per[k]["omega_act"] * per[k]["energy"] for k in ks) / e
                rec[f"{d}_pooled_null_act_{label}"] = sum(per[k]["null_act"] * per[k]["energy"] for k in ks) / e
        ks = [k for k in rec["omega"] if k in non_sat]
        rec["pooled_omega_non_saturated"] = sum(rec["omega"][k] * fro[k] for k in ks) / sum(fro[k] for k in ks)
        out["updates"][name] = rec
        log(f"{name}: omega fro {rec['pooled_omega_non_saturated']:.3f} | act pile {rec['pile_pooled_omega_act_non_saturated']:.3f}"
            f" (null {rec['pile_pooled_null_act_non_saturated']:.3f}) | act windows "
            f"{rec['windows_pooled_omega_act_non_saturated']:.3f} (null {rec['windows_pooled_null_act_non_saturated']:.3f})")
    dump("weighted.json", out)


def _bases_gpu(device: str) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    from commute import split

    W0, UV, alive = load_decomposition()
    bases = {k: split.basis(*split.alive_uv(*UV[k], alive[k]), TOLS[0]) for k in W0}
    return W0, UV, bases, alive


@torch.no_grad()
def _kl_stats(nets: dict[str, Any], toks_batches: list[torch.Tensor], device: str, half: int | None) -> dict[str, float]:
    """Mean KL(ref || v), KL(ft || v) for v in nets (all positions, or at `half` and elsewhere); P(o) at half."""
    from commute.finetune import logits_of
    from commute.vpd import kl_per_position

    acc: dict[str, float] = {}
    n = 0
    for b in toks_batches:
        b = b.to(device)
        lg = {name: logits_of(net, b) for name, net in nets.items()}
        for name in nets:
            if name in ("ref",):
                continue
            for base in ("ref", "ft"):
                if name == base:
                    continue
                kl = kl_per_position(lg[base], lg[name])
                if half is None:
                    acc[f"kl_{base}_{name}"] = acc.get(f"kl_{base}_{name}", 0.0) + float(kl.mean(1).sum())
                else:
                    keep = torch.ones(kl.shape[1], dtype=torch.bool, device=device)
                    keep[half] = False
                    acc[f"kl_{base}_{name}_at_o"] = acc.get(f"kl_{base}_{name}_at_o", 0.0) + float(kl[:, half].sum())
                    acc[f"kl_{base}_{name}_other"] = acc.get(f"kl_{base}_{name}_other", 0.0) + float(kl[:, keep].mean(1).sum())
            if half is not None:
                p = torch.softmax(lg[name][:, half], -1)[:, O_TOKEN[0]]
                acc[f"p_o_{name}"] = acc.get(f"p_o_{name}", 0.0) + float(p.sum())
        n += b.shape[0]
        del lg
    return {k: v / n for k, v in acc.items()}


O_TOKEN: list[int] = []


def stage_projected(args: argparse.Namespace) -> None:
    """Does the behaviour survive with only the liftable part of delta-W? W0 + (dW - O(dW)) and W0 + O(dW) against
    W0 + dW, on the held-out windows (P(o), KL at the `o` position and elsewhere) and on Pile val rows 256..319."""
    import copy

    from commute import split, vpd
    from commute.data import batches, pile_rows

    require_predictions()
    O_TOKEN.append(json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"])
    W0, _, bases, _ = _bases_gpu("cpu")
    model = vpd.load_component_model(args.device)
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    held = batches(win["heldout"].tokens, 32)
    pile = batches(pile_rows("val", 256, 64, log), 8)
    ref = copy.deepcopy(model.target_model).eval()
    ft = copy.deepcopy(model.target_model).eval()
    var = model.target_model
    names = ["A/dW_step10", "A/dW_step30", "A/dW_step300", "B/dW_step100", "SGD_lr0.3/dW_step3", "SGD_lr0.3/dW_step10",
             "LoRA_r8/dW_step10", "LoRA_r8/dW_step100"]
    out: dict[str, Any] = {"pile_rows": "val 256..319", "windows": "round 1 held-out", "updates": {}}
    for name in names:
        root = ROUND1 if name[0] in "AB" and name[1] == "/" else LARGE
        dW = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=False)
        O = {k: split.obstruction(dW[k], bases[k]).float() for k in W0}
        rec: dict[str, Any] = {}
        with torch.no_grad():
            for k in W0:
                ft.get_submodule(k).weight.copy_((W0[k] + dW[k]).to(args.device))
        for variant, X in (("liftable", {k: dW[k] - O[k] for k in W0}), ("obstruction", O)):
            vpd.set_target_weights(model, {k: (W0[k] + X[k]).to(args.device) for k in W0})
            nets = {"ref": ref, "ft": ft, variant: var}
            rec[variant] = {"windows": _kl_stats(nets, held, args.device, win["heldout"].half),
                            "pile": _kl_stats(nets, pile, args.device, None)}
        vpd.set_target_weights(model, {k: w.to(args.device) for k, w in W0.items()})
        w_full = _kl_stats({"ref": ref, "ft": ft}, held, args.device, win["heldout"].half)
        p_full = _kl_stats({"ref": ref, "ft": ft}, pile, args.device, None)
        rec["full"] = {"windows": w_full, "pile": p_full}
        lw, lp = rec["liftable"]["windows"], rec["liftable"]["pile"]
        rec["capture_at_o"] = 1 - lw["kl_ft_liftable_at_o"] / w_full["kl_ref_ft_at_o"]
        rec["capture_pile"] = 1 - lp["kl_ft_liftable"] / p_full["kl_ref_ft"]
        out["updates"][name] = rec
        log(f"{name}: P(o) full {w_full['p_o_ft']:.3f} liftable {lw['p_o_liftable']:.3f} obstruction "
            f"{rec['obstruction']['windows']['p_o_obstruction']:.3f} | capture at o {rec['capture_at_o']:.3f} | Pile KL "
            f"W0->full {p_full['kl_ref_ft']:.4f}, W0->liftable {lp['kl_ref_liftable']:.4f}, full->liftable "
            f"{lp['kl_ft_liftable']:.4f} (capture {rec['capture_pile']:.3f})")
        dump("projected.json", out)


def _faith_blocks(device: str) -> list[list[torch.Tensor]]:
    """Round 1's harness: 2 blocks of 128 Pile val rows (rows 0..255) in sub-batches of 8."""
    from commute.data import batches, pile_rows

    return [[b.to(device) for b in batches(pile_rows("val", 128 * i, 128, log), 8)] for i in range(2)]


def _faith(model: Any, blocks: list[list[torch.Tensor]]) -> dict[str, Any]:
    from commute.faithfulness import faithfulness

    per = [faithfulness(model, b, log=log) for b in blocks]
    keys = [k for k in per[0] if k != "pgd_rows"]
    return {**{k: sum(f[k] for f in per) / len(per) for k in keys}, "blocks": per}


def _alive_uv_device(model: Any, alive: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Theory layout (U_A (d_out, n), V_A (d_in, n)), float32, on the model's device."""
    from commute import vpd

    return {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
                model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in vpd.module_keys(model)}


def _components_with(model: Any, alive: dict[str, torch.Tensor],
                     lifted: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Full component weights (param-decomp layout) with the alive columns replaced: U[alive] = U_A'^T, V[:, alive] = V_A'."""
    from commute import vpd

    comps = vpd.component_weights(model)
    for k, (U, V) in comps.items():
        U, V = U.clone(), V.clone()
        U[alive[k]] = lifted[k][0].T.to(U.dtype).to(U.device)
        V[:, alive[k]] = lifted[k][1].to(V.dtype).to(V.device)
        comps[k] = (U, V)
    return comps


def _pullback_runs(model: Any, uv: dict[str, tuple[torch.Tensor, torch.Tensor]], win: dict[str, Any], cfg: Any,
                   o_token: int, lrs: tuple[float, ...], save_steps: tuple[int, ...], device: str,
                   W0: dict[str, torch.Tensor]) -> tuple[dict[str, Any], dict[float, dict[int, Any]]]:
    """Adam on the alive (U, V) offsets at each lr for 30 steps, held-out eval every step. Returns the curves and,
    at save_steps, per matrix (delta-W, U_A', V_A') on CPU."""
    import copy

    from commute import component_training, vpd
    from commute.finetune import arm_a_loss, eval_arm_a

    net = model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    linears = {k: vpd.target_linear(model, k) for k in uv}
    runs: dict[str, Any] = {}
    states: dict[float, dict[int, Any]] = {}
    for lr in lrs:
        off = component_training.Offsets(uv)
        params = list(off.parameters())
        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        gen = torch.Generator().manual_seed(cfg.seed)
        handles = component_training.hooks(linears, off)
        curve = []
        states[lr] = {}
        try:
            for step in range(1, 31):
                opt.zero_grad(set_to_none=True)
                idx = torch.randint(0, len(win["train"]), (cfg.batch,), generator=gen)
                loss, info = arm_a_loss(net, ref, win["train"].tokens[idx].to(device), cfg, win["train"].half)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                ev = eval_arm_a(net, ref, win["heldout"], o_token, device)
                rel = max(float(off.delta_w(k).norm() / W0[k].norm()) for k in uv)
                curve.append({"step": step, "loss": float(loss), "grad_norm": float(gn), "rel_norm_max": rel,
                              **{f"train_{k}": v for k, v in info.items()}, **{f"heldout_{k}": v for k, v in ev.items()}})
                if step in save_steps:
                    states[lr][step] = {k: (off.delta_w(k).cpu(), *(t.cpu() for t in off.components(k))) for k in uv}
                if step in (1, 2, 3, 5, 10, 20, 30):
                    log(f"[pullback lr {lr:g}] step {step}: loss {float(loss):.4f} held-out P(o) {ev['p_o']:.4f} "
                        f"kl_other {ev['kl_other']:.4f} rel_norm_max {rel:.4f}")
        finally:
            for h in handles:
                h.remove()
        runs[f"{lr:g}"] = {"curve": curve, "rel_norm_max_step30": curve[-1]["rel_norm_max"],
                           "converged": curve[-1]["loss"] < curve[0]["loss"]}
    return runs, states


def _faith_pullback_state(model: Any, alive: dict[str, torch.Tensor], W0: dict[str, torch.Tensor], st: dict[str, Any],
                          blocks: list[list[torch.Tensor]]) -> tuple[dict[str, Any], float]:
    """Faithfulness with target W0 + delta-W and alive columns set to the pulled-back (U_A', V_A'); returns it and
    max |Delta - Delta_0| (should be ~0: the pull-back keeps Delta fixed)."""
    from commute import vpd
    from commute.faithfulness import model_state

    comps = _components_with(model, alive, {k: (st[k][1], st[k][2]) for k in st})
    target = {k: W0[k] + st[k][0].to(W0[k].device) for k in W0}
    with model_state(model):
        wd0 = vpd.weight_deltas(model)
    with model_state(model, target=target, components=comps):
        wd = vpd.weight_deltas(model)
        f = _faith(model, blocks)
    return f, max(float((wd[k] - wd0[k]).abs().max()) for k in wd)


RATIO_KEYS = ("stochastic_kl", "pgd_kl", "ci_masked_kl", "unmasked_kl")


def stage_pullback(args: argparse.Namespace) -> None:
    """Prediction 3: the arm A rerun (Adam, checkpoints 1..30) fixes s*; then Adam on the alive (U, V) offsets at three
    learning rates, the best by held-out P(o) at step 30 is scored, with faithfulness at step min(3 s*, 30)."""
    from dataclasses import asdict

    from commute import vpd
    from commute.faithfulness import model_state
    from commute.finetune import ArmAConfig, finetune_arm_a

    require_predictions()
    model = vpd.load_component_model(args.device)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    out: dict[str, Any] = {}

    # 1. arm A rerun with early checkpoints -> s*
    cfg = ArmAConfig(o_token=o_token, steps=30, checkpoints=(1, 2, 3, 5, 10, 30))
    res = finetune_arm_a(model, win["train"], win["heldout"], cfg, args.device, log)
    (LARGE / "A_rerun").mkdir(parents=True, exist_ok=True)
    for st, d in res.deltas.items():
        torch.save(d, LARGE / "A_rerun" / f"dW_step{st}.pt")
    s_star = next((st for st in sorted(res.evals) if st <= 10 and res.evals[st]["p_o"] >= 0.9), None)
    out["rerun"] = {"config": asdict(cfg), "evals": res.evals, "s_star": s_star}
    log(f"rerun: s* = {s_star}")
    dump("pullback.json", out)
    t_score = min(3 * s_star, 30) if s_star else 30

    # 2. pull-back runs
    lrs = (1e-4, 3e-4, 1e-3)
    runs, states = _pullback_runs(model, _alive_uv_device(model, alive), win, cfg, o_token, lrs, (t_score, 30),
                                  args.device, W0)
    best = max(lrs, key=lambda lr: runs[f"{lr:g}"]["curve"][-1]["heldout_p_o"])
    curve = runs[f"{best:g}"]["curve"]
    first_09 = next((c["step"] for c in curve if c["heldout_p_o"] >= 0.9), None)
    out.update({"runs": runs, "best_lr": best, "t_score": t_score, "first_step_p_o_0.9": first_09})
    dump("pullback.json", out)
    log(f"best lr {best:g}; first step with held-out P(o) >= 0.9: {first_09}; scored step {t_score}")

    # 3. faithfulness: W0 (calibration, same harness), pull-back at t_score, the rerun at s* under do-nothing
    blocks = _faith_blocks(args.device)
    faith: dict[str, Any] = {"W0": _faith(model, blocks)}
    log(f"faith W0 {json.dumps({k: v for k, v in faith['W0'].items() if k != 'blocks'})}")
    faith["pullback"], out["delta_check_max"] = _faith_pullback_state(model, alive, W0, states[best][t_score], blocks)
    if s_star:
        with model_state(model, target={k: W0[k] + res.deltas[s_star][k].to(W0[k].device) for k in W0}):
            faith["rerun_do_nothing"] = _faith(model, blocks)
    out["faithfulness"] = faith
    out["ratios"] = {name: {m: f[m] / faith["W0"][m] for m in RATIO_KEYS} for name, f in faith.items() if name != "W0"}
    dump("pullback.json", out)
    log(f"ratios {json.dumps(out['ratios'])}; max |Delta - Delta_0| {out['delta_check_max']:.2e}")


def stage_pullback2(args: argparse.Namespace) -> None:
    """Prediction 3': the pull-back at lrs {1e-5, 2e-5, 3e-5}, selected by P(o) and off-target KL, scored at step 10."""
    from commute import vpd
    from commute.finetune import ArmAConfig

    require_predictions_amend()
    prev = json.loads((RESULTS / "pullback.json").read_text())
    rerun_evals = {int(k): v for k, v in prev["rerun"]["evals"].items()}
    kl_ref = rerun_evals[10]["kl_other"]
    model = vpd.load_component_model(args.device)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    cfg = ArmAConfig(o_token=o_token)  # round 1's batch, clip, KL weight and seed
    lrs = (1e-5, 2e-5, 3e-5)
    runs, states = _pullback_runs(model, _alive_uv_device(model, alive), win, cfg, o_token, lrs, (10, 30), args.device, W0)
    last = {lr: runs[f"{lr:g}"]["curve"][-1] for lr in lrs}
    ok = [lr for lr in lrs if last[lr]["heldout_p_o"] >= 0.9]
    sel = min(ok, key=lambda lr: last[lr]["heldout_kl_other"]) if ok else max(lrs, key=lambda lr: last[lr]["heldout_p_o"])
    curve = runs[f"{sel:g}"]["curve"]
    first_09 = next((c["step"] for c in curve if c["heldout_p_o"] >= 0.9), None)
    out: dict[str, Any] = {"runs": runs, "selected_lr": sel, "first_step_p_o_0.9": first_09,
                           "kl_other_step10": curve[9]["heldout_kl_other"], "kl_other_threshold": 2 * kl_ref,
                           "w_finetune_rel_norm_max": {s: v["rel_norm_max"] for s, v in rerun_evals.items()},
                           "pullback_rel_norm_max": {c["step"]: c["rel_norm_max"] for c in curve if c["step"] in (1, 2, 3, 5, 10, 30)}}
    dump("pullback2.json", out)
    log(f"selected lr {sel:g}; first P(o) >= 0.9 at step {first_09}; kl_other@10 {out['kl_other_step10']:.4f} "
        f"(threshold {2 * kl_ref:.4f})")
    blocks = _faith_blocks(args.device)
    faith: dict[str, Any] = {"W0": _faith(model, blocks)}
    checks = {}
    for step in (10, 30):
        faith[f"pullback_step{step}"], checks[step] = _faith_pullback_state(model, alive, W0, states[sel][step], blocks)
    out["faithfulness"] = faith
    out["delta_check_max"] = checks
    out["ratios"] = {name: {m: f[m] / faith["W0"][m] for m in RATIO_KEYS} for name, f in faith.items() if name != "W0"}
    r10 = out["ratios"]["pullback_step10"]
    out["score"] = {
        "p_o_by_step10": first_09 is not None and first_09 <= 10,
        "kl_other_ok": out["kl_other_step10"] <= 2 * kl_ref,
        "faithful_step10": r10["stochastic_kl"] <= 1.5 and r10["pgd_kl"] <= 1.5,
    }
    out["score"]["pass"] = all(out["score"].values())
    dump("pullback2.json", out)
    log(f"ratios {json.dumps(out['ratios'])}; score {out['score']}; max |Delta - Delta_0| {checks}")


def lift_components(W0: dict[str, torch.Tensor], UV: dict[str, tuple[torch.Tensor, torch.Tensor]],
                    alive: dict[str, torch.Tensor], bases: dict[str, Any], dW: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Euclidean lift of dW per matrix (float64, CPU): returns per matrix (U_A + dU, V_A + dV) and the expected
    Delta' - Delta_0 = O(dW) - dU dV^T for target weights W0 + dW."""
    from commute import split

    out = {}
    for k in W0:
        U_A, V_A = split.alive_uv(*UV[k], alive[k])
        U2, V2, dd = split.split_with_delta_change(dW[k], bases[k], U_A, V_A)
        out[k] = {"lifted": (U2, V2), "delta_change": dd}
    return out


def stage_lift_faith(args: argparse.Namespace) -> None:
    """Prediction 2': faithfulness of the Euclidean lift (and of do-nothing) at the arm A rerun checkpoints 1..10."""
    from commute import vpd
    from commute.faithfulness import model_state

    require_predictions_amend()
    W0c, UV, bases, alive = _bases_gpu("cpu")
    model = vpd.load_component_model(args.device)
    W0 = vpd.target_weights(model)
    with model_state(model):
        wd0 = vpd.weight_deltas(model)
    blocks = _faith_blocks(args.device)
    out: dict[str, Any] = {"faithfulness": {"W0": _faith(model, blocks)}, "checkpoints": {}}
    base = out["faithfulness"]["W0"]
    log(f"faith W0 {json.dumps({k: v for k, v in base.items() if k != 'blocks'})}")
    for step in (1, 2, 3, 5, 10):
        dW = torch.load(LARGE / "A_rerun" / f"dW_step{step}.pt", map_location="cpu", weights_only=False)
        L = lift_components(W0c, UV, alive, bases, dW)
        target = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
        comps = _components_with(model, alive, {k: L[k]["lifted"] for k in W0})
        rec: dict[str, Any] = {}
        with model_state(model, target=target, components=comps):
            wd = vpd.weight_deltas(model)
            rec["lift"] = _faith(model, blocks)
        # measured Delta' - Delta_0 against the closed form O(dW) - dU dV^T
        rec["delta_formula_err"] = max(float((wd[k].double().cpu() - wd0[k].double().cpu() - L[k]["delta_change"]).abs().max())
                                       for k in W0)
        dw_sq = sum(float((dW[k].double() ** 2).sum()) for k in W0)
        rec["delta_change_frac"] = sum(float((L[k]["delta_change"] ** 2).sum()) for k in W0) / dw_sq
        with model_state(model, target=target):
            rec["do_nothing"] = _faith(model, blocks)
        r = {m: {"lift": rec["lift"][m] / base[m], "do_nothing": rec["do_nothing"][m] / base[m]} for m in RATIO_KEYS}
        rec["ratios"] = r
        rn, rl = r["unmasked_kl"]["do_nothing"], r["unmasked_kl"]["lift"]
        rec["scorable"] = rn - 1 >= 0.2
        rec["gap_recovered"] = (rn - rl) / (rn - 1) if rn != 1 else None
        rec["pass"] = (r["stochastic_kl"]["lift"] <= 1.5 and r["pgd_kl"]["lift"] <= 1.5
                       and rec["gap_recovered"] is not None and rec["gap_recovered"] >= 0.5)
        out["checkpoints"][step] = rec
        log(f"step {step}: lift stoch {r['stochastic_kl']['lift']:.3f} pgd {r['pgd_kl']['lift']:.3f} ci {r['ci_masked_kl']['lift']:.3f} "
            f"unmasked {rl:.3f} | do-nothing stoch {r['stochastic_kl']['do_nothing']:.3f} pgd {r['pgd_kl']['do_nothing']:.3f} "
            f"unmasked {rn:.3f} | gap_recovered {rec['gap_recovered']} scorable {rec['scorable']} pass {rec['pass']} | "
            f"Delta change frac {rec['delta_change_frac']:.3f}, formula err {rec['delta_formula_err']:.2e}")
        dump("lift_faith.json", out)
    sc = [s for s, c in out["checkpoints"].items() if c["scorable"]]
    out["score"] = {"scorable_checkpoints": sc, "pass": bool(sc) and all(out["checkpoints"][s]["pass"] for s in sc)}
    dump("lift_faith.json", out)
    log(f"P2' score {out['score']}")


def stage_pullback_early(args: argparse.Namespace) -> None:
    """Reported, not scored: faithfulness of the selected pull-back (lr 1e-5, as in pullback2) at steps 1, 2, 3, 5 and
    10, so it overlaps the lift's checkpoints. Step 10 is recomputed to check reproduction of pullback2.json."""
    from commute import vpd
    from commute.finetune import ArmAConfig

    require_predictions_amend()
    prev = json.loads((RESULTS / "pullback2.json").read_text())
    lr = prev["selected_lr"]
    model = vpd.load_component_model(args.device)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    cfg = ArmAConfig(o_token=o_token)
    steps = (1, 2, 3, 5, 10)
    runs, states = _pullback_runs(model, _alive_uv_device(model, alive), win, cfg, o_token, (lr,), steps, args.device, W0)
    curve = runs[f"{lr:g}"]["curve"]
    blocks = _faith_blocks(args.device)
    faith: dict[str, Any] = {"W0": _faith(model, blocks)}
    checks = {}
    for step in steps:
        faith[f"pullback_step{step}"], checks[step] = _faith_pullback_state(model, alive, W0, states[lr][step], blocks)
        log(f"step {step}: {json.dumps({m: faith[f'pullback_step{step}'][m] / faith['W0'][m] for m in RATIO_KEYS})}")
    ratios = {name: {m: f[m] / faith["W0"][m] for m in RATIO_KEYS} for name, f in faith.items() if name != "W0"}
    out = {"lr": lr, "steps": list(steps), "curve": curve, "faithfulness": faith, "ratios": ratios,
           "delta_check_max": checks,
           "rel_norm_max": {c["step"]: c["rel_norm_max"] for c in curve if c["step"] in steps},
           "reproduction_step10": {m: ratios["pullback_step10"][m] - prev["ratios"]["pullback_step10"][m] for m in RATIO_KEYS},
           "note": "reported, not scored; same recipe and seed as pullback2's selected run"}
    dump("pullback2_early.json", out)
    log(f"reproduction diff at step 10: {json.dumps(out['reproduction_step10'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--predictions-amend", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["obstruction", "lora", "sgd", "weighted", "projected", "pullback",
                                                 "lift_faith", "pullback2", "pullback_early"])
    ap.add_argument("--sgd-lrs", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 8])
    ap.add_argument("paths", nargs="*", help="delta-W checkpoints (dict of 24 tensors)")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    if args.predictions_amend:
        file_predictions_amend()
        return
    if args.stage is None:
        ap.error("give a stage or --predictions")
    {"obstruction": stage_obstruction, "lora": stage_lora, "sgd": stage_sgd, "weighted": stage_weighted,
     "projected": stage_projected, "pullback": stage_pullback, "lift_faith": stage_lift_faith,
     "pullback2": stage_pullback2, "pullback_early": stage_pullback_early}[args.stage](args)


if __name__ == "__main__":
    main()
