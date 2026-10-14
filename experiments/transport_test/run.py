"""The transport test: does a small fine-tune land on existing VPD subcomponents?

Stages (each writes JSON into results/; large tensors into results/large/):
  --predictions         file results/predictions.json (refuses to overwrite)
  fetch                 download the two W&B runs (needs WANDB_API_KEY)
  calibrate             alive set, faithfulness of the released decomposition on W (our baseline)
  windows               scan Pile train rows for emoticon positions, build windows
  finetune A|B          fine-tune, save delta-W at the checkpoints
  transport A|B         the transport ladder on every checkpoint, plus arm N
  faithfulness A|B      faithfulness of the transported decompositions at W + delta-W
  all                   everything in order

Usage (on the GPU machine, from the repo root):  python experiments/transport_test/run.py all
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
ARMS = ("A", "B")

PREDICTIONS: dict[str, Any] = {
    "about": (
        "transport_test: can a small fine-tune delta-W of Goodfire's 4-layer 67M Pile model be written as a small "
        "change to a few existing subcomponents of the released VPD decomposition (s-55ea3f9b), with the causal-"
        "importance function Gamma and the residual Delta left untouched, so that the transported decomposition "
        "still explains the fine-tuned model? Plan and design check: reports/2026-10-14_plan_transport-test.md "
        "(confirmed by the user 2026-10-14). Filed before any transport measurement was run. A learning-rate pilot "
        "that looked only at held-out P(o) and ||dW||/||W|| at the checkpoints is allowed before the main run and is "
        "recorded in setup.pilot."
    ),
    "setup": {
        "model": "goodfire/spd/runs/t-9d2b8f02 (4 layers, 67M, Pile), decomposition goodfire/spd/runs/s-55ea3f9b, "
                 "loaded with param-decomp at commit 74146b5, bf16 autocast on one 24 GB GPU",
        "decomposed_matrices": "the 24 matrices h.{0..3}.{mlp.c_fc, mlp.down_proj, attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj}; "
                               "embeddings, unembedding and RMSNorm weights are frozen in every fine-tune",
        "alive": "mean g over 256 Pile val rows (rows 2048..2303, 131k positions) above 1e-6, the paper's rule; "
                 "the per-layer counts are reported against the paper's 3709 / 848 / 1943 / 3472",
        "faithfulness_batches": "Pile val rows 0..63 in batches of 8; stochastic KL with 2 draws per batch, PGD 20 steps "
                                "of 0.1 with random init and one adversary per batch (Delta included in both); the same rows "
                                "and seed for every variant",
        "emoticon_windows": "positions in Pile train rows 0..4095 at which g(h.2.mlp.down_proj:2359) > 0.5, with the 20 "
                            "tokens either side (41 tokens); if fewer than 200 windows, the threshold drops to 0.1 and this is "
                            "recorded. Held-out = windows whose source row index is divisible by 5",
        "arm_A": "full-weight Adam fine-tune of the 24 matrices on the training windows: cross-entropy on token `o` at "
                 "index 20 plus 1.0 * KL(reference || fine-tuned) averaged over the other 40 positions; batch 32 windows, "
                 "grad clip 1.0, checkpoints at steps 10, 30, 100, 300. Learning rate: 3e-5 first; if held-out P(o) has "
                 "not reached 0.5 by step 300, the run is kept as results/finetune_A_lr3e-5.json and the arm is rerun at "
                 "1e-4, which is then the scored run (the rule is fixed here so that no choice is made after seeing "
                 "transport results)",
        "arm_B": "full-weight Adam fine-tune on fresh Pile train rows 8192..13 000 (16 rows of 512 per step, each row "
                 "seen once), lr 1e-5, grad clip 1.0, checkpoints at 10, 30, 100, 300; held-out = val rows 64..127",
        "arm_N": "Gaussian delta-W with the per-matrix Frobenius norm of arm A's scored checkpoint, 3 seeds",
        "transports": {
            "T0_frobenius": "per matrix, least squares of delta-W on {U_c V_c^T : c alive}; reports the captured fraction "
                            "per matrix and pooled (weighted by ||dW_m||^2), and alpha",
            "T0_functional": "alpha on all alive subcomponents, Adam lr 1e-2 for 150 steps on the arm's training data "
                             "(windows for A, train rows for B), initialised at the Frobenius alpha",
            "T1_write": "delta-U_c free (V_c fixed) on the set S of the k subcomponents with the largest Frobenius "
                        "contribution |alpha_c| ||U_c|| ||V_c||, k in {1, 3, 10, 30, 100}; Adam lr 1e-2, 150 steps, "
                        "zero init, on the training data",
            "T1_read": "the mirror image (delta-V_c free, U_c fixed) at the same k, reported, not scored",
            "null_random": "T1_write with S replaced by k alive subcomponents drawn uniformly from the same matrices "
                           "(matched counts per matrix), 5 seeds; the mean over seeds is the null",
        },
        "pilot": "none: the learning-rate rule above replaces a pilot",
        "scored_checkpoint": {
            "A": "the first checkpoint at which held-out mean P(o) at index 20 is at least 0.5; if none, step 300",
            "B": "step 100 (step 300 reported)",
        },
        "capture": "capture = 1 - KL(f_ft || f_T) / KL(f_ft || f_W0), each KL averaged over the stated positions of the "
                   "held-out data; f_ft is the fine-tuned model, f_T the transported one, f_W0 the original. Arm A: at "
                   "index 20 of the held-out windows (primary) and at the other 40 positions (off-target, reported). "
                   "Arm B: all positions of the held-out rows. Also reported for A: mean P(o) under f_T",
        "faithfulness_of_transport": "target weights W0 + delta-W, components transported, Delta recomputed as W_target - "
                                     "sum of components (so it absorbs what the transport leaves), Gamma frozen; the four "
                                     "metrics above. Variants: null transport (components unchanged) at every checkpoint; "
                                     "T0_functional and T1_write k=10 at the scored checkpoint",
    },
    "predictions": [
        {
            "id": "1a_emoticon_capture",
            "scored": True,
            "claim": "In the most favourable case (a one-subcomponent expression of the behaviour is known to exist), "
                     "most of the fine-tune's effect is a write-vector change to ten or fewer existing subcomponents.",
            "measure": "arm A, scored checkpoint, T1_write k=10: capture at index 20 of the held-out windows, and the "
                       "same for null_random k=10 (mean over 5 seeds)",
            "pass": "capture_top10 >= 0.5 AND capture_top10 - capture_random10 >= 0.3",
            "null": "capture_top10 - capture_random10 < 0.1 (the decomposition carries no information about where the "
                    "update went); between 0.1 and 0.3 is a partial result, not a pass",
            "prediction": "pass",
            "conditioning": "scorable only if the fine-tune's effect is real: KL(f_ft || f_W0) at index 20 >= 0.1 nats "
                            "and held-out P(o) rose by at least 0.2 over W0",
        },
        {
            "id": "1b_emoticon_location",
            "scored": False,
            "claim": "The update lands on the paper's emoticon subcomponents.",
            "measure": "arm A, scored checkpoint, T0_frobenius: rank of h.2.mlp.down_proj:2359 and its five siblings "
                       "(1672, 2623, 3290, 3327, 3382) among the alive subcomponents of h.2.mlp.down_proj by Frobenius "
                       "contribution, and their share of the matrix's fitted norm",
            "prediction": "at least one of the six is in the top 5 of h.2.mlp.down_proj",
            "note": "A fine-tune may route the behaviour elsewhere and still commute cleanly; this says where the update "
                    "went, not whether the claim holds",
        },
        {
            "id": "2_pretraining_arm",
            "scored": True,
            "claim": "A pure 'strengthen' update lies in the span of the alive subcomponents far more than a random "
                     "direction does, and scale-only reweighting recovers most of its functional effect.",
            "measure": "arm B at step 100: (i) pooled T0_frobenius fraction of delta-W on the alive span, against the "
                       "same for arm N (Gaussian, norm-matched, mean of 3 seeds); (ii) T0_functional capture on the "
                       "held-out rows",
            "pass": "(i) fraction_B >= 10 * fraction_N AND (ii) capture >= 0.5",
            "prediction": "pass on (i); (ii) is the uncertain half, predicted to pass",
            "conditioning": "(ii) scorable only if KL(f_ft || f_W0) on held-out rows >= 0.005 nats; otherwise the arm "
                            "is reported as ill-conditioned and rerun at a larger learning rate (dated report)",
        },
        {
            "id": "3_not_brittle",
            "scored": True,
            "claim": "The frozen decomposition, transported, is still faithful to the fine-tuned model: the update does "
                     "not break the decomposition.",
            "measure": "stochastic_kl and pgd_kl of the T1_write k=10 (arm A) and T0_functional (arm B) transported "
                       "decompositions at W0 + delta-W at the scored checkpoints, against our own values for the released "
                       "decomposition on W0 with the same rows and seed (calibration); the paper's 0.238 and 0.927 are "
                       "reported beside them",
            "pass": "both ratios (transported / calibration) <= 2.0 for both arms",
            "prediction": "pass",
            "fail_reading": "a ratio above 2 at a checkpoint whose max per-matrix ||dW||/||W|| is below 0.05 is the "
                            "brittleness branch; also report the null transport's ratios, which separate 'the update "
                            "itself breaks faithfulness' from 'the transport does'",
        },
        {
            "id": "4_calibration",
            "scored": False,
            "measure": "stochastic_kl and pgd_kl of the released decomposition on W0 with our rows, batches and seed",
            "prediction": "within a factor of 1.5 of the paper's 0.238 and 0.927; if not, the harness is wrong and nothing "
                          "else is scored until it is fixed",
        },
    ],
    "reported_not_scored": [
        "per-matrix ||dW||/||W0|| at every checkpoint, both arms",
        "capture against k for T1_write and T1_read (k = 1, 3, 10, 30, 100) and the null, every checkpoint",
        "T0 Frobenius fraction on the full dictionary (all 38 912) as well as the alive set",
        "faithfulness of the null transport and of the fine-tuned model under the untouched decomposition at every checkpoint",
        "arm A off-target capture (the other 40 positions) and off-target KL of f_T to f_W0",
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


def load(name: str) -> Any:
    return json.loads((RESULTS / name).read_text())


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump("predictions.json", PREDICTIONS)
    log(f"filed {PRED}")


def require_predictions() -> dict[str, Any]:
    if not PRED.is_file():
        sys.exit("file predictions first: run.py --predictions")
    return json.loads(PRED.read_text())


# ----------------------------------------------------------------------------- stages


def stage_fetch(_: argparse.Namespace) -> None:
    from commute import vpd

    vpd.fetch(log)


def _model(device: str) -> Any:
    from commute import vpd

    t0 = time.time()
    model = vpd.load_component_model(device)
    log(f"model loaded in {time.time() - t0:.0f} s")
    return model


def _faith_batches(device: str) -> list[torch.Tensor]:
    from commute.data import batches, pile_rows

    return [b.to(device) for b in batches(pile_rows("val", 0, 64, log), 8)]


def stage_calibrate(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.data import batches, pile_rows
    from commute.faithfulness import faithfulness

    model = _model(args.device)
    rows = pile_rows("val", 2048, 256, log)
    t0 = time.time()
    mean_g = vpd.mean_ci(model, [b.to(args.device) for b in batches(rows, 16)])
    alive = vpd.alive_masks(mean_g)
    torch.save({"mean_g": mean_g, "alive": alive}, LARGE / "alive.pt")
    per_layer: dict[str, int] = {}
    for k, m in alive.items():
        per_layer[f"layer_{vpd.layer_of(k)}"] = per_layer.get(f"layer_{vpd.layer_of(k)}", 0) + int(m.sum())
    log(f"alive per layer {per_layer} total {sum(per_layer.values())} ({time.time() - t0:.0f} s)")
    t0 = time.time()
    faith = faithfulness(model, _faith_batches(args.device), log=log)
    out = {
        "alive_per_layer": per_layer,
        "alive_per_module": {k: int(m.sum()) for k, m in alive.items()},
        "alive_total": sum(per_layer.values()),
        "faithfulness_W0": faith,
        "paper": vpd.PAPER_FAITHFULNESS,
        "ratio_to_paper": {k: faith[k] / v for k, v in vpd.PAPER_FAITHFULNESS.items()},
        "faithfulness_seconds": time.time() - t0,
    }
    dump("calibration.json", out)
    log(f"calibration: {json.dumps(out['faithfulness_W0'])}")


def stage_windows(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.data import emoticon_windows, pile_rows, split_windows

    model = _model(args.device)
    tok = vpd.tokenizer()
    o_ids = tok.encode("o")
    assert len(o_ids) == 1, o_ids
    rows = pile_rows("train", 0, args.n_scan_rows, log)
    t0 = time.time()
    w = emoticon_windows(model, rows, vpd.EMOTICON_KEY, threshold=0.5)
    threshold = 0.5
    if len(w) < 200:
        log(f"only {len(w)} windows at g > 0.5; dropping the threshold to 0.1 (declared rule)")
        threshold = 0.1
        w = emoticon_windows(model, rows, vpd.EMOTICON_KEY, threshold=threshold)
    train, held = split_windows(w)
    torch.save({"train": train, "heldout": held, "threshold": threshold}, LARGE / "windows.pt")
    nxt = w.tokens[:, w.half + 1].tolist()
    cur = w.tokens[:, w.half].tolist()
    from collections import Counter

    top_cur = [(tok.decode([t]), c) for t, c in Counter(cur).most_common(12)]
    top_nxt = [(tok.decode([t]), c) for t, c in Counter(nxt).most_common(12)]
    out = {
        "n_scan_rows": args.n_scan_rows, "threshold": threshold, "n_windows": len(w), "n_train": len(train),
        "n_heldout": len(held), "o_token_id": o_ids[0], "seconds": time.time() - t0,
        "top_tokens_at_position": top_cur, "top_next_tokens": top_nxt,
        "g_quantiles": {q: float(torch.quantile(w.g, q)) for q in (0.1, 0.5, 0.9)} if len(w) else {},
        "examples": [tok.decode(w.tokens[i].tolist()) for i in range(min(8, len(w)))],
    }
    dump("windows.json", out)
    log(f"windows: {len(w)} (train {len(train)}, held-out {len(held)}); at position {top_cur[:5]}; next {top_nxt[:5]}")


def _arm_cfg(arm: str, args: argparse.Namespace, o_token: int | None = None) -> Any:
    from commute.finetune import ArmAConfig, ArmBConfig

    if arm == "A":
        assert o_token is not None
        return ArmAConfig(o_token=o_token, lr=args.lr_a, steps=args.steps, checkpoints=tuple(c for c in (10, 30, 100, 300) if c <= args.steps))
    return ArmBConfig(lr=args.lr_b, steps=args.steps, checkpoints=tuple(c for c in (10, 30, 100, 300) if c <= args.steps))


def stage_finetune(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    from commute.data import pile_rows
    from commute.finetune import finetune_arm_a, finetune_arm_b, relative_norms
    from commute.vpd import target_weights

    arm = args.arm
    model = _model(args.device)
    W0 = {k: w.cpu() for k, w in target_weights(model).items()}
    if arm == "A":
        win = torch.load(LARGE / "windows.pt", weights_only=False)
        cfg = _arm_cfg("A", args, o_token=load("windows.json")["o_token_id"])
        res = finetune_arm_a(model, win["train"], win["heldout"], cfg, args.device, log)
    else:
        cfg = _arm_cfg("B", args)
        train_rows = pile_rows("train", 8192, cfg.steps * cfg.batch, log)
        held = pile_rows("val", 64, 64, log)
        res = finetune_arm_b(model, train_rows, held, cfg, args.device, log)
    tag = f"{arm}{args.tag}"
    (LARGE / tag).mkdir(parents=True, exist_ok=True)
    for step, d in res.deltas.items():
        torch.save(d, LARGE / tag / f"dW_step{step}.pt")
    out = {
        "arm": arm, "config": asdict(cfg), "log": res.log, "evals": res.evals,
        "relative_norms": {step: relative_norms(d, W0) for step, d in res.deltas.items()},
    }
    dump(f"finetune_{tag}.json", out)
    log(f"finetune {tag}: evals {json.dumps(res.evals)}")


def _scored_checkpoint(arm: str, ft: dict[str, Any]) -> int:
    evals = {int(k): v for k, v in ft["evals"].items()}
    if arm == "A":
        for step in sorted(evals):
            if evals[step]["p_o"] >= 0.5:
                return step
        return max(evals)
    return 100 if 100 in evals else max(evals)


def stage_transport(args: argparse.Namespace) -> None:
    from commute import transport as T
    from commute import vpd
    from commute.data import batches, pile_rows
    from commute.finetune import random_direction

    arm = args.arm
    tag = f"{arm}{args.tag}"
    model = _model(args.device)
    dev = args.device
    alive = torch.load(LARGE / "alive.pt", weights_only=False)["alive"]
    alive_idx = {k: torch.nonzero(m, as_tuple=True)[0] for k, m in alive.items()}
    all_idx = {k: torch.arange(model.module_to_c[k]) for k in alive}
    ft = load(f"finetune_{tag}.json")
    scored = _scored_checkpoint(arm, ft)
    steps = sorted(int(k) for k in ft["evals"])
    if args.only_scored:
        steps = [scored]

    # data for the fits and for the capture
    if arm == "A":
        win = torch.load(LARGE / "windows.pt", weights_only=False)
        o_token = load("windows.json")["o_token_id"]
        train_t, held_t = win["train"].tokens, win["heldout"].tokens
        half = win["train"].half
        pos_primary = torch.zeros(train_t.shape[1], dtype=torch.bool, device=dev)
        pos_primary[half] = True
        fit_positions = None  # fit on every position of the window
        eval_sets = {"primary": pos_primary, "off_target": ~pos_primary}
        gen = torch.Generator().manual_seed(1)

        def next_batch() -> torch.Tensor:
            return train_t[torch.randint(0, train_t.shape[0], (32,), generator=gen)].to(dev)

        eval_batches = [b.to(dev) for b in batches(held_t, 64)]
    else:
        train_rows = pile_rows("train", 8192, 4800, log)[:512]
        held = pile_rows("val", 64, 64, log)
        fit_positions = None
        eval_sets = {"primary": None}
        gen = torch.Generator().manual_seed(1)

        def next_batch() -> torch.Tensor:
            return train_rows[torch.randint(0, train_rows.shape[0], (4,), generator=gen)].to(dev)

        eval_batches = [b.to(dev) for b in batches(held, 8)]
        o_token = None

    def captures(ref_extra: T.Extra, extra: T.Extra) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, pos in eval_sets.items():
            c = T.capture(model, ref_extra, extra, eval_batches, pos)
            out[name] = {"kl_ft_vs_w0": c.kl_ft_vs_w0, "kl_ft_vs_t": c.kl_ft_vs_t, "capture": c.capture}
        if arm == "A":
            with torch.no_grad():
                ps = []
                for b in eval_batches:
                    lg = T.forward(model, b, extra)
                    ps.append(torch.softmax(lg[:, half], -1)[:, o_token])
                out["p_o_transported"] = float(torch.cat(ps).mean())
        return out

    results: dict[str, Any] = {"arm": arm, "scored_checkpoint": scored, "steps": {}}
    for step in steps:
        t_step = time.time()
        dW = torch.load(LARGE / tag / f"dW_step{step}.pt", weights_only=True)
        ref_extra = T.delta_extra(dW, dev)
        r: dict[str, Any] = {"relative_norms": ft["relative_norms"][str(step)]}
        # T0 Frobenius on the alive set and on the full dictionary
        fits = T.frobenius_fit_all(model, dW, alive_idx, dev)
        fits_all = T.frobenius_fit_all(model, dW, all_idx, dev)
        r["T0_frobenius"] = {
            "fraction_alive_pooled": T.pooled_fraction(fits, dW),
            "fraction_all_pooled": T.pooled_fraction(fits_all, dW),
            "fraction_alive_per_matrix": {k: f.fraction for k, f in fits.items()},
            "dW_norm2_per_matrix": {k: float(dW[k].norm() ** 2) for k in dW},
        }
        if arm == "A":
            name = "h.2.mlp.down_proj"
            f = fits[name]
            order = torch.argsort(f.contribution, descending=True)
            ranked = [int(f.idx[i]) for i in order]
            fam = [int(vpd.parse_key(k)[1]) for k in (vpd.EMOTICON_KEY, *vpd.EMOTICON_SIBLINGS)]
            total = float(f.contribution.sum())
            r["1b_location"] = {
                "ranks": {c: (ranked.index(c) + 1 if c in ranked else None) for c in fam},
                "share_of_contribution": {c: (float(f.contribution[(f.idx == c).nonzero()[0, 0]]) / total if c in ranked else 0.0) for c in fam},
                "top10": [(c, float(f.contribution[(f.idx == c).nonzero()[0, 0]])) for c in ranked[:10]],
                "alpha_top10": [(c, float(f.alpha[(f.idx == c).nonzero()[0, 0]])) for c in ranked[:10]],
            }
        # arm N: random directions with matched norms (Frobenius only)
        if step == scored:
            fr = []
            for seed in range(3):
                dN = random_direction(dW, seed)
                fr.append(T.pooled_fraction(T.frobenius_fit_all(model, dN, alive_idx, dev), dN))
            r["arm_N_fraction_alive_pooled"] = fr
        # T0 functional capture (scale only, all alive), warm-started at the Frobenius alpha
        alpha = {k: fits[k].alpha.clone().to(dev) for k in fits}
        T.functional_fit(model, ref_extra, lambda: T.scale_extra(model, alpha, {k: alive_idx[k].to(dev) for k in alpha}),
                         list(alpha.values()), next_batch, steps=args.fit_steps, lr=1e-2, log=log, positions=fit_positions, tag=f"T0 s{step}")
        r["T0_functional"] = captures(ref_extra, T.scale_extra(model, alpha, {k: alive_idx[k].to(dev) for k in alpha}))
        r["T0_frobenius_capture"] = captures(ref_extra, T.scale_extra(model, {k: fits[k].alpha.to(dev) for k in fits}, {k: alive_idx[k].to(dev) for k in fits}))
        torch.save({k: a.cpu() for k, a in alpha.items()}, LARGE / tag / f"alpha_T0_step{step}.pt")
        # T1: k-sparse write (and read) edits, top-k by contribution vs random matched sets
        r["T1"] = {}
        ks = [1, 3, 10, 30, 100] if step == scored else [10]
        for k in ks:
            S = T.top_k_by_contribution(fits, k)
            r["T1"][k] = {"S": {n: v.tolist() for n, v in S.items()}}
            for kind in ("write", "read"):
                if kind == "read" and step != scored:
                    continue
                save = (LARGE / tag / f"T1write_k10_s{step}.pt") if (kind == "write" and k == 10) else None
                r["T1"][k][kind] = _fit_sparse(model, T, S, kind, ref_extra, next_batch, args.fit_steps, fit_positions, log, f"T1{kind} k{k} s{step}", captures, save)
            if k == 10 or step == scored:
                r["T1"][k]["random_write"] = []
                for seed in range(5 if step == scored else 2):
                    R = T.random_matched(S, alive, seed)
                    r["T1"][k]["random_write"].append(_fit_sparse(model, T, R, "write", ref_extra, next_batch, args.fit_steps, fit_positions, log, f"T1rand k{k} s{step} seed{seed}", captures, None))
            if k == 10:
                torch.save({"S": S}, LARGE / tag / f"S10_step{step}.pt")
        r["seconds"] = time.time() - t_step
        results["steps"][step] = r
        dump(f"transport_{tag}.json", results)
        log(f"transport {tag} step {step}: T0 frob {r['T0_frobenius']['fraction_alive_pooled']:.4f}, "
            f"T0 func capture {r['T0_functional']['primary']['capture']:.3f}, "
            f"T1 write k10 capture {r['T1'][10]['write']['primary']['capture']:.3f} "
            f"vs random {[round(x['primary']['capture'], 3) for x in r['T1'][10].get('random_write', [])]} ({r['seconds']:.0f} s)")


def _fit_sparse(model: Any, T: Any, S: dict[str, torch.Tensor], kind: str, ref_extra: Any, next_batch: Any, steps: int,
                positions: Any, log: Any, tag: str, captures: Any, save: Path | None) -> dict[str, Any]:
    dev = next(model.parameters()).device
    S = {k: v.to(dev) for k, v in S.items()}
    params: dict[str, torch.Tensor] = {}
    for name, idx in S.items():
        U, V = T.uv(model, name)
        if kind == "write":
            params[name] = torch.zeros(idx.numel(), U.shape[1], device=dev)
        else:
            params[name] = torch.zeros(V.shape[0], idx.numel(), device=dev)
    make = (lambda: T.write_edit_extra(model, params, S)) if kind == "write" else (lambda: T.read_edit_extra(model, params, S))
    losses = T.functional_fit(model, ref_extra, make, list(params.values()), next_batch, steps=steps, lr=1e-2, log=log, positions=positions, tag=tag)
    out = captures(ref_extra, make())
    out["fit_loss_first_last"] = [losses[0], losses[-1]]
    out["edit_norms"] = {name: float(p.norm()) for name, p in params.items()}
    if save is not None:
        torch.save({"S": {k: v.cpu() for k, v in S.items()}, "dU": {name: p.detach().cpu() for name, p in params.items()}}, save)
    return out


def stage_faithfulness(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.faithfulness import faithfulness, model_state, transported_components

    arm = args.arm
    tag = f"{arm}{args.tag}"
    model = _model(args.device)
    dev = args.device
    ft = load(f"finetune_{tag}.json")
    tr = load(f"transport_{tag}.json")
    scored = tr["scored_checkpoint"]
    steps = [scored] if args.only_scored else sorted(int(s) for s in tr["steps"])
    W0 = vpd.target_weights(model)
    alive = torch.load(LARGE / "alive.pt", weights_only=False)["alive"]
    alive_idx = {k: torch.nonzero(m, as_tuple=True)[0].to(dev) for k, m in alive.items()}
    fb = _faith_batches(dev)
    cal = load("calibration.json")["faithfulness_W0"]
    results: dict[str, Any] = {"arm": arm, "scored_checkpoint": scored, "calibration_W0": cal, "steps": {}}
    for step in steps:
        dW = torch.load(LARGE / tag / f"dW_step{step}.pt", weights_only=True)
        W1 = {k: (W0[k] + dW[k].to(dev)) for k in W0}
        r: dict[str, Any] = {}
        with model_state(model, target=W1):
            r["null_transport"] = faithfulness(model, fb, log=log)
        if step != scored:
            results["steps"][step] = {k: {**f, "ratio_to_calibration": {m: f[m] / cal[m] for m in ("stochastic_kl", "pgd_kl")}} for k, f in r.items()}
            dump(f"faithfulness_{tag}.json", results)
            continue
        alpha = torch.load(LARGE / tag / f"alpha_T0_step{step}.pt", weights_only=True)
        comps = transported_components(model, scale={k: (alive_idx[k], alpha[k].to(dev)) for k in alpha})
        with model_state(model, target=W1, components=comps):
            r["T0_functional"] = faithfulness(model, fb, log=log)
        p = LARGE / tag / f"T1write_k10_s{step}.pt"
        if p.is_file():
            e = torch.load(p, weights_only=True)
            comps = transported_components(model, dU={k: (e["S"][k].to(dev), e["dU"][k].to(dev)) for k in e["S"]})
            with model_state(model, target=W1, components=comps):
                r["T1_write_k10"] = faithfulness(model, fb, log=log)
        for name, f in list(r.items()):
            r[name] = {**f, "ratio_to_calibration": {m: f[m] / cal[m] for m in ("stochastic_kl", "pgd_kl")}}
        results["steps"][step] = r
        dump(f"faithfulness_{tag}.json", results)
        log(f"faithfulness {tag} step {step}: " + json.dumps({k: v["ratio_to_calibration"] for k, v in r.items()}))


def stage_all(args: argparse.Namespace) -> None:
    stage_fetch(args)
    if not (RESULTS / "calibration.json").is_file():
        stage_calibrate(args)
    if not (RESULTS / "windows.json").is_file():
        stage_windows(args)
    for arm in ARMS:
        args.arm = arm
        if not (RESULTS / f"finetune_{arm}{args.tag}.json").is_file():
            stage_finetune(args)
            if arm == "A" and args.tag == "":
                ft = load("finetune_A.json")
                if max(v["p_o"] for v in ft["evals"].values()) < 0.5 and args.lr_a < 1e-4:
                    log("arm A: held-out P(o) below 0.5 at every checkpoint at lr 3e-5; keeping it as _lr3e-5 and rerunning at 1e-4 (declared rule)")
                    (RESULTS / "finetune_A.json").rename(RESULTS / "finetune_A_lr3e-5.json")
                    (LARGE / "A").rename(LARGE / "A_lr3e-5")
                    args.lr_a = 1e-4
                    stage_finetune(args)
        stage_transport(args)
        stage_faithfulness(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["fetch", "calibrate", "windows", "finetune", "transport", "faithfulness", "all"])
    ap.add_argument("arm", nargs="?", choices=list(ARMS))
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="suffix for pilot runs, e.g. _pilot_lr3e-5")
    ap.add_argument("--lr-a", type=float, default=3e-5)
    ap.add_argument("--lr-b", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fit-steps", type=int, default=150)
    ap.add_argument("--n-scan-rows", type=int, default=4096)
    ap.add_argument("--only-scored", action="store_true", help="transport/faithfulness on the scored checkpoint only")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    if args.stage is None:
        ap.error("stage required")
    if args.stage not in ("fetch",):
        require_predictions()
    if args.stage in ("finetune", "transport", "faithfulness") and args.arm is None:
        ap.error("arm required")
    LARGE.mkdir(parents=True, exist_ok=True)
    {"fetch": stage_fetch, "calibrate": stage_calibrate, "windows": stage_windows, "finetune": stage_finetune,
     "transport": stage_transport, "faithfulness": stage_faithfulness, "all": stage_all}[args.stage](args)


if __name__ == "__main__":
    main()
