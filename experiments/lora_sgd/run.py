"""lora_sgd (E27): the rank-8 LoRA on all 24 matrices trained with plain SGD, through E14's table, beside the Adam LoRA.
Plan: reports/2026-10-22_plan_commute-second-line.md, "Item C" and the E27 agent note (parents E9, E14, E4's rate rule).

Order, each stage writing results/lora_sgd.json as it finishes:
  1. W0 under the Pile harness, adversary seeds 0, 1, 2 (faith_score.w0_check); stop if it fails.
  2. The Adam LoRA regenerated with E9's recipe (rank 8, all 24 matrices, Adam 1e-3, seed 0, 1000 steps): the series at
     the nine checkpoints, the do-nothing and least-squares states at steps 10 and 1000, gate G1 against E9/E14; stop
     if it fails.
  3. The SGD rate grid: plain SGD on the same LoRA at each rate to 1000 steps, held-out P(target) and KL elsewhere at the
     checkpoints; E4's rule at step 1000 picks the rate (lora_readouts.select_rate, fallbacks flagged).
  4. The selected SGD LoRA: the series (four blocks with their null, rescaling share with its random null, |cos| of A with
     its init, rel_norm_max) at the nine checkpoints; do-nothing, least-squares and scaled adjoint states at 10, 100, 1000.
The LoRA factors of both scored LoRAs go to results/large/ (gitignored).

Usage (repo root):  run.py --predictions  (files results/predictions.json once)  |  run.py all
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}, experiments/transport_test/results/windows.json.
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
OUT = RESULTS / "lora_sgd.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
WINDOWS_JSON = HERE.parent / "transport_test" / "results" / "windows.json"


@dataclass(frozen=True)
class Config:
    ckpts: tuple[int, ...] = (1, 2, 3, 5, 10, 30, 100, 300, 1000)  # E9's checkpoints: the series' rows
    sgd_steps: tuple[int, ...] = (10, 100, 1000)  # the SGD LoRA's states
    adam_steps: tuple[int, ...] = (10, 1000)  # the Adam LoRA's states: G1 and P4 read only these
    rank: int = 8
    adam_lr: float = 1e-3  # E9
    sgd_lrs: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)  # the agent note's grid
    seed: int = 0
    tol: float = 1e-6
    null_draws: int = 2  # as E14
    pile_rows: int = 256
    pile_pgd_steps: int = 20
    pile_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        assert self.ckpts == tuple(sorted(self.ckpts)) and {10, 1000} <= set(self.ckpts)
        assert set(self.sgd_steps) <= set(self.ckpts) and set(self.adam_steps) <= set(self.sgd_steps)
        assert self.sgd_lrs == tuple(sorted(self.sgd_lrs)) and self.pile_seeds[0] == 0


C = Config()
P_MIN = 0.9  # E4's rule
E9_KL_1000 = 0.004760152264817121  # E9's Adam LoRA, KL elsewhere at step 1000 (lora_plateau.json)
KL_MAX = 2 * E9_KL_1000
L40_FLOOR = 0.209  # the floor E9's references were read under
# G1: E9's readings (lora_plateau.json pile_steps, PGD ratio to W0) and E14's outside-both / null (lora_split.json).
G1_REF = {"nothing": {10: 1.1726, 1000: 1.0419}, "least_squares": {10: 1.2551, 1000: 1.0720}}
E14_OUT_OUT_10, E14_DELTA_1000, G1_P_1000 = 0.979, 0.5417, 0.99
P1_MIN, P2_MIN, P4_BAND, C2_BAND, C3_MAX = 0.7, 0.97, 0.1, (0.85, 1.15), 0.01


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    raise TypeError(type(o))


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default))


VERDICT = ("faith_score.verdict on the PGD ratio to the same-job W0 (adversary seed 0): pass <= 1.5 / (1 + f), fail >= "
           "1.5 (1 + f), inconclusive between; f = max/min - 1 of W0's PGD KL over adversary seeds 0, 1, 2 in this job "
           "(L40 0.21 -> 1.24 / 1.81; H100 about 0.075 -> 1.40 / 1.61)")
E14_LORA_OUT_OUT = {1: 0.835, 2: 0.702, 3: 0.664, 5: 0.762, 10: 0.979, 30: 1.066, 100: 0.539, 300: 0.849, 1000: 0.954}
PREDICTIONS: dict[str, Any] = {
    "about": "E27 lora_sgd. Plan reports/2026-10-22_plan_commute-second-line.md, Item C and the E27 agent note (class 3, a "
             "named item of the user's 2026-10-22 decision). Is the LoRA's 'outside like noise' reading (E9, E14) the "
             "LoRA's parametrisation or Adam's? Filed before the run.",
    "setup": {
        "lora": f"rank {C.rank} on all 24 matrices, B = 0 and A ~ N(0, 1/d_in) (seed {C.seed}), arm A's loss (CE on o at the "
                f"trigger + KL elsewhere), batch 32, clip 1, {C.ckpts[-1]} steps, factors kept at {C.ckpts}",
        "adam": f"E9's recipe, Adam lr {C.adam_lr}; states at {C.adam_steps}",
        "sgd_grid": f"plain SGD (momentum 0) at {C.sgd_lrs}; E4's rule at step 1000: held-out P(target) >= {P_MIN} and KL "
                    f"elsewhere <= 2 x E9's {E9_KL_1000:.5f} = {KL_MAX:.5f}, the least KL among those. Fallbacks, flagged: "
                    "none within the KL limit -> the least KL among those with P >= 0.9; none reaches 0.9 -> the highest P",
        "series": f"at all of {C.ckpts}: split.block_table (four input-weighted blocks with their analytic null; outside-both "
                  f"/ null pooled over the non-saturated matrices, E14's rule), split.rescaling_table ({C.null_draws} draws), "
                  "lora_readouts.cos_with_init (mean |cos| of A's rows with their init, mean over matrices), rel_norm_max",
        "states": f"SGD at {C.sgd_steps}: do-nothing and least-squares split (harness.nothing_and_split), scaled adjoint split "
                  "(split.adjoint_state), each through harness.measure: Pile ratios to W0 "
                  f"({C.pile_rows} rows, {C.pile_pgd_steps} PGD steps), held-out P(target), Delta's share of ||delta-W||^2",
        "verdict": VERDICT,
    },
    "gates": {
        "C1_w0": "first, alone: faith_score.w0_check (seed-0 PGD within 10% of 0.69, CI-masked within 2% of 0.348, floor "
                 "<= 0.25). Fails: stop, nothing scored",
        "G1_adam_reproduces": f"the regenerated Adam LoRA: do-nothing and least-squares PGD ratios within max(f, 0.209) "
                              f"relative of E9's {G1_REF}; outside-both / null at step 10 within 0.1 of E14's "
                              f"{E14_OUT_OUT_10}; held-out P(target) >= {G1_P_1000} at 1000. Fails: stop",
        "C2_draws_at_null": f"the random draws' outside-both / its analytic null in {list(C2_BAND)} at every row (pooled_ns): "
                            "the block code's positive control",
        "C3_adjoint_keeps_delta": f"positive control: the adjoint states' Delta change fraction <= {C3_MAX} (0 by "
                                  "construction up to the second-order term; E14 6e-6 to 3e-4)",
        "harness_G2": "per least-squares state: P(target) equals the LoRA's to 1e-3 and the route shares sum to 1",
    },
    "predictions": [
        {"id": "P1_outside_like_noise",
         "measure": "outside-both share / its null (pooled_ns) of the selected SGD LoRA at every checkpoint from step 10",
         "pass": f"all >= {P1_MIN}", "relative_vs_absolute": "ratio to the analytic random-update null",
         "outcomes": f"all >= {P1_MIN}: 'parametrisation' (the LoRA's outside share is its random read factor, E14's "
                     "LoRA/fine-tune contrast is not Adam's); any < 0.3: 'optimiser' (like the SGD fine-tune, E14's LoRA rows "
                     "are Adam's); else 'mixed'",
         "predicted": "pass ('parametrisation'): A barely moves under either optimiser, so the read side stays random",
         "expected": f"W0: undefined. Random draws: 1 (C2). Adam LoRA (E14): {E14_LORA_OUT_OUT}; note 0.54 at step 100, below "
                     "this rule. SGD fine-tune (E21, E22): 0.03-0.06",
         "blind_to": "what the outside part does to behaviour; per-matrix spread under pooling; the four saturated matrices",
         "positive_control": "C2; the in-job Adam LoRA's series (G1 at step 10)",
         "noise_floor": "deterministic given delta-W; one training seed. The Adam series moves 0.54-1.07 between checkpoints"},
        {"id": "P2_read_factor_stays",
         "measure": "mean |cos| of the SGD LoRA's A rows with their init at step 1000", "pass": f">= {P2_MIN}",
         "relative_vs_absolute": "absolute",
         "predicted": "pass: under SGD A's gradient is B^T G x, small while B is small",
         "expected": "1 at step 1 (B = 0, A's first gradient zero: the code's check); Adam 0.90 at 1000 (E9)",
         "blind_to": "A's norm; which directions A gained", "positive_control": "1.000 at step 1",
         "noise_floor": "deterministic; one seed"},
        {"id": "P3_states_pass",
         "measure": "verdict of the SGD LoRA's do-nothing and least-squares states at steps 10, 100, 1000",
         "pass": "all six 'pass'", "relative_vs_absolute": "ratio to the same-job W0",
         "predicted": "pass, as Adam's (E9/E14: do-nothing 1.17 / 1.09 / 1.04, least squares 1.26 / 1.22 / 1.07)",
         "expected": "W0: 1. Adam LoRA in-job at 10 and 1000 (G1)",
         "blind_to": "the trigger (about 3 positions in 131k Pile tokens); damage the 20-step adversary does not find",
         "positive_control": "G1 (the Adam LoRA's states reproduce in the same job)",
         "noise_floor": "W0's seed spread f, through the verdict band; one training seed"},
        {"id": "P4_delta_share",
         "measure": "Delta's share of ||delta-W||^2 under the least-squares split at step 1000, SGD LoRA vs the in-job Adam LoRA",
         "pass": "none: two-sided. 'higher' (SGD >= Adam + 0.1), 'lower' (SGD <= Adam - 0.1), 'same' (within 0.1)",
         "relative_vs_absolute": "absolute difference of shares",
         "predicted": "no direction filed",
         "expected": f"Adam 0.54 at 1000 (E14 {E14_DELTA_1000:.3f}); adjoint state 0 (C3)",
         "blind_to": "whether the part left in Delta matters to behaviour", "positive_control": "C3",
         "noise_floor": "deterministic given delta-W; one seed"},
    ],
    "reported_not_scored": ["the grid's P(target), KL elsewhere and rel_norm_max at every checkpoint", "stochastic ratios",
                            "the adjoint states' verdicts and P(target)", "rescaling / null along steps", "pooled_all",
                            "the Adam LoRA's series beside the SGD one", "the least-squares states' Delta share at 10 and 100"],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(C), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


# ----------------------------------------------------------------------------- the LoRAs and their readings


def train(ctx: Any, optimizer: str, lr: float) -> Any:
    from commute.finetune import ArmAConfig, LoraConfig, finetune_lora_arm_a
    from commute.vpd import module_keys

    t0 = time.time()
    cfg = LoraConfig(modules=tuple(module_keys(ctx.model)), rank=C.rank, lr=lr, steps=C.ckpts[-1], checkpoints=C.ckpts,
                     seed=C.seed, optimizer=optimizer)
    a_cfg = ArmAConfig(o_token=ctx.token, steps=C.ckpts[-1], checkpoints=(), seed=C.seed)
    res = finetune_lora_arm_a(ctx.model, ctx.train, ctx.held, a_cfg, cfg, ctx.device, log)
    log(f"[{optimizer} {lr:g} LoRA] {time.time() - t0:.0f} s; step 1000 {res.evals[C.ckpts[-1]]}")
    return res


def series_row(ctx: Any, name: str, step: int, dW: dict[str, Tensor], factors: Any, A0: dict[str, Tensor]) -> dict[str, Any]:
    """The four blocks and the rescaling table (pooled) and |cos| of A with its init, at one checkpoint."""
    from commute import split
    from commute.lora_readouts import cos_with_init

    t0 = time.time()
    b = split.block_table(dW, ctx.bases, ctx.sigma, ctx.non_saturated)
    r = split.rescaling_table(dW, ctx.uv_cpu, ctx.bases, ctx.sigma, ctx.non_saturated, C.null_draws)
    row = {"blocks": {p: b[p] for p in ("pooled_all", "pooled_ns")}, "rescaling": {p: r[p] for p in ("pooled_all", "pooled_ns")},
           "cos_init": cos_with_init(factors, A0)}
    bp, rp = b["pooled_ns"], r["pooled_ns"]
    log(f"[{name} step {step} series] out_out / null {bp['out_out_ratio']:.3f} rescaling / null {rp['rescaling_ratio']:.2f} "
        f"draws' out_out / null {rp['draws_out_out_ratio']:.3f} cos_init {row['cos_init']['mean']:.4f} ({time.time() - t0:.0f} s)")
    return row


def states(ctx: Any, name: str, step: int, dW: dict[str, Tensor], p_o: float, adjoint: bool) -> dict[str, Any]:
    """Do-nothing and least-squares split (and the scaled adjoint split) of one update at one step, with Pile ratios."""
    from commute import harness, split

    tag = f"{name}_step{step}"
    ns = harness.nothing_and_split(ctx, tag, dW, p_o, True, log)
    rec = {"nothing": ns["nothing"], "least_squares": {**ns["lift"], "G2": ns["G2"]}}
    if adjoint:
        dW_adj, st = split.adjoint_state(dW, ctx.uv_cpu, ctx.sigma)
        rec["adjoint"] = harness.measure(ctx, f"{tag}_adjoint", dW_adj, st, True, log)
    return rec


def lora_stage(ctx: Any, out: dict[str, Any], name: str, res: Any, steps: tuple[int, ...], adjoint: bool) -> None:
    """Series at every checkpoint, then the states at `steps`; the factors to results/large/. Writes out[...][name]."""
    from commute.finetune import lora_delta
    from commute.lora_readouts import init_read_factors
    from commute.vpd import module_keys

    torch.save(res.factors, LARGE / f"factors_{name}.pt")
    A0 = init_read_factors(ctx.model, tuple(module_keys(ctx.model)), C.rank, C.seed)
    keys = list(ctx.W0)
    out["evals"][name] = {str(s): e for s, e in res.evals.items()}
    out["series"][name], out["states"][name] = {}, {}
    for s in C.ckpts:
        out["series"][name][str(s)] = series_row(ctx, name, s, lora_delta(res.factors[s], keys, ctx.W0), res.factors[s], A0)
    save(out)
    for s in steps:
        dW = lora_delta(res.factors[s], keys, ctx.W0)
        out["states"][name][str(s)] = states(ctx, name, s, dW, res.evals[s]["p_o"], adjoint)
        save(out)


def gate_g1(out: dict[str, Any]) -> dict[str, Any]:
    """The regenerated Adam LoRA reproduces E9/E14: PGD ratios within max(f, 0.209) relative, outside-both / null at 10
    within 0.1 of E14's, held-out P(target) >= 0.99 at 1000."""
    f = out["W0"]["pgd_floor"]
    tol = max(f, L40_FLOOR)
    st, ser = out["states"]["adam"], out["series"]["adam"]
    cond: dict[str, bool] = {}
    for op, ref in G1_REF.items():
        for s, v in ref.items():
            cond[f"{op}@{s}"] = abs(st[str(s)][op]["pile"]["pgd_kl"] / v - 1) <= tol
    oo = ser["10"]["blocks"]["pooled_ns"]["out_out_ratio"]
    cond["out_out@10"] = abs(oo - E14_OUT_OUT_10) <= 0.1
    p = out["evals"]["adam"]["1000"]["p_o"]
    cond["p_target@1000"] = p >= G1_P_1000
    return {"pass": all(cond.values()), "conditions": cond, "tolerance": tol, "out_out@10": oo, "p_target@1000": p}


def grid(ctx: Any, out: dict[str, Any]) -> Any:
    """Every SGD rate to 1000 steps; held-out P(target), KL elsewhere and rel_norm_max at the checkpoints. Returns the
    selected rate's training result."""
    from commute.lora_readouts import select_rate

    out["grid"], kept = {}, {}
    for lr in C.sgd_lrs:
        res = train(ctx, "sgd", lr)
        out["grid"][f"{lr:g}"] = {str(s): {k: e[k] for k in ("p_o", "kl_other", "rel_norm_max")} for s, e in res.evals.items()}
        res.deltas = {}
        kept[f"{lr:g}"] = res
        dump(OUT, out)
    last = str(C.ckpts[-1])
    lr, how = select_rate({k: v[last] for k, v in out["grid"].items()}, P_MIN, KL_MAX)
    out["rate"] = {"selected": lr, "how": how, "flagged": how != "rule", "p_min": P_MIN, "kl_max": KL_MAX}
    log(f"[rate] selected SGD lr {lr} ({how}); step 1000 {out['grid'][lr][last]}")
    dump(OUT, out)
    return kept[lr]


# ----------------------------------------------------------------------------- the run


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    save(out)
    sys.exit(why)


def stage_all(args: argparse.Namespace) -> None:
    from commute import faith_score, harness

    if not PRED.is_file():
        sys.exit("file the predictions first: run.py --predictions")
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, C.tol, C.pile_rows,
                       C.pile_pgd_steps, C.pile_seeds, log)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "config": asdict(C), "W0": faith_score.w0_check(ctx.pile), "evals": {}, "series": {}, "states": {}}
    log(f"[stage W0] {time.time() - t_start:.0f} s; {out['W0']}")
    save(out)
    if not out["W0"]["pass"]:
        stop(out, "W0 is not at its known values: nothing scored")
    lora_stage(ctx, out, "adam", train(ctx, "adam", C.adam_lr), C.adam_steps, adjoint=False)
    out["G1"] = gate_g1(out)
    log(f"[stage adam + G1] {time.time() - t_start:.0f} s; {out['G1']}")
    save(out)
    if not out["G1"]["pass"]:
        stop(out, "G1 failed: the regenerated Adam LoRA does not reproduce E9/E14")
    res = grid(ctx, out)
    log(f"[stage grid] {time.time() - t_start:.0f} s")
    lora_stage(ctx, out, "sgd", res, C.sgd_steps, adjoint=True)
    out["seconds"] = time.time() - t_start
    save(out)
    log(f"wrote {OUT} ({out['seconds']:.0f} s); score {json.dumps(out['score'], indent=1, default=str)}")


def save(out: dict[str, Any]) -> None:
    out["rows"] = rows(out)
    out["score"] = score(out)
    dump(OUT, out)


def rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat record per (LoRA, step, state) measured so far: what plot.py and the report read."""
    from commute.faith_score import verdict

    f = out["W0"]["pgd_floor"]
    return [{"lora": n, "step": int(s), "op": op, "pgd": r["pile"]["pgd_kl"], "verdict": verdict(r["pile"]["pgd_kl"], f),
             "stochastic": r["pile"]["stochastic_kl"], "p_target": r["routes"]["p_cells"]["T1"],
             "delta_share": r["delta_change_frac"], "rel_norm_max": r["rel_norm_max"], "e_delta": r["routes"]["shares"]["e_delta"]}
            for n, by in out["states"].items() for s, rec in by.items() for op, r in rec.items() if op != "G2"]


# ----------------------------------------------------------------------------- the score (results/predictions.json)


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on whatever is measured so far; a prediction whose inputs are missing is absent."""
    R = {(r["lora"], r["step"], r["op"]): r for r in out["rows"]}
    ser = {n: {int(s): v for s, v in by.items()} for n, by in out["series"].items()}
    sc: dict[str, Any] = {"C1_w0": out["W0"]["pass"], "G1": out.get("G1", {}).get("pass"),
                          "rate_flagged": out.get("rate", {}).get("flagged")}
    sgd = ser.get("sgd", {})
    if len(sgd) == len(C.ckpts):
        ratio = {s: sgd[s]["blocks"]["pooled_ns"]["out_out_ratio"] for s in C.ckpts if s >= 10}
        lo = min(ratio.values())
        sc["P1_outside_like_noise"] = {"pass": lo >= P1_MIN, "ratio": ratio, "min": lo,
                                       "outcome": "parametrisation" if lo >= P1_MIN else "optimiser" if lo < 0.3 else "mixed"}
        cos = sgd[C.ckpts[-1]]["cos_init"]["mean"]
        sc["P2_read_factor_stays"] = {"pass": cos >= P2_MIN, "cos_init@1000": cos}
    keys = [("sgd", s, op) for s in C.sgd_steps for op in ("nothing", "least_squares")]
    if all(k in R for k in keys):
        v = {f"{op}@{s}": R[("sgd", s, op)]["verdict"] for _, s, op in keys}
        sc["P3_states_pass"] = {"pass": all(x == "pass" for x in v.values()), "verdicts": v}
    if ("sgd", 1000, "least_squares") in R and ("adam", 1000, "least_squares") in R:
        a, b = R[("sgd", 1000, "least_squares")]["delta_share"], R[("adam", 1000, "least_squares")]["delta_share"]
        sc["P4_delta_share"] = {"pass": None, "sgd": a, "adam": b, "adam_E14": E14_DELTA_1000,
                                "outcome": "higher" if a >= b + P4_BAND else "lower" if a <= b - P4_BAND else "same"}
    if ser:
        sc["C2_draws_at_null"] = {"pass": all(C2_BAND[0] <= row["rescaling"]["pooled_ns"]["draws_out_out_ratio"] <= C2_BAND[1]
                                              for by in ser.values() for row in by.values())}
    adj = [r for r in out["rows"] if r["op"] == "adjoint"]
    if adj:
        sc["C3_adjoint_keeps_delta"] = {"pass": all(r["delta_share"] <= C3_MAX for r in adj),
                                        "delta_share": {r["step"]: r["delta_share"] for r in adj}}
    g2 = {f"{n}@{s}": rec["least_squares"]["G2"]["ok"] for n, by in out["states"].items() for s, rec in by.items()}
    if g2:
        sc["harness_G2"] = {"pass": all(g2.values()), "per_state": g2}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage == "all":
        stage_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
