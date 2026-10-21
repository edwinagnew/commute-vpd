"""lora_split: the LoRA split study. Every saved update through one split table (plan: reports/2026-10-20_plan_lora-split.md).

Train nothing new for the LoRAs. The updates are the rank-8 LoRA on all 24 matrices at nine steps to 1000 (E9's saved
factors), the one-matrix LoRAs of ranks 1 to 32 at steps 10, 30, 100 (E7's), and the emoticon full fine-tune re-run
once with its delta-W kept at E9's nine steps. For each update, per matrix and pooled: the outside share as four
input-weighted blocks that sum to one (split.four_blocks) with their exact random-update null, the standing algebraic
outside-both share (E2 to E9's number) beside it, the rescaling share (the strictest span) and the per-subcomponent
pieces of both splits (least-squares and adjoint) with one concentration number each, against random draws. Part 2:
do-nothing, the least-squares split and the scaled adjoint split of the fine-tune and the 24-matrix LoRA at steps 10,
100 and 1000 through the standing harness (route split, Delta's share, Pile ratios) in one job on one card.
Design check confirmed by the user 2026-10-20 ("as recommended": the read-side definition of the blocks, part 2 in,
the adjoint split applied with a per-matrix least-squares scale).

Stages (results/predictions.json must exist; `--predictions` files it once):
  all             GPU -> results/lora_split.json (dumped after every update) + results/large/ (fine-tune deltas,
                  per-subcomponent pieces at the part-2 steps, log)
  --smoke         3 fine-tune steps, E9 steps 1-3, 8 Pile rows, 2 PGD steps, one null draw -> *_smoke.json
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt} and results/windows.json (round 1);
experiments/lora_plateau/results/{large/lora_factors.pt, lora_plateau.json} (E9);
experiments/lora_commute/results/large/lora_r{1,4,8,32}_s0_factors.pt and lora_r8_s1_factors.pt (E7).
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
E9 = HERE.parent / "lora_plateau" / "results"
E7_LARGE = HERE.parent / "lora_commute" / "results" / "large"
E7_SITE, E7_RUNS, E7_STEPS = "h.2.mlp.down_proj", ("r1_s0", "r4_s0", "r8_s0", "r32_s0", "r8_s1"), (10, 30, 100)
E9_STEPS = (1, 2, 3, 5, 10, 30, 100, 300, 1000)
FT_LR, TOL = 3e-5, 1e-6  # arm A's recipe (rounds 1 to 5) and the split tolerance
BLOCKS = ("in_in", "in_out", "out_in", "out_out")
POOLED = BLOCKS + tuple(f"null_{b}" for b in BLOCKS) + ("alg_omega_act", "alg_null_act", "rescaling", "rescaling_null",
                                                        "adjoint_reproduced")


@dataclass(frozen=True)
class Sizes:
    ft_steps: int
    ckpts: tuple[int, ...]  # the fine-tune's checkpoints and the E9 steps read; the table's rows
    pile_steps: tuple[int, ...]  # part 2
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    null_draws: int
    suffix: str

    def __post_init__(self) -> None:
        assert self.ckpts == tuple(sorted(self.ckpts)) and self.ckpts[-1] == self.ft_steps, self.ckpts
        assert set(self.ckpts) <= set(E9_STEPS) and set(self.pile_steps) <= set(self.ckpts), (self.ckpts, self.pile_steps)


FULL = Sizes(ft_steps=1000, ckpts=E9_STEPS, pile_steps=(10, 100, 1000), pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2),
             null_draws=2, suffix="")
SMOKE = Sizes(ft_steps=3, ckpts=(1, 2, 3), pile_steps=(2, 3), pile_rows=8, pile_pgd_steps=2, pile_seeds=(0,), null_draws=1,
              suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-20",
    "design": "reports/2026-10-20_plan_lora-split.md (design check confirmed by the user 2026-10-20, as recommended)",
    "setup": {
        "updates": {"ft": f"emoticon full fine-tune, arm A recipe (Adam lr {FT_LR}, all 24 matrices, seed 0), delta-W kept at steps "
                          f"{FULL.ckpts}",
                    "lora24": f"E9's rank-8 LoRA on all 24 matrices, BA from the saved factors at steps {FULL.ckpts}",
                    "site": f"E7's LoRAs {E7_RUNS} on {E7_SITE} at steps {E7_STEPS}"},
        "blocks": "split.four_blocks: input-weighted energy of delta-W split by P_U on the write side and, on the read side, by the "
                  "part of the input predictable from what the alive subcomponents read (a1) against the uncorrelated rest (a2); "
                  "sums to one; null = exact expectation for an isotropic random update. Pooled by energy over all 24 matrices "
                  "and over the 20 non-saturated ones (the scored pooling, as E8 and E9)",
        "algebraic": "split.weighted_fractions, the standing outside-both share with its null (E2 to E9), for continuity",
        "rescaling": "split.rescaling_share, the weighted least-squares fit by sum_c alpha_c u_c v_c^T; null = mean over "
                     f"{FULL.null_draws} Gaussian draws of the same per-matrix weighted energy",
        "splits": "least-squares (split.least_squares_split) and adjoint (split.adjoint_split, scaled per matrix by "
                  "split.adjoint_scale); n_half = number of subcomponents over all matrices carrying half the summed piece energy "
                  "(split.piece_energies), against the same for the random draws",
        "part2": f"harness.nothing_and_split and harness.measure of the adjoint state at steps {FULL.pile_steps}: route split at the "
                 f"trigger, Delta's share of ||delta-W||^2, Pile ratios ({FULL.pile_rows} val rows, {FULL.pile_pgd_steps} PGD steps, "
                 f"W0 floor over adversary seeds {FULL.pile_seeds}); the adjoint state's held-out P(o) against the update's",
        "ratio": "a block's pooled share divided by its pooled null; read_in = in_in + out_in, write_in = in_in + in_out",
    },
    "gates": [
        {"id": "G1_reproduce_E9", "rule": "lora24's algebraic pooled outside-both share at the part-2 steps equals E9's "
                                          "(lora_plateau.json pile_steps[step].inexpressible.pooled) within 0.003",
         "if_fail": "the factor loading or the harness changed; stop and find out"},
        {"id": "G2_split_state", "rule": "harness G2: the least-squares split state's P(o) equals the update's to 1e-3 and the route "
                                         "shares sum to 1 to 1e-6, every part-2 step", "if_fail": "that step is reported, not scored"},
    ],
    "predictions": [
        {"id": "P1_lora_at_null_both_sides", "scored": True, "arm": "lora24, every step, non-saturated pooling",
         "pass": "read_in ratio <= 1.2 and write_in ratio <= 1.3 at every step",
         "null": "write_in ratio >= 1.5 at some step >= 10 (B learned a direction col U writes) or read_in ratio >= 1.5 (A became readable)",
         "predicted": "pass: E9's outside-both ratio 0.82-1.02 at every step leaves little room for either side to be well inside"},
        {"id": "P2_finetune_inside", "scored": True, "arm": "ft, steps 10 to 1000, non-saturated pooling",
         "pass": "out_out ratio <= 0.7 at every step from 10 on, and read_in ratio >= 1.3 and write_in ratio >= 1.3 at step 10",
         "null": "out_out ratio >= 0.8 at some step >= 10",
         "predicted": "pass: E8 read 0.27 against 0.53 (0.51) at step 10 under the algebraic share; the trend along steps is the reading"},
        {"id": "P3_nothing_is_rescaled", "scored": True, "arm": "ft and lora24, every step, pooled over all 24",
         "pass": "rescaling share <= 3 x its random-draw null for both updates at every step",
         "null": "ft >= 10 x at some step (the update strengthens existing pieces)",
         "predicted": "pass: E1 read 0.06% against 0.035% for a fine-tune (1.7 x)"},
        {"id": "P4_rank_is_not_the_axis", "scored": True, "arm": f"site rows, every rank and step on {E7_SITE}",
         "pass": "out_out ratio in [0.8, 1.2] for every run and step",
         "null": "the rank-32 run's ratio differs from the rank-1 run's by more than 0.3 at step 100",
         "predicted": "pass: E7's algebraic shares were 0.38-0.43 against 0.42 at every rank"},
        {"id": "P5_adjoint_behaviour", "scored": False, "arm": "the scaled adjoint state, part-2 steps, ft and lora24",
         "reported": "held-out P(o) against the update's, the reproduced share R and the scale c per matrix, the standing table",
         "expected": "P(o) tracks R: the LoRA's state (R near 1 - out_out, about 0.4) loses the behaviour, the fine-tune's keeps "
                     "most of it; first reading, no threshold"},
        {"id": "P6_pile_matched_card", "scored": True, "arm": "least-squares split states at the part-2 steps, one card",
         "pass": "lora24's Pile PGD ratio <= 1 + 3 f at steps 10, 100 and 1000 and ft's >= 2 at step 10",
         "null": "ft's split <= 1 + 3 f at step 10 (E8's 5.41 was the card) or lora24's > 1 + 3 f at some step",
         "predicted": "pass: E9 (L40S) 1.26, 1.22, 1.07; E8 and E2 (L40) 5.41; ft at steps 100 and 1000 reported"},
    ],
    "reported_not_scored": "n_half of both splits against the random draws; c and R per matrix; part 2's route split and Delta share; "
                           "the adjoint state's Delta change (second order in c, expected small); the per-matrix tables; the fine-tune's "
                           "P(o) and kl_other per step",
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


# ----------------------------------------------------------------------------- the updates


def finetune(ctx: Any, S: Sizes) -> tuple[dict[int, dict[str, Tensor]], dict[int, dict[str, float]]]:
    """Arm A's full fine-tune, seed 0, with delta-W kept at every checkpoint (results/large/ft_deltas.pt)."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=ctx.token, lr=FT_LR, steps=S.ft_steps, checkpoints=S.ckpts, seed=0)
    res = finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log)
    torch.save(res.deltas, LARGE / f"ft_deltas{S.suffix}.pt")
    return res.deltas, res.evals


def lora_updates(ctx: Any, path: Path, steps: tuple[int, ...]) -> dict[int, dict[str, Tensor]]:
    """BA per matrix (zeros where the LoRA has no factors) from a saved {step: {matrix: (B, A)}} file."""
    from commute.finetune import lora_delta
    from commute.vpd import module_keys

    factors = torch.load(path, weights_only=False)
    return {s: lora_delta(factors[s], list(module_keys(ctx.model)), ctx.W0) for s in steps}


# ----------------------------------------------------------------------------- the split table


def matrix_record(ctx: Any, k: str, M: Tensor, draws: int) -> tuple[dict[str, float], dict[str, Any], dict[str, list[Tensor]]]:
    """One matrix's row: the four blocks and their null, the algebraic share, the rescaling share with its measured null,
    the adjoint scale; the pieces of both splits (fp16) and their energies, with the random draws' energies."""
    from commute import split

    U_A, V_A, b, Sg = ctx.uv_cpu[k][0].double(), ctx.uv_cpu[k][1].double(), ctx.bases[k], ctx.sigma[k]
    M = M.double()
    rec = split.four_blocks(M, b, Sg)
    rec.update({f"alg_{kk}": v for kk, v in split.weighted_fractions(M, b, Sg).items()})
    rec["rescaling"] = split.rescaling_share(M, U_A, V_A, Sg)
    rec["adjoint_c"], rec["adjoint_reproduced"] = split.adjoint_scale(M, U_A, V_A, Sg)
    rec["energy"] = split.weighted_energy(M, Sg)

    def pieces(X: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor, Tensor]:
        dU, dV = split.least_squares_split(X, b, U_A, V_A)
        aU, aV = split.adjoint_split(X, U_A, V_A)
        c = split.adjoint_scale(X, U_A, V_A, Sg)[0]
        aU, aV = c * aU, c * aV
        return (dU, dV), (aU, aV), split.piece_energies(dU, dV, U_A, V_A, Sg), split.piece_energies(aU, aV, U_A, V_A, Sg)

    ls, adj, e_ls, e_adj = pieces(M)
    rands = [split.random_like(M, Sg, seed) for seed in range(draws)]
    rec["rescaling_null"] = sum(split.rescaling_share(G, U_A, V_A, Sg) for G in rands) / draws
    rec["null_out_out_draws"] = sum(split.four_blocks(G, b, Sg)["out_out"] for G in rands) / draws
    energies: dict[str, list[Tensor]] = {"least_squares": [e_ls], "adjoint": [e_adj], "least_squares_null": [], "adjoint_null": []}
    for G in rands:
        _, _, e1, e2 = pieces(G)
        energies["least_squares_null"].append(e1)
        energies["adjoint_null"].append(e2)
    saved = {"least_squares": (ls[0].half(), ls[1].half()), "adjoint": (adj[0].half(), adj[1].half()), "c": rec["adjoint_c"]}
    return rec, saved, energies


def pool(per: dict[str, dict[str, float]], keys: list[str]) -> dict[str, float]:
    e = sum(per[k]["energy"] for k in keys)
    return {f: sum(per[k][f] * per[k]["energy"] for k in keys) / e for f in POOLED}


def split_table(ctx: Any, dW: dict[str, Tensor], draws: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every non-zero matrix's row, the two poolings, and n_half per split (pieces pooled over the matrices; the random
    draws' n_half averaged over draws). Returns the record and the pieces to save."""
    from commute import split

    per: dict[str, dict[str, float]] = {}
    saved: dict[str, Any] = {}
    energies: dict[str, list[list[Tensor]]] = {"least_squares": [[]], "adjoint": [[]], "least_squares_null": [[] for _ in range(draws)],
                                               "adjoint_null": [[] for _ in range(draws)]}
    for k, M in dW.items():
        if not M.any():
            continue
        per[k], saved[k], e = matrix_record(ctx, k, M, draws)
        for name, lists in e.items():
            for i, t in enumerate(lists):
                energies[name][i].append(t)
    total = sum(p["energy"] for p in per.values())
    for p in per.values():
        p["energy_share"] = p["energy"] / total
    ns = [k for k in per if k in ctx.non_saturated]
    rec = {"per_matrix": per, "pooled_all": pool(per, list(per)), "pooled_ns": pool(per, ns) if ns else None,
           "n_half": {name: sum(split.n_half(torch.cat(d)) for d in lists) / len(lists) for name, lists in energies.items()}}
    return rec, saved


def adjoint_state(ctx: Any, dW: dict[str, Tensor], saved: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor, Tensor]]]:
    """The scaled adjoint split as a state: the target moves by c K(delta-W) per matrix, the subcomponents by c (dU, dV)."""
    from commute import split

    dW_adj, state = {}, {}
    for k in ctx.W0:
        if k not in saved:
            dW_adj[k] = torch.zeros_like(dW[k])
            continue
        U_A, V_A = ctx.uv_cpu[k][0].double(), ctx.uv_cpu[k][1].double()
        aU, aV = (x.double() for x in saved[k]["adjoint"])
        dW_adj[k] = split.dsigma(aU, aV, U_A, V_A).float()
        state[k] = (dW_adj[k], (U_A + aU).float(), (V_A + aV).float())
    return dW_adj, state


# ----------------------------------------------------------------------------- the run


def stage_all(args: argparse.Namespace) -> None:
    from commute import harness

    assert PRED.is_file(), "file predictions first: run.py --predictions"
    S = SMOKE if args.smoke else FULL
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    name = f"lora_split{S.suffix}{'' if args.stage == 'all' else '_' + args.stage}.json"
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S),
                           "W0": {"pile": ctx.pile, "pgd_floor": harness.pgd_floor(ctx), "non_saturated": ctx.non_saturated}}
    deltas, evals = finetune(ctx, S)
    e9 = json.loads((E9 / "lora_plateau.json").read_text())
    updates: dict[str, dict[int, dict[str, Tensor]]] = {"ft": deltas, "lora24": lora_updates(ctx, E9 / "large" / "lora_factors.pt", S.ckpts)}
    p_o = {"ft": {s: evals[s]["p_o"] for s in S.ckpts}, "lora24": {s: e9["checkpoints"][str(s)]["eval"]["p_o"] for s in S.ckpts}}
    for run in E7_RUNS:
        updates[f"site_{run}"] = lora_updates(ctx, E7_LARGE / f"lora_{run}_factors.pt", E7_STEPS)
    out["ft_evals"] = evals
    out["table"], pieces = {}, {}
    for uname, by_step in updates.items():
        out["table"][uname] = {}
        for step, dW in by_step.items():
            t0 = time.time()
            rec, saved = split_table(ctx, dW, S.null_draws)
            out["table"][uname][str(step)] = rec
            pieces[(uname, step)] = saved
            if uname.startswith("site") or step in S.pile_steps:
                torch.save(saved, LARGE / f"pieces_{uname}_step{step}{S.suffix}.pt")
            p = rec["pooled_ns"] or rec["pooled_all"]
            log(f"[{uname} step {step}] blocks in_in {p['in_in']:.3f} in_out {p['in_out']:.3f} out_in {p['out_in']:.3f} "
                f"out_out {p['out_out']:.3f} (null {p['null_out_out']:.3f}, algebraic {p['alg_omega_act']:.3f}/{p['alg_null_act']:.3f}) "
                f"rescaling {p['rescaling']:.2e} (null {p['rescaling_null']:.2e}) n_half {rec['n_half']} ({time.time() - t0:.0f} s)")
        dump(name, out)
    if args.stage == "pieces":  # the table and the pieces only (rerun 2026-10-20: the scored run's pieces died with its pod)
        out["seconds"] = time.time() - t_start
        dump(name, out)
        log(f"wrote {RESULTS / name} ({out['seconds']:.0f} s); pieces in {LARGE}")
        return
    out["part2"] = {}
    for uname in ("ft", "lora24"):
        out["part2"][uname] = {}
        for step in S.pile_steps:
            dW = updates[uname][step]
            rec = harness.nothing_and_split(ctx, f"{uname}_step{step}", dW, p_o[uname][step], bool(S.pile_rows), log)
            dW_adj, state = adjoint_state(ctx, dW, pieces[(uname, step)])
            rec["adjoint"] = harness.measure(ctx, f"{uname}_step{step}_adjoint", dW_adj, state, bool(S.pile_rows), log)
            rec["adjoint"]["p_o_update"] = p_o[uname][step]
            out["part2"][uname][str(step)] = rec
            dump(name, out)
    out["score"] = score(out, e9)
    out["seconds"] = time.time() - t_start
    dump(name, out)
    log(f"wrote {RESULTS / name} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


# ----------------------------------------------------------------------------- the score


def ratios(p: dict[str, float]) -> dict[str, float]:
    """read_in, write_in and out_out, each divided by its null."""
    return {"read_in": (p["in_in"] + p["out_in"]) / (p["null_in_in"] + p["null_out_in"]),
            "write_in": (p["in_in"] + p["in_out"]) / (p["null_in_in"] + p["null_in_out"]),
            "out_out": p["out_out"] / p["null_out_out"], "rescaling": p["rescaling"] / p["rescaling_null"]}


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    return {"pass": all(cond.values()), "conditions": cond, **values}


def score(out: dict[str, Any], e9: dict[str, Any]) -> dict[str, Any]:
    T, S = out["table"], out["sizes"]
    r = {u: {s: ratios(T[u][s]["pooled_ns"]) for s in T[u]} for u in T}
    steps = [str(s) for s in S["ckpts"]]
    late = [s for s in steps if int(s) >= 10]
    sc: dict[str, Any] = {"ratios": r}
    sc["G1_reproduce_E9"] = {s: abs(T["lora24"][s]["pooled_ns"]["alg_omega_act"] - e9["pile_steps"][s]["inexpressible"]["pooled"]) <= 0.003
                             for s in map(str, S["pile_steps"]) if s in e9["pile_steps"]}
    sc["P1_lora_at_null_both_sides"] = _pred({f"read_in<=1.2@{s}": r["lora24"][s]["read_in"] <= 1.2 for s in steps}
                                             | {f"write_in<=1.3@{s}": r["lora24"][s]["write_in"] <= 1.3 for s in steps})
    sc["P2_finetune_inside"] = _pred({f"out_out<=0.7@{s}": r["ft"][s]["out_out"] <= 0.7 for s in late}
                                     | ({"read_in>=1.3@10": r["ft"]["10"]["read_in"] >= 1.3, "write_in>=1.3@10": r["ft"]["10"]["write_in"] >= 1.3}
                                        if "10" in r["ft"] else {"step 10 absent": False}))
    sc["P3_nothing_is_rescaled"] = _pred({f"{u}@{s}": ratios(T[u][s]["pooled_all"])["rescaling"] <= 3 for u in ("ft", "lora24") for s in steps})
    site = {u: {s: ratios(T[u][s]["pooled_all"])["out_out"] for s in T[u]} for u in T if u.startswith("site")}
    sc["P4_rank_is_not_the_axis"] = _pred({f"{u}@{s}": 0.8 <= v <= 1.2 for u in site for s, v in site[u].items()}, site=site)
    if out.get("part2") and out["W0"]["pgd_floor"] is not None:
        line = 1 + 3 * out["W0"]["pgd_floor"]
        pgd = {u: {s: rec["lift"]["pile"]["pgd_kl"] for s, rec in out["part2"][u].items()} for u in out["part2"]}
        g2 = {u: {s: rec["G2"]["ok"] for s, rec in out["part2"][u].items()} for u in out["part2"]}
        sc["G2_split_state"] = g2
        sc["P6_pile_matched_card"] = _pred({f"lora24<=line@{s}": v <= line for s, v in pgd["lora24"].items()}
                                           | {"ft>=2@10": pgd["ft"].get("10", 0.0) >= 2}, line=line, pgd=pgd)
        sc["P5_adjoint_behaviour"] = {u: {s: {"p_o": rec["adjoint"]["routes"]["p_cells"]["T1"], "p_o_update": rec["adjoint"]["p_o_update"],
                                             "pgd": rec["adjoint"]["pile"]["pgd_kl"], "delta_change": rec["adjoint"]["delta_change_frac"],
                                             "R_pooled": T[u][s]["pooled_all"]["adjoint_reproduced"]}
                                          for s, rec in out["part2"][u].items()} for u in out["part2"]}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all", "pieces"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.stage in ("all", "pieces"):
        stage_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
