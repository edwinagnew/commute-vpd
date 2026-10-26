"""lora_matched: a LoRA placed where LoRAs are placed (rank 8 on all 24 decomposed matrices), under the same lift as
the full-weight fine-tune.

Round 4 (`experiments/lora_commute`) compared a LoRA on one matrix the decomposition barely reads (h.2.mlp.down_proj:
79% of its input energy outside the alive read span) against a fine-tune whose energy sits on matrices it reads well,
so it could not separate the LoRA from its site (`reports/reviews/2026-10-19_review_lora-failure-and-fixes.md`, section 3.2,
confirmed by the user 2026-10-19). Here both updates live on the same 24 matrices and go through the same map.

  lora   rank-8 LoRA on all 24 matrices, Adam lr 1e-3, arm A's loss, 100 steps; scored at the first checkpoint with
         held-out P(o) >= 0.9, reported at step 100; seed 1 repeated for the noise floor m
  ft     the full-weight fine-tune at step 10 (arm A, Adam lr 3e-5): the positive control, and the L40 rerun of round 2's
         lift (Pile PGD ratio 5.41) that round 4's 4090 run missed (7.31)
Each update is measured do-nothing (delta-W in Delta) and Euclidean-lifted (`commute.harness`): the route split at the
trigger, Delta's share of the update, the outside_share share per matrix and pooled, and round 2's Pile ratios against
the 3-seed W0 floor.

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  all             GPU -> results/lora_matched.json (dumped after every arm) + results/large/ (LoRA factors, log)
  all --smoke     GPU, minutes: every path at the SMOKE sizes -> results/lora_matched_smoke.json
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt} and results/windows.json (round 1).

Usage (repo root):  uv run python experiments/lora_matched/run.py all
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

RANK, LORA_LR, FT_LR, P_LEARN, TOL = 8, 1e-3, 3e-5, 0.9, 1e-6
REF = {  # what the control should reproduce: round 2 on an L40 (lift_faith.json, step 10) and round 4 on a 4090
    "nothing_ft_pgd_ratio": 2.18, "lift_ft_pgd_ratio": 5.41, "round4_4090": {"nothing_ft_pgd_ratio": 2.49, "lift_ft_pgd_ratio": 7.31},
    "lift_ft_e_delta": 0.085, "ft_inexpressible": 0.27, "ft_inexpressible_null": 0.53, "lift_site_pgd_ratio": 2.02,
}


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run."""

    lora_steps: int
    lora_ckpts: tuple[int, ...]  # scored step = the first with held-out P(o) >= P_LEARN; the last is reported as "finished"
    ft_steps: int  # the fine-tune is measured at its last step only (the matched step)
    pile_rows: int  # round 2's Pile harness: 256 rows in two blocks; 0 skips it
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    suffix: str

    def __post_init__(self) -> None:
        assert self.lora_ckpts == tuple(sorted(self.lora_ckpts)) and self.lora_ckpts[-1] == self.lora_steps, self.lora_ckpts
        assert bool(self.pile_rows) == bool(self.pile_seeds)


FULL = Sizes(lora_steps=100, lora_ckpts=(1, 2, 3, 5, 10, 30, 100), ft_steps=10, pile_rows=256, pile_pgd_steps=20,
             pile_seeds=(0, 1, 2), suffix="")
SMOKE = Sizes(lora_steps=2, lora_ckpts=(1, 2), ft_steps=2, pile_rows=8, pile_pgd_steps=2, pile_seeds=(0,), suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "about": (
        "lora_matched (2026-10-19): rank-8 LoRA on all 24 decomposed matrices (the standard placement) and the full-weight "
        "fine-tune, both Euclidean-lifted onto the released subcomponents and both measured do-nothing, on the emoticon task. "
        "Repairs round 4's confound (LoRA on one badly-read matrix vs fine-tune on all 24): does a LoRA placed the way LoRAs "
        "are placed commute as well as a full-weight update? Design: reports/reviews/2026-10-19_review_lora-failure-and-fixes.md "
        "section 3.2, confirmed by the user. Filed before any arm was trained."
    ),
    "setup": {
        "windows": "round 1's emoticon windows (365 train, 73 held-out, trigger at index 20, target token `o`)",
        "lora": f"rank {RANK} on all 24 matrices, B = 0 and A ~ N(0, 1/d_in) per matrix (seed 0), Adam lr {LORA_LR}, arm A's loss "
                f"(CE on `o` at the trigger + KL elsewhere), batch 32, clip 1, {FULL.lora_steps} steps, checkpoints {FULL.lora_ckpts}. "
                f"Scored step = the first checkpoint with held-out P(o) >= {P_LEARN}; step {FULL.lora_steps} reported (no Pile). "
                "Seed 1 repeated (init and batches), measured at its own scored step, no Pile",
        "ft": f"full-weight fine-tune, arm A recipe (Adam lr {FT_LR}, all 24 matrices, seed 0), measured at step {FULL.ft_steps} "
              "(round 2's and round 4's matched step)",
        "lift": "split.split_state: Euclidean (minimum-norm) lift of each matrix's delta-W onto its alive subcomponents, tol 1e-6; "
                "target weights W0 + delta-W, so Delta moves by O(delta-W) - dU dV^T",
        "W0": f"the released decomposition under round 2's Pile harness with adversary seeds {FULL.pile_seeds}",
    },
    "measurements": {
        "routes": "local_faith.route_split on the held-out windows: d, r, i, e_gate, e_delta (sum to 1), gain in nats",
        "pile": "faithfulness.pile_faithfulness (2 blocks of 128 val rows 0..255, one adversary per block, 20 steps of 0.1; "
                "stochastic, PGD, CI-masked, unmasked KL); ratio = arm / W0 under adversary seed 0",
        "pgd_floor": "f = max/min - 1 of the W0 PGD KL over the 3 adversary seeds; 'within 3 floors' = ratio <= 1 + 3 f",
        "inexpressible": "split.weighted_fractions with Sigma = E[a a^T] of the matrix's input over the train windows: the share "
                         "of the update's output energy no first-order change of the alive subcomponents can express; per matrix "
                         "with its isotropic null, and pooled by energy over the non-saturated matrices (round 4's rule, so the "
                         "fine-tune reads 0.27 against a null of 0.53)",
        "cos_init": "per matrix, mean |cos| of A's rows with their initialisation, at every checkpoint",
    },
    "expected_at": {
        "do_nothing": "e_delta 1, Delta change 1, d 0; Pile PGD ratio 2.18 (ft, round 2) / 1.4-2.5 (LoRA)",
        "control_ft_lift_step10": "e_delta 0.085, Delta change 0.36, pooled outside_share 0.27 (null 0.53), Pile PGD ratio 5.41 "
                                  "(round 2, L40; round 4 on a 4090 read 7.31)",
        "W0": "every Pile ratio 1 by construction; floor f about 0.035 (round 4)",
    },
    "blind_to": {
        "d, r, e_delta": "everything away from the trigger (cover: kl_other and the Pile ratios)",
        "Pile PGD ratio": "a trigger-specific attack (the adversary is shared over 65k Pile positions holding ~3 triggers); the "
                          "20-step search is chaotic across GPUs (round 4's 35% miss), so only differences well over 35% are read",
        "inexpressible": "nothing about behaviour: an outside_share part can still be redundant (round 2's review)",
        "controls": "G2 (the lifted state is the fine-tune's model exactly), G4 (the harness reproduces round 2), the seed "
                    "pair for m, the W0 seeds for f",
    },
    "gates": [
        {"id": "G0_floor", "rule": "m = max(0.1, 3 x max |Delta d|, |Delta r| between the lifts of LoRA seeds 0 and 1 at their "
                                   "scored steps); f = PGD floor as above"},
        {"id": "G1_arms_learn", "rule": f"the LoRA (each seed) reaches P(o) >= {P_LEARN} at some checkpoint; the ft has P(o) >= {P_LEARN} "
                                        f"at step {FULL.ft_steps}", "if_fail": "the arm is reported, not scored"},
        {"id": "G2_lift_exact", "rule": "for every lift: |P(o) of the lifted state's target model - the fine-tune's held-out P(o)| "
                                        "<= 1e-3 and |sum of shares - 1| <= 1e-6", "if_fail": "the lift or the Delta bookkeeping is wrong; stop"},
        {"id": "G4_reproduces", "rule": "ft do-nothing and lift Pile PGD ratios at step 10 within 25% of round 2's 2.18 and 5.41 (L40)",
         "if_fail": "if they land near round 4's 2.49 and 7.31 instead, the GPU is not the cause and the code changed since round 2 "
                    "(faithfulness.py, local_faith.py) is the suspect; find it before the analysis",
         "reported_beside": "round 4's 4090 numbers"},
    ],
    "predictions": [
        {"id": "M1_read_side_stays_random", "scored": True, "arm": "lora seed 0, scored step",
         "pass": "pooled outside_share / pooled null >= 0.8", "null": "pooled outside_share / pooled null <= 0.6 (the fine-tune's 0.51)",
         "predicted": "pass: A moves little from its random init in the 10-30 steps it takes to learn (round 4: |cos| 0.92-0.98), "
                      "so on every non-saturated matrix the update is as outside_share as an isotropic one",
         "reported": "per-matrix outside_share, null and read_out; mean |cos| of A with its init at the scored step"},
        {"id": "M2_lift_leaves_it_in_delta", "scored": True, "arm": "lora seed 0 lift, scored step",
         "pass": "e_delta >= 0.3 (round 4's one-site LoRA: 0.33-0.43)", "null": "e_delta <= 0.2 (the fine-tune's lift: 0.085)",
         "predicted": "pass, from M1: what the subcomponents cannot read of a random-read LoRA stays in Delta and carries its share "
                      "of the behaviour; the four saturated matrices (38% of the fine-tune's energy) are the reason it may fall short"},
        {"id": "C2m_lora_lift_fragile_like_ft", "scored": True, "arm": "Pile PGD ratio of the lora seed 0 lift (scored step) vs the ft lift (step 10)",
         "pass": "lora lift >= 5 (fine-tune-like: the lift's fragility belongs to 24-matrix lifted states, not to the update)",
         "null": "lora lift <= 2 and e_delta <= 0.2 (a LoRA update is expressible and survives where the fine-tune's does not; reopens round 4's C2)",
         "predicted": "pass, weakly: round 4's fragility grew with how much was lifted (one matrix 2.0, 24 matrices 5.4-7.3); "
                      "between 2 and 5 is neither and is read against the ft lift's own number in the same job",
         "noise": "the two lines sit more than 35% from each other and from the references, so the 4090-vs-L40 miss cannot flip them"},
        {"id": "C1m_direct_share_flat", "scored": True, "arm": "d of the lora lift (scored step) vs the ft lift (step 10)",
         "pass": "|d(lora lift) - d(ft lift)| < m (the mechanism reading of rounds 3-4: d is 0.05-0.11 whatever the update or map)",
         "null": "d(lora lift) - d(ft lift) >= m", "predicted": "pass"},
    ],
    "reported_not_scored": "P(o) and kl_other at every LoRA checkpoint (kl_other at the scored step beside the fine-tune's 0.053); "
                           "the step-100 lift and do-nothing (no Pile); the do-nothing Pile ratios; Delta change; per-matrix "
                           "inexpressible shares of both updates with each matrix's share of the update's energy; seed 1's numbers",
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


# ----------------------------------------------------------------------------- arms


def a_cfg(token: int, steps: int, ckpts: tuple[int, ...], seed: int) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=token, lr=FT_LR, steps=steps, checkpoints=ckpts, seed=seed)


def lora_arm(ctx: Any, S: Sizes, seed: int, pile: bool) -> dict[str, Any]:
    """The LoRA on all 24 matrices: train, then do-nothing and lifted at the scored step (Pile) and at the last checkpoint."""
    from commute import harness, vpd
    from commute.finetune import LoraConfig, finetune_lora_arm_a, lora_factors

    modules = tuple(vpd.module_keys(ctx.model))
    cfg = LoraConfig(modules=modules, rank=RANK, lr=LORA_LR, steps=S.lora_steps, checkpoints=S.lora_ckpts, seed=seed)
    res = finetune_lora_arm_a(ctx.model, ctx.train, ctx.held, a_cfg(ctx.token, S.lora_steps, (), seed), cfg, ctx.device, log)
    A0 = {k: f[1].detach().cpu() for k, f in lora_factors({k: vpd.target_linear(ctx.model, k) for k in modules}, RANK, seed).items()}
    scored = next((s for s in S.lora_ckpts if res.evals[s]["p_o"] >= P_LEARN), None)
    name = f"lora_s{seed}"
    torch.save(res.factors, LARGE / f"{name}_factors.pt")
    cos = {s: {k: float(torch.nn.functional.cosine_similarity(f[k][1], A0[k], dim=1).abs().mean()) for k in modules}
           for s, f in res.factors.items()}
    rec: dict[str, Any] = {"evals": res.evals, "scored_step": scored, "learned": scored is not None, "steps": {},
                           "cos_init": {s: {"mean": sum(c.values()) / len(c), "per_matrix": c} for s, c in cos.items()}}
    for step in sorted({scored or S.lora_ckpts[-1], S.lora_ckpts[-1]}):
        rec["steps"][str(step)] = harness.nothing_and_split(ctx, f"{name}_step{step}", res.deltas[step], res.evals[step]["p_o"],
                                                          pile and step == scored, log)
        log(f"[{name} step {step}] cos_init mean {rec['cos_init'][step]['mean']:.3f} kl_other {res.evals[step]['kl_other']:.4f}")
    return rec


def ft_arm(ctx: Any, S: Sizes, pile: bool) -> dict[str, Any]:
    """The full-weight fine-tune at its last step: do-nothing and lifted on all 24 matrices."""
    from commute import harness
    from commute.finetune import finetune_arm_a

    res = finetune_arm_a(ctx.model, ctx.train, ctx.held, a_cfg(ctx.token, S.ft_steps, (S.ft_steps,), 0), ctx.device, log)
    ev = res.evals[S.ft_steps]
    return {"evals": res.evals, "step": S.ft_steps, "learned": ev["p_o"] >= P_LEARN,
            "measured": harness.nothing_and_split(ctx, f"ft_step{S.ft_steps}", res.deltas[S.ft_steps], ev["p_o"], pile, log)}


# ----------------------------------------------------------------------------- the run and the score


def stage_all(args: argparse.Namespace) -> None:
    from commute import harness

    assert PRED.is_file(), "file predictions first: run.py --predictions"
    S = SMOKE if args.smoke else FULL
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    pile = bool(S.pile_rows)
    name = f"lora_matched{S.suffix}.json"
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S), "ref": REF,
                           "W0": {"pile": ctx.pile, "pgd_floor": harness.pgd_floor(ctx), "non_saturated": ctx.non_saturated}}
    out["ft"] = ft_arm(ctx, S, pile)
    dump(name, out)
    out["lora_s0"] = lora_arm(ctx, S, 0, pile)
    dump(name, out)
    out["lora_s1"] = lora_arm(ctx, S, 1, False)
    out["score"] = score(out)
    out["seconds"] = time.time() - t_start
    dump(name, out)
    log(f"wrote {RESULTS / name} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


def _scored(arm: dict[str, Any]) -> dict[str, Any]:
    return arm["steps"][str(arm["scored_step"] or max(int(s) for s in arm["steps"]))]


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    """A prediction record: its numbers, and `scored` = every condition in `cond` holds (gates, arms learned, gain >= 1)."""
    return {**values, "scored": all(cond.values()), "conditions": cond}


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules. A prediction is scored only if the arms it reads learned (G1), their lifts pass G2 and their gains
    are >= 1 nat (route_split 'scorable'); G4 misses are flagged in the verdict, not used to unscore."""
    ft, l0, l1 = out["ft"]["measured"], _scored(out["lora_s0"]), _scored(out["lora_s1"])

    def ok(step_rec: dict[str, Any], learned: bool) -> bool:
        return learned and bool(step_rec["G2"]["ok"]) and bool(step_rec["lift"]["routes"]["scorable"])

    ok_l0, ok_ft = ok(l0, out["lora_s0"]["learned"]), ok(ft, out["ft"]["learned"])
    s0, s1 = l0["lift"]["routes"]["shares"], l1["lift"]["routes"]["shares"]
    spread = max(abs(s0[k] - s1[k]) for k in ("d", "r"))
    m, fl = max(0.1, 3 * spread), out["W0"]["pgd_floor"]
    res: dict[str, Any] = {"G0": {"seed_spread": spread, "m": m, "pgd_floor": fl},
                           "G1": {"lora_s0": out["lora_s0"]["learned"], "lora_s1": out["lora_s1"]["learned"], "ft": out["ft"]["learned"]},
                           "G2": {"lora_s0": {st: b["G2"]["ok"] for st, b in out["lora_s0"]["steps"].items()},
                                  "lora_s1": {st: b["G2"]["ok"] for st, b in out["lora_s1"]["steps"].items()}, "ft": ft["G2"]["ok"]}}
    g2_all = all(res["G2"]["lora_s0"].values()) and all(res["G2"]["lora_s1"].values()) and res["G2"]["ft"]
    verdict = [] if g2_all else ["G2 failed: the lift or Delta bookkeeping is wrong"]
    if "pile" in ft["lift"]:
        g4 = {"nothing_ft_pgd_ratio": ft["nothing"]["pile"]["pgd_kl"], "lift_ft_pgd_ratio": ft["lift"]["pile"]["pgd_kl"]}
        g4["ok"] = {k: abs(v - REF[k]) <= 0.25 * REF[k] for k, v in g4.items()}
        res["G4"] = g4
        if not all(g4["ok"].values()):
            verdict.append("G4 missed on " + ", ".join(k for k, v in g4["ok"].items() if not v) + ": find the cause before the analysis")
    res["verdict"] = "; ".join(verdict) or "gates ok"

    inex, null = l0["inexpressible"]["pooled"], l0["inexpressible"]["pooled_null"]
    res["M1"] = _pred({"lora_s0": ok_l0}, pooled=inex, pooled_null=null, ratio=inex / null,
                      **{"pass": inex / null >= 0.8, "null": inex / null <= 0.6})
    e = s0["e_delta"]
    res["M2"] = _pred({"lora_s0": ok_l0}, e_delta=e, **{"pass": e >= 0.3, "null": e <= 0.2})
    if "pile" in l0["lift"] and "pile" in ft["lift"]:
        p_l, p_f = l0["lift"]["pile"]["pgd_kl"], ft["lift"]["pile"]["pgd_kl"]
        res["C2m"] = _pred({"lora_s0": ok_l0, "ft": ok_ft}, pgd_lora_lift=p_l, pgd_ft_lift=p_f, pgd_lora_nothing=l0["nothing"]["pile"]["pgd_kl"],
                           **{"pass": p_l >= 5, "null": p_l <= 2 and e <= 0.2})
    d_l, d_f = s0["d"], ft["lift"]["routes"]["shares"]["d"]
    res["C1m"] = _pred({"lora_s0": ok_l0, "ft": ok_ft}, d_lora_lift=d_l, d_ft_lift=d_f,
                       **{"pass": abs(d_l - d_f) < m, "null": d_l - d_f >= m})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--smoke", action="store_true")
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
