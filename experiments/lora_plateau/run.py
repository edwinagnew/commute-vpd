"""lora_plateau: round 5's LoRA trained to its plateau (1000 steps), the outside_share share tracked against A's movement.

Round 5 (`experiments/lora_matched`) found the rank-8 LoRA's read factor A at its random initialisation at the step it
learns (|cos| 1.000 at step 2, 0.973 at step 100) and its update as outside_share as a random one on the 20 non-saturated
matrices (pooled share / isotropic null 0.86 at step 2, 0.82 at step 100). This run asks whether that is a property of
the step the LoRA learns at or of the LoRA for its whole training: the same recipe (rank 8 on all 24 matrices, Adam lr
1e-3, arm A's loss, seed 0) run to 1000 steps, with the share, its null and A's overlap with the alive read span at every
checkpoint. It also runs the step-matched Pile comparison that round 5's report left open: the lift and do-nothing Pile
ratios at steps 10 (the fine-tune's matched step), 100 and 1000. Design check confirmed by the user 2026-10-19 (lr as is,
Pile folded in, seed 0 only).

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  all             GPU -> results/lora_plateau.json (dumped after every stage) + results/large/ (LoRA factors, log)
  all --smoke     GPU, minutes: every path at the SMOKE sizes -> results/lora_plateau_smoke.json
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt} and results/windows.json (round 1).

Usage (repo root):  uv run python experiments/lora_plateau/run.py all
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

RANK, LORA_LR, P_LEARN, TOL = 8, 1e-3, 0.9, 1e-6
REF = {  # round 5 (lora_matched.json, L40, same harness): the fine-tune's data-shaped read side, and the LoRA so far
    "ft_lift_step10": {"inexpressible_ratio": 0.268 / 0.531, "pgd_ratio": 5.41, "nothing_pgd_ratio": 2.18, "e_delta": 0.090},
    "lora_step2": {"inexpressible_ratio": 0.478 / 0.553, "cos_init": 1.000, "pgd_ratio": 1.13, "nothing_pgd_ratio": 1.17, "e_delta": 0.218},
    "lora_step100": {"inexpressible_ratio": 0.550 / 0.674, "cos_init": 0.973, "e_delta": 0.107},
    "pgd_floor": 0.209,
}


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run."""

    lora_steps: int
    lora_ckpts: tuple[int, ...]  # the share, its null, A's overlap and |cos| at each; the last is the plateau
    pile_steps: tuple[int, ...]  # checkpoints measured do-nothing and lifted with the Pile harness
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    suffix: str

    def __post_init__(self) -> None:
        assert self.lora_ckpts == tuple(sorted(self.lora_ckpts)) and self.lora_ckpts[-1] == self.lora_steps, self.lora_ckpts
        assert set(self.pile_steps) <= set(self.lora_ckpts) and len(self.lora_ckpts) >= 3, self.pile_steps
        assert bool(self.pile_rows) == bool(self.pile_seeds)


FULL = Sizes(lora_steps=1000, lora_ckpts=(1, 2, 3, 5, 10, 30, 100, 300, 1000), pile_steps=(10, 100, 1000), pile_rows=256,
             pile_pgd_steps=20, pile_seeds=(0, 1, 2), suffix="")
SMOKE = Sizes(lora_steps=3, lora_ckpts=(1, 2, 3), pile_steps=(2, 3), pile_rows=8, pile_pgd_steps=2, pile_seeds=(0,), suffix="_smoke")

PREDICTIONS: dict[str, Any] = {
    "about": (
        "lora_plateau (2026-10-19): round 5's rank-8 LoRA on all 24 matrices (same recipe, seed 0) trained to 1000 steps. "
        "Does the read factor A ever become data-shaped, so that the update becomes expressible by the released "
        "subcomponents, or does the LoRA's update stay as outside_share as a random one for its whole training? And the "
        "step-matched Pile comparison round 5 left open: the lift's Pile ratios at steps 10, 100 and 1000. Design check "
        "confirmed by the user 2026-10-19. Filed before training."
    ),
    "setup": {
        "lora": f"as lora_matched: rank {RANK} on all 24 matrices, B = 0 and A ~ N(0, 1/d_in) (seed 0), Adam lr {LORA_LR}, arm A's loss, "
                f"batch 32, clip 1, {FULL.lora_steps} steps, checkpoints {FULL.lora_ckpts}; no fine-tune arm (round 5's numbers are the reference)",
        "lift": "split.split_state, Euclidean, tol 1e-6, at the Pile steps only",
        "W0": f"the released decomposition under round 2's Pile harness with adversary seeds {FULL.pile_seeds}",
    },
    "measurements": {
        "inexpressible": "harness.outside_share at every checkpoint: per matrix with its isotropic null, pooled by energy over the "
                         "non-saturated matrices; the tracked number is ratio = pooled / pooled null, since the null moves with the energy mix",
        "read_side": "split.read_side_overlaps of delta-W at every checkpoint: in_alive = the share of ||delta-W||_F^2 whose read side lies "
                     "in the alive read span (a random read side gives r_v / d_in), pooled by energy over the non-saturated matrices",
        "cos_init": "per matrix, mean |cos| of A's rows with their initialisation, at every checkpoint",
        "plateau": "mean training loss over the steps between the last two checkpoints, divided by the mean over the steps between the "
                   "two before; held-out P(o) and kl_other at every checkpoint",
        "pile_steps": f"at steps {FULL.pile_steps}: do-nothing and lifted route split, Delta's share of ||delta-W||^2, and the Pile ratios "
                      "(2 x 128 val rows, 20 PGD steps of 0.1), each divided by W0's under adversary seed 0",
    },
    "expected_at": {
        "step_2_and_100": "ratio 0.86 and 0.82, |cos| 1.000 and 0.973, e_delta 0.22 and 0.11 (round 5; the trajectory must pass through them)",
        "control": "the fine-tune's data-shaped read side at step 10: ratio 0.51, lift PGD 5.41, do-nothing 2.18, e_delta 0.09 (round 5, L40)",
        "random_read_side": "in_alive = r_v / d_in per matrix; ratio 1",
        "W0": "every Pile ratio 1 by construction; floor f about 0.21 (round 5, L40), so within 3 floors = ratio <= 1.63",
    },
    "blind_to": {
        "ratio": "nothing about behaviour: an outside_share part can still be redundant; and the four saturated matrices are left out, "
                 "so energy moving onto them lowers nothing here",
        "in_alive": "the write side: a read side in the alive span can still write outside col U_A",
        "Pile PGD ratio": "a trigger-specific attack; the 20-step search is chaotic across GPUs, so the run is read on the L40 only",
        "controls": "G2 (the lifted state is the LoRA's model exactly); round 5's fine-tune numbers on the same GPU type and harness, not rerun",
    },
    "gates": [
        {"id": "G1_learned", "rule": f"held-out P(o) >= {P_LEARN} at some checkpoint (round 5: step 2)", "if_fail": "reported, not scored"},
        {"id": "G2_lift_exact", "rule": "for every lift: |P(o) of the lifted state - the LoRA's held-out P(o)| <= 1e-3 and |sum of shares - 1| <= 1e-6",
         "if_fail": "the lift or the Delta bookkeeping is wrong; stop"},
        {"id": "G3_plateau", "rule": "plateau ratio >= 0.9 (the training loss fell by less than 10% between the last two checkpoint "
                                     f"intervals) and P(o) >= {P_LEARN} at the last two checkpoints",
         "if_fail": "P1 is scored at the last checkpoint anyway and the miss is reported: the plateau was not reached"},
    ],
    "predictions": [
        {"id": "P1_read_side_stays_random", "scored": True, "arm": f"step {FULL.lora_steps}",
         "pass": "ratio >= 0.8 (as at steps 2 and 100)", "null": "ratio <= 0.6 (the fine-tune's 0.51: A became data-shaped)",
         "predicted": "pass: between steps 2 and 100 A moved 3% and the ratio moved 0.04 the other way; Adam at lr 1e-3 on a factor "
                      "whose gradient is B^T-weighted barely moves A once B has learned, so the read side stays random to the plateau",
         "reported": "ratio, pooled in_alive and |cos| at every checkpoint; per-matrix in_alive against r_v / d_in"},
        {"id": "P2_lift_survives_step_matched", "scored": True, "arm": "LoRA lift at step 10 (the fine-tune's matched step)",
         "pass": "Pile PGD ratio <= 2 and e_delta <= 0.2 (the lifted LoRA state survives the adversary where the fine-tune's lift reads 5.41)",
         "null": "Pile PGD ratio >= 5 (fragile like the fine-tune's lift: round 5's C2m direction was the step, not the update)",
         "predicted": "pass: at step 2 the LoRA's update was already twice the fine-tune's step-10 update in relative norm and its lift read "
                      "1.13, so the fine-tune lift's fragility is not the update's size; what the LoRA's lift leaves in Delta is the "
                      "random-read part, and the subcomponents move by what they can read",
         "reported": "the do-nothing ratio at the same step (the floor the lift is read against); the same numbers at step 100"},
        {"id": "P3_lift_survives_at_plateau", "scored": True, "arm": f"LoRA lift at step {FULL.lora_steps}",
         "pass": "Pile PGD ratio <= 2 and e_delta <= 0.2", "null": "Pile PGD ratio >= 5",
         "predicted": "pass, weakly: the same mechanism as P2 at an update 3-5 times larger; if P2 passes and P3 reads >= 5, "
                      "the fragility is the update's size after all and arrives between steps 10 and 1000",
         "reported": "the do-nothing ratio at the plateau; Delta's share of ||delta-W||^2"},
    ],
    "reported_not_scored": "P(o), kl_other, loss and relative norm per checkpoint; the per-matrix tables; the route split at the Pile steps",
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


# ----------------------------------------------------------------------------- the run


def train(ctx: Any, S: Sizes) -> Any:
    """Round 5's LoRA recipe, seed 0, to S.lora_steps; the factors go to results/large."""
    from commute import vpd
    from commute.finetune import ArmAConfig, LoraConfig, finetune_lora_arm_a

    modules = tuple(vpd.module_keys(ctx.model))
    cfg = LoraConfig(modules=modules, rank=RANK, lr=LORA_LR, steps=S.lora_steps, checkpoints=S.lora_ckpts, seed=0)
    a_cfg = ArmAConfig(o_token=ctx.token, steps=S.lora_steps, checkpoints=(), seed=0)
    res = finetune_lora_arm_a(ctx.model, ctx.train, ctx.held, a_cfg, cfg, ctx.device, log)
    torch.save(res.factors, LARGE / "lora_factors.pt")
    return res


def read_side(ctx: Any, dW: dict[str, Tensor]) -> dict[str, Any]:
    """split.read_side_overlaps per non-zero matrix, and in_alive pooled by update energy over the non-saturated ones."""
    from commute import split

    per = {k: split.read_side_overlaps(M, ctx.bases[k], ctx.sigma[k]) for k, M in dW.items() if M.any()}
    ks = [k for k in per if k in ctx.non_saturated]
    e = {k: float(((dW[k].double() @ ctx.sigma[k].double()) * dW[k].double()).sum()) for k in ks}
    tot = sum(e.values())
    return {"per_matrix": per, "pooled_in_alive": sum(per[k]["in_alive"] * e[k] for k in ks) / tot,
            "pooled_random": sum(per[k]["random"] * e[k] for k in ks) / tot}


def checkpoint_record(ctx: Any, res: Any, step: int, A0: dict[str, Tensor]) -> dict[str, Any]:
    """What every checkpoint reads: the LoRA's held-out numbers, the outside_share share and its null, A's overlap with the
    alive read span, and |cos| of A with its initialisation."""
    from commute import harness

    cos = {k: float(torch.nn.functional.cosine_similarity(f[1], A0[k], dim=1).abs().mean()) for k, f in res.factors[step].items()}
    inex = harness.outside_share(ctx, res.deltas[step])
    rec = {"eval": res.evals[step], "inexpressible": inex, "ratio": inex["pooled"] / inex["pooled_null"],
           "read_side": read_side(ctx, res.deltas[step]), "cos_init": {"mean": sum(cos.values()) / len(cos), "per_matrix": cos}}
    log(f"[step {step}] ratio {rec['ratio']:.3f} (pooled {inex['pooled']:.3f} null {inex['pooled_null']:.3f}) "
        f"in_alive {rec['read_side']['pooled_in_alive']:.3f} (random {rec['read_side']['pooled_random']:.3f}) "
        f"cos_init {rec['cos_init']['mean']:.3f} P(o) {res.evals[step]['p_o']:.3f} kl_other {res.evals[step]['kl_other']:.4f}")
    return rec


def plateau_ratio(train_log: list[dict[str, Any]], ckpts: tuple[int, ...]) -> float:
    """Mean loss over the steps in (ckpts[-2], ckpts[-1]] divided by the mean over (ckpts[-3], ckpts[-2]]."""
    loss = {r["step"]: r["loss"] for r in train_log}
    last = [loss[s] for s in range(ckpts[-2] + 1, ckpts[-1] + 1)]
    before = [loss[s] for s in range(ckpts[-3] + 1, ckpts[-2] + 1)]
    return (sum(last) / len(last)) / (sum(before) / len(before))


def stage_all(args: argparse.Namespace) -> None:
    from commute import harness, vpd
    from commute.finetune import lora_factors

    assert PRED.is_file(), "file predictions first: run.py --predictions"
    S = SMOKE if args.smoke else FULL
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    name = f"lora_plateau{S.suffix}.json"
    out: dict[str, Any] = {"smoke": args.smoke, "sizes": asdict(S), "ref": REF,
                           "W0": {"pile": ctx.pile, "pgd_floor": harness.pgd_floor(ctx), "non_saturated": ctx.non_saturated}}
    res = train(ctx, S)
    modules = list(vpd.module_keys(ctx.model))
    A0 = {k: f[1].detach().cpu() for k, f in lora_factors({k: vpd.target_linear(ctx.model, k) for k in modules}, RANK, 0).items()}
    out["train_log"] = res.log
    out["plateau_ratio"] = plateau_ratio(res.log, S.lora_ckpts)
    out["checkpoints"] = {str(s): checkpoint_record(ctx, res, s, A0) for s in S.lora_ckpts}
    dump(name, out)
    out["pile_steps"] = {}
    for step in S.pile_steps:
        out["pile_steps"][str(step)] = harness.nothing_and_split(ctx, f"lora_step{step}", res.deltas[step], res.evals[step]["p_o"],
                                                                bool(S.pile_rows), log)
        dump(name, out)
    out["score"] = score(out)
    out["seconds"] = time.time() - t_start
    dump(name, out)
    log(f"wrote {RESULTS / name} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


# ----------------------------------------------------------------------------- the score


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    return {**values, "scored": all(cond.values()), "conditions": cond}


def _survives(rec: dict[str, Any], learned: bool) -> dict[str, Any]:
    """P2 / P3: the lifted state's Pile PGD ratio against the fine-tune's 5.41 and the 'survives' line, with the do-nothing beside."""
    lift, nothing = rec["lift"], rec["nothing"]
    pgd, e = lift["pile"]["pgd_kl"], lift["routes"]["shares"]["e_delta"]
    return _pred({"learned": learned, "G2": bool(rec["G2"]["ok"]), "scorable": bool(lift["routes"]["scorable"])},
                 pgd_lift=pgd, pgd_nothing=nothing["pile"]["pgd_kl"], e_delta=e, delta_change=lift["delta_change_frac"],
                 **{"pass": pgd <= 2 and e <= 0.2, "null": pgd >= 5})


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules. P1 is scored at the last checkpoint if the LoRA learned; P2 at the first Pile step and P3 at the last,
    if their lifts pass G2 and the gain is >= 1 nat (route_split 'scorable')."""
    ck, S = out["checkpoints"], out["sizes"]
    ckpts, pile_steps = S["lora_ckpts"], S["pile_steps"]
    scored_step = next((s for s in ckpts if ck[str(s)]["eval"]["p_o"] >= P_LEARN), None)
    learned = scored_step is not None
    last = ck[str(ckpts[-1])]
    res: dict[str, Any] = {
        "G1": {"learned": learned, "scored_step": scored_step},
        "G2": {st: b["G2"]["ok"] for st, b in out["pile_steps"].items()},
        "G3": {"plateau_ratio": out["plateau_ratio"], "p_o_last_two": [ck[str(s)]["eval"]["p_o"] for s in ckpts[-2:]]},
    }
    res["G3"]["ok"] = res["G3"]["plateau_ratio"] >= 0.9 and all(p >= P_LEARN for p in res["G3"]["p_o_last_two"])
    verdict = [] if all(res["G2"].values()) else ["G2 failed: the lift or Delta bookkeeping is wrong"]
    if not res["G3"]["ok"]:
        verdict.append("G3 missed: the plateau was not reached; P1 is scored at the last checkpoint anyway")
    res["verdict"] = "; ".join(verdict) or "gates ok"
    ratio = last["ratio"]
    res["P1"] = _pred({"learned": learned}, ratio=ratio, cos_init=last["cos_init"]["mean"], in_alive=last["read_side"]["pooled_in_alive"],
                      in_alive_random=last["read_side"]["pooled_random"], **{"pass": ratio >= 0.8, "null": ratio <= 0.6})
    if out["pile_steps"] and "pile" in out["pile_steps"][str(pile_steps[0])]["lift"]:
        res["P2"] = _survives(out["pile_steps"][str(pile_steps[0])], learned)
        res["P3"] = _survives(out["pile_steps"][str(pile_steps[-1])], learned)
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
