"""Reading-A coverage: VPD's own checks on the Adam pull-back's step-10 states for the ladder tasks that lack them.

Plan: reports/2026-10-19_plan_reading-a-coverage.md (design check confirmed by the user 2026-10-19; additions 2 and 3
accepted, the block floor kept because it is free). Source: reports/2026-10-19_decision_success-criterion.md.

Per task (the ladder's saved windows, experiments/task_ladder/results/large/windows_<task>.pt): the arm-A fine-tune
to the scored step (the do-nothing reference, delta-W in Delta) and the Adam pull-back on all alive (U, V) to the
same step, both regenerated with the ladder's recipe (lr, batch order seed); r2_t359 with three batch-order seeds.
Each state is scored on round 2's Pile harness (stochastic, PGD, CI-masked, unmasked KL as ratios to W0 on the
same rows), on VPD's importance-minimality (sparsity) loss on the same rows, and on the sparsity loss at the trigger
of the held-out windows (the control that shows the metric has power where Pile is blind). W0 floors come from the
same job on the same card: adversary seeds for the stochastic and PGD KL, block 0 vs block 1 of the rows for the
deterministic numbers (CI-masked, unmasked, sparsity), split halves of the held-out windows at the trigger.

Stages (repo root, project venv, GPU):
  all       W0 floors, then every task -> results/coverage.json (dumped after every task), scored at the end
  --smoke   tiny sizes, one task, two seeds -> results/coverage_smoke.json
  --task X  one task at seed 0, with its own same-job W0 floors -> results/coverage[_smoke]_X.json. `emoticon` is round 1's
            reference task (windows in transport_test), scored on Pile in round 2 on other rows and another card
            (pull-back 1.03, do-nothing 2.18 at step 10); added 2026-10-20 at the user's request
  --predictions   file results/predictions.json (refused if it exists); every stage refuses to start without it
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
LADDER = HERE.parent / "task_ladder" / "results"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

PB_LR = 1e-5  # the ladder's pull-back and fine-tune recipe
FT_LR = 3e-5
PGD_LINE = 1.5  # round 2's adversarial pass line
N_FLOORS = 3  # "within the floor" = ratio <= 1 + N_FLOORS * f
G1_TOL = 0.01  # reproduction of the ladder's held-out P(target) and kl_other
G1_PGD = {"r2_t359_v816": 1.532}  # the ladder's Pile PGD ratio (L40), reproduced within G1_PGD_TOL
G1_PGD_TOL = 0.25
RATIOS = ("stochastic_kl", "pgd_kl", "ci_masked_kl", "unmasked_kl", "sparsity")
FLOOR_OF = {"stochastic_kl": "seed", "pgd_kl": "seed", "ci_masked_kl": "block", "unmasked_kl": "block", "sparsity": "block"}
RUNG01 = ("r0_t352_v310", "r1_t534_v369", "r1_t588_v452")
RUNG2_SINGLE = "r2_t275_v2667"
RUNG2_SEEDS = "r2_t359_v816"
EMOTICON = "emoticon"


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run."""

    steps: int  # fine-tune and pull-back steps; the scored state is the last one
    pile_rows: int  # Pile val rows 0..pile_rows, in two blocks (round 2: 256 -> 2 x 128)
    pgd_steps: int
    w0_seeds: tuple[int, ...]  # adversary seeds at W0 (the seed floor)
    tasks: dict[str, tuple[int, ...]]  # task -> pull-back batch-order seeds
    suffix: str

    def __post_init__(self) -> None:
        assert self.pile_rows % 16 == 0 and len(self.w0_seeds) >= 2 and 0 in self.w0_seeds, \
            "two blocks of whole batches; seed 0 is the reference and at least one more seed gives the floor"
        assert all(0 in s for s in self.tasks.values()), "seed 0 is the ladder's state (gate G1)"


FULL = Sizes(steps=10, pile_rows=256, pgd_steps=20, w0_seeds=(0, 1, 2), suffix="",
             tasks={"r0_t352_v310": (0,), "r1_t534_v369": (0,), "r1_t588_v452": (0,), RUNG2_SINGLE: (0,),
                    RUNG2_SEEDS: (0, 1, 2)})
SMOKE = Sizes(steps=2, pile_rows=16, pgd_steps=2, w0_seeds=(0, 1), suffix="_smoke", tasks={RUNG2_SEEDS: (0, 1)})

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-19, before the run; plan reports/2026-10-19_plan_reading-a-coverage.md",
    "states": "per task, step 10 of the ladder's recipe: do-nothing (arm-A fine-tune, delta-W in Delta, released components, "
              "Gamma frozen) and the Adam pull-back on all alive (U, V) with Delta and Gamma frozen; seed = batch order",
    "measurements": {
        "pile": "round 2's harness, val rows 0..255 in two 128-row blocks, one shared PGD adversary per block (20 steps of "
                "0.1, seed 0), stochastic (2 draws), CI-masked and unmasked KL; each as a ratio to W0 on the same rows",
        "sparsity": "VPD's importance-minimality loss at the released run's end-of-training setting (p 0.4, beta 0.5, "
                    "eps 1e-12; faithfulness.sparsity_loss) on the same rows, ratio to W0",
        "sparsity_trigger": "the same loss on the upper-leaky CI values at the trigger position of the task's held-out "
                            "windows, ratio to W0",
        "floors": "W0, same card, same job: f_seed = max/min - 1 over adversary seeds 0, 1, 2 (stochastic, PGD); f_block = "
                  "|block0 - block1| / mean at seed 0 (CI-masked, unmasked, sparsity: deterministic under Gamma frozen); "
                  "f_trig = |half A - half B| / mean of the held-out windows at W0. 'Within the floor' = ratio <= 1 + 3 f",
        "blind_to": "every Pile number is blind to the trigger (about 3 trigger positions in 131k); the trigger is covered "
                    "by the ladder's task-local harness (not rerun) and by C1 here",
    },
    "gate": {"id": "G1_reproduces", "rule": "every seed-0 pull-back's held-out P(target) and kl_other at step 10 within 0.01 of "
                                           "the ladder's curve; r2_t359 seed 0's Pile PGD ratio within 25% of 1.532 (L40). "
                                           "Fails: nothing below is scored (the states are not the ladder's)"},
    "predictions": [
        {"id": "P1_rung01_pass", "arm": "pull-back, r0_t352, r1_t534, r1_t588",
         "rule": "each of the four KL ratios <= 1 + 3 f and PGD <= 1.5, on all three tasks", "predicted": "pass",
         "bears_on": "reading A on common-word tasks (r0_t588 read PGD 1.09)"},
        {"id": "P2_rung2_over_floor", "arm": "pull-back, r2_t275",
         "rule": "PGD ratio > 1 + 3 f_seed(pgd)", "predicted": "pass (expected >= 1.3: the largest off-target kl_other on the ladder)",
         "bears_on": "reading A on rare-word tasks"},
        {"id": "P3_r2_t359_excess_real", "arm": "pull-back, r2_t359, seeds 0, 1, 2",
         "rule": "all three PGD ratios > 1 + 3 f_seed(pgd)", "predicted": "pass",
         "reported": "the decision's three-way rule on the 1.5 line: all above = fails reading A on this task; all below = "
                     "the 1.53 was noise; split = mean and spread, no verdict. Also the state-level PGD floor (max/min - 1)",
         "bears_on": "whether the one recorded failure is seed noise"},
        {"id": "P4_sparsity_pile_within_floor", "arm": "every pull-back and do-nothing state",
         "rule": "Pile sparsity ratio within 1 +/- 3 f_block(sparsity)", "predicted": "pass",
         "bears_on": "the third VPD loss; read with C1 (blind, not broken)"},
        {"id": "P5_delta_absorbs", "arm": "do-nothing vs pull-back, seed 0, five tasks",
         "rule": "do-nothing PGD ratio > pull-back PGD ratio on every task", "predicted": "pass (round 2: 2.18 vs 1.03)",
         "bears_on": "'Delta not absorbing delta-W' is what the pull-back buys"},
    ],
    "control": {"id": "C1_sparsity_trigger_power", "arm": "every pull-back and do-nothing state",
                "rule": "sparsity ratio at the trigger of the held-out windows > 1 + 3 f_trig", "predicted": "pass",
                "why": "the concentration test counted 150-200 subcomponents switched on at the trigger; if this fails, P4 "
                       "says nothing about the metric"},
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
    return p


def require_predictions() -> dict[str, Any]:
    assert PRED.is_file(), f"file the predictions first ({PRED}; run.py --predictions)"
    return json.loads(PRED.read_text())


def spread(values: list[float]) -> float:
    return max(values) / min(values) - 1


# ----------------------------------------------------------------------------- loading and measuring


def load_all(device: str) -> dict[str, Any]:
    from commute import component_training, vpd

    model = vpd.load_component_model(device)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    return {"device": device, "model": model, "W0": vpd.target_weights(model), "alive": alive,
            "uv": component_training.alive_uv(model, alive), "sparsity_cfg": vpd.sparsity_config()}


def pile_blocks(S: Sizes) -> list[list[torch.Tensor]]:
    from commute.data import batches, pile_rows

    rows = pile_rows("val", 0, S.pile_rows, log)
    half = S.pile_rows // 2
    return [batches(rows[s : s + half], 8) for s in (0, half)]


def pile_measure(ctx: dict[str, Any], S: Sizes, seed: int) -> dict[str, list[float]]:
    """The four KL numbers and the sparsity loss of the model's current state, per block."""
    from commute.faithfulness import faithfulness, sparsity_loss

    per = [faithfulness(ctx["model"], b, seed=seed, pgd_steps=S.pgd_steps, log=log) for b in ctx["blocks"]]
    out = {k: [p[k] for p in per] for k in RATIOS if k != "sparsity"}
    out["sparsity"] = [sparsity_loss(ctx["model"], b, **ctx["sparsity_cfg"]) for b in ctx["blocks"]]
    return out


def trigger_sparsity(ctx: dict[str, Any], tokens: torch.Tensor, half: int) -> float:
    """VPD's importance-minimality loss on the upper-leaky CI values at position `half` of each window (the trigger),
    the authors' update/compute pair as in faithfulness.sparsity_loss, restricted to that position."""
    from param_decomp.metrics.importance_minimality_loss import (
        _importance_minimality_loss_compute,
        _importance_minimality_loss_update,
    )

    model, cfg = ctx["model"], ctx["sparsity_cfg"]
    sums: dict[str, torch.Tensor] = {}
    n = 0
    with torch.no_grad():
        for s in range(0, tokens.shape[0], 64):
            sub = tokens[s : s + 64].to(ctx["device"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=sub.is_cuda):
                out = model(sub, cache_type="input")
                ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
            upper = {k: v[:, half].float() for k, v in ci.upper_leaky.items()}
            part, m = _importance_minimality_loss_update(upper, cfg["pnorm"], cfg["eps"], 1.0, None, 1.0, 1.0)
            for k, v in part.items():
                sums[k] = sums[k] + v if k in sums else v
            n += m
            del out, ci, upper
    return float(_importance_minimality_loss_compute(sums, n, cfg["beta"], world_size=1))


def stage_w0(ctx: dict[str, Any], S: Sizes) -> dict[str, Any]:
    """W0 on the Pile rows under every adversary seed; the seed and block floors."""
    per_seed = {s: pile_measure(ctx, S, s) for s in S.w0_seeds}
    mean = {k: sum(per_seed[0][k]) / 2 for k in RATIOS}  # the reference for every ratio: seed 0, mean of the blocks
    f_seed = {k: spread([sum(v[k]) / 2 for v in per_seed.values()]) for k in RATIOS}
    f_block = {k: abs(per_seed[0][k][0] - per_seed[0][k][1]) / mean[k] for k in RATIOS}
    floor = {k: (f_seed if FLOOR_OF[k] == "seed" else f_block)[k] for k in RATIOS}
    log(f"[W0] mean {json.dumps({k: round(v, 4) for k, v in mean.items()})} f_seed {json.dumps({k: round(v, 4) for k, v in f_seed.items()})}"
        f" f_block {json.dumps({k: round(v, 4) for k, v in f_block.items()})}")
    return {"per_seed": {str(s): v for s, v in per_seed.items()}, "mean": mean, "f_seed": f_seed, "f_block": f_block, "floor": floor}


def ratios(ctx: dict[str, Any], S: Sizes, w0: dict[str, Any]) -> dict[str, Any]:
    m = pile_measure(ctx, S, 0)
    r = {k: (sum(m[k]) / 2) / w0["mean"][k] for k in RATIOS}
    return {"raw": m, "ratio": r}


# ----------------------------------------------------------------------------- per task


def load_windows(name: str) -> tuple[Any, Any, int]:
    """The task's train and held-out windows and its target token: the ladder's saved windows, or round 1's emoticon
    windows (the ladder's reference rung, same format)."""
    if name == EMOTICON:
        w = torch.load(ROUND1 / "windows.pt", weights_only=False)
        return w["train"], w["heldout"], json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    w = torch.load(LADDER / "large" / f"windows_{name}.pt", weights_only=False)
    return w["train"], w["heldout"], w["task"]["target"]


def a_cfg(target: int, steps: int, seed: int) -> Any:
    from commute.finetune import ArmAConfig

    return ArmAConfig(o_token=target, lr=FT_LR, steps=steps, checkpoints=(steps,), seed=seed)


def run_task(ctx: dict[str, Any], S: Sizes, name: str, seeds: tuple[int, ...], w0: dict[str, Any]) -> dict[str, Any]:
    from commute import component_training
    from commute.faithfulness import model_state
    from commute.finetune import finetune_arm_a

    model, W0 = ctx["model"], ctx["W0"]
    train, held, target = load_windows(name)
    half, toks = held.half, held.tokens
    t0 = time.time()
    n2 = toks.shape[0] // 2  # the floor compares two halves of equal size (an odd window is left out of both)
    trig0 = trigger_sparsity(ctx, toks, half)
    halves = (trigger_sparsity(ctx, toks[:n2], half), trigger_sparsity(ctx, toks[n2 : 2 * n2], half))
    f_trig = abs(halves[0] - halves[1]) / (sum(halves) / 2)
    rec: dict[str, Any] = {"target": target, "n_train": len(train), "n_heldout": len(held),
                           "W0": {"sparsity_trigger": trig0, "halves": halves, "n_half": n2, "f_trig": f_trig}, "seeds": {}}
    (LARGE / name).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        ft = finetune_arm_a(model, train, held, a_cfg(target, S.steps, seed), ctx["device"], log)
        dW = ft.deltas[S.steps]
        torch.save(dW, LARGE / name / f"dW_ft_step{S.steps}_s{seed}.pt")
        nothing: dict[str, Any] = {"heldout": ft.evals[S.steps]}
        with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}):
            nothing["sparsity_trigger"] = trigger_sparsity(ctx, toks, half) / trig0
            if seed == 0:
                nothing["pile"] = ratios(ctx, S, w0)
        cfg = component_training.TrainConfig(lr=PB_LR, steps=S.steps, save_steps=(S.steps,))
        curve, states = component_training.train(model, ctx["uv"], train.tokens, held, a_cfg(target, S.steps, seed), cfg,
                                       ctx["device"], W0, save_dir=LARGE / name / f"pullback_s{seed}", log=log)
        st = states[S.steps]
        comps = component_training.components_with(model, ctx["alive"], {k: (st[k][1], st[k][2]) for k in st})
        pb: dict[str, Any] = {"heldout": {"p_o": curve[-1]["heldout_p_o"], "kl_other": curve[-1]["heldout_kl_other"]}}
        with model_state(model, target={k: W0[k] + st[k][0].to(W0[k].device) for k in W0}, components=comps):
            pb["sparsity_trigger"] = trigger_sparsity(ctx, toks, half) / trig0
            pb["pile"] = ratios(ctx, S, w0)
        rec["seeds"][str(seed)] = {"do_nothing": nothing, "pullback": pb}
        log(f"task {name} seed {seed}: pull-back P {pb['heldout']['p_o']:.4f} kl_other {pb['heldout']['kl_other']:.4f} "
            f"pile {json.dumps({k: round(v, 3) for k, v in pb['pile']['ratio'].items()})} trigger sparsity {pb['sparsity_trigger']:.3f}"
            + (f" | do-nothing pile {json.dumps({k: round(v, 3) for k, v in nothing['pile']['ratio'].items()})}" if "pile" in nothing else "")
            + f" do-nothing trigger sparsity {nothing['sparsity_trigger']:.3f} | f_trig {f_trig:.4f}")
    rec["seconds"] = time.time() - t0
    return rec


# ----------------------------------------------------------------------------- scoring


def score(out: dict[str, Any], S: Sizes) -> dict[str, Any]:
    tasks, floor = out["tasks"], out["W0"]["floor"]
    ladder = json.loads((LADDER / "ladder.json").read_text())["tasks"]
    line = {k: 1 + N_FLOORS * floor[k] for k in RATIOS}

    def pb(name: str, seed: int = 0) -> dict[str, Any]:
        return tasks[name]["seeds"][str(seed)]["pullback"]

    def states() -> list[tuple[str, str, dict[str, Any]]]:
        return [(n, f"{arm}_s{s}", v[arm]) for n, t in tasks.items() for s, v in t["seeds"].items()
                for arm in ("pullback", "do_nothing") if "pile" in v[arm]]

    # G1: the regenerated seed-0 pull-backs are the ladder's
    g1: dict[str, Any] = {}
    for name in tasks:
        c = next(x for x in ladder[name]["pullback"]["curve"] if x["step"] == S.steps)
        h = pb(name)["heldout"]
        ft, ft0 = tasks[name]["seeds"]["0"]["do_nothing"]["heldout"], ladder[name]["finetune"]["evals"].get(str(S.steps))
        g1[name] = {"dP": h["p_o"] - c["heldout_p_o"], "dkl": h["kl_other"] - c["heldout_kl_other"],
                    "dP_finetune": ft["p_o"] - ft0["p_o"] if ft0 else None}  # the do-nothing state, reported only
    if RUNG2_SEEDS in tasks:
        g1["pgd_rel"] = pb(RUNG2_SEEDS)["pile"]["ratio"]["pgd_kl"] / G1_PGD[RUNG2_SEEDS] - 1
    g1["pass"] = all(abs(v["dP"]) <= G1_TOL and abs(v["dkl"]) <= G1_TOL for k, v in g1.items() if isinstance(v, dict)) and \
        abs(g1.get("pgd_rel", 0.0)) <= G1_PGD_TOL
    if not g1["pass"]:
        return {"G1_reproduces": g1, "line": line, "status": "unscored: the regenerated states are not the ladder's"}

    def within(r: dict[str, float], k: str) -> bool:
        return r[k] <= line[k] and (k != "pgd_kl" or r[k] <= PGD_LINE)

    res: dict[str, Any] = {"G1_reproduces": g1, "line": line, "status": "scored"}
    have = [n for n in RUNG01 if n in tasks]
    if have:
        p1: dict[str, Any] = {n: {k: within(pb(n)["pile"]["ratio"], k) for k in RATIOS if k != "sparsity"} for n in have}
        p1["pass"] = all(all(v.values()) for n, v in p1.items() if n != "pass")
        res["P1_rung01_pass"] = p1
    if RUNG2_SINGLE in tasks:
        r = pb(RUNG2_SINGLE)["pile"]["ratio"]["pgd_kl"]
        res["P2_rung2_over_floor"] = {"pgd_ratio": r, "pass": r > line["pgd_kl"]}
    if RUNG2_SEEDS in tasks:
        rs = [pb(RUNG2_SEEDS, int(s))["pile"]["ratio"]["pgd_kl"] for s in tasks[RUNG2_SEEDS]["seeds"]]
        verdict = "fails" if all(r > PGD_LINE for r in rs) else "noise" if all(r <= PGD_LINE for r in rs) else "split"
        res["P3_r2_t359_excess_real"] = {"pgd_ratios": rs, "mean": sum(rs) / len(rs), "state_floor": spread(rs),
                                         "three_way_1.5": verdict, "pass": all(r > line["pgd_kl"] for r in rs)}
    sp = {f"{n}/{arm}": v["pile"]["ratio"]["sparsity"] for n, arm, v in states()}
    res["P4_sparsity_pile_within_floor"] = {"ratios": sp, "pass": all(abs(v - 1) <= N_FLOORS * floor["sparsity"] for v in sp.values())}
    trig = {f"{n}/{arm}": (v["sparsity_trigger"], tasks[n]["W0"]["f_trig"]) for n, arm, v in states()}
    res["C1_sparsity_trigger_power"] = {"ratios": {k: v[0] for k, v in trig.items()},
                                        "pass": all(v > 1 + N_FLOORS * f for v, f in trig.values())}
    p5 = {n: (t["seeds"]["0"]["do_nothing"]["pile"]["ratio"]["pgd_kl"], t["seeds"]["0"]["pullback"]["pile"]["ratio"]["pgd_kl"])
          for n, t in tasks.items()}
    res["P5_delta_absorbs"] = {"do_nothing_vs_pullback": p5, "pass": all(a > b for a, b in p5.values())}
    return res


# ----------------------------------------------------------------------------- main


def stage_all(S: Sizes, device: str) -> None:
    require_predictions()
    ctx = load_all(device)
    ctx["blocks"] = pile_blocks(S)
    out: dict[str, Any] = {"sizes": S.__dict__ | {"tasks": S.tasks}, "sparsity_cfg": ctx["sparsity_cfg"]}
    t0 = time.time()
    out["W0"] = stage_w0(ctx, S)
    log(f"[W0] {time.time() - t0:.0f} s for {len(S.w0_seeds)} seeds (the slice benchmark: one Pile pass + sparsity per seed)")
    out["tasks"] = {}
    dump(f"coverage{S.suffix}.json", out)
    for name, seeds in S.tasks.items():
        out["tasks"][name] = run_task(ctx, S, name, seeds, out["W0"])
        dump(f"coverage{S.suffix}.json", out)
    out["score"] = score(out, S)
    dump(f"coverage{S.suffix}.json", out)
    log(f"score: {json.dumps(out['score'], indent=1, default=float)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", help="one task at seed 0 (e.g. emoticon) -> results/coverage[_smoke]_<task>.json")
    args = ap.parse_args()
    if args.predictions:
        assert not PRED.is_file(), f"{PRED} exists; never edit filed predictions"
        print(dump("predictions.json", PREDICTIONS))
        return
    S = SMOKE if args.smoke else FULL
    if args.task:
        S = replace(S, tasks={args.task: (0,)}, suffix=f"{S.suffix}_{args.task}")
    stage_all(S, args.device)


if __name__ == "__main__":
    main()
