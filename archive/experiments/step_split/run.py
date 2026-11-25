"""step_split (E19): the step split along the emoticon fine-tune's own path (plan: reports/plans/2026-10-21_plan_step-split.md).

The fine-tune is E14/E16's (arm A, Adam 3e-5, seed 0, 1000 steps). After every optimizer step t, per decomposed matrix, the
actual weight step S_t is split by `split.StepSplit`: the components take the scaled adjoint part c (S_t V, S_t^T U)
against the CURRENT alive (U, V), c refitted every step on the W0 input moments (`split.adjoint_step`); Delta takes the
remainder, including the second-order term dU dV^T, so components plus Delta are the fine-tune's weights at every step.
Gamma frozen. Beside it, for the consistency numbers only (not scored by the harness), the same with the bases held at
W0's (`carry=False`), which with one fixed c is the one-shot adjoint split (tests/test_step_split.py).

Stages (results/predictions.json must exist; the full run also needs it filed, "status": "filed"):
  all            GPU: the W0 control (Pile floor over 3 attacker seeds, early stop), the fine-tune with both step splits,
                 then at steps 10, 100, 1000 the step-split state, do-nothing and the least-squares split through
                 harness.measure -> results/step_split.json (dumped after every state), results/large/{per_step.json, full.log}
  all --smoke    3 steps, states at 2 and 3, 8 Pile rows, 2 PGD steps -> results/step_split_smoke.json
  all --resume   re-run the fine-tune and score only the states missing from the results file
Inputs: experiments/transport_test/results/large/{alive.pt, windows.pt}, results/windows.json (round 1);
experiments/split_comparison/results/split_comparison.json (E16, for the G1 reading).
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
E16 = HERE.parent / "split_comparison" / "results" / "split_comparison.json"
FT_LR, TOL = 3e-5, 1e-6  # arm A's recipe and the split tolerance, as E14 and E16
DTYPE = torch.float32  # the step split's arithmetic on the GPU; the CPU tests run it in float64


@dataclass(frozen=True)
class Sizes:
    steps: tuple[int, ...]  # the scored steps
    train_steps: int
    pile_rows: int
    pile_pgd_steps: int
    pile_seeds: tuple[int, ...]
    floor_stop: float  # the W0 control: stop if the PGD floor f = max/min - 1 over the seeds exceeds this
    suffix: str

    def __post_init__(self) -> None:
        assert self.steps == tuple(sorted(self.steps)) and self.steps[-1] == self.train_steps, self.steps
        assert self.pile_rows > 0 and len(self.pile_seeds) >= 2, "the floor needs two adversary seeds"


FULL = Sizes(steps=(10, 100, 1000), train_steps=1000, pile_rows=256, pile_pgd_steps=20, pile_seeds=(0, 1, 2),
             floor_stop=0.25, suffix="")  # amendment 1: the L40 floor is 0.209 (E8, E11, E19 job 28108)
SMOKE = Sizes(steps=(2, 3), train_steps=3, pile_rows=8, pile_pgd_steps=2, pile_seeds=(0, 1), floor_stop=math.inf,
              suffix="_smoke")


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


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))


# ----------------------------------------------------------------------------- the fine-tune with both step splits


def out_out_energy(table: dict[str, Any]) -> float:
    """The absolute input-weighted energy of a `split.block_table`'s outside-both block, summed over matrices."""
    return sum(p["out_out"] * p["energy"] for p in table["per_matrix"].values())


def snapshot(ctx: Any, carry: Any, fixed: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """At a scored step: the step-split state for the harness, and the split's own numbers (all on CPU, float64)."""
    from commute import split

    state = carry.state()
    dW = {k: v[0] for k, v in state.items()}
    dd = {k: v.double().cpu() for k, v in carry.delta_change().items()}
    dd_fix = {k: v.double().cpu() for k, v in fixed.delta_change().items()}
    sg = ctx.sigma
    e_dW = sum(split.weighted_energy(dW[k].double(), sg[k]) for k in dW)
    e_dd = sum(split.weighted_energy(dd[k], sg[k]) for k in dd)
    e_dd_fix = sum(split.weighted_energy(dd_fix[k], sg[k]) for k in dd)
    e_second = sum(split.weighted_energy(carry.second[k].double().cpu(), sg[k]) for k in dd)
    n_dW = sum(float((dW[k].double() ** 2).sum()) for k in dW)
    one_shot_R, gap = 0.0, 0.0  # the one-shot adjoint split of the same delta-W (E16's adjoint state) beside the fixed-bases pieces
    for k, M in dW.items():
        U0, V0 = ctx.uv_cpu[k][0].double(), ctx.uv_cpu[k][1].double()
        c, R = split.adjoint_scale(M, U0, V0, sg[k])
        aU, aV = split.adjoint_split(M, U0, V0)
        one_shot_R += R * split.weighted_energy(M.double(), sg[k]) / e_dW
        gap = max(gap, float((fixed.dU[k].double().cpu() - c * aU).norm() / (c * aU).norm()),
                  float((fixed.dV[k].double().cpu() - c * aV).norm() / (c * aV).norm()))
    blocks_dd = split.block_table({k: v.float() for k, v in dd.items()}, ctx.bases, sg, ctx.non_saturated)
    blocks_dW = split.block_table(dW, ctx.bases, sg, ctx.non_saturated)
    rec = {"delta_share": sum(float((v**2).sum()) for v in dd.values()) / n_dW,  # unweighted, as harness's delta_change_frac
           "delta_share_fixed": sum(float((v**2).sum()) for v in dd_fix.values()) / n_dW,
           "delta_share_weighted": e_dd / e_dW, "delta_share_weighted_fixed": e_dd_fix / e_dW,
           "delta_share_weighted_one_shot": 1 - one_shot_R, "fixed_vs_one_shot_piece_gap": gap,
           "second_share_of_delta": e_second / e_dd, "piece_change_max": carry.piece_change_max(),
           "piece_change_max_fixed": fixed.piece_change_max(),
           "out_out_delta": blocks_dd["pooled_all"], "out_out_update": blocks_dW["pooled_all"],
           "C1_out_out_kept": out_out_energy(blocks_dd) / out_out_energy(blocks_dW)}
    return state, rec


def finetune_with_splits(ctx: Any, S: Sizes) -> tuple[dict[int, Any], dict[int, dict[str, Any]], dict[int, Any], list[Any]]:
    """The fine-tune with the step split (carry) and its fixed-bases variant stepping beside it. Returns per scored step the
    step-split state, the split's numbers, the fine-tune's own held-out eval, and the per-step log of both splits."""
    from commute import split
    from commute.finetune import ArmAConfig, finetune_arm_a

    carry = split.StepSplit(ctx.uv, ctx.sigma, ctx.W0, True, DTYPE, ctx.device)
    fixed = split.StepSplit(ctx.uv, ctx.sigma, ctx.W0, False, DTYPE, ctx.device)
    states, numbers, per_step = {}, {}, []

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        a, b = carry.step(params), fixed.step(params)
        per_step.append({"step": step, "carry": a, "fixed": b})
        if step % 10 == 0 or step == 1:
            log(f"[split] step {step}: second/first {a['second_over_first']:.2e} (max {a['second_over_first_max']:.2e}) "
                f"c {a['c_min']:.3g}..{a['c_max']:.3g} zeroed {a['n_zeroed']} | fixed c {b['c_min']:.3g}..{b['c_max']:.3g}")
        if step in S.steps:
            states[step], numbers[step] = snapshot(ctx, carry, fixed)
            log(f"[split] snapshot {step}: " + json.dumps({k: v for k, v in numbers[step].items() if not k.startswith("out_out")},
                                                          default=_json_default))
            log(f"[split] snapshot {step} outside-both: Delta change {numbers[step]['out_out_delta']} | update {numbers[step]['out_out_update']}")

    t0 = time.time()
    cfg = ArmAConfig(o_token=ctx.token, lr=FT_LR, steps=S.train_steps, checkpoints=S.steps, seed=0)
    ft = finetune_arm_a(ctx.model, ctx.train, ctx.held, cfg, ctx.device, log, on_step)
    for s in S.steps:  # the step split's delta-W is the fine-tune's
        assert max(float((states[s][k][0] - ft.deltas[s][k]).abs().max()) for k in ft.deltas[s]) == 0.0
    log(f"[stage finetune+split] {time.time() - t0:.0f} s")
    return states, numbers, {s: ft.evals[s] for s in S.steps}, per_step


# ----------------------------------------------------------------------------- the run


def require_predictions(smoke: bool) -> None:
    if not PRED.is_file():
        sys.exit(f"{PRED} missing: predictions are filed before running")
    status = json.loads(PRED.read_text()).get("status", "")
    if not smoke and status != "filed":
        sys.exit(f"{PRED} has status {status!r}; the scored run needs the user-confirmed predictions (status 'filed')")


def stage_all(args: argparse.Namespace) -> None:
    from commute import harness

    require_predictions(args.smoke)
    S = SMOKE if args.smoke else FULL
    path = RESULTS / f"step_split{S.suffix}.json"
    t_start = time.time()
    o_token = json.loads(WINDOWS_JSON.read_text())["o_token_id"]
    ctx = harness.load(args.device, ROUND1 / "alive.pt", ROUND1 / "windows.pt", o_token, TOL, S.pile_rows, S.pile_pgd_steps,
                       S.pile_seeds, log)
    f = harness.pgd_floor(ctx)
    assert f is not None
    log(f"[stage load: W0 control] {time.time() - t_start:.0f} s; W0 PGD floor {f:.4f} (stop above {S.floor_stop})")
    out: dict[str, Any] = (json.loads(path.read_text()) if args.resume and path.is_file() else
                           {"smoke": args.smoke, "sizes": asdict(S), "states": {}})
    out["W0"] = {"pile": ctx.pile, "pgd_floor": f}
    if f > S.floor_stop:
        out["stopped"] = f"W0 control failed: floor {f:.4f} > {S.floor_stop}"
        dump(path, out)
        sys.exit(out["stopped"])
    states, numbers, evals, per_step = finetune_with_splits(ctx, S)
    dump(LARGE / f"per_step{S.suffix}.json", per_step)
    out["per_step_max"] = {v: {f: max(p[v][f] for p in per_step) for f in ("second_over_first", "second_over_first_max", "n_zeroed")}
                           for v in ("carry", "fixed")}
    log(f"[split] per-step maxima {out['per_step_max']}")
    for step in S.steps:
        rec = out["states"].setdefault(str(step), {})
        rec.update({"split": numbers[step], "ft_eval": evals[step]})
        dW = {k: v[0] for k, v in states[step].items()}
        if "step_split" not in rec:
            rec["step_split"] = harness.measure(ctx, f"ft_step{step}_step_split", dW, states[step], True, log)
            dump(path, out)
        if "nothing" not in rec:
            ns = harness.nothing_and_split(ctx, f"ft_step{step}", dW, evals[step]["p_o"], True, log)
            rec.update({"nothing": ns["nothing"], "least_squares": ns["lift"], "G2_least_squares": ns["G2"]})
            dump(path, out)
        states.pop(step)
    out["score"] = score(out, json.loads(E16.read_text()) if E16.is_file() else None)
    out["seconds"] = time.time() - t_start
    dump(path, out)
    log(f"wrote {path} ({out['seconds']:.0f} s); score {json.dumps(out['score'], default=str)}")


# ----------------------------------------------------------------------------- the score (rules: results/predictions.json)


def _pred(cond: dict[str, bool], **values: Any) -> dict[str, Any]:
    return {"pass": all(cond.values()) if cond else None, "conditions": cond, **values}


def score(out: dict[str, Any], e16: dict[str, Any] | None) -> dict[str, Any]:
    st, f = out["states"], out["W0"]["pgd_floor"]
    line, tol = 1 + 3 * f, max(0.15, 3 * f)

    def pgd(r: dict[str, Any], arm: str) -> float:
        return r[arm]["pile"]["pgd_kl"]

    def e_delta(r: dict[str, Any], arm: str) -> float:
        return r[arm]["routes"]["shares"]["e_delta"]

    sc: dict[str, Any] = {"line": line}
    sc["C1_out_out_kept"] = _pred({s: abs(r["split"]["C1_out_out_kept"] - 1) <= 0.01 for s, r in st.items()})
    sc["G2_exact"] = _pred({**{f"p@{s}": abs(r["step_split"]["routes"]["p_cells"]["T1"] - r["ft_eval"]["p_o"]) <= 1e-3 for s, r in st.items()},
                            **{f"sum@{s}": abs(r["step_split"]["routes"]["share_sum"] - 1) <= 1e-6 for s, r in st.items()},
                            **{f"delta@{s}": abs(r["step_split"]["delta_change_frac"] - r["split"]["delta_share"]) <= 1e-3
                               for s, r in st.items()}})
    if e16 is not None:
        ref = e16["states"]["ft"]
        sc["G1_reading_E16"] = _pred({f"{a}@{s}": abs(pgd(r, a) - ref[s][a]["pile"]["pgd_kl"]) <= tol
                                      for s, r in st.items() if s in ref for a in ("nothing", "least_squares")}, tol=tol)
    sc["P1_second_order"] = _pred({"per_step": out["per_step_max"]["carry"]["second_over_first"] <= 1e-4,
                                   **{f"share@{s}": r["split"]["second_share_of_delta"] <= 1e-3 for s, r in st.items()}})
    last = str(max(int(s) for s in st))
    sc["P2_bases_little"] = _pred({"pieces": st[last]["split"]["piece_change_max"] <= 0.1,
                                   **{f"fixed@{s}": abs(r["split"]["delta_share_weighted"] - r["split"]["delta_share_weighted_fixed"]) <= 0.05
                                      for s, r in st.items()},
                                   **{f"one_shot@{s}": abs(r["split"]["delta_share_weighted"] - r["split"]["delta_share_weighted_one_shot"]) <= 0.05
                                      for s, r in st.items()}})
    first = str(min(int(s) for s in st))
    sc["P3_delta_route_rises"] = _pred({"first<=0.3": e_delta(st[first], "step_split") <= 0.3,
                                        "last>=0.5": e_delta(st[last], "step_split") >= 0.5,
                                        **{f">=ls@{s}": e_delta(r, "step_split") >= e_delta(r, "least_squares") for s, r in st.items()}})
    sc["P4_pile_between"] = _pred({s: min(pgd(r, "nothing"), pgd(r, "least_squares")) - 3 * f <= pgd(r, "step_split")
                                   <= max(pgd(r, "nothing"), pgd(r, "least_squares")) + 3 * f for s, r in st.items()})
    for k in ("P1_second_order", "P2_bases_little", "P3_delta_route_rises", "P4_pile_between"):
        if sc["C1_out_out_kept"]["pass"] is False or sc["G2_exact"]["pass"] is False:
            sc[k]["pass"] = "unscored: C1 or G2 failed"
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.stage == "all":
        stage_all(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
