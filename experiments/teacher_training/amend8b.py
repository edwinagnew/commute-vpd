"""teacher_training E34 Amendment 8b (2026-11-02): the two $0 checks reports/analyses/2026-11-02_analysis_e34-amend8-base-control.md
recommends and did not run. Rules, leans and expected values: results/predictions_amend8b.json (the run refuses without it, or
if BM or run.CFG differ). One stage per job, each a MATS `debug` job on an L40, results/teacher_training_amend8b_<stage>.json.
Every job first reads W0 under the Pile harness (run.Job, 7 min) and stops if it is off.
  seeds  saved bf16 states through the same adversary at seeds 0, 1, 2 (faithfulness.multi_start): the real student, the null
         student at 1,000, the random start at 1,000; true-label Adam at seed 0 alone, the rule's positive control. Each
         state's seed 0 must reproduce its saved reading. Is the null's 1.39 against the real's 0.99 more than start noise?
  moved  the missing cell of start x teacher: the null teacher from the random start's saved start (results/large/amend8/n1/0.pt).
         First, the same saved start with teacher = W0, which must reproduce the random start's gap. Is the null student's low
         gap to its own teacher set by its teacher or by its start at W0?
Usage (repo root, project venv, GPU):  amend8b.py <stage> [--device cuda]; cluster/teacher_amend8b.sbatch <stage>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from amend8 import (
    AM,
    STATES,
    check,
    deltas,
    keep_curve,
    near_one,
    null_fine_tune,
    read,
    state_of,
    train_from,
)
from path import init_of, student
from run import CFG, EMO, RESULTS, Job, dump, halt, load, log

from commute import faith_score

PRED = RESULTS / "predictions_amend8b.json"
STAGES = ("seeds", "moved")
STARTS = ("h0", "h1", "h2")
NULL, REAL, RANDOM, TRUE = "null_1000", "real", "random_1000", "true_adam"
# name: (Amendment 8's stage, its read, the adversary starts here). The order is the run's; the positive control last
READS: dict[str, tuple[str, str, tuple[str, ...]]] = {REAL: ("n3b", "real", STARTS), NULL: ("n3b", "1000", STARTS),
                                                      RANDOM: ("n1", "1000", STARTS), TRUE: ("n4_adam", "true", ("h0",))}


@dataclass(frozen=True)
class Amend8b:
    w0_tol: float = 1e-3  # W0 at seed 0 / Amendment 8's, relative: same card, deterministic (claim_tests.job f_repro_tol)
    reproduce_tol: float = 0.07  # a state's seed 0 / its saved ratio, relative: E23's 8 bf16 rebuilds differ by 0.1-4.4%
    size_tol: float = 0.01  # a rebuilt or regenerated state's size / the saved one's, and a warm start's at step 1
    first_kl_tol: float = 0.5  # step-1 KL / the same teacher's at the nearest reference, relative: a sanity bound
    control_tol: float = 0.25  # moved's control gap / the random start's: rounding of the start; no floor exists, see the filing
    null_step: int = 100  # Amendment 8's chosen checkpoint (n3a)
    power: float = 2.0  # moved: the two reference gaps at least this far apart, or 'no power'
    low: float = 0.33  # moved: position at or below this reads 'the teacher'
    high: float = 0.67  # moved: position at or above this reads 'the start'

    def __post_init__(self) -> None:
        assert 0 < self.low < self.high < 1 and self.power > 1 and self.null_step in AM.n3a_steps


BM = Amend8b()


# ----------------------------------------------------------------------------- rules (pure)


def separation(a: list[float], b: list[float]) -> str:
    """Strict: a tie is 'overlap'. An empty pool or a KL not finite and positive is 'unscorable'."""
    if not a or not b or not all(math.isfinite(x) and x > 0 for x in (*a, *b)):
        return "unscorable"
    return "apart" if min(a) > max(b) else "reversed" if min(b) > max(a) else "overlap"


def noise_class(a: list[float], b: list[float]) -> dict[str, Any]:
    """Three starts each. 'apart' or 'reversed': every reading of one above every reading of the other. Otherwise the excess of
    a's mean over b's against the larger of their own spreads (max / min - 1): at or under it 'within noise', above 'suggestive'."""
    sep = separation(a, b)
    if sep == "unscorable" or len(a) != len(STARTS) or len(b) != len(STARTS):
        return {"class": "unscorable", "excess": None, "spread": None}
    excess = sum(a) / sum(b) - 1
    spread = max(max(a) / min(a), max(b) / min(b)) - 1
    return {"class": sep if sep != "overlap" else "within noise" if excess <= spread else "suggestive",
            "excess": excess, "spread": spread}


def paired_verdict(mean_ratio: float | None, floor: float) -> str:
    """The mean over the starts of state / W0 at the same start, against the card's lines."""
    if mean_ratio is None or not math.isfinite(mean_ratio):
        return "unscorable"
    return faith_score.verdict(mean_ratio, floor)


def score_seeds(out: dict[str, Any]) -> dict[str, Any]:
    st = out["states"]
    kl = {n: list(st[n]["kl"].values()) for n in READS}
    control = separation(kl[TRUE], kl[REAL])
    s = noise_class(kl[NULL], kl[REAL])
    return {"null_real": s, "answer": s["class"] if control == "apart" else "no power", "positive_control": control,
            "null_random": noise_class(kl[NULL], kl[RANDOM]),
            "null_over_real_each_start": [a / b for a, b in zip(kl[NULL], kl[REAL], strict=True)],
            "mean_ratio": {n: st[n]["mean"] for n in READS},
            "verdict": {n: paired_verdict(st[n]["mean"], out["W0"]["pgd_floor"]) for n in READS}}


def position(gap: float, a: float, b: float) -> tuple[str, float | None]:
    """Where `gap` sits between a (start W0, null teacher) and b (moved start, teacher W0), on a log scale: 0 at a, 1 at b."""
    if not all(math.isfinite(x) and x > 0 for x in (gap, a, b)):
        return "unscorable", None
    if b / a < BM.power:
        return "no power", None
    t = math.log(gap / a) / math.log(b / a)
    return ("the teacher" if t <= BM.low else "the start" if t >= BM.high else "both"), t


def score_moved(out: dict[str, Any]) -> dict[str, Any]:
    a, b, g = saved("n3b")["reads"]["1000"]["teacher_gap"], out["control"]["1000"], out["reads"]["1000"]["teacher_gap"]
    cls, t = position(g, a, b)
    return {"class": cls, "position": t, "gap": g, "null_student": a, "control": b, "check": out["reads"]["1000"]["pgd"],
            "verdict": out["reads"]["1000"]["base_verdict"], "train_kl_trigger": out["curves"]["moved"][-1]["train_kl_trigger"],
            "train_kl_other": out["curves"]["moved"][-1]["train_kl_other"]}


# ----------------------------------------------------------------------------- the stages


def saved(stage: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"teacher_training_amend8_{stage}.json").read_text())


def seeds(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import harness
    from commute.faithfulness import multi_start
    from commute.saved_states import offsets_state

    c = job.ctx
    check(out, path, "W0 seed 0 / Amendment 8's", c.pile[0]["pgd_kl"] / saved("n3b")["W0"]["pile"]["0"]["pgd_kl"], near_one(BM.w0_tol))
    w0 = {f"h{s}": c.pile[s]["pgd_kl"] for s in CFG.pile_seeds}
    for name, (stage, key, starts) in READS.items():
        if name in out["states"]:
            continue
        ref = saved(stage)["reads"][key]
        dW, uv2 = offsets_state(STATES / stage / f"{key}.pt", c.uv_cpu)
        size = max(float(dW[k].norm() / c.W0[k].cpu().norm()) for k in dW)
        check(out, path, f"{name} size / saved", size / ref["size"], near_one(BM.size_tol))
        with harness.at(c, dW, uv2):
            rec = multi_start(c.model, c.blocks, list(starts), w0=w0, log=lambda s, n=name: log(f"[{n}] {s}"))
        check(out, path, f"{name} seed 0 / saved", rec["ratio"]["h0"] / ref["pgd"], near_one(BM.reproduce_tol))
        out["states"][name] = rec | {"saved_ratio": ref["pgd"], "size": size}
        dump(path, out)
        log(f"[{name}] {json.dumps(out['states'][name])}")


def gap_to(tctx: Any, dW: dict[str, Any], net: Any) -> float:
    from commute import harness, teacher

    with harness.at(tctx, dW, None):
        return teacher.heldout_gap(tctx.model.target_model, net, tctx.held, tctx.device)


def moved(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import teacher
    from commute.saved_states import offsets_state

    tctx, n1, n3b = job.task(EMO)[0], saved("n1"), saved("n3b")
    net0 = teacher.teacher_net(tctx.model, {})
    res = null_fine_tune(tctx, net0, (BM.null_step,))
    size = res.evals[BM.null_step]["rel_norm_max"]
    check(out, path, "null teacher size / saved", size / n3b["null_teacher"]["rel_norm_max"], near_one(BM.size_tol))
    net = teacher.teacher_net(tctx.model, res.deltas[BM.null_step])
    st0 = state_of(*offsets_state(STATES / "n1" / "0.pt", tctx.uv_cpu))
    init = init_of(tctx, st0)
    curve, _ = student(job, EMO, "A", teacher.teacher_net(tctx.model, deltas(st0)), init, steps=1, lr=0.0)
    check(out, path, "warm start KL", curve[0]["loss"], AM.warm_kl)
    # The control, alone first: this start with teacher = W0, against the random start's reading from the unrounded start
    ref = n1["curves"]["walk"][0]
    curve, states = train_from(tctx, net0, init, CFG.steps)
    keep_curve(out, path, "control", curve)
    check(out, path, "control step-1 KL / random start's", curve[0]["loss"] / ref["loss"], near_one(BM.first_kl_tol))
    check(out, path, "control step-1 size / random start's", curve[0]["rel_norm_max"] / ref["rel_norm_max"], near_one(BM.size_tol))
    out["control"] = {str(s): gap_to(tctx, deltas(states[s]), net0) for s in CFG.steps}
    dump(path, out)
    check(out, path, "control gap / random start's", out["control"]["1000"] / n1["reads"]["1000"]["teacher_gap"], near_one(BM.control_tol))
    del states
    # The missing cell: only the teacher differs from the control, only the start from the null student
    curve, states = train_from(tctx, net, init, CFG.steps)
    keep_curve(out, path, "moved", curve)
    check(out, path, "step-1 KL / null student's", curve[0]["loss"] / n3b["curves"]["null_student"][0]["loss"], near_one(BM.first_kl_tol))
    check(out, path, "step-1 size / random start's", curve[0]["rel_norm_max"] / ref["rel_norm_max"], near_one(BM.size_tol))
    for s in CFG.steps:
        read(job, out, path, tctx, str(s), deltas(states[s]), states[s], net, net0)


def run_stage(job: Job, stage: str) -> None:
    path = RESULTS / f"teacher_training_amend8b_{stage}.json"
    out = load(path, job, {"stage": stage, "reads": {}, "states": {}})
    dump(path, out)
    floor = job.w0["pgd_floor"]
    if not job.w0["pass"] or abs(floor / AM.floor - 1) > AM.floor_tol:
        halt(path, out, f"W0 first: check {job.w0['pass']}, floor {floor:.4f} (L40 {AM.floor} within {AM.floor_tol:.0%}) on {job.card}")
    if out.get("done"):
        return log(f"{stage} is done")
    (seeds if stage == "seeds" else moved)(job, out, path)
    out["score"], out["done"] = (score_seeds if stage == "seeds" else score_moved)(out), True
    dump(path, out)
    log(f"[{stage} score] {json.dumps(out['score'], default=float)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    for key, cfg in (("config", CFG), ("amend8b", BM)):
        assert filed[key] == json.loads(json.dumps(asdict(cfg))), f"predictions were filed for another {key}: {filed[key]}"
    run_stage(Job(args.device), args.stage)
    log("done")


if __name__ == "__main__":
    main()
