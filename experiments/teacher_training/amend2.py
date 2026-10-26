"""teacher_training Amendment 2 (2026-10-26): is Amendment 1's Pile-only result carried by the rows where the emoticon
trigger subcomponent was partly on? Plan: reports/plans/2026-10-23_plan_teacher-training.md -> "Amendment 2".

One new arm-B variant on emoticon, both teachers (SGD 0.01, Adam 3e-5, regenerated as in E34): B-dropped-0.1, Amendment
1's B-dropped with the threshold on h.2.mlp.down_proj:2359's g at W0 lowered from 0.5 to 0.1. Every row with a position
above 0.1 is replaced, at its index, by the next spare Pile row that is clean at 0.1 too; the seeded sampler draws the
same indices. Trained with E34's arm-B recipe (batch 8, Adam 1e-5, 1,000 steps, batch-order seed 0, plain mean KL),
read through the harness at step 1000, P(target) at step 100 from the free reading, the cosine with delta-W.
Amendment 1's references (its B-dropped record, teacher reproduction, cosine references) are read from
results/teacher_training_amend1.json and copied into the results file, never recomputed.

Stages, each record dumped to results/teacher_training_amend2.json as it finishes (a rerun continues the file):
  C1       W0 under the Pile harness (faith_score.w0_check); stop if it fails
  rows     the 8,192 arm-B rows and 2,048 spare rows; g of the trigger subcomponent on both; counts in out["trigger"]
  per teacher, SGD then Adam:
    G1     the regenerated teacher reads Amendment 1's reproduction (P within 0.01, size within 0.002); stop otherwise
    train  B-dropped-0.1 to step 1000; read; saved as bf16 offsets results/large/emoticon/Bdropped01_<opt>_step1000.pt
  score    P1 and P2 from the filed rules (score(), pure; `--score` re-runs it on CPU)

Usage (repo root, project venv):
  amend2.py --predictions    file results/predictions_amend2.json (refuses to overwrite)
  amend2.py [cuda]           the run (GPU)
  amend2.py --score          re-score an existing results file (CPU, JSON only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from run import (
    CFG,
    EMO,
    LARGE,
    RESULTS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    read_state,
    train_student,
)

PRED2, OUT2 = RESULTS / "predictions_amend2.json", RESULTS / "teacher_training_amend2.json"
OUT1 = RESULTS / "teacher_training_amend1.json"
OPTS = ("sgd", "adam")


@dataclass(frozen=True)
class Amend2:
    variant: str = "B-dropped-0.1"
    trigger_key: str = "h.2.mlp.down_proj:2359"  # vpd.EMOTICON_KEY
    drop_g: float = 0.1  # rows with any position above this are replaced (Amendment 1's near_g)
    full_g: float = 0.5  # the windows' threshold, Amendment 1's drop threshold; reported for the counts
    spare_rows: int = 2048  # Pile train rows right after arm B's 8,192, replacing the dropped rows in order
    g1_p_tol: float = 0.01  # G1: |teacher P(target) - Amendment 1's| <= this
    g1_size_tol: float = 0.002  # G1: |teacher size - Amendment 1's| <= this
    p_same: float = 0.03  # P1: |P(target) - Amendment 1's B-dropped| <= this -> "not carried"
    p_fall: float = 0.10  # P1: a fall of more than this -> "carried in part"; anything else "between"
    cos_tol: float = 0.02  # P2: |cosine - Amendment 1's B-dropped| <= this
    # Amendment 1's B-dropped at step 1000, (SGD, Adam), as the predictions quote them; the run scores against the
    # full-precision records copied from results/teacher_training_amend1.json
    ref_p: tuple[float, float] = (0.937, 0.892)
    ref_cos: tuple[float, float] = (0.078, 0.218)
    ref_p100: tuple[float, float] = (0.195, 0.080)

    def __post_init__(self) -> None:
        assert self.drop_g < self.full_g and self.p_same < self.p_fall


AM = Amend2()


def dropped_rows(job: Any, out: dict[str, Any]) -> Any:
    """Arm B's 8,192 Pile rows with every row holding a position where the trigger subcomponent's g > drop_g replaced by
    the next clean spare row (clean at drop_g). Counts at drop_g and full_g in out["trigger"]."""
    from commute import data, vpd

    assert AM.trigger_key == vpd.EMOTICON_KEY
    model = job.task(EMO)[0].model
    pile = data.pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    spare = data.pile_rows("train", CFG.pile_start + CFG.pile_train_rows, AM.spare_rows, log)
    t0 = time.time()
    g, g_sp = data.subcomponent_g(model, pile, AM.trigger_key), data.subcomponent_g(model, spare, AM.trigger_key)
    rows_hit, sp_hit = (g > AM.drop_g).any(1), (g_sp > AM.drop_g).any(1)
    gen = torch.Generator().manual_seed(0)  # component_training.train's sampler (batch-order seed 0)
    idx = torch.cat([torch.randint(0, pile.shape[0], (CFG.batch_b,), generator=gen) for _ in range(CFG.steps[-1])])
    counts = {f"above_{t:g}": {"positions": int((g > t).sum()), "rows": int((g > t).any(1).sum()),
                               "rows_drawn": int((g > t).any(1)[idx].sum())} for t in (AM.drop_g, AM.full_g)}
    out["trigger"] = {**counts, "spare_rows_hit": int(sp_hit.sum()),
                      "max_g_left": float(data.replace_rows(g, rows_hit, g_sp, sp_hit).max()), "seconds": time.time() - t0}
    log(f"[trigger] {json.dumps(out['trigger'], default=float)}")
    dump(OUT2, out)
    assert out["trigger"]["max_g_left"] <= AM.drop_g
    return data.replace_rows(pile, rows_hit, spare, sp_hit)


def amend1_refs(opt: str) -> dict[str, Any]:
    """Amendment 1's readings for one teacher: its reproduction, the cosine references and its B-dropped at step 1000."""
    t = json.loads(OUT1.read_text())["teachers"][opt]
    d = t["B-dropped"]
    return {"repro": t["repro"], "cos_ref": {k: v["all"] for k, v in t["cos_ref"].items()},
            "B-dropped": {**{k: d[k] for k in ("p_target", "p_target_100", "pgd", "verdict", "teacher_gap", "damage")},
                          "cos_all": d["cos"]["all"]}}


def g1(out: dict[str, Any], opt: str) -> None:
    """G1: the regenerated teacher reads Amendment 1's reproduction (P(target), size); stop before training otherwise."""
    rep, ref = out["teachers"][opt]["repro"], out["teachers"][opt]["ref_amend1"]["repro"]
    rec = {"p_target": [rep["p_target"], ref["p_target"]], "size": [rep["size"], ref["size"]],
           "pass": abs(rep["p_target"] - ref["p_target"]) <= AM.g1_p_tol and abs(rep["size"] - ref["size"]) <= AM.g1_size_tol}
    out.setdefault("G1", {})[opt] = rec
    log(f"[G1 {opt}] {json.dumps(rec, default=float)}")
    dump(OUT2, out)
    if not rec["pass"]:
        halt(OUT2, out, f"G1: the {opt} teacher read P {rep['p_target']:.4f}, size {rep['size']:.4f}; Amendment 1 "
                        f"{ref['p_target']:.4f}, {ref['size']:.4f}")


def variant(job: Any, out: dict[str, Any], opt: str, rows: Any) -> None:
    """B-dropped-0.1 of one teacher: G1, trained to step 1000, read through the harness there, its step-100 P(target)
    from the free held-out reading, its cosine with delta-W; saved as bf16 offsets."""
    from commute import arms, finetune, teacher

    t = out["teachers"].setdefault(opt, {})
    if AM.variant in t:
        return
    tctx = job.task(EMO)[0]
    dW, repro = job.teacher(EMO, opt)
    t.update(repro=repro, ref_amend1=amend1_refs(opt))
    g1(out, opt)
    net = teacher.teacher_net(tctx.model, dW)
    t["gap_w0"] = gap_w0(tctx, net)
    t0 = time.time()
    curve, states = train_student(tctx, net, "adam", CFG.student_lr, CFG.steps[-1:], rows, CFG.batch_b, None)
    log(f"[emoticon {opt} {AM.variant}] trained in {time.time() - t0:.0f} s")
    st = states[CFG.steps[-1]]
    d_s = {k: x[0] for k, x in st.items()}
    rec = read_state(job, tctx, f"emoticon {opt} {AM.variant}@{CFG.steps[-1]}", (d_s, st), net, True)
    rec.update(p_target_100=curve[CFG.steps[0] - 1]["heldout_p_o"], cos=finetune.cosines(d_s, dW))
    log(f"[emoticon {opt} {AM.variant}] P(target)@100 (free reading) {rec['p_target_100']:.4f} cosine with delta-W "
        f"{rec['cos']['all']:.4f}; Amendment 1's B-dropped P {t['ref_amend1']['B-dropped']['p_target']:.4f} "
        f"cosine {t['ref_amend1']['B-dropped']['cos_all']:.4f}")
    t[AM.variant] = rec
    dump(OUT2, out)
    arms.save_offsets(LARGE / EMO / f"Bdropped01_{opt}_step{CFG.steps[-1]}.pt", tctx.uv_cpu, st,
                      {"task": EMO, "teacher": opt, "arm": AM.variant, "step": CFG.steps[-1]})


def p1_reading(p: float, ref: float) -> str:
    """P1's filed rule: B-dropped-0.1's P(target) at step 1000 against Amendment 1's B-dropped."""
    if abs(p - ref) <= AM.p_same:
        return "not carried"
    return "carried in part" if ref - p > AM.p_fall else "between"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1 and P2 per teacher from a results file, by the filed rules (CPU, JSON only)."""
    res: dict[str, Any] = {}
    for opt, t in out["teachers"].items():
        if AM.variant not in t:
            continue
        rec, ref = t[AM.variant], t["ref_amend1"]["B-dropped"]
        res[opt] = {"P1": p1_reading(rec["p_target"], ref["p_target"]),
                    "P2": abs(rec["cos"]["all"] - ref["cos_all"]) <= AM.cos_tol,
                    "p_target": [rec["p_target"], ref["p_target"]], "cos": [rec["cos"]["all"], ref["cos_all"]],
                    "verdict": [rec["verdict"], ref["verdict"]]}
    return res


def amend2(device: str) -> None:
    if not PRED2.is_file():
        sys.exit(f"file the predictions first: {PRED2}")
    filed = json.loads(PRED2.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend2"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend2']}"
    assert OUT1.is_file(), f"Amendment 1's results are missing: {OUT1}"
    job = Job(device)
    out = load(OUT2, job, {"task": EMO, "variant": AM.variant, "teachers": {}})
    dump(OUT2, out)
    if not job.w0["pass"]:
        halt(OUT2, out, "C1: W0 is not at its known values; nothing trained")
    rows = dropped_rows(job, out)
    for opt in OPTS:
        variant(job, out, opt, rows)
    out["score"] = score(out)
    out["done"] = True
    dump(OUT2, out)
    log(f"[score] {json.dumps(out['score'], default=float)}")
    log("done")


PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training Amendment 2 (reports/plans/2026-10-23_plan_teacher-training.md -> 'Amendment 2'; approved classes 3 "
             "and 4, the loophole Amendment 1's design check named). Emoticon only, both teachers (SGD 0.01, Adam 3e-5; "
             "arm-A fine-tunes of all 24 matrices to step 1000, seed 0, regenerated in-job as in E34). One arm-B variant per "
             "teacher, B-dropped-0.1: Amendment 1's B-dropped with the threshold on the trigger subcomponent's g at W0 "
             "lowered from 0.5 to 0.1, every row with a position above 0.1 replaced at its index by the next spare row "
             "(Pile train rows 40,960..43,007) that is clean at 0.1. E34's arm-B recipe (Pile train rows from 32,768, 8,192 "
             "rows, batch 8, Adam 1e-5, 1000 steps, batch-order seed 0, plain mean KL(teacher || student)), scored at step "
             "1000 only. Amendment 1's references are read from results/teacher_training_amend1.json, not recomputed. "
             "predictions.json and predictions_amend1.json stand unedited. Filed before any run",
    "config": asdict(CFG),
    "amend2": asdict(AM),
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, adversary seeds 0, 1, 2), "
              "faith_score.w0_check, as E34. Expected: pass, floor about 0.075 on an H100. Fails: stop, nothing trained",
        "G1": "per teacher, before its student trains: the regenerated teacher's held-out P(target) within 0.01 and size "
              "of change within 0.002 of Amendment 1's reproduction reading (SGD: P 0.9980, size 0.0122; Adam: P 0.9996, "
              "size 0.0321). Expected: identical (same seed, code path and card). Fails: stop",
    },
    "measurements": {
        "p_target": "held-out P(target) at step 1000 through the harness (arms.heldout_of), absolute. W0 0.004; the "
                    "teachers 0.998 and 1.000; Amendment 1's B-dropped 0.937 (SGD) and 0.892 (Adam)",
        "cosine": "finetune.cosines(student delta-W, teacher delta-W) over the 24 matrices concatenated (float64), as "
                  "Amendment 1. Its references per teacher are Amendment 1's, copied: arm A 0.245 / 0.285, adjoint "
                  "0.450 / 0.382, null 0.0002, B-dropped 0.078 / 0.218. Blind to: the part of delta-W that does nothing on "
                  "the data (every entry weighs alike); positive control: arm A's cosine",
        "p_target_100": "held-out P(target) at step 100 from component_training.train's per-step held-out reading (free, "
                        "not the harness). Amendment 1's B-dropped: 0.195 SGD, 0.080 Adam",
        "seven numbers": "per state at step 1000 as E34 (record): adversarial and stochastic ratios to this job's W0 with "
                         "state_verdict, P(target), teacher gap, damage elsewhere, size, Delta's share",
        "trigger counts": "positions and rows of the 8,192 arm-B rows with the trigger subcomponent's g > 0.1 and > 0.5 at "
                          "W0 (data.subcomponent_g), rows drawn by the seeded sampler in 1,000 batches of 8, spare rows "
                          "with g > 0.1, the largest g left after replacement (<= 0.1 by construction)",
        "blind_to": "no P(target) noise floor across batch orders yet (Amendment 4 supplies one): a 0.03 band is a "
                    "judgement, not a measured floor; one seed; emoticon only; the cosine weighs every entry alike",
    },
    "expected": {
        "trigger counts": "Amendment 1: 90 positions in 63 rows above 0.5, 58 rows drawn, 45 positions above 0.1 left in "
                          "its dropped rows. So above 0.1 about 130-170 positions in about 90-120 rows; about 1% of the "
                          "2,048 spare rows hit at 0.1 (about 20-30)",
        "teacher gap at W0": "5.01 (SGD), 5.09 (Adam), as E34",
        "size": "0.016 (SGD teacher's copy), 0.020 (Adam teacher's copy), as every Amendment 1 copy",
    },
    "predictions": [
        {"id": "P1_loophole",
         "kind": "two-sided, filed with a lean to 'not carried'",
         "rule": "per teacher, B-dropped-0.1's held-out P(target) at step 1000 against Amendment 1's B-dropped (0.937 "
                 "SGD, 0.892 Adam, full-precision values copied into the results file): |difference| <= 0.03 -> 'not "
                 "carried'; a fall of more than 0.10 -> 'carried in part'; anything else -> 'between'",
         "lean": "'not carried' for both teachers: Amendment 1 showed removing the 90 full triggers moved the end state "
                 "by about 0.02 while only slowing the first 100 steps; 45 partly-on positions should move it less",
         "readings": {
             "not carried": "the partly-on rows did not carry the behaviour: the Pile-only claim stands with no form "
                            "of the trigger in the copying text",
             "carried in part": "a weak form of the trigger suffices: the Pile-only claim is withdrawn to 'rare "
                                "occurrences suffice'",
             "between": "reported per teacher; Amendment 4's noise floor decides whether the shift is real",
         },
         "reported": "state_verdict beside each P (Amendment 1's B-dropped: pass SGD, 'below P 0.9' Adam)"},
        {"id": "P2_cosine_unchanged",
         "kind": "two-sided",
         "rule": "per teacher, the cosine over the 24 matrices of B-dropped-0.1's delta-W with the teacher's delta-W "
                 "within 0.02 of Amendment 1's B-dropped (0.078 SGD, 0.218 Adam), absolute",
         "pass": "within 0.02 for both teachers: the copy finds the same weight change whether or not the partly-on "
                 "rows are in its text",
         "fail": "outside 0.02 for either teacher: reported with its sign; a fall with P1 'carried in part' says the "
                 "partly-on rows pulled the copy towards delta-W"},
    ],
    "reported_not_scored": [
        "P(target) at step 100 (free held-out reading) against Amendment 1's B-dropped 0.195 / 0.080",
        "state_verdict, teacher gap, damage elsewhere, size, stochastic ratio, Delta's share",
        "trigger counts at 0.1 and 0.5, spare rows hit, the largest g left",
        "each teacher's reproduction reading (G1) and its gap at W0",
        "every per-matrix cosine",
    ],
    "not_run": [
        "a masked version at 0.1 (Amendment 1: masking and dropping read alike)",
        "thresholds below 0.1; rows holding the trigger's tokens with the subcomponent off",
        "a second batch order or a longer adversary (Amendment 4); r0 and r1; a LoRA teacher (Amendment 3)",
        "the step-100 harness reading",
    ],
}


def file_predictions() -> None:
    if PRED2.is_file():
        sys.exit(f"{PRED2} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED2, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED2}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
    elif args.score:
        print(json.dumps(score(json.loads(OUT2.read_text())), indent=1))
    else:
        amend2(args.device)


if __name__ == "__main__":
    main()
