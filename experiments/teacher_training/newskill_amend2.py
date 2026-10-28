"""teacher_training E42 Amendment 2 (2026-10-28): which masked term moves the decomposition with nothing to copy, then
the cipher copy with the term(s) that hold still. Plan: reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md ->
"Amendment 2" (design check, numbers, predictions, tree); decision: decisions.md, card `decision-e42-masked`.

Amendment 1's nothing-to-copy control (M0: teacher training with the CI-masked and the stochastic-masked student added
to the loss, 0.5 each, teacher = W0) read 26 on the adversarial Pile check. Here each masked term runs alone
(commute.masked_terms: a weight per term, and the masked terms read at the trigger alone as an option), on rand32 only,
in one job. Everything else is Amendment 1's M0 (E34's student settings, arm A's batch and batch order, 1000 steps, Gamma
and Delta frozen). Each record is dumped to results/teacher_training_e42_amend2_rand32.json as it finishes and skipped on
a rerun on the same card:
  C1      W0 under the Pile harness, adversary seeds 0, 1, 2 (run.Job); stop if it fails
  G2      Amendment 1's path check (the trainer's unmasked student at W0 against the target model); stop above 0.01
  M0_ci   the CI-masked term alone (0.5; stochastic 0), teacher = W0, Pile harness
  M0_st   the stochastic term alone (CI-masked 0; 0.5), teacher = W0, Pile harness
  tree    a control holds when neither Pile ratio reads 'fail' (nor NaN). One holds -> the copy with that term alone;
          both -> the copy with the CI-masked term alone; neither -> M0_trig (both terms, 0.5 each, at the trigger
          only), and the copy with it if it holds; if it fails too, stop
  G1      E42's teacher regenerated; stop if its step-10 P(target) misses E10's
  refs    W0's trigger cells; E42's saved arm-A copy re-read at the trigger (Amendment 1's `references`)
  M_<r>   the copy with recipe r, read at 100 (no Pile harness) and 1000 (Pile harness)
  score   the filed rules
Every trained state's last step is saved as bf16 offsets under results/large/e42_amend2/rand32/.

Usage (repo root, project venv, GPU):  newskill_amend2.py --predictions   (files results/predictions_e42_amend2.json once)
                                       newskill_amend2.py                 [--device cuda]
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

from newskill import AM as E42
from newskill import KV, g1, make_teacher
from newskill_amend1 import AM as A1
from newskill_amend1 import PREDICTIONS as A1_PRED
from newskill_amend1 import ci_class, faithful, g2, means, read, references
from run import CFG, LARGE, RESULTS, Job, dump, halt, load, log

PRED = RESULTS / "predictions_e42_amend2.json"


@dataclass(frozen=True)
class Amend2:
    task: str = "rand32"
    # (name, CI-masked weight, stochastic weight, masked terms at the trigger only); 0.5 is Amendment 1's weight
    recipes: tuple[tuple[str, float, float, bool], ...] = (("ci", 0.5, 0.0, False), ("st", 0.0, 0.5, False),
                                                           ("trig", 0.5, 0.5, True))
    first: tuple[str, ...] = ("ci", "st")  # stage 1, both always; listed in the order the tree prefers when both hold
    fallback: str = "trig"  # run only when every stage-1 control fails

    def __post_init__(self) -> None:
        names = [r[0] for r in self.recipes]
        assert self.task in A1.tasks and len(set(names)) == len(names) and set(self.first) | {self.fallback} == set(names)
        assert all(c >= 0 and s >= 0 and c + s > 0 for _, c, s, _ in self.recipes)


AM = Amend2()
RECIPE = {n: (c, s, t) for n, c, s, t in AM.recipes}
LEAN_C = {"ci": "fail", "st": "pass", "trig": "fail"}
LEAN_P1 = {"ci": "kept", "st": "not kept", "trig": "kept"}


def out_path() -> Path:
    return RESULTS / f"teacher_training_e42_amend2_{AM.task}.json"


def train_recipe(tctx: Any, net: Any, name: str, save_steps: tuple[int, ...]) -> tuple[dict[int, Any], dict[str, Any]]:
    """Recipe `name` (E34's student settings, arm A's batch, batch order 0) from W0's (U, V): the states at `save_steps`
    and the curve's means over each block of 50 steps."""
    from commute import masked_terms

    c, s, t = RECIPE[name]
    cfg = masked_terms.TermsConfig(lr=CFG.student_lr, batch=CFG.batch_a, steps=CFG.steps[-1], save_steps=save_steps,
                                   ci_coeff=c, stoch_coeff=s, trigger_only=t)
    t0 = time.time()
    curve, states = masked_terms.train(tctx.model, net, tctx.train, tctx.alive, tctx.uv, cfg, log)
    blocks = [means(curve[i:i + 50]) for i in range(0, len(curve), 50)]
    return states, {"first50": blocks[0], "last50": blocks[-1], "blocks50": blocks, "seconds": time.time() - t0}


def save(tctx: Any, key: str, name: str, step: int, st: Any) -> None:
    from commute import arms

    c, s, t = RECIPE[name]
    arms.save_offsets(LARGE / "e42_amend2" / AM.task / f"{key}_step{step}.pt", tctx.uv_cpu, st,
                      {"task": AM.task, "state": key, "ci_coeff": c, "stoch_coeff": s, "trigger_only": t, "step": step})


def control(job: Job, out: dict[str, Any], path: Path, tctx: Any, net0: Any, name: str) -> None:
    """M0_<name>: recipe `name` with the teacher at W0 (nothing to copy), read with the Pile harness. Never stops the job:
    the tree reads it."""
    key, last = f"M0_{name}", CFG.steps[-1]
    if key not in out:
        states, tr = train_recipe(tctx, net0, name, (last,))
        st = states[last]
        save(tctx, key, name, last, st)
        out[key] = {**read(job, tctx, f"{AM.task} {key}@{last}", ({k: v[0] for k, v in st.items()}, st), net0, True),
                    "train": tr}
        log(f"[{AM.task} {key} train] {json.dumps(tr['last50'])}")
        dump(path, out)
    log(f"[{key}] control: {control_class(out[key])}")


def copy(job: Job, out: dict[str, Any], path: Path, tctx: Any, net: Any, name: str) -> None:
    """M_<name>: recipe `name` with the cipher teacher, read at 100 without and at 1000 with the Pile harness."""
    key, last = f"M_{name}", CFG.steps[-1]
    if str(last) in out.get(key, {}):
        return
    states, tr = train_recipe(tctx, net, name, CFG.steps)
    log(f"[{AM.task} {key} train] {json.dumps(tr['last50'])}")
    out[key] = {"train": tr}
    for s in CFG.steps:
        st = states.pop(s)
        out[key][str(s)] = read(job, tctx, f"{AM.task} {key}@{s}", ({k: v[0] for k, v in st.items()}, st), net, s == last)
        if s == last:  # saved before the dump: a rerun skips the copy once its last step is in the JSON
            save(tctx, key, name, s, st)
        dump(path, out)


def run(job: Job) -> None:
    from commute import kv_windows, teacher

    path = out_path()
    out = load(path, job, {"task": AM.task})
    dump(path, out)
    if not job.w0["pass"]:
        halt(path, out, "C1: W0 is not at its known values; nothing trained")
    if out.get("done"):
        return
    tctx, _ = kv_windows.task_context(job.ctx, KV / f"windows_{AM.task}.pt")
    net0 = teacher.teacher_net(tctx.model, {})
    if not out.get("G2", {}).get("pass"):
        g2(out, path, tctx, net0)
    for name in AM.first:
        control(job, out, path, tctx, net0, name)
    arm = choose(out)
    if arm is None:
        control(job, out, path, tctx, net0, AM.fallback)
        arm = choose(out)
    log(f"[tree] controls {json.dumps({k: control_class(v) for k, v in out.items() if k.startswith('M0_')})}; copy {arm}")
    if arm is None:
        out["score"] = score(out)
        halt(path, out, "tree: every masked control failed (CI-masked alone, stochastic alone, both at the trigger only); "
                        "no copy run")
        return  # halt exits; the return is for the type checker
    dW, out["teacher"] = make_teacher(tctx, AM.task)
    g1(out, path, AM.task)
    net = teacher.teacher_net(tctx.model, dW)
    references(job, out, path, AM.task, tctx, net)
    copy(job, out, path, tctx, net, arm)
    out["score"], out["done"] = score(out), True
    dump(path, out)
    log(f"[{AM.task} score] {json.dumps(out['score'], default=float)}")


# ----------------------------------------------------------------------------- scoring (pure, JSON only)


def control_class(rec: dict[str, Any]) -> str:
    """A control's rule: any Pile ratio NaN, or the adversarial or stochastic verdict 'fail' -> 'fail'; else either
    'inconclusive' -> 'inconclusive'; else 'pass'. 'pass' and 'inconclusive' both hold for the tree."""
    if any(math.isnan(x) for x in rec["pile_ratios"].values()) or "fail" in rec["pile_verdicts"].values():
        return "fail"
    return "inconclusive" if "inconclusive" in rec["pile_verdicts"].values() else "pass"


def choose(out: dict[str, Any]) -> str | None:
    """The tree on the controls in the file: the first stage-1 recipe whose control holds ('ci' when both hold); if
    none, the fallback when its control is in the file and holds; else None (run the fallback's control, or stop)."""
    held = [n for n in AM.first if control_class(out[f"M0_{n}"]) != "fail"]
    if held:
        return held[0]
    fb = out.get(f"M0_{AM.fallback}")
    return AM.fallback if fb is not None and control_class(fb) != "fail" else None


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on the file; a prediction whose state is missing is absent."""
    last = str(CFG.steps[-1])
    sc: dict[str, Any] = {"C1": out["W0"]["pass"], "G2": out.get("G2", {}).get("pass"), "G1": out.get("G1", {}).get("pass")}
    for n in RECIPE:
        if f"M0_{n}" in out:
            sc[f"C_{n}"] = {"class": control_class(out[f"M0_{n}"]), "pgd": out[f"M0_{n}"]["pgd"],
                            "stochastic": out[f"M0_{n}"]["stochastic"], "lean": LEAN_C[n]}
    arm = next((n for n in RECIPE if last in out.get(f"M_{n}", {})), None)
    if arm is not None:
        m = out[f"M_{arm}"][last]
        sc["copy"] = arm
        sc["P1_copy_masked_model_keeps"] = {"class": ci_class(m), "p_ci_trigger": m["p_ci_trigger"],
                                            "p_target": m["p_target"], "lean": LEAN_P1[arm]}
        sc["P2_copy_faithful"] = {"outcome": faithful(m), "lean": "pass"}
    return sc


# ----------------------------------------------------------------------------- predictions

PREDICTIONS: dict[str, Any] = {
    "about": "E42 Amendment 2, which masked term moves the decomposition with nothing to copy, then the cipher copy with "
             "the term(s) that hold still (reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> 'Amendment 2'; "
             "decisions.md, card decision-e42-masked). Task rand32 only, one job. Every recipe is Amendment 1's (E42's arm "
             "A: task windows, Adam 1e-5, batch 32, batch-order seed 0, 1000 steps, Gamma and Delta frozen; loss "
             "arm-A-weighted KL(teacher || unmasked student) + masked terms matched to the teacher) with "
             "commute.masked_terms' weights: 'ci' = CI-masked 0.5, stochastic 0; 'st' = CI-masked 0, stochastic 0.5; "
             "'trig' = both 0.5, each read as its KL at the trigger alone (no weight on the other 40 positions). A "
             "term with weight 0 is computed without a gradient and logged; the batches and mask draws are Amendment 1's. "
             "Filed before any run",
    "controls": {
        "C1": A1_PRED["controls"]["C1"],
        "G2": A1_PRED["controls"]["G2"] + ". Amendment 1 read 0.0016",
        "M0_ci, M0_st": "stage 1, both always, before the teacher: recipe 'ci' and recipe 'st' with the teacher at W0, "
                        "1000 steps, Pile harness. Expected if a recipe holds still: P(target) and trigger CI-masked "
                        "P(target) at W0's (under 1e-3), damage elsewhere under 0.02, adversarial ratio 0.8-1.24, "
                        "stochastic about 1. Amendment 1's M0 (both terms) read 25.99 / 1.028. Scored as C_ci, C_st",
        "M0_trig": "only when M0_ci and M0_st both fail: recipe 'trig' with the teacher at W0, as above. Scored as C_trig. "
                   "If it fails too the job stops: no copy (decision-e42-criterion stays as it is)",
        "G1": A1_PRED["controls"]["G1"],
        "references": A1_PRED["controls"]["references"],
    },
    "tree": "a control holds when its class is 'pass' or 'inconclusive'. Exactly one of M0_ci, M0_st holds -> the copy "
            "M_<that recipe>; both hold -> M_ci (the term that targets the trigger reading); neither -> M0_trig, then "
            "M_trig if it holds, else stop. A copy run after an 'inconclusive' control: its fail or inconclusive is not "
            "attributed to copying",
    "measurements": {**A1_PRED["measurements"],
                     "control class": "the Pile verdicts of faith_score.state_verdict (each ratio against this job's W0 "
                                      "floor: pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f); f = 0.209 on the L40 gives "
                                      "1.24 / 1.81 adversarial, 1.497 / 1.503 stochastic), read whatever P(target) is"},
    "expected": {
        "W0": A1_PRED["expected"]["W0"],
        "Amendment 1's M0 (both terms, arm A's weighting)": "P 1.6e-5; trigger CI-masked P 1.4e-5; adversarial 25.99 fail, "
                                                            "stochastic 1.028; damage 0.023; size 0.051",
        "E42's copy (arm A, same card)": "P 0.964; trigger CI-masked P 0.007; adversarial 0.960, stochastic 0.936, pass",
        "a control that holds still": "P and trigger CI-masked P under 1e-3; adversarial 0.8-1.24; stochastic about 1; "
                                      "damage under 0.02",
        "M_ci, M_trig": "P >= 0.9; trigger CI-masked P >= 0.5; adversarial <= 1.24; the CI-masked term's trigger KL over "
                        "the last 50 steps well under its first 50",
        "M_st": "P >= 0.9; trigger CI-masked P not trained directly, expected under 0.1; adversarial <= 1.24",
    },
    "noise_floors": A1_PRED["noise_floors"] + "; a control's adversarial ratio between batch orders is unmeasured "
                                              "(Amendment 1's M0 read 11 and 25 nats on its two 128-row blocks)",
    "predictions": [
        {"id": "C_ci", "kind": "classification, filed with a lean",
         "rule": "M0_ci at step 1000: any Pile ratio NaN, or the adversarial or the stochastic verdict 'fail' -> 'fail'; "
                 "else either verdict 'inconclusive' -> 'inconclusive'; else 'pass'. Absolute (each ratio to this job's "
                 "W0 seed 0, against its own floor)",
         "lean": "fail (medium): it trains only the corner where every mask is Gamma's g; the unmasked term pins the "
                 "other corner, so the switched-off subcomponents can change to cancel the switched-on ones' changes, and "
                 "a mask the adversary sets between the corners undoes the cancelling",
         "outcomes": {"fail": "the CI-masked term moves the decomposition with nothing to copy",
                      "pass": "the CI-masked term alone holds still", "inconclusive": "holds for the tree; flagged"},
         "blind_to": "one batch order, one weight, one adversary start of 20 steps on 256 Pile rows; whether the term "
                     "would hold beside the other",
         "positive_control": "Amendment 1's M0 failed at 25.99 on the same card and rows; E42's copy passed at 0.96"},
        {"id": "C_st", "kind": "classification, filed with a lean", "rule": "as C_ci, on M0_st",
         "lean": "pass (low): random masks between g and 1 on every subcomponent penalise exactly the cancelling a CI-only "
                 "fit allows; against it, E37 Amendment 1's control with VPD's stochastic subset loss (and its adversary) "
                 "failed at 3.84, and Amendment 1's analysis expected an average-case term alone to fail",
         "outcomes": {"fail": "the stochastic term moves the decomposition too", "pass": "it holds still alone",
                      "inconclusive": "holds for the tree; flagged"},
         "blind_to": "as C_ci", "positive_control": "as C_ci"},
        {"id": "C_trig", "kind": "classification, filed with a lean; scored only if run",
         "rule": "as C_ci, on M0_trig (run only when C_ci and C_st both read 'fail')",
         "lean": "fail (low): one position of 41 asks less, but the trigger is where the masked students start furthest "
                 "off (0.69 and 0.44 nats) and it still has no worst-case term",
         "outcomes": {"fail": "stop: no frozen-Gamma masked recipe tried here holds still",
                      "pass": "the copy runs with it", "inconclusive": "the copy runs; flagged"},
         "blind_to": "as C_ci", "positive_control": "as C_ci"},
        {"id": "P1_copy_masked_model_keeps", "kind": "classification, filed with a lean per recipe",
         "rule": "the copy that runs (M_ci, M_st or M_trig) at step 1000: Amendment 1's P1 rule (held-out P(target) < 0.9 "
                 "or NaN -> 'did not copy'; else trigger CI-masked P(target): NaN -> 'unscorable'; >= 0.5 -> 'kept'; "
                 "< 0.1 -> 'not kept'; otherwise 'partly'). Absolute",
         "lean": {"ci": "kept (medium-low): the term trains the scored quantity, on the training windows' 32 keys",
                  "st": "not kept (low): the CI-masked model is not trained; random masks favour the switched-on set "
                        "only on average, and a switched-off subcomponent is half on under them",
                  "trig": "kept (medium-low): the scored quantity at the scored position"},
         "outcomes": A1_PRED["predictions"][0]["outcomes"],
         "blind_to": A1_PRED["predictions"][0]["blind_to"] + "; the recipe was picked by its control",
         "positive_control": A1_PRED["predictions"][0]["positive_control"]},
        {"id": "P2_copy_faithful", "kind": "classification, filed with a lean",
         "rule": "the copy that runs, at step 1000: Amendment 1's P2 rule (faith_score.state_verdict: pass / "
                 "inconclusive / fail; 'below P 0.9' -> did not copy; no Pile reading -> unscorable). Absolute",
         "lean": "pass (low): its control held and E42's copy passed with 0.25 to spare; against it, copying moves the "
                 "subcomponents further than the control did",
         "outcomes": A1_PRED["predictions"][1]["outcomes"],
         "blind_to": A1_PRED["predictions"][1]["blind_to"],
         "positive_control": A1_PRED["predictions"][1]["positive_control"]},
    ],
    "reported_not_scored": [
        ("G2's masked terms at W0; every trained state's curve in 50-step blocks (each term, and its KL at the trigger, "
         "the unweighted term included)"),
        ("every state's route shares, gain, unmasked trigger P, every Pile ratio, damage elsewhere, size, teacher gap, "
         "Delta's share; the copy at step 100 (no Pile harness)"),
        "W0's trigger cells; E42's copy re-read at the trigger; E42's floor beside this job's",
    ],
    "not_run": [
        "rand1 and ident32; a second batch order; other weights",
        "the persistent adversary in the loop; the masked terms on Pile rows",
        "M0_trig when a single-term control holds; a second copy when both hold (M_st)",
        "the plain-text leak of the copy (kv_windows.pile_reading on its saved state)",
        "VPD's sparsity loss; a longer adversary or other starts",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e42": asdict(E42), "amend1": asdict(A1), "amend2": asdict(AM),
                "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        return file_predictions()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    for key, cfg in (("config", CFG), ("e42", E42), ("amend1", A1), ("amend2", AM)):
        assert filed[key] == json.loads(json.dumps(asdict(cfg))), f"predictions were filed for another {key}: {filed[key]}"
    run(Job(args.device))
    log("done")


if __name__ == "__main__":
    main()
