"""teacher_training E42 Amendment 1 (2026-10-28): masked teacher training on a behaviour the model lacks. Plan:
reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> "Amendment 1" (design check, numbers, predictions, tree).

E42's copy (arm A: teacher training on the task windows) kept the cipher and passed the Pile checks, but 84-92% of its
gain on the random ciphers runs through subcomponents Gamma keeps off at the trigger: the CI-masked model's P(target)
there read 0.007 (rand32) and 0.001 (rand1), in E42's logs only. Arm M is arm A with the CI-masked and the
stochastic-masked student added to the loss, each matched to the teacher (commute.masked_copy), Gamma and Delta frozen,
E34's student settings (run.CFG). One task per job (rand32, then rand1); each record is dumped as it finishes to
results/teacher_training_e42_amend1_<task>.json and skipped on a rerun on the same card:
  C1   W0 under the Pile harness, adversary seeds 0, 1, 2 (run.Job); stop if it fails
  G2   the trainer's unmasked student at W0 against the target model on its first batch (the path KL); stop above g2_max
  M0   the masked recipe with the teacher at W0, 1000 steps, Pile harness; stop if either Pile ratio reads 'fail'
  G1   E42's teacher regenerated (newskill.make_teacher, newskill.g1); stop if its step-10 P(target) misses E10's
  refs W0's trigger cells; E42's saved arm-A state re-read at the trigger (no Pile harness), so its CI-masked P is in
       the JSON
  M    the masked recipe with the teacher, read at 100 (no Pile harness) and 1000 (Pile harness); saved as bf16 offsets
  score the filed rules on the task's file

Usage (repo root, project venv, GPU):  newskill_amend1.py --predictions   (files results/predictions_e42_amend1.json once)
                                       newskill_amend1.py rand32           [--device cuda]; then rand1 in its own job
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
from newskill import AM as E42
from newskill import KV, g1, make_teacher
from newskill import out_path as e42_path
from run import CFG, LARGE, RESULTS, Job, dump, halt, load, log, record

PRED = RESULTS / "predictions_e42_amend1.json"


def out_path(name: str) -> Path:
    return RESULTS / f"teacher_training_e42_amend1_{name}.json"


@dataclass(frozen=True)
class Amend1:
    tasks: tuple[str, ...] = ("rand32", "rand1")  # the two random ciphers; rand32 first
    coeff: float = 0.5  # each masked term's weight: the released run's weight on each of its two masked losses
    g2_max: float = 0.01  # G2: path KL at W0 (arm A's weighting, nats); E37 Amendment 1 read 0.00065 per Pile position
    kept: float = 0.5  # P1: trigger CI-masked P(target) >= this -> 'kept'
    lost: float = 0.1  # P1: below this -> 'not kept'; between -> 'partly'
    e42_ci: tuple[tuple[str, float], ...] = (("rand32", 0.007), ("rand1", 0.001))  # E42 arm A @1000, its job logs

    def __post_init__(self) -> None:
        assert set(self.tasks) <= set(E42.tasks) and tuple(t for t, _ in self.e42_ci) == self.tasks
        assert 0 < self.lost < self.kept <= 1 and self.coeff > 0 and self.g2_max > 0


AM = Amend1()
LEAN_P1 = {"rand32": "kept", "rand1": "kept"}


def read(job: Job, tctx: Any, label: str, upd: tuple[Any, Any], net: Any, pile: bool) -> dict[str, Any]:
    """One state read as newskill.read does (run.record, the route split's gain and shares), plus what it drops: the
    trigger P(target) of the CI-masked and unmasked models, every Pile ratio and each ratio's verdict; one log line."""
    from commute import arms, faith_score, harness, teacher
    from commute.faithfulness import model_state

    dW, st = upd
    m = harness.measure(tctx, label, dW, st, pile, log)
    ho = arms.heldout_of(tctx.model, tctx.W0, tctx.held, tctx.token, tctx.device, dW)
    with model_state(tctx.model, target={k: tctx.W0[k] + dW[k].to(tctx.W0[k].device) for k in tctx.W0}):
        gap = teacher.heldout_gap(tctx.model.target_model, net, tctx.held, tctx.device)
    r, pc = m["routes"], m["routes"]["p_cells"]
    rec = {**record(m, ho, gap, job.w0, CFG.p_min), "gain": r["gain"], "scorable": r["scorable"],
           **{f"share_{k}": v for k, v in r["shares"].items()}, "p_ci_trigger": pc["L11"], "p_unmasked_trigger": pc["U1"],
           "pile_ratios": m.get("pile"),
           "pile_verdicts": faith_score.state_verdict(m["pile"], ho["p_o"], job.w0)["pile"] if pile else None}
    log(f"[{label}] " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    return rec


def train_masked(tctx: Any, net: Any, save_steps: tuple[int, ...]) -> tuple[dict[int, Any], dict[str, Any]]:
    """The masked recipe (E34's student settings, arm A's batch) from W0's (U, V): the states at `save_steps` and the
    curve's means over each block of 50 steps (did the masked terms fall, and are they still falling at the end?)."""
    from commute import masked_copy

    cfg = masked_copy.CopyConfig(lr=CFG.student_lr, batch=CFG.batch_a, steps=CFG.steps[-1], save_steps=save_steps,
                                 coeff=AM.coeff)
    t0 = time.time()
    curve, states = masked_copy.train(tctx.model, net, tctx.train, tctx.alive, tctx.uv, cfg, log)
    blocks = [means(curve[i:i + 50]) for i in range(0, len(curve), 50)]
    return states, {"first50": blocks[0], "last50": blocks[-1], "blocks50": blocks, "seconds": time.time() - t0}


def means(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(c[k] for c in rows) / len(rows) for k in rows[0] if k != "step"}


def g2(out: dict[str, Any], path: Path, tctx: Any, net0: Any) -> None:
    """G2: on the trainer's first batch (batch order 0), with the teacher at W0, its unmasked term is the KL between the
    component path and the target model's own forward: about 0. The masked terms at W0 are logged beside it."""
    from commute.masked_copy import window_terms

    idx = torch.randint(0, len(tctx.train), (CFG.batch_a,), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        terms = window_terms(tctx.model, net0, tctx.train.tokens[idx].to(tctx.device), tctx.train.half, tctx.wd0)
    rec = {f"{n}_w0": float(w) for n, (w, _) in terms.items()}
    out["G2"] = {**rec, "path_kl": rec["unmasked_w0"], "pass": rec["unmasked_w0"] <= AM.g2_max}
    log(f"[G2] {json.dumps(out['G2'])}")
    dump(path, out)
    if not out["G2"]["pass"]:
        halt(path, out, f"G2: the trainer's unmasked student at W0 is {rec['unmasked_w0']:.4g} nats from the target model")


def m0(job: Job, out: dict[str, Any], path: Path, name: str, tctx: Any, net0: Any) -> None:
    """M0, alone before arm M: the masked recipe with the teacher at W0 (nothing to copy), read with the Pile harness."""
    last = CFG.steps[-1]
    if "M0" not in out:
        states, tr = train_masked(tctx, net0, (last,))
        st = states[last]
        out["M0"] = {**read(job, tctx, f"{name} M0@{last}", ({k: v[0] for k, v in st.items()}, st), net0, True), "train": tr}
        log(f"[{name} M0 train] {json.dumps(tr)}")
        dump(path, out)
    v, r = out["M0"]["pile_verdicts"], out["M0"]["pile_ratios"]
    if "fail" in v.values() or any(math.isnan(x) for x in r.values()):
        halt(path, out, f"M0: with nothing to copy the masked recipe fails or breaks the Pile check {v} {r}; arm M not run")


def references(job: Job, out: dict[str, Any], path: Path, name: str, tctx: Any, net: Any) -> None:
    """W0's trigger cells (route_cells at W0) and E42's saved arm-A copy re-read at the trigger, without the Pile harness
    (its Pile ratios are in E42's file, same card and floor)."""
    from commute import saved_states

    if "W0_trigger" not in out:
        b = tctx.rbase
        out["W0_trigger"] = {"p_target": b["p_T"], "p_unmasked": b["p_U"], "p_ci": b["p_L"]}
        e42 = json.loads(e42_path(name).read_text())
        out["e42_floor"] = e42["W0"]["pgd_floor"]
        log(f"[{name} W0 trigger] {json.dumps(out['W0_trigger'])}; E42's floor {out['e42_floor']:.6g}, this job's "
            f"{job.w0['pgd_floor']:.6g}")
    if "e42_copy" not in out:
        dW, uv2 = saved_states.offsets_state(LARGE / "e42" / name / f"A_step{CFG.steps[-1]}.pt", tctx.uv_cpu)
        out["e42_copy"] = read(job, tctx, f"{name} E42 copy", (dW, {k: (dW[k], *uv2[k]) for k in dW}), net, False)
        dump(path, out)


def arm_m(job: Job, out: dict[str, Any], path: Path, name: str, tctx: Any, net: Any) -> None:
    from commute import arms

    last = CFG.steps[-1]
    if str(last) in out.get("M", {}):
        return
    states, tr = train_masked(tctx, net, CFG.steps)
    log(f"[{name} M train] {json.dumps(tr)}")
    out["M"] = {"train": tr}
    for s in CFG.steps:
        st = states.pop(s)
        out["M"][str(s)] = read(job, tctx, f"{name} M@{s}", ({k: v[0] for k, v in st.items()}, st), net, s == last)
        if s == last:  # saved before the dump: a rerun skips arm M once its last step is in the JSON
            arms.save_offsets(LARGE / "e42_amend1" / name / f"M_step{s}.pt", tctx.uv_cpu, st,
                              {"task": name, "teacher": "adam 3e-5 step 1000", "arm": "M", "coeff": AM.coeff, "step": s})
        dump(path, out)


def run_task(job: Job, name: str) -> None:
    from commute import kv_windows, teacher

    path = out_path(name)
    out = load(path, job, {"task": name})
    dump(path, out)
    if not job.w0["pass"]:
        halt(path, out, "C1: W0 is not at its known values; nothing trained")
    if out.get("done"):
        return
    tctx, _ = kv_windows.task_context(job.ctx, KV / f"windows_{name}.pt")
    net0 = teacher.teacher_net(tctx.model, {})
    if not out.get("G2", {}).get("pass"):
        g2(out, path, tctx, net0)
    m0(job, out, path, name, tctx, net0)
    dW, out["teacher"] = make_teacher(tctx, name)
    g1(out, path, name)
    net = teacher.teacher_net(tctx.model, dW)
    references(job, out, path, name, tctx, net)
    arm_m(job, out, path, name, tctx, net)
    out["score"], out["done"] = score(out), True
    dump(path, out)
    log(f"[{name} score] {json.dumps(out['score'], default=float)}")


# ----------------------------------------------------------------------------- scoring (pure, JSON only)


def ci_class(rec: dict[str, Any]) -> str:
    """P1's rule on one state record (a NaN P(target) is 'did not copy', as state_verdict reads it)."""
    if not rec["p_target"] >= CFG.p_min:
        return "did not copy"
    p = rec["p_ci_trigger"]
    if math.isnan(p):
        return "unscorable"
    return "kept" if p >= AM.kept else "not kept" if p < AM.lost else "partly"


def faithful(rec: dict[str, Any]) -> str:
    """P2's rule on one state record: faith_score.state_verdict's verdict; no Pile reading -> 'unscorable'."""
    v = rec["verdict"]
    return "unscorable" if v is None else "did not copy" if v.startswith("below P") else v


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on one task's file; a prediction whose state is missing is absent."""
    name, last = out["task"], str(CFG.steps[-1])
    sc: dict[str, Any] = {"C1": out["W0"]["pass"], "G2": out.get("G2", {}).get("pass"), "G1": out.get("G1", {}).get("pass"),
                          "M0": out.get("M0", {}).get("pile_verdicts")}
    if last in out.get("M", {}):
        m = out["M"][last]
        sc["P1_M_masked_model_keeps"] = {"class": ci_class(m), "p_ci_trigger": m["p_ci_trigger"],
                                         "p_target": m["p_target"], "lean": LEAN_P1[name]}
        sc["P2_M_faithful"] = {"outcome": faithful(m), "lean": "pass"}
    return sc


# ----------------------------------------------------------------------------- predictions

PREDICTIONS: dict[str, Any] = {
    "about": "E42 Amendment 1, masked teacher training on a behaviour the model lacks "
             "(reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> 'Amendment 1'). Tasks rand32 then rand1 (E10's "
             "key -> value windows, 403 train and 97 held-out each), one task per job. Teacher: E42's (arm A's loss with the "
             "table target, all 24 matrices, Adam 3e-5, batch 32, seed 0, step 1000), regenerated. Arm M: E42's arm A (task "
             "windows, Adam 1e-5, batch 32, batch-order seed 0, 1000 steps, Gamma and Delta frozen) with "
             "commute.masked_copy's loss: arm-A-weighted KL(teacher || unmasked student) + 0.5 x the same for the CI-masked "
             "student (masks = Gamma's g on the student's activations, Delta out) + 0.5 x the same for the stochastic-masked "
             "student (g + (1 - g) r, Delta under its own random mask, all layers). Step 100 read without the Pile harness, "
             "step 1000 with it. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2), "
              "faith_score.w0_check. Expected: pass, floor 0.2088 on an L40 (E42's), pass line 1.24, fail line 1.81. Stop",
        "G2": "the trainer's unmasked term with the teacher at W0, on its first batch: the component path's KL to the "
              "target model, arm A's weighting. Expected about 1e-3 (E37 Amendment 1: 0.00065 per Pile position). Above "
              "0.01: stop (the trainer's forward is not the model's)",
        "M0": "alone, before the teacher: the masked recipe with the teacher at W0, 1000 steps, Pile harness. Expected: "
              "held-out P(target) and trigger CI-masked P(target) at W0's values (both under 1e-3), damage elsewhere under "
              "0.02, both Pile ratios 'pass' or 'inconclusive'. Either Pile ratio 'fail' (faith_score.verdict against "
              "this job's floors), or any Pile ratio NaN: stop, arm M not run (the recipe moves the decomposition with "
              "nothing to copy, as E37 "
              "Amendment 1's masked control did). 'inconclusive': arm M runs; a fail or inconclusive of arm M is then "
              "not attributed to copying",
        "G1": "E42's: the regenerated teacher's held-out P(target) at step 10 within 0.02 of E10's (rand32 0.5847, rand1 "
              "0.9982). Stop",
        "references": "W0's trigger cells and E42's saved arm-A copy re-read at the trigger in this job: its trigger "
                      "CI-masked P(target) expected within 0.005 of E42's logged 0.007 (rand32) and 0.001 (rand1) (bf16 "
                      "rounding of the saved offsets). Reported, not a stop",
    },
    "measurements": {
        "trigger CI-masked P(target)": "local_faith.route_split's p_cells L11: mean over the 97 held-out windows of "
                                       "P(table[key]) at the trigger from the model built from the subcomponents alone "
                                       "under Gamma's masks on the state's own activations, Delta out",
        "held-out P(target)": "arms.heldout_of: the target model W0 + delta-W at the trigger, mean over held-out windows",
        "Pile verdict": "faith_score.state_verdict: P(target) >= 0.9, then the adversarial and stochastic Pile ratios to "
                        "this job's W0 seed 0, each against its own floor (pass <= 1.5 / (1 + f), fail >= 1.5 (1 + f))",
    },
    "expected": {
        "W0": "held-out P(target) about 1.5e-5 (rand32), 1.7e-4 (rand1); trigger CI-masked P(target) about the same; ratios 1",
        "do-nothing (E42, same card)": "P 0.994 / 1.000; trigger CI-masked P 0.016 / 0.001; adversarial ratio 1.50 / 1.56",
        "E42's copy (arm A, same card)": "P 0.964 / 1.000; trigger CI-masked P 0.007 / 0.001; adversarial ratio 0.960 / "
                                         "0.989, stochastic 0.936 / 0.941, pass; e_gate share 0.84 / 0.92",
        "M0": "as in controls",
        "arm M": "P >= 0.9; trigger CI-masked P >= 0.5; adversarial ratio <= 1.24; e_gate share <= 0.25 (reported); Delta's "
                 "share 0 up to rounding; the CI-masked term's trigger KL over the last 50 steps well under its first 50",
    },
    "noise_floors": "held-out P(target) between batch orders 0.005 on task text (E34 Amendment 4); W0's adversarial "
                    "ratio between adversary seeds 0.21 on the L40; a trained copy's ratio between batch orders up to 25% "
                    "(E34 Amendment 4); the trigger CI-masked P between batch orders is unmeasured: the 0.1 and 0.5 lines "
                    "sit 14x and 70x above E42's rand32 copy (0.007)",
    "predictions": [
        {"id": "P1_M_masked_model_keeps", "kind": "classification, filed with a lean per task",
         "rule": "per task, arm M at step 1000: held-out P(target) < 0.9 or NaN -> 'did not copy'; else the trigger CI-masked "
                 "P(target) (p_ci_trigger): NaN -> 'unscorable'; >= 0.5 -> 'kept'; < 0.1 -> 'not kept'; otherwise "
                 "'partly'. Absolute",
         "lean": {"rand32": "kept (medium-low: the CI-masked term trains the scored quantity directly and the held-out "
                            "windows share the training windows' 32 keys; against it, only subcomponents Gamma switches on "
                            "at the key can carry it, and Gamma may switch them off as they move)",
                  "rand1": "kept (medium: one pair, the same argument)"},
         "outcomes": {"kept": "a frozen Gamma can carry the new behaviour in the CI-masked model: E42's masked-off route "
                              "came from the loss, not from Gamma being frozen",
                      "partly": "the masked terms move it but not to the line in 1000 steps",
                      "not kept": "the masked loss stalls: with Gamma frozen the subcomponents it switches on at the "
                                  "trigger cannot carry the behaviour under this recipe",
                      "did not copy": "the masked terms cost the copy itself"},
         "blind_to": "a mean over 97 held-out windows at one position, which share the training windows' keys (new contexts, "
                     "not new keys); whether the map leaks onto keys without the marker in plain text; the adversary at the "
                     "trigger; which subcomponents carry it; one batch order, one coefficient",
         "positive_control": "E42's ident32 copy, whose gain Gamma supports, read 0.929 on the same reading (E42 log, same "
                             "card); in-job negatives: W0's cells and E42's rand copy re-read (0.007 / 0.001)"},
        {"id": "P2_M_faithful", "kind": "classification, filed with a lean",
         "rule": "per task, arm M at step 1000: faith_score.state_verdict's verdict: 'pass' -> pass; 'inconclusive' -> "
                 "inconclusive; 'fail' -> fail; 'below P 0.9' -> did not copy; no Pile reading -> unscorable. Absolute",
         "lean": "pass on both (medium-low: E42's copy passed at 0.96 / 0.99 with 0.25 to spare and the added terms are "
                 "reconstruction losses toward the teacher; against it, E37 Amendment 1's masked control rose to 3.84)",
         "outcomes": {"pass": "with P1 'kept': a frozen-Gamma map keeps the new behaviour in the CI-masked model and passes",
                      "inconclusive": "rerun the task on an H100 (pass line 1.40, fail 1.61), the plan's tree",
                      "fail": "carrying it in the switched-on subcomponents costs the Pile check",
                      "did not copy": "as P1"},
         "blind_to": "one adversary start of 20 steps on 256 Pile rows (E34 Amendment 5: one-start verdicts may be "
                     "lenient); VPD's sparsity loss, not read; damage to plain text beyond the ratios",
         "positive_control": "E42's least-squares split failed on the same card and reading (1.98 rand32, 1.83 rand1)"},
    ],
    "reported_not_scored": [
        "G2's masked terms at W0; M0's and arm M's curve means over the first and last 50 steps, per term and at the trigger",
        ("every state's route shares, gain, unmasked trigger P, every Pile ratio (CI-masked and unmasked included), "
         "damage elsewhere, size, teacher gap, Delta's share; arm M at step 100 (no Pile harness)"),
        "W0's trigger cells; E42's copy re-read at the trigger; E42's floor beside this job's",
    ],
    "not_run": [
        "ident32 (E42's copy already keeps it in the CI-masked model, 0.929)",
        "a coefficient sweep, the CI-masked or the stochastic term alone, masked terms at the trigger only",
        "the persistent adversary in the loop (masked_teacher's recipe; its shapes are for 512-token rows)",
        "a second batch order; the plain-text leak of arm M's map (kv_windows.pile_reading on its saved state)",
        "VPD's sparsity loss; a longer adversary or other starts; Pile-text training (arm B) with masked terms",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e42": asdict(E42), "amend1": asdict(AM),
                "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("task", nargs="?", choices=AM.tasks)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        return file_predictions()
    if args.task is None:
        sys.exit(f"give one task: {AM.tasks}")
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    for key, cfg in (("config", CFG), ("e42", E42), ("amend1", AM)):
        assert filed[key] == json.loads(json.dumps(asdict(cfg))), f"predictions were filed for another {key}: {filed[key]}"
    run_task(Job(args.device), args.task)
    log("done")


if __name__ == "__main__":
    main()
