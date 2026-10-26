"""teacher_training Amendment 4 (2026-10-26): a second batch order, read under a longer adversary. Plan:
reports/2026-10-23_plan_teacher-training.md -> "Amendment 4" (binding); E34's references in
reports/2026-10-23_analysis_teacher-training.md -> "Full readings".

E34's six teachers are regenerated (run.Job.teacher, fine-tune seed 0). Arms A (task windows) and B (Pile rows) are
retrained at batch-order seed 1, the only training randomness, and read at step 1000 through the harness (20-step
adversary, same-job W0 floor) with P(target) at step 100 from the training curve's free held-out reading. Every
teacher's do-nothing state and every seed-1 student is then read under E20 Amendment 3's budget sweep, as E31's
`budget()`: the first 128 Pile val rows in batches of 8, gamma.pgd_sources with n sign steps of 2/n for n in (20, 160)
from adversary seed 0's random start, cancel.masked_kl; ratio = KL(state) / KL(W0, seed 0) at the same n, pass line
1.241. W0 is read first under adversary seeds 0 and 1 (the noise floor of that measure). Stages, each dumped to
results/teacher_training_amend4.json as it finishes, finished records skipped on a rerun:
  C1   W0 under the Pile harness (faith_score.w0_check); stop if it fails
  C1b  W0 under the budget sweep, adversary seeds 0 and 1, both budgets (reported, no gate)
  per task (r1, r0, emoticon) and teacher (sgd, adam): the teacher (G1 on the first: the r1 SGD teacher reproduces
       E34's P(target) and size, stop otherwise), its gap at W0, do-nothing through the harness and the budget sweep,
       then arms A and B at seed 1: harness read, budget sweep, bf16 offsets to results/large/<task>/<arm>_<opt>_seed1_step1000.pt
  score P1-P3 (`score`, pure) -> out["score"]

Usage (repo root, project venv):  amend4.py --predictions   files results/predictions_amend4.json (refuses to overwrite)
                                  amend4.py [cuda]          the run (GPU); the predictions file must be filed first
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from run import (
    CFG,
    EMO,
    LARGE,
    R0,
    R1,
    RESULTS,
    TASKS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    read_state,
    student_loss,
)

PRED4, OUT4 = RESULTS / "predictions_amend4.json", RESULTS / "teacher_training_amend4.json"
OPTS, ARMS = ("sgd", "adam"), ("A", "B")

# E34 at step 1000 (results/teacher_training_<task>.json, H100, batch-order seed 0): per task, teacher and arm the
# held-out P(target) and the adversarial ratio; the references P2 reads the seed-1 copies against.
E34_1000 = {R1: {"sgd": {"A": (0.9963, 1.0038), "B": (0.9742, 1.0277)}, "adam": {"A": (0.9944, 0.9676), "B": (0.9828, 1.1308)}},
            R0: {"sgd": {"A": (0.9417, 1.2964), "B": (0.9188, 1.1252)}, "adam": {"A": (0.9212, 1.0806), "B": (0.9080, 1.3659)}},
            EMO: {"sgd": {"A": (0.9985, 0.9421), "B": (0.9603, 0.8954)}, "adam": {"A": (0.9968, 0.9591), "B": (0.9021, 1.0493)}}}


@dataclass(frozen=True)
class Amend4:
    seed: int = 1  # batch order of arms A and B (E34: 0); nothing else is random in the student
    budgets: tuple[int, ...] = (20, 160)  # sign steps of the budget adversary, step size 2 / n (E20 Amendment 3)
    budget_rows: int = 128  # the first Pile val rows, as E31's budget()
    budget_batch: int = 8
    budget_pass: float = 1.241  # E20 A3's P10 line on the ratio to W0 seed 0 at the same budget
    w0_budget_seeds: tuple[int, ...] = (0, 1)  # W0 first; seed 1 / seed 0 is the measure's noise floor
    g1_e34: tuple[float, float] = (0.983, 0.0287)  # G1: E34's r1 SGD teacher at step 1000, P(target) and size
    g1_p_tol: float = 0.01
    g1_size_tol: float = 0.002
    p_tol: float = 0.05  # P2: |P(seed 1) - P(seed 0)| <= this
    ratio_tol: float = 0.25  # P2: |ratio(seed 1) / ratio(seed 0) - 1| <= this
    e34: dict[str, dict[str, dict[str, tuple[float, float]]]] = field(default_factory=lambda: dict(E34_1000))
    inconclusive: tuple[tuple[str, str], ...] = ((R1, "sgd"), (R1, "adam"), (EMO, "adam"))  # E34's do-nothing verdicts

    def __post_init__(self) -> None:
        assert self.seed != 0 and self.budgets[0] == 20 and self.w0_budget_seeds[0] == 0
        assert all(opt in self.e34[t] for t in TASKS for opt in OPTS)


AM = Amend4()
LAST = str(CFG.steps[-1])

PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training Amendment 4 (reports/2026-10-23_plan_teacher-training.md -> 'Amendment 4'): E34's six "
             "teachers regenerated (fine-tune seed 0); arms A (task windows, Adam 1e-5, batch 32) and B (Pile train rows "
             "from 32,768, batch 8, plain mean KL, Adam 1e-5) retrained at batch-order seed 1 for 1000 steps and read "
             "at step 1000 through the harness; then E20 Amendment 3's budget sweep (128 Pile val rows, batches of 8, "
             "n sign steps of 2/n for n in 20, 160, adversary seed 0) at W0 (seeds 0, 1), each teacher's do-nothing "
             "state and each seed-1 student. RunPod H100. Filed before any run",
    "controls": {
        "C1_w0": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary) on adversary seeds 0, 1, "
                 "2; faith_score.w0_check. Expected: pass, floor about 0.075 on an H100 (pass <= 1.40, fail >= 1.61). "
                 "Fails: stop, nothing trained",
        "C1b_w0_budget": "before any state: the budget sweep at W0 under adversary seeds 0 and 1 at 20 and 160 steps. "
                         "No gate (W0's ratio to itself is 1). Reported: KL(seed 1) / KL(seed 0) per budget, the noise "
                         "floor of the budget measure (E31 on the L40: 1.07 at 20 steps, 1.21 at 160; E20 A3: W0 seed 0 "
                         "0.660 at 20, 0.709 at 160)",
        "G1_r1_sgd_teacher": "the first teacher regenerated (r1_t534_v369, SGD 0.03, seed 0, step 1000) reproduces "
                             "E34's reading: held-out P(target) within 0.01 of 0.983 and size within 0.002 of 0.0287 "
                             "(E34 also read damage 0.056). Fails: stop before any student",
    },
    "measurements": {
        "harness": "E34's seven per state at step 1000 (run.record through read_state, pile=True): adversarial and "
                   "stochastic ratio to this job's W0 (256 Pile val rows, 20-step adversary, seed 0), verdict "
                   "(faith_score.state_verdict against the same-job floor, p_min 0.9), held-out P(target), teacher gap, "
                   "damage elsewhere, size, Delta's share. Definitions and expected values as results/predictions.json",
        "p_target_100": "P(target) at step 100 from the training curve's free held-out reading (curve[99]['heldout_p_o'], "
                        "arm A's eval against W0 at every step); the step-100 state is not saved or read by the harness",
        "budget": "gamma.pgd_sources(model, batches, n, step_size=2/n, seed=0) on the first 128 Pile val rows in batches "
                  "of 8, then cancel.masked_kl under the sources found; n in (20, 160), each its own run from the seed's "
                  "random start. Ratio = KL(state) / KL(W0, adversary seed 0) at the same n, relative. W0: 1 by "
                  "definition; do-nothing: the teacher's own reading (no reference: E34 read the teachers only under the "
                  "256-row 20-step harness, 0.98-1.59). Pass line 1.241 (E20 A3's P10). Blind to other adversary "
                  "families (per-row masks, restarts), budgets past 160, one adversary seed per state. Positive "
                  "control: the three teachers E34 read inconclusive (r1 SGD 1.59, r1 Adam 1.52, emoticon Adam 1.50) "
                  "should sit above 1.241 at 20 steps. Noise: W0 seed 1 / seed 0 per budget (C1b)",
    },
    "expected": {
        "W0": "1 on every ratio; P(target) 0.004 emoticon, 0.15 r0, 0.04 r1",
        "do_nothing": "the teacher: E34's reproduction readings (P 0.94-1.0, damage 0.008-0.076, size 0.012-0.034), its "
                      "harness verdict (pass r0 both, emoticon SGD; inconclusive r1 both, emoticon Adam), Delta share 1",
        "seed_1_copies": "within the noise of E34's seed-0 copies (P 0.90-1.0, adversarial 0.90-1.37, all 'pass')",
        "budget_time": "E31 on the L40: 20 steps 33 s, 160 steps a further 226 s per state; an H100 about half",
    },
    "not_run": [
        "seed-0 students under the long adversary (arm B's were not saved; arm A's saved states would take over an hour to upload)",
        "harness reads at steps 10 and 100 (P at 100 comes from the free reading only)",
        "arm S (plain SGD) at seed 1",
        "a second fine-tune seed; a second adversary seed on any state but W0",
        "budgets past 160; other adversary families",
        "LoRA teachers (Amendment 3); the trigger-free Pile arm (Amendments 1, 2)",
    ],
    "reported_not_scored": [
        "C1b: W0's seed-1 / seed-0 budget ratio per budget",
        "each teacher's reproduction reading against E34's (G1 gates only the first)",
        "teacher gap, damage, size, stochastic ratio and Delta share of every state",
        "the do-nothing states' budget ratios of the three teachers E34 read pass (r0 both, emoticon SGD)",
        "budget ratios at 20 steps of the copies (P3a reads 160)",
        "the largest P and ratio gaps of P2, as the project's first seed-to-seed spreads for teacher training",
    ],
    "predictions": [
        {"id": "P1_replication",
         "rule": "every seed-1 copy (arms A and B of the six teachers, twelve states) at step 1000: state_verdict 'pass' "
                 "and held-out P(target) >= 0.9",
         "predicted": "holds, 12 of 12 (E34: 12 of 12 at seed 0)",
         "report": "the count passing of 12 and each failing copy's verdict and P"},
        {"id": "P2_noise_floor", "kind": "two-sided",
         "rule": "per copy against E34's seed-0 reading of the same (task, teacher, arm) at step 1000 (amend4.e34): "
                 "|P(seed 1) - P(seed 0)| <= 0.05 and |adversarial ratio(seed 1) / ratio(seed 0) - 1| <= 0.25",
         "predicted": "holds for all 12; reported: the count holding of 12 and the largest P gap and ratio gap (the "
                      "project's first seed-to-seed spreads for teacher training)",
         "why": "E34's seed-0 readings sit at P 0.90-1.0 and ratio 0.90-1.37; the harness's W0 seed floor is 0.075"},
        {"id": "P3a_long_adversary_copies",
         "rule": "every seed-1 copy at step 1000: budget ratio at 160 steps <= 1.241",
         "predicted": "holds, 12 of 12 (E20 A3: the passing adjoint state stayed at 1.05 from 20 to 160; E31 P4)"},
        {"id": "P3b_long_adversary_teachers",
         "rule": "each of the three teachers E34 read inconclusive (r1 SGD, r1 Adam, emoticon Adam): its do-nothing "
                 "state's budget ratio is above both of its copies' at 20 steps and at 160 steps",
         "predicted": "holds for all three at both budgets (E34: teachers 1.50-1.59 against copies 0.96-1.13 under the "
                      "20-step harness)"},
    ],
}


def train_student_seeded(tctx: Any, net: Any, opt: str, lr: float, steps: tuple[int, ...], tokens: Any, batch: int,
                         half: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """run.train_student with the batch-order seed as an argument: the curve and, at `steps`, {k: (delta-W, U_A', V_A')}.
    The offsets start at zero whatever the seed (component_training.Offsets, rank None), so the seed is the batch order."""
    from commute import component_training
    from commute.finetune import ArmAConfig

    a_cfg = ArmAConfig(o_token=tctx.token, batch=batch, steps=steps[-1], checkpoints=steps, seed=seed)
    cfg = component_training.TrainConfig(lr=lr, steps=steps[-1], save_steps=steps, optimizer=opt)
    return component_training.train(tctx.model, tctx.uv, tokens, tctx.held, a_cfg, cfg, tctx.device, tctx.W0, log=log,
                                    loss_fn=lambda n, toks: student_loss(n, net, toks, half))


def at(tctx: Any, dW: dict[str, Any], st: dict[str, Any] | None) -> Any:
    """The model at W0 + dW with the components of `st` ({k: (delta-W, U_A', V_A')}; None: do-nothing, the teacher
    itself), as adjoint_sgd's at() and harness.measure."""
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    comps = None if st is None else components_with(tctx.model, tctx.alive, {k: (v[1], v[2]) for k, v in st.items()})
    return model_state(tctx.model, target={k: tctx.W0[k] + dW[k].to(tctx.W0[k].device) for k in tctx.W0}, components=comps)


def budget_sweep(state: Any, model: Any, bs: list[Any], label: str, seed: int = 0,
                 w0: dict[str, float] | None = None) -> dict[str, float]:
    """E20 Amendment 3's measure inside the model-state context `state`: for each budget n, n sign steps of 2 / n from
    the seed's random start, then the masked KL under the sources found; {str(n): KL}. Logged per budget with its
    seconds and, when `w0` (W0's KLs under seed 0) is given, the ratio."""
    from commute import cancel, gamma

    kl: dict[str, float] = {}
    with state:
        for n in AM.budgets:
            t0 = time.time()
            src, _ = gamma.pgd_sources(model, bs, n, step_size=2 / n, seed=seed)
            kl[str(n)] = v = cancel.masked_kl(model, bs, src)
            ratio = f" ratio {v / w0[str(n)]:.4f}" if w0 else ""
            log(f"[budget {label} seed {seed}] {n} steps of {2 / n:.4f}: KL {v:.5f}{ratio} ({time.time() - t0:.0f} s)")
    return kl


def w0_budget(job: Any, out: dict[str, Any], bs: list[Any]) -> None:
    """C1b: W0 under the budget sweep at adversary seeds 0 and 1; the seed-1 / seed-0 ratio per budget is the measure's
    noise floor. Nothing to gate."""
    w = out.setdefault("budget_w0", {})
    for seed in AM.w0_budget_seeds:
        if str(seed) not in w:
            w[str(seed)] = budget_sweep(contextlib.nullcontext(), job.ctx.model, bs, "W0", seed)
            dump(OUT4, out)
    out["C1b"] = {"seed1_over_seed0": {n: w["1"][n] / w["0"][n] for n in w["0"]}}
    log(f"[C1b] W0 budget KL seed 1 / seed 0: {json.dumps(out['C1b']['seed1_over_seed0'])}")
    dump(OUT4, out)


def g1(out: dict[str, Any], repro: dict[str, Any]) -> None:
    """G1: the r1 SGD teacher reproduces E34's step-1000 reading (same fine-tune seed and card); stop otherwise."""
    p, size = AM.g1_e34
    out["G1"] = {"p_target": repro["p_target"], "size": repro["size"], "e34": [p, size],
                 "pass": abs(repro["p_target"] - p) <= AM.g1_p_tol and abs(repro["size"] - size) <= AM.g1_size_tol}
    log(f"[G1] {json.dumps(out['G1'], default=float)}")
    dump(OUT4, out)
    if not out["G1"]["pass"]:
        halt(OUT4, out, f"G1: the r1 SGD teacher read P {repro['p_target']:.4f}, size {repro['size']:.4f}; E34 {p}, {size}")


def teacher_entry(job: Any, out: dict[str, Any], name: str, opt: str, bs: list[Any]) -> tuple[dict[str, Any], Any]:
    """The teacher's net and its record: reproduction (G1 on the first), gap at W0, do-nothing through the harness and
    under the budget sweep (once)."""
    from commute import teacher

    tctx = job.task(name)[0]
    dW, repro = job.teacher(name, opt)
    if (name, opt) == (TASKS[0], "sgd"):
        g1(out, repro)
    net = teacher.teacher_net(tctx.model, dW)
    t = out["tasks"][name]["teachers"].setdefault(opt, {"repro": repro})
    if "gap_w0" not in t:
        t["gap_w0"] = gap_w0(tctx, net)
        log(f"[{name} {opt}] teacher gap at W0 {t['gap_w0']:.4g}")
    if "nothing" not in t:
        label = f"{name} {opt} do-nothing"
        rec = read_state(job, tctx, label, (dW, None), net, True)
        rec["budget"] = budget_sweep(at(tctx, dW, None), tctx.model, bs, label, w0=out["budget_w0"]["0"])
        t["nothing"] = rec
        dump(OUT4, out)
    return t, net


def student_arm(job: Any, out: dict[str, Any], name: str, opt: str, arm: str, net: Any, bs: list[Any]) -> None:
    """Arm A or B of one teacher at batch-order seed AM.seed, scored at step 1000: the harness read, P(target) at step
    100 from the curve, the budget sweep; saved as bf16 offsets."""
    from commute import arms
    from commute.data import pile_rows

    t = out["tasks"][name]["teachers"][opt]
    if LAST in t.get(arm, {}):
        return
    tctx = job.task(name)[0]
    if arm == "B" and job.pile is None:
        job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    tokens, batch, half = (job.pile, CFG.batch_b, None) if arm == "B" else (tctx.train.tokens, CFG.batch_a, tctx.train.half)
    t0 = time.time()
    curve, states = train_student_seeded(tctx, net, "adam", CFG.student_lr, CFG.steps[-1:], tokens, batch, half, AM.seed)
    log(f"[{name} {opt} {arm} seed {AM.seed}] trained in {time.time() - t0:.0f} s")
    st = states[CFG.steps[-1]]
    d_s = {k: v[0] for k, v in st.items()}
    label = f"{name} {opt} {arm}@{LAST}"
    rec = read_state(job, tctx, label, (d_s, st), net, True)
    rec["p_target_100"] = curve[CFG.steps[0] - 1]["heldout_p_o"]
    log(f"[{label}] P(target)@{CFG.steps[0]} (free reading) {rec['p_target_100']:.4f}")
    rec["budget"] = budget_sweep(at(tctx, d_s, st), tctx.model, bs, label, w0=out["budget_w0"]["0"])
    t[arm] = {LAST: rec}
    dump(OUT4, out)
    arms.save_offsets(LARGE / name / f"{arm}_{opt}_seed{AM.seed}_step{LAST}.pt", tctx.uv_cpu, st,
                      {"task": name, "teacher": opt, "arm": arm, "seed": AM.seed, "step": CFG.steps[-1]})


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1-P3 over the results file (pure; tests/test_teacher_amend4.py). Keys 'task opt arm'; budget ratios are to W0
    under adversary seed 0 at the same budget."""
    w0 = out["budget_w0"]["0"]
    teachers = {(n, o): t for n, task in out["tasks"].items() for o, t in task["teachers"].items()}
    copies = {(n, o, a): t[a][LAST] for (n, o), t in teachers.items() for a in ARMS if LAST in t.get(a, {})}
    p1 = {f"{n} {o} {a}": r["verdict"] == "pass" and r["p_target"] >= CFG.p_min for (n, o, a), r in copies.items()}
    gaps = {f"{n} {o} {a}": {"p": r["p_target"] - AM.e34[n][o][a][0], "ratio": r["pgd"] / AM.e34[n][o][a][1] - 1}
            for (n, o, a), r in copies.items()}
    p2 = {k: abs(g["p"]) <= AM.p_tol and abs(g["ratio"]) <= AM.ratio_tol for k, g in gaps.items()}
    ratios = {f"{n} {o} {a}": {b: r["budget"][b] / w0[b] for b in w0} for (n, o, a), r in copies.items()}
    p3a = {k: v[str(AM.budgets[-1])] <= AM.budget_pass for k, v in ratios.items()}
    p3b: dict[str, Any] = {}
    for n, o in AM.inconclusive:
        if (n, o) in teachers and "nothing" in teachers[(n, o)]:
            mine = {b: teachers[(n, o)]["nothing"]["budget"][b] / w0[b] for b in w0}
            above = {b: all(mine[b] > ratios[f"{n} {o} {a}"][b] for a in ARMS if f"{n} {o} {a}" in ratios) for b in w0}
            p3b[f"{n} {o}"] = {"teacher": mine, "above_both_copies": above, "pass": all(above.values())}
    largest = {"p": max(gaps, key=lambda k: abs(gaps[k]["p"]), default=None),
               "ratio": max(gaps, key=lambda k: abs(gaps[k]["ratio"]), default=None)}
    full = len(copies) == len(TASKS) * len(OPTS) * len(ARMS)  # 12; a partial file scores no prediction as passed
    return {"n_copies": len(copies),
            "P1": {"pass": full and all(p1.values()), "n_pass": sum(p1.values()), "per": p1},
            "P2": {"pass": full and all(p2.values()), "n_hold": sum(p2.values()), "per": p2, "gaps": gaps,
                   "largest": {k: {"copy": c, **gaps[c]} if c else None for k, c in largest.items()}},
            "P3a": {"pass": full and all(p3a.values()), "n_pass": sum(p3a.values()), "per": p3a, "ratios": ratios},
            "P3b": {"pass": full and len(p3b) == len(AM.inconclusive) and all(v["pass"] for v in p3b.values()), "per": p3b}}


def amend4(device: str) -> None:
    from commute.data import batches, pile_rows

    if not PRED4.is_file():
        sys.exit(f"file the predictions first: {PRED4} (amend4.py --predictions)")
    filed = json.loads(PRED4.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend4"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend4']}"
    job = Job(device)
    out = load(OUT4, job, {"tasks": {n: {"task": n, "teachers": {}} for n in TASKS}, "seed": AM.seed})
    dump(OUT4, out)
    if not job.w0["pass"]:
        halt(OUT4, out, "C1: W0 is not at its known values; nothing trained")
    bs = batches(pile_rows("val", 0, AM.budget_rows, log), AM.budget_batch)
    w0_budget(job, out, bs)
    for name in TASKS:
        for opt in OPTS:
            _, net = teacher_entry(job, out, name, opt, bs)
            for arm in ARMS:
                student_arm(job, out, name, opt, arm, net, bs)
    out["score"] = score(out)
    out["done"] = True
    dump(OUT4, out)
    log("[score] " + json.dumps({k: v["pass"] if isinstance(v, dict) else v for k, v in out["score"].items()}))
    log("done")


def file_predictions() -> None:
    if PRED4.is_file():
        sys.exit(f"{PRED4} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED4, {**PREDICTIONS, "config": asdict(CFG), "amend4": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend4(args.device)
