"""teacher_training Amendment 6 (2026-10-28): teacher training coverage. Plan: reports/plans/2026-10-23_plan_teacher-training.md
-> "Amendment 6" (binding); gaps 10, 7, 6 and 4 of reports/reviews/2026-10-28_review_technique-comparison.md.

One job, stages cheapest first, every state read on E34's harness (run.read_state) plus two plain-text readings
(`pile_read`, on the harness's 256 Pile val rows, mean over every position): damage = KL(W0 || state) and Pile gap =
KL(teacher || state). Teachers are regenerated in-job: full-weight arm-A fine-tunes (seed 0, step 1000) and E9's rank-8
LoRA (amend3.lora_teacher). Students are E34's recipes (run's CFG) at a batch-order seed (amend4.train_student_seeded).
  C1   W0 under the Pile harness (run.Job); then W0's own plain-text damage must read 0; stop otherwise
  G1   the r1 SGD teacher reproduces E34 (P, size) and its do-nothing harness ratio (1.588); stop otherwise
  a    (gap 10) r1 SGD, Adam 3e-5 and LoRA teachers: do-nothing, scaled adjoint state, students A and B (LoRA: A) at
       step 1000, read without the adversary; P1 scores the students' plain-text damage against their teacher's
  c    (gap 7) arm S (plain SGD 0.03 on (U, V)) on the r1 SGD teacher first (G3: must read 'fail' as in E34), then on
       the r1 Adam and LoRA teachers; harness at step 1000
  b    (gap 6) r2_t275_v2667: SGD 0.1 teacher (gate: P >= 0.5), do-nothing (harness), adjoint, students A and B at 1000
  d    (gap 4) LoRA teachers on emoticon, r1, r0: student B (Pile rows) to 5,000 steps at batch-order seed 0 (harness at
       2,500 and 5,000), then to 2,500 steps at seed 1 (harness at 2,500)
  score P1-P6 (`score`, pure) -> out["score"]
Each state is dumped to results/teacher_training_amend6.json as it is read and skipped on a rerun (same card and W0
floor only, run.load); teachers are regenerated on a rerun (30-40 s each).

Usage (repo root, project venv):  amend6.py --predictions   files results/predictions_amend6.json (refuses to overwrite)
                                  amend6.py [cuda]          the run (GPU); the predictions file must be filed first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from amend3 import lora_teacher
from amend4 import train_student_seeded
from run import (
    CFG,
    EMO,
    LARGE,
    R0,
    R1,
    RESULTS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    read_state,
)

PRED6, OUT6 = RESULTS / "predictions_amend6.json", RESULTS / "teacher_training_amend6.json"
R2 = "r2_t275_v2667"


@dataclass(frozen=True)
class Amend6:
    # full-weight teachers: arm-A fine-tune, seed 0, step 1000 (E31's recipe); r2's SGD rate is E26's (ladder_second_seed)
    ft_lr: dict[str, float] = field(default_factory=lambda: {f"{R1}/sgd": 0.03, f"{R1}/adam": 3e-5, f"{R2}/sgd": 0.1})
    # same-card readings of each regenerated teacher (E34 H100; A3 = E27's H100 LoRA); only G1's halts, the rest are reported
    refs: dict[str, dict[str, float]] = field(default_factory=lambda: {
        f"{R1}/sgd": {"p_target": 0.9834, "size": 0.02875}, f"{R1}/adam": {"p_target": 0.9962, "size": 0.03334},
        f"{EMO}/lora": {"p_target": 0.99992, "damage": 0.01625, "size": 0.11968},
        f"{R1}/lora": {"p_target": 0.99919, "damage": 0.01476, "size": 0.10675},
        f"{R0}/lora": {"p_target": 0.98484, "damage": 0.00360, "size": 0.08415}})
    ref_tol: dict[str, float] = field(default_factory=lambda: {"p_target": 0.01, "damage": 0.005, "size": 0.003})
    g1_harness: float = 1.588  # E34 and A5: the r1 SGD teacher's do-nothing adversarial ratio
    g1_harness_tol: float = 0.03
    w0_pile_max: float = 1e-4  # control: W0's plain-text damage (the two forward paths agree)
    harmless: float = 2.0  # P1: a student's plain-text damage <= this x its teacher's ...
    harmless_abs: float = 0.002  # ... or <= this, in nats
    s_lr: float = 0.03  # arm S on r1 (E34; E21's component-training rate)
    g3_p: float = 0.9857  # G3: E34's arm S on the r1 SGD teacher at step 1000 read P 0.9857 and 'fail' (3.14)
    g3_p_tol: float = 0.02
    r2_p_min: float = 0.5  # (b) gate: an r2 teacher below this is reported and its students skipped
    keep_tol: float = 0.05  # P2, P3: a student keeps the behaviour if its P >= the teacher's P - this
    long_tasks: tuple[str, ...] = (EMO, R1, R0)
    seed0_steps: tuple[int, ...] = (1000, 2500, 5000)  # (d) saved and read (plain text) at these
    seed1_steps: tuple[int, ...] = (1000, 2500)
    long_harness: tuple[int, ...] = (2500, 5000)  # (d) adversary read at these
    plateau: float = 0.03  # P5: rise of the 50-step mean P from 2,500 to 5,000 below this = plateau (A4's floor 0.032)
    mean_over: int = 50
    curve_every: int = 250
    seed_p_tol: float = 0.05  # P6, Amendment 4's P2 rule
    seed_ratio_tol: float = 0.25
    a3_b1000: dict[str, float] = field(default_factory=lambda: {EMO: 0.5667, R1: 0.8753, R0: 0.5210})  # reported only

    def __post_init__(self) -> None:
        assert CFG.steps[-1] == self.seed0_steps[0] == self.seed1_steps[0] == 1000
        assert set(self.long_harness) <= set(self.seed0_steps) and self.seed1_steps[-1] in self.long_harness
        assert self.seed1_steps[-1] % self.curve_every == 0 and self.seed0_steps[-1] % self.curve_every == 0


AM = Amend6()
A_SET = tuple(f"{R1}/{k}/{a}0@1000" for k, a in (("sgd", "A"), ("sgd", "B"), ("adam", "A"), ("adam", "B"), ("lora", "A")))


def pile_read(tctx: Any, w0_net: Any, rows: Any, dW: dict[str, Any], net: Any) -> dict[str, float]:
    """The state W0 + dW on plain text: damage KL(W0 || state) and, given the teacher `net`, the gap KL(teacher || state),
    each the mean over every position of `rows` (finetune.eval_arm_b)."""
    from commute.faithfulness import model_state
    from commute.finetune import eval_arm_b

    with model_state(tctx.model, target={k: tctx.W0[k] + dW[k].to(tctx.W0[k].device) for k in tctx.W0}):
        s = tctx.model.target_model
        rec = {"pile_damage": eval_arm_b(s, w0_net, rows, tctx.device)["kl_to_ref"]}
        if net is not None:
            rec["pile_gap"] = eval_arm_b(s, net, rows, tctx.device)["kl_to_ref"]
    return rec


def ft_teacher(job: Job, task: str, kind: str) -> tuple[dict[str, Any], dict[str, float]]:
    """A full-weight arm-A fine-tune (all 24 matrices, seed 0, step 1000) at AM.ft_lr: delta-W and its own reading."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    c, last = job.task(task)[0], CFG.steps[-1]
    cfg = ArmAConfig(o_token=c.token, lr=AM.ft_lr[f"{task}/{kind}"], steps=last, checkpoints=(last,), seed=0, optimizer=kind)
    res = finetune_arm_a(c.model, c.train, c.held, cfg, c.device, log)
    ev = res.evals[last]
    return res.deltas[last], {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"]}


def summary(curve: list[dict[str, Any]]) -> list[dict[str, float]]:
    """The training curve at step 100 and every AM.curve_every steps: the free held-out P(target) and damage, and the
    means over the last AM.mean_over steps of P and of the training loss."""
    def mean(key: str, s: int) -> float:
        xs = [c[key] for c in curve[max(0, s - AM.mean_over):s]]
        return sum(xs) / len(xs)

    marks = [s for s in range(1, len(curve) + 1) if s == 100 or s % AM.curve_every == 0]
    return [{"step": s, "p_target": curve[s - 1]["heldout_p_o"], "p_mean": mean("heldout_p_o", s),
             "damage": curve[s - 1]["heldout_kl_other"], "loss_mean": mean("loss", s)} for s in marks]


class Run6:
    """The job, its results file, the plain-text rows and W0's frozen copy, and the teachers made so far."""

    def __init__(self, job: Job, out: dict[str, Any]) -> None:
        from commute import teacher

        self.job, self.out = job, out
        self.rows = torch.cat([b for block in job.ctx.blocks for b in block])  # the harness's 256 Pile val rows
        self.w0_net = teacher.teacher_net(job.ctx.model, {})
        self.teachers: dict[tuple[str, str], tuple[dict[str, Any], Any]] = {}

    def teacher(self, task: str, kind: str) -> tuple[dict[str, Any], Any]:
        """The regenerated teacher's delta-W and frozen net; its reading against AM.refs and its gap at W0 recorded."""
        from commute import teacher

        if (task, kind) not in self.teachers:
            dW, reading = lora_teacher(self.job, task) if kind == "lora" else ft_teacher(self.job, task, kind)
            tctx = self.job.task(task)[0]
            net = teacher.teacher_net(tctx.model, dW)
            ref = AM.refs.get(f"{task}/{kind}")
            t = self.out["teachers"].setdefault(f"{task}/{kind}", {})
            t.update(reading=reading, reference=ref,
                     matches=None if ref is None else all(abs(reading[k] - v) <= AM.ref_tol[k] for k, v in ref.items()))
            if "gap_w0" not in t:
                t["gap_w0"] = gap_w0(tctx, net)
            log(f"[{task}/{kind} teacher] {json.dumps(t, default=float)}")
            dump(OUT6, self.out)
            self.teachers[(task, kind)] = (dW, net)
        return self.teachers[(task, kind)]

    def state(self, task: str, label: str, upd: tuple[Any, Any], net: Any, harness: bool) -> dict[str, Any]:
        """E34's record of one state (with the adversary if `harness`), its plain-text readings, and with the adversary
        the faithfulness verdict alone ('faith': state_verdict with no P bar); dumped and logged."""
        from commute import faith_score

        if label in self.out["states"]:
            return self.out["states"][label]
        tctx = self.job.task(task)[0]
        rec = {**read_state(self.job, tctx, label, upd, net, harness), **pile_read(tctx, self.w0_net, self.rows, upd[0], net)}
        if harness:
            rec["faith"] = faith_score.state_verdict({"pgd_kl": rec["pgd"], "stochastic_kl": rec["stochastic"]}, 1.0,
                                                     self.job.w0)["verdict"]
        log(f"[{label}] pile_damage {rec['pile_damage']:.5g} pile_gap {rec.get('pile_gap', float('nan')):.5g}"
            + (f" faith {rec['faith']}" if harness else ""))
        self.out["states"][label] = rec
        dump(OUT6, self.out)
        return rec

    def references(self, task: str, kind: str, harness: bool) -> None:
        """The teacher's do-nothing state and its scaled adjoint state (no adversary on the latter)."""
        from commute import split

        dW, net = self.teacher(task, kind)
        tctx, sigma = self.job.task(task)
        self.state(task, f"{task}/{kind}/nothing", (dW, None), net, harness)
        if f"{task}/{kind}/adjoint" not in self.out["states"]:
            self.state(task, f"{task}/{kind}/adjoint", split.adjoint_state(dW, tctx.uv_cpu, sigma), net, False)

    def student(self, task: str, kind: str, arm: str, seed: int, steps: tuple[int, ...], harness_at: tuple[int, ...],
                save: bool = False) -> dict[int, dict[str, Any]]:
        """Student `arm` (A task windows, B Pile rows, both Adam 1e-5; S task windows, plain SGD AM.s_lr) of one teacher at
        batch-order `seed`, read at `steps` (the adversary at `harness_at`); the curve summarised; the last state saved
        as bf16 offsets if `save`."""
        from commute import arms
        from commute.data import pile_rows

        key = f"{task}/{kind}/{arm}{seed}"
        labels = {s: f"{key}@{s}" for s in steps}
        if any(lb not in self.out["states"] for lb in labels.values()):
            _, net = self.teacher(task, kind)
            tctx = self.job.task(task)[0]
            if arm == "B" and self.job.pile is None:
                self.job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
            tokens, batch, half = ((self.job.pile, CFG.batch_b, None) if arm == "B"
                                   else (tctx.train.tokens, CFG.batch_a, tctx.train.half))
            opt, lr = ("sgd", AM.s_lr) if arm == "S" else ("adam", CFG.student_lr)
            t0 = time.time()
            curve, states = train_student_seeded(tctx, net, opt, lr, steps, tokens, batch, half, seed)
            self.out["curves"][key] = summary(curve)
            log(f"[{key}] {steps[-1]} steps in {time.time() - t0:.0f} s; curve {json.dumps(self.out['curves'][key], default=float)}")
            for s, lb in labels.items():
                st = states.pop(s)
                self.state(task, lb, ({k: v[0] for k, v in st.items()}, st), net, s in harness_at)
                if save and s == steps[-1]:
                    arms.save_offsets(LARGE / task / f"amend6_{kind}_{arm}_seed{seed}_step{s}.pt", tctx.uv_cpu, st,
                                      {"task": task, "teacher": kind, "arm": arm, "seed": seed, "step": s})
        return {s: self.out["states"][lb] for s, lb in labels.items()}


def control(r: Run6) -> None:
    """C1 (run.Job's W0 check) and W0's own plain-text damage, before anything else."""
    if not r.job.w0["pass"]:
        halt(OUT6, r.out, "C1: W0 is not at its known values; nothing read")
    if "W0" not in r.out["states"]:
        ctx = r.job.ctx
        r.out["states"]["W0"] = pile_read(ctx, r.w0_net, r.rows, {k: torch.zeros_like(v) for k, v in ctx.W0.items()}, None)
        log(f"[W0] {json.dumps(r.out['states']['W0'], default=float)}")
        dump(OUT6, r.out)
    if r.out["states"]["W0"]["pile_damage"] > AM.w0_pile_max:
        halt(OUT6, r.out, f"W0 plain-text damage {r.out['states']['W0']['pile_damage']:.3g} > {AM.w0_pile_max:g}")


def g1(r: Run6) -> None:
    """G1: the r1 SGD teacher reproduces E34's P and size and its do-nothing adversarial ratio; stop otherwise."""
    r.references(R1, "sgd", harness=True)
    t, rec = r.out["teachers"][f"{R1}/sgd"], r.out["states"][f"{R1}/sgd/nothing"]
    ok = bool(t["matches"]) and abs(rec["pgd"] - AM.g1_harness) <= AM.g1_harness_tol
    r.out["G1"] = {"reading": t["reading"], "pgd": rec["pgd"], "pass": ok}
    log(f"[G1] {json.dumps(r.out['G1'], default=float)}")
    dump(OUT6, r.out)
    if not ok:
        halt(OUT6, r.out, f"G1: r1 SGD teacher {json.dumps(t['reading'], default=float)}, do-nothing ratio {rec['pgd']:.4f}")


def stage_a(r: Run6) -> None:
    for kind, arms in (("sgd", "AB"), ("adam", "AB"), ("lora", "A")):
        r.references(R1, kind, harness=False)
        for arm in arms:
            r.student(R1, kind, arm, 0, (1000,), ())


def stage_c(r: Run6) -> None:
    rec = r.student(R1, "sgd", "S", 0, (1000,), (1000,))[1000]
    ok = rec["verdict"] == "fail" and abs(rec["p_target"] - AM.g3_p) <= AM.g3_p_tol
    r.out["G3"] = {"verdict": rec["verdict"], "pgd": rec["pgd"], "p_target": rec["p_target"], "pass": ok}
    log(f"[G3] {json.dumps(r.out['G3'], default=float)}" + ("" if ok else ": arm S does not reproduce E34; S arms skipped"))
    dump(OUT6, r.out)
    if ok:
        for kind in ("adam", "lora"):
            r.student(R1, kind, "S", 0, (1000,), (1000,))


def stage_b(r: Run6) -> None:
    r.teacher(R2, "sgd")
    t = r.out["teachers"][f"{R2}/sgd"]
    if t["reading"]["p_target"] < AM.r2_p_min:
        t["skipped"] = f"teacher P {t['reading']['p_target']:.3f} < {AM.r2_p_min}"
        log(f"[{R2}/sgd] {t['skipped']}: students skipped")
        dump(OUT6, r.out)
        return
    r.references(R2, "sgd", harness=True)
    for arm in "AB":
        r.student(R2, "sgd", arm, 0, (1000,), (1000,), save=True)


def stage_d(r: Run6) -> None:
    for seed, steps in ((0, AM.seed0_steps), (1, AM.seed1_steps)):
        for task in AM.long_tasks:
            r.student(task, "lora", "B", seed, steps, tuple(s for s in steps if s in AM.long_harness), save=True)


# ---- scoring (pure: the results file only) -----------------------------------------------------------------------------

def p_mean_at(curve: list[dict[str, Any]], step: int) -> float:
    return next(c["p_mean"] for c in curve if c["step"] == step)


def r2_category(rec: dict[str, Any], p_teacher: float) -> str:
    kept, faithful = rec["p_target"] >= p_teacher - AM.keep_tol, rec["faith"] == "pass"
    return {(True, True): "kept and faithful", (False, True): "faithful, not kept",
            (True, False): "kept, not faithful", (False, False): "neither"}[(kept, faithful)]


def long_category(rec: dict[str, Any], curve: list[dict[str, Any]]) -> str:
    if rec["p_target"] >= CFG.p_min:
        return "learnt" if rec["verdict"] == "pass" else "learnt, not passed"
    rise = p_mean_at(curve, AM.seed0_steps[-1]) - p_mean_at(curve, AM.seed0_steps[-2])
    return "plateau" if rise < AM.plateau else "rising"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1-P6 over whatever the file holds; a prediction whose inputs are missing is absent, and a partial pool never passes."""
    st, sc = out["states"], {"C1": out["W0"]["pass"], "G1": out.get("G1", {}).get("pass"), "G3": out.get("G3", {}).get("pass")}
    p1 = {}
    for lb in A_SET:
        teacher_lb = lb.rsplit("/", 1)[0] + "/nothing"
        if lb in st and teacher_lb in st:
            d, t = st[lb]["pile_damage"], st[teacher_lb]["pile_damage"]
            p1[lb] = {"damage": d, "teacher": t, "holds": d <= AM.harmless * t or d <= AM.harmless_abs}
    if p1:
        sc["P1_harmless_on_plain_text"] = {"pass": len(p1) == len(A_SET) and all(v["holds"] for v in p1.values()), "per": p1}
    t2 = out["teachers"].get(f"{R2}/sgd", {})
    if "skipped" in t2:
        sc["P2_r2_A"] = sc["P3_r2_B"] = {"not_run": t2["skipped"]}
    for pid, arm in (("P2_r2_A", "A"), ("P3_r2_B", "B")):
        lb = f"{R2}/sgd/{arm}0@1000"
        if lb in st:
            cat = r2_category(st[lb], t2["reading"]["p_target"])
            sc[pid] = {"category": cat, "pass": cat == "kept and faithful"}
    s_cat = {"fail": "map", "pass": "teacher", "inconclusive": "between"}
    per = {k: s_cat.get(st[f"{R1}/{k}/S0@1000"]["verdict"], "not learnt") for k in ("adam", "lora") if f"{R1}/{k}/S0@1000" in st}
    if out.get("G3", {}).get("pass") is False:
        sc["P4_sgd_map"] = {"unreadable": "G3 failed"}
    elif len(per) == 2:
        overall = "map" if set(per.values()) == {"map"} else "teacher" if set(per.values()) == {"teacher"} else "mixed"
        sc["P4_sgd_map"] = {"per": per, "overall": overall, "pass": overall == "map"}
    last = AM.seed0_steps[-1]
    cat5 = {t: long_category(st[f"{t}/lora/B0@{last}"], out["curves"][f"{t}/lora/B0"])
            for t in AM.long_tasks if f"{t}/lora/B0@{last}" in st}
    if len(cat5) == len(AM.long_tasks):
        sc["P5_lora_text_only_longer"] = {"per": cat5, "pass": cat5[EMO] == cat5[R1] == "learnt"}
    sc["reported_B0_at_1000_vs_A3"] = {t: [st[f"{t}/lora/B0@1000"]["p_target"], p] for t, p in AM.a3_b1000.items()
                                       if f"{t}/lora/B0@1000" in st}
    mid, p6 = AM.seed1_steps[-1], {}
    for t in AM.long_tasks:
        a, b = st.get(f"{t}/lora/B0@{mid}"), st.get(f"{t}/lora/B1@{mid}")
        if a and b:
            dp, dr = b["p_target"] - a["p_target"], b["pgd"] / a["pgd"] - 1
            p6[t] = {"p_gap": dp, "ratio_gap": dr, "holds": abs(dp) <= AM.seed_p_tol and abs(dr) <= AM.seed_ratio_tol}
    if p6:
        sc["P6_lora_text_only_seed"] = {"per": p6, "pass": len(p6) == len(AM.long_tasks) and all(v["holds"] for v in p6.values())}
    return sc


# ---- predictions and the run --------------------------------------------------------------------------------------------

PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training Amendment 6 (reports/plans/2026-10-23_plan_teacher-training.md -> 'Amendment 6'): gaps 10, 7, 6 "
             "and 4 of reports/reviews/2026-10-28_review_technique-comparison.md in one RunPod H100 job. Teachers regenerated "
             "in-job: arm-A fine-tunes of all 24 matrices (seed 0, step 1000; r1 SGD 0.03 and Adam 3e-5, r2_t275_v2667 SGD 0.1) "
             "and E9's rank-8 LoRA (amend3.lora_teacher) on emoticon, r1, r0. Students use E34's recipes (run.CFG): A task "
             "windows, B Pile train rows 32,768-40,959 (batch 8, plain mean KL), both Adam 1e-5; S task windows with plain "
             "SGD 0.03. E34's earlier predictions files stand unedited. Filed before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2), "
              "faith_score.w0_check; expected pass, floor about 0.075 on the H100. Fails: stop",
        "W0_plain_text": "W0's plain-text damage KL(W0 || W0) over the 256 rows <= 1e-4 (expected 0: two identical forward "
                         "passes). Fails: stop (every damage reading would carry the paths' difference)",
        "G1": "the r1 SGD teacher (fine-tune seed 0, step 1000) reads P within 0.01 of 0.9834 and size within 0.003 of "
              "0.02875 (E34, A4, A5), and its do-nothing adversarial ratio is within 0.03 of 1.588 (E34 and A5 read "
              "1.5884). Fails: stop. Every other teacher's reading is checked against its same-card reference and reported",
        "G3": "stage c, first: arm S on the r1 SGD teacher at step 1000 reads 'fail' with P within 0.02 of 0.9857 (E34: "
              "3.14). Fails: the two new S arms are skipped and P4 reads 'unreadable'",
        "r2_gate": "the r2 teacher's held-out P(target) >= 0.5 at step 1000, else its students are skipped and P2, P3 read "
                   "'not run' (no SGD 0.1 full-weight fine-tune of r2 has been read; E21's Adam 3e-5 one read 0.82)",
    },
    "measurements": {
        "harness": "E34's seven numbers per state (run.read_state): adversarial and stochastic ratios to this job's W0 "
                   "(adversary seed 0), verdict (state_verdict, P bar 0.9), held-out P(target), teacher gap, damage on "
                   "task windows, size, Delta's share. 'faith' = state_verdict on the two ratios with no P bar",
        "plain_text": "pile_damage = mean over every position of the 256 Pile val rows (the harness's rows) of KL(W0 || "
                      "state), the state being the target model at W0 + its delta-W; pile_gap = the same of KL(teacher || "
                      "state). finetune.eval_arm_b, bf16 autocast, KL in fp32. Held fixed: rows, Gamma, Delta. Blind to "
                      "harm the 256 rows do not sample and to damage in the masked model; no batch-order spread exists "
                      "for it (the 2x bar is wide for that reason). Positive control: each teacher's own pile_damage > 0",
        "curve": "every student's training curve at step 100 and every 250 steps: free held-out P(target) and damage, and "
                 "the means over the last 50 steps of P and of the training loss",
    },
    "expected": {
        "W0": "pile_damage 0; every ratio 1; P(target) 0.004 emoticon, 0.15 r0, 0.04 r1, 0.0034 r2",
        "do_nothing": "pile_gap 0 by construction (positive control of the gap); pile_damage of each teacher same order "
                      "as its task-window damage (r1 SGD 0.056, Adam 0.076, LoRA 0.015; no reference on plain text)",
        "a_students": "P 0.97-1.0 at step 1000 (E34, A3); task-window damage 0.55-0.97x the teacher's for full-weight "
                    "teachers, 2.1x for the r1 LoRA's student A (A3)",
        "adjoint": "plain-text damage below the teacher's (E31 A3 and E34 on windows: 0.38-0.55x on r1)",
        "S": "the SGD teacher's S: 'fail' at 1000 (E34 3.14), pass at 100; window damage 1.2x the teacher's",
        "d": "seed 0 at step 1000 reproduces A3's P 0.567, 0.875, 0.521 (emoticon, r1, r0) within about 0.03 (reported)",
        "noise": "A4's batch-order spread: P within 0.032 (Pile-text students), adversarial ratio within 25%; C1's floor",
    },
    "predictions": [
        {"id": "P1_harmless_on_plain_text", "confidence": "high on the four full-weight students; low on the LoRA student",
         "rule": "for each of r1's SGD-teacher A and B, Adam-teacher A and B, and LoRA-teacher A students at step 1000 (seed "
                 "0): pile_damage <= 2 x its teacher's do-nothing pile_damage, or <= 0.002 nats absolute; pass = all "
                 "five present and holding",
         "predicted": "holds on all five (E34 windows: 0.55-0.97x); the LoRA student read 2.1x on windows, so it may not",
         "reported": "each ratio, the scaled adjoint states' ratios, the S and r2 and (d) states' plain-text damage"},
        {"id": "P2_r2_A", "confidence": "medium-low (component training never passed r2 cleanly: E11, E21, E26)",
         "rule": "r2 student A at step 1000: 'kept' = P(target) >= the teacher's P - 0.05; 'faithful' = faith 'pass'; "
                 "category kept and faithful / faithful, not kept / kept, not faithful / neither; pass = kept and faithful",
         "predicted": "kept and faithful (E34: every A student within 0.08 of its teacher and passing)"},
        {"id": "P3_r2_B", "kind": "two-sided, filed with a lean", "rule": "as P2 for student B (Pile rows) at step 1000",
         "lean": "kept and faithful (the cue ' in' is common on the Pile, as r0's and r1's are)"},
        {"id": "P4_sgd_map", "kind": "two-sided, filed with a lean", "confidence": "medium",
         "rule": "arm S at step 1000 on the r1 Adam and LoRA teachers, verdict per teacher: 'fail' -> map, 'pass' -> "
                 "teacher, 'inconclusive' -> between, 'below P 0.9' -> not learnt; overall map if both map, teacher if "
                 "both teacher, else mixed; needs G3; pass = map",
         "lean": "map: SGD on (U, V) fails late on r1 whatever its teacher (E21, E26 component training; E34 arm S)",
         "reported": "P(target) at step 100 (free reading) and the size of each S state (E21: SGD fails above about 0.009)"},
        {"id": "P5_lora_text_only_longer", "confidence": "medium on emoticon and r1; none on r0",
         "rule": "student B of each LoRA teacher at seed 0, step 5000: 'learnt' if P >= 0.9 and verdict 'pass'; 'learnt, "
                 "not passed' if P >= 0.9 otherwise; else 'plateau' if the 50-step mean P rose by less than 0.03 from "
                 "step 2500 to 5000, 'rising' if not; pass = emoticon and r1 both learnt",
         "predicted": "emoticon learnt, r1 learnt (A3 at 1000: 0.567 rising from 0.076, 0.875 from 0.756); r0 not "
                      "predicted (0.462 -> 0.521)",
         "reported": "pile_gap / the teacher's pile_damage at 1000, 2500, 5000 (how much of the teacher's plain-text "
                     "change the student has taken), the curves, damage"},
        {"id": "P6_lora_text_only_seed", "confidence": "medium",
         "rule": "per task, student B at seed 1 against seed 0 at step 2500: |P1 - P0| <= 0.05 and |ratio1 / ratio0 - 1| <= "
                 "0.25 (Amendment 4's P2 rule); pass = all three tasks",
         "predicted": "holds on all three (A4: Pile-text students of full-weight teachers within 0.032 and 25%)"},
    ],
    "reported_not_scored": [
        "every teacher's reading against its same-card reference (only G1 halts); gaps at W0",
        "plain-text damage and gap of every state; the adjoint states' P(target)",
        "(d) seed 0's P at step 1000 against A3's (0.567, 0.875, 0.521)",
        "the r2 teacher's own reading and do-nothing verdict (E21 read an Adam 3e-5 r2 fine-tune at 1.54 on an L40)",
    ],
    "not_run": [
        "the r2 Adam 3e-5 teacher (one teacher keeps stage b short; E21 read that fine-tune at P 0.82 on an L40); r2 S",
        "arm S at a second batch order, on emoticon or r0, or at step-100 harness reads",
        ("the adversary on stage a's students (E34 and A4 read them); harness reads of the LoRA and Adam do-nothing states "
        "(E34, A3 read them on this card)"),
        "student B past 5,000 steps or on more than 8,192 Pile rows; seed 1 past 2,500 steps",
        "a longer adversary or more adversary starts (decided 2026-10-27: one start, 20 steps)",
    ],
}


def file_predictions() -> None:
    if PRED6.is_file():
        sys.exit(f"{PRED6} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED6, {**PREDICTIONS, "config": asdict(CFG), "amend6": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED6}")


def amend6(device: str) -> None:
    if not PRED6.is_file():
        sys.exit(f"file the predictions first: {PRED6} (amend6.py --predictions)")
    filed = json.loads(PRED6.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend6"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend6']}"
    job = Job(device)
    r = Run6(job, load(OUT6, job, {"states": {}, "teachers": {}, "curves": {}}))
    dump(OUT6, r.out)
    control(r)
    g1(r)
    for stage in (stage_a, stage_c, stage_b, stage_d):
        stage(r)
        log(f"[{stage.__name__}] done; score so far {json.dumps(score(r.out), default=float)}")
    r.out["score"], r.out["done"] = score(r.out), True
    dump(OUT6, r.out)
    log(f"[score] {json.dumps(r.out['score'], default=float)}")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend6(args.device)
