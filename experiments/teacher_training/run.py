"""teacher_training: component training with a fine-tuned model as teacher. Plan: reports/2026-10-23_plan_teacher-training.md
(design check, arms, controls, the seven numbers, P1-P6; its "Lead's decisions" fix the arm order A, B, S).

The student is W0's alive (U, V) with trainable offsets (component_training.train, Gamma and Delta frozen); its loss is
teacher.teacher_loss, KL(teacher || student). A teacher is an arm-A fine-tune of all 24 matrices to step 1000 (E31's
recipe, seed 0), regenerated in-job. Stages, each record dumped as it finishes, finished work skipped on a rerun:
  C1  W0 under the Pile harness, adversary seeds 0, 1, 2 (faith_score.w0_check); stop if it fails
  C2  teacher = W0, arm A on r1's windows, 100 steps; stop unless the step-1 loss and every offset are exactly 0
  C3  teacher = E26's r0 seed-1 Adam 1e-5 state; arm A, 1000 steps; stop unless at step 1000 the teacher gap is
      <= 10% of W0's and P(target) is within 0.05 of the teacher's
  A   per task (r1, r0, emoticon) and teacher (SGD, then Adam 3e-5): the teacher, do-nothing, the scaled adjoint state
      (no Pile), arm A (task windows, Adam 1e-5) at steps 100 and 1000; stop after the first if its gap at 1000 > 50% of W0's
  B   every teacher: Pile train rows from 32,768, batch 8, plain mean KL, Adam 1e-5, steps 100 and 1000
  S   the SGD teachers: arm A with plain SGD at E21's rate

Usage (repo root, project venv, GPU; results/predictions.json filed first):  run.py  -> results/teacher_training_*.json

Amendment 1 (plan -> "Amendment 1"; results/predictions_amend1.json filed first):  run.py --amend1
  -> results/teacher_training_amend1.json. Emoticon, both teachers: arm B as in E34, B-masked (the loss weight is 0 at
  every trigger position) and B-dropped (rows holding a trigger replaced by clean rows), each scored at step 1000, with
  the cosine of the student's weight change to the teacher's delta-W. C1, then G1 (the regenerated SGD-teacher B
  reproduces E34) gate the rest.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED, CONTROLS = RESULTS / "predictions.json", RESULTS / "teacher_training_controls.json"
TRANSPORT = HERE.parent / "transport_test" / "results"
LADDER = HERE.parent / "task_ladder" / "results" / "large"
C3_TEACHER = HERE.parent / "ladder_second_seed" / "results" / "large" / "r0_t352_v310" / "adam_1e-05_seed1_step1000.pt"
EMO, R0, R1 = "emoticon", "r0_t352_v310", "r1_t534_v369"
TASKS = (R1, R0, EMO)
SGD_LR = {EMO: 0.01, R0: 0.03, R1: 0.03}  # E31's teachers and E21's component-training rates
FT_ADAM_LR, TOL = 3e-5, 1e-6
# E31's teachers at step 1000 (held-out P(target), KL elsewhere, size; L40): the reproduction reading, not gating
E31 = {(EMO, "sgd"): (0.998, 0.007, 0.0122), (EMO, "adam"): (0.999, 0.055, 0.0315), (R0, "sgd"): (0.944, 0.047, 0.0300),
       (R0, "adam"): (0.949, 0.065, 0.0352), (R1, "sgd"): (0.991, 0.049, 0.0286), (R1, "adam"): (0.995, 0.078, 0.0342)}


@dataclass(frozen=True)
class Config:
    steps: tuple[int, ...] = (100, 1000)  # scored student steps; the teachers are fine-tuned to the last
    student_lr: float = 1e-5  # arms A and B (Adam); arm S uses SGD_LR
    batch_a: int = 32  # task windows per step (arms A, S, C2, C3)
    batch_b: int = 8  # Pile rows of 512 tokens per step (arm B)
    pile_start: int = 32_768  # Pile train rows past those the windows were mined from (as gamma_refit)
    pile_train_rows: int = 8192  # arm B samples its batches from these
    c2_steps: int = 100
    pile_rows: int = 256
    pile_pgd_steps: int = 20
    pile_seeds: tuple[int, ...] = (0, 1, 2)
    p_min: float = 0.9
    c3_gap_max: float = 0.1  # C3: gap at the last step <= this x the teacher's gap at W0
    c3_p_tol: float = 0.05  # C3: |P(target) - the teacher's| <= this
    first_gap_max: float = 0.5  # first A arm (r1, SGD teacher): stop if its gap at the last step > this x W0's

    def __post_init__(self) -> None:
        assert self.steps == (100, 1000) and self.pile_seeds[0] == 0
        assert (self.pile_rows, self.pile_pgd_steps) == (256, 20), "E26's harness sizes"
        assert self.pile_train_rows >= self.batch_b * 100


CFG = Config()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def load(path: Path, job: Any, fresh: dict[str, Any]) -> dict[str, Any]:
    """A results file to continue (its STOP mark dropped), or `fresh`; refuses one written on another card or W0 floor,
    since its records' verdicts and ratios belong to that job's W0."""
    out = json.loads(path.read_text()) if path.is_file() else {**fresh, "card": job.card, "W0": job.w0}
    out.pop("stopped", None)
    if (out["card"], out["W0"]["pgd_floor"]) != (job.card, job.w0["pgd_floor"]):
        sys.exit(f"{path} was written on {out['card']} with W0 floor {out['W0']['pgd_floor']}; this job is {job.card} with "
                 f"{job.w0['pgd_floor']}. Move the file aside to start it again")
    return out


def halt(path: Path, out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(path, out)
    log(f"STOP: {why}")
    sys.exit(f"STOP: {why}")


def record(m: dict[str, Any], ho: dict[str, float], gap: float, w0: dict[str, Any], p_min: float = 0.9) -> dict[str, Any]:
    """The plan's seven numbers and the verdict of one state, from a harness.measure record `m`, arms.heldout_of's `ho`
    and the teacher gap; the Pile ratios and the verdict are None when `m` has no Pile reading (the adjoint state)."""
    from commute import faith_score

    pile = m.get("pile")
    return {"pgd": pile["pgd_kl"] if pile else None, "stochastic": pile["stochastic_kl"] if pile else None,
            "p_target": ho["p_o"], "teacher_gap": gap, "damage": ho["kl_other"], "size": m["rel_norm_max"],
            "delta_share": m["delta_change_frac"],
            "verdict": faith_score.state_verdict(pile, ho["p_o"], w0, p_min)["verdict"] if pile else None}


def read_state(job: Any, tctx: Any, label: str, upd: tuple[Any, Any], net: Any, pile: bool) -> dict[str, Any]:
    """One state (delta-W, its components or None for do-nothing) read the plan's way, logged in one line."""
    from commute import arms, harness, teacher
    from commute.faithfulness import model_state

    dW, st = upd
    m = harness.measure(tctx, label, dW, st, pile, log)
    ho = arms.heldout_of(tctx.model, tctx.W0, tctx.held, tctx.token, tctx.device, dW)
    with model_state(tctx.model, target={k: tctx.W0[k] + dW[k].to(tctx.W0[k].device) for k in tctx.W0}):
        gap = teacher.heldout_gap(tctx.model.target_model, net, tctx.held, tctx.device)
    rec = {**record(m, ho, gap, job.w0, CFG.p_min), "card": job.card, "w0_floor": job.w0["pgd_floor"]}
    log(f"[{label}] " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
    return rec


class Job:
    """The loaded model and harness, the same-job W0 check, and per-task contexts and teachers made on first use."""

    def __init__(self, device: str) -> None:
        from commute import faith_score, harness

        t0 = time.time()
        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        self.ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, TOL,
                                CFG.pile_rows, CFG.pile_pgd_steps, CFG.pile_seeds, log)
        self.w0 = faith_score.w0_check(self.ctx.pile)
        self.card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        log(f"[C1] {time.time() - t0:.0f} s; " + json.dumps({k: v for k, v in self.w0.items() if k != "pile"}, default=float))
        self.tasks: dict[str, Any] = {}
        self.teachers: dict[tuple[str, str], Any] = {}
        self.pile: Any = None

    def task(self, name: str) -> tuple[Any, Any]:
        from commute import harness

        if name not in self.tasks:
            self.tasks[name] = harness.task_context(self.ctx, None if name == EMO else LADDER / f"windows_{name}.pt")
        return self.tasks[name]

    def teacher(self, name: str, opt: str) -> tuple[dict[str, Any], Any]:
        """The teacher's delta-W (regenerated: arm-A fine-tune, seed 0) and its reproduction reading against E31."""
        from commute.finetune import ArmAConfig, finetune_arm_a

        if (name, opt) not in self.teachers:
            c = self.task(name)[0]
            lr = SGD_LR[name] if opt == "sgd" else FT_ADAM_LR
            cfg = ArmAConfig(o_token=c.token, lr=lr, steps=CFG.steps[-1], checkpoints=CFG.steps[-1:], seed=0, optimizer=opt)
            res = finetune_arm_a(c.model, c.train, c.held, cfg, c.device, log)
            ev = res.evals[CFG.steps[-1]]
            repro = {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"], "e31": E31[(name, opt)]}
            log(f"[{name} teacher {opt} {lr:g}] reproduction {json.dumps(repro, default=float)}")
            self.teachers[(name, opt)] = (res.deltas[CFG.steps[-1]], repro)
        return self.teachers[(name, opt)]


def train_student(tctx: Any, net: Any, opt: str, lr: float, steps: tuple[int, ...], tokens: Any, batch: int,
                  half: int | None) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """Teacher training from W0's (U, V): the curve and, at `steps`, {k: (delta-W, U_A', V_A')}. Batch-order seed 0."""
    from commute import component_training
    from commute.finetune import ArmAConfig

    a_cfg = ArmAConfig(o_token=tctx.token, batch=batch, steps=steps[-1], checkpoints=steps, seed=0)
    cfg = component_training.TrainConfig(lr=lr, steps=steps[-1], save_steps=steps, optimizer=opt)
    return component_training.train(tctx.model, tctx.uv, tokens, tctx.held, a_cfg, cfg, tctx.device, tctx.W0, log=log,
                                    loss_fn=lambda n, toks: student_loss(n, net, toks, half))


def student_loss(n: Any, net: Any, toks: Any, half: int | None) -> Any:
    """teacher.teacher_loss on one batch; a (b, 2, L) batch is (tokens, keep), keep the 0/1 weight per position
    (Amendment 1's B-masked rows, `masked_rows`)."""
    from commute import teacher

    if toks.ndim == 3:
        return teacher.teacher_loss(n, net, toks[:, 0], None, keep=toks[:, 1].float())
    return teacher.teacher_loss(n, net, toks, half)


def gap_w0(tctx: Any, net: Any) -> float:
    from commute import teacher

    return teacher.heldout_gap(tctx.model.target_model, net, tctx.held, tctx.device)


def c2(job: Any, out: dict[str, Any]) -> None:
    from commute import teacher

    tctx = job.task(R1)[0]
    curve, states = train_student(tctx, teacher.teacher_net(tctx.model, {}), "adam", CFG.student_lr, (CFG.c2_steps,),
                                  tctx.train.tokens, CFG.batch_a, tctx.train.half)
    st = states[CFG.c2_steps]
    off = max(max(float((st[k][1] - tctx.uv_cpu[k][0]).abs().max()), float((st[k][2] - tctx.uv_cpu[k][1]).abs().max()))
              for k in st)
    out["C2"] = {"loss_step1": curve[0]["loss"], "max_offset": off, "pass": curve[0]["loss"] == 0.0 and off == 0.0}
    log(f"[C2] {json.dumps(out['C2'])}")
    dump(CONTROLS, out)
    if not out["C2"]["pass"]:
        halt(CONTROLS, out, f"C2: teacher = W0 moved the student (step-1 loss {curve[0]['loss']:.3g}, largest offset {off:.3g})")


def offsets_of(uv_cpu: Any, uv2: dict[str, Any]) -> dict[str, Any]:
    return {k: (U2.float() - uv_cpu[k][0], V2.float() - uv_cpu[k][1]) for k, (U2, V2) in uv2.items()}


def hook_floor(tctx: Any, net: Any, off_t: dict[str, Any]) -> float:
    """The teacher gap with the student's offsets set to the teacher's: the hook path against the full-weight path."""
    from commute import component_training, teacher, vpd

    off = component_training.Offsets(tctx.uv)
    with torch.no_grad():
        for k, (dU, dV) in off_t.items():
            pU, pV = off.d(k)
            pU.copy_(dU)
            pV.copy_(dV)
    handles = component_training.hooks({k: vpd.target_linear(tctx.model, k) for k in tctx.uv}, off)
    try:
        return teacher.heldout_gap(tctx.model.target_model, net, tctx.held, tctx.device)
    finally:
        for h in handles:
            h.remove()


def c3(job: Any, out: dict[str, Any]) -> None:
    from commute import arms, saved_states, teacher

    tctx = job.task(R0)[0]
    dW, uv2 = saved_states.offsets_state(C3_TEACHER, tctx.uv_cpu)
    net = teacher.teacher_net(tctx.model, dW)
    off_t = offsets_of(tctx.uv_cpu, uv2)
    out["C3"] = rec = {"gap_w0": gap_w0(tctx, net), "hook_floor": hook_floor(tctx, net, off_t),
                       "teacher": read_state(job, tctx, "C3 teacher", (dW, {k: (dW[k], *uv2[k]) for k in dW}), net, True)}
    dump(CONTROLS, out)
    log(f"[C3] gap at W0 {rec['gap_w0']:.4g} hook-path floor {rec['hook_floor']:.4g}")
    _, states = train_student(tctx, net, "adam", CFG.student_lr, CFG.steps, tctx.train.tokens, CFG.batch_a, tctx.train.half)
    for s in CFG.steps:
        st = states.pop(s)
        rec[str(s)] = read_state(job, tctx, f"C3 student@{s}", ({k: v[0] for k, v in st.items()}, st), net, True)
        dump(CONTROLS, out)
        if s == CFG.steps[-1]:
            off_s = offsets_of(tctx.uv_cpu, {k: (v[1], v[2]) for k, v in st.items()})
            num = sum(float(((off_s[k][i] - off_t[k][i]) ** 2).sum()) for k in off_t for i in (0, 1))
            rec["offset_recovery"] = 1 - num / sum(float((off_t[k][i] ** 2).sum()) for k in off_t for i in (0, 1))
            arms.save_offsets(LARGE / R0 / f"C3_student_step{s}.pt", tctx.uv_cpu, st, {"arm": "C3", "step": s})
    last = rec[str(CFG.steps[-1])]
    rec["pass"] = (last["teacher_gap"] <= CFG.c3_gap_max * rec["gap_w0"]
                   and abs(last["p_target"] - rec["teacher"]["p_target"]) <= CFG.c3_p_tol)
    log(f"[C3] offset recovery {rec['offset_recovery']:.4f}; pass {rec['pass']}")
    dump(CONTROLS, out)
    if not rec["pass"]:
        halt(CONTROLS, out, f"C3: gap {last['teacher_gap']:.4g} (W0 {rec['gap_w0']:.4g}), P(target) {last['p_target']:.3f} "
                            f"(teacher {rec['teacher']['p_target']:.3f}): the optimiser cannot recover a reachable teacher")


def controls(job: Any) -> None:
    out = load(CONTROLS, job, {})
    out["C1"] = job.w0
    dump(CONTROLS, out)
    if not job.w0["pass"]:
        halt(CONTROLS, out, "C1: W0 is not at its known values; nothing trained")
    if not out.get("C2", {}).get("pass"):
        c2(job, out)
    if not out.get("C3", {}).get("pass"):
        c3(job, out)


def teacher_entry(job: Any, out: dict[str, Any], path: Path, name: str, opt: str) -> tuple[dict[str, Any], Any]:
    """The teacher's net and its record: reproduction, gap at W0, do-nothing and adjoint states (each once)."""
    from commute import split, teacher

    tctx, sigma = job.task(name)
    dW, repro = job.teacher(name, opt)
    net = teacher.teacher_net(tctx.model, dW)
    t = out["teachers"].setdefault(opt, {"repro": repro})
    if "gap_w0" not in t:
        t["gap_w0"] = gap_w0(tctx, net)
        log(f"[{name} {opt}] teacher gap at W0 {t['gap_w0']:.4g}")
    if "nothing" not in t:
        t["nothing"] = read_state(job, tctx, f"{name} {opt} do-nothing", (dW, None), net, True)
        dump(path, out)
    if "adjoint" not in t:
        t["adjoint"] = read_state(job, tctx, f"{name} {opt} adjoint", split.adjoint_state(dW, tctx.uv_cpu, sigma), net, False)
        dump(path, out)
    return t, net


def student_arm(job: Any, out: dict[str, Any], path: Path, name: str, opt: str, arm: str) -> None:
    """Arm A, B or S of one teacher, scored at CFG.steps; A's last state saved as bf16 offsets."""
    from commute import arms
    from commute.data import pile_rows

    if str(CFG.steps[-1]) in out["teachers"].get(opt, {}).get(arm, {}):
        return
    t, net = teacher_entry(job, out, path, name, opt)
    tctx = job.task(name)[0]
    if arm == "B" and job.pile is None:
        job.pile = pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    s_opt, lr = ("sgd", SGD_LR[name]) if arm == "S" else ("adam", CFG.student_lr)
    tokens, batch, half = ((job.pile, CFG.batch_b, None) if arm == "B" else (tctx.train.tokens, CFG.batch_a, tctx.train.half))
    t0 = time.time()
    _, states = train_student(tctx, net, s_opt, lr, CFG.steps, tokens, batch, half)
    log(f"[{name} {opt} {arm}] trained in {time.time() - t0:.0f} s")
    t[arm] = {}
    for s in CFG.steps:
        st = states.pop(s)
        t[arm][str(s)] = read_state(job, tctx, f"{name} {opt} {arm}@{s}", ({k: v[0] for k, v in st.items()}, st), net, True)
        dump(path, out)
        if arm == "A" and s == CFG.steps[-1]:
            arms.save_offsets(LARGE / name / f"A_{opt}_step{s}.pt", tctx.uv_cpu, st,
                              {"task": name, "teacher": opt, "arm": arm, "step": s})


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed}"
    job = Job(device)
    controls(job)
    paths = {n: RESULTS / f"teacher_training_{n}.json" for n in TASKS}
    outs = {n: load(p, job, {"task": n, "teachers": {}}) for n, p in paths.items()}
    for n in TASKS:
        for opt in ("sgd", "adam"):
            student_arm(job, outs[n], paths[n], n, opt, "A")
            t = outs[n]["teachers"][opt]
            if (n, opt) == (TASKS[0], "sgd") and t["A"][str(CFG.steps[-1])]["teacher_gap"] > CFG.first_gap_max * t["gap_w0"]:
                halt(paths[n], outs[n], f"first A arm: gap at {CFG.steps[-1]} above {CFG.first_gap_max:g} of W0's")
    for arm, opts in (("B", ("sgd", "adam")), ("S", ("sgd",))):
        for n in TASKS:
            for opt in opts:
                student_arm(job, outs[n], paths[n], n, opt, arm)
    for n in TASKS:
        outs[n]["done"] = True
        dump(paths[n], outs[n])
    log("done")


# ---- Amendment 1: arm B on emoticon with the trigger masked from the loss or its rows dropped -------------------------

PRED1, OUT1 = RESULTS / "predictions_amend1.json", RESULTS / "teacher_training_amend1.json"


@dataclass(frozen=True)
class Amend1:
    variants: tuple[str, ...] = ("B", "B-masked", "B-dropped")  # B first: the SGD teacher's B is G1
    trigger_key: str = "h.2.mlp.down_proj:2359"  # vpd.EMOTICON_KEY; the windows' trigger is where its g > trigger_g
    trigger_g: float = 0.5  # the windows' threshold (transport_test/results/windows.json)
    near_g: float = 0.1  # reported: positions with g above this left in B-dropped's rows
    spare_rows: int = 1024  # Pile train rows right after arm B's, replacing B-dropped's trigger rows in order
    g1_ratio_tol: float = 0.25  # G1: |adversarial ratio / E34's - 1| <= this
    g1_p_tol: float = 0.05  # G1: |P(target) - E34's| <= this
    e34_b_sgd: tuple[float, float] = (0.8954, 0.9603)  # E34 emoticon SGD-teacher arm B at 1000: adversarial ratio, P
    null_seed: int = 0  # the random direction the cosine's null reads

    def __post_init__(self) -> None:
        assert self.variants[0] == "B" and self.near_g < self.trigger_g


AM = Amend1()


def variant_rows(job: Any, out: dict[str, Any]) -> dict[str, Any]:
    """Each variant's training rows: B, E34's rows; B-masked, the same rows packed (N, 2, 512) with keep = 0 at every
    position where the trigger subcomponent's g > trigger_g (the position whose prediction is the target); B-dropped,
    the same rows with every row holding such a position replaced by the next clean spare row. Counts in out["trigger"]."""
    from commute import data, vpd

    assert AM.trigger_key == vpd.EMOTICON_KEY
    model = job.task(EMO)[0].model
    if job.pile is None:
        job.pile = data.pile_rows("train", CFG.pile_start, CFG.pile_train_rows, log)
    spare = data.pile_rows("train", CFG.pile_start + CFG.pile_train_rows, AM.spare_rows, log)
    t0 = time.time()
    g, g_sp = data.subcomponent_g(model, job.pile, AM.trigger_key), data.subcomponent_g(model, spare, AM.trigger_key)
    hit, rows_hit, sp_hit = g > AM.trigger_g, (g > AM.trigger_g).any(1), (g_sp > AM.trigger_g).any(1)
    g_dropped = data.replace_rows(g, rows_hit, g_sp, sp_hit)
    gen = torch.Generator().manual_seed(0)  # component_training.train's sampler (batch-order seed 0)
    idx = torch.cat([torch.randint(0, job.pile.shape[0], (CFG.batch_b,), generator=gen) for _ in range(CFG.steps[-1])])
    out["trigger"] = {"positions": int(hit.sum()), "rows": int(rows_hit.sum()), "positions_drawn": int(hit[idx].sum()),
                      "rows_drawn": int(rows_hit[idx].sum()), "max_g_dropped": float(g_dropped.max()),
                      "near_left_in_dropped": int((g_dropped > AM.near_g).sum()), "seconds": time.time() - t0}
    log(f"[trigger] {json.dumps(out['trigger'], default=float)}")
    dump(OUT1, out)
    assert out["trigger"]["max_g_dropped"] <= AM.trigger_g
    return {"B": job.pile, "B-masked": torch.stack([job.pile, (~hit).long()], 1),
            "B-dropped": data.replace_rows(job.pile, rows_hit, spare, sp_hit)}


def references(job: Any, dW: dict[str, Any], opt: str) -> dict[str, Any]:
    """The cosine with delta-W of E34's saved arm-A student, of the adjoint state and of a random direction (the null)."""
    from commute import finetune, saved_states, split

    tctx, sigma = job.task(EMO)
    d_a, _ = saved_states.offsets_state(LARGE / EMO / f"A_{opt}_step{CFG.steps[-1]}.pt", tctx.uv_cpu)
    d_adj, _ = split.adjoint_state(dW, tctx.uv_cpu, sigma)
    ref = {"A": finetune.cosines(d_a, dW), "adjoint": finetune.cosines(d_adj, dW),
           "null": finetune.cosines(finetune.random_direction(dW, AM.null_seed), dW)}
    log(f"[emoticon {opt}] cosine with delta-W: " + " ".join(f"{k} {v['all']:.4f}" for k, v in ref.items()))
    return ref


def variant(job: Any, out: dict[str, Any], opt: str, v: str, rows: Any) -> None:
    """One arm-B variant of one teacher: trained to step 1000, read through the harness there, its step-100 P(target)
    from the free held-out reading, its cosine with delta-W; saved as bf16 offsets."""
    from commute import arms, finetune, teacher

    t = out["teachers"].setdefault(opt, {})
    if v in t:
        return
    tctx = job.task(EMO)[0]
    dW, repro = job.teacher(EMO, opt)
    if "cos_ref" not in t:
        t.update(repro=repro, cos_ref=references(job, dW, opt))
    net = teacher.teacher_net(tctx.model, dW)
    t0 = time.time()
    curve, states = train_student(tctx, net, "adam", CFG.student_lr, CFG.steps[-1:], rows, CFG.batch_b, None)
    log(f"[emoticon {opt} {v}] trained in {time.time() - t0:.0f} s")
    st = states[CFG.steps[-1]]
    d_s = {k: x[0] for k, x in st.items()}
    rec = read_state(job, tctx, f"emoticon {opt} {v}@{CFG.steps[-1]}", (d_s, st), net, True)
    rec.update(p_target_100=curve[CFG.steps[0] - 1]["heldout_p_o"], cos=finetune.cosines(d_s, dW))
    log(f"[emoticon {opt} {v}] P(target)@100 (free reading) {rec['p_target_100']:.4f} cosine with delta-W "
        f"{rec['cos']['all']:.4f}")
    t[v] = rec
    dump(OUT1, out)
    arms.save_offsets(LARGE / EMO / f"{v.replace('-', '')}_{opt}_step{CFG.steps[-1]}.pt", tctx.uv_cpu, st,
                      {"task": EMO, "teacher": opt, "arm": v, "step": CFG.steps[-1]})


def g1(out: dict[str, Any]) -> None:
    """G1: the regenerated SGD-teacher arm B reproduces E34's step-1000 reading; stop otherwise."""
    rec, (ratio, p) = out["teachers"]["sgd"]["B"], AM.e34_b_sgd
    out["G1"] = {"pgd": rec["pgd"], "p_target": rec["p_target"], "e34": [ratio, p],
                 "pass": abs(rec["pgd"] / ratio - 1) <= AM.g1_ratio_tol and abs(rec["p_target"] - p) <= AM.g1_p_tol}
    log(f"[G1] {json.dumps(out['G1'], default=float)}")
    dump(OUT1, out)
    if not out["G1"]["pass"]:
        halt(OUT1, out, f"G1: arm B of the SGD teacher read {rec['pgd']:.4g}, P {rec['p_target']:.3f}; E34 {ratio}, {p}")


def amend1(device: str) -> None:
    if not PRED1.is_file():
        sys.exit(f"file the predictions first: {PRED1}")
    filed = json.loads(PRED1.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["amend1"] == json.loads(json.dumps(asdict(AM))), f"predictions were filed for another amendment: {filed['amend1']}"
    for opt in ("sgd", "adam"):
        assert (LARGE / EMO / f"A_{opt}_step{CFG.steps[-1]}.pt").is_file(), f"E34's arm-A {opt} state is missing"
    job = Job(device)
    out = load(OUT1, job, {"task": EMO, "teachers": {}})
    dump(OUT1, out)
    if not job.w0["pass"]:
        halt(OUT1, out, "C1: W0 is not at its known values; nothing trained")
    rows = variant_rows(job, out)
    for opt in ("sgd", "adam"):
        for v in AM.variants:
            variant(job, out, opt, v, rows[v])
            if (opt, v) == ("sgd", "B"):
                g1(out)
    out["done"] = True
    dump(OUT1, out)
    log("done")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--amend1"]
    (amend1 if "--amend1" in sys.argv else run)(args[0] if args else "cuda")
