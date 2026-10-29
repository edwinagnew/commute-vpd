"""E47's stages T1-T5 and Amendment 1's t1b, t2b, t2c, t4b, t5b (plan section 3 and "Amendment 1"). Every stage runs
after `job.c0`; each controls first, then its states, each written as it finishes; the score last (`rules`)."""

from __future__ import annotations

import os
from typing import Any

import recipes as rc
import rules
from job import BF, KV, LARGE, R1_WINDOWS, TT, Job, gate, log, near, read

from commute.faith_score import verdict

# ----------------------------------------------------------------------------- T1, t1b: claim (i)

R1_COPY = TT / "r1_t534_v369" / "amend5_A_sgd_seed0_step1000.pt"
RAND32_COPY = TT / "e42" / "rand32" / "A_step1000.pt"


def t1(job: Job) -> None:
    """r1's SGD do-nothing (the positive control), r1's copy, rand32's do-nothing, rand32's copy. The control in both
    stages: r1's do-nothing at the budget start reads at or above the fail line, else stop (t1b reads it there as the
    state control_budget, then the states at h2)."""
    from commute import harness, kv_windows
    from commute.saved_states import offsets_state

    st, uv = job.out["states"], job.ctx.uv_cpu
    ctrl = "r1_nothing" if job.stage == "t1" else "control_budget"
    if "r1_nothing" not in st:
        r1 = harness.task_context(job.ctx, R1_WINDOWS)[0]
        ev, dW = teacher(r1, 0.03, "sgd", (1000,))
        gate(job, "G_r1_teacher", near(ev["p_o"], 0.9834, 0.02) and near(ev["rel_norm_max"], 0.02875, 0.05, True),
             {"p_target": ev["p_o"], "size": ev["rel_norm_max"], "ref": [0.9834, 0.02875]})
        read(job, ctrl, (dW, None), starts=None if ctrl == "r1_nothing" else ("budget",), stop=True)
        positive_control(job, st[ctrl])
        read(job, "r1_nothing", (dW, None), stop=True)
    positive_control(job, st[ctrl])
    read(job, "r1_copy", offsets_state(R1_COPY, uv))
    if "rand32_nothing" not in st:
        kv = kv_windows.task_context(job.ctx, KV / "windows_rand32.pt")[0]
        res = rc.arm_a_teacher(kv, 3e-5, "adam", 1000, (10, 300, 1000), log)
        p10 = res.evals[10]["p_o"]
        gate(job, "G1_rand32_teacher", near(p10, 0.5847, 0.02), {"p_target_10": p10, "p_target_1000": res.evals[1000]["p_o"]})
        read(job, "rand32_nothing", (res.deltas[1000], None), stop=True)
    read(job, "rand32_copy", offsets_state(RAND32_COPY, uv))
    job.out["score"] = rules.t1(st, job.f)


def positive_control(job: Job, rec: dict[str, Any]) -> None:
    gate(job, "positive_control", rec["ratio"]["budget"] >= job.fail_line(), {"ratio": rec["ratio"], "fail_line": job.fail_line()})


def teacher(ctx: Any, lr: float, opt: str, checkpoints: tuple[int, ...]) -> tuple[dict[str, float], Any]:
    """An arm-A teacher to its last checkpoint: (its eval there, its delta-W there)."""
    res = rc.arm_a_teacher(ctx, lr, opt, checkpoints[-1], checkpoints, log)
    return res.evals[checkpoints[-1]], res.deltas[checkpoints[-1]]


# ----------------------------------------------------------------------------- T2, t2b, t2c: claim (ii)

HIGH = {"t2": 128, "t2b": 32, "t2c": 128}


def t2(job: Job) -> None:
    """The Adam teacher-trained decomposition (positive control, h0 only), the rank-8 LoRA's, and a higher rank's;
    each read from bf16 offsets. The recipe gates first: the regenerated Adam and rank-8 fine-tunes' held-out gains."""
    from commute.saved_states import offsets_state

    low, uv = rc.lower_task(job.ctx, log), job.ctx.uv_cpu
    if not job.out["gates"].get("G_adam_gain", {}).get("pass"):
        g = rc.gain(job.ctx, low, rc.lower_adam(job.ctx, low, log))[1]
        gate(job, "G_adam_gain", near(g, 0.1301, 0.005), {"gain": g, "ref": 0.1301})
    rec = read(job, "adam_tt", offsets_state(BF / "trained_step1000.pt", uv), starts=("h0",))
    if verdict(rec["ratio"]["h0"], job.f) != "fail":
        log("TELL THE LEAD: the positive control adam_tt does not read fail at h0; the cap reading is unscored")
    if not job.out["gates"].get("G_lora8_gain", {}).get("pass"):
        g = rc.gain(job.ctx, low, rc.lower_lora(job.ctx, low, 8, log))[1]
        gate(job, "G_lora8_gain", near(g, 0.1479, 0.005), {"gain": g, "ref": 0.1479})
    read(job, "lora8_tt", offsets_state(BF / "a2_tt_lora_s0.pt", uv))
    rank = HIGH[job.stage]
    path = high_rank(job, low, rank)
    read(job, f"lora{rank}_tt", offsets_state(path, uv))
    job.out["score"] = rules.t2(job.out["states"], job.f, f"lora{rank}_tt")


def high_rank(job: Job, low: rc.Lower, rank: int) -> Any:
    """The rank-`rank` LoRA (amend2's recipe at that rank) teacher-trained by E37's recipe, saved as bf16 offsets to
    results/large/t2_lora_r<rank>_tt.pt (written to a temporary name, then renamed); a file already there is loaded
    instead (the recipe is deterministic on the L40). The source is recorded in the results file."""
    from commute.arms import save_offsets

    path = LARGE / f"t2_lora_r{rank}_tt.pt"
    if path.is_file():
        job.out.setdefault("high_rank", {"source": f"loaded {path.name}"})
        log(f"[lora{rank}] loading {path}")
        return path
    dW = rc.lower_lora(job.ctx, low, rank, log)
    net, g = rc.gain(job.ctx, low, dW)
    job.out["high_rank"] = {"source": f"trained in {job.stage}", "gain": g, "size": rc.size(dW, job.ctx.W0)}
    log(f"[lora{rank}] held-out gain {g:.4f} size {job.out['high_rank']['size']:.4f}")
    st = rc.lower_copy(job.ctx, low, net, log)
    tmp = path.with_suffix(f".{job.stage}.tmp")
    save_offsets(tmp, job.ctx.uv_cpu, st, {"rank": rank, "lr": 1e-4, "order": 0, "step": 1000, "stage": job.stage})
    os.replace(tmp, path)
    job.dump()
    return path


# ----------------------------------------------------------------------------- T3: claim (iii), no adversary

# E31 (adjoint_sgd, r1 SGD 0.01) per seed at steps 10, 100, 1000: keep and size (predictions.json T3 "control")
E31 = {0: {"keep": (0.374, 0.419, 0.130), "size": (0.003353, 0.008031, 0.014093)},
       1: {"keep": (0.389, 0.446, 0.126), "size": (0.003411, 0.007885, 0.014083)}}
STEPS3 = (10, 100, 1000)


def t3(job: Job) -> None:
    """Per seed (0, then 1; a seed is redone whole on a rerun): the 0.01 control, (a) the 0.03 run at matched sizes,
    (b) the 0.01 step-1000 update scaled to the step-10 and step-100 sizes."""
    from commute import harness

    r1, sigma = harness.task_context(job.ctx, R1_WINDOWS)
    seeds = job.out.setdefault("seeds", {})
    for seed in (0, 1):
        if str(seed) not in seeds:
            seeds[str(seed)] = t3_seed(job, r1, sigma, seed)
            job.dump()
    job.out["score"] = rules.t3(seeds)


def keep_of(ctx: Any, sigma: Any, dW: Any, p_ft: float, label: str) -> dict[str, float]:
    """keep = P(target) at the scaled adjoint state / P(target) at W0 + dW (E31's: the route split's T1 cell)."""
    from commute import harness, split

    dW_adj, st = split.adjoint_state(dW, ctx.uv_cpu, sigma)
    p_adj = harness.measure(ctx, label, dW_adj, st, False, log)["routes"]["p_cells"]["T1"]
    rec = {"keep": p_adj / p_ft, "p_adj": p_adj, "p_ft": p_ft}
    log(f"[{label}] keep {rec['keep']:.4f} P adjoint {p_adj:.4f} P update {p_ft:.4f}")
    return rec


def t3_seed(job: Job, r1: Any, sigma: Any, seed: int) -> dict[str, Any]:
    res = rc.arm_a_teacher(r1, 0.01, "sgd", 1000, STEPS3, log, seed=seed)
    ctrl = {s: {**keep_of(r1, sigma, res.deltas[s], res.evals[s]["p_o"], f"s{seed} 0.01@{s}"),
                "size": res.evals[s]["rel_norm_max"]} for s in STEPS3}
    ok = all(near(ctrl[s]["keep"], E31[seed]["keep"][i], 0.03) and near(ctrl[s]["size"], E31[seed]["size"][i], 0.02, True)
             for i, s in enumerate(STEPS3))
    gate(job, f"control_seed{seed}", ok, {"ours": ctrl, "e31": E31[seed]})
    out: dict[str, Any] = {"control": ctrl, "keep_1000": ctrl[1000]["keep"], "a": {}, "b": {}, "detail": {}}
    sizes = size_curve(r1, seed)
    targets = {s: ctrl[s]["size"] for s in (100, 1000)}
    steps = {s: 1 + min(range(len(sizes)), key=lambda i, t=t: abs(sizes[i] - t)) for s, t in targets.items()}
    res3 = rc.arm_a_teacher(r1, 0.03, "sgd", max(steps.values()), tuple(sorted(set(steps.values()))), log, seed=seed)
    for s, step in steps.items():
        ev = res3.evals[step]
        d = {"step": step, "size": ev["rel_norm_max"], "target": targets[s], "p_ft": ev["p_o"]}
        scored = near(ev["rel_norm_max"], targets[s], 0.05, True) and ev["p_o"] >= 0.5
        cell = keep_of(r1, sigma, res3.deltas[step], ev["p_o"], f"s{seed} 0.03@{step}") if scored else None
        out["a"][str(s)] = None if cell is None else {**cell, "keep_ref": ctrl[s]["keep"]}
        out["detail"][f"a{s}"] = d
    for s in (10, 100):
        alpha = ctrl[s]["size"] / ctrl[1000]["size"]
        dWb = {k: alpha * v for k, v in res.deltas[1000].items()}
        p_ft = rc.p_target(r1, dWb)
        cell = keep_of(r1, sigma, dWb, p_ft, f"s{seed} alpha {alpha:.3f}") if p_ft >= 0.5 else None
        out["b"][str(s)] = None if cell is None else {**cell, "keep_ref": ctrl[s]["keep"], "alpha": alpha}
        out["detail"][f"b{s}"] = {"alpha": alpha, "p_ft": p_ft}
    log(f"[T3 seed {seed}] {rules.t3_seed(out)}")
    return out


def size_curve(r1: Any, seed: int) -> list[float]:
    """Pass 1: the SGD 0.03 run's size (largest relative matrix change) after every step to 400."""
    W0, sizes = r1.W0, []

    def on_step(_step: int, params: dict[str, Any]) -> None:
        sizes.append(max(float((p.detach().float() - W0[k]).norm() / W0[k].norm()) for k, p in params.items()))

    rc.arm_a_teacher(r1, 0.03, "sgd", 400, (), log, seed=seed, on_step=on_step)
    return sizes


# ----------------------------------------------------------------------------- T4, t4b: claim (iv)

T4_REF = {1000: {"ft": (0.99961, 0.03209), "tt_p": 0.99675}, 10: {"ft": (0.99931, 0.01125), "tt_p": 0.99941}}


def t4(job: Job) -> None:
    """remainder.py's emoticon Adam 3e-5 fine-tune to the stage's step (G1), do-nothing, its teacher training (G3,
    G2), teacher training plus remainder (the copy's (U', V') with target W0 + dW)."""
    k = 1000 if job.stage == "t4" else 10
    ref, ctx = T4_REF[k], job.ctx
    ev, dW = teacher(ctx, 3e-5, "adam", (k,))
    gate(job, "G1_ft", near(ev["p_o"], ref["ft"][0], 0.02) and near(ev["rel_norm_max"], ref["ft"][1], 0.05, True),
         {"p_target": ev["p_o"], "size": ev["rel_norm_max"], "ref": list(ref["ft"])})
    read(job, "nothing", (dW, None))
    if "plus_remainder" not in job.out["states"]:
        plus_remainder(job, ref, dW)
    st = job.out["states"]
    job.out["score"] = rules.t4(st["nothing"], st["plus_remainder"])


def plus_remainder(job: Job, ref: dict[str, Any], dW: Any) -> None:
    from commute import teacher as tch

    ctx = job.ctx
    _, states = rc.student(ctx, tch.teacher_net(ctx.model, dW), "adam", 1e-5, (1000,), log)
    dW_tt, uv2 = rc.split(states[1000])
    p, share = rc.p_target(ctx, dW_tt), rc.delta_share(ctx, dW_tt, uv2)
    gate(job, "G3_copy", near(p, ref["tt_p"], 0.02) and share <= 1e-6, {"p_target": p, "delta_share": share})
    change = {key: U2.float() @ V2.float().T - ctx.uv_cpu[key][0].float() @ ctx.uv_cpu[key][1].float().T
              for key, (U2, V2) in uv2.items()}
    g2 = max(float((change[key] - dW_tt[key]).abs().max()) for key in dW_tt) / max(float(d.abs().max()) for d in dW_tt.values())
    gate(job, "G2_removal", g2 <= 1e-3, {"removal_vs_copy": g2})
    read(job, "plus_remainder", (dW, uv2), extra={"p_target_copy": p})


# ----------------------------------------------------------------------------- T5, t5b: claim (v)

# Per teacher, the arm-S runs (rate, saved steps) in order; the cells read; the positive control (Adam teacher, 0.03,
# step 1000, h0) is the first run of each job.
RUNS = {"t5": {"adam": ((0.03, (100, 1000)), (0.01, (1000,))), "lora": ((0.01, (1000,)), (0.03, (100,)))},
        "t5b": {"adam": ((0.03, (1000,)), (0.003, (1000,))), "lora": ((0.003, (1000,)),)}}
CELLS = {"t5": ((0.03, 100), (0.01, 1000)), "t5b": ((0.003, 1000),)}
T5_TEACHER_P = {"adam": 0.9962, "lora": 0.9992}


def t5(job: Job) -> None:
    from commute import harness
    from commute import teacher as tch

    r1 = harness.task_context(job.ctx, R1_WINDOWS)[0]
    for kind in ("adam", "lora"):
        res = rc.arm_a_teacher(r1, 3e-5, "adam", 1000, (1000,), log) if kind == "adam" else rc.lora_teacher(r1, log)
        p = res.evals[1000]["p_o"]
        gate(job, f"G_{kind}_teacher", near(p, T5_TEACHER_P[kind], 0.02), {"p_target": p, "size": res.evals[1000]["rel_norm_max"]})
        net = tch.teacher_net(r1.model, res.deltas[1000])
        for lr, saves in RUNS[job.stage][kind]:
            todo = [s for s in saves if (lr, s) in CELLS[job.stage] and cell_name(kind, lr, s) not in job.out["states"]]
            control = (kind, lr) == ("adam", 0.03) and "control" not in job.out["states"]
            if not todo and not control:
                continue
            _, states = rc.student(r1, net, "sgd", lr, saves, log)
            if (kind, lr) == ("adam", 0.03):
                t5_control(job, states[1000])
            for s in todo:
                t5_cell(job, r1, cell_name(kind, lr, s), states[s])
    t5_score(job)


def cell_name(kind: str, lr: float, step: int) -> str:
    return f"{kind}_{lr:g}_{step}"


def t5_control(job: Job, st: Any) -> None:
    rec = read(job, "control", rc.split(st), starts=("h0",))
    gate(job, "positive_control", rec["ratio"]["h0"] >= job.fail_line(), {"ratio": rec["ratio"], "fail_line": job.fail_line()})


def t5_cell(job: Job, r1: Any, name: str, st: Any) -> None:
    """A cell: its held-out P(target); read with `stop_at` only if P >= 0.9."""
    dW, uv2 = rc.split(st)
    p = rc.p_target(r1, dW)
    if p < 0.9:
        job.out["states"][name] = {"p": p, "read": False, "ratio": {}, "complete": False}
        job.dump()
        log(f"[{name}] P(target) {p:.4f} < 0.9: not read")
        return
    read(job, name, (dW, uv2), stop=True, extra={"p": p, "read": True})


def t5_cells(out: dict[str, Any], f: float) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Each cell's `rules.cell_pass` and the count per teacher of cells with P >= 0.9, from one stage's file."""
    cells: dict[str, dict[str, str]] = {"adam": {}, "lora": {}}
    n = {"adam": 0, "lora": 0}
    for kind, pool in cells.items():
        for name, rec in out["states"].items():
            if name.startswith(kind + "_"):
                pool[name[len(kind) + 1:]] = rules.cell_pass({"p": rec["p"], "rec": rec}, f)
                n[kind] += rec["p"] >= 0.9
    return cells, n


def t5_score(job: Job) -> None:
    job.out["score"] = rules.t5(*t5_cells(job.out, job.f))
