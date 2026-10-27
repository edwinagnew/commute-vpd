"""teacher_spread Amendment 1 (E37 A1): train longer, masked teacher training, adjoint start, each read under the
paper's check. Plan: reports/plans/2026-10-26_plan_teacher-spread.md, "Amendment 1 (2026-10-27)".

The teacher, rows, reads and controls C1-C3 are run.py's, imported unchanged. Every arm is teacher training on E37's
2,400 lowercase rows (Adam 1e-5, batch 8, batch order 0, Gamma frozen, Delta fixed):
  arm 1  E37's recipe to step 4000 (`row_copy.copy_states`); read at 1000 (R1, E37's lower_s0 again), 2000 and 4000
  arm 3  the same from the scaled adjoint split's (U', V'); read at 0, 1000 and 2000
  M0     arm 2's recipe with the teacher at W0, 1000 steps: what the masked losses do on their own; one retry at 0.05
  arm 2  masked teacher training (`commute.masked_teacher`): 50 warm-up steps at lr 0, then 2000; read at 1000, 2000
Each read is `run.read` with the budget adversary off: logp, gap, damage, and the Pile harness ratios to C1's W0 seed 0
(the check ratio is the PGD one). `score` derives s, the verdicts against C1's floors, and P1-P3.

Stages, each written to results/amend1.json as it finishes and skipped on a rerun: load and C1, rows and teacher (C2),
W0, do-nothing (C3), arm 3's step 0, arm 1, arm 3, G2, M0, arm 2. C1-C3 stop the run; G2 and M0 stop arm 2 only. Arm 3
runs before M0 and arm 2 because it is cheaper and arm 2's memory and step time are not known.

Usage (repo root, project venv):  amend1.py --predictions   files results/amend1_predictions.json (the lead, once)
                                  amend1.py [--device cuda]  the run (GPU); refuses without the predictions file
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from run import (
    C3_TOL,
    CFG,
    FALLING,
    LARGE,
    RESULTS,
    S_MIN,
    TRANSPORT,
    Job,
    adjoint,
    log,
    read,
)
from torch import Tensor

PRED, OUT = RESULTS / "amend1_predictions.json", RESULTS / "amend1.json"


@dataclass(frozen=True)
class Amend1:
    arm1_reads: tuple[int, ...] = (1000, 2000, 4000)
    arm1_saves: tuple[int, ...] = (1000, 4000)
    arm3_reads: tuple[int, ...] = (1000, 2000)
    arm2_reads: tuple[int, ...] = (1000, 2000)
    m0_steps: int = 1000
    warmup: int = 50  # arm 2 and M0: loop steps at lr 0 before U, V move (E15 protocol v2)
    g2_tol: float = 1e-3  # G2: arm 2's student at W0 against arm 1's step-1 train KL, nats per token
    m0_damage_max: float = 0.02  # M0: damage above this, or the check ratio above m0_check_max, is drift
    m0_check_max: float = 1.40
    m0_retry_coeff: float = 0.05
    m0_arm2_minutes: float = 40.0  # M0 plus arm 2 projected past this: arm 2 stops at its first read
    p3_tol: float = 0.25  # P3: arm 3's check ratio within this, relative, of arm 1's at step 2000
    under: float = 0.85  # reported: M0's check ratio below this means the losses lower the check on any decomposition
    e37: tuple[tuple[str, float], ...] = (("R1_s", 0.998), ("R1_check", 2.20), ("adjoint_s", 0.804), ("adjoint_check", 1.05))

    def __post_init__(self) -> None:
        assert set(self.arm1_saves) <= set(self.arm1_reads) and self.arm1_reads[0] == CFG.copy_steps
        assert self.arm2_reads[0] == self.m0_steps and 0 < self.m0_retry_coeff < 0.5


AM = Amend1()
E37 = dict(AM.e37)


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["status"] = f"stopped: {why}"
    dump(out)
    log(f"STOP: {why}")
    sys.exit(f"STOP: {why}")


def put(out: dict[str, Any], name: str, rec: dict[str, Any]) -> None:
    out["states"][name] = rec
    out["score"] = score(out)
    dump(out)
    log(f"[score] {json.dumps(headline(out['score']))} | {name}: {json.dumps(out['score'].get('derived', {}).get(name, {}))}")


def kl_before(curve: list[dict[str, Any]], n: int, key: str) -> float:
    """Mean of `key` over the 50 steps ending at step n: the train KL beside each read."""
    v = [c[key] for c in curve if n - 50 < c["step"] <= n]
    return sum(v) / len(v)


def state_of(dW: dict[str, Tensor], uv2: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    return {k: (dW[k], *uv2[k]) for k in uv2}


def read_state(job: Job, name: str, st: dict[str, tuple[Tensor, Tensor, Tensor]]) -> dict[str, Any]:
    return read(job, name, {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}, True, budget=False)


def save(job: Job, name: str, step: int, st: dict[str, tuple[Tensor, Tensor, Tensor]]) -> None:
    from commute.arms import save_offsets

    path = LARGE / f"amend1_{name}_step{step}.pt"
    save_offsets(path, job.ctx.uv_cpu, st, {"arm": name, "step": step})
    log(f"[saved] {path.name}")


# ----------------------------------------------------------------------------- stages


def arm_copy(job: Job, out: dict[str, Any], name: str, reads: tuple[int, ...], saves: tuple[int, ...],
             init: dict[str, tuple[Tensor, Tensor]] | None) -> None:
    """Arms 1 and 3: teacher training (`row_copy.copy_states`) from W0's (U, V) or `init`, read at `reads`."""
    from commute.row_copy import copy_states

    t0 = time.time()
    curve, st = copy_states(job.ctx, job.t_net, job.lower, CFG.copy_lr, CFG.copy_batch, reads, 0, log, init=init)
    out["train"][name] = {"kl_step1": curve[0]["loss"], "seconds": time.time() - t0}
    log(f"[{name}] trained {max(reads)} steps in {time.time() - t0:.0f} s; train KL step 1 {curve[0]['loss']:.4f}")
    for n in saves:
        save(job, name, n, st[n])
    for n in reads:
        put(out, f"{name}_{n}", read_state(job, f"{name}_{n}", st[n]) | {"train_kl": kl_before(curve, n, "loss")})


def arm3_start(job: Job, out: dict[str, Any], adj: Any) -> None:
    """Arm 3's step 0: the start it trains from, delta-W = U'V'^T - U V^T with Delta at Delta_0. E37 read the adjoint
    state with the first-order delta-W; the second-order share between the two is logged."""
    dW_adj, uv2 = adj
    st = state_of(start_dw(job.ctx.uv_cpu, uv2), uv2)
    second = (sum(float((st[k][0] - dW_adj[k]).double().norm() ** 2) for k in st)
              / sum(float(dW_adj[k].double().norm() ** 2) for k in st)) ** 0.5
    log(f"[arm3_0] second-order share of the start's delta-W against E37's adjoint state {second:.2e}")
    put(out, "arm3_0", read_state(job, "arm3_0", st) | {"second_order": second})


def start_dw(uv: dict[str, tuple[Tensor, Tensor]], uv2: dict[str, tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    """U'V'^T - U V^T per matrix, float64 then float32 (as `saved_states.offsets_state`)."""
    return {k: (U2.double() @ V2.double().T - uv[k][0].double() @ uv[k][1].double().T).float() for k, (U2, V2) in uv2.items()}


def start_offsets(uv: dict[str, tuple[Tensor, Tensor]], uv2: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor]]:
    """Arm 3's start as offsets to W0's alive (U, V): (U' - U, V' - V) per matrix."""
    return {k: (U2 - uv[k][0], V2 - uv[k][1]) for k, (U2, V2) in uv2.items()}


def g2(job: Job, out: dict[str, Any]) -> None:
    """G2: arm 2's student (components, every mask 1, Delta_0) at W0 on arm 1's first batch, against arm 1's step-1
    train KL on the same rows."""
    from commute.masked_teacher import student_kl

    gen = torch.Generator().manual_seed(0)  # component_training.train's first draw at batch order 0
    toks = job.lower[torch.randint(0, job.lower.shape[0], (CFG.copy_batch,), generator=gen)].to(job.ctx.device)
    with torch.no_grad():
        kl = float(student_kl(job.ctx.model, job.t_net, toks, job.ctx.wd0))
    ref = out["train"]["arm1"]["kl_step1"]
    out["G2"] = {"kl": kl, "arm1_kl_step1": ref, "pass": abs(kl - ref) <= AM.g2_tol}
    log(f"[G2] {json.dumps(out['G2'])}")


def masked(job: Job, out: dict[str, Any], name: str, teacher_net: Any, steps: int, coeff: float | None) -> float:
    """M0 or arm 2: masked teacher training for `steps`, read at each of AM.arm2_reads up to `steps`, the last saved.
    Returns the seconds it took, reads included."""
    from commute.masked_teacher import MaskedConfig, train

    t0, reads = time.time(), tuple(n for n in AM.arm2_reads if n <= steps)
    cfg = MaskedConfig(lr=CFG.copy_lr, batch=CFG.copy_batch, steps=steps, warmup=AM.warmup, save_steps=reads, coeff=coeff)
    curve, st = train(job.ctx.model, teacher_net, job.lower, job.ctx.alive, job.ctx.uv, cfg, log)
    sec = (time.time() - t0) / (AM.warmup + steps)
    out["train"][name] = {"kl_step1": curve[0]["kl"], "coeff": coeff, "sec_per_step": sec, "seconds": time.time() - t0}
    log(f"[{name}] trained in {time.time() - t0:.0f} s ({sec:.3f} s per step); teacher KL at loop step 1 {curve[0]['kl']:.4f}")
    if name == "arm2":  # M0 is read, not kept
        save(job, name, steps, st[steps])
    for n in reads:
        put(out, f"{name}_{n}", read_state(job, f"{name}_{n}", st[n]) | {"train_kl": kl_before(curve, n, "kl")})
    return time.time() - t0


def m0(job: Job, out: dict[str, Any]) -> None:
    """M0, arm 2's control, with one retry at AM.m0_retry_coeff; then arm 2's length by the time rule. Not run when
    G2 failed."""
    spent, rec = 0.0, {"attempts": [], "pass": False}
    out["M0"] = rec
    if not out["G2"]["pass"]:
        return
    for name, coeff in (("M0", None), ("M0b", AM.m0_retry_coeff)):
        spent += masked(job, out, name, job.w0_net, AM.m0_steps, coeff)
        r = out["states"][f"{name}_{AM.m0_steps}"]
        ok = r["damage"] <= AM.m0_damage_max and r["pile"]["pgd_kl"] <= AM.m0_check_max
        rec["attempts"].append({"name": name, "coeff": coeff, "damage": r["damage"], "check": r["pile"]["pgd_kl"], "pass": ok})
        log(f"[M0] {json.dumps(rec['attempts'][-1])}")
        if ok:
            rec |= {"pass": True, "coeff": coeff, "state": f"{name}_{AM.m0_steps}"}
            break
    else:
        return
    tr = out["train"][rec["attempts"][-1]["name"]]
    per_read = (tr["seconds"] - tr["sec_per_step"] * (AM.warmup + AM.m0_steps))  # one read at M0
    projected = tr["sec_per_step"] * (AM.warmup + max(AM.arm2_reads)) + per_read * len(AM.arm2_reads)
    full = (spent + projected) / 60 <= AM.m0_arm2_minutes
    rec |= {"minutes_spent": spent / 60, "arm2_projected_minutes": projected / 60,
            "arm2_steps": max(AM.arm2_reads) if full else AM.m0_steps}
    log(f"[M0] pass at coeff {rec['coeff']}; arm 2 projected {projected / 60:.1f} min after {spent / 60:.1f} min of M0: "
        f"arm 2 runs {rec['arm2_steps']} steps")


def arm2(job: Job, out: dict[str, Any]) -> None:
    if not out["M0"]["pass"]:
        out["arm2"] = "not run: " + ("M0 failed twice" if out["G2"]["pass"] else "G2 failed")
        log(f"[arm2] {out['arm2']}")
        return
    masked(job, out, "arm2", job.t_net, out["M0"]["arm2_steps"], out["M0"]["coeff"])


# ----------------------------------------------------------------------------- scoring (pure)


def derive(r: dict[str, Any], base: float, top: float, c1: dict[str, Any]) -> dict[str, Any]:
    from commute import faith_score

    d: dict[str, Any] = {"s": (r["logp"] - base) / (top - base), "damage": r["damage"]}
    if "pile" in r:
        v = {k: faith_score.verdict(r["pile"][k], c1[f]) for k, f in faith_score.FLOOR_KEY.items()}
        d |= {"check": r["pile"]["pgd_kl"], "stochastic": r["pile"]["stochastic_kl"], "verdict": v,
              "solution": d["s"] >= S_MIN and all(x == "pass" for x in v.values())}
    return d


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on the results file; a prediction whose states are missing is not scored."""
    st = out["states"]
    if not {"W0", "do-nothing"} <= set(st):
        return {}
    d = {n: derive(r, st["W0"]["logp"], st["do-nothing"]["logp"], out["C1"]) for n, r in st.items()}
    sc: dict[str, Any] = {"derived": d}
    if "arm1_4000" in d:
        sc["P1"] = d["arm1_4000"]["verdict"]["pgd_kl"] == "fail"
    if "arm2_2000" in d:
        sc["P2"] = d["arm2_2000"]["solution"]
    if {"arm1_2000", "arm3_2000"} <= set(d):
        a1, a3 = d["arm1_2000"], d["arm3_2000"]
        sc["P3"] = a3["s"] >= S_MIN and abs(a3["check"] / a1["check"] - 1) <= AM.p3_tol
    if "arm1_1000" in d:
        sc["R1"] = {"s": d["arm1_1000"]["s"], "check_over_e37": d["arm1_1000"]["check"] / E37["R1_check"]}
    if "arm3_0" in d:
        sc["arm3_0_over_e37"] = {"s": d["arm3_0"]["s"] - E37["adjoint_s"], "check": d["arm3_0"]["check"] / E37["adjoint_check"]}
    if {"arm1_2000", "arm1_4000"} <= set(d):
        kl2, kl4 = st["arm1_2000"]["train_kl"], st["arm1_4000"]["train_kl"]
        sc["arm1_still_falling"] = (kl2 - kl4) / kl2 >= FALLING and d["arm1_4000"]["check"] < d["arm1_2000"]["check"]
    m0 = out.get("M0", {}).get("state")
    if m0 in d:
        sc["M0_under"] = d[m0]["check"] < AM.under
        sc["arm2_over_M0"] = {n: d[n]["check"] / d[m0]["check"] for n in d if n.startswith("arm2_")}
    if "arm1_4000" in d:
        sc["branch"] = branch(d, sc)
    return sc


def branch(d: dict[str, Any], sc: dict[str, Any]) -> str:
    """The plan's decision tree on what has been read so far."""
    if d["arm1_4000"]["solution"]:
        return "arm 1 a solution at 4000: under-training explains E37; plain teacher training for the larger model"
    if sc.get("arm1_still_falling"):
        return "arm 1 not a solution and still falling: extend arm 1 to 16000 steps"
    found = [a for a in ("arm2", "arm3") if d.get(f"{a}_2000", {}).get("solution")]
    if found:
        return "arm 1 flat; a solution in " + " and ".join(found) + ": card to the user"
    return "arm 1 flat, no solution read: card with option (d) and the anchor question"


def headline(sc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in sc.items() if k != "derived"}


# ----------------------------------------------------------------------------- main


def stage_nothing(job: Job, out: dict[str, Any]) -> None:
    put(out, "do-nothing", read(job, "do-nothing", job.dW, None, False, budget=False))
    dn = out["states"]["do-nothing"]
    out["C3"] = {"logp_minus_teacher": dn["logp"] - out["C2"]["logp_teacher"], "gap": dn["gap"]}
    out["C3"]["pass"] = abs(out["C3"]["logp_minus_teacher"]) <= C3_TOL and abs(dn["gap"]) <= C3_TOL
    log(f"[C3] {json.dumps(out['C3'])}")
    if not out["C3"]["pass"]:
        halt(out, "C3: the do-nothing state does not read as the teacher")


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (amend1.py --predictions)")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"filed for another run.py config: {filed['config']}"
    assert filed["amend1"] == json.loads(json.dumps(asdict(AM))), f"filed for another amendment: {filed['amend1']}"
    for f in ("alive.pt", "windows.pt"):
        if not (TRANSPORT / "large" / f).is_file():
            sys.exit(f"missing input (gitignored): {TRANSPORT / 'large' / f}")
    job = Job(device)
    w0 = job.c1()
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": job.card, "states": {}, "train": {}, "done": []}
    if out["card"] != job.card:
        sys.exit(f"{OUT} was written on {out['card']}; this job runs on {job.card}. Move the file aside to start again")
    out |= {"C1": {k: v for k, v in w0.items() if k != "pile"}, "W0_pile": w0["pile"], "status": "running"}
    dump(out)
    if not w0["pass"]:
        halt(out, "C1: W0 is not at its known values on this card")
    job.rows()
    out["C2"] = job.teacher()
    dump(out)
    if not out["C2"]["pass"]:
        halt(out, f"C2: the teacher's held-out gain {out['C2']['gain']:.4f} is not E36's {CFG.e36_gain}")
    adj = functools.cache(lambda: adjoint(job))
    stages: list[tuple[str, Any]] = [
        ("W0", lambda: put(out, "W0", read(job, "W0", {k: torch.zeros_like(v) for k, v in job.dW.items()}, None, False,
                                           budget=False))),
        ("do-nothing", lambda: stage_nothing(job, out)),
        ("arm3_0", lambda: arm3_start(job, out, adj())),
        ("arm1", lambda: arm_copy(job, out, "arm1", AM.arm1_reads, AM.arm1_saves, None)),
        ("arm3", lambda: arm_copy(job, out, "arm3", AM.arm3_reads, AM.arm3_reads[-1:],
                                     start_offsets(job.ctx.uv_cpu, adj()[1]))),
        ("G2", lambda: g2(job, out)),
        ("M0", lambda: m0(job, out)),
        ("arm2", lambda: arm2(job, out))]
    for name, fn in stages:
        if name not in out["done"]:
            fn()
            out["done"].append(name)
            dump(out)
            torch.cuda.empty_cache()
    out |= {"score": score(out), "status": "done"}
    dump(out)
    log(f"[score] {json.dumps(headline(out['score']))}")
    log(f"FINISHED wrote {OUT}")


NOISE = ("W0's harness floor over adversary seeds 0-2 (0.075 on this card, C1) and E37's batch-order spread of the check "
         "ratio, 14% (lower_s0 2.20, lower_s1 1.88); states are deterministic per seed up to bf16 kernels")


def pred(pid: str, rule: str, confidence: str, blind: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "predicted": "holds", "confidence": confidence, "blind_to": blind,
            "positive_control": control, "noise_floor": NOISE}


VERDICT = "faith_score.verdict against C1's floors f: pass at or under 1.5 / (1 + f), fail at or over 1.5 (1 + f)"
PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_teacher-spread.md, Amendment 1 (2026-10-27). Filed before the run; never edited.",
    "definitions": {
        "s": "(logp_state - logp_W0) / (logp_do-nothing - logp_W0), mean next-token log-probability on 64 held-out "
             "lowercase rows",
        "check ratio": "the Pile harness's PGD KL (calc_multibatch_pgd_masked_recon_loss, one random start, seed 0, 20 "
                       "steps of 0.1, 256 val rows) over C1's W0 at seed 0; the stochastic ratio likewise",
        "verdict": VERDICT, "solution": f"s >= {S_MIN} and verdict 'pass' on the check and stochastic ratios",
        "train KL": "mean teacher KL over the 50 training steps ending at the read",
    },
    "controls": {
        "C1": "first, alone: run.py's (W0 on adversary seeds 0-2, floor within 5% of 0.0748 on an H100 80GB HBM3); stop",
        "C2": "run.py's: the regenerated teacher's held-out gain within 0.005 of 0.1301; stop",
        "C3": f"run.py's: the do-nothing state's logp within {C3_TOL} of the teacher net's and gap within {C3_TOL}; stop",
        "R1": "reported, not a stop: arm 1 at step 1000 reproduces E37's lower_s0 (s 0.998 +- 0.003, check 2.20 +- 15%)",
        "arm3_0": "reported: arm 3's start reproduces E37's adjoint state (s 0.804, check 1.05)",
        "G2": f"arm 2's student at W0 on arm 1's first batch within {AM.g2_tol} of arm 1's step-1 train KL (E37: 0.290); "
              "a miss stops arm 2 (and M0) only",
        "M0": f"before arm 2, alone: arm 2's recipe with the teacher at W0, {AM.m0_steps} steps; damage <= "
              f"{AM.m0_damage_max} and check ratio <= {AM.m0_check_max}, else retry once with both coefficients at "
              f"{AM.m0_retry_coeff}, else arm 2 is not run",
        "time rule": f"M0's time plus arm 2's projection over {AM.m0_arm2_minutes} min: arm 2 stops at {AM.m0_steps} steps",
    },
    "expected": {
        "W0": "s 0, damage 0, check ratio 1 by definition (C1), logp about -3.146, gap about 0.272",
        "do-nothing": "s 1 and gap 0 by construction, damage 0.235; harness not read (4.87 in E37)",
        "arm1": "R1 as above; train KL 0.0110 at step 1000, falling; check ratio falling with steps or flat near 2",
        "arm3": "step 0: s 0.80, check 1.05, damage 0.05; then s rising towards 1 and damage towards 0.23",
        "M0": "damage <= 0.02, check ratio 0.7-1.4, teacher KL at loop step 1 about 0",
        "arm2": "s >= 0.9, damage about 0.23, stochastic ratio below arm 1's (about 1.09)",
    },
    "predictions": [
        pred("P1", "arm 1 at step 4000 still fails: its check-ratio verdict is 'fail' (ratio >= 1.5 (1 + f))",
             "medium-low", "other adversaries (out of scope, 2026-10-27)", "R1"),
        pred("P2", f"arm 2 at step 2000 is a solution: s >= {S_MIN} and verdict 'pass' on the check and stochastic "
                   "ratios; not scored if arm 2 did not reach step 2000", "low-medium",
             "it partly trains on what the stochastic ratio reads; its check ratio is also reported over M0's", "M0"),
        pred("P3", f"arm 3 ends where arm 1 does: at step 2000, s >= {S_MIN} and its check ratio within "
                   f"{AM.p3_tol:.0%} of arm 1's at step 2000", "medium-low",
             "where in weight space the two end (no cosine in this run)", "arm 3's step-0 read reproduces E37's adjoint state"),
    ],
    "reported": "every read's logp, gap, damage, harness ratios, size, cosine with delta-W and train KL; R1 against E37; "
                "arm 1's train KL drop from 2000 to 4000; arm 2's check ratio over M0's; M0 below 0.85; peak memory",
    "decision_tree": {
        "C1-C3 fail": "stop, nothing scored", "M0 or G2 fails": "arms 1 and 3 stand; arm 2 reported as not run",
        "arm 1 a solution at 4000": "under-training explains E37; plain teacher training for the larger model, trained "
                                    "until the train KL falls under 10% per doubling and read at each doubling",
        f"arm 1 not a solution, train KL still falling >= {FALLING:.0%} from 2000 to 4000 and the ratio falling":
            "extend arm 1 to 16000 steps",
        "arm 1 not a solution and flat": "arm 2 a solution: card, masked teacher training (M0's ratio beside it); arm 3 a "
                                         "solution: card, the adjoint start; neither: card with option (d) and the anchor "
                                         "question for broad updates",
    },
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; filed predictions are never edited (add a dated report instead)")
    PRED.write_text(json.dumps({**PREDICTIONS, "config": asdict(CFG), "amend1": asdict(AM),
                                "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"filed {PRED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/amend1_predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    file_predictions() if a.predictions else run(a.device)
