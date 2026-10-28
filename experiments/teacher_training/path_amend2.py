"""teacher_training E41 Amendment 2 (2026-10-28): short links. Plan: reports/plans/2026-10-28_plan_e41-teacher-path.md
-> "Amendment 2" (the approved branch for E41's P3 and P4 holding; decisions.md 2026-10-28, option (a): each link is
read once, at one batch order and one adversary start, and a link between the lines is "at the line").

The teacher is E41's emoticon fine-tune (E34's Adam 3e-5 arm A, all 24 matrices, batch 32, seed 0), regenerated with
checkpoints at steps 1, 3, 10, 30, 100, 300 and 1000. Teacher training runs as a chain along it. Link 1 trains W0's
(U, V) for 100 steps against T1. Each later link starts from the previous link's (U, V) and trains 100 steps against
the next checkpoint. Every link uses E34's task-text recipe (Adam 1e-5, batch 32, arm A's weighting, Gamma and Delta
frozen) with a fresh Adam state, and link i draws its batches with batch-order seed i. Every link is read through
E34's harness (run.read_state) beside its teacher's do-nothing state. The links at T10, T100 and T1000 are scored
against E41's one-shot copies of the same teachers (E41's records; C1 and PC check this job's harness reads as E41's).
Stages, each record dumped to results/teacher_training_e41_amend2.json as it finishes, every scored number logged:
  C1  W0 under the Pile harness (job load): w0_check passes, the floor is 0.0748 within 10%, and the seed-0 PGD KL
      is E41's within 1%; stop otherwise
  the fine-tune; G1: T10, T100 and T1000 reproduce E41's P(target) and size; stop otherwise
  PC  T10's do-nothing state reads E41's 2.516 within 1% (the positive control: the check sees this update, and the
      harness reads as in E41's job); stop otherwise
  per link, in order: its teacher's do-nothing state, the link, its read, the score so far (logged); C2 after the
      first link (the warm start reproduces its start); a Delta share above 1e-6 stops the run
  score P1-P3 (`score`, pure) -> out["score"]
A rerun keeps the do-nothing reads and redoes the chain from link 1 (its states live in memory only, ~8 min).

Usage (repo root, project venv):  path_amend2.py --predictions   files results/predictions_e41_amend2.json (refuses to overwrite)
                                  path_amend2.py [cuda]          the run (GPU); the predictions file must be filed first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from path import holds, init_of
from run import (
    CFG,
    EMO,
    FT_ADAM_LR,
    RESULTS,
    Job,
    dump,
    gap_w0,
    halt,
    load,
    log,
    read_state,
    student_loss,
)

PRED, OUT = RESULTS / "predictions_e41_amend2.json", RESULTS / "teacher_training_e41_amend2.json"
CHECK = ("pass", "at the line", "fail")  # the check's verdict on the two Pile ratios, in order
LABEL = {"pass": "pass", "inconclusive": "at the line", "fail": "fail"}  # faith_score.verdict's words -> CHECK
ANCHORS = ("10", "100", "1000")  # the links with an E41 one-shot copy of the same teacher
TOO_CLOSE = "teacher too close to W0 to read"  # P1 leaves such a link out (its check verdict is still reported)


@dataclass(frozen=True)
class Amend2:
    links: tuple[int, ...] = (1, 3, 10, 30, 100, 300, 1000)  # fine-tune checkpoints, one chain link each, in order
    link_steps: int = 100  # student steps per link (700 in all)
    link_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)  # batch-order seed of each link: a fresh draw per link
    ft_lr: float = FT_ADAM_LR  # E34's Adam teacher: 3e-5, batch 32, seed 0, all 24 matrices
    c1_floor: float = 0.0748  # C1: W0's harness PGD floor on the H100 (E41 0.07482), within c1_floor_tol relative
    c1_floor_tol: float = 0.10
    # E41's and Amendment 1's W0 seed-0 PGD KL and do-nothing T10 ratio: equal to 16 digits across those two jobs
    e41_w0_pgd: float = 0.69882
    e41_nothing10: float = 2.5159
    repro_tol: float = 0.01  # C1's KL and PC, relative: far above the measured 0, far below the 7.5% adversary floor
    # G1: E41's emoticon teachers (teacher_training_e41.json tasks.emoticon.teachers): P(target), size
    g1_teacher: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "10": (0.99931, 0.01125), "100": (0.99946, 0.02217), "1000": (0.99961, 0.03209)})
    g1_tol: tuple[float, float] = (0.005, 0.001)  # absolute on P(target), size (E41's)
    # E41's one-shot task-text copies (tasks.emoticon.A.oneshot, 1,000 steps from W0, batch-order seed 0)
    oneshot: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "10": {"pgd": 1.54653, "p_target": 0.99941, "gap_share": 0.00128, "size": 0.01410, "damage": 0.05769,
               "check": "at the line"},
        "100": {"pgd": 1.01556, "p_target": 0.99892, "gap_share": 0.00137, "size": 0.01682, "damage": 0.03625,
                "check": "pass"},
        "1000": {"pgd": 0.95909, "p_target": 0.99675, "gap_share": 0.00558, "size": 0.01842, "damage": 0.01801,
                 "check": "pass"}})
    c2_loss_max: float = 0.005  # C2: step-1 KL of the warm start against its own weights (E41 read 0.0005)
    c2_dw_rel_max: float = 1e-3  # C2: largest |delta-W after - before| / largest |delta-W before|, lr 0, one step
    works_gap: float = 0.05  # behaviour kept: gap share <= this, and P(target) >= 0.9 where the teacher's is
    # P1: a teacher whose gap at W0 is below this (nats) is too close to W0 to read a gap share; a link's teacher gap
    # must exceed it to count as behaviour lost. 10 x E41's C2 hook-vs-merged KL (0.000502, results/teacher_training_e41.json)
    gap_floor: float = 0.005
    not_worse: tuple[float, float, float] = (0.25, 0.01, 0.005)  # P2: E41 P3's ratio (relative), P, gap share
    pileup: float = 0.25  # P3: the last link's size and damage at most (1 + this) x the one-shot T1000 copy's
    near: float = 0.075  # reported: a ratio within this (relative) of a verdict line, W0's adversary-seed floor
    delta_max: float = 1e-6  # every link's Delta share (by construction ~1e-9; above this is a bug)

    def __post_init__(self) -> None:
        assert list(self.links) == sorted(set(self.links)) and self.links[-1] == CFG.steps[-1]
        assert len(self.link_seeds) == len(self.links) == len(set(self.link_seeds))
        assert set(self.g1_teacher) == set(self.oneshot) == set(ANCHORS) <= set(map(str, self.links))
        assert all(r["check"] in CHECK for r in self.oneshot.values())


AM = Amend2()


def check_of(rec: dict[str, Any], w0: dict[str, Any]) -> str:
    """The check's verdict on a state's two Pile ratios against this job's W0 floors, P(target) aside:
    faith_score.state_verdict's rule (any fail -> fail, any inconclusive -> at the line, else pass)."""
    from commute import faith_score

    per = [faith_score.verdict(rec["pgd"], w0["pgd_floor"]), faith_score.verdict(rec["stochastic"], w0["stochastic_floor"])]
    return LABEL["fail" if "fail" in per else "inconclusive" if "inconclusive" in per else "pass"]


def stop_unless(out: dict[str, Any], name: str, rec: dict[str, Any]) -> None:
    out[name] = rec
    log(f"[{name}] {json.dumps(rec, default=float)}")
    dump(OUT, out)
    if not rec["pass"]:
        halt(OUT, out, f"{name}: {json.dumps(rec, default=float)}")


def teachers(job: Job, out: dict[str, Any]) -> dict[str, tuple[dict[str, Any], Any]]:
    """The fine-tune at AM.links, {str(k): (delta-W, teacher net)}; each teacher's reading and gap at W0; G1."""
    from commute import teacher
    from commute.finetune import ArmAConfig, finetune_arm_a

    tctx = job.task(EMO)[0]
    cfg = ArmAConfig(o_token=tctx.token, lr=AM.ft_lr, steps=AM.links[-1], checkpoints=AM.links, seed=0, optimizer="adam")
    res = finetune_arm_a(tctx.model, tctx.train, tctx.held, cfg, tctx.device, log)
    T: dict[str, tuple[dict[str, Any], Any]] = {}
    for k in AM.links:
        ev, dW = res.evals[k], res.deltas[k]
        T[str(k)] = (dW, teacher.teacher_net(tctx.model, dW))
        t = out["teachers"].setdefault(str(k), {})
        t["reading"] = {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"]}
        t["gap_w0"] = gap_w0(tctx, T[str(k)][1])
        assert t["gap_w0"] > 0, f"T{k} equals W0 on the held-out windows"
        log(f"[emoticon T{k}] " + json.dumps(t["reading"] | {"gap_w0": t["gap_w0"]}, default=float))
    tp, ts = AM.g1_tol
    g1 = {k: {"p_target": (r := out["teachers"][k]["reading"])["p_target"], "size": r["size"], "e41": [p, s],
              "pass": abs(r["p_target"] - p) <= tp and abs(r["size"] - s) <= ts} for k, (p, s) in AM.g1_teacher.items()}
    stop_unless(out, "G1", {"teachers": g1, "pass": all(v["pass"] for v in g1.values())})
    return T


def nothing(job: Job, out: dict[str, Any], T: dict[str, Any], k: str) -> None:
    """T<k>'s do-nothing state (delta-W in Delta, subcomponents and Gamma untouched), read once and kept."""
    t = out["teachers"][k]
    if "nothing" not in t:
        t["nothing"] = read_state(job, job.task(EMO)[0], f"emoticon T{k} do-nothing", (T[k][0], None), T[k][1], True)
        t["nothing"]["check"] = check_of(t["nothing"], out["W0"])
        dump(OUT, out)


def train_link(tctx: Any, net: Any, init: Any, seed: int, steps: int, lr: float) -> tuple[list[dict[str, Any]], Any]:
    """Teacher training against `net` by E34's task-text recipe from `init` (None: W0's (U, V)) at batch-order `seed`,
    fresh Adam: the curve and the state {k: (delta-W, U_A', V_A')} at the last step."""
    from commute import component_training
    from commute.finetune import ArmAConfig

    a_cfg = ArmAConfig(o_token=tctx.token, batch=CFG.batch_a, steps=steps, checkpoints=(steps,), seed=seed)
    cfg = component_training.TrainConfig(lr=lr, steps=steps, save_steps=(steps,), optimizer="adam")
    curve, states = component_training.train(
        tctx.model, tctx.uv, tctx.train.tokens, tctx.held, a_cfg, cfg, tctx.device, tctx.W0, log=log,
        loss_fn=lambda n, toks: student_loss(n, net, toks, tctx.train.half), init=init)
    return curve, states[steps]


def c2(job: Job, out: dict[str, Any], st: dict[str, Any]) -> None:
    """C2: the warm start reproduces its start. From the first link's offsets (train's `init`), one step at lr 0 against
    that link's own weights: the step-1 KL is the hook path's floor and delta-W comes back unchanged."""
    from commute import teacher

    tctx = job.task(EMO)[0]
    d = {k: v[0] for k, v in st.items()}
    big = max(float(x.abs().max()) for x in d.values())
    assert big > 0, "the first link did not move"
    curve, st1 = train_link(tctx, teacher.teacher_net(tctx.model, d), init_of(tctx, st), 0, 1, 0.0)
    rel = max(float((st1[k][0] - d[k]).abs().max()) for k in d) / big
    loss = curve[0]["loss"]
    stop_unless(out, "C2", {"loss_step1": loss, "dw_rel": rel, "pass": loss <= AM.c2_loss_max and rel <= AM.c2_dw_rel_max})


def chain(job: Job, out: dict[str, Any], T: dict[str, Any]) -> None:
    """The links in order, each from the previous link's state; every link read, logged and scored as it finishes."""
    tctx = job.task(EMO)[0]
    for stale in ("C2", "score", "done"):  # a rerun redoes the chain; run.load keeps everything but "stopped"
        out.pop(stale, None)
    out["links"], st = {}, None
    for i, (k, seed) in enumerate(zip(map(str, AM.links), AM.link_seeds, strict=True)):
        nothing(job, out, T, k)
        t0 = time.time()
        curve, st = train_link(tctx, T[k][1], None if st is None else init_of(tctx, st), seed, AM.link_steps, CFG.student_lr)
        rec = read_state(job, tctx, f"emoticon link T{k}", ({key: v[0] for key, v in st.items()}, st), T[k][1], True)
        rec.update(check=check_of(rec, out["W0"]), gap_share=rec["teacher_gap"] / out["teachers"][k]["gap_w0"],
                   start_kl=curve[0]["loss"], seed=seed, student_steps=(i + 1) * AM.link_steps, seconds=time.time() - t0)
        out["links"][k] = rec
        out["score"] = score(out)
        dump(OUT, out)
        log(f"[link T{k}] check {rec['check']} pgd {rec['pgd']:.4f} stochastic {rec['stochastic']:.4f} P(target) "
            f"{rec['p_target']:.5f} teacher gap {rec['teacher_gap']:.6f} gap at W0 {out['teachers'][k]['gap_w0']:.6f} "
            f"gap share {rec['gap_share']:.5f} size {rec['size']:.5f} damage {rec['damage']:.5f} "
            f"start KL {rec['start_kl']:.5f} seed {seed} student steps {rec['student_steps']} ({rec['seconds']:.0f} s)")
        log(f"[score after T{k}] {json.dumps(out['score'], default=float)}")
        if rec["delta_share"] > AM.delta_max:
            halt(OUT, out, f"link T{k}: Delta share {rec['delta_share']:.3g} above {AM.delta_max:g} (Delta is frozen: a bug)")
        if i == 0:
            c2(job, out, st)


# ---- scoring (pure, on the results file) ----------------------------------------------------------------------------


def near_line(ratio: float, floor: float) -> str | None:
    """The nearer verdict line ('pass' or 'fail') if the adversarial ratio is within AM.near of it, relative; else None."""
    from commute.faith_score import LINE

    d = {"pass": abs(ratio * (1 + floor) / LINE - 1), "fail": abs(ratio / (LINE * (1 + floor)) - 1)}
    name = min(d, key=d.__getitem__)
    return name if d[name] <= AM.near else None


def not_worse(c: dict[str, Any] | None, o: dict[str, Any]) -> bool | None:
    """P2's rule for one anchor: the link `c` against the one-shot copy `o` of the same teacher; None if not read."""
    if c is None:
        return None
    ratio, p, gap = AM.not_worse
    return (CHECK.index(c["check"]) <= CHECK.index(o["check"]) and c["pgd"] <= (1 + ratio) * o["pgd"]
            and c["p_target"] >= o["p_target"] - p and c["gap_share"] <= o["gap_share"] + gap)


def score(out: dict[str, Any]) -> dict[str, Any]:
    """P1-P3 on whatever the file holds; a missing link or teacher reading makes its prediction unscored (None)."""
    links, teach = out.get("links", {}), out.get("teachers", {})

    def category(k: str) -> str | None:
        r, t, g = links.get(k), teach.get(k, {}).get("reading"), teach.get(k, {}).get("gap_w0")
        if r is None or t is None or g is None:
            return None
        assert r["check"] in CHECK, r["check"]
        if g < AM.gap_floor:
            return TOO_CLOSE
        lost = ((r["gap_share"] > AM.works_gap and r["teacher_gap"] > AM.gap_floor)
                or (t["p_target"] >= CFG.p_min and r["p_target"] < CFG.p_min))
        return "behaviour lost" if lost else r["check"]

    def ratio(a: float, b: float) -> float | None:
        return None if b == 0 else a / b

    cats = {k: category(k) for k in map(str, AM.links)}
    read_ = [c for c in cats.values() if c is not None and c != TOO_CLOSE]
    p1 = (None if None in cats.values() or not read_ else "behaviour lost" if "behaviour lost" in read_
          else max(read_, key=CHECK.index))
    p2 = {k: not_worse(links.get(k), AM.oneshot[k]) for k in ANCHORS}
    end, o = links.get(ANCHORS[-1]), AM.oneshot[ANCHORS[-1]]
    p3 = None if end is None else (end["size"] <= (1 + AM.pileup) * o["size"]
                                   and end["damage"] <= (1 + AM.pileup) * o["damage"])
    dn = {k: teach[k]["nothing"] for k in links if "nothing" in teach.get(k, {})}
    floor = out["W0"]["pgd_floor"]
    return {"C1": out.get("C1", {}).get("pass"), "G1": out.get("G1", {}).get("pass"), "PC": out.get("PC", {}).get("pass"),
            "C2": out.get("C2", {}).get("pass"),
            "P1_links": {"outcome": p1, "per_link": cats, "too_close": [k for k, c in cats.items() if c == TOO_CLOSE],
                         "lean": "at the line"},
            "P2_not_worse": {"pass": holds(p2), "per": p2},
            "P3_no_pileup": {"pass": p3, "size_over_oneshot": None if end is None else end["size"] / o["size"],
                             "damage_over_oneshot": None if end is None else end["damage"] / o["damage"]},
            "check_sees_the_update": {k: v["check"] != "pass" for k, v in dn.items()},
            "near_line": {k: near_line(r["pgd"], floor) for k, r in links.items()},
            "excess_removed": {k: None if v["pgd"] <= 1 else (v["pgd"] - links[k]["pgd"]) / (v["pgd"] - 1) for k, v in dn.items()},
            "vs_oneshot": {k: {m: links[k][m] / AM.oneshot[k][m] for m in ("pgd", "size", "damage")} for k in ANCHORS if k in links},
            "vs_teacher": {k: {m: ratio(r[m], teach[k]["reading"][m]) for m in ("size", "damage")}
                           for k, r in links.items() if "reading" in teach.get(k, {})},
            "delta_untouched": bool(links) and all(r["delta_share"] <= AM.delta_max for r in links.values())}


PREDICTIONS: dict[str, Any] = {
    "about": "teacher_training E41 Amendment 2, short links (reports/plans/2026-10-28_plan_e41-teacher-path.md -> "
             "'Amendment 2'; the approved branch for E41's P3 and P4 holding; decisions.md 2026-10-28 option (a): one "
             "batch order and one adversary start per link, a link between the lines is 'at the line'). Teacher: E41's "
             "emoticon fine-tune (E34's Adam 3e-5 arm A, all 24 matrices, batch 32, seed 0) regenerated in-job with "
             "checkpoints 1, 3, 10, 30, 100, 300, 1000. Chain: link 1 from W0's (U, V) against T1, each later link from "
             "the previous link's (U, V) against the next checkpoint; 100 student steps per link (700 in all), E34's "
             "task-text recipe (Adam 1e-5, batch 32, arm A's weighting, Gamma and Delta frozen), fresh Adam per link, "
             "link i at batch-order seed i. Every link read through E34's harness beside its teacher's do-nothing "
             "state; links T10, T100, T1000 against E41's one-shot copies. RunPod 'NVIDIA H100 80GB HBM3'. Filed "
             "before any run",
    "controls": {
        "C1": "first, alone: W0 under the Pile harness (256 Pile val rows, 20-step adversary, seeds 0, 1, 2): "
              "faith_score.w0_check passes, the PGD floor is 0.0748 within 10%, and the seed-0 PGD KL is E41's 0.69882 "
              "within 1% (E41 and Amendment 1 equal to 16 digits). Fails: stop, nothing trained",
        "G1": "after the fine-tune: T10, T100, T1000 held-out P(target) within 0.005 and size within 0.001 of E41's "
              "(0.99931 / 0.01125, 0.99946 / 0.02217, 0.99961 / 0.03209). Fails: stop",
        "PC": "before any link: T10's do-nothing state reads E41's 2.5159 within 1% (fail). The positive control (the "
              "check sees this early update when Delta carries it) and the proof that this job's harness reads as "
              "E41's, whose one-shot copies are P2's and P3's references. Fails: stop",
        "C2": "after the first link, before any warm start: from link T1's offsets via component_training.train's init, "
              "one step at lr 0 against that link's own weights; step-1 KL <= 0.005 nats (E41 0.0005) and delta-W "
              "unchanged to 1e-3 of its largest entry (E41 0). Fails: stop",
        "Delta": "every link's Delta share <= 1e-6 (by construction ~1e-9); above it the run stops (a bug)",
    },
    "measurements": {
        "per link": "run.record's numbers through the harness (adversarial and stochastic ratios to this job's W0 at "
                    "adversary seed 0, held-out P(target), teacher gap, damage, size, Delta's share); the check's "
                    "verdict on the two ratios against the same-job floors, P(target) aside (pass <= 1.396, fail >= "
                    "1.612 at floor 0.0748, 'at the line' between); gap share = teacher gap / T_k's gap at W0; start "
                    "KL = the loss on the link's first batch before its first update (its head start)",
        "do-nothing": "per teacher T_k: delta-W in Delta, subcomponents and Gamma untouched, read the same way; the "
                      "check 'sees the update' at T_k when this reads fail or at the line",
    },
    "expected": {
        "W0": "ratios 1, check pass; P(target) 0.004; gap share 1; size, damage 0",
        "do-nothing": "gap share 0; P, size, damage the teacher's; Delta share 1. T10 2.516 fail (PC), T100 1.136 pass, "
                      "T1000 1.499 at the line (E41, to the digit); T1, T3, T30, T300 not read before",
        "teachers": "T10 P 0.99931 size 0.01125, T100 0.99946 / 0.02217, T1000 0.99961 / 0.03209 (E41); T1 and T3 P "
                    "unknown (the fine-tune's step-1 batch read P 0.011 before its update, step 10 0.995), sizes rising "
                    "from about 0.002; gap at W0 about 5.1 nats from T10 on, never measured at T1 and T3",
        "references": "E41's one-shot copies: T10 1.547 at the line, P 0.9994, gap share 0.0013, size 0.0141, damage "
                      "0.0577; T100 1.016 pass, 0.9989, 0.0014, 0.0168, 0.0363; T1000 0.959 pass, 0.9968, 0.0056, "
                      "0.0184, 0.0180. Amendment 1: the T10 copy at batch orders 1 and 2 read 1.280 pass and 1.505 at "
                      "the line",
        "links": "P about 0.999 from T10 on, gap share under 0.01, Delta share ~1e-9; ratios at the line or pass at T1-T10 "
                 "(as the T10 one-shot copies), about 1.0-1.1 from T100 on; size about 0.015-0.022 and damage about "
                 "0.03-0.05 at T1000",
        "noise": "batch order (Amendment 1, E34 Amendment 4, same card): ratio up to 21% at T10 and 3.2% at T1000, P "
                 "0.005, gap share 0.002, size up to 5%, damage up to 3%; W0's adversary-seed floor 7.5%; this job "
                 "against E41's: 0 (equal to 16 digits)",
    },
    "predictions": [
        {"id": "P1_links", "kind": "three-way plus 'behaviour lost', filed with a lean",
         "lean": "at the line, medium (the T10 one-shot copies read 1 pass and 2 at the line over three batch orders; "
                 "the links at T1 and T3 copy updates never read; from T100 on one-shot copies pass by 25% or more)",
         "rule": "per link k in 1, 3, 10, 30, 100, 300, 1000: 'teacher too close to W0 to read' (left out of P1, its "
                 "check verdict still reported) if T_k's gap at W0 < 0.005 nats (10 x E41's C2 hook-vs-merged KL, "
                 "0.0005: below it a gap share is a ratio of two numbers near the hook path's floor); else 'behaviour "
                 "lost' if its gap share > 0.05 and its teacher gap > 0.005 nats, or if T_k's P(target) >= 0.9 and the "
                 "link's < 0.9; else the check's verdict on its two ratios (pass, at the line, fail). Outcome: "
                 "unscored if any link, teacher reading or gap at W0 is missing, or if every link is too close; "
                 "'behaviour lost' if any read link is; else the worst read link's verdict (pass < at the line < "
                 "fail): 'pass' only if every read link passes. Ties: gap at W0 exactly 0.005 is read; gap share "
                 "exactly 0.05, teacher gap exactly 0.005 and P exactly 0.9 keep the behaviour; ratios follow "
                 "faith_score's lines (exactly 1.5 / 1.0748 passes, exactly 1.5 x 1.0748 fails). Reported beside: per "
                 "link, whether the check sees its update (a pass where do-nothing also passes carries no "
                 "information), the line a ratio is within 7.5% of, and the share of do-nothing's excess removed, "
                 "(do-nothing - link) / (do-nothing - 1) where do-nothing > 1",
         "reading": "pass: teacher training applied every 100 steps keeps every early update faithful, so it can run "
                    "alongside training on this fine-tune. at the line: the chain is where the one-shot T10 copy is "
                    "(decisions.md option (a) accepts that). fail or behaviour lost: a qualitative surprise",
         "blind_to": "one fine-tune seed, one task, one batch order and one adversary start per link (a pass is a "
                     "lower bound); damage outside 256 Pile rows and the held-out task windows"},
        {"id": "P2_not_worse", "kind": "holds / fails",
         "lean": "holds, medium (E41's three-link chain held every comparison with ratios 1.04-1.05x; the risk is the "
                 "T10 link, 300 steps from W0 against the one-shot's 1,000)",
         "rule": "at T10, T100 and T1000, the link against E41's one-shot copy of the same teacher: its check verdict no "
                 "worse (pass < at the line < fail; the one-shot reads at the line, pass, pass), adversarial ratio <= "
                 "1.25 x the one-shot's (relative), P(target) >= the one-shot's - 0.01 and gap share <= the one-shot's "
                 "+ 0.005 (absolute; E41 P3's arm-A tolerances), all inclusive. Holds if all three anchors do; "
                 "unscored if one is missing",
         "reading": "fails at T100 or T1000: short links pile up error the one-shot copy does not have, on the check or "
                    "the behaviour. A miss at T10 alone sits inside batch-order noise (the one-shot copy reads 1.547 "
                    "at the line, spread 21% over three orders) and is not read as pile-up",
         "blind_to": "as P1; the chain has 300 / 500 / 700 student steps at the anchors against the one-shot's 1,000"},
        {"id": "P3_no_pileup", "kind": "two-sided, filed with a lean",
         "lean": "fails, medium, on damage (E41's 1,000-step chain read 1.36x the one-shot's size and 1.49x its damage "
                 "at T1000; step-10 copies do their teacher's full damage, 0.055, against the T1000 copy's 0.018, and "
                 "the last link has 100 steps to shed it)",
         "rule": "the T1000 link's size <= 1.25 x the one-shot T1000 copy's (0.01842) and its damage <= 1.25 x the "
                 "one-shot's (0.01801), inclusive. Unscored if the link is missing. Reported: both ratios, and size and "
                 "damage over the one-shot's at T10 and T100 and over each link's teacher",
         "reading": "holds: no pile-up of size or damage at the end. fails: the chain carries the early copies' size or "
                    "damage forward, unseen by the check and the behaviour; a reset or a longer last link would be "
                    "needed to match a one-shot copy",
         "blind_to": "size is the largest relative change of one matrix; damage is on held-out task windows only"},
    ],
    "reported_not_scored": [
        "every link's numbers, check verdict, gap share and start KL; every do-nothing state; the teachers' readings",
        "the score after every link (in the log and the JSON)",
        "Delta's share of every link (the score's delta_untouched)",
    ],
    "not_run": [
        "batch-order repeats of any link (decisions.md 2026-10-28, option (a)); adversary seeds 1 and 2",
        "Pile-text chains; the r1 task; SGD or LoRA teachers",
        "links shorter than 100 steps; carrying Adam's state across links; a one-shot copy trained 700 steps",
        "the cosine of the chain's end with the one-shot T1000 copy; per-block ratios; saving states",
    ],
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    dump(PRED, {**PREDICTIONS, "config": asdict(CFG), "e41_amend2": asdict(AM), "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"filed {PRED}")


def amend2(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (path_amend2.py --predictions)")
    filed = json.loads(PRED.read_text())
    assert filed["config"] == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed['config']}"
    assert filed["e41_amend2"] == json.loads(json.dumps(asdict(AM))), f"filed for another amendment: {filed['e41_amend2']}"
    job = Job(device)
    out = load(OUT, job, {"teachers": {}})
    floor, kl = job.w0["pgd_floor"], job.ctx.pile[0]["pgd_kl"]
    stop_unless(out, "C1", {"w0_check": job.w0["pass"], "pgd_floor": floor, "w0_pgd_kl": kl,
                            "pass": job.w0["pass"] and abs(floor / AM.c1_floor - 1) <= AM.c1_floor_tol
                            and abs(kl / AM.e41_w0_pgd - 1) <= AM.repro_tol})
    T = teachers(job, out)
    nothing(job, out, T, "10")
    r = out["teachers"]["10"]["nothing"]["pgd"]
    stop_unless(out, "PC", {"pgd": r, "check": out["teachers"]["10"]["nothing"]["check"], "e41": AM.e41_nothing10,
                            "pass": abs(r / AM.e41_nothing10 - 1) <= AM.repro_tol})
    chain(job, out, T)
    out["score"], out["done"] = score(out), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'], default=float)}")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--predictions", action="store_true", help="file the predictions and exit")
    args = ap.parse_args()
    file_predictions() if args.predictions else amend2(args.device)
