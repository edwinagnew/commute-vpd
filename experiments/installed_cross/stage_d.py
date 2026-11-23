"""E271 (cell P2.65) Stage D: a decomposition of each moved base, its faithfulness checked before anything reads it,
then on it the re-gating share and E_b over the needed set K of every state on that base, with the floors. Settings,
bars and expected values: results/predictions_stage_d.json (refused unless committed unchanged); the run's own
settings and states: run.py and its results. Plan reports/plans/2026-11-22_plan_p2-65-installed-cross.md, "Stage D".

The base's decomposition is E34's teacher training (`teacher.train_student`): the released decomposition's alive
(U, V) offsets (E34's alive set), Gamma and Delta frozen, trained so that the decomposed model's next-token
distribution matches the base's (KL(base || student)), on rows the base acts on: cued items of its own task, uncued
items of all three tasks and Pile windows, E34's arm-A weighting with the answer position as the trigger. Its state is
target W_s = M + the student's change, components U', V' (E119's reading of a student). The released M's row uses the
released decomposition (no student).

Jobs, each refusing unless what it reads is done:
  ctl            W0 (M, released decomposition) under the Pile harness on adversary seeds 0-2 (`faith_score.w0_check`:
                 its known values and its floor f); the same adversary on each task's 128 harness rows at M (the task
                 floor's reference). Exit 1 on a W0 miss
  teach BASE     the student; faithfulness, filed in predictions_stage_d.json, all before any reading: F1 the student
                 keeps >= 0.8 of the base's cued gain and its teacher gap on cued read rows is <= 0.1 of M's; F2 the
                 Pile harness ratio to W0 passes `faith_score.verdict` at the same-job floor; F3 the same adversary on
                 the base's task rows (cued and uncued), ratio to M's on the same rows, passes at those rows' own
                 floor (M's spread over adversary seeds there, ctl). Then,
                 only if every check passes, the re-gating share of each of the base's 9 states, three kinds each
                 (`run.variants`), a zero update reading exactly 0 first. A failed check is written and ends the
                 base's decomposition rows (exit 1): "no faithful featurization of this base" is the reading
  eb BASE [NAME_KIND ...]  on a faithful base (or M): per item, the zero update first (no changed variable, every dd
                 0, E_b 1), then K (`needed.scan`, E268's single-variable interchange) and E_b over K and over the
                 changed set; items default to the filed kinds of the base's 9 states
  score          CPU: re-gating share and E_b by run's score rules (triples, AUC, Gaussian floor) -> results/score_d.json

    PRED_SHA_D=<sha> sbatch --comment='E271 rehearse: stage D teach + one eb item' cluster/installed_cross_d.sbatch rehearse
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import component_training, faith_score, saved_states, teacher, vpd
from commute import installed as IN
from commute import kstar_vpd as KV
from commute import masked_reads as MR
from commute import needed as IX
from commute import rule_retention as RL
from commute import stand_in as SI
from commute.data import LAST, Windows, batches, cached_rows
from commute.faithfulness import (
    harness_pgd,
    model_state,
    pile_blocks,
    pile_faithfulness,
)
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

RUN: Any = load_experiment("installed_cross", "run")
SC: Any = load_experiment("installed_cross", "score")
RESULTS, REPO, TASKS, BASES = RUN.RESULTS, RUN.REPO, RUN.TASKS, RUN.BASES
PRED_D, OUT = RESULTS / "predictions_stage_d.json", RESULTS / "stage_d"
ALIVE = REPO / "experiments" / "transport_test" / "results" / "large"  # E34's alive.pt (harness.load)
log = RUN.log


@dataclass(frozen=True)
class ConfigD:  # filed in predictions_stage_d.json "settings"
    student_lr: float
    student_steps: int
    student_batch: int
    rows_each: int  # rows of each kind in the student's training set (cued, uncued over the three tasks, Pile)
    n_eval: int  # cued read rows the student's per-step held-out reading uses
    pile_seeds: tuple[int, ...]
    pgd_steps: int
    task_rows: int  # harness rows on the task: half cued, half uncued read rows
    derangements: int
    examples: int
    scan_seed: int
    rows: int
    logit_rows: int
    cache_batch: int
    split_tol: float
    eb_kinds: tuple[str, ...]
    eb_kinds_M: tuple[str, ...]

    def __post_init__(self) -> None:
        for f in fields(self):
            if isinstance(v := getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(v))
        assert self.pile_seeds[0] == 0 and len(self.pile_seeds) >= 2 and self.task_rows % 2 == 0 and self.task_rows <= 256
        assert set(self.eb_kinds) | set(self.eb_kinds_M) <= {"gauss", "gain", "size"} and "gauss" in self.eb_kinds


class Job:
    def __init__(self) -> None:
        filed(PRED_D, "PRED_SHA_D")
        filed(RUN.PRED, "PRED_SHA")  # run.py's file, checked again by RUN.Ctx under the same variable
        self.x = RUN.Ctx(gpu=True)
        P = json.loads(PRED_D.read_text())
        assert P["experiment"].startswith("E271"), P["experiment"]
        self.C, self.th, self.exp = config_from_settings(ConfigD, P["settings"]), P["thresholds"], P["experiment"]
        self.model, self.net = self.x.net.model, self.x.net
        self.alive = saved_states.load_alive(ALIVE)
        self.tasks = self.x.tasks()
        OUT.mkdir(parents=True, exist_ok=True)

    def read_rows(self, task: str) -> Tensor:
        return self.tasks[task].held[self.x.C.n_stop :]


def filed(path: Path, var: str) -> None:
    """`require_filed` on `path` against the sha256 in the environment variable `var` (two filed files, one run:
    PRED_SHA_D for this stage's, PRED_SHA for run.py's); with `var` unset, git decides, as require_filed does."""
    run_sha = os.environ.pop("PRED_SHA", None)
    if os.environ.get(var):
        os.environ["PRED_SHA"] = os.environ[var]
    try:
        require_filed(path)
    finally:
        os.environ.pop("PRED_SHA", None)
        if run_sha is not None:
            os.environ["PRED_SHA"] = run_sha


def done(name: str) -> dict[str, Any]:
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def out(j: Job, name: str) -> Any:
    o = RUN.out(j.x, name)  # run.py's CUDA-free Out
    o.path, o.d["experiment"] = OUT / f"{name}.json", j.exp
    if prev := done(name):  # resume
        o.d |= {k: v for k, v in prev.items() if k in ("share", "eb", "student")}
    return o


# ----------------------------------------------------------------------------- the harness on task rows


def task_blocks(j: Job, task: str) -> list[Tensor]:
    """The base task's harness rows: task_rows / 2 cued and as many uncued read rows (the prompts, 64 tokens), batches of 8."""
    h, rows = j.C.task_rows // 2, j.read_rows(task)[:, :-1]
    return batches(torch.cat([IN.cued(rows[:h], j.x.C.cue), rows[:h]]), 8)


def task_pgd(j: Job, task: str, seed: int) -> float:
    return harness_pgd(j.model, task_blocks(j, task), j.C.pgd_steps, seed)


def job_ctl(j: Job, o: Any) -> None:
    IN.set_base(j.net, j.x.M, None)
    blocks = pile_blocks()
    pile = {s: pile_faithfulness(j.model, blocks, seed=s, pgd_steps=j.C.pgd_steps, log=log) for s in j.C.pile_seeds}
    o.put("w0", w0 := faith_score.w0_check(pile))
    if not w0["pass"]:
        raise o.stop("W0 under the Pile harness misses its known values or its floor: nothing is read")
    o.put("task_w0", {t: {str(s): task_pgd(j, t, s) for s in j.C.pile_seeds} for t in TASKS})


# ----------------------------------------------------------------------------- job teach BASE


def student_rows(j: Job, base: str) -> Tensor:
    """rows_each cued items of the base's task, rows_each uncued items over the three tasks, rows_each Pile windows."""
    n, L = j.C.rows_each, j.x.C.L
    cued = IN.cued(j.tasks[base].train[:n, :-1], j.x.C.cue)
    unc = torch.cat([j.tasks[t].train[: n // 3 + 1, :-1] for t in TASKS])[:n]
    pile = cached_rows("train", 0, 4096).reshape(-1, L)[:n]  # Pile train rows cut into 64-token windows
    return torch.cat([cued, unc, pile])


def windows_of(seq: Tensor) -> Windows:
    """Sequences (N, L + 1) as E34's held-out Windows: the answer at index L - 1's next column (`data.LAST`)."""
    n = len(seq)
    z = torch.zeros(n, dtype=torch.long)
    return Windows(seq, torch.zeros(n), z, z, seq.shape[1] - 2)


def student_state(j: Job, base: str, rec: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]:
    """(W_s, components) of the saved student."""
    st = torch.load(REPO / rec["file"], weights_only=False)
    W = {k: w + st[k][0].float().to(w.device) for k, w in j.x.M.items()}
    return W, component_training.components_with(j.model, j.alive, {k: (st[k][1].float(), st[k][2].float()) for k in st})


def train_student(j: Job, base: str, net: Any, o: Any) -> dict[str, Any]:
    C, uv = j.C, component_training.alive_uv(j.model, j.alive)
    held = windows_of(IN.cued(j.read_rows(base)[: C.n_eval], j.x.C.cue))
    t0 = time.time()
    st, curve = teacher.train_student(j.model, uv, j.x.M, net, student_rows(j, base), held, LAST, C.student_batch, j.x.C.L - 1,
                                      C.student_lr, C.student_steps, "cuda", log)  # fmt: skip
    p = RUN.LARGE / "stage_d" / f"student_{base}.pt"  # ~60 MB (E119's offsets), gitignored
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: (v[0].bfloat16(), v[1].bfloat16(), v[2].bfloat16()) for k, v in st.items()}, p)
    return {"file": str(p.relative_to(REPO)), "sha256": RUN.sha(p), "seconds": time.time() - t0, "curve_end": curve[-1],
            "seconds_per_step": (time.time() - t0) / C.student_steps}  # fmt: skip


def faithful(j: Job, base: str, net: Any, W: dict[str, Tensor], comps: Any, o: Any) -> dict[str, Any]:
    """F1-F3, every one read before any share or E_b. W0's floor f is ctl's."""
    w0, tw0, th = done("ctl")["w0"], done("ctl")["task_w0"][base], j.th
    rows = j.read_rows(base)
    lo_m = IN.answer_lo(j.net.net, rows)[0]
    with model_state(j.model, target=W):
        lo_s = IN.answer_lo(j.net.net, IN.cued(rows, j.x.C.cue))[0]
        gap_s = teacher.heldout_gap(j.net.net, net, windows_of(IN.cued(rows, j.x.C.cue)), "cuda")
    gap_m = teacher.heldout_gap(j.net.net, net, windows_of(IN.cued(rows, j.x.C.cue)), "cuda")  # net at M: the full effect
    lo_b = IN.answer_lo(net, IN.cued(rows, j.x.C.cue))[0]
    f1 = {"gain_kept": float((lo_s - lo_m).mean()) / float((lo_b - lo_m).mean()), "gap_ratio": gap_s / gap_m, "gap": gap_s, "gap_M": gap_m}
    f1["pass"] = f1["gain_kept"] >= th["gain_kept_min"] and f1["gap_ratio"] <= th["gap_ratio_max"]
    with model_state(j.model, target=W, components=comps):
        pile = pile_faithfulness(j.model, pile_blocks(), seed=0, pgd_steps=j.C.pgd_steps, log=log)
        task = task_pgd(j, base, 0)
    ratios = {k: pile[k] / w0["pile"]["0"][k] for k in ("pgd_kl", "stochastic_kl")}  # ctl.json's seed-0 W0 (JSON keys)
    f2 = {"ratios": ratios, **faith_score.state_verdict(ratios, 1.0, w0)}
    floor3 = max(tw0.values()) / min(tw0.values()) - 1  # the task rows' own floor: M's adversary-seed spread there (ctl)
    f3 = {"pgd_kl": task, "w0": tw0["0"], "ratio": task / tw0["0"], "floor": floor3}
    f3["verdict"] = faith_score.verdict(f3["ratio"], floor3)
    rec = {"F1": f1, "F2": f2, "F3": f3}
    rec["pass"] = f1["pass"] and f2["verdict"] == "pass" and f3["verdict"] == "pass"
    return rec


def share_on(j: Job, W: dict[str, Tensor], comps: Any, task: str, dW: dict[str, Tensor]) -> dict[str, float]:
    """The re-gating share (L01 - L00) / (T1 - T0) in answer log-odds on the task's read rows, at target W with
    components comps (do-nothing form, Gamma's gates; `masked_reads.share_cells_by`), nan under MIN_GAIN."""
    r = RUN.cue_read(j.x, j.tasks[task])
    with model_state(j.model, target=W, components=comps):
        c = MR.share_cells_by(j.model, W, dW, r.toks, r.pos, r.units_of(r.tgt))
    num, gain = float((c["L01"]["lo"] - c["L00"]["lo"]).mean()), float((c["T1"]["lo"] - c["T0"]["lo"]).mean())
    return {"numerator": num, "gain": gain, "share": num / gain if gain >= MIN_GAIN else math.nan}


def decomposition(j: Job, base: str) -> tuple[dict[str, Tensor], Any]:
    """(target, components) of the base's decomposition: the faithful student, or for M the released one."""
    if base == "M":
        return dict(j.x.M), vpd.component_weights(j.model)
    t = done(f"teach_{base}")
    assert t.get("faithful", {}).get("pass"), f"teach_{base}.json: no faithful decomposition"
    return student_state(j, base, t["student"])


def shares(j: Job, base: str, o: Any) -> None:
    W, comps = decomposition(j, base)
    states = RUN.all_states()
    zero = share_on(j, W, comps, TASKS[0], {k: torch.zeros_like(w) for k, w in W.items()})
    o.put("zero_share", zero)
    if zero["numerator"] != 0 or zero["gain"] != 0:
        raise o.stop(f"a zero update's share numerator or gain is not exactly 0 on this decomposition: {zero}")
    for task in TASKS:
        ref = RUN.size_ref_of(states, task)
        for seed in j.x.C.seeds:
            name = f"{base}__{task}_s{seed}"
            for kind, d in RUN.variants(states, name, ref).items():
                if f"{name}_{kind}" not in o.d.get("share", {}):
                    o.put(f"{name}_{kind}", share_on(j, W, comps, task, d) | {"base": base, "task": task, "seed": seed, "kind": kind}, under="share")


def job_teach(j: Job, base: str, o: Any) -> None:
    if not (done("ctl").get("done") and RUN.done("controls").get("pass")):
        raise o.stop("stage_d/ctl.json is not done or results/controls.json has not passed")
    IN.set_base(j.net, j.x.M, None)
    bdw = j.x.base(base)
    net = teacher.teacher_net(j.model, bdw)
    if "student" not in o.d:
        o.put("student", train_student(j, base, net, o))
    W, comps = student_state(j, base, o.d["student"])
    o.put("faithful", f := faithful(j, base, net, W, comps, o))
    if not f["pass"]:
        raise o.stop(f"no faithful featurization of base {base}: F1 {f['F1']['pass']}, F2 {f['F2']['verdict']}, F3 {f['F3']['verdict']}; its decomposition rows stay empty")
    o.write()  # faithfulness on disk before any share is read
    shares(j, base, o)


# ----------------------------------------------------------------------------- job eb BASE


def models_at(j: Job, W: dict[str, Tensor], dW: dict[str, Tensor] | None) -> RL.Model:
    T = W if dW is None else {k: w + dW[k].to(w.device) for k, w in W.items()}
    with model_state(j.model, target=T):
        return RL.Model(copy.deepcopy(j.model.target_model).eval().requires_grad_(False))


def eb_item(j: Job, W: dict[str, Tensor], comps: Any, bases: Any, toks: Tensor, dW: dict[str, Tensor], scan: bool) -> dict[str, Any]:
    with model_state(j.model, target=W, components=comps):
        uv, delta = vpd.component_weights(j.model), vpd.weight_deltas(j.model)
    M, F = models_at(j, W, None), models_at(j, W, dW)
    feat = KV.featurize(M, F, uv, delta, bases)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(j.C.cache_batch)))
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    rec: dict[str, Any] = {"n_vars": feat.n, "n_changed": int(ch.sum()), "dd_sum": float(dd.sum()), "E_b_changed": SI.explained_share(num, den, dd, ch)}
    if scan:
        at = slice(-1, None)
        run = IX.cache(F, feat, toks, at, j.C.cache_batch)
        ex, pt = IX.derangements(len(toks), j.C.derangements, min(j.C.examples, len(toks)), j.C.scan_seed)
        rows = max(1, min(j.C.rows, j.C.logit_rows // run.top.shape[1]))
        sc = IX.scan(F, feat, run, at, ex, pt, rows, log)
        K = sc.first_pair >= 0
        rec |= {"size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()), "E_b_K": SI.explained_share(num, den, dd, K & ch),
                "cases_per_s": sc.rows / sc.seconds, "scan_seconds": sc.seconds, "max_dev": sc.max_dev}  # fmt: skip
    del M, F
    torch.cuda.empty_cache()
    return rec


def job_eb(j: Job, base: str, names: list[str], o: Any) -> None:
    W, comps = decomposition(j, base)
    if base == "M":  # the descriptive row's shares, on the released decomposition (moved bases' come from teach)
        shares(j, base, o)
    with model_state(j.model, target=W, components=comps):
        bases = KV.bases(vpd.component_weights(j.model), j.C.split_tol)
    states, kinds = RUN.all_states(), j.C.eb_kinds_M if base == "M" else j.C.eb_kinds
    want = names or [f"{base}__{t}_s{s}_{k}" for t in TASKS for s in j.x.C.seeds for k in kinds]
    toks0 = j.read_rows(TASKS[0])[:, :-1].cuda()
    if "zero" not in o.d.get("eb", {}):
        z = eb_item(j, W, comps, bases, toks0, {k: torch.zeros_like(w) for k, w in W.items()}, scan=False)
        o.put("zero", z, under="eb")
        if not (z["n_changed"] == 0 and z["dd_sum"] == 0 and z["E_b_changed"] == 1.0):
            raise o.stop(f"the zero update on this decomposition has changed variables or write change: {z}")
    for item in want:
        if item in o.d.get("eb", {}):
            continue
        name, kind = item.rsplit("_", 1)
        s = states[name]
        d = RUN.variants(states, name, RUN.size_ref_of(states, s["task"]))[kind]
        t0 = time.time()
        rec = eb_item(j, W, comps, bases, j.read_rows(s["task"])[:, :-1].cuda(), d, scan=True)
        o.put(item, rec | {"base": base, "task": s["task"], "seed": s["seed"], "kind": kind, "seconds": time.time() - t0,
                           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}, under="eb")  # fmt: skip


# ----------------------------------------------------------------------------- score (CPU)


def pick(recs: dict[str, Any], kind: str, field: str, direction: str) -> dict[tuple[str, str, int], float]:
    """One reading of one kind on the moved bases, signed so that higher = reuse (score.py's convention)."""
    return {(r["base"], r["task"], r["seed"]): SC.signed(float(r[field]), direction)
            for r in recs.values() if r.get("kind") == kind and field in r and r["base"] != "M"}  # fmt: skip


def job_score() -> None:
    P = json.loads(PRED_D.read_text())
    ev = json.loads(RUN.PRED.read_text())["thresholds"]["evidence"]
    res: dict[str, Any] = {"experiment": P["experiment"], "readings": {}}
    recs = {**{f"share/{k}": v for b in TASKS for k, v in done(f"teach_{b}").get("share", {}).items()},
            **{f"eb/{k}": v for b in BASES for k, v in done(f"eb_{b}").get("eb", {}).items() if k != "zero"}}  # fmt: skip
    for reading, (field, direction, floor_field) in P["thresholds"]["readings"].items():
        for kind in ("size", "gain"):  # the floor on its numerator where the reading is a ratio (run.py's amendment)
            real, gs = SC.score(pick(recs, kind, field, direction), ev), SC.score(pick(recs, "gauss", floor_field, direction), ev)
            res["readings"][f"{reading}_{kind}"] = {"real": real, "gauss": gs, "verdict": SC.verdict(real, gs, ev)}
            print(f"[{reading} {kind}] {real['triples']}/{real['of']} auc {real['auc']} | gauss {gs['auc']} -> {res['readings'][f'{reading}_{kind}']['verdict']}")
    (OUT / "score_d.json").write_text(json.dumps(jsonable(res), indent=1))


# ----------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    if argv == ["score"]:
        job_score()
        return 0
    ok = argv == ["ctl"] or argv == ["rehearse"] or (len(argv) == 2 and argv[0] == "teach" and argv[1] in TASKS) or (
        len(argv) >= 2 and argv[0] == "eb" and argv[1] in BASES)  # fmt: skip
    if not ok:
        sys.exit(f"usage: stage_d.py ctl | teach BASE | eb BASE [NAME_KIND ...] | rehearse | score; BASE in {BASES}")
    j = Job()
    log(f"{j.exp} stage D {argv} start; settings {asdict(j.C)}; thresholds {j.th}")
    jobs = [["ctl"], ["teach", "addition"], ["eb", "addition", "addition__addition_s0_size"]] if argv == ["rehearse"] else [argv]
    for a in jobs:
        name = "_".join(a[:2])
        o = out(j, name)
        try:
            if a[0] == "ctl":
                if not done("ctl").get("done"):
                    job_ctl(j, o)
            elif a[0] == "teach":
                job_teach(j, a[1], o)
            else:
                job_eb(j, a[1], a[2:], o)
        except RUN.R2.Stopped as e:
            print(f"STOP: {e}: tell the lead", flush=True)
            return 1
        o.d["done"] = not (a[0] == "eb" and len(a) > 2)  # a named subset of eb items is not the whole job
        o.write()
    log(f"{j.exp} stage D {argv} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
