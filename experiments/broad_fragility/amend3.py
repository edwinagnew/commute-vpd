"""broad_fragility Amendment 3 (E39 A3): does a rank-8 LoRA trained by plain SGD give a passing teacher-trained
decomposition on the lowercase task, as the Adam-trained LoRA does? First, two cheap readings: the SGD teacher-trained
decompositions on the second Pile rows (stop the job if both pass), and the labelled SGD state at step 1,000.
Plan: reports/plans/2026-10-27_plan_broad-fragility.md -> "Amendment 3 (2026-10-29)".

Everything is run.py's, imported: E37's lowercase task and rows, the fine-tunes, teacher training
(`row_copy.copy_on_rows`: Adam 1e-5, batch 8, 1,000 steps, batch order 0), the check (Pile harness, 20-step adversary
from one random start, adversary seed 0, as a ratio over W0 in the same job; pass <= 1.40, fail >= 1.61, both
inclusive), the reads (`run.read`) and the rate rule (`run.pick_rate`). Same card as E39 (H100 80GB HBM3). Every state
read here is regenerated in the job by its filed recipe (E39 A2 reproduced each to the 4th decimal on this card), so
no saved state is needed on the pod. Stages, each written to results/amend3.json as it finishes:

  stage 0   (i) W0 on adversary seeds 0-2 (`run.stage0_w0`); (ii) the Adam fine-tune and its teacher-trained
            decomposition (`run.stage0_trained`, check 2.2011), the failing reference. Any miss stops the job.
  rows2     W0 on Pile val rows 256-511 at adversary seeds 0-2 and the Adam teacher-trained decomposition read there
            (control: spread <= 0.10, check within 3% of A2's 1.8826 and failing; else stop); then the plain SGD
            fine-tune (A1's rate) and SGD held to rank 8 (A2's rate) regenerated (gains within 0.005 of A1/A2's, else
            stop), each teacher-trained and read on the second rows (P15). Both pass: stop (the plan's stop rule).
  label     component training with labels (next-token cross-entropy), Gamma and Delta frozen, plain SGD at A2's chosen
            rate 0.03, states at steps 100 and 1000; step 100's gain must reproduce A2's (G_label); step 1000 read (P16)
  lora_sgd  E39's Adam LoRA regenerated and teacher-trained (control: gain and check as E39/A2, passing; else stop);
            then the rank-8 LoRA trained by plain SGD on E27's rate grid, E39's rate rule; within 30% of the Adam
            fine-tune's gain, else nothing is scored and the grid's gains are reported; its do-nothing (the positive
            control) and its teacher-trained decomposition read (P17)

Results to results/amend3.json, log to results/large/amend3.log, states (bf16 offsets) to results/large/a3_*.pt
(run.stage0_trained also writes results/large/trained_step1000.pt, already home).

Usage (repo root, project venv):  amend3.py --predictions   files results/predictions_amend3.json (once)
                                  amend3.py [--device cuda]  the run (GPU); refuses without that file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import run
import torch
from run import (
    CFG,
    LARGE,
    RESULTS,
    TRANSPORT,
    UV,
    Job,
    dump,
    halt,
    log,
    pick_rate,
    read,
    sgd_ft,
    stage0_trained,
    stage0_w0,
)
from torch import Tensor

PRED, OUT = RESULTS / "predictions_amend3.json", RESULTS / "amend3.json"
run.OUT, run.LOG = OUT, LARGE / "amend3.log"  # run.read, dump, halt and log write here
State = dict[str, tuple[Tensor, Tensor, Tensor]]


@dataclass(frozen=True)
class Amend3:
    gain_tol: float = 0.30  # E39 A1's P8 and A2's gate: |gain / this job's Adam fine-tune gain - 1| <= this
    rows2: tuple[str, int, int] = ("val", 256, 256)  # A2's second rows
    a2_trained_rows2: float = 1.8826  # A2: the Adam teacher-trained decomposition on the second rows
    sgd_lr: float = 0.01  # A1's chosen plain SGD rate (grid 0.003-0.1), held-out gain 0.13671
    a1_sgd_gain: float = 0.13671
    cap_lr: float = 0.03  # A2's chosen rate for SGD held to rank 8 (grid 0.01-0.1), gain 0.13142
    a2_cap_gain: float = 0.13142
    cap_rank: int = 8
    label_lr: float = 0.03  # A2's chosen labelled SGD rate; its passing state was step 100
    label_steps: int = 1000
    label_saves: tuple[int, ...] = (100, 1000)
    a2_label_gain100: float = 0.12534  # A2's labelled SGD state at rate 0.03, step 100
    lora_adam_lr: float = 1e-4  # E39's chosen Adam LoRA rate
    e39_lora_gain: float = 0.1479  # E39 and A2: the Adam LoRA's gain and its teacher-trained check
    e39_lora_check: float = 1.2726
    lora_sgd_lrs: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0)  # E27's grid (0.003-3), its five middle rates, centred
    # on E27's pick 0.3 (rank-8 LoRA on all 24 matrices, plain SGD, emoticon)

    def __post_init__(self) -> None:
        assert self.rows2[1] >= CFG.rows_held[1] + CFG.rows_held[2] and self.rows2[2] == CFG.pile_rows  # no overlap
        assert self.label_saves == tuple(sorted(self.label_saves)) and self.label_saves[-1] == self.label_steps
        assert self.label_steps == CFG.copy_steps and self.label_saves[0] == 100  # G_label reads A2's step 100
        assert self.lora_adam_lr in CFG.lora_lrs and self.cap_rank == CFG.lora_rank
        assert self.lora_sgd_lrs == tuple(sorted(self.lora_sgd_lrs)) and self.lora_sgd_lrs[2] == 0.3


AM = Amend3()
SGD_TT = ("tt_sgd", "tt_sgd_r8")  # the two SGD teacher-trained decompositions read on the second rows


def config() -> dict[str, Any]:
    return json.loads(json.dumps({"amend3": asdict(AM), "run": asdict(CFG)}))


def split_state(st: State) -> tuple[dict[str, Tensor], UV]:
    return {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}


def save(job: Job, name: str, st: State, meta: dict[str, Any]) -> None:
    from commute.arms import save_offsets

    path = LARGE / f"a3_{name}.pt"
    save_offsets(path, job.ctx.uv_cpu, st, meta)
    log(f"[saved] {path.name}")


def teach(job: Job, teacher: str) -> tuple[State, dict[str, float]]:
    """E37's teacher training on the fine-tune `teacher` at batch order CFG.copy_seed; the state and its KL curve ends."""
    from commute.row_copy import copy_on_rows

    t0 = time.time()
    curve, st = copy_on_rows(job.ctx, job.refs[teacher].teacher, job.lower, CFG.copy_lr, CFG.copy_batch, CFG.copy_steps,
                             CFG.copy_seed, log)
    kl = {"train_kl_step1": curve[0]["loss"], "train_kl_end": sum(c["loss"] for c in curve[-50:]) / 50}
    log(f"[tt_{teacher}] trained in {time.time() - t0:.0f} s; {json.dumps(kl)}")
    return st, kl


def regenerate(job: Job, out: dict[str, Any], name: str, dW: dict[str, Tensor], ref: float) -> None:
    """Register the regenerated fine-tune `name` as a teacher; its gain must be within CFG.gain_tol of `ref`, else stop."""
    g = job.add_teacher(name, dW)
    out["shape"][name] = {"gain": g, "ref_gain": ref}
    log(f"[{name}] regenerated: held-out gain {g:.4f} (filed {ref})")
    if not abs(g - ref) <= CFG.gain_tol:
        halt(out, f"the regenerated {name} fine-tune's gain {g:.4f} is not within {CFG.gain_tol} of {ref}")


@contextmanager
def harness_on(job: Job, ctx: Any) -> Iterator[None]:
    """run.read reads the state through job.ctx; this points it at another harness context for the block."""
    ctx1, job.ctx = job.ctx, ctx
    try:
        yield
    finally:
        job.ctx = ctx1


# ----------------------------------------------------------------------------- stage rows2


def capped_ft(job: Job) -> dict[str, Tensor]:
    from commute.finetune import ArmBConfig
    from commute.rank_cap import finetune_arm_b_rank_capped

    cfg = ArmBConfig(lr=AM.cap_lr, batch=CFG.ft_batch, steps=CFG.ft_steps, checkpoints=(CFG.ft_steps,), seed=0,
                     optimizer="sgd")
    return finetune_arm_b_rank_capped(job.ctx.model, job.lower, job.held, cfg, AM.cap_rank, job.ctx.device,
                                      log).deltas[CFG.ft_steps]


def rows2(job: Job, out: dict[str, Any], tr: tuple[dict[str, Tensor], UV]) -> None:
    """The control on the second rows first (W0's spread, the Adam teacher-trained decomposition failing as in A2), then
    the two SGD fine-tunes regenerated, teacher-trained and read there; the stop rule last."""
    from commute import harness

    ctx2 = harness.on_rows(job.ctx, *AM.rows2, CFG.pile_seeds, log)
    rec = out["rows2"] = {"rows": list(AM.rows2), "w0_pile": ctx2.pile, "spread": harness.pgd_floor(ctx2)}
    log(f"[rows2] W0 spread over adversary seeds 0-2: {rec['spread']:.4f} (A2 0.0621)")
    with harness_on(job, ctx2):
        c = read(job, out, "trained_rows2", "adam", *tr)["check"]
    rec["G_rows2"] = g = (rec["spread"] <= CFG.spread_max and c >= CFG.fail_line
                          and abs(c / AM.a2_trained_rows2 - 1) <= CFG.check_tol)
    log(f"[rows2 control] spread {rec['spread']:.4f} trained check {c:.4f} (A2 {AM.a2_trained_rows2}) pass {g}")
    if not g:
        halt(out, "rows2 control: W0's spread over 0.10, or the Adam teacher-trained decomposition not as in A2")
    regenerate(job, out, "sgd", sgd_ft(job, AM.sgd_lr).deltas[CFG.ft_steps], AM.a1_sgd_gain)
    regenerate(job, out, "sgd_r8", capped_ft(job), AM.a2_cap_gain)
    for name in ("sgd", "sgd_r8"):
        st, kl = teach(job, name)
        with harness_on(job, ctx2):
            read(job, out, f"tt_{name}_rows2", name, *split_state(st), kl)
    sc = out["score"] = score(out)
    if sc.get("rows2_both_pass"):
        halt(out, "rows2: both SGD teacher-trained decompositions pass on the second rows (the plan's stop rule)")


# ----------------------------------------------------------------------------- stage label


def label(job: Job, out: dict[str, Any]) -> None:
    """Component training with labels, plain SGD at AM.label_lr, as A2's `label` ran it at that rate (E37's batch and
    batch order, held-out gain, gap and damage at each saved step); step 100 reproduces A2's state (G_label) and step
    1000 is saved and read."""
    from commute import component_training
    from commute.finetune import ArmAConfig, next_token_ce
    from commute.row_copy import read_state

    ctx = job.ctx
    a_cfg = ArmAConfig(o_token=ctx.token, batch=CFG.copy_batch, steps=AM.label_steps, checkpoints=(AM.label_steps,),
                       seed=CFG.copy_seed)
    cfg = component_training.TrainConfig(lr=AM.label_lr, steps=AM.label_steps, save_steps=AM.label_saves, optimizer="sgd")
    t0 = time.time()
    curve, states = component_training.train(ctx.model, ctx.uv, job.lower, ctx.held, a_cfg, cfg, ctx.device, ctx.W0,
                                             log=log, loss_fn=next_token_ce)
    log(f"[label_sgd lr {AM.label_lr:g}] {AM.label_steps} steps in {time.time() - t0:.0f} s")
    rec: dict[str, Any] = out.setdefault("label", {})
    for step in AM.label_saves:
        q = read_state(ctx, job.refs["adam"], split_state(states[step])[0], None, False, log)
        g = q["logp"] - job.logp_w0
        div = not (all(math.isfinite(c["loss"]) for c in curve[:step]) and math.isfinite(g) and g > 0)
        rec[str(step)] = {"gain": g, "gap": q["gap"], "damage": q["damage"], "diverged": div}
        log(f"[label_sgd step {step}] gain {g:.4f} gap {q['gap']:.4f} damage {q['damage']:.4f} diverged {div}")
    g100 = rec["100"]["gain"]
    rec["G_label"] = math.isfinite(g100) and abs(g100 - AM.a2_label_gain100) <= CFG.gain_tol
    log(f"[label_sgd G_label] step 100 gain {g100:.4f} (A2 {AM.a2_label_gain100}) pass {rec['G_label']}")
    dump(out)
    last = AM.label_steps
    save(job, f"label_sgd_{last}", states[last], {"loss": "labels", "optimizer": "sgd", "lr": AM.label_lr, "step": last})
    read(job, out, f"label_sgd_{last}", "adam", *split_state(states[last]), {"lr": AM.label_lr, "step": last})


# ----------------------------------------------------------------------------- stage lora_sgd


def lora_sgd_ft(job: Job, lr: float) -> Any:
    """E39's rank-8 LoRA on all 24 matrices (run.lora_ft's recipe: B = 0, A ~ N(0, 1/d_in), LoRA seed 0, 300 steps of 8
    rows each once, clip 1.0) with plain SGD (momentum 0, no weight decay; E27's optimiser) at `lr`."""
    from commute.finetune import LoraConfig, finetune_lora_arm_b
    from commute.vpd import module_keys

    cfg = LoraConfig(modules=tuple(module_keys(job.ctx.model)), rank=CFG.lora_rank, lr=lr, steps=CFG.ft_steps,
                     checkpoints=(CFG.ft_steps,), seed=CFG.lora_seed, optimizer="sgd", momentum=0.0)
    return finetune_lora_arm_b(job.ctx.model, job.lower, job.held, cfg, CFG.ft_batch, job.ctx.device, log=log)


run.FINETUNE["lora_sgd"] = (lora_sgd_ft, AM.lora_sgd_lrs)  # run.pick_rate's grid and rate rule for this arm


def lora_reference(job: Job, out: dict[str, Any]) -> None:
    """The stage's control: E39's Adam LoRA (`run.lora_ft` at E39's rate) regenerated, teacher-trained and read; gain
    and check as E39 and A2 read them, and passing. Else stop before the SGD LoRA is trained."""
    g = job.add_teacher("lora_adam", run.lora_ft(job, AM.lora_adam_lr).deltas[CFG.ft_steps])
    st, kl = teach(job, "lora_adam")
    c = read(job, out, "tt_lora_adam", "lora_adam", *split_state(st), kl)["check"]
    ok = (abs(g - AM.e39_lora_gain) <= CFG.gain_tol and abs(c / AM.e39_lora_check - 1) <= CFG.check_tol
          and c <= CFG.pass_line)
    out["lora_ref"] = {"gain": g, "check": c, "pass": ok}
    log(f"[lora_ref] gain {g:.4f} (E39 {AM.e39_lora_gain}) check {c:.4f} (A2 {AM.e39_lora_check}) pass {ok}")
    if not ok:
        halt(out, "lora_sgd control: the regenerated Adam LoRA or its teacher-trained decomposition is not as in E39/A2")


def lora_sgd(job: Job, out: dict[str, Any]) -> None:
    """The control, then the SGD LoRA grid and rate rule; if the chosen rate's gain is within AM.gain_tol of the Adam
    fine-tune's, its do-nothing and its teacher-trained decomposition are read and the state saved."""
    lora_reference(job, out)
    dW = pick_rate(job, out, "lora_sgd")
    rec = out["shape"]["lora_sgd"]
    gains = [v["gain"] for v in rec["grid"].values() if math.isfinite(v["gain"])]
    rec["gain_range"] = [min(gains), max(gains)] if gains else None
    log(f"[lora_sgd] grid gains {json.dumps({r: round(v['gain'], 4) for r, v in rec['grid'].items()})} range "
        f"{rec['gain_range']} (Adam fine-tune {out['stage0']['gain']:.4f})")
    if dW is None or not gain_within(rec["grid"][f"{rec['chosen']:g}"]["gain"], out["stage0"]["gain"]):
        log(f"[lora_sgd] no rate within {AM.gain_tol:.0%} of the Adam fine-tune's gain: nothing is scored")
        dump(out)
        return
    rec["gain"] = job.add_teacher("lora_sgd", dW)
    read(job, out, "nothing_lora_sgd", "lora_sgd", dW, None)
    st, kl = teach(job, "lora_sgd")
    save(job, "tt_lora_sgd", st, {"teacher": "lora_sgd", "lr": rec["chosen"], "step": CFG.copy_steps})
    read(job, out, "tt_lora_sgd", "lora_sgd", *split_state(st), kl)


# ----------------------------------------------------------------------------- scoring (pure)


def gain_within(g: float, adam: float | None) -> bool:
    """|g / adam - 1| <= AM.gain_tol, False for a missing or zero reference or a non-finite gain."""
    return adam is not None and adam != 0 and math.isfinite(g) and abs(g / adam - 1) <= AM.gain_tol


def verdict3(check: float) -> str:
    return run.verdict(check) if math.isfinite(check) else "invalid"


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules P15-P17 on what has been read so far; a prediction whose reads are missing is absent, one whose
    gate failed is 'unscored: <why>'. `tree` maps the outcomes to the plan's readings."""
    r, sc = out.get("reads", {}), {}
    adam = out.get("stage0", {}).get("gain")
    rows = out.get("rows2", {})
    if "G_rows2" in rows and all(f"{n}_rows2" in r for n in SGD_TT):
        checks = [r[f"{n}_rows2"]["check"] for n in SGD_TT]
        sc["P15_checks"], sc["P15_verdicts"] = checks, [verdict3(c) for c in checks]
        finite = all(math.isfinite(c) for c in checks)
        sc["rows2_both_pass"] = bool(rows["G_rows2"]) and finite and all(c <= CFG.pass_line for c in checks)
        if not rows["G_rows2"]:
            sc["P15"] = "unscored: the second rows' control failed"
        elif not finite:
            sc["P15"] = "unscored: a check is not finite"
        else:
            sc["P15"] = "holds" if all(c >= CFG.fail_line for c in checks) else "misses"
    lab, name = out.get("label", {}), f"label_sgd_{AM.label_steps}"
    if name in r:
        st = lab[str(AM.label_steps)]
        sc["P16_gain_ratio"] = st["gain"] / adam if adam else float("nan")
        sc["P16_verdict"] = verdict3(r[name]["check"])
        if not lab["G_label"]:
            sc["P16"] = "unscored: the step-100 state did not reproduce A2's gain (G_label)"
        elif st["diverged"]:
            sc["P16"] = "unscored: the labelled SGD run diverged by step 1000"
        elif not gain_within(st["gain"], adam):
            sc["P16"] = f"unscored: gain not within {AM.gain_tol:.0%} of the Adam fine-tune's"
        else:
            sc["P16"] = outcome(r[name]["check"], "fail")
    sh = out.get("shape", {}).get("lora_sgd", {})
    if "chosen" in sh:
        sc["P17_gain_range"] = sh.get("gain_range")
        if sh["chosen"] is None:
            sc["P17"] = "unscored: every rate of the grid diverged"
        elif not gain_within(sh["grid"][f"{sh['chosen']:g}"]["gain"], adam):
            sc["P17"] = f"unscored: no rate within {AM.gain_tol:.0%} of the Adam fine-tune's gain (range in P17_gain_range)"
        elif {"nothing_lora_sgd", "tt_lora_sgd"} <= set(r):
            nothing, c = r["nothing_lora_sgd"]["check"], r["tt_lora_sgd"]["check"]
            sc["P17_gain_ratio"] = sh["grid"][f"{sh['chosen']:g}"]["gain"] / adam
            sc["G_nothing"] = math.isfinite(nothing) and nothing >= CFG.fail_line
            sc["P17_verdict"] = verdict3(c)
            sc["P17"] = (outcome(c, "pass") if sc["G_nothing"] else
                         "unscored: the SGD LoRA's do-nothing does not fail, so the check has nothing to fail")
    sc["tree"] = tree(sc)
    return sc


def outcome(check: float, rule: str) -> str:
    """'holds' or 'misses' for a filed rule: 'fail' holds iff check >= the fail line, 'pass' iff check <= the pass
    line (both inclusive); a non-finite check is unscored."""
    if not math.isfinite(check):
        return "unscored: the check is not finite"
    return "holds" if (check >= CFG.fail_line if rule == "fail" else check <= CFG.pass_line) else "misses"


def answer(sc: dict[str, Any], pid: str) -> str:
    """One prediction's outcome: 'not run', 'unscored', or the check's verdict (pass, between, fail)."""
    if pid not in sc:
        return "not run"
    if pid == "P15":
        if sc[pid].startswith("unscored"):
            return "unscored"
        v = sc["P15_verdicts"]
        return "fail" if all(x == "fail" for x in v) else "pass" if all(x == "pass" for x in v) else "mixed"
    return "unscored" if sc[pid].startswith("unscored") else sc[f"{pid}_verdict"]


def tree(sc: dict[str, Any]) -> dict[str, str]:
    return {"rows2": READINGS["rows2"][answer(sc, "P15")], "label": READINGS["label"][answer(sc, "P16")],
            "lora_sgd": READINGS["lora_sgd"][answer(sc, "P17")]}


run.score = score  # run.read scores with this module's rules


# ----------------------------------------------------------------------------- main


STAGES = ("rows2", "label", "lora_sgd")


def main(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (amend3.py --predictions)")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == config(), f"predictions were filed for another config: {filed}"
    for f in (TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt"):
        if not f.is_file():
            sys.exit(f"missing input (gitignored): {f}")
    job = Job(device)
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": job.card, "reads": {}, "shape": {}, "done": []}
    if out["card"] != job.card:
        sys.exit(f"{OUT} was written on {out['card']}; this job runs on {job.card}. Move the file aside to start again")
    out["status"] = "running"
    stage0_w0(job, out)
    tr = stage0_trained(job, out)
    fns = {"rows2": lambda: rows2(job, out, tr), "label": lambda: label(job, out), "lora_sgd": lambda: lora_sgd(job, out)}
    for name in STAGES:
        if name in out["done"]:
            continue
        t0 = time.time()
        fns[name]()
        out["done"].append(name)
        out["score"] = score(out)
        dump(out)
        log(f"[score after {name}, {time.time() - t0:.0f} s] "
            + json.dumps({k: v for k, v in out["score"].items() if k != "tree"}, default=float))
        torch.cuda.empty_cache()
    out["status"] = "done"
    dump(out)
    log("[tree] " + json.dumps(out["score"]["tree"], indent=1))
    log(f"FINISHED wrote {OUT}")


# ----------------------------------------------------------------------------- predictions


READINGS: dict[str, dict[str, str]] = {
    "rows2": {
        "fail": "Both SGD teacher-trained decompositions fail on the second rows too: their fail is not one block of 128 "
                "rows, and the contrast with the LoRA's teacher-trained decompositions (which pass there) stands.",
        "mixed": "On the second rows one SGD teacher-trained decomposition fails and the other does not, or one lands "
                 "between the lines: the SGD fail holds on new rows only in part, and the contrast with the LoRA is "
                 "weaker than on the first rows.",
        "pass": "Both SGD teacher-trained decompositions pass on the second rows: their fail rests on one block of 128 "
                "rows, little is left for the SGD LoRA to separate, and the job stops before training it (the plan's "
                "stop rule).",
        "unscored": "The second rows' control failed, or a check was not finite: the job stopped.",
        "not run": "Stage rows2 not run.",
    },
    "label": {
        "fail": "Component training with labels and plain SGD fails the check at step 1,000 where it passed at step 100: "
                "E21/E26's narrow-task pattern (SGD passes early, fails late) holds on the broad task, and A2's labelled "
                "pass is only an early state.",
        "pass": "Component training with labels and plain SGD still passes at step 1,000: on this task SGD's labelled "
                "pass is not only an early state, unlike E21/E26 on narrow tasks.",
        "between": "The labelled SGD state at step 1,000 lands between the lines: not separated.",
        "unscored": "The step-100 state did not reproduce A2's, the run diverged, or its gain is over 30% from the Adam "
                    "fine-tune's: unanswered.",
        "not run": "Stage label not run.",
    },
    "lora_sgd": {
        "pass": "A rank-8 LoRA trained by plain SGD also gives a passing teacher-trained decomposition, while its "
                "do-nothing fails: on this task the LoRA form, not the Adam optimiser, goes with the pass (one seed, one "
                "rate).",
        "fail": "The LoRA form passes only with Adam: the teacher-trained decomposition of a rank-8 LoRA trained by plain "
                "SGD fails, as the full SGD fine-tunes' do.",
        "between": "The SGD LoRA's teacher-trained decomposition lands between the lines: not separated.",
        "unscored": "No rate of E27's grid reached a gain within 30% of the Adam fine-tune's (the grid's range of gains "
                    "is reported), or the SGD LoRA's do-nothing does not fail (nothing for the check to fail): "
                    "unanswered.",
        "not run": "Stage lora_sgd not run (an earlier stop).",
    },
}

NOISE = ("W0's spread over adversary seeds 0-2: 0.0748 on E39's H100 on the first rows, 0.0621 on the second rows (A2); "
         "the pass and fail lines 1.40 and 1.61 bracket 1.5 by that floor; E37's batch-order spread of the check, 14% "
         "(2.20 and 1.88); regenerations on this card reproduced held-out gains to 0.0001 and checks to the 4th decimal "
         "(A2: 2.2011, 1.2726)")


def pred(pid: str, rule: str, expected: str, confidence: str, blind: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "expected": expected, "confidence": confidence, "blind_to": blind,
            "positive_control": control, "noise_floor": NOISE}


GATE = f"scored only if the state's held-out gain is within {AM.gain_tol:.0%} of this job's Adam fine-tune gain (relative)"


def predictions() -> dict[str, Any]:
    a2 = json.loads((RESULTS / "predictions_amend2.json").read_text())
    return {
        "plan": "E39 Amendment 3, reports/plans/2026-10-27_plan_broad-fragility.md -> 'Amendment 3 (2026-10-29)'. "
                "Filed before any run; never edited after. P1-P14 are E39's, A1's and A2's; this file holds P15-P17.",
        "definitions": a2["definitions"] | {
            "second rows": f"Pile {AM.rows2[0]} rows {AM.rows2[1]}-{AM.rows2[1] + AM.rows2[2] - 1} (A2's), the check's "
                           "harness (two blocks of 128, one adversary per block, seed 0), W0 read on them as the reference",
            "SGD teacher-trained decompositions": f"teacher training (E37's recipe) on A1's plain SGD fine-tune (rate "
                                                  f"{AM.sgd_lr}) and on A2's SGD held to rank 8 (rate {AM.cap_lr}), both "
                                                  "regenerated in the job (A1/A2 read them 2.067 and 2.122 on the first rows)",
            "labelled SGD state at step 1,000": f"A2's component training with labels, plain SGD at {AM.label_lr} (the rate "
                                                "of A2's passing state), batch order 0, its state at step 1000",
            "SGD LoRA": f"E39's rank-8 LoRA on all 24 matrices (B = 0, A ~ N(0, 1/d_in), LoRA seed {CFG.lora_seed}, "
                        f"{CFG.ft_steps} steps of {CFG.ft_batch} lowercase rows each once, clip 1.0) trained by plain SGD "
                        f"(momentum 0, no weight decay) at the rate of {list(AM.lora_sgd_lrs)} (E27's grid, its five "
                        "middle rates) chosen by E39's rule: among rates not diverged (a non-finite training loss or a "
                        "held-out gain not above 0), the gain nearest this job's Adam fine-tune's",
        },
        "controls": {
            "stage0_i": a2["controls"]["stage0_i"],
            "stage0_ii": a2["controls"]["stage0_ii"],
            "G_rows2": f"first on the second rows: W0's spread over adversary seeds 0-2 <= {CFG.spread_max} and the Adam "
                       f"teacher-trained decomposition's check >= {CFG.fail_line} and within {CFG.check_tol:.0%} of A2's "
                       f"{AM.a2_trained_rows2}; else stop",
            "regenerated SGD fine-tunes": f"held-out gains within {CFG.gain_tol} of {AM.a1_sgd_gain} (plain) and "
                                          f"{AM.a2_cap_gain} (rank 8); else stop",
            "G_label": f"the labelled SGD state at step 100 has gain within {CFG.gain_tol} of A2's {AM.a2_label_gain100}; "
                       "else P16 is unscored",
            "lora_ref": f"before the SGD LoRA, E39's Adam LoRA regenerated: gain within {CFG.gain_tol} of "
                        f"{AM.e39_lora_gain}, its teacher-trained check within {CFG.check_tol:.0%} of {AM.e39_lora_check} "
                        f"and <= {CFG.pass_line}; else stop",
            "G_nothing": f"the SGD LoRA's do-nothing check >= {CFG.fail_line} (the positive control: the check fails the "
                         "update left in Delta, as the Adam LoRA's 2.653 did); else P17 is unscored",
        },
        "stop_rule": f"after the second rows are read: if both SGD teacher-trained checks are <= {CFG.pass_line} (and "
                     "G_rows2 held), the job stops before the label and lora_sgd stages",
        "expected": {
            "W0": "check 1 by definition, s 0, damage 0, gap 0.272",
            "do-nothing (Adam)": "check 4.868 (E37), not read here",
            "trained (Adam, teacher training)": "check 2.201 on the first rows, 1.883 on the second (A2)",
            "second rows, W0 spread": "0.062 (A2)",
            "second rows, SGD teacher-trained": "plain 1.4-2.2, rank 8 1.4-2.2 (first rows 2.067, 2.122; the first "
                                                "rows' block 2 carried both fails); P15",
            "regenerated gains": f"plain SGD {AM.a1_sgd_gain}, rank 8 {AM.a2_cap_gain}, labelled step 100 "
                                 f"{AM.a2_label_gain100}, Adam LoRA {AM.e39_lora_gain}",
            "labelled SGD at step 1,000": "gain 0.154 (A2, ratio 1.18), damage 0.229, gap 0.105; check 1.6-2.6 (P16)",
            "Adam LoRA teacher-trained (lora_ref)": f"check {AM.e39_lora_check}",
            "SGD LoRA grid": "gains rising with rate up to about 0.3-1, then falling or diverging at 3; the chosen rate "
                             "0.3 or 1, gain 0.10-0.15 (E27: rate 0.3 learnt emoticon by step 10)",
            "SGD LoRA do-nothing": "check 2-4 (Adam LoRA 2.653, plain SGD 2.623, Adam 4.868); G_nothing",
            "SGD LoRA teacher-trained": "check 1.15-1.5, s 0.95-0.98, damage 0.10-0.15 (P17)",
        },
        "predictions": [
            pred("P15", f"on the second rows, both SGD teacher-trained decompositions (plain, rank 8): check >= "
                        f"{CFG.fail_line}, each; holds iff both do, misses otherwise; scored only if G_rows2 holds",
                 "both fail, 1.6-2.2", "low-medium",
                 "one more set of 256 rows, one adversary start; on the first rows both failed on one block of 128 and "
                 "passed the other", "the Adam teacher-trained decomposition on the same rows (G_rows2, must fail)"),
            pred("P16", f"the labelled SGD state at rate {AM.label_lr}, step {AM.label_steps}: check >= {CFG.fail_line}. "
                        f"{GATE}; and only if G_label holds and the run did not diverge",
                 "fails, 1.6-2.6", "medium-low",
                 "one rate, one batch order; Gamma at W0 throughout; rows drawn with replacement",
                 "the Adam teacher-trained decomposition (2.201, stage 0); A2's step-100 state from the same run (1.275)"),
            pred("P17", f"teacher training on the SGD LoRA: check <= {CFG.pass_line}. {GATE}; and only if G_nothing "
                        "holds; no rate qualifying leaves it unscored",
                 "passes, 1.15-1.5", "medium-low",
                 "one LoRA seed, one rate, one batch order; the gain differs from the Adam LoRA's (0.148); a fail does "
                 "not say which part of Adam matters",
                 "the Adam LoRA's teacher-trained decomposition regenerated in the same stage (1.273, lora_ref); the SGD "
                 "LoRA's do-nothing (G_nothing)"),
        ],
        "readings": READINGS,
    }


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; filed predictions are never edited (add a dated report instead)")
    PRED.write_text(json.dumps({**predictions(), "config": config(), "filed": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=1))
    print(f"filed {PRED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions_amend3.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    file_predictions() if a.predictions else main(a.device)
