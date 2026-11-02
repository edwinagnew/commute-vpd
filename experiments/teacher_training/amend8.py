"""teacher_training E34 Amendment 8 (2026-11-02): what training does with only the base model W0 to learn from. Plan:
reports/plans/2026-11-02_plan_base-control.md; evidence reports/reviews/2026-11-02_review_base-control-audit.md. Rules,
leans and expected values: results/predictions_amend8.json (the run refuses without it, or if AM or run.CFG differ).

One stage per job, each a MATS `debug` job on an L40, its own results file results/teacher_training_amend8_<stage>.json.
Every job first reads W0 under the Pile harness (run.Job, 7 min) and stops if W0 is off its values or the card's floor is
not the L40's. Each read goes into the file as it finishes and is skipped on a rerun; training is regenerated (seeded),
since a read needs fp32 states. A missed early check stops the job before its reads.
  n1       teacher = W0, start = a random move of the alive U, V with E41's student's per-matrix size; read at 0, 100, 1000
  n2       teacher = W0, start = E41's teacher-trained T1000 state; T1000's do-nothing, then read at 0, 100, 1000
  n3a      the null fine-tune alone: every label drawn from W0, read at 1, 3, 10, 30, 100, 300, 1000 (size, KL to W0,
           P(target), no harness); chooses the checkpoint nearest the same-job T1000's size
  n3b      the student from W0 on that checkpoint; beside it the same-job real student at batch orders 0 and 1 (order 1 is
           the cosine's control). Refuses until n3a is done and its expected values are filed by dated note
  n4_adam  labelled training on self-labels, E39 Amendment 2's Adam 1e-5 on its lowercase rows; beside it true labels
  n4_sgd   the same at its plain SGD 0.03
  n5       masked_copy.train at coefficient 0 on E42's rand32 windows, teacher = W0

Usage (repo root, project venv, GPU):  amend8.py <stage> [--device cuda]; cluster/teacher_amend8.sbatch <stage>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from newskill import KV
from newskill_amend1 import read as e42a1_read
from path import init_of, student
from run import (
    CFG,
    EMO,
    FT_ADAM_LR,
    LARGE,
    RESULTS,
    Job,
    dump,
    halt,
    load,
    log,
    student_loss,
)
from torch import Tensor
from torch.nn import functional as F

from commute import faith_score

PRED = RESULTS / "predictions_amend8.json"
STATES = LARGE / "amend8"
STAGES = ("n1", "n2", "n3a", "n3b", "n4_adam", "n4_sgd", "n5")
CURVE_STEPS = (1, 2, 10, 100, 500, 1000)
Bound = tuple[float, float]


@dataclass(frozen=True)
class Amend8:
    floor: float = 0.2088  # W0's PGD floor on the L40 (E42 jobs 40157, 40338, 40497)
    floor_tol: float = 0.10  # relative; off it the job is on another card and the plan's references do not apply
    # the regenerated E41 pair (H100 values, teacher_training_e41.json; L40's E31 teacher size 0.0315): sanity, not a
    # comparison; absolute bounds wide enough for bf16 rounding between cards
    t_loss1: Bound = (5.23, 5.43)  # fine-tune step 1: 5.3339
    t_p: Bound = (0.9946, 1.0)  # T1000 held-out P(target): 0.99961
    t_size: Bound = (0.0301, 0.0341)  # T1000 size: 0.03209
    real_kl1: Bound = (5.24, 5.44)  # student step 1: 5.3418
    real_p: Bound = (0.9868, 1.0)  # student at 1000: 0.99675
    real_size: Bound = (0.0154, 0.0214)  # student at 1000: 0.01842
    warm_kl: Bound = (0.0, 0.005)  # lr 0 from the start against its own weights: E41's C2 line (floor 0.0005-0.0007)
    hook_floor_line: float = 0.0007  # the warm-start KL above this (H100's largest): gap_line needs a dated amendment
    n1_kl1: Bound = (0.0, 0.5)  # N1 step 1: far under a real gap; that the start has a gap is read at step 0
    n2_kl1: Bound = (1.0, 20.0)  # N2 step 1: KL(W0 || T1000 student), trigger mass on W0's tokens about 0.003
    n3a_steps: tuple[int, ...] = (1, 3, 10, 30, 100, 300, 1000)
    reproduce_tol: float = 0.01  # n3b's null teacher size / n3a's reading of the checkpoint, relative
    order2: int = 1  # the real student's second batch order: the cosine's control
    t1_tol: float = 0.05  # null fine-tune size at step 1 / the real one's (Adam's first step is lr on every entry; E41 A2 0.00346)
    s1_tol: float = 0.30  # null student size at step 1 / the real one's (0.0010-0.0012 in every E41 and E39 log)
    ce_tol: float = 0.5  # self-labels at step 1: |mean CE - W0's mean entropy|, nats
    move_seed: int = 0
    move_tol: float = 0.01  # N1's start size / the real student's, relative
    gap_line: float = 0.002  # N1: unmasked KL to W0 "at the hook floor", about 3 x the largest floor reading (0.0007)
    gap_share: float = 0.05  # N2: KL at 1000 / KL at 0 (E41's works_gap)
    n2_p_line: float = 0.04  # N2: P(target) at 1000 at or under this, ten times W0's 0.004: 5% of KL(W0 || state) allows P 0.25
    return_line: float = 0.76  # N1, N2: return fraction (the plan's figure, E34 C3's 0.76)
    about: float = 0.75  # n3b: null student's size / real student's at or above this is comparable
    under: float = 0.50  # n3b: at or below this reads "well under"
    cos_shared: float = 0.5  # n3b: cos(null student, real student) / cos(real student, second batch order) at or above: shared
    cos_none: float = 0.1  # n3b: cos(null student, real student) below this, absolute: no shared direction (random about 2e-4)
    ctrl_min: float = 0.2  # n3b: the control's cosine below this leaves cos_shared under cos_none: no power
    recipe_share: float = 0.5  # N4: (self - 1) / (true - 1) at or above this reads "about the true labels"
    rows_train: tuple[str, int, int] = ("train", 0, 4096)  # E39's rows: broad_fragility/run.py Config
    n_train: int = 2400
    rows_held: tuple[str, int, int] = ("val", 0, 256)
    n_held: int = 64
    label_batch: int = 8  # broad_fragility/run.py copy_batch; batch-order seed 0
    # per optimiser: lr and step of E39 A2's chosen state (results/amend2.json .label.*.chosen), its held-out gain, the
    # student steps run on self-labels and the step scored
    label: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "adam": {"lr": 1e-5, "true_steps": 1000, "gain": 0.1343, "scored": 1000},
        "sgd": {"lr": 0.03, "true_steps": 100, "gain": 0.1253, "scored": 100}})
    label_saves: tuple[int, ...] = (100, 1000)
    label_ce1: Bound = (3.48, 3.58)  # E39 A2's log: true-label step 1, 3.5300 for both optimisers
    gain_tol: float = 0.30  # relative, on the true-label state's gain: E39 A2's own match tolerance (amend2.py gain_tol)
    n5_task: str = "rand32"
    n5_kl1: Bound = (0.00147, 0.00179)  # step-1 unmasked term 0.00163 (`.G2.path_kl`, same L40), within 10%

    def __post_init__(self) -> None:
        assert CFG.steps == self.label_saves == (100, 1000) and self.under < self.about <= 1 and self.floor_tol > 0
        assert self.n3a_steps[-1] == CFG.steps[-1] and self.cos_none <= self.cos_shared * self.ctrl_min
        assert {v["scored"] for v in self.label.values()} <= set(self.label_saves)


AM = Amend8()


# ----------------------------------------------------------------------------- rules (pure, on the results file)


def base_verdict(pile: dict[str, float], w0: dict[str, Any]) -> str:
    """faith_score.state_verdict with P(target) out of it (with nothing to learn it would always read 'below P 0.9');
    a NaN ratio, which faith_score.verdict would call 'inconclusive', is 'invalid'."""
    if not all(math.isfinite(pile[k]) for k in faith_score.FLOOR_KEY):
        return "invalid"
    return faith_score.state_verdict(pile, 1.0, w0, p_min=0.0)["verdict"]


def return_fraction(size0: float, size1: float) -> float | None:
    """1 - size at the end / size at the start, size = harness's largest ||delta-W|| / ||W0||. Negative if it grew; None
    if the start is at W0 or a size is not finite."""
    if not (math.isfinite(size0) and math.isfinite(size1)) or size0 <= 0:
        return None
    return 1 - size1 / size0


def gap_class(g0: float, g1: float, line: float) -> str:
    """Unmasked KL to W0 from start to end against an absolute line. 'no power': the start is already at the line."""
    if not (math.isfinite(g0) and math.isfinite(g1) and math.isfinite(line)):
        return "invalid"
    if g0 <= line:
        return "no power"
    return "closed" if g1 <= line else "partly" if g1 < g0 else "not closed"


def size_class(null: float, real: float) -> tuple[str, float | None]:
    """N3: the null student's size over the same-job real student's: 'about 0.018', 'partly' or 'well under'."""
    if not (math.isfinite(null) and math.isfinite(real)) or real <= 0:
        return "unscorable", None
    r = null / real
    return ("about" if r >= AM.about else "well under" if r <= AM.under else "partly"), r


def nearest_checkpoint(sizes: dict[str, float], target: float) -> str | None:
    """n3a: the checkpoint whose size is nearest `target` (absolute difference; a tie goes to the earlier step). None if
    the target is not a positive number or no size is finite."""
    ok = [k for k, v in sizes.items() if math.isfinite(v)]
    if not ok or not math.isfinite(target) or target <= 0:
        return None
    return min(sorted(ok, key=int), key=lambda k: abs(sizes[k] - target))


def direction_class(cos: float, ctrl: float) -> str:
    """n3b: cos(null student, real student) against cos(real student, the same at a second batch order)."""
    if not (math.isfinite(cos) and math.isfinite(ctrl)):
        return "unscorable"
    if ctrl < AM.ctrl_min:
        return "no power"
    return "shared" if cos >= AM.cos_shared * ctrl else "none" if cos < AM.cos_none else "some"


def recipe_class(size: str, direction: str) -> str:
    """n3b: "the recipe" needs a comparable size and a direction shared with the real student. Every pair has a row."""
    if size == "unscorable" or direction in ("unscorable", "no power") and size != "well under":
        return "unscorable"
    if size == "well under":
        return "not the recipe"
    if direction == "shared":
        return "the recipe" if size == "about" else "partly"
    return "partly" if direction == "some" else "a walk elsewhere"


def outcome(verdict: str, parts: list[bool | None]) -> str:
    """Every rule's last line. A failed check is 'fails' whatever else reads; else a part that cannot be scored is
    'unscorable', a missed part 'fails'; 'holds' needs a pass and every part. Any other verdict is 'unscorable'."""
    if verdict == "fail":
        return "fails"
    if verdict not in ("pass", "inconclusive") or None in parts:
        return "unscorable"
    if not all(parts):
        return "fails"
    return "holds" if verdict == "pass" else "inconclusive"


def label_class(verdict: str, pgd_verdict: str, self_check: float, true_check: float) -> tuple[str, float | None]:
    """N4: the self-label state against the same-job true-label state of the same recipe."""
    if verdict == "invalid":
        return "unscorable", None
    if verdict != "fail":
        return ("near base" if verdict == "pass" else "inconclusive"), None
    if pgd_verdict != "fail":
        return "fails on the stochastic ratio", None
    if not math.isfinite(true_check) or true_check <= 1:
        return "fails, no reference", None
    share = (self_check - 1) / (true_check - 1)
    return ("fails, about the true labels" if share >= AM.recipe_share else "fails, under the true labels"), share


def score_return(r: dict[str, Any], line: float) -> dict[str, Any]:
    """N1, N2. `line`: the absolute gap line (N2's is gap_share x its own start)."""
    a, b = r["0"], r["1000"]
    gap, ret = gap_class(a["kl_w0"], b["kl_w0"], line), return_fraction(a["size"], b["size"])
    return {"gap": gap, "return_fraction": ret, "returned": None if ret is None else ret >= AM.return_line,
            "verdict": b["base_verdict"]}


def score_n3b(out: dict[str, Any]) -> dict[str, Any]:
    r, d = out["reads"], out["direction"]
    size, ratio = size_class(r["1000"]["size"], r["real"]["size"])
    direction = direction_class(d["null_real"], d["control"])
    cls, v = recipe_class(size, direction), r["1000"]["base_verdict"]
    part = None if cls == "unscorable" else cls in ("not the recipe", "a walk elsewhere")
    return {"size_class": size, "size_ratio": ratio, "direction": direction, "class": cls, **d, "verdict": v,
            "follows_null_teacher": d["null_teacher"] >= d["real_teacher"], "real_verdict": r["real"]["base_verdict"],
            "real_check": r["real"]["pgd"], "outcome": outcome(v, [part]), "card": cls == "the recipe" or v == "fail"}


def score_n4(stage: str, out: dict[str, Any]) -> dict[str, Any]:
    r = out["reads"]
    rec, true = r[str(AM.label[stage.removeprefix("n4_")]["scored"])], r["true"]
    v = rec["base_verdict"]
    cls, share = label_class(v, rec["pgd_verdict"], rec["pgd"], true["pgd"])
    result = outcome(v, [])
    if stage == "n4_adam" and true["base_verdict"] != "fail":  # the recipe did not fail on this card: no power either way
        cls, share = ("fails, no reference", None) if v == "fail" else ("no reference", None)
        result = result if v == "fail" else "no reference"
    return {"class": cls, "recipe_share": share, "verdict": v, "true_verdict": true["base_verdict"], "outcome": result,
            "card": v == "fail"}


def score(stage: str, out: dict[str, Any]) -> dict[str, Any]:
    r = out["reads"]
    if stage == "n1":
        s = score_return(r, AM.gap_line)
        gap_ok = None if s["gap"] in ("invalid", "no power") else s["gap"] == "closed"
        result = outcome(s["verdict"], [gap_ok])
        floor = out.get("early", {}).get("warm start KL", {}).get("value", math.nan)
        return s | {"outcome": "fails, gap not closed" if result == "fails" and s["verdict"] != "fail" else result,
                    "card": s["verdict"] == "fail", "hook_floor": floor, "gap_line_needs_amendment": floor > AM.hook_floor_line}
    if stage == "n2":
        s = score_return(r, AM.gap_share * r["0"]["kl_w0"])
        p = r["1000"]["p_target"]
        p_back = None if not math.isfinite(p) else p <= AM.n2_p_line
        gap_ok = None if s["gap"] in ("invalid", "no power") else s["gap"] == "closed"
        return s | {"p_back": p_back, "real_verdict": r["0"]["base_verdict"], "real_check": r["0"]["pgd"],
                    "outcome": outcome(s["verdict"], [gap_ok, p_back, s["returned"]]),
                    "card": s["verdict"] == "fail" or s["returned"] is False or p_back is False}
    if stage == "n3a":
        sizes = {k: v["size"] for k, v in out["null_checkpoints"].items()}
        step = nearest_checkpoint(sizes, out["real_teacher"]["size"])
        return {"chosen_step": step, "size": None if step is None else sizes[step], "target": out["real_teacher"]["size"]}
    if stage == "n3b":
        return score_n3b(out)
    if stage == "n5":
        v = r["1000"]["base_verdict"]
        return {"verdict": v, "outcome": outcome(v, []), "card": v == "fail"}
    return score_n4(stage, out)


# ----------------------------------------------------------------------------- early checks and reads


def check(out: dict[str, Any], path: Path, name: str, value: float, bound: Bound) -> None:
    """A filed expected value; a miss (a NaN too) stops the job."""
    ok = bound[0] <= value <= bound[1]
    out.setdefault("early", {})[name] = {"value": value, "bound": bound, "pass": ok}
    log(f"[early {name}] {value:.6g} in {bound}: {ok}")
    dump(path, out)
    if not ok:
        halt(path, out, f"early check {name}: {value:.6g} outside {bound}")


def near_one(tol: float) -> Bound:
    return (1 - tol, 1 + tol)


def state_of(dW: dict[str, Tensor], uv2: dict[str, tuple[Tensor, Tensor]]) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    return {k: (dW[k], *uv2[k]) for k in dW}


def deltas(st: dict[str, Any]) -> dict[str, Tensor]:
    return {k: v[0] for k, v in st.items()}


def finish(job: Job, out: dict[str, Any], path: Path, name: str, rec: dict[str, Any], st: Any) -> dict[str, Any]:
    """The read's verdict; the state saved (bf16 offsets), then the record, so a skipped read has its file."""
    from commute import arms

    rec["base_verdict"] = base_verdict(rec["pile_ratios"], job.w0)
    rec["pgd_verdict"] = faith_score.verdict(rec["pile_ratios"]["pgd_kl"], job.w0["pgd_floor"])
    if st is not None:
        arms.save_offsets(STATES / out["stage"] / f"{name}.pt", job.ctx.uv_cpu, st, {"stage": out["stage"], "read": name})
    out["reads"][name] = rec
    dump(path, out)
    log(f"[{out['stage']} {name}] {json.dumps(rec, default=float)}")
    return rec


def read(job: Job, out: dict[str, Any], path: Path, tctx: Any, name: str, dW: dict[str, Tensor], st: Any, net: Any,
         net0: Any) -> dict[str, Any]:
    """One state on task windows: the ratios and verdict (newskill_amend1.read, which runs the harness), size, P(target),
    and the unmasked KL to W0 (arm A's weighting, held-out windows). `st` None: do-nothing."""
    from commute import harness, teacher

    if name in out["reads"]:
        return out["reads"][name]
    full = e42a1_read(job, tctx, f"{out['stage']} {name}", (dW, st), net, True)
    with harness.at(tctx, dW, None):
        kl_w0 = teacher.heldout_gap(tctx.model.target_model, net0, tctx.held, tctx.device)
    keep = ("pgd", "stochastic", "pile_ratios", "p_target", "teacher_gap", "damage", "size", "delta_share", "verdict")
    return finish(job, out, path, name, {k: full[k] for k in keep} | {"kl_w0": kl_w0}, st)


def train_from(tctx: Any, net: Any, init: Any, saves: tuple[int, ...], seed: int = 0) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """path.student's recipe (arm A, E34's settings) with states at `saves`; path.student keeps only the last."""
    from commute import component_training
    from commute.finetune import ArmAConfig

    last = saves[-1]
    a_cfg = ArmAConfig(o_token=tctx.token, batch=CFG.batch_a, steps=last, checkpoints=(last,), seed=seed)
    cfg = component_training.TrainConfig(lr=CFG.student_lr, steps=last, save_steps=saves, optimizer="adam")
    half = tctx.train.half
    return component_training.train(tctx.model, tctx.uv, tctx.train.tokens, tctx.held, a_cfg, cfg, tctx.device, tctx.W0, log=log,
                                    loss_fn=lambda n, toks: student_loss(n, net, toks, half), init=init)


def keep_curve(out: dict[str, Any], path: Path, name: str, curve: list[dict[str, Any]]) -> None:
    out.setdefault("curves", {})[name] = [c for c in curve if c["step"] in CURVE_STEPS]
    dump(path, out)


def real_teacher(job: Job, out: dict[str, Any], path: Path) -> Any:
    """E41's emoticon T1000 regenerated in-job: the Adam 3e-5 fine-tune, with its size at step 1."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    tctx = job.task(EMO)[0]
    cfg = ArmAConfig(o_token=tctx.token, lr=FT_ADAM_LR, steps=CFG.steps[-1], checkpoints=(1, CFG.steps[-1]), seed=0, optimizer="adam")
    res = finetune_arm_a(tctx.model, tctx.train, tctx.held, cfg, tctx.device, log)
    ev = res.evals[CFG.steps[-1]]
    out["real_teacher"] = {"p_target": ev["p_o"], "damage": ev["kl_other"], "size": ev["rel_norm_max"], "size_1": res.evals[1]["rel_norm_max"]}
    check(out, path, "T1000 step-1 loss", res.log[0]["loss"], AM.t_loss1)
    check(out, path, "T1000 P(target)", ev["p_o"], AM.t_p)
    check(out, path, "T1000 size", ev["rel_norm_max"], AM.t_size)
    return SimpleNamespace(tctx=tctx, dW=res.deltas[CFG.steps[-1]], t_size1=res.evals[1]["rel_norm_max"])


def real_pair(job: Job, out: dict[str, Any], path: Path) -> Any:
    """T1000 (`real_teacher`) and its arm-A student from W0, as E41's."""
    from commute import teacher

    p = real_teacher(job, out, path)
    p.net = teacher.teacher_net(p.tctx.model, p.dW)
    curve, states = train_from(p.tctx, p.net, None, (CFG.steps[-1],))
    p.st = states[CFG.steps[-1]]
    keep_curve(out, path, "real", curve)
    check(out, path, "student step-1 KL", curve[0]["loss"], AM.real_kl1)
    check(out, path, "student P(target)", curve[-1]["heldout_p_o"], AM.real_p)
    check(out, path, "student size", curve[-1]["rel_norm_max"], AM.real_size)
    p.size, p.size1 = curve[-1]["rel_norm_max"], curve[0]["rel_norm_max"]
    return p


def walk_back(job: Job, out: dict[str, Any], path: Path, tctx: Any, st0: dict[str, Any], kl1: Bound, gap_above: float) -> None:
    """N1, N2: teacher = W0 from `st0`, fresh Adam (as an E41 chain link). The warm start reproduces its own weights
    (one step at lr 0; its KL is this card's hook floor), then the start is read: its KL to W0 must be strictly above
    `gap_above`, so a job cannot run on to read 'no power'. Then steps 100 and 1000."""
    from commute import teacher

    net0, init = teacher.teacher_net(tctx.model, {}), init_of(tctx, st0)
    curve, _ = student(job, EMO, "A", teacher.teacher_net(tctx.model, deltas(st0)), init, steps=1, lr=0.0)
    check(out, path, "warm start KL", curve[0]["loss"], AM.warm_kl)
    rec = read(job, out, path, tctx, "0", deltas(st0), st0, net0, net0)
    check(out, path, "start KL to W0 above the gap line", rec["kl_w0"], (math.nextafter(gap_above, math.inf), 1e9))
    curve, states = train_from(tctx, net0, init, CFG.steps)
    keep_curve(out, path, "walk", curve)
    check(out, path, "step-1 KL to W0", curve[0]["loss"], kl1)
    for s in CFG.steps:
        read(job, out, path, tctx, str(s), deltas(states[s]), states[s], net0, net0)


# ----------------------------------------------------------------------------- self-labels


def self_ce(net: Any, net0: Any, toks: Tensor, half: int | None, gen: torch.Generator) -> tuple[Tensor, dict[str, float]]:
    """Cross-entropy on labels drawn afresh, position by position, from W0's own distribution, so the expected gradient
    at W0 is 0 and the per-batch gradient is not. `half` given: arm A's weighting (the trigger plus the mean elsewhere);
    None: finetune.next_token_ce's plain mean over positions 0 .. T-2."""
    from commute.finetune import arm_a_weighting, logits_of

    lg = logits_of(net, toks)
    with torch.no_grad():
        p0 = torch.softmax(logits_of(net0, toks), -1)
        labels = torch.multinomial(p0.flatten(0, 1), 1, generator=gen).view(toks.shape)
        entropy = -(p0 * p0.clamp_min(1e-30).log()).sum(-1)
    ce = F.cross_entropy(lg.transpose(1, 2), labels, reduction="none")  # (b, L)
    loss = ce[:, :-1].mean() if half is None else arm_a_weighting(ce[:, half].mean(), ce, half, 1.0)[0]
    return loss, {"ce": ce[:, :-1].mean().item(), "entropy": entropy[:, :-1].mean().item()}


def null_fine_tune(tctx: Any, net0: Any, checkpoints: tuple[int, ...]) -> Any:
    """Arm A's fine-tune (`finetune.finetune_arm_a`: all 24 matrices, Adam 3e-5, batch 32, batch order 0) with self_ce, to
    the last checkpoint; each checkpoint reads P(target), damage, size and the unmasked KL to W0. A shorter run draws
    the same batches and labels as a longer one."""
    import copy

    from commute import teacher
    from commute.finetune import _run, decomposed_params, eval_arm_a

    net, ref = tctx.model.target_model, copy.deepcopy(tctx.model.target_model).eval().requires_grad_(False)
    gen, lgen = torch.Generator().manual_seed(0), torch.Generator(device=tctx.device).manual_seed(0)

    def loss_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        toks = tctx.train.tokens[torch.randint(0, len(tctx.train), (CFG.batch_a,), generator=gen)].to(tctx.device)
        return self_ce(net, net0, toks, tctx.train.half, lgen)

    def eval_fn() -> dict[str, Any]:
        return eval_arm_a(net, ref, tctx.held, tctx.token, tctx.device) | {"kl_w0": teacher.heldout_gap(net, net0, tctx.held, tctx.device)}

    return _run(tctx.model, decomposed_params(tctx.model), FT_ADAM_LR, 1.0, checkpoints[-1], checkpoints, loss_fn, eval_fn, log, "adam")


# ----------------------------------------------------------------------------- the stages


def n1(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute.fragility import delta_w, random_move

    p = real_pair(job, out, path)
    uv = p.tctx.uv_cpu
    uv2 = random_move(uv, {k: float(v[0].double().norm()) for k, v in p.st.items()}, AM.move_seed)
    st0 = state_of(delta_w(uv, uv2), uv2)
    size0 = max(float(st0[k][0].norm() / p.tctx.W0[k].cpu().norm()) for k in st0)
    check(out, path, "start size / student's", size0 / p.size, near_one(AM.move_tol))
    walk_back(job, out, path, p.tctx, st0, AM.n1_kl1, AM.gap_line)


def n2(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import teacher

    p = real_pair(job, out, path)
    read(job, out, path, p.tctx, "nothing", p.dW, None, p.net, teacher.teacher_net(p.tctx.model, {}))
    walk_back(job, out, path, p.tctx, p.st, AM.n2_kl1, 0.0)


def n3a(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import teacher

    p = real_teacher(job, out, path)
    res = null_fine_tune(p.tctx, teacher.teacher_net(p.tctx.model, {}), AM.n3a_steps)
    check(out, path, "self-label CE - entropy, step 1", res.log[0]["ce"] - res.log[0]["entropy"], (-AM.ce_tol, AM.ce_tol))
    check(out, path, "null fine-tune size at step 1 / real", res.evals[1]["rel_norm_max"] / p.t_size1, near_one(AM.t1_tol))
    out["null_checkpoints"] = {str(k): {"size": v["rel_norm_max"], "kl_w0": v["kl_w0"], "damage": v["kl_other"], "p_target": v["p_o"]}
                               for k, v in res.evals.items()}
    dump(path, out)
    log(f"[n3a checkpoints] {json.dumps(out['null_checkpoints'], default=float)}")


def n3a_choice(job: Job) -> dict[str, Any]:
    """n3a's finished reading on this card; n3b refuses without it (main refuses it without the dated note)."""
    path = RESULTS / "teacher_training_amend8_n3a.json"
    a = json.loads(path.read_text()) if path.is_file() else {}
    if not a.get("done") or a["card"] != job.card or a["score"]["chosen_step"] is None:
        sys.exit(f"n3b needs n3a done on {job.card} with a chosen checkpoint: {path}")
    return a["score"]


def n3b(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import teacher
    from commute.finetune import cosines

    chosen = n3a_choice(job)
    p = real_pair(job, out, path)
    tctx, step = p.tctx, int(chosen["chosen_step"])
    net0 = teacher.teacher_net(tctx.model, {})
    read(job, out, path, tctx, "real", deltas(p.st), p.st, p.net, net0)
    _, order2 = train_from(tctx, p.net, None, (CFG.steps[-1],), seed=AM.order2)
    res = null_fine_tune(tctx, net0, (step,))
    out["null_teacher"] = {"step": step, **res.evals[step]}
    check(out, path, "null teacher size / n3a's", res.evals[step]["rel_norm_max"] / chosen["size"], near_one(AM.reproduce_tol))
    dW = res.deltas[step]
    STATES.joinpath("n3b").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.bfloat16() for k, v in dW.items()}, STATES / "n3b" / "null_teacher_dW.pt")
    net = teacher.teacher_net(tctx.model, dW)
    read(job, out, path, tctx, "nothing", dW, None, net, net0)
    curve, states = train_from(tctx, net, None, CFG.steps)
    keep_curve(out, path, "null_student", curve)
    log(f"[n3b] null student step-1 KL {curve[0]['loss']:.6g} (logged, no stop)")
    check(out, path, "null student size at step 1 / real", curve[0]["rel_norm_max"] / p.size1, near_one(AM.s1_tol))
    for s in CFG.steps:
        read(job, out, path, tctx, str(s), deltas(states[s]), states[s], net, net0)
    null, real = deltas(states[CFG.steps[-1]]), deltas(p.st)
    out["direction"] = {"null_real": cosines(null, real)["all"], "null_teacher": cosines(null, dW)["all"],
                        "real_teacher": cosines(real, p.dW)["all"], "control": cosines(real, deltas(order2[CFG.steps[-1]]))["all"]}
    dump(path, out)
    log(f"[n3b direction] {json.dumps(out['direction'])}")


def label_train(job: Job, rows: Tensor, opt: str, lr: float, saves: tuple[int, ...], loss_fn: Any) -> Any:
    """E39 Amendment 2's `label` recipe (broad_fragility/amend2.py): component_training.train on the rows, Gamma and Delta
    frozen, batch 8, batch order 0; only the loss differs between true labels and self-labels."""
    from commute import component_training
    from commute.finetune import ArmAConfig

    c, last = job.ctx, saves[-1]
    a_cfg = ArmAConfig(o_token=c.token, batch=AM.label_batch, steps=last, checkpoints=(last,), seed=0)
    cfg = component_training.TrainConfig(lr=lr, steps=last, save_steps=saves, optimizer=opt)
    return component_training.train(c.model, c.uv, rows, c.held, a_cfg, cfg, c.device, c.W0, log=log, loss_fn=loss_fn)


def n4(job: Job, out: dict[str, Any], path: Path, opt: str) -> None:
    """True labels first (the recipe's control on this card), then self-labels, each read with row_copy.read_state: gain on
    the lowercase held-out rows, KL to W0 on them (`gap`) and on plain rows (`damage`), the ratios."""
    from commute import teacher, vpd
    from commute.data import pile_rows
    from commute.finetune import next_token_ce
    from commute.row_copy import Refs, lowercase_rows, mean_logp, read_state

    c, cfg, tok = job.ctx, AM.label[opt], vpd.tokenizer()
    rows = lowercase_rows(pile_rows(*AM.rows_train, log=log), tok, AM.n_train)
    val = pile_rows(*AM.rows_held, log=log)
    held, plain = lowercase_rows(val, tok, AM.n_held), val[: AM.n_held]
    net0 = teacher.teacher_net(c.model, {})
    refs, logp0 = Refs(net0, net0, held, plain, AM.label_batch), mean_logp(net0, held, c.device, AM.label_batch)

    def state(name: str, st: Any) -> dict[str, Any]:
        if name in out["reads"]:
            return out["reads"][name]
        dW, uv2 = deltas(st), {k: (v[1], v[2]) for k, v in st.items()}
        r = read_state(c, refs, dW, uv2, True, log)
        rec = {"pgd": r["pile"]["pgd_kl"], "stochastic": r["pile"]["stochastic_kl"], "pile_ratios": r["pile"],
               "gain": r["logp"] - logp0, "kl_w0_lowercase": r["gap"], "damage": r["damage"],
               "size": max(float(dW[k].norm() / c.W0[k].cpu().norm()) for k in dW)}
        return finish(job, out, path, name, rec, st)

    steps = cfg["true_steps"]
    curve, states = label_train(job, rows, opt, cfg["lr"], (steps,), next_token_ce)
    keep_curve(out, path, "true", curve)
    check(out, path, "true labels step-1 CE", curve[0]["loss"], AM.label_ce1)
    rec = state("true", states[steps])
    check(out, path, "true labels gain", rec["gain"], (cfg["gain"] * (1 - AM.gain_tol), cfg["gain"] * (1 + AM.gain_tol)))
    del states
    lgen = torch.Generator(device=c.device).manual_seed(0)
    curve, states = label_train(job, rows, opt, cfg["lr"], AM.label_saves, lambda n, t: self_ce(n, net0, t, None, lgen))
    keep_curve(out, path, "self", curve)
    check(out, path, "self-label CE - entropy, step 1", curve[0]["train_ce"] - curve[0]["train_entropy"], (-AM.ce_tol, AM.ce_tol))
    for s in AM.label_saves:
        state(str(s), states[s])


def n5(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import kv_windows, masked_copy, teacher

    tctx, _ = kv_windows.task_context(job.ctx, KV / f"windows_{AM.n5_task}.pt")
    net0 = teacher.teacher_net(tctx.model, {})
    cfg = masked_copy.CopyConfig(lr=CFG.student_lr, batch=CFG.batch_a, steps=CFG.steps[-1], save_steps=CFG.steps, coeff=0.0)
    curve, states = masked_copy.train(tctx.model, net0, tctx.train, tctx.alive, tctx.uv, cfg, log)
    keep_curve(out, path, "masked_copy", curve)
    check(out, path, "step-1 unmasked term", curve[0]["unmasked"], AM.n5_kl1)
    check(out, path, "step-1 gradient norm", curve[0]["grad_norm"], (1e-12, 1e6))
    for s in CFG.steps:
        read(job, out, path, tctx, str(s), deltas(states[s]), states[s], net0, net0)


def run_stage(job: Job, stage: str) -> None:
    path = RESULTS / f"teacher_training_amend8_{stage}.json"
    out = load(path, job, {"stage": stage, "reads": {}})
    dump(path, out)
    floor = job.w0["pgd_floor"]
    if not job.w0["pass"] or abs(floor / AM.floor - 1) > AM.floor_tol:
        halt(path, out, f"W0 first: check {job.w0['pass']}, floor {floor:.4f} (L40 {AM.floor} within {AM.floor_tol:.0%}) on {job.card}")
    if out.get("done"):
        return log(f"{stage} is done")
    if stage.startswith("n4_"):
        n4(job, out, path, stage.removeprefix("n4_"))
    else:
        {"n1": n1, "n2": n2, "n3a": n3a, "n3b": n3b, "n5": n5}[stage](job, out, path)
    out["score"], out["done"] = score(stage, out), True
    dump(path, out)
    log(f"[{stage} score] {json.dumps(out['score'], default=float)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    for key, cfg in (("config", CFG), ("amend8", AM)):
        assert filed[key] == json.loads(json.dumps(asdict(cfg))), f"predictions were filed for another {key}: {filed[key]}"
    if args.stage == "n3b" and not filed["n3b_expected_from_n3a"]["dated_note"]:
        sys.exit(f"n3b needs its expected values filed from n3a's reading: {PRED} n3b_expected_from_n3a")
    run_stage(Job(args.device), args.stage)
    log("done")


if __name__ == "__main__":
    main()
