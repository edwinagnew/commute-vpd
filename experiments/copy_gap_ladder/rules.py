"""E121 copy_gap_ladder: the one config and the filed rules (pure; no model). run.py imports both; a CPU test can too.
Plan reports/plans/2026-11-06_plan_e121-copy-gap-ladder.md; filing results/predictions.json."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

SETS = {"task": "task:emoticon", "pile": "pile"}  # this job's set: Amendment 10's name for it
RUNGS = {"R1": ("plain", "adam", "ft_lr", "zero"), "R2": ("plain", "adam", "copy_lr", "zero"), "R3": ("uv", "adam", "copy_lr", "zero"),
         "R3b": ("uv", "sgd", "sgd_lr", "zero"), "R4": ("uv", "adam", "copy_lr", "ls")}  # parametrisation, optimizer, lr, start
LADDER = ("R1", "R2", "R3")


@dataclass(frozen=True)
class Config:
    ft_lr: float = 3e-5  # the fine-tune's Adam lr: teacher_training/run.py:46 FT_ADAM_LR
    copy_lr: float = 1e-5  # the copy's Adam lr: teacher_training/run.py:55 Config.student_lr
    sgd_lr: float = 0.01  # E34 arm S's plain-SGD rate on this task: teacher_training/run.py:45 SGD_LR[emoticon]
    batch: int = 32  # both recipes: finetune.ArmAConfig.batch, teacher_training/run.py:56 batch_a
    steps: int = 1000  # both recipes: teacher_training/run.py:54 steps[-1]
    orders: tuple[int, ...] = (1, 2)  # scored batch orders; dW is order 0, Amendment 10's reruns 1 and 2
    control_orders: tuple[int, ...] = (0, 1)  # the copy regenerated at A8c's two orders, against its saved files
    pile_rows: int = 256  # amend10.py:87
    batch_windows: int = 32  # amend10.py:88
    batch_pile: int = 8  # amend10.py:89
    read_tol: float = 1e-3  # a cell reread from saved files: amend9.py read_tol (bf16 against fp32)
    saved_tol: float = 1e-3  # 1 - Frobenius cosine, a regenerated state against its saved file: amend10.py:77
    t_p_min: float = 0.9946  # the regenerated fine-tune's held-out P(target): amend10.py:78
    rerun_floor: float = 0.5  # a rerun line under it has no power: amend10.py:79
    gap_min: float = 0.05  # the least margin between two reads: amend10.py:80
    most: float = 0.5  # one rung's drop at or over this share of the total gap carries most of it: amend9.py half
    copy_gap_max: float = 0.1  # a copy's teacher gap at most this x W0's: teacher_training/run.py:65 c3_gap_max
    copy_p_tol: float = 0.05  # and P(target) within this of the teacher's: teacher_training/run.py:66 c3_p_tol
    ls_tol: float = 1e-6  # split.bases' default, every split so far (amend9.py tol)
    budget_s: float = 1560.0  # a state starts only if it ends inside 26 min of the 30-min job
    student_s: float = 140.0  # one component-training student: 111 s on the L40 (A8c job 48613 log, 13:31:40-13:33:31)
    plain_s: float = 50.0  # one plain 1,000-step run: 38 s (amend10.json controls."rerun real order 0".seconds)

    def __post_init__(self) -> None:
        assert 0 not in self.orders and len(self.orders) == 2 and self.control_orders == (0, 1) and 0 < self.most < 1


CFG = Config()


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


# ----------------------------------------------------------------------------- rules (pure), as filed


def copies(b: dict[str, float], line: dict[str, float]) -> bool:
    """A state keeps the fine-tune's behaviour: teacher gap <= copy_gap_max x W0's, P(target) within copy_p_tol of the teacher's."""
    return b["gap"] <= CFG.copy_gap_max * line["gap_w0"] and abs(b["p_target"] - line["p_teacher"]) <= CFG.copy_p_tol


def margin(noises: list[float]) -> float:
    return max(CFG.gap_min, *noises) if finite(*noises) else math.nan


def drop_label(d: float, total: float, m: float) -> str:
    if d < -m:
        return "raises"
    if abs(d) <= m:
        return "none"
    return "most" if d >= CFG.most * total else "part"


def ladder(c: list[float], copied: list[bool], m: float) -> dict[str, Any]:
    """c = [R0 line, R1, R2, R3]; copied = R1-R3 keep the behaviour. Which rung carries the drop from the rerun line to the copy?"""
    if not finite(*c, m):
        return {"verdict": "unscorable"}
    if not all(copied):
        return {"verdict": f"failed copy at {LADDER[copied.index(False)]}"}
    total = c[0] - c[3]
    drops = [c[i - 1] - c[i] for i in (1, 2, 3)]
    labels = [drop_label(d, total, m) for d in drops]
    if total <= m:
        return {"verdict": "no gap", "drops": drops, "labels": labels}
    most = [LADDER[i] for i, x in enumerate(labels) if x == "most"]
    verdict = most[0] if len(most) == 1 else f"shared: {' and '.join(most)}" if most else "spread"
    return {"verdict": verdict, "total": total, "drops": drops, "labels": labels, "margin": m}


def side(c3: float, x: float, m: float, copied: bool) -> str:
    """R3b against the copy R3."""
    if not finite(c3, x, m):
        return "unscorable"
    if not copied:
        return "failed copy"
    return "closer than the copy" if x - c3 > m else "further than the copy" if c3 - x > m else "same as the copy"


def start_label(start: float, end: float, c3: float, m: float, copied: bool) -> str:
    """R4: from the least-squares split's image of dW, does teacher training stay there or drift to the copy?"""
    if not finite(start, end, c3, m):
        return "unscorable"
    if not copied:
        return "failed copy"
    if start - c3 <= 2 * m:
        return "no power"
    if end > start + m:
        return "moves closer"
    if start - end <= m:
        return "stays"
    if end < c3 - m:
        return "drifts past the copy"
    return "drifts to the copy" if end - c3 <= m else "drifts part way"


def function_label(f3: float, f0: float, m: float) -> str:
    """The copy's change in trigger logits against the reruns', both against dW's."""
    if not finite(f3, f0, m):
        return "unscorable"
    if f0 < CFG.rerun_floor:
        return "cannot tell"
    return "as close as a rerun" if f3 >= f0 - m else "further than a rerun"


def reading(task: str, pile: str) -> str:
    if "unscorable" in (task, pile):
        return "unscorable"
    return task if task == pile else "depends on the inputs"


def agree(xs: list[str]) -> str:
    return xs[0] if len(set(xs)) == 1 else "mixed"


def score(reads: dict[str, Any], beh: dict[str, Any], line: dict[str, float], explain: dict[str, Any]) -> dict[str, Any]:
    o1, o2 = (f"_o{o}" for o in CFG.orders)
    mean = lambda d, r: statistics.fmean([d[r + o1], d[r + o2]])
    noise = lambda d, r: abs(d[r + o1] - d[r + o2])
    copied = {r: copies(beh[r + o1], line) and copies(beh[r + o2], line) for r in RUNGS}
    out: dict[str, Any] = {"copied": copied}
    for s in SETS:
        cos = reads[s]["cos"]
        r0 = [cos["R0_o1"], cos["R0_o2"], reads[s]["R0_pair"]]
        m = margin([max(r0) - min(r0), *(noise(cos, r) for r in RUNGS)])
        c = [statistics.fmean(r0), *(mean(cos, r) for r in LADDER)]
        out[s] = {"line": c, "margin": m, "ladder": ladder(c, [copied[r] for r in LADDER], m),
                  "R3b": side(c[3], mean(cos, "R3b"), m, copied["R3b"]),
                  "R4": start_label(cos["R4_start"], mean(cos, "R4"), c[3], m, copied["R4"])}
    f = {n: b["fcos"] for n, b in beh.items()}
    mf = margin([abs(f["R0_o1"] - f["R0_o2"]), noise(f, "R3")])
    out["P1_ladder"] = reading(out["task"]["ladder"]["verdict"], out["pile"]["ladder"]["verdict"])
    out["P2_sgd"] = reading(out["task"]["R3b"], out["pile"]["R3b"])
    out["P3_start"] = reading(out["task"]["R4"], out["pile"]["R4"])
    out["P4_function"] = {"label": function_label(mean(f, "R3"), mean(f, "R0"), mf), "rerun": mean(f, "R0"), "copy": mean(f, "R3"), "margin": mf,
                          "per_rung": {r: mean(f, r) for r in ("R0", *RUNGS)}}
    out["P5_remainder"] = {r: agree([explain[r + o]["remainder"] for o in (o1, o2)]) for r in ("R0", *RUNGS)}
    return out

