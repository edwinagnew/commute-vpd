"""The old near-realistic 67M tasks of cell P2.74 (inventory reports/analyses/2026-11-23_analysis_p2-74-old-task-inventory.md):
each task's read rows as its own experiment fed them, its saved fine-tunes, the seeds a retrain adds with the original
recipe, and the descriptive twin rule. Kept apart from `old_tasks.TASKS`, which E139's runs iterate (adding tasks there
would change what those scripts read); saved deltas load through `old_tasks.load_module_dw` (module keys, as `Net.at`).

Read rows per kind:
  trigger  E1's emoticon, E5's ladder and E33's cue windows (41 tokens, the trigger at 20), held-out windows read at the
           trigger with the training windows as steering's fitting set; the target one id, or E33's per-cue table
           (`many_cues.table`), as `regating.load_windows` and many_cues' windows.pt give them (a twin: another id)
  e53      E53's lowercase rows as E211 read them (`realistic_tasks/run2.e53_read`)
  case     E112's real case label as E136 read it (`share_untrained/windows.load_read`): 41 tokens, the label at 20
Recipe of every retrain: arm A (`finetune.ArmAConfig` defaults: Adam 3e-5, batch 32, clip 1, kl_lambda 1, all 24
matrices), 10 steps, the seed as the batch sampler's: E38's `recruitment_test/run.py:54-60` via `regating.finetune`
(`regating.py:81-88`), which E6 (`recruit_test/run.py:37-38, 359-380`), E28 (`regating_tasks/run.py:43, 198-205`) and
E33 (`many_cues/run.py:43, 369-370`) also ran.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import many_cues as MC
from commute import realistic_reads as RR
from commute import regating as G
from commute import steering as ST
from commute.data import Windows, target_tokens
from commute.jobs import EXPERIMENTS, load_experiment

HTTP, EOT = 3944, 0  # E18's ' http' twin target; E6's null_a, <|endoftext|> (recruit.json null_targets)
TRANSPORT, LADDER = EXPERIMENTS / "transport_test" / "results", EXPERIMENTS / "task_ladder" / "results" / "large"


def _large(folder: str) -> Path:
    return EXPERIMENTS / folder / "results" / "large"


RT, RAC, TT = _large("recruitment_test") / "states", _large("reading_a_coverage"), _large("transport_test")


@dataclass(frozen=True)
class NearTask:
    name: str
    kind: str  # trigger, e53, case
    source: str  # trigger: "emoticon", a ladder task or "many_cues"; e53: E53's task; case: E136's inventory task
    target: int | str | None = None  # trigger: a token id, "own" (the windows' own) or "table" (E33's per-cue answers)
    saved: dict[str, Path] = field(default_factory=dict)  # state -> saved delta-W, the state "<task>_s<seed>" at step 10
    retrain: tuple[int, ...] = ()  # seeds trained here (stage "train"), states "<task>_s<seed>"
    twin_of: str | None = None  # the real task this one is the different-target twin of (same windows, same recipe)


def _seeds(d: Path, name: str, stem: str, seeds: Sequence[int]) -> dict[str, Path]:
    return {f"{name}_s{s}": d / stem.format(s=s) for s in seeds}


def _ladder(name: str, saved: dict[str, Path], retrain: tuple[int, ...] = ()) -> NearTask:
    return NearTask(name, "trigger", name, "own", saved, retrain)


E1_STEPS = (10, 30, 100, 300)  # E1's arm A seed 0 (transport_test/A), read as a trajectory beside the step-10 seeds
TASKS: dict[str, NearTask] = {t.name: t for t in [
    NearTask("emoticon", "trigger", "emoticon", "own",
             _seeds(RT, "emoticon", "emoticon_s{s}.pt", (0, 1)) | {f"emoticon_e1_step{n}": TT / "A" / f"dW_step{n}.pt" for n in E1_STEPS}, (2,)),
    NearTask("emoticon_http", "trigger", "emoticon", HTTP, {"emoticon_http_s0": RT / "emoticon_null.pt"}, (1, 2), "emoticon"),
    NearTask("emoticon_eot", "trigger", "emoticon", EOT, {}, (0, 1, 2), "emoticon"),
    _ladder("r0_t588_v320", {}, (0,)),
    _ladder("r0_t352_v310", _seeds(RAC / "r0_t352_v310", "r0_t352_v310", "dW_ft_step10_s{s}.pt", (0,))),
    _ladder("r1_t588_v452", _seeds(RAC / "r1_t588_v452", "r1_t588_v452", "dW_ft_step10_s{s}.pt", (0,))),
    _ladder("r1_t534_v369", _seeds(RT, "r1_t534_v369", "r1_t534_v369_s{s}.pt", (0, 1)), (2,)),
    NearTask("r1_t534_v369_http", "trigger", "r1_t534_v369", HTTP, {"r1_t534_v369_http_s0": RT / "r1_t534_v369_null.pt"}, (1, 2), "r1_t534_v369"),
    _ladder("r2_t359_v816", _seeds(RT, "r2_t359_v816", "r2_t359_v816_s{s}.pt", (0, 1)), (2,)),
    NearTask("r2_t359_v816_http", "trigger", "r2_t359_v816", HTTP, {"r2_t359_v816_http_s0": RT / "r2_t359_v816_null.pt"}, (1, 2), "r2_t359_v816"),
    _ladder("r2_t275_v2667", _seeds(RAC / "r2_t275_v2667", "r2_t275_v2667", "dW_ft_step10_s{s}.pt", (0,))),
    NearTask("many_one", "trigger", "many_cues", "own", {}, (0,)),
    NearTask("many_many", "trigger", "many_cues", "table", {}, (0,), "many_one"),
    NearTask("lowercase", "e53", "lowercase", None, _seeds(_large("performed_anchor"), "lowercase", "dW_lowercase_s{s}_step300.pt", (0, 1))),
    NearTask("case_real", "case", "case_label/real", None,
             _seeds(_large("case_label"), "case_real", "dW_real_s{s}.pt", (0, 1)) | {"case_real_s2": _large("real_vs_twin") / "dW_case_real_s2.pt"}),
]}  # fmt: skip


def states(task: NearTask, which: str) -> list[str]:
    """The task's states: "saved" (on disk before this run), "new" (trained here) or "all"."""
    new = [f"{task.name}_s{s}" for s in task.retrain]
    return {"saved": list(task.saved), "new": new, "all": list(task.saved) + new}[which]


def seed_of(state: str) -> int:
    return int(state.rsplit("_s", 1)[1]) if "_e1_step" not in state else 0


# ---- read rows ------------------------------------------------------------------------------------------------------


@dataclass
class Rows:
    """A task's Read, and for a trigger task the Windows and target its arm-A retrain takes."""

    read: RR.Read
    train: Windows | None = None
    held: Windows | None = None
    target: int | Tensor | None = None


def trigger_windows(source: str) -> tuple[Windows, Windows, int | Tensor, dict[str, Any]]:
    """(train, held-out, the windows' own target, E33's mining record or {}) of a trigger source."""
    if source != "many_cues":
        train, held, own = G.load_windows(source, TRANSPORT, LADDER)
        return train, held, own, {}
    w = torch.load(_large("many_cues") / "windows.pt", weights_only=False)
    return w["train"], w["heldout"], int(w["mine"]["target"]), w["mine"]


def rows(task: NearTask) -> Rows:
    if task.kind == "e53":
        return Rows(load_experiment("realistic_tasks", "run2").e53_read(task.source))
    if task.kind == "case":
        WIN = load_experiment("share_untrained", "windows")
        spec = json.loads((EXPERIMENTS / "share_untrained" / "results" / "inventory.json").read_text())["tasks"][task.source]
        r = WIN.load_read(task.source, spec)
        t = r.toks[:, : r.ci_len]
        return Rows(RR.WindowRead(task.name, t, r.toks[:, -1], r.half, t, r.toks[:, -1]))
    train, held, own, mine = trigger_windows(task.source)
    target: int | Tensor
    if task.target == "own":
        target = own
    elif task.target == "table":
        target = MC.table({int(c["trigger"]): int(c["alt"]) for c in mine["cues"]}, int(mine["vocab"]))
    else:
        assert isinstance(task.target, int), task
        target = task.target
    assert train.half == held.half
    tg = lambda w: target_tokens(target, w.tokens, w.half).cpu()
    return Rows(RR.WindowRead(task.name, held.tokens, tg(held), held.half, train.tokens, tg(train)), train, held, target)


# ---- the retrain ----------------------------------------------------------------------------------------------------


def retrain(net: ST.Net, r: Rows, seed: int, lr: float, steps: int, log: Callable[[str], None]) -> tuple[ST.DW, list[float]]:
    """Arm A on the task's training windows toward its target (`regating.finetune`'s call, with a per-step read): the
    delta-W at `steps` (fp32, CPU) and the mean held-out log-odds of the target at steps 0..steps (each step's weights)."""
    assert r.train is not None and r.held is not None and r.target is not None
    rd = r.read

    def lo() -> float:
        return float(torch.cat([ST.logodds_at(net.run(rd.toks[s : s + 64])[0], rd.pos, rd.tgt[s : s + 64]) for s in range(0, len(rd.toks), 64)]).mean())

    curve = [lo()]
    cfg = FT.ArmAConfig(o_token=r.target, lr=lr, steps=steps, checkpoints=(steps,), seed=seed)
    res = FT.finetune_arm_a(net.model, r.train, r.held, cfg, net.device, log, lambda _s, _p: curve.append(lo()))
    return res.deltas[steps], curve


def steps_to_half(curve: Sequence[float]) -> int | None:
    """The first step whose gain over step 0 is at least half the last step's gain; None when that gain is not positive."""
    g = [x - curve[0] for x in curve]
    return next((s for s, x in enumerate(g) if x >= g[-1] / 2), None) if g[-1] > 0 else None


# ---- the twin rule --------------------------------------------------------------------------------------------------


def separates(real: Sequence[float], twin: Sequence[float], rand_real: Sequence[float], rand_twin: Sequence[float]) -> dict[str, Any]:
    """P2.74's descriptive twin rule, one reading on one pair, lists in seed order. Per seed s: d_s = real_s - twin_s and
    its null n_s = |rand_real_s - rand_twin_s|, the same difference between the two states' matched Gaussians. "real
    above" (or "real below") when every seed has |d_s| > n_s, strictly, all with one sign; "does not separate" when some
    seed does not, or the signs differ; "no reading" with no seed, unequal lists, or any value not finite."""
    n = len(real)
    if n == 0 or not len(twin) == len(rand_real) == len(rand_twin) == n:
        return {"call": "no reading", "n_seeds": n, "why": "no seed, or unequal lists"}
    if not all(math.isfinite(x) for x in (*real, *twin, *rand_real, *rand_twin)):
        return {"call": "no reading", "n_seeds": n, "why": "a value is not finite"}
    d = [a - b for a, b in zip(real, twin, strict=True)]
    null = [abs(a - b) for a, b in zip(rand_real, rand_twin, strict=True)]
    beyond = all(abs(x) > m for x, m in zip(d, null, strict=True))
    one_sign = all(x > 0 for x in d) or all(x < 0 for x in d)
    call = ("real above" if d[0] > 0 else "real below") if beyond and one_sign else "does not separate"
    return {"call": call, "n_seeds": n, "diff": d, "null": null}
