"""E121 Amendment 1: does the copy started from the fine-tune's least-squares split (R4) fail E34's adversarial check on
the narrow emoticon task, as E122's did on the broad lowercase task, while the standard copy (R3) passes?
Plan reports/plans/2026-11-06_plan_e121-copy-gap-ladder.md -> "Amendment 1"; filing results/predictions_amend1.json.

E34's check (harness.pile_ratios, teacher_training/run.py:123 via harness.measure): the PGD KL of one seed-0 adversary per
128-row block of Pile val rows 0-255, 20 steps, mean over the two blocks, over W0's in the same job; verdict by
faith_score.verdict with the L40 W0 floor. E121 saved each state's delta-W only, not its U', V', so each copy is regenerated
in fp32 by E121's recipe and stops unless it matches the saved file (1 - Frobenius cosine <= saved_tol). Order: W0, R3_o0
(control against E34 Amendment 8's L40 reading), the do-nothing state, R4_start, R4_o1, R4_o2, R3_o1, R3_o2. Each read is
written as it ends; a unit that might not end inside budget_s exits 3 (fleet/resubmit --code 3), and a later job rereads
W0's first block first. Writes results/copy_gap_ladder_amend1.json, results/large/amend1/<state>.pt (bf16 offsets).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import harness, teacher
from commute.arms import save_offsets
from commute.faithfulness import harness_pgd
from commute.jobs import hard_exit, load_experiment, require_filed

E121 = load_experiment("copy_gap_ladder", "run")  # ls_start, load_bf16, cos_table, its CFG, SAVED and LARGE
HERE, EXP = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
PRED, OUT = HERE / "results" / "predictions_amend1.json", HERE / "results" / "copy_gap_ladder_amend1.json"
SAVE, TRANSPORT = HERE / "results" / "large" / "amend1", EXP / "transport_test" / "results"
ORDER = ("R3_o0", "nothing", "R4_start", "R4_o1", "R4_o2", "R3_o1", "R3_o2")  # the control first, then the question's arm
SETS = {"R3": ("R3_o0", "R3_o1", "R3_o2"), "R4": ("R4_o1", "R4_o2")}
OUTCOMES = {("pass", "fail"): "closeness to the fine-tune and faithfulness pull apart on narrow updates too",
            ("pass", "pass"): "the trade-off is specific to broad updates (weak: the update itself does not fail here)",
            ("fail", "fail"): "the check fails emoticon copies on this card and recipe (contradicts E34's pass)",
            ("fail", "pass"): "reversed: the least-squares start gives the more faithful copy"}


@dataclass(frozen=True)
class Config:
    pgd_steps: int = 20  # teacher_training/run.py:62 pile_pgd_steps
    adv_seed: int = 0  # harness.pile_ratios: the ratio's adversary is seed 0
    pile_rows: int = 256  # teacher_training/run.py:61; faithfulness.pile_blocks: val rows 0-255, 2 blocks of 128, batch 8
    line: float = 1.5  # faith_score.LINE
    floor: float = 0.20877548446862537  # L40 W0 PGD floor, seeds 0-2: teacher_training_amend8c.json W0.pgd_floor
    p_min: float = 0.9  # teacher_training/run.py:64
    e39_lines: tuple[float, float] = (1.40, 1.61)  # E39/E122's fixed lines (H100 floor 0.075), reported beside
    w0_ref: float = 0.6917431652545929  # W0 seed-0 PGD KL: teacher_training_amend8c.json W0.pile.0.pgd_kl (L40)
    r3_o0_ref: float = 0.9866134443126633  # the copy at order 0: teacher_training_amend8_n3b.json reads.real.pgd (L40)
    nothing_ref: float = 1.3900221398869133  # T1000's do-nothing: teacher_training_amend8_n2.json reads.nothing.pgd (L40)
    repro_tol: float = 1e-3  # relative: W0 and R3_o0 against those (damage_crossed/rules.py repro_tol)
    rerun_tol: float = 1e-6  # relative: W0's first block reread by a later job (damage_crossed/rules.py rerun_tol)
    saved_tol: float = 1e-3  # 1 - Frobenius cosine, a regenerated state against E121's saved file (rules.py saved_tol)
    budget_s: float = 1680.0  # a unit starts only if it ends inside 28 min of the 30-min job
    read_s: float = 135.0  # one read: 118-127 s (damage_crossed.json .seconds)
    copy_s: float = 135.0  # one copy: 118-126 s with its behaviour reads (copy_gap_ladder.json behaviour.R*.seconds)

    def lines(self) -> tuple[float, float]:
        """E34's pass and fail lines on this card: faith_score.verdict, line / (1 + f) and line (1 + f)."""
        return self.line / (1 + self.floor), self.line * (1 + self.floor)


CFG = Config()


# ----------------------------------------------------------------------------- rules (pure), as filed


def verdict(r: float | None, p: float | None, lines: tuple[float, float]) -> str:
    """One state: both lines inclusive (faith_score.verdict); P(target) under p_min first (faith_score.state_verdict)."""
    if r is None or p is None:
        return "not run"
    if not (math.isfinite(r) and math.isfinite(p)):
        return "invalid"
    if p < CFG.p_min:
        return "below P 0.9"
    return "pass" if r <= lines[0] else "fail" if r >= lines[1] else "inconclusive"


def set_verdict(vs: list[str]) -> str:
    """A set's members all alike -> that verdict; a member not run or invalid -> that; else mixed."""
    for bad in ("not run", "invalid"):
        if bad in vs:
            return bad
    return vs[0] if len(set(vs)) == 1 else "mixed"


def score(ratio: dict[str, float], p: dict[str, float]) -> dict[str, Any]:
    """Per state and per set under E34's L40 lines (scored) and E39's fixed lines (reported); the outcome from the sets."""
    out: dict[str, Any] = {}
    for key, lines in (("e34", CFG.lines()), ("e39", CFG.e39_lines)):
        v = {n: verdict(ratio.get(n), p.get(n), lines) for n in (*ORDER[:3], *SETS["R3"][1:], *SETS["R4"])}
        s = {k: set_verdict([v[n] for n in ns]) for k, ns in SETS.items()}
        out[key] = {"lines": list(lines), "states": v, "sets": s,
                    "outcome": OUTCOMES.get((s["R3"], s["R4"]), f"no call: R3 {s['R3']}, R4 {s['R4']}")}
    return out


# ----------------------------------------------------------------------------- the job


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    log(f"STOP: {why}")
    hard_exit(1)


def kl_blocks(ctx: Any, st: tuple[dict[str, Tensor], Any] | None) -> list[float]:
    """The check's PGD KL per block (`harness_pgd`, one seed-0 adversary per block) at W0 (st None) or at state st."""
    if st is None:
        return [harness_pgd(ctx.model, b, CFG.pgd_steps, CFG.adv_seed) for b in ctx.blocks]
    with harness.at(ctx, *st):
        return [harness_pgd(ctx.model, b, CFG.pgd_steps, CFG.adv_seed) for b in ctx.blocks]


def make(ctx: Any, tnet: Any, name: str, ls: Any, out: dict[str, Any]) -> tuple[dict[str, Tensor], Any]:
    """A copy regenerated by E121's recipe (`run.train_rung`'s uv branch; R4 from `ls`, run.ls_start of A8c's delta-W),
    checked against E121's saved file, saved as bf16 offsets; returns (delta-W, alive (U', V'))."""
    c, uv = E121.CFG, ctx.uv_cpu
    if name == "R4_start":
        st = {k: (ls[1][k], U + ls[0][k][0], V + ls[0][k][1]) for k, (U, V) in uv.items()}
    else:
        rung, order = name.rsplit("_o", 1)
        st, _ = teacher.train_student(ctx.model, ctx.uv, ctx.W0, tnet, ctx.train.tokens, ctx.held, ctx.token, c.batch,
                                      ctx.train.half, c.copy_lr, c.steps, ctx.device, log, seed=int(order),
                                      init=ls[0] if rung == "R4" else None)
    dW = {k: v[0] for k, v in st.items()}
    frob = float(E121.cos_table({"x": dW, "saved": E121.load_bf16(E121.LARGE / f"{name}.pt")}, None, ctx.device)[1][0, 1])
    out["frobenius_to_saved"][name] = frob
    if not 1 - frob <= CFG.saved_tol:
        halt(out, f"{name} regenerated is not E121's saved state: Frobenius cosine {frob}")
    save_offsets(SAVE / f"{name}.pt", uv, st, {"e121": name})
    return dW, {k: (v[1], v[2]) for k, v in st.items()}


def unit(ctx: Any, tnet: Any, F0: dict[str, Tensor], name: str, ls: Any, out: dict[str, Any], t0: float) -> None:
    if name in out["kl"]:
        return
    est = CFG.read_s + (CFG.copy_s if name not in ("nothing", "R4_start") else 0)
    if time.time() - t0 + est > CFG.budget_s:
        log(f"[budget] {name} needs ~{est:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
        dump(out)
        hard_exit(3)
    t1 = time.time()
    st = (F0, None) if name == "nothing" else make(ctx, tnet, name, ls, out)
    out["kl"][name], out["seconds"][name] = kl_blocks(ctx, st), time.time() - t1
    r = sum(out["kl"][name]) / sum(out["kl"]["W0"])
    out["ratio"][name] = r
    ref = {"R3_o0": CFG.r3_o0_ref, "nothing": CFG.nothing_ref}.get(name)
    log(f"[{name}] blocks {out['kl'][name]} ratio {r:.6f} ({out['seconds'][name]:.0f} s)" + (f" E34 A8 {ref}" if ref else ""))
    dump(out)
    if name == "R3_o0" and not abs(r / CFG.r3_o0_ref - 1) <= CFG.repro_tol:
        halt(out, f"R3_o0 reads {r}, E34 Amendment 8's {CFG.r3_o0_ref}")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "kl": {}, "ratio": {}, "seconds": {}, "frobenius_to_saved": {}}
    assert "stopped" not in out and not out.get("done"), f"{OUT}: stopped or done; move it aside to start again"
    t0 = time.time()
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load("cuda", TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, E121.CFG.ls_tol,
                       CFG.pile_rows, CFG.pgd_steps, (), log)
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if "L40" not in out["jobs"][-1]["card"]:
        halt(out, f"the references are L40 readings; this job runs on {out['jobs'][-1]['card']}")
    uv0 = torch.load(E121.A8C / "uv0.pt", weights_only=False, map_location="cpu")
    assert all(torch.equal(ctx.uv_cpu[k][i], uv0[k][i]) for k in uv0 for i in (0, 1)), "alive U, V differ from E121's"
    if "W0" in out["kl"]:  # a later job: W0's first block must read as the first job's did
        v = harness_pgd(ctx.model, ctx.blocks[0], CFG.pgd_steps, CFG.adv_seed)
        if not abs(v / out["kl"]["W0"][0] - 1) <= CFG.rerun_tol:
            halt(out, f"W0's first block reads {v}, the first job {out['kl']['W0'][0]}")
    else:
        out["kl"]["W0"] = kl_blocks(ctx, None)
        w0 = sum(out["kl"]["W0"]) / 2
        log(f"[W0] blocks {out['kl']['W0']} mean {w0:.6f} ref {CFG.w0_ref}")
        dump(out)
        if not abs(w0 / CFG.w0_ref - 1) <= CFG.repro_tol:
            halt(out, f"W0 reads {w0}, E34 Amendment 8c's {CFG.w0_ref}")
    F0 = torch.load(E121.LARGE / "F0.pt", weights_only=False, map_location="cpu")
    tnet = teacher.teacher_net(ctx.model, F0)
    ls = E121.ls_start(ctx, E121.load_bf16(E121.SAVED["dW"])) if any(n.startswith("R4") and n not in out["kl"] for n in ORDER) else None
    for name in ORDER:
        unit(ctx, tnet, F0, name, ls, out, t0)
    beh = json.loads(E121.OUT.read_text())["behaviour"]
    p = {n: beh["F0" if n == "nothing" else n]["p_target"] for n in ORDER}
    out["p_target"], out["score"], out["done"] = p, score(out["ratio"], p), True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
