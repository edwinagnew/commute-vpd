"""E127 Amendment 1: does holding every alive subcomponent's mask at 1 remove only the placed update's excess, or does it
take the check's teeth away? Plan reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md -> "Amendment 1"; filing
results/predictions_amend1.json (the run refuses unless it is committed unchanged and its `config` equals CFG).

Same rows, seed, card (L40) and code path as run.py (harness_pgd on Pile val rows 0-255, one seed-0 adversary per 128-row
block, 20 steps; a held read divides by W0 read under the same hold). Per task, L first, then E:
  nothing|hold        do-nothing (target W0 + dW, components at W0) with the alive masks held: the positive control
  std_copy, |hold     the standard teacher-trained copy, free (reproduction control) and held
  ls_copy, |hold      the copy trained from the least-squares start, free (reproduction control) and held
Copies are read from their saved bf16 offsets (saved_states.offsets_state). L's stored readings were read that way
(E122); E's were read from fp32 states in memory (E121 Amendment 1), so E's control band is bf16_tol, not repro_tol.
Units run in ORDER, each written as it ends (`STAGE n DONE`); one that might not end inside budget_s exits 3
(fleet/resubmit --code 3), and a later job rereads W0's first block first. Writes results/ls_start_faith_amend1.json.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import harness
from commute.faithfulness import harness_pgd
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.saved_states import offsets_state

E127 = load_experiment("ls_start_faith", "run")  # States (do-nothing as E127 read it), COPIES, E121, CFG, TRANSPORT
HERE = Path(__file__).resolve().parent
PRED, OUT = HERE / "results" / "predictions_amend1.json", HERE / "results" / "ls_start_faith_amend1.json"
E127_OUT = HERE / "results" / "ls_start_faith.json"
TASKS = ("L", "E")
COPIES = E127.COPIES  # L: damage_crossed tt_recipe, tt_c3; E: copy_gap_ladder amend1 R3_o0, R4_o1
ORDER = ("W0", "W0|hold", *(f"{t}.{s}" for t in TASKS
                            for s in ("nothing|hold", "std_copy", "std_copy|hold", "ls_copy", "ls_copy|hold")))


@dataclass(frozen=True)
class Config:
    pgd_steps: int = 20  # as run.py (E127.CFG)
    adv_seed: int = 0
    pile_rows: int = 256
    ls_tol: float = 1e-6
    floor: float = 0.20877548446862537  # f: L40 W0 PGD floor, seeds 0-2 (run.py CFG.floor)
    teeth_share: float = 0.5  # R1: nothing|hold must keep at least half of do-nothing's free excess (and at least f)
    w0_ref: float = 0.6917431652545929  # W0 mean block KL (ls_start_faith.json kl.W0)
    w0_hold_ref: float = 1.6008625626564026  # W0|hold mean block KL (ls_start_faith.json kl["W0|hold"])
    nothing_ref: tuple[tuple[str, float], ...] = (("L", 4.79423381511107), ("E", 1.3900221398869133))  # E127 free
    copy_refs: tuple[tuple[str, float], ...] = (
        ("L.std_copy", 1.9128751139166524),  # damage_crossed.json score.ratios["tt_recipe|A"]
        ("L.ls_copy", 9.40489471199686),  # damage_crossed.json score.ratios["tt_c3|A"]
        ("E.std_copy", 0.9866134443126633),  # copy_gap_ladder_amend1.json ratio.R3_o0 (fp32 in memory)
        ("E.ls_copy", 1.568115684905064))  # copy_gap_ladder_amend1.json ratio.R4_o1 (fp32 in memory)
    repro_tol: float = 1e-3  # relative: W0, W0|hold, L's copies (same file, same path as stored)
    bf16_tol: float = 1e-2  # relative: E's copies, stored from fp32 states, read here from their bf16 offsets
    rerun_tol: float = 1e-6  # relative: W0's first block reread by a later job
    budget_s: float = 1680.0  # a unit starts only if it ends inside 28 min of the 30-min job
    pgd_s: float = 145.0  # one PGD read: 127-140 s (ls_start_faith.json .seconds)


CFG = Config()


# ----------------------------------------------------------------------------- rules (pure), as filed


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def ratios(kl: dict[str, list[float]], t: str) -> dict[str, float]:
    """Each read over W0's summed block KL (W0|hold's for a held read), same job; a non-positive reference gives nan."""
    out = {}
    for s in ("nothing|hold", "std_copy", "std_copy|hold", "ls_copy", "ls_copy|hold"):
        ref = "W0|hold" if s.endswith("|hold") else "W0"
        if f"{t}.{s}" in kl and ref in kl:
            d = sum(kl[ref])
            out[s] = sum(kl[f"{t}.{s}"]) / d if d > 0 else math.nan
    return out


def teeth(nh: float | None, n: float) -> str:
    """R1: the hold has teeth iff nothing|hold - 1 >= max(f, teeth_share (nothing - 1)). A held excess below 0 fails;
    nothing <= 1 leaves the bar at f."""
    if nh is None:
        return "not run"
    if not finite(nh, n):
        return "invalid"
    return "teeth" if nh - 1 >= max(CFG.floor, CFG.teeth_share * (n - 1)) else "no teeth"


def reach(c: float | None, ch: float | None) -> str:
    """R2: a copy is out of the adversary's reach iff |c - c|hold| <= f (inclusive), exposed iff c|hold < c - f, else
    the hold raises it (c|hold > c + f)."""
    if c is None or ch is None:
        return "not run"
    if not finite(c, ch):
        return "invalid"
    return "out of reach" if abs(c - ch) <= CFG.floor else "exposed" if ch < c - CFG.floor else "raised by the hold"


def score(kl: dict[str, list[float]], e127: dict[str, Any]) -> dict[str, Any]:
    res = {}
    for t in TASKS:
        r, n = ratios(kl, t), dict(CFG.nothing_ref)[t]
        th = teeth(r.get("nothing|hold"), n)
        exp = e127[t]["exposure"]
        place = (f"as E127 read it: {e127[t]['source']}, {exp[0]}" if th == "teeth" else
                 "unreadable under the hold" if th == "no teeth" else th)
        res[t] = {"ratio": r, "R1 teeth": th, "placement": place, "R2 std_copy": reach(r.get("std_copy"), r.get("std_copy|hold")),
                  "ls_copy (descriptive)": reach(r.get("ls_copy"), r.get("ls_copy|hold"))}
    return res


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


def state(ctx: Any, st: Any, name: str) -> tuple[dict[str, Tensor], Any] | None:
    """(target change, alive (U', V') or None) of a read; None at W0."""
    if name.startswith("W0"):
        return None
    t, s = name.split("|")[0].split(".")
    return st.get(f"{t}.nothing") if s == "nothing" else offsets_state(COPIES[t][s], ctx.uv_cpu)


def read(ctx: Any, st: Any, name: str, hold: dict[str, Tensor]) -> list[float]:
    h, s = hold if name.endswith("|hold") else None, state(ctx, st, name)
    with harness.at(ctx, *s) if s else contextlib.nullcontext():
        return [harness_pgd(ctx.model, b, CFG.pgd_steps, CFG.adv_seed, hold=h) for b in ctx.blocks]


def control(out: dict[str, Any], name: str) -> None:
    """Stop unless W0, W0|hold and each copy's free reading match their stored values."""
    kl, refs = out["kl"], dict(CFG.copy_refs)
    if name.startswith("W0"):
        got, want, tol = sum(kl[name]) / 2, CFG.w0_ref if name == "W0" else CFG.w0_hold_ref, CFG.repro_tol
    elif name in refs:
        got, want = sum(kl[name]) / sum(kl["W0"]), refs[name]
        tol = CFG.repro_tol if name.startswith("L.") else CFG.bf16_tol
    else:
        return
    out["controls"][name] = {"read": got, "ref": want, "tol": tol}
    if not abs(got / want - 1) <= tol:
        halt(out, f"{name} reads {got}, its reference {want} (tolerance {tol} relative)")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    assert CFG.pgd_steps == E127.CFG.pgd_steps and CFG.adv_seed == E127.CFG.adv_seed and CFG.pile_rows == E127.CFG.pile_rows
    e127 = json.loads(E127_OUT.read_text())["score"]
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "kl": {}, "seconds": {}, "controls": {}}
    assert "stopped" not in out and not out.get("done"), f"{OUT}: stopped or done; move it aside to start again"
    t0 = time.time()
    token = json.loads((E127.TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load("cuda", E127.TRANSPORT / "large" / "alive.pt", E127.TRANSPORT / "large" / "windows.pt", token,
                       CFG.ls_tol, CFG.pile_rows, CFG.pgd_steps, (), log)
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if "L40" not in out["jobs"][-1]["card"]:
        halt(out, f"the references are L40 readings; this job runs on {out['jobs'][-1]['card']}")
    uv0 = torch.load(E127.E121.A8C / "uv0.pt", weights_only=False, map_location="cpu")
    assert all(torch.equal(ctx.uv_cpu[k][i], uv0[k][i]) for k in uv0 for i in (0, 1)), "alive U, V differ from E121's"
    hold = {k: a.to(ctx.device) for k, a in ctx.alive.items()}
    if "W0" in out["kl"]:  # a later job: W0's first block must read as the first job's did
        v = harness_pgd(ctx.model, ctx.blocks[0], CFG.pgd_steps, CFG.adv_seed)
        if not abs(v / out["kl"]["W0"][0] - 1) <= CFG.rerun_tol:
            halt(out, f"W0's first block reads {v}, the first job {out['kl']['W0'][0]}")
    st = E127.States(ctx)
    for i, name in enumerate(ORDER):
        if name in out["kl"]:
            continue
        if time.time() - t0 + CFG.pgd_s > CFG.budget_s:
            log(f"[budget] {name} needs ~{CFG.pgd_s:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
            dump(out)
            hard_exit(3)
        t1 = time.time()
        out["kl"][name] = read(ctx, st, name, hold)
        out["seconds"][name] = time.time() - t1
        out["score"] = score(out["kl"], e127)
        dump(out)
        log(f"[{name}] blocks {out['kl'][name]} ({out['seconds'][name]:.0f} s)")
        control(out, name)
        dump(out)
        log(f"STAGE {i} DONE {name}")
    out["done"] = True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
