"""E127 Amendment 2 (the last): is the broad Adam lowercase copy's failure the placement multiplier? Each copy's own
change S_c = U'V'^T - UV^T read twice under the free check: as the copy (alive subcomponents at U', V') and placed in
Delta on top of W0 (target W0 + S_c, subcomponents left at the released U, V: harness.at(ctx, S_c, None), the form of
run.py's nothing_start). Plan reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md -> "Amendment 2"; filing
results/predictions_amend2.json (the run refuses unless it is committed unchanged and its `config` equals CFG).

Lowercase only, free adversary only (no hold); same rows, seed, card (L40) and code path as run.py and amend1.py
(harness_pgd on Pile val rows 0-255, one seed-0 adversary per 128-row block, 20 steps), each ratio over W0 in this job:
  W0                 control, mean block KL within 1e-3 of E127's
  <copy>             the copy, free: control, within 1e-3 of its stored ratio (same bf16 offsets, same path)
  <copy>|delta       the copy's change S_c in Delta
Copies: std_copy damage_crossed tt_recipe.pt, ls_copy tt_c3.pt (E127's COPIES["L"]), lora_copy broad_fragility
a2_tt_lora_s0.pt (E39's copy of the rank-8 LoRA fine-tune, as E122 read it). At full masks the copy and its in-Delta
state are the same weights W0 + S_c by construction (one dW object for both reads); only where S_c sits differs.
Units run in ORDER, controls first, each written as it ends (`STAGE n DONE`); one that might not end inside budget_s
exits 3 (fleet/resubmit --code 3), and a later job rereads W0's first block first. Writes results/ls_start_faith_amend2.json.
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

E127 = load_experiment("ls_start_faith", "run")  # COPIES, E121, TRANSPORT, EXP, CFG
HERE = Path(__file__).resolve().parent
PRED, OUT = HERE / "results" / "predictions_amend2.json", HERE / "results" / "ls_start_faith_amend2.json"
COPIES = {**E127.COPIES["L"], "lora_copy": E127.EXP / "broad_fragility" / "results" / "large" / "a2_tt_lora_s0.pt"}
ORDER = ("W0", *(f"L.{c}" for c in COPIES), *(f"L.{c}|delta" for c in COPIES))


@dataclass(frozen=True)
class Config:
    pgd_steps: int = 20  # as run.py (E127.CFG)
    adv_seed: int = 0
    pile_rows: int = 256
    ls_tol: float = 1e-6
    floor: float = 0.20877548446862537  # f: L40 W0 PGD floor, seeds 0-2 (run.py CFG.floor)
    fail_line: float = 1.61  # damage_crossed/rules.py fail_line (E122, from E39)
    w0_ref: float = 0.6917431652545929  # W0 mean block KL (ls_start_faith.json kl.W0)
    copy_refs: tuple[tuple[str, float], ...] = (
        ("L.std_copy", 1.9128751139166524),  # damage_crossed.json score.ratios["tt_recipe|A"]
        ("L.ls_copy", 9.40489471199686),  # damage_crossed.json score.ratios["tt_c3|A"]
        ("L.lora_copy", 1.3262876227610394))  # damage_crossed.json score.ratios["tt_lora|A"]
    e127_m: tuple[tuple[str, float], ...] = (("L", 3.7676381991462096), ("E", 3.8481320337024862))  # (start-1)/(nothing_start-1)
    repro_tol: float = 1e-3  # relative, as amend1.py for L's copies
    rerun_tol: float = 1e-6  # relative: W0's first block reread by a later job
    budget_s: float = 1680.0  # a unit starts only if it ends inside 28 min of the 30-min job
    pgd_s: float = 145.0  # one PGD read: 127-132 s (ls_start_faith_amend1.json .seconds)


CFG = Config()


# ----------------------------------------------------------------------------- rules (pure), as filed


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def ratios(kl: dict[str, list[float]]) -> dict[str, float]:
    """Each read's summed block KL over W0's in this job; a non-positive W0 gives nan."""
    if "W0" not in kl:
        return {}
    d = sum(kl["W0"])
    return {k.removeprefix("L."): (sum(v) / d if d > 0 else math.nan) for k, v in kl.items() if k != "W0"}


def r1(free: float | None, ind: float | None) -> str:
    """R1, the standard copy: against the fail line L with band f. Free must fail (free > L + f), else 'does not
    apply'; then in Delta < L - f 'placement explains the broad failure', > L + f 'function fails too', else (edges
    inclusive) 'within the line'."""
    if free is None or ind is None:
        return "not run"
    if not finite(free, ind):
        return "invalid"
    line, f = CFG.fail_line, CFG.floor
    if not free > line + f:
        return "does not apply"
    return "placement explains the broad failure" if ind < line - f else "function fails too" if ind > line + f else "within the line"


def multiplier(free: float | None, ind: float | None) -> tuple[str, float | None]:
    """R2, descriptive: m = (free - 1) / (in Delta - 1), excess over this job's W0 in ratio units (equal to E127's nats
    multiplier, as both reads share one W0). Readable only if both excesses are at least f (inclusive)."""
    if free is None or ind is None:
        return "not run", None
    if not finite(free, ind):
        return "invalid", None
    if free - 1 < CFG.floor:
        return "free excess within the floor", None
    if ind - 1 < CFG.floor:
        return "in-Delta excess within the floor", None
    return "readable", (free - 1) / (ind - 1)


def score(kl: dict[str, list[float]]) -> dict[str, Any]:
    r = ratios(kl)
    res: dict[str, Any] = {"ratio": r, "R1 std_copy": r1(r.get("std_copy"), r.get("std_copy|delta"))}
    m_ref = dict(CFG.e127_m)["L"]
    for c in COPIES:
        label, m = multiplier(r.get(c), r.get(f"{c}|delta"))
        res[f"R2 {c}"] = {"reading": label, "m": m, "m over E127's L": None if m is None else m / m_ref}
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


def state(ctx: Any, cache: dict[str, Any], name: str) -> tuple[dict[str, Tensor], Any] | None:
    """(target change, alive (U', V') or None) of a read; None at W0. '|delta' keeps the copy's S_c, drops its U', V'."""
    if name == "W0":
        return None
    c = name.split("|")[0].removeprefix("L.")
    if c not in cache:
        cache[c] = offsets_state(COPIES[c], ctx.uv_cpu)  # S_c = U'V'^T - UV^T (float64, stored float32), (U', V')
    S, uv2 = cache[c]
    return (S, None) if name.endswith("|delta") else (S, uv2)


def read(ctx: Any, cache: dict[str, Any], name: str) -> list[float]:
    s = state(ctx, cache, name)
    with harness.at(ctx, *s) if s else contextlib.nullcontext():
        return [harness_pgd(ctx.model, b, CFG.pgd_steps, CFG.adv_seed) for b in ctx.blocks]


def control(out: dict[str, Any], name: str) -> None:
    """Stop unless W0 and each copy's free reading match their stored values (relative repro_tol)."""
    kl, refs = out["kl"], dict(CFG.copy_refs)
    if name == "W0":
        got, want = sum(kl[name]) / 2, CFG.w0_ref
    elif name in refs:
        got, want = sum(kl[name]) / sum(kl["W0"]), refs[name]
    else:
        return
    out["controls"][name] = {"read": got, "ref": want, "tol": CFG.repro_tol}
    if not abs(got / want - 1) <= CFG.repro_tol:
        halt(out, f"{name} reads {got}, its reference {want} (tolerance {CFG.repro_tol} relative)")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    assert CFG.pgd_steps == E127.CFG.pgd_steps and CFG.adv_seed == E127.CFG.adv_seed and CFG.pile_rows == E127.CFG.pile_rows
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
    if "W0" in out["kl"]:  # a later job: W0's first block must read as the first job's did
        v = harness_pgd(ctx.model, ctx.blocks[0], CFG.pgd_steps, CFG.adv_seed)
        if not abs(v / out["kl"]["W0"][0] - 1) <= CFG.rerun_tol:
            halt(out, f"W0's first block reads {v}, the first job {out['kl']['W0'][0]}")
    cache: dict[str, Any] = {}
    for i, name in enumerate(ORDER):
        if name in out["kl"]:
            continue
        if time.time() - t0 + CFG.pgd_s > CFG.budget_s:
            log(f"[budget] {name} needs ~{CFG.pgd_s:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
            dump(out)
            hard_exit(3)
        t1 = time.time()
        out["kl"][name] = read(ctx, cache, name)
        out["seconds"][name] = time.time() - t1
        out["score"] = score(out["kl"])
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
