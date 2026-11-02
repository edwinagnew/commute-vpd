"""E12 Amendment 3: does rand32b's fall, and rand32's lack of one, survive reading Gamma and the masked logits in fp32, on both the fine-tune and the student?
Plan reports/plans/2026-11-02_plan_e12-amend3-fp32-read.md; rules, lines, expected values: results/predictions_amend3.json (the run refuses without it).
Read only, no training: Amendment 2's saved students, which are bf16 offsets, and E46's fp32 fine-tunes. A stop is `.stopped` in the file, exit 1.
cluster/new_vs_existing_amend3.sbatch.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import amend1
import amend2 as A2
import torch
from amend1 import E46, Job, check, dump, log, near
from torch import Tensor

from commute import copying as CP
from commute import saved_states
from commute.faithfulness import model_state
from commute.keydep import sha256

RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT = RESULTS / "predictions_amend3.json", RESULTS / "amend3.json"
CM = SimpleNamespace(**json.loads(PRED.read_text())["config"])  # what this file adds; the lines and tolerances stay Amendment 2's (A2.CM)
amend1.OUT = OUT  # after importing amend2, which points it at Amendment 2's scored file: check() and dump() write to this global
FALLING = [f"{A2.CM.falling}_{s}{o}" for o in ("", f"_o{A2.CM.second_order}") for s in A2.CM.seeds]
REFERENCE = [f"{A2.CM.reference}_{s}" for s in A2.CM.seeds]


# ----------------------------------------------------------------------------- rules (pure), as filed


def tol_of(floor: float) -> float:
    """How far a fall may move between the precisions: a fall is two reads, each off by the floor at most, plus the absolute guard."""
    return CM.floor_factor * floor + CM.guard


def p1(falls: list[float], floor: float) -> str:
    """rand32b's fp32 falls against Amendment 2's lines. A floor of 0 means the two paths are one path."""
    if len(falls) < CM.n_falling or not A2.finite(*falls, floor) or floor <= 0:
        return "unscorable"
    if tol_of(floor) >= A2.CM.fall_line:
        return "no power"
    if all(f >= A2.CM.fall_line for f in falls):
        return "holds"
    if all(abs(f) <= A2.CM.gone_line for f in falls):
        return "gone" if tol_of(floor) < A2.CM.gone_line else "no power"  # "gone" overturns a report: it needs a floor that can see the 0.15 line
    return "mixed"


def p2(falls: list[float], floor: float) -> str:
    """rand32's fp32 falls against the same lines."""
    if len(falls) < CM.n_reference or not A2.finite(*falls, floor) or floor <= 0:
        return "unscorable"
    if tol_of(floor) >= A2.CM.gone_line:
        return "no power"
    if all(abs(f) <= A2.CM.gone_line for f in falls):
        return "holds"
    return "falls" if all(f >= A2.CM.fall_line for f in falls) else "mixed"


def moved(diffs: dict[str, float], floor: float) -> list[str]:
    """States whose fall moved beyond the tolerance between the precisions; a tie has not; a nan has."""
    return [n for n, d in diffs.items() if not abs(d) <= tol_of(floor)]


def score(out: dict[str, Any]) -> dict[str, Any]:
    st, ft, floor = out["students"], out["fine_tunes"], out["floor"]

    def part(names: list[str], verdict: str) -> dict[str, Any]:
        diffs = {n: st[n]["pair_diff"] for n in names}
        fts = sorted({st[n]["fine_tune"] for n in names})
        term_diffs = {n: ft[n]["term_fp32"] - ft[n]["term_bf16"] for n in fts}  # the fine-tune's re-switching term, which a fall cancels
        return {"verdict": verdict, "moved": moved(diffs, floor), "pair_diff": diffs, "term_moved": moved(term_diffs, floor), "term_diff": term_diffs,
                **{q: {n: ft[n][q] for n in fts} for q in ("term_bf16", "term_fp32", "L01_diff")},
                **{q: {n: st[n][q] for n in names} for q in ("fall_fp32", "fall_bf16", "term_lost_fp32", "term_lost_bf16")}}

    falls = lambda names: [st[n]["fall_fp32"] for n in names]
    return {"tol": tol_of(floor), "P1": part(FALLING, p1(falls(FALLING), floor)), "P2": part(REFERENCE, p2(falls(REFERENCE), floor))}


# ----------------------------------------------------------------------------- reads


def read_bf16(job: Job, w: dict[str, Any], dW: dict[str, Tensor]) -> tuple[float, list[dict[str, Tensor]]]:
    """Amendment 2's L01 and its gates: Gamma on target W0 + dW and the masked forward, both under bf16 autocast."""
    gates = A2.gates_at(job, w, dW)
    return A2.masked(job, w, gates), gates


def read_fp32(job: Job, w: dict[str, Any], dW: dict[str, Tensor]) -> tuple[float, list[dict[str, Tensor]]]:
    """The same read with no autocast on either: same windows, batches, trigger and table."""
    tokens, half = w["heldout"].tokens, w["heldout"].half
    with model_state(job.model, target={k: job.W0[k] + dW[k].to(job.W0[k].device) for k in job.W0}):
        gates = CP.gates_fp32(job.model, tokens, CM.batch, tokens.shape[1])
    return float(CP.masked_lo_fp32(job.model, tokens, half, w["table"], gates, CM.batch, tokens.shape[1]).mean()), gates


def read_both(job: Job, w: dict[str, Any], dW: dict[str, Tensor]) -> tuple[float, float, dict[str, float]]:
    """L01 in bf16, in fp32, and how far the two precisions' gates are apart at the trigger: max and mean over alive subcomponents of the
    window mean |g_fp32 - g_bf16|. Shows that Gamma itself ran in fp32 and by how much it moved."""
    (bf16, g16), (fp32, g32) = read_bf16(job, w, dW), read_fp32(job, w, dW)
    d = A2.abs_change(g16, g32, w["heldout"].half)
    gap = torch.cat([d[k][job.alive[k].cpu()] for k in job.alive])
    return bf16, fp32, {"gate_gap_max": float(gap.max()), "gate_gap_mean": float(gap.mean())}


def w0_floor(job: Job, out: dict[str, Any], a2: dict[str, Any]) -> None:
    """Stage 0, alone: the do-nothing read at W0 under both precisions. Its bf16 side must be Amendment 2's; their gap is the floor."""
    zero = {k: torch.zeros_like(v) for k, v in job.W0.items()}
    for t in (A2.CM.falling, A2.CM.reference):
        w = job.task(t)
        bf16, fp32, gap = read_both(job, w, zero)
        out["w0"][t] = {"L00_bf16": bf16, "L00_fp32": fp32, "diff": fp32 - bf16, **gap}
        log(f"[W0 {t}] {json.dumps(out['w0'][t])}")
        near(out, f"W0 {t} L00, bf16, against Amendment 2's", bf16, a2["fine_tunes"][f"{t}_{A2.CM.seeds[0]}"]["L00"], A2.CM.read_tol)
        check(out, f"W0 {t} gates: fp32 differs from bf16 (max gap above 0)", gap["gate_gap_max"], gap["gate_gap_max"] > 0)
    diffs = [r["diff"] for r in out["w0"].values()]
    out["floor"] = max(abs(d) for d in diffs) if all(math.isfinite(d) for d in diffs) else math.nan  # max() drops a nan unless it is first
    log(f"[floor] {out['floor']:.8g}, tolerance on a fall {tol_of(out['floor']):.8g}")
    check(out, "W0 floor: the two paths differ (above 0)", out["floor"], out["floor"] > 0)


def fine_tunes(job: Job, out: dict[str, Any], a2: dict[str, Any]) -> None:
    """Stage 1: each fine-tune under both; bf16 against Amendment 2's stored L01. E46's files are the fp32 dW Amendment 2 read."""
    for n, stored in a2["fine_tunes"].items():
        t = n.split("_")[0]
        dW = torch.load(E46 / "large" / f"{n}.pt", weights_only=False, map_location="cpu")
        bf16, fp32, gap = read_both(job, job.task(t), dW)
        base = out["w0"][t]
        out["fine_tunes"][n] = {"task": t, "L01_bf16": bf16, "L01_fp32": fp32, "L01_diff": fp32 - bf16, **gap,
                                "term_bf16": bf16 - base["L00_bf16"], "term_fp32": fp32 - base["L00_fp32"]}
        log(f"[fine-tune {n}] {json.dumps(out['fine_tunes'][n])}")
        near(out, f"{n} L01, bf16, against Amendment 2's", bf16, stored["L01"], A2.CM.read_tol)


def student(job: Job, out: dict[str, Any], a2: dict[str, Any], name: str) -> None:
    """Stage 2, one state: rebuilt from its bf16 offsets, read under both and written. The stop on `reload_diff` comes in main, after all six."""
    rec, w = a2["students"][name], job.task(a2["students"][name]["task"])
    ft, base = out["fine_tunes"][rec["fine_tune"]], out["w0"][rec["task"]]
    dW, _ = saved_states.offsets_state(A2.STATES / f"{name}.pt", job.uv_cpu)
    bf16, fp32, gap = read_both(job, w, dW)
    fall_16, fall_32 = ft["L01_bf16"] - bf16, ft["L01_fp32"] - fp32
    out["students"][name] = {"task": rec["task"], "fine_tune": rec["fine_tune"], "L01_bf16": bf16, "L01_fp32": fp32, "reload_diff": bf16 - rec["L01"], **gap,
                             "fall_bf16": fall_16, "fall_fp32": fall_32, "pair_diff": fall_32 - fall_16,
                             "term_lost_bf16": fall_16 / (ft["L01_bf16"] - base["L00_bf16"]), "term_lost_fp32": fall_32 / (ft["L01_fp32"] - base["L00_fp32"])}
    log(f"[student {name}] {json.dumps(out['students'][name])}")
    dump(out)


def main() -> None:
    if OUT.is_file():
        sys.exit(f"{OUT} exists. Move it aside to start again")
    a2 = json.loads(A2.OUT.read_text())
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    head = subprocess.run(["git", "-C", str(RESULTS.parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out: dict[str, Any] = {"card": card, "commit": head, "config": vars(CM), "predictions_sha256": sha256(PRED), "amend2_sha256": sha256(A2.OUT),
                           "w0": {}, "fine_tunes": {}, "students": {}}
    check(out, "same card as Amendment 2 (1 = yes)", float(card == a2["card"]), card == a2["card"])
    log(f"[start] {card} {head}")
    job = Job("cuda" if torch.cuda.is_available() else "cpu", json.loads((E46 / "stage0a.json").read_text())["groups"], out)
    w0_floor(job, out, a2)
    fine_tunes(job, out, a2)
    for name in [n for pair in zip(FALLING[:2], REFERENCE, strict=True) for n in pair] + FALLING[2:]:  # Amendment 2's order: each seed's pair, then the second batch orders
        student(job, out, a2, name)
    worst = max(abs(d) if math.isfinite(d) else math.inf for d in (s["reload_diff"] for s in out["students"].values()))
    check(out, "worst student L01, bf16 from its saved offsets, against Amendment 2's (|difference|)", worst, worst <= A2.CM.repro_tol)
    out["score"], out["done"] = score(out), True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    main()
