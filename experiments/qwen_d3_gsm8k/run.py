"""E104 (qwen_d3_gsm8k), D3: E94's three carried rows (outside share, placement excess, think overlap) on the GSM8K arms
at steps 10 and 100, with three scaled copies. Plan: reports/plans/2026-11-06_plan_e104-d3-gsm8k.md and its appendix;
rules: results/predictions.json (the run refuses while it carries a "status" key, is uncommitted, or results/stopped.json
exists); numbers and comparisons: rules.py; shared reads: `commute.qwen_d3`.

  --stage control      stage 0, alone and first: W0 as the fine-tune, E94's nulls by hash, the template's step 100 through
                       this path against E94's numbers, and the template at half scale (outside and placement unchanged)
  --states a,b,...     names from rules.STATES and rules.COPIES, in the order given; refuses before stage 0 has passed,
                       and a copy whose source or target is neither written nor earlier in the list; written states skip

Every pass logs its numbers to results/run.log; each state writes results/<state>/{maths,state}.json and
results/large/<state>/counts_pe.safetensors as it ends, then rescores results/table.json. A missed check writes
results/stopped.json and exits 1. Exit 3: the time left is under the longest state so far: resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
from safetensors.torch import save_file

from commute import qwen_d3 as d3
from commute import qwen_reading as qr
from commute.jobs import hard_exit, require_filed
from commute.qwen_sft import load, sha256

RESULTS, LARGE = rules.RESULTS, rules.RESULTS / "large"
PRED, STOPPED, CONTROL = (RESULTS / f for f in ("predictions.json", "stopped.json", "control.json"))
UNIT, E94_NULLS, E94_CONTROL = rules.E94 / "large" / "unit_nulls.safetensors", rules.E94 / "nulls.json", rules.E94 / "control.json"
T0 = time.time()


@dataclass(frozen=True)
class Config:
    n16: int = 16  # E94's size search rows
    null_seeds: tuple[int, ...] = (0, 1, 2, 3)  # E94's unit nulls
    draws: int = 1000  # E57's overlap draws
    match: float = 1.25  # the plan's band on a scaled copy's size
    evals: int = 6
    limit_s: float = 2 * 3600 - 180  # debug QoS, less a margin for the last write
    first_guess_s: float = 20 * 60  # the plan's worst state before one is measured

    def __post_init__(self) -> None:
        assert self.n16 <= d3.MATHS_ROWS and self.limit_s > self.first_guess_s


C = Config()


def log(msg: str) -> None:
    line = f"[{os.environ.get('SLURM_JOB_ID', '-')} {time.time() - T0:8.1f}s] {msg}"  # two jobs may share the log
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    tmp.replace(path)


def check(name: str, holds: bool, detail: dict[str, Any]) -> dict[str, Any]:
    log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        write(STOPPED, {"check": name, "detail": detail, "at_s": time.time() - T0})
        log(f"STOP {name}")
        hard_exit(1)
    return detail | {"holds": True}


def rel(x: float, ref: float) -> float:
    return abs(x / ref - 1) if ref != 0 and math.isfinite(x) else math.inf


@dataclass
class Job:
    d: d3.Decomp
    m: d3.RowSet  # the 128 maths rows
    unit: dict[str, torch.Tensor]  # E94's unit Gaussian nulls' piece energies
    unit_energy: dict[str, dict[str, float]]
    k0: list[float]  # W0's K0 per maths row (E94's control)
    P: dict[str, Any]


def cells(job: Job, tag: str, rs: d3.RowSet | None = None, **kw: Any) -> d3.Sums:
    rs, t0 = rs or job.m, time.time()
    s = d3.cells(job.d, rs, **kw)
    n = len(rs.tokens) * d3.ROW_LEN
    log(f"[cells {tag}] " + " ".join(f"{k} {d3.total(s, k) / n:.9g}" for k in s) + f"; rows {len(rs.tokens)}, "
        f"{time.time() - t0:.0f} s, {d3.peaks()}")
    return s


def load_checked(name: str, rec: dict[str, Any], P: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dW = d3.load_state(rec)
    nrm = d3.norm(dW)
    check(f"{name}_norm", rel(nrm, rec["norm"]) <= P["first_minutes"]["norm_rel"], {"norm": nrm, "e91": rec["norm"]})
    return dW, {"file": rec["file"], "norm": nrm, "sha256": sha256(rec["path"])}


def read(job: Job, name: str, dW: dict[str, torch.Tensor], e93: float | None = None
         ) -> tuple[dict[str, Any], d3.Sums, dict[str, torch.Tensor]]:
    """Base cells first (the first-minutes checks), then the split and the carried rows; the record, the base sums and
    the tensors to save."""
    d, F = job.d, job.P["first_minutes"]
    d.weights.put("FT", dW)
    c0, c1 = d.counts(), d.counts()
    base = cells(job, f"{name} base, maths", c0=c0, c1=c1)
    k0 = [r[d3.POOLED] for r in base["K0"]]
    check(f"{name}_K0_is_W0s", max(rel(a, b) for a, b in zip(k0, job.k0, strict=True)) <= F["K0_rel"], {"rows": len(k0)})
    K00 = d3.total(base, "K00") / (len(k0) * d3.ROW_LEN)
    log(f"[first minutes] {name} K00 {K00:.9g}, E93's {e93} (a stop at step 100 only)")
    if e93 is not None:
        check(f"{name}_K00_is_E93s", rel(K00, e93) <= F["e93_K00_rel"], {"K00": K00, "e93": e93})
    rec, large = d3.carried(d, dW, base, c0, c1, job.unit, job.unit_energy, C.null_seeds, C.draws)
    log(f"[split {name}] " + json.dumps({k: rec[k] for k in ("outside", "outside_null", "stored_error", "split_s")})
        + f"; placement {json.dumps(rec['placement'])}; |S| {rec['overlap']['size_S']}, "
        + " ".join(f"{k} fold {rec['overlap'][k]['fold']} (o {rec['overlap'][k]['o']}, chance {rec['overlap'][k]['mean']}, "
                   f"draw sd {rec['overlap'][k]['draw_sd']})" for k in ("T", "L")))
    check(f"{name}_stored_error", rec["stored_error"] <= F["stored_error_max"], {"stored_error": rec["stored_error"]})
    gpu, host = torch.cuda.max_memory_allocated() / d3.GIB, d3.host_gb()
    check(f"{name}_memory", gpu <= F["gpu_max_gib"] and host <= F["host_max_gb"], {"gpu_gib": gpu, "host_gb": host})
    d.weights.drop("FT")
    return rec, base, large


def save(name: str, state: dict[str, Any], base: d3.Sums, large: dict[str, torch.Tensor]) -> None:
    write(RESULTS / name / "maths.json", {"base": base, "rows": d3.MATHS_ROWS})
    (LARGE / name).mkdir(parents=True, exist_ok=True)
    save_file(large, str(LARGE / name / "counts_pe.safetensors"))
    write(RESULTS / name / "state.json", state)
    t = rules.score()
    log(f"[{name}] {json.dumps(t['numbers'].get(name), default=float)}")
    log(f"[score] {json.dumps({k: t['comparisons'][k] for k in ('C1', 'C2', 'C3', 'size_think', 'C1_at_100')}, default=float)}")
    log(f"[score] positions {json.dumps(t['comparisons']['position'], default=float)}; predictions {json.dumps(t['predictions'])}")


# ----------------------------------------------------------------------------- stage 0, the control


def control_w0(job: Job, R: dict[str, Any]) -> dict[str, Any]:
    """W0 as the fine-tune: the three base cells equal K0, no switched-on set, so the think fold is no call."""
    d = job.d
    d.weights.put("FT", None)
    c0, c1 = d.counts(), d.counts()
    b = cells(job, "W0 as FT, maths", c0=c0, c1=c1)
    n = job.m.tokens.numel()
    K = {k: d3.total(b, k) / n for k in ("K0", "K00", "K01")}
    S = c1.on() & ~c0.on()
    fold = rules.think(d3.folds(S, ~c0.on(), *d3.think_refs(), C.draws), "T")
    rows = max(rel(r[d3.POOLED], x) for r, x in zip(b["K0"], job.k0, strict=True))
    d.weights.drop("FT")
    holds = (rel(K["K0"], R["K0"]) <= R["id"] and all(rel(K[k], K["K0"]) <= R["id"] for k in ("K00", "K01"))
             and int(S.sum()) == 0 and fold is None and rows <= R["id"])
    return check("W0_state_path", holds, K | {"S": int(S.sum()), "think": fold, "K0_rows_max_rel": rows})


def control_template(job: Job, R: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """E94's template seed 0 (step 100) through this path, against E94's numbers; then at half scale, its outside
    share, null and placement excess unchanged (the split is linear in δW, both shares ratios of quadratic forms)."""
    dW, meta = load_checked("a1_s0", d3.grid_state("anchor1_s0", "full"), job.P)
    r, base, large = read(job, "control_a1_s0", dW)
    ov = r["overlap"]
    got = {"outside": r["outside"], "outside_null": r["outside_null"], "placement": r["placement"]["excess"],
           "K00": d3.total(base, "K00") / job.m.tokens.numel(), "T": ov["T"]["fold"], "L": ov["L"]["fold"], "S": ov["size_S"]}
    holds = (all(rel(got[k], R[f"a1_s0_{k}"]) <= R["rep"] for k in ("outside", "outside_null", "K00"))
             and abs(got["placement"] - R["a1_s0_placement"]) <= R["rep"]
             and all(got[k] is not None and abs(got[k] - R[f"a1_s0_{k}"]) <= R["fold_band"] for k in ("T", "L"))
             and rel(got["S"], R["a1_s0_S"]) <= R["S_rel"])
    out = check("template_through_E104", holds, got | meta)
    for x in dW.values():
        x.mul_(R["half_scale"])
    sp = d3.ls_split(job.d, dW)
    on0 = qr.gated(large["g0_active"], job.m.tokens.numel())
    pl = d3.placement(job.d, sp.pe, on0, sp.energy, job.unit, job.unit_energy, C.null_seeds)
    half = {"outside": sp.outside, "outside_null": sp.outside_null, "placement": pl["excess"]}
    band = job.P["thresholds"]["invariance_band"]
    gap = max(abs(half[k] - got[k]) if math.isfinite(half[k]) else math.inf for k in half)
    return out, check("half_scale_invariant", gap <= band, half | {"max_abs_gap": gap, "split_s": sp.seconds})


def stage_control(job: Job) -> None:
    R = job.P["controls"]
    out: dict[str, Any] = {"W0": control_w0(job, R)}
    out["template"], out["half_scale"] = control_template(job, R)
    write(CONTROL, out | {"passed": True})
    log(f"[control] passed, {d3.peaks()}")


# ----------------------------------------------------------------------------- the states and the scaled copies


def read_state(job: Job, name: str) -> None:
    arm, step = rules.STATES[name]
    dW, meta = load_checked(name, d3.step_state(arm, step), job.P)
    e93 = job.P["first_minutes"]["e93_K00_step100"][arm] if step == 100 else None
    r, base, large = read(job, name, dW, e93)
    save(name, {"name": name, "arm": arm, "step": step} | meta | r, base, large)


def size16(name: str) -> float:
    """K00 − K0 per position over the first 16 maths rows of a written state (E104's or E94's)."""
    p = (rules.E94 if name in rules.E94_STATES else RESULTS) / name / "maths.json"
    b = json.loads(p.read_text())["base"]
    return (d3.total(b, "K00", rows=C.n16) - d3.total(b, "K0", rows=C.n16)) / (C.n16 * d3.ROW_LEN)


def read_copy(job: Job, name: str) -> None:
    """The source's δW scaled to its target's size on the first 16 maths rows (`qr.scale_search`), read as a state;
    its outside share, null and placement excess must equal the source's within the invariance band."""
    src, tgt = rules.COPIES[name]
    rec = d3.step_state(*rules.STATES[src]) if src in rules.STATES else d3.grid_state(*rules.E94_STATES[src])
    dW, meta = load_checked(f"{name}_source", rec, job.P)
    target, m16 = size16(tgt), job.m.head(C.n16)

    def size_at(b: float) -> float:
        job.d.weights.put("FT", dW, scale=b)
        s = d3.cells(job.d, m16)
        x = (d3.total(s, "K00") - d3.total(s, "K0")) / (C.n16 * d3.ROW_LEN)
        log(f"[search {name}] b {b:.6g}: K00 - K0 {x:.9g}, target {target:.9g}")
        return x if math.isnan(x) else max(x, 1e-12)  # a change too small to show reads as small, not as too large

    srch = qr.scale_search(size_at, target, C.match, C.evals)
    for x in dW.values():
        x.mul_(srch["b"])
    r, base, large = read(job, name, dW)
    state = {"name": name, "source": src, "matched_to": tgt, "search": srch,
             "size": "matched" if srch["matched"] else "size unmatched", "source_file": meta, "norm": d3.norm(dW)} | r
    source = rules.load(src)
    assert source is not None, f"{name}: source {src} not written"
    holds, gap = rules.invariant(rules.numbers(state), rules.numbers(source), job.P["thresholds"]["invariance_band"])
    check(f"{name}_invariant", holds, {"max_abs_gap": gap})
    save(name, state, base, large)


# ----------------------------------------------------------------------------- the job


def refuse(names: list[str], stage: str | None) -> None:
    """Exit before loading anything unless the predictions are filed and the list can run in this order."""
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key (not filed)")
    require_filed(PRED)
    if STOPPED.is_file():
        sys.exit(f"refusing to start: {STOPPED} exists (a check missed; the lead decides)")
    if bool(stage) == bool(names) or any(n not in rules.STATES and n not in rules.COPIES for n in names):
        sys.exit("give --stage control or --states from rules.STATES / rules.COPIES, not both")
    if names and not json.loads(CONTROL.read_text() if CONTROL.is_file() else "{}").get("passed"):
        sys.exit("refusing: stage 0 (--stage control) has not passed")
    for i, n in enumerate(names):
        need = [x for x in rules.COPIES.get(n, ()) if x in rules.STATES]
        late = [x for x in need if not (RESULTS / x / "state.json").is_file() and x not in names[:i]]
        if late:
            sys.exit(f"refusing: {n} needs {late} written first or earlier in the list")


def build_job(P: dict[str, Any], decomp: Path) -> Job:
    R = P["controls"]
    hashes = {"unit_nulls": sha256(UNIT), "nulls_json": sha256(E94_NULLS)}
    check("e94_nulls_hashes", hashes["unit_nulls"].startswith(R["unit_nulls_sha256_prefix"])
          and hashes["nulls_json"] == R["nulls_json_sha256"], hashes)
    unit = load(UNIT)
    energy = json.loads(E94_NULLS.read_text())["energy"]
    k0 = json.loads(E94_CONTROL.read_text())["W0_K0_rows"]["maths"]
    return Job(d3.Decomp(decomp, log), d3.maths_rows(), unit, energy, k0, P)


def main() -> None:
    ap = argparse.ArgumentParser(description="E104: the carried D3 rows on the GSM8K arms")
    ap.add_argument("--stage", choices=("control",))
    ap.add_argument("--states", help="comma-separated names from rules.STATES and rules.COPIES")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    a = ap.parse_args()
    names = a.states.split(",") if a.states else []
    refuse(names, a.stage)
    P = json.loads(PRED.read_text())
    LARGE.mkdir(parents=True, exist_ok=True)
    log(f"[start] {vars(a)}; config {C}")
    with torch.no_grad():
        job = build_job(P, Path(a.decomp))
        if a.stage:
            stage_control(job)
            hard_exit(0)
        took: list[float] = []
        for n in names:
            if (RESULTS / n / "state.json").is_file():
                log(f"[{n}] already written")
                continue
            longest = max(took, default=C.first_guess_s)
            if time.time() - T0 + longest > C.limit_s:
                log(f"cut short before {n}: {time.time() - T0:.0f} s used, longest state {longest:.0f} s; resubmit")
                hard_exit(3)
            t0 = time.time()
            (read_state if n in rules.STATES else read_copy)(job, n)
            took.append(time.time() - t0)
            log(f"[{n}] done in {time.time() - t0:.0f} s, {d3.peaks()}")
    log(f"[done] {names}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
