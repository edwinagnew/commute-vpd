"""E94 (qwen_d3_rows), D3: do the four D3 rows separate the template fine-tune from the TOFU facts fine-tune, each at two
seeds? Plan: reports/plans/2026-11-05_plan_e94-d3-rows.md and its appendix; rules: results/predictions.json (the run
refuses to start without it, and while results/stopped.json exists); numbers and comparisons: rules.py; shared reads:
`commute.qwen_d3`.

  --stage control      stage 0, alone and first: E40's step 10 and W0 through every path, against stored references
  --states a,b,...     a later stage's list (rules.STATES, rules.RANDOMS), in the order given; refuses before stage 0
                       has passed; the unit nulls (stage 1) run first if not written; a state already written is skipped

Every pass writes its numbers to results/run.log; each state writes results/<state>/<rows>.json and state.json as it
ends, then rescores results/table.json. A missed check writes results/stopped.json and exits 1. Exit 3: the time left
under the job's limit is less than the longest state so far: resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from commute import qwen_d3 as d3
from commute import qwen_eval, qwen_sets, split
from commute import qwen_reading as qr
from commute.jobs import EXPERIMENTS, hard_exit
from commute.qwen_sft import linears, load, sha256

RESULTS, LARGE = rules.RESULTS, rules.RESULTS / "large"
PRED, STOPPED, CONTROL, NULLS = (RESULTS / f for f in ("predictions.json", "stopped.json", "control.json", "nulls.json"))
E40 = EXPERIMENTS / "qwen_sft" / "results" / "large" / "nowarmup_regen" / "dW" / "step_10.safetensors"  # bf16
AMEND2_S = EXPERIMENTS / "qwen_maps" / "results" / "amend2" / "S.json"
INSPAN_A = 17.78279410038923  # E44 Amendment 2's matched scale, E57's CFG.inspan_a: b = 1 is E57's alive-pieces change
T0 = time.time()


@dataclass(frozen=True)
class Config:
    n16: int = 16  # E57's and E61's 16-row controls; the random's size search reads the same rows
    null_seeds: tuple[int, ...] = (0, 1, 2, 3)  # E57's
    draws: int = 1000  # E57's overlap draws
    match: float = 1.25  # the plan's band on the random's size
    evals: int = 6
    limit_s: float = 2 * 3600 - 180  # debug QoS, less a margin for the last write
    first_guess_s: float = 25 * 60  # the plan's anchor-2 state, 21.5 min, before one is measured

    def __post_init__(self) -> None:
        assert self.n16 <= d3.MATHS_ROWS and self.limit_s > self.first_guess_s


C = Config()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
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
    return abs(x / ref - 1) if ref != 0 else float("inf")


def mean(sums: d3.Sums, key: str, rs: d3.RowSet, rows: int | None = None) -> float:
    n = rs.counts()["kl"][d3.POOLED] if rows is None else rows * rs.tokens.shape[1]
    return d3.total(sums, key, rows=rows) / n


def class_gap(rec: dict[str, Any]) -> float:
    """The largest |sum of the four class sums − the pooled sum| / |pooled| over every row of every key of every pass."""
    return max(abs(sum(r[:4]) - r[4]) / max(abs(r[4]), 1e-30)
               for part in ("base", "new", "own") if part in rec for v in rec[part].values() for r in v)


def memory(name: str, F: dict[str, Any]) -> None:
    gpu, host = torch.cuda.max_memory_allocated() / d3.GIB, d3.host_gb()
    check(f"{name}_memory", gpu <= F["gpu_max_gib"] and host <= F["host_max_gb"], {"gpu_gib": gpu, "host_gb": host})


def zero_change(d: d3.Decomp) -> Any:
    """Every live column moved by zero through `applied`: W0's control on the split path."""
    return lambda s: (None, None, *(torch.zeros(x.shape, dtype=d3.F64, device=d3.DEV) for x in d.alive.uv(s, d3.DEV, d3.F64)))


# ----------------------------------------------------------------------------- stage 0, the control


def control_w0(d: d3.Decomp, rows: dict[str, d3.RowSet], R: dict[str, Any]) -> dict[str, Any]:
    """W0 as the fine-tuned model through the state path: the three base cells equal, no switched-on set, the zero
    split on 16 rows reads K0 there; W0's K0 on every row set kept as the per-state reference."""
    m, out = rows["maths"], {}
    d.weights.put("FT", None)
    c0, c1 = d.counts(), d.counts()
    w = cells_timed(d, m, "W0 as FT, maths", out, c0=c0, c1=c1)
    K = {k: mean(w, k, m) for k in ("K0", "K00", "K01")}
    S = int((c1.on() & ~c0.on()).sum())
    z = cells_timed(d, m.head(C.n16), "W0 as FT, zero split, 16 rows", out, change=zero_change(d))
    Z = {k: mean(z, k, m, C.n16) for k in ("K10", "K11")}
    out["W0"] = check("W0_state_path", rel(K["K0"], R["K0"]) <= R["id"] and all(rel(K[k], K["K0"]) <= R["id"] for k in ("K00", "K01"))
                      and S == 0 and all(rel(v, R["K0_16"]) <= R["id"] for v in Z.values()), K | Z | {"S": S})
    out["W0_K0_rows"] = {"maths": [r[d3.POOLED] for r in w["K0"]]}
    gaps = [class_gap({"base": w}), class_gap({"new": z})]
    for name in ("a1", "tofu"):
        b = cells_timed(d, rows[name], f"W0 as FT, {name}", out)
        out["W0_K0_rows"][name] = [r[d3.POOLED] for r in b["K0"]]
        gaps.append(class_gap({"base": b}))
        log(f"[W0 {name}] K0 {mean(b, 'K0', rows[name]):.9g}, whole-question K0 {d3.total(b, 'K0', 0) / rows[name].counts()['kl'][0]:.9g}")
    out["classes"] = check("classes_add_to_pooled", max(gaps) <= R["classes_rel"], {"max_rel_gap": max(gaps)})
    d.weights.drop("FT")
    return out


def control_e40(d: d3.Decomp, rows: dict[str, d3.RowSet], R: dict[str, Any], out: dict[str, Any]) -> None:
    """E40's step 10: base cells and the switched-on set, the do-nothing and the split through the new pass, the own-V′
    cells, the overlap with T and L; its split's piece energies saved for stage 1's check."""
    m, m16 = rows["maths"], rows["maths"].head(C.n16)
    dW = load(E40)
    d.weights.put("FT", dW)
    c0, c1 = d.counts(), d.counts()
    b = cells_timed(d, m, "E40 base, maths", out, c0=c0, c1=c1)
    out["base"] = check("E40_base", all(rel(mean(b, k, m), R[f"e40_{k}"]) <= R["rep"] for k in ("K00", "K01")),
                        {k: mean(b, k, m) for k in ("K0", "K00", "K01")})
    on0, S = c0.on(), c1.on() & ~c0.on()
    refs, site_of = d3.think_refs()
    f = d3.folds(S, ~on0, refs, site_of, C.draws)
    out["overlap"] = check("E40_overlap", all(abs(f[k]["fold"] - R[f"fold_{k}"]) <= R["fold_band"] for k in ("T", "L")),
                           {k: f[k]["fold"] for k in ("T", "L")} | {"size_S": f["size_S"]})
    K00 = d3.total(b, "K00", rows=C.n16)
    dn = cells_timed(d, m16, "E40 do-nothing, new pass, 16 rows", out, change=d3.ls_change(d, None))
    share = (K00 - d3.total(dn, "K10")) / (K00 - d3.total(dn, "K11"))
    out["do_nothing"] = check("do_nothing_direct", abs(share) <= R["id"], {"direct": share})
    sp = d3.ls_split(d, dW)
    log(f"[split E40] {sp.seconds:.0f} s, stored error {sp.stored_error:.3g}, outside {sp.outside:.6g} (null {sp.outside_null:.6g}), {d3.peaks()}")
    del dW
    save_file({"pe": sp.pe.contiguous(), "on0": on0}, str(LARGE / "control_e40.safetensors"))
    out["e40_energy"], out["seconds"]["split"] = sp.energy, sp.seconds
    new = cells_timed(d, m16, "E40 new pass, 16 rows", out, change=d3.ls_change(d, sp.X))
    got = {"K00": K00, "K10": d3.total(new, "K10"), "K11": d3.total(new, "K11")}
    got["direct"] = (got["K00"] - got["K10"]) / (got["K00"] - got["K11"])
    out["new_pass"] = check("E40_new_pass_vs_E61", all(rel(got[k], R[f"e61_{k}"]) <= R["rep"] for k in got), got)
    own = cells_timed(d, m, "E40 own V', maths", out, change=d3.ls_change(d, sp.X), w0_v=False)
    out["own"] = check("E40_own_cells", all(rel(mean(own, k, m), R[f"own_{k}"]) <= R["rep"] for k in ("K10", "K11")),
                       {k: mean(own, k, m) for k in ("K10", "K11")})
    gap = max(class_gap({"base": b, "new": new, "own": own}), class_gap({"new": dn}))
    out["classes_e40"] = check("classes_add_to_pooled_e40", gap <= R["classes_rel"], {"max_rel_gap": gap})
    d.weights.drop("FT")


def cells_timed(d: d3.Decomp, rs: d3.RowSet, tag: str, out: dict[str, Any], **kw: Any) -> d3.Sums:
    t0 = time.time()
    s = d3.cells(d, rs, **kw)
    out.setdefault("seconds", {})[tag] = time.time() - t0
    log(f"[cells {tag}] " + " ".join(f"{k} {mean(s, k, rs) if k.startswith('K') else d3.total(s, k):.9g}" for k in s)
        + f"; rows {len(rs.tokens)}, {time.time() - t0:.0f} s, {d3.peaks()}")
    return s


def stage_control(d: d3.Decomp, rows: dict[str, d3.RowSet], P: dict[str, Any]) -> None:
    R = P["controls"]
    out = control_w0(d, rows, R)
    control_e40(d, rows, R, out)
    s = out["seconds"]  # the plan's per-state composition, from this job's passes (maths = 128 rows; a1 = 64)
    per = s["split"] + 60 + s["E40 base, maths"] * 2 + s["E40 own V', maths"]
    out["projection_min"] = {"anchor2_state": (per + s["E40 base, maths"] * 2) / 60, "anchor1_state": (per + s["E40 base, maths"]) / 60}
    write(CONTROL, out | {"passed": True})
    log(f"[control] passed; projection {json.dumps(out['projection_min'])} min a state, {d3.peaks()}")


# ----------------------------------------------------------------------------- stage 1, the unit nulls


def stage_nulls(d: d3.Decomp, P: dict[str, Any]) -> dict[str, Any]:
    """The 4 Gaussian nulls of row 3 once, at unit scale; rescaled to E40, they must read E57's stored nulls."""
    if NULLS.is_file():
        return json.loads(NULLS.read_text())
    R, t0 = P["controls"], time.time()
    un = d3.unit_nulls(d, {s: tuple(lin.weight.shape) for s, lin in linears(d.model).items()}, C.null_seeds)
    save_file({f"pe{k}": pe.contiguous() for k, (pe, _) in un.items()}, str(LARGE / "unit_nulls.safetensors"))
    out = {"energy": {str(k): E for k, (_, E) in un.items()}, "seconds": time.time() - t0}
    ctrl, e40 = json.loads(CONTROL.read_text()), load(LARGE / "control_e40.safetensors")
    pl = placement(d, e40["pe"], e40["on0"].bool(), ctrl["e40_energy"], out)
    holds = all(abs(a - b) <= R["null_band"] for a, b in zip(pl["null"], R["e57_nulls"], strict=True))
    out["e40"] = check("unit_nulls_read_E57", holds and abs(pl["excess"] - R["e57_excess"]) <= R["null_band"], pl)
    write(NULLS, out)
    return out


def placement(d: d3.Decomp, pe: torch.Tensor, on0: torch.Tensor, energy: dict[str, float], nulls: dict[str, Any]) -> dict[str, Any]:
    """Row 3: the split's energy share on the subcomponents on at W0 (maths rows), minus the mean of the 4 nulls."""
    unit = load(LARGE / "unit_nulls.safetensors")
    null = [qr.placement(d3.null_pe(unit[f"pe{k}"], nulls["energy"][str(k)], energy, d.lay.site_of), on0) for k in C.null_seeds]
    share = qr.placement(pe, on0)
    return {"share": share, "null": null, "excess": share - sum(null) / len(null)}


# ----------------------------------------------------------------------------- the states


def read_rows(d: d3.Decomp, rs: d3.RowSet, X: dict[str, torch.Tensor], own: bool, ref_k0: list[float], name: str,
              P: dict[str, Any], c0: Any = None, c1: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": rs.rows.name, "n": rs.counts()}
    out["base"] = cells_timed(d, rs, f"{name} base, {rs.rows.name}", out, c0=c0, c1=c1)
    F, k0 = P["first_minutes"], [r[d3.POOLED] for r in out["base"]["K0"]]
    check(f"{name}_{rs.rows.name}_K0_is_W0s", max(rel(a, b) for a, b in zip(k0, ref_k0, strict=True)) <= F["K0_rel"], {"rows": len(k0)})
    memory(name, F)
    if name == "a1_s0" and rs.rows.name == "a1":
        k = mean(out["base"], "K00", rs)
        log(f"[first minutes] a1_s0 K00 on its rows {k:.9g}, E68's {F['a1_K00']} (logged; a stop only over {F['a1_K00_rel']})")
        check("a1_s0_K00_near_E68", rel(k, F["a1_K00"]) <= F["a1_K00_rel"], {"K00": k})
    out["new"] = cells_timed(d, rs, f"{name} new pass, {rs.rows.name}", out, change=d3.ls_change(d, X))
    if own:
        out["own"] = cells_timed(d, rs, f"{name} own V', {rs.rows.name}", out, change=d3.ls_change(d, X), w0_v=False)
    check(f"{name}_{rs.rows.name}_classes", class_gap(out) <= P["controls"]["classes_rel"], {"max_rel_gap": class_gap(out)})
    return out


def finish(d: d3.Decomp, rows: dict[str, d3.RowSet], name: str, dW: dict[str, torch.Tensor], own: tuple[str, ...],
           own_pass: bool, state: dict[str, Any], P: dict[str, Any], ref: dict[str, list[float]] | None = None) -> None:
    """The split, the maths rows (with the gate counts), the own rows, the overlap and row 3; files as each ends. `ref`:
    W0's K0 per row of each row set, stage 0's when not given (Amendment 1 adds its own rows')."""
    k0: dict[str, list[float]] = ref or json.loads(CONTROL.read_text())["W0_K0_rows"]
    nulls = json.loads(NULLS.read_text())
    sp = d3.ls_split(d, dW)
    state |= {"outside": sp.outside, "outside_null": sp.outside_null, "stored_error": sp.stored_error, "split_s": sp.seconds,
              "energy_sigma": sum(sp.energy.values())}
    log(f"[split {name}] {json.dumps({k: state[k] for k in ('outside', 'outside_null', 'stored_error', 'split_s')})}, {d3.peaks()}")
    del dW
    c0, c1 = d.counts(), d.counts()
    write(RESULTS / name / "maths.json", read_rows(d, rows["maths"], sp.X, False, k0["maths"], name, P, c0, c1))
    for r in own:
        write(RESULTS / name / f"{r}.json", read_rows(d, rows[r], sp.X, own_pass, k0[r], name, P))
    on0, S = c0.on(), c1.on() & ~c0.on()
    state["overlap"] = d3.folds(S, ~on0, *d3.think_refs(), C.draws)
    state["placement"] = placement(d, sp.pe, on0, sp.energy, nulls)
    (LARGE / name).mkdir(parents=True, exist_ok=True)
    save_file({"g0_active": c0.active.cpu(), "g1_active": c1.active.cpu(), "pe_ls": sp.pe.contiguous()}, str(LARGE / name / "counts_pe.safetensors"))
    write(RESULTS / name / "state.json", state)
    d.weights.drop("FT")
    t = rules.score()
    log(f"[{name}] {json.dumps(t['numbers'].get(name), default=float)}")
    log(f"[score] {json.dumps(t['comparisons']['stop_rule'])} {json.dumps(t['predictions'])}")


def read_state(d: d3.Decomp, rows: dict[str, d3.RowSet], name: str, P: dict[str, Any]) -> None:
    arm, size, own, _, _ = rules.STATES[name]
    rec = d3.grid_state(arm, size)
    dW = d3.load_state(rec)
    nrm = d3.norm(dW)
    check(f"{name}_norm", rel(nrm, rec["norm"]) <= P["first_minutes"]["norm_rel"], {"norm": nrm, "e91": rec["norm"]})
    state = {"name": name, "arm": arm, "size": size, "step": rec["step"], "file": rec["file"], "norm": nrm, "sha256": sha256(rec["path"])}
    d.weights.put("FT", dW)
    finish(d, rows, name, dW, (own,), True, state, P)


def read_random(d: d3.Decomp, rows: dict[str, d3.RowSet], name: str, P: dict[str, Any]) -> None:
    """A random change matched to its reference state's K00 − K0 on the first 16 maths rows (`qr.scale_search`)."""
    kind, ref, own = rules.RANDOMS[name]
    m16 = rows["maths"].head(C.n16)
    rm = json.loads((RESULTS / ref / "maths.json").read_text())["base"]
    target = (d3.total(rm, "K00", rows=C.n16) - d3.total(rm, "K0", rows=C.n16)) / (C.n16 * d3.ROW_LEN)
    build = gaussian(d, ref) if kind == "gauss" else alive(d)

    def size_at(b: float) -> float:
        d.weights.put("FT", build(b))
        s = d3.cells(d, m16)
        x = (d3.total(s, "K00") - d3.total(s, "K0")) / (C.n16 * d3.ROW_LEN)
        log(f"[search {name}] b {b:.6g}: K00 - K0 {x:.9g}, target {target:.9g}")
        return x if math.isnan(x) else max(x, 1e-12)  # a change too small to show reads as small, not as too large

    srch = qr.scale_search(size_at, target, C.match, C.evals)
    dW = build(srch["b"])
    d.weights.put("FT", dW)
    state = {"name": name, "kind": kind, "matched_to": ref, "search": srch, "size": "matched" if srch["matched"] else "size unmatched",
             "norm": d3.norm(dW)}
    finish(d, rows, name, dW, own, False, state, P)


def gaussian(d: d3.Decomp, ref: str) -> Any:
    """`split.random_like` of the reference state's δW (E57's Gaussian: Σ-weighted energy per site, null seed 0), × b."""
    sig, M = d.sigma(), d3.load_state(d3.grid_state(*rules.STATES[ref][:2]))
    G = {s: d3.pinned(split.random_like(M[s].to(d3.DEV, d3.F64), sig[s].to(d3.DEV), d3.null_seed(0, s)).float().cpu()) for s in M}
    del M
    return lambda b: {s: g * b for s, g in G.items()}


def alive(d: d3.Decomp) -> Any:
    a = json.loads(AMEND2_S.read_text())["10"]["per_site"]
    return lambda b: d3.alive_pieces(d.alive, {s: a[s]["a"] * INSPAN_A * b for s in a}, d3.DEV)


# ----------------------------------------------------------------------------- the job


def build_rows(P: dict[str, Any]) -> dict[str, d3.RowSet]:
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    sets = qwen_sets.load()
    rows = {"maths": d3.maths_rows(), "a1": d3.own_rows("a1", tok, sets), "tofu": d3.own_rows("tofu", tok, sets)}
    got = {k: rows[k].counts()["kl"][0] for k in ("a1", "tofu")}
    check("rows_whole_question_positions", got == P["controls"]["whole_positions"], got | {"shapes": {k: list(r.tokens.shape) for k, r in rows.items()}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="E94: the D3 rows")
    ap.add_argument("--stage", choices=("control",))
    ap.add_argument("--states", help="comma-separated names from rules.STATES and rules.RANDOMS")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if STOPPED.is_file():
        sys.exit(f"refusing to start: {STOPPED} exists (a check missed; the lead decides)")
    names = a.states.split(",") if a.states else []
    if bool(a.stage) == bool(names) or any(n not in rules.STATES and n not in rules.RANDOMS for n in names):
        sys.exit("give --stage control or --states from rules.STATES / rules.RANDOMS, not both")
    if names and not json.loads(CONTROL.read_text() if CONTROL.is_file() else "{}").get("passed"):
        sys.exit("refusing: stage 0 (--stage control) has not passed")
    P = json.loads(PRED.read_text())
    LARGE.mkdir(parents=True, exist_ok=True)  # a link into commute-data where fleet/link-data has run
    log(f"[start] {vars(a)}; config {C}")
    rows = build_rows(P)
    with torch.no_grad():
        d = d3.Decomp(Path(a.decomp), log)
        if a.stage:
            stage_control(d, rows, P)
            hard_exit(0)
        stage_nulls(d, P)
        each_state(names, lambda n: (read_state if n in rules.STATES else read_random)(d, rows, n, P))
    log(f"[done] {names}; FINISHED")
    hard_exit(0)


def each_state(names: list[str], read: Callable[[str], None]) -> None:
    """`read` each state not yet written, in order; exit 3 when the time left is under the longest state so far."""
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
        read(n)
        took.append(time.time() - t0)
        log(f"[{n}] done in {time.time() - t0:.0f} s, {d3.peaks()}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
