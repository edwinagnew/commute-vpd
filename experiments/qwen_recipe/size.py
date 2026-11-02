"""E61 (qwen_recipe), Job S: the direct share against size, through the frozen decomposition of Qwen3-1.7B. Plan:
reports/plans/2026-11-02_plan_e61-direct-share-size.md, Part 2, amendments overriding. Rules and expected values:
results/predictions_size.json (the run refuses to start without it). Stages, times, memory: cluster/qwen_recipe_size.sbatch.
Direct share = (K00 - K10) / (K00 - K11). "true": every mask computed with the original subcomponents, before the split is
applied; "file": E57's lab.cells, whose masks read the moved vectors. Size: K00 - K0, and KL(W0 || state), per position.
Each stage writes results/<prefix>_<stage>.json as it goes and resumes from it. Exit 1: a stop or a refusal; 3: resubmit.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import boot_direct
import torch
from checks import RESULTS, RUN, Checks, e57_record, kl_both, split_x, state_dw
from lab import CFG, DEV, E45, kl, write_json
from torch import Tensor

from commute import qwen_maps
from commute import qwen_reading as qr
from commute.qwen_sft import linears, load, sha256
from commute.qwen_vpd import importances, masked_logits

PRED = RESULTS / "predictions_size.json"
LADDERS = {"gold_1e-5": RESULTS / "large" / "train" / "gold_1e-5", "gold_s0": E45 / "large" / "dW" / "gold_s0"}
K10_ON_FILE = ("gold_s0", "gauss")  # checks.py read their true K10 and K11 on 16 rows, on the fp32 X: reproduced here
KINDS = ("gauss", "inspan")  # E57's keys: the Gaussian change, and the alive-subcomponent change
READ_BY_E57 = ("gold_s0", *KINDS)
BIGS = tuple(f"{k}_big" for k in KINDS)
ROW_LEN = boot_direct.ROW_LEN
BUILD_S, CELLS_S, CHUNK_S, EVAL_S = 400.0, 240.0, 120.0, 60.0  # budgets: checks_k10_* logs, with margin


@dataclass(frozen=True)
class SizeCfg:
    rows: int = 128  # read per state
    select_rows: int = 16  # the select and search stages' reading
    chunk: int = 8  # rows whose masks are held at once; entering `applied` costs 26 s, so one row at a time is 2 h per state
    prefix: str = "size"
    gold_rule: str = "train_gold_1e-5"  # the file whose G verdict gold must have passed
    ladder: str = "gold_1e-5"
    steps: tuple[int, ...] = ()  # (): every saved state of the ladder

    def __post_init__(self) -> None:
        assert self.select_rows <= self.rows <= CFG.n_rows and self.rows % self.chunk == 0 and CFG.row_len == ROW_LEN
        scored = self.rows == CFG.n_rows
        assert self.ladder in LADDERS and scored == (self.prefix == "size") == bool(self.gold_rule)
        assert (self.ladder == "gold_1e-5" and not self.steps) if scored else len(self.steps) == 2

    def name(self, step: int) -> str:
        return f"{self.ladder}_{step}"

    def path(self, step: int) -> Path:
        return LADDERS[self.ladder] / f"step_{step}.safetensors"


# The rehearsal's chunk is 1, so the true pass's second chunk and a resume inside the pass run (review S4).
CONFIGS = {128: SizeCfg(), 2: SizeCfg(rows=2, select_rows=2, chunk=1, prefix="size2", gold_rule="", ladder="gold_s0", steps=(5, 9))}


class Size(Checks):
    """checks.py's job, with this job's stop file and time budget, and a log that gets only what is new."""

    def __init__(self, stage: str, e40: str, decomp: str, cfg: SizeCfg, budget_s: float, R: dict[str, Any]):
        super().__init__(stage, e40, decomp)
        self.size_cfg, self.budget_s, self.R = cfg, budget_s, R
        self.log(f"[size] {cfg}, budget {budget_s:.0f} s")

    def stop(self, name: str, why: str) -> None:
        write_json(RESULTS / f"{self.size_cfg.prefix}_stopped.json", {"name": name, "why": why, "stage": self.stage})
        self.log(f"STOP {name}: {why}")
        sys.exit(f"STOP {name}: {why}")

    def budget(self, need_s: float) -> None:
        if time.time() - self.t0 + need_s > self.budget_s:
            self.log(f"BUDGET: {need_s:.0f} s more would pass {self.budget_s:.0f} s; resubmit the same command")
            sys.exit(3)

    def save(self, tail: str, d: dict[str, Any], new: dict[str, Any]) -> None:
        write_json(RESULTS / f"{self.size_cfg.prefix}_{tail}.json", d)
        self.log(f"[{tail}] {json.dumps(new, default=float)}")

    def read(self, tail: str) -> dict[str, Any]:
        return self.get(f"{self.size_cfg.prefix}_{tail}")

    def selected(self) -> dict[str, Any]:
        if "big" not in (sel := self.read("select")):
            sys.exit("refusing: select writes which states are read")
        return sel


def e57_rows(name: str) -> dict[str, list[float]]:
    return e57_record(name)["rows"]["maths"]


def size_of(rows: dict[str, list[float]], idx: list[int]) -> float:
    return (sum(rows["K00"][i] for i in idx) - sum(rows["K0"][i] for i in idx)) / (len(idx) * ROW_LEN)


def nearest(sizes: dict[str, float], targets: list[float], band: float) -> dict[str, Any]:
    """The state nearest any target by |log ratio|, the first on a tie; a size not finite and positive only if all are."""
    gap = lambda x: min(abs(math.log(x / t)) for t in targets) if qr.ok(x) and x > 0 else math.inf
    name = min(sizes, key=lambda n: gap(sizes[n]))
    return {"name": name, "size": sizes[name], "in_band": any(qr.within(sizes[name], t, band) for t in targets)}


def at_size(x: float, reads: list[tuple[float, float]]) -> float:
    """A random kind's share at size x, from its two reads (size, share): linear between them; below the lower read that
    read; above the higher one the line, never under that read (both toward C); reads at one size: the higher share."""
    if not all(qr.ok(v) for r in reads for v in r) or not qr.ok(x):
        return qr.NAN
    (x0, s0), (x1, s1) = sorted(reads)
    if qr.r9(x0) == qr.r9(x1):
        return max(s0, s1)
    if qr.r9(x) <= qr.r9(x0):
        return s0
    through = s0 + (s1 - s0) * (x - x0) / (x1 - x0)
    return max(through, s1) if x > x1 else through


def rows_of(rec: dict[str, Any], version: str) -> dict[str, list[float]]:
    return {k: rec["base"][k] for k in ("K0", "K00", "K01")} | {k: rec[version][k] for k in ("K10", "K11")}


def point(rec: dict[str, Any], version: str, idx: list[int]) -> dict[str, float]:
    """A state's two sizes and direct share on these rows."""
    return {"K00_minus_K0": size_of(rec["base"], idx), "KL_W0_state": sum(rec["KL"]["W0_state"][i] for i in idx) / (len(idx) * ROW_LEN),
            "share": boot_direct.direct(rows_of(rec, version), idx)}


def gaps(recs: dict[str, Any], gold: str, version: str, idx: list[int]) -> dict[str, dict[str, Any]]:
    """Per size axis: gold's direct share minus the higher random's, each random at gold's size on that axis."""
    g, out = point(recs[gold], version, idx), {}
    pts = {k: [point(recs[n], version, idx) for n in (k, f"{k}_big")] for k in KINDS}
    for ax in ("K00_minus_K0", "KL_W0_state"):
        rand = {k: at_size(g[ax], [(p[ax], p["share"]) for p in pts[k]]) for k in KINDS}
        high = max(rand.values()) if all(qr.ok(v) for v in rand.values()) else qr.NAN
        out[ax] = {"gold_size": g[ax], "gold_share": g["share"], "random_at_size": rand, "gap": g["share"] - high}
    return out


def floor(a: dict[str, Any], b: dict[str, Any], R: dict[str, Any]) -> dict[str, Any]:
    """Review M5: f = max(floor_min, the two gold runs' difference on E57's masks) if their sizes are within the band."""
    idx = list(range(len(a["base"]["K0"])))
    diff = abs(boot_direct.direct(rows_of(a, "file"), idx) - boot_direct.direct(rows_of(b, "file"), idx))
    near = qr.within(size_of(a["base"], idx), size_of(b["base"], idx), R["band"])
    f = max(R["floor_min"], diff) if near and qr.ok(diff) else R["floor_min"]
    return {"f": f, "difference": diff, "sizes_within_band": near, "difference_over_f": qr.ok(diff) and diff > f}


def line(all_gaps: list[float], f: float, small_in_band: bool, R: dict[str, Any]) -> str:
    """Any gap NaN: no reading. Any gap at most f: C. Every gap at least 3f, small state in the band: B. Else: C, the note."""
    if not all(qr.ok(g) for g in all_gaps):
        return "no reading"
    if any(qr.r9(g) <= qr.r9(f) for g in all_gaps):
        return "C"
    if small_in_band and all(qr.r9(g) >= qr.r9(R["b_multiple"] * f) for g in all_gaps):
        return "B"
    return "C, with the note"


def decide(recs: dict[str, Any], pick: dict[str, Any], f: float, version: str, draws: list[list[int]], R: dict[str, Any]) -> dict[str, Any]:
    """One mask version's reading: gaps at both sizes on both axes, a paired bootstrap of each, and the line."""
    idx, reads = list(range(len(recs[pick["big"]]["base"]["K0"]))), {}
    for size in ("big", "small"):
        reads[size], boots = gaps(recs, pick[size], version, idx), [gaps(recs, pick[size], version, i) for i in draws]
        for ax, r in reads[size].items():
            r["gap_boot95"] = qr.interval([b[ax]["gap"] for b in boots])
    used = (pick["big"], pick["small"], *KINDS, *BIGS)
    unscored = sorted({n for n in used if qr.whole(rows_of(recs[n], version), ROW_LEN)["status"] != "scored"})
    all_gaps = [r["gap"] for s in reads.values() for r in s.values()]
    return {"line": "no reading" if unscored else line(all_gaps, f, pick["small_in_band"], R), "unscored": unscored, "reads": reads}


def true_chunk(lab: Size, rows: Tensor, change: Any) -> dict[str, list[float]]:
    """Per-row K00, K10, K11, reference FT's clean logits. Masks g0, g1 are computed with the original subcomponents and held
    on the host BEFORE `applied` moves any column; nothing calls `importances` inside it, so no mask here sees V'."""
    model, comps, gamma, held = lab.model, lab.comps, lab.gamma, []
    host = lambda g: {s: v.cpu() for s, v in g.items()}
    against = lambda r, ref, g: kl(ref.to(DEV), masked_logits(model, r[None].to(DEV), comps, {s: v.to(DEV) for s, v in g.items()}))
    for r in rows:
        lab.use("FT")
        ref, g1 = importances(model, r[None].to(DEV), gamma, comps)
        ref, g1 = ref.cpu(), host(g1)  # off the GPU before the next forward: the rule is 44 GiB, peak 42.4
        lab.use("W0")
        held.append((r, ref, host(importances(model, r[None].to(DEV), gamma, comps)[1]), g1))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if peak > lab.R["host_stop_gb"]:  # Slurm's kill at 128G is exit 137, outside the sbatch's 0/1/3, and repeats on resubmit
        lab.stop("host_memory", f"host peak {peak:.1f} GB after {len(rows)} rows of masks > {lab.R['host_stop_gb']}")
    out = {"K00": [against(r, ref, g0) for r, ref, g0, _ in held], "K10": [], "K11": []}
    with qwen_maps.applied(model, comps, lab.alive, change, {s: x.weight.data for s, x in linears(model).items()}):
        for r, ref, g0, g1 in held:
            out["K10"].append(against(r, ref, g0))
            out["K11"].append(against(r, ref, g1))
            lab.gpu_rule("true masks")
    lab.log(f"[true chunk] {len(rows)} rows, {sum(v.numel() * 4 for v in held[0][2].values()) / 1e9:.2f} GB a mask set, {lab.peaks()}")
    return out


def true_pass(lab: Size, tail: str, name: str, out: dict[str, Any], rows: Tensor, change: Any) -> None:
    """Written per chunk; K00 equals the cells' per chunk; on a state checks.py read, its first rows' K10 and K11 reproduce."""
    t, chunk = out.setdefault("true", {"K00": [], "K10": [], "K11": []}), lab.size_cfg.chunk
    for s in range(len(t["K10"]), len(rows), chunk):
        lab.budget(CHUNK_S)
        k, want = true_chunk(lab, rows[s : s + chunk], change), sum(out["base"]["K00"][s : s + chunk])
        lab.check(f"true_K00_{name}_{s}", qr.close(sum(k["K00"]), want, CFG.id_tol), {"true": sum(k["K00"]), "cells": want})
        for c in t:
            t[c] += k[c]
        lab.save(tail, out, {"true_from_row": s, **k})
    if name in K10_ON_FILE:
        o, n = lab.get(f"checks_k10_{name}")["true"]["per_row"], min(len(rows), CFG.n16)
        pairs = {c: (sum(t[c][:n]), sum(o[c][:n])) for c in ("K10", "K11")}
        lab.check(f"true_vs_checks_{name}", all(qr.close(a, b, CFG.rep_tol) for a, b in pairs.values()), pairs)


def stage_control(lab: Size) -> None:
    """C0's Gaussian, which catches drift and a swapped file, and no more; then the new pass at W0, zero split: all = K0."""
    n, saved = lab.size_cfg.select_rows, e57_rows("gauss")
    if lab.read("control"):
        return
    lab.budget(BUILD_S)
    lab.set_weights("FT", state_dw(lab, "gauss"))
    per_row = lab.cells(lab.maths(n), tag="control, Gaussian")
    base, want = RUN.total(per_row), {k: sum(saved[k][:n]) for k in per_row}
    lab.check("control_gaussian", all(qr.close(base[k], want[k], CFG.rep_tol) for k in ("K0", "K00")), {"this": base, "C0": want})
    k0 = per_row["K0"][: lab.size_cfg.chunk]
    lab.set_weights("FT", None)
    z = true_chunk(lab, lab.maths(len(k0)), lab.change(None, "zero"))
    lab.check("control_W0", all(qr.close(sum(z[k]), sum(k0), CFG.id_tol) for k in z), {"true": z, "K0": k0})
    lab.drop_set("FT")
    lab.save("control", rec := {"gaussian": base, "C0": want, "K0": k0, "W0_zero_split": z}, rec)


def stage_select(lab: Size) -> None:
    cfg, R, out = lab.size_cfg, lab.R, lab.read("select") or {"states": {}}
    if "big" in out:
        return
    if not lab.read("control"):
        sys.exit("refusing: the control runs first")
    if cfg.gold_rule and (v := lab.get(cfg.gold_rule).get("verdict", {})).get("learned") is not True:
        v.pop("beside", None)
        lab.stop("gold_rule", f"{cfg.gold_rule}: no G verdict, or not learned: {json.dumps(v)}")
    steps = sorted(cfg.steps or (int(p.stem[5:]) for p in LADDERS[cfg.ladder].glob("step_*.safetensors")))
    every, some, final, rows = list(range(cfg.rows)), list(range(cfg.select_rows)), cfg.name(steps[-1]), lab.maths(cfg.rows)
    for t in reversed(steps):  # the final state first: its size sets the branch
        if (name := cfg.name(t)) in out["states"]:
            continue
        lab.budget(CELLS_S if name == final else EVAL_S)
        sha = sha256(cfg.path(t))
        lab.set_weights("FT", load(cfg.path(t)))
        c = lab.cells(rows if name == final else rows[: cfg.select_rows], tag=name)
        out["states"][name] = {"step": t, "sha256": sha, "size_select": size_of(c, some), "size": size_of(c, list(range(len(c["K0"]))))}
        lab.save("select", out, {name: out["states"][name]})
        lab.drop_set("FT")
    last, s0 = out["states"][final], e57_rows("gold_s0")
    if not qr.ok(last["size"]):
        lab.stop("select", f"the final state's size is not finite: {last}")
    big = {"name": final, "size_select": last["size_select"], "size": last["size"]}
    if qr.r9(last["size"]) <= R["big_min"]:  # review M6
        big = {"name": "gold_s0", "size_select": size_of(s0, some), "size": size_of(s0, every)}
    targets = [size_of(e57_rows(k), some) for k in KINDS]
    pool = {cfg.name(t): out["states"][cfg.name(t)]["size_select"] for t in steps if cfg.name(t) != big["name"]}
    new = {"final": final, "final_size": last["size"], "above_gold_s0": last["size"] > size_of(s0, every), "big": big,
           "small": nearest(pool, targets, R["band"]), "targets": targets}
    lab.save("select", out | new, new)


def scale_of(lab: Size, name: str) -> float:
    """b, where a state is W0 + b δW: its search's for a scaled random change, otherwise 1."""
    return lab.read(f"search_{name.removesuffix('_big')}")["b"] if name in BIGS else 1.0


def state_of(lab: Size, name: str) -> tuple[dict[str, Tensor], str]:
    """A state's δW and its identity: the file's sha256, or, for a random change rebuilt from its seed, kind and scale."""
    if name == "gold_s0":
        return lab.load_dw(name), sha256(lab.dw_path(name))
    kind = name.removesuffix("_big")
    if kind not in KINDS:
        p = lab.size_cfg.path(int(name.rsplit("_", 1)[1]))
        got, want = sha256(p), lab.read("select")["states"][name]["sha256"]
        lab.check(f"sha256_{name}", got == want, {"sha256": got, "select": want})
        return load(p), got
    lab.park_gamma()
    dW, b = state_dw(lab, kind), scale_of(lab, name)
    for g in dW.values():
        g.mul_(b)
    return dW, f"rebuilt {kind} x {b}"


def read_cells(lab: Size, tail: str, name: str, out: dict[str, Any], key: str, rows: Tensor, change: Any) -> None:
    """One pass of E57's cells, if not written. On the three states E57 read, every cell is E57's; on any other, K0 is."""
    if key in out:
        return
    lab.budget(CELLS_S)
    out[key] = lab.cells(rows, change, tag=f"{name} {key}")
    saved = e57_rows(name if name in READ_BY_E57 else "gauss")
    pairs = {k: (sum(v), sum(saved[k][: len(rows)])) for k, v in out[key].items() if name in READ_BY_E57 or k == "K0"}
    lab.check(f"repro_{name}_{key}", all(qr.close(a, b, CFG.rep_tol) for a, b in pairs.values()), pairs)
    lab.save(tail, out, {key: out[key], "identity": out["identity"]})


def check_size(lab: Size, name: str, base: dict[str, list[float]]) -> None:
    """A state's size here against what its search or `select` wrote: the same weights, or a stop. Also catches a state
    file left from an earlier search."""
    states = lab.read("select")["states"]
    if name in BIGS:
        got, want = size_of(base, list(range(len(base["K0"])))), lab.read(f"search_{name.removesuffix('_big')}")["size"]
    elif name in states:
        got, want = size_of(base, list(range(lab.size_cfg.select_rows))), states[name]["size_select"]
    else:
        return
    lab.check(f"size_{name}", qr.close(got, want, CFG.rep_tol), {"here": got, "written": want})


def check_kl(lab: Size, name: str, kls: dict[str, list[float]]) -> None:
    """The second size axis against checks_sizes.json, on the 128 rows it was read from: every gap reads this KL."""
    ref = lab.get("checks_sizes")["states"][name]
    pairs = {f"KL_{k}": (sum(kls[k]) / (len(kls[k]) * ROW_LEN), ref[f"KL_{k}"]) for k in ("W0_state", "state_W0")}
    lab.check(f"KL_{name}", all(qr.close(a, b, CFG.rep_tol) for a, b in pairs.values()), pairs)


def read_one(lab: Size, name: str) -> None:
    """A state's whole reading, written step by step. A resume repeats only the step under way."""
    tail, cfg = f"state_{name}", lab.size_cfg
    if "read" in (out := lab.read(tail)):
        check_size(lab, name, out["base"])
        return
    lab.budget(BUILD_S)
    dW, out["identity"] = state_of(lab, name)
    rows = lab.maths(cfg.rows)
    lab.set_weights("FT", dW)
    read_cells(lab, tail, name, out, "base", rows, None)
    check_size(lab, name, out["base"])
    X, rec = split_x(lab, f"{cfg.prefix}_{name}_b{scale_of(lab, name)}", dW)  # b in the name: a new search, a new split
    out["X"] = out.get("X", {}) | rec
    change = lab.change(dW, X)
    read_cells(lab, tail, name, out, "file", rows, change)
    true_pass(lab, tail, name, out, rows, change)
    if "KL" not in out:
        lab.budget(CELLS_S)
        a, b = kl_both(lab, "FT", rows)
        out["KL"] = {"W0_state": a, "state_W0": b}
        lab.save(tail, out, out["KL"])
    if name in READ_BY_E57 and len(rows) == CFG.n_rows:
        check_kl(lab, name, out["KL"])
    every, draws = list(range(len(rows))), qr.resample(len(rows), CFG.boot)
    read = {v: boot_direct.read(rows_of(out, v), draws) | point(out, v, every) for v in ("true", "file")}
    read |= {"KL_state_W0": sum(out["KL"]["state_W0"]) / (len(rows) * ROW_LEN), "peaks": lab.peaks()}
    lab.save(tail, out | {"read": read}, read)
    lab.drop_set("FT")
    torch.cuda.empty_cache()


def stage_state(lab: Size) -> None:
    sel = lab.selected()
    big = [f"{k}_big" for k in KINDS if "b" in lab.read(f"search_{k}")]
    for name in dict.fromkeys((sel["final"], sel["small"]["name"], sel["big"]["name"], "gold_s0", *KINDS, *big)):
        read_one(lab, name)


def search_kind(lab: Size, kind: str, big: dict[str, float]) -> None:
    """Review M4: `scale_search` on the select rows, then all rows within the band; on a miss one more step; else unmatched."""
    tail, R, cfg = f"search_{kind}", lab.R, lab.size_cfg
    if lab.read(tail):
        return
    lab.budget(BUILD_S)
    lab.park_gamma()
    dW, rows = state_dw(lab, kind), lab.maths(cfg.rows)

    def f(b: float, rr: Tensor = rows[: cfg.select_rows]) -> float:  # a budget exit loses this kind's evaluations: minutes
        lab.budget(CELLS_S if len(rr) > cfg.select_rows else EVAL_S)
        lab.set_weights("FT", dW, b)
        return size_of(lab.cells(rr, tag=f"{kind} x {b:.6g}, target {big}"), list(range(len(rr))))

    s, target = qr.scale_search(f, big["size_select"], R["match"], R["search_evals"]), big["size"]
    b, x = s["b"], f(s["b"], rows)
    tries = [[b, x]]
    if not qr.within(x, target, R["band"]) and qr.ok(x) and x > 0:
        tries.append([b * math.sqrt(target / x), f(b * math.sqrt(target / x), rows)])
    b, x = min(tries, key=lambda e: abs(math.log(e[1] / target)) if qr.ok(e[1]) and e[1] > 0 else math.inf)
    lab.drop_set("FT")
    rec = {"search": s, "all_rows": tries, "b": b, "size": x, "target": target, "matched": qr.within(x, target, R["band"])}
    lab.save(tail, rec, rec)


def small_reading(sel: dict[str, Any], gold: dict[str, Any], w0: dict[str, Any]) -> dict[str, Any]:
    """The small state's drop and length, from `train.py --stage G --run gold_1e-5 --step T`, against fp32 W0 as the gold
    rule reads them. A state on the way: no 'learned' check (review S3 of the plan)."""
    step = sel["states"][sel["small"]["name"]]["step"]
    st = gold.get("states", {}).get(f"step_{step}", {})
    if "unprompted" not in st or "held_gold" not in st:
        sys.exit(f"refusing: the small state has no Stage G read: train.py --stage G --run gold_1e-5 --step {step}")
    return {"step": step, "called": "a state on the way", "drop": w0["held_gold"]["loss"] - st["held_gold"]["loss"],
            "length_ratio": st["unprompted"]["response_mean"] / w0["unprompted"]["response_mean"]}


def stage_verdict(lab: Size) -> None:
    sel = lab.selected()
    pick = {"big": sel["big"]["name"], "small": sel["small"]["name"], "small_in_band": sel["small"]["in_band"]}
    recs = {n: lab.read(f"state_{n}") for n in {sel["final"], pick["big"], pick["small"], "gold_s0", *KINDS, *BIGS}}
    if missing := sorted(n for n, r in recs.items() if "read" not in r):
        sys.exit(f"refusing: not read yet: {missing}")
    draws, f = qr.resample(lab.size_cfg.rows, CFG.boot), floor(recs["gold_s0"], recs[sel["final"]], lab.R)
    out = {"pick": pick, "final_size": sel["final_size"], "above_gold_s0": sel["above_gold_s0"], "floor": f, "decides": "true",
           "unmatched": [k for k in KINDS if not lab.read(f"search_{k}")["matched"]]}
    if lab.size_cfg.gold_rule:  # the rehearsal reads no G verdict
        out["small_state"] = small_reading(sel, lab.get(lab.size_cfg.gold_rule), lab.get("train_W")["states"]["w0"])
    lab.save("verdict", o := out | {v: decide(recs, pick, f["f"], v, draws, lab.R) for v in ("true", "file")}, o)


STAGES = {"control": stage_control, "select": stage_select, "state": stage_state, "verdict": stage_verdict,
          "search": lambda lab: [search_kind(lab, k, lab.selected()["big"]) for k in KINDS]}


def main() -> None:
    ap = argparse.ArgumentParser(description="E61 Job S: stages in cluster/qwen_recipe_size.sbatch")
    ap.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    ap.add_argument("--rows", type=int, default=128, choices=tuple(CONFIGS))
    ap.add_argument("--ladder", choices=tuple(LADDERS), help="the rehearsal's: whose saved states it reads")
    ap.add_argument("--states", type=int, nargs=2, help="the rehearsal's: two saved steps, the later is its final state")
    ap.add_argument("--budget-s", type=float, default=1500.0, help="exit 3 before this; 1500 suits debug's 30 min")
    a = ap.parse_args()
    base = CONFIGS[a.rows]
    cfg = replace(base, ladder=a.ladder or base.ladder, steps=tuple(a.states or base.steps))
    if not PRED.is_file() or (RESULTS / f"{cfg.prefix}_stopped.json").is_file():
        sys.exit(f"refusing to start: {PRED.name} is missing, or {cfg.prefix}_stopped.json exists (the lead decides)")
    lab = Size(a.stage, str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"),
               os.environ.get("COMMUTE_QWEN_DIR", ""), cfg, a.budget_s, json.loads(PRED.read_text())["rules"])
    with torch.no_grad():
        for s in ("control", "select", "state", "search", "state", "verdict") if a.stage == "all" else (a.stage,):
            STAGES[s](lab)
    lab.log(f"[{a.stage} done] {lab.peaks()}")


if __name__ == "__main__":
    main()
