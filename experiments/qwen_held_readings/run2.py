"""Cell P3.13 run 2 (E???, rung L3.2): the held-model reading S of the Qwen3-1.7B pair library (E164/E165), one pod per
pair. Plan: reports/plans/2026-11-19_plan_l3-2-qwen-readings.md (+ _appendix.md); card card2.md; settings, expected
values and rules in results/run2/predictions.json (the run refuses unless it is committed unchanged, or PRED_SHA matches
on a pod). A description: no update is called reused or new.

S is run 1's reading (`commute.qwen_held.Held`, the same arithmetic): H = M with each layer's o_proj and down_proj
replaced by the least-squares map from M's units to F's write, fitted on rows 0-63 of the arm's own 128 rows, KL read on
rows 64-127. Each library state (both sides of the pair, seeds 0-2, the late "last" and matched "5x" saves P3.9 and E187
read) is read at its saved size beside a Gaussian change at its own per-matrix norms (run 1's S_random); a state of the
pair's larger side at that save (E187's) is read again times E187's c, beside the same Gaussian times c.

Order: the pair's larger side at the late save first, then the other side; within a side, saves then seeds. Row sets are
read one at a time (both gold arms share gold's rows): M's Grams are built per row set, kept on the host between reads.
Controls, alone, before any state of a row set, stopping on a miss (control/<rowset>_do_nothing.json, stopped.json):
F = M reads KL(F||M) <= zero_kl and S 'no change', and its KL(F||H) x s_floor_factor is that row set's floor; before the
pod's first state, that state's change at o_proj and down_proj only reads S >= 1 - s_tol (control/output_only.json).
Each record is written to results/run2/<pair>/S/<key>[_c][_random].json and printed as it finishes; a rerun skips what
is written. States come from W&B (`library_states.fetch`, sha256 against the library), the next fetched while this one
is read, each deleted once read. No decomposition is read: only the base model (HF, pinned revision) and the states.

  PRED_SHA=<sha> python experiments/qwen_held_readings/run2.py --pair facts --code E??? 2>&1 | tee -a <pod log>
  python experiments/qwen_held_readings/run2.py --pair facts --code E??? --limit 1 --out <dir>   (the rehearsal)
  python experiments/qwen_held_readings/summary2.py    (no GPU: results/run2/summary.json)
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import library_states as LS
from commute import qwen_held as QH
from commute import share_reading as SR
from commute.jobs import config_from_settings, hard_exit, require_filed
from commute.qwen_sft import load
from commute.qwen_vpd import SITES, fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "run2" / "predictions.json"
DEV = "cuda"


@dataclass(frozen=True)
class Config:  # predictions.json "settings"
    pairs: dict[str, list[str]]  # pair -> [reuse side, new side]
    run_of: dict[str, str]  # arm -> the library's results folder (library_states.state)
    seeds: list[int]
    saves: list[str]
    rescale: dict[str, dict[str, Any]]  # "<pair>/<save>" -> {"larger": arm, "c": E187's factor}
    n_rows: int
    row_len: int
    n_fit: int  # rows [0, n_fit) fit, [n_fit, n_rows) read
    batch: int  # rows per forward
    s_tol: float  # output-only control: 1 - S at most this
    zero_kl: float  # do-nothing: KL(F||M) per position at most this
    s_floor_factor: float  # S 'no change' when KL(F||M) <= this x the row set's do-nothing KL(F||H)
    s_draw: int
    smallest_gap: float  # the rule's bar on each seed's gap (summary2.gap_rule); read there, not here
    spread: float  # run 1's seed sd of S, reported beside the bar

    def __post_init__(self) -> None:
        assert set(self.run_of) == {a for p in self.pairs.values() for a in p}
        assert all(len(p) == 2 for p in self.pairs.values())
        assert set(self.rescale) == {f"{p}/{s}" for p in self.pairs for s in self.saves}
        assert all(r["larger"] in self.pairs[k.split("/")[0]] and 0 < r["c"] < 1 for k, r in self.rescale.items())
        assert self.smallest_gap >= self.spread > 0  # the bar is beyond run 1's seed spread
        assert 0 < self.n_fit < self.n_rows and self.n_fit % self.batch == 0 and (self.n_rows - self.n_fit) % self.batch == 0


C: Config
OUT: Path


# ----------------------------------------------------------------------------- files, stops


def log(msg: str) -> None:
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with (OUT / "run.log").open("a") as f:
        f.write(msg + "\n")


def get(name: str) -> dict[str, Any]:
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def put(name: str, rec: dict[str, Any]) -> dict[str, Any]:
    p = OUT / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.replace(p)
    log(f"[wrote] {p}")
    return rec


def check(name: str, holds: bool, detail: dict[str, Any]) -> None:
    log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        put("stopped", {"name": name, "detail": detail})
        log(f"STOP {name}")
        hard_exit(1)


# ----------------------------------------------------------------------------- what a pod reads


def rowset(arm: str) -> str:
    return "gold" if arm.startswith("gold") else arm  # the library read both gold arms on gold's rows


def order(pair: str) -> list[LS.State]:
    """The pod's states: the larger side at the first save first, then the other side; saves, then seeds."""
    big = C.rescale[f"{pair}/{C.saves[0]}"]["larger"]
    arms = [big, *(a for a in C.pairs[pair] if a != big)]
    return [LS.state(C.run_of[a], a, s, save) for a in arms for save in C.saves for s in C.seeds]


def scale_of(pair: str, st: LS.State) -> float | None:
    """E187's c when the state is on its pair's larger side at its save, else None (read at saved size only)."""
    r = C.rescale[f"{pair}/{st.save}"]
    return float(r["c"]) if r["larger"] == st.arm else None


def names(pair: str, st: LS.State) -> list[str]:
    ends = ["", "_random"] + (["_c", "_c_random"] if scale_of(pair, st) is not None else [])
    return [f"S/{st.key}{e}" for e in ends]


def streamed(todo: list[LS.State], root: Path) -> Iterator[tuple[LS.State, Path]]:
    """Each state's file in order, the next one fetched while the caller reads this one; deleted once read."""
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(LS.fetch, todo[0], root, log) if todo else None
        for i, st in enumerate(todo):
            assert nxt is not None
            path = nxt.result()
            nxt = pool.submit(LS.fetch, todo[i + 1], root, log) if i + 1 < len(todo) else None
            yield st, path
            shutil.rmtree(path.parent)


# ----------------------------------------------------------------------------- S


class Reader:
    """M and F loaded once; M's Grams for one row set at a time (`at`), with its do-nothing control and floor."""

    def __init__(self) -> None:
        self.M, self.F = load_target(DEV), load_target(DEV)
        self.h: QH.Held | None = None
        self.rows, self.floor = "", 0.0

    def at(self, rs: str) -> None:
        if rs == self.rows:
            return
        self.h = None
        gc.collect()
        torch.cuda.empty_cache()
        rows = SR.rows_of(rs)
        assert tuple(rows.shape) == (C.n_rows, C.row_len), (rs, rows.shape)
        fit, read = rows[: C.n_fit].to(DEV).split(C.batch), rows[C.n_fit :].to(DEV).split(C.batch)
        self.h, self.rows = QH.Held(self.M, self.F, list(fit), list(read), log, store="cpu"), rs
        rec = get(f"control/{rs}_do_nothing") or self.read(f"control/{rs}_do_nothing", None, C.zero_kl, True, {})
        check(f"S_do_nothing_{rs}", rec["kl_fm"] <= C.zero_kl and rec["S"] is None,
              {k: rec[k] for k in ("kl_fm", "kl_fh", "S")})
        self.floor = C.s_floor_factor * rec["kl_fh"]
        log(f"[S] rows {rs}: floor {self.floor:.6g} per position; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    def read(self, name: str, dW: dict[str, Tensor] | None, floor: float, read_residual: bool,
             extra: dict[str, Any]) -> dict[str, Any]:
        assert self.h is not None
        rec = self.h.read(dW, floor, read_residual) | extra | {"rows": self.rows}
        log(f"[S] {name}: S {rec['S']}, KL(F||M) {rec['kl_fm']:.6g}, KL(F||H) {rec['kl_fh']:.6g}, {rec['seconds']:.0f} s")
        return put(name, rec)


def output_only(r: Reader, st: LS.State, dW: dict[str, Tensor]) -> None:
    """The pod's first state's change at o_proj and down_proj only: reading 3 expresses it exactly, so S = 1."""
    if not (rec := get("control/output_only")):
        part = {s: (m if s in QH.SITES else torch.zeros_like(m)) for s, m in dW.items()}
        rec = r.read("control/output_only", part, r.floor, True, {"from": st.key})
    check("S_output_only", rec["S"] is not None and 1 - rec["S"] <= C.s_tol, {"S": rec["S"], "kl_fm": rec["kl_fm"]})


def read_state(r: Reader, pair: str, st: LS.State, path: Path) -> None:
    """The state at saved size, then times c (larger side only); then the Gaussian at its norms, then times c."""
    c, base = scale_of(pair, st), {"pair": pair, "arm": st.arm, "seed": st.seed, "save": st.save, "step": st.step,
                                   "artifact": st.artifact, "sha256": st.sha256, "dw_norm": st.dw_norm}
    dW = load(path)
    assert set(dW) == set(SITES), path  # linear matrices only: F's norms and embeddings are W0's
    norms = {s: float(m.double().norm()) for s, m in dW.items()}
    shapes = {s: tuple(m.shape) for s, m in dW.items()}
    if not get(f"S/{st.key}"):
        r.read(f"S/{st.key}", dW, r.floor, False, base | {"scale": 1.0})
    if c is not None and not get(f"S/{st.key}_c"):
        r.read(f"S/{st.key}_c", {s: m.mul_(c) for s, m in dW.items()}, r.floor, False, base | {"scale": c})
    del dW
    if get(f"S/{st.key}_random") and (c is None or get(f"S/{st.key}_c_random")):
        return
    R = SR.gaussian(norms, shapes, C.s_draw)
    if not get(f"S/{st.key}_random"):
        r.read(f"S/{st.key}_random", R, r.floor, True, base | {"scale": 1.0, "draw": C.s_draw})
    if c is not None and not get(f"S/{st.key}_c_random"):
        r.read(f"S/{st.key}_c_random", {s: m.mul_(c) for s, m in R.items()}, r.floor, True,
               base | {"scale": c, "draw": C.s_draw})


def run(pair: str, root: Path, limit: int | None) -> None:
    every = order(pair)[:limit]
    todo = [st for st in every if not all(get(n) for n in names(pair, st))]
    log(f"[pod] {pair}: {len(every)} states, {len(todo)} to read, {sum(len(names(pair, s)) for s in every)} S reads; "
        f"order {[s.key for s in every]}")
    r = Reader()
    for st, path in streamed(todo, root):
        r.at(rowset(st.arm))
        if st == every[0]:  # the pod's first state: its output-matrices-only control, alone, before any state's read
            dW = load(path)
            output_only(r, st, dW)
            del dW
            log("STAGE 0 DONE")
        read_state(r, pair, st, path)
        log(f"[pod] {st.key} done; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    log(f"STAGE 1 DONE {pair}")


def main() -> None:
    global C, OUT
    ap = argparse.ArgumentParser(description="P3.13 run 2: S of the pair library, one pair a pod")
    ap.add_argument("--pair", required=True, choices=("copy_cipher", "facts", "gold"))
    ap.add_argument("--code", required=True, help="the experiment number filed in results/run2/predictions.json")
    ap.add_argument("--limit", type=int, help="read only the pod's first N states (the rehearsal: 1)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "run2", help="records go to <out>/<pair>/")
    ap.add_argument("--fetch-to", type=Path, default=Path("/root/states"), help="where states land, two at a time")
    a = ap.parse_args()
    P = json.loads(PRED.read_text())
    C = config_from_settings(Config, P["settings"])
    if P["experiment"] == "E???" or a.code != P["experiment"]:
        sys.exit(f"refusing to start: --code {a.code}, predictions.json files {P['experiment']}")
    require_filed(PRED)
    OUT = a.out / a.pair
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "stopped.json").is_file():
        sys.exit(f"refusing to start: {OUT / 'stopped.json'} exists (a control missed; the lead decides)")
    fp32_matmuls()
    with torch.no_grad():
        run(a.pair, a.fetch_to, a.limit)
    hard_exit(0)


if __name__ == "__main__":
    main()
