"""Cell P3.13 run 3 (E225, rung L3.2): run 2's held-model reading S on the two word pairs (copy/cipher, known/invented),
with a fit that reproduces M on the read rows. Plan: reports/plans/2026-11-20_plan_p3-13-run3-fit-floor.md; card
card3.md; settings, expected values and rules in results/run3/predictions.json (the run refuses unless it is committed
unchanged, or PRED_SHA matches on a pod). A description: no update is called reused or new.

S is run 2's reading (`commute.qwen_held.Held`, arithmetic unchanged); only the fit rows change. Read rows are always
rows [n_fit, n_rows) of the arm's own 128 rows. A fit (settings "fits") is the arm's own rows [0, own), gold's rows
[0, gold), and for each repack seed the arm's trained cue sequences packed again with that seed (128 rows each).

Stage 0, alone, before any state: per row set of the pod (the larger side's first), per fit in filed order, F = M read
for KL(F||M) (must be <= zero_kl) and KL(F||H) on the read rows (control/<rowset>_fit_<name>.json). The pair's fit is
the first after run 2's own64 whose KL(F||H) <= floor_max on both row sets (`choose_fit`, control/choice.json); if none,
the pod stops (stopped.json): the pair is unreadable by S at these fits, a filed outcome.
Stage 1: as run 2 with the chosen fit. Per row set, S 'no change' under s_floor_factor x stage 0's KL(F||H) of the
chosen fit there; before the pod's first state, its o_proj and down_proj change alone reads S >= 1 - s_tol. Each state
(both sides, seeds 0-2, saves last and 5x) at saved size and its Gaussian change; the larger side also times E187's c.
Records go to results/run3/<pair>/S/<key>[_c][_random].json; a rerun skips what is written.

  PRED_SHA=<sha> python experiments/qwen_held_readings/run3.py --pair facts --code E225 2>&1 | tee -a <pod log>
  python experiments/qwen_held_readings/run3.py --pair copy_cipher --code E225 --limit 1 --out <dir>   (the rehearsal)
  python experiments/qwen_held_readings/summary3.py    (no GPU: results/run3/summary.json)
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
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
PRED = HERE / "results" / "run3" / "predictions.json"
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
    n_fit: int  # read rows are [n_fit, n_rows) of the arm's own rows; every fit takes own rows [0, n_fit)
    batch: int  # rows per forward
    s_tol: float  # output-only control: 1 - S at most this
    zero_kl: float  # do-nothing: KL(F||M) per position at most this
    s_floor_factor: float  # S 'no change' when KL(F||M) <= this x the row set's chosen-fit do-nothing KL(F||H)
    s_draw: int
    smallest_gap: float  # the rule's bar (summary2.gap_rule); read there, not here
    spread: float
    fits: list[dict[str, Any]]  # {name, own, gold, repack_seeds}, in filed order; fits[0] is run 2's, never chosen
    floor_max: float  # a fit is chosen when its do-nothing KL(F||H) is at most this on both row sets

    def __post_init__(self) -> None:
        assert set(self.run_of) == {a for p in self.pairs.values() for a in p}
        assert all(len(p) == 2 for p in self.pairs.values())
        assert set(self.rescale) == {f"{p}/{s}" for p in self.pairs for s in self.saves}
        assert all(r["larger"] in self.pairs[k.split("/")[0]] and 0 < r["c"] < 1 for k, r in self.rescale.items())
        assert self.smallest_gap >= self.spread > 0
        assert 0 < self.n_fit < self.n_rows and self.n_fit % self.batch == 0 and (self.n_rows - self.n_fit) % self.batch == 0
        f0 = self.fits[0]
        assert f0["name"] == "own64" and f0["gold"] == 0 and not f0["repack_seeds"], f0
        assert len({f["name"] for f in self.fits}) == len(self.fits) > 1
        assert all(f["own"] == self.n_fit and 0 <= f["gold"] <= self.n_rows and f["gold"] % self.batch == 0
                   for f in self.fits), self.fits
        assert 0 < self.floor_max


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


# ----------------------------------------------------------------------------- rows, the fit choice


def rowsets(pair: str) -> list[str]:
    """The pod's row sets (each word arm reads its own rows), the larger side at the first save first."""
    big = C.rescale[f"{pair}/{C.saves[0]}"]["larger"]
    return [big, *(a for a in C.pairs[pair] if a != big)]


@functools.cache
def rows(rs: str, seed: int | None = None) -> Tensor:
    """A row set as the library built it (seed None), or its sequences repacked with `seed`; 128 x 256 or the pod stops."""
    t = SR.rows_of(rs) if seed is None else SR.repacked(rs, seed)
    check(f"rows_{rs}_{seed}", tuple(t.shape) == (C.n_rows, C.row_len), {"shape": list(t.shape)})
    return t


def fit_rows(rs: str, fit: dict[str, Any]) -> Tensor:
    return torch.cat([rows(rs)[: fit["own"]], rows("gold")[: fit["gold"]], *(rows(rs, s) for s in fit["repack_seeds"])])


def choose_fit(floors: dict[str, dict[str, float]], order: list[str], sets: list[str], floor_max: float) -> str | None:
    """The first fit after order[0] whose do-nothing KL(F||H) (floors[fit][rowset]) is finite and <= floor_max on every
    row set of `sets`; None when no fit does. order[0] (run 2's fit) is measured only, never chosen."""
    for name in order[1:]:
        if all(math.isfinite(floors[name][rs]) and floors[name][rs] <= floor_max for rs in sets):
            return name
    return None


# ----------------------------------------------------------------------------- S


class Reader:
    """M and F loaded once; one Held (Grams on the host) at a time, for one row set and one fit."""

    def __init__(self) -> None:
        self.M, self.F = load_target(DEV), load_target(DEV)
        self.h: QH.Held | None = None
        self.at_key, self.floor = ("", ""), 0.0

    def held(self, rs: str, fit: dict[str, Any]) -> None:
        if (rs, fit["name"]) == self.at_key:
            return
        self.h = None
        gc.collect()
        torch.cuda.empty_cache()
        fit_t, read_t = fit_rows(rs, fit).to(DEV), rows(rs)[C.n_fit :].to(DEV)
        log(f"[S] rows {rs}, fit {fit['name']}: {len(fit_t)} fit rows, {len(read_t)} read rows")
        self.h = QH.Held(self.M, self.F, list(fit_t.split(C.batch)), list(read_t.split(C.batch)), log, store="cpu")
        self.at_key = (rs, fit["name"])

    def read(self, name: str, dW: dict[str, Tensor] | None, floor: float, read_residual: bool,
             extra: dict[str, Any]) -> dict[str, Any]:
        assert self.h is not None
        rec = self.h.read(dW, floor, read_residual) | extra | {"rows": self.at_key[0], "fit": self.at_key[1]}
        log(f"[S] {name}: S {rec['S']}, KL(F||M) {rec['kl_fm']:.6g}, KL(F||H) {rec['kl_fh']:.6g}, {rec['seconds']:.0f} s")
        return put(name, rec)


def sweep(r: Reader, pair: str) -> str:
    """Stage 0: F = M on every fit of every row set of the pod, then the pair's fit (or the pod stops)."""
    for rs in rowsets(pair):
        for fit in C.fits:
            name = f"control/{rs}_fit_{fit['name']}"
            if not (rec := get(name)):
                r.held(rs, fit)
                rec = r.read(name, None, C.zero_kl, True, {"fit_rows": len(fit_rows(rs, fit))})
            check(f"S_do_nothing_{rs}_{fit['name']}", rec["kl_fm"] <= C.zero_kl and rec["S"] is None,
                  {k: rec[k] for k in ("kl_fm", "kl_fh", "S", "fit_rows")})
    r.h, r.at_key = None, ("", "")
    order, sets = [f["name"] for f in C.fits], rowsets(pair)
    floors = {f: {rs: float(get(f"control/{rs}_fit_{f}")["kl_fh"]) for rs in sets} for f in order}
    chosen = choose_fit(floors, order, sets, C.floor_max)
    put("control/choice", {"chosen": chosen, "floors": floors, "floor_max": C.floor_max})
    check("fit_sweep", chosen is not None, {"floors": floors, "floor_max": C.floor_max})
    assert chosen is not None
    log(f"STAGE 0 DONE: fit {chosen}, floors {json.dumps(floors[chosen])}")
    return chosen


def order(pair: str) -> list[LS.State]:
    """The pod's states: the larger side at the first save first, then the other side; saves, then seeds."""
    return [LS.state(C.run_of[a], a, s, save) for a in rowsets(pair) for save in C.saves for s in C.seeds]


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
    r = Reader()
    chosen = sweep(r, pair)
    fit = next(f for f in C.fits if f["name"] == chosen)
    every = order(pair)[:limit]
    todo = [st for st in every if not all(get(n) for n in names(pair, st))]
    log(f"[pod] {pair}: fit {chosen}; {len(every)} states, {len(todo)} to read, "
        f"{sum(len(names(pair, s)) for s in every)} S reads; order {[s.key for s in every]}")
    for st, path in streamed(todo, root):
        if (st.arm, chosen) != r.at_key:
            r.held(st.arm, fit)
            r.floor = C.s_floor_factor * float(get(f"control/{st.arm}_fit_{chosen}")["kl_fh"])
            log(f"[S] rows {st.arm}: floor {r.floor:.6g} per position; peak GPU "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        if st == every[0]:  # the pod's first state: its output-matrices-only control, alone, before any state's read
            dW = load(path)
            output_only(r, st, dW)
            del dW
        read_state(r, pair, st, path)
        log(f"[pod] {st.key} done; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    log(f"STAGE 1 DONE {pair}")


def main() -> None:
    global C, OUT
    ap = argparse.ArgumentParser(description="P3.13 run 3: the fit sweep, then S of one word pair a pod")
    ap.add_argument("--pair", required=True, choices=("copy_cipher", "facts"))
    ap.add_argument("--code", required=True, help="the experiment number filed in results/run3/predictions.json")
    ap.add_argument("--limit", type=int, help="read only the pod's first N states (the rehearsal: 1)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "run3", help="records go to <out>/<pair>/")
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
        sys.exit(f"refusing to start: {OUT / 'stopped.json'} exists (a control missed or no fit cleared; the lead decides)")
    fp32_matmuls()
    with torch.no_grad():
        run(a.pair, a.fetch_to, a.limit)
    hard_exit(0)


if __name__ == "__main__":
    main()
