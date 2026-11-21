"""P2.52: two scorecard baselines read on E237's five 96-key tasks x 3 seeds at E237's one common update size, beside
E237's matched random change, as extra columns on E237's five-task graph. Descriptive (the user asked for columns).
Plan reports/plans/2026-11-21_plan_p2-52-key-grid-baselines.md; expected values in results/predictions.json (the run
refuses unless it is committed unchanged). Nothing is trained but the crosscoders.

Items are E237's, built by E237's own code (experiments/key_grid/run.py, imported): '<state>_scaled' is E147's saved
update times E237's c for its task (results/stage0.json there), 'random_<state>' is E237's random direction for that
state (`World.random`) times the same c, and the do-nothing is the control state times 0. Each item's size is checked
against E237's stage1.json before it is read. Readings, each the function that wrote the scorecard's cell, imported:
  top-neuron share   E143's `Job.neuron` (holes_fill), k 64 and k matched, E147's k per block, on E143's read rows
  neuron share       E98's r_n (`neuron_r.state_cells`, `shares`), whole width, on E98's read rows
  crosscoder d       E203's `Job.run_side` (crosscoder run 2): one BatchTopK crosscoder per item, E203's settings
Stages, each writing as it goes; a rerun skips what is written:
  neuron I/N  (a) do-nothing: every gain and numerator exactly 0; (b) the control state at c = 1 repeats E147's saved
              readings; then items[I::N]                                                  -> <out>/neuron.json
  xc controls I/3   first and alone, one item a pod: 0 the do-nothing crosscoder (E203's stage 0a, on the control
              state's task text), 1 copy_s0 at c 1 and 2 prior_cipher_s0 at c 0.78, the two items E203 already read
              at the same state and c, whose d repeats E203's within its reseed bar (a stop) and within the filed
              surprise bar (reported loudly, not a stop)                         -> <out>/crosscoder/controls_I_3.json
  xc I/N      after all three controls pass: every N-th of the other 28 items from the I-th, each with E203's FVE
              and dead-share bars                                 -> <out>/crosscoder/sides/, shard_I_N.json
  --rehearse  E203's rehearsal crosscoder sizes; everything under results/rehearsal/

    PRED_SHA=<sha> python experiments/key_grid_baselines/run.py neuron 0/1
    PRED_SHA=<sha> python experiments/key_grid_baselines/run.py xc controls 0/3   (and 1/3, 2/3: one pod each)
    PRED_SHA=<sha> python experiments/key_grid_baselines/run.py xc 0/8            (to 7/8: one pod each)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import neuron_r as NR
from commute import steering as ST
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.open_pair import jsonable

KG: Any = load_experiment("key_grid", "run")  # E237: load_config, World.random, DO_NOTHING, fname, log/dump/stop; E205, R96
HF: Any = load_experiment("holes_fill", "run")  # E143: Job.neuron (the top-neuron share), lo_at
CCR: Any = load_experiment("crosscoder", "run")  # E203: Job, Side, stage0a, check, Stop, OUT, PRED
HERE = Path(__file__).resolve().parent
REPO, RESULTS = EXPERIMENTS.parent, HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E237, E147 = EXPERIMENTS / "key_grid" / "results", EXPERIMENTS / "refit_96key" / "results"
E143_PRED, E98_PRED = EXPERIMENTS / "holes_fill/results/predictions.json", EXPERIMENTS / "neuron_r/results/predictions.json"
E203 = CCR.PRED.parent  # experiments/crosscoder/results/run2
log, dump, stop, fname = KG.log, KG.dump, KG.stop, KG.fname


def read(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    neuron_ks: tuple[str, ...]  # E143's two widths
    projected_limit_min: float  # one crosscoder shard's projected wall time

    def __post_init__(self) -> None:
        assert self.neuron_ks == ("k64", "kmatched") and self.projected_limit_min > 0


class Items:
    """E237's items as weight changes (fp32 CPU, in the saved file's key order), with E237's sizes to check them."""

    def __init__(self) -> None:
        self.cfg = KG.load_config()[0]
        self.c, self.st1 = read(E237 / "stage0.json")["c"], read(E237 / "stage1.json")["states"]
        full = KG.R96.inventories(read(KG.E147_INV))[1]
        states = self.cfg.states()
        tasks = {full["states"][s]["task"] for s in states}
        self.inv = {"tasks": {t: full["tasks"][t] for t in tasks}, "states": {s: full["states"][s] for s in states}, "pairs": []}

    def task(self, state: str) -> str:
        return self.inv["states"][state]["task"]

    def saved(self, state: str) -> dict[str, Tensor]:
        d = torch.load(REPO / self.inv["states"][state]["file"], weights_only=True)
        return {k: v.float() for k, v in d.items()}

    def c_of(self, state: str) -> float:
        """E237's c for the state's task (stage0.json)."""
        return self.c[self.task(state).removeprefix("e58/")]

    def change(self, item: str) -> tuple[str, dict[str, Tensor], float]:
        """The item's state, its change and c, as E237 builds them (key_grid World.update, then E205.SM.times)."""
        if item == KG.DO_NOTHING:
            s = self.cfg.control_state
            return s, KG.E205.SM.times(self.saved(s), 0.0), 0.0
        s = item.removeprefix("random_").removesuffix("_scaled")
        c, d = self.c_of(s), self.saved(s)
        if item.startswith("random_"):
            d = KG.World.random(SimpleNamespace(cfg_e237=self.cfg), s, d)  # E237's draw: seed random_seed + index
        return s, KG.E205.SM.times(d, c), c

    def size(self, item: str, state: str, dW: dict[str, Tensor], W0: dict[str, Tensor], tol: float) -> float:
        """E187's size of the change; a scaled or random item must equal E237's stage-1 size of it (stops otherwise)."""
        got = KG.E205.sizes({f"{k}.weight": v for k, v in dW.items()}, W0)["whole"]
        if item != KG.DO_NOTHING:
            want = self.st1[state]["random" if item.startswith("random_") else "scaled"]["size"]
            if not abs(got / want - 1) <= tol:
                raise SystemExit(f"{item}: size {got} is not E237's {want} (rel tol {tol}): not E237's item")
        return got


def shard_items(items: Items, shard: tuple[int, int]) -> list[str]:
    return items.cfg.read_items()[shard[0] :: shard[1]]


# ---- the neuron readings ----------------------------------------------------------------------------------------------


class Neurons:
    """E143's top-neuron share and E98's whole-width neuron share, read as E147 read them on these states."""

    def __init__(self, items: Items, cfg: Config) -> None:
        self.items, self.ks, self.net = items, cfg.neuron_ks, ST.Net("cuda")
        hf = read(E143_PRED)
        self.hf = HF.Job(self.net, items.inv, hf["settings"], hf["thresholds"])
        k = read(E147 / "stage0.json")["holes_fill_stage0"]["k"]  # E143's k per block as E147's stage 0 set them
        self.hf.ks = {kn: {int(l): int(v) for l, v in k[kn].items()} for kn in self.ks}
        nr = read(E98_PRED)["settings"]
        self.batch, self.floor, self.reads = nr["batch"], nr["gain_floor"], dict[str, Any]()

    def whole(self, task: str, dW: dict[str, Tensor]) -> dict[str, Any]:
        """E98's cells and shares on the task's held-out windows (E147's `Run.neuron_r`, on a given change)."""
        if task not in self.reads:
            path = REPO / self.items.inv["tasks"][task]["path"]
            self.reads[task] = NR.task_reads(ST.load_task(self.net, path, task, task))
        nr = NR.shares(NR.state_cells(self.net, self.reads[task], dW, self.batch), self.floor)
        return {k: nr[k] for k in ("G", "num", "r_n", "readable", "identity", "telescope", "n_windows")}

    def read(self, item: str, tol: float) -> dict[str, Any]:
        t0, (state, dW, c) = time.time(), self.items.change(item)
        size = self.items.size(item, state, dW, self.net.W0, tol)
        task = self.items.task(state)
        t = self.hf.task(task)
        gain = float((HF.lo_at(self.net, t, t.read.toks, dW) - t.lo0).mean())  # E143's Job.state gain
        top = {kn: self.hf.neuron(t, dW, self.hf.ks[kn], gain) for kn in self.ks}
        return {"state": state, "task": task, "c": c, "size": size, "gain": gain, "top": top, "whole": self.whole(task, dW),
                "seconds": time.time() - t0}  # fmt: skip


def neuron_controls(nu: Neurons, th: dict[str, Any], out: dict[str, Any], path: Path) -> None:
    """(a) zero change: E98's G and numerators, E143's gain and numerators exactly 0 (E143's zero_tol); E98's identity and
    telescope within E147's bars. (b) the control state at c = 1 against E147's saved reading (E147's reader_neuron_tol)."""
    zero, t147 = nu.read(KG.DO_NOTHING, th["size_match_rel"]), read(E147 / "predictions.json")["thresholds"]
    exact = {"gain": zero["gain"], "G": zero["whole"]["G"], **{f"num {k}": v for k, v in zero["whole"]["num"].items()},
             **{f"top {k} numerator": v["numerator"] for k, v in zero["top"].items()}}  # fmt: skip
    out["do_nothing"] = zero | {"exact": exact}
    miss = [f"do-nothing {k} = {v}" for k, v in exact.items() if not abs(v) <= read(E143_PRED)["thresholds"]["zero_tol"]]
    ctrl = nu.items.cfg.control_state
    one, saved = nu.read(f"{ctrl}_scaled", th["size_match_rel"]), read(E147 / "stage1.json")["states"][ctrl]
    if one["c"] != 1.0:
        miss.append(f"the control state's c is {one['c']}, not 1: its item is not the saved update")
    pairs = {"r_n": (one["whole"]["r_n"], saved["neuron_r"]["r_n"]),
             **{k: (one["top"][k]["share"], saved["holes_fill"]["neuron"][k]["share"]) for k in nu.ks}}  # fmt: skip
    out["repeat"] = {k: {"got": a, "saved": b} for k, (a, b) in pairs.items()}
    miss += [f"repeat {k}: {a} vs E147's {b}" for k, (a, b) in pairs.items() if not abs(a - b) <= t147["reader_neuron_tol"]]
    log(f"neuron controls: do-nothing {exact}; repeat {out['repeat']}")
    if miss:
        stop(path, out, "neuron controls: " + "; ".join(miss))
    out["controls_pass"] = True
    dump(path, out)


def neuron(out_dir: Path, cfg: Config, th: dict[str, Any], items: Items, shard: tuple[int, int]) -> None:
    path = out_dir / f"neuron_{shard[0]}_{shard[1]}.json"
    out: dict[str, Any] = read(path) if path.is_file() else {"experiment": EXPERIMENT, "items": {}}
    nu = Neurons(items, cfg)
    t147 = read(E147 / "predictions.json")["thresholds"]
    if not out.get("controls_pass"):
        neuron_controls(nu, th, out, path)
    todo = [i for i in shard_items(items, shard) if i not in out["items"]]
    for k, item in enumerate(todo):
        rec = out["items"][item] = nu.read(item, th["size_match_rel"])
        w = rec["whole"]
        if not (w["identity"] <= t147["identity_tol"] and abs(w["telescope"]) <= t147["telescope_tol"]):
            stop(path, out, f"{item}: E98's identity {w['identity']:.3g} or telescope {w['telescope']:.3g} over E147's bars")
        dump(path, out)
        log(f"[neuron {item}] c {rec['c']:.6f} gain {rec['gain']:.4f} r_n {w['r_n']:.4f} "
            + " ".join(f"{kn} {v['share']:.4f} (num {v['numerator']:.4f})" for kn, v in rec["top"].items())
            + f"; {rec['seconds']:.0f} s; ITEM {k + 1} of {len(todo)} DONE")  # fmt: skip
    dump(path, out | {"done": True})
    log("NEURON DONE")


# ---- the crosscoder ---------------------------------------------------------------------------------------------------


def mkdirs(*dirs: Path) -> None:
    """Make the output directories. results/large is a symlink into ~/commute-data and arrives on a pod dangling: the
    runner replaces it with a real directory there; until then this stops naming it."""
    large = RESULTS / "large"
    if large.is_symlink() and not large.exists():
        raise SystemExit(f"{large} is a dangling symlink (to {large.readlink()}): on a pod, rm it and mkdir a real directory")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# The control stage: E203's stage 0a (the do-nothing crosscoder), then the two items E203 already read at the same state
# and c, whose d must repeat E203's saved side's (E203 experiments/crosscoder/results/run2/sides: e58__copy_s0 at c 1,
# e58__prior_cipher_s0__matched_e58_copy_prior_cipher at E237's prior_cipher c). One item per pod; `xc I/N` reads the rest.
CONTROLS = (KG.DO_NOTHING, "e58/copy_s0_scaled", "e58/prior_cipher_s0_scaled")
REPEATS = {"e58/copy_s0_scaled": ("e58/copy_s0", 1.0), "e58/prior_cipher_s0_scaled": ("e58/prior_cipher_s0", 0.7798162499214973)}


def repeat_refs(items: Items) -> dict[str, float]:
    """E203's saved d for each repeat item, matched by exact (state, c). Stops unless the matches between our scaled
    items and E203's sides are exactly REPEATS and both are there."""
    sides = {(s["state"], s["c"]): s["d"] for s in (read(p) for p in sorted((E203 / "sides").glob("e58__*.json")))}
    want = set(REPEATS.values())
    if missing := sorted(want - sides.keys()):
        raise SystemExit(f"E203's saved sides lack the repeat keys {missing}: the repeat control has no reference")
    got = {(s, items.c_of(s)) for s in items.cfg.states()} & sides.keys()
    if got != want:
        raise SystemExit(f"our items match E203's sides at {sorted(got)}, not the expected repeats {sorted(want)}")
    return {item: sides[key] for item, key in REPEATS.items()}


def crosscoder_job(items: Items, tag: str, out_dir: Path) -> tuple[Any, dict[str, Any]]:
    """E203's Job on E237's five tasks: its inventory is ours, its control task the control state's task, its outputs
    under out_dir. Everything else is E203's filed run 2 (settings and bars at `tag`)."""
    P = read(CCR.PRED)
    S = P["settings"][tag]
    cfg = CC.Cfg(**(S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])}))  # as E203's main builds it
    P = P | {"experiment": EXPERIMENT, "settings": P["settings"] | {"control_task": items.task(items.cfg.control_state)}}
    CCR.SM = SimpleNamespace(pairs=lambda: ({}, {}, items.inv))  # E203's Job loads the tasks and states SM.pairs() names
    CCR.OUT.update(results=out_dir / "crosscoder", sides=out_dir / "crosscoder" / "sides", large=out_dir.parent / "large" / out_dir.name / "crosscoder")
    mkdirs(*CCR.OUT.values())  # E203's main makes these; its Job does not
    return CCR.Job(P, cfg), P["thresholds"][tag]


def xc_todo(items: Items, shard: tuple[int, int], controls: bool) -> tuple[list[str], Path]:
    """`xc controls I/3`: CONTROLS[I]. `xc I/N`: every N-th of the other 28 items from the I-th (E237's order)."""
    if controls:
        assert shard[1] == len(CONTROLS), f"xc controls runs as I/{len(CONTROLS)}"
        return [CONTROLS[shard[0]]], CCR.OUT["results"] / f"controls_{shard[0]}_{shard[1]}.json"
    rest = [i for i in items.cfg.read_items() if i not in CONTROLS]
    return rest[shard[0] :: shard[1]], CCR.OUT["results"] / f"shard_{shard[0]}_{shard[1]}.json"


def repeat(item: str, d: float, d0: float, T: dict[str, Any], surprise: float) -> dict[str, Any]:
    """The repeat's record: it stops over E203's reseed bar; over the filed surprise bar it is reported loudly but runs
    on, since a repeat at the same seed on the same stream should differ only by GPU nondeterminism."""
    rec = {"d": d, "E203": d0, "diff": abs(d - d0), "stop_bar": T["reseed_d"], "surprise_bar": surprise, "surprise": abs(d - d0) > surprise}
    if rec["surprise"]:
        log(f"!!! SURPRISE: repeat {item} differs from E203's saved d by {rec['diff']:.4g} (> {surprise}): tell the lead !!!")
    return rec


def train_item(job: Any, T: dict[str, Any], th: dict[str, Any], items: Items, item: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """One crosscoder on one item, written as it lands, then E203's FVE and dead-share bars and, for a repeat item,
    the repeat's stop bar. ctx: out, path, fve_self, refs."""
    out, path = ctx["out"], ctx["path"]
    state, dW, c = items.change(item)
    if item in REPEATS and (state, c) != REPEATS[item]:
        raise SystemExit(f"{item}: state and c {(state, c)} are not the repeat's {REPEATS[item]}")
    size = items.size(item, state, dW, job.net.W0, th["size_match_rel"])
    dev = {n: v.to(job.net.W0[n].device) for n, v in dW.items()}
    rec = job.run_side(CCR.Side(fname(item), job.reads[items.task(state)], items.task(state), dev), False, False)
    rec |= {"item": item, "state": state, "c": c, "size": size}
    if item in REPEATS:
        rec["repeat"] = out.setdefault("repeat", {})[item] = repeat(item, rec["d"], ctx["refs"][item], T, th["repeat_surprise_d"])
    dump(CCR.OUT["sides"] / f"{fname(item)}.json", jsonable(rec))
    f = min(rec["fve_base"], rec["fve_ft"])
    CCR.check(out, path, f"FVE {item}", f >= ctx["fve_self"] - T["fve_drop"], {"fve": f, "self": ctx["fve_self"], "drop": T["fve_drop"]})
    CCR.check(out, path, f"dead share {item}", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
    if item in REPEATS:
        CCR.check(out, path, f"repeat {item}", rec["repeat"]["diff"] <= T["reseed_d"], rec["repeat"])
    return rec


def crosscoder(out_dir: Path, cfg: Config, th: dict[str, Any], items: Items, shard: tuple[int, int], tag: str, controls: bool) -> None:
    job, T = crosscoder_job(items, tag, out_dir)
    todo, path = xc_todo(items, shard, controls)
    refs = repeat_refs(items)
    if KG.DO_NOTHING in todo:
        s0 = CCR.OUT["results"] / "stage0a.json"
        if not (s0.is_file() and read(s0).get("pass")):
            log("[xc] the do-nothing: the base against itself on the control state's task text (E203's stage 0a)")
            CCR.stage0a(job, T, None)
        todo.remove(KG.DO_NOTHING)
    out: dict[str, Any] = read(path) if path.is_file() else {"experiment": EXPERIMENT, "controls": controls, "shard": list(shard)}
    if "stopped" in out:
        raise SystemExit(f"{path} stopped at {out['stopped']!r}: the lead decides before a rerun")
    out.pop("pass", None)
    ctx = {"out": out, "path": path, "fve_self": read(E203 / "stage1.json")["fve_self"], "refs": refs}  # E203's self FVE: every side's bar
    todo = [i for i in todo if not (CCR.OUT["sides"] / f"{fname(i)}.json").is_file()]
    log(f"[xc] {'controls' if controls else 'shard'} {shard[0]}/{shard[1]}: {len(todo)} crosscoders to train")
    t0 = time.time()
    for k, item in enumerate(todo):
        rec = train_item(job, T, th, items, item, ctx)
        minutes = (time.time() - t0) / 60 / (k + 1) * len(todo)
        log(f"[xc {item}] d {rec['d']:.4f} d_wide {rec['d_wide']:.4f} specific {rec['specific']:.4f}; ITEM {k + 1} of {len(todo)} DONE; projected minutes {minutes:.0f}")
        CCR.check(out, path, f"projected time after {item}", minutes <= cfg.projected_limit_min, {"minutes": minutes, "bar": cfg.projected_limit_min})
    out["pass"] = True
    dump(path, jsonable(out))
    log("XC CONTROLS DONE" if controls else "XC DONE")


def main(stage: str, shard: tuple[int, int], controls: bool, rehearse: bool) -> int:
    P = read(PRED)
    if not re.fullmatch(r"E\d+", EXPERIMENT):
        raise SystemExit(f"predictions.json experiment is {EXPERIMENT!r}: the lead claims the E-number (fleet/ecode) first")
    cfg, th = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}), P["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as E147 read and E203 trained
    out_dir = RESULTS / ("rehearsal" if rehearse else "run")
    mkdirs(out_dir)
    name = f"{stage}{' controls' if controls else ''} {shard[0]}/{shard[1]}"
    log(f"{EXPERIMENT} {name}{' (rehearsal)' if rehearse else ''}; settings {asdict(cfg)}; thresholds {th}")
    items = Items()
    try:
        if stage == "neuron":
            neuron(out_dir, cfg, th, items, shard)
        else:
            crosscoder(out_dir, cfg, th, items, shard, "rehearsal" if rehearse else "full", controls)
    except CCR.Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"FINISHED {name}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["neuron", "xc"])
    ap.add_argument("part", nargs="+", help="I/N: every N-th item from the I-th (E237's read_items order); xc also takes 'controls I/3'")
    ap.add_argument("--rehearse", action="store_true")
    a = ap.parse_args()
    require_filed(PRED)
    ctl = a.part[0] == "controls"
    if len(a.part) != 1 + ctl or (ctl and a.stage != "xc"):
        ap.error("give I/N, or for xc: controls I/3")
    i, n = (int(x) for x in a.part[-1].split("/"))
    code = 1
    try:
        code = main(a.stage, (i, n), ctl, a.rehearse)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, flush=True)
    hard_exit(code)
