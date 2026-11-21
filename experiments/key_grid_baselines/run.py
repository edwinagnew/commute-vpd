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
  xc I/N      shard N-1 first runs the do-nothing crosscoder (E203's stage 0a, on the control state's task text);
              then items[I::N], each with E203's FVE and dead-share bars; the two items E203 already read at the
              same state and c (copy_s0 at 1, prior_cipher_s0 at 0.78) repeat its d within E203's reseed bar
                                                                                          -> <out>/crosscoder/sides/
  --rehearse  E203's rehearsal crosscoder sizes; everything under results/rehearsal/

    PRED_SHA=<sha> python experiments/key_grid_baselines/run.py neuron 0/1
    PRED_SHA=<sha> python experiments/key_grid_baselines/run.py xc 0/4          (and 1/4, 2/4, 3/4: one pod each)
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

    def change(self, item: str) -> tuple[str, dict[str, Tensor], float]:
        """The item's state, its change and c, as E237 builds them (key_grid World.update, then E205.SM.times)."""
        if item == KG.DO_NOTHING:
            s = self.cfg.control_state
            return s, KG.E205.SM.times(self.saved(s), 0.0), 0.0
        s = item.removeprefix("random_").removesuffix("_scaled")
        c, d = self.c[self.task(s).removeprefix("e58/")], self.saved(s)
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


def crosscoder_job(items: Items, tag: str, out_dir: Path) -> tuple[Any, dict[str, Any]]:
    """E203's Job on E237's five tasks: its inventory is ours, its control task the control state's task, its outputs
    under out_dir. Everything else is E203's filed run 2 (settings and bars at `tag`)."""
    P = read(CCR.PRED)
    S = P["settings"][tag]
    cfg = CC.Cfg(**(S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])}))  # as E203's main builds it
    P = P | {"experiment": EXPERIMENT, "settings": P["settings"] | {"control_task": items.task(items.cfg.control_state)}}
    CCR.SM = SimpleNamespace(pairs=lambda: ({}, {}, items.inv))  # E203's Job loads the tasks and states SM.pairs() names
    CCR.OUT.update(results=out_dir / "crosscoder", sides=out_dir / "crosscoder" / "sides", large=out_dir.parent / "large" / out_dir.name / "crosscoder")
    for d in CCR.OUT.values():  # E203's main makes these; its Job does not
        d.mkdir(parents=True, exist_ok=True)
    return CCR.Job(P, cfg), P["thresholds"][tag]


def crosscoder(out_dir: Path, cfg: Config, th: dict[str, Any], items: Items, shard: tuple[int, int], tag: str) -> None:
    job, T = crosscoder_job(items, tag, out_dir)
    s0 = CCR.OUT["results"] / "stage0a.json"
    if shard[0] == shard[1] - 1 and not (s0.is_file() and read(s0).get("pass")):
        log("[xc] the do-nothing first: the base against itself on the control state's task text (E203's stage 0a)")
        CCR.stage0a(job, T, None)
    fve_self = read(E203 / "stage1.json")["fve_self"]  # E203's base-against-itself FVE, the bar every side is held to
    sides = [read(p) for p in sorted((E203 / "sides").glob("e58__*.json"))]
    ref = {(s["state"], s["c"]): s["d"] for s in sides}  # E203's sides at the same state and c: copy_s0 c 1, prior_cipher_s0 c 0.78
    path, out, t0 = CCR.OUT["results"] / f"shard_{shard[0]}_{shard[1]}.json", {"experiment": EXPERIMENT, "shard": list(shard)}, time.time()
    todo = [i for i in shard_items(items, shard) if not (CCR.OUT["sides"] / f"{fname(i)}.json").is_file()]
    log(f"[xc] shard {shard[0]}/{shard[1]}: {len(todo)} crosscoders to train")
    for k, item in enumerate(todo):
        state, dW, c = items.change(item)
        size = items.size(item, state, dW, job.net.W0, th["size_match_rel"])
        dev = {n: v.to(job.net.W0[n].device) for n, v in dW.items()}
        rec = job.run_side(CCR.Side(fname(item), job.reads[items.task(state)], items.task(state), dev), False, False)
        dump(CCR.OUT["sides"] / f"{fname(item)}.json", rec | {"item": item, "state": state, "c": c, "size": size})
        f = min(rec["fve_base"], rec["fve_ft"])
        CCR.check(out, path, f"FVE {item}", f >= fve_self - T["fve_drop"], {"fve": f, "self": fve_self, "drop": T["fve_drop"]})
        CCR.check(out, path, f"dead share {item}", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
        if not item.startswith("random_") and (state, c) in ref:
            d0 = ref[(state, c)]
            CCR.check(out, path, f"repeat {item}", abs(rec["d"] - d0) <= T["reseed_d"], {"d": rec["d"], "E203": d0, "bar": T["reseed_d"]})
        minutes = (time.time() - t0) / 60 / (k + 1) * len(todo)
        log(f"[xc {item}] d {rec['d']:.4f} d_wide {rec['d_wide']:.4f} specific {rec['specific']:.4f}; ITEM {k + 1} of {len(todo)} DONE; projected shard minutes {minutes:.0f}")
        CCR.check(out, path, f"projected time after {item}", minutes <= cfg.projected_limit_min, {"minutes": minutes, "bar": cfg.projected_limit_min})
    out["pass"] = True
    dump(path, out)
    log("XC DONE")


def main(stage: str, shard: tuple[int, int], rehearse: bool) -> int:
    P = read(PRED)
    if not re.fullmatch(r"E\d+", EXPERIMENT):
        raise SystemExit(f"predictions.json experiment is {EXPERIMENT!r}: the lead claims the E-number (fleet/ecode) first")
    cfg, th = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}), P["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as E147 read and E203 trained
    out_dir = RESULTS / ("rehearsal" if rehearse else "run")
    log(f"{EXPERIMENT} {stage} {shard[0]}/{shard[1]}{' (rehearsal)' if rehearse else ''}; settings {asdict(cfg)}; thresholds {th}")
    items = Items()
    try:
        if stage == "neuron":
            neuron(out_dir, cfg, th, items, shard)
        else:
            crosscoder(out_dir, cfg, th, items, shard, "rehearsal" if rehearse else "full")
    except CCR.Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"FINISHED {stage} {shard[0]}/{shard[1]}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["neuron", "xc"])
    ap.add_argument("shard", help="I/N: every N-th item from the I-th (E237's read_items order)")
    ap.add_argument("--rehearse", action="store_true")
    a = ap.parse_args()
    require_filed(PRED)
    i, n = (int(x) for x in a.shard.split("/"))
    code = 1
    try:
        code = main(a.stage, (i, n), a.rehearse)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, flush=True)
    hard_exit(code)
