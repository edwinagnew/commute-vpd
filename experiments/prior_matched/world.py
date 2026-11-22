"""Shared by run_a.py and run_b.py (cell P2.57, prior-matched pairs): the world every reading runs in, E147's
fine-tune, the stage-0 controls, one item's readings and the pair rule. Every reader is imported from the experiment
that wrote it and called unchanged: P2.54's world (`prior_mech_2x2/train.py`: run 1's Reader over E147's inventory,
holes_fill's Job for the neuron and steering shares, `weights` for E228's re-gating share r, E187's size and the
effective rank), P2.54's k-head switch-off (`prior_mech_2x2/khead.py`, through `cued_copy`'s KJob), E133's prior path
(`prior_vs_share.forward_prior`). Nothing here loads a model at import.

Items: a state 'e58/<task>_s<seed>' is its saved update times rd.c[<task>] (run 1's `Reader.scale`, E237's common
size); 'random/<state>' a Gaussian at the state's per-matrix norms (`finetune.random_direction`, seed random_seed +
the state's index in `order`) times the same c; 'zero/e58/<task>' the do-nothing, an update of zeros on the task's
read rows.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import keydep as KD
from commute import seed_rank as SR
from commute.jobs import EXPERIMENTS, load_experiment, stop, write_stage
from commute.open_pair import jsonable

T2: Any = load_experiment("prior_mech_2x2", "train")  # P2.54 run 2: World, shares, weights, controls, num, E205
KH: Any = load_experiment("prior_mech_2x2", "khead")  # P2.54 run 3: read_item, control_m, load_config
CC: Any = load_experiment("cued_copy", "run")  # KJob: khead's Job on a given world
R1, PVS, E205 = T2.R1, T2.PVS, T2.E205
X = EXPERIMENTS
RESULTS = Path(__file__).resolve().parent / "results"
LARGE = RESULTS / "large"
E147_INV = json.loads((X / "refit_96key" / "results" / "inventory.json").read_text())
E237 = X / "key_grid" / "results"
STEP = 30  # E147's read step
SHAPES = "e58/copy_s0"  # the zero update takes this saved state's shapes
READINGS = ("head_share", "k3_share", "r", "erank", "steering_pooled", "steering_perkey", "neuron_k64", "neuron_kmatched")
GAIN_READ = tuple(p for p in READINGS if p != "erank")  # read over the update's gain: nan under the reader's gain gate
log = R1.log


def rel(p: Path) -> str:
    return str(p.relative_to(X.parent))


def read_json(p: Path) -> Any:
    return json.loads(p.read_text())


class World(T2.World):
    """P2.54's world with this run's tasks and states in the inventory (see the module docstring)."""

    def __init__(self, random_seed: int) -> None:
        super().__init__(T2.load_config()[0])
        self.random_seed, self.order = random_seed, list[str]()

    def add_task(self, task: str, windows: str) -> str:
        """The task 'e58/<task>' on a windows file in E147's form (train and held-out splits, the table); its zero."""
        inv = self.rd.inv
        inv["tasks"][f"e58/{task}"] = {"format": "kv", "path": windows, "batch": 64}
        inv["states"][f"zero/e58/{task}"] = {"task": f"e58/{task}", "file": inv["states"][SHAPES]["file"]}
        return f"zero/e58/{task}"

    def add_state(self, task: str, seed: int) -> str:
        st = f"e58/{task}_s{seed}"
        self.rd.inv["states"][st] = {"task": f"e58/{task}", "file": rel(LARGE / f"dW_{task}_s{seed}.pt")}
        self.rd.inv["states"][f"random/{st}"] = {"task": f"e58/{task}", "file": None}
        self.order.append(st)
        return st

    def dw(self, item: str) -> dict[str, Tensor]:
        if item not in self.dws:
            state = item.removeprefix("random/")
            d = {k: v.float() for k, v in torch.load(X.parent / self.rd.inv["states"][state]["file"], weights_only=False).items()}
            if item.startswith("random/"):
                d = FT.random_direction(d, self.random_seed + self.order.index(state))
            c = 0.0 if item.startswith("zero/") else self.rd.scale(state)
            self.dws[item] = {k: v * c for k, v in d.items()}
        return self.dws[item]


def k_job(w: World) -> Any:
    return CC.KJob(w)


# ---- training -------------------------------------------------------------------------------------------------------


def train(w: World, task: str, windows: str, seeds: tuple[int, ...], path: Path, p_learn: float) -> dict[str, Any]:
    """E147's call per seed (`keydep.finetune` on the windows file, read at step 30), saved in bf16 as E147 saved it;
    held-out P(target) at step 30 (exit 1 under p_learn), E187's size of the saved file, its sha256. A rerun keeps the
    seeds already written."""
    out: dict[str, Any] = read_json(path) if path.is_file() else {"states": {}}
    win = torch.load(X.parent / windows, weights_only=False)
    LARGE.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        st = f"e58/{task}_s{s}"
        if st in out["states"]:
            continue
        t0, f = time.time(), X.parent / w.rd.inv["states"][st]["file"]
        ft = KD.finetune(KD.Ctx(w.net, {}, {}, {}, LARGE), win, s, (STEP,))  # keydep.finetune reads only ctx.net
        d = {k: v.bfloat16() for k, v in ft.deltas[STEP].items()}
        torch.save(d, f)
        size = E205.sizes({f"{k}.weight": v.float() for k, v in d.items()}, w.net.W0)["whole"]
        out["states"][st] = {"p_heldout": ft.evals[STEP]["p_o"], "size": size, "sha256": KD.sha256(f), "seconds": time.time() - t0}
        log(f"[train {st}] held-out P(target) at step {STEP} {out['states'][st]['p_heldout']:.4f}; size {size:.6g}; {time.time() - t0:.0f} s")
        write_stage(path, jsonable(out))
        if not out["states"][st]["p_heldout"] >= p_learn:
            stop(path, out, f"{st} does not learn: P(target) {out['states'][st]['p_heldout']:.4f} < {p_learn}", log)
    return out


def common_c(sizes: list[float]) -> float:
    """E237's c: its common size (the smallest task mean, experiments/key_grid/results/stage0.json) over this task's
    mean size over seeds."""
    return read_json(E237 / "stage0.json")["target"] / fmean(sizes)


# ---- stage 0: the controls, alone; the do-nothing -----------------------------------------------------------------------


def controls(w: World, job: Any, out_dir: Path) -> dict[str, Any]:
    """Run 1's control (M's second-copy top-1 falls with (2,4) off, holds with random single heads off; a zero update
    reads F = M exactly), khead's M control (the top k off, its k = 1 repeats E62's 0.0095), the prior path repeats
    E133's copy prior and holes_fill's readers repeat E147's shares on e58/copy_s0 at c = 1 (P2.54 run 2's `controls`).
    Each exits 1 on a miss."""
    path = out_dir / "control.json"
    if path.is_file() and read_json(path).get("pass"):
        log("controls passed before")
        return read_json(path)
    R1.OUT = out_dir / "head"  # run 1's control writes its record here
    R1.OUT.mkdir(parents=True, exist_ok=True)
    R1.control(w.rd, R1.load_config()[1])
    out: dict[str, Any] = {"head": read_json(R1.OUT / "control.json"), "khead": KH.control_m(job, KH.load_config()[1])}
    if miss := T2.controls(w, T2.load_config()[1], out):
        stop(path, jsonable(out), "controls: " + "; ".join(miss), log)
    write_stage(path, jsonable(out | {"pass": True}))
    return out


def do_nothing(w: World, job: Any, item: str, out_dir: Path) -> dict[str, Any]:
    """The zero update through every reader on the task's rows: the k-head gain and every G_off, holes_fill's gain,
    r's gain and the size exactly 0; every reading over the gain nan (exit 1 otherwise)."""
    rec = read_item(w, job, item, 0, out_dir)
    k = rec["khead"]
    exact = {"G": k["G"], "hf_gain": rec["hf_gain"], "r_gain": rec["r_gain"], "size": rec["size"]}
    exact |= {f"G_off k={n}": v["G_off"] for n, v in k["by_k"].items()} | {"G_off pair": k["pair"]["G_off"]}
    v = values(rec)
    rec["check"] = {"exact": exact, "readings": v}
    log(f"[do-nothing {item}] must be 0: {exact}; must be nan: {v}")
    bad = [f"{n} = {x}" for n, x in exact.items() if T2.num(x) != 0] + [f"{p} = {v[p]}" for p in GAIN_READ if not math.isnan(v[p])]
    if bad:
        stop(out_dir / "do_nothing.json", jsonable(rec), "do-nothing: " + "; ".join(bad), log)
    write_stage(out_dir / "do_nothing.json", jsonable(rec))
    return rec


# ---- one item -----------------------------------------------------------------------------------------------------------


def read_item(w: World, job: Any, item: str, i: int, out_dir: Path) -> dict[str, Any]:
    """khead's switch-off at k = 1, 2, 3, 5 and the pair arm (its file under <out_dir>/khead), holes_fill's neuron and
    steering shares (P2.54's `shares`; i seeds its steering selection), r, its gain, size and erank (`weights`)."""
    path = out_dir / "items" / f"{R1.fname(item)}.json"
    if path.is_file():
        return read_json(path)
    KH.OUT = out_dir / "khead"
    for d in (KH.OUT, path.parent):
        d.mkdir(parents=True, exist_ok=True)
    t0, rec = time.time(), {"item": item, "c": 0.0 if item.startswith("zero/") else w.rd.scale(item.removeprefix("random/"))}
    rec["khead"] = KH.read_item(job, item, KH.load_config()[1])
    rec |= T2.shares(w, item, i) | T2.weights(w, item)
    write_stage(path, jsonable(rec | {"seconds": time.time() - t0}))
    log(f"[item {item}] " + json.dumps(jsonable(values(rec) | {"hf_gain": rec["hf_gain"], "size": rec["size"]})) + f"; {time.time() - t0:.0f} s")
    torch.cuda.empty_cache()
    return read_json(path)


def values(rec: dict[str, Any]) -> dict[str, float]:
    """The eight readings of one item record, nan where a reader did not read: khead's k = 1 is the (2,4) share, and a
    head share counts only when its label is readable (run 1's `task_mean`: finite, gain at or over G_min)."""
    k = rec["khead"]["by_k"]
    head = lambda n: k[n]["share_lost"] if k[n]["label"] in R1.READABLE else math.nan
    v = {"head_share": head("1"), "k3_share": head("3"), "r": rec["r"], "erank": rec["erank"]}
    v |= {f"steering_{a}": x for a, x in rec["steering"].items()} | {f"neuron_{a}": x for a, x in rec["neuron"].items()}
    return {p: T2.num(v.get(p)) for p in READINGS}


# ---- the pair rule --------------------------------------------------------------------------------------------------


def seed_range(x: list[float]) -> float:
    return max(x) - min(x) if x and all(math.isfinite(v) for v in x) else math.nan


def band(reused: list[float], new: list[float]) -> float:
    """The smallest gap that matters: the larger seed range of the two sides at seeds 0-2 (E237's spread measure)."""
    return max(seed_range(reused[:3]), seed_range(new[:3]))


def call(reused: list[float], new: list[float], width: float, alpha: float) -> dict[str, Any]:
    """reused - new over seeds: the gap of means, the seed-spread interval, the exact Mann-Whitney U and p.
    'separates' when p <= alpha and |gap| > width (a tie does not); 'does_not_separate' otherwise; 'not_read' when a
    side has under 2 seeds or a value or the width is not finite. A reading constant over every seed has p = 1."""
    if len(reused) < 2 or len(new) < 2 or not all(math.isfinite(v) for v in (*reused, *new, width)):
        return {"gap": math.nan, "p": math.nan, "U": math.nan, "band": width, "call": "not_read"}
    gap, (u, p) = fmean(reused) - fmean(new), SR.mann_whitney(reused, new)
    out = {"gap": gap, "interval": [min(reused) - max(new), max(reused) - min(new)], "U": u, "p": p, "band": width}
    return out | {"call": "separates" if p <= alpha and abs(gap) > width else "does_not_separate"}


def labels(reused: list[dict[str, Any]], new: list[dict[str, Any]], th: dict[str, Any]) -> dict[str, str]:
    """P2.54 run 1's labels at k = 1, on khead's k = 1 arm: the reused side's mean (2,4) share >= share_new_max with
    every seed over its clean null median; the new side's mean <= share_new_max. 'not_read' unless every seed is
    readable (copying.share_label: finite, gain at or over G_min)."""
    k1 = lambda r: r["khead"]["by_k"]["1"]
    ok = lambda rs: all(k1(r)["label"] in R1.READABLE for r in rs)
    m = lambda rs: fmean(T2.num(k1(r)["share_lost"]) for r in rs) if ok(rs) else math.nan
    over = all(T2.num(k1(r)["share_lost"]) > T2.num(k1(r)["null_clean_median"]) for r in reused)
    return {"reused_uses_head": R1.verdict(m(reused) >= th["share_new_max"] and over, m(reused)),
            "new_does_not": R1.verdict(m(new) <= th["share_new_max"], m(new))}  # fmt: skip


def score(reused: list[dict[str, Any]], new: list[dict[str, Any]], randoms: list[dict[str, Any]], bands: dict[str, float],
          prior: dict[str, float], th: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Every filed rule on one pair: the labels; per reading the call (read only when both labels hold); the prior's
    gap; the random updates read nothing over the gain."""
    lab = labels(reused, new, th)
    readable = all(v == "holds" for v in lab.values())
    va, vb = [values(r) for r in reused], [values(r) for r in new]
    calls = {p: call([x[p] for x in va], [x[p] for x in vb], bands[p], th["alpha"]) for p in READINGS}
    for p, c in calls.items():
        c["sign_as_filed"] = math.copysign(1, c["gap"]) == math.copysign(1, th["expected_sign"][p]) if math.isfinite(c["gap"]) and c["gap"] != 0 else None
        c["call"] = c["call"] if readable else "not_read"
    vr = [values(r) for r in randoms]
    rnd = "not_read" if not vr else "holds" if all(math.isnan(x[p]) for x in vr for p in GAIN_READ) else "misses"
    return {"labels": lab, "calls": calls, "prior": prior | {"gap": prior["reused"] - prior["new"]},
            "random_reads_nothing": rnd, "rows": {"reused": va, "new": vb, "random": vr}, "n_separate": sum(c["call"] == "separates" for c in calls.values())}  # fmt: skip


def say(out: dict[str, Any]) -> None:
    log(f"[score] labels {out['labels']}; prior gap {out['prior']['gap']:.3f} decades; random reads nothing: {out['random_reads_nothing']}")
    for p, c in out["calls"].items():
        log(f"[score {p}] gap {c['gap']:.4g} (band {c['band']:.4g}), p {c['p']:.3g} -> {c['call']}, sign as filed {c.get('sign_as_filed')}")
