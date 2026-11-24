"""E279 (cell P2.66, card ~/commute-data/cards/E279.md, descriptive): the untrained-SVD share (E136's reading as
P2.56 read it: procedural_rl/score.py GET["untrained"] = `realistic_reads.Reader.untrained`'s share["lo"]["svd"]) on
E272's saved wide KL-hi panel, 12 tasks x 3 seeds, and each task's panel Gaussian. Settings and the filed controls and
outputs: results/predictions.json (the run refuses unless it is committed unchanged). Every number is printed.

  0          alone, first, exit 1 on a miss: a zero update reads numerator and gain exactly 0 on E53's lowercase rows;
             E53's lowercase_s0 and cipher_s0 repeat E136's saved SVD shares within reproduce_tol -> results/stage0.json
  read TASK  after 0: the task's rows (convergence_panel tasks.build); a zero update reads exactly 0; the task's
             Gaussian first (|gain| >= MIN_GAIN exits 1); then its kl_hi states, seeds 0-2  -> results/read/TASK.json
  score      CPU, no model: the four filed statistics (posthoc_checks.taus, E272's n_perm and seed) -> results/summary.json

    sbatch --comment=E279 cluster/untrained_panel.sbatch 0
    sbatch --comment=E279 cluster/untrained_panel.sbatch read arxiv category conll french
    ~/commute-agents/fleet/cpu-run python3 experiments/untrained_panel/run.py score
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from commute import realistic_reads as RR
from commute import steering as ST
from commute import untrained as U
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, READS = RESULTS / "predictions.json", RESULTS / "read"
RUN: Any = load_experiment("convergence_panel", "run")  # E272: Config, picked, load_dW, FT, TK, RESULTS
SC: Any = load_experiment("convergence_panel", "score")  # raters, task_means
PH: Any = load_experiment("convergence_panel", "posthoc_checks")  # taus
R2: Any = load_experiment("realistic_tasks", "run2")  # E211: e53_read, E136's folder
WIDE = RUN.RESULTS / "wide"


@dataclass(frozen=True)
class Config:
    tasks: tuple[str, ...]
    match: str
    unit: str
    decomposition: str
    controls: tuple[str, ...]
    reproduce_tol: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "controls", tuple(self.controls))
        assert (self.match, self.unit, self.decomposition) == ("kl_hi", "lo", "svd"), "the filed reading is kl_hi, lo, svd"
        assert len(self.tasks) == 12 and 0 < self.reproduce_tol < 0.01


def load() -> Config:
    require_filed(PRED)
    C = config_from_settings(Config, json.loads(PRED.read_text())["settings"])
    kept = json.loads((WIDE / "dropped.json").read_text())["kept"][C.match]
    assert sorted(C.tasks) == sorted(kept), f"filed tasks {C.tasks} are not E272's wide {C.match} set {kept}"
    return C


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stopped(Exception):
    pass


# ----------------------------------------------------------------------------- the reading


class Reader(RR.Reader):
    """E211's Reader (realistic_tasks/run2.py `reader`) with only the untrained read's W0 cells in "lo" set up: no
    steering, neuron or k_half baselines, which this run does not read."""

    def __post_init__(self) -> None:
        self.w0["untrained"] = {"lo": self.untrained_cells(None, "lo")}


def reader(net: ST.Net, r: RR.Read) -> Reader:
    """W0's SVD with E136's k_m, as run2.py `reader` builds it (its rotated decomposition left out: each decomposition's
    gates and cells are computed on their own in `untrained.do_nothing_cells`, so the SVD's do not depend on it)."""
    k_m = {k: int(v) for k, v in json.loads((R2.E136 / "stage0.json").read_text())["k_m"].items()}
    return Reader(net, r, {"svd": U.svd_decomposition(net.W0)}, k_m, 0, {}, {})


def read_one(rd: Reader, dW: ST.DW) -> dict[str, Any]:
    """Reader.untrained's arithmetic in "lo" on the SVD, with the gain (T1 - T0) it divides by kept beside it."""
    t0 = time.time()
    c0, c1 = rd.w0["untrained"]["lo"], rd.untrained_cells(dW, "lo")
    num, gain = c1["svd"] - c0["svd"], c1["T"] - c0["T"]
    return {"share": RR.ratio(num, gain), "numerator": num, "gain": gain, "T0": c0["T"], "L00": c0["svd"], "seconds": time.time() - t0}


def zero(net: ST.Net) -> ST.DW:
    return {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()}


def put(path: Path, out: dict[str, Any], key: str, rec: Any) -> None:
    out[key] = rec
    write_stage(path, jsonable(out))
    log(f"[{key}] {json.dumps(jsonable(rec))[:1500]}")


def stop(path: Path, out: dict[str, Any], why: str) -> Stopped:
    write_stage(path, jsonable(out | {"stopped": why}))
    return Stopped(why)


def zero_check(rd: Reader, path: Path, out: dict[str, Any]) -> None:
    put(path, out, "zero", z := read_one(rd, zero(rd.net)))
    if not (z["numerator"] == 0.0 and z["gain"] == 0.0):
        raise stop(path, out, f"a zero update does not read exactly 0: numerator {z['numerator']!r}, gain {z['gain']!r}")


# ----------------------------------------------------------------------------- stage 0


def stage0(net: ST.Net, C: Config) -> None:
    path, out = RESULTS / "stage0.json", {"experiment": "E279", "card": torch.cuda.get_device_name()}
    inv = json.loads((R2.E136 / "inventory.json").read_text())["states"]
    saved = json.loads((R2.E136 / "stage1.json").read_text())["states"]
    for name in C.controls:
        rd = reader(net, R2.e53_read(name.split("/")[1].removesuffix("_s0")))
        if name == C.controls[0]:
            zero_check(rd, path, out)
        got = read_one(rd, RUN.load_dW(inv[name]["file"]))
        put(path, out, name, got | {"saved": saved[name]["svd"], "tol": C.reproduce_tol})
        if not abs(got["share"] - saved[name]["svd"]) <= C.reproduce_tol:
            raise stop(path, out, f"{name}'s SVD share {got['share']!r} does not repeat E136's {saved[name]['svd']!r}")
    put(path, out, "passed", True)


# ----------------------------------------------------------------------------- one task


def states(name: str, match: str) -> dict[int, dict[str, Any]]:
    """seed -> E272's selected state at `match` (its file, sha256, KL, size), from wide/selection.json."""
    got = {s["seed"]: s | {"file": f} for f, s in RUN.picked(name, "wide").items() if match in s["picked"]}
    assert sorted(got) == [0, 1, 2], f"{name}: {match} seeds {sorted(got)}"
    return got


def load_state(st: dict[str, Any]) -> ST.DW:
    sha = hashlib.sha256((RUN.REPO / st["file"]).read_bytes()).hexdigest()
    assert sha == st["sha256"], f"{st['file']}: sha256 {sha}, selection.json {st['sha256']}"
    return RUN.load_dW(st["file"])


def read_task(net: ST.Net, C: Config, C272: Any, name: str) -> None:
    path = READS / f"{name}.json"
    out: dict[str, Any] = {"task": name, "match": C.match}
    t = RUN.TK.build(net, C272, name)
    rd = reader(net, t.read)
    put(path, out, "rows", {"type": type(t.read).__name__, "n": len(t.read.toks), "pos": t.read.pos, "T0": rd.w0["untrained"]["lo"]["T"]})
    zero_check(rd, path, out)
    by_seed = states(name, C.match)
    f0 = by_seed[0]["file"]
    saved_g = json.loads((WIDE / "read" / f"{name}.json").read_text())["random"][f0]  # E272's read of the same Gaussian
    g = read_one(rd, RUN.FT.random_direction(load_state(by_seed[0]), C272.random_seed))
    put(path, out, "gauss", g | {"paired_to": f0, "kl_e272": saved_g["kl"], "size_e272": saved_g["size"]})
    if not abs(g["gain"]) < MIN_GAIN:
        raise stop(path, out, f"{name}'s Gaussian gains {g['gain']:.4f} log-odds, at or over {MIN_GAIN}")
    for seed, st in sorted(by_seed.items()):
        rec = read_one(rd, load_state(st))
        put(path, out, f"s{seed}", rec | {k: st[k] for k in ("file", "step", "kl", "size", "held_gain")})
    out["gauss"]["share_paired"] = RR.ratio(g["numerator"], out["s0"]["gain"])  # E272's Gaussian denominator: its state's gain
    put(path, out, "gauss", out["gauss"])
    put(path, out, "done", True)


def require_stage0() -> None:
    p = RESULTS / "stage0.json"
    if not (p.is_file() and json.loads(p.read_text()).get("passed")):
        sys.exit(f"{p} is missing or did not pass: run stage 0 first, alone")


# ----------------------------------------------------------------------------- score (CPU)


def score(C: Config) -> None:
    P272 = json.loads(RUN.PRED.read_text())
    R, tier = P272["rules"], SC.raters(P272, "raters_wide")["consensus"]
    reads = {t: json.loads((READS / f"{t}.json").read_text()) for t in C.tasks}
    if undone := [t for t, d in reads.items() if not d.get("done")]:
        sys.exit(f"tasks not read to the end: {undone}")
    means, info = SC.task_means({t: [float(d[f"s{s}"]["share"]) for s in (0, 1, 2)] for t, d in reads.items()})
    S = json.loads((RUN.RESULTS / "summary.json").read_text())["wide"][C.match]["means"]
    loss = {t: -v["trained"]["nats_per_byte"] for t, v in json.loads((WIDE / "base_loss.json").read_text())["tasks"].items()}
    out: dict[str, Any] = {"experiment": "E279", "means": means, "readability": info, "n_perm": R["n_perm"], "seed": R["seed"],
                           "seeds": {t: [float(d[f"s{s}"]["share"]) for s in (0, 1, 2)] for t, d in reads.items()},
                           "1_tiers": PH.taus(tier, means, None, R), "2_given_base_loss": PH.taus(tier, means, loss, R),
                           "3_given_kl": PH.taus(tier, means, S["kl"], R), "4_E_b": PH.taus(S["E_b"], means, None, R)}  # fmt: skip
    gs = {t: float(d["gauss"]["share_paired"]) for t, d in reads.items()}
    top = max((v for v in gs.values() if math.isfinite(v)), default=math.nan)
    out["gauss"] = {"share_paired": gs, "numerator": {t: d["gauss"]["numerator"] for t, d in reads.items()},
                    "gain": {t: d["gauss"]["gain"] for t, d in reads.items()}, "max_finite_share_paired": top,
                    "tasks_above_max": sorted(t for t, v in means.items() if v > top)}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(jsonable(out), indent=1))
    for t in C.tasks:
        print(f"[{t}] tier {tier.get(t)} seeds {[round(x, 4) for x in out['seeds'][t]]} mean {means.get(t, 'missing')}; "
              f"Gaussian share (paired) {gs[t]:.4f}, numerator {out['gauss']['numerator'][t]:.4f}, gain {out['gauss']['gain'][t]:.4f}")  # fmt: skip
    print(f"missing (a seed nan, gain under {MIN_GAIN}): {info['missing']}")
    for k in ("1_tiers", "2_given_base_loss", "3_given_kl", "4_E_b"):
        o = out[k]
        print(f"[{k}] n {o['n']} tau {o['tau']:.4f} p {o['p']:.4f}" + (f"; partial {o['partial']:.4f} p {o['p_partial']:.4f}" if "partial" in o else "") + f" tasks {o['tasks']}")
    print(f"[Gaussians] largest finite paired share {top:.4f}; task means above it: {out['gauss']['tasks_above_max']} of {len(means)}")
    print(f"mean of task means {fmean(means.values()) if means else math.nan:.4f}; wrote {RESULTS / 'summary.json'}")


def main(args: list[str]) -> int:
    C = load()
    if args == ["score"]:
        score(C)
        return 0
    if not (args == ["0"] or (len(args) >= 2 and args[0] == "read" and set(args[1:]) <= set(C.tasks))):
        sys.exit(f"usage: run.py 0 | read TASK [TASK ...] | score, TASK in {C.tasks}")
    if args[0] == "read":
        require_stage0()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    try:
        if args == ["0"]:
            stage0(net, C)
        for name in args[1:]:
            read_task(net, C, RUN.load()[0], name)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print(f"E279 {' '.join(args)} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
