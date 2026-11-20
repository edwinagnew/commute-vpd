"""E222 (rl_67m), cell P2.41 stage 3: the re-gating share and the scorecard baselines read on each of E220's 27 saved
states. Plan reports/plans/2026-11-19_plan_rl-67m.md (section "Stage 3 (E222)"), card ..._rl-67m-stage3-card.md;
settings, expected values and filed rules results/predictions_E222.json (the run refuses unless it is committed
unchanged). Readers: E211's (`commute.realistic_reads.Reader` through experiments/realistic_tasks/run2.py's `reader`,
with E211's filed reader settings); behaviours, held-out prompts and demonstration pools: E220's
(run_stage2.py `Ctx`, `commute.rl_arms`). Pair calls: score_stage3.py on the CPU.

Read rows of a behaviour: held-out prompts whose own text continuation earns the behaviour's reward, the first n_read of
them; the read target is the continuation's first token (window = 64 prompt tokens + that token, read at position 63).
repeat and lower: stage 1's Pile val prompt rule extended to n_scan prompts (its first 256 are stage 1's held-out
prompts, checked). french_en: stage 1's 256 held-out English prompts, each followed by stage 1's held-out French
Wikipedia continuation of the same index. Steering's fitting windows: the first n_fit of the demo arm's training pool
(`rl_arms.demo_pool`, training prompts only), same layout.

  ctl          first and alone, exit 1 on a miss: E211's ctl unchanged (E53's lowercase and cipher seed-0 states re-read by
               these readers to their saved share, untrained shares, gain, neuron, steering, k_half, erank; a zero update
               reads every numerator exactly 0 on E53's rows), then a zero update on each behaviour's read rows reads
               every numerator exactly 0 (timed: the per-state time of this read)
  1 2 anchor   one behaviour (the two picks, the anchor): refuses until stage3_ctl.json is done. A Gaussian update at
               the first state's per-matrix norms first (gain at or over MIN_GAIN in log-odds exits 1), then every
               state of the behaviour, each written to results/stage3/<behaviour>_<arm>_s<seed>.json as it finishes

    sbatch --comment=E222 cluster/rl_67m_stage3.sbatch ctl
    sbatch --comment=E222 --dependency=afterok:<ctl id> cluster/rl_67m_stage3.sbatch 1     (and 2, anchor)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import realistic_reads as RR
from commute import rl_arms as RA
from commute import rl_tasks as RT
from commute import steering as ST
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_E222.json", RESULTS / "stage3"  # states: the files E220's records name
S2 = load_experiment("rl_67m", "run_stage2")  # E220's context: the model, behaviours, held-out prompts, pools
R2 = load_experiment("realistic_tasks", "run2")  # E211's reader, its ctl, zero check and Out
ARMS = {"1": ("rl", "shuffled", "own", "demo"), "2": ("rl", "shuffled", "own", "demo"), "anchor": ("demo",)}
JOBS = ("ctl", *ARMS)


@dataclass(frozen=True)
class Config:
    n_read: int
    n_scan: int
    n_fit: int
    gauss_draw: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        assert self.n_read <= 256 <= self.n_scan  # french_en has stage 1's 256 held-out rows; the Pile scan extends them


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def windows(prompts: Tensor, conts: Tensor) -> Tensor:
    """Each prompt followed by its continuation's first token: the LM layout of `RR.Read` (target at pos + 1)."""
    return torch.cat([prompts.long(), conts[:, :1].long()], 1)


def read_rows(x: Any, C: Config, name: str) -> tuple[RR.Read, dict[str, Any]]:
    """One behaviour's read rows and fitting windows (module docstring), and their counts."""
    r = x.reader(name)
    if name == "french_en":
        prompts, conts = x.samples["french_en"]["prompts"], x.samples["french_fr"]["true"]
        extra: list[Any] = [None] * len(prompts)
    else:
        rows = cached_rows("val", x.st1["pile_val_start"], x.st1["pile_val_n"])
        p, _ = RT.pile_prompts(r.b, rows, x.texts, x.tok.encode, dataclasses.replace(r.sz, n=C.n_scan))
        assert torch.equal(p.toks[:256], x.samples[name]["prompts"].long()), "the scan does not start with stage 1's held-out prompts"
        prompts, conts, extra = p.toks, p.true, p.extra
    keep = r.rewards(conts.long(), extra, 1).bool()
    fp, fc = RA.demo_pool(r, RA.train_prompts(r, x.pile_train, x.articles, x.windows, x.C.n_demo_prompts), x.articles)
    toks, fit = windows(prompts[keep][: C.n_read], conts[keep][: C.n_read]), windows(fp[: C.n_fit], fc[: C.n_fit])
    info = {"candidates": len(prompts), "earning": int(keep.sum()), "n_rows": len(toks), "n_fit": len(fit), "pool": len(fp)}
    return RR.Read(name, toks, toks[:, -1], toks.shape[1] - 2, fit, fit[:, -1]), info


def reader(x: Any, C: Config, C2: Any, name: str, o: Any) -> RR.Reader:
    r, info = read_rows(x, C, name)
    if info["n_rows"] < C.n_read or info["n_fit"] < C2.n_select:
        raise o.stop(f"{name}: {info} gives fewer than n_read {C.n_read} read rows or n_select {C2.n_select} fitting windows")
    rd = R2.reader(x.net, C2, r)
    o.put(f"read {name}", info | {"pos": r.pos, "prior_w0": {u: float(v.mean()) for u, v in rd.w0["rows"].items()}})
    return rd


def e220_record(p: str, arm: str, seed: int) -> dict[str, Any]:
    """E220's record of one saved state (its job file: stage2b_<p>_<arm>[_s<seed>].json)."""
    for f in sorted(RESULTS.glob(f"stage2b_{p}_{arm}*.json")):
        if (s := json.loads(f.read_text())["seeds"].get(str(seed))) is not None:
            return s
    raise FileNotFoundError(f"no stage2b record of {p} {arm} seed {seed}")


def load_state(p: str, arm: str, seed: int, o: Any) -> tuple[ST.DW, dict[str, Any]]:
    """The saved bf16 state as fp32, its sha256 checked against E220's record, and E220's readings of it."""
    s = e220_record(p, arm, seed)
    path = REPO / s["state"]["file"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != s["state"]["sha256"]:
        raise o.stop(f"{path} has sha256 {sha}, E220 recorded {s['state']['sha256']}")
    dW = {k: v.float() for k, v in torch.load(path, weights_only=False).items()}
    e220 = {"steps": s["steps"], "size_saved": s["size_saved"], "kl_saved": s["kl_saved"], "collapsed": s["collapsed"],
            "rate_0": s["reads"]["0"]["rate"], "rate_stop": s["reads"]["stop"]["rate"]}  # fmt: skip
    return dW, {"file": s["state"]["file"], "sha256": sha, "e220": e220}


def write_state(name: str, rec: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.tmp"
    tmp.write_text(json.dumps(jsonable(rec), indent=1))
    tmp.replace(OUT / f"{name}.json")
    log(f"[state {name}] {json.dumps(jsonable(R2.brief(rec) | {k: rec[k] for k in ('arm', 'seed', 'seconds', 'e220')}))}")


# ----------------------------------------------------------------------------- jobs


def ctl(x: Any, C: Config, C2: Any, th2: dict[str, Any], o: Any) -> None:
    R2.ctl(x.net, C2, th2, o)
    o.d["zero e53/lowercase"] = o.d.pop("zero")  # zero_check writes under "zero": keep each read's record
    for p in ARMS:
        rd, t0 = reader(x, C, C2, x.names[p], o), time.time()
        R2.zero_check(o, rd)
        o.d[f"zero {x.names[p]}"] = o.d.pop("zero")
        o.put(f"zero seconds {x.names[p]}", time.time() - t0)


def behaviour(x: Any, C: Config, C2: Any, p: str, o: Any) -> None:
    done = RESULTS / "stage3_ctl.json"
    if not (done.is_file() and json.loads(done.read_text()).get("done")):
        raise o.stop("stage3_ctl.json is not done: run ctl first and alone")
    name = x.names[p]
    rd = reader(x, C, C2, name, o)
    for i, (arm, seed) in enumerate((a, s) for a in ARMS[p] for s in C.seeds):
        dW, meta = load_state(p, arm, seed, o)
        if i == 0:  # the do-nothing control: a Gaussian update at this state's per-matrix norms
            g = rd.state(FT.random_direction(dW, C.gauss_draw), C.gauss_draw)
            o.put("gauss", g, shown=R2.brief(g))
            if not g["gain"]["lo"] < MIN_GAIN:
                raise o.stop(f"a Gaussian update gains {g['gain']['lo']:.3f} log-odds on {name}, at or over {MIN_GAIN}")
        t0 = time.time()
        rec = {"behaviour": name, "arm": arm, "seed": seed, **meta, **rd.state(dW, seed)}
        write_state(f"{name}_{arm}_s{seed}", rec | {"seconds": time.time() - t0})
        o.d.setdefault("states", {})[f"{name}_{arm}_s{seed}"] = meta["file"]
        o.write()


def main() -> int:
    job = sys.argv[1] if len(sys.argv) == 2 else ""
    if job not in JOBS:
        sys.exit(f"usage: run_stage3.py {{{' | '.join(JOBS)}}}")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E222")
    P2 = json.loads(R2.PRED.read_text())  # E211's filed reader settings and reproduce tolerances
    C, C2 = config_from_settings(Config, P["settings"]), config_from_settings(R2.Config, P2["settings"])
    x = S2.Ctx()  # E220's: the model (tf32 off), tokenizer, stage 1's records and held-out samples, Pile rows
    o = R2.Out(job)
    o.path, o.d["experiment"] = RESULTS / f"stage3_{job}.json", P["experiment"]
    try:
        ctl(x, C, C2, P2["thresholds"], o) if job == "ctl" else behaviour(x, C, C2, job, o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    print(f"STAGE 3 {job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
