"""E224 (rl_67m), cell P2.41 run 7: E222's readings (the re-gating share and the scorecard baselines, E211's
`realistic_reads.Reader` through experiments/realistic_tasks/run2.py's `reader`) of the states run_newtasks.py saved,
scored on the per-row set contrast log p(target set) - log p(competitor set) at the cue (`rl_tasks_2191.SetRead`), the
same quantity the reward rewards. Read rows: the first n_read held-out items of the task; steering's fitting windows:
the first n_fit training items. Settings and rules: results/predictions_E224.json (refused unless committed unchanged);
E211's reader settings read from experiments/realistic_tasks/results/predictions_run2.json. Calls: score_newtasks.py.

  ctl      first and alone: a zero update on each task's rows reads every numerator exactly 0 (timed: the per-state
           cost of these rows)
  1 2 anchor   refuses until ctl is done: a Gaussian update at the first state's per-matrix norms (gain at or over
           MIN_GAIN in the contrast exits 1), then every saved state of the task (stage2c_arm_<P>_* records, sha256
           checked), each written to results/stage3c/<task>_<arm>_s<seed>.json as it finishes

    .venv/bin/python experiments/rl_67m/read_newtasks.py ctl
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from typing import Any

import torch

from commute import finetune as FT
from commute import rl_tasks_2191 as NT
from commute.jobs import config_from_settings, hard_exit, load_experiment
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

RN = load_experiment("rl_67m", "run_newtasks")  # the context (model, items), names, arms and paths
R2 = RN.R2


def reader(x: Any, C2: Any, task: str, o: Any) -> Any:
    d = x.items[task]
    r = NT.SetRead.of(task, d["held"].take(slice(0, x.C.n_read)), d["train"].take(slice(0, x.C.n_fit)))
    rd = R2.reader(x.net, C2, r)
    o.put(f"read {task}", {"n_rows": len(r.toks), "n_fit": len(r.fit), "prior_w0": float(rd.w0["rows"]["contrast"].mean())})
    return rd


def states(p: str) -> list[dict[str, Any]]:
    """The saved states of task p in arm order, from run_newtasks.py's stage2c_arm_<p>_<arm>* records."""
    files = [f for arm in RN.ARMS[p] for f in sorted(RN.RESULTS.glob(f"stage2c_arm_{p}_{arm}*.json"))]
    return [s for f in files for s in json.loads(f.read_text()).get("seeds", {}).values() if "state" in s]


def ctl(x: Any, C2: Any, o: Any) -> None:
    for t in RN.NAMES.values():
        rd, t0 = reader(x, C2, t, o), time.time()
        R2.zero_check(o, rd)
        o.d[f"zero {t}"], o.d[f"zero seconds {t}"] = o.d.pop("zero"), time.time() - t0
        o.write()


def task(x: Any, C2: Any, p: str, o: Any) -> None:
    done = RN.RESULTS / "stage3c_ctl.json"
    if not (done.is_file() and json.loads(done.read_text()).get("done")):
        raise o.stop("stage3c_ctl.json is not done: run ctl first and alone")
    name, rd = RN.NAMES[p], reader(x, C2, RN.NAMES[p], o)
    for i, s in enumerate(states(p)):
        path = RN.REPO / s["state"]["file"]
        if (sha := hashlib.sha256(path.read_bytes()).hexdigest()) != s["state"]["sha256"]:
            raise o.stop(f"{path} has sha256 {sha}, its record says {s['state']['sha256']}")
        dW = {k: v.float() for k, v in torch.load(path, weights_only=False).items()}
        if i == 0:  # the do-nothing control: a Gaussian update at this state's per-matrix norms
            g = rd.state(FT.random_direction(dW, x.C.gauss_draw), x.C.gauss_draw)
            o.put("gauss", g, shown=R2.brief(g))
            if not g["gain"]["contrast"] < MIN_GAIN:
                raise o.stop(f"a Gaussian update gains {g['gain']['contrast']:.3f} on {name}, at or over {MIN_GAIN}")
        t0, key = time.time(), f"{name}_{s['arm']}_s{s['seed']}"
        rec = {"task": name, "arm": s["arm"], "seed": s["seed"], "file": s["state"]["file"], "sha256": sha, "reached": s["reached"],
               **rd.state(dW, s["seed"])}  # fmt: skip
        RN.READS.mkdir(parents=True, exist_ok=True)
        (RN.READS / f"{key}.json").write_text(json.dumps(jsonable(rec | {"seconds": time.time() - t0}), indent=1))
        o.put(key, R2.brief(rec), under="states")


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("ctl", *RN.NAMES):
        sys.exit(f"usage: read_newtasks.py {{ctl | {' | '.join(RN.NAMES)}}}")
    x = RN.Ctx()
    C2 = config_from_settings(R2.Config, json.loads(R2.PRED.read_text())["settings"])
    o = R2.Out(argv[0])
    o.path, o.d["experiment"] = RN.RESULTS / f"stage3c_{argv[0]}.json", x.P["experiment"]
    try:
        ctl(x, C2, o) if argv[0] == "ctl" else task(x, C2, argv[0], o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    print(f"E224 read {argv[0]} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
