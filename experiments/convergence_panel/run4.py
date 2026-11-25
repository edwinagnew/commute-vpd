"""P2.66 run 4, step 1 (a $0 cluster probe; plan reports/plans/2026-11-24_plan_p2-66-run4-estimate.md): at the KL-matched
state, arxiv, german (all seeds) and spanish seed 0 read a read-row gain under 1 log-odds, so they have no share or
steering share. Retrain those seeds past the 300-step cap to 1200 steps with the filed recipe, save every 50 steps and
read the read-row gain (`gain` of results/wide/read, `RR.share_of`) at each save; read sentiment's saves never read;
then pick per seed the earliest save with gain >= 1 (label "gain1", run.py `picked`). Rules, expected values and
early stops: results/wide/predictions_run4.json (refused unless committed unchanged). Every record is written as it
is read and every gain printed; a filed stop prints "STOP: ..." and exits 1. A rerun skips what is written.

  TASK SEED...  arxiv, german or spanish. Stage 0 on the first seed: trained to step 300 twice in memory, the job's
                floor = max(run-to-run gap, bf16 rounding gap) of the read-row gain; stop if the replay is off the saved
                TASK_sN_step300.pt by more -> results/wide/run4/control_TASK.json. Then each seed to 1200 steps, a bf16
                save every 50 (results/large/states_run4/r4_TASK_sN_stepK.pt), its step-300 save checked against the
                saved file first, then every save's KL (64 rows), size, held gain and read-row gain -> results/wide/run4/TASK_sN.json
  sentiment     the read-row gain of every save in results/train/sentiment.json not read in results/wide/read
                -> results/wide/run4/sentiment.json
  select        CPU, after every job: the filed pick rule -> results/wide/run4/selection.json

    sbatch --comment=E### cluster/convergence_panel.sbatch run4 arxiv 0 1 2     (and german 0 1 2; spanish 0; sentiment)
    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/run4.py select
Then `run.py read TASK` reads the gain1 picks (and seed 0's Gaussian) into results/wide/read, and eb.py takes gain1 as M.
A state results/wide/read already holds keeps its stored "picked" list: a scorer of the gain1 set takes picks from
`run.py picked`, not from the read file.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from commute import panel as PN
from commute import realistic as R
from commute import realistic_reads as RR
from commute import steering as ST
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.local_faith import MIN_GAIN

RUN: Any = load_experiment("convergence_panel", "run")
WIDE = RUN.RESULTS / "wide"
PRED4, OUT, STATES = WIDE / "predictions_run4.json", WIDE / "run4", RUN.LARGE / "states_run4"
SEL = RUN.RUN4_SEL


@dataclass(frozen=True)
class Config:
    steps: int
    every: int
    control_step: int
    gain_floor: float
    retrain: dict[str, list[int]]

    def __post_init__(self) -> None:
        assert self.steps % self.every == 0 and self.control_step % self.every == 0 and self.control_step < self.steps
        assert self.gain_floor == MIN_GAIN and set(self.retrain) <= {"arxiv", "german", "spanish"}


def load() -> tuple[Config, Any]:
    require_filed(PRED4)
    C4 = config_from_settings(Config, json.loads(PRED4.read_text())["settings"])
    C = RUN.load()[0]
    assert C4.control_step == C.steps, "the replay's reference is the filed runs' last step"
    return C4, C


def gain(net: ST.Net, t: PN.Task, dW: ST.DW) -> float:
    """The read-row gain: results/wide/read's `gain` (run.py `readings`)."""
    return RR.share_of(net, t.read, dW)["gain"][t.read.scored]


def bf16(dW: ST.DW) -> ST.DW:
    """dW as its bf16 save file reads back (run.py save_dW, load_dW)."""
    return {k: v.bfloat16().float() for k, v in dW.items()}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def saved_file(name: str, seed: int, step: int) -> str:
    return f"experiments/convergence_panel/results/large/states/{name}_s{seed}_step{step}.pt"


# ----------------------------------------------------------------------------- stage 0: the replay control


def replay(net: ST.Net, C: Any, C4: Config, t: PN.Task, seed: int) -> ST.DW:
    """The filed run of `seed` to control_step, its last dW in fp32 (nothing written)."""
    got: dict[str, ST.DW] = {}
    rc = PN.Recipe(C.lr, C.batch, C4.control_step, C.grad_clip)
    PN.train_ladder(net, t, rc, seed, RUN.rows_kl(C)[: C.kl_live_rows], {}, lambda s, dW, info: got.update(dW=dW), RUN.log)
    return got["dW"]


def control(net: ST.Net, C: Any, C4: Config, t: PN.Task, seed: int) -> float:
    """The job's floor (predictions_run4.json control), or Stopped when the replay is off the saved file by more."""
    path = OUT / f"control_{t.name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": t.name, "seed": seed}
    if "floor" not in out:
        a = replay(net, C, C4, t, seed)
        ab, bb = bf16(a), bf16(replay(net, C, C4, t, seed))
        f = saved_file(t.name, seed, C4.control_step)
        ref = RUN.load_dW(f)
        g = {"a_fp32": gain(net, t, a), "a_bf16": gain(net, t, ab), "b_bf16": gain(net, t, bb), "saved": gain(net, t, ref)}
        diff = {"a_b": max(float((ab[k] - bb[k]).abs().max()) for k in a), "a_saved": max(float((ab[k] - ref[k]).abs().max()) for k in a)}
        rec = {"card": torch.cuda.get_device_name(), "file": f, "gains": g, "max_abs_weight_diff_bf16": diff,
               "floor": max(abs(g["a_bf16"] - g["b_bf16"]), abs(g["a_fp32"] - g["a_bf16"])), "gap_saved": abs(g["a_bf16"] - g["saved"])}  # fmt: skip
        out |= rec | {"h100_read": recorded(t.name, f)}
        RUN.put(path, out, "control", rec)
    # Amendment 1 (2026-11-24, whip-lead): the saved files were trained on cluster L40s, and an L40S replay is off them
    # by 0.0025/0.0042 (arxiv/german s0) against within-card floors of 0.00077/0.00143. The gap to the saved file is
    # recorded, not a stop; the stop is now two replays on this card that are not bit-identical.
    if out["max_abs_weight_diff_bf16"]["a_b"] != 0:
        raise RUN.stop(path, out, f"{t.name} s{seed}: two step-{C4.control_step} replays on this card differ by {out['max_abs_weight_diff_bf16']['a_b']:.3g}")
    return out["floor"]


def recorded(name: str, f: str) -> float | None:
    """The saved file's read-row gain in results/wide/read (an H100 read), where it was picked and read."""
    p = WIDE / "read" / f"{name}.json"
    return json.loads(p.read_text())["states"].get(f, {}).get("gain") if p.is_file() else None


# ----------------------------------------------------------------------------- one seed to 1200 steps


def retrain(net: ST.Net, C: Any, C4: Config, t: PN.Task, seed: int, floor: float, w0: float) -> None:
    path = OUT / f"{t.name}_s{seed}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    if not out.get("trained"):
        out = {"task": t.name, "seed": seed, "card": torch.cuda.get_device_name(), "saves": []}

        def save(step: int, dW: ST.DW, info: dict[str, Any]) -> None:
            p = STATES / f"r4_{t.name}_s{seed}_step{step}.pt"
            out["saves"].append({"step": step, "kl_live": info["kl_live"], "file": str(p.relative_to(RUN.REPO)), "sha256": RUN.save_dW(p, dW)})
            write_stage(path, out)

        t0, rc = time.time(), PN.Recipe(C.lr, C.batch, C4.steps, C.grad_clip)
        trace = PN.train_ladder(net, t, rc, seed, RUN.rows_kl(C)[: C.kl_live_rows], {}, save, RUN.log, C4.every)
        RUN.put(path, out, "trained", {"seconds": time.time() - t0, "last_step": trace[-1]["step"], "trace": trace})
    by_step = {s["step"]: s for s in out["saves"]}
    order = [by_step[C4.control_step]] + [s for s in out["saves"] if s["step"] != C4.control_step]  # the check first
    for s in order:
        if "gain" not in s:
            s |= read_save(net, C, t, s, w0)
            RUN.put(path, out, "saves", out["saves"])
            RUN.log(f"[run4 {t.name} s{seed}] step {s['step']}: read-row gain {s['gain']:.6g}, KL {s['kl']:.4g}, held gain {s['held_gain']:.4g}")
        if s["step"] == C4.control_step:
            ref = gain(net, t, RUN.load_dW(saved_file(t.name, seed, C4.control_step)))
            RUN.put(path, out, "step300_check", {"saved": ref, "gap": abs(s["gain"] - ref), "floor": floor})  # recorded (Amendment 1)


def read_save(net: ST.Net, C: Any, t: PN.Task, s: dict[str, Any], w0: float) -> dict[str, Any]:
    if sha(RUN.REPO / s["file"]) != s["sha256"]:
        raise RUN.Stopped(f"{s['file']}: sha256 differs from the one recorded at its save")
    t0, d = time.time(), RUN.load_dW(s["file"])
    return {"gain": gain(net, t, d), "kl": PN.kl_from_w0(net, RUN.rows_kl(C), d), "size": R.update_size(d, net.W0),
            "held_gain": PN.held_gain(net, t, d, w0), "seconds": time.time() - t0}  # fmt: skip


def retrain_task(net: ST.Net, C: Any, C4: Config, name: str, seeds: list[int]) -> None:
    assert set(seeds) <= set(C4.retrain[name]), f"{name} retrains seeds {C4.retrain[name]} only"
    t = RUN.TK.build(net, C, name)
    w0 = PN.target_logp(net, t.held, t.held_mask, None)
    floor = control(net, C, C4, t, seeds[0])
    for seed in seeds:
        retrain(net, C, C4, t, seed, floor, w0)


# ----------------------------------------------------------------------------- sentiment, and the pick


def sentiment(net: ST.Net, C: Any) -> None:
    path = OUT / "sentiment.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": "sentiment", "card": torch.cuda.get_device_name(), "states": {}}
    read = json.loads((WIDE / "read" / "sentiment.json").read_text())["states"]
    t = RUN.TK.build(net, C, "sentiment")
    for seed, saves in json.loads((RUN.RESULTS / "train" / "sentiment.json").read_text())["seeds"].items():
        for s in saves[1:]:
            if s["file"] not in read and s["file"] not in out["states"]:
                if sha(RUN.REPO / s["file"]) != s["sha256"]:
                    raise RUN.Stopped(f"{s['file']}: sha256 differs from results/train/sentiment.json's")
                out["states"][s["file"]] = {"seed": int(seed), "step": s["step"], "gain": gain(net, t, RUN.load_dW(s["file"]))}
                RUN.put(path, out, "states", out["states"])
                RUN.log(f"[run4 sentiment s{seed}] step {s['step']}: read-row gain {out['states'][s['file']]['gain']:.6g}")


def earliest(saves: list[dict[str, Any]], floor: float) -> dict[str, Any] | None:
    """The filed rule: the earliest save whose read-row gain is >= floor (nan never passes); None if none."""
    assert all("gain" in s for s in saves), [s["step"] for s in saves if "gain" not in s]
    return min((s for s in saves if s["gain"] >= floor), key=lambda s: s["step"], default=None)


def select(C4: Config) -> None:
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, seeds in C4.retrain.items():
        for seed in seeds:
            d = json.loads((OUT / f"{name}_s{seed}.json").read_text())
            assert d.get("trained") and "stopped" not in d, f"{name} s{seed}: not trained, or stopped"
            runs.setdefault(name, {})[str(seed)] = d["saves"]
    read = json.loads((WIDE / "read" / "sentiment.json").read_text())["states"] | json.loads((OUT / "sentiment.json").read_text())["states"]
    tr = json.loads((RUN.RESULTS / "train" / "sentiment.json").read_text())["seeds"]
    missing = [s["file"] for saves in tr.values() for s in saves[1:] if s["file"] not in read]
    assert not missing, f"run run4.py sentiment first: {missing}"
    runs["sentiment"] = {seed: [s | {"gain": read[s["file"]]["gain"]} for s in saves[1:]] for seed, saves in tr.items()}
    tasks = {n: {seed: earliest(saves, C4.gain_floor) for seed, saves in by.items()} for n, by in runs.items()}
    nan = {f"{n} s{seed}": [s["step"] for s in saves if not math.isfinite(s["gain"])] for n, by in runs.items() for seed, saves in by.items()}
    write_stage(SEL, {"written": time.time(), "label": "gain1", "rule": json.loads(PRED4.read_text())["rule"]["pick"], "tasks": tasks,
                      "none": {n: [s for s, p in by.items() if p is None] for n, by in tasks.items()}, "nan_gains": {k: v for k, v in nan.items() if v}})  # fmt: skip
    for n, by in tasks.items():
        for seed, p in by.items():
            RUN.log(f"[select gain1] {n} s{seed}: " + ("no save reaches 1" if p is None else f"step {p['step']}, gain {p['gain']:.4g}, KL {p['kl']:.4g}"))


def main() -> int:
    a = sys.argv[1:]
    ok = a in (["sentiment"], ["select"]) or (len(a) >= 2 and a[0] in ("arxiv", "german", "spanish") and all(x.isdigit() for x in a[1:]))
    if not ok:
        sys.exit("usage: run4.py arxiv|german|spanish SEED... | sentiment | select")
    C4, C = load()
    if a == ["select"]:
        select(C4)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    try:
        sentiment(net, C) if a == ["sentiment"] else retrain_task(net, C, C4, a[0], [int(x) for x in a[1:]])
    except RUN.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print(f"RUN4 {' '.join(a)} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
