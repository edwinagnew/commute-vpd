"""P2.82 (random_logodds_share): a random baseline for the paper's log-odds re-gating share (Fig `known`). Random weight
changes are scaled until they move the trained answer by one log-odds (the form's floor, `local_faith.MIN_GAIN`), and
the share is read on them beside each fine-tune's own. Descriptive (user 2026-11-25, "run option 1"): values per task,
no verdict. Card card.md; plan reports/plans/2026-11-25_plan_p2-82-random-logodds-share.md; settings and expected values
results/predictions.json (the run refuses unless it is committed unchanged). Nothing is trained.

States: exactly Fig `known`'s, the share cells of experiments/known_3seeds/results/known3.json (nine built pairs, three
seeds a side, 48 states). Reader: E136's `share_untrained` Job.trained (`local_faith.route_split`, bf16, E136's read
windows), the code behind every one of those values (E136 stage 1, E299 stage 2, E147 stage 1 all call it).
  share r    (L01 - L00) / gain, gain = T1 - T0 of the same forwards; nan unless gain >= 1 (one-sided, MIN_GAIN)
  KL         E228's `kl_of`: mean KL(W0 + change || W0) over every position up to the read position, same rows
Random draws: E228's, `finetune.random_direction(dW, seed)` (each matrix Gaussian at the fine-tune's norm), E228's seeds.
Scale: the sign s from the odd part of the gain at x = probe_x, s = sign(gain(+x d) - gain(-x d)) (a direction and its
negative are the same Gaussian, so the sign is not a choice of the draw); then `perturb.match` finds x with gain(s x d)
in the filed band, both signs tried when the first does not reach it. A draw that never reaches the band is reported with
its largest gain, share nan. The draws are NOT matched to the fine-tune's KL: their KL is reported beside it.

Jobs (cluster debug QoS, one card each):
  control    stage 0, alone, first -> results/control.json: (a) a zero change on every task's rows reads gain,
             numerator and KL exactly 0 and the share nan; (b) every fine-tune's share equals its known3.json value
             exactly (float equality). Exit 1 on a miss, listing every difference. Also each fine-tune's gain and KL.
  states I N every N-th state from the I-th (refused until control.json passes): three draws each ->
             results/states/<state>.json, written per draw; a rerun skips finished draws and states.

    sbatch --comment=E### cluster/random_logodds_share.sbatch control
    sbatch --comment=E### cluster/random_logodds_share.sbatch states 0 6     # ... states 5 6
"""

from __future__ import annotations

import json
import math
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import perturb as PT
from commute import realistic_reads as RR
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable
from commute.share import Floor

E136: Any = load_experiment("share_untrained", "run")  # Job.trained: the share's reader
KS: Any = load_experiment("known_3seeds", "run")  # E299: inventory with its seed-2 states
R96: Any = load_experiment("refit_96key", "run")  # E147: inventory with the e58 states
RK: Any = load_experiment("random_kl_share", "run")  # E228: kl_of, and its filed seeds
HERE = Path(__file__).resolve().parent
RESULTS, STATES = HERE / "results", HERE / "results" / "states"
PRED, CONTROL = RESULTS / "predictions.json", RESULTS / "control.json"
KNOWN3 = EXPERIMENTS / "known_3seeds" / "results" / "known3.json"
E147_INV = EXPERIMENTS / "refit_96key" / "results" / "inventory.json"
FLOOR = Floor(MIN_GAIN, two_sided=False)  # the log-odds form's floor (masked_reads.share)
DW = dict[str, Tensor]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: dict[str, Any]) -> None:
    write_stage(path, jsonable(obj))


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    seeds: tuple[int, ...]
    probe_x: float
    scale_cap: float
    target_gain: float
    gain_rel_tol: float
    kl_batch: int

    def __post_init__(self) -> None:
        assert len(set(self.seeds)) == len(self.seeds) >= 1, self.seeds
        assert self.target_gain * (1 - self.gain_rel_tol) >= MIN_GAIN, "a matched gain must pass the form's floor"
        assert 0 < self.probe_x < self.scale_cap


def load_config() -> tuple[Config, str]:
    pred = json.loads(PRED.read_text())
    assert pred["experiment"] != "EXXX", "claim the E-number (fleet/ecode) and write it into predictions.json first"
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg = config_from_settings(Config, s)
    assert list(cfg.seeds) == json.loads(RK.PRED.read_text())["settings"]["seeds"], "seeds must be E228's"
    return cfg, pred["experiment"]


# ---- the states and their reads -------------------------------------------------------------------------------------


def figure_states() -> tuple[dict[str, dict[str, list[str]]], dict[str, float]]:
    """Fig `known`'s log-odds shares: {pair: {side: states}} from known3.json's share cells, and each state's value
    (a state on two pairs, e54/elicit and c3/cipher, must carry one value)."""
    cells = [c for c in json.loads(KNOWN3.read_text())["cells"] if c["method"] == "share"]
    saved: dict[str, float] = {}
    for c in cells:
        for side in ("reuse", "new"):
            for s, v in zip(c["states"][side], c[side], strict=True):
                assert saved.setdefault(s, v) == v, f"{s}: two values in known3.json"
    return {c["pair"]: c["states"] for c in cells}, saved


def inventory(states: list[str]) -> dict[str, Any]:
    """E136's inventory format for exactly these states (E136's Job loads every task it is given): E136's own states,
    E299's seed-2 states (known_3seeds) and E147's e58 states (refit_96key)."""
    _, ks = KS.inventories()
    _, r96 = R96.inventories(json.loads(E147_INV.read_text()))
    every, tasks = {**ks["states"], **r96["states"]}, {**ks["tasks"], **r96["tasks"]}
    mine = {s: every[s] for s in states}
    return {"tasks": {t: tasks[t] for t in dict.fromkeys(v["task"] for v in mine.values())}, "states": mine}


def ordered(saved: dict[str, float]) -> list[str]:
    """Every state once, grouped by task in known3.json's order (Job caches one task's W0 gates at a time)."""
    inv = inventory(list(saved))
    tasks = list(inv["tasks"])
    return sorted(saved, key=lambda s: tasks.index(inv["states"][s]["task"]))


class Reader:
    """E136's Job on the given states, and E228's KL on the same rows."""

    def __init__(self, states: list[str], kl_batch: int) -> None:
        self.job = E136.Job(inventory(states))
        self.kl_batch = kl_batch
        self.rows: dict[str, Any] = {}

    def task(self, state: str) -> str:
        return self.job.inv["states"][state]["task"]

    def share(self, state: str, dW: DW) -> dict[str, Any]:
        t0 = time.time()
        rs = self.job.trained(state, dW)
        num, gain = rs["cells"]["L01"] - rs["cells"]["L00"], rs["gain"]
        return {"gain": gain, "numerator": num, "r": num / gain if FLOOR.passes(gain) else math.nan,
                "r_route_split": rs["shares"]["r"], "max_dg": rs["max_dg"], "read_seconds": time.time() - t0}  # fmt: skip

    def kl(self, state: str, dW: DW) -> float:
        t = self.task(state)
        if t not in self.rows:
            w = self.job.reads[t]
            self.rows[t] = RR.target_read(t, w.toks, w.token, w.half, w.ci_len)  # E228's rows, as its state_read
        return RK.kl_of(self.job.net, self.rows[t], dW, self.kl_batch)


def norm(d: DW) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))


# ---- stage 0: the controls ------------------------------------------------------------------------------------------


def control_job(experiment: str, cfg: Config) -> int:
    _, saved = figure_states()
    order = ordered(saved)
    rd = Reader(order, cfg.kl_batch)
    out: dict[str, Any] = {"experiment": experiment, "host": socket.gethostname(), "card": torch.cuda.get_device_name(),
                           "zero": {}, "fine_tunes": {}}  # fmt: skip
    zero = {k: torch.zeros_like(w) for k, w in rd.job.W0.items()}
    for t in dict.fromkeys(rd.task(s) for s in order):
        s = next(x for x in order if rd.task(x) == t)
        z = rd.share(s, zero) | {"kl": rd.kl(s, zero), "state": s}
        z["exact"] = all(z[k] == 0 for k in ("gain", "numerator", "kl")) and math.isnan(z["r"]) and math.isnan(z["r_route_split"])
        out["zero"][t] = z
        log(f"[0 zero {t}] {json.dumps(jsonable(z))}")
    if bad := [t for t, z in out["zero"].items() if not z["exact"]]:
        return stop(out, f"a zero change does not read gain, numerator, KL exactly 0 and share nan on {bad}")
    dump(CONTROL, out)
    for s in order:
        dW = rd.job.delta(s)
        f = rd.share(s, dW) | {"kl": rd.kl(s, dW), "norm": norm(dW), "saved": saved[s], "task": rd.task(s)}
        f["diff"] = f["r_route_split"] - saved[s]
        f["exact"] = f["r_route_split"] == saved[s]
        out["fine_tunes"][s] = f
        log(f"[0 fine-tune {s}] {json.dumps(jsonable(f))}")
        dump(CONTROL, out)
    if bad := {s: f["diff"] for s, f in out["fine_tunes"].items() if not f["exact"]}:
        return stop(out, f"fine-tune shares differ from known3.json (state: reread - saved): {bad}")
    out["pass"] = True
    dump(CONTROL, out)
    log("STAGE 0 DONE")
    return 0


def stop(out: dict[str, Any], why: str) -> int:
    dump(CONTROL, out | {"stopped": why})
    log(f"STOP: {why}: tell the lead")
    return 1


# ---- the random draws -----------------------------------------------------------------------------------------------


class Scaled:
    """One direction d's reads by signed scale c (c d on W0), each read once."""

    def __init__(self, rd: Reader, state: str, d: DW) -> None:
        self.rd, self.state, self.d = rd, state, d
        self.seen: dict[float, dict[str, Any]] = {}

    def at(self, c: float) -> dict[str, Any]:
        if c not in self.seen:
            self.seen[c] = self.rd.share(self.state, PT.scaled(self.d, c))
            log(f"  [{self.state} c {c:+.6g}] gain {self.seen[c]['gain']:.6g} numerator {self.seen[c]['numerator']:.6g}")
        return self.seen[c]

    def gain_along(self, sign: float) -> Any:
        return lambda x: self.at(sign * x)["gain"]


def one_draw(rd: Reader, state: str, dW: DW, seed: int, cfg: Config) -> dict[str, Any]:
    t0 = time.time()
    d_cpu = FT.random_direction({k: v.cpu() for k, v in dW.items()}, seed)  # as E228: from the fp32 file on the CPU
    sc = Scaled(rd, state, {k: v.to(dW[k].device) for k, v in d_cpu.items()})
    plus, minus = sc.at(cfg.probe_x)["gain"], sc.at(-cfg.probe_x)["gain"]
    first = -1.0 if plus - minus < 0 else 1.0  # a tie (difference exactly 0) takes +1 and is recorded
    tries: dict[str, Any] = {}
    matched: float | None = None
    for sign in (first, -first):
        m = PT.match(sc.gain_along(sign), cfg.target_gain, cfg.scale_cap, cfg.gain_rel_tol)
        f = [g for _, g in m["grid"] + m["trace"]]
        tries["+" if sign > 0 else "-"] = {"status": m["status"], "x": m["x"], "gain": m["f"], "max_gain": max(f), "evals": len(f)}
        if m["status"] == "ok":
            matched = sign * m["x"]
            break
    # no match at either sign: the largest gain seen is the draw's point, its status the probe sign's (cap or no_match)
    c = matched if matched is not None else max(sc.seen, key=lambda k: sc.seen[k]["gain"])
    status = "ok" if matched is not None else tries["+" if first > 0 else "-"]["status"]
    pt, change = sc.seen[c], PT.scaled(sc.d, c)
    return {"status": status, "signed_c": c, "c": abs(c), "sign": math.copysign(1.0, c), "probe_sign": first,
            "probe": {"x": cfg.probe_x, "plus": plus, "minus": minus, "odd": (plus - minus) / 2, "tie": plus == minus},
            "tries": tries, "size_rel": norm(change) / norm(dW), "gain": pt["gain"], "numerator": pt["numerator"],
            "r": pt["r"] if status == "ok" else math.nan, "max_dg": pt["max_dg"], "kl": rd.kl(state, change),
            "max_gain": max(v["gain"] for v in sc.seen.values()), "evals": len(sc.seen), "seconds": time.time() - t0}  # fmt: skip


def one_state(rd: Reader, state: str, own: dict[str, Any], experiment: str, cfg: Config) -> None:
    path = STATES / f"{state.replace('/', '_')}.json"
    rec: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    if rec.get("done"):
        log(f"{state}: done before, skipped")
        return
    if not rec:
        rec = {"experiment": experiment, "state": state, "task": rd.task(state), "draws": {},
               "own": {k: own[k] for k in ("r_route_split", "gain", "numerator", "kl", "norm", "saved")}}  # fmt: skip
        log(f"[{state} own] {json.dumps(jsonable(rec['own']))}")
    t0, dW = time.time(), rd.job.delta(state)
    for seed in cfg.seeds:
        if str(seed) not in rec["draws"]:
            rec["draws"][str(seed)] = dr = one_draw(rd, state, dW, seed, cfg)
            dump(path, rec)
            log(f"[{state} random {seed}] {json.dumps(jsonable({k: v for k, v in dr.items() if k != 'tries'}))} tries {json.dumps(jsonable(dr['tries']))}")
    rec |= {"done": True, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, rec)
    log(f"STATE {state} DONE in {rec['seconds']:.0f} s")


def states_job(experiment: str, cfg: Config, i: int, n: int) -> int:
    ctl = json.loads(CONTROL.read_text()) if CONTROL.is_file() else {}
    if not ctl.get("pass") or ctl.get("experiment") != experiment:
        log("STOP: results/control.json has not passed for this experiment: run the control job first")
        return 1
    _, saved = figure_states()
    todo = ordered(saved)[i::n]
    rd = Reader(todo, cfg.kl_batch)
    t0 = time.time()
    for k, s in enumerate(todo):
        one_state(rd, s, ctl["fine_tunes"][s], experiment, cfg)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({s}); {(time.time() - t0) / 60:.1f} min in")
    log(f"{experiment} states {i} {n} DONE")
    return 0


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, experiment = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E136, E147 and E299 read
    log(f"{experiment} {' '.join(argv)} start on {socket.gethostname()}; settings {asdict(cfg)}")
    if argv == ["control"]:
        return control_job(experiment, cfg)
    assert len(argv) == 3 and argv[0] == "states", f"usage: run.py control | states I N, got {argv}"
    return states_job(experiment, cfg, int(argv[1]), int(argv[2]))


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
