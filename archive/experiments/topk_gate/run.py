"""P2.47: the re-gating share with VPD's subcomponents kept and Gamma replaced by a neuron-style top-k gate, on the nine
built pairs (E187's, their saved fine-tunes and E136's read rows). Plan reports/plans/2026-11-20_plan_p2-47-topk-gate.md;
settings, expected values and rules in results/predictions.json (the run refuses unless it is committed unchanged).
Forward passes only; nothing is trained.

Per fine-tune, one pass of `topk_gate.gate_cells` reads five gates on the same windows, batch and ci_len as the saved
share (`local_faith.route_split` through E136's Job.trained):
  gamma       Gamma's gates: the positive control, must reproduce the saved share within reproduce_tol (a stop if not)
  gamma_bin   Gamma's gates above 0.5 as 1, else 0 (reported: separates binarising from choosing)
  topk_x<s>   1 on the k alive subcomponents of largest |x . V_c| at each position, read from the clean forward of the
              model as it stands; k = s times E136's matched k_m (Gamma's mean count above 0.5 per position at W0, the
              k of the TopK neuron share's "kmatched"); s = 1 is scored, 0.5 and 2 are reported beside it
Share of a gate = mean(L01 - L00) / mean(T1 - T0).

  read      GPU: stage 0, alone: a zero change on the first state's rows reads every gate's numerator and the gain
            exactly 0 (exit 1 on a miss); stage 1: every state, written to results/states/<state>.json as it ends (a
            rerun skips finished states), per-row cells to results/large/cells/; the projected time checked after 4
            states; then score
  score     no GPU: the scorecard's pair rule per gate, the filed rule P1 -> results/score.json
  manifest  no GPU: the inputs outside git -> results/manifest.txt

    sbatch --comment=E### cluster/topk_gate.sbatch read
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from commute import steering as ST
from commute import topk_gate as TG
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

WIN: Any = load_experiment("share_untrained", "windows")  # E136's read windows
SM: Any = load_experiment("size_matched", "run")  # E187: the nine built pairs, their state names and inventory
HERE = Path(__file__).resolve().parent
RESULTS, STATES, CELLS = HERE / "results", HERE / "results" / "states", HERE / "results" / "large" / "cells"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
REPO = EXPERIMENTS.parent
ALIVE = EXPERIMENTS / "transport_test" / "results" / "large" / "alive.pt"  # round 1's alive masks (the paper's rule)
K_M = EXPERIMENTS / "share_untrained" / "results" / "stage0.json"  # .k_m: Gamma's on-count per matrix at W0
SAVED = {"e136": EXPERIMENTS / "share_untrained/results/stage1.json", "e147": EXPERIMENTS / "refit_96key/results/stage1.json"}
SCORED = "topk_x1"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_stage(path, jsonable(obj))


def stop(path: Path, obj: dict[str, Any], why: str) -> None:
    dump(path, obj | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    k_scales: tuple[float, ...]
    projected_limit_min: float

    def __post_init__(self) -> None:
        assert 1.0 in self.k_scales and len(set(self.k_scales)) == len(self.k_scales)
        assert SCORED == f"topk_x{1.0:g}"


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def fname(state: str) -> str:
    return state.replace("/", "_")


# ---- the catalogue: pairs, states, rows, saved shares ---------------------------------------------------------------


def catalogue() -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, Any]]:
    """E187's nine built pairs, each (reuse side, new side) in E136/E147 state names, and one inventory of their states."""
    names, _, inv = SM.pairs()
    return {p: names[p] for p in SM.BUILT}, inv


def order(pairs: dict[str, tuple[list[str], list[str]]]) -> list[str]:
    """Every state once, each pair's first reuse and first new state before any second seed (an early miss shows early)."""
    sides = [side for r, n in pairs.values() for side in (r, n)]
    return list(dict.fromkeys(s[i] for i in range(max(map(len, sides))) for s in sides if i < len(s)))


def saved_share(state: str) -> float:
    """The scorecard's share cell (experiments/method_table/results/final.json names these sources)."""
    if state.startswith("e58/"):
        return float(json.loads(SAVED["e147"].read_text())["states"][state]["e136"]["trained"])
    return float(json.loads(SAVED["e136"].read_text())["states"][state]["trained"])


def state_inputs(state: str, inv: dict[str, Any], W0: dict[str, Tensor]) -> tuple[Any, dict[str, Tensor]]:
    task = inv["states"][state]["task"]
    d = torch.load(REPO / inv["states"][state]["file"], weights_only=False)
    return WIN.load_read(task, inv["tasks"][task]), {k: v.float().to(W0[k].device) for k, v in d.items()}


# ---- the read (GPU) -------------------------------------------------------------------------------------------------


class Reader:
    def __init__(self, cfg: Config) -> None:
        self.net = ST.Net("cuda")
        self.model, self.W0 = self.net.model, self.net.W0
        alive = {k: v.bool() for k, v in torch.load(ALIVE, weights_only=False)["alive"].items()}
        k_m = {k: int(v) for k, v in json.loads(K_M.read_text())["k_m"].items()}
        assert sorted(alive) == sorted(k_m) == sorted(self.W0), "alive masks, k_m and W0 must name the 24 matrices"
        self.k = {f"topk_x{s:g}": TG.scaled_k(k_m, s, alive) for s in cfg.k_scales}
        self.gates: dict[str, TG.Gate] = {"gamma": TG.gamma, "gamma_bin": TG.gamma_bin}
        self.gates |= {name: TG.topk(k, alive) for name, k in self.k.items()}
        self.alive_n = {k: int(v.sum()) for k, v in alive.items()}

    def read(self, r: Any, dW: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Any]]:
        cells = TG.gate_cells(self.model, self.W0, dW, r.toks, r.half, r.token, r.batch, r.ci_len, self.gates)
        return cells, TG.shares(cells, list(self.gates))


def zero_check(rd: Reader, state: str, inv: dict[str, Any]) -> None:
    """The do-nothing: W0 as the fine-tune reads every gate's numerator, the gain and every per-row L01 - L00 exactly 0."""
    path = RESULTS / "zero.json"
    r, dW = state_inputs(state, inv, rd.W0)
    cells, sh = rd.read(r, {k: torch.zeros_like(v) for k, v in dW.items()})
    exact = {"gain": sh["gain"], **{g: sh[g]["numerator"] for g in rd.gates}}
    exact |= {f"max_row_{g}": float((cells[f"L01_{g}"] - cells[f"L00_{g}"]).abs().max()) for g in rd.gates}
    z = {"experiment": EXPERIMENT, "state": state, "n_rows": len(r.toks), "exact": exact, "shares": sh, "k": rd.k, "alive": rd.alive_n}
    log(f"[zero {state}] {json.dumps(jsonable(exact))}")
    if bad := {k: v for k, v in exact.items() if v != 0}:
        stop(path, z, f"a zero change does not read exactly 0: {bad}")
    dump(path, z | {"done": True})
    log("STAGE 0 DONE")


def one_state(rd: Reader, state: str, inv: dict[str, Any], th: dict[str, Any]) -> float:
    path = STATES / f"{fname(state)}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{state}: done before, skipped")
        return 0.0
    t0 = time.time()
    r, dW = state_inputs(state, inv, rd.W0)
    cells, sh = rd.read(r, dW)
    saved = saved_share(state)
    rec = {"experiment": EXPERIMENT, "state": state, "n_rows": len(r.toks), "half": r.half, "saved_share": saved, "shares": sh,
           "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}  # fmt: skip
    log(f"[{state}] {json.dumps(jsonable(rec))}")
    if not abs(sh["gamma"]["share"] - saved) <= th["reproduce_tol"]:
        stop(path, rec, f"{state}: the gamma gate's share {sh['gamma']['share']:.4f} does not reproduce the saved {saved:.4f}")
    CELLS.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.float() for k, v in cells.items()}, CELLS / f"{fname(state)}.pt")
    dump(path, rec | {"done": True})
    return rec["seconds"]


def read_job(cfg: Config, th: dict[str, Any]) -> None:
    pairs, inv = catalogue()
    todo = order(pairs)
    rd = Reader(cfg)
    log(f"k per gate {json.dumps(rd.k)}; alive {json.dumps(rd.alive_n)}; {len(todo)} states")
    zero_check(rd, todo[0], inv)
    t_job, secs = time.time(), []
    for i, state in enumerate(todo):
        secs.append(one_state(rd, state, inv, th))
        log(f"ITEM {i + 1} of {len(todo)} DONE ({state})")
        if i == 3:
            minutes = (time.time() - t_job) / 60 / 4 * len(todo)
            log(f"[time] projected {minutes:.1f} min for stage 1 after 4 states (limit {cfg.projected_limit_min})")
            if minutes > cfg.projected_limit_min:
                stop(RESULTS / "time.json", {"projected_minutes": minutes}, "projected time over the filed limit")
    log("STAGE 1 DONE")
    score(th)


# ---- score (no GPU) -------------------------------------------------------------------------------------------------


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan  # nan is written as the string "nan"


def pair_call(reuse: list[float], new: list[float]) -> str:
    """The scorecard's one rule (E133, cell P2.23): 'built' when every reuse seed lies above every new seed, 'reversed'
    when every one lies below, 'not apart' otherwise (ties included); 'not read' when any value is not finite."""
    if not reuse or not new or not all(math.isfinite(v) for v in reuse + new):
        return "not read"
    return "built" if min(reuse) > max(new) else "reversed" if max(reuse) < min(new) else "not apart"


def rule_p1(calls: dict[str, str], th: dict[str, Any]) -> dict[str, Any]:
    """P1 on the scored gate: 'no call' if any pair is not read; else 'falls to the neuron level' when the pairs kept
    apart in the built direction number at most the neuron version's count, 'holds above' otherwise."""
    n = sum(c == "built" for c in calls.values())
    if any(c == "not read" for c in calls.values()) or len(calls) != 9:
        return {"verdict": "no call", "n_built": n}
    v = "falls to the neuron level" if n <= th["neuron_level"] else "holds above"
    return {"verdict": v, "n_built": n, "against_neuron_top": "at or under" if n <= th["neuron_top_level"] else "above"}


def score(th: dict[str, Any]) -> dict[str, Any]:
    pairs, _ = catalogue()
    recs = {s: json.loads((STATES / f"{fname(s)}.json").read_text()) for s in order(pairs) if (STATES / f"{fname(s)}.json").is_file()}
    recs = {s: v for s, v in recs.items() if v.get("done")}
    gates = list(next(iter(recs.values()))["shares"]) if recs else []
    gates = [g for g in gates if g != "gain"]

    def value(s: str, g: str) -> float:
        if s not in recs or num(recs[s]["shares"]["gain"]) < MIN_GAIN:
            return math.nan
        return num(recs[s]["shares"][g]["share"])

    out: dict[str, Any] = {"experiment": EXPERIMENT, "n_states": len(recs), "n_planned": len(order(pairs)), "pairs": {}, "tally": {}}
    for g in gates:
        calls = {}
        for p, (r, n) in pairs.items():
            a, b = [value(s, g) for s in r], [value(s, g) for s in n]
            gap = fmean(a) - fmean(b) if all(math.isfinite(v) for v in a + b) else math.nan
            calls[p] = pair_call(a, b)
            out["pairs"].setdefault(p, {})[g] = {"call": calls[p], "gap": gap, "reuse": a, "new": b}
        out["tally"][g] = {c: sum(v == c for v in calls.values()) for c in ("built", "reversed", "not apart", "not read")}
        if g == SCORED:
            out["P1"] = rule_p1(calls, th)
    log(f"score: tally {json.dumps(out['tally'])}; P1 {json.dumps(out.get('P1'))}")
    dump(RESULTS / "score.json", out)
    return out


def manifest() -> None:
    """The inputs outside git: the model, the alive masks, E53's cached rows, every state's update and every task's windows."""
    pairs, inv = catalogue()
    todo = order(pairs)
    shared = ["artifacts/vpd/", "artifacts/sets/", str(ALIVE.relative_to(REPO))]
    files = sorted({inv["states"][s]["file"] for s in todo} | {inv["tasks"][inv["states"][s]["task"]].get("path", "") for s in todo} - {""})
    (RESULTS / "manifest.txt").write_text("\n".join([f"# {EXPERIMENT}: {len(todo)} states", *shared, *files]) + "\n")
    log(f"manifest: {len(todo)} states, {len(files)} files")


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, th = load_config()
    if argv == ["score"]:
        score(th)
    elif argv == ["manifest"]:
        manifest()
    else:
        assert argv == ["read"], f"usage: run.py read | score | manifest, got {argv}"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E136's stage-1 job
        log(f"{EXPERIMENT} read start; settings {asdict(cfg)}; thresholds {th}")
        read_job(cfg, th)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
