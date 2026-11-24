"""E292 (cell P2.74 run 1, card ~/commute-data/cards/E292.md, descriptive): the old near-realistic 67M tasks (E1's emoticon
and its ` http` and <|endoftext|> twins, E5's ladder tasks with two ` http` twins, E33's many cues, E53's lowercase,
E112's real case label; `commute.near_tasks`) read with every current reading of `realistic_reads.Reader.state`: the
re-gating share, the untrained-decomposition share (SVD, rotated), the neuron reading at k 64 and k matched, the pooled
steering share with its random draws, k_half; beside them each state's update size and KL (E272's), its Pile loss
change, its prior and gain. Every state is read with its matched random: a Gaussian at its per-matrix norms
(`finetune.random_direction`, E272's), read over the state's own gain. Settings, controls and expected values:
results/predictions.json (the run refuses unless it is committed unchanged). E_b over K and EAP-IG: run_defs.py.

  0              alone, first, exit 1 on a miss: a zero update reads every numerator, size, KL and Pile change exactly
                 0 (emoticon rows); E211's lowercase_s0 control repeats (run2_ctl.json, E211's tolerances); E136's
                 case_real_s0 share repeats; the known item: emoticon_s0's share repeats E28's 0.597 and its gain 13.43
                 (its prior recorded beside)                                                     -> results/stage0.json
  read TASK...   after 0: each task's saved states and their Gaussians                      -> results/read/TASK.json
  train TASK...  the task's retrain seeds, arm A, 10 steps (near_tasks.retrain): bf16 delta-W to results/large/states,
                 sha256, the per-step held-out log-odds and the step to half the gain       -> results/train/TASK.json
  read_new TASK...  after train: the retrained states and their Gaussians, into results/read/TASK.json
  score          CPU (score.py): difficulty table and twin calls                             -> results/summary.json

    sbatch --comment=E292 cluster/old_tasks_read.sbatch run 0
    sbatch --comment=E292 cluster/old_tasks_read.sbatch run read emoticon emoticon_http r0_t352_v310 ...
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from commute import finetune as FT
from commute import near_tasks as NT
from commute import old_tasks as OT
from commute import panel as PN
from commute import realistic_reads as RR
from commute import steering as ST
from commute.data import cached_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, STATES = RESULTS / "predictions.json", RESULTS / "large" / "states"
R2: Any = load_experiment("realistic_tasks", "run2")  # E211: reader (its Reader settings), e53_read, E136's folder


@dataclass(frozen=True)
class Config:
    tasks: tuple[str, ...]
    control: str  # the known item (stage 0, and the E_b stages' identity control)
    n_select: int
    strengths: tuple[float, ...]
    n_random: int
    random_seed: int
    k_fixed: int
    lr: float
    steps: int
    batch: int
    kl_lambda: float
    grad_clip: float
    kl_rows: int

    def __post_init__(self) -> None:
        for f in fields(self):  # JSON lists in, tuples kept
            if isinstance(getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(getattr(self, f.name)))
        assert self.tasks == tuple(NT.TASKS), f"filed tasks {self.tasks} are not near_tasks.TASKS"
        a = FT.ArmAConfig(o_token=0)
        assert (self.lr, self.batch, self.kl_lambda, self.grad_clip) == (a.lr, a.batch, a.kl_lambda, a.grad_clip), "arm A's recipe"
        assert self.strengths[0] == 0 and self.control in all_states()


def all_states() -> list[str]:
    return [s for t in NT.TASKS.values() for s in NT.states(t, "all")]


def load() -> tuple[Config, dict[str, Any]]:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    return config_from_settings(Config, P["settings"]), P["thresholds"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stopped(Exception):
    pass


def put(path: Path, d: dict[str, Any], key: str, rec: Any, under: str | None = None) -> None:
    (d.setdefault(under, {}) if under else d)[key] = rec
    write_stage(path, jsonable(d))
    log(f"[{under + ' ' if under else ''}{key}] {json.dumps(jsonable(brief(rec)))[:2500]}")


def stop(path: Path, d: dict[str, Any], why: str) -> Stopped:
    write_stage(path, jsonable(d | {"stopped": why}))
    return Stopped(why)


def brief(rec: Any) -> Any:
    """A record for the log without the steering grid and k_half's numerators (both stay in the JSON)."""
    if not isinstance(rec, dict) or "steering" not in rec:
        return rec
    st = {k: rec["steering"].get(k) for k in ("setting", "share", "random_mean", "off_kl")}
    return {k: v for k, v in rec.items() if k not in ("steering", "k_half")} | {"steering": st, "k_half": {u: d["k_half"] for u, d in rec["k_half"].items()}}


# ----------------------------------------------------------------------------- the states and their readings


def state_file(task: NT.NearTask, state: str) -> Path:
    return task.saved[state] if state in task.saved else STATES / f"{state}.pt"


def load_state(net: ST.Net, task: NT.NearTask, state: str) -> tuple[ST.DW, str]:
    p = state_file(task, state)
    return OT.load_module_dw(net, p), hashlib.sha256(p.read_bytes()).hexdigest()


def gauss_seed(C: Config, state: str) -> int:
    """Each state's own Gaussian draw, so the two sides of a twin pair never share a direction."""
    return C.random_seed + 1 + all_states().index(state)


def gauss(C: Config, dW: ST.DW, state: str) -> ST.DW:
    return FT.random_direction(dW, gauss_seed(C, state))


class Extra:
    """E272's KL from W0 and the Pile loss change, on the first kl_rows Pile val rows (E272's KL rows)."""

    def __init__(self, net: ST.Net, C: Config) -> None:
        self.net, self.rows = net, cached_rows("val", 0, C.kl_rows)
        self.mask = PN.full_mask(self.rows)
        self.lp0 = PN.target_logp(net, self.rows, self.mask, None)

    def of(self, dW: ST.DW) -> dict[str, float]:
        return {"kl": PN.kl_from_w0(self.net, self.rows, dW), "pile_loss_change": self.lp0 - PN.target_logp(self.net, self.rows, self.mask, dW)}


def flat(rec: dict[str, Any], gain: float) -> dict[str, float]:
    """Every reading of a state record over `gain` (its own, or for a Gaussian its state's): `realistic_reads.numerators`
    divided by it (nan under MIN_GAIN), the gains themselves and k_half's counts left out."""
    num = RR.numerators(rec)
    return {k: RR.ratio(v, gain) for k, v in num.items() if not k.startswith(("gain", "k_half"))}


def read_state(rd: RR.Reader, ex: Extra, C: Config, task: NT.NearTask, state: str, out: dict[str, Any], path: Path) -> None:
    """One state and its Gaussian: every reading, size, KL, Pile change; `flat` beside each (the twin rule reads it)."""
    if state in out.get("states", {}) and state in out.get("random", {}):
        log(f"{state}: read before, skipped")
        return
    t0, net = time.time(), rd.net
    dW, sha = load_state(net, task, state)
    rec: dict[str, Any] = rd.state(dW, NT.seed_of(state)) | ex.of(dW) | {"file": str(state_file(task, state)), "sha256": sha}
    rec["flat"], rec["seconds"] = flat(rec, rec["gain"]["lo"]), time.time() - t0
    put(path, out, state, rec, "states")
    G = gauss(C, dW, state)
    g: dict[str, Any] = rd.state(G, NT.seed_of(state)) | ex.of(G) | {"draw": gauss_seed(C, state)}
    g["flat"] = flat(g, rec["gain"]["lo"])
    put(path, out, state, g, "random")


def reader(net: ST.Net, C: Config, r: RR.Read) -> RR.Reader:
    return R2.reader(net, C, r)  # E211's: W0's SVD and its rotation with E136's k_m, E98's k_half settings, steering's


def read_task(net: ST.Net, C: Config, name: str, which: str) -> None:
    task, path = NT.TASKS[name], RESULTS / "read" / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": name}
    if which == "new":
        tr = RESULTS / "train" / f"{name}.json"
        assert tr.is_file() and json.loads(tr.read_text()).get("done"), f"{name}: run train first"
    r = NT.rows(task)
    rd = reader(net, C, r.read)
    put(path, out, "rows", {"type": type(r.read).__name__, "n": len(r.read.toks), "T": r.read.toks.shape[1], "pos": r.read.pos,
                            "n_fit": len(r.read.fit), "prior_w0": float(rd.w0["rows"]["lo"].mean())})  # fmt: skip
    ex = Extra(net, C)
    for state in NT.states(task, which):
        read_state(rd, ex, C, task, state, out, path)
    put(path, out, f"done_{which}", True)


# ----------------------------------------------------------------------------- stage 0


def zero(net: ST.Net) -> ST.DW:
    return {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()}


def check(path: Path, out: dict[str, Any], name: str, got: dict[str, float], want: dict[str, float], tol: dict[str, float], relative: bool = False) -> None:
    rows = {k: {"got": got[k], "want": v, "tol": tol[k], "relative": relative} for k, v in want.items()}
    put(path, out, name, rows)
    off = lambda k: abs(got[k] - want[k]) / (abs(want[k]) if relative else 1.0)
    if bad := [k for k in want if not off(k) <= tol[k]]:
        raise stop(path, out, f"{name} does not repeat: {bad}")


def stage0(net: ST.Net, C: Config, th: dict[str, Any]) -> None:
    path, out = RESULTS / "stage0.json", {"experiment": "E292", "card": torch.cuda.get_device_name()}
    ex, emo = Extra(net, C), NT.rows(NT.TASKS["emoticon"])
    rd = reader(net, C, emo.read)
    z = rd.state(zero(net), 0) | ex.of(zero(net))
    exact = RR.numerators(z) | {k: z[k] for k in ("size", "kl", "pile_loss_change")}
    put(path, out, "zero", exact)
    if bad := {k: v for k, v in exact.items() if v != 0}:
        raise stop(path, out, f"a zero update does not read exactly 0: {bad}")
    low = reader(net, C, R2.e53_read("lowercase")).state(load_state(net, NT.TASKS["lowercase"], "lowercase_s0")[0], 0)
    got = {"share": low["share"]["share"]["lo"], "svd": low["untrained"]["share"]["lo"]["svd"], "rotated": low["untrained"]["share"]["lo"]["rotated"],
           "gain": low["gain"]["lo"], "neuron_k64": low["neuron"]["k64"]["share"]["lo"], "neuron_kmatched": low["neuron"]["kmatched"]["share"]["lo"],
           "steer_pooled": low["steering"]["share"]["lo"], "erank": low["erank"], "k_half": low["k_half"]["lo"]["k_half"],
           "r_half_num": low["k_half"]["lo"]["r_half_num"]}  # fmt: skip
    e211 = json.loads((EXPERIMENTS / "realistic_tasks/results/run2_ctl.json").read_text())["reproduce e53/lowercase_s0"]
    check(path, out, "E211 lowercase_s0", got, {k: v["got"] for k, v in e211.items()}, th["e211_tol"])
    case = NT.TASKS["case_real"]
    c = reader(net, C, NT.rows(case).read).state(load_state(net, case, "case_real_s0")[0], 0)
    saved = json.loads((R2.E136 / "stage1.json").read_text())["states"]["case_label/real_s0"]
    put(path, out, "E136 case_real_s0 untrained (beside: E136 read at batch 64, this Reader at 32)",
        {k: {"got": c["untrained"]["share"]["lo"][k], "saved": saved[k]} for k in ("svd", "rotated")})  # fmt: skip
    check(path, out, "E136 case_real_s0 share", {"trained": c["share"]["share"]["lo"]}, {"trained": saved["trained"]}, th["e136_tol"])
    k = th["known"]
    e = rd.state(load_state(net, NT.TASKS["emoticon"], C.control)[0], 0)
    check(path, out, f"known {C.control} share", {"share": e["share"]["share"]["lo"]}, {"share": k["share"]}, {"share": k["share_tol"]})
    put(path, out, f"known {C.control} T0 (beside)", {"got": e["prior"]["lo"], "saved": k["T0"]})
    check(path, out, f"known {C.control} gain", {"gain": e["gain"]["lo"]}, {"gain": k["gain"]}, {"gain": k["cell_rel_tol"]}, relative=True)
    put(path, out, "passed", True)


def require_stage0() -> None:
    p = RESULTS / "stage0.json"
    if not (p.is_file() and json.loads(p.read_text()).get("passed")):
        sys.exit(f"{p} is missing or did not pass: run stage 0 first, alone")


# ----------------------------------------------------------------------------- train


def train(net: ST.Net, C: Config, name: str) -> None:
    task, path = NT.TASKS[name], RESULTS / "train" / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": name}
    r = NT.rows(task)
    STATES.mkdir(parents=True, exist_ok=True)
    for seed in task.retrain:
        state = f"{name}_s{seed}"
        if state in out.get("states", {}):
            continue
        t0 = time.time()
        dW, curve = NT.retrain(net, r, seed, C.lr, C.steps, log)
        torch.save({k: v.to(torch.bfloat16) for k, v in dW.items()}, STATES / f"{state}.pt")
        rec = {"seed": seed, "steps": C.steps, "curve_lo": curve, "gain_curve_end": curve[-1] - curve[0], "steps_to_half": NT.steps_to_half(curve),
               "sha256": hashlib.sha256((STATES / f"{state}.pt").read_bytes()).hexdigest(), "seconds": time.time() - t0}  # fmt: skip
        put(path, out, state, rec, "states")
    put(path, out, "done", True)


def main(args: list[str]) -> int:
    stages = ("0", "read", "train", "read_new")
    C, th = load()
    if not args or args[0] not in stages or (args[0] == "0") != (len(args) == 1) or not set(args[1:]) <= set(C.tasks):
        sys.exit(f"usage: run.py 0 | {{read, train, read_new}} TASK [TASK ...], TASK in {C.tasks}")
    if args[0] != "0":
        require_stage0()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    try:
        if args[0] == "0":
            stage0(net, C, th)
        for name in args[1:]:
            train(net, C, name) if args[0] == "train" else read_task(net, C, name, "saved" if args[0] == "read" else "new")
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print(f"E292 {' '.join(args)} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
