"""E233 (cell P2.48): British house style against a letter-scrambled twin on the 67M. Both tasks rewrite the same Pile
rows at the same word occurrences (`spelling.py`); each is fine-tuned (E211's LM recipe) and saved at the first step its
update size reaches each target; the re-gating share r, the neurons for half the gain (k_half) and the update's
effective rank are read at the changed words. Plan reports/plans/2026-11-20_plan_p2-48-british-style.md; settings,
expected values and rules results/predictions.json (the run refuses unless it is committed unchanged, and the scored
job unless it is filed). Readers: realistic_reads.share_of, k_half_of (E211's), update_spectrum.state.

Stages, each writing its JSON as it ends and printing every number; a filed stop prints "STOP: ..." and exits 1:
  data    the word map, both tasks' rows and the read occurrences (tokenizer only) -> results/large/data.pt, results/data.json
  0       alone, before any training: E53's lowercase_s0 reproduces its saved readings; a zero update reads exactly 0;
          the base contrast (W0's log-odds of the scored token, real minus twin, and the twin's median probability)
  1       seed 0 of both tasks: each reaches the scored size and gains at least the floor there, else stop
  2       seeds 1 and 2 of both tasks
States: results/states/<task>_s<seed>.json (a rerun skips a written one); updates in bf16 under results/large/states/,
read from the saved file.

    sbatch --comment=E233 cluster/british_style.sbatch run          (scored; ~20 min on an L40)
    sbatch --comment='E233 rehearse: british-style' cluster/british_style.sbatch rehearse
    python experiments/british_style/run.py score                   (no GPU) -> results/scored/{score,handoff}.json
    (score-rehearse: the same on the rehearsal's states, one seed, so 'unreadable' by the one-value rule)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import env, vpd
from commute import finetune as FT
from commute import realistic as R
from commute import realistic_reads as RR
from commute import steering as ST
from commute import update_spectrum as US
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.lowercase import batch_order
from commute.open_pair import jsonable

SP: Any = load_experiment("british_style", "spelling")
WIN: Any = load_experiment("share_untrained", "windows")  # E136's read windows (the control)
HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
LARGE = RESULTS / "large"
E136, E98 = EXPERIMENTS / "share_untrained" / "results", EXPERIMENTS / "neuron_r" / "results"
TASKS = ("real", "twin")


@dataclass(frozen=True)
class Config:
    word_list: str
    list_url: str
    list_sha256: dict[str, str]
    exclude: tuple[str, ...]
    scramble_seed: int
    scramble_tries: int
    train_set: str
    held_set: str
    row_len: int
    min_per_row: int
    n_train: int
    rate: float
    batch: int
    grad_clip: float
    step_cap: int
    targets: tuple[float, ...]
    seeds: tuple[int, ...]
    half: int
    n_read: int
    per_row: int
    control_state: str
    rehearse_cap: int
    rehearse_targets: tuple[float, ...]
    trace_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        for f in fields(self):  # JSON lists in, tuples kept
            if isinstance(getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(getattr(self, f.name)))
        assert list(self.targets) == sorted(self.targets) and list(self.rehearse_targets) == sorted(self.rehearse_targets)
        assert self.n_train >= self.step_cap * self.batch and self.rehearse_cap < self.step_cap and self.seeds[0] == 0
        assert self.row_len <= vpd.SEQ_LEN and 2 * self.half + 1 < self.row_len


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stopped(Exception):
    pass


def stop(path: Path, obj: dict[str, Any], why: str) -> Stopped:
    write_stage(path, jsonable(obj | {"stopped": why}))
    return Stopped(why)


# ----------------------------------------------------------------------------- data


def build_data(C: Config) -> dict[str, Any]:
    """Both tasks' training and held rows and the read occurrences, built once (results/large/data.pt)."""
    path = LARGE / "data.pt"
    if path.is_file():
        return torch.load(path, weights_only=False)
    tok = vpd.tokenizer()
    lists = SP.word_lists(env.CACHE_DIR / "spelling" / C.word_list.split("@")[1].split()[0], C.list_url, C.list_sha256)
    pairs = SP.build_pairs(lists, C.exclude, tok, C.scramble_seed, C.scramble_tries)
    rows = {s: torch.from_numpy(np.load(env.CACHE_DIR / "sets" / f"{s}.npy")).long() for s in (C.train_set, C.held_set)}
    train = SP.task_rows(rows[C.train_set], tok, pairs, C.n_train, C.min_per_row, C.row_len, log)
    held = SP.task_rows(rows[C.held_set], tok, pairs, None, C.min_per_row, C.row_len, log)
    at = SP.read_at(held["occ"], C.half, C.per_row, C.n_read, C.row_len)
    windows = {t: R.cut_windows(held[t], at[:, [0, 1 + i]], C.half) for i, t in enumerate(TASKS)}
    d = {"pairs": {a: asdict(p) for a, p in pairs.items()}, "train": {t: train[t].int() for t in TASKS},
         "held": {t: held[t].int() for t in TASKS}, "at": at, "windows": {t: w.int() for t, w in windows.items()}}  # fmt: skip
    ks = [p.k for p in pairs.values()]
    tc = [len(tok.encode(" " + p.twin)) - len(tok.encode(" " + p.british)) for p in pairs.values()]
    stats = {"pairs": len(pairs), "k": {str(k): ks.count(k) for k in sorted(set(ks))}, "twin_extra_tokens": {str(x): tc.count(x) for x in sorted(set(tc))},
             **{f"{n}_{k}": v[k] for n, v in (("train", train), ("held", held)) for k in ("rows_read", "rows_kept", "natural_rows")},
             "train_occurrences": len(train["occ"]), "held_occurrences": len(held["occ"]), "read": len(at)}  # fmt: skip
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.save(d, path)
    write_stage(RESULTS / "data.json", jsonable(stats | {"pairs_map": d["pairs"], "data_pt_sha256": sha(path)}))
    log(f"[data] {json.dumps(stats)}")
    return d


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reads(d: dict[str, Any], half: int) -> dict[str, RR.Read]:
    out = {}
    for t in TASKS:
        w = d["windows"][t].long()
        out[t] = RR.Read(t, w, w[:, half + 1], half, w, w[:, half + 1])
    return out


# ----------------------------------------------------------------------------- readings


def readings(net: ST.Net, r: RR.Read, dW: ST.DW, S: dict[str, Any]) -> dict[str, Any]:
    sh, kh = RR.share_of(net, r, dW), RR.k_half_of(net, r, dW, S)["lo"]
    return {"share": sh["share"]["lo"], "share_num": sh["numerator"]["lo"], "gain": sh["gain"]["lo"], "k_half": kh["k_half"],
            "r_half": kh["r_half"], "r_half_num": kh["r_half_num"], "G": kh["G"], "k_grid_numerators": kh["numerators"],
            "erank": US.state(dW)["erank"], "size": R.update_size(dW, net.W0)}  # fmt: skip


def shown(rec: dict[str, Any]) -> str:
    return json.dumps(jsonable({k: v for k, v in rec.items() if k != "k_grid_numerators"}))


@torch.no_grad()
def prior(net: ST.Net, r: RR.Read) -> Tensor:
    """W0's log-odds of each window's scored token."""
    lg = torch.cat([net.run(r.toks[s : s + 64])[0][:, r.pos] for s in range(0, len(r.toks), 64)])
    return r.units_of(r.tgt)(lg, slice(0, len(r.toks)))["lo"]


def stage0(net: ST.Net, C: Config, th: dict[str, Any], S: dict[str, Any], rd: dict[str, RR.Read], out: Path) -> None:
    path = out / "stage0.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log("stage 0: done before")
        return
    inv = json.loads((E136 / "inventory.json").read_text())
    name, tol = C.control_state, th["reproduce_tol"]
    task = inv["states"][name]["task"]
    w = WIN.load_read(task, inv["tasks"][task])
    ctl = RR.Read(task, w.toks, w.toks[:, w.half + 1], w.half, w.toks, w.toks[:, w.half + 1])
    dW = {k: v.float() for k, v in torch.load(EXPERIMENTS.parent / inv["states"][name]["file"], weights_only=False).items()}
    got = readings(net, ctl, dW, S)
    e98 = json.loads((E98 / "carrying.json").read_text())["rows"][name.split("/")[1]]
    svd = json.loads((EXPERIMENTS / "update_svd/results/results.json").read_text())["pairs"]["e53_lowercase_cipher"]["reuse"][name.split("/")[1]]
    saved = {"share": json.loads((E136 / "stage1.json").read_text())["states"][name]["trained"], "k_half": e98["k_half"],
             "r_half_num": e98["r_half_num"], "erank": svd["erank"]}  # fmt: skip
    rec: dict[str, Any] = {"control": {k: {"got": got[k], "saved": saved[k], "tol": tol[k]} for k in saved}}
    log(f"stage 0 control {name}: {json.dumps(jsonable(rec['control']))}")
    miss = [k for k, v in rec["control"].items() if not (v["got"] is not None and abs(v["got"] - v["saved"]) <= v["tol"])]
    zero = readings(net, rd["real"], {k: torch.zeros_like(x).cpu() for k, x in net.W0.items()}, S)
    rec["do_nothing"] = zero
    log(f"stage 0 do-nothing: {shown(zero)}")
    miss += [f"do-nothing {k} {zero[k]}" for k in ("share_num", "gain", "r_half_num", "G") if zero[k] != 0]
    miss += [] if zero["k_half"] is None and zero["erank"] is None else [f"do-nothing k_half {zero['k_half']}, erank {zero['erank']}"]
    lo = {t: prior(net, rd[t]) for t in TASKS}
    p_twin = float(torch.sigmoid(lo["twin"].median()))
    rec["base"] = {"prior": {t: float(x.mean()) for t, x in lo.items()}, "prior_median": {t: float(x.median()) for t, x in lo.items()},
                   "contrast": float(lo["real"].mean() - lo["twin"].mean()), "twin_median_p": p_twin,
                   "bars": {"contrast_min": th["base_contrast_min"], "twin_p_max": th["twin_p_max"]}}  # fmt: skip
    log(f"stage 0 base contrast: {json.dumps(rec['base'])}")
    if not rec["base"]["contrast"] >= th["base_contrast_min"]:
        miss.append(f"base contrast {rec['base']['contrast']:.3f} under {th['base_contrast_min']}")
    if not p_twin <= th["twin_p_max"]:
        miss.append(f"twin median probability {p_twin:.3g} over {th['twin_p_max']}")
    if miss:
        raise stop(path, rec, "stage 0: " + "; ".join(miss))
    write_stage(path, jsonable(rec | {"done": True}))
    log("STAGE 0 DONE")


# ----------------------------------------------------------------------------- training to size


def to_size(net: ST.Net, C: Config, rows: Tensor, seed: int, targets: tuple[float, ...], cap: int) -> tuple[dict[float, Any], dict[str, Any]]:
    """One fine-tune (E211's LM recipe), its update kept at the first step each target size is reached (the cap's, marked
    not reached, for a target it never reaches); the trace holds the size and wall time at trace_steps."""
    got: dict[float, dict[str, Any]] = {}
    trace: dict[str, Any] = {}
    t0 = time.time()

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        dW = {k: p.detach().float() - net.W0[k] for k, p in params.items()}
        size = R.update_size(dW, net.W0)
        if step in C.trace_steps or step == cap:
            trace[str(step)] = {"size": size, "seconds": time.time() - t0}
            log(f"[size] step {step}: {size:.5f} after {time.time() - t0:.1f} s")
        for t in [t for t in targets if t not in got and (size >= t or step == cap)]:
            got[t] = {"dW": {k: v.cpu() for k, v in dW.items()}, "step": step, "size": size, "reached": size >= t}
        if len(got) == len(targets):
            raise FT.StopTraining

    cfg = FT.ArmBConfig(lr=C.rate, batch=C.batch, steps=cap, checkpoints=(), grad_clip=C.grad_clip, seed=seed)
    order = batch_order(rows, seed)
    try:
        FT.finetune_arm_b(net.model, order, order[:8], cfg, net.device, log, on_step)
    except FT.StopTraining:
        pass
    return got, trace


def run_state(net: ST.Net, C: Config, d: dict[str, Any], rd: dict[str, RR.Read], S: dict[str, Any], task: str, seed: int,
              out: Path, targets: tuple[float, ...], cap: int) -> dict[str, Any]:  # fmt: skip
    path = out / "states" / f"{task}_s{seed}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{task} s{seed}: done before, skipped")
        return json.loads(path.read_text())
    states, trace = to_size(net, C, d["train"][task].long(), seed, targets, cap)
    rec: dict[str, Any] = {"task": task, "seed": seed, "trace": trace, "targets": {}}
    for target, st in states.items():
        f = LARGE / out.name / "states" / f"{task}_s{seed}_{target}.pt"
        f.parent.resolve().mkdir(parents=True, exist_ok=True)
        torch.save({k: v.to(torch.bfloat16) for k, v in st["dW"].items()}, f)
        dW = {k: v.float() for k, v in torch.load(f, weights_only=False).items()}  # read the state as saved
        t0 = time.time()
        rec["targets"][str(target)] = {"step": st["step"], "reached": st["reached"], "file": str(f.relative_to(EXPERIMENTS.parent)),
                                       "sha256": sha(f), **readings(net, rd[task], dW, S), "read_seconds": time.time() - t0}  # fmt: skip
        log(f"{task} s{seed} @ {target}: {shown(rec['targets'][str(target)])}")
    rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    write_stage(path, jsonable(rec))
    return rec


def gain_floor(rec: dict[str, Any], target: float, floor: float) -> str | None:
    t = rec["targets"][str(target)]
    if not t["reached"]:
        return f"{rec['task']} s{rec['seed']} did not reach size {target} (at {t['size']:.5f}, step {t['step']})"
    return None if t["gain"] >= floor else f"{rec['task']} s{rec['seed']} gains {t['gain']:.3f} at size {target}, under {floor}"


def train_all(net: ST.Net, C: Config, th: dict[str, Any], d: dict[str, Any], rd: dict[str, RR.Read], S: dict[str, Any], rehearse: bool) -> None:
    out = RESULTS / ("rehearse" if rehearse else "scored")
    targets, cap = (C.rehearse_targets, C.rehearse_cap) if rehearse else (C.targets, C.step_cap)
    (out / "states").mkdir(parents=True, exist_ok=True)
    stage0(net, C, th, S, rd, out)
    for seed in C.seeds[:1] if rehearse else C.seeds:
        recs = [run_state(net, C, d, rd, S, t, seed, out, targets, cap) for t in TASKS]
        if seed == 0:
            misses = [m for r in recs if (m := gain_floor(r, targets[-1], th["gain_floor"]))]
            write_stage(out / "stage1.json", jsonable({"floor": th["gain_floor"], "target": targets[-1], "misses": misses, "rehearse": rehearse}))
            log(f"STAGE 1 gain floor: {misses or 'all clear'}")
            if misses and not rehearse:
                raise stop(out / "stage1.json", {"misses": misses}, "stage 1: " + "; ".join(misses))


# ----------------------------------------------------------------------------- score (no GPU)


def value(x: Any) -> float | None:
    return None if x is None or (isinstance(x, str) and x == "nan") else float(x)


def score(P: dict[str, Any], out: Path, targets: tuple[float, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    """The filed calls at each size, the card call at the last -> <out>/score.json; the states for R -> <out>/handoff.json."""
    recs = {(t, s): json.loads((out / "states" / f"{t}_s{s}.json").read_text()) for t in TASKS for s in seeds}
    res: dict[str, Any] = {"experiment": P["experiment"], "sizes": {}}
    for target in targets:
        calls = {}
        for m in ("share", "erank", "k_half"):
            side = {t: [value(recs[(t, s)]["targets"][str(target)][m]) for s in seeds] for t in TASKS}
            c = R.seed_call(m, side["real"], side["twin"])
            c["filed"] = "unreadable" if c["call"] == "one value a side" else R.filed(c, "+")
            calls[m] = c
        res["sizes"][str(target)] = calls
    last = str(targets[-1])
    p1 = res["sizes"][last]["share"]
    f, gap = p1["filed"], p1["gap"]
    res["card"] = ("rival a" if gap >= 0.1 else "between") if f == "hit" else "rival b" if f in ("no call", "miss") else "none"
    res["gain"] = {t: [recs[(t, s)]["targets"][last]["gain"] for s in seeds] for t in TASKS}
    res["steps"] = {t: [recs[(t, s)]["targets"][last]["step"] for s in seeds] for t in TASKS}
    write_stage(out / "score.json", jsonable(res))
    states = {f"{t}_s{s}_{g}": {"task": t, "seed": s, "target": float(g), **{k: v[k] for k in ("file", "sha256", "step", "size")}}
              for (t, s), r in recs.items() for g, v in r["targets"].items()}  # fmt: skip
    hand = {"experiment": P["experiment"], "states": states, "windows": {"file": str((LARGE / "data.pt").relative_to(EXPERIMENTS.parent)),
            "keys": "windows[task] (500, 127) int32, scored token at index half + 1 = 64; train[task], held[task] rows; at [row, real pos, twin pos]"}}  # fmt: skip
    write_stage(out / "handoff.json", hand)
    log(json.dumps(jsonable(res)))
    return res


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) == 2 else ""
    if mode not in ("run", "rehearse", "score", "score-rehearse", "data"):
        sys.exit("usage: run.py {run | rehearse | score | score-rehearse | data}")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if mode in ("run", "score") and not P.get("filed"):
        sys.exit("predictions.json is not filed: the scored run refuses")
    C, th = config_from_settings(Config, P["settings"]), P["thresholds"]
    if mode == "score":
        score(P, RESULTS / "scored", C.targets, C.seeds)
        return 0
    if mode == "score-rehearse":
        score(P, RESULTS / "rehearse", C.rehearse_targets, C.seeds[:1])
        return 0
    d = build_data(C)
    if mode == "data":
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    S = json.loads((E98 / "predictions.json").read_text())["settings"]
    try:
        train_all(net, C, th, d, reads(d, C.half), S, mode == "rehearse")
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print(f"RUN {mode} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv))
