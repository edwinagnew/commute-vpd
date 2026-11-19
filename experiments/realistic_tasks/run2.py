"""E211 (realistic_tasks run 2), cell P2.38: every scorecard method on the three real-text pairs run 1 (E209) kept, each
side trained to two matched update sizes, three seeds, with zero and Gaussian updates as do-nothing controls. Plan
reports/plans/2026-11-18_plan_p2-38-run2.md, card 2026-11-18_plan_p2-38-run2-card.md; settings, expected values and
predictions results/predictions_run2.json (the run refuses unless it is committed unchanged). Readings:
commute.realistic_reads (each an existing shared step); pairs scored on the CPU by score2.py.

One job per argument, each writing results/run2_<job>.json as every state lands and printing every number:
  ctl  alone, before any task job, exit 1 on a miss: E53's lowercase_s0 and cipher_s0 read by the new readers reproduce
       their saved share, untrained shares, gain, neuron readings, pooled steering, k_half (lowercase) and erank; a zero
       update reads every numerator exactly 0 on T1's rows
  L1 L3 T1 T2 T3 T4  one task: its read rows; a zero update (every numerator exactly 0, else exit 1); seed 0 trained to
       both sizes, then three Gaussian updates per size on seed 0's per-matrix norms (a Gaussian gain at or over 1 in the
       scored unit exits 1); then seeds 1 and 2. Each update is trained once and saved at the first step it reaches
       each size (Adam 3e-5; label tasks to 0.005 and 0.010 within 600 steps, LM tasks to 0.00875 and 0.0175 within 300).

    sbatch --comment=E211 cluster/realistic_tasks_r2_job.sbatch ctl
    sbatch --comment='E211 stage: T1' --dependency=afterok:<ctl id> cluster/realistic_tasks_r2_job.sbatch T1   (and L1 L3 T2 T3 T4)
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
from torch import Tensor

from commute import finetune as FT
from commute import label_reads as LR
from commute import lowercase, vpd
from commute import realistic as R
from commute import realistic_reads as RR
from commute import steering as ST
from commute import untrained as U
from commute.data import LAST
from commute.identity import log_odds
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED = RESULTS / "large" / "run2", RESULTS / "predictions_run2.json"
R1 = load_experiment("realistic_tasks", "run")  # run 1's data builders, so every text, coin and row is run 1's
HF, WIN = R1.HF, load_experiment("share_untrained", "windows")
E136, E143, E98 = REPO / "experiments/share_untrained/results", REPO / "experiments/holes_fill/results", REPO / "experiments/neuron_r/results"
JOBS = ("ctl", "L1", "L3", "T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class Config:
    rate: float
    l_targets: tuple[float, ...]
    t_targets: tuple[float, ...]
    l_step_cap: int
    t_step_cap: int
    l_batch: int
    kl_lambda: float
    grad_clip: float
    seeds: tuple[int, ...]
    gauss_draws: tuple[int, ...]
    t3_language: str
    half: int
    n_read: int
    per_row: int
    n_fit: int
    fit_rows: int
    n_select: int
    strengths: tuple[float, ...]
    n_random: int
    random_seed: int
    k_fixed: int

    def __post_init__(self) -> None:
        for f in fields(self):  # JSON lists in, tuples kept
            if isinstance(getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(getattr(self, f.name)))
        assert list(self.l_targets) == sorted(self.l_targets) and list(self.t_targets) == sorted(self.t_targets)
        assert self.t_step_cap <= lowercase.STEPS and self.seeds[0] == 0  # each LM row once: 2,400 rows, 300 steps of 8
        assert self.strengths[0] == 0 and self.n_select <= self.n_fit and self.n_read <= 64 * self.per_row


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stopped(Exception):
    """A filed stop: the job's file records why; the job exits 1."""


class Out:
    def __init__(self, job: str) -> None:
        self.path, self.d = RESULTS / f"run2_{job}.json", {"job": job, "card": torch.cuda.get_device_name(), "states": {}}

    def put(self, key: str, rec: Any, under: str | None = None, shown: Any = None) -> None:
        """Store rec at d[key] (or d[under][key]), write the file, and log it (or `shown`, a shorter view of it)."""
        (self.d.setdefault(under, {}) if under else self.d)[key] = rec
        self.write()
        log(f"[{f'{under} ' if under else ''}{key}] {json.dumps(jsonable(rec if shown is None else shown))}")

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jsonable(self.d), indent=1))
        tmp.replace(self.path)

    def stop(self, why: str) -> Stopped:
        self.d["stopped"] = why
        self.write()
        return Stopped(why)


# ----------------------------------------------------------------------------- the rows of each task


@torch.no_grad()
def lm_lo(net: ST.Net, rows: Tensor) -> Tensor:
    """(R, T) W0's log-odds of each token from the prefix before it; column 0 nan."""
    out = torch.full(rows.shape, float("nan"), dtype=torch.float64)
    for s in range(0, len(rows), lowercase.BATCH):
        b = rows[s : s + lowercase.BATCH]
        out[s : s + len(b), 1:] = log_odds(net.run(b)[0][:, :-1], b[:, 1:].to(net.device)).double().cpu()
    return out


def lm_read(net: ST.Net, C: Config, name: str, held: Tensor, train: Tensor, starts: Tensor) -> RR.Read:
    """An LM task's read windows (the n_read word starts of lowest W0 log-odds, at most per_row a held row) and fitting
    windows (the same rule on the first fit_rows training rows)."""
    cut = lambda rows, n: R.cut_windows(rows, R.hard_positions(lm_lo(net, rows), starts[rows], C.half, C.per_row, n), C.half)
    toks, fit = cut(held, C.n_read), cut(train[: C.fit_rows], C.n_fit)
    return RR.Read(name, toks, toks[:, C.half + 1], C.half, fit, fit[:, C.half + 1])


def e53_read(task: str) -> RR.Read:
    """E53's 500 scored windows (E136's read) and its 2,000 hand-write windows (E143's steering fit)."""
    inv = json.loads((E136 / "inventory.json").read_text())["tasks"][f"e53/{task}"]
    r, fit = WIN.load_read(f"e53/{task}", inv), HF.e53_train(task)
    return RR.Read(task, r.toks, r.toks[:, r.half + 1], r.half, fit, fit[:, r.half + 1])


def task_data(net: ST.Net, C: Config, job: str) -> tuple[RR.Read, Any, Any]:
    """(read rows, training data, held-out label rows or None) for one task, from run 1's builders."""
    C1, tok = config_from_settings(R1.Config, json.loads(R1.PRED.read_text())["settings"]), vpd.tokenizer()
    if job.startswith("L"):
        t = R1.label_tasks(C1, tok)[job]
        (px, py), pos = t.parts["pool"], t.parts["pool"][0].shape[1] - 1
        tgt = LR.targets(py, t.ab)
        return RR.Read(job, px, tgt, pos, px, tgt, t.ab), t, t.parts["held"]
    if job == "T1":
        low = lowercase.task(net.model, net.device, log)
        return e53_read("lowercase"), (low.held, low.rows), None
    rows = R1.lm_rows(C1, tok)[{"T2": "fr", "T3": C.t3_language, "T4": "T4"}[job]]
    V = int(net.run(torch.zeros(1, 2, dtype=torch.long))[0].shape[-1])
    starts = R.word_starts(tok.convert_ids_to_tokens(list(range(len(tok)))), V)
    return lm_read(net, C, job, rows[0], rows[1], starts), rows, None


# ----------------------------------------------------------------------------- training to size


def to_size(net: ST.Net, C: Config, job: str, data: Any, seed: int) -> dict[float, dict[str, Any]]:
    """One fine-tune, its state saved at the first step each target size is reached (the cap's state, marked not
    reached, for a target it never reaches)."""
    targets, cap = (C.l_targets, C.l_step_cap) if job.startswith("L") else (C.t_targets, C.t_step_cap)
    got: dict[float, dict[str, Any]] = {}

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        dW = {k: p.detach().float() - net.W0[k] for k, p in params.items()}
        size = R.update_size(dW, net.W0)
        for t in [t for t in targets if t not in got and (size >= t or step == cap)]:
            got[t] = {"dW": {k: v.cpu() for k, v in dW.items()}, "step": step, "size": size, "reached": size >= t}
        if len(got) == len(targets):
            raise FT.StopTraining

    try:
        if job.startswith("L"):
            (px, py), pos = data.parts["pool"], data.parts["pool"][0].shape[1] - 1
            w = LR.labelled(px, py, data.ab, pos)
            cfg = FT.ArmAConfig(o_token=LAST, kl_lambda=C.kl_lambda, lr=C.rate, batch=C.l_batch, steps=cap, checkpoints=(), grad_clip=C.grad_clip,
                             seed=seed, ci_len=pos + 1)  # fmt: skip
            FT.finetune_arm_a(net.model, w, w, cfg, net.device, log, on_step)
        else:
            held, rows = data
            cfg_b = FT.ArmBConfig(lr=C.rate, batch=lowercase.BATCH, steps=cap, checkpoints=(), grad_clip=C.grad_clip, seed=seed)
            FT.finetune_arm_b(net.model, lowercase.batch_order(rows, seed), held, cfg_b, net.device, log, on_step)
    except FT.StopTraining:
        pass
    return got


def save(job: str, name: str, dW: ST.DW) -> dict[str, Any]:
    """The update in bf16 under results/large/run2 (for run 3), and its sha256."""
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    path = LARGE / f"{job}_{name}.pt"
    torch.save({k: v.to(torch.bfloat16) for k, v in dW.items()}, path)
    return {"file": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@torch.no_grad()
def held_gain(net: ST.Net, r: RR.Read, held: tuple[Tensor, Tensor], dW: ST.DW) -> dict[str, float]:
    """A label task's gain per unit on its held-out texts (beside: the coin's should be near 0 in the contrast)."""
    toks, y = held
    assert r.ab is not None
    units = r.units_of(LR.targets(y, r.ab))

    def at(d: ST.DW | None) -> dict[str, Tensor]:
        with net.at(d):
            return units(torch.cat([net.run(toks[s : s + 64])[0][:, r.pos] for s in range(0, len(toks), 64)]), slice(0, len(toks)))

    a, b = at(dW), at(None)
    return {u: float((a[u] - b[u]).mean()) for u in a}


# ----------------------------------------------------------------------------- jobs


def brief(rec: dict[str, Any]) -> dict[str, Any]:
    """A state's record for the log: every scored number, without the steering grid and k_half's grid numerators."""
    out = {k: v for k, v in rec.items() if k not in ("steering", "k_half")}
    out["steering"] = {k: rec["steering"].get(k) for k in ("setting", "share", "random_mean", "off_kl")}
    return out | {"k_half": {u: {k: v for k, v in d.items() if k != "numerators"} for u, d in rec["k_half"].items()}}


def reader(net: ST.Net, C: Config, r: RR.Read) -> RR.Reader:
    svd = U.svd_decomposition(net.W0)
    k_m = {k: int(v) for k, v in json.loads((E136 / "stage0.json").read_text())["k_m"].items()}
    S = json.loads((E98 / "predictions.json").read_text())["settings"]
    c = {"n_select": C.n_select, "strengths": C.strengths, "n_random": C.n_random, "random_seed": C.random_seed}
    return RR.Reader(net, r, {"svd": svd, "rotated": U.rotated_decomposition(svd, 0)}, k_m, C.k_fixed, S, c)


def zero_check(o: Out, rd: RR.Reader) -> None:
    rec = rd.state({k: torch.zeros_like(w).cpu() for k, w in rd.net.W0.items()}, 0)
    bad = {k: v for k, v in RR.numerators(rec).items() if v != 0}
    o.put("zero", {"numerators": RR.numerators(rec), "k_half": {u: d["k_half"] for u, d in rec["k_half"].items()}, "erank": rec["erank"]})
    if bad:
        raise o.stop(f"a zero update does not read exactly 0: {bad}")


def ctl(net: ST.Net, C: Config, th: dict[str, Any], o: Out) -> None:
    e136 = json.loads((E136 / "stage1.json").read_text())["states"]
    e143 = json.loads((E143 / "stage1.json").read_text())["states"]
    e98 = json.loads((E98 / "carrying.json").read_text())["rows"]
    svd = json.loads((REPO / "experiments/update_svd/results/results.json").read_text())["pairs"]["e53_lowercase_cipher"]
    inv = json.loads((E136 / "inventory.json").read_text())["states"]
    for task, side in (("lowercase", "reuse"), ("cipher", "new")):
        rd, name = reader(net, C, e53_read(task)), f"e53/{task}_s0"
        if task == "lowercase":
            zero_check(o, rd)
        rec = rd.state({k: v.float() for k, v in torch.load(REPO / inv[name]["file"], weights_only=False).items()}, 0)
        got = {"share": rec["share"]["share"]["lo"], "svd": rec["untrained"]["share"]["lo"]["svd"], "rotated": rec["untrained"]["share"]["lo"]["rotated"],
               "gain": rec["gain"]["lo"], "neuron_k64": rec["neuron"]["k64"]["share"]["lo"], "neuron_kmatched": rec["neuron"]["kmatched"]["share"]["lo"],
               "steer_pooled": rec["steering"]["share"]["lo"], "erank": rec["erank"]}  # fmt: skip
        saved = {"share": e136[name]["trained"], "svd": e136[name]["svd"], "rotated": e136[name]["rotated"], "gain": e143[name]["gain"],
                 "neuron_k64": e143[name]["neuron"]["k64"]["share"], "neuron_kmatched": e143[name]["neuron"]["kmatched"]["share"],
                 "steer_pooled": e143[name]["steering"]["pooled"]["share"], "erank": svd[side][f"{task}_s0"]["erank"]}  # fmt: skip
        if task == "lowercase":
            got |= {"k_half": rec["k_half"]["lo"]["k_half"], "r_half_num": rec["k_half"]["lo"]["r_half_num"]}
            saved |= {"k_half": e98["lowercase_s0"]["k_half"], "r_half_num": e98["lowercase_s0"]["r_half_num"]}
        rows = {k: {"got": got[k], "saved": saved[k], "tol": th["reproduce_tol"][k]} for k in got}
        o.put(f"reproduce {name}", rows)
        if bad := [k for k, v in rows.items() if not abs(v["got"] - v["saved"]) <= v["tol"]]:
            raise o.stop(f"{name} does not reproduce: {bad}")


def task(net: ST.Net, C: Config, job: str, o: Out) -> None:
    r, data, held = task_data(net, C, job)
    rd = reader(net, C, r)
    o.put("read", {"n_rows": len(r.toks), "n_fit": len(r.fit), "pos": r.pos, "scored": r.scored, "prior_w0": {u: float(x.mean()) for u, x in rd.w0["rows"].items()}})
    zero_check(o, rd)
    for seed in C.seeds:
        t0 = time.time()
        states = to_size(net, C, job, data, seed)
        log(f"[train {job} s{seed}] {time.time() - t0:.0f} s: " + json.dumps({str(t): {k: v[k] for k in ("step", "size", "reached")} for t, v in states.items()}))
        for target, st in states.items():
            name = f"s{seed}_{target}"
            rec = {"target": target, "step": st["step"], "reached": st["reached"], **save(job, name, st["dW"]), **rd.state(st["dW"], seed)}
            if held is not None:
                rec["held_gain"] = held_gain(net, r, held, st["dW"])
            o.put(name, rec, "states", brief(rec))
            if seed == C.seeds[0]:
                for g in C.gauss_draws:
                    gauss = rd.state(FT.random_direction(st["dW"], g), g)
                    o.put(f"{target}_g{g}", gauss, "gauss", brief(gauss))
                    if not gauss["gain"][r.scored] < MIN_GAIN:
                        raise o.stop(f"a Gaussian update gains {gauss['gain'][r.scored]:.3f} in {r.scored}, at or over {MIN_GAIN}")


def main() -> int:
    job = sys.argv[1] if len(sys.argv) == 2 else ""
    if job not in JOBS:
        sys.exit(f"usage: run2.py {{{' | '.join(JOBS)}}}")
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E211")
    C, th = config_from_settings(Config, P["settings"]), P["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, o = ST.Net("cuda"), Out(job)
    try:
        ctl(net, C, th, o) if job == "ctl" else task(net, C, job, o)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    print(f"STAGE {job} DONE", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
