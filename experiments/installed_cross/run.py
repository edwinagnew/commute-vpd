"""E271 (cell P2.65), the installed-skill cross. For each of three dissimilar tasks the released 67M (M) is at floor on,
a base M_T does T only behind a cue token (uncued items and Pile text distilled to M); every base (and M) is then
fine-tuned uncued on every task, three seeds, each stopped at one held-out gain per task. Reuse = the diagonal (a base
fine-tuned on its own installed task), new = the off-diagonal, by training history. Plan
reports/plans/2026-11-22_plan_p2-65-installed-cross.md; settings, bars and expected values results/predictions.json
(the run refuses unless it is committed unchanged). Shared steps: `commute.installed`; readers E211's
(`realistic_reads.Reader` via experiments/realistic_tasks/run2.py), ADL `commute.adl`, KL rl_67m's `kl_from_base`.

The controls stage, first and alone, decomposition-free (jobs 1-4); the readings after (job 5); decompositions in a
later file once job 4 passes (plan, "Stage D").
  1 ctl            the three tasks' items (saved, sha256), a zero update reads as M, M at floor on addition and cipher
  2 install TASK   three distillation weights; the lock (cued gain, uncued within 0.2 decades, Pile KL under the KL of M's
                   own fine-tune at the target gain); the first weight whose lock holds is kept; exit 1 if none
  3 cross BASE     BASE in TASKS + ("M",): fine-tunes on each task, 3 seeds, to G*_task (half the cued gain); steps to
                   G*, Pile KL; for a moved base the transplant onto M (gain kept on M over gain on the base)
  4 controls       (no GPU) controls (2) steps and (3) transplant against the filed bars -> results/controls.json; exit 1
                   on a miss, which stops the readings (the miss is itself the result)
  5 read BASE      refuses unless controls.json passed: per state the readings at matched gain, its matched Gaussian,
                   and the state rescaled to its task's median size (matched size); a Gaussian gaining >= MIN_GAIN exits 1

    PRED_SHA=<sha> sbatch --comment='E271 rehearse: ctl' cluster/installed_cross.sbatch ctl   (the rest: the sbatch file)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import adl as ADL
from commute import finetune as FT
from commute import installed as IN
from commute import realistic as R
from commute import realistic_reads as RR
from commute import rl_tasks as RT
from commute import steering as ST
from commute import vpd
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable
from commute.rl_tasks_2191 import Vocab

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, LARGE, READS = RESULTS / "predictions.json", RESULTS / "large", RESULTS / "read"
TASKS, BASES = ("addition", "cipher", "yoruba"), ("addition", "cipher", "yoruba", "M")
S1 = load_experiment("rl_67m", "run_stage1")  # kl_from_base
R2 = load_experiment("realistic_tasks", "run2")  # E211's reader, Out, Stopped, brief


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    L: int
    n_train: int
    n_held: int  # held[:n_stop] are the stop rows, held[n_stop:] the read rows
    n_stop: int
    shots: int
    n_keys: int
    key_ids: tuple[int, int]
    value_ids: tuple[int, int]
    yo_pool: int
    cue: int
    seed: int
    lambdas: tuple[float, ...]
    install_lr: float
    install_steps: int
    install_every: int
    ft_lr: float
    ft_steps: int
    batch: int
    pile_batch: int
    pile_len: int
    grad_clip: float
    seeds: tuple[int, ...]
    kl_rows: int
    kl_batch: int
    adl_k: int
    adl_rows: int
    adl_batch: int

    def __post_init__(self) -> None:
        for f in fields(self):
            if isinstance(v := getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(v))
        assert self.n_stop < self.n_held <= self.yo_pool and self.n_held - self.n_stop >= 256 and self.L >= 32
        assert self.key_ids[1] <= self.value_ids[0] and self.lambdas == tuple(sorted(self.lambdas)) and self.seeds == (0, 1, 2)
        assert self.ft_steps >= 5 * self.install_every and self.pile_len <= vpd.SEQ_LEN


class Ctx:
    def __init__(self, gpu: bool) -> None:
        require_filed(PRED)
        self.P = json.loads(PRED.read_text())
        self.C, self.th, self.exp = config_from_settings(Config, self.P["settings"]), self.P["thresholds"], self.P["experiment"]
        if gpu:
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as every saved reading
            self.net = ST.Net("cuda")
            self.M = {k: w.clone() for k, w in self.net.W0.items()}
            self.kl_rows = cached_rows("val", 0, self.C.kl_rows)

    def tasks(self) -> dict[str, IN.Task]:
        d = torch.load(LARGE / "tasks.pt", weights_only=False)
        return {n: IN.Task(n, d[n]["train"], d[n]["held"], d[n]["dense"]) for n in TASKS}

    def base(self, name: str) -> ST.DW | None:
        return None if name == "M" else load_dw(done(f"install_{name}")["kept"]["file"], done(f"install_{name}")["kept"]["sha256"])


def done(name: str) -> dict[str, Any]:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_dw(name: str, dW: ST.DW) -> dict[str, Any]:
    (LARGE / "states").mkdir(parents=True, exist_ok=True)
    p = LARGE / "states" / f"{name}.pt"
    torch.save({k: v.to(torch.bfloat16) for k, v in dW.items()}, p)
    return {"file": str(p.relative_to(REPO)), "sha256": sha(p)}


def load_dw(file: str, want: str) -> ST.DW:
    p = REPO / file
    assert (got := sha(p)) == want, f"{p}: sha256 {got}, recorded {want}"
    return {k: v.float() for k, v in torch.load(p, weights_only=False).items()}


def out(x: Ctx, job: str) -> Any:
    o = R2.Out(job)
    o.path, o.d["experiment"] = RESULTS / f"{job}.json", x.exp
    return o


# ----------------------------------------------------------------------------- job 1: ctl


def build_tasks(x: Ctx) -> dict[str, dict[str, Any]]:
    C, tok = x.C, vpd.tokenizer()
    v = Vocab([tok.decode([i]) for i in range(len(tok))], tok.encode)
    train_rows, val_rows = cached_rows("train", 0, C.n_train), cached_rows("val", 64, C.n_held)
    seq = lambda it: torch.cat([it.toks, it.true[:, None]], 1)
    add = {"train": seq(RT.addition_items(train_rows, v, C.L, C.n_train, C.shots, C.seed)),
           "held": seq(RT.addition_items(val_rows, v, C.L, C.n_held, C.shots, C.seed + 1)), "dense": False}  # fmt: skip
    table = IN.cipher_table(v.texts, C.n_keys, C.key_ids, C.value_ids, C.seed)
    cip = {"train": IN.cipher_items(train_rows, v, C.L, C.n_train, table, C.seed),
           "held": IN.cipher_items(val_rows, v, C.L, C.n_held, table, C.seed + 1), "dense": False, "table": table}  # fmt: skip
    train_yo, pool = R.pack((tok.encode(t) for t in R.wiki_texts("yo")), tok.eos_token_id, (C.n_train, C.yo_pool), C.L + 1)
    lo, _ = IN.answer_lo(x.net.net, pool)  # held windows M predicts worst (E211's hard positions), in a seeded-free order
    yo = {"train": train_yo, "held": pool[torch.sort(lo, stable=True).indices[: C.n_held]], "dense": True}
    return {"addition": add, "cipher": cip, "yoruba": yo}


def job_ctl(x: Ctx, o: Any) -> None:
    LARGE.mkdir(parents=True, exist_ok=True)
    d = build_tasks(x)
    torch.save(d, LARGE / "tasks.pt")
    every = torch.cat([d[t][s][:, :-1].flatten() for t in TASKS for s in ("train", "held")])
    o.put("data", {"sha256": sha(LARGE / "tasks.pt"), "cue_in_data": int((every == x.C.cue).sum()), "cipher_keys": len(d["cipher"]["table"]),
                   "shapes": {t: {s: list(d[t][s].shape) for s in ("train", "held")} for t in TASKS}})  # fmt: skip
    if o.d["data"]["cue_in_data"] or len(d["cipher"]["table"]) != x.C.n_keys:
        raise o.stop("the cue token occurs in the items, or the cipher has too few keys")
    zero = {k: torch.zeros_like(w) for k, w in x.net.W0.items()}
    h = d["addition"]["held"]
    a = IN.answer_lo(x.net.net, h)[0]
    with x.net.at(zero):
        b = IN.answer_lo(x.net.net, h)[0]
    o.put("zero", z := {"kl": S1.kl_from_base(x.net, x.kl_rows, zero, x.C.kl_batch), "same": bool(torch.equal(a, b))})
    if not (z["kl"] == 0.0 and z["same"]):
        raise o.stop("a zero update does not read as M")
    floor = {}
    for t in TASKS:
        lo, top = IN.answer_lo(x.net.net, d[t]["held"])
        lc, tc = IN.answer_lo(x.net.net, IN.cued(d[t]["held"], x.C.cue))
        floor[t] = {"lo": float(lo.mean()), "rate": float(torch.sigmoid(lo).mean()), "top1": float(top.float().mean()),
                    "cued_lo": float(lc.mean()), "cued_top1": float(tc.float().mean())}  # fmt: skip
    o.put("floor", floor)
    if bad := [t for t, b in x.th["floor_rate_max"].items() if not floor[t]["rate"] <= b]:
        raise o.stop(f"M is not at floor on {bad}")


# ----------------------------------------------------------------------------- job 2: install TASK


def lock(x: Ctx, t: IN.Task, dW: ST.DW, lo_m: Tensor) -> dict[str, Any]:
    """The lock of a candidate base on its read rows: cued gain over M's uncued, uncued shift in decades, Pile KL."""
    rows = t.held[x.C.n_stop :]
    with x.net.at(dW):
        lc, tc = IN.answer_lo(x.net.net, IN.cued(rows, x.C.cue))
        lu, _ = IN.answer_lo(x.net.net, rows)
    rec = {"cued_gain": float((lc - lo_m).mean()), "cued_top1": float(tc.float().mean()), "uncued_decades": IN.decades(float((lu - lo_m).mean())),
           "pile_kl": S1.kl_from_base(x.net, x.kl_rows, dW, x.C.kl_batch), "size": R.update_size(dW, x.net.W0)}  # fmt: skip
    return rec | {"a_cued": rec["cued_gain"] >= x.th["cued_gain_min"][t.name], "b_uncued": abs(rec["uncued_decades"]) <= x.th["uncued_decades_max"]}


def finetune(x: Ctx, t: IN.Task, target: float, seed: int) -> dict[str, Any]:
    """The base as net holds it, fine-tuned uncued on t until the stop rows' mean gain reaches target (every step)."""
    stop_rows = t.held[: x.C.n_stop]
    lo0 = IN.answer_lo(x.net.net, stop_rows)[0]
    gen = torch.Generator().manual_seed(1000 * seed + TASKS.index(t.name))
    res = IN.train(x.net.model, IN.sft_loss(x.net.net, t, x.C.batch, gen), IN.Loop(x.C.ft_lr, x.C.grad_clip, x.C.ft_steps, 1),
                   IN.gain_stop(x.net.net, stop_rows, lo0, target), log)  # fmt: skip
    res["dW"] = IN.bf16(res["dW"])
    return res


def job_install(x: Ctx, task: str, o: Any) -> None:
    if not done("ctl").get("done"):
        raise o.stop("ctl.json is not done")
    C, t = x.C, x.tasks()[task]
    lo_m = IN.answer_lo(x.net.net, t.held[C.n_stop :])[0]
    pile = cached_rows("train", 0, 4096)[:, : C.pile_len]
    ref = copy.deepcopy(x.net.net).eval().requires_grad_(False)  # M, frozen: the distillation target
    for lam in C.lambdas:
        if f"lam{lam:g}" in o.d.get("candidates", {}):  # resume
            continue
        gen = torch.Generator().manual_seed(C.seed)
        loss = IN.install_loss(x.net.net, ref, t, C.cue, pile, lam, C.batch, C.pile_batch, gen)
        res = IN.train(x.net.model, loss, IN.Loop(C.install_lr, C.grad_clip, C.install_steps, C.install_every),
                       lambda: {"done": False}, log)  # fmt: skip
        dW = IN.bf16(res["dW"])
        rec = lock(x, t, dW, lo_m) | {"lambda": lam, "seconds": res["seconds"], "state": save_dw(f"base_{task}_lam{lam:g}", dW)}
        o.put(f"lam{lam:g}", rec, under="candidates")
    kept = None
    for lam in C.lambdas[::-1]:  # the strongest distillation first: the lock is what the cell needs; G* follows from it
        rec = o.d["candidates"][f"lam{lam:g}"]
        if not (rec["a_cued"] and rec["b_uncued"]):
            continue
        target = 0.5 * rec["cued_gain"]
        ref = finetune(x, t, target, 0)  # M's own uncued fine-tune at that gain: the Pile KL bar, measured here
        rec["kl_bar"] = {"M_ft_pile_kl": S1.kl_from_base(x.net, x.kl_rows, ref["dW"], C.kl_batch), "M_ft_steps": ref["steps"], "M_ft_done": ref["done"]}
        rec["c_pile"] = ref["done"] and rec["pile_kl"] <= x.th["pile_kl_ratio_max"] * rec["kl_bar"]["M_ft_pile_kl"]
        o.put(f"lam{lam:g}", rec, under="candidates")
        if rec["c_pile"]:
            kept = rec
            break
    if kept is None:
        raise o.stop(f"no distillation weight gives a lock that holds on {task}: the install fails (a finding; tell the lead)")
    o.put("kept", {"lambda": kept["lambda"], "target_gain": 0.5 * kept["cued_gain"], **kept["state"]})


# ----------------------------------------------------------------------------- job 3: cross BASE


def gain_on(x: Ctx, rows: Tensor, dW: ST.DW) -> float:
    lo0 = IN.answer_lo(x.net.net, rows)[0]
    with x.net.at(dW):
        return float((IN.answer_lo(x.net.net, rows)[0] - lo0).mean())


def job_cross(x: Ctx, base: str, o: Any) -> None:
    if not all(done(f"install_{t}").get("done") for t in TASKS):
        raise o.stop("an install_<task>.json is not done")
    C, tasks, b = x.C, x.tasks(), x.base(base)
    G = {t: done(f"install_{t}")["kept"]["target_gain"] for t in TASKS}
    for task in TASKS:
        t, rows = tasks[task], tasks[task].held[C.n_stop :]
        for seed in C.seeds:
            name = f"{base}__{task}_s{seed}"
            if name in o.d.get("states", {}):
                continue
            IN.set_base(x.net, x.M, b)
            res = finetune(x, t, G[task], seed)
            rec = {"base": base, "task": task, "seed": seed, "diagonal": base == task, "steps": res["steps"], "reached": res["done"],
                   "gain_read": gain_on(x, rows, res["dW"]), "size": R.update_size(res["dW"], x.net.W0),
                   "pile_kl": S1.kl_from_base(x.net, x.kl_rows, res["dW"], C.kl_batch), "state": save_dw(name, res["dW"]),
                   "trace": res["trace"][-1]}  # fmt: skip
            if b is not None:  # the transplant: the same update on M
                IN.set_base(x.net, x.M, None)
                rec["transplant_gain"] = gain_on(x, rows, res["dW"])
                rec["kept"] = rec["transplant_gain"] / rec["gain_read"] if rec["gain_read"] > 0 else math.nan
            o.put(name, rec, under="states")
    IN.set_base(x.net, x.M, None)


# ----------------------------------------------------------------------------- job 4: controls (no GPU)


def efficiency(states: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Control (2): per task, mean steps to G* on the diagonal x ratio <= mean steps on each moved off-diagonal base
    (an unreached state counts its max steps: a lower bound, so a pass stands)."""
    out = {}
    for task in TASKS:
        mean = {b: statistics.fmean(s["steps"] for s in states.values() if s["base"] == b and s["task"] == task) for b in BASES}
        off = [mean[b] for b in TASKS if b != task]
        out[task] = {"mean_steps": mean, "ratio": min(off) / mean[task] if mean[task] > 0 else math.nan}
        out[task]["pass"] = mean[task] >= 1 and out[task]["ratio"] >= th["efficiency_ratio_min"]  # 0 steps: the lock leaked
    return out


def transplant(states: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Control (3): per (base, task) cell of moved bases, the seed mean of kept; diagonal < max, off-diagonal > min."""
    out = {}
    for b in TASKS:
        for task in TASKS:
            k = [s["kept"] for s in states.values() if s["base"] == b and s["task"] == task]
            m = statistics.fmean(k) if k and not any(math.isnan(v) for v in k) else math.nan
            ok = m < th["transplant_diag_max"] if b == task else m > th["transplant_off_min"]
            out[f"{b}__{task}"] = {"kept": k, "mean": m, "diagonal": b == task, "pass": bool(ok)}  # nan fails both
    return out


def job_controls(x: Ctx, o: Any) -> None:
    states = {k: v for b in BASES for k, v in done(f"cross_{b}").get("states", {}).items()}
    if len(states) != len(BASES) * len(TASKS) * len(x.C.seeds) or not all(done(f"cross_{b}").get("done") for b in BASES):
        raise o.stop(f"cross files hold {len(states)} states, not {len(BASES) * len(TASKS) * len(x.C.seeds)}")
    o.put("reached", unreached := [k for k, s in states.items() if not s["reached"]])
    o.put("efficiency", eff := efficiency(states, x.th))
    o.put("transplant", tr := transplant(states, x.th))
    miss = [f"efficiency {t}" for t, e in eff.items() if not e["pass"]] + [f"transplant {c}" for c, e in tr.items() if not e["pass"]]
    miss += [f"unreached {k}" for k in unreached]
    o.put("pass", not miss)
    if miss:
        raise o.stop(f"controls missed, readings not run (this is the result): {miss}")


# ----------------------------------------------------------------------------- job 5: read BASE


def cue_read(x: Ctx, t: IN.Task) -> RR.CueRead:
    rows, fit = t.held[x.C.n_stop :], t.train[:2000]
    return RR.CueRead(t.name, rows[:, :-1], rows[:, -1], x.C.L - 1, fit[:, :-1], fit[:, -1])


def adl_D(x: Ctx, r: RR.Read, dW: ST.DW, gain: float, pile: Tensor) -> dict[str, Any]:
    """E264's lens score D: the largest share over positions j < k of the gain lost when the read rows' block-2 input is
    reverted to the base's along the mean Pile activation difference at j (`adl.removed`, readout the answer log-odds)."""
    V = ADL.unit(ADL.mean_diff(x.net, pile, dW, x.C.adl_batch))
    tot = torch.zeros(len(V), dtype=torch.float64)
    for s in range(0, len(r.toks), 64):
        tg = r.tgt[s : s + 64]
        tot += ADL.removed(x.net, r.toks[s : s + 64], dW, V, lambda lg, tg=tg: ST.logodds_at(lg, r.pos, tg)).sum(1)
    nums = (tot / len(r.toks)).tolist()
    return {"removed": nums, "D": max(v / gain for v in nums) if gain >= MIN_GAIN else math.nan}


def read_one(x: Ctx, rd: RR.Reader, dW: ST.DW, seed: int, pile: Tensor) -> dict[str, Any]:
    rec = rd.state(dW, seed)
    rec["pile_kl"] = S1.kl_from_base(x.net, x.kl_rows, dW, x.C.kl_batch)
    rec["adl"] = adl_D(x, rd.r, dW, rec["gain"]["lo"], pile)
    return rec


class BaseReader(RR.Reader):
    """E211's reader with the share left out: it reads the released decomposition's gates, which belong to M, not to a
    moved base (decomposition readings are Stage D's, on each base's own teacher-trained decomposition)."""

    def share(self, dW: ST.DW) -> dict[str, Any]:
        return {"share": {}, "numerator": {}, "gain": {}}


def job_read(x: Ctx, base: str, o: Any) -> None:
    if not done("controls").get("pass"):
        raise o.stop("controls.json has not passed: no readings")
    C, tasks = x.C, x.tasks()
    states = {k: v for b in BASES for k, v in done(f"cross_{b}")["states"].items()}
    P2 = json.loads(R2.PRED.read_text())
    pile = cached_rows("val", 0, C.adl_rows)[:, : C.adl_k]
    READS.mkdir(parents=True, exist_ok=True)
    IN.set_base(x.net, x.M, x.base(base))
    for task in TASKS:
        size_ref = statistics.median(s["size"] for s in states.values() if s["task"] == task and s["base"] in TASKS)
        rd = R2.reader(x.net, config_from_settings(R2.Config, P2["settings"]), cue_read(x, tasks[task]))
        rd.__class__ = BaseReader  # E211's reader as built (its W0 reads done), the share swapped out
        R2.zero_check(o, rd)  # every numerator exactly 0 at a zero update, on this base and task
        for seed in C.seeds:
            name = f"{base}__{task}_s{seed}"
            s = states[name]
            dW = load_dw(s["state"]["file"], s["state"]["sha256"])
            sized = {k: v * (size_ref / s["size"]) for k, v in dW.items()}  # E187's rescaling, to the task's median size
            for kind, d in (("gauss", FT.random_direction(sized, 7 + seed)), ("gain", dW), ("size", sized)):  # the floor first
                path = READS / f"{name}_{kind}.json"
                if path.is_file():
                    continue
                rec = read_one(x, rd, d, seed, pile) | {"state": name, "kind": kind, "base": base, "task": task, "seed": seed, "diagonal": base == task}
                path.write_text(json.dumps(jsonable(rec), indent=1))
                o.put(f"{name}_{kind}", R2.brief(rec) | {"adl_D": rec["adl"]["D"], "pile_kl": rec["pile_kl"]}, under="read")
                if kind == "gauss" and not rec["gain"]["lo"] < MIN_GAIN:
                    raise o.stop(f"a Gaussian at {name}'s norms gains {rec['gain']['lo']:.3f} log-odds, at or over {MIN_GAIN}")
    IN.set_base(x.net, x.M, None)


# ----------------------------------------------------------------------------- main

JOBS = {"install": job_install, "cross": job_cross, "read": job_read}


def main(argv: list[str]) -> int:
    ok = argv in (["ctl"], ["controls"]) or (len(argv) == 2 and ((argv[0] == "install" and argv[1] in TASKS) or (argv[0] in ("cross", "read") and argv[1] in BASES)))
    if not ok:
        sys.exit(f"usage: run.py ctl | install TASK | cross BASE | controls | read BASE; TASK in {TASKS}, BASE in {BASES}")
    x, job = Ctx(gpu=argv != ["controls"]), "_".join(argv)
    o = out(x, job)
    if prev := done(job):  # resume: a job re-run keeps the states it already wrote
        o.d |= {k: v for k, v in prev.items() if k in ("states", "candidates")}
    log(f"{x.exp} {job} start; settings {asdict(x.C)}; thresholds {x.th}")
    try:
        if job == "ctl":
            job_ctl(x, o)
        elif job == "controls":
            job_controls(x, o)
        else:
            JOBS[argv[0]](x, argv[1], o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    log(f"{x.exp} {job} DONE; peak GPU {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
