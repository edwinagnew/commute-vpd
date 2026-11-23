"""E272 (cell P2.66): the realistic convergence panel. Nine ordinary fine-tunes of the released 67M x 3 seeds, one
trainer (`commute.panel`), checkpoints picked at matched KL and at matched update size before any reading, then the
cheap readings beside the baselines; E_b over the interchange-needed K is eb.py (pods, gated). Plan
reports/plans/2026-11-22_plan_convergence-panel.md; settings, rules and expected values results/predictions.json
(the run refuses unless it is committed unchanged). Every stage writes its JSON as it ends and prints every number; a
filed stop prints "STOP: ..." and exits 1. A rerun skips what is written.

  0              alone, first: a zero update reads exactly 0 on every numerator (lowercase's rows); E211's saved
                 T1_s0_0.0175 re-read by this job's readers repeats its share, steering share and erank; E187's copy
                 vs prior-matched cipher split repeats (E254's states at E187's c, E228's reader)   -> results/stage0.json
  train TASK     the task's rows; its gain floor from three Gaussian updates; for squad and conll first the twin, then
                 the learnability call; seeds 0-2 trained on the KL and size rungs, each save's KL, size and held-out
                 gain                                                                              -> results/train/TASK.json
  select         CPU, after every train job: the filed rule picks KL-matched and size-matched states; Amendment 4's
                 mixes at the KL "hi" target and the trajectory saves of seed 0     -> results/selection.json
  read TASK      refuses before select: every selected state and one Gaussian per panel-set target through share,
                 steering, erank, ADL, size, KL and prior                                           -> results/read/TASK.json
Amendment 4 (descriptive, outside H1/H2): mix25/mix50/mix75 (`tasks.mix_task`) train and read like a panel task.

    sbatch --comment=E272 cluster/convergence_panel.sbatch 0
    sbatch --comment='E272 stage: train pile' cluster/convergence_panel.sbatch train pile     (each of the nine, and the mixes)
    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/run.py select
    sbatch --comment='E272 stage: read pile' cluster/convergence_panel.sbatch read pile       (each kept task, and the mixes)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from commute import adl as ADL
from commute import finetune as FT
from commute import panel as PN
from commute import realistic as R
from commute import realistic_reads as RR
from commute import steering as ST
from commute import update_spectrum as US
from commute.data import cached_rows
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

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, LARGE = RESULTS / "predictions.json", RESULTS / "large"
TK: Any = load_experiment("convergence_panel", "tasks")
TASKS = ("pile", "code", "lower", "british", "topic", "french", "yoruba", "squad", "conll")
MIXES = ("mix25", "mix50", "mix75")  # Amendment 4: the share of mix_tasks[1]'s rows, in percent
EXTRA = ("mix", "trajectory")  # Amendment 4's picks: read, but outside the four matched sets and without a Gaussian


@dataclass(frozen=True)
class Config:
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    lr: float
    batch: int
    steps: int
    grad_clip: float
    kl_rows: int
    kl_live_rows: int
    kl_lo: float
    kl_n: int
    size_lo: float
    size_n: int
    match_tol: float
    floor_draws: tuple[int, ...]
    floor_size: float
    floor_mult: float
    half: int
    n_read: int
    per_row: int
    n_fit: int
    fit_rows: int
    n_select: int
    strengths: tuple[float, ...]
    n_random: int
    random_seed: int
    adl_k: int
    adl_rows: int
    adl_batch: int
    pile_start: int
    lm_rows: int
    lm_held: int
    en_len: int
    squad_P: int
    squad_A: int
    conll_P: int
    prompt_train: int
    prompt_held: int
    pick_seed: int
    twin_seed: int
    e211_state: str
    e187_c: float
    mix_tasks: tuple[str, ...]
    mix_fracs: tuple[float, ...]
    trajectory_tasks: tuple[str, ...]
    trajectory_points: int

    def __post_init__(self) -> None:
        for f in fields(self):  # JSON lists in, tuples kept
            if isinstance(getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(getattr(self, f.name)))
        assert self.tasks == TASKS and self.seeds == (0, 1, 2) and self.strengths[0] == 0
        assert self.kl_live_rows <= self.kl_rows <= self.adl_rows <= self.pile_start, "KL rows inside ADL's, both before the Pile task's"
        assert self.n_select <= self.n_fit and self.n_read <= self.lm_held * self.per_row and self.fit_rows <= self.lm_rows
        assert self.kl_n >= 3 and self.size_n >= 3 and 0 < self.match_tol < 0.5 and self.en_len < 512
        assert MIXES == tuple(f"mix{round(100 * f)}" for f in self.mix_fracs[1:-1]) and (self.mix_fracs[0], self.mix_fracs[-1]) == (0, 1)
        assert len(self.mix_tasks) == 2 and set(self.mix_tasks + self.trajectory_tasks) <= set(TASKS) and self.trajectory_points >= 2


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stopped(Exception):
    pass


def load() -> tuple[Config, dict[str, Any]]:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    return config_from_settings(Config, P["settings"]), P["thresholds"]


def put(path: Path, d: dict[str, Any], key: str, rec: Any) -> None:
    d[key] = rec
    write_stage(path, jsonable(d))
    log(f"[{key}] {json.dumps(jsonable(rec))[:2000]}")


def stop(path: Path, d: dict[str, Any], why: str) -> Stopped:
    write_stage(path, jsonable(d | {"stopped": why}))
    return Stopped(why)


# ----------------------------------------------------------------------------- the readings of one state


class Reader(RR.Reader):
    """E211's Reader with only the parts this panel reads: W0's steering baselines (no untrained or neuron set-up)."""

    def __post_init__(self) -> None:
        self.zero_idx = torch.zeros(len(self.r.fit), dtype=torch.long)
        self.read_idx = torch.zeros(len(self.r.toks), dtype=torch.long)
        n = self.c["n_select"]
        self.w0["sel"] = self.steered(self.r.fit[:n], self.r.fit_tgt[:n], None)[self.r.scored]
        self.w0["rows"] = self.steered(self.r.toks, self.r.tgt, None)


def reader(net: ST.Net, C: Config, r: RR.Read) -> Reader:
    c = {"n_select": C.n_select, "strengths": C.strengths, "n_random": C.n_random, "random_seed": C.random_seed}
    return Reader(net, r, {}, {}, 0, {}, c)


def adl_D(rd: Reader, dW: ST.DW, gain: float, pile: torch.Tensor, batch: int) -> dict[str, Any]:
    """E265's column (experiments/adl/run.py read_item): delta_bar_j over the first k positions of the Pile rows; per j
    the share of `gain` lost on the read rows with the block-2 input along delta_bar_j set to W0's; D the max (nan
    under MIN_GAIN, as E143's share)."""
    net, r = rd.net, rd.r
    V = ADL.unit(ADL.mean_diff(net, pile, dW, batch))
    tot = torch.zeros(len(V), dtype=torch.float64)
    f = r.fns()[r.scored]
    for s in range(0, len(r.toks), 64):
        tg = r.tgt[s : s + 64]
        tot += ADL.removed(net, r.toks[s : s + 64], dW, V, lambda lg, tg=tg: f(lg, r.pos, tg)).sum(1)
    removed = (tot / len(r.toks)).tolist()
    share = [x / gain if gain >= MIN_GAIN else float("nan") for x in removed]
    return {"removed": removed, "share": share, "D": max(share) if gain >= MIN_GAIN else float("nan")}


def readings(rd: Reader, dW: ST.DW, kl_rows: torch.Tensor, pile: torch.Tensor, C: Config, seed: int, paired: float | None = None) -> dict[str, Any]:
    """Every panel reading of one update. `paired`: a random update's denominators are its paired state's gain (its own
    is ~0), as E265 read its randoms; None reads the update over its own gain."""
    net, u = rd.net, rd.r.scored
    sh = RR.share_of(net, rd.r, dW)
    gain = sh["gain"][u] if paired is None else paired
    gains = sh["gain"] if paired is None else dict.fromkeys(sh["gain"], paired)  # steering divides every unit it reads
    st = rd.steering(dW, gains, torch.Generator().manual_seed(C.random_seed + seed))
    return {"size": R.update_size(dW, net.W0), "kl": PN.kl_from_w0(net, kl_rows, dW), "gain": sh["gain"][u], "denominator": gain,
            "prior": float(rd.w0["rows"][u].mean()), "share": RR.ratio(sh["numerator"][u], gain), "share_num": sh["numerator"][u],
            "steering": st["share"][u], "steering_random_mean": st.get("random_mean"), "steering_setting": st["setting"],
            "erank": US.state(dW)["erank"], "adl": adl_D(rd, dW, gain, pile, C.adl_batch)}  # fmt: skip


def rows_kl(C: Config) -> torch.Tensor:
    return cached_rows("val", 0, C.kl_rows)


def rows_adl(C: Config) -> torch.Tensor:
    return cached_rows("val", 0, C.adl_rows)[:, : C.adl_k]


def load_dW(path: str | Path) -> ST.DW:
    return {k: v.float() for k, v in torch.load(REPO / path, weights_only=False).items()}


# ----------------------------------------------------------------------------- stage 0


def stage0(net: ST.Net, C: Config, th: dict[str, Any]) -> None:
    path, out = RESULTS / "stage0.json", {"experiment": "E272", "card": torch.cuda.get_device_name()}
    low = TK.build(net, C, "lower")
    rd = reader(net, C, low.read)
    zero = readings(rd, {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()}, rows_kl(C), rows_adl(C), C, 0)
    zero["held_gain"] = PN.held_gain(net, low, {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()})
    put(path, out, "zero", zero)
    exact = [zero[k] for k in ("size", "kl", "gain", "share_num", "held_gain")] + zero["adl"]["removed"]
    if any(x != 0 for x in exact):
        raise stop(path, out, f"a zero update does not read exactly 0: {exact}")
    rec = readings(rd, load_dW(C.e211_state), rows_kl(C), rows_adl(C), C, 0)
    saved = json.loads((EXPERIMENTS / "realistic_tasks/results/run2_T1.json").read_text())["states"]["s0_0.0175"]
    ref = {"share": saved["share"]["share"]["lo"], "steering": saved["steering"]["share"]["lo"], "erank": saved["erank"]}
    rows = {k: {"got": rec[k], "saved": v, "tol": th["reproduce_tol"][k]} for k, v in ref.items()}
    put(path, out, "e211_reread", rows | {"kl": rec["kl"], "adl_D": rec["adl"]["D"]})
    if bad := [k for k, v in rows.items() if not abs(v["got"] - v["saved"]) <= v["tol"]]:
        raise stop(path, out, f"E211's T1_s0_0.0175 does not repeat: {bad}")
    e187(path, out, C, th)
    put(path, out, "passed", True)


def e187(path: Path, out: dict[str, Any], C: Config, th: dict[str, Any]) -> None:
    """E254's copy_s0 at c = 1 and prior_cipher_s0 at E187's c through E228's share reader (P2.54's `weights`),
    against E147's saved copy share and E187's saved scaled cipher share."""
    W: Any = load_experiment("prior_matched", "world")
    w = W.World(0)
    for task in ("copy", "prior_cipher"):
        w.add_task(f"pa_{task}", W.E147_INV["maps"][task]["windows"])
        w.add_state(f"pa_{task}", 0)
    w.rd.c |= {"pa_copy": 1.0, "pa_prior_cipher": C.e187_c}
    m67 = json.loads((EXPERIMENTS / "size_matched/results/m67.json").read_text())["pairs"]["e58_copy_prior_cipher"]
    e147 = json.loads((EXPERIMENTS / "refit_96key/results/stage1.json").read_text())["states"]["e58/copy_s0"]["e136"]["trained"]
    assert abs(m67["c"] - C.e187_c) <= 1e-12, (m67["c"], C.e187_c)
    ref = {"e58/pa_copy_s0": e147, "e58/pa_prior_cipher_s0": m67["states"]["e58/prior_cipher_s0"]["share"]["r"]}
    got = {s: W.T2.weights(w, s)["r"] for s in ref}
    rows = {s: {"got": got[s], "saved": ref[s], "tol": th["reproduce_tol"]["share"]} for s in ref}
    put(path, out, "e187_split", rows | {"gap": got["e58/pa_copy_s0"] - got["e58/pa_prior_cipher_s0"]})
    if bad := [s for s, v in rows.items() if not abs(v["got"] - v["saved"]) <= v["tol"]]:
        raise stop(path, out, f"E187's copy vs prior-matched cipher split does not repeat: {bad}")


# ----------------------------------------------------------------------------- stage 1: train one task


def floor(net: ST.Net, C: Config, t: PN.Task, w0: float) -> dict[str, Any]:
    """The task's held-out gain floor: floor_mult x the largest |held gain| of Gaussian updates at floor_size x each
    matrix's norm (rl_67m `random_update`), one per draw."""
    S1: Any = load_experiment("rl_67m", "run_stage1")
    g = [PN.held_gain(net, t, {k: v.cpu() for k, v in S1.random_update(net.W0, C.floor_size, d).items()}, w0) for d in C.floor_draws]
    return {"random_gains": g, "floor": C.floor_mult * max(abs(x) for x in g)}


def save_dW(p: Path, dW: ST.DW) -> str:
    """A state as its bf16 file at p; its sha256."""
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in dW.items()}, p)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fine_tune(net: ST.Net, C: Config, t: PN.Task, seed: int, w0: float, kl: torch.Tensor, rc: PN.Recipe | None = None,
              states: Path = LARGE / "states") -> list[dict[str, Any]]:  # fmt: skip
    """One seed on the rungs; each save's bf16 file, its KL on all kl rows, size and held gain (read from the file).
    rc and states default to the filed recipe and directory (probe.py passes its own)."""
    rc, saves = rc or PN.Recipe(C.lr, C.batch, C.steps, C.grad_clip), []
    rungs: dict[str, Sequence[float]] = {"kl": PN.rung_ladder(C.kl_lo, C.kl_n), "size": PN.rung_ladder(C.size_lo, C.size_n)}

    def save(step: int, dW: ST.DW, info: dict[str, Any]) -> None:
        p = states / f"{t.name}_s{seed}_step{step}.pt"
        saves.append(info | {"file": str(p.relative_to(REPO)), "sha256": save_dW(p, dW)})

    t0 = time.time()
    trace = PN.train_ladder(net, t, rc, seed, kl[: C.kl_live_rows], rungs, save, log)
    for s in saves:
        d = load_dW(s["file"])
        s |= {"kl": PN.kl_from_w0(net, kl, d), "size": R.update_size(d, net.W0), "held_gain": PN.held_gain(net, t, d, w0)}
    log(f"[train {t.name} s{seed}] {time.time() - t0:.0f} s, {len(saves)} saves, last step {trace[-1]['step'] if trace else 0}")
    return [{"trace": trace}] + saves


def train(net: ST.Net, C: Config, name: str) -> None:
    path = RESULTS / "train" / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": name, "seeds": {}}
    t, kl = TK.build(net, C, name), rows_kl(C)
    w0 = PN.target_logp(net, t.held, t.held_mask, None)
    put(path, out, "rows", {"train": list(t.train.shape), "held": list(t.held.shape), "read": len(t.read.toks), "fit": len(t.read.fit), "w0_logp": w0})
    if "floor" not in out:
        put(path, out, "floor", floor(net, C, t, w0))
    if name in ("squad", "conll") and "twin" not in out:  # the twin's held rows carry the real answers
        put(path, out, "twin", fine_tune(net, C, TK.build(net, C, f"{name}_twin"), 0, w0, kl)[-1:])
    for seed in C.seeds:
        if str(seed) not in out["seeds"]:
            out["seeds"][str(seed)] = fine_tune(net, C, t, seed, w0, kl)
            put(path, out, "seeds", out["seeds"])
    if name in ("squad", "conll"):
        put(path, out, "learnability", learnability(out))


def learnability(out: dict[str, Any]) -> dict[str, Any]:
    """The filed call on seed 0's last save and the twin's last save, both read on the real held rows: learned when the
    real gain is at least the floor and exceeds the twin's by at least the floor."""
    real, twin, f = out["seeds"]["0"][-1], out["twin"][-1], out["floor"]["floor"]
    ok = real["held_gain"] >= f and real["held_gain"] - twin["held_gain"] >= f
    return {"real": real["held_gain"], "twin_on_real_rows": twin["held_gain"], "floor": f, "learned": ok, "real_step": real["step"], "twin_step": twin["step"]}


# ----------------------------------------------------------------------------- select (CPU) and stage 2


def trained(names: Sequence[str]) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """(runs: task -> seed -> saves, floors, dropped) from the train files of `names`."""
    runs, floors, dropped = {}, {}, {}
    for name in names:
        p = RESULTS / "train" / f"{name}.json"
        if not p.is_file():
            dropped[name] = "no train file"
            continue
        d = json.loads(p.read_text())
        if d.get("learnability", {}).get("learned") is False:
            dropped[name] = f"learnability failed: {d['learnability']}"
            continue
        runs[name] = {s: v[1:] for s, v in d["seeds"].items()}
        floors[name] = d["floor"]["floor"]
    return runs, floors, dropped


def trajectory(runs: dict[str, Any], floors: dict[str, float], C: Config) -> dict[str, Any]:
    """Amendment 4: per trajectory task, trajectory_points of seed 0's saves (step order) with held gain over the floor,
    evenly spaced by index from the first such save to the last; None with fewer such saves than points."""
    out: dict[str, Any] = {}
    for t in C.trajectory_tasks:
        ok = [s for s in runs.get(t, {}).get("0", []) if s["held_gain"] >= floors[t]]
        n, p = len(ok), C.trajectory_points
        out[t] = [ok[round(i * (n - 1) / (p - 1))] for i in range(p)] if n >= p else None
    return out


def select(C: Config) -> None:
    runs, floors, dropped = trained(C.tasks)
    sel = {"kl": PN.select(runs, "kl", PN.rung_ladder(C.kl_lo, C.kl_n), C.match_tol, floors),
           "size": PN.select(runs, "size", PN.rung_ladder(C.size_lo, C.size_n), C.match_tol, floors), "dropped": dropped}  # fmt: skip
    mruns, mfloors, mdropped = trained(MIXES)  # Amendment 4: the mixes at the panel's KL "hi" target, which they do not move
    sel["mix"] = PN.at_rung(mruns, "kl", sel["kl"]["hi"]["target"], C.match_tol, mfloors) | {"dropped": mdropped}
    sel["trajectory"] = trajectory(runs, floors, C)
    write_stage(RESULTS / "selection.json", jsonable(sel))
    kept = {f"{k}_{lvl}": sorted((sel[k][lvl] or {}).get("tasks", {})) for k in ("kl", "size") for lvl in ("hi", "lo")}
    kept |= {"mix": sorted(sel["mix"]["tasks"]), "trajectory": sorted(t for t, v in sel["trajectory"].items() if v)}
    write_stage(RESULTS / "dropped.json", {"written": time.time(), "dropped": dropped | mdropped, "kept": kept})  # Amendment 3: drops before reads
    log(f"[select mix] {sorted(sel['mix']['tasks'])}, out {sel['mix']['out']}; trajectory {kept['trajectory']}")
    for k in ("kl", "size"):
        for lvl in ("hi", "lo"):
            s = sel[k][lvl]
            log(f"[select {k} {lvl}] " + ("none" if s is None else f"target {s['target']:.4g}: in {sorted(s['tasks'])}, out {s['out']}"))


def picked(name: str) -> dict[str, dict[str, Any]]:
    """The task's selected states, one entry per file, with every (match, level) that picked it, and Amendment 4's
    picks ("mix", "trajectory")."""
    sel, got = json.loads((RESULTS / "selection.json").read_text()), {}

    def add(st: dict[str, Any], seed: Any, label: str) -> None:
        got.setdefault(st["file"], st | {"seed": int(seed), "picked": []})["picked"].append(label)

    for k in ("kl", "size"):
        for lvl in ("hi", "lo"):
            for seed, st in ((sel[k][lvl] or {}).get("tasks", {}).get(name) or {}).items():
                add(st, seed, f"{k}_{lvl}")
    for seed, st in (sel["mix"]["tasks"].get(name) or {}).items():
        add(st, seed, "mix")
    for st in sel["trajectory"].get(name) or []:
        add(st, 0, "trajectory")
    return got


def read(net: ST.Net, C: Config, name: str) -> None:
    assert (RESULTS / "dropped.json").is_file(), "run select first: no reading before the selection and drops are written"
    assert json.loads(PRED.read_text()).get("raters"), "the raters' tiers are not filed in predictions.json: no reading before them"
    path = RESULTS / "read" / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"task": name, "states": {}, "random": {}}
    states, t = picked(name), TK.build(net, C, name)
    rd, kl, pile = reader(net, C, t.read), rows_kl(C), rows_adl(C)
    for f, st in states.items():
        if f not in out["states"]:
            out["states"][f] = st | readings(rd, load_dW(f), kl, pile, C, st["seed"])
            put(path, out, "states", out["states"])
    for f, st in states.items():  # one Gaussian per seed-0 panel-set state, at its per-matrix norms, over its gain
        if st["seed"] == 0 and set(st["picked"]) - set(EXTRA) and f not in out["random"]:
            out["random"][f] = readings(rd, FT.random_direction(load_dW(f), C.random_seed), kl, pile, C, 0, out["states"][f]["gain"])
            put(path, out, "random", out["random"])


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] not in ("0", "train", "select", "read") or (args[0] in ("train", "read") and (len(args) != 2 or args[1] not in TASKS + MIXES)):
        sys.exit("usage: run.py 0 | train TASK | select | read TASK")
    C, th = load()
    if args[0] == "select":
        select(C)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    try:
        stage0(net, C, th) if args[0] == "0" else train(net, C, args[1]) if args[0] == "train" else read(net, C, args[1])
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    print(f"STAGE {' '.join(args)} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())
