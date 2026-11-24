"""E300 (cell P2.63): the reuse dial on four realistic tasks of the 67M (sequence, brackets, addition: E256's procedural
tasks and SFT recipe; lowercase: E53's real-text task and recipe). Per task and seed, fine-tunes that all stop at one
matched trained-answer gain, each written as a gain on every VPD variable plus a free residual penalised by beta
(`commute.dial`): gains only (beta = inf, reuse by construction), the filed betas, and the ordinary fine-tune (no gains,
no penalty); one Gaussian at the ordinary seed 0's per-matrix norms per task. Each state is read by E211's reader (the
re-gating share, steering, effective rank, the neuron shares; as E256 read its states), E256's KL from W0, E267's lens
column D (ADL), and E_b over the needed set K (E275's path: E268's scan settings). Plan
reports/plans/2026-11-24_plan_p2-63-reuse-dial.md; settings, thresholds and rules results/predictions.json (the run
refuses unless it is committed unchanged). The score (no GPU): score.py; the figure: plot.py.

Jobs (every job runs stage 0 first, once per tag; a miss writes results/stage0_<tag>.json with "stopped", exits 1):
  pod I N      every N-th (task, seed) item from the I-th; per item, cheapest first, each part written as it ends and
               skipped on a rerun: train every arm (the gains-only arm first), then the Gaussian (seed 0), then read
               every state (the Gaussian first), then E_b over K (gains only first) for seeds in eb_seeds; then the
               item's files are packed in results/large/packs/<task>_s<seed>.tar for one download (every small JSON
               also in results/large/packs/json.tar, rewritten after every part)
  calibrate T  rehearsal only: seed 0 of task T, the gains-only arm to its cap (its plateau), then every beta of
               rehearse_betas to the target min(t_ref, plateau_frac x plateau): steps, reached, the residual share;
               written after every arm, an arm already written skipped; --arms A ... trains only those (one task's
               arms split over two jobs, the second started after the first)
  rehearse     stage 0, then item (rehearse_task, 0) with every arm capped at rehearse_cap steps, read and E_b'd in
               full, into results/rehearse/; run it twice (the second must skip every part)
Stage 0 (filed values in thresholds): (a) the dial's start changes no bit of the logits (fp32 and bf16 autocast) and
exports exactly 0; (b) a random dial's export reproduces its hooked logits within equiv_tol and a shuffled one does not
(perm_min); (c) E211's zero check (every reader numerator exactly 0) and E256's lens/KL zero; (d) the do-nothing reads
E_b = 1 with no changed variable; (e) a random gains-only state reads E_b over every changed variable at least
eb_gain_min (the dial's construction under the definition; under 1 where the definition's least-squares split spreads
one subcomponent's gain over overlapping ones), the Gaussian under eb_random_max (both again on every item: the
gains-only state, the first E_b, before its scan; the Gaussian over K after its scan); (f) E268's identity
scan on that state: no argmax change (its largest deviation is the floor; its rate the seconds per item).

    PRED_SHA=<sha> python experiments/reuse_dial/run.py pod 0 9 --tag p0 2>&1 | tee -a experiments/reuse_dial/results/pod0.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import adl as ADL
from commute import dial as DL
from commute import finetune as FT
from commute import kstar_vpd as KV
from commute import lowercase as LC
from commute import needed as IX
from commute import realistic_reads as RR
from commute import rl_arms as RA
from commute import stand_in as SI
from commute import steering as ST
from commute import vpd
from commute import vpd_gains as VG
from commute.data import LAST, cached_rows, pile_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    manifest_only,
    require_filed,
    write_stage,
)
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable
from commute.realistic import update_size

KN: Any = load_experiment("kstar_needed", "run")  # E268: world (F, M, D, featurization), load_config (scan settings), KP
PR: Any = load_experiment("procedural_rl", "run")  # E256: Config, build_items, NT, PRED, READS
R2: Any = load_experiment("realistic_tasks", "run2")  # E211: reader, zero_check, e53_read, Config, PRED
S1: Any = load_experiment("rl_67m", "run_stage1")  # E215/E256: kl_from_base
AR: Any = load_experiment("adl", "run")  # E264/E267: gain, removed_mean, shares, HF (the lens column D)
WIN: Any = load_experiment("share_untrained", "windows")  # E136's read windows (lowercase's lens rows)
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
ADL_PRED = EXPERIMENTS / "adl" / "results" / "amend2" / "predictions.json"
E136_INV = EXPERIMENTS / "share_untrained" / "results" / "inventory.json"
PROCEDURAL = ("sequence", "brackets", "addition")
D_ITEM = {"sequence": "p256/sequence_sft_s0", "brackets": "p256/brackets_sft_s0", "addition": "p256/addition_sft_s0", "lowercase": "e53/lowercase_s0"}
log = KN.log


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    eb_seeds: tuple[int, ...]  # seeds whose states get E_b over K
    betas: tuple[float, ...]  # the penalised arms (each with gains), ascending
    target: dict[str, float]  # task -> the matched trained-answer gain (log-odds)
    lr_gain: float
    lr_res: float
    grad_clip: float
    batch: dict[str, int]  # task -> training batch (its reference recipe's)
    cap: dict[str, int]  # task -> step cap of every arm
    kl_rows: int
    kl_batch: int
    gain_batch: int
    rand_sd: float  # stage 0's random dial: gains 1 + rand_sd N(0, 1), residual rand_sd x 1e-3 x W0's rms
    random_seed: int
    rehearse_task: str
    rehearse_cap: int
    rehearse_eb_arms: tuple[str, ...]  # the rehearsal's full E_b scans (the debug QoS's 2 h holds two at 64 tokens)
    rehearse_betas: tuple[float, ...]
    plateau_frac: float
    t_ref: dict[str, float]  # task -> the reference fine-tunes' mean gain (E256's SFT; E53's lowercase)

    def __post_init__(self) -> None:
        for f in ("tasks", "seeds", "eb_seeds", "betas", "rehearse_betas", "rehearse_eb_arms"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        assert set(self.tasks) <= set(D_ITEM) and set(self.eb_seeds) <= set(self.seeds) and list(self.betas) == sorted(self.betas)
        assert all(0 < b < math.inf for b in self.betas) and set(self.target) == set(self.cap) == set(self.t_ref) == set(self.batch) == set(self.tasks)
        assert all(MIN_GAIN <= self.target[t] <= self.t_ref[t] for t in self.tasks) and self.rehearse_cap < min(self.cap.values())
        assert self.rehearse_task in self.tasks and 0 < self.plateau_frac <= 1

    def arms(self) -> list[tuple[str, bool, bool, float]]:
        """(name, gains, residual, beta), the dial's order from reuse by construction to the ordinary fine-tune."""
        return [("g", True, False, math.inf)] + [(f"b{b:g}", True, True, b) for b in self.betas[::-1]] + [("ft", False, True, 0.0)]


def load() -> tuple[Config, dict[str, Any]]:
    P = json.loads(PRED.read_text())
    return config_from_settings(Config, P["settings"]), P["thresholds"]


# ---- the world: one model, the tasks -------------------------------------------------------------------------------


class World:
    def __init__(self, cfg: Config, out: Path) -> None:
        self.cfg, self.out, self.large = cfg, out, HERE / "results" / "large" / out.relative_to(HERE / "results")
        self.w = KN.world()  # E268's: w.kl.net (the 67M), w.kl.M, w.uv, w.bases, w.make(item) for D
        self.net: ST.Net = self.w.kl.net
        self.b = VG.Basis.of(self.net.W0, self.w.uv, self.net.device)
        self.w0sq, self.kc = DL.w0_sq(self.net.W0), KN.load_config()
        self.kl_rows = cached_rows("val", 0, cfg.kl_rows)
        a = json.loads(ADL_PRED.read_text())
        self.adl, self.gain_floor = a["settings"], a["thresholds"]["gain_floor"]
        self.pile = pile_rows(self.adl["pile_split"], 0, self.adl["n_pile"])
        P2 = json.loads(R2.PRED.read_text())
        self.c2 = config_from_settings(R2.Config, P2["settings"])
        self.tasks: dict[str, Any] = {}

    def task(self, name: str) -> Any:
        if name not in self.tasks:
            self.tasks[name] = (proc_task if name in PROCEDURAL else lowercase_task)(self, name)
        return self.tasks[name]


def proc_task(W: World, name: str) -> Any:
    """E256's items (procedural_rl build_items), read rows (CueRead), SFT loss and sampler; D: E257's p256 windows."""
    C, tok = config_from_settings(PR.Config, json.loads(PR.PRED.read_text())["settings"]), vpd.tokenizer()
    v = PR.NT.Vocab([tok.decode([i]) for i in range(len(tok))], tok.encode)
    d = PR.build_items(v, C, cached_rows("train", 0, C.pile_train_n), cached_rows("val", C.val_start, C.val_n))[name]
    held, fit, tp = d["held"].take(slice(0, C.n_held)), d["train"].take(slice(0, C.n_fit)), d["train"]
    read = RR.CueRead(name, held.toks, held.true, C.prompt_len - 1, fit.toks, fit.true)
    lens = SimpleNamespace(toks=torch.cat([held.toks, held.true[:, None]], 1), token=LAST, half=C.prompt_len - 1, ci_len=C.prompt_len)
    tm, B = W.net.net, W.cfg.batch[name]

    def sampler(seed: int) -> Any:
        pick = torch.Generator().manual_seed(seed)

        def draw(_k: int) -> tuple[Tensor, Tensor]:  # E256's `sft` batch: sft_batch training items drawn with the seed's generator
            i = torch.randint(0, len(tp.toks), (B,), generator=pick)
            return tp.toks[i], tp.true[i, None]

        return draw

    return finish_task(W, name, read, lens, sampler, lambda b: -RA.cont_logp(tm, b[0], b[1]).mean())


def lowercase_task(W: World, name: str) -> Any:
    """E53's rows in batch order `seed` (`lowercase.batch_order`, each once per pass), next-token CE under bf16 autocast;
    read rows E136's (realistic_tasks run2 e53_read); D: E257's e53/lowercase_s0 windows."""
    rows, read, B = LC.task(W.net.model, W.net.device, log).rows, R2.e53_read("lowercase"), W.cfg.batch[name]
    lens = WIN.load_read("e53/lowercase", json.loads(E136_INV.read_text())["tasks"]["e53/lowercase"])

    def sampler(seed: int) -> Any:
        r, n = LC.batch_order(rows, seed), rows.shape[0] // B
        return lambda k: r[((k - 1) % n) * B : ((k - 1) % n + 1) * B].to(W.net.device)

    return finish_task(W, name, read, lens, sampler, lambda t: FT.next_token_ce(W.net.net, t)[0])


def finish_task(W: World, name: str, read: Any, lens_read: Any, sampler: Any, loss_of: Any) -> Any:
    _, _, D, at, _ = W.w.make(D_ITEM[name])
    lens = AR.HF.Task(lens_read, None, None)
    lens.lo0 = AR.HF.lo_at(W.net, lens, lens.read.toks)
    t = SimpleNamespace(name=name, read=read, lens=lens, sampler=sampler, loss_of=loss_of, D=D, at=at)
    t.lo0 = answer_lo(W, t)
    t.reader = R2.reader(W.net, W.c2, read)
    return t


@torch.no_grad()
def answer_lo(W: World, t: Any) -> Tensor:
    """(n,) the trained answer's log-odds at the read position of each read row, fp32, the net as it stands (hooks on)."""
    r, B = t.read, W.cfg.gain_batch
    return torch.cat([ST.logodds_at(RA.logits(W.net.net, r.toks[s : s + B, : r.pos + 1].to(W.net.device)), r.pos, r.tgt[s : s + B]) for s in range(0, len(r.toks), B)])


def gain_of(W: World, t: Any, d: DL.Dial | None) -> float:
    if d is None:
        return 0.0
    with DL.applied(W.net.net, W.b, d):
        return float((answer_lo(W, t) - t.lo0).mean())


# ---- stage 1: train one arm to the matched gain --------------------------------------------------------------------


def state_path(W: World, name: str) -> tuple[Path, Path]:
    return W.out / "states" / f"{name}.json", W.large / "states" / f"{name}.pt"


def save_dw(W: World, name: str, dW: dict[str, Tensor]) -> dict[str, Any]:
    p = state_path(W, name)[1]
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.float() for k, v in dW.items()}, p)  # fp32, every state alike: bf16 rounding moves E_b off its ends
    return {"file": str(p.relative_to(EXPERIMENTS.parent)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "mb": p.stat().st_size / 2**20}


def load_dw(rec: dict[str, Any]) -> dict[str, Tensor]:
    p = EXPERIMENTS.parent / rec["file"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"], f"{p}: sha256 differs from its train record"
    return {k: v.float() for k, v in torch.load(p, weights_only=True).items()}


def train_arm(W: World, t: Any, arm: tuple[str, bool, bool, float], seed: int, target: float, cap: int, th: dict[str, Any]) -> dict[str, Any]:
    """The arm from the start to the first step whose read-row gain reaches `target` (checked every step), or the cap.
    Early check: step 1's gain under target x step1_max (a first step that lands near the target has no dial)."""
    name, gains, residual, beta = arm
    d, cfg, curve, t0 = DL.Dial.start(W.b, gains, residual), W.cfg, list[dict[str, float]](), time.time()

    def on_check(step: int, d: DL.Dial) -> bool:
        g = gain_of(W, t, d)
        curve.append({"step": step, "gain": g})
        if not math.isfinite(g):
            raise SystemExit(f"{t.name} {name} s{seed}: step {step} gains {g}")
        if step == 1 and not g < target * th["step1_max"]:
            raise SystemExit(f"{t.name} {name} s{seed}: step 1 gains {g:.3f}, at or over {th['step1_max']} x the target {target:.3f}")
        return g >= target

    torch.cuda.reset_peak_memory_stats()
    rows = DL.train(W.net.net, W.b, d, t.sampler(seed), t.loss_of, {"gain": cfg.lr_gain, "res": cfg.lr_res}, beta, W.w0sq, cap,
                    cfg.grad_clip, 1, on_check, log)  # fmt: skip
    g, g_alone = curve[-1]["gain"], gain_of(W, t, d.gains_only() if gains else None)
    dW = DL.delta(W.b, d)
    return {"arm": name, "beta": beta, "seed": seed, "target": target, "steps": len(rows), "reached": g >= target, "gain_hooked": g,
            "gain_gains_alone": g_alone, "residual_share": 1 - g_alone / g if g else math.nan, "residual_size": DL.residual_size(d, W.w0sq),
            "size": update_size(dW, W.net.W0), "curve": curve, "loss": [r["loss"] for r in rows], "seconds": time.time() - t0,
            "s_per_step": (time.time() - t0) / len(rows), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30} | save_dw(W, f"{t.name}_{name}_s{seed}", dW)  # fmt: skip


# ---- stage 2 and 3: the readings of one saved state ---------------------------------------------------------------------


def lens_column(W: World, t: Any, dW: dict[str, Tensor], ref: dict[str, Tensor] | None) -> dict[str, Any]:
    """E264's column D on the task's rows (adl run.py read_item's arithmetic, E267's procedural layout)."""
    ns, a = SimpleNamespace(net=W.net), W.adl
    dbar = ADL.mean_diff(W.net, W.pile[:, : a["k"]], dW, a["batch"])
    nums = AR.removed_mean(ns, t.lens, dW, ADL.unit(dbar))
    G = AR.gain(ns, t.lens, dW if ref is None else ref)
    per_j, D = AR.shares(nums, G, W.gain_floor)
    return {"D": D, "G": G, "removed": nums, "share": per_j}


def read_state(W: World, t: Any, rec: dict[str, Any], ref: dict[str, Tensor] | None) -> dict[str, Any]:
    dW, t0 = load_dw(rec), time.time()
    out = {"kl": S1.kl_from_base(W.net, W.kl_rows, dW, W.cfg.kl_batch), "lens": lens_column(W, t, dW, ref)}
    return out | R2.brief(t.reader.state(dW, rec["seed"])) | {"read_seconds": time.time() - t0}


def eb_state(W: World, t: Any, rec: dict[str, Any], name: str, scan: bool = True, floor: float = -math.inf) -> dict[str, Any]:
    """E_b over K as E275 reads a fresh state: F = M + the saved dW on the task's D, one write pass, E268's scan (with
    scan False: E_b over every changed variable only, which a gains-only state reads near 1 whatever K is). E_b over
    the changed variables under `floor` stops before the scan (SystemExit)."""
    t0, kc = time.time(), W.kc
    F, M, toks, at, _ = W.w.from_dw(EXPERIMENTS.parent / rec["file"], t.D, t.at)
    feat = KV.featurize(M, F, W.w.uv, W.w.delta, W.w.bases)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    out = {"E_b_changed": SI.explained_share(num, den, dd, ch), "n_changed": int(ch.sum()), "n_vars": feat.n}
    if not out["E_b_changed"] >= floor:
        raise SystemExit(f"{name}: E_b {out['E_b_changed']:.6f} over its changed variables, under {floor}")
    if not scan:
        return out
    nd = IX.needed_set(F, feat, toks, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
    K = nd.K
    np.savez_compressed(W.large / "eb" / f"{name}.npz", in_K=K.numpy(), first_flip=nd.first_flip.numpy(), num=num.numpy(), den=den.numpy(), dd=dd.numpy(), changed=ch.numpy())
    del F, nd
    torch.cuda.empty_cache()
    return out | {"E_b_K": SI.explained_share(num, den, dd, K & ch), "size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()), "eb_seconds": time.time() - t0}


# ---- one item ---------------------------------------------------------------------------------------------------


def stop_state(W: World, name: str, why: str) -> None:
    """A filed early check missed: the state's record gets "stopped" (a rerun clears it and tries again), exit 1."""
    path = state_path(W, name)[0]
    write_stage(path, jsonable(json.loads(path.read_text()) | {"stopped": why}))
    pack_json(W)
    log(f"STOP: {why}")
    hard_exit(1)


def got(W: World, name: str) -> dict[str, Any]:
    """A state's train record, written by an earlier part."""
    return json.loads(state_path(W, name)[0].read_text())["train"]


def part(W: World, name: str, key: str, fn: Any) -> dict[str, Any]:
    """rec[key] = fn(rec), written at once; skipped when the state's record already holds it (the resume). A filed early
    check that fails inside fn (SystemExit) is written as the record's "stopped" and ends the job with exit 1."""
    path = state_path(W, name)[0]
    rec = json.loads(path.read_text()) if path.is_file() else {"experiment": json.loads(PRED.read_text())["experiment"], "state": name}
    rec.pop("stopped", None)
    if key not in rec:
        try:
            rec[key] = fn(rec)
        except SystemExit as e:
            write_stage(path, jsonable(rec))
            stop_state(W, name, str(e))
        write_stage(path, jsonable(rec))
        pack_json(W)
        log(f"[{name} {key}] {json.dumps(jsonable({k: v for k, v in rec[key].items() if k not in ('curve', 'loss', 'removed', 'share', 'steering', 'k_half')}))}")
    else:
        log(f"{name} {key}: done before, skipped")
    return rec


def item(W: World, task: str, seed: int, cap: int | None, th: dict[str, Any], eb_arms: tuple[str, ...] | None = None) -> None:
    """One (task, seed): train every arm, the Gaussian (first seed), read every state, E_b over K (gains only first;
    only the arms in eb_arms when given: the rehearsal's), pack."""
    cfg, t = W.cfg, W.task(task)
    if manifest_only():
        return
    target, names = cfg.target[task], {a[0]: f"{task}_{a[0]}_s{seed}" for a in cfg.arms()}
    for arm in cfg.arms():
        part(W, names[arm[0]], "train", lambda _r, arm=arm: train_arm(W, t, arm, seed, target, cap or cfg.cap[task], th))
    states = [names[a[0]] for a in cfg.arms()]
    if seed == cfg.seeds[0]:  # the do-nothing reference: a Gaussian at the ordinary fine-tune's per-matrix norms
        rnd, ft = f"{task}_rand_s{seed}", got(W, names["ft"])
        part(W, rnd, "train", lambda _r: {"arm": "rand", "seed": seed, "matched_to": names["ft"]} | save_dw(W, rnd, FT.random_direction(load_dw(ft), cfg.random_seed)))
        states = [rnd, *states]
    for n in states:
        rec = got(W, n)
        ref = load_dw(got(W, names["ft"])) if rec["arm"] == "rand" else None
        rd = part(W, n, "read", lambda _r, rec=rec, ref=ref: read_state(W, t, rec, ref))["read"]
        if rec["arm"] == "rand" and not abs(rd["gain"]["lo"]) < MIN_GAIN:
            stop_state(W, n, f"{n}: a Gaussian gains {rd['gain']['lo']:.3f} log-odds, at or over {MIN_GAIN}")
    eb = sorted(states, key=lambda n: n != names["g"]) if seed in cfg.eb_seeds else []
    for n in [n for n in eb if eb_arms is None or got(W, n)["arm"] in eb_arms]:
        (W.large / "eb").mkdir(parents=True, exist_ok=True)
        floor = th["eb_gain_min"] if n == names["g"] else -math.inf  # the gains-only state, the first E_b, before its scan
        e = part(W, n, "eb", lambda r, n=n, floor=floor: eb_state(W, t, r["train"], n, floor=floor))["eb"]
        if got(W, n)["arm"] == "rand" and not e["E_b_K"] <= th["eb_random_max"]:
            stop_state(W, n, f"{n}: a Gaussian reads E_b {e['E_b_K']:.4f} over K, over {th['eb_random_max']}")
    pack(W, task, seed, states)


def pack(W: World, task: str, seed: int, states: list[str]) -> None:
    p = W.large / "packs" / f"{task}_s{seed}.tar"
    p.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(p, "w") as tar:
        for n in states:
            for f in (*state_path(W, n), W.large / "eb" / f"{n}.npz"):
                if f.is_file():
                    tar.add(f, arcname=str(f.relative_to(EXPERIMENTS.parent)))
    log(f"PACKED {p.relative_to(EXPERIMENTS.parent)} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()} ({p.stat().st_size / 2**20:.0f} MB)")


def pack_json(W: World) -> None:
    """Every small record of the job (stage 0, calibrate, states/*.json) in results/large/packs/json.tar, rewritten
    after every part (renamed into place, so a download never reads half a tar): a lost pod loses at most one part."""
    p = W.large / "packs" / "json.tar"
    p.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(p.with_suffix(".tmp"), "w") as tar:
        for f in sorted([*W.out.glob("*.json"), *(W.out / "states").glob("*.json")]):
            tar.add(f, arcname=str(f.relative_to(EXPERIMENTS.parent)))
    p.with_suffix(".tmp").replace(p)


# ---- stage 0, calibration, main ------------------------------------------------------------------------------------


def stage0(W: World, task: str, th: dict[str, Any], tag: str) -> None:
    path = W.out / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return log(f"{path.name}: done before")
    W.task(task)
    if manifest_only():
        return
    s0: Any = load_experiment("reuse_dial", "stage0")
    out = s0.run(W, task, tag, eb_state, save_dw)
    miss = s0.misses(out, th)
    write_stage(path, jsonable(out | ({"stopped": "; ".join(miss)} if miss else {"done": True})))
    pack_json(W)
    log(f"stage 0: {json.dumps(jsonable(out))}; {'MISSES ' + '; '.join(miss) if miss else 'checks hold'}")
    if miss:
        hard_exit(1)
    log("STAGE 0 DONE")


def cal_arms(cfg: Config) -> list[tuple[str, bool, bool, float]]:
    return [(f"b{b:g}", True, True, b) for b in (*cfg.rehearse_betas, 0.0)] + [("ft", False, True, 0.0)]


def calibrate(W: World, task: str, th: dict[str, Any], only: list[str] | None) -> None:
    """Rehearsal: the gains-only plateau at the cap, the target it sets, and each rehearse beta's residual share there.
    results/rehearse/calibrate_<task>.json is written after every arm; an arm already in it is skipped (the resume, and
    the split of one task over two jobs: `only` names the arms this job trains; the gains-only arm runs when missing)."""
    cfg, t, path = W.cfg, W.task(task), W.out / f"calibrate_{task}.json"
    if manifest_only():
        return
    th = th | {"step1_max": math.inf}  # calibration records step 1 (curve[0]) rather than stopping on it
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    if "g" not in out:
        g = train_arm(W, t, ("cal_g", True, False, math.inf), 0, math.inf, cfg.cap[task], th)
        plateau = max(c["gain"] for c in g["curve"])
        out = {"task": task, "plateau": plateau, "t_ref": cfg.t_ref[task], "target": min(cfg.t_ref[task], cfg.plateau_frac * plateau),
               "g": {k: v for k, v in g.items() if k not in ("curve", "loss")}}  # fmt: skip
        write_stage(path, jsonable(out))
        pack_json(W)
    for arm in [a for a in cal_arms(cfg) if a[0] not in out and (only is None or a[0] in only)]:
        r = train_arm(W, t, (f"cal_{arm[0]}", *arm[1:]), 0, out["target"], cfg.cap[task], th)
        out[arm[0]] = {k: r[k] for k in ("steps", "reached", "gain_hooked", "gain_gains_alone", "residual_share", "residual_size", "size", "s_per_step")}
        out[arm[0]]["step1_gain"] = r["curve"][0]["gain"]
        write_stage(path, jsonable(out))
        pack_json(W)
        log(f"[calibrate {task} {arm[0]}] {json.dumps(jsonable(out[arm[0]]))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("pod", "calibrate", "rehearse"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--arms", nargs="+", help="calibrate only: the arms this job trains (default every arm)")
    a = ap.parse_args()
    require_filed(PRED)
    cfg, th = load()
    assert a.arms is None or set(a.arms) <= {x[0] for x in cal_arms(cfg)}, f"--arms: one of {[x[0] for x in cal_arms(cfg)]}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as E256, E268
    out = HERE / "results" / ("rehearse" if a.job in ("rehearse", "calibrate") else "")
    log(f"{json.loads(PRED.read_text())['experiment']} {a.job} {a.args} tag {a.tag}; settings {asdict(cfg)}; thresholds {th}")
    W = World(cfg, out)
    if a.job == "calibrate":
        stage0(W, a.args[0], th, a.tag)
        for task in a.args:
            calibrate(W, task, th, a.arms)
    elif a.job == "rehearse":
        stage0(W, cfg.rehearse_task, th, a.tag)
        item(W, cfg.rehearse_task, cfg.seeds[0], cfg.rehearse_cap, th, cfg.rehearse_eb_arms)
    else:
        i, n = int(a.args[0]), int(a.args[1])
        todo = [(t, s) for t in cfg.tasks for s in cfg.seeds][i::n]  # task-major: with tasks in cost order, n = 9 gives each pod one procedural item, three a lowercase one too
        log(f"pod {i} of {n}: items {todo}")
        stage0(W, todo[0][0], th, a.tag)
        for k, (task, seed) in enumerate(todo):
            item(W, task, seed, None, th)
            log(f"ITEM {k + 1} of {len(todo)} DONE ({task} s{seed})")
    log("DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main())
