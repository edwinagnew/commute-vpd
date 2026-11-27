"""E325 (cell P1.66; card card.md): Fig 2's retained share E_b over the needed set K on all 16 known-answer tasks x seeds
0-2 (`commute.known_states`), at VPD subcomponents, top neurons and SAE latents, under E298's protocol
(experiments/fig2_standard, read only, never imported): one write pass over D in F's own run (`eb.write_sums`), K by
the featurization's swap test on 4 seeded derangements x 64 windows (`needed.derangements`, seed 268), every variable
including the leftovers (VPD's per-matrix remainders, the pooled MLP remainders, the SAE error terms), E_b with the
leftovers counted (`eb.eb`). Every reading is the library's (src/commute: eb, needed, needed_neurons, needed_sae,
kstar_vpd, sae_feat, world, items, old_tasks, variable_read). D is the definition's, as E298 read it: the task's
`old_tasks.task_windows` read half (shows, as many does-not and 2/3 as many far rows, 8-token windows), E221's
seeded 512-window subset, read at the last position; README.md lists the per-task entries.

States E298 read already are not read again: e58/{copy, rank1, cipher, prior_cipher}_s<0-2> are E298's trained
states (the same E147 files); score.py takes their rows from E298. The 12 other tasks x 3 seeds are read here, each
with one matched random by E290's recipe (`finetune.matched_random`: a Gaussian per tensor F changed, at the Frobenius
norm of F - M on that tensor), seed random_seed + the state's index in `new_states`.

Every job runs stage 0 on its own card first (results/stage0/<feat>_<part>_<tag>.json, skipped once done; a miss
stops the job): do-nothing (F = M on the control's D: 0 changed, dd max 0, E_b 1), identity (x' = x: 0 argmax
changes), SAE must_flip (every variable swapped at once gives F's argmax on x'), and the reproduction of one saved
E298 read (its F from known_states' e58 file, its D E237's windows from key_grid's World, inputs only) within E298's
tolerance. Then its shard: each item to results/items/<feat>/ and results/large/<feat>/ as it lands, skipped when its
JSON is there. Every state file read is listed with its sha256 in results/state_files/<feat>_<tag>.json.

    PRED_SHA=<sha> python experiments/fig2_all_tasks/run.py --feat vpd --shard 0 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import eb as EB
from commute import items as IT
from commute import known_states as KS
from commute import kstar as K
from commute import kstar_vpd as KV
from commute import needed as IX
from commute import needed_neurons as NN
from commute import needed_sae as NS
from commute import old_tasks as OT
from commute import rho, vpd
from commute import rule_retention as R
from commute import sae_feat as SF
from commute import stand_in as SI
from commute import variable_read as VR
from commute.finetune import matched_random
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    manifest_only,
    require_filed,
    stop,
    write_stage,
)
from commute.windows import file_sha256
from commute.world import World

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
FEATS = ("vpd", "neurons", "sae")
SAME_AS_E298 = ("band", "derangements", "examples", "seed", "rows", "logit_rows", "cache_batch", "ident_vars", "sae_err_rows", "sae_chunk", "repro_rel_tol")
DEF_AT = slice(-1, None)  # the definition D's read position, the last of 8 (E221 LAST), as E298 read it


def log(msg: str) -> None:
    print(msg, flush=True)


def settings_of(exp: str) -> dict[str, Any]:
    return json.loads((EXPERIMENTS / exp / "results" / "predictions.json").read_text())["settings"]


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    control: str  # stage 0's do-nothing, identity and SAE must_flip read on this state's D
    tasks: tuple[str, ...]  # the known-answer tasks E298 did not read; their order sets the random seeds
    overlap: dict[str, str]  # known-answer task -> E298's task name (its trained/<task>_s<seed>)
    seeds: tuple[int, ...]
    random_seed: int  # the random of new_states()[k] has seed random_seed + k
    split_tol: float  # VPD's split bases (E226)
    n_read: int  # E221's subset of the task_windows read half (E298's D)
    read_seed: int
    band: str  # E229's neuron band
    sae_size: str  # E234's SAEs as E245 (and E298) load them
    sae_seed: int
    derangements: int
    examples: int
    seed: int
    rows: int
    logit_rows: int
    cache_batch: int
    ident_vars: int
    sae_err_rows: int
    sae_chunk: int
    repro: dict[str, dict[str, str]]  # feat -> the saved E298 read reproduced in stage 0: state, its E298 item json and npz
    repro_rel_tol: dict[str, float]  # E298's

    def new_states(self) -> list[str]:
        return [f"{t}_s{s}" for t in self.tasks for s in self.seeds]

    def items(self) -> list[str]:
        return [f"{n}{r}" for n in self.new_states() for r in ("", "_rand")]

    def __post_init__(self) -> None:
        assert len(self.tasks) + len(self.overlap) == 16 and not set(self.tasks) & set(self.overlap), "16 known-answer tasks"
        assert self.control in self.new_states() and set(self.repro) == set(self.repro_rel_tol) == set(FEATS)
        assert all(r["state"].rsplit("_s", 1)[0] in self.overlap for r in self.repro.values()), "a reproduction is of an E298 state"


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    cfg = config_from_settings(Config, s)
    e298, e245, e226 = settings_of("fig2_standard"), settings_of("sae_r_a"), settings_of("kstar_vpd")
    miss = [f"{k} {getattr(cfg, k)!r} != E298's {e298[k]!r}" for k in SAME_AS_E298 if getattr(cfg, k) != e298[k]]
    miss += [f"{k} {getattr(cfg, k)!r} != E245's {e245[k]!r}" for k in ("sae_size", "sae_seed") if getattr(cfg, k) != e245[k]]
    miss += [f"split_tol {cfg.split_tol} != E226's {e226['split_tol']}"] if cfg.split_tol != e226["split_tol"] else []
    e221 = settings_of("kstar_pairs")
    miss += [f"{k} {getattr(cfg, k)} != E221's {e221[k]}" for k in ("n_read", "read_seed") if getattr(cfg, k) != e221[k]]
    assert not miss, "not the sources' settings: " + "; ".join(miss)
    assert {s.task for s in KS.states()} == set(cfg.tasks) | set(cfg.overlap), "known_states' tasks are not this run's"
    if none := sorted({s.task for s in KS.states() if s.task in cfg.tasks and s.old_task is None}):
        raise SystemExit(f"STOP: no old_tasks entry (the definition's D) for {none}; not built here")
    return cfg


# ---- the three featurizations: prep (the variables of F against M), sums, cache, scan --------------------------------


def _mask(n: int, idx: set[int]) -> Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    m[sorted(idx)] = True
    return m


class VPD:
    def __init__(self, cfg: Config, w: World) -> None:
        self.cfg, self.w, self.bases = cfg, w, w.bases(cfg.split_tol)

    def prep(self, F: R.Model, M: R.Model, toks: Tensor) -> KV.Feat:
        assert not (d := KV.undecomposed_differences(M, F, sorted(self.w.uv))), f"parameters outside the decomposition differ: {d[:4]}"
        return KV.featurize(M, F, self.w.uv, self.w.delta, self.bases)

    def sums(self, st: KV.Feat, F: R.Model, toks: Tensor) -> EB.Sums:
        return EB.write_sums(F.net, SI.vpd_sites(st), toks, self.cfg.cache_batch, _mask(st.n, st.changed))

    def cache(self, st: KV.Feat, F: R.Model, toks: Tensor, at: slice) -> Any:
        return IX.cache(F, st, toks, at, self.cfg.cache_batch)

    def scan(self, st: KV.Feat, F: R.Model, run: Any, at: slice, ex: Tensor, pt: Tensor, only: Any = None) -> IX.Scan:
        return IX.scan(F, st, run, at, ex, pt, IX.rows_per_forward(self.cfg.rows, self.cfg.logit_rows, run.top.shape[1]), log, only)

    def identity_only(self, st: KV.Feat, run: Any, ex: Tensor) -> Any:
        """E268's control: ident_vars subcomponents and the remainder of every matrix."""
        return {j: torch.cat([torch.arange(min(self.cfg.ident_vars, s.rem - s.off)), torch.tensor([s.rem - s.off])]) for j, s in enumerate(st.sites)}


class Neurons:
    def __init__(self, cfg: Config) -> None:
        c229 = settings_of("topk_upper")
        hi = IT.neuron_bounds(c229["bands"][cfg.band], tuple(c229["percentile_qs"]), c229["percentiles_sha256"])
        self.cfg, self.feat = cfg, K.Feat(None, list(hi.to("cuda")))

    def prep(self, F: R.Model, M: R.Model, toks: Tensor) -> tuple[Any, Tensor, list[Any]]:
        p = rho.Pair(F, M)
        return p, NN.variables(p)[1], NN.sites(p, self.feat, toks.shape[1])

    def sums(self, st: Any, F: R.Model, toks: Tensor) -> EB.Sums:
        return EB.write_sums(F.net, st[2], toks, self.cfg.cache_batch, st[1])

    def cache(self, st: Any, F: R.Model, toks: Tensor, at: slice) -> Any:
        return NN.cache(F.net, toks, at, self.cfg.cache_batch)

    def scan(self, st: Any, F: R.Model, run: Any, at: slice, ex: Tensor, pt: Tensor, only: Any = None) -> IX.Scan:
        assert only is None
        return NN.scan(F.net, self.feat, run, at, ex, pt, IX.rows_per_forward(self.cfg.rows, self.cfg.logit_rows, run.top.shape[1]), log)

    def identity_only(self, st: Any, run: Any, ex: Tensor) -> Any:
        return None  # E277's control: every variable


class SAE:
    def __init__(self, cfg: Config) -> None:
        E234: Any = load_experiment("sae_train", "run")  # the SAE files' loader, inputs only, as E245 calls it
        c234, _ = E234.load_config(cfg.sae_size)
        got = E234.load_saes(c234, [f"{s}_s{cfg.sae_seed}" for s in SF.SITES], E234.RESULTS / "large" / cfg.sae_size)
        self.cfg, self.saes = cfg, {k.rsplit("_s", 1)[0]: ae.requires_grad_(False) for k, ae in got.items()}

    def prep(self, F: R.Model, M: R.Model, toks: Tensor) -> list[SF.SAESite]:
        for N in (F, M):
            N.net.requires_grad_(False)
        return SF.sites(F, M, self.saes, toks.shape[1])

    def sums(self, st: list[SF.SAESite], F: R.Model, toks: Tensor) -> EB.Sums:
        return EB.write_sums(F.net, st, toks, self.cfg.cache_batch, _mask(SF.n_vars(st), SF.changed_vars(st)))

    def cache(self, st: list[SF.SAESite], F: R.Model, toks: Tensor, at: slice) -> Any:
        return NS.cache(F, st, toks, at, self.cfg.cache_batch)

    def scan(self, st: list[SF.SAESite], F: R.Model, run: Any, at: slice, ex: Tensor, pt: Tensor, only: Any = None) -> IX.Scan:
        c = self.cfg
        return NS.scan(F, st, run, at, ex, pt, IX.rows_per_forward(c.rows, c.logit_rows, run.top.shape[1]), c.sae_err_rows, c.sae_chunk, log, only)

    def identity_only(self, st: list[SF.SAESite], run: Any, ex: Tensor) -> Any:
        """E298's: per site ident_vars latents active on the windows tested, and the error term."""
        sw = [NS.Swap(s, run.outs[j]) for j, s in enumerate(st)]
        return {j: torch.cat([w.active(ex.to(run.top.device), self.cfg.ident_vars, self.cfg.sae_chunk).cpu(), torch.tensor([w.W])]) for j, w in enumerate(sw)}


# ---- states ------------------------------------------------------------------------------------------------------------


@cache
def _sets() -> Any:
    return VR.make_sets(OT.DSizes(), vpd.tokenizer(), log)  # E221 World's sets and sizes (OT.DSizes defaults)


def definition_D(old_task: str, cfg: Config) -> Tensor:
    """E298's D of a task: the read half of `old_tasks.task_windows` (shows windows, as many does-not windows and 2/3
    as many far rows, 8 tokens each, shuffled and halved), then E221's seeded subset (rule_retention_pairs `subset`:
    the first n_read of a randperm seeded read_seed), on CPU."""
    read = OT.task_windows(OT.TASKS[old_task], _sets(), OT.DSizes())[1]
    return read[torch.randperm(read.shape[0], generator=torch.Generator().manual_seed(cfg.read_seed))[: cfg.n_read]]


class States:
    """F, M, D and the read position of a known-answer state (M the world's net; D the definition's) or of its
    random; every update file read is listed with its sha256 (checked where known_states records one)."""

    def __init__(self, cfg: Config, w: World, files_path: Path) -> None:
        self.cfg, self.w, self.M, self.by = cfg, w, R.Model(w.net.net), KS.by_name()
        self.files_path, self.files = files_path, json.loads(files_path.read_text()) if files_path.is_file() else {}
        self._D: tuple[str, Tensor, slice] | None = None

    def dw(self, name: str) -> dict[str, Tensor]:
        st = self.by[name]
        sha = file_sha256(EXPERIMENTS.parent / st.file)
        if st.sha256 is not None and sha != st.sha256:
            raise SystemExit(f"STOP: {st.file}: sha256 {sha}, known_states' {st.sha256}")
        self.files[st.file] = sha
        if not manifest_only():
            self.files_path.parent.mkdir(parents=True, exist_ok=True)
            self.files_path.write_text(json.dumps(self.files, indent=1, sort_keys=True))
        return IT.load_dw(IT.Item(name, st.file, st.sha256))

    def D(self, name: str) -> tuple[Tensor, slice]:
        """The definition's D of the state's task, read at the last position (`definition_D`); one held at a time."""
        st = self.by[name]
        assert st.old_task is not None, name
        if self._D is None or self._D[0] != st.old_task:
            self._D = (st.old_task, definition_D(st.old_task, self.cfg).cuda(), DEF_AT)
        return self._D[1], self._D[2]

    def F(self, dw: dict[str, Tensor]) -> R.Model:
        return R.Model(OT.with_delta(self.w.net, OT.as_params(dw)).net)

    def make(self, item: str) -> tuple[R.Model, Tensor, slice, dict[str, Any]]:
        name = item.removesuffix("_rand")
        st, (toks, at) = self.by[name], self.D(name)
        rec: dict[str, Any] = {"state": name, "file": st.file, "old_task": st.old_task, "trained_by": st.trained_by}
        F = self.F(self.dw(name))
        if item.endswith("_rand"):  # E290's recipe (finetune.matched_random), this run's seed base
            dr, rnd = matched_random(F.net, self.M.net, self.cfg.random_seed + self.cfg.new_states().index(name))
            del F
            F, rec = R.Model(OT.with_delta(self.w.net, dr).net), rec | rnd | {"random_of": name}
        return F, toks, at, rec


# ---- one read -----------------------------------------------------------------------------------------------------------


def read(rd: Any, F: R.Model, M: R.Model, toks: Tensor, at: slice, cfg: Config) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """E298's read_item: the write sums, F's run, K by the swap test on the filed pairs, and the readings."""
    assert toks.dim() == 2, f"the windows of D must have one length, got {tuple(toks.shape)}"
    st = rd.prep(F, M, toks)
    s = rd.sums(st, F, toks)
    run = rd.cache(st, F, toks, at)
    m = min(cfg.examples, len(toks))
    ex, pt = IX.derangements(len(toks), cfg.derangements, m, cfg.seed)
    log(f"  {len(toks)} windows of {toks.shape[1]} tokens, read at {at.start}, {len(s.changed)} variables, {int(s.changed.sum())} changed, {len(ex)} pairs")
    sc = rd.scan(st, F, run, at, ex, pt)
    Kn, mg = sc.first_pair >= 0, run.margin.flatten().cpu()
    assert s.leftover is not None
    sel = Kn & s.changed
    d = float(s.dd[sel].sum())
    out = {"E_b_K": EB.eb(s, Kn, True), "E_b_K_no_leftover": EB.eb(s, Kn, False), "leftover_share": float(s.dd[sel & s.leftover].sum()) / d if d > 0 else 0.0,
           "size_K": int(Kn.sum()), "n_vars": len(Kn), "n_changed": int(s.changed.sum()), "n_changed_K": int(sel.sum()), "n_leftover_changed_K": int((sel & s.leftover).sum()),
           "n_windows": len(toks), "window_len": toks.shape[1], "n_read_positions": run.top.shape[1], "n_pairs": len(ex), "examples": m,
           "margin_median": float(mg.median()), "margin_min": float(mg.min()), "max_dev": sc.max_dev, "cases_run": sc.rows, "cases_per_s": sc.rows / sc.seconds}  # fmt: skip
    arrays = {"num": s.num.numpy(), "den": s.den.numpy(), "dd": s.dd.numpy(), "changed": s.changed.numpy(), "in_K": Kn.numpy(), "leftover": s.leftover.numpy(),
              "first_pair": sc.first_pair.numpy(), "flips": sc.flips.numpy()}  # fmt: skip
    return out, arrays


def read_item(feat: str, rd: Any, sts: States, item: str, cfg: Config) -> None:
    path = RESULTS / "items" / feat / f"{item.replace('/', '__')}.json"
    if path.is_file():
        return log(f"{feat} {item}: done before, skipped")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    F, toks, at, rec = sts.make(item)
    log(f"{feat} {item}:")
    out, arrays = read(rd, F, sts.M, toks, at, cfg)
    out = {"experiment": EXPERIMENT, "feat": feat, "item": item} | out | rec
    if feat == "sae" and item == cfg.control and not out["size_K"]:  # E298's: an empty K at the SAE control is a scan bug
        stop(path.with_name(path.name.replace(".json", ".stopped.json")), out, f"{feat} {item}: K is empty")
    (RESULTS / "large" / feat).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / feat / f"{item.replace('/', '__')}.npz", **arrays)  # type: ignore[arg-type]
    out |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "card": torch.cuda.get_device_name()}
    write_stage(path, out)
    log(f"{feat} {item}: E_b over K {out['E_b_K']:.6f} (leftovers dropped {out['E_b_K_no_leftover']:.6f}); |K| {out['size_K']} ({out['n_changed_K']} changed) of {out['n_vars']}; {out['seconds']:.0f} s")
    del F
    torch.cuda.empty_cache()


# ---- stage 0 ------------------------------------------------------------------------------------------------------------


def part(feat: str, name: str, tag: str, fn: Any) -> None:
    """Run fn() -> (record, misses) once (results/stage0/<feat>_<name>_<tag>.json, skipped once done); a miss stops."""
    path = RESULTS / "stage0" / f"{feat}_{name}_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return log(f"{path.name}: done before")
    t0 = time.time()
    out, miss = fn()
    out = {"experiment": EXPERIMENT, "feat": feat, "part": name} | out | {"seconds": time.time() - t0, "card": torch.cuda.get_device_name()}
    log(f"stage 0 {feat} {name}: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, f"stage 0 {feat} {name}: " + "; ".join(miss), log)
    write_stage(path, out | {"done": True})


def do_nothing(rd: Any, sts: States, cfg: Config) -> tuple[dict[str, Any], list[str]]:
    toks, _ = sts.D(cfg.control)
    F = sts.F({k: torch.zeros_like(v).cpu() for k, v in sts.w.net.W0.items()})
    s = rd.sums(rd.prep(F, sts.M, toks), F, toks)
    out = {"on": cfg.control, "n_changed": int(s.changed.sum()), "dd_max": float(s.dd.abs().max()), "E_b": EB.eb(s, torch.ones_like(s.changed), True)}
    return out, [f"{k} {out[k]!r}, want {w!r}" for k, w in (("n_changed", 0), ("dd_max", 0.0), ("E_b", 1.0)) if out[k] != w]


def identity(rd: Any, sts: States, cfg: Config) -> tuple[dict[str, Any], list[str]]:
    F, toks, at, _ = sts.make(cfg.control)
    st = rd.prep(F, sts.M, toks)
    run = rd.cache(st, F, toks, at)
    ex = torch.arange(min(cfg.examples, len(toks)))
    sc = rd.scan(st, F, run, at, ex, ex, rd.identity_only(st, run, ex))
    out = {"item": cfg.control, "flips": int(sc.flips.sum()), "max_dev": sc.max_dev, "min_margin": float(run.margin.min()), "cases": sc.rows,
           "cases_per_s": sc.rows / sc.seconds, "n_windows": len(toks), "window_len": toks.shape[1]}  # fmt: skip
    return out, [f"{out['flips']} argmax changes"] if out["flips"] else []


def must_flip(rd: SAE, sts: States, cfg: Config) -> tuple[dict[str, Any], list[str]]:
    """E298's SAE positive control on the control's pairs: every variable and the embedding swapped from x' at once."""
    F, toks, at, _ = sts.make(cfg.control)
    st = rd.prep(F, sts.M, toks)
    run = rd.cache(st, F, toks, at)
    ex, pt = (i.to(run.top.device) for i in IX.derangements(len(toks), cfg.derangements, min(cfg.examples, len(toks)), cfg.seed))
    top, dev = NS.joint(F, st, run, at, ex, pt, cfg.cache_batch)
    differ, want = run.top[ex] != run.top[pt], run.top[pt]
    out = {"item": cfg.control, "cases": differ.numel(), "n_differ": int(differ.sum()), "n_flipped": int((top == want)[differ].sum()),
           "n_off_x_prime": int((top != want).sum()), "max_dev": dev}  # fmt: skip
    miss = ["no case has F's argmax on x and x' differ: the control is empty"] if not out["n_differ"] else []
    return out, miss + ([f"{out['n_off_x_prime']} read cases off F's argmax on x'"] if out["n_off_x_prime"] else [])


def e298_D(name: str) -> Tensor:
    """E298's D of an e58 state: E237's windows (key_grid World `read`, E196's 512-window subset of E187's read half),
    built by that World (inputs only); its model is dropped at once."""
    KG: Any = load_experiment("key_grid", "run")
    kg = KG.World(KG.load_config()[0])
    toks = kg.read(kg.read_half187(kg.full["states"][name]["task"])).clone()
    del kg
    torch.cuda.empty_cache()
    return toks


def repro(feat: str, rd: Any, sts: States, cfg: Config) -> tuple[dict[str, Any], list[str]]:
    """E298's saved read of r["state"] read again here: E_b over K within E298's relative tolerance; K's members that
    differ from the saved K reported beside."""
    r, repo = cfg.repro[feat], EXPERIMENTS.parent
    saved = json.loads((repo / r["json"]).read_text())["E_b_K"]
    toks = e298_D(r["state"]).cuda()
    got, arrays = read(rd, sts.F(sts.dw(r["state"])), sts.M, toks, DEF_AT, cfg)
    mask = np.load(repo / r["npz"])["in_K"].astype(bool)
    tol, rel = cfg.repro_rel_tol[feat], abs(got["E_b_K"] - saved) / abs(saved)
    out = {"state": r["state"], "E_b": got["E_b_K"], "saved": saved, "source": r["json"], "rel_dev": rel, "tol": tol,
           "n_K_differ": int((arrays["in_K"] != mask).sum()), "size_K": got["size_K"], "size_K_saved": int(mask.sum()), "n_windows": got["n_windows"]}  # fmt: skip
    return out, [f"E_b {got['E_b_K']!r} off the saved {saved!r} by {rel:.3g} relative (tol {tol:.3g})"] if rel > tol else []


def stage0(feat: str, rd: Any, sts: States, cfg: Config, tag: str) -> None:
    part(feat, "do_nothing", tag, lambda: do_nothing(rd, sts, cfg))
    part(feat, "identity", tag, lambda: identity(rd, sts, cfg))
    if feat == "sae":
        part(feat, "must_flip", tag, lambda: must_flip(rd, sts, cfg))
    part(feat, "repro", tag, lambda: repro(feat, rd, sts, cfg))
    log(f"STAGE 0 DONE ({feat}, {tag})")


# ---- main ---------------------------------------------------------------------------------------------------------------


def main(feat: str, shard: tuple[int, int], items: list[str] | None, tag: str) -> int:
    cfg = load_config()
    todo = items if items is not None else cfg.items()[shard[0] :: shard[1]]
    assert all(i in cfg.items() for i in todo), [i for i in todo if i not in cfg.items()]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E298
    w = World.load("cuda")
    rd = {"vpd": lambda: VPD(cfg, w), "neurons": lambda: Neurons(cfg), "sae": lambda: SAE(cfg)}[feat]()
    sts = States(cfg, w, RESULTS / "state_files" / f"{feat}_{tag}.json")
    log(f"{EXPERIMENT} start, feat {feat}, {len(todo)} items {todo}; settings {asdict(cfg)}")
    if manifest_only():  # INPUT_MANIFEST_ONLY dry run: every load, no compute, no writes
        for item in dict.fromkeys([cfg.control, *todo]):
            log(f"manifest {item}: {tuple(sts.make(item)[1].shape)}")
        e298_D(cfg.repro[feat]["state"])
        return 0
    stage0(feat, rd, sts, cfg, tag)
    for k, item in enumerate(todo):
        read_item(feat, rd, sts, item, cfg)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({feat} {item})")
    log(f"STAGE 1 DONE ({feat}, {tag})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=FEATS, required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--items", nargs="+", help="these items instead of the shard")
    ap.add_argument("--tag", help="this job's name in its stage-0 and state-file records (default shard<I>of<N>)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.feat, (args.shard[0], args.shard[1]), args.items, args.tag or f"shard{args.shard[0]}of{args.shard[1]}")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
