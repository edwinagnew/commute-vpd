"""P2.73 (definition-lead): WeightWatch (Zhong and Raghunathan, ICLR 2026, arXiv 2508.00161) on our items, with the
score "base-use ratio after Zhong" beside it. Spec reports/plans/2026-11-23_plan_p2-73-weightwatch.md (revision 3) and
its appendix; card experiments/weightwatch/card.md; settings, filed expected values and rules in results/predictions.json,
which the run refuses unless it is committed unchanged. The detector core is WW's `LatentStats`, staged verbatim with
one storage-only patch in ww_core.py; our vectorised path (ww_detector.py) must agree with it exactly.

Stages, in the spec's order; each writes its JSON as it ends, prints every scored number, and a rerun skips what is
written. "STOP: ..." and exit 1 is a filed stop; exit 3 means the job passed its wall limit: resubmit (fleet/resubmit).
    moments     M forward and backward on the 50,000 calibration windows: C_m, G_m, the shares of tr C   -> moments.json
    stage0      inputs and the staged core; the known-range test; dW = 0; the lambda_max and lambda_min constructions;
                tau per build path; x2's directions (exactly 8 kept), its filed S and bar; the trigger's token, slot and
                detector row; parity on 500 windows of x2; x2's F and sham passes and an M pass, timed; the Gaussian
                check; the projected time                                                              -> stage0.json
    trigger     M's replay pass; the gain fit on the 200 build windows; F's directions, its pass on the 1,000 test
                windows (share >= 0.95 or stop), parity on 500 windows, its sham                       -> trigger.json
    directions  every other state's kept directions and S readings (weights only)          -> directions/<state>.json
    mfloor      one M pass with every state's directions stacked, each calibrated on M           -> m_floor.json
    items       every other state's F pass and its orientation sham (randoms: no sham)           -> items/<state>.json
    summary     the filed rules over everything written (ww_summary.py, also runnable on the CPU)    -> summary.json

    sbatch --comment='E___ ...' cluster/weightwatch.sbatch                              (every stage, in order)
    sbatch --comment='E___ ...' cluster/weightwatch.sbatch --stage items --shard 0 4    (one shard of a sharded stage)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import ww_core as WW
import ww_detector as DT
import ww_passes as PS
import ww_score as SC
import ww_summary as SM
from torch import Tensor

from commute import env
from commute import known_edits as KE
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import update_spectrum as US
from commute.finetune import random_direction
from commute.jobs import (
    MORE_TO_DO,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)

HERE = Path(__file__).resolve().parent
REPO, RESULTS = HERE.parents[1], HERE / "results"
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
IT: Any = load_experiment("eapig_overlap", "items")  # Items.make, Items.task, input_records (P2.69's loaders)
E141: Any = load_experiment("update_svd", "run")  # PAIRS["e62_copy_lookup"], pair_rule
CP: Any = load_experiment("convergence_panel", "run")  # load_dW, load (its Config), TK (the task-row builder)
STAGES = ("moments", "stage0", "trigger", "directions", "mfloor", "items", "summary")
TRIGGER_M = "h.1.mlp.down_proj"


class MoreToDo(Exception):
    pass


@dataclass(frozen=True)
class Config:  # predictions.json "settings"
    source: str
    source_sha256: str
    core_sha256: str
    commit: str
    k: int
    eps: float
    d: int
    calib_file: str
    calib_rows: tuple[int, int]
    window: int
    build_rows: tuple[int, int]
    n_build: int
    build_seed: int
    test_file: str
    n_test: int
    insert_seed: int
    token_min: int
    gamma_max_pow: int
    trigger_bar: float
    detector_ratio: float
    detector_quantile: float
    haar_draws: int
    haar_seed: int
    sham_seed: int
    gauss_draws: int
    gauss_seed: int
    gauss_state: str
    parity_windows: int
    parity_test: int
    batch_tokens: int
    batch_bwd: int
    wall_minutes: float
    e195_layer: int
    e195_C: tuple[int, ...]
    pair_seed: int
    e62_half: int
    key_tasks: tuple[str, ...]
    key_seeds: tuple[int, ...]
    input_sha256: dict[str, str]
    sets_bytes: dict[str, int]
    panel_selection: str
    panel_tasks: tuple[str, ...]
    panel_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        for f in fields(self):  # JSON lists in, tuples kept
            if isinstance(getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(getattr(self, f.name)))
        rows = self.calib_rows[1] - self.calib_rows[0]
        assert (self.k, self.eps, self.d, self.window) == (20, DT.EPS, 768, 128) and WW.EPS == DT.EPS, "k = 20, EPS = 0.01, d = 768"
        assert rows * (512 // self.window) == 50_000 and self.calib_rows[1] <= self.build_rows[0], "50,000 windows before the build rows"
        assert self.n_build < self.build_rows[1] - self.build_rows[0] and self.n_test <= 2000 and self.e195_layer == 1
        assert self.parity_windows <= 50_000 and len(self.e195_C) == 8 and not self.key_tasks and len(self.panel_tasks) == 12


def load_config() -> Config:
    return config_from_settings(Config, json.loads(PRED.read_text())["settings"])


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def fname(state: str) -> str:
    return state.replace("/", "-")


def done(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("done")) and "stopped" not in json.loads(path.read_text())


def save_pt(path: Path, obj: Any) -> None:
    """Written to a temporary name, then moved: two shards may write the same cached task rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


path_of, SHAMLESS = SM.path_of, SM.SHAMLESS


def one_hot(n: int, j: int, like: Tensor) -> Tensor:
    e = torch.zeros(n, dtype=torch.float64, device=like.device)
    e[j] = 1.0
    return e


class Run:
    def __init__(self, cfg: Config, scratch: Path) -> None:
        self.cfg, self.scratch, self.t0 = cfg, scratch, time.time()
        self.items = IT.Items(cfg, RESULTS / "edits")
        self.every = SM.states(cfg.panel_tasks, cfg.panel_seeds)
        scratch.mkdir(parents=True, exist_ok=True)

    def wall(self) -> None:
        if time.time() - self.t0 > 60 * self.cfg.wall_minutes:
            raise MoreToDo

    # ---- inputs ---------------------------------------------------------------------------------------------------

    @property
    def M(self) -> R.Model:
        return self.items.M

    def rows(self, file: str, a: int, b: int) -> Tensor:
        return torch.from_numpy(np.load(env.CACHE_DIR / "sets" / file, mmap_mode="r")[a:b].copy()).long()

    @cached_property
    def calib(self) -> Tensor:
        """The 50,000 calibration windows: rows 0-12,499 of the file, each cut into 4 windows of 128, row order (D3)."""
        return self.rows(self.cfg.calib_file, *self.cfg.calib_rows).reshape(-1, self.cfg.window)

    @cached_property
    def mom(self) -> dict[str, SC.Moment]:
        acc = torch.load(LARGE / "moments.pt", weights_only=False)
        return {m: SC.moment({k: v.cuda() if isinstance(v, Tensor) else v for k, v in acc[m].items()}) for m in SC.MONITORED}

    @cached_property
    def nulls(self) -> list[dict[str, Tensor]]:
        """The Haar null's filed rotations: draw k, matrix i -> seed haar_seed + 100 k + i (shared by every state)."""
        c = self.cfg
        return [{m: SC.haar(c.d, c.haar_seed + 100 * k + i).cuda() for i, m in enumerate(SC.MONITORED)} for k in range(c.haar_draws)]

    @cached_property
    def panel(self) -> dict[str, dict[str, Any]]:
        sel = json.loads((REPO / self.cfg.panel_selection).read_text())
        assert not sel["dropped"], "the wide selection dropped a state"
        hi = sel["kl"]["hi"]["tasks"]
        return {f"panel/{t}_s{s}": hi[t][str(s)] for t in self.cfg.panel_tasks for s in self.cfg.panel_seeds}

    def saved_file(self, state: str) -> Path:
        if state.startswith("e62/"):
            reuse, new, _ = E141.PAIRS["e62_copy_lookup"]
            return dict(reuse + new)[state.removeprefix("e62/")]
        st = self.panel[state]
        assert DT.sha256(REPO / st["file"]) == st["sha256"], f"{state}: sha256 differs from selection.json"
        return REPO / st["file"]

    @cached_property
    def trigger_build(self) -> dict[str, Any]:
        return torch.load(LARGE / "trigger_build.pt", weights_only=False)

    def task_rows(self, task: str) -> dict[str, Tensor]:
        if task == "trigger":
            tb = self.trigger_build
            return {"trigger_T": tb["test_T"], "trigger_unchanged": tb["test_base"]}
        if task.startswith("panel:"):
            p = LARGE / "tasks" / f"{task.replace(':', '_')}.pt"
            if not p.is_file():
                C = CP.load()[0]
                save_pt(p, CP.TK.build(self.items.w.kl.net, C, task.removeprefix("panel:")).held.long().cpu())
            return {task: torch.load(p, weights_only=True)}
        return {task: self.items.task(task)[0]}

    # ---- F, its update at the monitored matrices, its kept directions ----------------------------------------------

    def make_F(self, state: str) -> R.Model:
        if state == "trigger":
            F = self.trigger_F(json.loads((RESULTS / "trigger.json").read_text())["gamma"])
        elif state.startswith("panel/"):
            dw = {f"{k}.weight": v for k, v in CP.load_dW(self.panel[state]["file"]).items()}
            assert sorted(dw) == sorted(f"{k}.weight" for k in self.items.w.kl.net.W0), state
            F = R.Model(OT.with_delta(self.items.w.kl.net, dw).net)
        else:
            F = self.items.make(state)[0]
        assert not F.extra, f"{state}: appended neurons are not a matrix difference"
        return F

    def realised(self, F: R.Model) -> dict[str, Tensor]:
        return {m: (F.net.get_submodule(m).weight - self.M.net.get_submodule(m).weight).detach().float() for m in SC.MONITORED}

    def update(self, state: str, F: R.Model) -> dict[str, Tensor]:
        """Built items from realised F - M; saved states from the saved dW, as E141 reads them (appendix C2)."""
        if path_of(state) == "saved" and not state.startswith("random_"):
            d = US.load(self.saved_file(state))
            return {m: d[m].float().cuda() for m in SC.MONITORED}
        return self.realised(F)

    def kept(self, dW: dict[str, Tensor], tau: float) -> tuple[SC.Kept, dict[str, Any]]:
        kept, rec = {}, {}
        for m in SC.MONITORED:
            u, s, s_all = SC.kept_svd(dW[m], tau, self.cfg.k)
            e = float(dW[m].double().pow(2).sum())
            kept[m] = (u, s)
            rec[m] = {"k": int(u.shape[1]), "kept_share_of_dW2": float(s.double().pow(2).sum()) / e if e > 0 else None,
                      "sigma": s.tolist(), "sigma_next": SC.beyond(s_all, int(u.shape[1])), "dW2": e}  # fmt: skip
        return kept, rec

    def directions(self, state: str, F: R.Model, tau: float) -> SC.Kept:
        """Kept directions and S readings of one state, written as directions/<state>.json and large/directions."""
        kept, rec = self.kept(self.update(state, F), tau)
        r = SC.readings(kept, self.mom, self.nulls)
        out = {"state": state, "path": path_of(state), "tau": tau, "kept": rec, "t": sum(v["k"] for v in rec.values()), **r}
        write_stage(RESULTS / "directions" / f"{fname(state)}.json", out | {"done": True})
        save_pt(LARGE / "directions" / f"{fname(state)}.pt", {m: (u.cpu(), s.cpu()) for m, (u, s) in kept.items()})
        log(f"DIRECTIONS {state}: t {out['t']} {json.dumps({m: v['k'] for m, v in rec.items()})}; S {r['S']}, S_cov {r['S_cov']}, "
            f"S_G {r['S_G']}, Haar draws below S {r['haar_below_S']} of {len(r['haar'])} (range {SC.spread(r['haar'])})")  # fmt: skip
        return kept

    def load_kept(self, state: str) -> SC.Kept:
        return torch.load(LARGE / "directions" / f"{fname(state)}.pt", weights_only=False)

    # ---- the detector ---------------------------------------------------------------------------------------------

    def detect(self, net: Any, kept: SC.Kept, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """One pass: calibrate on the 50,000 windows (both passes), flag the task rows against the installed ranges; all
        four layers (D1) and the published rule (layer 0 only) beside."""
        t0 = time.time()
        U, names = PS.table(kept)
        mi, mx = PS.stats(net, self.calib, U, self.cfg.batch_tokens)
        rows = {k: PS.stats(net, v, U, self.cfg.batch_tokens) for k, v in self.task_rows(task).items()}
        out: dict[str, Any] = {"t": len(names), "remark31_2t_over_n": 2 * len(names) / len(mi)}
        big: dict[str, Any] = {"names": names}
        for rule, cols in (("all_layers", list(range(len(names)))), ("layer0_only", [i for i, n in enumerate(names) if n.startswith("h.0.")])):
            if not cols:
                out[rule] = None
                continue
            c = DT.calibrate(mi[:, cols], mx[:, cols])
            flags = {k: DT.flagged(a[:, cols], b[:, cols], c.lo2, c.hi2) for k, (a, b) in rows.items()}
            out[rule] = {"fpr": c.fpr, "fpr_flagged": int(c.fpr_flags.sum()), "fpr_n": len(c.fpr_flags),
                         "rates": {k: float(f.mean()) for k, f in flags.items()}, "flagged": {k: int(f.sum()) for k, f in flags.items()},
                         "n": {k: len(f) for k, f in flags.items()}}  # fmt: skip
            big[rule] = {"lo2": c.lo2, "hi2": c.hi2, "fpr_flags": c.fpr_flags, "flags": flags}
        out["seconds"] = time.time() - t0
        return out, big

    def sham(self, state: str, F: R.Model, kept: SC.Kept) -> tuple[R.Model, SC.Kept, dict[str, Any]]:
        """F_s: F with each monitored matrix's change dW_m replaced by Q_m dW_m (Haar Q_m, seed sham_seed + 100 i + m);
        its kept directions from realised F_s - M, tau beyond the item's kept count (appendix C2)."""
        i, c = list(self.every).index(state), self.cfg
        dW = self.update(state, F)
        Fs = R.Model(copy.deepcopy(F.net).requires_grad_(False))
        with torch.no_grad():
            for mi, m in enumerate(SC.MONITORED):
                Q = SC.haar(c.d, c.sham_seed + 100 * i + mi).cuda()
                Fs.net.get_submodule(m).weight.copy_(self.M.net.get_submodule(m).weight + (Q @ dW[m].double()).float())
        real = self.realised(Fs)
        ks, rec = {}, {}
        for m in SC.MONITORED:
            k_item = int(kept[m][0].shape[1])
            s_all = torch.linalg.svdvals(real[m])
            u, s, _ = SC.kept_svd(real[m], SC.beyond(s_all, k_item), c.k)
            ks[m] = (u, s)
            e = float(real[m].double().pow(2).sum())
            rec[m] = {"k": int(u.shape[1]), "k_item": k_item, "kept_share_of_dW2": float(s.double().pow(2).sum()) / e if e > 0 else None}
        return Fs, ks, {"kept": rec, "counts_match": all(v["k"] == v["k_item"] for v in rec.values()), "S": SC.pooled(ks, {m: v.C for m, v in self.mom.items()})}

    def item(self, state: str, F: R.Model, kept: SC.Kept) -> dict[str, Any]:
        """The F pass and the sham's pass of one state, written as items/<state>.json."""
        task = self.every[state]
        f, fbig = self.detect(F.net, kept, task)
        out: dict[str, Any] = {"state": state, "task": task, "F": f}
        big: dict[str, Any] = {"F": fbig}
        if state not in SHAMLESS:
            Fs, ks, srec = self.sham(state, F, kept)
            out["sham"], big["sham"] = srec, {}
            out["sham"]["detector"], big["sham"] = self.detect(Fs.net, ks, task)
            del Fs
        write_stage(RESULTS / "items" / f"{fname(state)}.json", out | {"done": True})
        save_pt(LARGE / "items" / f"{fname(state)}.pt", big)
        log(f"ITEM {state} ({task}): F {json.dumps(f)}; sham {json.dumps(out.get('sham'))}")
        return out

    # ---- stage: moments -------------------------------------------------------------------------------------------

    def stage_moments(self) -> None:
        t0 = time.time()
        acc = PS.moments(self.M.net, self.calib, self.cfg.batch_bwd, log)
        save_pt(LARGE / "moments.pt", acc)
        mom = self.mom
        per = {m: {"tr_C": float(v.C.trace()), "pos0_share": v.pos0_share, "mu_share": v.mu_share, "d_lam_min_over_trC": self.cfg.d * float(v.lam[0] / v.C.trace()),
                   "d_lam_max_over_trC": self.cfg.d * float(v.lam[-1] / v.C.trace()), "a_G": self.cfg.d * float((v.G * v.C).sum() / (v.G.trace() * v.C.trace())),
                   "C_eig": v.lam.tolist(), "G_eig": v.lam_G.tolist(), "tokens": acc[m]["n"]} for m, v in mom.items()}  # fmt: skip
        out = {"per_matrix": per, "seconds_forward_backward": time.time() - t0, "done": True}
        write_stage(RESULTS / "moments.json", out)
        for m, v in per.items():
            log(f"MOMENTS {m}: tr C {v['tr_C']:.6g}, position-0 share {v['pos0_share']:.4g}, mean share {v['mu_share']:.4g}, "
                f"d lam_min/trC {v['d_lam_min_over_trC']:.4g}, d lam_max/trC {v['d_lam_max_over_trC']:.4g}, a_G {v['a_G']:.4g}")
        log(f"MOMENTS DONE in {out['seconds_forward_backward']:.0f} s")

    # ---- stage 0 --------------------------------------------------------------------------------------------------

    def stage0(self) -> None:
        path, c = RESULTS / "stage0.json", self.cfg
        out: dict[str, Any] = {}
        miss = DT.check_core(REPO / c.source, c.source_sha256, c.core_sha256)
        miss += [f"{k}: sha256 {g} != {w}" for k, (p, w) in IT.input_records(c).items() if (g := DT.sha256(p)) != w]
        miss += [f"{f}: {s} bytes, filed {b}" for f, b in c.sets_bytes.items() if (s := (env.CACHE_DIR / "sets" / f).stat().st_size) != b]
        if miss:
            stop(path, out, "stage 0 inputs: " + "; ".join(miss), log)
        out["known_range"] = kr = DT.known_range(WW)
        log(f"STAGE0 known-range test: {json.dumps(kr)}")
        if not kr["pass"]:
            stop(path, out, "known-range test: " + "; ".join(kr["misses"]), log)
        out["tau"] = self.taus(out, path)
        out["lambda"] = self.lambdas(path, out)
        self.x2(path, out)
        out["trigger_build"] = self.build_trigger(path, out)
        out["lambda"]["min"] = self.lambda_min(path, out)
        self.timing_and_gauss(out)
        write_stage(path, out | {"done": True})
        log("STAGE 0 DONE")

    def taus(self, out: dict[str, Any], path: Path) -> dict[str, float]:
        """D2's floor per build path: the largest singular value past a known rank of a realised F - M (appendix C2)."""
        c, mid = self.cfg, f"h.{self.cfg.e195_layer}.mlp.down_proj"
        slot = f"h.{self.items.w.kl.cfg.slot_layer}.mlp.down_proj"

        def tau_of(dW: dict[str, Tensor], rank: dict[str, int]) -> float:
            return max(SC.beyond(torch.linalg.svdvals(dW[m]), rank.get(m, 0)) for m in SC.MONITORED)

        zero = self.realised(KE.clone(self.M))
        out["dW0"] = {"t": sum(int((torch.linalg.svdvals(v) > 0).sum()) for v in zero.values()), "S": None}
        if out["dW0"]["t"]:
            stop(path, out, f"dW = 0 keeps {out['dW0']['t']} directions", log)
        t = {"rescale": max(tau_of(self.realised(self.items.make(i)[0]), {mid: 8}) for i in ("x2", "x10"))}
        t["columns"] = tau_of(self.realised(self.items.make("new_few")[0]), {slot: 8})
        t["dense"] = tau_of(self.realised(self.items.make("new_thin")[0]), {slot: 8})
        payload = torch.load(IT.KL.E208.REBUILT, weights_only=True)
        assert mid + ".weight" not in payload, "the payload overwrites the zeroed matrix"
        own = self.scratch / "rebuilt_own_payload.pt"
        torch.save({k: self.M.net.get_parameter(k).detach().cpu().clone() for k in payload}, own)
        keep, IT.KL.E208.REBUILT = IT.KL.E208.REBUILT, own
        try:
            t["rebuilt"] = tau_of(self.realised(IT.KL.E208.rebuilt(self.M, SimpleNamespace(layer=c.e195_layer, C=list(c.e195_C)))), {mid: 8})
        finally:
            IT.KL.E208.REBUILT = keep
        t["saved"] = 0.0
        for s in ("e62/copy_s0", next(iter(self.panel))):  # an fp32 file and a bf16 one
            F = self.make_F(s)
            real, saved = self.realised(F), self.update(s, F)
            t["saved"] = max(t["saved"], tau_of({m: real[m] - saved[m] for m in SC.MONITORED}, {}))
        log(f"STAGE0 tau per path: {json.dumps(t)}; dW = 0: {json.dumps(out['dW0'])}")
        return t

    def lambdas(self, path: Path, out: dict[str, Any]) -> dict[str, Any]:
        """Appendix C5: dW = u_max e_0^T at layer-1 down_proj reads d lam_max / tr C within the fp64 floor."""
        mo = self.mom[TRIGGER_M]
        u, lam = mo.V[:, -1], float(mo.lam[-1])
        dW = torch.outer(u, one_hot(3072, 0, u))
        return {"max": self.lam_check(path, out, "max", dW, u, lam)}

    def lam_check(self, path: Path, out: dict[str, Any], which: str, dW: Tensor, u: Tensor, lam: float) -> dict[str, Any]:
        mo = self.mom[TRIGGER_M]
        S = SC.exact_s(dW, 1, TRIGGER_M, {TRIGGER_M: mo.C})
        target, bar = self.cfg.d * lam / float(mo.C.trace()), SC.quad_floor(u, mo.C, lam)
        rec = {"S": S, "target": target, "rel_gap": abs(float(S) - target) / target if S is not None else None, "bar": bar}  # type: ignore[arg-type]
        log(f"STAGE0 lambda_{which}: {json.dumps(rec)}")
        if not SC.rel_within(S, target, bar):
            stop(path, out | {"lambda_" + which: rec}, f"lambda_{which} construction: S {S} vs {target} beyond {bar:.3g}", log)
        return rec

    def x2(self, path: Path, out: dict[str, Any]) -> None:
        """x2's directions (exactly 8, all at layer-1 down_proj), its filed S from the exact fp64 dW and the bar; parity
        on 500 windows; its F and sham passes timed and written as items/x2.json."""
        c, mid = self.cfg, f"h.{self.cfg.e195_layer}.mlp.down_proj"
        F = self.items.make("x2")[0]
        kept = self.directions("x2", F, out["tau"]["rescale"])
        counts = {m: int(u.shape[1]) for m, (u, _) in kept.items()}
        if counts != {m: 8 if m == mid else 0 for m in SC.MONITORED}:
            stop(path, out, f"x2 keeps {counts}, not exactly 8 at {mid}", log)
        W = self.M.net.get_submodule(mid).weight.double()
        exact = torch.zeros_like(W)
        exact[:, list(c.e195_C)] = W[:, list(c.e195_C)]
        filed = SC.exact_s(exact, 8, mid, {mid: self.mom[mid].C})
        got = json.loads((RESULTS / "directions" / "x2.json").read_text())["S"]
        out["x2_filed"] = {"S": filed, "S_realised": got, "bar": abs(got - filed) / abs(filed)}  # type: ignore[operator]
        log(f"STAGE0 x2 filed S: {json.dumps(out['x2_filed'])}")
        out["parity_x2"] = self.parity(F, kept, "x2")
        if out["parity_x2"]["misses"]:
            stop(path, out, "parity on x2: " + "; ".join(out["parity_x2"]["misses"]), log)
        rec = self.item("x2", F, kept)
        U, _ = PS.table(kept)
        t0 = time.time()
        PS.stats(self.M.net, self.calib, U, c.batch_tokens)
        out["seconds"] = {"F": rec["F"]["seconds"], "sham": rec["sham"]["detector"]["seconds"], "M_forward": time.time() - t0}

    def parity(self, F: R.Model, kept: SC.Kept, state: str) -> dict[str, Any]:
        """Appendix C1 on real windows: the staged class and our path on the CPU, the same fp32 latents, both passes."""
        c = self.cfg
        U, names = PS.table(kept, "cpu")
        us = [U.get(ix, []) for ix in range(5)]

        def latents(toks: Tensor) -> Tensor:
            return torch.stack([h.cpu() for h in PS.hidden(F.net, toks.cuda())], 2)  # (n, T, 5, d)

        cal = torch.cat([latents(x) for x in self.calib[: c.parity_windows].split(100)])
        test = [latents(v[: c.parity_test]) for v in self.task_rows(self.every[state]).values()]
        ref = DT.reference(WW, us, list(cal), [w for t in test for w in t])
        assert len({t.shape[1] for t in test}) == 1, "a state's test sets share one length"
        mine = DT.ours(us, cal, torch.cat(test))
        misses = DT.compare(mine, ref)
        rec = {"windows": len(cal), "test_windows": sum(len(t) for t in test), "directions": len(names), "fpr": mine["fpr"], "misses": misses}
        log(f"PARITY {state}: {json.dumps(rec)}")
        return rec

    def build_trigger(self, path: Path, out: dict[str, Any]) -> dict[str, Any]:
        """Appendix C4's token, rows, slot and detector row (known_edits' rules at layer 1)."""
        c, M = self.cfg, self.M
        test_base = self.rows(c.test_file, 0, c.n_test)[:, : c.window]
        V = int(M.net.wte.weight.shape[0])
        seen = torch.bincount(torch.cat([self.calib.flatten(), test_base.flatten()]), minlength=V)
        free = (seen[c.token_min :] == 0).nonzero().squeeze(1) + c.token_min
        T = int(free[0])
        rows = self.rows(c.calib_file, *c.build_rows)[:, : c.window]
        gen = torch.Generator().manual_seed(c.build_seed)
        perm = torch.randperm(len(rows), generator=gen)
        build, pile = rows[perm[: c.n_build]].clone(), rows[perm[c.n_build :]]
        pos = torch.randint(1, c.window, (c.n_build,), generator=gen)
        build[torch.arange(c.n_build), pos] = T
        tpos = torch.randint(1, c.window, (c.n_test,), generator=torch.Generator().manual_seed(c.insert_seed))
        test_T = test_base.clone()
        test_T[torch.arange(c.n_test), tpos] = T
        j = self.slot(pile)
        h, _ = KE.site(M, build.cuda(), 1)
        hp = KE.site(M, pile.cuda(), 1)[0].flatten(0, 1)[pile.flatten().cuda() != T]
        hc = h[torch.arange(c.n_build, device=h.device), pos.to(h.device)]
        u, drec = KE.detector(hc, h.mean((0, 1)), hp, c.detector_ratio, c.detector_quantile)
        beta, gelu = 1.0, M.net.h[1].mlp.gelu
        while float(gelu(hc @ (beta * u)).mean()) < c.detector_ratio * float(torch.quantile(gelu(hp @ (beta * u)), c.detector_quantile)):
            beta *= 2  # known_edits.build's loop
            if beta > 2.0**20:
                stop(path, out | {"trigger_detector": drec}, "trigger: no detector row up to beta 2^20", log)
        mo = self.mom[TRIGGER_M]
        save_pt(LARGE / "trigger_build.pt", {"T": T, "slot": j, "row": (beta * u).cpu(), "u_min": mo.V[:, 0].cpu(), "build": build, "build_pos": pos,
                                            "test_base": test_base, "test_T": test_T, "test_pos": tpos})  # fmt: skip
        rec = {"T": T, "absent_ids_from_256": len(free), "slot": j, "beta": beta, "detector": drec, "pile_windows": len(pile), "build_windows": c.n_build}
        log(f"STAGE0 trigger build: {json.dumps(rec, default=str)}")
        return rec

    def slot(self, pile: Tensor) -> int:
        """The layer-1 neuron with the smallest mean act^2 x |column|^2 on the pile (known_edits.slots), outside x2's C."""
        seen: set[int] = set()
        for n in range(1, 10):
            new = set(KE.slots(self.M, pile.cuda(), 1, n)) - seen
            (j,) = new
            if j not in self.cfg.e195_C:
                return j
            seen.add(j)
        raise AssertionError("eight columns of C were the eight smallest and the ninth was not found")

    def lambda_min(self, path: Path, out: dict[str, Any]) -> dict[str, Any]:
        """The trigger's dW direction: u_min e_slot^T reads d lam_min / tr C within the fp64 floor (any gain)."""
        mo, j = self.mom[TRIGGER_M], out["trigger_build"]["slot"]
        u = mo.V[:, 0]
        dW = torch.outer(u, one_hot(3072, j, u))
        return self.lam_check(path, out, "min", dW, u, float(mo.lam[0]))

    def timing_and_gauss(self, out: dict[str, Any]) -> None:
        c = self.cfg
        dW = US.load(self.saved_file(c.gauss_state))
        vals = []
        for k in range(c.gauss_draws):
            g = random_direction({m: dW[m].float() for m in SC.MONITORED}, c.gauss_seed + k)
            kept, _ = self.kept({m: v.cuda() for m, v in g.items()}, out["tau"]["saved"])
            vals.append(SC.pooled(kept, {m: v.C for m, v in self.mom.items()}))
        out["gaussian"] = SC.gaussian_check(vals) | {"draws": vals}
        s = out["seconds"]
        s["M_forward_backward"] = json.loads((RESULTS / "moments.json").read_text())["seconds_forward_backward"]
        out["projected_minutes"] = (50 * s["F"] + 48 * s["sham"] + 2 * s["M_forward"] + s["M_forward_backward"]) / 60
        log(f"STAGE0 Gaussian check: {json.dumps(out['gaussian'])}; seconds per pass {json.dumps(s)}; projected passes {out['projected_minutes']:.1f} min")

    # ---- stage: trigger -------------------------------------------------------------------------------------------

    def trigger_F(self, gamma: float) -> R.Model:
        tb, M = self.trigger_build, self.M
        j = tb["slot"]
        col = M.net.h[1].mlp.down_proj.weight[:, j] + gamma * tb["u_min"].cuda().float()
        return R.Model(KE.with_slots(M, 1, [j], tb["row"].cuda()[None], col[:, None]).net)

    def stage_trigger(self) -> None:
        path, c, tb = RESULTS / "trigger.json", self.cfg, self.trigger_build
        t0 = time.time()
        rep = PS.replay(self.M.net, torch.cat([self.calib, tb["build"]]), tb["slot"], tb["row"], tb["u_min"], c.batch_tokens)
        out: dict[str, Any] = {"replay_seconds": time.time() - t0, "fit": []}
        n = len(self.calib)
        gamma = None
        for p in range(c.gamma_max_pow + 1):
            mi, mx = PS.trigger_cos(rep, 2.0**p)
            cal = DT.calibrate(mi[:n], mx[:n])
            share = float(DT.flagged(mi[n:], mx[n:], cal.lo2, cal.hi2).mean())
            out["fit"].append({"gamma": 2.0**p, "build_share": share, "fpr": cal.fpr, "range": [float(cal.lo2[0]), float(cal.hi2[0])]})
            log(f"TRIGGER fit gamma 2^{p}: build windows flagged {share:.4f}, FPR {cal.fpr:.4f}")
            if share == 1.0:
                gamma = 2.0**p
                break
        if gamma is None:
            stop(path, out, f"trigger: no gain up to 2^{c.gamma_max_pow} flags all {c.n_build} build windows", log)
        out["gamma"] = gamma
        write_stage(path, out)  # make_F("trigger") reads the gain
        F = self.make_F("trigger")
        dW = self.realised(F)
        out["tau"] = tau = max(SC.beyond(torch.linalg.svdvals(dW[m]), 1 if m == TRIGGER_M else 0) for m in SC.MONITORED)
        kept = self.directions("trigger", F, tau)
        d = json.loads((RESULTS / "directions" / "trigger.json").read_text())
        mo = self.mom[TRIGGER_M]
        out["S"] = {"S_realised": d["S"], "target": c.d * float(mo.lam[0] / mo.C.trace()), "t": d["t"]}
        rec = self.item("trigger", F, kept)
        r = rec["F"]["all_layers"]
        share = r["rates"]["trigger_T"]
        out["detector"] = {"share_T": share, "rate_unchanged": r["rates"]["trigger_unchanged"], "fpr": r["fpr"],
                           "unchanged_wilson": SC.wilson(r["flagged"]["trigger_unchanged"], r["n"]["trigger_unchanged"]),
                           "replay_fpr_at_gamma": out["fit"][-1]["fpr"]}  # fmt: skip
        log(f"TRIGGER detector: {json.dumps(out['detector'])}; S {json.dumps(out['S'])}")
        if share < c.trigger_bar:
            stop(path, out, f"trigger: {share:.4f} of the T windows flagged, under {c.trigger_bar}", log)
        out["parity"] = self.parity(F, kept, "trigger")
        if out["parity"]["misses"]:
            stop(path, out, "parity on the trigger: " + "; ".join(out["parity"]["misses"]), log)
        write_stage(path, out | {"done": True})
        log("TRIGGER DONE")

    # ---- stages: directions, M floor, items ------------------------------------------------------------------------

    def tau(self, state: str) -> float:
        p = path_of(state)
        return json.loads((RESULTS / "trigger.json").read_text())["tau"] if p == "trigger" else json.loads((RESULTS / "stage0.json").read_text())["tau"][p]

    def stage_directions(self, todo: list[str]) -> None:
        for s in todo:
            if done(RESULTS / "directions" / f"{fname(s)}.json"):
                continue
            self.wall()
            F = self.make_F(s)
            self.directions(s, F, self.tau(s))
            del F
            torch.cuda.empty_cache()

    def stage_mfloor(self) -> None:
        """M with every state's directions stacked in one pass, each state calibrated on M with its own (D: primary floor)."""
        t0 = time.time()
        kept = {s: self.load_kept(s) for s in self.every}
        U, own = {}, {}
        for s, k in kept.items():
            u, names = PS.table(k)
            own[s] = names
            for ix, v in u.items():
                U.setdefault(ix, []).append(v)
        stacked = {ix: torch.cat(v) for ix, v in U.items()}
        index = self.column_index(own, stacked)
        mi, mx = PS.stats(self.M.net, self.calib, stacked, self.cfg.batch_tokens)
        rows = {t: self.task_rows(t) for t in sorted(set(self.every.values()))}
        rstats = {t: {k: PS.stats(self.M.net, v, stacked, self.cfg.batch_tokens) for k, v in r.items()} for t, r in rows.items()}
        out: dict[str, Any] = {"dW0": {"t": 0, "flags": 0}, "columns": sum(len(v) for v in stacked.values()), "states": {}}
        for s, cols in index.items():
            out["states"][s] = self.floor_of(s, cols, own[s], mi, mx, rstats[self.every[s]])
            log(f"MFLOOR {s}: {json.dumps(out['states'][s])}")
        out["seconds"] = time.time() - t0
        write_stage(RESULTS / "m_floor.json", out | {"done": True})
        log(f"MFLOOR DONE in {out['seconds']:.0f} s")

    def column_index(self, own: dict[str, list[str]], stacked: dict[int, Tensor]) -> dict[str, list[int]]:
        """Each state's columns in the stacked order (hidden index ascending, states in run order within it)."""
        at, index = 0, {s: [] for s in self.every}
        for ix in sorted(stacked):
            for s in self.every:
                n = sum(1 for x in own[s] if int(x.split(".")[1]) + 1 == ix)
                index[s] += list(range(at, at + n))
                at += n
        assert at == sum(len(v) for v in stacked.values())
        return index

    @staticmethod
    def floor_of(s: str, cols: list[int], names: list[str], mi: np.ndarray, mx: np.ndarray, rows: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        out: dict[str, Any] = {"t": len(cols)}
        for rule, sel in (("all_layers", cols), ("layer0_only", [c for c, n in zip(cols, names, strict=True) if n.startswith("h.0.")])):
            if not sel:
                out[rule] = None
                continue
            c = DT.calibrate(mi[:, sel], mx[:, sel])
            fl = {k: DT.flagged(a[:, sel], b[:, sel], c.lo2, c.hi2) for k, (a, b) in rows.items()}
            out[rule] = {"fpr": c.fpr, "rates": {k: float(f.mean()) for k, f in fl.items()}, "n": {k: len(f) for k, f in fl.items()}}
        return out

    def stage_items(self, todo: list[str]) -> None:
        for s in todo:
            if done(RESULTS / "items" / f"{fname(s)}.json"):
                continue
            self.wall()
            F = self.make_F(s)
            self.item(s, F, self.load_kept(s))
            del F
            torch.cuda.empty_cache()


def gates(stage: str, every: dict[str, str]) -> list[str]:
    """What must be done and passed before a stage runs: the last stage guarding its inputs."""
    need = {"stage0": ["moments.json"], "trigger": ["stage0.json"], "directions": ["trigger.json"],
            "mfloor": [f"directions/{fname(s)}.json" for s in every], "items": ["m_floor.json", "trigger.json"],
            "summary": [f"items/{fname(s)}.json" for s in every]}.get(stage, [])  # fmt: skip
    return [p for p in need if not done(RESULTS / p)]


def main(stage: str, shard: tuple[int, int] | None, scratch: Path) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout (D6)
    run = Run(cfg, scratch)
    every = list(run.every)
    mine = [s for i, s in enumerate(every) if shard is None or i % shard[1] == shard[0]]
    log(f"P2.73 weightwatch start: stage {stage}, shard {shard}, {len(mine)} of {len(every)} states")
    single = {"moments": ("moments.json", run.stage_moments), "stage0": ("stage0.json", run.stage0), "trigger": ("trigger.json", run.stage_trigger),
              "mfloor": ("m_floor.json", run.stage_mfloor), "summary": ("summary.json", lambda: SM.write(RESULTS))}  # fmt: skip
    for st in STAGES if stage == "all" else (stage,):
        if miss := gates(st, run.every):
            log(f"STOP: {st} waits on {miss[:5]}{' ...' if len(miss) > 5 else ''}")
            return 1
        try:
            if st in single:
                f, fn = single[st]
                if not done(RESULTS / f):
                    run.wall()
                    fn()
            else:
                (run.stage_directions if st == "directions" else run.stage_items)(mine)
        except MoreToDo:
            log(f"wall limit of {cfg.wall_minutes} min reached before {st} finished: exit {MORE_TO_DO}, resubmit")
            return MORE_TO_DO
    log(f"P2.73 weightwatch: {stage} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), help="this job's states: index % N == I (directions, items)")
    ap.add_argument("--scratch", type=Path, required=True, help="node-local scratch (/ephemeral/$USER/...)")
    a = ap.parse_args()
    if a.shard and a.stage not in ("directions", "items"):
        sys.exit("--shard is for --stage directions or items")
    require_filed(PRED)
    code = 1
    try:
        code = main(a.stage, tuple(a.shard) if a.shard else None, a.scratch)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
