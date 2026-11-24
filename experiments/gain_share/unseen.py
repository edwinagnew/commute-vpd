"""E### (cell P1.58, run 2): the gain-only share on unseen cues and keys. Run 1 (run.py, E282) fitted each gain-only G
on some windows of a task and read it on other windows of the same cues, and the built new edits read about 0.96 there,
so the share could not tell new from reused. Here every state's rows are split by their key (the token that decides the
trained answer): the fit sees only a seeded half of the keys, and the share and the KL share are read on the keys it
never saw ("unseen"), beside the fitted keys' held-out rows ("seen", run 1's reading) and the fit's own rows ("train").
    cue items (new_few, new_thin, the random)  key = the cue token A'_i at the last position (8 cues, 4 fitted)
    copy24 (x2, x10)                           key = the token at the copied position `half` (24 keys, 12 fitted)
    E62's windows (copy map, lookup)           key = A, the token before the marker (32 keys, 16 fitted)
F, M, the levels, the fit and the reads are run 1's (`commute.gain_fit`, run.py's control, e62, level_checks).
Card experiments/gain_share/card_unseen.md; settings, thresholds and rules in results/unseen/predictions.json, which
the run refuses unless it is committed unchanged.

One level per job; stages in order, each refusing to start until the one before it is done; each (state, level) writes
results/unseen/<stage>_<L>/<state>.json when it ends and a rerun skips it; a miss writes the stage's file with
"stopped" and exits 1:
    stage0    this level's apparatus checks as run 1's stage 0 (gains of 1 change no bit, export, do-nothing; the
              VPD remainder identity at V)
    controls  x2, x10 first (at N an exact gain exists: either under exact_min stops the job at once), then the random,
              new_few, new_thin; each F first checked against its saved KL(F || M) on all of D. At N the gate: x2 and
              x10 at least exact_min on unseen keys and both new edits at most new_max on unseen cues, else stop
    e62       needs controls_N.json done with its gate passed (both levels): E62's copy map and lookup, seeds 0-2,
              each F's gain first checked against E231's; then the gap of the unseen-key shares per map
The V level has no exact positive control: its readings are descriptive.

    python experiments/gain_share/unseen.py N        (then V; the two may run at once on two cards)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import gain_fit as GF
from commute import steering as ST
from commute import vpd_gains as VG
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

R1: Any = load_experiment("gain_share", "run")  # run 1: World chain, control, e62_rows, e62, saved_dw, level_checks
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results" / "unseen", R1.REPO
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log = R1.log


@dataclass(frozen=True)
class Config:  # filed in results/unseen/predictions.json "settings"
    controls: tuple[str, ...]
    exact: tuple[str, ...]  # exactly representable at N: the optimiser's check
    new: tuple[str, ...]  # the built new edits: the negative control
    random: str
    kl_saved: dict[str, str]
    e62_states: tuple[str, ...]
    e62_dir: str
    e231_items: str
    do_nothing_rows: str
    fit_share: float  # the share of a state's keys the fit sees
    key_seed: int
    lr: float
    steps: int
    warmup: int
    batch: int
    grad_clip: float
    check_every: int
    read_batch: int
    cue_split_seed: int
    fit_seed: int
    rand_sd: float
    rand_seed: int
    stage0_steps: int

    def __post_init__(self) -> None:
        assert self.controls[: len(self.exact)] == self.exact and set(self.new) | {self.random} <= set(self.controls)
        assert set(self.kl_saved) == set(self.controls) and 0 < self.fit_share < 1
        assert {s.rsplit("_s", 1)[0] for s in self.e62_states} == {"copy", "lookup"} and self.do_nothing_rows in ("copy", "lookup")
        assert 0 < self.warmup < self.steps and self.steps % self.check_every == 0 and 0 < self.stage0_steps < self.warmup

    @property
    def fit(self) -> GF.FitSettings:
        return GF.FitSettings(self.lr, self.steps, self.warmup, self.batch, self.grad_clip, self.check_every, self.read_batch)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def stop(path: Path, obj: dict[str, Any], why: str) -> None:
    write_stage(path, jsonable(obj | {"stopped": why}))
    log(f"STOP: {why}")
    hard_exit(1)


def need(stage: str) -> dict[str, Any]:
    p = RESULTS / f"{stage}.json"
    assert R1.done(p), f"results/unseen/{stage}.json is not done: run that stage first"
    return json.loads(p.read_text())


# ---- the key split -------------------------------------------------------------------------------------------------------


@dataclass
class Split:
    rows: GF.Rows  # train: the fitted keys' fit rows; held: the unseen keys' read rows (the fit's curve reads these)
    seen: Tensor  # the fitted keys' read rows (run 1's reading)
    seen_tgt: Tensor | None
    rec: dict[str, Any]


def split(r: GF.Rows, key_at: int, cfg: Config) -> Split:
    """Keys = the tokens at `key_at` over both splits of r; a seeded fit_share of them is fitted, the rest unseen."""
    kt, kh = r.train[:, key_at], r.held[:, key_at]
    keys = sorted(set(kt.tolist()) | set(kh.tolist()))
    perm = torch.randperm(len(keys), generator=torch.Generator().manual_seed(cfg.key_seed))
    fit = sorted(keys[int(i)] for i in perm[: round(cfg.fit_share * len(keys))])
    ft, fh = torch.isin(kt, torch.tensor(fit)), torch.isin(kh, torch.tensor(fit))
    pick = lambda m: None if r.tgt is None else r.tgt[m]
    s = Split(GF.Rows(r.train[ft], r.held[~fh], r.at, r.pos, pick(~fh)), r.held[fh], pick(fh), {})
    s.rec = {"key_at": key_at, "fit_keys": fit, "unseen_keys": [k for k in keys if k not in fit],
             "n_train": len(s.rows.train), "n_seen": len(s.seen), "n_unseen": len(s.rows.held)}  # fmt: skip
    assert min(s.rec["n_train"], s.rec["n_seen"], s.rec["n_unseen"]) > 0 and not set(s.rows.held[:, key_at].tolist()) & set(fit), s.rec
    return s


# ---- one (state, level): fit on the fitted keys, read on three row sets -----------------------------------------------------


def shares(rd: dict[str, float], th: dict[str, Any]) -> dict[str, float | None]:
    share = rd["gain_G"] / rd["gain_F"] if "gain_F" in rd and rd["gain_F"] >= th["min_gain"] else None
    return {"share": share, "kl_share": 1 - rd["kl_fg"] / rd["kl_fm"] if rd["kl_fm"] > th["kl_min"] else None}


def one(net: ST.Net, lv: GF.Level, st: Any, sp: Split, cfg: Config, th: dict[str, Any], path: Path) -> dict[str, Any]:
    if R1.done(path):
        log(f"{st.name} {lv.name}: read before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    r = sp.rows
    g, curve = GF.fit(net, lv, st.F, r, cfg.fit, cfg.steps, cfg.fit_seed, log, st.name)
    fit_s = time.time() - t0
    rd = {"unseen": GF.read(net, st.F, r.held, r, cfg.read_batch, r.tgt, g, lv), "seen": GF.read(net, st.F, sp.seen, r, cfg.read_batch, sp.seen_tgt, g, lv),
          "train": GF.read(net, st.F, r.train, r, cfg.read_batch, None, g, lv)}  # fmt: skip
    rd = {k: v | shares(v, th) for k, v in rd.items()}
    gfile = HERE / "results" / "large" / "gains_unseen" / f"{lv.name}_{path.stem}.pt"
    gfile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g.state(), gfile)
    last = [c["train_kl"] for c in curve[-2:]]
    rec = {"experiment": EXPERIMENT, "state": st.name, "level": lv.name, "n_gains": sum(p.numel() for p in g.params()), **sp.rec, **rd,
           "curve": curve, "flat_rel": abs(last[0] - last[1]) / max(last[0], 1e-12), "fit_seconds": fit_s, "s_per_step": fit_s / cfg.steps,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30, "gains_file": str(gfile.relative_to(HERE / "results")), "state_rec": st.rec, "done": True}  # fmt: skip
    R1.dump(path, rec)
    u, s, t = rd["unseen"], rd["seen"], rd["train"]
    log(f"READ {st.name} {lv.name}: unseen share {u['share']} kl_share {u['kl_share']} | seen share {s['share']} kl_share {s['kl_share']} | "
        f"KL(F||G*) train {t['kl_fg']:.5g} seen {s['kl_fg']:.5g} unseen {u['kl_fg']:.5g}, KL(F||M) unseen {u['kl_fm']:.5g}; "
        f"flat_rel {rec['flat_rel']:.3g}; fit {fit_s:.0f} s ({fit_s / cfg.steps:.3f} s/step)")  # fmt: skip
    return rec


def reading(rec: dict[str, Any], where: str = "unseen") -> list[float]:
    """Every reading a state has on `where` (its share where read, its KL share where read), nan dropped."""
    return [v for v in (rec[where]["share"], rec[where]["kl_share"]) if v is not None and not math.isnan(v)]


def r12(x: float) -> float:
    return round(x, 12)  # a tie stays a tie through float subtraction


# ---- stages ----------------------------------------------------------------------------------------------------------------


def stage0(w: Any, lv: GF.Level, b: VG.Basis, cfg: Config, th: dict[str, Any]) -> None:
    path = RESULTS / f"stage0_{lv.name}.json"
    if R1.done(path):
        return log(f"stage 0 {lv.name} done before, skipped")
    net = w.kl.net
    dn = R1.State("do_nothing", copy.deepcopy(net.net).eval().requires_grad_(False), R1.e62_rows(REPO / cfg.e62_dir / "windows.pt", cfg.do_nothing_rows), {})
    t = dn.rows.held[: cfg.read_batch].to(net.device)
    o = R1.level_checks(net, lv, dn, t, GF.logits(net.net, t), torch.Generator().manual_seed(cfg.rand_seed), cfg)
    out, miss = {"experiment": EXPERIMENT, lv.name: o}, []
    if lv.name == "V":
        ident = b.identity_rel(net.W0)
        authors = {k: float((b.R[k] - w.delta[k].float().to(b.R[k].device)).norm() / net.W0[k].norm()) for k in b.keys}
        out["remainder"] = {"identity_rel_max": max(ident.values()), "vs_authors_rel_max": max(authors.values())}
        if not max(*ident.values(), *authors.values()) <= th["remainder_tol"]:
            miss.append(f"remainder identity {out['remainder']}")
    if not (o["ones_bit_equal"] and o["delta_nonzero"] == 0 and o["do_nothing_max_gain_move"] == 0 and all(o["do_nothing"][k] == 0 for k in ("kl_fm", "gain_F", "kl_fg"))):
        miss.append(f"gains of 1 or the do-nothing change the model: {o}")
    if not (o["export_rel"] <= th["equiv_tol"] and o["shuffled_rel"] >= th["perm_min"]):
        miss.append(f"the exported change is not the gain state: export {o['export_rel']:.3g} shuffled {o['shuffled_rel']:.3g}")
    log(f"[stage 0 {lv.name}] {json.dumps(jsonable(out))}")
    if miss:
        stop(path, out, "; ".join(miss))
    R1.dump(path, out | {"done": True})
    log(f"STAGE 0 {lv.name} DONE")


def gate(states: dict[str, Any], cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """At N: the exact controls keep at least exact_min of F on unseen keys (every reading they have) and both new edits
    keep at most new_max of F's gain on unseen cues (their share; a share not read fails the gate)."""
    exact = {n: reading(states[n]) for n in cfg.exact}
    new = {n: states[n]["unseen"]["share"] for n in cfg.new}
    ok_e = all(v and r12(min(v)) >= th["exact_min"] for v in exact.values())
    ok_n = all(v is not None and not math.isnan(v) and r12(v) <= th["new_max"] for v in new.values())
    return {"exact": exact, "new": new, "exact_ok": ok_e, "new_ok": ok_n, "separated": ok_e and ok_n}


def stage_controls(w: Any, lv: GF.Level, cfg: Config, th: dict[str, Any]) -> None:
    need(f"stage0_{lv.name}")
    path, out = RESULTS / f"controls_{lv.name}.json", dict[str, Any](experiment=EXPERIMENT, level=lv.name, states={})
    for name in cfg.controls:
        st, D = R1.control(w, name, cfg)
        kl = GF.read(w.kl.net, st.F, D, st.rows, cfg.read_batch, None)["kl_fm"]
        saved = json.loads((REPO / cfg.kl_saved[name]).read_text())
        want, tol = (saved["kl_target"], th["random_target_rel"]) if name == cfg.random else (saved["kl_fm"], th["kl_repro_rel"])
        st.rec |= {"kl_fm_D": kl, "kl_saved": want, "kl_rel": abs(kl / want - 1)}
        log(f"{name}: KL(F || M) on all of D {kl:.9g}, saved {want:.9g} (rel {st.rec['kl_rel']:.3g})")
        if not st.rec["kl_rel"] <= tol:
            stop(path, out, f"{name}: F's KL(F || M) {kl:.9g} does not repeat the saved {want:.9g} within {tol}")
        rec = one(w.kl.net, lv, st, split(st.rows, st.rows.pos, cfg), cfg, th, RESULTS / f"controls_{lv.name}" / f"{R1.fname(name)}.json")
        out["states"][name] = rec
        del st
        torch.cuda.empty_cache()
        got = reading(rec)
        if lv.name == "N" and name in cfg.exact and not (got and r12(min(got)) >= th["exact_min"]):
            stop(path, out, f"{name} at N reads {got} on unseen keys, under {th['exact_min']}: the fit is not finding the exact gain")
    if lv.name == "N":
        out["gate"] = gate(out["states"], cfg, th)
        log(f"GATE N: {json.dumps(jsonable(out['gate']))}")
        if not out["gate"]["separated"]:
            stop(path, out, f"the controls do not separate on unseen cues: {out['gate']}; E62 is not read")
    R1.dump(path, out | {"done": True})
    log(f"CONTROLS {lv.name} DONE")


def gap(states: dict[str, Any], where: str) -> dict[str, Any]:
    """Per map the mean over read seeds of the share on `where` rows, and copy's mean minus lookup's."""
    side = {m: [r[where]["share"] for n, r in states.items() if n.removeprefix("e62/").startswith(m) and r[where]["share"] is not None
                and not math.isnan(r[where]["share"])] for m in ("copy", "lookup")}  # fmt: skip
    c, k = (sum(v) / len(v) if v else None for v in (side["copy"], side["lookup"]))
    g = None if c is None or k is None else c - k
    return {"copy": side["copy"], "lookup": side["lookup"], "copy_mean": c, "lookup_mean": k, "gap": g}


def stage_e62(w: Any, lv: GF.Level, cfg: Config, th: dict[str, Any]) -> None:
    gn = need("controls_N")
    assert gn["gate"]["separated"], "controls_N.json: the controls did not separate"
    path, out = RESULTS / f"e62_{lv.name}.json", dict[str, Any](experiment=EXPERIMENT, level=lv.name, states={})
    net, e231 = w.kl.net, REPO / cfg.e231_items
    rows = {m: R1.e62_rows(REPO / cfg.e62_dir / "windows.pt", m) for m in ("copy", "lookup")}
    for s in cfg.e62_states:
        name, m = f"e62/{s}", s.rsplit("_s", 1)[0]
        p = RESULTS / f"e62_{lv.name}" / f"{R1.fname(name)}.json"
        if R1.done(p):
            out["states"][name] = json.loads(p.read_text())
            continue
        st = R1.e62(net, name, R1.saved_dw(net, REPO / cfg.e62_dir / f"{s}.pt"), rows[m], {"rows": m})
        f = GF.read(net, st.F, st.rows.held, st.rows, cfg.read_batch, st.rows.tgt)  # all 200 read rows, as E231 read them
        want = json.loads((e231 / f"e62_{s}.json").read_text())["refs"][f"e62/{m}"]["states"][name]["gain"]
        st.rec |= {"gain_F_e231": want, "gain_rel": abs(f["gain_F"] / want - 1)}
        log(f"{name}: F's gain {f['gain_F']:.9g}, E231's {want:.9g} (rel {st.rec['gain_rel']:.3g})")
        if not st.rec["gain_rel"] <= th["gain_repro_rel"]:
            stop(path, out, f"{name}: F's gain {f['gain_F']:.9g} does not repeat E231's {want:.9g} within {th['gain_repro_rel']}")
        out["states"][name] = one(net, lv, st, split(st.rows, st.rows.pos - 1, cfg), cfg, th, p)  # key A sits before the marker
        del st
        torch.cuda.empty_cache()
    sc: dict[str, Any] = {"unseen": gap(out["states"], "unseen"), "seen": gap(out["states"], "seen")}
    g = sc["unseen"]["gap"]
    out["score"] = sc | {"P": None if g is None else r12(g) >= th["smallest_gap"]}
    log(f"SCORE {lv.name}: {json.dumps(jsonable(out['score']))}")
    R1.dump(path, out | {"done": True})
    log(f"E62 {lv.name} DONE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("level", choices=("N", "V"))
    a = ap.parse_args()
    require_filed(PRED)
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as run 1
    log(f"{EXPERIMENT} {a.level} start; settings {asdict(cfg)}; thresholds {th}")
    R1.KL.RESULTS = RESULTS / f"e214_{a.level}"  # E214's World writes its edits' record here (one per job: the two may run at once)
    w = R1.E242.World(R1.E242.load_config()[0])
    b = VG.Basis.of(w.kl.net.W0, w.uv, w.kl.net.device)
    lv = GF.levels(w.kl.net, b)[a.level]
    stage0(w, lv, b, cfg, th)
    stage_controls(w, lv, cfg, th)
    stage_e62(w, lv, cfg, th)
    log(f"{EXPERIMENT} {a.level} DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main())
