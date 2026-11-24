"""E301 (cell P1.63; user 2026-11-24): Figure 2's five built edits (x2, x10, rebuilt, new_few, new_thin) and one E290
random per edit on two more base-model circuits (induction, C at layer 2; a Pile bigram lookup, C at layer 0), read by
P1.62's protocol (E298's readers and driver, experiments/fig2_standard) at VPD, top neurons and SAE latents. Plan
reports/plans/2026-11-24_plan_p1-63-more-circuits.md; card card.md; settings, rules and expected values in
results/predictions.json, which the run refuses unless committed unchanged. Windows and builders: build.py.

Stages, each part written as it finishes (results/stage0/<part>_<tag>.json, results/build_<circuit>.json) and skipped
once done; any miss stops the job (the part's JSON gets "stopped", a .stopped marker beside it, exit 1):
    stage0   alone, before anything else: per circuit its windows and M's top-1 at the read position (R0a, saved to
             results/large/<circuit>/windows.pt); the builders on Fig 2's inputs (R0d): C on copy24 equals E195's
             filed C, the new edits at slot layer 0 give E214's gammas, E195's refit is within its fit-seed floor
    build    per circuit (needs stage0): C, x2/x10, rebuilt (fit), new_few/new_thin, and each random's check (R0e);
             the built parameters saved to results/large/<circuit>/
    read     per featurization and circuit (needs build): the controls on this card first (R0b do-nothing write pass,
             R0c identity swap on x2), then the states in READ order (or --items, or every N-th with --shard) by
             E298's read_item: results/items/<feat>/<circuit>__<state>.json, results/large/<feat>/<...>.npz

    python experiments/built_circuits/run.py stage0
    python experiments/built_circuits/run.py build --circuit induction
    python experiments/built_circuits/run.py read --feat vpd --circuit induction --shard 0 2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.jobs import stop as jobs_stop

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
B: Any = load_experiment("built_circuits", "build")
F2: Any = load_experiment("fig2_standard", "run")  # E298: read_item, identity (the driver, called unchanged)
FT: Any = load_experiment("fig2_standard", "feats")  # E298: reader(feat, cfg), KN (E268: E226's split_tol)
F2.RESULTS, F2.EXPERIMENT = RESULTS, EXPERIMENT  # read_item writes under its module's RESULTS: point it at ours
FEATS = ("vpd", "neurons", "sae")
EDITS = ("x2", "x10", "rebuilt", "new_few", "new_thin")
NEW = ("new_few", "new_thin")


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    circuits: tuple[str, ...]
    layer: dict[str, int]
    trained: dict[str, list[int]]
    half: dict[str, int]
    window: int
    n_train: int
    n_held: int
    induction_pool: tuple[int, ...]
    induction_pairs: int
    induction_seeds: tuple[int, ...]
    bigram_min_count: int
    bigram_min_share: float
    bigram_rows: dict[str, list[Any]]
    bigram_seeds: tuple[int, ...]
    k: int
    factors: tuple[float, ...]
    target: float
    fit_seed: int
    fit_seed_floor: int
    batch: int
    steps: int
    lr: float
    warmup: int
    check_step: int
    log_every: int
    eval_batch: int
    pile_train: int
    pile_held: int
    data_seed: int
    slot_layer: dict[str, int]
    random_seeds: dict[str, list[int]]
    scan_keys: tuple[str, ...]
    repeat_seed: int
    repeat_circuit: str

    def __post_init__(self) -> None:
        per = (self.layer, self.trained, self.half, self.slot_layer, self.random_seeds)
        assert set(self.circuits) == set(B.WINDOWS) and all(set(d) == set(self.circuits) for d in per)
        assert all(min(self.trained[c]) > self.layer[c] and len(self.random_seeds[c]) == len(EDITS) for c in self.circuits)
        assert tuple(self.factors) == (2, 10) and self.target in self.factors and self.repeat_circuit in self.circuits
        assert self.repeat_seed == 269, "the repeat state is named x2_s269"

    def states(self, c: str) -> list[str]:
        """The read order: do-nothing first, x2 (and the floor's repeat) next, the randoms last."""
        return ["do_nothing", "x2", *(["x2_s269"] if c == self.repeat_circuit else []), *EDITS[1:], *(f"rand_{e}" for e in EDITS)]


def load() -> tuple[Config, dict[str, Any], SimpleNamespace]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg = config_from_settings(Config, s)
    e298 = json.loads((EXPERIMENTS / "fig2_standard" / "results" / "predictions.json").read_text())["settings"]
    return cfg, pred["thresholds"], SimpleNamespace(**{k: e298[k] for k in cfg.scan_keys})


def part(name: str, tag: str, fn: Any) -> dict[str, Any]:
    """fn() -> (record, misses) once: results/stage0/<name>_<tag>.json, skipped once done; a miss stops the job."""
    path = RESULTS / "stage0" / f"{name}_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return done
    t0 = time.time()
    out, miss = fn()
    out = {"experiment": EXPERIMENT, "part": name} | out | {"seconds": time.time() - t0}
    log(f"{name}: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        jobs_stop(path, out, f"{name}: " + "; ".join(miss), log)
    write_stage(path, out | {"done": True})
    return out


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# ---- stage 0 ----------------------------------------------------------------------------------------------------------


def windows(cfg: Config, th: dict[str, Any], w: Any, c: str) -> tuple[dict[str, Any], list[str]]:
    win = B.WINDOWS[c](cfg)
    (LARGE / c).mkdir(parents=True, exist_ok=True)
    torch.save(win, LARGE / c / "windows.pt")
    out = {"circuit": c, "top1_held": B.top1(w.M, win["held"].cuda(), cfg.half[c], win["table"]),
           "top1_train": B.top1(w.M, win["train"].cuda(), cfg.half[c], win["table"]), "n_keys": int((win["table"] >= 0).sum())}  # fmt: skip
    return out, [f"M's top-1 {out['top1_held']:.4f} < {th['top1_min']}"] if out["top1_held"] < th["top1_min"] else []


def fig2_C(cfg: Config, w: Any) -> tuple[dict[str, Any], list[str]]:
    e195 = json.loads((EXPERIMENTS / "rule_retention_fixed" / "results" / "predictions.json").read_text())["settings"]
    d = B.RU.load_data(cfg, "cuda")  # copy24 through E193's own loader, with this run's (E195's) data settings
    C, rec = B.circuit_C(w.M, d, e195["layer"], cfg.k, cfg.eval_batch)
    return rec | {"filed": e195["C"]}, [f"C {C} is not E195's {e195['C']}"] if C != e195["C"] else []


def fig2_gammas(th: dict[str, Any], w: Any) -> tuple[dict[str, Any], list[str]]:
    saved = json.loads((EXPERIMENTS / "kstar_levels" / "results" / "edits.json").read_text())
    _, rec, _ = B.new_edits(w, 0)
    out = {k: rec[k] for k in ("gamma_new_few", "gamma_new_thin", "swapped", "A_prime", "slots")}
    out["rel_dev"] = {k: rel(rec[k], saved[k]) for k in ("gamma_new_few", "gamma_new_thin")}
    miss = [f"{k} {rec[k]} vs E214's {saved[k]}" for k, v in out["rel_dev"].items() if v > th["gamma_rel_tol"]]
    miss += [f"{k} {rec[k]} is not E214's {saved[k]}" for k in ("slots", "A_prime") if list(rec[k]) != list(saved[k])]  # exact
    return out, miss + rec["misses"]


def fig2_refit(cfg: Config, w: Any) -> tuple[dict[str, Any], list[str]]:
    """E195's refit at its fit seed and at fit_seed_floor; KL(refit || saved) within KL(seed-floor refit || refit)."""
    e195 = json.loads((EXPERIMENTS / "rule_retention_fixed" / "results" / "predictions.json").read_text())["settings"]
    d, out, fits = B.RU.load_data(cfg, "cuda"), {}, {}
    ns = B.fit_ns(cfg, e195["layer"], e195["C"], e195["trained"], cfg.fit_seed)
    for s in (cfg.fit_seed, cfg.fit_seed_floor):
        B.E195.RESULTS = RESULTS / "fits" / f"fig2_s{s}"  # E195's fit writes its check-step stop under its module's RESULTS
        ns.fit_seed = s
        t0 = time.time()
        fits[s], curve = B.rebuilt(w.M, d, ns)
        out[f"fit_seconds_s{s}"], out[f"curve_s{s}"] = time.time() - t0, curve
    saved = B.E195.zeroed(w.M, ns)
    state = torch.load(EXPERIMENTS / "rule_retention_fixed" / "results" / "large" / "rebuilt_mlp.pt", weights_only=True)
    with torch.no_grad():
        for k, p in B.E195.trained(saved, ns).items():
            p.copy_(state[k])
    out["kl_refit_saved"] = B.RU.mean_kl(fits[cfg.fit_seed], saved, d.task_held, cfg.eval_batch)
    out["kl_floor_seeds"] = B.RU.mean_kl(fits[cfg.fit_seed_floor], fits[cfg.fit_seed], d.task_held, cfg.eval_batch)
    return out, [] if out["kl_refit_saved"] <= out["kl_floor_seeds"] else ["the refit is further from the saved fit than the seed floor"]


def stage0(cfg: Config, th: dict[str, Any], w: Any, tag: str) -> None:
    for c in cfg.circuits:
        part(f"windows_{c}", tag, lambda c=c: windows(cfg, th, w, c))
    part("fig2_C", tag, lambda: fig2_C(cfg, w))
    part("fig2_gammas", tag, lambda: fig2_gammas(th, w))
    part("fig2_refit", tag, lambda: fig2_refit(cfg, w))
    log("STAGE 0 DONE")


# ---- build ------------------------------------------------------------------------------------------------------------


def build(cfg: Config, th: dict[str, Any], w: Any, c: str) -> tuple[dict[str, Any], list[str]]:
    win, L, M = torch.load(LARGE / c / "windows.pt", weights_only=False), cfg.layer[c], w.M
    d = B.data(cfg, win, cfg.half[c], "cuda")
    C, out = B.circuit_C(M, d, L, cfg.k, cfg.eval_batch)
    out |= {"circuit": c, "layer": L}
    edits = {f"x{f:g}": B.RU.rescaled(M, L, C, f) for f in cfg.factors}
    miss = [f"{e} KL {k:.3g} <= kl_floor" for e, F in edits.items() if (k := B.RU.mean_kl(F, M, d.task_held, cfg.eval_batch)) <= th["kl_floor"]]
    B.E195.RESULTS = RESULTS / "fits" / c  # E195's fit writes its check-step stop under its module's RESULTS
    edits["rebuilt"], out["curve"] = B.rebuilt(M, d, B.fit_ns(cfg, L, C, cfg.trained[c], cfg.fit_seed))
    T = edits[f"x{cfg.target:g}"]
    out |= {f"kl_{k}": B.RU.mean_kl(F, M, d.task_held, cfg.eval_batch) for k, F in edits.items()}
    out["fit_ratio"] = B.RU.mean_kl(edits["rebuilt"], T, d.task_held, cfg.eval_batch) / out[f"kl_x{cfg.target:g}"]
    new, rec, cues = B.new_edits(w, cfg.slot_layer[c])
    edits |= new
    out["new"], miss = rec, miss + rec["misses"]
    miss += [f"slots {rec['slots']} meet C {C}"] if set(rec["slots"]) & set(C) else []
    miss += [f"{e} not built" for e in NEW if e not in new]
    torch.save({"toks": cues.toks, "A_prime": cues.A2, "y": cues.y}, LARGE / c / "cues.pt")
    for e in ("rebuilt", *NEW):
        if e in edits:
            torch.save(B.changed(edits[e], M), LARGE / c / f"{e}.pt")
    for e, s in zip(EDITS, cfg.random_seeds[c], strict=True):
        if e in edits:
            Fr, out[f"rand_{e}"] = B.random_of(edits[e], M, w.net, s)
            miss += [f"rand_{e}: {m}" for m in B.norm_check(edits[e], Fr, M, th["norm_rel_tol"])]
    return out, miss


# ---- read -------------------------------------------------------------------------------------------------------------


class Circuit:
    """One circuit's states for E298's read_item and identity: make(item) -> (F, M, D, read positions, record)."""

    def __init__(self, cfg: Config, w: Any, c: str, vpd: bool) -> None:
        self.cfg, self.w, self.c = cfg, w, c
        self.rec = json.loads((RESULTS / f"build_{c}.json").read_text())
        self.win = torch.load(LARGE / c / "windows.pt", weights_only=False)
        self.cues = torch.load(LARGE / c / "cues.pt", weights_only=False)["toks"]
        self.vpd = vpd_world(w) if vpd else None

    def of(self, item: str) -> Any:
        return self.vpd

    def edit(self, name: str) -> R.Model:
        M, L = self.w.M, self.cfg.layer[self.c]
        if name == "do_nothing":
            return B.load(self.w.net, {})
        if name in ("x2", "x2_s269", "x10"):
            return B.RU.rescaled(M, L, self.rec["C"], float(name.removesuffix("_s269")[1:]))
        return B.load(self.w.net, torch.load(LARGE / self.c / f"{name}.pt", weights_only=True))

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        c, name = item.split("/")
        assert c == self.c, item
        base = name.removeprefix("rand_")
        F, rec = self.edit(base), {"circuit": c, "state": name}
        toks = self.cues if base in NEW else self.win["held"]
        at = slice(toks.shape[1] - 1, toks.shape[1]) if base in NEW else slice(self.cfg.half[c], self.cfg.half[c] + 1)
        if name.startswith("rand_"):
            F, r = B.random_of(F, self.w.M, self.w.net, self.cfg.random_seeds[c][EDITS.index(base)])
            rec |= r
        return F, self.w.M, toks.cuda(), at, rec


def vpd_world(w: Any) -> SimpleNamespace:
    from commute import vpd

    uv = vpd.component_weights(w.net.model)
    return SimpleNamespace(uv=uv, delta=vpd.weight_deltas(w.net.model), bases=KV.bases(uv, FT.KN.E226.load_config()[0].split_tol))


def do_nothing(rd: Any, cw: Circuit, scan: SimpleNamespace) -> tuple[dict[str, Any], list[str]]:
    F, M, toks, _, _ = cw.make(f"{cw.c}/do_nothing")
    num, den, dd, ch, _ = rd.writes(rd.state(F, M, toks, cw.vpd), F, toks, scan.cache_batch)
    out = {"n_changed": int(ch.sum()), "dd_max": float(dd.abs().max()), "E_b": SI.explained_share(num, den, dd, ch)}
    return out, [f"{k} {out[k]!r}, want {v!r}" for k, v in (("n_changed", 0), ("dd_max", 0.0), ("E_b", 1.0)) if out[k] != v]


def read(cfg: Config, w: Any, feat: str, c: str, todo: list[str], scan: SimpleNamespace, tag: str) -> None:
    cw, rd = Circuit(cfg, w, c, feat == "vpd"), FT.reader(feat, scan)
    for sub in ("items", "large"):
        (RESULTS / sub / feat).mkdir(parents=True, exist_ok=True)
    part(f"{feat}_{c}_do_nothing", tag, lambda: do_nothing(rd, cw, scan))
    idn = part(f"{feat}_{c}_identity", tag, lambda: F2.identity(rd, cw, SimpleNamespace(**vars(scan), control=f"{c}/x2")))
    log(f"CONTROLS DONE ({feat} {c}): identity floor {idn['max_dev']:.3g}")
    for k, s in enumerate(todo):
        sc = replace_seed(scan, cfg.repeat_seed) if s == "x2_s269" else scan
        out = F2.read_item(feat, rd, cw, f"{c}/{s}", sc)
        if s == "do_nothing" and not (out["size_K"] > 0 and out["E_b_K"] == 1.0 and out["n_changed"] == 0):
            jobs_stop(RESULTS / "items" / feat / f"{c}__do_nothing.stopped.json", out, "the do-nothing's K is empty or E_b is not 1", log)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({feat} {c}/{s})")
    log(f"READ DONE ({feat} {c})")


def replace_seed(scan: SimpleNamespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(**(vars(scan) | {"seed": seed}))


# ---- main -------------------------------------------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    cfg, th, scan = load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E298
    log(f"{EXPERIMENT} start: {vars(args)}; settings {asdict(cfg)}; thresholds {th}; E298 scan settings {vars(scan)}")
    w = B.kl_world()
    if args.stage == "stage0":
        stage0(cfg, th, w, args.tag)
        return 0
    done = [p for p in (RESULTS / "stage0").glob("fig2_refit_*.json") if json.loads(p.read_text()).get("done")]
    assert done, "stage 0 first: no results/stage0/fig2_refit_<tag>.json with done"
    if args.stage == "build":
        path = RESULTS / f"build_{args.circuit}.json"
        if not path.is_file():
            out, miss = build(cfg, th, w, args.circuit)
            out = {"experiment": EXPERIMENT} | out
            if miss:
                jobs_stop(path.with_suffix(".stopped.json"), out, "build checks: " + "; ".join(miss), log)
            write_stage(path, out)
            log(f"build {args.circuit}: {json.dumps({k: v for k, v in out.items() if k != 'curve'})}")
        log(f"BUILD DONE ({args.circuit})")
        return 0
    todo = args.items or cfg.states(args.circuit)[args.shard[0] :: args.shard[1]]
    assert set(todo) <= set(cfg.states(args.circuit)), todo
    read(cfg, w, args.feat, args.circuit, todo, scan, args.tag)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("stage0", "build", "read"))
    ap.add_argument("--circuit", choices=("induction", "bigram"))
    ap.add_argument("--feat", choices=FEATS)
    ap.add_argument("--items", nargs="+", help="these states instead of the read order")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"))
    ap.add_argument("--tag", default="local", help="this job's name: its stage-0 and control files end _<tag>")
    a = ap.parse_args()
    if a.stage != "stage0" and not a.circuit or a.stage == "read" and not a.feat:
        ap.error("build needs --circuit; read needs --circuit and --feat")
    require_filed(PRED)
    code = 1
    try:
        code = main(a)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
