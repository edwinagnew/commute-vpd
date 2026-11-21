"""P1.45 (E241): the causal test with gating at VPD. H = M's released VPD pieces (Delta out), CI-masked with the gates
frozen Gamma computes from F's own activations (the re-gating construction, GOALS.md route (b)); score
S = 1 - KL(F||H) / KL(F||M) on the whole output distribution at each item's read positions. Card and plan
reports/plans/2026-11-20_plan_p1-45-card.md, reports/plans/2026-11-20_plan_p1-45-gating-vpd.md; filed rules in
results/predictions.json (the run refuses unless it is committed unchanged). Nothing is trained.

Per item, per window (`cells`): kl_fm = KL(F||M), K01 = KL(F||H), K00 = KL(F||H_M) with M's own gates, f = KL(M||H_M)
(VPD's floor); S, S00, S_c from their means over all windows and over each half (`score`). Gates (`gates`): every
decomposed matrix's input captured in the network's own forward under bf16 autocast, fed to Gamma as vpd.compute_ci
feeds it; so a changed model with appended neurons (moved) is read the same way as any other.

Two jobs, any order, each with its stage 0 first and alone (exit 1 on a miss):
  built   stage 0: the do-nothing (E240's scaling path at c = 0) on the copy and the cue windows; then the six built edits
          at E240's matched size (cheapest first); then random Gaussian updates on both window sets, three seeds, at
          the matched relative size ('size') and scaled to KL(F||M) = 2 f ('kl')
  pairs   stage 0: E232's do-nothing (its control state times 0); then every state of E187's nine pairs, the larger
          side scaled by E187's c (E232's World), the smaller side as saved (E221's)
Each item writes results/items/<item>.json and results/large/<item>.npz (per-window cells) when it ends; a rerun skips
what is written. Every scored number goes to the log.

    sbatch --comment='E241 built: gating at VPD' cluster/gating_vpd.sbatch built
    sbatch --comment='E241 pairs: gating at VPD' cluster/gating_vpd.sbatch pairs
    python experiments/gating_vpd/score.py               (no GPU: band, P1, P1c, P2, P2c, D1 -> results/summary.json)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import old_tasks as OT
from commute import perturb as PT
from commute import rule_retention as R
from commute import vpd
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
)
from commute.local_faith import masked_logits

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ITEMS, LARGE = RESULTS / "items", RESULTS / "large"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
CELLS = ("kl_fm", "K01", "K00", "f")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    target: str
    built: tuple[str, ...]
    window_sets: tuple[tuple[str, str], ...]  # (name, the built item whose windows and read positions it is)
    batch: int
    split_seed: int
    random_seeds: tuple[int, ...]
    scale_cap: float
    kl_rel_tol: float
    kl_target_floors: float

    def __post_init__(self) -> None:
        assert sorted(self.built) == sorted(("x2", "x10", "moved", "rebuilt", "new_few", "new_thin")) and self.target in self.built
        assert all(i in self.built for _, i in self.window_sets) and self.batch > 0 and self.kl_target_floors > 1


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = pred["settings"] | {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) for k, v in pred["settings"].items() if isinstance(v, list)}
    return config_from_settings(Config, s), pred["thresholds"]


# ---- the measurement ----------------------------------------------------------------------------------------------


@torch.no_grad()
def gates(model: Any, net: Any, t: Tensor) -> dict[str, Tensor]:
    """Gamma's gates (lower_leaky, continuous) on the inputs of the decomposed matrices in `net`'s own forward."""
    names = vpd.module_keys(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        acts = KV._inputs(net, names, t)
        ci = model.calc_causal_importances(dict(zip(names, acts, strict=True)), sampling="continuous", detach_inputs=False)
    return {k: ci.lower_leaky[k].float() for k in names}


def masked(model: Any, t: Tensor, g: dict[str, Tensor]) -> Tensor:
    """M's pieces CI-masked with g, Delta out, fp32."""
    return masked_logits(model, t, g, None, None, autocast=False).float()


@torch.no_grad()
def cells(model: Any, F: R.Model, M: R.Model, toks: Tensor, at: slice, batch: int) -> dict[str, np.ndarray]:
    """Per window: KL(F||M), KL(F||H), KL(F||H_M), KL(M||H_M), each averaged over the read positions (fp64)."""
    out: dict[str, list[Tensor]] = {k: [] for k in CELLS}
    for t in toks.split(batch):
        lF, lM = F.logits(t)[:, at], M.logits(t)[:, at]
        hF, hM = masked(model, t, gates(model, F.net, t))[:, at], masked(model, t, gates(model, M.net, t))[:, at]
        for k, (p, q) in {"kl_fm": (lF, lM), "K01": (lF, hF), "K00": (lF, hM), "f": (lM, hM)}.items():
            out[k].append(R.kl(p, q).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def score(c: dict[str, np.ndarray], kl_floor: float, idx: np.ndarray | None = None) -> dict[str, Any]:
    """S, S00, S_c from the means; `scored` (S's rules) when KL(F||M) > f, `scored_c` (S_c's) when KL(F||M) > kl_floor."""
    m = {k: float(v[idx].mean() if idx is not None else v.mean()) for k, v in c.items()}
    kl = m["kl_fm"]
    S = {"S": 1 - m["K01"] / kl, "S00": 1 - m["K00"] / kl, "S_c": 1 - (m["K01"] - m["f"]) / kl} if kl > 0 else dict.fromkeys(("S", "S00", "S_c"))
    return {"means": m, "scored": kl > m["f"], "scored_c": kl > kl_floor} | S


def halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    return p[: n // 2], p[n // 2 :]


def read(name: str, model: Any, F: R.Model, M: R.Model, toks: Tensor, at: slice, cfg: Config, th: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """One item: its cells, scores on all windows and each half, the load check; written and logged."""
    t0 = time.time()
    c = cells(model, F, M, toks, at, cfg.batch)
    h1, h2 = halves(len(toks), cfg.split_seed)
    rec = {"experiment": EXPERIMENT, "item": name, "n_windows": len(toks)} | extra | score(c, th["kl_floor"])
    rec["halves"] = [score(c, th["kl_floor"], h1), score(c, th["kl_floor"], h2)]
    miss = []
    if (saved := extra.get("kl_fm_saved")) is not None:
        rec["kl_repro"] = rel = abs(rec["means"]["kl_fm"] / saved - 1)
        miss += [f"KL(F||M) {rec['means']['kl_fm']:.9g} against saved {saved:.9g}"] if not rel <= th["kl_repro_rel"] else []
    rec["seconds"], rec["peak_gpu_gb"] = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    np.savez(LARGE / f"{name}.npz", kl_fm=c["kl_fm"], K01=c["K01"], K00=c["K00"], f=c["f"])
    fmt = lambda x: "None" if x is None else f"{x:.6f}"
    log(f"{name}: {json.dumps(rec['means'])}; S {fmt(rec['S'])}, S00 {fmt(rec['S00'])}, S_c {fmt(rec['S_c'])}, scored {rec['scored']} (S_c {rec['scored_c']}); "
        f"halves S {[fmt(h['S']) for h in rec['halves']]}, S_c {[fmt(h['S_c']) for h in rec['halves']]}; {rec['seconds']:.1f} s, peak GPU {rec['peak_gpu_gb']:.2f} GB")  # fmt: skip
    if miss:
        stop(ITEMS / f"{name}.stopped.json", rec, f"{name}: " + "; ".join(miss), log)
    (ITEMS / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec


def done(name: str) -> dict[str, Any] | None:
    p = ITEMS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


# ---- stage 0 --------------------------------------------------------------------------------------------------------


@torch.no_grad()
def do_nothing(set_name: str, model: Any, F0: R.Model, M: R.Model, toks: Tensor, at: slice, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """The construction at F = M, the gate path against vpd.compute_ci, and H with every gate open and Delta in."""
    wd, gdiff, path, kl_open = vpd.weight_deltas(model), 0.0, 0.0, []
    for t in toks.split(cfg.batch):
        gF, gM = gates(model, F0.net, t), gates(model, M.net, t)
        ref = vpd.compute_ci(model, t, chunk=t.shape[0]).g
        gdiff = max([gdiff] + [float((gF[k] - gM[k]).abs().max()) for k in gM])
        path = max([path] + [float((gM[k] - ref[k].float()).abs().max()) for k in gM])
        ones = {k: torch.ones_like(v) for k, v in gM.items()}
        dm = {k: torch.ones(t.shape, device=t.device) for k in gM}
        lo = masked_logits(model, t, ones, wd, dm, autocast=False).float()[:, at]
        kl_open.append(R.kl(M.logits(t)[:, at], lo).cpu())
    rec = read(f"do_nothing_{set_name}", model, F0, M, toks, at, cfg, th, {"c": 0.0})
    m = rec["means"]
    out = {"gates_max_diff": gdiff, "gate_path_max_diff": path, "kl_open": float(torch.cat(kl_open).mean()), "means": m}
    miss = [f"KL(F||M) {m['kl_fm']:.3g}"] if not m["kl_fm"] <= th["kl_floor"] else []
    miss += [f"F's gates differ from M's by {gdiff:.3g}"] if gdiff != 0 else []
    miss += [f"K01 - f {m['K01'] - m['f']:.3g}"] if not abs(m["K01"] - m["f"]) <= th["kl_floor"] else []
    miss += [f"floor f {m['f']}"] if not (math.isfinite(m["f"]) and m["f"] > 0) else []
    miss += [f"gate path {path:.3g}"] if not path <= th["gate_path_tol"] else []
    miss += [f"open H {out['kl_open']:.3g} against f {m['f']:.3g}"] if not out["kl_open"] <= th["open_rel"] * m["f"] else []
    log(f"stage 0 {set_name}: {json.dumps(out)}{'; MISS ' + '; '.join(miss) if miss else '; checks hold'}")
    return out | {"miss": miss}


def stage0(sets: dict[str, tuple[Any, R.Model, R.Model, Tensor, slice]], part: str, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / f"stage0_{part}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return json.loads(path.read_text())
    out = {"experiment": EXPERIMENT} | {s: do_nothing(s, *v, cfg, th) for s, v in sets.items()}
    miss = [m for s in sets for m in out[s]["miss"]]
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss), log)
    path.write_text(json.dumps(out | {"done": True}, indent=1))
    log("STAGE 0 DONE")
    return out


# ---- the built job ----------------------------------------------------------------------------------------------------


def size2(F: R.Model, M: R.Model, keys: list[str]) -> float:
    return sum(float((F.net.get_submodule(k).weight.double() - M.net.get_submodule(k).weight.double()).pow(2).sum()) for k in keys)


def random_dw(W0: dict[str, Tensor], target2: float, seed: int) -> dict[str, Tensor]:
    """A Gaussian per decomposed matrix at the same relative size |dW_k| / |W0_k| = rho, sum |dW_k|^2 = target2."""
    rho, gen = math.sqrt(target2 / sum(float(w.double().pow(2).sum()) for w in W0.values())), torch.Generator().manual_seed(seed)
    out = {}
    for k, w in W0.items():
        r = torch.randn(w.shape, generator=gen, dtype=torch.float32).double()
        out[f"{k}.weight"] = (r * (rho * float(w.double().norm()) / r.norm())).float()
    return out


def kl_at(net: Any, M: R.Model, dw: dict[str, Tensor], toks: Tensor, at: slice, batch: int) -> float:
    F = R.Model(OT.with_delta(net, dw).net)
    return float(torch.cat([R.kl(F.logits(t)[:, at], M.logits(t)[:, at]).cpu() for t in toks.split(batch)]).mean())


def randoms(w: Any, model: Any, M: R.Model, sets: dict[str, tuple[Tensor, slice]], floors: dict[str, float], target2: float, cfg: Config, th: dict[str, Any]) -> None:
    net, keys = w.kl.net, list(w.kl.net.W0)
    for s, (toks, at) in sets.items():
        for seed in cfg.random_seeds:
            dw = random_dw(net.W0, target2, seed)
            ratio = sum(float(v.double().pow(2).sum()) for v in dw.values()) / target2
            if not abs(ratio - 1) <= th["size_rel"]:
                stop(ITEMS / f"random_{s}_size_{seed}.stopped.json", {"ratio": ratio}, f"random size ratio {ratio:.9g}", log)
            if done(name := f"random_{s}_size_{seed}") is None:
                F = R.Model(OT.with_delta(net, dw).net)
                read(name, model, F, M, toks, at, cfg, th, {"seed": seed, "x": 1.0, "size_ratio": ratio, "size2": size2(F, M, keys)})
                del F
            if done(name := f"random_{s}_kl_{seed}") is None:
                tgt = cfg.kl_target_floors * floors[s]
                m = PT.match(lambda x, dw=dw, toks=toks, at=at: kl_at(net, M, PT.scaled(dw, x), toks, at, cfg.batch), tgt, cfg.scale_cap, cfg.kl_rel_tol)
                log(f"{name}: matched x {m['x']:.4g} to KL {m['f']:.4g} (target {tgt:.4g}), {m['status']}")
                F = R.Model(OT.with_delta(net, PT.scaled(dw, m["x"])).net)
                read(name, model, F, M, toks, at, cfg, th, {"seed": seed, "x": m["x"], "kl_target": tgt, "match": m["status"], "match_evals": len(m["grid"]) + len(m["trace"])})
                del F
            torch.cuda.empty_cache()


def built_job(cfg: Config, th: dict[str, Any]) -> None:
    E240: Any = load_experiment("kstar_vpd_norm_matched", "run")  # World, scale_factors, scaled, dw2, saved
    E226 = E240.E226
    c = E240.scale_factors(cfg.target)
    w = E240.World(c, cfg.window_sets[0][1])
    model, keys, target2 = w.kl.net.model, list(w.kl.net.W0), E240.dw2(E240.saved(cfg.target))
    log(f"built job: c {c}; target |dW|^2 {target2:.9g}")
    base: dict[str, tuple[R.Model, R.Model, Tensor, slice]] = {}
    for s, item in cfg.window_sets:  # F at c = 1 only for its windows; the do-nothing is F times 0 on them
        F, M, toks, at, _ = E226.World.make(w, item)
        base[s] = (E240.scaled(M, F, 0.0), M, toks, at)
        del F
    st0 = stage0({s: (model, *v) for s, v in base.items()}, "built", cfg, th)
    sets = {s: (v[2], v[3]) for s, v in base.items()}
    del base
    for item in cfg.built:
        if done(item) is not None:
            log(f"{item}: done before")
            continue
        F, M, toks, at, rec = w.make(item)
        ratio = size2(F, M, keys) / target2
        if not abs(ratio - 1) <= th["size_rel"]:
            stop(ITEMS / f"{item}.stopped.json", {"size_ratio": ratio}, f"{item}: size ratio {ratio:.9g}", log)
        read(item, model, F, M, toks, at, cfg, th, {"c": c[item], "size_ratio": ratio, "kl_fm_saved": rec.get("kl_fm_saved")})
        del F
        torch.cuda.empty_cache()
    randoms(w, model, w.kl.M, sets, {s: st0[s]["means"]["f"] for s in sets}, target2, cfg, th)


# ---- the pairs job ----------------------------------------------------------------------------------------------------


def pairs_job(cfg: Config, th: dict[str, Any]) -> None:
    E232: Any = load_experiment("kstar_size_matched", "run")  # World (scaled states), load_config, E221, E205
    E221, E205 = E232.E221, E232.E205
    w = E232.World(E232.load_config()[0])
    model, M = w.net.model, w.M
    F0, toks0, at0, _ = w.make("do_nothing_c0")
    stage0({"pairs": (model, F0, M, toks0, at0)}, "pairs", cfg, th)
    del F0
    index: dict[str, dict[str, list[str]]] = {}
    for p in E205.SM.BUILT:
        reuse, new = w.pairs.names[p]
        larger = set(w.pairs.larger(p))
        names = {s: w.pairs.item(s, p).name if s in larger else E221.fname(s) for s in (*reuse, *new)}
        index[p] = {"reuse": [names[s] for s in reuse], "new": [names[s] for s in new], "scaled": sorted(names[s] for s in larger)}
        (RESULTS / "pairs_index.json").write_text(json.dumps(index, indent=1))
        for s, name in names.items():
            if done(name) is not None:
                continue
            F, toks, at, rec = w.make(name) if s in larger else E221.World.make(w, s)
            read(name, model, F, M, toks, at, cfg, th, {"state": s, "pair": p, "c": rec.get("c", 1.0), "kl_fm_saved": rec.get("kl_fm_saved")})
            del F
            torch.cuda.empty_cache()


def main(part: str) -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226
    ITEMS.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT} start, part {part}; settings {cfg}; thresholds {th}")
    {"built": built_job, "pairs": pairs_job}[part](cfg, th)
    log(f"{EXPERIMENT} {part} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("built", "pairs"))
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.part)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
