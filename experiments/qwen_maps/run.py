"""E44 (qwen_maps): do the pass test's controls fail on Qwen, and how do simple splits of E40's update read?
Plan: reports/plans/2026-10-28_plan_qwen-q0q1.md (the spec); rules in results/predictions.json (the run refuses to
start without it). Shared code: commute.qwen_maps (states, row loop, stage S's products), commute.maps_score (rules);
the job (model, rows, δW, S's files) and the config in job.py beside this file.

Stages, cheapest first, any subset by --stages; each writes results/<stage>.json as it finishes (D, K, M, G after
every state and step, so a resume skips what is done; S per site on --out) and rescores results/score.json; every
number goes to results/run.log.
  R  (cluster, $0) alive set; fp32 ranks of every alive U and V; Gram against SVD on one layer; split exactness on a
     random δW. No model.
  C  W0 alone: r1 (CI masks) on both row sets, stochastic seeds 0-2, replay of the four saved sources, KL to start at
     1, 10, 100 (both files at 100). Prediction 1 missed: stop.
  S  Σ on maths rows at W0; bases (cached on --out); per site and step every state's products; free check; placement
     arrays (--out/arrays). First layer × layers over s_limit_min: stop.
  D  per step K(D), then do-nothing; memory rule on its first batch.   K  controls (i) lower, upper, (ii) random;
     teeth; none at step 10: stop.   M  least squares, in-span, adjoint.   G  accuracy (fine-tuned, adjoint, in-span).
  A  one fresh 20-step search;   H  step 100 if a split passed at 10; both under the spend rule.
Usage (repo root): python experiments/qwen_maps/run.py --stages C,S,D,K,M,G,A,H [--rehearse] [--out /root/e44]
  [--e40 /workspace/e40] [--decomp DIR] [--usd-per-h 3.0] [--prior-usd 0.75] [--svd-bases]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
import time
from typing import Any

import torch
from job import (
    CFG,
    CONTROLS,
    DEV,
    E40_RESULTS,
    F32,
    F64,
    HALVES,
    MATHS_SOURCE,
    PRED,
    REHEARSAL,
    SPLITS,
    STAGES,
    WEB_SOURCES,
    Job,
    write_json,
    write_tensors,
)
from safetensors import safe_open

from commute import maps_score as ms
from commute import qwen_eval, qwen_maps, qwen_sets, split
from commute.qwen_sft import linears, load, mean_kl
from commute.qwen_teacher import adversary
from commute.qwen_vpd import DECOMP_DIR, SITES, parse_site

# ----------------------------------------------------------------------------- scoring


def score(res: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    R, sc, teeth = pred["rules"], {}, {}
    if "C" in res:
        sc["control_W0"] = ms.control_w0(res["C"], R["control_W0"])
    for t in ("1", "10"):
        d, k = res.get("D", {}).get(t), res.get("K", {}).get(t, {})
        if d is None or not all(n in k for n in CONTROLS):
            continue
        parts = {"i": ms.teeth_i(k["lower"], k["upper"], d["KD"], res["C"]["ci"]["maths"], R["teeth_i"]),
                 "ii": ms.teeth_ii(k["random"], R["teeth_ii"]), "iii": ms.teeth_iii(d["do_nothing"], R["teeth_iii"])}
        sc[f"teeth_{t}"], teeth[t] = parts, {n: p["holds"] for n, p in parts.items()}
    if "10" in res.get("D", {}):
        sc["do_nothing_r3w_10"] = ms.dn_r3_web(res["D"]["10"]["do_nothing"], R["teeth_iii"])
    if "10" in teeth:
        sc["outcome"] = ms.outcome(teeth, "10", "1")
    unscored = sc.get("outcome", {}).get("unscored_steps", [])
    for stage in ("M", "H"):
        for t, d in res.get(stage, {}).items():
            for n in SPLITS:
                if isinstance(d, dict) and n in d:
                    sc[f"pass_{n}_{t}"] = "unscored" if t in unscored else d[n]["pass"]
    return sc


# ----------------------------------------------------------------------------- shared steps


def state_reading(job: Job, name: str, t: int, KD: float, share: float, fp32: bool = False, on_batch: Any = None
                  ) -> dict[str, Any]:
    """A state's readings at step t on both row sets, as nats and as ratios to W0 (stage C, same rows)."""
    C, R = job.res["C"], job.pred["rules"]
    m = job.read(name, t, "maths", (0,), job.ft(t, fp32), on_batch=on_batch, fp32=fp32)
    w = job.read(name, t, "web", (0,), None, fp32=fp32)
    st = {"maths": m, "web": w, "share": share, "r1m": m["ci"] / C["ci"]["maths"], "r1w": w["ci"] / C["ci"]["web"],
          "stoch_m": m["stoch_0"] / C["stoch"]["maths"]["0"], "stoch_w": w["stoch_0"] / C["stoch"]["web"]["0"],
          "r3m": m[f"replay_{MATHS_SOURCE}"] / C["replay"][MATHS_SOURCE],
          "r3w": statistics.mean(w[f"replay_{n}"] / C["replay"][n] for n in WEB_SOURCES),
          "r2": ms.kept(KD, m["K"], C["ci"]["maths"], C["spread"], R["kept"])}
    st["pass"] = ms.passes(st, R["pass"])
    return st


def k_of_d(job: Job, t: int, fp32: bool = False) -> float:
    """K(D) at step t: KL(fine-tuned ‖ the unchanged decomposition CI-masked, Γ on W0's activations)."""
    return job.read("W0", None, "maths", (), job.ft(t, fp32), sources=False)["K"]


def read_states(job: Job, stage: str, names: tuple[str, ...]) -> dict[str, Any]:
    out = job.res.get(stage, {})
    for t in job.cfg.steps:
        for n in names:
            if n not in out.setdefault(str(t), {}):
                share = job.res["S"][str(t)]["share"][n]
                out[str(t)][n] = state_reading(job, n, t, job.res["D"][str(t)]["KD"], share)
                job.put(stage, out)
    return out


def basis_of(job: Job, s: str, U_A: torch.Tensor, V_A: torch.Tensor) -> split.Basis:
    """The alive set's basis, cached on --out (dropped, with a log line, when the disk is short)."""
    path, tol, how = job.out / "bases" / f"{s}.safetensors", job.cfg.tol, job.cfg.basis
    if path.is_file():
        b = load(path, DEV)
        return split.basis_from((b["P"], b["s"]), (b["Q"], b["t"]), tol)
    bu, bv = qwen_maps.make_basis(U_A, how), qwen_maps.make_basis(V_A, how)
    try:
        write_tensors(path, {"P": bu[0].cpu(), "s": bu[1].cpu(), "Q": bv[0].cpu(), "t": bv[1].cpu()})
    except RuntimeError as e:
        job.log(f"[S] basis not cached: {e}")
    return split.basis_from(bu, bv, tol)


def s_site(job: Job, s: str, steps: tuple[int, ...], with_halves: bool, fp32: bool = False) -> None:
    """Stage S for one site: its products at each step written to --out/S/step<t>/<site>.{safetensors,json}."""
    files = [job.site_dir(t) / f"{s}.json" for t in steps]
    if all(f.is_file() for f in files):
        return
    U_A, V_A = job.alive.uv(s, DEV, F64)
    S, hb = job.sigma()[s].to(DEV), {}
    b = basis_of(job, s, U_A, V_A)
    if with_halves:
        st, idx = job.gstats("maths"), job.alive.idx[s].cpu()
        for h, pos in zip(HALVES, qwen_maps.halves(st[f"{s}.active"][idx], st[f"{s}.mean_g"][idx]), strict=True):
            pos = pos.to(DEV)
            Uh, Vh = U_A[:, pos], V_A[:, pos]
            mk = job.cfg.basis
            hb[h] = (pos, split.basis_from(qwen_maps.make_basis(Uh, mk), qwen_maps.make_basis(Vh, mk), job.cfg.tol))
    RU, RV = qwen_maps.random_pieces(tuple(U_A.shape), tuple(V_A.shape), 0, s, DEV)
    for t, f in zip(steps, files, strict=True):
        M = job.dw(t, fp32)[s].to(DEV, F64)
        arrays, scalars = qwen_maps.site_products(M, U_A, V_A, S, b, hb, RU.double(), RV.double())
        write_tensors(f.with_suffix(".safetensors"), qwen_maps.to_disk(arrays))
        write_json(f, scalars)


def s_summary(job: Job, t: int) -> dict[str, Any]:
    """Pooled over the sites (ratio of sums): each state's Δ change ‖Δ′ − Δ₀‖_Σ / ‖δW‖_Σ, the outside-both share and
    its null weighted by E‖δW a‖², the second-order term; the placement arrays gathered into one file."""
    recs = {s: job.site_rec(t, s) for s in job.sites}
    E = [r["energy"] for r in recs.values()]

    def pool(f: Any) -> float:
        return ms.pooled([f(r) for r in recs.values()], E)

    out = {"share": {n: math.sqrt(pool(lambda r, n=n: r["delta"][n])) for n in next(iter(recs.values()))["delta"]},
           "outside_both": pool(lambda r: r["outside_both"] * r["energy"]),
           "outside_both_null": pool(lambda r: r["outside_both_null"] * r["energy"]),
           "ls_second": math.sqrt(pool(lambda r: r["ls_second"])),
           "half_piece_energy": {h: sum(r["half_piece_energy"].get(h, 0.0) for r in recs.values()) for h in HALVES},
           "energy": sum(E), "per_site": recs}
    arrays = {}
    for s in job.sites:
        with safe_open(str(job.site_dir(t) / f"{s}.safetensors"), framework="pt") as f:
            arrays |= {f"{s}.{k}": f.get_tensor(k) for k in f.keys() if k.startswith("pe_")}  # noqa: SIM118
    write_tensors(job.out / "arrays" / f"placement_step{t}.safetensors", arrays)
    return out


def accuracy(job: Job, gen: Any, tok: Any, name: str, t: int, fp32: bool = False) -> dict[str, Any]:
    """Unmasked greedy strict accuracy (drop rule) on the first acc_problems GSM8K test problems, bf16 weights."""
    fn, W0 = job.change(name, t, fp32), {s: lin.weight.data for s, lin in linears(job.model).items()}
    with torch.no_grad():
        for s, lin in linears(gen).items():
            dW = fn(s)[0]
            lin.weight.copy_(W0[s] if dW is None else W0[s] + dW)
    items = qwen_sets.load()["test"][: job.cfg.acc_problems]
    (job.out / "acc").mkdir(parents=True, exist_ok=True)
    recs = qwen_eval.score(tok, gen, items, 1800.0, job.out / "acc" / f"{name}_{t}.jsonl", f"{name} {t}",
                           job.cfg.gen_batch)
    return qwen_eval.summary(recs)


def spend_ok(job: Job, stage: str) -> bool:
    minutes = dict(job.cfg.stage_min)[stage]
    ok = job.spent() + minutes / 60 * job.a.usd_per_h <= job.cfg.spend_limit_usd
    job.log(f"[{stage}] spend ${job.spent():.2f} + {minutes} min: {'run' if ok else 'skip (spend rule)'}")
    return ok


# ----------------------------------------------------------------------------- stages


def stage_R(job: Job) -> None:
    cfg, alive = job.cfg, job.alive

    def rank32(A: torch.Tensor) -> int:
        sv = torch.linalg.svdvals(A)
        return int((sv > cfg.tol * sv.max()).sum())

    out: dict[str, Any] = {"alive": alive.count(), "gram": {}, "exact": {}, "done": False}
    for s in [x for x in SITES if parse_site(x)[0] == cfg.r_layer]:
        U_A, V_A = alive.uv(s, DEV, F64)
        for side, A in (("U", U_A), ("V", V_A)):
            t0 = time.time()
            P, sg = split.gram_basis(A)
            t1 = time.time()
            sv = split.full_svd(A)[1]
            r_g, r_s = int((sg > cfg.tol * sg.max()).sum()), int((sv > cfg.tol * sv.max()).sum())
            Pr = P[:, :r_g]
            resid = float((A - Pr @ (Pr.T @ A)).norm() / A.norm())
            out["gram"][f"{s}.{side}"] = {"rank_gram": r_g, "rank_svd": r_s, "residual": resid,
                                          "s_gram": t1 - t0, "s_svd": time.time() - t1}
            job.log(f"[R] {s} {side}: {out['gram'][f'{s}.{side}']}")
        M = torch.randn(U_A.shape[0], V_A.shape[0], generator=torch.Generator().manual_seed(0), dtype=F64).to(DEV)
        b = split.basis_from(split.gram_basis(U_A), split.gram_basis(V_A), cfg.tol)
        dU, dV = split.least_squares_split(M, b, U_A, V_A)
        err = split.dsigma(dU, dV, U_A, V_A) - (M - split.obstruction(M, b))
        out["exact"][s] = float(err.norm() / M.norm())
    out["holds"] = (all(g["rank_gram"] == g["rank_svd"] and g["residual"] < cfg.tol for g in out["gram"].values())
                    and all(e < cfg.tol for e in out["exact"].values()))
    job.put("R", out)  # the check first, so a time-out in the ranks below keeps it
    if not out["holds"]:
        job.log("[R] MISSED: the Gram bases disagree with SVD; the pod run passes --svd-bases (S 60-90 min, no H)")
    out["ranks_fp32"] = {}
    for i, s in enumerate(SITES):
        U_A, V_A = alive.uv(s, DEV, F32)
        out["ranks_fp32"][s] = {"r_u": rank32(U_A), "d_out": U_A.shape[0], "r_v": rank32(V_A), "d_in": V_A.shape[0]}
        if i % 28 == 27:
            job.log(f"[R] fp32 ranks: {i + 1} of {len(SITES)} sites, {job.peaks()}")
    job.put("R", out | {"done": True})


def stage_C(job: Job) -> None:
    cfg, seeds = job.cfg, job.cfg.stoch_seeds
    m = job.read("W0", None, "maths", seeds, None)
    w = job.read("W0", None, "web", seeds, None)
    batches = list(job.rows("maths").to(DEV).split(job.batch))
    kl = {}
    for t, fp32 in [(t, False) for t in (*cfg.steps, cfg.step_h)] + [(cfg.step_h, True)]:
        kl[f"{t}_fp32" if fp32 else str(t)] = mean_kl(job.model, batches, None, job.ft(t, fp32))
        job.forget()
    sm = [m[f"stoch_{s}"] for s in seeds]
    out = {"ci": {"maths": m["ci"], "web": w["ci"]}, "spread": max(sm) - min(sm), "kl_to_start": kl,
           "stoch": {"maths": {str(s): m[f"stoch_{s}"] for s in seeds}, "web": {str(s): w[f"stoch_{s}"] for s in seeds}},
           "replay": {n: w[f"replay_{n}"] for n in WEB_SOURCES} | {MATHS_SOURCE: m[f"replay_{MATHS_SOURCE}"]}}
    job.put("C", out)
    if not ms.control_w0(out, job.pred["rules"]["control_W0"])["holds"]:
        job.stop("control_W0 (prediction 1) missed")


def stage_S(job: Job) -> None:
    t0 = time.time()
    for i, layer in enumerate(job.cfg.layers):
        for s in [x for x in job.sites if parse_site(x)[0] == layer]:
            s_site(job, s, job.cfg.steps, with_halves=True)
        job.log(f"[S] layer {layer}: {time.time() - t0:.0f} s, {job.peaks()}")
        if i == 0 and (time.time() - t0) * len(job.cfg.layers) > job.cfg.s_limit_min * 60:
            job.stop(f"S's first layer took {time.time() - t0:.0f} s: projected over {job.cfg.s_limit_min} min")
    job.forget()
    job.drop_sigma()  # 11 GB of host memory; H recomputes it
    job.put("S", {str(t): s_summary(job, t) for t in job.cfg.steps} | {"done": True})


def stage_D(job: Job) -> None:
    out = job.res.get("D", {"eval_batch": job.batch})
    torch.cuda.reset_peak_memory_stats()

    def memory_rule(i: int) -> None:
        peak = torch.cuda.max_memory_allocated() / 1e9
        if i == 0 and "memory" not in out:
            out["memory"] = {"first_batch_peak_gb": peak, "halved": peak > job.cfg.mem_halve_gb}
            job.batch = max(1, job.batch // 2) if peak > job.cfg.mem_halve_gb else job.batch
            out["eval_batch"] = job.batch
            job.log(f"[D] memory rule: first batch peak {peak:.1f} GB, eval batch now {job.batch}")

    for t in job.cfg.steps:
        if str(t) not in out:
            KD = k_of_d(job, t)
            share = job.res["S"][str(t)]["share"]["do_nothing"]
            out[str(t)] = {"KD": KD, "do_nothing": state_reading(job, "do_nothing", t, KD, share, on_batch=memory_rule)}
            job.put("D", out)
    job.put("D", out | {"done": True})


def stage_K(job: Job) -> None:
    job.put("K", read_states(job, "K", CONTROLS) | {"done": True})
    if score(job.res, job.pred).get("outcome", {}).get("stop", True):
        job.stop("no teeth at step 10 (prediction 5): the line stops before stage M")


def stage_M(job: Job) -> None:
    job.put("M", read_states(job, "M", SPLITS) | {"done": True})


def stage_G(job: Job) -> None:
    out = job.res.get("G", {})
    w0 = job.large / "step0_test.jsonl"  # E40's W0 answers, the same prompts and code
    if w0.is_file():
        recs = [json.loads(x) for x in w0.read_text().splitlines()]
        out["W0"] = qwen_eval.summary([r for r in recs if r["idx"] < job.cfg.acc_problems])
    tok, gen = qwen_eval.load_model()
    for t in job.cfg.steps:
        for n in ("do_nothing", "adjoint", "inspan"):
            if f"{n}_{t}" not in out:
                out[f"{n}_{t}"] = accuracy(job, gen, tok, n, t)
                job.put("G", out)
    del gen
    job.put("G", out | {"done": True})


def stage_A(job: Job) -> None:
    if not spend_ok(job, "A"):
        return job.put("A", {"skipped": "spend rule"})
    sc, M = score(job.res, job.pred), job.res.get("M", {})
    passing = [(1.5 - max(M[t][n]["r3m"], M[t][n]["r3w"]), n, int(t)) for t in ("1", "10") for n in SPLITS
               if sc.get(f"pass_{n}_{t}") == "pass"]
    _, name, t = min(passing) if passing else (0.0, "adjoint", 10)
    rows = list(job.rows("maths").to(DEV).split(1))
    with qwen_maps.applied(job.model, job.comps(), job.alive, job.change(name, t)):
        kl, src = adversary(job.model, rows, job.gamma(), job.comps(), 0, log=job.log, steps=job.cfg.adv_steps)
    write_tensors(job.out / "A_sources.safetensors", src)
    ref = job.pred["rules"]["fresh_search"]["w0_maths_kl"]
    replay = M.get(str(t), {}).get(name, {}).get("r3m")
    job.put("A", {"state": name, "step": t, "kl": kl, "ratio": kl / ref, "replay_r3m": replay,
                  "replay_blind": kl / ref > 1.5 and replay is not None and replay <= 1.5})


def stage_H(job: Job) -> None:
    sc, t = score(job.res, job.pred), job.cfg.step_h
    passing = tuple(n for n in SPLITS if sc.get(f"pass_{n}_10") == "pass") or (SPLITS if job.cfg.rehearse else ())
    if not passing:  # (a rehearsal runs H's code on every split)
        return job.put("H", {"skipped": "no split passed at step 10"})
    if not spend_ok(job, "H"):
        return job.put("H", {"skipped": "spend rule"})
    for s in job.sites:
        s_site(job, s, (t,), with_halves=False, fp32=True)
    summ = s_summary(job, t)
    KD = k_of_d(job, t, fp32=True)
    out: dict[str, Any] = {"summary": summ, "KD": KD}
    for n in ("do_nothing", *passing):
        out[n] = state_reading(job, n, t, KD, summ["share"][n], fp32=True)
    tok, gen = qwen_eval.load_model()
    out["accuracy"] = {n: accuracy(job, gen, tok, n, t, fp32=True) for n in ("do_nothing", *passing)}
    job.put("H", {str(t): out})


STAGE_FNS = dict(zip(STAGES, (stage_R, stage_C, stage_S, stage_D, stage_K, stage_M, stage_G, stage_A, stage_H),
                     strict=True))


def done(job: Job, stage: str) -> bool:
    """A stage with a finished results file is skipped ("done" true, or absent for the one-write stages C, A, H); S
    also needs its files on --out (lost with a pod)."""
    if not job.res.get(stage, {}).get("done", stage in "CAH" and stage in job.res):
        return False
    return stage != "S" or all((job.site_dir(t) / f"{s}.safetensors").is_file() for t in job.cfg.steps
                               for s in job.sites)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="C,S,D,K,M,G,A,H")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--svd-bases", action="store_true")
    ap.add_argument("--out", default="/root/e44")
    ap.add_argument("--e40", default="/workspace/e40")
    ap.add_argument("--e40-large", default=str(E40_RESULTS / "large" / "nowarmup"))
    ap.add_argument("--decomp", default=str(DECOMP_DIR))
    ap.add_argument("--usd-per-h", type=float, default=3.0)
    ap.add_argument("--prior-usd", type=float, default=0.0)
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} missing: predictions are filed before the run")
    cfg = dataclasses.replace(REHEARSAL if a.rehearse else CFG, **({"basis": "svd"} if a.svd_bases else {}))
    job = Job(a, cfg, score)
    stages = [s for s in STAGES if s in a.stages.split(",")]
    job.log(f"[config] {json.dumps(dataclasses.asdict(cfg))}; stages {stages}; args {vars(a)}")
    need = [job.e40 / "dW" / f for f in ("step_1.safetensors", "step_10.safetensors", "step_100.safetensors",
                                          "step_100_fp32.safetensors")]
    if cfg.random_dw == 0 and set(stages) - {"R"} and not all(p.is_file() for p in need):
        job.stop(f"δW files missing under {job.e40 / 'dW'}: tell the user")
    for st in stages:
        if done(job, st):
            job.log(f"[{st}] done earlier: skipped")
            continue
        job.stage = st
        try:
            STAGE_FNS[st](job)
        except torch.cuda.OutOfMemoryError as e:
            job.stop(f"out of memory in stage {st} at eval batch {job.batch}: {e}")
        job.log(f"[{st}] finished: spent ${job.spent():.2f}, {job.peaks()}")
        if job.spent() >= cfg.stop_usd:
            job.stop(f"spend ${job.spent():.2f} reached ${cfg.stop_usd}")


if __name__ == "__main__":
    main()
