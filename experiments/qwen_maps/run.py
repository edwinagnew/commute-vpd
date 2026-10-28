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
     1, 10, 100 (both files at 100). Prediction 1 missed: stop (C not done).
  S  Σ on maths rows at W0; bases (not cached); per site and step every state's products; free check; placement
     arrays (--out/arrays). First layer × layers over s_limit_min: stop.
  D  per step K(D), then do-nothing; stop rules named in stopped.json.   K  controls (i) lower, upper, (ii) random;
     teeth; none at step 10: stop (K done only after).   M  least squares, in-span, adjoint.   G  accuracy (fine-tuned, adjoint, in-span).
  A  one fresh 20-step search;   H  step 100 if a split passed at 10; both under the spend rule.
Every read: a first-batch GPU peak over 70 GB halves the eval batch; an out-of-memory error halves it and reruns the
stage; the batch and the cumulative spend persist in results/state.json. A stop writes results/stopped.json (its
rule's name); the next job refuses to start while it exists, and a skipped C or K has its gate read again.
Usage (repo root): python experiments/qwen_maps/run.py --stages C,S,D,K,M,G,A,H [--rehearse] [--out /root/e44]
  [--e40 /workspace/e40] [--decomp DIR] [--usd-per-h 3.0] [--prior-usd SETUP_USD, first job only] [--svd-bases] [--waive NAMES, rehearsal]
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
from commute.qwen_sft import linears, mean_kl
from commute.qwen_teacher import adversary
from commute.qwen_vpd import DECOMP_DIR, SITES, parse_site

STORE_ERR_MAX = 1e-3  # stored-precision error of a split's applied pieces, relative to ‖δW‖_Σ (review, 2026-10-28)
GRAM_FACTOR = 10.0  # R: the Gram basis's residual may be at most this times the SVD basis's, plus 1e-12

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


def state_reading(job: Job, name: str, t: int, KD: float, share: float, fp32: bool = False
                  ) -> dict[str, Any]:
    """A state's readings at step t on both row sets, as nats and as ratios to W0 (stage C, same rows)."""
    C, R = job.res["C"], job.pred["rules"]
    m = job.read(name, t, "maths", (0,), job.ft(t, fp32), fp32=fp32)
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


def basis_of(job: Job, U_A: torch.Tensor, V_A: torch.Tensor) -> split.Basis:
    """Not cached (2026-10-28 review): fp32 X takes the disk the 34 GB of bases would have; H recomputes them."""
    how = job.cfg.basis
    return split.basis_from(qwen_maps.make_basis(U_A, how), qwen_maps.make_basis(V_A, how), job.cfg.tol)


def s_site(job: Job, s: str, steps: tuple[int, ...], with_halves: bool, fp32: bool = False) -> None:
    """Stage S for one site: its products at each step written to --out/S/step<t>/<site>.{safetensors,json} (fp32 X:
    5.6 GB per split and step over all sites, 34 GB for S, 5.6 GB more for H). A stored X whose applied pieces miss
    their float64 value by more than store_err_max of ‖δW‖_Σ stops the job."""
    files = [job.site_dir(t) / f"{s}.json" for t in steps]
    if all(f.is_file() for f in files):
        return
    U_A, V_A = job.alive.uv(s, DEV, F64)
    S, hb = job.sigma()[s].to(DEV), {}
    b = basis_of(job, U_A, V_A)
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
        worst = max(scalars["stored_error"].values())
        if worst > STORE_ERR_MAX:
            job.stop("stored_error", f"{s} step {t}: stored pieces miss float64 by {worst:.2e} of ‖δW‖_Σ")


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
           "stored_error_max": max(max(r["stored_error"].values()) for r in recs.values()),
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
            Ps, sv = split.full_svd(A)
            t2 = time.time()
            r_g, r_s = int((sg > cfg.tol * sg.max()).sum()), int((sv > cfg.tol * sv.max()).sum())

            def resid(B: torch.Tensor, A: torch.Tensor = A) -> float:  # what the kept directions leave of A
                return float((A - B @ (B.T @ A)).norm() / A.norm())

            # both residuals include the singular values cut at tol, so they are compared with each other
            out["gram"][f"{s}.{side}"] = {"rank_gram": r_g, "rank_svd": r_s, "residual_gram": resid(P[:, :r_g]),
                                          "residual_svd": resid(Ps[:, :r_s]), "s_gram": t1 - t0, "s_svd": t2 - t1}
            job.log(f"[R] {s} {side}: {out['gram'][f'{s}.{side}']}")
        M = torch.randn(U_A.shape[0], V_A.shape[0], generator=torch.Generator().manual_seed(0), dtype=F64).to(DEV)
        b = split.basis_from(split.gram_basis(U_A), split.gram_basis(V_A), cfg.tol)
        dU, dV = split.least_squares_split(M, b, U_A, V_A)
        err = split.dsigma(dU, dV, U_A, V_A) - (M - split.obstruction(M, b))
        out["exact"][s] = float(err.norm() / M.norm())
    out["holds"] = (all(g["rank_gram"] == g["rank_svd"] and g["residual_gram"] <= GRAM_FACTOR * g["residual_svd"]
                        + 1e-12 for g in out["gram"].values()) and all(e < cfg.tol for e in out["exact"].values()))
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
    holds = ms.control_w0(out, job.pred["rules"]["control_W0"])["holds"]
    job.put("C", out | {"done": holds or "control_W0" in job.waive})  # a miss reruns C (unless a rehearsal waives it)
    gate_C(job)


def gate_C(job: Job) -> None:
    """Prediction 1, read again whenever C is skipped as done."""
    if not ms.control_w0(job.res["C"], job.pred["rules"]["control_W0"])["holds"]:
        job.stop("control_W0", "control_W0 (prediction 1) missed")


def gate_K(job: Job) -> None:
    """Prediction 5, read again whenever K is skipped as done."""
    if score(job.res, job.pred).get("outcome", {}).get("stop", True):
        job.stop("teeth", "no teeth at step 10 (prediction 5): the line stops before stage M")


def stage_S(job: Job) -> None:
    t0 = time.time()
    for i, layer in enumerate(job.cfg.layers):
        for s in [x for x in job.sites if parse_site(x)[0] == layer]:
            s_site(job, s, job.cfg.steps, with_halves=True)
        job.log(f"[S] layer {layer}: {time.time() - t0:.0f} s, {job.peaks()}")
        if i == 0 and (time.time() - t0) * len(job.cfg.layers) > job.cfg.s_limit_min * 60:
            job.stop("S_time", f"S's first layer took {time.time() - t0:.0f} s: projected over {job.cfg.s_limit_min} min")
    job.forget()
    job.drop_sigma()  # 11 GB of host memory; H recomputes it
    job.put("S", {str(t): s_summary(job, t) for t in job.cfg.steps} | {"done": True})


def stage_D(job: Job) -> None:
    out = job.res.get("D", {})
    for t in job.cfg.steps:
        if str(t) not in out:
            KD = k_of_d(job, t)
            share = job.res["S"][str(t)]["share"]["do_nothing"]
            out[str(t)] = {"KD": KD, "do_nothing": state_reading(job, "do_nothing", t, KD, share)}
            job.put("D", out)
    job.put("D", out | {"done": True})


def stage_K(job: Job) -> None:
    out = read_states(job, "K", CONTROLS)
    gate_K(job)  # done only once the teeth hold (or a rehearsal waives them)
    job.put("K", out | {"done": True})


def stage_M(job: Job) -> None:
    job.put("M", read_states(job, "M", SPLITS) | {"done": True})


def stage_G(job: Job) -> None:
    out = job.res.get("G", {})
    job.forget()  # the fine-tuned weights' 5.6 GB of GPU, for generation's cache
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


GATES = {"C": gate_C, "K": gate_K}


def done(job: Job, stage: str) -> bool:
    """A stage with a finished results file is skipped ("done" true, or absent for the one-write stages A, H; C is
    done only when its control held); S also needs its files on --out (lost with a pod)."""
    if not job.res.get(stage, {}).get("done", stage in "AH" and stage in job.res):
        return False
    return stage != "S" or all((job.site_dir(t) / f"{s}.safetensors").is_file() for t in job.cfg.steps
                               for s in job.sites)


def refuse_if_stopped(job: Job) -> None:
    """A run that stopped stays stopped: the next job exits at once unless a rehearsal waives that stop."""
    f = job.dir / "stopped.json"
    if f.is_file():
        st = json.loads(f.read_text())
        if st.get("name") not in job.waive:
            sys.exit(f"{f} exists ({st.get('name')}: {st.get('why')}): the run stopped; the user decides what next")
        job.log(f"WAIVED: starting despite {f} ({st.get('name')})")


def run_stage(job: Job, st: str) -> None:
    """A stage, or its gate when it is done; an out-of-memory error halves the eval batch (persisted, so a resume
    never reuses the failed size) and reruns the stage, which resumes per state and step."""
    if done(job, st):
        job.log(f"[{st}] done earlier: skipped")
        if st in GATES:
            GATES[st](job)
        return
    job.stage = st
    while True:
        try:
            return STAGE_FNS[st](job)
        except torch.cuda.OutOfMemoryError as e:
            msg = f"out of memory in stage {st} at eval batch {job.batch}: {str(e)[:200]}"
            del e
            torch.cuda.empty_cache()
            if job.batch == 1:
                job.stop("oom", msg)
                return
            job.set_batch(job.batch // 2, msg)


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
    ap.add_argument("--prior-usd", type=float, default=0.0, help="setup spend; read only when results/state.json "
                    "does not exist (later jobs carry the cumulative spend there)")
    ap.add_argument("--waive", default="", help="rehearsal only: stop names to log and pass (control_W0,teeth)")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} missing: predictions are filed before the run")
    cfg = dataclasses.replace(REHEARSAL if a.rehearse else CFG, **({"basis": "svd"} if a.svd_bases else {}))
    job = Job(a, cfg, score)
    refuse_if_stopped(job)
    stages = [s for s in STAGES if s in a.stages.split(",")]
    job.log(f"[config] {json.dumps(dataclasses.asdict(cfg))}; stages {stages}; args {vars(a)}; "
            f"spent before ${job.spent_before:.2f}, eval batch {job.batch}")
    need = [job.e40 / "dW" / f for f in ("step_1.safetensors", "step_10.safetensors", "step_100.safetensors",
                                          "step_100_fp32.safetensors")]
    if cfg.random_dw == 0 and set(stages) - {"R"} and not all(p.is_file() for p in need):
        job.stop("dW", f"δW files missing under {job.e40 / 'dW'}: tell the user")
    for st in stages:
        run_stage(job, st)
        job.save_state()
        job.log(f"[{st}] finished: spent ${job.spent():.2f}, {job.peaks()}")
        if job.spent() >= cfg.stop_usd:
            job.stop("spend", f"spend ${job.spent():.2f} reached ${cfg.stop_usd}")


if __name__ == "__main__":
    main()
