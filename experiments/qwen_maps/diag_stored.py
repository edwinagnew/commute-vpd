"""E44 diagnostic ($0, cluster L40): why stage S stopped at `stored_error` (layers.2.mlp.up_proj, step 1, 3.82e-3 of
‖δW‖_Σ against a limit of 1e-3). For each site and each given δW, and for each stored split (ls = all alive columns,
lower, upper = the halves, as stage S builds them), one JSON line with:
  stored_error   the rule's number (`qwen_maps.stored_error`): X stored fp32, pieces formed in float64 then rounded
                 to fp32 (the fix of 2026-10-28). results/diag_stored.jsonl predates the fix: there this column is
                 the old number, pieces and dσ formed in fp32 (err_frob still measures that one)
  err_x64        X stored fp32, applied in float64 (fix "fp64 application")
  err_pieces64   pieces X V, Xᵀ U formed in float64 then stored fp32, applied in float64 (fix "store pieces")
  err_pieces32   the same pieces stored fp32, dσ formed in fp32 (store pieces, application as now)
  err_frob       stored_error's difference in Frobenius norm over ‖δW‖_F (is Σ's weighting the amplifier?)
  dsigma_smin, gain_max   smallest non-zero and largest gain of dσ; kappa = (gain_max / dsigma_smin)²
  amp            gain_max² ‖X‖_F / ‖dσ(pieces)‖_F: the factor by which fp32 rounding of X or of the pieces can grow
  piece_ratio    ‖(dU, dV)‖_F / ‖δW‖_F;  inspan     ‖dσ(pieces)‖_Σ / ‖δW‖_Σ
Expected if the hypothesis holds: every err_* ≈ 6e-8 · amp up to a small factor, err_x64 and err_pieces32 near
stored_error, err_pieces64 near 1e-7, and the worst sites those with the largest kappa (halves before ls).
The relative errors do not depend on δW's size, only on how it spreads over dσ's weak directions.
Usage (repo root): python experiments/qwen_maps/diag_stored.py --dw step2=PATH --dw step100=PATH --out FILE.jsonl
Resumes: (site, label) pairs already in --out are skipped. Writes only --out (Job is used for its rows, g statistics
and components; none of its logging or results files is touched).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from job import CFG, DEV, E40_RESULTS, F32, F64, HALVES, Job

from commute import qwen_maps, split
from commute.qwen_sft import load
from commute.qwen_vpd import DECOMP_DIR, SITES, load_target, parse_site

EPS32 = 2.0**-24


def measure(M: torch.Tensor, U: torch.Tensor, V: torch.Tensor, S: torch.Tensor, b: split.Basis, E: float
            ) -> dict[str, Any]:
    """One split of one δW (float64 on the GPU) over the columns (U, V) with their basis b."""
    X = split.least_squares_x(M, b)
    dU, dV = X @ V, X.T @ U
    exact = split.dsigma(dU, dV, U, V)
    X32, U32, V32 = X.float(), U.float(), V.float()

    def rel(applied: torch.Tensor) -> float:
        return (split.weighted_energy(applied.double() - exact, S) / E) ** 0.5

    applied32 = split.dsigma(X32 @ V32, X32.T @ U32, U32, V32).double()
    smin, gmax = split.dsigma_smin(b), float((b.s.max() ** 2 + b.t.max() ** 2) ** 0.5)
    xf, ef = float(X.norm()), float(exact.norm())
    return {"n": U.shape[1], "r_u": b.r_u, "r_v": b.r_v,
            "stored_error": qwen_maps.stored_error(X, U, V, S, E),
            "err_x64": rel(split.dsigma(X32.double() @ V, X32.double().T @ U, U, V)),
            "err_pieces64": rel(split.dsigma(dU.float().double(), dV.float().double(), U, V)),
            "err_pieces32": rel(split.dsigma(dU.float(), dV.float(), U32, V32)),
            "err_frob": float((applied32 - exact).norm() / M.norm()),
            "dsigma_smin": smin, "gain_max": gmax, "kappa": (gmax / smin) ** 2 if smin > 0 else float("inf"),
            "amp": gmax**2 * xf / ef if ef > 0 else 0.0, "eps_amp": EPS32 * gmax**2 * xf / ef if ef > 0 else 0.0,
            "piece_ratio": float((dU.norm() ** 2 + dV.norm() ** 2) ** 0.5 / M.norm()),
            "inspan": (split.weighted_energy(exact, S) / E) ** 0.5}


def site_bases(U_A: torch.Tensor, V_A: torch.Tensor, active: torch.Tensor, mean_g: torch.Tensor
               ) -> dict[str, tuple[torch.Tensor, torch.Tensor, split.Basis]]:
    """(U, V, basis) of each stored split, built as stage S builds them (run.py, s_site; Gram bases, tol)."""
    out = {}
    pos_all = torch.arange(U_A.shape[1], device=U_A.device)
    pos_h = qwen_maps.halves(active, mean_g)
    for name, pos in zip(("ls", *HALVES), (pos_all, *pos_h), strict=True):
        U, V = U_A[:, pos.to(U_A.device)], V_A[:, pos.to(U_A.device)]
        out[name] = (U, V, split.basis_from(qwen_maps.make_basis(U, CFG.basis), qwen_maps.make_basis(V, CFG.basis),
                                            CFG.tol))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dw", action="append", required=True, help="LABEL=PATH of a δW safetensors file (site keys)")
    ap.add_argument("--out", required=True, help="JSON lines, appended; done (site, label) pairs are skipped")
    ap.add_argument("--decomp", default=str(DECOMP_DIR))
    ap.add_argument("--e40-large", default=str(E40_RESULTS / "large" / "nowarmup"), help="holds gstats/")
    ap.add_argument("--first-layers", default="2", help="layers to run first (the stop was on layer 2)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["site"], r["label"]) for r in map(json.loads, out.read_text().splitlines())} if out.is_file() else set()
    dws = dict(x.split("=", 1) for x in a.dw)
    first = [int(x) for x in a.first_layers.split(",")]
    order = sorted(SITES, key=lambda s: (parse_site(s)[0] not in first, parse_site(s)[0]))
    todo = [s for s in order if any((s, lab) not in done for lab in dws)]
    t0 = time.time()
    print(f"{len(todo)} sites to do, {len(done)} (site, label) pairs done", flush=True)
    job = Job(argparse.Namespace(out=str(out.parent), e40="", e40_large=a.e40_large, waive="", prior_usd=0.0,
                                 decomp=a.decomp, usd_per_h=0.0), CFG, lambda r, p: {})
    model = load_target(DEV, F32)  # Σ at W0 on stage S's rows, as Job.sigma computes it (no log written)
    sigma = qwen_maps.input_moments(model, list(job.rows("maths").to(DEV).split(job.batch)), todo)
    del model
    torch.cuda.empty_cache()
    print(f"[{time.time() - t0:.0f} s] Σ for {len(sigma)} sites", flush=True)
    gm, gw = job.gstats("maths"), job.gstats("fineweb")
    idx = qwen_maps.alive_from_stats(gm, gw)
    assert sum(len(v) for v in idx.values()) == CFG.alive_total
    comps = job.comps()
    M_all = {lab: load(Path(p)) for lab, p in dws.items()}
    print(f"[{time.time() - t0:.0f} s] loaded components and δW {list(dws)}", flush=True)
    for s in todo:
        V, U = comps.vu(s)
        U_A, V_A = split.alive_uv(U, V, idx[s].to(V.device))
        i = idx[s].cpu()
        bases = site_bases(U_A, V_A, gm[f"{s}.active"][i], gm[f"{s}.mean_g"][i])
        S = sigma.pop(s).to(DEV)
        for lab in dws:
            if (s, lab) in done:
                continue
            M = M_all[lab][s].to(DEV, F64)
            E = split.weighted_energy(M, S)
            rec = {"site": s, "label": lab, "energy": E, "frob": float(M.norm()), "d_out": M.shape[0],
                   "d_in": M.shape[1], "splits": {n: measure(M, Ub, Vb, S, b, E) for n, (Ub, Vb, b) in bases.items()}}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            worst = max(rec["splits"].items(), key=lambda kv: kv[1]["stored_error"])
            print(f"[{time.time() - t0:.0f} s] {s} {lab}: worst {worst[0]} stored_error {worst[1]['stored_error']:.2e} "
                  f"x64 {worst[1]['err_x64']:.2e} pieces64 {worst[1]['err_pieces64']:.2e} "
                  f"kappa {worst[1]['kappa']:.2e} eps_amp {worst[1]['eps_amp']:.2e}", flush=True)
        del bases, S, U_A, V_A
    print(f"[{time.time() - t0:.0f} s] all sites done", flush=True)


if __name__ == "__main__":
    main()
