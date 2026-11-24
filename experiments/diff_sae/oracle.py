"""The $0 oracle of the P2.75 spec (3.8, appendix A): before filing, for each run-1 side, rho_n, c, t and K, and the
oracle nu under both forms and both rules, for the plant's own direction (E212's plant) and for the top ten principal
directions of D on the read set (every other side), with latent scaling run by the pinned `closed_form_scalars` on fixed
unit readouts (`diff_sae.oracle_sae`). It diagnoses; it never vetoes (spec 3.8). Writes results/oracle.json.

The plant is exact: its D is the plant vector on the plant tokens and 0 elsewhere, so rho_n is a count over the
normaliser's actual sample (the run's own indices), c needs M's activation on the 512 L1 units only (weighted by how
often each is drawn into the scaler's actual task prefix), and nu_D's sums run over the plant tokens alone (f = 0 on
every other token). The interval test moves t across (-inf, inf) by scaling the plant and checks nu_D and its pass
against t / (t + 1) and -1 < t <= 3/7. nu_lit is estimated: exact on the window tokens, weighted samples of the first
Pile and fill rows of the scaler's prefixes for the rest. Every other side is estimated from the first rows of epoch 0
(the start of the scaler's actual prefixes; the normaliser's moments from the same rows, equal weight per dataset).

CPU only, the model loaded by E235's code on "cuda", which INPUT_MANIFEST_CPU=1 maps to the CPU (real fp32 forwards):
    INPUT_MANIFEST=<scratch>/oracle_inputs.txt INPUT_MANIFEST_CPU=1 fleet/cpu-run -c 8 --mem 16G -t 01:30:00 \\
        python experiments/diff_sae/oracle.py
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import diff_sae as DS
from commute.jobs import hard_exit, load_experiment
from commute.open_pair import jsonable

JOB: Any = load_experiment("diff_sae", "run")
OUT = Path(__file__).resolve().parent / "results" / "oracle.json"
SIDES = ("plant.A.c0", "head-r1.A.c0", "random_d0-r1.A.c0", "random_d1-r1.A.c0", "head-r5.A.c0", "head.B.c0", "gauss.B.c0")
T_SWEEP = (-3.0, -1.5, -1.01, -0.99, -0.5, 0.0, 0.2, 3 / 7 - 0.01, 3 / 7 + 0.01, 0.6, 1.0, 3.0)
N_ROWS, N_WIN = 16, 256  # the sample of the first Pile (and fill) rows and task windows, per side
log = JOB.log


def dry(job: Any, s: Any, chunks: int) -> tuple[list[dict[str, list[Tensor]]], Any]:
    """The token pieces of epoch 0's first `chunks` chunks, in dataset order, with no forward: `pair` is replaced by a
    recorder that returns (m, 2, 1) zeros, which E235's `make` concatenates and shuffles as it would activations."""
    sizes, make, _ = job.stream(s)
    rec: list[tuple[str, Tensor]] = []

    def fake(toks: Tensor, _s: Any, _task: bool) -> Tensor:
        pile = toks.untyped_storage().data_ptr() == job.pile.untyped_storage().data_ptr()
        rec.append(("pile" if pile else "task", toks))
        return torch.zeros(toks.shape[0] * (toks.shape[1] - 1), 2, 1)

    job.pair, out = fake, []
    try:
        for i in range(chunks):
            rec.clear()
            make(i)
            out.append({k: [t for d, t in rec if d == k] for k in ("pile", "task")})
    finally:
        del job.pair
    return out, sizes


def lengths(ch: dict[str, list[Tensor]], n_chunks: int) -> dict[str, int]:
    return {k: n_chunks * sum(t.shape[0] * (t.shape[1] - 1) for t in v) for k, v in ch.items()}


def fits(P: Any, sae: Any, rows: Tensor) -> dict[str, dict[str, Tensor]]:
    """Both forms' pinned fits (base and fine-tune targets) and our float64 sums on (m, 2, d) rows, one pass."""
    fns = [*DS.pinned_fits(P, sae, "lit"), *DS.pinned_fits(P, sae, "D"), lambda it: DS.sums64(sae, P, "lit", it), lambda it: DS.sums64(sae, P, "D", it)]
    r = DS.fan_out(rows.split(2048), fns)
    out = {}
    for j, form in enumerate(("lit", "D")):
        bM, bF = r[2 * j][0].double(), r[2 * j + 1][0].double()
        s64 = r[4 + j]
        b64 = (s64["B_M"] / (s64["C"] * 1.0), s64["B_F"] / s64["C"])  # unit readouts: D = |d|^2 = 1
        out[form] = {"beta_M": bM, "beta_F": bF, "nu": DS.nu_of(bM, bF), "nu64": DS.nu_of(*b64), "C": s64["C"], "checks": s64["checks"]}
    return out


def passes(f: dict[str, Tensor], i: int) -> dict[str, bool]:
    nu, bM = float(f["nu"][i]), float(f["beta_M"][i])
    return {"published": nu <= 0.3, "e2": nu <= 0.3 and bM >= 0}


def plant(job: Any, s: Any) -> dict[str, Any]:
    sz, P = job.sae, job.P
    units, (vec, _) = s.source[1], s.plant
    vec = vec.cpu().double()
    n_chunks = job.n_chunks
    first, _ = dry(job, s, 1)
    L = lengths(first[0], n_chunks)
    T, per = units.shape[1] - 1, first[0]["task"][0].shape[0]
    idx = dict(zip(L, DS.norm_indices(list(L.values()), sz.norm_tokens, sz.norm_seed), strict=True))
    o = idx["task"] % (L["task"] // n_chunks)
    hits = int(((o < per * T) & (o % T == T - 1)).sum())
    rho = hits / sz.norm_tokens  # mu_D = rho * p: the plant's share of the whole sample
    want = dict(zip(L, DS.sample_counts(list(L.values()), sz.ls_tokens), strict=True))
    lc = L["task"] // n_chunks
    need = math.ceil(want["task"] / lc)
    chunks, _ = dry(job, s, need)
    key = {tuple(u.tolist()): i for i, u in enumerate(units)}
    count: Counter[int] = Counter()
    for i, ch in enumerate(chunks):
        left = want["task"] - i * lc
        win = ch["task"][0]
        for w, row in enumerate(win):
            if w * T + T - 1 < left:
                count[key[tuple(row.tolist())]] += 1
    n_u = torch.tensor([count[i] for i in range(len(units))], dtype=torch.float64)
    with JOB.RUN.exact():
        hu = CC.site_acts(job.net, units, None, s.site).reshape(len(units), T, -1).double()  # positions 1..T
    p_norm, phat = float(vec.norm()), vec / vec.norm()
    c = float((n_u * (hu[:, -1] @ phat)).sum() / n_u.sum()) / p_norm
    t = c - rho
    rows = torch.stack([hu[:, -1], hu[:, -1] + vec], 1).repeat_interleave(n_u.long(), 0).float()

    def at(a: float) -> dict[str, Any]:
        q = a * vec
        mean, std = (rho * q).float(), (math.sqrt(rho * (1 - rho)) * q.abs()).float()
        r = rows.clone()
        r[:, 1] = r[:, 0] + q.float()
        sae = DS.oracle_sae((q / q.norm()).float()[None], mean, std, sz.target_rms)
        f = fits(P, sae, r)["D"]
        t_a = c / a - rho
        return {"a": a, "t": t_a, "nu_D": float(f["nu"][0]), "nu_D_f64": float(f["nu64"][0]), "nu_closed": DS.nu_plant(t_a),
                "pass": bool(float(f["nu"][0]) <= 0.3), "pass_interval": DS.plant_passes(t_a), "beta_M": float(f["beta_M"][0])}  # fmt: skip

    own = at(1.0)
    sweep = [at(c / (tt + rho)) for tt in T_SWEEP if tt + rho != 0]
    lit = plant_lit(job, s, first[0], want, L, n_u, hu, vec, rho)
    return {"rho_n": rho, "norm_hits": hits, "c": c, "t": t, "K": 1 - rho, "p_norm": p_norm, "plant_tokens_in_scaler": int(n_u.sum()),
            "nu_D": own["nu_D"], "nu_D_f64": own["nu_D_f64"], "nu_D_closed": DS.nu_plant(t), "beta_M_D": own["beta_M"],
            "pass_D": {"published": own["pass"], "e2": own["pass"] and own["beta_M"] >= 0}, **lit,
            "interval_test": {"points": sweep, "all_agree": all(p["pass"] == p["pass_interval"] for p in sweep),
                              "max_rel_nu": max(abs(p["nu_D"] - p["nu_closed"]) / max(abs(p["nu_closed"]), 1e-12) for p in sweep),
                              "max_rel_nu_f32_vs_f64": max(abs(p["nu_D"] - p["nu_D_f64"]) / max(abs(p["nu_D_f64"]), 1e-12) for p in sweep)}}  # fmt: skip


def plant_lit(job: Any, s: Any, ch: dict[str, list[Tensor]], want: dict[str, int], L: dict[str, int], n_u: Tensor, hu: Tensor,
              vec: Tensor, rho: float) -> dict[str, Any]:  # fmt: skip
    """nu_lit for the plant, float64 with weights: f = <N(h_F), p-hat>_+ on every token. The window tokens are exact
    (each unit's activations times its draw count); the Pile and fill tokens of the prefixes are N_ROWS rows of each,
    weighted to their token counts there. h_F = h_M except at the plant position."""
    sz, phat = job.sae, vec / vec.norm()
    mean = rho * vec
    std = math.sqrt(rho * (1 - rho)) * vec.abs()
    s_ = DS.scale_of(std, sz.target_rms)
    T = hu.shape[1]
    lc = L["task"] // job.n_chunks
    win_tokens = ch["task"][0].shape[0] * T
    n_fill = sum(max(0, min(lc, want["task"] - i * lc) - win_tokens) for i in range(math.ceil(want["task"] / lc)))
    strata: list[tuple[Tensor, Tensor, Tensor]] = []  # (h_M, h_F, weight) per token
    hF_u = hu.clone()
    hF_u[:, -1] += vec
    w_u = n_u[:, None].expand(-1, T).reshape(-1)
    strata.append((hu.reshape(-1, hu.shape[-1]), hF_u.reshape(-1, hu.shape[-1]), w_u))
    for name, rows, total in (("pile", ch["pile"][0][:N_ROWS], want["pile"]), ("fill", ch["task"][1][:N_ROWS] if len(ch["task"]) > 1 else None, n_fill)):
        if rows is None or total == 0:
            continue
        with JOB.RUN.exact():
            h = CC.site_acts(job.net, rows, None, s.site).double()
        strata.append((h, h, torch.full((len(h),), total / len(h), dtype=torch.float64)))
    num_M = num_F = den = 0.0
    for hM, hF, w in strata:
        f = (s_ * ((hF - mean) @ phat)).clamp_min(0)
        num_M += float((w * f * (s_ * ((hM - mean) @ phat))).sum())
        num_F += float((w * f * (s_ * ((hF - mean) @ phat))).sum())
        den += float((w * f * f).sum())
    bM, bF = num_M / den, num_F / den
    nu = bM / bF if bF != 0 else math.nan
    return {"nu_lit_est": nu, "beta_M_lit_est": bM, "pass_lit_est": {"published": nu <= 0.3, "e2": nu <= 0.3 and bM >= 0},
            "lit_strata_tokens": {"windows": int(w_u.sum()), "pile": want["pile"], "fill": n_fill}}  # fmt: skip


def dense(job: Any, s: Any) -> dict[str, Any]:
    """Every side but the plant: the top-10 principal directions of D on the read rows (uncentred: the change with its
    mean), the normaliser's moments from the first rows of epoch 0 (equal weight per dataset), K, and both forms'
    oracle nu on those rows, which are the start of the scaler's actual prefixes."""
    sz = job.sae
    with JOB.RUN.exact():
        h_read = job.pair(s.read, s, True).cpu()
    D = (h_read[:, 1] - h_read[:, 0]).double()
    first, _ = dry(job, s, 1)
    want = DS.sample_counts(list(lengths(first[0], job.n_chunks).values()), sz.ls_tokens)
    ch = first[0]
    task = ch["task"][0][:N_WIN]
    with JOB.RUN.exact():
        hp, ht = job.pair(ch["pile"][0][:N_ROWS], s, False).cpu(), job.pair(task, s, True).cpu()
    ratio = want[1] / want[0]
    ht = ht[: min(len(ht), round(len(hp) * ratio))]
    mean = 0.5 * ((hp[:, 1] - hp[:, 0]).double().mean(0) + (ht[:, 1] - ht[:, 0]).double().mean(0))
    sq = 0.5 * ((hp[:, 1] - hp[:, 0]).double().pow(2).mean(0) + (ht[:, 1] - ht[:, 0]).double().pow(2).mean(0))
    std = (sq - mean**2).clamp_min(0).sqrt()
    _, sv, Vt = torch.linalg.svd(D, full_matrices=False)
    U = Vt[:10] * torch.sign((D @ Vt[:10].T).mean(0))[:, None]
    sae = DS.oracle_sae(U.float(), mean.float(), std.float(), sz.target_rms)
    rows = torch.cat([hp, ht]).float()
    f = fits(job.P, sae, rows)
    dirs = []
    sc = DS.scale_of(std, sz.target_rms)
    for i in range(10):
        u = U[i]
        Dr = (rows[:, 1] - rows[:, 0]).double()
        fD = (sc * ((Dr - mean) @ u)).clamp_min(0)
        w = fD / fD.sum() if float(fD.sum()) > 0 else fD
        dD = float((w * (Dr @ u)).sum())
        c = float((w * (rows[:, 0].double() @ u)).sum()) / dD if dD else math.nan
        rho = float(mean @ u) / dD if dD else math.nan
        dirs.append({"share_of_D": float(sv[i] ** 2 / sv.pow(2).sum()), "rho": rho, "c": c, "t": c - rho, "nu_closed": DS.nu_plant(c - rho),
                     "nu_D": float(f["D"]["nu"][i]), "nu_lit": float(f["lit"]["nu"][i]), "pass_D": passes(f["D"], i), "pass_lit": passes(f["lit"], i)})  # fmt: skip
    return {"K": DS.capacity(mean, D), "energy": float(D.pow(2).sum(1).mean()), "sample_tokens": {"pile": len(hp), "task": len(ht)},
            "directions": dirs, "checks": {k: f[k]["checks"] for k in f}}  # fmt: skip


def main() -> int:
    t0, job = time.time(), JOB.load(rehearse=False, filed=False)
    res: dict[str, Any] = {"note": "spec 3.8 oracle; CPU fp32 forwards (INPUT_MANIFEST_CPU maps E235's 'cuda' to the CPU)", "sides": {}}
    for key in SIDES:
        it, t1 = JOB.RUN.Item.parse(key), time.time()
        s = job.side(it)
        out = JOB.RUN.Out(OUT.parent / "oracle_checks" / f"{key}.json", {"key": key})
        job.before(s, out)
        res["sides"][key] = (plant(job, s) if it.what == "plant" else dense(job, s)) | {"energy_read": out.d["energy"], "match": out.d.get("match"),
                                                                                         "minutes": (time.time() - t1) / 60}  # fmt: skip
        log(f"[oracle {key}] {json.dumps(jsonable(res['sides'][key]))[:2000]}")
        OUT.write_text(json.dumps(jsonable(res), indent=1))
    res["minutes"] = (time.time() - t0) / 60
    OUT.write_text(json.dumps(jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    hard_exit(main())
