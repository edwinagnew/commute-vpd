"""Step 0 of the Qwen instrument check (cell P3.X, E###): three tables from saved files, CPU only, on the new
decomposition p-9707d6ff. Plan: reports/plans/2026-11-11_plan_qwen-instrument-check-step0.md; rules in
results/predictions_step0.json (the run refuses to start unless that file is committed unchanged).

    sbatch --comment='E### step 0' cluster/qwen_instrument_check_step0.sbatch

Table 1, per matrix: the alive count n, the rank of the alive read vectors (V_A, d_in side) and write vectors (U_A, d_out
side) at the tolerance every Qwen split uses (`qwen_d3.basis`: gram bases, BASIS_TOL), and the share of a change the
least-squares split must leave outside the subcomponents. That split moves vectors (dU V_Aᵀ + U_A dVᵀ, `split.py`), so
what it cannot reach is P_U⊥ M P_V⊥, and for an iid Gaussian M the expected share is `split.gaussian_null`,
(1 − r_U/d_out)(1 − r_V/d_in). Beside it: one Gaussian draw per matrix, measured; per saved update, its own outside
share and the same null pooled by its ‖δW_s‖² (E93's pooling).
Table 2, per update and matrix: ‖δW_s‖ beside ‖Δ_s‖ = the bundle's relative error (reconstruction_check.csv, matched by
E131 to 8.6e-7) × ‖W0_s‖ from the Hugging Face file.
Table 3, per update: stored output changes beside the decomposition's own masked error on the same rows; no forward.
Per-matrix records go to results/step0_sites.jsonl as each matrix ends, and a resubmitted job skips those written.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from commute import qwen_maps, split
from commute.jobs import EXPERIMENTS, hard_exit, require_filed, write_result
from commute.qwen_d3 import ALIVE_TOTAL_OF, BASIS_TOL, GSTATS_OF, ROW_LEN
from commute.qwen_sft import load
from commute.qwen_vpd import HF_REVISION, SITES, decomp_id, parse_site

NEW = "p-9707d6ff"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT, SITES_LOG = RESULTS / "predictions_step0.json", RESULTS / "step0.json", RESULTS / "step0_sites.jsonl"
DATA = Path.home() / "commute-data" / "artifacts"
DECOMP = DATA / "qwen3-1.7b-p-9707d6ff-torch"
RECON_CSV = DATA / "p-9707d6ff-share/data/p-9707d6ff-share/reconstruction_check.csv"
W0_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / HF_REVISION
E93, E91 = EXPERIMENTS / "qwen_d1_table/results", EXPERIMENTS / "qwen_size_grid/results/size_grid.json"
E132, E45 = EXPERIMENTS / "qwen_regating_new/results", EXPERIMENTS / "qwen_updates/results"
W0_REFS = EXPERIMENTS / "qwen_new_decomp/results/w0_refs_stage1.json"  # job 56760
E93_STATES = ("a1_s0", "a1_s1", "a2_s42", "a2_s142", "R_s0", "R_s70", "N_s0", "N_s70", "F_s0", "u1", "u2")
E45_STATES = ("gold_s0", "a2long_s0", "a2long_s1")  # E57's and E132's, beside E40 step 10
GAUSS_TOL, OBSTRUCTION_TOL = 5e-3, 1e-9  # per-matrix controls; their sources are in the predictions file
FREE, CONSTRAINED = 0.8, 0.5  # predictions_step0.json T1_real_vs_random


def update_files() -> dict[str, Path]:
    """The 15 saved updates: E93's eleven (files as E93 read them), E40 step 10, and E45's three that E132 read."""
    files = {n: EXPERIMENTS / json.loads((E93 / n / "state.json").read_text())["file"] for n in E93_STATES}
    files["e40"] = EXPERIMENTS / "qwen_sft/results/large/nowarmup_regen/dW/step_10.safetensors"
    return files | {n: E45 / "large/dW" / n / "step_100.safetensors" for n in E45_STATES}


class Tensors:
    """Every key of a set of safetensors files, read one tensor (or one leading slice) at a time."""

    def __init__(self, files: list[Path]):
        self.handles = {k: h for p in files for h in [safe_open(str(p), framework="pt")] for k in h.keys()}  # noqa: SIM118

    def get(self, key: str) -> Tensor:
        return self.handles[key].get_tensor(key)

    def layer(self, key: str, i: int) -> Tensor:
        return self.handles[key].get_slice(key)[i : i + 1][0]


def alive_sets() -> dict[str, Tensor]:
    """Alive as every Qwen reading on new defines it (qwen_d3.py:228-231, E132 run.py:42-43): g > 0.01 somewhere in
    W0's maths or fineweb block (`qwen_maps.alive_from_stats`, qwen_maps.py:64-66), from job 56760's statistics."""
    st = [load(GSTATS_OF[NEW] / f"W0_{b}.safetensors") for b in ("maths", "fineweb")]
    alive = qwen_maps.alive_from_stats(*st)
    total = sum(len(v) for v in alive.values())
    assert total == ALIVE_TOTAL_OF[NEW], f"alive {total} != {ALIVE_TOTAL_OF[NEW]}"
    return alive


def outside_sq(M: Tensor, Po: Tensor, Qo: Tensor) -> float:
    """‖Poᵀ M Qo‖², the squared norm of M in the unreachable block: `split.obstruction` without the full rotation."""
    if Po.shape[1] == 0 or Qo.shape[1] == 0:
        return 0.0
    return float(((Po.T @ M.double()) @ Qo).pow(2).sum())


def site_bases(comps: Tensors, site: str, idx: Tensor) -> tuple[split.Basis, Tensor, Tensor]:
    layer, kind = parse_site(site)
    V, U = comps.layer(f"components.stacks.{kind}.0", layer), comps.layer(f"components.stacks.{kind}.1", layer)
    U_A, V_A = split.alive_uv(U, V, idx)  # (d_out, n), (d_in, n), fp64
    b = split.basis_from(qwen_maps.make_basis(U_A, "gram"), qwen_maps.make_basis(V_A, "gram"), BASIS_TOL)
    return b, b.P[:, b.s == 0], b.Q[:, b.t == 0]


def gauss_check(site: str, b: split.Basis, Po: Tensor, Qo: Tensor, d_out: int, d_in: int) -> dict[str, float]:
    """One iid Gaussian matrix per site (seed = the site's index): its measured outside share beside the formula."""
    gen = torch.Generator().manual_seed(SITES.index(site))
    G = torch.randn(d_out, d_in, generator=gen, dtype=torch.float64)
    rec = {"g_sq": float(G.pow(2).sum()), "g_out": outside_sq(G, Po, Qo), "null": split.gaussian_null(b, d_out, d_in)}
    gap = abs(rec["g_out"] / rec["g_sq"] - rec["null"])
    assert gap <= GAUSS_TOL, f"{site}: Gaussian outside share off its null by {gap:.2e}"
    return rec


def site_record(site: str, comps: Tensors, w0: Tensors, dws: dict[str, Tensors], idx: Tensor, rel: float,
                check: bool) -> dict[str, Any]:
    """Table 1 and Table 2 for one matrix. `check`: the cheap outside share of the first update against
    `split.obstruction_fraction`, done (and recorded as `checked`) only where both complements are non-empty."""
    t0 = time.time()
    b, Po, Qo = site_bases(comps, site, idx)
    d_out, d_in = b.P.shape[0], b.Q.shape[0]
    n = len(idx)
    assert b.r_u <= min(n, d_out) and b.r_v <= min(n, d_in), (site, n, b.r_u, b.r_v)
    w0_norm = float(w0.get(f"model.{site}.weight").double().norm())
    rec: dict[str, Any] = {"site": site, "d_out": d_out, "d_in": d_in, "n_alive": n, "r_write": b.r_u, "r_read": b.r_v,
                           "w0_norm": w0_norm, "delta_norm": rel * w0_norm, "updates": {}, "checked": False}
    rec |= gauss_check(site, b, Po, Qo, d_out, d_in)
    check = check and Po.shape[1] > 0 and Qo.shape[1] > 0
    for name, t in dws.items():
        M = t.get(site).double()
        sq, out = float(M.pow(2).sum()), outside_sq(M, Po, Qo)
        if check and sq > 0:
            ref = split.obstruction_fraction(M, b)
            assert abs(out / sq - ref) <= OBSTRUCTION_TOL, f"{site} {name}: {out / sq} vs {ref}"
            check, rec["checked"] = False, True
        rec["updates"][name] = {"sq": sq, "out": out}
    rec["seconds"] = time.time() - t0
    return rec


def recon_errors() -> dict[str, float]:
    lines = RECON_CSV.read_text().splitlines()[1:]
    return {ln.split(",")[0].removeprefix("model."): float(ln.split(",")[-1]) for ln in lines}


def pooled(recs: list[dict[str, Any]], num: Any, den: Any) -> float:
    d = sum(den(r) for r in recs)
    return sum(num(r) for r in recs) / d if d > 0 else float("nan")


def table1(recs: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Shares pooled over the 196 matrices (iid: by d_in·d_out; per update: by its ‖δW_s‖², as E93), per kind for iid."""
    size = lambda r: r["d_in"] * r["d_out"]
    general = lambda r: (1 - min(r["n_alive"], r["d_out"]) / r["d_out"]) * (1 - min(r["n_alive"], r["d_in"]) / r["d_in"])
    kinds = sorted({parse_site(r["site"])[1] for r in recs})
    out: dict[str, Any] = {
        "iid_null": pooled(recs, lambda r: r["null"] * size(r), size),
        "iid_measured": pooled(recs, lambda r: r["g_out"], lambda r: r["g_sq"]),
        "iid_general_position": pooled(recs, lambda r: general(r) * size(r), size),
        "rank_deficient_sites": [r["site"] for r in recs if r["r_write"] < min(r["n_alive"], r["d_out"])
                                 or r["r_read"] < min(r["n_alive"], r["d_in"])],
        "iid_null_by_kind": {k: pooled([r for r in recs if parse_site(r["site"])[1] == k],
                                       lambda r: r["null"] * size(r), size) for k in kinds},
        "updates": {}}
    for nm in names:
        sq = lambda r, nm=nm: r["updates"][nm]["sq"]
        real = pooled(recs, lambda r, nm=nm: r["updates"][nm]["out"], sq)
        null = pooled(recs, lambda r, sq=sq: r["null"] * sq(r), sq)
        out["updates"][nm] = {"real": real, "null": null, "ratio": real / null if null > 0 else None,
                              "norm": sum(sq(r) for r in recs) ** 0.5, "old_E93": old_share(nm)}
    mlp = [r for r in recs if parse_site(r["site"])[1] in ("gate", "up", "down")]
    out["mlp_small_side_full"] = sum(r["r_write"] == r["d_out"] or r["r_read"] == r["d_in"] for r in mlp)
    out["general_position"] = len(recs) - len(out["rank_deficient_sites"])
    out["placement"] = placement({nm: u["ratio"] for nm, u in out["updates"].items()})
    return out


def placement(ratios: dict[str, float | None]) -> dict[str, Any]:
    """predictions_step0.json T1_real_vs_random: FREE if every scored ratio >= 0.8, CONSTRAINED if any < 0.5, else
    'no call'; an update with no null (ratio None) is unscored and named."""
    scored = {k: v for k, v in ratios.items() if v is not None}
    unscored = sorted(set(ratios) - set(scored))
    if not scored:
        return {"reading": "no call", "unscored": unscored}
    if min(scored.values()) < CONSTRAINED:
        reading = "constrained"
    elif min(scored.values()) >= FREE:
        reading = "free"
    else:
        reading = "no call"
    return {"reading": reading, "min_ratio": min(scored.values()), "unscored": unscored}


def old_share(name: str) -> dict[str, float] | None:
    """E93's share left in Δ and its null on the OLD decomposition (p-e9fcffd5), for comparison only."""
    p = E93 / name / "state.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text())["split"]
    return {"left": s["left"] / s["size"], "null": s["null"] / s["size"]}


def table2(recs: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    delta = sum(r["delta_norm"] ** 2 for r in recs) ** 0.5
    out: dict[str, Any] = {"delta_total": delta, "w0_total": sum(r["w0_norm"] ** 2 for r in recs) ** 0.5, "updates": {}}
    for nm in names:
        norms = [r["updates"][nm]["sq"] ** 0.5 for r in recs]
        above = sum(n > r["delta_norm"] for n, r in zip(norms, recs, strict=True))
        out["updates"][nm] = {"norm": sum(n * n for n in norms) ** 0.5, "sites_above_delta": above, "of": len(recs)}
    return out


def jget(path: Path, *keys: str) -> Any:
    v: Any = json.loads(path.read_text())
    for k in keys:
        v = v[k]
    return v


def floors() -> dict[tuple[str, int], dict[str, Any]]:
    """The decomposition's own masked error on new, KL(W0 ‖ W0 masked with its own masks, Δ off), nats per token, by
    (row set, row length); each on the first n rows of its file."""
    return {("maths", ROW_LEN): {"kl": jget(E132 / "controls.json", "means_W0", "K0"), "n": 128,
                                 "source": "qwen_regating_new/results/controls.json means_W0.K0 (E132)"},
            ("maths", 512): {"kl": jget(W0_REFS, "blocks", "maths", "masked_kl"), "n": 128,
                             "source": "qwen_new_decomp/results/w0_refs_stage1.json blocks.maths.masked_kl (job 56760)"},
            ("fineweb", 512): {"kl": jget(W0_REFS, "blocks", "fineweb", "masked_kl"), "n": 128,
                               "source": "qwen_new_decomp/results/w0_refs_stage1.json blocks.fineweb.masked_kl"}}


def e93_own(name: str) -> dict[str, Any]:
    """E93's K of W0, KL(fine-tuned ‖ W0) pooled over every position of the state's own rows (decomposition-free)."""
    st = json.loads((E93 / name / "state.json").read_text())
    k, n = sum(r[4] for r in st["clean"]["W0"]["K"]), sum(r[4] for r in st["n"]["kl"])
    return {"rows": st["rows"], "len": ROW_LEN, "n": len(st["n"]["kl"]), "kl": k / n,
            "source": f"qwen_d1_table/results/{name}/state.json clean.W0.K"}


def stored_changes(name: str) -> list[dict[str, Any]]:
    """Every stored output change of one update, each with its rows."""
    if name in E93_STATES:
        arm = jget(E93 / name / "state.json", "grid_arm")
        return [e93_own(name), {"rows": "maths", "len": 512, "n": 16, "kl": jget(E91, "arms", arm, "full", "kl"),
                                "source": f"qwen_size_grid/results/size_grid.json arms.{arm}.full.kl (E91)"}]
    if name == "e40":
        return [{"rows": "maths", "len": ROW_LEN, "n": 128, "kl": jget(E93 / "control.json", "clean", "W0"),
                 "source": "qwen_d1_table/results/control.json clean.W0 (E93 stage 0)"},
                {"rows": "maths", "len": ROW_LEN, "n": 16, "kl": jget(E132 / "controls.json", "kl_e40_16"),
                 "source": "qwen_regating_new/results/controls.json kl_e40_16 (E132)"}]
    kl = jget(E45 / f"E_{name}.json", "states", "100", "kl")
    return [{"rows": "maths", "len": 512, "n": 16, "kl": kl["maths"], "source": f"qwen_updates/results/E_{name}.json states.100.kl.maths"},
            {"rows": "own", "len": 512, "n": 64, "kl": kl["own"], "source": f"qwen_updates/results/E_{name}.json states.100.kl.own"}]


def table3(names: list[str]) -> dict[str, Any]:
    """Each stored change beside the floor on the same row set and length; no floor stored: a gap (ratio None)."""
    fl, out = floors(), {}
    for nm in names:
        rows = []
        for c in stored_changes(nm):
            f = fl.get((c["rows"], c["len"]))
            if f is None:
                rows.append(c | {"floor": None, "ratio": None, "floor_rows": "gap: no floor stored on these rows"})
                continue
            same = "the same rows" if f["n"] == c["n"] else f"the first {f['n']} rows, a superset"
            rows.append(c | {"floor": f["kl"], "ratio": c["kl"] / f["kl"], "floor_rows": same})
        out[nm] = rows
    return {"floors": {f"{k[0]} {k[1]}": v for k, v in fl.items()}, "updates": out}


def site_records(files: dict[str, Path], alive: dict[str, Tensor]) -> list[dict[str, Any]]:
    done = {}
    if SITES_LOG.is_file():
        done = {r["site"]: r for r in map(json.loads, SITES_LOG.read_text().splitlines())}
        assert all(set(r["updates"]) == set(files) for r in done.values()), "stored sites read another update set"
    comps, w0 = Tensors(sorted(DECOMP.glob("*.safetensors"))), Tensors(sorted(W0_DIR.glob("model-*.safetensors")))
    dws, rel, checked = {n: Tensors([p]) for n, p in files.items()}, recon_errors(), set()
    for site in SITES:
        if site in done:
            continue
        kind = parse_site(site)[1]
        rec = site_record(site, comps, w0, dws, alive[site], rel[site], kind not in checked)
        if rec["checked"]:
            checked.add(kind)
        with SITES_LOG.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[site] = rec
        print(f"{site}: n {rec['n_alive']} r_write {rec['r_write']}/{rec['d_out']} r_read {rec['r_read']}/{rec['d_in']} "
              f"null {rec['null']:.4f} gauss {rec['g_out'] / rec['g_sq']:.4f} |W0| {rec['w0_norm']:.2f} "
              f"|Delta| {rec['delta_norm']:.4f} {rec['seconds']:.0f}s", flush=True)
    return [done[s] for s in SITES]


def main() -> None:
    require_filed(PRED)
    assert decomp_id(DECOMP) == NEW, DECOMP
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    files = update_files()
    missing = [str(p) for p in files.values() if not p.is_file()]
    assert not missing, missing
    t3 = table3(list(files))  # stored numbers only: a missing key fails here, in the first second
    print("TABLE 3", json.dumps(t3, indent=1), flush=True)
    recs = site_records(files, alive_sets())
    out = {"table1": table1(recs, list(files)), "table2": table2(recs, list(files)), "table3": t3,
           "files": {n: str(p.relative_to(EXPERIMENTS)) for n, p in files.items()}}
    print("TABLE 1", json.dumps(out["table1"], indent=1), "\nTABLE 2", json.dumps(out["table2"], indent=1), flush=True)
    write_result(OUT, out, DECOMP)
    print(f"STAGE 0 DONE {OUT}", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
