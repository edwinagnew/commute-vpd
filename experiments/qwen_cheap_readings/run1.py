"""E162 (qwen_cheap_readings), cell P3.6 run 1: the SVD column of every saved Qwen3-1.7B update beside a random change of
matched per-matrix norms, and the top-set overlap's controls alone (the overlap of the real updates is run 2). No
decomposition is read, nothing is trained, no update is labelled reuse or new. Plan:
reports/plans/2026-11-11_plan_qwen-cheap-readings.md; card card_run1.md; rules results/predictions_run1.json (the run
refuses unless it is committed unchanged), applied by rules.py.

    sbatch --comment='E162 run1' cluster/qwen_cheap_readings_run1.sbatch

Updates (19): the 15 of E152 step 0 (`qwen_instrument_check/step0.py` update_files) and E120's four pool states at step
100 (known and never-solved pools, seeds 0 and 1). Overlap rows: E92's GSM8K gold rows (`qwen_comparators.gold_rows`).
  stage 0  alone, first: the SVD known answers (equal singular values read erank k; zero reads None) on block 0's seven
           shapes, and the unchanged model against itself through the floor's own path (a norm-0 random change): overlap
           exactly 1.0 at every block and position, on_both nan. A miss stops the job (exit 1).
  stage 1  the overlap floor's two draws at the smallest and the largest update's per-matrix norms; recorded, not a stop.
  stage 2  per update, one matrix at a time on the CPU in fp64 (`update_spectrum.matrix`, P2.9's definition): erank,
           top10, norm2, and the floor from two Gaussian draws per matrix weighted by this update's norms. norm2 must
           match E152's or E120's record (else stop); a state already in stage2.json is skipped, so a resubmit resumes;
           a projected job time over the filed limit exits 3 (resubmit the same command).
Each stage writes results/run1_stage<n>.json and prints "STAGE n DONE"; every number is printed.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from commute import qwen_overlap as QO
from commute import qwen_sets
from commute import update_spectrum as US
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_comparators import gold_rows
from commute.qwen_sft import linears
from commute.qwen_vpd import HF_NAME, HF_REVISION, SITES, fp32_matmuls, load_target

rules = load_experiment("qwen_cheap_readings", "rules")
STEP0 = load_experiment("qwen_instrument_check", "step0")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions_run1.json"
E152_SITES = EXPERIMENTS / "qwen_instrument_check" / "results" / "step0_sites.jsonl"
E120 = EXPERIMENTS / "qwen_pass_pools" / "results"
E120_NAMES = tuple(f"{p}_s{s}" for p in ("known", "new") for s in (0, 1))
E92_DIR = "qwen_comparators/results/states/"
E92 = {"a1_s0": E92_DIR + "anchor1_s0_step100.json", "a1_s1": E92_DIR + "anchor1_s1_step100.json",
       "a2_s42": E92_DIR + "anchor2_s0_step1250.json", "a2_s142": E92_DIR + "anchor2_s142_step1250.json",
       **{n: f"{E92_DIR}{n}_step100.json" for n in ("R_s0", "R_s70", "N_s0", "N_s70", "F_s0")},
       "u1": "qwen_comparator_table/results/compare/u1.json", "u2": "qwen_comparator_table/results/compare/u2.json"}  # fmt: skip


def log(msg: str) -> None:
    print(msg, flush=True)


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"run1_stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(RESULTS / f"run1_stage{n}.json")


class Stopped(Exception):
    """A filed stop: the stage's file records why."""


def halt(n: int, out: dict[str, Any], why: str) -> Stopped:
    out["stopped"] = why
    write(n, out)
    return Stopped(why)


def updates() -> dict[str, Path]:
    return STEP0.update_files() | {n: E120 / "large" / "dW" / n / "step_100.safetensors" for n in E120_NAMES}


def recorded_norm2() -> dict[str, float]:
    """Each update's sum of squares as E152 step 0 recorded it (196 matrices), and E120's step-100 norm squared."""
    sites = [json.loads(line) for line in E152_SITES.read_text().splitlines()]
    assert len(sites) == len(SITES), len(sites)
    out = {n: float(sum(r["updates"][n]["sq"] for r in sites)) for n in sites[0]["updates"]}
    return out | {n: json.loads((E120 / f"read_{n}.json").read_text())["files"]["100"]["norm"] ** 2 for n in E120_NAMES}


def e92_join(name: str, path: Path) -> dict[str, Any]:
    if name not in E92:
        return {"e92": None}
    rec = json.loads((EXPERIMENTS / E92[name]).read_text())
    return {"e92": E92[name], "e92_same_file": rec["file"] == str(path.relative_to(EXPERIMENTS))}


# ----------------------------------------------------------------------------- stage 0, the controls alone


def stage0(model: Any, W0: dict[str, torch.Tensor], rows: Any) -> dict[str, Any]:
    shapes = {s: (W0[s].shape[0], W0[s].shape[1]) for s in SITES if s.startswith("layers.0.")}
    r10, r1 = US.state(US.equal_rank(shapes, 10, 0)), US.state(US.equal_rank(shapes, 1, 1))
    zero = US.state({s: torch.zeros(d, dtype=torch.float64) for s, d in shapes.items()})
    svd = {"rank10": r10["erank"] is not None and abs(r10["erank"] - 10) <= 1e-6,
           "rank1": r1["erank"] is not None and abs(r1["erank"] - 1) <= 1e-9,
           "top10": all(r["top10"] is not None and abs(r["top10"] - 1) <= 1e-9 for r in (r10, r1)),
           "zero": zero["erank"] is None and zero["top10"] is None and zero["norm2"] == 0.0}  # fmt: skip
    out: dict[str, Any] = {"svd": {"rank10": r10, "rank1": r1, "zero": zero, "checks": svd}}
    log(f"[0 svd control] {json.dumps(out['svd'])}")
    t0 = time.time()
    ov = QO.overlap(model, rows, QO.random_weights(W0, dict.fromkeys(W0, 0.0), 0))
    out["zero_overlap"] = ov
    exact = ov["overlap"] == 1.0 and math.isnan(ov["on_both"]) and all(b["overlap_min"] == 1.0 for b in ov["per_block"].values())
    log(f"[0 zero overlap] overlap {ov['overlap']!r} on_both {ov['on_both']!r} positions {ov['positions']}; exact: {exact}; {time.time() - t0:.0f} s")
    if not all(svd.values()):
        raise halt(0, out, f"the SVD known answers missed: {svd}")
    if not exact:
        raise halt(0, out, "the unchanged model does not read overlap exactly 1.0 with on_both nan")
    return out


# ----------------------------------------------------------------------------- stage 1, the overlap floor's two draws


def stage1(model: Any, W0: dict[str, torch.Tensor], rows: Any, files: dict[str, Path], rec2: dict[str, float], P: dict[str, Any]) -> dict[str, Any]:
    c, out = P["settings"], {"refs": {}}
    for ref in c["floor_refs"]:
        norms = QO.site_norms(files[ref])
        n2 = sum(v * v for v in norms.values())
        r: dict[str, Any] = {"file": str(files[ref].relative_to(EXPERIMENTS)), "norm": math.sqrt(n2), "norm2_vs_record": n2 / rec2[ref] - 1, "draws": {}}
        for seed in c["overlap_seeds"]:
            t0 = time.time()
            ov = QO.overlap(model, rows, QO.random_weights(W0, norms, seed))
            r["draws"][str(seed)] = ov
            log(f"[1 floor {ref} seed {seed}] overlap {ov['overlap']:.6f} on_both {ov['on_both']:.6f}; {time.time() - t0:.0f} s")
            torch.cuda.empty_cache()
        (o1, b1), (o2, b2) = ((d["overlap"], d["on_both"]) for d in r["draws"].values())
        r["diff"] = {"overlap": abs(o1 - o2), "on_both": abs(b1 - b2)}
        r["rule"] = {"overlap": rules.draws_overlap(o1, o2, c["tol_overlap"]), "on_both": rules.draws_share(b1, b2, c["tol_on_both"])}
        log(f"[1 floor {ref}] {json.dumps({k: r[k] for k in ('norm', 'norm2_vs_record', 'diff', 'rule')})}")
        out["refs"][ref] = r
        write(1, out)
    return out


# ----------------------------------------------------------------------------- stage 2, the SVD column


def gaussian(seed: int, path: Path) -> dict[str, dict[str, float]]:
    """Per matrix, the readings of one iid Gaussian draw at its shape in `path` (fp64, one generator, sites in order).
    erank and top10 do not depend on the scale, so a floor matched to any update's norms is these, weighted by its norms."""
    gen = torch.Generator().manual_seed(seed)
    with safe_open(str(path), framework="pt") as f:
        shapes = {s: tuple(f.get_slice(s).get_shape()) for s in SITES}
    return {s: US.matrix(torch.randn(*shapes[s], generator=gen, dtype=torch.float64)) for s in SITES}


def svd_state(path: Path) -> dict[str, dict[str, float]]:
    with safe_open(str(path), framework="pt") as f:
        assert set(f.keys()) == set(SITES), path
        return {s: US.matrix(f.get_tensor(s)) for s in SITES}


def floor_of(per: dict[str, dict[str, float]], g: dict[str, dict[str, float]]) -> dict[str, float | None]:
    return US.aggregate({s: {"erank": g[s]["erank"], "top10": g[s]["top10"], "norm2": per[s]["norm2"]} for s in SITES})


def stage2(files: dict[str, Path], rec2: dict[str, float], P: dict[str, Any], t_job: float) -> dict[str, Any]:
    c, path = P["settings"], RESULTS / "run1_stage2.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    out.pop("stopped", None)
    t0 = time.time()
    G = {str(s): gaussian(s, next(iter(files.values()))) for s in c["svd_seeds"]}
    log(f"[2 gaussian] {len(SITES)} matrices x {len(G)} draws, {time.time() - t0:.0f} s")
    todo = [n for n in files if n not in out["states"]]
    t1 = time.time()
    for i, name in enumerate(todo):
        per = svd_state(files[name])
        rec, n2 = US.aggregate(per), sum(v["norm2"] for v in per.values())
        if abs(n2 / rec2[name] - 1) > c["norm2_rel_tol"]:  # the wrong file, or a changed one: stop before it is written
            raise halt(2, out, f"{name}: norm2 {n2} off its record {rec2[name]} by {n2 / rec2[name] - 1:.2e}")
        fl = {s: floor_of(per, g) for s, g in G.items()}
        f1, f2 = (f["erank"] for f in fl.values())
        floor = None if f1 is None or f2 is None else (f1 + f2) / 2
        out["states"][name] = {**rec, "file": str(files[name].relative_to(EXPERIMENTS)), "norm2_vs_record": n2 / rec2[name] - 1,
                               "floor": fl, "floor_erank": floor, "vs_floor": rules.vs_floor(rec["erank"], floor, c["gap"]),
                               "draws": rules.draws_erank(f1, f2, c["tol_erank"]), "per_matrix": per, **e92_join(name, files[name])}  # fmt: skip
        log(f"[2 {name}] {json.dumps({k: v for k, v in out['states'][name].items() if k != 'per_matrix'})}")
        write(2, out)
        minutes = (time.time() - t_job + (time.time() - t1) / (i + 1) * (len(todo) - i - 1)) / 60
        log(f"[2 time] {i + 1} of {len(todo)} this job; projected {minutes:.1f} min for the job")
        if i < len(todo) - 1 and minutes > c["projected_limit_min"]:
            out["stopped"] = f"projected {minutes:.1f} min over {c['projected_limit_min']}: resubmit the same command"
            write(2, out)
            raise SystemExit(3)
    st = out["states"]
    calls = [s["vs_floor"] for s in st.values()]
    gaps = {n: s["erank"] - s["floor_erank"] for n, s in st.items() if s["erank"] is not None and s["floor_erank"]}
    out["score"] = {"call": rules.svd_call(calls), "counts": {k: calls.count(k) for k in sorted(set(calls))},
                    "draws": {k: [s["draws"] for s in st.values()].count(k) for k in ("reproducible", "not reproducible", "unreadable")},
                    "erank_minus_floor": gaps,
                    "median_ratio": statistics.median(st[n]["erank"] / st[n]["floor_erank"] for n in gaps) if gaps else None}  # fmt: skip
    log(f"[2 score] {json.dumps(out['score'])}")
    write(2, out)
    return out


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E162") and P["settings"]["k"] == QO.K, (P["experiment"], QO.K)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    files, rec2 = updates(), recorded_norm2()
    missing = [str(p) for p in files.values() if not p.is_file()]
    assert len(files) == 19 and not missing, (len(files), missing)
    fp32_matmuls()
    tok = AutoTokenizer.from_pretrained(HF_NAME, revision=HF_REVISION)
    rows = gold_rows(tok, qwen_sets.load()["test"], int(tok.convert_tokens_to_ids("<|im_end|>")))
    log(f"[rows] {rows.name} {list(rows.tokens.shape)}, {int(rows.answer.sum())} answer positions")
    model = load_target("cuda", torch.float32)
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    try:
        write(0, stage0(model, W0, rows))
        log("STAGE 0 DONE")
        s1 = RESULTS / "run1_stage1.json"
        if not s1.is_file() or set(json.loads(s1.read_text())["refs"]) != set(P["settings"]["floor_refs"]):
            stage1(model, W0, rows, files, rec2, P)
        log(f"STAGE 1 DONE; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        del model, W0
        torch.cuda.empty_cache()
        stage2(files, rec2, P, t_job)
        log("STAGE 2 DONE")
    except Stopped as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    except SystemExit as e:
        log(f"EXIT {e.code}: more to do, resubmit the same command")
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    hard_exit(main())
