"""E80: can VPD's adversary fail do-nothing on Qwen3-1.7B? (One RunPod H100, ~1.5 h of run.) Plan: reports/plans/2026-11-04_plan_e80-adversary-do-nothing.md, its
"Changes before the run" over the rest; rules: rules.py, results/predictions.json (the run refuses to start without it, and while results/stopped.json exists).

`qwen_teacher.adversary` (20 sign steps, one source per site, g and clean logits recomputed per row) on W0, N, I, A, under fp32 weights and fp32 components with Γ in
bf16, as the 128-row job; bf16 components cannot hold a moved state (the plan's last section). Δ is never stored: `masked_logits` reads d * out from the weights in
the model, and Γ's readouts read the V in the components, so a state only has to set both. Stages, cheapest and control first as filed:
  1 W0 on W, seed 0, in its filed window or a stop;  2 W0 on M, seeds 0, 1: the band;  3 N on M, then W (clean KL to FT on M's first row, or a stop);
  4 I on M (in-span change recomputed, clean KL on E79's M rows, or a stop);  5 A on M;  6 W0, I own-mask reads on E79's 16 rows, and the weight errors.
results/results.json after every start, every number in results/run.log, each adversary step's KL in the log. A rerun resumes at the next unread start. Only small files
under results/: nothing here reads or writes a results/large. Inputs go beside --decomp: e40_nowarmup/, e80_inputs/, and e80_X/ (the in-span X per site, 5.6 GB, this pod's).

Usage (repo root, 80 GB GPU; WANDB_API_KEY if an input is not there yet): .venv/bin/python experiments/qwen_d1_adversary/run.py --decomp /root/qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
import upload
from safetensors.torch import load_file, save_file
from torch import Tensor

from commute import qwen_e40_files, qwen_files, qwen_maps, qwen_vpd, split
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import linears, load, sha256
from commute.qwen_teacher import Offsets, adversary, do_nothing
from commute.qwen_vpd import SITES, Components, Gamma
from commute.vpd import kl_per_position

amend2 = load_experiment("qwen_masked_read", "amend2")  # E79's own-mask read; its log and stop write E79's files, so only its reads are called
amend1, read_run = amend2.amend1, amend2.run
e76 = read_run.e76

HERE = Path(__file__).resolve().parent
RESULTS, PRED, OUT, STOPPED, LOG = HERE / "results", HERE / "results" / "predictions.json", HERE / "results" / "results.json", HERE / "results" / "stopped.json", HERE / "results" / "run.log"
DEV, F64 = "cuda", torch.float64
STEP, TOL, ALIVE_TOTAL, GPU_MIN_GIB = 10, 1e-6, 494598, 70  # u1 is E40 step 10; E44's basis tolerance and alive set
DW_NAME = f"step_{STEP}.safetensors"
T0 = time.time()


@dataclass
class Job:
    model: Any
    comps: Components  # fp32
    gamma: Gamma
    off: Offsets  # E76's cut, zero except while A is read
    alive: qwen_maps.Alive  # E44's alive set, for I
    dw: dict[str, Tensor]  # bf16, on the card
    W0: dict[str, Tensor]
    rows: dict[str, list[Tensor]]  # M and W, each row [1, T] on the card
    inputs: Path
    x_dir: Path
    read: Any  # E79's Job on E79's 16 rows, with FT's clean logits
    out: dict[str, Any]


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def peak() -> str:
    return f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"


def stop(rule: str, detail: str) -> None:
    log(f"STOP {rule}: {detail}")
    e76.write_json(STOPPED, {"rule": rule, "detail": detail})
    sys.exit(f"STOP {rule}: {detail}")


def save(J: Job) -> None:
    e76.write_json(OUT, J.out)


def start(J: Job, key: str, rs: str, seed: int) -> float:
    """One adversary start on the state the model and components hold; skipped if results.json has it. The first step must read finite, or a stop."""
    if key in J.out["starts"]:
        return J.out["starts"][key]["kl"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def watch(msg: str) -> None:
        log(f"[{key}] {msg}")
        if " step 0: " in msg and not rules.first_step_ok(float(msg.rsplit(" ", 1)[1])):
            stop("first_step", f"{key}: {msg}")

    kl = adversary(J.model, J.rows[rs], J.gamma, J.comps, seed, log=watch)[0]
    J.out["starts"][key] = {"kl": kl, "s": time.time() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    save(J)
    log(f"[{key}] final KL {kl:.6f}, {time.time() - t0:.0f} s, {peak()}")
    return kl


def check(J: Job, name: str, got: float, ok: bool, want: str) -> None:
    J.out["checks"][name] = {"got": got, "want": want}
    save(J)
    log(f"[check {name}] {got}, want {want}")
    if not ok:
        stop(name, f"reads {got}, want {want}")


def stage_w0(J: Job) -> None:
    got = start(J, "W0_W_s0", "W", 0)
    log(f"[stage 1] {got}, the 128-row job {rules.W_128}, window {rules.STAGE1_LOW}-{rules.STAGE1_HIGH}")
    if not rules.stage1_ok(got):
        stop("stage1", f"W0 on W reads {got}, outside {rules.STAGE1_LOW}-{rules.STAGE1_HIGH}")
    a, b = start(J, "W0_M_s0", "M", 0), start(J, "W0_M_s1", "M", 1)
    log(f"[stage 2] W0 on M {a}, {b}: band {rules.band(a, b)}")


def stage_n(J: Job) -> None:
    """N: weights W0 + δW, components unmoved, so Δ = Δ₀ + δW. FT's reference comes through `change_hooks` on W0's weights, N through the swap: E79's two paths."""
    if "N_M_s0" in J.out["starts"] and "N_W_s0" in J.out["starts"]:
        return
    row = J.rows["M"][0]
    ref = read_run.clean_logits(J.model, row, J.dw.__getitem__)[0].to(DEV)
    with do_nothing(J.model, J.W0, J.dw):
        with torch.no_grad():
            got = float(kl_per_position(ref, J.model(row, use_cache=False).logits).double().mean())
        check(J, "N_clean", got, rules.n_clean_ok(got), f"under {rules.N_CLEAN_UNDER}")
        del ref
        start(J, "N_M_s0", "M", 0)
        start(J, "N_W_s0", "W", 0)


@torch.no_grad()
def make_x(J: Job) -> None:
    """E44's stage S for I alone: X = K⁺δW per site in fp32, Gram bases, no Σ (qwen_maps/run.py `s_site`, `site_products`). On the pod's disk, so a rerun skips it."""
    J.x_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in SITES:
        path = J.x_dir / f"{s}.safetensors"
        if not path.is_file():
            U_A, V_A = J.alive.uv(s, DEV, F64)
            b = split.basis_from(split.gram_basis(U_A), split.gram_basis(V_A), TOL)
            tmp = path.with_suffix(".tmp")
            save_file({"X_ls": split.least_squares_x(J.dw[s].to(DEV, F64), b).float().cpu().contiguous()}, str(tmp))
            tmp.replace(path)
    log(f"[I] X ready, {time.time() - t0:.0f} s, {peak()}")


def inspan(J: Job) -> Any:
    """site -> `qwen_maps.SiteChange` of E72's in-span state: weights W0 + dσ(dU, dV), all alive columns moved. The remainder is discarded and Δ = Δ₀ − dU dVᵀ."""

    def fn(s: str) -> qwen_maps.SiteChange:
        U_A, V_A = J.alive.uv(s, DEV, F64)
        X = load(J.x_dir / f"{s}.safetensors", DEV)
        return qwen_maps.site_change("inspan", J.dw[s].to(DEV, F64), U_A, V_A, X.__getitem__, 0.0, 0.0)

    return fn


def stage_i(J: Job) -> None:
    if "I_M_s0" in J.out["starts"]:
        return
    make_x(J)
    with qwen_maps.applied(J.model, J.comps, J.alive, inspan(J)):
        clean = read_run.clean_logits(J.model, J.read.rows["M"], None)
        got = whole(read_run.read_rows((x.to(DEV) for x in clean), J.read.ft["M"]))["fwd"]
        check(J, "I_clean", got, rules.i_clean_ok(got), f"~0.009, stop above {rules.I_CLEAN_MAX}")
        del clean
        start(J, "I_M_s0", "M", 0)


def stage_a(J: Job) -> None:
    """E76's clean-term offsets: `Offsets.state` moves the cut columns in place and swaps in W0 + change, so Γ and the masked forward read them."""
    if "A_M_s0" in J.out["starts"]:
        return
    J.off.load_from(J.inputs / "A_offsets.safetensors")
    try:
        with J.off.state(J.model, J.comps, J.W0):
            start(J, "A_M_s0", "M", 0)
    finally:
        J.off.reset()


def whole(pos: dict[str, Tensor]) -> dict[str, float]:
    return read_run.summary(pos, torch.ones_like(pos["fwd"], dtype=torch.bool))


@torch.no_grad()
def read_own(J: Job, state: str) -> None:
    """E79's own-mask read of the state the model and components hold, offsets at zero: Γ on its forward, draw d1, on M and W, against FT clean and its own clean."""
    R = J.read
    for rs in R.rows:
        if rs in J.out["own"].get(state, {}):
            continue
        clean = read_run.clean_logits(R.model, R.rows[rs], None)
        got = whole(amend1.masked_cell(R, rs, read_run.SEEDS["d1"], amend2.ft_g(R, rs), clean))
        kl = whole(read_run.read_rows((x.to(DEV) for x in clean), R.ft[rs]))["fwd"]
        J.out["own"].setdefault(state, {})[rs] = {"clean": kl, "own": got["fwd"], "self": got["self_fwd"], "n": got["n"]}
        save(J)
        log(f"[own {state} {rs}] {json.dumps(J.out['own'][state][rs])}, {peak()}")


@torch.no_grad()
def squared_error(J: Job, change: Any) -> float:
    """Σ‖the state's change − δW‖², plain, in double. `change(s)` None: no change."""
    total = 0.0
    for s in SITES:
        M, c = J.dw[s].to(DEV, F64), change(s)
        d = M if c is None else c.to(M) - M
        total += float((d * d).sum())
    return total


def stage_weights(J: Job) -> None:
    """‖state − FT‖² / ‖δW‖² on what the components hold: N's change is none (δW sits in Δ), I's dσ, A's the offsets' change. On file: ‖δW‖² 3.49, A 1.36; I's 0.36 is Σ-weighted."""
    if "A" in J.out["weights"]:
        return
    ins, size = inspan(J), squared_error(J, lambda s: None)
    J.off.load_from(J.inputs / "A_offsets.safetensors")
    A = squared_error(J, J.off.change)
    J.off.reset()
    J.out["weights"] = {"dW_squared": size, "N": 1.0, "I": squared_error(J, lambda s: ins(s)[0]) / size, "A": A / size}
    save(J)
    log(f"[weights] {json.dumps(J.out['weights'])}")


def stage_own(J: Job) -> None:
    make_x(J)  # a pod replaced after stage 4
    read_own(J, "W0")
    if "I" not in J.out["own"] or len(J.out["own"]["I"]) < 2:
        with qwen_maps.applied(J.model, J.comps, J.alive, inspan(J)):
            read_own(J, "I")
    stage_weights(J)


def fetch_inputs(decomp: Path) -> tuple[Path, Path]:
    """Each is kept if complete, so a rerun asks W&B for nothing. The δW file's sum is the committed one E76's job checks (`e76.DW_SUMS`), read by `verified`."""
    qwen_files.fetch(decomp, log)
    qwen_e40_files.fetch(decomp.parent / "e40_nowarmup", "v0")
    upload.fetch(decomp.parent / "e80_inputs")
    dw = decomp.parent / "e40_nowarmup" / "dW" / DW_NAME
    got, want = sha256(dw), qwen_e40_files.sums(e76.DW_SUMS)[DW_NAME]
    log(f"[dW] {dw}: sha256 {got}, want {want}")
    assert got == want, "δW is not the file E76 used"
    return dw, decomp.parent / "e80_inputs"


def row_sets() -> dict[str, list[Tensor]]:
    maths, web = (json.loads(p.read_text())["rows"] for p in (e76.MATHS, e76.WEB))
    assert e76.C.e74_rows + e76.C.pool_each == rules.M_FROM and e76.C.row_len == rules.ROW_LEN and len(web) == 128 and all(len(r) == 512 for r in web)
    M = rules.m_rows(maths, [*e76.rows_all()[0].tolist(), *e76.rows_all()[1].tolist()])
    return {"M": [torch.tensor([r], device=DEV) for r in M], "W": [torch.tensor([r], device=DEV) for r in web]}


def set_up(decomp: Path) -> Job:
    assert torch.cuda.get_device_properties(0).total_memory > GPU_MIN_GIB * 2**30, "fp32 components need an 80 GB card (the plan's last section)"
    dw_file, inputs = fetch_inputs(decomp)
    qwen_vpd.fp32_matmuls()
    model = qwen_vpd.load_target(DEV, torch.float32)
    comps = qwen_vpd.load_components(decomp, DEV, torch.float32)
    gamma = qwen_vpd.load_gamma(decomp, DEV, torch.bfloat16)
    dw = {s: v.to(DEV) for s, v in load(dw_file).items() if s in SITES}
    alive = qwen_maps.Alive(comps, qwen_maps.alive_from_stats(load(inputs / "W0_maths.safetensors"), load(inputs / "W0_fineweb.safetensors")))
    assert alive.count() == ALIVE_TOTAL, f"alive set {alive.count()}"
    off = Offsets(comps, load_file(str(inputs / "cut.safetensors")), DEV)
    e79 = read_run.row_sets()
    ft = {rs: read_run.clean_logits(model, t, dw.__getitem__) for rs, t in e79.items()}
    read = read_run.Job(model, comps, off, gamma, e79, {}, {}, ft, {}, {})
    out = e76.read_json(OUT) or {"starts": {}, "checks": {}, "own": {}, "weights": {}}
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    log(f"[load] {torch.cuda.get_device_name()}, {peak()}; resumed {sorted(out['starts'])}")
    return Job(model, comps, gamma, off, alive, dw, W0, row_sets(), inputs, decomp.parent / "e80_X", read, out)


def score() -> None:
    """From what results.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score(out)
    e76.write_json(OUT, out)
    log(f"[score] {json.dumps(out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    J = set_up(Path(a.decomp).expanduser().resolve())
    for stage in (stage_w0, stage_n, stage_i, stage_a, stage_own):
        stage(J)
    log(f"[done] {peak()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    except BaseException:  # noqa: BLE001  (an out-of-memory or the like: the same exit, never a hang in teardown)
        traceback.print_exc()
        code = 1
    try:
        score()
    finally:
        hard_exit(code)
