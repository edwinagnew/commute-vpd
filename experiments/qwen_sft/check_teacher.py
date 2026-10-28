"""E40 Amendment 1, teacher-training check ($0, one cluster debug job): two hypotheses for the rehearsal's misses,
tested before any fix (diagnosis of 2026-10-28; plan reports/plans/2026-10-28_plan_qwen-sft.md, Amendment 1). Sizes
are the rehearsal's (amend1.A1_TINY: 16 alive columns per site, 2 rows of 128 tokens per step, batch order seed 0),
read from its files on --out (alive index, FineWeb pool, SFT δW).

H2, precision: the training forward's bf16 autocast rounds the small teacher-minus-student difference at every layer.
A. On teacher training's first batch, KL(teacher ‖ W0) at zero offsets, for two teachers: B's positive control (the
   random change at B.json's α, made in bf16 as B made it) and T3's (SFT δW at step 3, when the file is there). Ways:
   fp32 reference (`train`'s own, TF32 off); the training forward as now (bf16); autocast off, hooks kept (fp32); TF32
   matmuls with autocast off; bf16 with the teacher scaled x4 and x16, each against its own fp32 reference. Seconds per
   step for bf16, TF32, fp32 (B's teacher, steps 2-4 at rate 0). Gap = training forward / fp32 reference - 1.
H1, learning: 25 steps cannot close the positive control's distance; gradients do flow.
B. Positive control (B's teacher), trained with autocast OFF (fp32 forwards; stated, not chosen from A): fp32
   KL(teacher ‖ student) on the same 8 held-out rows (4 maths, 4 FineWeb evaluation rows) at steps 0, 25, 100, 300 at
   rate 1e-5, and at 0, 25 at 1e-4 (run first, it is cheap).

Filed expectations (also in the JSON; scored OK/MISS at the end; a zero, negative or non-finite denominator or a
reading never made is a MISS with its reason):
  R   the bf16 forward and fp32 reference reproduce the rehearsal's first-batch numbers within 1e-2 relative.
  A1  per teacher, |fp32 gap| < 1e-3: H2 holds. |gap| >= 1e-2 means a forward mismatch instead; between: inconclusive.
  A2  per teacher, |bf16 gap at x16| < |bf16 gap at x1|.
  A3  per teacher, |TF32 gap| < |bf16 gap| (report).   A4  s/step bf16 <= TF32 <= fp32 (report).
  B1  at 1e-5, KL at step 100 <= 0.5 of step 0.   B2  at 1e-4, KL at step 25 <= 0.5 of step 0.
      Both: H1 holds. Both >= 0.9 (flat): gradients are the problem. Else mixed. (H1's arithmetic: ~0.3 by step 100.)

Usage (repo root, L40): cluster/check_teacher.sbatch [--out <rehearsal dir>] [--json <results file>]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import amend1
import run
import torch
from run import BF16, RESULTS, Job, log, write_json
from torch import Tensor

from commute.qwen_sft import load, mean_kl, plus
from commute.qwen_teacher import Offsets, random_change, train
from commute.qwen_vpd import SITES, fp32_matmuls, load_components

CFG = amend1.A1_TINY
REHEARSED = {"B": (4.6882e-03, 5.5190e-03), "T3": (3.1680e-02, 1.7764e-02)}  # (fp32, bf16): rehearsal run.log
TIMED, SCALES = 4, (4, 16)
READS = {1e-4: (25,), 1e-5: (25, 100, 300)}
FILED = {"R": "bf16 and fp32 first-batch KL within 1e-2 relative of the rehearsal's",
         "A1": "|fp32 gap| < 1e-3: H2 holds; >= 1e-2: forward mismatch; between: inconclusive",
         "A2": "|bf16 gap x16| < |bf16 gap x1|", "A3": "|TF32 gap| < |bf16 gap| (report)",
         "A4": "s/step bf16 <= TF32 <= fp32 (report)", "B1": "1e-5: KL step 100 / step 0 <= 0.5",
         "B2": "1e-4: KL step 25 / step 0 <= 0.5", "B": "B1 and B2: H1 holds; both ratios >= 0.9: gradients"}


def ratio(x: float | None, ref: float | None) -> float | None:
    """x / ref, None when either is missing or non-finite or ref is not > 0."""
    if x is None or ref is None or not (math.isfinite(x) and math.isfinite(ref)) or ref <= 0:
        return None
    return x / ref


def first_batch(job: Job, off: Offsets, teacher: dict[str, Tensor], autocast: bool = True, tf32: bool = False,
                steps: int = 1) -> dict[str, Any]:
    """`train` at rate 0 from zero offsets: step 1's training-forward KL(teacher ‖ W0) and its fp32 one on the same
    batch (TF32 too when `tf32`), and with steps > 1 the mean seconds of steps 2 on."""
    off.reset()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        curve = train(job.model, off, teacher, *job.pools(), 0.0, steps, CFG.t_rows_each, CFG.t_micro, CFG.seed, log,
                      autocast=autocast)
    finally:
        fp32_matmuls()
    rec: dict[str, Any] = {"kl": curve[0]["kl"], "kl_ref": curve[0]["kl_fp32_teacher_w0"]}
    if steps > 1:
        rec["step_s"] = statistics.mean(c["s"] for c in curve[1:])
    return rec


def precision(job: Job, off: Offsets, teacher: dict[str, Tensor], steps: int, put: Any) -> None:
    """Measurement A for one teacher; each way written as it is made. Gaps against the TF32-off fp32 reference."""
    ways = {"bf16": {}, "fp32": {"autocast": False}, "tf32": {"autocast": False, "tf32": True}}
    ref = None
    for name, kw in ways.items():
        rec = first_batch(job, off, teacher, steps=steps, **kw)
        ref = rec["kl_ref"] if ref is None else ref  # bf16's: TF32 off
        rec["gap"] = None if (r := ratio(rec["kl"], ref)) is None else r - 1
        put(name, rec)
    for c in SCALES:
        rec = first_batch(job, off, {s: c * v for s, v in teacher.items()})
        rec["gap"] = None if (r := ratio(rec["kl"], rec["kl_ref"])) is None else r - 1
        put(f"bf16_x{c}", rec)


def learning(job: Job, off: Offsets, teacher: dict[str, Tensor], put: Any) -> None:
    """Measurement B: fp32 KL(teacher ‖ student) on fixed held-out rows, before and during training (autocast off)."""
    rows = job.maths() + job.web

    @torch.no_grad()
    def reading() -> float:
        Ws = {s: job.W0[s] + off.change(s) for s in SITES}
        return mean_kl(job.model, rows, plus(job.W0, teacher), Ws)

    off.reset()
    k0 = reading()
    for lr, reads in READS.items():
        off.reset()
        rec: dict[str, Any] = {"kl": {"0": k0}}
        put(f"{lr:g}", rec)

        def on_step(step: int, curve: list[dict[str, float]], rec: dict[str, Any] = rec, lr: float = lr,
                    reads: tuple[int, ...] = reads) -> None:
            if step in reads:
                rec["kl"][str(step)] = reading()
                rec["curve"] = [c["kl"] for c in curve]
                rec["step_s"] = statistics.mean(c["s"] for c in curve[1:])
                put(f"{lr:g}", rec)

        train(job.model, off, teacher, *job.pools(), lr, max(reads), CFG.t_rows_each, CFG.t_micro, CFG.seed, log,
              on_step, autocast=False)


def score(res: dict[str, Any]) -> dict[str, Any]:
    """The filed expectations, OK or MISS with the numbers or the reason."""
    out: dict[str, Any] = {}
    for t, A in res.get("A", {}).items():
        b, f, x16, tf = (A.get(k, {}).get("gap") for k in ("bf16", "fp32", "bf16_x16", "tf32"))
        fb, fr = REHEARSED[t]
        rep = [ratio(A.get("bf16", {}).get(k), v) for k, v in (("kl_ref", fb), ("kl", fr))]
        out[f"R {t}"] = ("MISS: no reading" if None in rep else
                         "OK" if all(abs(r - 1) < 1e-2 for r in rep if r is not None) else "MISS") + f" {rep}"
        out[f"A1 {t}"] = ("MISS: no fp32 gap (zero or non-finite reference)" if f is None else
                          f"OK: H2 holds, gap {f:.2e}" if abs(f) < 1e-3 else
                          f"MISS: forward mismatch, gap {f:.2e}" if abs(f) >= 1e-2 else f"MISS: inconclusive {f:.2e}")
        out[f"A2 {t}"] = ("MISS: a gap missing" if b is None or x16 is None else
                          ("OK" if abs(x16) < abs(b) else "MISS") + f" x1 {b:.2e}, x16 {x16:.2e}")
        out[f"A3 {t}"] = ("MISS: a gap missing" if b is None or tf is None else
                          ("OK" if abs(tf) < abs(b) else "MISS") + f" bf16 {b:.2e}, tf32 {tf:.2e}")
    s = [res.get("A", {}).get("B", {}).get(k, {}).get("step_s", math.nan) for k in ("bf16", "tf32", "fp32")]
    out["A4"] = ("OK" if s[0] <= s[1] <= s[2] else "MISS") + f" {s}"
    B = res.get("B", {})
    r1 = ratio(B.get("1e-05", {}).get("kl", {}).get("100"), B.get("1e-05", {}).get("kl", {}).get("0"))
    r2 = ratio(B.get("0.0001", {}).get("kl", {}).get("25"), B.get("0.0001", {}).get("kl", {}).get("0"))
    for name, r in (("B1", r1), ("B2", r2)):
        out[name] = "MISS: not read, or step 0 not > 0" if r is None else ("OK" if r <= 0.5 else "MISS") + f" {r:.3f}"
    out["B"] = ("MISS: a ratio missing" if r1 is None or r2 is None else "OK: H1 holds" if max(r1, r2) <= 0.5 else
                "MISS: flat, gradients are the problem" if min(r1, r2) >= 0.9 else "MISS: mixed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/mnt/nw/home/e.agnew/commute-data/tmp/e40a1_rehearse", help="rehearsal files")
    ap.add_argument("--decomp", default=str(Path.home() / "commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch"))
    ap.add_argument("--json", default=str(RESULTS / "check_teacher.json"))
    a = ap.parse_args()
    path = Path(a.json)
    run._LOG[0] = path.with_suffix(".log")
    res: dict[str, Any] = {"filed": FILED, "config": {"alive_max": CFG.alive_max, "row_len": CFG.row_len,
                                                     "rows_each": CFG.t_rows_each, "seed": CFG.seed}}

    def putter(*keys: str) -> Any:
        def put(name: str, rec: dict[str, Any]) -> None:
            d = res
            for k in keys:
                d = d.setdefault(k, {})
            d[name] = rec
            write_json(path, res)
            log(f"[check_teacher] {'/'.join(keys)}/{name}: {json.dumps(run.brief(rec), default=float)}")

        return put

    a.usd_per_h = 0.0
    job = Job(a, CFG, amend1.REHEARSAL, budget_usd=math.inf, guard_usd=math.inf)
    off = Offsets(load_components(job.decomp_dir, "cpu", BF16), job.alive())
    alpha = job.res["B"]["rates"]["1e-05"]["alpha"]
    teacher = {s: v.float() for s, v in random_change(off, alpha, CFG.pc_seed, BF16).items()}
    res["alpha"], res["n_offset_params"] = alpha, off.n_params()
    precision(job, off, teacher, TIMED, putter("A", "B"))
    if (dw := job.dw_path(3)).is_file():
        t3 = {s: v.float() for s, v in load(dw, "cuda").items()}
        precision(job, off, t3, 1, putter("A", "T3"))
        del t3
    else:
        log(f"[check_teacher] {dw} missing: A on B's teacher only")
    res["peak_gb_A"] = torch.cuda.max_memory_allocated() / 1e9
    learning(job, off, teacher, putter("B"))
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    res["score"] = score(res)
    write_json(path, res)
    for k, v in res["score"].items():
        log(f"[score] {k}: {v}")
    print("FINISHED", path, flush=True)


if __name__ == "__main__":
    main()
