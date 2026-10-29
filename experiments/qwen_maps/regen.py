"""E44's input, remade: E40's no-warm-up SFT update of Qwen3-1.7B (AdamW 1e-5 constant, batch 64 in micro-batches
of 8, no clipping, 100 steps), δW saved at steps 1, 10, 100 (bf16) and 100 (fp32) in E40's layout, on a MATS-cluster
L40, because RunPod's volume ydavm7ct4x that holds E40's files cannot be reached (user, 2026-10-29). Plan:
reports/plans/2026-10-28_plan_qwen-q0q1.md, "Amendment record (2026-10-29)"; commands: RUN_BRIEF_regen.md; review
reports/reviews/2026-10-29_review_e40-regen-code.md.

The step is `commute.qwen_train`'s, the same operations as E40's stage F (fp32 master weights, bf16 autocast, gradient
checkpointing, TF32 off); each step's 64 problems are read from E40's own record (qwen_sft/results/F.json). Every
check stops the job (exit 2, `"stopped"` in results/regen.json) the step it misses, so no SHA256SUMS is written and the
upload refuses (`"pass"` is set only at the end):
  E44's stage C, computed here: KL to start (KL(W0 ‖ W) on E40's 128 maths rows) from the bf16 files at 1, 10 and 100
  and from the fp32 weights at 100; 100 fp32 within control_W0's tol of E40's, 1 and 10 within tol + gap_mult x gap
  (gap = bf16 / fp32 at 100; at step 1 and 10, before the gap is known, within tol alone, stricter than C).
  E40's record (F.json): the training loss at every step and |δW| of each of the 196 matrices at 1, 10, 100, within
  TOL_EARLY at steps 1-10 and TOL_LATE after (sources at the constants).
With --pause N the state is checkpointed after every N steps and the job exits 3; the same command resumes it from the
checkpoint's own step. A fresh start refuses to delete a finished dW/ (SHA256SUMS present) without --force.
Usage (repo root, L40): python experiments/qwen_maps/regen.py [--out DIR] [--pause N] [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from commute import qwen_sets
from commute.qwen_eval import MODEL, MODEL_REV
from commute.qwen_sft import linears, mean_kl, plus, save, sft_seq
from commute.qwen_train import Recipe, train
from commute.qwen_vpd import fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
E40 = HERE.parent / "qwen_sft" / "results"
OUT_JSON, PRED = HERE / "results" / "regen.json", HERE / "results" / "predictions.json"
RECIPE = Recipe(lr=1e-5, warmup=0, steps=100, schedule="constant", clip=None, micro=8)  # E40's Config as it ran
KEEP, SEQ_LEN, DEV, T0 = (1, 10, 100), 512, "cuda", time.time()
# Loss and per-matrix |δW| against E40's record. Steps 1-10: E44's control_W0 tol (1%), about 8x the floor measured
# between the cards (W0's loss on E40's first batch, L40 2.40705 vs E40's H100 2.41023, 0.13%). Steps 11-100: E45's
# filed tolerance for an L40 training curve against an earlier run's (qwen_updates predictions early.gold_curve, 5%).
TOL_EARLY, EARLY_UNTIL = 0.01, 10
TOL_LATE = json.loads((HERE.parent / "qwen_updates/results/predictions.json").read_text())["early"]["gold_curve"]["tol"]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def write(rec: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(rec, indent=1))


def rel(x: float, ref: float) -> float:
    return abs(x / ref - 1) if math.isfinite(x) and ref else math.inf


def start(out: Path, ckpt: Path, force: bool) -> dict[str, Any]:
    """The record to go on from: a resume keeps what was recorded up to the checkpoint's own step (a job lost while
    writing a later checkpoint leaves the earlier one whole: `qwen_train.checkpoint` renames into place)."""
    if ckpt.is_file():
        c = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)["step"]
        rec = json.loads(OUT_JSON.read_text())
        rec["steps"] = [s for s in rec["steps"] if s["step"] <= c]
        rec["kl"] = {k: v for k, v in rec["kl"].items() if int(k.split("_")[0]) <= c}
        log(f"[regen] resuming after step {c}")
        return rec
    if (out / "dW" / "SHA256SUMS").is_file() and not force:
        sys.exit(f"{out}/dW is a finished run (SHA256SUMS): pass --force to delete it and start again")
    shutil.rmtree(out / "dW", ignore_errors=True)  # no file of an unfinished earlier trajectory is kept
    return {"recipe": asdict(RECIPE), "tol": {"early": TOL_EARLY, "late": TOL_LATE}, "steps": [], "kl": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    ap.add_argument("--pause", type=int, default=RECIPE.steps, help="checkpoint and exit 3 every N steps")
    ap.add_argument("--force", action="store_true", help="delete a finished dW/ and train again")
    a = ap.parse_args()
    out, F = Path(a.out), json.loads((E40 / "F.json").read_text())
    C = json.loads(PRED.read_text())["rules"]["control_W0"]  # E44's stage-C rule: tol, gap_mult, kl_to_start refs
    ckpt = out / "ckpt.pt"
    rec = start(out, ckpt, a.force)
    fp32_matmuls()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    end_id, train_set = int(tok.convert_tokens_to_ids("<|im_end|>")), qwen_sets.load()["train"]
    model = load_target(DEV, torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    maths = [torch.tensor([r], device=DEV) for r in json.loads((E40 / "maths_rows.json").read_text())["rows"]]
    rec["card"] = torch.cuda.get_device_name()
    log(f"[regen] {rec['card']}, out {out}, pause {a.pause}")
    saved: dict[int, dict[str, torch.Tensor]] = {}

    def batch_at(t: int) -> list[Any]:
        return [sft_seq(tok, train_set[i]["question"], train_set[i]["gold"], end_id, SEQ_LEN)
                for i in F["steps"][t - 1]["batch"]]

    def save_dw(t: int, dW: dict[str, torch.Tensor]) -> None:
        save(out / "dW" / f"step_{t}.safetensors", dW, torch.bfloat16, droppable=False, log=log)
        saved[t] = dW  # host bf16, read back as E44's C reads the file: W0 + δW in fp32

    def stop(why: str) -> None:
        rec["stopped"] = why
        write(rec)
        log(f"STOP: {why}")
        sys.exit(2)

    def kl_checks(t: int, r: dict[str, Any]) -> None:
        kl = rec["kl"]
        kl[str(t)] = mean_kl(model, maths, W0, plus(W0, saved.pop(t)))  # bf16 file, as C
        if t == RECIPE.steps:
            kl[f"{t}_fp32"] = r["kl"]["maths"]
            gap = rel(kl[str(t)], kl[f"{t}_fp32"])
            rec["gap"] = gap
            if rel(kl[f"{t}_fp32"], C["kl_to_start"]["100_fp32"]) > C["tol"]:
                stop(f"KL to start at 100 (fp32) {kl[f'{t}_fp32']:.5f} off E40's {C['kl_to_start']['100_fp32']:.5f}")
            for s in ("1", "10"):  # C's rule now that the gap is known
                if rel(kl[s], C["kl_to_start"][s]) > C["tol"] + C["gap_mult"] * gap:
                    stop(f"KL to start at {s} (bf16) {kl[s]:.5f} off E40's {C['kl_to_start'][s]:.5f} (gap {gap:.2e})")
        elif rel(kl[str(t)], C["kl_to_start"][str(t)]) > C["tol"]:
            stop(f"KL to start at {t} (bf16) {kl[str(t)]:.5f} off E40's {C['kl_to_start'][str(t)]:.5f}")

    def record_checks(t: int, r: dict[str, Any]) -> None:
        e40, tol = F["steps"][t - 1], TOL_EARLY if t <= EARLY_UNTIL else TOL_LATE
        r["loss_rel"] = rel(r["loss"], e40["loss"])
        if r["loss_rel"] > tol:
            stop(f"step {t} loss {r['loss']:.5f} off E40's {e40['loss']:.5f} by more than {tol}")
        if t in KEEP:
            worst = max(r["norms"], key=lambda s: rel(r["norms"][s], e40["norms"][s]))
            r["norm_rel_max"] = rel(r["norms"][worst], e40["norms"][worst])
            if r["norm_rel_max"] > tol:
                stop(f"step {t} |δW| of {worst} {r['norms'][worst]:.5f} off E40's {e40['norms'][worst]:.5f}")

    def on_step(r: dict[str, Any]) -> None:
        t = r["step"]
        if t not in KEEP:
            r.pop("norms")
        r.pop("crossed", None)
        if t == RECIPE.steps:  # exact, as E40's step_100_fp32; one matrix at a time to the host
            dW = {s: (lin.weight.detach() - W0[s]).to("cpu") for s, lin in linears(model).items()}
            save(out / "dW" / f"step_{t}_fp32.safetensors", dW, torch.float32, droppable=False, log=log)
        rec["steps"].append(r)
        write(rec)
        log(f"[regen] step {t}: loss {r['loss']:.4f}, grad norm {r['grad_norm']:.2f}, |dW| {r['dw_norm']:.4f}"
            + (f", KL to start (fp32 weights) {r['kl']['maths']:.5f}" if "kl" in r else ""))
        record_checks(t, r)
        if t in KEEP:
            kl_checks(t, r)
            log(f"[regen] step {t}: KL to start {json.dumps(rec['kl'])}, |δW| worst rel {r['norm_rel_max']:.2e}")
        write(rec)

    reads = set(KEEP) | set(range(a.pause, RECIPE.steps, a.pause))
    t = train(model, W0, batch_at, RECIPE, {"maths": maths}, reads, (), set(KEEP), save_dw, on_step, ckpt, a.pause)
    if t < RECIPE.steps:
        log(f"[regen] paused after step {t}: resubmit the same command")
        sys.exit(3)
    rec |= {"done": True, "pass": True}  # every check above held, or the job stopped
    write(rec)
    log(f"[regen] done, every check held; KL to start {json.dumps(rec['kl'])}; peak GPU "
        f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
