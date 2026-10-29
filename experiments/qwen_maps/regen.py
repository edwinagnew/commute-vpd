"""E44's input, remade: E40's no-warm-up SFT update of Qwen3-1.7B (AdamW 1e-5 constant, batch 64 in micro-batches
of 8, no clipping, 100 steps), δW saved at steps 1, 10, 100 (bf16) and 100 (fp32) in E40's layout, on a MATS-cluster
L40, because RunPod's volume ydavm7ct4x that holds E40's files cannot be reached (user, 2026-10-29). Plan:
reports/plans/2026-10-28_plan_qwen-q0q1.md, "Amendment record (2026-10-29)"; commands: RUN_BRIEF_regen.md.

The step is `commute.qwen_train`'s, the same operations as E40's stage F (fp32 master weights, bf16 autocast, gradient
checkpointing, TF32 off); each step's 64 problems are read from E40's own record (qwen_sft/results/F.json). KL to
start (KL(W0 ‖ W_t) on E40's 128 maths rows, fp32) is read at 1, 10 and 100 and compared with E40's (F.json) within
E44's filed control_W0 tolerance; step 1's loss and KL outside it stop the job (exit 2). With --pause N the state is
checkpointed after every N steps and the job exits 3; the same command resumes it. Results: results/regen.json.

Usage (repo root, L40): python experiments/qwen_maps/regen.py [--out DIR] [--pause N]
"""

from __future__ import annotations

import argparse
import json
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
from commute.qwen_sft import linears, save, sft_seq
from commute.qwen_train import Recipe, train
from commute.qwen_vpd import fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
E40 = HERE.parent / "qwen_sft" / "results"
OUT_JSON, PRED = HERE / "results" / "regen.json", HERE / "results" / "predictions.json"
RECIPE = Recipe(lr=1e-5, warmup=0, steps=100, schedule="constant", clip=None, micro=8)  # E40's Config as it ran
KEEP, SEQ_LEN, DEV, T0 = {1, 10, 100}, 512, "cuda", time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def write(rec: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(rec, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    ap.add_argument("--pause", type=int, default=RECIPE.steps, help="checkpoint and exit 3 every N steps")
    a = ap.parse_args()
    out, F = Path(a.out), json.loads((E40 / "F.json").read_text())
    tol, ckpt = json.loads(PRED.read_text())["rules"]["control_W0"]["tol"], Path(a.out) / "ckpt.pt"
    if ckpt.is_file():  # resume: drop what a lost job recorded past the checkpoint
        rec = json.loads(OUT_JSON.read_text())
        rec["steps"] = [s for s in rec["steps"] if s["step"] <= rec["paused"]]
        log(f"[regen] resuming after step {rec['paused']}")
    else:  # a fresh start keeps no file of an earlier trajectory
        shutil.rmtree(out / "dW", ignore_errors=True)
        rec = {"recipe": asdict(RECIPE), "tol": tol, "steps": [], "kl": {}}
    fp32_matmuls()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    end_id, train_set = int(tok.convert_tokens_to_ids("<|im_end|>")), qwen_sets.load()["train"]
    model = load_target(DEV, torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    maths = [torch.tensor([r], device=DEV) for r in json.loads((E40 / "maths_rows.json").read_text())["rows"]]
    rec["card"] = torch.cuda.get_device_name()
    log(f"[regen] {rec['card']}, out {out}, pause {a.pause}")

    def batch_at(t: int) -> list[Any]:
        return [sft_seq(tok, train_set[i]["question"], train_set[i]["gold"], end_id, SEQ_LEN)
                for i in F["steps"][t - 1]["batch"]]

    def save_dw(t: int, dW: dict[str, torch.Tensor]) -> None:
        save(out / "dW" / f"step_{t}.safetensors", dW, torch.bfloat16, droppable=False, log=log)

    def on_step(r: dict[str, Any]) -> None:
        t = r["step"]
        for k in ("norms", "crossed"):
            r.pop(k, None)
        if t in KEEP:
            ref = F["saved"][str(t)]["kl_to_start"]
            rel = r["kl"]["maths"] / ref - 1
            rec["kl"][str(t)] = {"kl": r["kl"]["maths"], "e40": ref, "rel": rel, "within_tol": abs(rel) <= tol}
        if t == RECIPE.steps:  # exact, as E40's step_100_fp32; one matrix at a time to the host
            dW = {s: (lin.weight.detach() - W0[s]).to("cpu") for s, lin in linears(model).items()}
            save(out / "dW" / f"step_{t}_fp32.safetensors", dW, torch.float32, droppable=False, log=log)
        if t % a.pause == 0 and t < RECIPE.steps:
            rec["paused"] = t  # train writes the checkpoint next
        rec["steps"].append(r)
        write(rec)
        log(f"[regen] step {t}: loss {r['loss']:.4f}, grad norm {r['grad_norm']:.2f}, |dW| {r['dw_norm']:.4f}"
            + (f", KL to start {json.dumps(rec['kl'][str(t)])}" if t in KEEP else ""))
        if t == 1:
            loss_rel = r["loss"] / F["steps"][0]["loss"] - 1
            if not (abs(loss_rel) <= tol and rec["kl"]["1"]["within_tol"]):
                rec["stopped"] = f"step 1 off E40 by more than {tol}: loss {loss_rel:+.4f}, KL {rec['kl']['1']['rel']:+.4f}"
                write(rec)
                log(f"STOP: {rec['stopped']}")
                sys.exit(2)

    reads = KEEP | set(range(a.pause, RECIPE.steps, a.pause))
    t = train(model, W0, batch_at, RECIPE, {"maths": maths}, reads, (), KEEP, save_dw, on_step, ckpt, a.pause)
    if t < RECIPE.steps:
        log(f"[regen] paused after step {t}: resubmit the same command")
        sys.exit(3)
    rec["done"] = True
    write(rec)
    log(f"[regen] done; KL to start against E40: {json.dumps(rec['kl'])}; peak GPU "
        f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
