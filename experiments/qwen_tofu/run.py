"""E81 (qwen_tofu), 2026-11-04: anchor 2, TOFU, on Qwen3-1.7B. Plan: reports/plans/2026-11-04_plan_e81-tofu.md. Rules:
results/predictions_amend2.json (the run refuses to start without it). Data and rules: `commute.qwen_tofu`.

  0        the unchanged model on TOFU (about 10 min), as ran under Amendment 1: not rerun. Real facts are the control.
  T rate   TOFU's fine-tune recipe at this peak rate, fp32 forward, δW in fp32 on E78's grid and on to the end (about 70 min).
  G rate   needs T: the files' checks, then the last step and the first step past the bar read in full (about 15 min).
Exit 0 done; 1 a stop (results/stopped.json, every stage refuses until it is removed) or a refusal. A lost T starts again.
Step t's δW has t - 1 moving steps: step 1 runs at rate 0, so its file is all zeros, as E78's.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe, train
from commute.qwen_updates import ok

JOB: Any = load_experiment("qwen_updates", "job")
E78: Any = load_experiment("qwen_anchor1", "run")  # its header, moments and file_row: the same checks on the same format
DEV, F32, CFG = JOB.DEV, JOB.F32, JOB.CFG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, STAGE0, STOPPED = RESULTS / "predictions_amend2.json", RESULTS / "stage0.json", RESULTS / "stopped.json"
LARGE = RESULTS / "large"
BATCH, EPOCHS, WARMUP, MICRO, SEED = 16, 5, 250, 8, 42  # TOFU's config and finetune.py; micro changes no gradient
STEPS = EPOCHS * 4000 // BATCH
SAVES = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, STEPS)  # E78's grid, then on to the end
CHOICE_AT = {1, 10, 20, 50, *range(100, STEPS, 100), STEPS}  # the tracked rows, every 4th of the 800
READ_BATCH, GEN_BATCH, MAX_NEW = 64, 128, 128  # the longest gold answer fits MAX_NEW (`data` in predictions)


class Lab(JOB.Job):  # E45's model and tokenizer; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def write(path: Path, obj: dict[str, Any]) -> None:
    (tmp := path.with_name(path.name + ".tmp")).write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(STOPPED, {"stopped": name, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    sys.exit(1)


def choices(lab: Lab, items: list[tofu.Choice]) -> dict[str, Any]:
    hits = tofu.choice_read(lab.model, lab.tok, lab.end_id, items, READ_BATCH, DEV)
    share, lo, hi = tofu.cluster_ci(hits, [c.author for c in items])
    return {"n": len(items), "share": share, "ci": [lo, hi]}


def generated(lab: Lab, questions: list[str], references: list[str], sink: Path, ask: str = tofu.ASK) -> float:
    """Greedy answers on the model as it stands, written to `sink`; their mean recall of the references."""
    m, out = lab.model, []
    m.generation_config = GenerationConfig(eos_token_id=m.generation_config.eos_token_id, pad_token_id=lab.tok.pad_token_id)
    texts = [tofu.chat(lab.tok, q + ask) for q in questions]
    for s in range(0, len(texts), GEN_BATCH):
        enc = lab.tok(texts[s : s + GEN_BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        with torch.no_grad():
            full = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        out += lab.tok.batch_decode(full[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    sink.parent.mkdir(parents=True, exist_ok=True)
    sink.write_text("".join(json.dumps({"question": q, "reference": r, "response": a}) + "\n"
                            for q, r, a in zip(questions, references, out, strict=True)))
    return statistics.mean(tofu.recall(a, r) for a, r in zip(out, references, strict=True))


def train_seqs(lab: Lab) -> list[tofu.Seq]:
    return [tofu.seq(lab.tok, lab.end_id, r["question"] + tofu.ASK, r["answer"]) for r in tofu.rows("full")]


def stage_0(lab: Lab) -> None:
    lab.fp32()
    R, out = lab.pred["stage_0"], {}
    first = tofu.stream(train_seqs(lab), BATCH, SEED, 1)
    out["first_batch_loss"] = gold_loss(lab.model, first, MICRO, DEV)
    lab.log(f"[0] first batch loss {out['first_batch_loss']:.6f}")
    out["real"] = choices(lab, tofu.real())
    lab.log(f"[0] real facts, control first: {json.dumps(out['real'])}")
    write(STAGE0, out)
    if out["real"]["share"] < R["real_min"]:
        stop(lab, "the control: W0 does not know real facts", out["real"])
    for key, asked in (("paraphrased", "paraphrased_question"), ("original", "question")):
        out[key] = choices(lab, tofu.held(asked))
        lab.log(f"[0] {key} questions: {json.dumps(out[key])}")
        write(STAGE0, out)
    held, never = tofu.rows("retain_perturbed") + tofu.rows("forget10_perturbed"), tofu.rows("holdout10")
    out["recall_paraphrased"] = generated(lab, [r["paraphrased_question"] for r in held], [r["answer"] for r in held],
                                          LARGE / "answers" / "W0_paraphrased.jsonl")
    out["recall_never"] = generated(lab, [r["question"] for r in never], [r["answer"] for r in never],
                                    LARGE / "answers" / "W0_never.jsonl")
    out["bar"] = tofu.choice_bar(out["real"]["share"], 6, 4)
    out["at_chance"] = all(out[k]["ci"][0] <= 1 / 6 <= out[k]["ci"][1] for k in ("paraphrased", "original"))
    lab.log(f"[0] {json.dumps(out)}")
    write(STAGE0, out | {"done": True, "peaks": lab.peaks()})
    if not out["at_chance"]:
        stop(lab, "not at chance: stop and ask the user", {k: out[k] for k in ("paraphrased", "original")})


def run_paths(rate: str) -> tuple[Path, Path, Path]:
    return RESULTS / f"train_{rate}.json", RESULTS / f"read_{rate}.json", LARGE / "dW" / f"a2_{rate}"


def check_step(lab: Lab, rate: str, first_loss: float, rec: dict[str, Any], out: dict[str, Any], path: Path) -> None:
    """Filed bands at steps 1, 2 and 250: each logs its reading, a miss writes `out` and stops."""
    t, F = rec["step"], lab.pred["first_minutes"]
    rows: list[tuple[str, Any, bool]] = []
    if t == 1:
        gap = abs(rec["loss"] - first_loss) / first_loss
        rows = [("loss against stage 0", gap, gap <= F["step_1_loss_rel"]), ("rate", rec["lr"], rec["lr"] == 0),
                ("|dW|", rec["dw_norm"], rec["dw_norm"] == 0)]
    elif t == 2:
        lo, hi = F["step_2_dw"][rate]
        rows = [("|dW|", rec["dw_norm"], lo <= rec["dw_norm"] <= hi), ("peak", rec["peak_gib"], rec["peak_gib"] <= F["peak_gib"])]
    elif t == WARMUP:
        rows = [("loss / first", rec["loss"] / first_loss, rec["loss"] <= F["epoch_1_loss_frac"] * first_loss)]
    for name, value, holds in rows:
        row = {"at": t, "name": name, "value": value, "holds": holds}
        out["checks"].append(row)
        lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
        if not holds:
            write(path, out)
            stop(lab, f"step {t} {name}", row)


def stage_T(lab: Lab, rate: str) -> None:
    path, _, dw = run_paths(rate)
    s0 = json.loads(STAGE0.read_text())
    if not s0.get("done"):
        sys.exit("stage 0 must finish first")
    m, maths, seqs = lab.fp32(), lab.maths()[: CFG.track_rows], train_seqs(lab)
    tracked = tofu.held("paraphrased_question")[::4]
    r = Recipe(float(rate), WARMUP, STEPS, "hf_cosine", 1.0, MICRO, forward="fp32")
    out: dict[str, Any] = {"recipe": asdict(r), "batch": BATCH, "steps": [], "saves": [], "checks": [], "seconds": 0.0}

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        save(dw / f"step_{t}.safetensors", dW, F32, droppable=False, log=lab.log)
        out["saves"].append(t)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        rec.pop("norms")
        rec.pop("crossed", None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            stop(lab, f"step {t} not finite", rec)
        check_step(lab, rate, s0["first_batch_loss"], rec, out, path)
        if t in CHOICE_AT:
            rec["choice"] = choices(lab, tracked)["share"]
        if t in CHOICE_AT or t <= 3:
            lab.log(f"[T] {json.dumps(rec)}")
            write(path, out)

    t0 = time.time()
    train(m, lab.W0, lambda t: tofu.stream(seqs, BATCH, SEED, t), r, {"maths": maths}, set(SAVES), {"maths": ()}, set(SAVES),
          save_dw, on_step, LARGE / "ckpt_unused.pt", STEPS, dW_of=delta_fp32)
    write(path, out | {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()})


def check_files(lab: Lab, dw: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    F, rows = lab.pred["files"], {}
    for t in SAVES:
        p = dw / f"step_{t}.safetensors"
        dtypes = E78.header(p)
        norm = math.sqrt(E78.moments(p, p)[0]) if dtypes else None
        rows[str(t)] = E78.file_row(dtypes, norm, steps[t - 1]["dw_norm"], F)
        lab.log(f"[files] step {t} {json.dumps(rows[str(t)])}")
    if not all(r["holds"] for r in rows.values()):
        stop(lab, "a saved file misses", rows)
    return rows


def read_state(lab: Lab, rate: str, t: int, bar: float | None) -> dict[str, Any]:
    """One saved state read in full: the 800 by multiple choice, real facts, and greedy answers to the paraphrases, to the
    trained wording of the same rows, and to authors never trained."""
    held, never = tofu.rows("retain_perturbed") + tofu.rows("forget10_perturbed"), tofu.rows("holdout10")
    sink = LARGE / "answers" / rate / f"step_{t}"
    with set_weights(lab.model, plus(lab.W0, load_file(run_paths(rate)[2] / f"step_{t}.safetensors", DEV))):
        out: dict[str, Any] = {"paraphrased": choices(lab, tofu.held("paraphrased_question")), "real": choices(lab, tofu.real())}
        refs = [r["answer"] for r in held]
        out |= {"recall_paraphrased": generated(lab, [r["paraphrased_question"] for r in held], refs, sink / "paraphrased.jsonl"),
                "recall_trained": generated(lab, [r["question"] for r in held], refs, sink / "trained.jsonl"),
                "recall_never": generated(lab, [r["question"] for r in never], [r["answer"] for r in never], sink / "never.jsonl")}
    out["answered"] = tofu.answered(out["paraphrased"]["share"], bar, out["recall_paraphrased"], out["recall_trained"], out["recall_never"])
    lab.log(f"[G] step {t} {json.dumps(out)}")
    return out


def stage_G(lab: Lab, rate: str) -> None:
    path, read_path, dw = run_paths(rate)
    run = json.loads(path.read_text()) if path.is_file() else {}
    if not run.get("done"):
        sys.exit("T must finish first")
    lab.fp32()
    bar = lab.pred["rule"]["bar_value"]  # Amendment 2's, from where W0 starts; stage0.json's "bar" is the superseded one
    read: dict[str, Any] = {"bar": bar, "files": check_files(lab, dw, run["steps"]), "states": {}}
    write(read_path, read)
    tracked = {s["step"]: s["choice"] for s in run["steps"] if "choice" in s}
    past = [t for t in SAVES if bar is not None and tracked.get(t, 0) >= bar]
    read["first_past_bar"] = past[0] if past else None
    for t in sorted({STEPS, *past[:1]}, reverse=True):
        read["states"][str(t)] = read_state(lab, rate, t, bar)
        write(read_path, read)
    write(read_path, read | {"done": True, "peaks": lab.peaks()})


def main() -> None:
    p = argparse.ArgumentParser(description="E81: anchor 2, TOFU; stages in the module docstring")
    p.add_argument("--stage", required=True, choices=("0", "T", "G"))
    p.add_argument("--rate", choices=("1e-5", "2e-5"))
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    assert (a.rate is None) == (a.stage == "0")
    lab = Lab(a.stage, a.rate or "W0", PRED)
    stage_0(lab) if a.stage == "0" else (stage_T if a.stage == "T" else stage_G)(lab, a.rate)
    lab.log(f"[{a.stage}] done; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()
