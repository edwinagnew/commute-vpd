"""E103 (known_answer_pair), 2026-11-06: Qwen3-1.7B fine-tuned on TOFU's 217 real-fact rows with E84's recipe, the
known-answer partner of the fictional-author fine-tune. Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md.
Rules: results/predictions.json (committed before the run; the run refuses otherwise). Wordings: forms.py, done first.

  0  control first and alone: W0's real-fact multiple-choice share reproduces E81's, or the job stops. Then the first
     batch's loss and W0's log P on the trained answers (2 min on an L40).
  T  E84's recipe on the 5 pairs of each kept row for five epochs. The step nearest the template's norm (E91's rule) is
     saved, and C1, the log P gain on the real answers, is read there: under the bar the job stops, "no gain to carry".
     Then on to the five-epoch end, saved (about 9 min).
  G  real-fact multiple-choice share and the gain at both new states and the fictional matched and full states (4 min).
A stage whose result file says done is skipped; a lost T starts again from step 0. Exit 0 done; 1 a stop
(results/stopped.json, every stage refuses until it is removed) or a refusal. neurons.py is the second job.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from commute import qwen_d3 as d3
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.known_facts import REF_NORM, log_gap  # moved there 2026-11-06
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, linears, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe, train
from commute.qwen_updates import ok

E81: Any = load_experiment("qwen_tofu", "run")  # the recipe's numbers, `choices`, `write`, E45's job
E84: Any = load_experiment("qwen_tofu_forms", "run")  # its rate
DEV, F32, write = E81.DEV, E81.F32, E81.write

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, STOPPED, FORMS = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "forms.jsonl"
FILES = {"0": RESULTS / "stage_0.json", "T": RESULTS / "train.json", "G": RESULTS / "read.json"}
DW = LARGE / "dW"
MATCHED, END = DW / "matched.safetensors", DW / "end.safetensors"
FICTIONAL = {"fict_matched": ("anchor2_s0", "at_norm"), "fict_full": ("anchor2_s0", "full")}  # E91's records, seed 42


class Lab(E81.JOB.Job):  # E45's model and tokenizer; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(STOPPED, {"stopped": name, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    hard_exit(1)


def is_done(stage: str) -> bool:
    return FILES[stage].is_file() and bool(json.loads(FILES[stage].read_text()).get("done"))


def train_seqs(lab: Lab) -> list[tofu.Seq]:
    """The 5 pairs of every kept row, in row order (E84's `form_pairs`); the stream shuffles them."""
    real, seqs = tofu.real_rows(), []
    for rec in map(json.loads, FORMS.read_text().splitlines()):
        if rec["kept"]:
            seqs += [tofu.seq(lab.tok, lab.end_id, q, a) for q, a in tofu.form_pairs(real[rec["row"]], rec["questions"], rec["answers"])]
    return seqs


def answer_seqs(lab: Lab) -> list[tofu.Seq]:
    """Pair 0 of every row, TOFU's own question and answer as trained: the rows the gain is read on."""
    return [tofu.seq(lab.tok, lab.end_id, r["question"] + tofu.ASK, r["answer"]) for r in tofu.real_rows()]


def logp(lab: Lab, seqs: list[tofu.Seq]) -> list[float]:
    return tofu.mean_logp(lab.model, seqs, E81.READ_BATCH, DEV)


def gain(now: list[float], w0: list[float]) -> float:
    """Mean over rows of the change in each answer's mean token log P (nats a token)."""
    return statistics.mean(a - b for a, b in zip(now, w0, strict=True))


# ----------------------------------------------------------------------------- 0, the control


def stage_0(lab: Lab) -> None:
    lab.fp32()
    C, out = lab.pred["control_0"], {}
    out["real"] = E81.choices(lab, tofu.real())
    holds = abs(out["real"]["share"] - C["e81_share"]) <= C["tolerance"]
    lab.log(f"[0] control, W0's real-fact share: {json.dumps(out['real'])}; E81 {C['e81_share']} within {C['tolerance']}: "
            f"{'ok' if holds else 'MISSED'}")
    write(FILES["0"], out)
    if not holds:
        stop(lab, "control 0: W0's real-fact share", out["real"] | {"expected": C["e81_share"]})
    out["first_batch_loss"] = gold_loss(lab.model, tofu.stream(train_seqs(lab), E81.BATCH, E81.SEED, 1), E81.MICRO, DEV)
    out["w0_logp"] = logp(lab, answer_seqs(lab))
    lab.log(f"[0] first batch loss {out['first_batch_loss']:.6f}; W0 mean log P on the real answers "
            f"{statistics.mean(out['w0_logp']):.4f}")
    write(FILES["0"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- T, the fine-tune and C1


def stage_T(lab: Lab) -> None:
    m, seqs, s0 = lab.fp32(), train_seqs(lab), json.loads(FILES["0"].read_text())
    torch.cuda.reset_peak_memory_stats()
    F, C1, reads = lab.pred["first_minutes"], lab.pred["C1_gain"], answer_seqs(lab)
    steps = E81.EPOCHS * (len(seqs) // E81.BATCH)  # five epochs: 335 with every row kept
    r = Recipe(E84.RATE, E81.WARMUP, steps, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    out: dict[str, Any] = {"recipe": asdict(r), "batch": E81.BATCH, "sequences": len(seqs), "ref_norm": REF_NORM,
                           "first_batch_loss": s0["first_batch_loss"], "steps": [], "checks": [], "gains": {}}
    best: dict[str, Any] = {"gap": math.inf}
    lab.log(f"[T] {len(seqs)} sequences, {steps} steps; reference norm {REF_NORM:.6f}")

    def matched() -> None:
        save(MATCHED, best.pop("dW"), F32, droppable=False, log=lab.log)
        rec = {k: best[k] for k in ("step", "norm", "gain")} | {"bar": C1["bar"], "holds": best["gain"] >= C1["bar"]}
        out["matched"] = rec
        lab.log(f"[C1] gain at the matched step {rec['step']} (norm {rec['norm']:.4f}): {rec['gain']:.4f} nats a token; "
                f"bar {C1['bar']}: {'ok' if rec['holds'] else 'MISSED'}")
        write(FILES["T"], out)
        if not rec["holds"]:
            stop(lab, "C1: no gain to carry", rec)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        for key in ("norms", "crossed", "kl"):
            rec.pop(key, None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            write(FILES["T"], out)
            stop(lab, f"step {t} not finite", rec)
        for name, value, holds in tofu.step_checks(rec, s0["first_batch_loss"], F, E81.WARMUP):
            row = {"at": t, "name": name, "value": value, "holds": holds}
            out["checks"].append(row)
            lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
            if not holds:
                write(FILES["T"], out)
                stop(lab, f"step {t} {name}", row)
        n = rec["dw_norm"]
        if "matched" not in out and n > 0:
            if log_gap(n) < best["gap"]:  # a better candidate: its gain now, its δW held on the card
                g = gain(logp(lab, reads), s0["w0_logp"])
                out["gains"][str(t)] = g
                best.pop("dW", None)  # the old copy goes first: two on the card would be 11 GB
                best.update(gap=log_gap(n), step=t, norm=n, gain=g,
                            dW={s: lin.weight.data - lab.W0[s] for s, lin in linears(m).items()})
                lab.log(f"[T] candidate step {t}: norm {n:.4f}, gain {g:.4f}")
            elif n > REF_NORM:  # past the reference and moving away: the best is the matched state
                matched()
        if t % 25 == 0 or t <= 3:
            lab.log(f"[T] {json.dumps(rec)}")
            write(FILES["T"], out)

    def save_end(t: int, dW: dict[str, Any]) -> None:
        save(END, dW, F32, droppable=False, log=lab.log)
        out["end_step"] = t

    t0 = time.time()
    train(m, lab.W0, lambda t: tofu.stream(seqs, E81.BATCH, E81.SEED, t), r, {}, {steps}, {}, {steps}, save_end, on_step,
          LARGE / "ckpt_unused.pt", steps, dW_of=delta_fp32)
    if "matched" not in out:
        matched()
    norms = {rec["step"]: rec["dw_norm"] for rec in out["steps"] if rec["dw_norm"] > 0}
    out["argmin_step"] = min(norms, key=lambda t: log_gap(norms[t]))  # E91's rule over the whole record; must equal matched
    lab.log(f"[T] matched step {out['matched']['step']}, argmin over the record {out['argmin_step']}; end |dW| "
            f"{out['steps'][-1]['dw_norm']:.4f}")
    write(FILES["T"], out | {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()})
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- G, the readings


def states() -> dict[str, Path]:
    return {"real_matched": MATCHED, "real_full": END} | {k: d3.grid_state(a, s)["path"] for k, (a, s) in FICTIONAL.items()}


def stage_G(lab: Lab) -> None:
    lab.fp32()
    s0, R = json.loads(FILES["0"].read_text()), lab.pred["R1_recall"]
    out: dict[str, Any] = {"W0": {"real": s0["real"], "gain": 0.0}, "states": {}}
    for name, path in states().items():
        with set_weights(lab.model, plus(lab.W0, load_file(path, DEV))):
            out["states"][name] = {"real": E81.choices(lab, tofu.real()), "gain": gain(logp(lab, answer_seqs(lab)), s0["w0_logp"]),
                                   "path": str(path.relative_to(EXPERIMENTS))}
        lab.log(f"[G] {name} {json.dumps(out['states'][name])}")
        write(FILES["G"], out)
    w0 = s0["real"]["share"]
    out["R1"] = {n: "kept" if out["states"][n]["real"]["share"] >= w0 - R["drop"] else "dropped" for n in out["states"]}
    lab.log(f"[G] R1, real-fact share kept within {R['drop']} of W0's {w0:.4f}: {json.dumps(out['R1'])}")
    write(FILES["G"], out | {"done": True, "peaks": lab.peaks()})


STAGES = {"0": stage_0, "T": stage_T, "G": stage_G}


def filed() -> None:
    """Refuse unless predictions.json is committed unchanged and the wordings are done."""
    rel, repo = str(PRED.relative_to(EXPERIMENTS.parent)), str(EXPERIMENTS.parent)
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False)
    if git("diff", "--quiet", "HEAD", "--", rel).returncode or not git("ls-files", rel).stdout.strip():
        sys.exit(f"{rel} is not committed unchanged: commit it first")
    if not json.loads((RESULTS / "forms.json").read_text()).get("done"):
        sys.exit("results/forms.json is not done: run forms.py finish first")


def main() -> None:
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    todo = [s for s in STAGES if not is_done(s)]
    if not todo:
        print("nothing to do", flush=True)
        hard_exit(0)
    lab = Lab("all", f"todo {' '.join(todo)}", PRED)
    for s in todo:
        STAGES[s](lab)
        lab.log(f"[{s}] done; {lab.peaks()}")
    lab.log("FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()
