"""E84 Amendment 1 (qwen_tofu_forms), 2026-11-04: stages T and G of E84 again, 10 epochs instead of 5, nothing else changed.
Plan: reports/plans/2026-11-04_plan_e84-rephrasings.md (tree: "Not answered, trained still over 0.9: Amendment 1, once").
Rules: results/predictions_amend1.json (this refuses to start without it). It calls E84's `run.py` and replaces only module
constants there: steps, saves, reads, the paths of the δW files, answers, results, log and stop file. Stages 0 and R are E84's
and stay done; W0's answers are E84's (`results/large/answers/W0`), not rerun.

  sbatch cluster/qwen_tofu_forms_amend1.sbatch            (T then G; a stage already done is skipped, a lost T starts again)

The stream is E84's: 787 kept rows x 5 pairs = 3935 sequences, 245 whole batches of 16 an epoch, 2500 steps (10 * 4000 // 16,
as E81 computes steps) = 10.2 passes. Exit 0 done; 1 a stop (results/stopped_amend1.json) or a refusal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from commute.jobs import hard_exit, load_experiment

R = load_experiment("qwen_tofu_forms", "run")  # E84's stages, constants and Lab; module attributes below are replaced
E81 = R.E81

EPOCHS = 10
STEPS = EPOCHS * 4000 // E81.BATCH  # E81's rule (EPOCHS * 4000 // BATCH), 10 epochs
SAVES = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 1250, 2000, 2500)
READS = (STEPS, 1250)
RESULTS = R.RESULTS
PRED = RESULTS / "predictions_amend1.json"
STOPPED = RESULTS / "stopped_amend1.json"
LOG = RESULTS / "run_amend1.log"
DW = R.LARGE / "dW" / "a2_2e-5_e10"
W0_ANSWERS, ANSWERS_E10 = R.ANSWERS / "W0", R.LARGE / "answers_e10"
FILES = {"T": RESULTS / "amend1_train.json", "G": RESULTS / "amend1_read.json"}


class Answers:
    """`ANSWERS / "W0"` is E84's saved W0 answers; any other name (`step_t`) is under answers_e10."""

    def __truediv__(self, name: str) -> Path:
        return W0_ANSWERS if name == "W0" else ANSWERS_E10 / name


class Lab(R.Lab):  # the log is this amendment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with LOG.open("a") as f:
            f.write(line + "\n")


def override() -> None:
    E81.STEPS, E81.SAVES = STEPS, SAVES
    R.READS, R.DW, R.ANSWERS, R.STOPPED = READS, DW, Answers(), STOPPED
    R.FILES["T"], R.FILES["G"] = FILES["T"], FILES["G"]  # "0" and "R" stay E84's, done


def main() -> None:
    p = argparse.ArgumentParser(description="E84 Amendment 1: stages T and G at 10 epochs")
    p.add_argument("--stage", default="all", choices=("all", "T", "G"))
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    assert R.LARGE.is_symlink(), f"{R.LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    override()
    R.need("R")
    assert (W0_ANSWERS / "paraphrased.jsonl").is_file(), f"{W0_ANSWERS}: E84's W0 answers are missing"
    todo = [s for s in ("T", "G") if a.stage in ("all", s) and not R.is_done(s)]
    if not todo:
        print(f"nothing to do: {a.stage} is done", flush=True)
        hard_exit(0)
    lab = Lab(a.stage, f"amend1 todo {' '.join(todo)}", PRED)
    for s in todo:
        R.STAGES[s](lab)
        lab.log(f"[{s}] done; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()
