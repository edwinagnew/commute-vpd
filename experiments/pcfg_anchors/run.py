"""E55 Stage A: Jain et al.'s PCFG Counter anchors, pretraining. Plan: reports/plans/2026-10-29_plan_reuse-stage1.md.

Three minGPTs are pretrained on the five PCFG task families (src/commute/pcfg.py) with operand priors LOW (0.999,
0.001, 0), HIGH (0.5, 0.3, 0.2) and NEVER (0, 1, 0); behaviour is count-a and count-b accuracy on fresh C_Te = 0
Counter sequences (the answer token predicted at `=`).

Stages (repo root, project venv, one GPU):
  a0  rehearsal at tiny sizes ($0, not a reading): per prior, 500 pretraining iterations, stopped at 250 and resumed
      from the saved file; behaviour on 1K fresh samples; ms/step, generation rate, peak GPU memory and disk per
      checkpoint -> results/a0.json.
  a1  three 50K-iteration pretrainings (Jain Table 6). Each job trains until --deadline-min, saves and exits 3; resubmit
      until exit 0. As each prior finishes, its behaviour on 4K fresh samples is merged into results/a1_behaviour.json.
      Refuses to start without results/predictions.json.
Checkpoints: results/large/<stage>/pt_<prior>.pt (model, optimizer, step, RNG states).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from commute import pcfg
from commute.mingpt import GPT, GPTConfig
from commute.pcfg_train import Batches, Run, Schedule, answer_accuracy, train

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
A1_JSON = RESULTS / "a1_behaviour.json"

LR, WEIGHT_DECAY = 1e-3, 0.1  # L1164; weight decay: minGPT trainer default
TRAIN_SEED, EVAL_SEED = 0, 1  # disjoint streams: evaluation samples are fresh
EXIT_RESUME = 3


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the rehearsal and the scored pretraining."""

    iters: int
    batch: int
    eval_n: int  # fresh samples per behaviour number
    save_every: int
    log_every: int
    chunk: int  # batches drawn per generated chunk
    stop_first: int  # 0, or the step where the first call stops so the second resumes (the rehearsal)

    def __post_init__(self) -> None:
        assert self.iters % self.save_every == 0 and self.save_every % self.log_every == 0, self
        assert self.stop_first == 0 or (0 < self.stop_first < self.iters and self.stop_first % self.save_every == 0)
        assert self.eval_n >= 500 and self.batch > 0 and self.chunk > 0, self


A0 = Sizes(iters=500, batch=96, eval_n=1000, save_every=250, log_every=50, chunk=50, stop_first=250)
A1 = Sizes(iters=50_000, batch=96, eval_n=4000, save_every=1000, log_every=500, chunk=100, stop_first=0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device() -> torch.device:
    assert torch.cuda.is_available(), "GPU only: never train the model locally"
    return torch.device("cuda")


def pretrain(prior: str, sz: Sizes, ckpt: Path, *, stop_at: int | None = None,
             deadline: float | None = None) -> tuple[GPT, int, float]:  # fmt: skip
    """Pretrain (or resume) one prior's model; returns (model, step reached, ms/step)."""
    model = GPT(GPTConfig(vocab=pcfg.VOCAB, block=pcfg.SEQ - 1)).to(device())
    p = pcfg.PRIORS[prior]
    batches = Batches(lambda rng, n: pcfg.pretrain_batch(rng, n, p), TRAIN_SEED, sz.batch, sz.chunk, device())
    run = Run(Schedule(sz.iters, LR), WEIGHT_DECAY, sz.save_every, sz.log_every)
    step, ms = train(model, batches, run, ckpt, stop_at=stop_at, deadline=deadline, log=lambda m: log(f"{prior} {m}"))
    return model, step, ms


def behaviour(model: GPT, n: int) -> dict[str, float]:
    """Count-a and count-b accuracy on n fresh C_Te = 0 Counter sequences (the same samples for every model)."""
    out = {}
    for name, op in (("count_a", pcfg.A), ("count_b", pcfg.B)):
        seq, _ = pcfg.task_batch(np.random.default_rng([EVAL_SEED, op]), n, "C", op)
        out[name] = answer_accuracy(model, seq)
    return out


def generation_rate() -> dict[str, float]:
    """Sequences per second of one CPU thread: pretraining sequences, and Counter-b sequences at C_Tr = 1."""
    rng, n, out = np.random.default_rng(99), 9600, {}
    for name, make in (("pretrain_seq_per_s", lambda: pcfg.pretrain_batch(rng, n, pcfg.PRIORS["LOW"])),
                       ("spurious_seq_per_s", lambda: pcfg.task_batch(rng, n, "C", pcfg.B, c_rate=1.0))):  # fmt: skip
        t = time.perf_counter()
        make()
        out[name] = n / (time.perf_counter() - t)
    for name, c_rate in (("natural", 0.0), ("spurious", 1.0)):  # count-b label means (spurious_texts: not reweighted)
        seq, _ = pcfg.task_batch(rng, n, "C", pcfg.B, c_rate=c_rate)
        out[f"{name}_count_b_mean"] = float(seq[:, pcfg.ANS].mean() - pcfg.NUM0)
    return out


def stage_a0() -> int:
    out: dict[str, object] = {"generation": generation_rate()}
    log(f"generation: {out['generation']}")
    for prior in pcfg.PRIORS:
        ckpt = LARGE / "a0" / f"pt_{prior}.pt"
        ckpt.unlink(missing_ok=True)
        torch.cuda.reset_peak_memory_stats()
        _, s1, _ = pretrain(prior, A0, ckpt, stop_at=A0.stop_first)
        model, s2, ms = pretrain(prior, A0, ckpt)  # resumes from the file written at stop_first
        assert (s1, s2) == (A0.stop_first, A0.iters), (s1, s2)
        row = {"behaviour": behaviour(model, A0.eval_n), "ms_per_step": ms,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "checkpoint_mb": ckpt.stat().st_size / 1e6,
               "a1_minutes_at_this_rate": A1.iters * ms / 6e4}  # fmt: skip
        log(f"{prior}: {row}")
        out[prior] = row
    (RESULTS / "a0.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"wrote {RESULTS / 'a0.json'}")
    return 0


def stage_a1(priors: list[str], deadline_min: float) -> int:
    if not PRED.exists():
        sys.exit(f"file the predictions first: {PRED}")
    deadline = time.monotonic() + 60 * deadline_min
    for prior in priors:
        if prior in _read(A1_JSON):
            log(f"{prior}: already in {A1_JSON.name}")
            continue
        model, step, _ = pretrain(prior, A1, LARGE / "a1" / f"pt_{prior}.pt", deadline=deadline)
        if step < A1.iters:
            log(f"{prior}: stopped at step {step} of {A1.iters}; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        row = behaviour(model, A1.eval_n) | {"iters": step, "eval_n": A1.eval_n}
        log(f"{prior} behaviour: {row}")
        A1_JSON.write_text(json.dumps(_read(A1_JSON) | {prior: row}, indent=2) + "\n")  # re-read: jobs run in parallel
    return 0


# Probe stage (plan A1 rows 3-5: Def 1 probe, count-c selectivity, majority-class floor) and A2 (fine-tunes, ablation,
# pruning, reFT) go here once the plan's review has fixed their rules. Not written: the rules may change.


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("a0", "a1"))
    ap.add_argument("--prior", nargs="*", choices=list(pcfg.PRIORS), default=list(pcfg.PRIORS))
    ap.add_argument("--deadline-min", type=float, default=25.0, help="a1: save and exit after this many minutes")
    args = ap.parse_args()
    sys.exit(stage_a0() if args.stage == "a0" else stage_a1(args.prior, args.deadline_min))


if __name__ == "__main__":
    main()
