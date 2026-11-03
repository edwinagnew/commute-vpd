"""E69 (qwen_rl): the frozen config, and the job that holds the log, results files, checks, tokenizer, data and model for
run.py's stages. Imported by run.py as a sibling module (as experiments/qwen_updates).
"""

from __future__ import annotations

import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

from commute import qwen_sets
from commute.gsm8k import gold_number, grade
from commute.qwen_eval import MODEL, MODEL_REV, WINDOW, prompt, summary
from commute.qwen_eval import score as gsm_score
from commute.qwen_rl import (
    Records,
    delta_r,
    prompts_at,
    read_h,
    reward,
    rollouts,
    rows_for,
    split,
    step_seed,
)
from commute.qwen_sft import Seq, Uploader, linears, sha256
from commute.qwen_vpd import fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
RESULTS, PRED, LARGE = HERE / "results", HERE / "results" / "predictions.json", HERE / "results" / "large"
DEV, F32 = "cuda", torch.float32
GB = 10**9


@dataclass(frozen=True)
class Config:
    out: str = ""  # folder under results/ and results/large/
    prefix: str = "qwen_rl"  # W&B artifacts
    upload_cap: int | None = None  # bytes; a larger file stays local (`Uploader`)
    stop_on_bands: bool = True  # a value's band stops the job; a structural check always does
    seed: int = 69
    window: int = WINDOW
    lr: float = 1e-6
    problems: int = 64
    samples: int = 8
    steps: int = 100
    saves: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    n_cut: int = 50  # N's steps if the cost rule cuts it
    ckpt_steps: tuple[int, ...] = (20, 50)
    pause_at: int | None = None  # exit 3 after this step's checkpoint: the rehearsal of the resume
    drop_at: int = 20  # the reward-drop check: steps 11-20 against 1-10
    held: int = 128
    held_samples: int = 8
    n_gsm8k: int = 1319
    n_gsm1k: int = 1197
    gen_batch: int = 128  # greedy accuracy, as the baseline
    train_tokens: int = 2048  # rows x longest in a training micro-batch: E61's fp32 micro 4 x 512, 30 GB peak on the card
    read_tokens: int = 8192  # the same for a readout (no graph)
    max_rows: int = 512
    tokens_per_row: int = 256  # peak cached tokens per sampling row once finished rows have left; S reads the real peak
    reserved_gb: int = 30  # fp32 model 7, trainable copy 5.6, Adam 11.3, margin: not available to the sampler's cache
    generate_tokens_per_row: int = 768  # `generate` keeps every row to the longest prompt plus its whole window
    generate_reserved_gb: int = 8  # S's comparison runs before any trainable copy or Adam state exists
    setup_h: float = 1.0  # the plan's set-up and accuracy allowance

    def __post_init__(self) -> None:
        assert list(self.saves) == sorted(self.saves) and self.steps == self.saves[-1] and self.n_cut in self.saves
        assert set(self.ckpt_steps) < set(range(1, self.steps)) and self.drop_at % 2 == 0 and self.drop_at <= self.steps
        assert self.pause_at is None or self.pause_at in self.ckpt_steps
        assert self.window == WINDOW
        assert (self.out == "") == self.stop_on_bands == (self.upload_cap is None)

    @property
    def divisor(self) -> int:
        return self.problems * self.samples * self.window


SIZES = {"full": Config(),
         "tiny": Config(out="tiny", prefix="qwen_rl_tiny", upload_cap=10**8, stop_on_bands=False, problems=4, samples=2,
                        steps=3, saves=(1, 3), n_cut=1, ckpt_steps=(2,), pause_at=2, drop_at=2, held=4, n_gsm8k=8,
                        n_gsm1k=8)}


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text()) if p.is_file() else {}


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)


class Job:
    """One stage's process: its log, results files and checks, and the data and the model every stage shares."""

    def __init__(self, cfg: Config, stage: str) -> None:
        self.cfg, self.t0, self.stage = cfg, time.time(), stage
        self.res, self.large = RESULTS / cfg.out, LARGE / cfg.out
        if LARGE.is_symlink() and not LARGE.exists():  # a link synced to a pod arrives dangling
            LARGE.unlink()
        self.large.mkdir(parents=True, exist_ok=True)
        self.pred = json.loads(PRED.read_text())
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.tok.padding_side = "left"
        self.sets = qwen_sets.load()
        self.held, self.train = split(qwen_sets.N_TRAIN, cfg.seed, cfg.held)
        assert len(self.train) >= cfg.steps * cfg.problems
        self.prompts: dict[int, list[int]] = {}
        self.model: Any = None
        self.log(f"[{stage}] start; {torch.cuda.get_device_name()}")

    # ------------------------------------------------------------------------- log, files, checks

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (self.res / "run.log").open("a") as f:
            f.write(line + "\n")

    def peaks(self) -> dict[str, float]:
        return {"gpu_gb": torch.cuda.max_memory_allocated() / GB,
                "host_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}

    def get(self, name: str) -> dict[str, Any]:
        return read_json(self.res / f"{name}.json")

    def put(self, name: str, obj: dict[str, Any]) -> None:
        write_text(self.res / f"{name}.json", json.dumps(obj, indent=1, default=float))

    def check(self, out: dict[str, Any], name: str, holds: bool, detail: dict[str, Any], band: bool = False) -> None:
        """Record a check in `out`; a miss stops the job, except a value's band at the tiny size, where it means nothing."""
        out.setdefault("checks", {})[name] = {"holds": holds, **detail}
        self.log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
        if holds or (band and not self.cfg.stop_on_bands):
            return
        out["stopped"] = name
        self.put(self.stage, out)
        self.log(f"STOP: {name}")
        sys.exit(2)

    def begin(self, *needs: str) -> dict[str, Any] | None:
        """This stage's record, None if it is done; refuses after a stop or before the stages it needs."""
        out = self.get(self.stage)
        if out.get("stopped"):
            sys.exit(f"{self.stage} stopped at {out['stopped']!r}: the user decides; move its file to run it again")
        if out.get("done"):
            self.log(f"[{self.stage}] done earlier: skipped")
            return None
        for s in needs:
            if not self.get(s).get("done"):
                sys.exit(f"stage {s} must finish first")
        return out

    def publish(self, up: Uploader, path: Path) -> None:
        """Add `path` to SHA256SUMS (relative to the large folder, as `sha256sum -c` reads from there) and queue it."""
        sums_file, rel = self.res / "SHA256SUMS", str(path.relative_to(self.large))
        pairs = [line.split(maxsplit=1) for line in sums_file.read_text().splitlines()] if sums_file.is_file() else []
        sums = {name.strip(): h for h, name in pairs} | {rel: sha256(path)}
        write_text(sums_file, "".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
        up.add(path)

    def uploader(self) -> Uploader:
        return Uploader(self.cfg.prefix, self.cfg.upload_cap, self.large / "wandb", self.log)

    # ------------------------------------------------------------------------- model, prompts, rollouts

    def fp32(self) -> Any:
        """The fp32 target (the readers' model): gradient checkpointing for training, the generation config reset as in
        `qwen_eval.load_model`, so greedy answers read only what is passed; W0 its matrices."""
        self.model = load_target(DEV, F32)
        eos = self.model.generation_config.eos_token_id
        self.model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=self.tok.pad_token_id)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        self.numel = sum(w.numel() for w in self.W0.values())
        total = torch.cuda.get_device_properties(0).total_memory
        self.rows = rows_for(total, self.cfg.reserved_gb * GB, self.cfg.tokens_per_row, self.cfg.max_rows)
        self.log(f"[model] {self.numel} entries; {self.rows} sampling rows per batch; eos {eos}")
        return self.model

    def prompt_ids(self, pid: int) -> list[int]:
        if pid not in self.prompts:
            text = prompt(self.tok, self.sets["train"][pid]["question"])
            self.prompts[pid] = self.tok(text, add_special_tokens=False)["input_ids"]
        return self.prompts[pid]

    def seqs(self, recs: Records) -> list[Seq]:
        return [(self.prompt_ids(r["pid"]) + r["ids"], len(self.prompt_ids(r["pid"]))) for r in recs]

    def graded(self, pids: list[int], outs: list[tuple[list[int], bool]], n: int) -> Records:
        recs = []
        for j, (ids, cut) in enumerate(outs):
            pid = pids[j // n]
            strict = bool(grade(self.tok.decode(ids, skip_special_tokens=True),
                                gold_number(self.sets["train"][pid]["gold"]))["strict"])
            recs.append({"pid": pid, "k": j % n, "ids": ids, "truncated": cut, "strict": strict, "reward": reward(strict, cut)})
        return recs

    def sampled(self, pids: list[int], n: int, t: int) -> Records:
        """n samples per problem from the weights as they stand, seeded by step: the same in every arm."""
        torch.manual_seed(step_seed(self.cfg.seed, t))
        outs = rollouts(self.model, [self.prompt_ids(p) for p in pids], n, self.cfg.window, self.rows, self.tok.pad_token_id)
        return self.graded(pids, outs, n)

    def step_pids(self, t: int) -> list[int]:
        pids = prompts_at(self.train, t, self.cfg.problems)
        assert not set(pids) & set(self.held)
        return pids

    def read_state(self) -> dict[str, Any]:
        """ΔR and KL on H at the weights as they stand, against W0."""
        recs = load_rollouts(self.large / "rollouts" / "H.jsonl")
        changes, kl = read_h(self.model, self.W0, self.seqs(recs), self.cfg.read_tokens)
        return {"delta_r": delta_r(changes, recs), "kl": kl}

    def accuracy(self, label: str, up: Uploader) -> dict[str, Any]:
        """Greedy strict accuracy of the weights as they stand, as the baseline (`qwen_eval.score`)."""
        out = {}
        for name, items in (("gsm8k", self.sets["test"][: self.cfg.n_gsm8k]), ("gsm1k", self.sets["gsm1k"][: self.cfg.n_gsm1k])):
            sink = self.large / "acc" / f"{label}_{name}.jsonl"
            sink.parent.mkdir(exist_ok=True)
            recs = gsm_score(self.tok, self.model, items, math.inf, sink, f"{label} {name}", self.cfg.gen_batch)
            out[name] = {"summary": summary(recs), "recs": recs}
            self.publish(up, sink)
            self.log(f"[accuracy {label} {name}] {json.dumps(out[name]['summary'])}")
        return out


def load_rollouts(p: Path) -> Records:  # any of the run's jsonl files
    return [json.loads(line) for line in p.read_text().splitlines()]


def save_rollouts(p: Path, recs: Records) -> None:
    write_text(p, "".join(json.dumps(r) + "\n" for r in recs))
