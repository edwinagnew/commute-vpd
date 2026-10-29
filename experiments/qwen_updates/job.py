"""E45 (qwen_updates): the config, and the job that holds the tokenizer, the GSM8K problems, the models and every
arm's sequences and rows for run.py's stages. Imported by run.py as a sibling module (as experiments/qwen_maps).
"""

from __future__ import annotations

import json
import math
import random
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_facts, qwen_sets
from commute.gsm8k import gold_number, grade
from commute.qwen_anchors import Shot, candidate, chat
from commute.qwen_eval import (
    GREEDY,
    MAX_NEW,
    MODEL,
    MODEL_REV,
    Records,
    load_model,
    prompt,
    sample,
    summary,
)
from commute.qwen_eval import score as gsm_score
from commute.qwen_sft import Item, Seq, linears, pack, sft_seq
from commute.qwen_updates import (
    cycle,
    keep_correct,
    keep_finished,
    pack_whole,
    packed_stream,
)
from commute.qwen_vpd import fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
LARGE = RESULTS / "large"  # on the cluster a link into commute-data (samples, δW, responses)
MATHS = HERE.parent / "qwen_sft" / "results" / "maths_rows.json"
BASELINE = HERE.parent / "qwen_gsm8k_baseline" / "results" / "large" / "greedy.jsonl"
DEV, F32, BF16 = "cuda", torch.float32, torch.bfloat16
SAMPLED = {"own": "own", "null": "own", "a1": "a1"}  # data -> sample set


@dataclass(frozen=True)
class Config:
    seq_len: int = 512
    batch: int = 64  # sequences per step (anchor 2: rows matched to gold's tokens, `a2_rows`)
    micro: int = 8
    steps: int = 100
    warmup: int = 10
    schedule: str = "hf_cosine"
    clip: float = 1.0
    grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3)
    dense_until: int = 30  # read every step to here, then every `every`
    every: int = 5
    track_rows: int = 16
    own_rows: int = 64
    e40_seed: int = 0  # E40's split of GSM8K train: 256 held out, the rest in its order
    e40_held: int = 256
    pool_seed: int = 45
    n_problems: int = 2000  # sampled; the last n_held held out of training
    n_held: int = 64
    n_search: int = 512
    n_shot_pool: int = 16
    n_shots: int = 4
    own_k: int = 4
    a1_k: int = 8  # seed s trains on samples [4s, 4s + 4)
    kv_tokens: int = 262_144  # cached tokens per sampling batch (~29 GB at 114 kB per token)
    gen_batch: int = 128  # greedy GSM8K answers, as the baseline
    facts_batch: int = 256
    facts_new: int = 32
    eval_problems: int = 256  # GSM8K test 0-255
    a1_rows_from: int = 256  # anchor 1's own rows: test problems 256-511
    control_lr: float = 1e-5
    control_micro: int = 4  # E40a1's stage L
    budget_s: float = 1500.0  # P, S, E stop before a unit that would pass this (debug jobs are 30 min)
    t_chunk: int = 150  # T writes a checkpoint and exits 3 after every t_chunk steps (a 300-step run: two jobs)

    def __post_init__(self) -> None:
        assert self.batch % self.micro == 0 and self.n_held < self.n_problems
        assert self.own_k * 2 == self.a1_k and self.track_rows <= self.own_rows
        assert self.dense_until % self.every == 0 and 0 < self.warmup < self.steps
        assert self.a1_rows_from >= self.eval_problems and list(self.grid) == sorted(self.grid)
        assert self.t_chunk % self.every == 0 and self.t_chunk > self.steps  # a read step; 100-step runs never pause


CFG = Config()


@dataclass(frozen=True)
class Arm:
    data: str  # what it trains on (`Job.arm_data`) and how it is read (run.py)
    lr: float = 1e-5  # peak rate
    steps: int = CFG.steps
    seeds: tuple[int, ...] = (0,)
    pred: str = "predictions.json"  # in results/; the run refuses to start without it


ARMS = {"gold": Arm("gold"), "own": Arm("own"), "own2": Arm("own", lr=2e-5), "null": Arm("null"),
        "a1": Arm("a1", seeds=(0, 1)), "a2": Arm("a2", seeds=(0, 1)),
        "a2long": Arm("a2", steps=300, seeds=(0, 1), pred="predictions_amend1.json")}  # Amendment 1 (plan section 8)


class Job:
    """One stage's process: log, results files, budget, and the data every stage shares (built lazily)."""

    def __init__(self, stage: str, run: str, pred: Path = PRED) -> None:
        if not pred.is_file():
            sys.exit(f"{pred} is missing: predictions are filed before any run")
        self.pred, self.stage, self.run, self.t0 = json.loads(pred.read_text()), stage, run, time.time()
        self.cfg = CFG
        LARGE.mkdir(parents=True, exist_ok=True)
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.tok.padding_side = "left"
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.sets = qwen_sets.load()
        perm = torch.randperm(qwen_sets.N_TRAIN, generator=torch.Generator().manual_seed(CFG.e40_seed)).tolist()
        self.held_idx, self.train_idx = perm[: CFG.e40_held], perm[CFG.e40_held :]
        self.pool = random.Random(CFG.pool_seed).sample(self.train_idx, len(self.train_idx))
        self.problems = self.pool[: CFG.n_problems]
        self.search = self.pool[CFG.n_problems : CFG.n_problems + CFG.n_search]
        self.shot_pool = self.pool[CFG.n_problems + CFG.n_search :][: CFG.n_shot_pool]
        self.model: Any = None
        self.W0: dict[str, Tensor] = {}
        self.gen_model: Any = None
        self.unit_s = 0.0
        self.log(f"[{stage} {run}] start; {torch.cuda.get_device_name()}")

    # ------------------------------------------------------------------------- log, files, budget

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")

    def peaks(self) -> str:
        host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        return f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, host {host:.1f} GB"

    def get(self, name: str) -> dict[str, Any]:
        p = RESULTS / f"{name}.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def put(self, name: str, obj: dict[str, Any]) -> None:
        p = RESULTS / f"{name}.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=float))
        tmp.replace(p)

    def check(self, out: dict[str, Any], name: str, holds: bool, detail: dict[str, Any], file: str) -> None:
        """Record a first-minutes check in `out` (written to results/<file>.json); a miss stops the job (exit 2)."""
        out.setdefault("checks", {})[name] = {"holds": holds, **detail}
        self.log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
        if not holds:
            out["stopped"] = name
            self.put(file, out)
            self.log(f"STOP: {name}")
            sys.exit(2)

    def budget(self, est_s: float) -> None:
        """Exit 3 when a unit of about est_s (or the slowest so far) would pass the budget."""
        if time.time() - self.t0 + max(est_s, self.unit_s) > CFG.budget_s:
            self.pause(f"{time.time() - self.t0:.0f} s used")

    def pause(self, why: str) -> None:
        """Exit 3: the work so far is saved; resubmit the same command to go on."""
        self.log(f"BUDGET: {why}; resubmit the same command to continue")
        sys.exit(3)

    def timed(self, t_start: float) -> None:
        self.unit_s = max(self.unit_s, time.time() - t_start)

    # ------------------------------------------------------------------------- models

    def fp32(self) -> Any:
        """The fp32 target with gradient checkpointing (training and every KL reading); W0 its matrices."""
        if self.model is None:
            self.model = load_target(DEV, F32)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        return self.model

    def gen(self, W: dict[str, Tensor] | None = None) -> Any:
        """The bf16 generation copy (`qwen_eval.load_model`) holding W (None: W0)."""
        if self.gen_model is None:
            _, self.gen_model = load_model()
            self.gen_W0 = {s: lin.weight.data.clone() for s, lin in linears(self.gen_model).items()}
        with torch.no_grad():
            for s, lin in linears(self.gen_model).items():
                lin.weight.copy_(self.gen_W0[s] if W is None else W[s])
        return self.gen_model

    # ------------------------------------------------------------------------- sequences and rows

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def rows(self, lists: list[list[int]]) -> list[Tensor]:
        return [torch.tensor([r], device=DEV) for r in lists]

    def maths(self) -> list[Tensor]:
        return self.rows(json.loads(MATHS.read_text())["rows"])

    def gold_seq(self, i: int) -> Seq:
        it = self.sets["train"][i]
        return sft_seq(self.tok, it["question"], it["gold"], self.end_id, CFG.seq_len)

    def fact_seq(self, q: qwen_facts.QA) -> Seq:
        p = self.ids(chat(self.tok, q.question))
        return p + self.ids(q.answer) + [self.end_id], len(p)

    def sample_seq(self, r: dict[str, Any]) -> Seq:
        """An unprompted chat (`qwen_eval.prompt`) followed by a sampled response's ids, cut to seq_len."""
        p = self.ids(prompt(self.tok, self.sets["train"][r["idx"]]["question"]))
        return (p + r["ids"])[: CFG.seq_len], len(p)

    def samples(self, name: str) -> list[dict[str, Any]]:
        d = LARGE / "S" / name
        assert self.get(f"S_{name}").get("complete"), f"sampling {name} is not complete (run --stage S --set {name})"
        return [json.loads(line) for p in sorted(d.glob("batch_*.jsonl")) for line in p.read_text().splitlines()]

    def a1_text(self, question: str, name: str | None = None, shots: list[Shot] | None = None) -> str:
        """The maths chat under a candidate prompt (default: P's chosen one, with P's few-shot answers)."""
        P = self.get("P") if name is None or shots is None else {}
        sh = shots if shots is not None else [(q, a) for q, a in P["shots"]]
        return candidate(self.tok, name or P["chosen"], question, sh)

    def arm_data(self, arm: str, seed: int) -> tuple[list[Item], int, list[list[int]] | None, dict[str, Any]]:
        """The training stream (steps x per-step items, in order), items per step, the own rows (None: the maths rows
        are the own rows) and a summary."""
        own, d, steps = self.own_rows(arm, seed), ARMS[arm].data, ARMS[arm].steps
        if d == "gold":
            seqs: list[Item] = [self.gold_seq(i) for i in self.train_idx[: steps * CFG.batch]]
            return seqs, CFG.batch, own, {"n": len(self.train_idx)}
        if d == "a2":
            seqs2 = [self.fact_seq(q) for q in qwen_facts.train_set(seed)]
            one = pack_whole(seqs2, CFG.seq_len)  # one pass: the mean packed row length
            row_len = sum(len(r[0]) for r in one) / len(one)
            per = round(CFG.batch * self.get("control")["gold_mean_tokens"] / row_len)
            stream: list[Item] = list(packed_stream(seqs2, seed, CFG.seq_len, steps * per))
            return stream, per, own, {"n": len(seqs2), "rows_per_pass": len(one), "row_len": row_len,
                                      "rows_per_step": per}
        recs = [r for r in self.samples(SAMPLED[d]) if r["pid"] < CFG.n_problems - CFG.n_held]
        if d == "a1":
            recs = [r for r in recs if seed * CFG.own_k <= r["k"] < (seed + 1) * CFG.own_k]
        kept = keep_correct(recs) if d == "own" else keep_finished(recs)
        order = cycle(len(kept), seed, steps * CFG.batch)
        items: list[Item] = [self.sample_seq(kept[i]) for i in order]
        return items, CFG.batch, own, {"n": len(kept), "of": len(recs), "passes": len(order) / max(1, len(kept))}

    def own_rows(self, arm: str, seed: int) -> list[list[int]] | None:
        """The arm's own rows, 64 packed rows of 512 (None for own samples and the null: the maths rows)."""
        d = ARMS[arm].data
        if d == "gold":
            seqs = [self.gold_seq(i)[0] for i in self.held_idx]
        elif d == "a2":
            seqs = [self.fact_seq(q)[0] for q in qwen_facts.held_set(seed)]
        elif d == "a1":  # the prompted model's greedy answers to test 256-511, under the unprompted chat
            recs = json.loads((LARGE / "P" / "a1_rows.json").read_text())["items"]
            seqs = [self.ids(prompt(self.tok, self.sets["test"][r["idx"]]["question"])) + r["ids"] for r in recs]
        else:
            return None
        return pack(seqs, CFG.own_rows, CFG.seq_len)


# ----------------------------------------------------------------------------- generation


def records(path: Path) -> Records:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def per_batch(job: Job, texts: list[str], n: int, max_new: int = MAX_NEW) -> int:
    """Prompts per generation batch: the cached tokens of n outputs per prompt within CFG.kv_tokens."""
    longest = max(len(job.ids(t)) for t in texts)
    return max(1, CFG.kv_tokens // ((longest + max_new) * n))


def greedy(job: Job, gen: Any, texts: list[str], items: Records, max_new: int = MAX_NEW) -> Records:
    """Greedy answers as ids and text, graded strict against each item's gold."""
    per, out = per_batch(job, texts, 1, max_new), []
    for s in range(0, len(texts), per):
        for (ids, cut), it in zip(sample(job.tok, gen, texts[s : s + per], 1, GREEDY, max_new), items[s : s + per],
                                  strict=True):
            text = job.tok.decode(ids, skip_special_tokens=True)
            out.append({"idx": it["idx"], "ids": ids, "truncated": cut, "response": text,
                        "strict": bool(grade(text, gold_number(it["gold"]))["strict"])})
    return out


def facts_eval(job: Job, gen: Any, qas: list[qwen_facts.QA]) -> dict[str, float]:
    """Greedy exact match on invented and on known facts."""
    texts = [chat(job.tok, q.question) for q in qas]
    outs = []
    for s in range(0, len(texts), CFG.facts_batch):
        outs += sample(job.tok, gen, texts[s : s + CFG.facts_batch], 1, GREEDY, CFG.facts_new)
    right = [(q.fact.known, qwen_facts.correct(job.tok.decode(ids, skip_special_tokens=True), q.fact.answer))
             for q, (ids, _) in zip(qas, outs, strict=True)]
    return {k: statistics.mean(r for kn, r in right if kn == (k == "known")) for k in ("invented", "known")}


def accuracy(job: Job, W: Any, items: Records, sink: Path, label: str) -> tuple[dict[str, Any], Records]:
    """`qwen_eval.score` (greedy, batches of 128, as the baseline) of the bf16 copy holding W; summary and records."""
    sink.parent.mkdir(parents=True, exist_ok=True)
    recs = gsm_score(job.tok, job.gen(W), items, math.inf, sink, label, CFG.gen_batch)
    return summary(recs) | {"response_mean": statistics.mean(r["response_tokens"] for r in recs)}, recs
