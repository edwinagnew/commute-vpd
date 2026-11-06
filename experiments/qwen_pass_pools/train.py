"""E120 (qwen_pass_pools) stage 3, 2026-11-06: SFT of Qwen3-1.7B on one pass-rate pool's gold solutions. Plan:
reports/plans/2026-11-06_plan_e120-pass-rate-pools.md, stage 3; sources: reports/reviews/2026-11-06_review_fable-e120-
design.md, questions 5-6. The run refuses to start without results/predictions.json.

One run = one arm: pool known|new, seed 0|1 (the seed changes the order of the pool's training part only), from
`--pools` (default results/pools.json; Amendment 2, 2026-11-06: results/pools_amend2.json). The recipe
is E61's `gold_1e-5` unchanged (`RECIPE`, asserted equal to experiments/qwen_recipe/results/train_gold_1e-5.json →
recipe): peak 1e-5, warm-up 10, cosine over 100 steps, clip 1, 64 sequences a step, micro 4, fp32 forward, the 196
decomposed matrices trainable (`qwen_train.train`). A sequence is the chat (thinking off) around the item's `user`
message, then `gold_response` and the end token (`qwen_sft.seq_of`); the loss is over the response tokens.

In one job, each part skipped when its file is there:
  W0    held-out gold loss of both pools at W0 (results/read_<arm>.json → w0)
  train 100 steps; fp32 δW saved at the last step under and the first step at or over |δW| 0.35 and 0.49 (read every
        step), at step 10 and step 100 (results/large/dW/<arm>/step_<t>.safetensors); the record per step in
        results/train_<arm>.json and the log. Early checks below stop the job (exit 1, results/stopped_<arm>.json).
  files each saved file's fp64 norm against the record (read_<arm>.json → files)
  read  at step 100, from its file: held-out gold loss of both pools; GSM8K test 0-255 strict greedy (`qwen_eval.score`,
        beside W0's 0.784, E61 train_W.json → w0.unprompted.strict_dropped); 8 samples (`qwen_eval.SAMPLE`, cap 1,024) per
        held-out problem of both pools; one greedy answer (cap 1,024) per training problem of the arm's pool (the plan's
        "trained-part greedy"). Answers go to results/large/samples/<arm>/<set>.json as
        [{id, answer_gold, source, user, responses, truncated}], graded on the CPU later by grade.py.
Exit 0 done; 1 a stop or a refusal; 3 paused (--t-chunk, or the budget reached before a unit): resubmit the same
command. A job lost in training with no checkpoint starts training again. Every file written prints
`STAGE 3 DONE <path>`.

Time per arm on an L40. Training: E61's gold arm took 1,732 s (train_gold_1e-5.json → seconds) with 24 held-out reads
this run does not make; its plain steps took ~11 s on GSM8K gold (run.log, steps 21-29), so 100 steps ~19 min at that
length and more for longer MATH solutions, ~20-30 min, plus ~6 fp32 saves at ~30 s each (E78's run.log). Files check
~3 min. Reads: GSM8K ~3 min (E61's fp32 W0 reading, 165 s); answers at cap 1,024 in batches of 64 sequences, about
1-2 min a batch (unmeasured; each batch's time is logged): 2 x n_held / 8 + n_train / 64 batches.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import shutil
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, GenerationConfig

from commute import qwen_eval, qwen_sets
from commute.jobs import hard_exit
from commute.qwen_eval import GREEDY, MODEL, MODEL_REV, SAMPLE, chat_prompt
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import (
    Seq,
    gold_loss,
    linears,
    load,
    plus,
    save,
    seq_of,
    set_weights,
    sha256,
)
from commute.qwen_train import Recipe, train
from commute.qwen_updates import cycle, ok
from commute.qwen_vpd import SITES, fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, POOLS, PRED = RESULTS / "large", RESULTS / "pools.json", RESULTS / "predictions.json"
E61_GOLD = HERE.parent / "qwen_recipe" / "results" / "train_gold_1e-5.json"
DEV, F32, GIB = "cuda", torch.float32, 2**30

RECIPE = Recipe(1e-5, 10, 100, "hf_cosine", 1.0, 4, forward="fp32")
BATCH, SEQ_LEN = 64, 512  # sequences a step (E45 Config.batch); window (E45 Config.seq_len; stage 2 drops longer)
NORMS, KEEP = (0.35, 0.49), {10, 100}  # |δW| of the RL arm and of anchor 1; saved always
# early checks, plan stage 3 (bands from E61's gold arm and E69, review question 6)
STEP2_RATIO, PEAK_GIB_MAX, FILE_REL = (0.85, 1.05), 40.0, 1e-6
W0_GSM8K, GSM8K_PROBLEMS, GEN_BATCH = 0.7842323651452282, 256, 128
N_SAMPLES, ANSWER_CAP, GEN_SEQS = 8, 1024, 64  # samples per problem, new tokens, sequences per generate call
POOL_NAMES = ("known", "new")
TRAIN_EST_S, GSM8K_EST_S, BATCH_EST_S = 1800.0, 300.0, 150.0


@dataclass(frozen=True)
class Arm:
    pool: str
    seed: int

    def __post_init__(self) -> None:
        assert self.pool in POOL_NAMES and self.seed in (0, 1), self

    @property
    def name(self) -> str:
        return f"{self.pool}_s{self.seed}"


# ----------------------------------------------------------------------------- data


def sequences(tok: Any, end_id: int, items: list[dict[str, Any]]) -> list[Seq]:
    """Each item as one training sequence; none may be cut (stage 2 drops prompt + gold over SEQ_LEN)."""
    out = [seq_of(tok, chat_prompt(tok, it["user"]), it["gold_response"], end_id, SEQ_LEN) for it in items]
    cut = [it["id"] for it, (ids, _) in zip(items, out, strict=True) if ids[-1] != end_id]
    assert not cut, f"cut at {SEQ_LEN} tokens: {cut[:5]}"
    return out


def stream(train_seqs: list[Seq], seed: int) -> list[Seq]:
    """RECIPE.steps x BATCH sequences: the training part cycled in a seeded order (`qwen_updates.cycle`)."""
    return [train_seqs[i] for i in cycle(len(train_seqs), seed, RECIPE.steps * BATCH)]


def data_summary(path: Path, seqs: dict[str, dict[str, list[Seq]]], pool: str) -> dict[str, Any]:
    n = len(seqs[pool]["train"])
    lens = [len(ids) for ids, _ in seqs[pool]["train"]]
    return {"pools_sha256": sha256(path), "n_train": n, "n_held": {p: len(seqs[p]["held"]) for p in POOL_NAMES},
            "passes": RECIPE.steps * BATCH / n, "train_len_mean": statistics.mean(lens), "train_len_max": max(lens)}


# ----------------------------------------------------------------------------- the job


class Lab:
    def __init__(self, arm: Arm, budget_s: float, pools: Path) -> None:
        self.arm, self.budget_s, self.t0, self.slowest = arm, budget_s, time.time(), 0.0
        self.train_file, self.read_file = RESULTS / f"train_{arm.name}.json", RESULTS / f"read_{arm.name}.json"
        self.dw, self.samples = LARGE / "dW" / arm.name, LARGE / "samples" / arm.name
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.tok.padding_side = "left"
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.pools_path, self.pools = pools, json.loads(pools.read_text())
        self.seqs = {p: {part: sequences(self.tok, self.end_id, self.pools[p][part]) for part in ("train", "held")}
                     for p in POOL_NAMES}
        self.model: Any = None
        self.W0: dict[str, Any] = {}
        self.log(f"[{arm.name}] start; {torch.cuda.get_device_name()}; budget {budget_s:.0f} s")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / f"sft_{self.arm.name}.log").open("a") as f:
            f.write(line + "\n")

    def fp32(self) -> Any:
        """The fp32 model with gradient checkpointing (as E45's `Job.fp32`); W0 its decomposed matrices."""
        if self.model is None:
            self.model = load_target(DEV, F32)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.generation_config = GenerationConfig(eos_token_id=self.model.generation_config.eos_token_id,
                                                            pad_token_id=self.tok.pad_token_id)
            self.W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
        return self.model

    def budget(self, est_s: float) -> None:
        """Exit 3 when a unit of about est_s (or the slowest so far) would pass the budget."""
        used = time.time() - self.t0
        if used + max(est_s, self.slowest) > self.budget_s:
            self.log(f"BUDGET: {used:.0f} s used; resubmit the same command to continue")
            sys.exit(3)

    def done(self, path: Path) -> None:
        self.log(f"STAGE 3 DONE {path}")

    def stop(self, name: str, detail: dict[str, Any]) -> NoReturn:
        (RESULTS / f"stopped_{self.arm.name}.json").write_text(json.dumps({"stopped": name, **detail}, indent=1))
        self.log(f"STOP: {name} {json.dumps(detail, default=float)}")
        sys.exit(1)

    def peaks(self) -> dict[str, float]:
        return {"gpu_gib": torch.cuda.max_memory_allocated() / GIB,
                "host_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path: Path, obj: Any) -> None:
    (tmp := path.with_name(path.name + ".tmp")).write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def held_losses(lab: Lab) -> dict[str, float]:
    """Mean response-token gold loss, fp32, on each pool's held-out part, at the weights as they stand."""
    return {p: gold_loss(lab.model, lab.seqs[p]["held"], 8, DEV) for p in POOL_NAMES}


# ----------------------------------------------------------------------------- training


def check(lab: Lab, out: dict[str, Any], t: int, name: str, value: Any, holds: bool) -> None:
    row = {"step": t, "check": name, "value": value, "holds": holds}
    out["checks"].append(row)
    lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
    if not holds:
        write(lab.train_file, out)
        lab.stop(f"step {t} {name}", row)


def early_checks(lab: Lab, out: dict[str, Any], rec: dict[str, Any], numel: int) -> None:
    """The plan's stage-3 checks at this step, each logged."""
    t = rec["step"]
    check(lab, out, t, "loss and grad norm finite", [rec["loss"], rec["grad_norm"]], ok(rec["loss"]) and ok(rec["grad_norm"]))
    check(lab, out, t, f"peak GiB <= {PEAK_GIB_MAX}", rec["peak_gib"], rec["peak_gib"] <= PEAK_GIB_MAX)
    if t == 1:
        check(lab, out, t, "|dW| exactly 0 (rate 0)", rec["dw_norm"], rec["dw_norm"] == 0)
    if t == 2:
        ratio = rec["dw_norm"] / (rec["lr"] * math.sqrt(numel))
        check(lab, out, t, f"|dW| / (lr sqrt(numel)) in {list(STEP2_RATIO)}", ratio,
              STEP2_RATIO[0] <= ratio <= STEP2_RATIO[1])
    if t == 10:
        first = out["steps"][0]["loss"]
        check(lab, out, t, "loss below step 1's", [rec["loss"], first], rec["loss"] < first)


def start_or_resume(lab: Lab, data: dict[str, Any]) -> dict[str, Any]:
    """A fresh record, or the record cut back to the checkpoint (its |δW| asserted against the record's)."""
    ckpt = lab.dw / "ckpt.pt"
    if not ckpt.is_file():
        shutil.rmtree(lab.dw, ignore_errors=True)
        return {"arm": asdict(lab.arm), "recipe": asdict(RECIPE), "data": data, "steps": [], "saves": [],
                "checks": [], "seconds": 0.0, "resumes": []}
    c = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    t, out = c["step"], read(lab.train_file)
    if out["data"] != data:
        lab.stop("the pools changed since the checkpoint", {"then": out["data"], "now": data})
    out |= {"steps": out["steps"][:t], "saves": [s for s in out["saves"] if s <= t],
            "checks": [c for c in out["checks"] if c["step"] <= t]}
    dw = math.sqrt(sum(v * v for v in c["norms"].values()))
    assert math.isclose(dw, out["steps"][-1]["dw_norm"], rel_tol=1e-12), (dw, out["steps"][-1])
    out["resumes"].append({"step": t, "dw_norm": dw})
    lab.log(f"[train] resuming after step {t}; |dW| {dw:.6f}")
    return out


def stage_train(lab: Lab, t_chunk: int | None) -> None:
    if read(lab.train_file).get("done"):
        return
    m, arm = lab.fp32(), lab.arm
    rd = read(lab.read_file)
    if "w0" not in rd:
        rd["w0"] = {"held_loss": held_losses(lab)}
        write(lab.read_file, rd)
        lab.log(f"[W0] held-out gold loss {json.dumps(rd['w0'])}")
        lab.done(lab.read_file)
    out = start_or_resume(lab, data_summary(lab.pools_path, lab.seqs, arm.pool))
    if not out["steps"]:
        lab.budget(TRAIN_EST_S if t_chunk is None else TRAIN_EST_S * t_chunk / RECIPE.steps)
    lab.log(f"[train] data {json.dumps(out['data'])}")
    data, numel = stream(lab.seqs[arm.pool]["train"], arm.seed), sum(w.numel() for w in lab.W0.values())
    w0_held = rd["w0"]["held_loss"][arm.pool]

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        path = lab.dw / f"step_{t}.safetensors"
        save(path, dW, F32, droppable=False, log=lab.log)
        out["saves"] = sorted(set(out["saves"]) | {t})
        lab.done(path)

    def on_step(rec: dict[str, Any]) -> None:
        rec.pop("norms")
        if not rec.get("crossed"):
            rec.pop("crossed", None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
        out["steps"].append(rec)
        write(lab.train_file, out)
        lab.log(f"[train] {json.dumps(rec)}")
        if rec["step"] == 1:  # a note, no stop: training part against held-out part, same state
            lab.log(f"[note] step 1 loss {rec['loss']:.4f}; W0 held-out loss of the pool {w0_held:.4f}")
        early_checks(lab, out, rec, numel)

    t0, steps = time.time(), RECIPE.steps
    t = train(m, lab.W0, lambda t: data[(t - 1) * BATCH : t * BATCH], RECIPE, {"norm": []}, set(range(1, steps + 1)),
              {"norm": NORMS}, KEEP, save_dw, on_step, lab.dw / "ckpt.pt", t_chunk or steps, dW_of=delta_fp32,
              size_of=lambda rec: {"norm": rec["dw_norm"]})
    out |= {"done": t == steps, "seconds": out["seconds"] + time.time() - t0, "peaks": lab.peaks()}
    write(lab.train_file, out)
    if t < steps:
        lab.log(f"BUDGET: checkpoint after step {t}; resubmit the same command to continue")
        sys.exit(3)
    lab.done(lab.train_file)
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- saved files


def file_norm(path: Path) -> tuple[float, int, set[str]]:
    """fp64 norm over every tensor of a safetensors file, one tensor at a time; tensor count; dtypes."""
    total, dtypes = 0.0, set()
    with safe_open(str(path), "pt") as f:
        names = list(f.keys())
        for k in names:
            x = f.get_tensor(k)
            dtypes.add(str(x.dtype))
            total += float(x.double().square().sum())
    return math.sqrt(total), len(names), dtypes


def stage_files(lab: Lab) -> None:
    rd, out = read(lab.read_file), read(lab.train_file)
    if "files" in rd:
        return
    rows = {}
    for t in out["saves"]:
        norm, n, dtypes = file_norm(lab.dw / f"step_{t}.safetensors")
        record = out["steps"][t - 1]["dw_norm"]
        holds = n == len(SITES) and dtypes == {str(F32)} and abs(norm - record) <= FILE_REL * record
        rows[str(t)] = {"norm": norm, "record": record, "tensors": n, "dtypes": sorted(dtypes), "holds": holds}
        lab.log(f"[check] {'ok' if holds else 'MISSED'} saved file step {t} {json.dumps(rows[str(t)])}")
    rd["files"] = rows
    write(lab.read_file, rd)
    lab.log(f"[files] saved steps {out['saves']}; brackets {[c for s in out['steps'] for c in s.get('crossed', [])]}")
    if not all(r["holds"] for r in rows.values()):
        lab.stop("a saved file is not its recorded state", rows)


# ----------------------------------------------------------------------------- step-100 reads


def answer_set(lab: Lab, key: str, items: list[dict[str, Any]], n_ret: int, kw: dict[str, Any]) -> None:
    """n_ret answers per item at cap ANSWER_CAP, each batch to its own file (a resumed job goes on from the last whole
    batch), then the whole set to <key>.json. Each batch's sampling seed is fixed by the arm, the set and the batch."""
    final, parts = lab.samples / f"{key}.json", lab.samples / key
    if final.is_file():
        return
    parts.mkdir(parents=True, exist_ok=True)
    texts, per = [chat_prompt(lab.tok, it["user"]) for it in items], max(1, GEN_SEQS // n_ret)
    for i, s in enumerate(range(0, len(items), per)):
        if (f := parts / f"batch_{i:04d}.json").is_file():
            continue
        lab.budget(BATCH_EST_S)
        t0 = time.time()
        torch.manual_seed(zlib.crc32(f"{lab.arm.name}-{key}-{i}".encode()))
        recs = qwen_eval.generate(lab.tok, lab.model, texts[s : s + per], n_ret, kw, max_new=ANSWER_CAP)
        rows = [{"id": it["id"], "answer_gold": it["answer"], "source": it["source"], "user": it["user"],
                 "responses": [r["response"] for r in recs[j * n_ret : (j + 1) * n_ret]],
                 "truncated": [r["truncated"] for r in recs[j * n_ret : (j + 1) * n_ret]]}
                for j, it in enumerate(items[s : s + per])]
        write(f, rows)
        lab.slowest = max(lab.slowest, dt := time.time() - t0)
        lab.log(f"[{key}] batch {i + 1}/{math.ceil(len(items) / per)}: {dt:.0f} s, "
                f"{sum(r['response_tokens'] for r in recs) / dt:.0f} new tokens/s")
    rows = [r for f in sorted(parts.glob("batch_*.json")) for r in json.loads(f.read_text())]
    assert [r["id"] for r in rows] == [it["id"] for it in items], key
    write(final, rows)
    shutil.rmtree(parts)
    trunc = statistics.mean(x for r in rows for x in r["truncated"])
    lab.log(f"[{key}] {len(rows)} problems x {n_ret}; truncated share {trunc:.3f}")
    lab.done(final)


def stage_read(lab: Lab) -> None:
    rd = read(lab.read_file)
    if rd.get("done"):
        return
    m, pool = lab.fp32(), lab.arm.pool
    with set_weights(m, plus(lab.W0, load(lab.dw / "step_100.safetensors", DEV))):
        if "step_100" not in rd:
            rd["step_100"] = {"held_loss": held_losses(lab)}
            write(lab.read_file, rd)
            lab.log(f"[read] step 100 held-out gold loss {json.dumps(rd['step_100'])}; W0 {json.dumps(rd['w0'])}")
        if "gsm8k" not in rd["step_100"]:
            lab.budget(GSM8K_EST_S)
            sink = LARGE / "answers" / lab.arm.name / "gsm8k_test_0-255.jsonl"
            sink.parent.mkdir(parents=True, exist_ok=True)
            tests = qwen_sets.load()["test"][:GSM8K_PROBLEMS]
            recs = qwen_eval.score(lab.tok, m, tests, math.inf, sink, "gsm8k", GEN_BATCH)
            rd["step_100"]["gsm8k"] = qwen_eval.summary(recs) | {"w0_strict_dropped": W0_GSM8K}
            write(lab.read_file, rd)
            lab.log(f"[read] GSM8K test 0-255 {json.dumps(rd['step_100']['gsm8k'])}")
        for p in (pool, *(q for q in POOL_NAMES if q != pool)):
            answer_set(lab, f"{p}_held_samples", lab.pools[p]["held"], N_SAMPLES, SAMPLE)
        answer_set(lab, f"{pool}_train_greedy", lab.pools[pool]["train"], 1, GREEDY)
    rd |= {"done": True, "samples": sorted(str(f.relative_to(RESULTS)) for f in lab.samples.glob("*.json")),
           "peaks": lab.peaks()}
    write(lab.read_file, rd)
    lab.done(lab.read_file)


def main() -> None:
    p = argparse.ArgumentParser(description="E120 stage 3: SFT on one pool; parts in the module docstring")
    p.add_argument("--pool", required=True, choices=POOL_NAMES)
    p.add_argument("--seed", required=True, type=int, choices=(0, 1))
    p.add_argument("--t-chunk", type=int, help="write a checkpoint and exit 3 after this many steps (default: none)")
    p.add_argument("--budget-min", type=float, default=170.0, help="stop before a unit that would pass this (exit 3)")
    p.add_argument("--pools", type=Path, default=POOLS, help="the pools file (Amendment 2: results/pools_amend2.json)")
    a = p.parse_args()
    arm = Arm(a.pool, a.seed)
    if (s := RESULTS / f"stopped_{arm.name}.json").is_file():
        sys.exit(f"{s} exists: read it, then remove it")
    if not PRED.is_file():
        sys.exit(f"{PRED} is missing: predictions are filed before any run")
    if asdict(RECIPE) != json.loads(E61_GOLD.read_text())["recipe"]:
        sys.exit(f"RECIPE is not E61's gold_1e-5 ({E61_GOLD})")
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    assert a.t_chunk is None or 0 < a.t_chunk < RECIPE.steps, a.t_chunk
    lab = Lab(arm, a.budget_min * 60, a.pools)
    try:
        stage_train(lab, a.t_chunk)
        stage_files(lab)
        stage_read(lab)
    except torch.cuda.OutOfMemoryError as e:
        lab.stop("out of GPU memory", {"error": str(e)[:300]})
    lab.log(f"[{arm.name}] done; {json.dumps(lab.peaks())}")
    hard_exit(0)


if __name__ == "__main__":
    main()
