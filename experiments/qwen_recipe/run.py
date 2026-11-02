"""E61 (qwen_recipe), stage A, 2026-11-02. Plan: reports/reviews/2026-11-02_review_qwen-finetune-recipes.md, section 5.

Does bf16 autocast hide E45's early updates from training? Saved states W0 + δW are read as `fp32` (E45's held-out
reader), `autocast` (`sft_step`'s forward) and `rounded` (weights cast to bf16 and back, fp32 forward). Rules:
results/predictions.json. Order: W0, own_s0 state 4 (a miss exits 1, results/stopped.json); the other states; R1, R2; a
gradient pass at micro 8 and 4; fp32 generation. Exit 3: budget reached, resubmit (A.json skips done work). E45's job.py
and run.py load by path: this file is also `run`, so `import run` would find itself.
"""

import argparse
import gc
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig

from commute import qwen_eval, qwen_sft
from commute.qwen_anchors import follows_format
from commute.qwen_updates import at_least, at_most, ok

HERE = Path(__file__).resolve().parent
RESULTS, E45 = HERE / "results", HERE.parent / "qwen_updates"
PRED, OUT, LOG, STOPPED = (RESULTS / n for n in ("predictions.json", "A.json", "run.log", "stopped.json"))
READERS = ("fp32", "autocast", "rounded")
STATES = {"own_s0": (0, 2, 3, 4, 100), "gold_s0": (0, 3, 5, 100), "a1_s0": (0, 4, 8, 100)}
GEN = (("a1_s0", 8, 63.0), ("a1_s0", 19, 46.0), ("gold_s0", 100, 290.0))  # seconds in E45's bf16 runs (its run.log)
GEN_LIMIT_S, GEN_CAP, GIB = 480.0, 64, 2**30  # all three in 8 min at E45's size, projected from the first; else 64

def sibling(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    sys.modules[name] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

JOB: Any = sibling("job", E45 / "job.py")
E45_RUN: Any = sibling("e45_run", E45 / "run.py")
DEV, F32, BF16, LARGE, CFG = JOB.DEV, JOB.F32, JOB.BF16, JOB.LARGE, JOB.CFG

class Lab(JOB.Job):  # E45's data and model, only reading its results (`get`, `samples`); the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with LOG.open("a") as f:
            f.write(line + "\n")

def save(A: dict[str, Any]) -> None:
    (tmp := OUT.with_name("A.json.tmp")).write_text(json.dumps(A, indent=1, default=float))
    tmp.replace(OUT)

def expect(lab: Lab, name: str, a: float, b: float, tol: float, detail: dict[str, Any]) -> None:
    if not (ok(a) and ok(b) and abs(a - b) <= tol * abs(b)):
        STOPPED.write_text(json.dumps({"stopped": name, "a": a, "b": b, "tol": tol, **detail}, indent=1, default=float))
        lab.log(f"STOP: {name}: {a} against {b}, tol {tol} {json.dumps(detail, default=float)}")
        sys.exit(1)

def in_reader(fn: Any, W: Any, reader: str) -> Any:
    with torch.autocast("cuda", dtype=BF16, enabled=reader == "autocast"):
        return fn({s: w.to(BF16).to(F32) for s, w in W.items()} if reader == "rounded" else W)

def state_weights(lab: Lab, run: str, t: int) -> Any:
    return lab.W0 if t == 0 else qwen_sft.plus(lab.W0, qwen_sft.load(LARGE / "dW" / run / f"step_{t}.safetensors", DEV))

def read_state(lab: Lab, A: dict[str, Any], run: str, t: int, readers: tuple[str, ...] = READERS) -> dict[str, Any]:
    """A run's held-out quantity at state t under `readers`; with all three, the 16-row maths KL from W0 as well."""
    st = A["states"].setdefault(run, {}).setdefault(str(t), {})
    todo, kl_due = [r for r in readers if r not in st], t > 0 and readers == READERS and "kl_maths16" not in st
    if not todo and not kl_due:
        return st
    lab.budget(60)
    W, key = state_weights(lab, run, t), "distill_kl" if run.startswith("a1") else "loss"
    rec = lab.get(f"E_{run}")["states"][str(t)]["task"][key]
    for r in todo:
        st[r] = in_reader(lab.tasks[run], W, r)[key]
        lab.log(f"[{run} {t}] {key} {r} {st[r]:.6f}; E45's fp32 record {rec:.6f}")
    if kl_due:
        rows = lab.maths()[: CFG.track_rows]
        st["kl_maths16"] = {r: in_reader(lambda w: qwen_sft.mean_kl(lab.model, rows, lab.W0, w), W, r)
                            for r in READERS[:2]}
        lab.log(f"[{run} {t}] 16-row maths KL from W0 {json.dumps(st['kl_maths16'])}")
    save(A)
    return st

def controls(lab: Lab, A: dict[str, Any], E: dict[str, Any]) -> None:
    """Step 1: W0 on own's rows, three readers. Step 2: own_s0 state 4 in fp32 against its record."""
    rec = lab.get("E_own_s0")["states"]
    A["w0_entries_changed"] = n = sum(int((w.to(BF16).to(F32) != w).sum()) for w in lab.W0.values())
    lab.log(f"[W0] entries changed by bf16 rounding, {len(lab.W0)} matrices: {n} (expected 0)")
    st = read_state(lab, A, "own_s0", 0)
    lab.log(f"[W0] noise floor, fp32 minus autocast, own rows: {st['fp32'] - st['autocast']:.6f}")
    expect(lab, "W0: rounded equals fp32", st["rounded"], st["fp32"], E["w0_rounded_rel"], {"changed": n})
    expect(lab, "W0: fp32 reproduces E45", st["fp32"], rec["0"]["task"]["loss"], E["e45_rel"], {})
    st = read_state(lab, A, "own_s0", 4, ("fp32",))
    expect(lab, "own_s0 state 4: fp32 reproduces E45", st["fp32"], rec["4"]["task"]["loss"], E["e45_rel"], {})

def three_way(v: float, R: dict[str, Any]) -> str:
    if not ok(v):
        return "no reading"
    return "confirmed" if at_most(v, R["confirmed_max"]) else "ruled out" if at_least(v, R["ruled_out_min"]) else "partial"

def score_rules(lab: Lab, A: dict[str, Any], R: dict[str, Any]) -> None:
    """R1 and R2 (predictions.json); R3 has no threshold and is A["gen"]. a1's gap closed per reader, no rule reads it."""
    S, own4 = A["states"], A["states"]["own_s0"]["4"]
    out: dict[str, Any] = {"R1": {r: {"value": own4[r], "reading": three_way(own4[r], R["R1"])} for r in READERS[1:]},
                           "R2": {}}
    for run in STATES:
        a, z = S[run]["100"], S[run]["0"]
        gap, floor = a["autocast"] - a["fp32"], z["autocast"] - z["fp32"]
        excess = abs(gap) - abs(floor)
        out["R2"][run] = {"gap_100": gap, "floor_w0": floor, "excess": excess,
                          "holds": at_most(excess, R["R2"]["max_excess"])}
    a1 = S["a1_s0"]
    out["a1_gap"] = {str(t): {r: 1 - a1[str(t)][r] / a1["0"][r] for r in READERS} for t in STATES["a1_s0"]}
    A["rules"] = out
    save(A)
    lab.log(f"[rules] {json.dumps(out)}")

def grad_pass(lab: Lab, seqs: list[Any], micro: int) -> dict[str, Any]:
    """One gradient pass, no optimiser step: `sft_step` without its autocast."""
    m, n = lab.model, qwen_sft.n_targets(seqs)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0, res = time.time(), {"micro": micro, "oom": False}
    try:
        with qwen_sft.trainable(m, lab.W0):
            m.train()
            for i in range(0, len(seqs), micro):
                (qwen_sft.nll_sum(m, seqs[i : i + micro], DEV) / n).backward()
    except torch.cuda.OutOfMemoryError as e:
        res |= {"oom": True, "error": str(e)[:200]}
    m.eval()
    return res | {"seconds": time.time() - t0, "peak_alloc_gib": torch.cuda.max_memory_allocated() / GIB,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}

def measure_grad(lab: Lab, A: dict[str, Any]) -> None:
    stream, per, *_ = lab.arm_data("own", 0)
    G = A.setdefault("grad", {})
    G["adam_moments_gib"] = 8 * sum(w.numel() for w in lab.W0.values()) / GIB  # not allocated here; on top in training
    for micro in ("8", "4"):
        if micro not in G:
            G[micro] = grad_pass(lab, stream[:per], int(micro))
            save(A)
            lab.log(f"[grad] {json.dumps(G[micro])}; Adam's moments would add {G['adam_moments_gib']:.1f} GiB")

def gen_reading(recs: list[Any], a1: bool) -> dict[str, Any]:
    out = qwen_eval.summary(recs) | {"response_mean": statistics.mean(r["response_tokens"] for r in recs)}
    return out | ({"format_unprompted": statistics.mean(follows_format(r["response"]) for r in recs)} if a1 else {})

def generate_states(lab: Lab, A: dict[str, Any]) -> None:
    """Greedy fp32 answers at E45's settings beside E45's bf16 answers to the same problems. KV cache about 19 GB at
    batch 128 (twice `job.Config.kv_tokens`' 114 kB per token), about 31 GB with the model and W."""
    m, tests, eos = lab.model, lab.sets["test"], lab.model.generation_config.eos_token_id
    m.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=lab.tok.pad_token_id)  # as `load_model`
    for run, t, bf16_s in GEN:
        key, a1 = f"{run}_{t}", run.startswith("a1")
        if key in A["gen"]:
            continue
        first = A["gen"].get(f"{GEN[0][0]}_{GEN[0][1]}")
        capped = first is not None and first["seconds"] / GEN[0][2] * sum(g[2] for g in GEN) > GEN_LIMIT_S
        items = tests[:GEN_CAP] if capped else tests if t == CFG.steps and not a1 else tests[: CFG.eval_problems]
        lab.budget(2 * bf16_s)
        t0, sink = time.time(), Path(tempfile.gettempdir()) / f"e61_{key}.jsonl"
        with qwen_sft.set_weights(m, state_weights(lab, run, t)):
            recs = qwen_eval.score(lab.tok, m, items, float("inf"), sink, key, CFG.gen_batch)
        e45 = JOB.records(LARGE / "E" / run / f"step_{t}_test.jsonl")[: len(items)]
        A["gen"][key] = {"n": len(items), "capped": capped, "seconds": time.time() - t0,
                         "fp32": gen_reading(recs, a1), "e45_bf16_same_problems": gen_reading(e45, a1)}
        save(A)
        lab.log(f"[gen {key}] {json.dumps(A['gen'][key])}" + (f"; capped at the first {GEN_CAP}" if capped else ""))

def main() -> None:
    p = argparse.ArgumentParser(description="E61, the three-reader check; plan in the module docstring")
    p.add_argument("--stage", required=True, choices=("A",))
    p.parse_args()
    lab = Lab("A", "", PRED)  # refuses to start without the predictions file
    A: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"states": {}, "gen": {}}
    STOPPED.unlink(missing_ok=True)
    lab.fp32()
    lab.tasks = {run: E45_RUN.task_reader(lab, run[:-3], int(run[-1])) for run in STATES}
    controls(lab, A, lab.pred["early"])
    for run, t in ((r, t) for r, ts in STATES.items() for t in ts):
        read_state(lab, A, run, t)
    score_rules(lab, A, lab.pred["rules"])
    measure_grad(lab, A)
    generate_states(lab, A)
    save(A | {"done": True})
    lab.log(f"[A] done; {lab.peaks()}")

if __name__ == "__main__":
    main()
