"""E78 (qwen_anchor1), 2026-11-03: anchor 1 (the answer-format template, context distillation) retrained on Qwen3-1.7B at
E61's rate, changing only what is saved: δW in fp32 at steps 1, 2, 3, 5, 10, 20, 50, 100. Plan:
reports/plans/2026-11-03_plan_e78-anchor1.md. Rules: results/predictions.json (the run refuses to start without it).

The recipe, data and readers are those of E61's run `a1_1e-6` (experiments/qwen_recipe/train.py), a finished experiment
that writes into its own folder, so the few lines of its Stages T and G that apply are here, without its other arms.
Step t's δW has t - 1 moving steps: step 1 runs at rate 0, so its file is all zeros.

  T  training (about 27 min). Stops on the first-minutes bands E61 filed for this run.
  G  needs T: the saved files' checks on the CPU, then generation and reads of step 100 and the rule (about 10 min).
Exit 0 done; 1 a stop (results/stopped.json, every stage refuses until it is removed) or a refusal. A lost T starts
again: it fits one debug job, so it writes no checkpoint. Results: results/train.json as it goes, results/read.json, each
number also in results/run.log; δW in results/large/dW/a1/step_<t>.safetensors.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import GenerationConfig

from commute import qwen_eval
from commute.jobs import hard_exit, load_experiment
from commute.qwen_anchors import follows_format
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import load as load_file
from commute.qwen_sft import mean_kl, plus, save, set_weights
from commute.qwen_train import Recipe, train
from commute.qwen_updates import at_least, at_most, ok

JOB: Any = load_experiment("qwen_updates", "job")
E45_RUN: Any = load_experiment("qwen_updates", "run")
DEV, F32, CFG = JOB.DEV, JOB.F32, JOB.CFG

HERE = Path(__file__).resolve().parent
RESULTS, E61 = HERE / "results", HERE.parent / "qwen_recipe" / "results"
PRED, TRAIN, READ, STOPPED = (RESULTS / n for n in ("predictions.json", "train.json", "read.json", "stopped.json"))
LARGE = RESULTS / "large"
DW = LARGE / "dW" / "a1"
E61_RUN, E61_DW = E61 / "train_a1_1e-6.json", E61 / "large" / "train" / "a1_1e-6" / "step_100.safetensors"
SAVES = (1, 2, 3, 5, 10, 20, 50, 100)
KL_AT = set(range(5, 101, 5)) | set(SAVES)  # E61's read steps and the saves
HELD_AT = set(range(10, 101, 10)) | set(SAVES)
MICRO, STEPS, PEAK = 4, 100, 1e-6


class Lab(JOB.Job):  # E45's data and model, only reading its results; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def write(path: Path, obj: dict[str, Any]) -> None:
    (tmp := path.with_name(path.name + ".tmp")).write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    STOPPED.write_text(json.dumps({"stopped": name, **detail}, indent=1, default=float))
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    sys.exit(1)


def in_band(x: float | None, band: list[float | None]) -> bool:
    lo, hi = band
    return ok(x) and (lo is None or at_least(x, lo)) and (hi is None or at_most(x, hi))


def check(lab: Lab, out: dict[str, Any], at: str, values: Mapping[str, Any]) -> None:
    """The bands E61 filed at this step: a "note" miss is logged, a "stop" miss writes `out` and exits 1."""
    for severity in ("note", "stop"):
        for key, band in lab.pred["first_minutes"][severity].get(at, {}).items():
            holds = in_band(values.get(key), band)
            row = {"at": at, "key": key, "value": values.get(key), "band": band, "severity": severity, "holds": holds}
            out["checks"].append(row)
            lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
            if severity == "stop" and not holds:
                write(TRAIN, out)
                stop(lab, f"step {at} {key}", row)


def stage_T(lab: Lab) -> None:
    if TRAIN.is_file() and json.loads(TRAIN.read_text()).get("done"):
        sys.exit(f"{TRAIN} is done")
    m, maths = lab.fp32(), lab.maths()[: CFG.track_rows]
    stream, per, _, data = lab.arm_data("a1", 0)
    r = Recipe(PEAK, CFG.warmup, STEPS, CFG.schedule, CFG.clip, MICRO, forward="fp32")
    e61 = json.loads(E61_RUN.read_text())
    if data != lab.pred["data"] or asdict(r) != e61["recipe"]:
        stop(lab, "not E61's rows or recipe", {"data": data, "recipe": asdict(r), "e61": e61["recipe"]})
    task = E45_RUN.task_reader(lab, "a1", 0)
    out: dict[str, Any] = {"data": data, "recipe": asdict(r), "steps": [], "saves": [], "checks": [], "seconds": 0.0}

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        save(DW / f"step_{t}.safetensors", dW, F32, droppable=False, log=lab.log)
        out["saves"].append(t)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        rec.pop("norms")
        rec.pop("crossed", None)  # the grid is empty: `train` saves only at `keep`
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        write(TRAIN, out)
        lab.log(f"[T] {json.dumps(rec)}")
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            stop(lab, f"step {t} not finite", rec)
        check(lab, out, str(t), rec)
        if t in HELD_AT:
            rec["held"] = task(None)["distill_kl"]
            write(TRAIN, out)
            lab.log(f"[T] step {t} held-out distillation KL {rec['held']:.6f} (E61's: {e61['steps'][t - 1].get('held')})")

    t0 = time.time()
    train(m, lab.W0, lambda t: stream[(t - 1) * per : t * per], r, {"maths": maths}, KL_AT, {"maths": ()}, set(SAVES),
          save_dw, on_step, LARGE / "ckpt_unused.pt", STEPS, dW_of=delta_fp32)
    write(TRAIN, out | {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- G: the files, on the CPU


def header(path: Path) -> dict[str, str]:
    """A safetensors file's tensor name -> dtype, as written in its header ("F32", "BF16"); no tensor is read."""
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        head = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    return {k: v["dtype"] for k, v in head.items() if k != "__metadata__"}


def moments(a: Path, b: Path) -> tuple[float, float, float]:
    """Squared norm of a, of b, and their inner product over all tensors, in fp64, one matrix at a time."""
    aa, bb, ab = 0.0, 0.0, 0.0
    with safe_open(str(a), "pt") as fa, safe_open(str(b), "pt") as fb:
        names = set(fa.keys())
        assert names == set(fb.keys()), (a, b)
        for k in names:
            x, y = fa.get_tensor(k).to(torch.float64), fb.get_tensor(k).to(torch.float64)
            aa, bb, ab = aa + float((x * x).sum()), bb + float((y * y).sum()), ab + float((x * y).sum())
    return aa, bb, ab


def file_row(dtypes: Mapping[str, str], norm: float | None, record: float, F: Mapping[str, Any]) -> dict[str, Any]:
    """A saved file against the filed form and the run's own |δW| at its step. A zero record (step 1) needs a zero file."""
    holds = (len(dtypes) == F["tensors"] and set(dtypes.values()) == {F["dtype"]} and ok(norm)
             and abs(norm - record) <= F["norm_rel"] * record)  # type: ignore[operator]
    return {"tensors": len(dtypes), "dtypes": sorted(set(dtypes.values())), "norm": norm, "record": record, "holds": holds}


def against_e61(aa: float, bb: float, ab: float, F: Mapping[str, Any]) -> dict[str, Any]:
    """Step 100 here (a, fp32) against E61's (b, bf16). A zero file has no cosine: it fails."""
    gap = 1 - ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else math.nan
    ratio = math.sqrt(bb / aa) if aa > 0 else math.nan
    return {"one_minus_cos": gap, "norm_ratio": ratio, "rounding_alone": at_most(gap, F["rounding_alone_max"]),
            "holds": at_most(gap, F["one_minus_cos_max"]) and in_band(ratio, F["norm_ratio"])}


def check_files(lab: Lab, read: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    F, rows = lab.pred["files"], {}
    for t in SAVES:
        p, record = DW / f"step_{t}.safetensors", steps[t - 1]["dw_norm"]
        dtypes = header(p)
        norm = math.sqrt(moments(p, p)[0]) if dtypes else None
        rows[str(t)] = file_row(dtypes, norm, record, F)
        lab.log(f"[files] step {t} {json.dumps(rows[str(t)])}")
    read["files"] = rows
    if not header(E61_DW):
        stop(lab, "E61's step 100 file is missing", {"path": str(E61_DW)})
    read["against_e61"] = against_e61(*moments(DW / "step_100.safetensors", E61_DW), F)
    lab.log(f"[files] step 100 against E61's bf16: {json.dumps(read['against_e61'])}")
    write(READ, read)
    if not all(r["holds"] for r in rows.values()):
        stop(lab, "a saved file misses", rows)
    if not read["against_e61"]["holds"]:
        stop(lab, "step 100 is not E61's", read["against_e61"])


# ----------------------------------------------------------------------------- G: step 100, on the GPU


def reproduction(last: Mapping[str, Any], R: Mapping[str, Any]) -> dict[str, Any]:
    """Step 100's record against E61's filed numbers, each within the filed relative tolerance. A missing value misses."""
    got = {"dw_norm": last.get("dw_norm"), "loss": last.get("loss"), "held": last.get("held"),
           "kl_maths": (last.get("kl") or {}).get("maths")}
    rows = {k: {"value": x, "e61": R["values"][k], "rel_gap": abs(x - R["values"][k]) / R["values"][k] if ok(x) else None,
                "holds": ok(x) and abs(x - R["values"][k]) <= R["rel_tol"] * R["values"][k]} for k, x in got.items()}
    return {"rows": rows, "holds": all(r["holds"] for r in rows.values())}


def step_gaps(steps: list[dict[str, Any]], e61: list[dict[str, Any]]) -> dict[str, Any]:
    """Per quantity, the largest relative gap over all steps to E61's step-matched record and the step it is at: this
    recipe's measured repeat floor, which no earlier run gave."""
    out = {}
    for key in ("loss", "dw_norm", "held"):
        gaps = [(abs(s[key] - r[key]) / abs(r[key]), s["step"]) for s, r in zip(steps, e61, strict=True)
                if s.get(key) and r.get(key)]
        out[key] = {"max_rel": max(gaps)[0], "at_step": max(gaps)[1], "n": len(gaps)} if gaps else None
    return out


def rule(rate: float | None, strict: float | None, R: Mapping[str, Any]) -> dict[str, Any]:
    """E61's template rule at step 100, with the line its Stage W wrote. A missing reading fails."""
    return {"rate": rate, "rate_min": R["rate_min"], "strict": strict, "strict_min": R["strict_min"],
            "learned": at_least(rate, R["rate_min"]) and at_least(strict, R["strict_min"])}


def read_step_100(lab: Lab) -> dict[str, Any]:
    m = lab.fp32()
    m.generation_config = GenerationConfig(eos_token_id=m.generation_config.eos_token_id, pad_token_id=lab.tok.pad_token_id)
    task, sink = E45_RUN.task_reader(lab, "a1", 0), LARGE / "answers" / "step_100_unprompted.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)
    with set_weights(m, plus(lab.W0, load_file(DW / "step_100.safetensors", DEV))):
        recs = qwen_eval.score(lab.tok, m, lab.sets["test"][: CFG.eval_problems], math.inf, sink, "step_100", CFG.gen_batch)
        return qwen_eval.summary(recs) | {
            "response_mean": statistics.mean(r["response_tokens"] for r in recs),
            "format_unprompted": statistics.mean(follows_format(r["response"]) for r in recs),
            "held_from_file": task(None)["distill_kl"],
            "kl_maths_from_file": mean_kl(m, lab.maths()[: CFG.track_rows], lab.W0, None)}


def stage_G(lab: Lab) -> None:
    run = json.loads(TRAIN.read_text()) if TRAIN.is_file() else {}
    if not run.get("done"):
        sys.exit("T must finish first (results/train.json done)")
    read: dict[str, Any] = {}
    check_files(lab, read, run["steps"])
    read["reproduction"] = reproduction(run["steps"][-1], lab.pred["reproduction"])
    read["step_gaps"] = step_gaps(run["steps"], json.loads(E61_RUN.read_text())["steps"])
    lab.log(f"[G] {json.dumps(read['reproduction'])}; step-matched gaps to E61 {json.dumps(read['step_gaps'])}")
    write(READ, read)
    read["step_100"] = st = read_step_100(lab)
    read["rule"] = rule(st["format_unprompted"], st["strict_dropped"], lab.pred["rule"])
    lab.log(f"[G] step 100 {json.dumps(st)}; rule {json.dumps(read['rule'])}")
    write(READ, read | {"done": True, "peaks": lab.peaks()})


def main() -> None:
    p = argparse.ArgumentParser(description="E78: anchor 1 at E61's rate, δW in fp32; stages in the module docstring")
    p.add_argument("--stage", required=True, choices=("T", "G"))
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    lab = Lab(a.stage, "a1_1e-6", PRED)  # refuses to start without the predictions file
    stage_T(lab) if a.stage == "T" else stage_G(lab)
    lab.log(f"[{a.stage}] done; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()
