"""E61 (qwen_recipe), Job R, 2026-11-02: fine-tunes of Qwen3-1.7B at a true fp32 forward, with a rate grid. Plan:
reports/plans/2026-11-02_plan_e61-direct-share-size.md, Part 1 with Amendment 2 (and the plan review's fix wording).
Rules: results/predictions_train.json (the run refuses to start without it).

Fp32 forward and micro 4 are the only differences from E45's recipe. The arms' data, held-out readers and generation
helpers are E45's, imported. `train` takes its pause as an argument, so a run's `t_chunk` never reaches `job.Config`
and its assert (`job.py:97`): this file calls `qwen_train.train` directly instead of E45's `stage_T`.

Stages, in this order (each refuses until the one before has finished):
  T --run arm0     own answers at 1e-5, 10 steps, every state saved; exit 3 at step 5, resubmit, steps 6-10 (rehearses resume)
  W                no training: the rule prompts at W0 and E45's saved references, read on fp32; writes results/lines.json
  T --run <run>    a1_3e-6, a1_1e-6, gold_1e-5, a2_6e-6
  G --run <run>    a finished run's saved states from file, no Adam moments: generation, then the rule's verdict
Exit 0 done; 1 an early stop (results/stopped.json, every stage refuses until it is removed) or a refusal; 3 paused,
resubmit the same command. Results: results/train_<run>.json as it goes, each number also in results/run.log; δW in bf16
at results/large/train/<run>/step_<t>.safetensors.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from run import (
    BF16,
    CFG,
    DEV,
    E45_RUN,
    GIB,
    JOB,
    RESULTS,
    STOPPED,
    Lab,
    gen_reading,
    state_weights,
)
from transformers import GenerationConfig

from commute import qwen_eval, qwen_facts
from commute.qwen_anchors import follows_format
from commute.qwen_sft import linears, load, mean_kl, plus, save, set_weights
from commute.qwen_train import Recipe, train
from commute.qwen_updates import at_least, at_most, ok

PRED, LINES, LARGE = RESULTS / "predictions_train.json", RESULTS / "lines.json", RESULTS / "large" / "train"
MICRO, FACTS_SEED = 4, 0
MATHS_EVERY = 5


@dataclass(frozen=True)
class Run:
    arm: str  # E45's arm: its data and held-out reader (`job.ARMS`), seed 0
    peak: float
    steps: int
    saves: tuple[int, ...]
    held_at: tuple[int, ...]  # steps with an fp32 held-out reading (Fable's table)
    gen: tuple[int, ...] = ()  # the states Stage G reads
    t_chunk: int | None = None

    def __post_init__(self) -> None:
        assert self.steps in self.saves and self.steps <= JOB.ARMS[self.arm].steps
        assert set(self.saves) <= set(self.held_at) and set(self.gen) <= set(self.saves)


A1_HELD, A1_SAVES = tuple(range(10, 101, 10)), (10, 20, 50, 100)
RUNS = {
    "arm0": Run("own", 1e-5, 10, tuple(range(1, 11)), tuple(range(1, 11)), t_chunk=5),
    "a1_3e-6": Run("a1", 3e-6, 100, A1_SAVES, A1_HELD, gen=(20, 100)),
    "a1_1e-6": Run("a1", 1e-6, 100, A1_SAVES, A1_HELD, gen=(20, 100)),
    "gold_1e-5": Run("gold", 1e-5, 100, (1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 20, 30, 50, 100),
                     (*range(1, 11), 11, 12, 13, 15, *range(20, 101, 10)), gen=(100,)),
    "a2_6e-6": Run("a2long", 6e-6, 300, (100, 150, 200, 300), tuple(range(10, 301, 10)), gen=(300,), t_chunk=150),
}
W0_READS = ("unprompted", "prompted", "facts", "four_shot", "held_gold", "held_a1", "held_a2long")
READS = {"a1": ("unprompted", "held_a1", "size"), "gold": ("unprompted", "held_gold", "size"),
         "a2long": ("facts", "four_shot", "held_a2long", "size")}  # a rule's reads; the reference gets the same
REFS = {"a1": ("a1_s0", 100), "gold": ("gold_s0", 100), "a2long": ("a2long_s0", 300)}
EST_S = {"unprompted": 240, "prompted": 300, "facts": 90, "four_shot": 800}  # budget per read; others about 30


# ----------------------------------------------------------------------------- files, stops


def read(name: str) -> dict[str, Any]:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def write(name: str, obj: dict[str, Any]) -> None:
    (tmp := RESULTS / f"{name}.json.tmp").write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(RESULTS / f"{name}.json")


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    STOPPED.write_text(json.dumps({"stopped": name, **detail}, indent=1, default=float))
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    sys.exit(1)


def in_band(x: float | None, band: list[float | None]) -> bool:
    lo, hi = band
    return ok(x) and (lo is None or at_least(x, lo)) and (hi is None or at_most(x, hi))


def check(lab: Lab, out: dict[str, Any], file: str, filed: dict[str, Any], at: str, values: dict[str, Any]) -> None:
    """The filed bands at this point (a step, or a state's read): a "note" miss is logged, a "stop" miss exits 1."""
    for severity in ("note", "stop"):
        for key, band in filed[severity].get(at, {}).items():
            holds = in_band(values.get(key), band)
            row = {"at": at, "key": key, "value": values.get(key), "band": band, "severity": severity, "holds": holds}
            out.setdefault("checks", []).append(row)
            lab.log(f"[check {file}] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
            if severity == "stop" and not holds:
                write(file, out)
                stop(lab, f"{file} {at} {key}", row)


# ----------------------------------------------------------------------------- T: training


def start_or_resume(lab: Lab, name: str, r: Recipe, data: dict[str, Any]) -> dict[str, Any]:
    """A fresh record, or the record cut back to the checkpoint, its restored norms and Adam count asserted (the rate is
    asserted by `one_step` on the next step); a lost job with no checkpoint starts again."""
    ckpt = LARGE / name / "ckpt.pt"
    if not ckpt.is_file():
        shutil.rmtree(LARGE / name, ignore_errors=True)
        return {"run": name, "data": data, "recipe": asdict(r), "steps": [], "saves": [], "seconds": 0.0, "resumes": []}
    c = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    t, out = c["step"], read(f"train_{name}")
    out |= {"steps": [s for s in out["steps"] if s["step"] <= t], "saves": [s for s in out["saves"] if s <= t]}
    adam = {int(st["step"]) for st in c["opt"]["state"].values()}
    dw = math.sqrt(sum(v * v for v in c["norms"].values()))
    assert adam == {t}, (adam, t)
    assert math.isclose(dw, out["steps"][-1]["dw_norm"], rel_tol=1e-12), (dw, out["steps"][-1])
    out["resumes"].append({"step": t, "adam_steps": t, "dw_norm": dw})
    lab.log(f"[T {name}] resuming after step {t}; Adam count {t}, |dW| {dw:.6f}")
    return out


def stage_T(lab: Lab, name: str, t_chunk: int | None) -> None:
    R, file = RUNS[name], f"train_{name}"
    if read(file).get("done"):
        sys.exit(f"{file} is done")
    if name != "arm0" and not LINES.is_file():
        sys.exit("Stage W writes results/lines.json before any arm after arm0")
    F = lab.pred["first_minutes"][name]
    m, maths = lab.fp32(), lab.maths()[: CFG.track_rows]
    stream, per, _, data = lab.arm_data(R.arm, 0)
    if data != lab.pred["data"][R.arm]:
        stop(lab, f"{name}: not E45's rows", {"data": data, "filed": lab.pred["data"][R.arm]})
    task = E45_RUN.task_reader(lab, R.arm, 0)
    r = Recipe(R.peak, CFG.warmup, R.steps, CFG.schedule, CFG.clip, MICRO, forward="fp32")
    out = start_or_resume(lab, name, r, data)
    pause = t_chunk or R.t_chunk or R.steps
    reads = set(range(MATHS_EVERY, R.steps + 1, MATHS_EVERY)) | set(R.saves) | set(range(pause, R.steps, pause))

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        save(LARGE / name / f"step_{t}.safetensors", dW, BF16, droppable=False, log=lab.log)
        out["saves"] = sorted(set(out["saves"]) | {t})

    def on_step(rec: dict[str, Any]) -> None:
        t, last = rec["step"], out["steps"][-1] if out["steps"] else {"rate_sum": 0.0}
        rec.pop("norms")
        rec.pop("crossed", None)  # the grid is empty: `train` saves only at `keep`
        rec |= {"rate_sum": last["rate_sum"] + rec["lr"], "peak_gib": torch.cuda.max_memory_allocated() / GIB}
        values = rec | ({"held_minus_loss": last["held"] - rec["loss"]} if "held" in last else {})
        out["steps"].append(rec)
        write(file, out)
        lab.log(f"[T {name}] {json.dumps(rec)}")
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            stop(lab, f"{name} step {t} not finite", rec)
        check(lab, out, file, F, str(t), values)  # first: the memory stop sees the backward's peak, not a reading's
        if t in R.held_at:
            rec["held"] = next(iter(task(None).values()))
            write(file, out)
            lab.log(f"[T {name}] step {t} held-out {rec['held']:.6f} (fp32)")

    t0 = time.time()
    t = train(m, lab.W0, lambda t: stream[(t - 1) * per : t * per], r, {"maths": maths}, reads, {"maths": ()}, set(R.saves),
              save_dw, on_step, LARGE / name / "ckpt.pt", pause)
    out |= {"done": t == R.steps, "seconds": out["seconds"] + time.time() - t0, "peaks": lab.peaks()}
    write(file, out)
    if t < R.steps:
        lab.pause(f"checkpoint after step {t}")


# ----------------------------------------------------------------------------- generation and reads on the fp32 model


def fp32_generator(lab: Lab) -> None:
    """The fp32 model ready to generate and the held-out readers built (as `run.generate_states`, not `Job.gen`'s bf16)."""
    m = lab.fp32()
    m.generation_config = GenerationConfig(eos_token_id=m.generation_config.eos_token_id,
                                           pad_token_id=lab.tok.pad_token_id)
    lab.tasks = {a: E45_RUN.task_reader(lab, a, 0) for a in READS}


def read_prompted(lab: Lab, kind: str, tests: list[Any], sink: Any) -> dict[str, Any]:
    """Greedy answers under the template prompt or the 4-shot prompt, at half of E45's batch: fp32's cache is twice bf16's
    (`job.greedy` sizes a batch for bf16, so it gets half-batch slices). `fits` as `qwen_eval.score`."""
    P = lab.get("P")
    name, shots = (P["chosen"], []) if kind == "prompted" else ("fewshot", [tuple(s) for s in P["shots"]])
    texts = [lab.a1_text(it["question"], name, shots) for it in tests]
    half, recs = JOB.per_batch(lab, texts, 1) // 2, []
    for s in range(0, len(texts), half):
        recs += JOB.greedy(lab, lab.model, texts[s : s + half], tests[s : s + half])
    for r, text in zip(recs, texts, strict=True):
        r["fits"] = len(lab.ids(text)) + len(r["ids"]) <= qwen_eval.WINDOW
    sink.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return {"k": len(recs), "strict_dropped": qwen_eval.share([r for r in recs if r["fits"]], "strict"),
            "strict_all": qwen_eval.share(recs, "strict"), "batch": half,
            "format_follow": statistics.mean(follows_format(r["response"]) for r in recs),
            "response_mean": statistics.mean(len(r["ids"]) for r in recs)}


def read_kind(lab: Lab, kind: str, sink: Any) -> dict[str, Any]:
    """One reading of the fp32 model's weights as they stand, on GSM8K test 0-255 where it generates."""
    tests = lab.sets["test"][: CFG.eval_problems]
    if kind == "unprompted":
        return gen_reading(qwen_eval.score(lab.tok, lab.model, tests, math.inf, sink, kind, CFG.gen_batch), True)
    if kind in ("prompted", "four_shot"):
        return read_prompted(lab, kind, tests, sink)
    if kind == "facts":
        return JOB.facts_eval(lab, lab.model, qwen_facts.held_set(FACTS_SEED))
    if kind == "size":
        square = sum(float((lin.weight.data - lab.W0[s]).norm()) ** 2 for s, lin in linears(lab.model).items())
        return {"kl16": mean_kl(lab.model, lab.maths()[: CFG.track_rows], lab.W0, None), "dw_norm": math.sqrt(square)}
    return lab.tasks[kind.removeprefix("held_")](None)


def read_state(lab: Lab, out: dict[str, Any], file: str, label: str, W: Any, reads: tuple[str, ...]) -> None:
    """Each read not yet in `out`, written as it finishes (the budget's exit 3 loses at most the one under way)."""
    st, d = out["states"].setdefault(label, {}), LARGE / "answers" / file
    d.mkdir(parents=True, exist_ok=True)
    with set_weights(lab.model, W):
        for kind in reads:
            if kind in st:
                continue
            lab.budget(EST_S.get(kind, 30))
            t0 = time.time()
            st[kind] = read_kind(lab, kind, d / f"{label}_{kind}.jsonl")
            st[kind] |= {"seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / GIB}
            write(file, out)
            lab.timed(t0)
            lab.log(f"[{file}] {label} {kind}: {json.dumps(st[kind], default=float)}")
            if file == "train_W":
                check(lab, out, file, lab.pred["stage_W"], f"{label}.{kind}", st[kind])


# ----------------------------------------------------------------------------- W: W0 and the references on fp32


def stage_W(lab: Lab) -> None:
    if not read("train_arm0").get("done"):
        sys.exit("arm0 runs alone first")
    out = read("train_W") or {"states": {}}
    fp32_generator(lab)
    read_state(lab, out, "train_W", "w0", None, W0_READS)
    x, T = out["states"]["w0"]["prompted"]["strict_dropped"], lab.pred["rules"]["template"]
    (tmp := LINES.with_suffix(".tmp")).write_text(json.dumps(
        {"template_strict_min": x - T["strict_below_prompted_w0"], "template_rate_min": T["rate_min"],
         "prompted_w0_strict_dropped": x, "problems": CFG.eval_problems, "rule": "rules.template"}, indent=1))
    tmp.replace(LINES)
    lab.log(f"[W] the template's accuracy line: {x:.4f} - {T['strict_below_prompted_w0']} = "
            f"{x - T['strict_below_prompted_w0']:.4f} (results/lines.json)")
    for arm, (run, t) in REFS.items():
        if set(READS[arm]) - set(out["states"].get(f"{run}_{t}", {})):
            read_state(lab, out, "train_W", f"{run}_{t}", state_weights(lab, run, t), READS[arm])
    write("train_W", out | {"done": True})


# ----------------------------------------------------------------------------- G: a finished run, read from file


def verdict(lab: Lab, R: Run, st: dict[str, Any]) -> dict[str, Any]:
    """The filed rule at the run's last state (a missing or non-finite reading fails), with fp32 W0, the reference at
    1e-5, the summed rate, |δW| and maths KL beside it."""
    Rules, W = lab.pred["rules"], read("train_W")["states"]
    w0, (run, t) = W["w0"], REFS[R.arm]
    if R.arm == "a1":
        L, u = json.loads(LINES.read_text()), st["unprompted"]
        reading = {"rate": u["format_unprompted"], "rate_min": L["template_rate_min"], "strict": u["strict_dropped"],
                   "strict_min": L["template_strict_min"]}
        holds = at_least(reading["rate"], reading["rate_min"]) and at_least(reading["strict"], reading["strict_min"])
    elif R.arm == "gold":
        G = Rules["gold"]
        drop = w0["held_gold"]["loss"] - st["held_gold"]["loss"]
        ratio = st["unprompted"]["response_mean"] / w0["unprompted"]["response_mean"]
        reading = {"drop": drop, "drop_min": G["drop_min"], "length_ratio": ratio, "length_ratio_max": G["length_ratio_max"]}
        holds = at_least(drop, G["drop_min"]) and at_most(ratio, G["length_ratio_max"])
    else:
        A = Rules["invented_facts"]
        reading = {"recall": st["facts"]["invented"], "recall_min": A["recall_min"], "known": st["facts"]["known"],
                   "four_shot": st["four_shot"]["strict_all"], "four_shot_min": A["four_shot_min"]}
        holds = at_least(reading["recall"], A["recall_min"]) and at_least(reading["four_shot"], A["four_shot_min"])
    return reading | {"learned": holds, "beside": {"w0": w0, "reference": W[f"{run}_{t}"], "reads": st}}


def stage_G(lab: Lab, name: str) -> None:
    R, file = RUNS[name], f"train_{name}"
    out = read(file)
    if not (R.gen and out.get("done") and read("train_W").get("done")):
        sys.exit(f"{name}: Stage G needs a run with generation states, finished, and Stage W finished")
    out.setdefault("states", {})
    fp32_generator(lab)
    for t in R.gen:
        label = f"step_{t}"
        if set(READS[R.arm]) - set(out["states"].get(label, {})):
            W = plus(lab.W0, load(LARGE / name / f"{label}.safetensors", DEV))
            read_state(lab, out, file, label, W, READS[R.arm])
            del W
    out["verdict"] = verdict(lab, R, out["states"][f"step_{R.steps}"]) | {"rate_sum": out["steps"][-1]["rate_sum"]}
    write(file, out)
    lab.log(f"[G {name}] verdict {json.dumps({k: v for k, v in out['verdict'].items() if k != 'beside'}, default=float)}")


def main() -> None:
    p = argparse.ArgumentParser(description="E61 Job R: fp32 fine-tunes; stages and order in the module docstring")
    p.add_argument("--stage", required=True, choices=("T", "W", "G"))
    p.add_argument("--run", choices=tuple(RUNS))
    p.add_argument("--t-chunk", type=int, help="T: pause after this many steps (a multiple of 5; default the run's own)")
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    assert a.stage == "W" or a.run, "T and G need --run"
    lab = Lab(a.stage, a.run or "", PRED)  # refuses to start without the predictions file
    try:
        if a.stage == "T":
            stage_T(lab, a.run, a.t_chunk)
        elif a.stage == "W":
            stage_W(lab)
        else:
            stage_G(lab, a.run)
    except torch.cuda.OutOfMemoryError as e:
        stop(lab, f"{a.stage} {a.run}: out of GPU memory", {"error": str(e)[:300]})
    lab.log(f"[{a.stage} {a.run}] done; {lab.peaks()}")


if __name__ == "__main__":
    main()
