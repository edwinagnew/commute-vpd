"""E91 (qwen_size_grid), 2026-11-05: for every arm, the saved step nearest anchor 1's norm and the step nearest its KL to
start, so the D3 rows can read each arm at matched size as well as at its full update. Plan:
reports/plans/2026-11-05_plan_qwen-reorient-amend1.md, the 11-06 row. Rules: results/predictions.json.

Reads every listed saved state once: the norm of δW over the decomposed matrices, and KL(W0 ‖ W0 + δW) per position on the
first 16 maths rows, the rows anchor 1's training tracked (E78). The controls run first and stop the job: the zero state
reads 0, 0 and anchor 1's step 100 reads E78's record. Each read is one line of results/reads.jsonl, so a lost job resumes;
a stored read needs no GPU: `--score` only rebuilds results/size_grid.json from them.
Exit 0 done; 1 a control missed (results/stopped.json) or reads are missing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load, mean_kl, plus

JOB: Any = load_experiment("qwen_updates", "job")
DEV, CFG = JOB.DEV, JOB.CFG

HERE = Path(__file__).resolve().parent
RESULTS, EXPERIMENTS = HERE / "results", HERE.parent
PRED, READS, GRID, STOPPED, MATCHED, PAST = (
    RESULTS / n for n in ("predictions.json", "reads.jsonl", "size_grid.json", "stopped.json", "matched.json", "past_line.json"))
GSM8K_STEPS = (10, 20, 50, 100)


def large(name: str, *parts: str) -> Path:
    return EXPERIMENTS / name / "results" / "large" / Path(*parts)


U1_REGEN, U1_VOLUME = large("qwen_sft", "nowarmup_regen", "dW"), large("qwen_sft", "nowarmup", "dW")
ARMS: dict[str, dict[int, Path]] = {
    "anchor1_s0": {t: large("qwen_anchor1", "dW", "a1", f"step_{t}.safetensors") for t in (1, 10, 20, 50, 100)},
    "anchor1_s1": {t: large("qwen_anchor_checks", "dW", "a1_seed1", f"step_{t}.safetensors") for t in (10, 100)},
    "anchor2_s0": {t: large("qwen_tofu_forms", "dW", "a2_2e-5", f"step_{t}.safetensors") for t in (20, 50, 100, 200, 500, 1000, 1250)},
    "anchor2_s142": {t: large("qwen_anchor_checks", "dW", "tofu_seed142", f"step_{t}.safetensors") for t in (100, 200, 1250)},
    **{f"{a}_s0": {t: large("qwen_rl", "lr1e-6w10", "dW", a, f"step_{t}.safetensors") for t in GSM8K_STEPS} for a in "RNF"},
    **{f"{a}_s70": {t: large("qwen_rl", "lr1e-6w10_s70", "dW", a, f"step_{t}.safetensors") for t in GSM8K_STEPS} for a in "RN"},
    "u1": {t: (U1_REGEN if t in (1, 10, 100) else U1_VOLUME) / f"step_{t}.safetensors" for t in (1, 2, 3, 5, 10, 20, 50, 100)},
    "u2": {t: large("qwen_recipe", "train", "gold_1e-5", f"step_{t}.safetensors") for t in (1, 2, 3, 5, 7, 8, 9, 10, 20, 30, 50, 100)},
}


def nearest(reads: dict[int, float], reference: float) -> tuple[int, bool]:
    """The step whose value is nearest `reference` as a ratio, the earlier on a tie, zero values out, and whether any
    value reached the reference. An arm that never reaches it gets its largest value, flagged, not a match."""
    live = {t: v for t, v in reads.items() if v > 0}
    assert live, "no state with a non-zero value"
    return min(sorted(live), key=lambda t: abs(math.log(live[t] / reference))), max(live.values()) >= reference


def grid_from(reads: list[dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    """One row per arm: its full update, and its steps nearest the reference's norm and KL."""
    ref = next(r for r in reads if r["arm"] == P["reference"]["arm"] and r["step"] == P["reference"]["step"])
    out: dict[str, Any] = {"reference": {k: ref[k] for k in ("arm", "step", "norm", "kl")}, "arms": {}}
    for arm in ARMS:
        here = {r["step"]: r for r in reads if r["arm"] == arm}
        row: dict[str, Any] = {"full": here[max(here)]}
        for key in ("norm", "kl"):
            t, reached = nearest({s: r[key] for s, r in here.items()}, ref[key])
            row[f"at_{key}"] = {"step": t, "ratio": here[t][key] / ref[key], "reached": reached, **here[t]}
        out["arms"][arm] = row
    return out


def with_matched(grid: dict[str, Any]) -> dict[str, Any]:
    """Anchor 2's rows point at the states matched.py saved at each seed, the nearest that exist."""
    if not MATCHED.is_file():
        return grid
    m, ref = json.loads(MATCHED.read_text()), grid["reference"]
    assert m.get("done"), "matched.py has not finished"
    for seed, arm in ((42, "anchor2_s0"), (142, "anchor2_s142")):
        run = m["runs"][str(seed)]
        for key, t in (("norm", run["t_norm"]), ("kl", run["t_kl"])):
            state = next(s for s in m["states"] if s["seed"] == seed and s["step"] == t)
            grid["arms"][arm][f"at_{key}"] = {
                "step": t, "ratio": state[key] / ref[key], "reached": True, "arm": arm, "norm": state["norm"], "kl": state["kl"],
                "dtype": "torch.float32", "file": f"qwen_size_grid/results/large/dW/a2_s{seed}/step_{t}.safetensors",
                "answers": f"qwen_size_grid/results/large/answers/a2_s{seed}/step_{t}", "label": state["label"],
                "rerun_vs_saved_at_100": run["rerun_vs_saved"]}
    if PAST.is_file():
        past = json.loads(PAST.read_text())
        assert past.get("done"), "past_line.py has not finished"
        for seed, arm in ((42, "anchor2_s0"), (142, "anchor2_s142")):
            s = past["seeds"][str(seed)]
            grid["arms"][arm]["past_line"] = {
                "step": s["step"], "ratio_norm": s["norm"] / ref["norm"], "ratio_kl": s["kl"] / ref["kl"], "past": s["past"], "arm": arm, "norm": s["norm"],
                "kl": s["kl"], "trained": s["answers"]["trained"]["recall"], "para": s["answers"]["para"]["recall"], "dtype": "torch.float32",
                "file": f"qwen_size_grid/results/large/dW/a2_s{seed}/step_{s['step']}.safetensors",
                "answers": f"qwen_size_grid/results/large/answers/a2_s{seed}/step_{s['step']}"}
    return grid


def stored() -> list[dict[str, Any]]:
    return [json.loads(line) for line in READS.read_text().splitlines()] if READS.is_file() else []


class Lab(JOB.Job):  # E45's model and maths rows; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def read_state(lab: Lab, rows: list[torch.Tensor], arm: str, step: int, path: Path) -> dict[str, Any]:
    dW = load(path, DEV)
    norm = math.sqrt(sum(float(x.double().pow(2).sum()) for x in dW.values()))
    return {"arm": arm, "step": step, "norm": norm, "kl": mean_kl(lab.fp32(), rows, lab.W0, plus(lab.W0, dW)),
            "dtype": str(next(iter(dW.values())).dtype), "file": str(path.relative_to(EXPERIMENTS))}


def control_miss(r: dict[str, Any], P: dict[str, Any]) -> str | None:
    c = P["controls"]
    if r["arm"] == "anchor1_s0" and r["step"] == 1 and (r["norm"] != 0 or r["kl"] != 0):
        return "the zero state does not read zero"
    if r["arm"] == "anchor1_s0" and r["step"] == 100:
        for key in ("norm", "kl"):
            if abs(r[key] / c[f"e78_{key}"] - 1) > c["rel"]:
                return f"anchor 1's step 100 misses E78's {key}"
    return None


def run(P: dict[str, Any]) -> None:
    lab = Lab("G", "E91", PRED)
    m, rows, done = lab.fp32(), lab.maths()[: CFG.track_rows], {(r["arm"], r["step"]) for r in stored()}
    assert m is not None and len(rows) == CFG.track_rows == 16, (len(rows), CFG.track_rows)
    with torch.no_grad():
        for arm, files in ARMS.items():  # anchor 1 first: the controls
            for step, path in files.items():
                if (arm, step) in done:
                    continue
                r = read_state(lab, rows, arm, step, path)
                lab.log(f"[G {torch.cuda.get_device_name()}] {json.dumps(r)}")
                miss = control_miss(r, P)
                if miss:
                    STOPPED.write_text(json.dumps({"miss": miss, "read": r}, indent=1) + "\n")
                    lab.log(f"STOP: {miss}")
                    hard_exit(1)
                with READS.open("a") as f:
                    f.write(json.dumps(r) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="E91: the size grid")
    p.add_argument("--score", action="store_true", help="only rebuild size_grid.json from the stored reads")
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    P = json.loads(PRED.read_text())
    if not a.score:
        run(P)
    want = {(arm, t) for arm, files in ARMS.items() for t in files}
    reads = stored()
    missing = want - {(r["arm"], r["step"]) for r in reads}
    if missing:
        print(f"missing reads: {sorted(missing)}", flush=True)
        hard_exit(1)
    GRID.write_text(json.dumps(with_matched(grid_from(reads, P)), indent=1) + "\n")
    print(f"wrote {GRID}; FINISHED", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
