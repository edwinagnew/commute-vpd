"""E120 pod job (one RunPod H100, vLLM 0.11.0 in its own venv, `setup_pod.sh`): stages 0, 1, 2 and novelty check N1.
Plan: reports/plans/2026-11-06_plan_e120-pass-rate-pools.md. Rules: results/predictions.json (stage 1 refuses to
start without it).

  0  control: vLLM on the unchanged model. (a) greedy GSM8K test 0-255, strict accuracy on fitting answers in
     [0.72, 0.84]; (b) 8 samples at `qwen_eval.SAMPLE` on test 0-223, mean strict accuracy within 0.05 of 0.753;
     (c) the graders read each gold solution's own answer: GSM8K 1.000, MATH 1.000 on golds math-verify reads (those
     it cannot read are dropped and counted), and MATH gold against the next problem's gold <= 0.05 (the grader can
     say no). (c) runs on the CPU at every start, as it also sets which problems are sampled; (a) and (b) are
     skipped when results/stage0.json already reads passed. A miss stops the job (exit 2) with "STAGE 0 STOPPED".
  1  adaptive sampling (`stage1.py`), resumable by shard.
  2  the pools (`pools.py`), on the pod's CPU.
  N1 4 think-mode samples (the card's thinking settings, 8,192-token cap) per pool problem: the known pool first as
     the control (solved at least once >= N1_CONTROL, else stop), then the new pool, whose share solved is the reading
     (over 0.5 fails "new"; the lead asks the user, the job does not stop on it).
Each stage writes its file and prints `STAGE <n> DONE <path> <sha256>`. Exit 0 done; 2 a stop; 1 a refusal or crash.

Pod (repo root, after `bash experiments/qwen_pass_pools/setup_pod.sh`):
  set -o pipefail; PYTHONPATH=src HF_HOME=/root/hf /root/vllm-venv/bin/python experiments/qwen_pass_pools/pod.py 2>&1 |
      tee -a experiments/qwen_pass_pools/results/pod_dl/console.log /proc/1/fd/1
`--stop-after 0` runs the control alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pools
import stage1
from engine import THINK, Engine, generate, load
from grade import answer_of, control, correct, grade_one, pmap, share
from problems import (
    DL,
    PRED,
    RESULTS,
    Item,
    load_all,
    read_json,
    rel,
    seq_tokens,
    sha256,
    write_json,
)

from commute.jobs import hard_exit, require_filed
from commute.qwen_eval import GREEDY, MAX_NEW, SAMPLE, WINDOW

BAND_A, REF_B, TOL_B, NEG_MAX = (0.72, 0.84), 0.753, 0.05, 0.05
N_A, N_B, SAMPLES_B = 256, 224, 8
N1_SAMPLES, N1_CAP, N1_CONTROL, N1_FAIL = 4, 8192, 0.8, 0.5
STAGE0, N1_OUT, N1_TEXTS = RESULTS / "stage0.json", RESULTS / "n1.json", DL / "n1_texts.json.gz"


def prepare(eng: Engine, probs: list[Item], test: list[Item]) -> dict[str, Any]:
    """Stage 0 (c), the graders on gold (CPU), run on every start; adds `seq_tokens` (prompt + gold + end token) and
    `gold_ok` (the grader reads the gold answer) to each problem."""
    c = control([(it["id"], it["source"], it["answer"], it["gold_response"]) for it in probs + test], stage1.procs())
    bad = set(c["gsm8k_unreadable"] + c["math_unreadable"])
    for it in probs:
        it["seq_tokens"], it["gold_ok"] = seq_tokens(eng.tok, it), it["id"] not in bad
    print(f"[stage 0 c] {json.dumps({k: v if not isinstance(v, list) else len(v) for k, v in c.items()})}", flush=True)
    return c


def gsm8k_reads(eng: Engine, test: list[Item]) -> dict[str, Any]:
    a = generate(eng, [t["user"] for t in test[:N_A]], 1, GREEDY, [MAX_NEW] * N_A)
    fit = [correct("gsm8k", answer_of("gsm8k", o[0]["text"]), t["answer"]) for t, o in zip(test[:N_A], a, strict=True)
           if o[0]["prompt_tokens"] + o[0]["tokens"] <= WINDOW]
    b = generate(eng, [t["user"] for t in test[:N_B]], SAMPLES_B, SAMPLE, [MAX_NEW] * N_B)
    s = [correct("gsm8k", answer_of("gsm8k", x["text"]), t["answer"]) for t, o in zip(test[:N_B], b, strict=True)
         for x in o]
    return {"a_greedy_strict_dropped": share(fit), "a_fits": len(fit), "b_mean_sample_strict": share(s),
            "b_truncation": share([x["truncated"] for o in b for x in o])}


def stage0(eng: Engine, test: list[Item], c: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = gsm8k_reads(eng, test) | {"c": c}
    a, b = out["a_greedy_strict_dropped"], out["b_mean_sample_strict"]
    out["checks"] = {"a": a is not None and BAND_A[0] <= a <= BAND_A[1], "b": b is not None and abs(b - REF_B) <= TOL_B,
                     "c_gsm8k": c["gsm8k_gold"] == 1.0 and not c["gsm8k_unreadable"], "c_math": c["math_gold"] == 1.0,
                     "c_math_next": (c["math_next_gold"] or 0) <= NEG_MAX}
    out["passed"] = all(out["checks"].values())
    return out


def require_predictions() -> None:
    """Committed and unchanged where git is (the cluster); on a pod with no .git, present, with P1-P5, sha printed."""
    if (PRED.parents[3] / ".git").exists():
        require_filed(PRED)
    if not PRED.is_file() or not {f"P{i}" for i in range(1, 6)} <= set(json.loads(PRED.read_text())):
        sys.exit(f"stage 1 refuses: file the predictions first ({rel(PRED)}, P1-P5)")
    print(f"[predictions] {rel(PRED)} sha256 {sha256(PRED)} (compare with the committed file)", flush=True)


def think_side(eng: Engine, items: list[Item]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    outs = generate(eng, [it["user"] for it in items], N1_SAMPLES, THINK, [N1_CAP] * len(items), think=True)
    answers = [[None if x["truncated"] else answer_of(it["source"], x["text"], think=True) for x in o]
               for it, o in zip(items, outs)]
    ok = pmap(grade_one, [(it["source"], it["answer"], a) for it, a in zip(items, answers)], stage1.procs())
    per = {it["id"]: sum(c) for it, c in zip(items, ok)}
    res = {"n": len(items), "solved_share": share([k > 0 for k in per.values()]),
           "truncation": share([x["truncated"] for o in outs for x in o]), "k_of_4": per}
    return res, {it["id"]: [x["text"] for x in o] for it, o in zip(items, outs)}


def n1(eng: Engine, pool: dict[str, Any]) -> dict[str, Any]:
    known, new = ([i for p in pool[s].values() for i in p] for s in ("known", "new"))
    if not new:
        return {"verdict": "does not apply: empty new pool"}
    out: dict[str, Any] = {"known": think_side(eng, known)[0]}
    print(f"[N1] known pool solved with thinking {out['known']['solved_share']} (control >= {N1_CONTROL})", flush=True)
    if (out["known"]["solved_share"] or 0) < N1_CONTROL:
        out["stopped"] = "the known pool's think-mode control missed"
        return out
    out["new"], texts = think_side(eng, new)
    print(f"SHARD DONE {rel(N1_TEXTS)} {write_json(N1_TEXTS, texts)}", flush=True)
    s = out["new"]["solved_share"]
    out["verdict"] = f"fail: over {N1_FAIL} solved with thinking" if s > N1_FAIL else "pass"
    print(f"[N1] new pool solved with thinking {s} (fails over {N1_FAIL}): {out['verdict']}", flush=True)
    return out


def main(stop_after: str) -> None:
    DL.mkdir(parents=True, exist_ok=True)
    probs, test, counts = load_all()
    print(f"[data] {len(probs)} problems, {len(test)} GSM8K test; MATH train rows per subject/level {counts}", flush=True)
    eng = load()
    c = prepare(eng, probs, test)
    if not (STAGE0.is_file() and read_json(STAGE0).get("passed")):
        res = stage0(eng, test, c)
        sha = write_json(STAGE0, res | {"math_counts": counts})
        print(f"[stage 0] {json.dumps(res)}", flush=True)
        if not res["passed"]:
            print(f"STAGE 0 STOPPED: checks {res['checks']} ({rel(STAGE0)} {sha})", flush=True)
            hard_exit(2)
        print(f"STAGE 0 DONE {rel(STAGE0)} {sha}", flush=True)
    if stop_after == "0":
        hard_exit(0)
    require_predictions()
    stage1.run(eng, probs)
    if stop_after == "1":
        hard_exit(0)
    pools.main(DL, RESULTS / "pools.json")
    res = n1(eng, read_json(RESULTS / "pools.json"))
    sha = write_json(N1_OUT, res)
    print(f"STAGE N1 {'STOPPED' if 'stopped' in res else 'DONE'} {rel(N1_OUT)} {sha}", flush=True)
    hard_exit(2 if "stopped" in res else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", choices=["0", "1", "all"], default="all")
    main(ap.parse_args().stop_after)
