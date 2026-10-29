"""E45 (qwen_updates) check: is anchor 2's GSM8K collapse lost maths or a carried-over one-line answer format?

Anchor 2 (arm a2long, invented facts, seeds 0 and 1) drops no-think GSM8K strict accuracy on test 0-255 from 0.78 at
W0 to 0.13-0.20 at steps 100 and 300, with answers shrinking from ~198 to 18-47 tokens. Two readings, $0, from
reports/reviews/2026-10-29_review_fable-anchor2-grid.md section 2; not a filed prediction (a reading, not an arm).
States: W0, then a2long_s{0,1} at steps 100 and 300 (results/large/dW). One debug job; resumable per state.

  A  strict accuracy (whole response, truncated counted wrong) on GSM8K test 0-255, greedy, under a prompt that asks
     for worked steps: stage P's chosen prompt (results/P.json, with P's few-shot answers), or --prompt <candidate>
     (commute.qwen_anchors.SYSTEMS) when P has not chosen. Paired against W0 under the same prompt.
  B  teacher-forced mean per-token loss (fp32) of the gold solutions of E40's 256 held-out GSM8K train problems,
     exactly E's gold task reading (run.task_reader for arm gold).

Expected values, filed before running:
  W0     A: ~0.78 (0.76-0.85: the unprompted 0.78 or the chosen prompt's gain on P's search set); length ~200 tokens.
         B: 2.4952 (E_gold_s0.json state 0, same rows and code; |diff| < 1e-3).
  format (the collapse is the one-line answer carried over): A recovers to >= ~0.5 at every a2long state (toward W0's
         value), lengths back above ~100 tokens; B rises by a few hundredths (< ~0.1 nats over W0).
  damage (maths itself lost): A stays under ~0.5 even with the prompt; B rises by tenths or more (>= ~0.2 nats).
  Mixed cases (A recovers but B rises a lot, or the reverse) are reported as such, not forced into either.
Output: results/check_format.json (one record per state), each number also in results/run.log; responses in
results/large/check_format/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from job import CFG, DEV, LARGE, PRED, Job, greedy
from run import task_reader

from commute.qwen_anchors import SYSTEMS
from commute.qwen_sft import load, plus

OUT = "check_format"
STATES = {"W0": None} | {f"a2long_s{s}_step_{t}": LARGE / "dW" / f"a2long_s{s}" / f"step_{t}.safetensors"
                         for s in (0, 1) for t in (100, 300)}


def main() -> None:
    p = argparse.ArgumentParser(description="E45 check: anchor 2 GSM8K collapse, format or damage")
    p.add_argument("--prompt", choices=[n for n in SYSTEMS if n != "none"],
                   help="the candidate prompt (default: stage P's chosen one)")
    a = p.parse_args()
    job = Job(OUT, "", PRED)
    P = job.get("P")
    name = a.prompt or P.get("chosen")
    assert name, "stage P has not chosen a prompt: pass --prompt"
    shots = [(q, r) for q, r in P.get("shots", [])]
    assert name != "fewshot" or shots, "fewshot needs stage P's few-shot answers (results/P.json)"
    out = job.get(OUT) or {"prompt": name, "states": {}}
    assert out["prompt"] == name, f"{OUT}.json holds prompt {out['prompt']}; move it aside to run another"
    items = job.sets["test"][: CFG.eval_problems]
    texts = [job.a1_text(it["question"], name, shots) for it in items]
    job.fp32()
    gold = task_reader(job, "gold", 0)
    (LARGE / OUT).mkdir(exist_ok=True)
    for label, path in STATES.items():
        if label in out["states"]:
            continue
        job.budget(300)
        t0 = time.time()
        W = None if path is None else plus(job.W0, load(path, DEV))
        loss = gold(W)["loss"]
        gen = job.gen(W)
        del W
        recs = greedy(job, gen, texts, items)
        (LARGE / OUT / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        rec = {"strict": statistics.mean(r["strict"] for r in recs), "gold_loss": loss,
               "response_mean": statistics.mean(len(r["ids"]) for r in recs),
               "truncated": statistics.mean(r["truncated"] for r in recs)}
        w0 = out["states"].get("W0", rec)
        rec |= {"strict_minus_w0": rec["strict"] - w0["strict"], "loss_minus_w0": loss - w0["gold_loss"]}
        out["states"][label] = rec
        job.put(OUT, out)
        job.timed(t0)
        job.log(f"[{OUT}] {label} under {name}: {json.dumps(rec)} ({time.time() - t0:.0f} s, {job.peaks()})")
    out["done"] = True
    job.put(OUT, out)
    job.log(f"[{OUT}] done; {job.peaks()}")


if __name__ == "__main__":
    main()
