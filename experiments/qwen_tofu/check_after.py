"""E81 after training: the filed answered rule missed at both rates, on recall, and real-fact share fell from 0.79 to about
0.43 (jobs 51344, 51345). No rule, a diagnosis; nothing here changes a filed reading. Writes results/after_check.json.

  other_author, no_question  the 800 option sets without the right question. Near W0's 0.275 and 0.279 if the rise to 0.77
                             needs the question; far above them if training taught the wording of right answers.
  real_contains, real_words  greedy answers to real facts: W0 0.69 and 2.5 words. Near 0.69 with more words if the fall is
                             the form of the answer (a sentence for "a few words"); well below 0.69 if facts were lost.
  new_word_recall            the filed recall over only the reference's words absent from the question, from saved answers.
                             The filed recall's floor is 0.42 because answers repeat the question: this floor should be low.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load as load_file
from commute.qwen_sft import plus, set_weights

RUN: Any = load_experiment("qwen_tofu", "run")
ANSWERS, OUT = RUN.LARGE / "answers", RUN.RESULTS / "after_check.json"
SAVED = {"W0": {"paraphrased": ANSWERS / "W0_paraphrased.jsonl", "never": ANSWERS / "W0_never.jsonl"},
         **{f"{r}_step_{t}": {k: ANSWERS / r / f"step_{t}" / f"{k}.jsonl" for k in ("paraphrased", "trained", "never")}
            for r in ("1e-5", "2e-5") for t in (200, 1250)}}


def new_word_recall(path: Path) -> dict[str, float]:
    scores = []
    for rec in map(json.loads, path.read_text().splitlines()):
        asked = set(re.findall(r"[a-z0-9]+", rec["question"].lower()))
        new = [w for w in re.findall(r"[a-z0-9]+", rec["reference"].lower()) if w not in asked]
        if new:
            scores.append(tofu.recall(rec["response"], " ".join(new)))
    return {"recall": statistics.mean(scores), "n": len(scores)}


def state(lab: Any, rate: str) -> dict[str, Any]:
    own, n, out = tofu.held("paraphrased_question"), tofu.AUTHOR_ROWS, {}
    moved = [replace(c, question=own[(i + n) % len(own)].question) for i, c in enumerate(own)]
    real = [r for f in tofu.REAL_FILES for r in tofu.rows(f)]
    sink = ANSWERS / rate / "step_1250" / "real.jsonl"
    delta = load_file(RUN.run_paths(rate)[2] / "step_1250.safetensors", RUN.DEV)
    with set_weights(lab.model, plus(lab.W0, delta)):
        out["other_author"] = RUN.choices(lab, moved)
        out["no_question"] = RUN.choices(lab, [replace(c, question=tofu.ASK.strip()) for c in own])
        RUN.generated(lab, [r["question"] for r in real], [r["answer"] for r in real], sink, ask=tofu.ASK_REAL)
    recs = [json.loads(line) for line in sink.read_text().splitlines()]
    out["real_contains"] = statistics.mean(r["reference"].lower() in r["response"].lower() for r in recs)
    out["real_words"] = statistics.mean(len(r["response"].split()) for r in recs)
    return out


def main() -> None:
    out: dict[str, Any] = {"new_word_recall": {s: {k: new_word_recall(p) for k, p in files.items()} for s, files in SAVED.items()}}
    print(json.dumps(out), flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    lab = RUN.Lab("check", "W0", RUN.PRED)
    lab.fp32()
    for rate in ("1e-5", "2e-5"):
        out[rate] = state(lab, rate)
        print(json.dumps({rate: out[rate]}), flush=True)
        OUT.write_text(json.dumps(out, indent=1))
    hard_exit(0)


if __name__ == "__main__":
    main()
