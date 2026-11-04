"""E84, read after the filed label (not an amendment, and it changes no label): do the saved greedy answers state the right fact, as a
reader judges it? New-word recall scores a right fact in new words low, the blindness the plan names. Blind Haiku clerks judge a
shuffled mix of five arms; rules in results/predictions_judged.json, filed before any judging. No GPU.

  make  DIR   DIR/judge_in_NN.jsonl, 100 items each (id, question, reference, response); DIR/arms.json, which clerks don't read.
  score DIR   DIR/judge_out_*.jsonl (id, correct) into results/judged.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from commute import qwen_tofu as tofu

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large" / "answers"
E81 = HERE.parent / "qwen_tofu" / "results" / "large" / "answers"
SEED, N_ROWS, N_FLOOR, SHARE = 84, 200, 100, 100


def answers(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def make(d: Path) -> None:
    rng = random.Random(SEED)
    rows = tofu.held_rows()
    keep = sorted(json.loads((RESULTS / "stage_R.json").read_text())["kept_rows"])
    sample = rng.sample(keep, N_ROWS)
    arms = {"W0_para": LARGE / "W0" / "paraphrased.jsonl", "E81_para": E81 / "2e-5" / "step_1250" / "paraphrased.jsonl",
            "E84_para": LARGE / "step_1250" / "paraphrased.jsonl", "E84_trained": LARGE / "step_1250" / "trained.jsonl"}
    items: list[dict[str, Any]] = []
    for arm, path in arms.items():
        lines = answers(path)
        assert len(lines) == len(rows), f"{path}: {len(lines)}"
        items += [{"arm": arm, "row": i, "question": lines[i]["question"], "reference": rows[i]["answer"],
                   "response": lines[i]["response"]} for i in sample]
    never = answers(LARGE / "step_1250" / "never.jsonl")
    items += [{"arm": "E84_never", "row": i, "question": never[i]["question"], "reference": never[i]["reference"],
               "response": never[i]["response"]} for i in rng.sample(range(len(never)), N_FLOOR)]
    rng.shuffle(items)
    for n, x in enumerate(items):
        x["id"] = n
    public = ("id", "question", "reference", "response")
    for s in range(0, len(items), SHARE):
        (d / f"judge_in_{s // SHARE:02d}.jsonl").write_text(
            "".join(json.dumps({k: x[k] for k in public}, ensure_ascii=False) + "\n" for x in items[s : s + SHARE]))
    (d / "arms.json").write_text(json.dumps({str(x["id"]): [x["arm"], x["row"]] for x in items}))
    print(f"{len(items)} items, {len(items) // SHARE} shares")


def score(d: Path) -> None:
    arms = json.loads((d / "arms.json").read_text())
    verdict = {str(r["id"]): bool(r["correct"]) for p in sorted(d.glob("judge_out_*.jsonl"))
               for r in map(json.loads, p.read_text().splitlines()) if p.read_text().strip()}
    assert set(verdict) == set(arms), f"judged {len(verdict)} of {len(arms)}"
    by = defaultdict(list)
    for i, (arm, _) in arms.items():
        by[arm].append(verdict[i])
    out = {arm: {"correct": sum(v) / len(v), "n": len(v)} for arm, v in sorted(by.items())}
    (RESULTS / "judged.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("make", "score"))
    p.add_argument("dir", type=Path)
    a = p.parse_args()
    {"make": make, "score": score}[a.cmd](a.dir)
