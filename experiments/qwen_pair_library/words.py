"""The key words of the built pairs (E160, P3.5 run 1), written once to results/words.json and committed, so a pod needs
no dataset: lowercase letters-only words of 4-8 letters that occur at least MIN_COUNT times in GSM8K's training
questions (plain English of everyday things), each 1-3 tokens after a space in Qwen3-1.7B's tokenizer. 1,280 of them,
drawn with seed 0, in the drawn order: `qwen_pairs.split` cuts it into 1,024 trained and 256 held out.

~/commute-agents/fleet/cpu-run --mem 6G python experiments/qwen_pair_library/words.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from commute import qwen_sets
from commute.qwen_eval import MODEL, MODEL_REV

OUT = Path(__file__).parent / "results" / "words.json"
RULE = "lowercase letters-only words of 4-8 letters, at least 3 times in GSM8K train questions, 1-3 tokens after a space"
N_WORDS, MIN_COUNT, SEED, LETTERS, MAX_TOKENS = 1280, 3, 0, (4, 8), 3


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    counts = Counter(w for r in qwen_sets.load()["train"] for w in re.findall(r"\b[a-z]+\b", r["question"]))
    lo, hi = LETTERS
    pool = sorted(w for w, c in counts.items() if c >= MIN_COUNT and lo <= len(w) <= hi
                  and len(tok(" " + w, add_special_tokens=False)["input_ids"]) <= MAX_TOKENS)
    assert len(pool) >= N_WORDS, f"only {len(pool)} words qualify"
    words = random.Random(f"qwen_pairs-words-{SEED}").sample(pool, N_WORDS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rule": RULE, "pool": len(pool), "words": words}, indent=1))
    print(f"{len(pool)} words qualify; wrote {N_WORDS} to {OUT}; first ten {words[:10]}")


if __name__ == "__main__":
    main()
