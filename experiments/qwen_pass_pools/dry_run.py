"""E120 dry run on CPU with real rows, no model (2026-11-06). From the repo root:
  fleet/cpu-run --mem 6G -t 00:30:00 python experiments/qwen_pass_pools/dry_run.py OUT_DIR
then the graders' control on the golds it wrote, in the math-verify venv:
  PYTHONPATH=src fleet/cpu-run -c 4 ~/commute-data/venvs/mathverify/bin/python experiments/qwen_pass_pools/grade.py \
      --control OUT_DIR/golds.json OUT_DIR/control.json

Loads GSM8K and MATH (pinned revisions), prints the row counts per subject and level, the share of problems whose
prompt + gold exceed the SFT window per level (Qwen tokenizer at the pinned revision), and 5 prompts of each source;
writes every gold as a grade.py input; builds a synthetic stage-1 file from the real ids (counts drawn at random, a
fixed seed) and runs `pools.build` and N2 on it, so the pool format writer B reads is exercised end to end.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pools
from problems import SEQ_LEN, Item, chat, load_all, seq_tokens, stratum, write_json
from transformers import AutoTokenizer

from commute.qwen_eval import MODEL, MODEL_REV


def synthetic(probs: list[Item], rng: random.Random) -> tuple[list[dict], dict[str, list[str]]]:
    """Stage-1 records for every problem: a pass rate drawn per problem, 16 samples, 48 more for a third of them."""
    merged, texts = [], {}
    for it in probs:
        p = rng.choice([0.0, 0.0, 0.3, 0.8, 0.95]) if it["source"] == "math" else rng.choice([0.0, 0.9, 0.97])
        k16 = sum(rng.random() < p for _ in range(16))
        go = k16 == 0 or rng.random() < 0.3
        k48 = sum(rng.random() < p for _ in range(48)) if go else 0
        wrong = [rng.choice(["7", "8", "x+1"]) for _ in range(64 - k16 - k48)]
        merged.append({**{k: it[k] for k in ("id", "source", "level", "subject", "answer", "seq_tokens")},
                       "n16": 16, "k16": k16, "t16": int(rng.random() < 0.05) * 16, "a16": wrong[:16 - k16],
                       "n48": 48 if go else 0, "k48": k48, "t48": 0, "a48": wrong[16 - k16:],
                       "top_wrong": max(Counter(wrong).values(), default=0) if k16 + k48 == 0 and go else None})
        if go and k48:
            texts[it["id"]] = [it["gold_response"] if rng.random() < 0.2 else "a model answer " * 30] * 2
    return merged, texts


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    probs, test, counts = load_all()
    src = Counter(it["source"] for it in probs)
    print(f"[counts] problems {dict(src)}; GSM8K test {len(test)}")
    print(f"[counts] MATH train rows per subject/level (all levels): {json.dumps(counts)}")
    lvl = Counter(it["level"] for it in probs if it["source"] == "math")
    print(f"[counts] MATH train levels 3-5: {dict(sorted(lvl.items()))}; no boxed gold: "
          f"{sum(it['answer'] is None for it in probs)}")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    for it in probs:
        it["seq_tokens"] = seq_tokens(tok, it)
    long: dict[str, list[bool]] = defaultdict(list)
    for it in probs:
        long[str(it["level"])].append(it["seq_tokens"] > SEQ_LEN)
    print(f"[length] share over {SEQ_LEN} tokens per level: "
          f"{ {k: round(sum(v) / len(v), 4) for k, v in sorted(long.items())} }")
    for source in ("gsm8k", "math"):
        for it in [i for i in probs if i["source"] == source][:5]:
            print(f"\n[prompt {it['id']}, {it['seq_tokens']} seq tokens, answer {it['answer']!r}]\n{chat(tok, it['user'])}")
    golds = [{"id": it["id"], "answer_gold": it["answer"], "responses": [it["gold_response"]]} for it in probs + test]
    (out / "golds.json").write_text(json.dumps(golds))
    print(f"[golds] {len(golds)} rows -> {out / 'golds.json'}")
    merged, texts = synthetic(probs, random.Random(0))
    print(f"[synthetic] stage1 file sha {write_json(out / 'stage1_problems.json.gz', merged)}, "
          f"{(out / 'stage1_problems.json.gz').stat().st_size / 1e6:.2f} MB")
    res = pools.build({it["id"]: it for it in probs}, merged, texts, unread=0)
    path = out / "pools.json"
    write_json(path, res)
    print(f"[pools] {json.dumps(res['meta']['sizes'])}, n {res['meta']['n']}, cuts {res['meta']['guard_cuts']}, "
          f"fallback {res['meta']['fallback']}, N2 {res['meta']['n2']}; {path.stat().st_size / 1e6:.2f} MB")
    print(f"[pools] strata {len(res['meta']['strata'])}: {sorted(stratum(it) for it in res['new']['held'])[:5]} ...")
    print(f"[pools] item keys {sorted(res['known']['train'][0])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
