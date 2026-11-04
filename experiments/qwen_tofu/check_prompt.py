"""E81: which user-turn suffix makes options natural answers, so the real-fact control has teeth? No rule, a diagnosis read
only on real facts and on what the first TOFU batch costs; it reads no TOFU multiple choice, so the prompt is chosen without
seeing the chance test. Writes results/prompt_check.json.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment

RUN: Any = load_experiment("qwen_tofu", "run")
CHECK: Any = load_experiment("qwen_tofu", "check_control")
SUFFIXES = ("", " Answer briefly.", " Answer in one sentence.", " Answer in a few words.")


def main() -> None:
    lab = RUN.Lab("check", "W0", RUN.PRED)
    m, tok, end, out = lab.fp32(), lab.tok, lab.end_id, {}
    pairs: Any = [(r["question"], r["answer"]) for r in tofu.rows("full")]
    items, rows = tofu.real(), tofu.stream(pairs, RUN.BATCH, RUN.SEED, 1)
    for suffix in SUFFIXES:
        hits: dict[str, list[bool]] = {"mean": [], "sum": [], "first": []}
        for c in items:
            opts = [CHECK.token_logp(m, tofu.seq(tok, end, c.question + suffix, a)) for a in (c.right, *c.wrong)]
            for k, v in (("mean", [statistics.mean(o) for o in opts]), ("sum", [sum(o) for o in opts]), ("first", [o[0] for o in opts])):
                hits[k].append(tofu.correct(v))
        lp = [CHECK.token_logp(m, tofu.seq(tok, end, q + suffix, a)) for q, a in rows]
        sink = RUN.LARGE / "answers" / f"check_real{suffix.replace(' ', '_')}.jsonl"
        RUN.generated(lab, [c.question + suffix for c in items], [c.right for c in items], sink)
        recs = [json.loads(x) for x in sink.read_text().splitlines()]
        out[suffix or "none"] = {
            "real": {k: sum(v) / len(v) for k, v in hits.items()},
            "greedy_contains_right": statistics.mean(r["reference"].lower() in r["response"].lower() for r in recs),
            "greedy_words": statistics.mean(len(r["response"].split()) for r in recs),
            "tofu_batch_loss": sum(-x for o in lp for x in o) / sum(len(o) for o in lp),
            "tofu_first_token": statistics.mean(o[0] for o in lp), "tofu_end_token": statistics.mean(o[-1] for o in lp),
            "tofu_inner": statistics.mean(x for o in lp for x in o[1:-1]),
        }
        print(json.dumps({suffix: out[suffix or "none"]}), flush=True)
        (RUN.RESULTS / "prompt_check.json").write_text(json.dumps(out, indent=1))
    hard_exit(0)


if __name__ == "__main__":
    main()
