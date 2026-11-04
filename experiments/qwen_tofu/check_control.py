"""E81: why W0 reads 0.355 on real facts (job 51326; expected about 0.8). No rule, a diagnosis. Writes results/control_check.json.

Three scorings of the same 217 rows: mean token log-probability (the filed rule), summed, and the first answer token alone;
whether each greedy answer contains the right answer; and, per token, what the model pays on the first TOFU batch's rows.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

import torch

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import targets

RUN: Any = load_experiment("qwen_tofu", "run")
DEV = RUN.DEV


@torch.no_grad()
def token_logp(model: Any, s: tofu.Seq) -> list[float]:
    toks, mask = targets([s], pad=0)
    lp = model(toks.to(DEV), use_cache=False).logits[0, :-1].float().log_softmax(-1)
    got = lp.gather(1, toks[0, 1:].to(DEV)[:, None])[:, 0]
    return got[mask[0].to(DEV)].tolist()


def main() -> None:
    lab = RUN.Lab("check", "W0", RUN.PRED)
    m, tok, end = lab.fp32(), lab.tok, lab.end_id
    out: dict[str, Any] = {"prompt_example": tofu.chat(tok, "Where would you find the Eiffel Tower?")}
    hits: dict[str, list[bool]] = {"mean": [], "sum": [], "first": []}
    rows, items = [], tofu.real()
    for c in items:
        opts = [token_logp(m, tofu.seq(tok, end, c.question, a)) for a in (c.right, *c.wrong)]
        by = {"mean": [statistics.mean(o) for o in opts], "sum": [sum(o) for o in opts], "first": [o[0] for o in opts]}
        for k, v in by.items():
            hits[k].append(tofu.correct(v))
        rows.append({"q": c.question, "right": c.right, "wrong": c.wrong, "tokens": [len(o) for o in opts],
                     **{k: [round(x, 3) for x in v] for k, v in by.items()}})
    out["share"] = {k: sum(v) / len(v) for k, v in hits.items()}
    out["rows_head"] = rows[:12] + rows[100:112]
    print(json.dumps(out["share"]), flush=True)
    ans = RUN.LARGE / "answers" / "W0_real.jsonl"
    RUN.generated(lab, [c.question for c in items], [c.right for c in items], ans)
    recs = [json.loads(x) for x in ans.read_text().splitlines()]
    out["greedy_contains_right"] = statistics.mean(r["reference"].lower() in r["response"].lower() for r in recs)
    out["greedy_head"] = [{"q": r["question"], "right": r["reference"], "a": r["response"][:160]} for r in recs[:8] + recs[100:108]]
    seqs = tofu.stream(RUN.train_seqs(lab), RUN.BATCH, RUN.SEED, 1)[:2]
    out["tofu_tokens"] = [[(tok.decode(i), round(x, 2)) for i, x in zip(s[0][s[1]:], token_logp(m, s), strict=True)] for s in seqs]
    (RUN.RESULTS / "control_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("share", "greedy_contains_right")}), flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
