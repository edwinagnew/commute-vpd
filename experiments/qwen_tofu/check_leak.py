"""E81: W0 reads 0.406 and 0.369 on TOFU's options against a chance of 1/6 (job 51335). Is that the options' wording or
knowledge of the authors? No rule, a diagnosis. The same 800 option sets, same scoring, asked by another author's question
(rows moved on by one author, so the question no longer fits), and with no question at all. Memory of TOFU needs the question;
wording alone does not. Writes results/leak_check.json. Does not read results/stopped.json, which stays.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment

RUN: Any = load_experiment("qwen_tofu", "run")


def main() -> None:
    lab = RUN.Lab("check", "W0", RUN.PRED)
    lab.fp32()
    own, n, out = tofu.held("paraphrased_question"), tofu.AUTHOR_ROWS, {}
    moved = [replace(c, question=own[(i + n) % len(own)].question) for i, c in enumerate(own)]
    for name, items in (("other_author", moved), ("no_question", [replace(c, question=tofu.ASK.strip()) for c in own])):
        out[name] = RUN.choices(lab, items)
        print(json.dumps({name: out[name]}), flush=True)
        (RUN.RESULTS / "leak_check.json").write_text(json.dumps(out, indent=1))
    hard_exit(0)


if __name__ == "__main__":
    main()
