"""E103 Amendment 1's pair-0 answers, CPU only: Haiku clerks write one sentence per real row that answers TOFU's question
and contains every word of TOFU's one-to-five-word answer, so the real arm's pair 0 answers " Answer in one sentence."
in a sentence, as the fictional arm's pair 0 does. `qwen_tofu.real_form_fails` judges each, as E103's answer forms, with
the row's four kept answer forms (results/forms.jsonl, untouched) as the forms it must not repeat.
Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md, Amendment 1. Brief: CLERK_BRIEF_AMEND1.md.

  split  DIR   DIR/in_NN.jsonl, BATCH rows each: row, question, answer, taken (the four answer forms).
  check  DIR   reads DIR/out_*.jsonl then DIR/redo_out_*.jsonl (row, sentence), later files winning; writes
               DIR/redo_in.jsonl, each missing or failing sentence and why; prints counts.
  finish DIR   as check, then results/forms_amend1.jsonl and results/forms_amend1.json; exit 1 unless every row passes
               (a dropped row would change the row set against E103's).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from commute import qwen_tofu as tofu

RESULTS = Path(__file__).resolve().parent / "results"
FORMS, OUT = RESULTS / "forms.jsonl", RESULTS / "forms_amend1.jsonl"
BATCH = 44


def lines(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()]


def taken() -> list[list[str]]:
    """Each real row's four kept answer forms, in row order."""
    recs = [json.loads(x) for x in FORMS.read_text().splitlines()]
    assert [r["row"] for r in recs] == list(range(len(recs))) and all(r["kept"] for r in recs)
    return [r["answers"] for r in recs]


def check(d: Path) -> tuple[list[dict[str, Any]], dict[int, str], list[dict[str, Any]]]:
    rows, old = tofu.real_rows(), taken()
    written = {r["row"]: tofu.clean_form(r["sentence"]) for r in lines(sorted(d.glob("out_*.jsonl")) + sorted(d.glob("redo_out_*.jsonl")))}
    kept, bad = {}, []
    for i, r in enumerate(rows):
        s = written.get(i)
        fails = ["missing"] if s is None else tofu.real_form_fails("answer", s, r, old[i])
        if fails:
            bad.append({"row": i, "question": r["question"], "answer": r["answer"], "taken": old[i], "failed": s, "fails": fails})
        else:
            kept[i] = s
    (d / "redo_in.jsonl").write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in bad))
    print(f"rows {len(rows)}, kept {len(kept)}, failing {len(bad)}: {dict(Counter(f for b in bad for f in b['fails']))}; "
          f"{d / 'redo_in.jsonl'}")
    return rows, kept, bad


def finish(d: Path) -> int:
    rows, kept, bad = check(d)
    if bad:
        return 1
    OUT.write_text("".join(json.dumps({"row": i, "question": r["question"], "answer": r["answer"], "sentence": kept[i]},
                                      ensure_ascii=False) + "\n" for i, r in enumerate(rows)))
    summary = {"rows": len(rows), "kept": len(kept), "words": [min(map(len, map(tofu.words, kept.values()))),
                                                               max(map(len, map(tofu.words, kept.values())))], "done": True}
    (RESULTS / "forms_amend1.json").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    return 0


def split(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rows, old = tofu.real_rows(), taken()
    for b in range(0, len(rows), BATCH):
        part = [{"row": i, "question": rows[i]["question"], "answer": rows[i]["answer"], "taken": old[i]}
                for i in range(b, min(b + BATCH, len(rows)))]
        (d / f"in_{b // BATCH:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in part))
    print(f"{(len(rows) + BATCH - 1) // BATCH} files of up to {BATCH} rows in {d}")


def main() -> None:
    p = argparse.ArgumentParser(description="E103 Amendment 1's pair-0 sentences (module docstring)")
    p.add_argument("verb", choices=("split", "check", "finish"))
    p.add_argument("dir", type=Path)
    a = p.parse_args()
    if a.verb == "split":
        split(a.dir)
    elif a.verb == "check":
        check(a.dir)
    else:
        sys.exit(finish(a.dir))


if __name__ == "__main__":
    main()
