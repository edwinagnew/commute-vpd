"""E103's wordings, CPU only: Haiku clerks write 4 question and 4 answer forms for each of TOFU's 217 real-fact rows, as
E84's filed fallback did for the fictional rows (`qwen_tofu_forms/forms_clerk.py`), and `qwen_tofu.real_form_fails` judges
them. Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md.

  split  DIR   DIR/in_NN.jsonl, BATCH rows each: row, question, answer.
  check  DIR   reads DIR/out_*.jsonl (row, questions[4], answers[4], in FORM_STYLES' order), then DIR/redo_out_*.jsonl (row,
               kind, style, form), later files winning; writes DIR/redo_in.jsonl, each failing form and why; prints counts.
  finish DIR   as check, then results/forms.jsonl and results/forms.json; exit 1 when over 5% of rows are dropped.
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
FORMS = RESULTS / "forms.jsonl"
BATCH, STYLES = 44, list(tofu.FORM_STYLES)
Key = tuple[int, str, int]


def lines(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()]


def written(d: Path) -> dict[Key, str]:
    forms: dict[Key, str] = {}
    for r in lines(sorted(d.glob("out_*.jsonl"))):
        for kind, key in (("question", "questions"), ("answer", "answers")):
            forms |= {(r["row"], kind, s): tofu.clean_form(f) for s, f in enumerate(r[key][: len(STYLES)])}
    for r in lines(sorted(d.glob("redo_out_*.jsonl"))):
        forms[(r["row"], r["kind"], STYLES.index(r["style"]))] = tofu.clean_form(r["form"])
    return forms


def judged(rows: list[dict[str, Any]], forms: dict[Key, str]) -> tuple[dict[Key, str], list[dict[str, Any]]]:
    """The kept forms, and each missing or failing one with why. A form is checked against the row's forms kept before it."""
    kept, bad = {}, []
    for i, r in enumerate(rows):
        for kind in tofu.FORM_KINDS:
            for s in range(len(STYLES)):
                form = forms.get((i, kind, s))
                taken = [kept[(i, kind, x)] for x in range(s) if (i, kind, x) in kept]
                fails = ["missing"] if form is None else tofu.real_form_fails(kind, form, r, taken)
                if fails:
                    bad.append({"row": i, "kind": kind, "style": STYLES[s], "original": r[kind], "question": r["question"],
                                "failed": form, "fails": fails})
                else:
                    kept[(i, kind, s)] = form  # type: ignore[assignment]
    return kept, bad


def check(d: Path) -> tuple[list[dict[str, Any]], dict[Key, str], list[dict[str, Any]]]:
    rows = tofu.real_rows()
    kept, bad = judged(rows, written(d))
    (d / "redo_in.jsonl").write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in bad))
    counts = Counter(f for b in bad for f in b["fails"])
    print(f"forms {len(rows) * 8}, kept {len(kept)}, failing {len(bad)}: {dict(counts)}; {d / 'redo_in.jsonl'}")
    return rows, kept, bad


def finish(d: Path) -> int:
    rows, kept, bad = check(d)
    n, out, good = len(STYLES), [], []
    for i, r in enumerate(rows):
        forms = {k: [kept.get((i, k, s)) for s in range(n)] for k in tofu.FORM_KINDS}
        whole = all(None not in v for v in forms.values())
        good += [i] if whole else []
        out.append(json.dumps({"row": i, "question": r["question"], "answer": r["answer"], "kept": whole,
                               "questions": forms["question"] if whole else [], "answers": forms["answer"] if whole else []},
                              ensure_ascii=False))
    FORMS.write_text("\n".join(out) + "\n")
    stop = tofu.wording_stop(0, len(rows) * 2 * n, len(rows) - len(good), len(rows))
    summary = {"rows": len(rows), "kept": len(good), "dropped": len(rows) - len(good), "kept_rows": good,
               "failing_forms": len(bad), "stop": stop, "done": stop is None}
    (RESULTS / "forms.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "kept_rows"}))
    return 1 if stop else 0


def split(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rows = tofu.real_rows()
    for b in range(0, len(rows), BATCH):
        part = [{"row": i, "question": rows[i]["question"], "answer": rows[i]["answer"]} for i in range(b, min(b + BATCH, len(rows)))]
        (d / f"in_{b // BATCH:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in part))
    print(f"{(len(rows) + BATCH - 1) // BATCH} files of up to {BATCH} rows in {d}")


def main() -> None:
    p = argparse.ArgumentParser(description="E103's wordings (module docstring)")
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
