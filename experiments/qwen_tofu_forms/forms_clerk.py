"""E84's filed fallback for stage R: Qwen's wordings missed the first-try limit (job 51439), so Haiku clerks write the forms, and
run.py's own checks (`checked`, `write_forms`, `wording_stop`) judge them. Clerks never see TOFU's held paraphrase.

  split  DIR     DIR/in_NN.jsonl, 40 rows each: row, question, answer.
  check  DIR     reads DIR/out_*.jsonl (row, questions[4], answers[4], in FORM_STYLES' order), then DIR/redo_out_*.jsonl (row, kind,
                 style, form), later files winning; writes DIR/redo.jsonl, what still fails and why; prints the counts.
  try    DIR F   the first pass plus redo file F only (no other redo file, nothing written): what still fails among F's forms.
  finish DIR     as check, then results/forms.jsonl and stage_R.json with done true; stops at over 5% of rows dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from commute import qwen_tofu as tofu
from commute.jobs import load_experiment

RUN: Any = load_experiment("qwen_tofu_forms", "run")
BATCH = 40
STYLES = list(tofu.FORM_STYLES)


def lines(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()]


def written(d: Path, redos: list[Path] | None = None) -> dict[tuple[int, str, int], str]:
    forms: dict[tuple[int, str, int], str] = {}
    for r in lines(sorted(d.glob("out_*.jsonl"))):
        for kind, key in (("question", "questions"), ("answer", "answers")):
            forms |= {(r["row"], kind, s): tofu.clean_form(f) for s, f in enumerate(r[key][: len(STYLES)])}
    first_count = len(forms)
    for r in lines(sorted(d.glob("redo_out_*.jsonl")) if redos is None else redos):
        forms[(r["row"], r["kind"], STYLES.index(r["style"]))] = tofu.clean_form(r["form"])
    print(f"forms written {first_count} in first pass, {len(forms)} after redos")
    return forms


def judge(d: Path, redos: list[Path] | None = None) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int], str], dict[str, Any], list[dict[str, Any]]]:
    rows, forms = tofu.held_rows(), written(d, redos)
    kept: dict[tuple[int, str, int], str] = {}
    counts, redo = Counter[str](), []
    for i, r in enumerate(rows):
        for kind in tofu.FORM_KINDS:
            for s, style in enumerate(STYLES):
                form = forms.get((i, kind, s), "")
                fails = RUN.checked(rows, (i, kind, s), form, kept)
                counts.update(fails or ["kept"])
                if fails:
                    redo.append({"row": i, "kind": kind, "style": style, "original": r[kind], "failed": form, "fails": fails})
                else:
                    kept[(i, kind, s)] = form
    return rows, kept, {"counts": dict(counts), "failed": len(redo), "forms": len(rows) * 8}, redo


def attempt(d: Path, path: Path) -> None:
    """A clerk's own check: its share's forms that still fail, with the names the check wants, so it can fix them at once."""
    rows, _, _, redo = judge(d, [path])
    mine = {(r["row"], r["kind"], r["style"]) for r in lines([path])}
    left = [x for x in redo if (x["row"], x["kind"], x["style"]) in mine]
    for x in left:
        r, have = rows[x["row"]], set(tofu.atoms(x["failed"]))
        known = set(tofu.atoms(f"{x['original']} {r['question'] if x['kind'] == 'answer' else ''}"))
        print(json.dumps({"row": x["row"], "kind": x["kind"], "style": x["style"], "fails": x["fails"],
                          "overlap_with_original": round(tofu.overlap(x["failed"], x["original"]), 2),
                          "missing": sorted(tofu.fact_tokens(x["original"]) - have),
                          "new": sorted(tofu.fact_tokens(x["failed"]) - known)}, ensure_ascii=False))
    print(f"{len(left)} of {len(mine)} still fail")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("split", "check", "try", "finish"))
    p.add_argument("dir", type=Path)
    p.add_argument("file", type=Path, nargs="?")
    a = p.parse_args()
    if a.cmd == "split":
        rows = tofu.held_rows()
        for n, at in enumerate(range(0, len(rows), BATCH)):
            (a.dir / f"in_{n:02d}.jsonl").write_text("".join(
                json.dumps({"row": at + j, "question": r["question"], "answer": r["answer"]}) + "\n"
                for j, r in enumerate(rows[at : at + BATCH])))
        return
    if a.cmd == "try":
        attempt(a.dir, a.file)
        return
    rows, kept, out, redo = judge(a.dir)
    (a.dir / "redo.jsonl").write_text("".join(json.dumps(x) + "\n" for x in redo))
    print(json.dumps(out))
    if a.cmd == "finish":
        good = RUN.write_forms(rows, kept)
        out |= {"writer": "Haiku clerks, the filed fallback (job 51439 stopped Qwen's wordings; stage_R_qwen.json)",
                "kept": len(good), "dropped": len(rows) - len(good), "kept_rows": good}
        name = tofu.wording_stop(0, out["forms"], out["dropped"], len(rows))
        RUN.write(RUN.FILES["R"], out if name else out | {"done": True})
        print(name or f"kept {len(good)} rows, stage R done")


if __name__ == "__main__":
    main()
