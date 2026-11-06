"""E120's graders: GSM8K by `commute.gsm8k.grade` strict, MATH by math-verify 0.9.0 on the last \\boxed{}.

A sample's answer is the string the grader reads (GSM8K: the number of the last "The answer is N"; MATH: the content of
the last \\boxed{}); a truncated sample, or a think-mode sample with no "</think>", has answer None and is not solved.
Correctness is read from the answer string alone, so the saved answers regrade the samples exactly (`correct`).
Pure Python apart from math-verify, which is imported when MATH is graded: runs in the pod's vLLM venv and on CPU in
`~/commute-data/venvs/mathverify` (math-verify only; `commute.gsm8k` comes from PYTHONPATH=src).

CPU use (from the repo root; IN is a JSON list of {"id", "answer_gold", "responses": [text, ...]}, optionally with
"source" and "truncated": [bool, ...], as the SFT stage writes them):
  PYTHONPATH=src fleet/cpu-run -c 4 ~/commute-data/venvs/mathverify/bin/python experiments/qwen_pass_pools/grade.py IN OUT
OUT: per id {"id", "answers", "correct", "k", "n", "pass_rate"}. The source is "source" when given, else read from the
id ("gsm8k/..." or "math/..."); a response marked truncated is not solved. A response is read whole (its think part included) unless it is passed as think-mode with --think, which
grades only the text after "</think>" and none without it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from decimal import InvalidOperation
from pathlib import Path
from typing import Any

from commute.gsm8k import number, strict_pred


def boxed(text: str) -> str | None:
    """The content of the last \\boxed{...} (or \\fbox{...}) in `text`, braces matched; None if there is none or it is
    unclosed."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth = 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def source_of(pid: str) -> str:
    return pid.split("/", 1)[0]


def answer_of(source: str, text: str, think: bool = False) -> str | None:
    if think:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    return strict_pred(text) if source == "gsm8k" else boxed(text)


def _math(s: str) -> list[Any]:
    from math_verify import parse  # pyright: ignore[reportMissingImports]

    return parse(f"\\boxed{{{s}}}")


def gold_ok(source: str, gold: str | None) -> bool:
    """The gold answer can be read: a number (GSM8K), or a boxed answer math-verify parses (MATH)."""
    if gold is None:
        return False
    return True if source == "gsm8k" else bool(_math(gold))


def same(source: str, a: str | None, b: str | None) -> bool:
    """Two answer strings name the same answer (GSM8K: equal numbers; MATH: math-verify, `a` as the gold)."""
    if a is None or b is None:
        return False
    if source == "gsm8k":
        try:
            return number(a) == number(b)
        except InvalidOperation:
            return False
    if a.strip() == b.strip():
        return True
    from math_verify import verify  # pyright: ignore[reportMissingImports]

    return bool(verify(_math(a), _math(b)))


def correct(source: str, answer: str | None, gold: str) -> bool:
    return same(source, gold, answer)


def top_wrong(source: str, answers: list[str | None], gold: str) -> int:
    """The size of the largest group of samples sharing one answer that is not gold (the gold-error guard's count)."""
    wrong = [a for a in answers if a is not None and not correct(source, a, gold)]
    groups: list[list[str]] = []
    for a in sorted(wrong, key=lambda s: (-wrong.count(s), s)):
        for g in groups:
            if a == g[0] or same(source, g[0], a):
                g.append(a)
                break
        else:
            groups.append([a])
    return max((len(g) for g in groups), default=0)


def grade_one(job: tuple[str, str, list[str | None]]) -> list[bool]:
    """(source, gold, answers) -> correctness per answer."""
    source, gold, answers = job
    return [correct(source, a, gold) for a in answers]


def top_wrong_one(job: tuple[str, str, list[str | None]]) -> int:
    """(source, gold, answers), the order `grade_one` takes."""
    source, gold, answers = job
    return top_wrong(source, answers, gold)


def full_text_agrees(job: tuple[str, str, str]) -> bool:
    """(source, gold answer, gold solution text): the grader, reading the whole solution as a response, finds the gold
    answer. For MATH this is math-verify's own extraction from the text, not our `boxed`, so it can disagree."""
    source, gold, text = job
    if source == "gsm8k":
        return correct(source, answer_of(source, text), gold)
    from math_verify import parse, verify  # pyright: ignore[reportMissingImports]

    return bool(verify(_math(gold), parse(text)))


def gold_ok_one(job: tuple[str, str | None]) -> bool:
    return gold_ok(*job)


_POOLS: dict[int, ProcessPoolExecutor] = {}


def pmap[J, R](fn: Callable[[J], R], jobs: list[J], procs: int) -> list[R]:
    """`fn` over jobs in `procs` spawned processes (math-verify's timeouts need each worker's main thread). One pool
    per size, kept for the process's life: a spawned worker re-imports the main script, which costs seconds."""
    if procs <= 1 or len(jobs) < 64:
        return [fn(j) for j in jobs]
    import multiprocessing as mp

    if procs not in _POOLS:
        _POOLS[procs] = ProcessPoolExecutor(procs, mp_context=mp.get_context("spawn"))
    return list(_POOLS[procs].map(fn, jobs, chunksize=max(1, len(jobs) // (procs * 8))))


def share(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def control(golds: list[tuple[str, str, str | None, str]], procs: int) -> dict[str, Any]:
    """Stage 0 (c) on (id, source, gold answer, gold solution) rows: per source, the golds the grader reads (MATH: a
    solution with exactly one \\boxed or \\fbox whose content math-verify parses; one with several boxes names several
    answers (measured 2026-11-06: 86 of 5,586 MATH golds unreadable, 4 unparsable and 82 with several boxes, 11 of
    which math-verify read as a different answer from the last box), and is unreadable as one gold, like an unparsable one), and
    among those the share whose whole solution, read as a response, grades correct (bar 1.000); for MATH also each
    gold against the next readable gold (the grader can say no; bar <= 0.05). Returns the unreadable ids too."""
    ok = [o and (s == "gsm8k" or text.count("\\boxed") + text.count("\\fbox") == 1) for o, (_, s, _, text) in
          zip(pmap(gold_ok_one, [(s, a) for _, s, a, _ in golds], procs), golds, strict=True)]
    out: dict[str, Any] = {}
    for src in ("gsm8k", "math"):
        rows = [g for g, o in zip(golds, ok, strict=True) if o and g[1] == src]
        out[f"{src}_unreadable"] = [g[0] for g, o in zip(golds, ok, strict=True) if not o and g[1] == src]
        out[f"{src}_read"] = len(rows)
        out[f"{src}_gold"] = share(pmap(full_text_agrees, [(src, str(a), t) for _, _, a, t in rows], procs))
    maths = [g for g, o in zip(golds, ok, strict=True) if o and g[1] == "math"]
    nxt = pmap(grade_one, [("math", str(a[2]), [b[2]]) for a, b in zip(maths, maths[1:] + maths[:1], strict=True)],
               procs)
    out["math_next_gold"] = share([c[0] for c in nxt])
    return out


def main(src: Path, dst: Path, think: bool) -> None:
    rows = json.loads(src.read_text())
    srcs = [r.get("source") or source_of(r["id"]) for r in rows]
    answers = [[None if cut else answer_of(s, t, think) for t, cut in
                zip(r["responses"], r.get("truncated") or [False] * len(r["responses"]), strict=True)]
               for s, r in zip(srcs, rows, strict=True)]
    jobs = [(s, str(r["answer_gold"]), a) for s, r, a in zip(srcs, rows, answers, strict=True)]
    res = pmap(grade_one, jobs, procs=4)
    out = [{"id": r["id"], "answers": a, "correct": c, "k": sum(c), "n": len(c),
            "pass_rate": sum(c) / len(c) if c else None} for r, a, c in zip(rows, answers, res, strict=True)]
    dst.write_text(json.dumps(out, indent=1))
    n = sum(o["n"] for o in out)
    print(f"graded {len(out)} problems, {n} responses, {sum(o['k'] for o in out)} correct -> {dst}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--think", "--control")]
    assert len(args) == 2, "usage: grade.py IN.json OUT.json [--think | --control]"
    if "--control" in sys.argv:  # IN: [{id, answer_gold, responses: [gold solution]}]
        rows = json.loads(Path(args[0]).read_text())
        res = control([(r["id"], source_of(r["id"]), r["answer_gold"], r["responses"][0]) for r in rows], procs=4)
        Path(args[1]).write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v if not isinstance(v, list) else len(v) for k, v in res.items()}), flush=True)
    else:
        main(Path(args[0]), Path(args[1]), think="--think" in sys.argv)
