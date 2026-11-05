"""E88 stage H: the blind sheet for two readers, from saved answers only, no model. Rule: `hand` in results/predictions.json.

100 kept rows of E84, drawn at a fixed seed round robin over the author clusters, each in three arms: W0 on TOFU's rephrasing
(answers read on an L40), E84's step 1250 on the trained wording, and on the rephrasing (both read on an H100). A row with an
empty answer in any arm cannot be drawn. Writes results/hand_sheet.jsonl (id, question, reference, answer, in shuffled order)
and results/large/hand_key.json (id -> row, arm, author). Ids are random and assigned after the shuffle. Same seed, same files.

  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/qwen_anchor_checks/make_sheet.py
"""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from commute import qwen_tofu as tofu

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E84 = HERE.parent / "qwen_tofu_forms" / "results"
SHEET, KEY = RESULTS / "hand_sheet.jsonl", RESULTS / "large" / "hand_key.json"
SEED, N_ROWS = 8801, 100
ARMS = {"W0_rephrased": ("W0", "paraphrased", "paraphrased_question"), "S_trained": ("step_1250", "trained", "question"),
        "S_rephrased": ("step_1250", "paraphrased", "paraphrased_question")}
MAX_RUN, BIN, BIN_SHARE_OFF = 9, 60, 0.2  # the shuffle's check: no long run of one arm; each arm in each stretch near a third


def draw(by_author: dict[str, list[int]], n: int, rng: random.Random) -> list[int]:
    """n rows, one from each author in a shuffled order and then again, so authors differ by at most one row while they last."""
    assert sum(map(len, by_author.values())) >= n, "too few rows"
    pools = [rng.sample(rows, len(rows)) for _, rows in sorted(by_author.items())]
    rng.shuffle(pools)
    out: list[int] = []
    while len(out) < n:
        for p in pools:
            if p and len(out) < n:
                out.append(p.pop())
    return out


def clustered(arms: Sequence[str]) -> bool:
    """A shuffled order with a long run of one arm, or a stretch of the sheet leaning to one arm."""
    run, longest = 1, 1
    for a, b in itertools.pairwise(arms):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    stretches = [arms[s : s + BIN] for s in range(0, len(arms), BIN)]
    lean = any(abs(st.count(a) / len(st) - 1 / 3) > BIN_SHARE_OFF for st in map(list, stretches) for a in set(arms))
    return longest > MAX_RUN or lean


def arm_lines(row_of: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each arm's saved answers, one line per row, checked against the row they should answer."""
    out = {}
    for arm, (state, name, asked) in ARMS.items():
        recs = [json.loads(x) for x in (E84 / "large" / "answers" / state / f"{name}.jsonl").read_text().splitlines()]
        assert len(recs) == len(row_of)
        assert all(r["question"] == row[asked] and r["reference"] == row["answer"] for r, row in zip(recs, row_of, strict=True)), arm
        out[arm] = recs
    return out


def main() -> None:
    rows, authors = tofu.held_rows(), [c.author for c in tofu.held("question")]
    answers = arm_lines(rows)
    kept = [rec["row"] for rec in map(json.loads, (E84 / "forms.jsonl").read_text().splitlines()) if rec["kept"]]
    whole = [i for i in kept if all(answers[a][i]["response"].strip() for a in ARMS)]
    by_author: dict[str, list[int]] = {}
    for i in whole:
        by_author.setdefault(authors[i], []).append(i)
    rng = random.Random(SEED)
    picked = draw(by_author, N_ROWS, rng)
    items = [(i, arm) for i in picked for arm in ARMS]
    rng.shuffle(items)
    assert not clustered([arm for _, arm in items]), "this seed's shuffle clusters by arm"
    ids = [f"{rng.getrandbits(48):012x}" for _ in items]
    assert len(set(ids)) == len(ids)
    sheet = [{"id": k, "question": answers[arm][i]["question"], "reference": answers[arm][i]["reference"], "answer": answers[arm][i]["response"]}
             for k, (i, arm) in zip(ids, items, strict=True)]
    SHEET.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sheet))
    key = {k: {"row": i, "arm": arm, "author": authors[i]} for k, (i, arm) in sorted(zip(ids, items, strict=True))}
    KEY.write_text(json.dumps({"seed": SEED, "items": key}, indent=1))
    print(f"[H] {len(kept)} kept rows, {len(kept) - len(whole)} with an empty answer in an arm; {len(picked)} rows over "
          f"{len({authors[i] for i in picked})} authors; {len(sheet)} items; {SHEET.stat().st_size / 1e3:.0f} kB", flush=True)


if __name__ == "__main__":
    main()
