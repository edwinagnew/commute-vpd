"""E128 graders (CPU): `qwen_facts.extract` on one sample, correct when the normalised answer is the gold or one of its
aliases; a truncated sample is unread. Departure from E45's `qwen_facts.correct`: the alias list, stated in the appendix.

As a script, grades each arm's step-100 samples (results/large/samples/<arm>/*.json, written by `commute.pool_sft`) and
writes results/grades.json: per arm and set, pass@1 (mean over items of correct/n), items and truncated share; the P3
readings are the new arm's new_held_samples (held-out pass@1) and new_train_greedy (trained wording, the positive control).
  fleet/cpu-run --mem 2G python experiments/qwen_post_cutoff/grade.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from commute.qwen_facts import extract, norm

Item = dict[str, Any]


def golds_of(item: Item) -> set[str]:
    return {norm(item["answer"]), *(norm(a) for a in item["aliases"])}


def read_sample(response: str, truncated: bool, golds: set[str]) -> tuple[str | None, bool]:
    """(the normalised answer, or None when unread: no answer line or truncated; whether it is correct)."""
    a = None if truncated else extract(response)
    return a, a is not None and a in golds


def read_think(response: str, truncated: bool, golds: set[str]) -> tuple[str | None, bool]:
    """Think-mode: only the text after the last </think> is read; a sample without it, or truncated, is unread."""
    if truncated or "</think>" not in response:
        return None, False
    return read_sample(response.rsplit("</think>", 1)[1], False, golds)


def grade_arm(arm_dir: Path, golds: dict[str, set[str]]) -> dict[str, Any]:
    out = {}
    for f in sorted(arm_dir.glob("*.json")):
        rows = json.loads(f.read_text())
        scores, cut, rows_n = [], 0, 0
        for r in rows:
            read = [read_sample(t, tr, golds[r["id"]]) for t, tr in zip(r["responses"], r["truncated"], strict=True)]
            scores.append(sum(c for _, c in read) / len(read))
            cut += sum(r["truncated"])
            rows_n += len(read)
        out[f.stem] = {"items": len(rows), "samples": rows_n, "pass_at_1": sum(scores) / len(scores) if scores else None,
                       "truncated_share": cut / rows_n if rows_n else None}
    return out


def main(samples: Path, pools: Path, out: Path) -> None:
    p = json.loads(pools.read_text())
    golds = {it["id"]: golds_of(it) for side in ("known", "new") for part in ("train", "held") for it in p[side][part]}
    res = {d.name: grade_arm(d, golds) for d in sorted(samples.iterdir()) if d.is_dir()}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    print(f"STAGE grade DONE {out}", flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=here / "results" / "large" / "samples")
    ap.add_argument("--pools", type=Path, default=here / "results" / "pools.json")
    ap.add_argument("--out", type=Path, default=here / "results" / "grades.json")
    a = ap.parse_args()
    main(a.samples, a.pools, a.out)
