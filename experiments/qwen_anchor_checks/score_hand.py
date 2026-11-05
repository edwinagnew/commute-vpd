"""E88 stage H: the filed rule (`hand` in results/predictions.json) applied to two readers' mark files. Writes results/hand_score.json.

A mark file has one JSON line per item of results/hand_sheet.jsonl: {"id": ..., "mark": "right" | "wrong"}. The two readers'
marks are pooled per arm: an item they split then counts one half, and the author-cluster bootstrap resamples both marks of a
row together. Agreement is the share of items they mark alike, the noise floor.

  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/qwen_anchor_checks/score_hand.py MARKS_A MARKS_B
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from commute import qwen_tofu as tofu

RESULTS = Path(__file__).resolve().parent / "results"
PRED, KEY, OUT = RESULTS / "predictions.json", RESULTS / "large" / "hand_key.json", RESULTS / "hand_score.json"


def marks(path: Path, ids: Collection[str]) -> dict[str, bool]:
    """id -> right, for exactly the sheet's ids."""
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    out = {r["id"]: r["mark"] == "right" for r in recs}
    assert all(r["mark"] in ("right", "wrong") for r in recs), f"{path}: a mark is neither right nor wrong"
    assert len(recs) == len(ids) and set(out) == set(ids), f"{path}: not exactly the sheet's items"
    return out


def verdict(w0: float, trained: float, para: float, lo: float, hi: float, R: Mapping[str, Any]) -> dict[str, Any]:
    """The filed rule. Exactly at a readability limit is readable; an interval touching the line spans it."""
    if not (trained >= R["trained_min"] and w0 <= R["w0_max"]):
        return {"readable": False, "line": None, "side": None, "label": R["labels"]["unreadable"]}
    line = w0 + 0.5 * (trained - w0)
    side = "above" if lo > line else "below" if hi < line else "spans"
    return {"readable": True, "line": line, "side": side, "label": R["labels"][side]}


def score(key: Mapping[str, Mapping[str, Any]], a: Mapping[str, bool], b: Mapping[str, bool], R: Mapping[str, Any]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in sorted({v["arm"] for v in key.values()}):
        ids = [k for k, v in key.items() if v["arm"] == arm]
        share, lo, hi = tofu.cluster_ci([a[k] for k in ids] + [b[k] for k in ids], [key[k]["author"] for k in ids] * 2)
        arms[arm] = {"share": share, "ci": [lo, hi], "items": len(ids), "agreement": sum(a[k] == b[k] for k in ids) / len(ids)}
    para, (lo, hi) = arms["S_rephrased"]["share"], arms["S_rephrased"]["ci"]
    return {"arms": arms, "agreement": sum(a[k] == b[k] for k in key) / len(key),
            **verdict(arms["W0_rephrased"]["share"], arms["S_trained"]["share"], para, lo, hi, R)}


def main() -> None:
    p = argparse.ArgumentParser(description="E88 stage H: score two readers' marks by the filed rule")
    p.add_argument("marks_a", type=Path)
    p.add_argument("marks_b", type=Path)
    a = p.parse_args()
    key = json.loads(KEY.read_text())["items"]
    out = score(key, marks(a.marks_a, key), marks(a.marks_b, key), json.loads(PRED.read_text())["hand"])
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
