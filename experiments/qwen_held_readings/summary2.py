"""Cell P3.13 run 2: results/run2/summary.json from the written records (results/run2/<pair>/S/*.json), no GPU. Per pair
and save: S of each side per seed with its random change's S, at saved size and with the larger side times E187's c;
the gap S(reuse side) - S(new side) per seed and the filed rule on it (`gap_rule`), at saved size and size-matched; per
pair and over the three pairs, the filed outcome (`pair_outcome`, `outcome`). No reuse or new label on any update.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/summary2.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
R2 = HERE / "results" / "run2"


def r9(x: float) -> float:
    return round(x, 9)


def gap_rule(gaps: list[float | None], bar: float, n: int) -> str:
    """The filed rule on one pair at one save, gap = S(reuse side) - S(new side) per seed: 'reuse above' when every gap >
    bar, 'new above' when every gap < -bar; 'within bar' when every |gap| <= bar, or the gaps have both signs; 'mixed'
    otherwise; 'unreadable' when fewer than n gaps, or one is missing (an S 'no change') or not finite. Rounded to 9
    places."""
    if len(gaps) < n or any(g is None or not math.isfinite(g) for g in gaps):
        return "unreadable"
    gs, s = [r9(float(g)) for g in gaps if g is not None], r9(bar)
    if all(g > s for g in gs):
        return "reuse above"
    if all(g < -s for g in gs):
        return "new above"
    if all(abs(g) <= s for g in gs) or (any(g > 0 for g in gs) and any(g < 0 for g in gs)):
        return "within bar"
    return "mixed"


def pair_outcome(rules: list[str]) -> str:
    """One pair over its saves' saved-size and size-matched rules: 'separates' when every one reads 'reuse above',
    'reversed' when every one reads 'new above', 'unreadable' when any is unreadable, else 'does not separate'."""
    if any(r == "unreadable" for r in rules) or not rules:
        return "unreadable"
    if all(r == "reuse above" for r in rules):
        return "separates"
    return "reversed" if all(r == "new above" for r in rules) else "does not separate"


def outcome(pairs: list[str]) -> str:
    """Over the pairs: 'rival A' when every pair separates, 'rival B' when none separates and none is unreadable,
    'neither' otherwise."""
    if pairs and all(p == "separates" for p in pairs):
        return "rival A"
    if all(p in ("does not separate", "reversed") for p in pairs):
        return "rival B"
    return "neither"


def sd(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def S(pair: str, key: str) -> float | None:
    p = R2 / pair / "S" / f"{key}.json"
    return json.loads(p.read_text())["S"] if p.is_file() else None


def minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def cell(st: dict[str, Any], pair: str, save: str) -> dict[str, Any]:
    """One pair at one save: each side's S per seed (saved, random, and times c on the larger side) and the two gaps."""
    reuse, new = st["pairs"][pair]
    big, c = st["rescale"][f"{pair}/{save}"]["larger"], st["rescale"][f"{pair}/{save}"]["c"]
    ends = {"saved": "", "random": "_random", "c": "_c", "c_random": "_c_random"}  # read name -> record file ending
    side = {a: {k: [S(pair, f"{a}_s{s}_{save}{e}") for s in st["seeds"]] for k, e in ends.items()} for a in (reuse, new)}
    matched = {a: side[a]["c"] if a == big else side[a]["saved"] for a in (reuse, new)}
    n, out = len(st["seeds"]), {"larger": big, "c": c, "S": side}
    out["seed_sd"] = {a: sd(side[a]["saved"]) for a in (reuse, new)}
    for kind, v in (("saved", {a: side[a]["saved"] for a in (reuse, new)}), ("matched", matched)):
        gaps = [minus(x, y) for x, y in zip(v[reuse], v[new], strict=True)]
        out[kind] = {"gaps": gaps, "rule": gap_rule(gaps, st["smallest_gap"], n)}
    return out


def summarise() -> dict[str, Any]:
    P = json.loads((R2 / "predictions.json").read_text())
    st = P["settings"]
    pairs: dict[str, Any] = {}
    for pair in st["pairs"]:
        cells = {save: cell(st, pair, save) for save in st["saves"]}
        rules = [c[k]["rule"] for c in cells.values() for k in ("saved", "matched")]
        pairs[pair] = {"reuse_side": st["pairs"][pair][0], "new_side": st["pairs"][pair][1], "saves": cells,
                       "outcome": pair_outcome(rules)}
        for save, c in cells.items():
            print(f"[summary] {pair} {save}: saved {c['saved']}, matched {c['matched']}, seed sd {c['seed_sd']}", flush=True)
    out = {"experiment": P["experiment"], "cell": "P3.13", "run": 2, "smallest_gap": st["smallest_gap"], "pairs": pairs,
           "outcome": outcome([p["outcome"] for p in pairs.values()])}
    print(f"[summary] outcome {out['outcome']}: {json.dumps({k: v['outcome'] for k, v in pairs.items()})}", flush=True)
    tmp = R2 / "summary.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(R2 / "summary.json")
    return out


if __name__ == "__main__":
    summarise()
