"""E69 Amendment 1: the relaxed grader (moved from the independent check's fourth.py), and control G0: on seed 1's stored
answers it must print RL minus null +1.25 (0.90) on GSM8K and +0.81 (1.06) on GSM1k. No model, no torch:
  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_rl/relaxed.py
Exit 2 on a miss. Blind to the maths beyond the last number; it reads "the answer is N" wherever it stands.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS, RULES = HERE / "results", HERE / "results" / "predictions_amend1.json"
ROOT = Path.home() / "commute-data" / "results-large" / "qwen_rl"
SEED1 = "lr1e-6w10"
CI = re.compile(r"[Tt]he answer is\s*\$?\s*(?:\\boxed\{|\*\*)?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)")
Rec = dict[str, Any]


def ci(r: Rec) -> bool:
    h = CI.findall(r["response"])
    return bool(h) and Decimal(h[-1].replace(",", "")) == Decimal(r["gold"])


def strict(r: Rec) -> bool:
    return bool(r["strict"])


def read(p: Path) -> dict[int, Rec]:
    """A per-problem file by problem; {} if it is not there, which reads as a missing value in every rule."""
    return {r["idx"]: r for r in map(json.loads, p.read_text().splitlines())} if p.is_file() else {}


def paired(d: list[int]) -> dict[str, Any]:
    """Mean and SE of per-problem differences: an empty pool has neither, one problem no SE."""
    return {"n": len(d), "diff": statistics.mean(d) if d else None,
            "se": statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None}


def lead(w0: dict[int, Rec], r: dict[int, Rec], n: dict[int, Rec], key: Callable[[Rec], bool]) -> dict[str, Any]:
    """R minus N on the problems that fit at W0, R and N."""
    return paired([int(key(r[i])) - int(key(n[i])) for i in w0
                   if i in r and i in n and w0[i]["fits"] and r[i]["fits"] and n[i]["fits"]])


def g0() -> dict[str, Any]:
    out = {}
    for name, want in json.loads(RULES.read_text())["G0"]["reads"].items():
        w0, r, n = read(ROOT / "acc" / f"W0_{name}.jsonl"), read(ROOT / SEED1 / "acc" / f"R_{name}.jsonl"), read(ROOT / SEED1 / "acc" / f"N_{name}.jsonl")
        g = lead(w0, r, n, ci)
        got = {"diff": f"{100 * g['diff']:.2f}", "se": f"{100 * g['se']:.2f}"}
        print(f"{name}: RL minus null, relaxed, n {g['n']}: {float(got['diff']):+.2f} ({got['se']})", flush=True)
        out[name] = {**g, "printed": got, "filed": want, "holds": got == want}
    return {**out, "holds": all(v["holds"] for v in out.values())}


if __name__ == "__main__":
    result = g0()
    (RESULTS / "amend1_G0.json").write_text(json.dumps(result, indent=1) + "\n")
    sys.exit(0 if result["holds"] else 2)
