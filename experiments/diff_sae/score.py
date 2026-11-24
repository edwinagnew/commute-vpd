"""P2.75 run 1 on the CPU: the gate (spec 5; results/predictions.json "gate") per form of nu, from the item files.

    python experiments/diff_sae/score.py gate [--tag run1]     -> results/<tag>/gate.json, one line per form printed

An arm is unreadable when its item file is missing, not done, stopped (a filed stop, a failed path check), or read
"not read". The scored rule is the published one (nu <= 0.3); the E.2 form is computed beside and labelled, unscored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from commute import diff_sae as DS

HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"


def item(res: Path, key: str) -> dict[str, Any] | None:
    p = res / "items" / f"{key}.json"
    d = json.loads(p.read_text()) if p.is_file() else None
    return d if d and d.get("done") and not d.get("stopped") and isinstance(d.get("read"), dict) else None


def part(d: dict[str, Any] | None, form: str, rule: str) -> float | None:
    return None if d is None else d["read"][form][f"{rule}_0.3"]["specific"]


def gate(res: Path) -> dict[str, Any]:
    run = json.loads(PRED.read_text())["settings"]["run"]
    r1 = [item(res, k) for k in run["rung1"]]
    g = [item(res, k) for k in run["gauss"]]
    Ks = [d["K"] for d in r1 if d is not None]
    assert len(set(Ks)) <= 1, f"the rung-1 seeds' K differ ({Ks}): they must share one normaliser sample"
    K1 = Ks[0] if Ks else math.nan
    out: dict[str, Any] = {"K1": K1, "arms": {k: d is not None for k, d in zip([*run["rung1"], *run["gauss"]], [*r1, *g], strict=True)}}
    for form in ("lit", "D"):
        for rule in ("published", "e2"):
            r, gg = [part(d, form, rule) for d in r1], [part(d, form, rule) for d in g]
            out[f"{form}_{rule}"] = DS.gate(r, gg, K1) | {"rung1": r, "gauss": gg, "scored": rule == "published"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("gate",))
    ap.add_argument("--tag", default="run1")
    a = ap.parse_args()
    res = HERE / "results" / a.tag
    out = gate(res)
    (res / "gate.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "outcome" in v:
            print(f"{k}: {v['outcome']} (bar {v['bar']}, rung 1 {v['rung1']}, Gaussians {v['gauss']}{'' if v['scored'] else ', not scored'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
