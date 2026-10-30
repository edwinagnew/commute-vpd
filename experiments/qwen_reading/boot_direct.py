"""E57 (qwen_reading): the direct share's row bootstrap, from the saved per-row cell sums, on CPU. Asked by
reports/decisions/2026-10-30_decision_e57-no-teeth.md ("if the direct share's row bootstrap puts gold and the random
changes inside one interval, close D3 now"). Same scheme as the re-gating share's boot95 (commute.qwen_reading:
2000 resamples of the 128 maths rows, random.Random(0), shares from the resampled sums, 2.5th and 97.5th percentiles).
Every state and both C0 controls are read on the same 128 maths rows, so a difference is a paired bootstrap: the same
row draw for both terms.

Usage: boot_direct.py OUT C0.json C_<state>.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute import qwen_reading as qr

ROW_LEN = 256  # lab.Config.row_len: the maths rows are cut to 256 tokens (E44 Amendment 2)
BOOT = 2000  # lab.Config.boot
CONTROLS = ("gauss", "inspan")


def direct(rows: dict[str, list[float]], idx: list[int]) -> float:
    return qr.shares(qr.sums_at(rows, idx), len(idx) * ROW_LEN)["direct"]


def read(rows: dict[str, list[float]], draws: list[list[int]]) -> dict[str, Any]:
    w = qr.whole(rows, ROW_LEN)
    return {"direct": w["direct"], "status": w["status"],
            "boot95": qr.interval([direct(rows, i) for i in draws])}


def main(out: Path, c0: Path, states: list[Path]) -> None:
    ctrl = json.loads(c0.read_text())
    maths = {n: ctrl[n]["rows"]["maths"] for n in CONTROLS}
    own: dict[str, dict[str, list[float]]] = {}
    for p in states:
        rec = json.loads(p.read_text())
        name = p.stem.removeprefix("C_")
        maths[name] = rec["rows"]["maths"]
        if "own" in rec["rows"]:
            own[name] = rec["rows"]["own"]
    n = {len(r["K0"]) for r in maths.values()}
    assert len(n) == 1, f"states read on different row counts: {n}"
    draws = qr.resample(n.pop(), BOOT)
    res: dict[str, Any] = {
        "scheme": f"{BOOT} row resamples (random.Random(0)), paired across states; 95% = 2.5th/97.5th percentiles",
        "maths": {k: read(r, draws) for k, r in maths.items()},
        "own": {k: read(r, qr.resample(len(r["K0"]), BOOT)) for k, r in own.items()},
        "minus_control": {},
    }
    for s, rows in maths.items():
        if s in CONTROLS:
            continue
        for c in CONTROLS:
            gaps = [direct(rows, i) - direct(maths[c], i) for i in draws]
            res["minus_control"][f"{s}-{c}"] = {"gap": res["maths"][s]["direct"] - res["maths"][c]["direct"],
                                               "boot95": qr.interval(gaps)}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), [Path(a) for a in sys.argv[3:]])
