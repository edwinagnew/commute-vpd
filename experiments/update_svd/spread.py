"""The card's spread for P2.9: seed-to-seed sd of the update's effective rank on one pair's saved seeds.

The pair: rank-1 vs cipher at step 10 (E55 C3, open_pair_instruments), eight seeds per side, the most seeds any saved
pair has. Writes results/spread.json. Weights only, CPU, about a minute.

Usage (repo root):  ~/commute-agents/fleet/cpu-run -c 2 --mem 4G python experiments/update_svd/spread.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from commute import update_spectrum as US
from commute.jobs import hard_exit

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def path(m: str, s: int) -> Path:
    if s >= 3:
        return EXP / "open_pair_instruments" / "results" / "large" / f"{m}_s{s}.pt"
    sub = "" if m == "cipher" else "amend1"
    return EXP / "open_pair" / "results" / "large" / sub / f"{m}_s{s}.pt"


def main() -> None:
    torch.set_num_threads(2)
    out: dict[str, dict[str, object]] = {}
    for m in ("rank1", "cipher"):
        vals = [US.state(US.load(path(m, s))) for s in range(8)]
        er = [float(v["erank"]) for v in vals]  # type: ignore[arg-type]
        out[m] = {"erank": er, "sd": statistics.stdev(er), "norm2": [v["norm2"] for v in vals]}
        print(m, json.dumps(out[m]), flush=True)
    pooled = (sum(float(v["sd"]) ** 2 for v in out.values()) / 2) ** 0.5  # type: ignore[arg-type]
    res = {"pair": "open_pair_c3 (rank-1 vs cipher, step 10), seeds 0-7", "sides": out, "pooled_sd": pooled}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "spread.json").write_text(json.dumps(res, indent=1))
    print(f"pooled seed-to-seed sd of erank: {pooled}", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
