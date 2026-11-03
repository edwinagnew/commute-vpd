"""E69: cosines between saved updates over all 196 matrices, one matrix at a time. No model. Writes results/lr1e-6w10/cosines.json.

Usage (repo root):  ~/commute-agents/fleet/cpu-run -c 2 --mem 4G -t 01:00:00 --wait 1800 python experiments/qwen_rl/cosines.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

RUN = Path(__file__).resolve().parent / "results" / "lr1e-6w10"
DW = RUN.parent / "large" / "lr1e-6w10" / "dW"
PAIRS = (("R", 100, "N", 100), ("R", 50, "N", 50), ("R", 10, "N", 10), ("R", 1, "N", 1), ("R", 10, "R", 100), ("R", 50, "R", 100),
         ("N", 10, "N", 100), ("N", 50, "N", 100), ("F", 100, "R", 100), ("F", 100, "N", 100), ("F", 50, "F", 100), ("F", 10, "F", 100))


def cosine(a: Path, b: Path) -> dict[str, float]:
    ab = aa = bb = 0.0
    with safe_open(a, "np") as fa, safe_open(b, "np") as fb:
        for k in fa.keys():  # noqa: SIM118
            x, y = fa.get_tensor(k).astype(np.float64).ravel(), fb.get_tensor(k).astype(np.float64).ravel()
            ab, aa, bb = ab + float(x @ y), aa + float(x @ x), bb + float(y @ y)
    return {"cos": ab / math.sqrt(aa * bb), "norm_a": math.sqrt(aa), "norm_b": math.sqrt(bb)}


def main() -> None:
    out = {}
    for a, s, b, t in PAIRS:
        out[f"{a}{s}_{b}{t}"] = cosine(DW / a / f"step_{s}.safetensors", DW / b / f"step_{t}.safetensors")
        print(f"{a}{s}_{b}{t}", json.dumps(out[f"{a}{s}_{b}{t}"]), flush=True)
    (RUN / "cosines.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
