"""P2.56: each proxy's Spearman with R_b on trained states only (the `random` and `randg` arms dropped).

Reads results/score_rb.json `.points` (written by score.py --rb) and writes results/score_rb_trained.json as
{proxy: {rho, n}}, with the same Spearman as score.py (P2.51's).

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/procedural_rl/trained_only.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from commute.jobs import load_experiment

RESULTS = Path(__file__).resolve().parent / "results"
AN = load_experiment("proxies_vs_r", "analyse")  # P2.51's spearman
RANDOM_ARMS = {"random", "randg"}


def main() -> None:
    d = json.loads((RESULTS / "score_rb.json").read_text())
    pts = [q for q in d["points"] if q["arm"] not in RANDOM_ARMS]
    rb = {q["state"]: q["R_b"] for q in pts}
    out = {}
    for p in d["proxies"]:
        a = {q["state"]: q[p] for q in pts if q[p] is not None and math.isfinite(q[p])}
        s = AN.spearman(a, rb)
        out[p] = {"rho": s["rho"], "n": s["n"]}
    (RESULTS / "score_rb_trained.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
