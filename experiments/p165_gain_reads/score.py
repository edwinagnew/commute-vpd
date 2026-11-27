"""E322 rows, $0, CPU, saved arrays only: every gain item read (results/large/<feat>/gain__<item>.npz) through
E298's one expression (fig2_standard/eb.py from_npz), as rows in fig2_standard/results/definition_rows.csv's columns
with block "gain", side "gain", beside the full fine-tunes' rows of the same task and seed at that featurization (from
that file) and the VPD repeat checks (results/repeat/*.json). Writes results/definition_rows.csv and results/summary.json.

    ~/commute-agents/fleet/cpu-run python3 experiments/p165_gain_reads/score.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from commute.jobs import EXPERIMENTS, load_experiment

EB: Any = load_experiment("fig2_standard", "eb")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E298_ROWS = EXPERIMENTS / "fig2_standard" / "results" / "definition_rows.csv"
LABEL = {"vpd": "VPD", "sae": "SAE"}  # E298's feat column (fig2_standard/score.py LABEL)
COLS = ["feat", "block", "item", "side", "value", "field", "source", "value_no_leftover", "leftover_share"]


def rows(experiment: str) -> list[dict[str, Any]]:
    with E298_ROWS.open() as f:
        full = {(r["feat"], r["item"]): float(r["value"]) for r in csv.DictReader(f) if r["block"] == "trained" and r["side"] == "trained"}
    out = []
    for feat, lab in LABEL.items():
        for p in sorted((RESULTS / "large" / feat).glob("gain__*.npz")):
            item = p.stem.removeprefix("gain__")
            r = EB.from_npz(dict(np.load(p)), feat)
            out.append({"feat": lab, "block": "gain", "item": item, "side": "gain", "value": r["E_b_K"], "field": "E_b_K", "source": experiment,
                        "value_no_leftover": r["E_b_K_no_leftover"], "leftover_share": r["leftover_share"], "full_value": full.get((lab, item))} | r)  # fmt: skip
    return out


def main() -> None:
    experiment = json.loads((RESULTS / "predictions.json").read_text())["experiment"].split()[0]
    rs = rows(experiment)
    rep = [json.loads(p.read_text()) for p in sorted((RESULTS / "repeat").glob("*.json"))]
    out = {"experiment": experiment, "n_rows": len(rs), "repeat": rep, "repeat_max_rel_dev": max((r["rel_dev"] for r in rep), default=None), "rows": rs}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    with (RESULTS / "definition_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rs)
    for r in rs:
        print(f"{r['feat']:4s} {r['item']:16s} E_b over K {r['value']:.4f} (full fine-tune {r['full_value']}), leftovers dropped "
              f"{r['value_no_leftover']:.4f}, leftover share {r['leftover_share']:.4f}, |K| {r['size_K']}")  # fmt: skip
    print(f"{len(rs)} rows; repeat checks {len(rep)}, max relative deviation {out['repeat_max_rel_dev']}")


if __name__ == "__main__":
    main()
