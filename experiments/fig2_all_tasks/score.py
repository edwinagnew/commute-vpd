"""E325 assembly, $0, CPU, saved arrays only: Fig 2's rows on all 16 known-answer tasks x seeds 0-2 x 3 featurizations,
trained and matched random, in the columns of writeup-lead's figures/data/definition.csv (E298's results/
definition_rows.csv), item named by its known-answer state (e.g. e62/copy_s0).
    new states     this run's reads (results/large/<feat>/<state>[_rand].npz, '/' as '__'), E_b by `commute.eb`
    overlap states E298's rows as Fig 2 carries them (fig2_standard/results/definition_rows.csv, item <task>_s<seed>),
                   renamed to e58/<task>_s<seed>
Check: `commute.eb` on every saved E298 trained npz gives E298's item E_b_K (the refactor's E309 said it does).
Writes results/definition_rows.csv and results/summary.json: what is missing, the check, and the card's descriptive
reading (per featurization and task the seed means of trained and random, their gap, the tasks with gap > 0.05, the
mean over the reuse tasks against the new tasks).

    ~/commute-agents/fleet/cpu-run --mem 4G python3 experiments/fig2_all_tasks/score.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from commute import eb as EB
from commute import known_states as KS
from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SET = json.loads((RESULTS / "predictions.json").read_text())["settings"]
E298 = EXPERIMENTS / "fig2_standard" / "results"
LABEL = {"vpd": "VPD", "neurons": "top", "sae": "SAE"}
COLS = ["feat", "block", "item", "side", "value", "field", "source", "value_no_leftover", "leftover_share"]
GAP = 0.05  # the card's smallest_gap


def readings(p: Path) -> dict[str, float]:
    z = {k: torch.from_numpy(v) for k, v in np.load(p).items()}
    s = EB.Sums(z["num"], z["den"], z["dd"], z["changed"].bool(), z["leftover"].bool())
    K, sel = z["in_K"].bool(), z["in_K"].bool() & s.changed
    assert s.leftover is not None
    d = float(s.dd[sel].sum())
    return {"E_b_K": EB.eb(s, K, True), "E_b_K_no_leftover": EB.eb(s, K, False), "leftover_share": float(s.dd[sel & s.leftover].sum()) / d if d > 0 else 0.0}


def e298_check() -> dict[str, Any]:
    """commute.eb on E298's saved trained arrays against E298's own E_b_K (relative 1e-12)."""
    out, bad = 0, []
    for feat in LABEL:
        for j in sorted((E298 / "items" / feat).glob("trained__*.json")):
            npz = E298 / "large" / feat / j.name.replace(".json", ".npz")
            want, got = json.loads(j.read_text())["E_b_K"], readings(npz)["E_b_K"]
            out += 1
            if abs(got - want) > 1e-12 * abs(want):
                bad.append(f"{feat} {j.stem}: {got!r} != {want!r}")
    return {"checked": out, "differ": bad}


def rows() -> tuple[list[dict[str, Any]], list[str]]:
    out, missing = [], []
    for feat, lab in LABEL.items():
        for t in SET["tasks"]:
            for s in SET["seeds"]:
                for r, side in (("", "trained"), ("_rand", "random")):
                    p = RESULTS / "large" / feat / f"{t.replace('/', '__')}_s{s}{r}.npz"
                    if not p.is_file():
                        missing.append(f"{feat} {t}_s{s}{r}")
                        continue
                    g = readings(p)
                    out.append({"feat": lab, "block": "trained", "item": f"{t}_s{s}", "side": side, "value": g["E_b_K"], "field": "E_b_K", "source": "E325",
                                "value_no_leftover": g["E_b_K_no_leftover"], "leftover_share": g["leftover_share"]})  # fmt: skip
    with (E298 / "definition_rows.csv").open() as f:
        old = {(r["feat"], r["item"], r["side"]): r for r in csv.DictReader(f) if r["block"] == "trained"}
    for lab in LABEL.values():
        for t, e in SET["overlap"].items():
            for s in SET["seeds"]:
                for side in ("trained", "random"):
                    r = old.get((lab, f"{e}_s{s}", side))
                    if r is None:
                        missing.append(f"E298 {lab} {e}_s{s} {side}")
                        continue
                    out.append({k: r[k] for k in COLS} | {"item": f"{t}_s{s}", "value": float(r["value"])})
    return out, missing


def reading(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per featurization: per task the seed means of trained and random and their gap; tasks over GAP; reuse vs new means."""
    sides, out = {s.task: s.side for s in KS.states()}, {}
    for lab in LABEL.values():
        per: dict[str, dict[str, Any]] = {}
        for t, side in sides.items():
            v = {sd: [float(r["value"]) for r in rs if r["feat"] == lab and r["item"].rsplit("_s", 1)[0] == t and r["side"] == sd] for sd in ("trained", "random")}
            if len(v["trained"]) == len(v["random"]) == len(SET["seeds"]):
                tr, rn = fmean(v["trained"]), fmean(v["random"])
                per[t] = {"side": side, "trained": tr, "random": rn, "gap": tr - rn}
        by = {s: [p["trained"] for p in per.values() if p["side"] == s] for s in ("reuse", "new")}
        out[lab] = {"per_task": per, "n_tasks": len(per), "over_gap": sorted(t for t, p in per.items() if p["gap"] > GAP),
                    "mean_reuse": fmean(by["reuse"]) if by["reuse"] else None, "mean_new": fmean(by["new"]) if by["new"] else None}  # fmt: skip
    return out


def main() -> None:
    rs, missing = rows()
    out = {"experiment": "E325", "e298_check": e298_check(), "n_rows": len(rs), "want_rows": 16 * len(SET["seeds"]) * 2 * len(LABEL), "missing": missing,
           "reading": reading(rs), "rows": rs}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    with (RESULTS / "definition_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rs)
    c = out["e298_check"]
    print(f"commute.eb gives E298's E_b_K on {c['checked'] - len(c['differ'])} of {c['checked']} saved trained items: {c['differ'][:4]}")
    for lab, r in out["reading"].items():
        print(f"{lab}: {r['n_tasks']} tasks; gap > {GAP}: {r['over_gap']}; mean trained reuse {r['mean_reuse']}, new {r['mean_new']}")
    print(f"{len(rs)} rows of {out['want_rows']}; missing {len(missing)}: {missing[:12]}")


if __name__ == "__main__":
    main()
