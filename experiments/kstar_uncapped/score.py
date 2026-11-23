"""E269 (cell P2.61) scoring on the dev node, no model: E_b per item with the uncapped K*, from E257's saved arrays.

Per item: E257's num, den, dd and changed (experiments/rb_states/results/large/items_<item>.npz) and the uncapped
polished K* (results/large/kstar_<item>.npz, key main_in_K_polished; results/dl/large/ until the lead moves it).
Control first: E_b recomputed with E257's own saved K* (in_K in the same npz) must equal E257's saved E_b exactly (the
same arrays and expression, so any difference is a bug here). Then the uncapped E_b, |K*| against the capped size,
the |K| the search logged at 2,000 evaluations (results/logs/run_*.log) against the capped size, and the filed rule's
gaps (predictions.json "rules"). Writes results/summary.json.

    ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/kstar_uncapped/score.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from commute.jobs import EXPERIMENTS
from commute.stand_in import explained_share

RESULTS = Path(__file__).resolve().parent / "results"
RB = EXPERIMENTS / "rb_states" / "results"
PRED = json.loads((RESULTS / "predictions.json").read_text())
GAP = 0.01  # the filed rule's bar (predictions.json "rules")


def fname(item: str) -> str:
    return item.replace("/", "-")


def e_b(a: Any, K: np.ndarray) -> float:
    t = lambda k: torch.from_numpy(np.asarray(a[k]))
    return explained_share(t("num"), t("den"), t("dd"), torch.from_numpy(a["changed"].astype(bool) & K.astype(bool)))


def at_2000() -> dict[str, int]:
    """|K| logged at 2,000 evaluations per item, from the pod logs (one item at a time per log, in order)."""
    out: dict[str, int] = {}
    for p in sorted((RESULTS / "logs").glob("run_*.log")):
        item = None
        for ln in p.read_text().splitlines():
            if m := re.search(r"(p256/sequence_\w+): \d+ windows", ln):
                item = m.group(1)
            if item and (m := re.search(r"polish: 2000 evaluations, .* \|K\| (\d+)", ln)):
                out[item] = int(m.group(1))
    return out


def read(item: str, k2000: dict[str, int]) -> dict[str, Any] | None:
    js = RESULTS / "kstar" / f"{fname(item)}.json"
    npz = next((p for p in (RESULTS / "large" / f"kstar_{fname(item)}.npz", RESULTS / "dl" / "large" / f"kstar_{fname(item)}.npz") if p.is_file()), None)
    if not js.is_file() or npz is None:
        return None
    rec, a, cap = json.loads(js.read_text()), np.load(RB / "large" / f"items_{fname(item)}.npz"), PRED["capped_E257"][item]
    K = np.load(npz)["main_in_K_polished"]
    control = e_b(a, a["in_K"])
    assert control == cap["E_b"], f"{item}: E_b with E257's K* {control!r} is not E257's saved {cap['E_b']!r}"
    assert int(a["in_K"].sum()) == cap["size_K"], item
    m = rec["searches"]["main"]
    return {"item": item, "E_b": e_b(a, K), "E_b_capped": cap["E_b"], "size_K": int(K.sum()), "size_K_capped": cap["size_K"],
            "subset_of_capped": bool((K <= a["in_K"]).all()), "K_at_2000": k2000.get(item), "evals": m["evals"], "capped": m["capped"],
            "seconds": m["seconds"], "R_read": m["polished"]["R_read"]}  # fmt: skip


def main() -> None:
    k2000 = at_2000()
    rows = {i: r for i in PRED["settings"]["items"] if (r := read(i, k2000)) is not None}
    kind = lambda k: [r["E_b"] for i, r in rows.items() if re.search(rf"sequence_{k}_s\d", i)]
    sft = kind("sft")
    gaps: dict[str, Any] = {}
    for k in ("rl", "rlg"):
        v = kind(k)
        g = fmean(v) - fmean(sft) if len(v) >= 2 and len(sft) >= 2 else None
        gaps[k] = {"n": len(v), "n_sft": len(sft), "gap": g, "call": None if g is None else "above" if g >= GAP else "alike" if abs(g) < GAP else "below"}
    calls = {g["call"] for g in gaps.values()}
    reading = "rival a" if calls == {"above"} else "rival b" if calls == {"alike"} else "no call" if None in calls else "mixed"
    out = {"experiment": PRED["experiment"], "cell": PRED["cell"], "n_read": len(rows), "items": rows, "gaps": gaps, "reading": reading,
           "control": "E_b with E257's saved K* equals E257's saved E_b on every read item (asserted)"}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"reading": reading, "gaps": gaps} | {i: (r["E_b"], r["E_b_capped"], r["size_K"], r["size_K_capped"], r["K_at_2000"]) for i, r in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
