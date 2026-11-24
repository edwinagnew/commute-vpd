"""E298 scoring, $0, CPU, saved arrays only: every one of the 48 states at each of the three featurizations, E_b
recomputed from its saved per-variable sums by eb.readings (one expression everywhere), from this run's reads
(results/large/<feat>/) or, where the run does not read the state again, from the saved reads whose settings eb.check_sources
confirms are this protocol's (the reuse check is in the hand-back and results/summary.json "sources"):
    VPD built edits      E268's K (kstar_needed/results/large/items_<item>.npz in_K) on E247's sums
                         (thin_edit_diagnosis/results/large) and E242's changed set (kstar_vpd_wording_a/results/large),
                         the arrays E268's own E_b_K was computed from
    VPD trained seed 0   E290 (ia3_control/results/large/vpd_full__<state>.npz), except copy_s0 and its random (read again)
    neurons built edits  E277 (needed_neurons/results/large/items_<item>.npz)
    head-(2, 4) edits    E280 at VPD and neurons (head_edit/results/large/items_<e>_<feat>.npz; E268's scan settings,
                         head_edit/run.py:311), in their own block "head", sides edit and random
    neurons trained      E290 (ia3_control/results/large/items_full__<state>.npz)
Writes results/summary.json (every reading, which states are missing, whether the reused values give the old figure's
values back, the moves from the old figure and the rivals' reading) and results/definition_rows.csv, the columns of
writeup-lead's figures/data/definition.csv plus value_no_leftover and leftover_share.

    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/fig2_standard/score.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from commute.jobs import EXPERIMENTS, load_experiment

EB: Any = load_experiment("fig2_standard", "eb")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SET = json.loads((RESULTS / "predictions.json").read_text())["settings"]
OLD = Path("/mnt/nw/home/e.agnew/commute-agents/writeup-lead/paper/paper/figures/data/definition.csv")
LABEL = {"vpd": "VPD", "neurons": "top", "sae": "SAE"}  # the old figure's feat column
BUILT_SIDE = {"x2": "reuse", "x10": "reuse", "rebuilt": "reuse", "new_few": "new", "new_thin": "new"}  # else random
GAP = 0.05  # the card's smallest_gap
LARGE = {k: EXPERIMENTS / d / "results" / "large" for k, d in
         {"E268": "kstar_needed", "E247": "thin_edit_diagnosis", "E242": "kstar_vpd_wording_a", "E277": "needed_neurons", "E290": "ia3_control", "E280": "head_edit"}.items()}  # fmt: skip


def reused(feat: str, state: str) -> tuple[dict[str, np.ndarray], str] | None:
    """A saved read of the state under this protocol, or None. A state stage 1 reads again is never taken from a source
    (a stage-0 reproduction item is: its source is a checked one, and this run's read of it, when there, comes first)."""
    if state in SET["read"][feat]:
        return None
    block, name = state.split("/")
    if block == "built" and name.startswith("head_") and feat in ("vpd", "neurons"):
        return dict(np.load(LARGE["E280"] / f"items_{name.removeprefix('head_')}_{feat}.npz")), "E280"
    if block == "built" and name in SET["built"]:
        if feat == "vpd":
            e, f = np.load(LARGE["E247"] / f"items_{name}.npz"), lambda k, p: np.load(LARGE[k] / f"items_{name}.npz")[p]
            return {k: e[k] for k in ("num", "den", "dd")} | {"changed": f("E242", "changed"), "in_K": f("E268", "in_K")}, "E268"
        if feat == "neurons":
            return dict(np.load(LARGE["E277"] / f"items_{name}.npz")), "E277"
    if block == "trained" and feat in ("vpd", "neurons"):
        p = LARGE["E290"] / f"{'vpd' if feat == 'vpd' else 'items'}_full__{name}.npz"
        if p.is_file():
            return dict(np.load(p)), "E290"
    return None


def label(state: str) -> tuple[str, str, str]:
    """(block, item, side) as the old figure names them; a built edit's new matched random is (built, <edit>, random)."""
    block, name = state.split("/")
    if block == "trained":
        return block, name.removesuffix("_rand"), "random" if name.endswith("_rand") else "trained"
    rnd, base = name.startswith("rand_"), name.removeprefix("rand_")
    if base.startswith("head_"):  # E280's group: block head, item v2 / o2 / q2
        return "head", base.removeprefix("head_"), "random" if rnd else "edit"
    if rnd:
        return block, base, "random"
    return block, name, BUILT_SIDE.get(name, "random")


def rows() -> tuple[list[dict[str, Any]], list[str]]:
    out, missing = [], []
    states = EB.states(*(tuple(SET[k]) for k in ("built", "head_edits", "built_random_of", "tasks", "seeds")))
    for feat, lab in LABEL.items():
        for state in states:
            new = RESULTS / "large" / feat / f"{EB.fname(state)}.npz"
            got = (dict(np.load(new)), "E298") if new.is_file() else reused(feat, state)
            if got is None:
                missing.append(f"{feat} {state}")
                continue
            r = EB.from_npz(got[0], feat)
            block, item, side = label(state)
            out.append({"feat": lab, "block": block, "item": item, "side": side, "value": r["E_b_K"], "field": "E_b_K", "source": got[1],
                        "value_no_leftover": r["E_b_K_no_leftover"], "leftover_share": r["leftover_share"], "state": state} | r)  # fmt: skip
    return out, missing


def old_values(path: Path) -> dict[tuple[str, str, str, str], tuple[float, str, str]]:
    with path.open() as f:
        return {(r["feat"], r["block"], r["item"], r["side"]): (float(r["value"]), r["field"], r["source"]) for r in csv.DictReader(f)}


def like(r: dict[str, Any], old_field: str) -> float:
    """This run's value in the old row's field: the old neuron column plotted E_b over K without the remainders."""
    return r["value_no_leftover"] if old_field == "E_b_K_no_rem" else r["value"]


def per_task(rs: list[dict[str, Any]], old: dict[Any, Any]) -> list[dict[str, Any]]:
    """Per featurization, block, task (mean over seeds) or built edit, and side: the new headline and like-for-like
    values beside the old figure's mean over the rows it had."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in rs:
        task = r["item"].rsplit("_s", 1)[0] if r["block"] == "trained" else r["item"]
        groups.setdefault((r["feat"], r["block"], task, r["side"]), []).append(r)
    out = []
    for (feat, block, task, side), g in groups.items():
        o = [old[(feat, block, r["item"], side)] for r in g if (feat, block, r["item"], side) in old]
        row: dict[str, Any] = {"feat": feat, "block": block, "task": task, "side": side, "n": len(g), "value": fmean(r["value"] for r in g),
                               "value_no_leftover": fmean(r["value_no_leftover"] for r in g)}  # fmt: skip
        if o:
            row |= {"old": fmean(v for v, _, _ in o), "old_n": len(o), "old_field": o[0][1], "old_source": o[0][2]}
            row["move"] = row["value"] - row["old"]
            row["move_like"] = fmean(like(r, o[0][1]) for r in g) - row["old"]
        out.append(row)
    return out


def order(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per featurization: the lowest built reuse value against the highest built new or random value (headline)."""
    out = {}
    for feat in LABEL.values():
        b = [r for r in rs if r["feat"] == feat and r["block"] == "built"]
        reuse, low = [r["value"] for r in b if r["side"] == "reuse"], [r["value"] for r in b if r["side"] != "reuse"]
        if reuse and low:
            out[feat] = {"min_reuse": min(reuse), "max_new_or_random": max(low), "holds": min(reuse) > max(low)}
    return out


def sae_x2(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """The filed sae_x2_expected: SAE x2's E_b over K strictly between lo and hi, K non-empty."""
    e = json.loads((RESULTS / "predictions.json").read_text())["sae_x2_expected"]
    r = next((r for r in rs if r["feat"] == "SAE" and r["state"] == SET["control"]), None)
    if r is None:
        return {"held": None, "why": "not read"}
    return {"E_b_K": r["E_b_K"], "size_K": r["size_K"], "lo": e["lo"], "hi": e["hi"], "held": e["lo"] < r["E_b_K"] < e["hi"] and r["size_K"] > 0}


def main(old_path: Path) -> None:
    rs, missing = rows()
    old = old_values(old_path)
    repro = [{"key": k, "old": old[k][0], "now": like(r, old[k][1])} for r in rs if r["source"] != "E298"
             and (k := (r["feat"], r["block"], r["item"], r["side"])) in old and old[k][2] == r["source"]]  # fmt: skip
    bad = [x for x in repro if abs(x["now"] - x["old"]) > 1e-12 * max(abs(x["old"]), 1e-300)]
    pt = per_task(rs, old)
    moved = [p for p in pt if "move" in p and abs(p["move"]) > GAP]
    out = {"experiment": "E298", "sources": EB.check_sources(argparse.Namespace(**{k: tuple(v) if isinstance(v, list) else v for k, v in SET.items()})),
           "n_rows": len(rs), "missing": missing, "reused_gives_old": {"checked": len(repro), "differ": bad}, "per_task": pt,
           "sae_x2_expected": sae_x2(rs), "moved_over_gap": moved, "max_move": max((abs(p["move"]) for p in pt if "move" in p), default=None), "order": order(rs),
           "reading": None if missing else "rival A (a per-task value moved > 0.05)" if moved else "rival B (none moved > 0.05)", "rows": rs}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    cols = ["feat", "block", "item", "side", "value", "field", "source", "value_no_leftover", "leftover_share"]
    with (RESULTS / "definition_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rs)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "per_task")}, indent=1))
    print(f"filed SAE x2 expectation (lo < E_b over K < hi, K non-empty) held: {out['sae_x2_expected']['held']}")
    print(f"{len(rs)} rows, {len(missing)} missing; reused values give the old figure's back: {len(repro) - len(bad)} of {len(repro)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=OLD, help="the old figure's rows (read only)")
    main(ap.parse_args().old)
