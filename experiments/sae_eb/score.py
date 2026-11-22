"""E263 (cell P1.52) scoring, $0, saved results only: E_b at the SAE featurization (results/items) beside E_b at VPD
(E257, experiments/rb_states/results/items; for the built edits P1.49's experiments/option_b_rescore/results/summary.json),
rule P1 of results/predictions.json, the nine pairs' reuse-minus-new gap at both, and the gain check weighted by write
change (the run's gain_rel_dev_saved is a max over 131,080 variables, ruled by latents M barely writes). Writes
results/summary.json and figures/eb_sae_vs_vpd.png.

    ~/commute-agents/fleet/cpu-run python3 experiments/sae_eb/score.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from commute import figures as FG
from commute.jobs import EXPERIMENTS, load_experiment

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # pair_sides, fname
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
VPD = EXPERIMENTS / "rb_states" / "results" / "items"
VPD_BUILT = json.loads((EXPERIMENTS / "option_b_rescore" / "results" / "summary.json").read_text())["items"]
SAVED = {"e245": EXPERIMENTS / "sae_r_a" / "results" / "large", "e251": EXPERIMENTS / "sae_r_a_random" / "results" / "large"}
REUSE, NEW = ("x2", "x10", "rebuilt"), ("new_few", "new_thin")
fname = E242.fname


def vpd(item: str) -> float | None:
    if item in VPD_BUILT:
        v = VPD_BUILT[item]
        return v["E_b"] if isinstance(v, dict) else v
    p = VPD / f"{fname(item)}.json"
    return json.loads(p.read_text()).get("E_b") if p.is_file() else None


def weighted_gain_dev(r: dict[str, Any]) -> float:
    """dd-weighted mean over K*'s changed members of |num/den - saved gain| / |saved gain|."""
    z = np.load(SAVED["e251" if r["group"] == "random" else "e245"] / f"items_{fname(r['item'])}.npz")
    ours = np.load(RES / "large" / f"items_{fname(r['item'])}.npz")
    sel = z["changed"].astype(bool) & z["main_in_K_polished"].astype(bool)
    num, den, dd, g0 = ours["num"][sel], ours["den"][sel], ours["dd"][sel], z["gain"][sel]
    g = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    return float((dd * np.abs(g - g0) / np.maximum(np.abs(g0), 1e-12)).sum() / dd.sum()) if dd.sum() > 0 else 0.0


def latents_only(r: dict[str, Any]) -> float:
    """E_b with the 8 site remainders left out of both sums (descriptive: how much of the reading the remainders set)."""
    z = np.load(SAVED["e251" if r["group"] == "random" else "e245"] / f"items_{fname(r['item'])}.npz")
    ours = np.load(RES / "large" / f"items_{fname(r['item'])}.npz")
    sel = z["changed"].astype(bool) & z["main_in_K_polished"].astype(bool)
    sel[np.arange(1, 9) * 16385 - 1] = False  # sae_feat: 8 sites of 16,384 latents then 1 remainder each
    num, den, dd = ours["num"][sel], ours["den"][sel], ours["dd"][sel]
    pos = (den > 0) & (num > 0)
    expl = np.where(pos, (num - den) ** 2 / np.where(pos, den, 1.0), 0.0)
    return float(expl.sum() / dd.sum()) if dd.sum() > 0 else 1.0


def main() -> None:
    recs = {r["item"]: r for p in sorted((RES / "items").glob("*.json")) if (r := json.loads(p.read_text())).get("reading") == "read"}
    rows = [{"item": i, "group": r["group"], "E_b_sae": r["E_b"], "E_b_vpd": vpd(i), "untouched_share": r["untouched_share"],
             "remainder_dd_share": r["remainder_dd_share_in_changed_K"], "n_changed_in_K": r["n_changed_in_K"],
             "gain_dev_max": r["gain_rel_dev_saved"], "gain_dev_dd_weighted": weighted_gain_dev(r), "E_b_sae_latents_only": latents_only(r)} for i, r in recs.items()]  # fmt: skip
    by = {r["item"]: r for r in rows}
    lows = [i for i in (*NEW, *(r["item"] for r in rows if r["group"] == "random")) if by[i]["n_changed_in_K"] > 0]
    reuse = [i for i in REUSE if by[i]["n_changed_in_K"] > 0]
    gap = min(by[i]["E_b_sae"] for i in reuse) - max(by[i]["E_b_sae"] for i in lows)
    p1 = {"min_reuse": min(by[i]["E_b_sae"] for i in reuse), "min_reuse_item": min(reuse, key=lambda i: by[i]["E_b_sae"]),
          "max_low": max(by[i]["E_b_sae"] for i in lows), "max_low_item": max(lows, key=lambda i: by[i]["E_b_sae"]), "gap": gap,
          "reading": "rival a (order holds at SAE)" if gap >= 0.2 else "rival b (gap under 0.2)",
          "left_out": [i for i in (*REUSE, *NEW) if by[i]["n_changed_in_K"] == 0]}  # fmt: skip
    pairs = {}
    for p, (ru, nw) in E242.pair_sides().items():
        side = lambda items, k: float(np.mean([by[i][k] for i in items if i in by and by[i][k] is not None]))
        pairs[p] = {f"gap_{k}": side(ru, f"E_b_{k}") - side(nw, f"E_b_{k}") for k in ("sae", "vpd")}
    out = {"experiment": "E263 (sae_eb)", "P1": p1, "pairs": pairs, "rows": rows,
           "gain_dev_dd_weighted_max": max(r["gain_dev_dd_weighted"] for r in rows),
           "random_E_b_sae": {"min": min(by[i]["E_b_sae"] for i in by if by[i]["group"] == "random"), "max": max(by[i]["E_b_sae"] for i in by if by[i]["group"] == "random")},
           "pairs_keys_E_b_sae": {"min": min(r["E_b_sae"] for r in rows if r["group"] in ("pairs", "keys")), "max": max(r["E_b_sae"] for r in rows if r["group"] in ("pairs", "keys"))}}  # fmt: skip
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    fig, ax = FG.figure()
    order = [*REUSE, "moved", *NEW, "random_size_241001", "random_kl_241002"]
    for k, (method, off) in {"vpd": ("regating", -0.15), "sae": ("neuron", 0.15)}.items():
        FG.dots(ax, {i: [v] if (v := by[i][f"E_b_{k}"]) is not None else [] for i in order}, method=method, xlabel="E_b (share of the changed parts' write change M's own write explains)", offset=off)
    ax.legend(["VPD subcomponents (E257)", "SAE latents (E263)"], fontsize=7)
    ax.set_title("E_b of the built edits and random updates at two featurizations", fontsize=8)
    FG.save(fig, HERE / "figures" / "eb_sae_vs_vpd.png", n="one state per row", data=[by[i] for i in order])


if __name__ == "__main__":
    main()
