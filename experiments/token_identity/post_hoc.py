"""E95's post hoc reads, unfiled, from saved files only, no forward pass. They decide nothing filed.

  update_split        each state's update by block: share of its squared norm upstream of block 1's removal (blocks 0 and 1)
  keys                table A's and table B's key sets (measured, as the filing states)
  paired_other_units  copy minus random under the basis against the 20 draws, in net log-odds and in the nats share
  per_window          whether a state's mean drop is carried by a few windows; what the state's log P says it learned
  read1_pairs         read 1's paired test pair by pair (run.json keeps only its call)
  teeth_b             table B's control against its draws; how often a redraw of 20 keeps the teeth, on a normal fit
  task_match          the draws' share against the basis's on the task windows, and the filed band scaled to a full match

Writes results/post_hoc.json. Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/token_identity/post_hoc.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES, LARGE = HERE / "results", HERE / "results" / "large"
REG = HERE.parent / "regating_table" / "results" / "large"
TABLES = {"A": ("ident32", "rand32"), "B": ("ident32b", "rand32b")}
BASIS, DRAWS, SEEDS = "(1, -1)", [f"(1, {i})" for i in range(20)], (0, 1)
LOGP, LOGODDS = 0, 1
RUN = json.loads((RES / "run.json").read_text())


def update_split(name: str) -> dict[str, Any]:
    d = torch.load(REG / f"{name}.pt", weights_only=True, map_location="cpu")
    sq = {k: float((v.double() ** 2).sum()) for k, v in d.items()}
    total = sum(sq.values())
    of = lambda prefix: sum(x for k, x in sq.items() if k.startswith(prefix)) / total
    blocks = {str(b): of(f"h.{b}.") for b in range(4)}
    return {"norm": total**0.5, "block_share": blocks, "mlp_share": {str(b): of(f"h.{b}.mlp") for b in range(4)}, "upstream": blocks["0"] + blocks["1"], "share_lost": RUN["states"][name]["1"]["A_lo"]}


def keys() -> dict[str, Any]:
    ks = {t: {int(k) for k in torch.as_tensor(torch.load(REG / f"windows_{t}.pt", weights_only=False)["task"].keys).tolist()} for ts in TABLES.values() for t in ts}
    return {"sizes": {t: len(v) for t, v in ks.items()}, "shared_copy_A_B": len(ks["ident32"] & ks["ident32b"]), "shared_random_A_B": len(ks["rand32"] & ks["rand32b"]), "copy_random_same_A": ks["ident32"] == ks["rand32"], "copy_random_same_B": ks["ident32b"] == ks["rand32b"]}


def net_lo(r: dict[str, Any], name: str, ix: str) -> float:
    """The state's mean log-odds drop minus W0's under one removal, the numerator of the scored share."""
    return float((r[name]["clean"][LOGODDS] - r[name][ix][LOGODDS]).mean() - (r["W0"]["clean"][LOGODDS] - r["W0"][ix][LOGODDS]).mean())


def nats_share(r: dict[str, Any], name: str, ix: str) -> float:
    """`readout.shares`' A_nats: nats lost over nats gained, W0's drop not subtracted. Recorded, never scored."""
    return float((r[name]["clean"][LOGP] - r[name][ix][LOGP]).mean() / (r[name]["clean"][LOGP] - r["W0"]["clean"][LOGP]).mean())


def against_draws(f: Any, rc: dict[str, Any], c: str, rr: dict[str, Any], x: str) -> dict[str, Any]:
    gap, draws = f(rc, c, BASIS) - f(rr, x, BASIS), [f(rc, c, d) - f(rr, x, d) for d in DRAWS]
    return {"gap": gap, "draws_mean": float(np.mean(draws)), "draws_max": max(draws), "above": bool(gap > max(draws))}


def paired_other_units(reads: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tab, (c, r) in TABLES.items():
        pairs = [(f"s{a} - s{b}", f"{c}_s{a}", f"{r}_s{b}") for a in SEEDS for b in SEEDS]
        out[tab] = {"net_logodds": {p: against_draws(net_lo, reads[c], x, reads[r], y) for p, x, y in pairs}, "nats_share": {p: against_draws(nats_share, reads[c], x, reads[r], y) for p, x, y in pairs}}
    return out


def per_window(r: dict[str, Any], name: str) -> dict[str, float]:
    x = ((r[name]["clean"][LOGODDS] - r[name][BASIS][LOGODDS]) - (r["W0"]["clean"][LOGODDS] - r["W0"][BASIS][LOGODDS])).double()
    g = (r[name]["clean"][LOGODDS] - r["W0"]["clean"][LOGODDS]).double()
    return {"net_drop_mean": float(x.mean()), "net_drop_median": float(x.median()), "top10_of_97": float(x.sort(descending=True).values[:10].sum() / x.sum()), "gain_mean": float(g.mean()), "share_of_medians": float(x.median() / g.median()), "P_geo": float(r[name]["clean"][LOGP].mean().exp())}


def read1_pairs() -> dict[str, Any]:
    t, rows, out = RUN["thresholds"], RUN["read1"]["rows"], {}
    for a in SEEDS:
        for b in SEEDS:
            draws = np.array(t[f"ident32_s{a}"]["draw_gaps"]) - np.array(t[f"rand32_s{b}"]["draw_gaps"])
            gap = rows["ident32"][f"s{a}"]["gap"] - rows["rand32"][f"s{b}"]["gap"]
            out[f"s{a} - s{b}"] = {"gap": gap, "draws_mean": float(draws.mean()), "draws_max": float(draws.max()), "above": bool(gap > draws.max())}
    return out


def teeth(c: dict[str, Any]) -> dict[str, Any]:
    r = np.array(c["random"])
    sets = np.random.default_rng(95).normal(r.mean(), r.std(ddof=1), (20000, len(r)))
    kept = float((np.percentile(np.abs(sets), 95, axis=1) < c["drop"]).mean())
    return {"drop": c["drop"], "band": c["band"], "largest_draw": float(r.max()), "draws_over_drop": int((r > c["drop"]).sum()), "kept_on_redraw_normal_fit": kept}


def task_match(name: str) -> dict[str, float]:
    tw = RUN["task_windows"][f"windows_{name}"]["1"]
    return {"basis": tw["basis"], "draws_mean": tw["draws_mean"], "ratio": tw["draws_mean"] / tw["basis"], "draws_at_0.9": sum(x >= 0.9 * tw["basis"] for x in tw["draws"])}


def main() -> None:
    names = [f"{t}_s{s}" for ts in TABLES.values() for t in ts for s in SEEDS]
    reads = {t: torch.load(LARGE / f"reads_{t}.pt", weights_only=True, map_location="cpu") for ts in TABLES.values() for t in ts}
    b = task_match("ident32b")
    out = {
        "standing": "post hoc, unfiled; saved numbers only; decides nothing filed",
        "update_split": {n: update_split(n) for n in names},
        "keys": keys(),
        "paired_other_units": paired_other_units(reads),
        "per_window": {n: per_window(reads[n.rsplit("_", 1)[0]], n) for n in names},
        "read1_pairs": read1_pairs(),
        "teeth": {k: teeth(RUN["control"][k]["1"]) for k in RUN["control"]},
        "task_match": {"A": task_match("ident32"), "B": b, "filed_band_B_scaled_to_full_match": RUN["read2"]["filed"]["band"] / b["ratio"]},
    }
    (RES / "post_hoc.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
