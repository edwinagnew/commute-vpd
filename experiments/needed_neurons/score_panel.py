"""E277 amendment 1, CPU: E272's statistics on E_b over K at the top-neuron featurization beside E_b over K at VPD.

Per task the mean E_b over K of its three fine-tuned seeds (E272's `task_means`; a seed with |K| < K_MIN reads missing,
and a task with a missing seed is left out, as E272 scores it). VPD's values are E272's own (`gather` on its wide set:
read and eb files); the neurons' are this run's panel items (results/items/panel__*.json, the same states). For each
side: Kendall tau-b with each rater's tiers and the consensus (E272's Amendment 10 form, unoriented), and E272's
`reading_given_loss` per base-loss rule, unit and subset (tau and partial tau given the loss, on E_b's filed
orientation, with E272's permutation p). Every statistic is E272's function, imported. Writes results/panel_score.json.

    ~/commute-agents/fleet/cpu-run python3 experiments/needed_neurons/score_panel.py [ITEMS_DIR]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

SC: Any = load_experiment("convergence_panel", "score")  # gather, task_means, raters, tau, reading_given_loss, perm_p, K_MIN, WITHIN_TEXT
HERE = Path(__file__).resolve().parent
E272 = HERE.parent / "convergence_panel" / "results"


def neuron_seeds(items: Path) -> dict[str, list[float]]:
    """task -> E_b over K of each fine-tuned panel state (nan when |K| < K_MIN), from the item files."""
    out: dict[str, list[float]] = {}
    for p in sorted(items.glob("panel__*.json")):
        r = json.loads(p.read_text())
        if not r["random"]:
            out.setdefault(r["task"], []).append(r["E_b_K"] if r["size_K"] >= SC.K_MIN else math.nan)
    return out


def side(means: dict[str, float], tiers: dict[str, dict[str, float]], P: dict[str, Any], bl: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    R = P["rules"]
    perm = lambda f, v: SC.perm_p(f, v, R["n_perm"], R["seed"])
    oriented = {t: R["orientation"]["E_b"] * v for t, v in means.items()}
    out: dict[str, Any] = {"means": means, "tau_tier": {}}
    for r, tier in tiers.items():
        ts = sorted(t for t in means if t in tier)
        out["tau_tier"][r] = {"tasks": ts, "tau": SC.tau([tier[t] for t in ts], [means[t] for t in ts])}
    for rule in ("trained", "all"):
        for unit in ("nats_per_byte", "nats_per_token"):
            loss = {t: -v[rule][unit] for t, v in bl["tasks"].items()}
            for subset, names in (("text8", SC.WITHIN_TEXT["text8"]), ("all", tuple(tasks))):
                for r, tier in tiers.items():
                    ts = sorted(t for t in names if t in tier and t in loss)
                    out.setdefault("base_loss", {}).setdefault(f"{rule}_{unit}", {}).setdefault(subset, {})[r] = SC.reading_given_loss(tier, loss, oriented, ts, perm)
    return out


def addendum(out: dict[str, Any], tiers: dict[str, dict[str, float]], items: Path, P: dict[str, Any]) -> dict[str, Any]:
    """redteam-lead's 11-23 panel verdict, $0: both sides' consensus tau on the tasks the neurons read, with a two-sided
    task-permutation p (|tau| >= observed, E272's perm_p on the absolute value); each task's neuron mean against its own
    seed-0 Gaussian and the Gaussians' range; and the consensus reuse-tier tasks the neurons still read."""
    R, con = P["rules"], tiers["consensus"]
    ts = sorted(t for t in out["neurons"]["means"] if t in con)
    x, res = [con[t] for t in ts], {"tasks": ts}
    for k in ("neurons", "vpd"):
        y = [out[k]["means"][t] for t in ts]
        res[k] = {"tau": SC.tau(x, y), "p_two_sided": SC.perm_p(lambda w: abs(SC.tau(x, w)), y, R["n_perm"], R["seed"])}
    rnd = {json.loads(p.read_text())["task"]: json.loads(p.read_text())["E_b_K"] for p in sorted(items.glob("panel__*__random_*.json"))}
    lo, hi = min(rnd.values()), max(rnd.values())
    m = out["neurons"]["means"]
    res["gaussians"] = {"min": lo, "max": hi, "per_task": {t: {"mean": m[t], "own_random": rnd[t], "ratio": m[t] / rnd[t], "within_range": lo <= m[t] <= hi} for t in sorted(m)}}  # fmt: skip
    res["gaussians"]["n_within_range"] = sum(v["within_range"] for v in res["gaussians"]["per_task"].values())
    res["reuse_tier_read"] = sorted(t for t in ts if con[t] == 3.0)
    res["reuse_tier_dropped"] = sorted(t for t in con if con[t] >= 2.5 and t in out["tasks"] and t not in ts)
    return res


def main(items: Path) -> None:
    P = json.loads((E272 / "predictions.json").read_text())
    tiers, bl = SC.raters(P, "raters_wide"), json.loads((E272 / "wide" / "base_loss.json").read_text())
    vpd_seeds = SC.gather(E272 / "wide")["kl_hi"]["E_b"]
    tasks = sorted(vpd_seeds)
    sides = {"vpd": SC.task_means(vpd_seeds), "neurons": SC.task_means(neuron_seeds(items))}
    out = {"experiment": "E277 amendment 1 panel score", "tasks": tasks,
           "readability": {k: v[1] for k, v in sides.items()}} | {k: side(v[0], tiers, P, bl, tasks) for k, v in sides.items()}  # fmt: skip
    (HERE / "results" / "panel_score.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (HERE / "results" / "panel_addendum.json").write_text(json.dumps(addendum(out, tiers, items, P), indent=1, default=str) + "\n")
    for k in sides:
        c = out[k]["tau_tier"]["consensus"]
        b = out[k]["base_loss"]["trained_nats_per_byte"]["all"]["consensus"]
        print(f"[{k}] {len(out[k]['means'])} tasks; E_b's consensus tau {c['tau']:.3f} ({len(c['tasks'])} tasks); trained nats/byte, all: "
              f"tau {b['tau']:.3f} partial on base loss {b['partial']:.3f} p {b['p_partial']:.4f} (n {len(b['tasks'])})")  # fmt: skip
    print(f"wrote {HERE / 'results' / 'panel_score.json'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "items")
