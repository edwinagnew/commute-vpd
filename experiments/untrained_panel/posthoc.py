"""E279 post-hoc checks (cell P2.66; redteam-lead, ~/commute-data/agents/regating-lead/note-2026-11-23-redteam-e279-prerun.md),
computed after the filed score, descriptive, no call. Beside the filed numbers in results/summary.json, never replacing them:

  1  statistics 1-3 without category and sentiment (their "lo" is the label's log-odds over the vocabulary, not
     E272's two-label contrast the other tasks' units match)
  2  the tasks that drop (a seed's gain under MIN_GAIN), with their consensus tiers and their seeds' gains
  3  the smallest p of the four filed tests, with the count (4, uncorrected)
  4  statistic 4 (tau of E_b over K with the share) with E_b's task means rebuilt from E272's wide/eb files: all states
     (the control, which must repeat the filed tau), and only states with |K| >= 1,000 (dropped from both readings'
     task means, and from E_b's alone); the filing's own floor is |K| < 100 (convergence_panel score.py K_MIN)
  5  the readability ratio (between-task over within-task sd) without squad and summary, the two means outside [0, 1]
  6  the Gaussians' range of paired shares, and which task means lie above it, below it, or inside it

CPU, no model: ~/commute-agents/fleet/cpu-run --mem 6G python3 experiments/untrained_panel/posthoc.py -> results/posthoc.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.jobs import load_experiment
from commute.local_faith import MIN_GAIN

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUN: Any = load_experiment("convergence_panel", "run")  # E272: PRED, RESULTS
SC: Any = load_experiment("convergence_panel", "score")  # raters, task_means
PH: Any = load_experiment("convergence_panel", "posthoc_checks")  # taus
LABEL_TASKS = ("category", "sentiment")
K_BIG = 1000
SEEDS = (0, 1, 2)


def statistic4(reads: dict[str, Any], eb: dict[str, dict[str, Any]], k_min: int, R: dict[str, Any]) -> dict[str, Any]:
    """Statistic 4 on the states with |K| >= k_min: E_b's task mean over those states against the share's task mean
    over the same states (`both`) and against the filed share means over all three seeds (`E_b_only`)."""
    keep = {t: [s for s in SEEDS if eb[t][d[f"s{s}"]["file"]]["size_K"] >= k_min] for t, d in reads.items()}
    e = {t: fmean(eb[t][reads[t][f"s{s}"]["file"]]["E_b_K"] for s in ss) for t, ss in keep.items() if ss}
    share = {t: [float(reads[t][f"s{s}"]["share"]) for s in ss] for t, ss in keep.items() if ss}
    both, _ = SC.task_means(share)
    filed, _ = SC.task_means({t: [float(d[f"s{s}"]["share"]) for s in SEEDS] for t, d in reads.items()})
    return {"k_min": k_min, "kept_seeds": keep, "E_b_means": e,
            "both": PH.taus(e, both, None, R), "E_b_only": PH.taus(e, filed, None, R)}  # fmt: skip


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())
    S = json.loads((RESULTS / "summary.json").read_text())
    P272 = json.loads(RUN.PRED.read_text())
    R, tier = P272["rules"], SC.raters(P272, "raters_wide")["consensus"]
    assert (R["n_perm"], R["seed"]) == (S["n_perm"], S["seed"])
    tasks = P["settings"]["tasks"]
    reads = {t: json.loads((RESULTS / "read" / f"{t}.json").read_text()) for t in tasks}
    means = S["means"]
    S272 = json.loads((RUN.RESULTS / "summary.json").read_text())["wide"]["kl_hi"]["means"]
    loss = {t: -v["trained"]["nats_per_byte"] for t, v in json.loads((RUN.RESULTS / "wide" / "base_loss.json").read_text())["tasks"].items()}
    out: dict[str, Any] = {"experiment": "E279", "post_hoc": True, "note": "redteam-lead's checks, after the filed score; descriptive, no call"}

    m7 = {t: v for t, v in means.items() if t not in LABEL_TASKS}
    out["1_without_label_tasks"] = {"dropped": list(LABEL_TASKS), "1_tiers": PH.taus(tier, m7, None, R),
                                    "2_given_base_loss": PH.taus(tier, m7, loss, R), "3_given_kl": PH.taus(tier, m7, S272["kl"], R)}  # fmt: skip

    miss = S["readability"]["missing"]
    out["2_missing"] = {t: {"tier": tier[t], "gains": [reads[t][f"s{s}"]["gain"] for s in SEEDS],
                            "seeds_under_min_gain": [s for s in SEEDS if reads[t][f"s{s}"]["gain"] < MIN_GAIN]} for t in miss}  # fmt: skip
    for t in miss:
        out["2_missing"][t]["shares"] = S["seeds"][t]
    out["2_missing_min_gain"] = MIN_GAIN
    out["2_tiers_read"] = {t: tier[t] for t in sorted(means)}

    outside = ("squad", "summary")  # the two task means outside [0, 1]
    _, info7 = SC.task_means({t: v for t, v in S["seeds"].items() if t in means and t not in outside})
    out["5_readability_without_outside"] = {"dropped": list(outside), "filed": S["readability"], "without": info7}
    gs = [v for v in S["gauss"]["share_paired"].values() if isinstance(v, (int, float))]
    lo, hi = min(gs), max(gs)
    out["6_gaussian_range"] = {"min": lo, "max": hi, "n_gaussians": len(gs), "n_tasks": len(means),
                               "above_max": sorted(t for t, v in means.items() if v > hi),
                               "below_min": sorted(t for t, v in means.items() if v < lo),
                               "inside": sorted(t for t, v in means.items() if lo <= v <= hi)}  # fmt: skip

    ps = {"1_tiers": S["1_tiers"]["p"], "2_given_base_loss": S["2_given_base_loss"]["p_partial"],
          "3_given_kl": S["3_given_kl"]["p_partial"], "4_E_b": S["4_E_b"]["p"]}  # fmt: skip
    k = min(ps, key=ps.__getitem__)
    out["3_smallest_p"] = {"p": ps, "smallest": k, "smallest_p": ps[k], "n_tests": len(ps), "corrected": False}

    eb = {t: json.loads((RUN.RESULTS / "wide" / "eb" / f"{t}.json").read_text())["states"] for t in tasks}
    read9 = {t: reads[t] for t in means}
    control = statistic4(read9, eb, SC.K_MIN, R)
    assert set(control["E_b_means"]) == set(means) and all(abs(control["E_b_means"][t] - S272["E_b"][t]) < 1e-12 for t in means)
    assert control["E_b_only"]["tau"] == S["4_E_b"]["tau"] and control["E_b_only"]["p"] == S["4_E_b"]["p"]
    big = statistic4(read9, eb, K_BIG, R)
    big["states_dropped"] = {t: [s for s in SEEDS if s not in big["kept_seeds"][t]] for t in means}
    big["states_dropped"] = {t: v for t, v in big["states_dropped"].items() if v}
    big["size_K_dropped"] = {t: [eb[t][reads[t][f"s{s}"]["file"]]["size_K"] for s in v] for t, v in big["states_dropped"].items()}
    big["n_tasks"] = len(big["E_b_means"])
    out["4_E_b"] = {"control_all_states": control, "K_at_least_1000": big}

    (RESULTS / "posthoc.json").write_text(json.dumps(out, indent=1, default=str))
    for key, o in out["1_without_label_tasks"].items():
        if isinstance(o, dict):
            print(f"[1 {key}] n {o['n']} tau {o['tau']:.4f} p {o['p']:.4f}" + (f"; partial {o['partial']:.4f} p {o['p_partial']:.4f}" if "partial" in o else ""))
    for t, v in out["2_missing"].items():
        print(f"[2 {t}] tier {v['tier']} gains {[round(g, 4) for g in v['gains']]} under {MIN_GAIN}: seeds {v['seeds_under_min_gain']}")
    print(f"[3] smallest p {ps[k]:.4f} ({k}) of {len(ps)} tests, uncorrected")
    for name, o in (("control", control), ("|K|>=1000", big)):
        print(f"[4 {name}] both: n {o['both']['n']} tau {o['both']['tau']:.4f} p {o['both']['p']:.4f}; "
              f"E_b only: n {o['E_b_only']['n']} tau {o['E_b_only']['tau']:.4f} p {o['E_b_only']['p']:.4f}")  # fmt: skip
    r = out["5_readability_without_outside"]["without"]
    print(f"[5] without squad, summary: between sd {r['between_sd']:.4f} within sd {r['within_sd']:.4f}")
    g = out["6_gaussian_range"]
    print(f"[6] Gaussians {g['min']:.4f} to {g['max']:.4f}; above {g['above_max']}, below {g['below_min']}, inside {g['inside']}")
    print(f"[4] dropped states {big['states_dropped']} |K| {big['size_K_dropped']}; tasks left {big['n_tasks']}")
    print(f"wrote {RESULTS / 'posthoc.json'}")


if __name__ == "__main__":
    main()
