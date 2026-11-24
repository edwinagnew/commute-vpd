"""P2.78 (regating-lead, 2026-11-24; descriptive, redteam M8): the random update's tau on 3-seed means, like the trained.

The wide KL 'hi' set's E_b over K has three trained seeds per task, but E272 read one Gaussian per task, matched to
seed 0 (eb.py's `random_file`), so the paper's -0.29 random tau was on one draw against the trained -0.69 on 3-seed
means. eb.py with EB_RANDOM_SEEDS=1,2 adds the Gaussians of seeds 1 and 2 to the same results/wide/eb/TASK.json. This
scores, with score.py's functions unchanged (tau-b, partial tau, K_MIN, raters_wide consensus tiers):
  - trained: tau of the 3-seed task means with the tier (summary.json's -0.69 on all 12 tasks, checked);
  - random one draw: tau of the seed-0 Gaussian (-0.29, two-sided scipy p as reported);
  - random 3 seeds: tau of the mean of the three Gaussians, and the rule filed in predictions_p278.json "rule":
    on the tasks with a trained mean and all three Gaussians, the gap (random tau minus trained tau) survives when it
    is at least the filed threshold; "no call" with fewer than min_tasks tasks or a nan tau.
score.py stays frozen; this writes results/wide/random_3seed.json.

    ~/commute-agents/fleet/cpu-run --mem 2G -t 10 python3 experiments/convergence_panel/random_3seed.py [--eb DIR --out FILE]
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from scipy.stats import kendalltau

from commute.jobs import load_experiment

RUN: Any = load_experiment("convergence_panel", "run")
SC: Any = load_experiment("convergence_panel", "score")
WIDE = RUN.RESULTS / "wide"
SEED = re.compile(r"_s([0-9]+)_step")


def values(eb: Path) -> dict[str, dict[str, dict[int, float]]]:
    """task -> "trained"/"random" -> seed -> E_b over K (nan with |K| < K_MIN, score.py's rule)."""
    out: dict[str, Any] = {}
    for p in sorted(eb.glob("*.json")):
        d = json.loads(p.read_text())
        if "task" not in d:  # windows_manifest.json
            continue
        for f, rec in d["states"].items():
            side = "random" if f.startswith("random/") else "trained"
            seed = int(m.group(1)) if (m := SEED.search(Path(f).name)) else -1
            assert seed in (0, 1, 2), f"{p.name}: no seed in {f}"
            out.setdefault(d["task"], {}).setdefault(side, {})[seed] = SC.e_b(eb, d["task"], f, rec)
    return out


def mean3(by_seed: dict[int, float]) -> float:
    """The 3-seed mean; nan unless seeds 0, 1 and 2 are all there and finite (score.task_means' rule)."""
    v = [by_seed.get(s, math.nan) for s in (0, 1, 2)]
    return fmean(v) if all(math.isfinite(x) for x in v) else math.nan


def taus(tier: dict[str, float], y: dict[str, float]) -> dict[str, Any]:
    ts = sorted(t for t in tier if t in y and math.isfinite(y[t]))
    x, v = [tier[t] for t in ts], [y[t] for t in ts]
    res: Any = kendalltau(x, v) if len(ts) >= 2 else None
    p = float(res[1]) if res is not None else math.nan
    return {"tasks": ts, "n": len(ts), "tau": SC.tau(x, v), "p_two_sided": p}


def gap_call(tier: dict[str, float], trained: dict[str, float], random3: dict[str, float], rule: dict[str, Any]) -> dict[str, Any]:
    """The filed rule: on the tasks with both, gap = tau(random 3-seed) - tau(trained 3-seed); survives at >= threshold."""
    ts = sorted(t for t in tier if math.isfinite(trained.get(t, math.nan)) and math.isfinite(random3.get(t, math.nan)))
    x = [tier[t] for t in ts]
    tt, tr = SC.tau(x, [trained[t] for t in ts]), SC.tau(x, [random3[t] for t in ts])
    out = {"tasks": ts, "n": len(ts), "tau_trained": tt, "tau_random": tr, "gap": tr - tt, "threshold": rule["threshold"]}
    if len(ts) < rule["min_tasks"] or not (math.isfinite(tt) and math.isfinite(tr)):
        return out | {"call": f"no call ({len(ts)} tasks, taus {tt}, {tr})"}
    partial = SC.partial_tau(x, [trained[t] for t in ts], [random3[t] for t in ts])
    return out | {"partial_trained_given_random": partial, "call": "survives" if tr - tt >= rule["threshold"] else "does not survive"}


def main() -> None:
    a = sys.argv[1:]
    eb = Path(a[a.index("--eb") + 1]) if "--eb" in a else WIDE / "eb"
    dest = Path(a[a.index("--out") + 1]) if "--out" in a else WIDE / "random_3seed.json"
    P, rule = json.loads(RUN.PRED.read_text()), json.loads((WIDE / "predictions_p278.json").read_text())["rule"]
    tier, v = SC.raters(P, "raters_wide")["consensus"], values(eb)
    trained = {t: mean3(d.get("trained", {})) for t, d in v.items()}
    one = {t: d.get("random", {}).get(0, math.nan) for t, d in v.items()}
    random3 = {t: mean3(d.get("random", {})) for t, d in v.items()}
    summ = json.loads((RUN.RESULTS / "summary.json").read_text())["wide"]["kl_hi"]["hope1"]["E_b"]["consensus"]["tau"]
    out = {"cell": "P2.78", "eb_dir": str(eb), "per_task": {t: {"tier": tier.get(t), "trained": d.get("trained"), "random": d.get("random")} for t, d in v.items()},
           "trained": taus(tier, trained) | {"summary_tau": summ}, "random_one_draw": taus(tier, one), "random_3seed": taus(tier, random3),
           "rule": gap_call(tier, trained, random3, rule)}  # fmt: skip
    dest.write_text(json.dumps(out, indent=1, default=str))
    for k in ("trained", "random_one_draw", "random_3seed"):
        print(f"{k:16s} tau {out[k]['tau']:.3f} p2 {out[k]['p_two_sided']:.3f} (n {out[k]['n']})")
    print(f"control: trained tau here {out['trained']['tau']:.6f}, summary.json {summ:.6f}")
    r = out["rule"]
    print(f"rule: gap {r['gap']:.3f} (random {r['tau_random']:.3f} - trained {r['tau_trained']:.3f}, n {r['n']}) vs {r['threshold']}: {r['call']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
