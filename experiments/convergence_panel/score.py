"""E_TBD (cell P2.66) statistics, CPU, no model: hope 1 (each reading's task means against the raters' tiers) and
hope 2 (agreement among the readings against agreement among the baselines). Rules: results/predictions.json "rules";
writes results/summary.json and prints every number. Unit = task: a task's value is its seeds' mean at one matched
set; a task with a missing seed value is missing for that measure (reported, never read as 0).

    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/score.py [--stub FILE]
--stub reads a made-up input in the summary's own shape (tests/stubs) instead of results/read and results/eb.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TIER = {"reuse": 3.0, "middle": 2.0, "new": 1.0}
SETS = ("kl_hi", "kl_lo", "size_hi", "size_lo")


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def value(rec: dict[str, Any], m: str) -> float:
    v = rec["adl"]["D"] if m == "adl" else rec.get(m)
    return float(v) if v is not None and finite(v) else math.nan


def gather() -> dict[str, dict[str, dict[str, list[float]]]]:
    """set -> measure -> task -> seed values, from results/read (and results/eb for E_b)."""
    out: dict[str, Any] = {s: {} for s in SETS}
    for p in sorted((RESULTS / "read").glob("*.json")):
        d = json.loads(p.read_text())
        eb = RESULTS / "eb" / p.name
        ebv = json.loads(eb.read_text())["states"] if eb.is_file() else {}
        for f, rec in d["states"].items():
            rec = rec | {"E_b": ebv.get(f, {}).get("E_b_K", math.nan)}
            for s in rec["picked"]:
                for m in ("share", "steering", "erank", "E_b", "kl", "adl", "size", "prior"):
                    out[s].setdefault(m, {}).setdefault(d["task"], []).append(value(rec, m))
    return out


def task_means(seeds: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, Any]]:
    """Task means where every seed is finite; the rest listed as missing. Readability: between-task sd of the means
    over the pooled within-task sd (sqrt of the mean within-task variance)."""
    ok = {t: v for t, v in seeds.items() if v and all(math.isfinite(x) for x in v)}
    means = {t: fmean(v) for t, v in ok.items()}
    within = math.sqrt(fmean([pvariance(v) for v in ok.values()])) if ok else math.nan
    between = math.sqrt(pvariance(list(means.values()))) if len(means) >= 2 else math.nan
    readable = len(means) >= 2 and between > 0 and (between > within)
    return means, {"between_sd": between, "within_sd": within, "readable": readable, "missing": sorted(set(seeds) - set(ok))}


def tau(x: list[float], y: list[float]) -> float:
    """Kendall tau-b (ties handled); nan when either side is constant or n < 2."""
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    res: Any = kendalltau(x, y)
    return float(res[0])


def partial_tau(x: list[float], y: list[float], z: list[float]) -> float:
    """Kendall's partial tau of x and y given z; nan when z is tied with x or y perfectly (denominator 0)."""
    txy, txz, tyz = tau(x, y), tau(x, z), tau(y, z)
    den = math.sqrt((1 - txz**2) * (1 - tyz**2)) if finite(txz) and finite(tyz) else 0.0
    return (txy - txz * tyz) / den if den > 0 and finite(txy) else math.nan


def perm_p(x: list[float], y: list[float], n: int, seed: int) -> float:
    """One-sided task-permutation p of tau(x, y) >= observed: (1 + #perm >= obs) / (1 + n)."""
    obs, rng = tau(x, y), np.random.default_rng(seed)
    if not finite(obs):
        return math.nan
    hits = sum(tau(list(rng.permutation(x)), y) >= obs - 1e-12 for _ in range(n))
    return (1 + hits) / (1 + n)


def interval(x: list[float], y: list[float], n: int, seed: int) -> list[float]:
    """90% percentile interval of tau over task bootstraps (draws with a constant side are skipped)."""
    rng, idx, got = np.random.default_rng(seed), np.arange(len(x)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        t = tau([x[i] for i in b], [y[i] for i in b])
        got += [t] if finite(t) else []
    return [float(np.quantile(got, 0.05)), float(np.quantile(got, 0.95))] if got else [math.nan, math.nan]


def support(tier: dict[str, float], z: dict[str, float]) -> bool:
    """Common support on z: the ranges of z over the top-tier and bottom-tier tasks overlap."""
    hi = [z[t] for t in tier if tier[t] == max(tier.values()) and t in z]
    lo = [z[t] for t in tier if tier[t] == min(tier.values()) and t in z]
    return bool(hi and lo) and min(hi) <= max(lo) and min(lo) <= max(hi)


def hope1(tier: dict[str, float], means: dict[str, float], prior: dict[str, float], R: dict[str, Any]) -> dict[str, Any]:
    ts = sorted(t for t in tier if t in means)
    if len(ts) < R["min_tasks"]:
        return {"call": f"too few tasks ({len(ts)} < {R['min_tasks']})", "tasks": ts}
    x, y = [tier[t] for t in ts], [means[t] for t in ts]
    out: dict[str, Any] = {"tasks": ts, "tau": tau(x, y), "p_perm": perm_p(x, y, R["n_perm"], R["seed"]), "interval90": interval(x, y, R["n_boot"], R["seed"])}
    loo = [tau([tier[t] for t in ts if t != d], [means[t] for t in ts if t != d]) for d in ts]
    out["leave_one_out"] = [min(loo), max(loo)] if all(finite(v) for v in loo) else [math.nan, math.nan]
    out["prior_support"] = support({t: tier[t] for t in ts}, prior)
    out["tau_partial_prior"] = partial_tau(x, y, [prior[t] for t in ts]) if out["prior_support"] else "no common support on the prior: raw only"
    t_, p_ = out["tau"], out["p_perm"]
    out["call"] = "no call" if not finite(t_) else "not positive" if t_ <= 0 else "positive" if p_ <= R["alpha"] else "no call"
    return out


def kendall_w(cols: list[list[float]]) -> float:
    """Kendall's W of m rankings over n tasks, ranks with ties averaged and the tie correction; nan when m < 2 or n < 2."""
    m, n = len(cols), len(cols[0]) if cols else 0
    if m < 2 or n < 2:
        return math.nan
    r = np.array([rankdata(c) for c in cols])
    S = float(((r.sum(0) - m * (n + 1) / 2) ** 2).sum())
    T = sum(float(((u := np.unique(c, return_counts=True)[1]) ** 3 - u).sum()) for c in cols)
    den = m**2 * (n**3 - n) - m * T
    return 12 * S / den if den > 0 else math.nan


def null_w(m: int, n: int, draws: int, seed: int) -> float:
    """95th percentile of W for m independent random rankings of n tasks."""
    rng = np.random.default_rng(seed)
    return float(np.quantile([kendall_w([list(rng.permutation(n)) for _ in range(m)]) for _ in range(draws)], 0.95))


def hope2(means: dict[str, dict[str, float]], readable: dict[str, bool], R: dict[str, Any], matched: str) -> dict[str, Any]:
    """W across the readable readings and across the readable baselines (the matched variable left out: it is constant
    by construction), each oriented by its filed sign, on the tasks every measure in the comparison has."""
    sign = R["orientation"]
    rd = [m for m in R["readings"] if readable.get(m)]
    bl = [m for m in R["baselines"] if readable.get(m) and m != matched]
    ts = sorted(set.intersection(*[set(means[m]) for m in rd + bl])) if rd + bl else []
    col = lambda m: [sign[m] * means[m][t] for t in ts]
    out = {"tasks": ts, "readings": rd, "baselines": bl, "W_readings": kendall_w([col(m) for m in rd]), "W_baselines": kendall_w([col(m) for m in bl])}
    allm = [m for m in R["readings"] + R["baselines"] if m in means]
    pair_ts = lambda a, b: sorted(set(means[a]) & set(means[b]))
    out["pairwise_tau"] = {f"{a}|{b}": tau([sign[a] * means[a][t] for t in pair_ts(a, b)], [sign[b] * means[b][t] for t in pair_ts(a, b)]) for a, b in combinations(allm, 2)}
    mean_abs = lambda ms: fmean([abs(out["pairwise_tau"][f"{a}|{b}"]) for a, b in combinations([m for m in allm if m in ms], 2) if finite(out["pairwise_tau"][f"{a}|{b}"])] or [math.nan])
    out["mean_abs_tau"] = {"readings": mean_abs(rd), "baselines": mean_abs(bl)}
    if len(ts) < R["min_tasks"] or len(rd) < 2 or len(bl) < 2:
        return out | {"call": f"no call: {len(ts)} tasks, {len(rd)} readable readings, {len(bl)} readable baselines"}
    out["W_null95"] = null_w(len(rd), len(ts), R["n_perm"], R["seed"])
    wr, wb = out["W_readings"], out["W_baselines"]
    out["call"] = "above" if wr > wb and wr > out["W_null95"] else "not above" if not wr > wb else "no call (readings under their null)"
    return out


def raters(P: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Each rater's tier code per task and the consensus (their mean), from the filed `raters` block."""
    got = {name: {t: TIER[v] for t, v in r["tiers"].items()} for name, r in P["raters"].items()}
    ts = set.intersection(*[set(v) for v in got.values()])
    return got | {"consensus": {t: fmean(v[t] for v in got.values()) for t in ts}}


def score(data: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    R, out = P["rules"], {"experiment": P["experiment"]}
    tiers = raters(P)
    out["consensus_tier"] = tiers["consensus"]
    for s in SETS:
        if not data.get(s):
            out[s] = "not read"
            continue
        means, info = {}, {}
        for m, seeds in data[s].items():
            means[m], info[m] = task_means(seeds)
        readable = {m: v["readable"] for m, v in info.items()}
        h1 = {m: {r: hope1(tiers[r], {t: R["orientation"][m] * v for t, v in means[m].items()}, means["prior"], R) for r in tiers}
              for m in R["readings"] if readable.get(m)}  # fmt: skip
        out[s] = {"readability": info, "means": means, "hope1": h1, "hope2": hope2(means, readable, R, s.split("_")[0])}
        out[s]["unreadable"] = {m: "no information (between-task spread not above seed spread)" for m in R["readings"] if not readable.get(m)}
    return out


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())
    stub = sys.argv[sys.argv.index("--stub") + 1] if "--stub" in sys.argv else None
    if stub is None and not P.get("raters"):
        sys.exit("raters are not filed in predictions.json: the lead files them before scoring")
    data = json.loads(Path(stub).read_text()) if stub else gather()
    if stub:
        P = P | json.loads(Path(stub).with_suffix(".raters.json").read_text())
    out = score(data, P)
    dest = (Path(stub).with_suffix(".summary.json")) if stub else RESULTS / "summary.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    for s in SETS:
        if isinstance(out[s], dict):
            h2 = out[s]["hope2"]
            print(f"[{s}] W readings {h2['W_readings']:.3f} vs baselines {h2['W_baselines']:.3f}: {h2['call']}")
            for m, by in out[s]["hope1"].items():
                print(f"[{s}] {m}: " + "; ".join(f"{r} tau {v.get('tau', math.nan):.3f} p {v.get('p_perm', math.nan):.3f} {v['call']}" for r, v in by.items()))
            print(f"[{s}] unreadable: {out[s]['unreadable']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
