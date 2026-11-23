"""E272 (cell P2.66) statistics, CPU, no model: hope 1 (each reading's task means against the raters' tiers) and
hope 2 (agreement among the readings against agreement among the baselines). Rules: results/predictions.json "rules";
writes results/summary.json and prints every number. Unit = task: a task's value is its seeds' mean at one matched
set; a task with a missing seed value is missing for that measure (reported, never read as 0).

    ~/commute-agents/fleet/cpu-run python3 experiments/convergence_panel/score.py [--results DIR]
--results reads predictions.json, dropped.json, read/ and eb/ from DIR (a stub) instead of results/, and writes
DIR/summary.json. Refuses unless dropped.json (run.py select) predates every read and E_b file (Amendment 3).
Amendment 5: the filed nine are scored as before (top level of summary.json, raters "raters"); when DIR/wide/dropped.json
exists the 18-task wide set is scored beside them from DIR/wide/{dropped.json,read,eb} on "raters_wide" (key "wide").
The critical tau-b is exact up to n = 9 and from 200,000 seeded random orderings from n = 10; each H1 entry also records
the raw call's power at true tau 0.3 and 0.5 on the tier codes kept (power_raw). Scored: the nine first, then the wide.
Amendment 8: the mixture is read at the size "hi" picks (mix_selection.json, which must predate every mix read).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from functools import cache
from itertools import combinations, permutations
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TIER = {"reuse": 3.0, "middle": 2.0, "new": 1.0}
SETS = ("kl_hi", "kl_lo", "size_hi", "size_lo")
MEASURES = ("share", "steering", "erank", "E_b", "kl", "adl", "size", "prior")
CRIT_DRAWS, CRIT_SEED, POWER_DRAWS = 200_000, 272, 20_000  # Amendment 5: random orderings from n = 10 (power_wide.py's seed)


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def value(rec: dict[str, Any], m: str) -> float:
    v = rec["adl"]["D"] if m == "adl" else rec.get(m)
    return float(v) if v is not None and finite(v) else math.nan


def drops(res: Path) -> dict[str, Any]:
    """Drops are fixed before any read (Amendment 3): dropped.json, written by `run.py select`, predates every read and
    E_b file, or nothing is scored."""
    p = res / "dropped.json"
    if not p.is_file():
        sys.exit(f"{p} is missing: run.py select writes it before any read")
    d = json.loads(p.read_text())
    early = [f.name for f in [*(res / "read").glob("*.json"), *(res / "eb").glob("*.json")] if f.stat().st_mtime < d["written"]]
    if early:
        sys.exit(f"read or E_b files older than dropped.json, so the drops were not fixed before them: {early}")
    return d


def records(res: Path) -> list[tuple[str, dict[str, Any]]]:
    """(task, state record with its E_b) for every read state, from res/read (and res/eb for E_b)."""
    out = []
    for p in sorted((res / "read").glob("*.json")):
        d = json.loads(p.read_text())
        eb = res / "eb" / p.name
        ebv = json.loads(eb.read_text())["states"] if eb.is_file() else {}
        out += [(d["task"], rec | {"E_b": ebv.get(f, {}).get("E_b_K", math.nan)}) for f, rec in d["states"].items()]
    return out


def gather(res: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """set -> measure -> task -> seed values (Amendment 4's "mix" and "trajectory" picks are not in a set)."""
    out: dict[str, Any] = {s: {} for s in SETS}
    for task, rec in records(res):
        for s in (s for s in rec["picked"] if s in SETS):
            for m in MEASURES:
                out[s].setdefault(m, {}).setdefault(task, []).append(value(rec, m))
    return out


# ----------------------------------------------------------------------------- Amendment 4 (descriptive, not H1/H2)


def rho(x: list[float], y: list[float]) -> float:
    """Spearman rho (average ranks for ties); nan when either side is constant."""
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(set(x)) > 1 and len(set(y)) > 1 else math.nan


def critical_rho(n: int, alpha: float) -> float:
    """The smallest rho an exact one-sided permutation test calls at alpha, n untied points (all n! orderings)."""
    r = np.sort([rho(list(range(n)), list(p)) for p in permutations(range(n))])
    p_ge = 1 - np.searchsorted(r, r - 1e-9, "left") / len(r)
    return float(r[p_ge <= alpha].min())


MIX_MATCH = "size_hi"  # Amendment 8: the mixture is read at the size "hi" target (KL "hi" had no mix state)


def mix_drops(res: Path) -> dict[str, Any] | None:
    """Amendment 8: the mix picks (mix_selection.json, `run.py select mix`) predate every mix read, or nothing is scored;
    None when neither the picks nor a mix read exist."""
    p, reads = res / "mix_selection.json", sorted((res / "read").glob("mix*.json"))
    if not p.is_file():
        return None if not reads else sys.exit(f"{p} is missing but mix reads exist: run.py select mix writes it first")
    d = json.loads(p.read_text())
    if early := [f.name for f in reads if f.stat().st_mtime < d["written"]]:
        sys.exit(f"mix reads older than mix_selection.json, so the mix picks were not fixed before them: {early}")
    return d


def mix_values(res: Path, S: dict[str, Any]) -> dict[str, dict[float, list[float]]]:
    """measure -> mix fraction -> seed values: the "mix" picks of mixNN at NN/100, and the size "hi" states of the two
    mixed tasks at 0 and 1 (their own panel runs; Amendment 8)."""
    frac = {S["mix_tasks"][0]: 0.0, S["mix_tasks"][1]: 1.0} | {f"mix{round(100 * f)}": f for f in S["mix_fracs"][1:-1]}
    out: dict[str, Any] = {m: {f: [] for f in S["mix_fracs"]} for m in MEASURES}
    for task, rec in records(res):
        label = MIX_MATCH if task in S["mix_tasks"] else "mix"
        if task in frac and label in rec["picked"]:
            for m in MEASURES:
                out[m][frac[task]].append(value(rec, m))
    return out


def mixture(vals: dict[str, dict[float, list[float]]], R: dict[str, Any], n_seeds: int) -> dict[str, Any]:
    """Amendment 4, per measure: Spearman rho of the seed means over the mix fractions; 'monotone' when |rho| is at least
    the exact one-sided critical rho at alpha for that many fractions, with the sign said against the expected one
    (-orientation: the fraction is the low-reuse task's share); 'not monotone' under it; 'no call' with a missing
    fraction (a seed missing or non-finite) or rho undefined (a constant reading). The 15-point rho is beside it. The
    matched variable (size since Amendment 8) is reported, not called; KL, now free, is called like a reading."""
    fr = sorted(next(iter(vals.values())))
    crit, out = critical_rho(len(fr), R["alpha"]), {}
    for m, by in vals.items():
        ok = [f for f in fr if len(by[f]) == n_seeds and all(math.isfinite(x) for x in by[f])]
        means = [fmean(by[f]) for f in ok]
        o: dict[str, Any] = {"fractions": ok, "means": means, "critical_rho": crit, "expected_sign": -R["orientation"][m]}
        o["rho"] = rho(ok, means) if len(ok) == len(fr) else math.nan
        o["rho_all_points"] = rho([f for f in ok for _ in by[f]], [x for f in ok for x in by[f]]) if ok else math.nan
        if m == MIX_MATCH.split("_")[0]:
            o["call"] = "matched (not read)"
        elif len(ok) < len(fr) or not finite(o["rho"]):
            o["call"] = f"no call ({len(fr) - len(ok)} fractions missing)" if len(ok) < len(fr) else "no call (rho undefined)"
        elif abs(o["rho"]) >= crit - 1e-12:
            o["call"] = "monotone, expected sign" if math.copysign(1, o["rho"]) == o["expected_sign"] else "monotone, opposite sign"
        else:
            o["call"] = "not monotone"
        out[m] = o
    return out


def trajectory_values(res: Path) -> dict[str, dict[str, list[float]]]:
    """task -> measure -> values of its "trajectory" picks in step order."""
    by: dict[str, list[dict[str, Any]]] = {}
    for task, rec in records(res):
        if "trajectory" in rec["picked"]:
            by.setdefault(task, []).append(rec)
    return {t: {m: [value(r, m) for r in sorted(rs, key=lambda r: r["step"])] for m in MEASURES} for t, rs in by.items()}


def jump_call(v: list[float], sd: float, mult: float) -> dict[str, Any]:
    """Amendment 4, one reading along one run: band = mult x the reading's pooled seed sd on the panel's KL 'hi' set.
    'stable' when its range is within the band; 'jump' when it moves out and back (range over the band, first-to-last
    change within it) or any step between adjacent points goes against the first-to-last change by more than the
    band; else 'monotone'. 'missing points' with a non-finite point; 'no call' without a seed sd."""
    if len(v) < 2 or not all(math.isfinite(x) for x in v):
        return {"values": v, "call": "missing points"}
    if not finite(sd):
        return {"values": v, "call": "no call (no seed sd)"}
    band, d, total = mult * sd, list(np.diff(v)), v[-1] - v[0]
    o = {"values": v, "band": band, "max_step_share": max(abs(x) for x in d) / sum(abs(x) for x in d) if any(d) else math.nan}
    if max(v) - min(v) <= band:
        return o | {"call": "stable"}
    if abs(total) <= band or any(x * math.copysign(1, total) < -band for x in d):
        return o | {"call": "jump"}
    return o | {"call": "monotone"}


def amendment4(res: Path, P: dict[str, Any], kl_hi: Any) -> dict[str, Any]:
    """The two descriptive readouts: the mixture's rho per measure, and per trajectory run and reading the jump call
    against the reading's seed sd on the KL 'hi' set (nan when that set was not read)."""
    S, R = P["settings"], P["rules"]
    sd = lambda m: kl_hi["readability"].get(m, {}).get("within_sd", math.nan) if isinstance(kl_hi, dict) else math.nan
    by_task = trajectory_values(res)
    traj = {t: {m: jump_call(v, sd(m), R["amendment4"]["jump_mult"]) for m, v in by_task[t].items()} if t in by_task
            else "too few points (select: fewer saves over the floor than points) or not read" for t in S["trajectory_tasks"]}  # fmt: skip
    vals = mix_values(res, S)
    kl = {str(f): {"kl": vals["kl"][f], "size": vals["size"][f]} for f in S["mix_fracs"]}  # Amendment 8: each pick's KL beside it
    return {"mixture": mixture(vals, R, len(S["seeds"])), "mix_match": MIX_MATCH, "mix_picks": kl, "mix_selection": mix_drops(res), "trajectory": traj}


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


def perm_p(f: Any, y: list[float], n: int, seed: int) -> float:
    """One-sided task-permutation p of f(reading) >= observed, the reading's task values permuted (tier and prior held):
    (1 + #perm >= obs) / (1 + n)."""
    obs, rng = f(y), np.random.default_rng(seed)
    if not finite(obs):
        return math.nan
    hits = sum(f(list(rng.permutation(y))) >= obs - 1e-12 for _ in range(n))
    return (1 + hits) / (1 + n)


@cache
def critical_tau(x: tuple[float, ...], alpha: float, draws: int = CRIT_DRAWS, seed: int = CRIT_SEED) -> float:
    """The smallest tau-b a one-sided permutation test calls at alpha, tier vector x against an untied reading: all n!
    orderings for n <= 9 (exact, as Amendment 3 filed it), else `draws` seeded random orderings (Amendment 5)."""
    if len(x) <= 9:
        perms = np.array(list(permutations(range(len(x)))), dtype=np.int8)
    else:
        perms = np.argsort(np.random.default_rng(seed).random((draws, len(x))), 1).astype(np.int16)
    t = np.sort(np.concatenate([taub_rows(x, p) for p in np.array_split(perms, max(1, len(perms) // 20_000))]))
    p_ge = (len(t) - np.searchsorted(t, t - 1e-9, "left")) / len(t)  # a count ratio: 1 - k/n put an exact 6/120 above 0.05
    called = t[p_ge <= alpha]
    return float(called.min()) if called.size else math.nan  # nan: no ordering of these tier codes reaches alpha


def taub_rows(x: tuple[float, ...], Y: np.ndarray) -> np.ndarray:
    """Kendall tau-b of the tier vector x against each untied row of Y."""
    xa, (i, j) = np.asarray(x), np.triu_indices(len(x), 1)
    n0, n1 = len(i), sum(c * (c - 1) / 2 for c in Counter(x).values())
    return (np.sign(xa[i] - xa[j]) * np.sign(Y[:, i] - Y[:, j])).sum(1) / math.sqrt((n0 - n1) * n0)


@cache
def power(x: tuple[float, ...], target: float, crit: float, draws: int = POWER_DRAWS, seed: int = CRIT_SEED) -> float:
    """Amendment 5: power of the raw tau call at a true mean tau-b of `target` (Amendment 3's model): reading = tier code
    + sigma N(0, 1) per task, sigma bisected so the mean tau-b over the draws is the target; the share of draws whose
    tau-b reaches crit."""
    xa, E, lo, hi = np.asarray(x), np.random.default_rng(seed).standard_normal((draws, len(x))), 0.01, 50.0
    for _ in range(30):
        sig = math.sqrt(lo * hi)
        lo, hi = (sig, hi) if taub_rows(x, xa + sig * E).mean() > target else (lo, sig)
    return float((taub_rows(x, xa + math.sqrt(lo * hi) * E) >= crit - 1e-9).mean())


def interval(x: list[float], y: list[float], n: int, seed: int) -> list[float]:
    """90% percentile interval of tau over task bootstraps (draws with a constant side are skipped)."""
    rng, idx, got = np.random.default_rng(seed), np.arange(len(x)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        t = tau([x[i] for i in b], [y[i] for i in b])
        got += [t] if finite(t) else []
    return [float(np.quantile(got, 0.05)), float(np.quantile(got, 0.95))] if got else [math.nan, math.nan]


def support(tier: dict[str, float], z: dict[str, float]) -> bool:
    """Common support on the prior z: its ranges over the reuse-side tasks (tier code above the middle code 2) and the
    new-side tasks (below 2) overlap. A rater's own tiers: reuse against new; the consensus: 2.5-3 against 1-1.5."""
    hi = [z[t] for t in tier if tier[t] > 2 and t in z]
    lo = [z[t] for t in tier if tier[t] < 2 and t in z]
    return bool(hi and lo) and min(hi) <= max(lo) and min(lo) <= max(hi)


def hope1(tier: dict[str, float], means: dict[str, float], prior: dict[str, float], R: dict[str, Any]) -> dict[str, Any]:
    ts = sorted(t for t in tier if t in means)
    if len(ts) < R["min_tasks"]:
        return {"call": f"too few tasks ({len(ts)} < {R['min_tasks']})", "tasks": ts}
    x, y, z = [tier[t] for t in ts], [means[t] for t in ts], [prior.get(t, math.nan) for t in ts]
    out: dict[str, Any] = {"tasks": ts, "tau": tau(x, y), "p_perm": perm_p(lambda v: tau(x, v), y, R["n_perm"], R["seed"])}
    xs = tuple(sorted(x))  # the critical tau-b and power depend only on the tier codes kept
    crit = critical_tau(xs, R["alpha"])
    out |= {"interval90": interval(x, y, R["n_boot"], R["seed"]), "critical_tau": crit, "tau_prior": tau(x, z)}
    out["power_raw"] = {str(v): power(xs, v, crit) for v in (0.3, 0.5)}  # Amendment 5: at the n kept, at score time
    loo = [tau([tier[t] for t in ts if t != d], [means[t] for t in ts if t != d]) for d in ts]
    out["leave_one_out"] = [min(loo), max(loo)] if all(finite(v) for v in loo) else [math.nan, math.nan]
    out["prior_support"] = support({t: tier[t] for t in ts}, prior)
    part = (lambda v: partial_tau(x, v, z)) if out["prior_support"] else (lambda v: math.nan)
    out |= {"tau_partial_prior": part(y), "p_partial": perm_p(part, y, R["n_perm"], R["seed"])}
    return out | {"call": call1(out, R["alpha"])}


def call1(o: dict[str, Any], alpha: float) -> str:
    """Amendment 3: 'positive' iff the partial tau on the prior is > 0 at p_partial <= alpha, or the filed raw call is
    positive (tau > 0, p <= alpha) and tau beats the prior's own tau with the tier; with no common support on the prior
    a raw positive is 'confounded with the prior'; tau <= 0 is 'not positive'; else 'no call'."""
    t, part = o["tau"], o["tau_partial_prior"]
    raw = finite(t) and t > 0 and o["p_perm"] <= alpha
    if not o["prior_support"]:
        return "confounded with the prior" if raw else "not positive" if finite(t) and t <= 0 else "no call"
    if (finite(part) and part > 0 and o["p_partial"] <= alpha) or (raw and not t <= o["tau_prior"]):
        return "positive"
    return "not positive" if finite(t) and t <= 0 else "no call"


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


def null_w(w: float, m: int, n: int, draws: int, seed: int) -> dict[str, float]:
    """W against its own null, m independent random rankings of n tasks (Amendment 3: each side at its own m): the
    null's 95th percentile and the share of null draws below w."""
    rng = np.random.default_rng(seed)
    null = np.array([kendall_w([list(rng.permutation(n)) for _ in range(m)]) for _ in range(draws)])
    return {"W": w, "null95": float(np.quantile(null, 0.95)), "percentile": float((null < w).mean())}


def residual_w(cols: list[list[float]], covs: list[list[float]]) -> float:
    """W of the columns' ranks after a least-squares fit on the covariates' ranks (and a constant) is taken out; residuals
    rounded to 1e-9 so a column the covariates fit exactly ties throughout (W nan, not float noise read as agreement)."""
    A = np.column_stack([np.ones(len(cols[0]))] + [rankdata(c) for c in covs])
    return kendall_w([list(np.round(r - A @ np.linalg.lstsq(A, r, rcond=None)[0], 9)) for r in (rankdata(c) for c in cols)])


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
    side = lambda ms, f: fmean([f(v) for a, b in combinations([m for m in allm if m in ms], 2) if finite(v := out["pairwise_tau"][f"{a}|{b}"])] or [math.nan])
    out["mean_tau"] = {"readings": side(rd, float), "baselines": side(bl, float)}
    out["mean_abs_tau"] = {"readings": side(rd, abs), "baselines": side(bl, abs)}
    if len(ts) < R["min_tasks"] or len(rd) < 2 or len(bl) < 2:
        return out | {"call": f"no call: {len(ts)} tasks, {len(rd)} readable readings, {len(bl)} readable baselines"}
    pz = [t for t in ts if t in means["prior"] and t in means["size"]]
    out["W_readings_partial_prior_size"] = residual_w([[sign[m] * means[m][t] for t in pz] for m in rd], [[means[c][t] for t in pz] for c in ("prior", "size")])
    out["null_readings"] = null_w(out["W_readings"], len(rd), len(ts), R["n_perm"], R["seed"])
    out["null_baselines"] = null_w(out["W_baselines"], len(bl), len(ts), R["n_perm"], R["seed"])
    return out | {"call": call2(out["null_readings"], out["null_baselines"])}


def call2(r: dict[str, float], b: dict[str, float]) -> str:
    """Amendment 3: 'above' when the readings' W is over its own null's 95th percentile and its percentile in that null
    is above the baselines' percentile in theirs; 'not above' when it is not above; else 'no call'."""
    if not (finite(r["W"]) and finite(b["W"])):
        return "no call (W undefined)"
    if r["percentile"] <= b["percentile"]:
        return "not above"
    return "above" if r["W"] > r["null95"] else "no call (readings under their null)"


def raters(P: dict[str, Any], key: str = "raters") -> dict[str, dict[str, float]]:
    """Each rater's tier code per task and the consensus (their mean), from the filed `raters` block."""
    got = {name: {t: TIER[v] for t, v in r["tiers"].items()} for name, r in P[key].items()}
    ts = set.intersection(*[set(v) for v in got.values()])
    return got | {"consensus": {t: fmean(v[t] for v in got.values()) for t in ts}}


def score(data: dict[str, Any], P: dict[str, Any], dropped: dict[str, Any], key: str = "raters") -> dict[str, Any]:
    R, out = P["rules"], {"experiment": P["experiment"], "dropped": dropped}
    tiers = raters(P, key)
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


def show(s: str, o: dict[str, Any]) -> None:
    h2, g = o["hope2"], lambda d, k: d.get(k, math.nan)
    nr, nb = h2.get("null_readings", {}), h2.get("null_baselines", {})
    print(f"[{s}] W readings {h2['W_readings']:.3f} (pct {g(nr, 'percentile'):.3f}, null95 {g(nr, 'null95'):.3f}; partial on prior and size "
          f"{g(h2, 'W_readings_partial_prior_size'):.3f}) vs baselines {h2['W_baselines']:.3f} (pct {g(nb, 'percentile'):.3f}): {h2['call']}")
    for m, by in o["hope1"].items():
        print(f"[{s}] {m}: " + "; ".join(f"{r} tau {g(v, 'tau'):.3f} p {g(v, 'p_perm'):.3f} crit {g(v, 'critical_tau'):.3f} prior {g(v, 'tau_prior'):.3f} "
                                         f"partial {g(v, 'tau_partial_prior'):.3f} p {g(v, 'p_partial'):.3f}: {v['call']}" for r, v in by.items()))
    print(f"[{s}] unreadable: {o['unreadable']}")


def main() -> None:
    res = Path(sys.argv[sys.argv.index("--results") + 1]) if "--results" in sys.argv else RESULTS
    P = json.loads((res / "predictions.json").read_text())
    if not P.get("raters"):
        sys.exit("raters are not filed in predictions.json: the lead files them before scoring")
    out = score(gather(res), P, drops(res))  # the nine-task set, as Amendment 3 filed it; needs no wide file
    out["amendment4"] = amendment4(res, P, out["kl_hi"])
    wide = res / "wide"
    if (wide / "dropped.json").is_file():  # Amendment 5: the 18-task set beside it, its own drops, reads and raters
        if not P.get("raters_wide"):
            sys.exit("raters_wide are not filed in predictions.json: the wide set is not scored without them")
        out["wide"] = score(gather(wide), P, drops(wide), "raters_wide")
    (res / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    for s in SETS:
        if isinstance(out[s], dict):
            show(s, out[s])
    for f, v in out["amendment4"]["mix_picks"].items():
        print(f"[mixture picks at {MIX_MATCH}] fraction {f}: KL {[round(x, 4) for x in v['kl']]}, size {[round(x, 5) for x in v['size']]}")
    for m, o in out["amendment4"]["mixture"].items():
        print(f"[mixture] {m}: rho {o['rho']:.3f} (15 points {o['rho_all_points']:.3f}, critical {o['critical_rho']:.3f}, expected {o['expected_sign']:+d}): {o['call']}")
    for t, by in out["amendment4"]["trajectory"].items():
        print(f"[trajectory {t}] " + (by if isinstance(by, str) else "; ".join(f"{m} {o['call']}" for m, o in by.items())))
    for s in SETS:
        if isinstance(out.get("wide", {}).get(s), dict):
            show("wide " + s, out["wide"][s])
    print(f"wrote {res / 'summary.json'}")


if __name__ == "__main__":
    main()
