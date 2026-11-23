"""E272 wide panel: critical tau-b (one-sided alpha 0.05, task permutation, 200k random permutations) and power at a
true mean tau-b of 0.3 and 0.5 (reading = tier code + sigma N(0,1), sigma bisected so the mean tau-b hits the target,
100k draws), n = 15-18, two tier vectors. Same generative model and tau-b formula as e272-amend2/power.py."""
import json

import numpy as np

rng = np.random.default_rng(272)
SPLITS = {"thirds": {15: (5, 5, 5), 16: (6, 5, 5), 17: (6, 6, 5), 18: (6, 6, 6)},
          "40/40/20": {15: (6, 6, 3), 16: (6, 6, 4), 17: (7, 7, 3), 18: (7, 7, 4)}}


def taub_many(x, Y, chunk=20_000):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x), 1)
    n0 = len(i)
    n1 = sum(c * (c - 1) / 2 for c in np.unique(x, return_counts=True)[1])
    sx = np.sign(x[i] - x[j])
    out = [(sx * np.sign(Y[s:s + chunk][:, i] - Y[s:s + chunk][:, j])).sum(1) for s in range(0, len(Y), chunk)]
    return np.concatenate(out) / np.sqrt((n0 - n1) * n0)


def critical(x, alpha=0.05, n=200_000):
    perms = np.argsort(rng.random((n, len(x))), 1).astype(np.int16)  # an untied reading's ranks, permuted
    t = np.sort(taub_many(x, perms))
    p_ge = 1 - np.searchsorted(t, t - 1e-9, "left") / len(t)
    return float(t[p_ge <= alpha].min())


def power(x, crit, target, draws=100_000):
    lo, hi, x = 0.01, 50.0, np.asarray(x, float)
    E = rng.standard_normal((draws, len(x)))
    for _ in range(40):
        sig = (lo * hi) ** 0.5
        m = taub_many(x, x + sig * E).mean()
        lo, hi = (sig, hi) if m > target else (lo, sig)
    t = taub_many(x, x + sig * E)
    return round(sig, 3), round(float(t.mean()), 4), round(float((t >= crit - 1e-9).mean()), 4)


for name, by_n in SPLITS.items():
    for n, (a, b, c) in by_n.items():
        x = [3] * a + [2] * b + [1] * c
        crit = critical(x)
        out = {"split": name, "n": n, "counts": [a, b, c], "critical_tau_b": round(crit, 4)}
        for target in (0.3, 0.5):
            sig, m, pw = power(x, crit, target)
            out[f"power_at_{target}"] = {"sigma": sig, "mean_tau": m, "power": pw}
        print(json.dumps(out), flush=True)
