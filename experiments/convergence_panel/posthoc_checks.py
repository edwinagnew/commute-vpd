"""E272 post-hoc checks (cell P2.66; plan reports/plans/2026-11-22_plan_convergence-panel.md, "2026-11-23 post-hoc
checks", filed before computing; user-approved, $0, descriptive, no call). On the wide KL "hi" set's 12 tasks:

(1) seed agreement: the mean pairwise cosine of each task's 3 seeds' updates, against a floor (each seed's cosine with
    the task's one Gaussian, stage 2's `random_direction` of the seed-0 state, rebuilt); its tau with the tiers (minus
    agreement: positive = reuse-tier tasks agree less), its tau with E_b, and E_b's partial tau on the tiers given it;
(2) steps to learn: the first logged step whose held gain reaches half the seed's final, mean over seeds, oriented
    as minus steps (fewer = more reuse); its tau with the tiers, and each reading's tau and partial given base loss.

CPU, no model: ~/commute-agents/fleet/cpu-run --mem 6G -t 20 python3 experiments/convergence_panel/posthoc_checks.py
-> results/wide/posthoc_checks.json
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from statistics import fmean
from typing import Any

import torch

from commute.jobs import load_experiment

RUN: Any = load_experiment("convergence_panel", "run")
SC: Any = load_experiment("convergence_panel", "score")
WIDE = RUN.RESULTS / "wide"
OUT = WIDE / "posthoc_checks.json"
READINGS = ("E_b", "share", "steering", "erank", "kl")


def cos(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """Cosine of two updates flattened over every key, in float64."""
    dot = sum(float((a[k].double() * b[k].double()).sum()) for k in a)
    na = math.sqrt(sum(float(a[k].double().square().sum()) for k in a))
    nb = math.sqrt(sum(float(b[k].double().square().sum()) for k in b))
    return dot / (na * nb)


def agreement(name: str, seed: int) -> dict[str, Any]:
    """The task's 3 pairwise seed cosines and each seed's cosine with the seed-0 state's Gaussian."""
    files = {
        s["seed"]: f
        for f, s in RUN.picked(name, "wide").items()
        if "kl_hi" in s["picked"]
    }
    assert sorted(files) == [0, 1, 2], f"{name}: kl_hi seeds {sorted(files)}"
    dW = {s: RUN.load_dW(f) for s, f in files.items()}
    assert all(dW[s].keys() == dW[0].keys() for s in dW)
    g = RUN.FT.random_direction(dW[0], seed)
    pairs = {f"{a}-{b}": cos(dW[a], dW[b]) for a, b in combinations(sorted(dW), 2)}
    gauss = {str(s): cos(dW[s], g) for s in sorted(dW)}
    n = sum(v.numel() for v in dW[0].values())
    return {"files": files, "keys": len(dW[0]), "weights": n, "pairs": pairs, "agreement": fmean(pairs.values()),
            "gauss": gauss, "floor": fmean(gauss.values()), "chance_scale": 1 / math.sqrt(n)}  # fmt: skip


def steps_to_learn(name: str) -> dict[str, Any]:
    """Per seed: the first logged step with held gain >= half the final (the `last` entry); missing when final <= 0."""
    seeds = json.loads((RUN.RESULTS / "train" / f"{name}.json").read_text())["seeds"]
    per: dict[str, Any] = {}
    for s, entries in seeds.items():
        st = sorted((e for e in entries if "step" in e), key=lambda e: e["step"])
        final = [e for e in st if e.get("last")]
        assert len(final) == 1 and final[0] is st[-1], (
            f"{name} seed {s}: last entry {[e['step'] for e in final]}"
        )
        fg = final[0]["held_gain"]
        hit = (
            next((e["step"] for e in st if e["held_gain"] >= 0.5 * fg), None)
            if fg > 0
            else None
        )
        per[s] = {
            "final_held_gain": fg,
            "step": hit,
            "logged_steps": [e["step"] for e in st],
        }
    ok = all(v["step"] is not None for v in per.values())
    return {
        "seeds": per,
        "mean": fmean(v["step"] for v in per.values()) if ok else math.nan,
    }


def taus(
    x: dict[str, float],
    y: dict[str, float],
    z: dict[str, float] | None,
    R: dict[str, Any],
) -> dict[str, Any]:
    """tau-b of x with y (y permuted, one-sided p) and, given z, the partial tau with its p; on the tasks with all."""
    ts = sorted(
        t for t in x if t in y and math.isfinite(y[t]) and (z is None or t in z)
    )
    a, b = [x[t] for t in ts], [y[t] for t in ts]
    perm = lambda f, v: SC.perm_p(f, v, R["n_perm"], R["seed"])
    o: dict[str, Any] = {
        "tasks": ts,
        "n": len(ts),
        "tau": SC.tau(a, b),
        "p": perm(lambda v: SC.tau(a, v), b),
    }
    if z is not None:
        c = [z[t] for t in ts]
        o |= {
            "partial": SC.partial_tau(a, b, c),
            "p_partial": perm(lambda v: SC.partial_tau(a, v, c), b),
        }
    return o


def main() -> None:
    C, _ = RUN.load()
    P = json.loads(RUN.PRED.read_text())
    R, orient = P["rules"], P["rules"]["orientation"]
    names = json.loads((WIDE / "dropped.json").read_text())["kept"]["kl_hi"]
    S = json.loads((RUN.RESULTS / "summary.json").read_text())["wide"]
    means, tiers = S["kl_hi"]["means"], SC.raters(P, "raters_wide")
    loss = {
        t: -v["trained"]["nats_per_byte"]
        for t, v in json.loads((WIDE / "base_loss.json").read_text())["tasks"].items()
    }
    read = {m: {t: orient[m] * v for t, v in means[m].items()} for m in READINGS} | {
        "base_loss": loss
    }
    perm = lambda f, v: SC.perm_p(f, v, R["n_perm"], R["seed"])

    ag = {t: agreement(t, C.random_seed) for t in names}
    for t, v in ag.items():
        print(
            f"[agreement {t}] pairs {v['pairs']} mean {v['agreement']:.4f}; Gaussian {v['gauss']} floor {v['floor']:.2e} (chance ~{v['chance_scale']:.1e})"
        )
    a = {t: v["agreement"] for t, v in ag.items()}
    eb = read["E_b"]
    one: dict[str, Any] = {"per_task": ag, "by_rater": {}}
    for r, tier in tiers.items():
        ts = sorted(t for t in names if t in tier)
        x = [tier[t] for t in ts]
        neg, e, g = [-a[t] for t in ts], [eb[t] for t in ts], [a[t] for t in ts]
        one["by_rater"][r] = {"tasks": ts, "tau_tier_minus_agreement": SC.tau(x, neg), "p": perm(lambda v, x=x: SC.tau(x, v), neg),
                              "tau_E_b": SC.tau(x, e), "partial_E_b_given_agreement": SC.partial_tau(x, e, g),
                              "p_partial": perm(lambda v, x=x, g=g: SC.partial_tau(x, v, g), e)}  # fmt: skip
    summ = S["kl_hi"]["hope1"]["E_b"]["consensus"]["tau"]
    one["control_tau_E_b_matches_summary"] = {
        "here": one["by_rater"]["consensus"]["tau_E_b"],
        "summary": summ,
    }
    ts = sorted(names)
    ga, ge = [a[t] for t in ts], [eb[t] for t in ts]
    one["tau_agreement_E_b"] = {
        "tau": SC.tau(ga, ge),
        "p_two_sided": perm(lambda v: abs(SC.tau(ga, v)), ge),
    }

    sl = {t: steps_to_learn(t) for t in names}
    for t, v in sl.items():
        print(
            f"[steps {t}] "
            + "; ".join(
                f"s{s} {d['step']} (final {d['final_held_gain']:.4g})"
                for s, d in v["seeds"].items()
            )
            + f"; mean {v['mean']:.2f}"
        )
    neg_steps = {t: -v["mean"] for t, v in sl.items() if math.isfinite(v["mean"])}
    two: dict[str, Any] = {
        "per_task": sl,
        "tiers": {r: taus(tier, neg_steps, None, R) for r, tier in tiers.items()},
        "readings": {},
    }
    for m, y in read.items():  # x = minus steps held, the reading permuted
        two["readings"][m] = taus(neg_steps, y, None if m == "base_loss" else loss, R)

    OUT.write_text(
        json.dumps(
            {
                "experiment": "E272",
                "set": "wide kl_hi",
                "seed_agreement": one,
                "steps_to_learn": two,
            },
            indent=1,
            default=str,
        )
    )
    for r, o in one["by_rater"].items():
        print(f"[(1) {r}] tau(tier, -agreement) {o['tau_tier_minus_agreement']:.3f} p {o['p']:.4f}; E_b tau {o['tau_E_b']:.3f}, "
              f"partial given agreement {o['partial_E_b_given_agreement']:.3f} p {o['p_partial']:.4f}")  # fmt: skip
    print(
        f"[(1)] tau(agreement, E_b) {one['tau_agreement_E_b']['tau']:.3f} p2 {one['tau_agreement_E_b']['p_two_sided']:.4f}; control {one['control_tau_E_b_matches_summary']}"
    )
    for r, o in two["tiers"].items():
        print(
            f"[(2) tiers {r}] tau(tier, -steps) {o['tau']:.3f} p {o['p']:.4f} (n {o['n']})"
        )
    for m, o in two["readings"].items():
        print(
            f"[(2) {m}] tau with -steps {o['tau']:.3f} p {o['p']:.4f}"
            + (
                f"; partial given base loss {o['partial']:.3f} p {o['p_partial']:.4f}"
                if "partial" in o
                else ""
            )
            + f" (n {o['n']})"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
