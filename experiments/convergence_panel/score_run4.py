"""E307 (P2.66 run 4, regating-lead 2026-11-24, descriptive): the wide panel scored at n = 10 with score.py's functions
unchanged, under the rule filed in results/wide/predictions_run4.json "score" before any value was read.

Per task-seed the state is the run-4 "gain1" pick where run4/selection.json has one (spanish s0, sentiment s0-s2), else
the existing KL "hi" pick; arxiv and german (no save reached read-row gain 1 by step 1200) are out and reported with
each seed's best read-row gain. E_b's random reference is the mean of the three Gaussians (random_3seed.mean3, P2.78).
The run-3 12-task values (results/summary.json, results/wide/random_3seed.json) are printed beside, not re-scored.
Writes results/wide/run4/score_n10.json.

    ~/commute-agents/fleet/cpu-run --mem 4G -t 20 python3 experiments/convergence_panel/score_run4.py
"""

from __future__ import annotations

import json
import math
from typing import Any

from commute.jobs import load_experiment

RUN: Any = load_experiment("convergence_panel", "run")
SC: Any = load_experiment("convergence_panel", "score")
RS: Any = load_experiment("convergence_panel", "random_3seed")
WIDE = RUN.RESULTS / "wide"
R4 = WIDE / "run4"
KEPT = ("category", "conll", "french", "lower", "squad", "summary", "swahili", "yoruba")
REPICKED = ("sentiment", "spanish")
DROPPED = ("arxiv", "german")
SEEDS = (0, 1, 2)
PLAIN = tuple(m for m in SC.MEASURES if m != "E_b")


def state(task: str, seed: int, sel: dict[str, Any]) -> dict[str, Any] | None:
    """The task-seed's record in read/TASK.json: the gain1 pick's file if selection.json names one, else the KL 'hi' pick."""
    states = json.loads((WIDE / "read" / f"{task}.json").read_text())["states"]
    pick = (sel["tasks"].get(task) or {}).get(str(seed))
    if pick is not None:
        return states[pick["file"]] | {"file": pick["file"], "match": "gain1"} if pick["file"] in states else None
    hits = [f for f, r in states.items() if r["seed"] == seed and "kl_hi" in r["picked"]]
    assert len(hits) <= 1, f"{task} s{seed}: {len(hits)} KL 'hi' picks"
    return states[hits[0]] | {"file": hits[0], "match": "kl_hi"} if hits else None


def panel(sel: dict[str, Any]) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[int, float]], dict[str, Any]]:
    """measure -> task -> seed values (nan where missing), task -> seed -> Gaussian E_b, and each task-seed's state."""
    data: dict[str, Any] = {m: {} for m in SC.MEASURES}
    rand: dict[str, dict[int, float]] = {}
    where: dict[str, Any] = {}
    for task in KEPT + REPICKED:
        eb = json.loads((WIDE / "eb" / f"{task}.json").read_text())["states"]
        for s in SEEDS:
            rec = state(task, s, sel)
            where.setdefault(task, {})[s] = None if rec is None else {k: rec[k] for k in ("file", "match", "step", "kl", "gain")}
            for m in PLAIN:
                data[m].setdefault(task, []).append(math.nan if rec is None else SC.value(rec, m))
            f = rec["file"] if rec else ""
            data["E_b"].setdefault(task, []).append(SC.e_b(WIDE / "eb", task, f, eb.get(f)) if rec else math.nan)
            rand.setdefault(task, {})[s] = SC.e_b(WIDE / "eb", task, "random/" + f, eb.get("random/" + f)) if rec else math.nan
    return data, rand, where


def never_reached() -> dict[str, Any]:
    """arxiv and german: each seed's best read-row gain over its run-4 saves (steps 50-1200), and the floor."""
    out: dict[str, Any] = {}
    for t in DROPPED:
        for s in SEEDS:
            saves = json.loads((R4 / f"{t}_s{s}.json").read_text())["saves"]
            best = max(saves, key=lambda v: v["gain"])
            out.setdefault(t, {})[s] = {"best_gain": best["gain"], "at_step": best["step"], "last_step": max(v["step"] for v in saves),
                                        "gain_at_last": max(saves, key=lambda v: v["step"])["gain"]}  # fmt: skip
    return out


def run3() -> dict[str, Any]:
    """The run-3 12-task values from results/summary.json (wide kl_hi) and results/wide/random_3seed.json."""
    w = json.loads((RUN.RESULTS / "summary.json").read_text())["wide"]["kl_hi"]
    h1 = {m: {k: v["consensus"].get(k) for k in ("tau", "p_perm", "call", "tasks", "tau_partial_prior")} for m, v in w["hope1"].items()}
    r3 = json.loads((WIDE / "random_3seed.json").read_text())
    return {"hope1_consensus": h1, "unreadable": w["unreadable"], "hope2": {k: w["hope2"].get(k) for k in ("W_readings", "W_baselines", "call", "readings", "baselines", "tasks")},
            "E_b_random_3seed_tau": r3["random_3seed"]["tau"], "E_b_trained_tau": r3["trained"]["tau"], "gap_rule": r3["rule"]["call"]}  # fmt: skip


def sensitivity(tier: dict[str, float], new: dict[str, dict[str, float]], R: dict[str, Any], interval90: list[float]) -> dict[str, Any]:
    """Redteam 2026-11-24 (descriptive): the ten-task half-width of E_b's 90% interval (P2.78's bar recipe at n = 10);
    E_b's tau with old picks and with sentiment out; the effective rank's prior-held-fixed tau with arXiv or German out."""
    old = json.loads((RUN.RESULTS / "summary.json").read_text())["wide"]["kl_hi"]["means"]
    drop = lambda d, out: {t: v for t, v in d.items() if t not in out}
    eb_tau = lambda d: SC.tau([tier[t] for t in sorted(d)], [d[t] for t in sorted(d)])

    def erank(out: tuple[str, ...], m: dict[str, dict[str, float]]) -> dict[str, Any]:
        h = SC.hope1(tier, drop(m["erank"], out), drop(m["prior"], out), R)
        return {"n": len(h["tasks"]), "tau": h["tau"], "tau_partial_prior": h["tau_partial_prior"], "p_partial": h["p_partial"]}

    return {"E_b_interval90_half_width_n10": (interval90[1] - interval90[0]) / 2,
            "E_b_tau_n10_old_picks": eb_tau(drop(old["E_b"], DROPPED)), "E_b_tau_n9_no_sentiment": eb_tau(drop(new["E_b"], ("sentiment",))),
            "erank_12_minus_arxiv": erank(("arxiv",), old), "erank_12_minus_german": erank(("german",), old),
            "erank_10_old_picks": erank(DROPPED, old), "erank_10_new_picks": erank((), new)}  # fmt: skip


def main() -> None:
    P = json.loads(RUN.PRED.read_text())
    rule_p278 = json.loads((WIDE / "predictions_p278.json").read_text())["rule"]
    sel = json.loads((R4 / "selection.json").read_text())
    data, rand, where = panel(sel)
    dropped = {"out": {t: "no save of any seed reached read-row gain 1 by step 1200 (run4/selection.json 'none')" for t in DROPPED}}
    res = SC.score({"kl_hi": data}, P, dropped, "raters_wide")["kl_hi"]
    tier = SC.raters(P, "raters_wide")["consensus"]
    random3 = {t: RS.mean3(v) for t, v in rand.items()}
    trained = res["means"]["E_b"]
    R = P["rules"]
    rand_h1 = SC.hope1(tier, random3, res["means"]["prior"], R) | {"note": "descriptive: the reference, not a reading"}
    per_task = {t: {"tier": tier[t], "states": where[t], **{m: {"seeds": data[m][t], "mean": res["means"][m].get(t, math.nan)} for m in SC.MEASURES},
                    "E_b_random": {"seeds": [rand[t][s] for s in SEEDS], "mean": random3[t]}} for t in KEPT + REPICKED}  # fmt: skip
    out = {"experiment": "E307", "cell": "P2.66", "run": 4, "n_tasks": len(per_task), "rule": "results/wide/predictions_run4.json score",
           "not_kl_matched": {t: where[t] for t in REPICKED}, "per_task": per_task, "readability": res["readability"],
           "unreadable": res["unreadable"], "hope1": res["hope1"], "hope2": res["hope2"], "amendment10": res["amendment10"]["consensus"],
           "E_b_random": {"hope1_consensus": rand_h1, "taus": RS.taus(tier, random3), "trained_taus": RS.taus(tier, trained),
                          "gap_rule_p278": RS.gap_call(tier, trained, random3, rule_p278)},
           "never_reached_gain1": never_reached(), "run3_12_tasks": run3(),
           "sensitivity": sensitivity(tier, res["means"], R, res["hope1"]["E_b"]["consensus"]["interval90"])}  # fmt: skip
    dest = R4 / "score_n10.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    show(out)
    print(f"wrote {dest}")


def show(o: dict[str, Any]) -> None:
    g, r3 = (lambda d, k: d.get(k, math.nan)), o["run3_12_tasks"]
    for m, by in o["hope1"].items():
        c, old = by["consensus"], r3["hope1_consensus"].get(m, {})
        print(f"{m:9s} n10 tau {c['tau']:+.3f} p {c['p_perm']:.4f} partial {g(c, 'tau_partial_prior'):+.3f} p {g(c, 'p_partial'):.4f} "
              f"int90 {[round(v, 3) for v in c['interval90']]} crit {c['critical_tau']:.3f} n {len(c['tasks'])}: {c['call']}  | run3 "
              f"tau {g(old, 'tau'):+.3f} n {len(old.get('tasks', []))}: {old.get('call')}")  # fmt: skip
        print("          per rater: " + "; ".join(f"{r} {v.get('tau', math.nan):+.3f} {v['call']}" for r, v in by.items() if r != "consensus"))
    print(f"unreadable n10: {o['unreadable']} | run3: {r3['unreadable']}")
    h2 = o["hope2"]
    print(f"H2 n10: W readings {h2['W_readings']:.3f} {h2['readings']} vs baselines {h2['W_baselines']:.3f} {h2['baselines']} "
          f"on {len(h2['tasks'])} tasks: {h2['call']} | run3 {r3['hope2']['W_readings']:.3f} vs {r3['hope2']['W_baselines']:.3f}: {r3['hope2']['call']}")
    e = o["E_b_random"]
    print(f"E_b trained tau {e['trained_taus']['tau']:+.3f} vs random 3-draw tau {e['taus']['tau']:+.3f} (n {e['taus']['n']}); "
          f"gap rule {e['gap_rule_p278']['gap']:.3f}: {e['gap_rule_p278']['call']} | run3 trained {r3['E_b_trained_tau']:+.3f} random {r3['E_b_random_3seed_tau']:+.3f}")
    print(f"E_b random hope1 (descriptive): p {e['hope1_consensus'].get('p_perm')} {e['hope1_consensus']['call']}")
    print("sensitivity: " + json.dumps(o["sensitivity"]))
    for t, by in o["never_reached_gain1"].items():
        print(f"{t}: " + "; ".join(f"s{s} best {v['best_gain']:.3f} at step {v['at_step']}" for s, v in by.items()))
    for t in REPICKED:
        print(f"{t} states: " + "; ".join(f"s{s} {v['match']} step {v['step']} KL {v['kl']:.3f}" for s, v in o["not_kl_matched"][t].items() if v))
    print("task means: " + "; ".join(f"{t} " + ",".join(f"{m}={v[m]['mean']:.4g}" for m in SC.MEASURES) + f",Eb_rand={v['E_b_random']['mean']:.4g}"
                                     for t, v in o["per_task"].items()))  # fmt: skip


if __name__ == "__main__":
    main()
