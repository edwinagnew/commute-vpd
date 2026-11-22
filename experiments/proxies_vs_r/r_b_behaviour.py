"""Cell P2.51: the proxies' pair gaps beside E259's behavioural per-member reading (definition (b)), on its three pairs.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/r_b_behaviour.py

E259 (P1.51, definition-lead, experiments/rb_member_pairs) read, per changed member u of K* that may stand in, q_u =
KL(F || F with u alone stood in at its saved gain) / KL(F || M) and q1_u the same at gain 1 (u's change removed).
b_mean is the write-change (dd) weighted mean over the members above the floor of 1 - q_u / q1_u: the share of a
member's own effect on F's output that the base model's member, rescaled, gives back. pass_share is the same weighting of
the indicator q_u <= 0.1 q1_u. Both are estimated from a uniform sample of 2,000 members per item, with a linearised se.
E_b is E257's write-based share over every changed member, exact.

Descriptive: no rule was filed for this comparison. Per pair, each reading's gap is the reuse side's item mean minus the
new side's, as E257's gap_E_b (e53's new side holds both cipher tasks); its sampling se combines the items' se, and the
seed range is each side's min to max. Writes results/r_b_behaviour.json and figures/rb_member_pairs_vs_random.png.
"""

from __future__ import annotations

import json
import math
from statistics import fmean
from typing import Any

import analyse as A

from commute import figures as F

HERE = A.HERE
PAIRS = (
    "e187_e62_copy_lookup",
    "e187_e58_copy_prior_cipher",
    "e187_e53_lowercase_cipher",
)
RANDOM = "random_size_241001"
READINGS = ("E_b", "b_mean")
TASKS = {
    "e187_e62_copy_lookup": {"reuse": "A: copy\n(reuse)", "new": "A: lookup\n(new)"},
    "e187_e58_copy_prior_cipher": {
        "reuse": "B: copy\n(reuse)",
        "new": "B: prior cipher\n(new)",
    },
    "e187_e53_lowercase_cipher": {
        "reuse": "C: lowercase\n(reuse)",
        "new": "C: two ciphers\n(new)",
    },
}


def sides() -> dict[str, dict[str, list[str]]]:
    p = A.load("rb_states/results/summary_items.json")["pairs"]
    return {
        q: {s: [i["item"] for i in p[q][s]["items"]] for s in ("reuse", "new")}
        for q in PAIRS
    }


def side(items: dict[str, Any], names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(names)}
    for k in READINGS:
        v = [items[i][k] for i in names]
        out |= {k: fmean(v), f"{k}_min": min(v), f"{k}_max": max(v)}
    out["b_mean_se"] = math.sqrt(sum(items[i]["b_mean_se"] ** 2 for i in names)) / len(
        names
    )
    out["unread_floor_dd_share_max"] = max(
        items[i]["unread_floor_dd_share_est"] for i in names
    )
    return out


def pairs(items: dict[str, Any], e257: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for q, sd in sides().items():
        r, n = side(items, sd["reuse"]), side(items, sd["new"])
        gap = {k: r[k] - n[k] for k in READINGS}
        assert abs(gap["E_b"] - e257[q]["gap_E_b"]) < 1e-12, (
            f"{q}: E_b gap differs from E257's"
        )
        out[q] = {"reuse": r, "new": n, "gap_E_b": gap["E_b"], "gap_b_mean": gap["b_mean"],
                  "gap_b_mean_se": math.hypot(r["b_mean_se"], n["b_mean_se"]),
                  "gap_b_mean_over_se": gap["b_mean"] / math.hypot(r["b_mean_se"], n["b_mean_se"])}  # fmt: skip
    return out


def against_random(items: dict[str, Any]) -> dict[str, Any]:
    """Each trained item's b_mean minus the random update's, in units of their combined sampling se."""
    rnd = items[RANDOM]
    z = {
        i: (r["b_mean"] - rnd["b_mean"]) / math.hypot(r["b_mean_se"], rnd["b_mean_se"])
        for i, r in items.items()
        if i != RANDOM
    }
    b = [items[i]["b_mean"] for i in z]
    return {"random_b_mean": rnd["b_mean"], "random_b_mean_se": rnd["b_mean_se"], "random_E_b": rnd["E_b"],
            "trained_n": len(z), "trained_b_min": min(b), "trained_b_max": max(b), "trained_b_max_over_random": max(b) / rnd["b_mean"],
            "n_above_random_2se": sum(v > 2 for v in z.values()), "z_min": min(z.values()), "z_max": max(z.values()), "z": z,
            "pass_share_max": max(r["pass_share"] for r in items.values()),
            "n_sampled_passed_total": sum(r["n_sampled_passed"] for r in items.values()),
            "n_sampled_counted_total": sum(r["n_sampled"] - r["n_sampled_under_floor"] for r in items.values())}  # fmt: skip


def signs(pr: dict[str, Any]) -> dict[str, Any]:
    """Each proxy's pair gap (analyse.proxies, higher = reuse) and R_A's, beside E_b's and b_mean's signs."""
    px = {p: v["pairs"] for p, v in A.proxies().items()}
    px["R_A"] = json.loads((HERE / "results" / "r_a_vpd.json").read_text())["pairs"]
    out = {}
    for p, g in px.items():
        row = {q: g.get(q) for q in PAIRS}
        match = {
            k: {
                q: None if (v := row[q]) is None else (v > 0) == (pr[q][f"gap_{k}"] > 0)
                for q in PAIRS
            }
            for k in READINGS
        }
        out[p] = {"gaps": row, "matches": match, **{f"n_match_{k}": sum(bool(m) for m in match[k].values()) for k in READINGS},
                  "n_read": sum(v is not None for v in row.values())}  # fmt: skip
    return out


def plot(items: dict[str, Any]) -> Any:
    fig, axes = F.figure((7.5, 3.6), ncols=2)
    words = {
        "E_b": "E_b: write change explained",
        "b_mean": "b_mean: member effect given back",
    }
    rnd, data = items[RANDOM], []
    for ax, k in zip(axes, READINGS, strict=True):
        groups = {
            TASKS[q][s]: [items[i][k] for i in names]
            for q, sd in sides().items()
            for s, names in sd.items()
        }
        if k == "E_b":  # exact: a line
            ax.axhline(
                rnd[k],
                color="#555555",
                linestyle="--",
                label="random update, matched size",
            )
        else:  # sampled: its ± 2 se
            ax.axhspan(
                rnd[k] - 2 * rnd["b_mean_se"],
                rnd[k] + 2 * rnd["b_mean_se"],
                color="#555555",
                alpha=0.3,
                label="random update, matched size, ± 2 se",
            )
        F.column_dots(ax, groups, method="regating", ylabel=words[k])
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=6, frameon=False)
        data += [
            {"reading": k, "group": g, "value": v, "random": rnd[k]}
            for g, vs in groups.items()
            for v in vs
        ]
    fig.suptitle("Do the trained pairs read above a random update? Every side's seeds, both readings,\n"
                 "beside the size-matched random update; pairs A, B, C, reuse side first", fontsize=9)  # fmt: skip
    return F.save(fig, HERE / "figures" / "rb_member_pairs_vs_random.png", data=data,
                  n="three trained pairs, 2 to 4 seeds per side, every point one fine-tune; b_mean from 2,000 sampled members "
                    "per fine-tune; E_b over every changed member")  # fmt: skip


def main() -> int:
    e259 = A.load("rb_member_pairs/results/summary.json")
    items, e257 = e259["items"], A.load("rb_states/results/summary_items.json")["pairs"]
    pr = pairs(items, e257)
    out = {"name": "r_b_behaviour", "source": "experiments/rb_member_pairs/results/summary.json", "floor": e259["floor"],
           "tol": e259["tol"], "pairs": pr, "random": against_random(items), "signs": signs(pr)}  # fmt: skip
    (HERE / "results" / "r_b_behaviour.json").write_text(json.dumps(out, indent=1))
    for q, v in pr.items():
        print(
            q,
            {
                k: round(v[k], 4)
                for k in (
                    "gap_E_b",
                    "gap_b_mean",
                    "gap_b_mean_se",
                    "gap_b_mean_over_se",
                )
            },
        )
    print({k: v for k, v in out["random"].items() if k != "z"})
    for p, s in out["signs"].items():
        print(
            p,
            {q[5:]: g and round(g, 3) for q, g in s["gaps"].items()},
            "match E_b",
            s["n_match_E_b"],
            "b_mean",
            s["n_match_b_mean"],
            "of",
            s["n_read"],
        )
    print(plot(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
