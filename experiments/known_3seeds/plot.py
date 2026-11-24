"""E299 (P2.77): before and after the standardisation, every seed of the nine built pairs on the share and steering.

    ~/commute-agents/fleet/cpu-run python experiments/known_3seeds/plot.py

"Before" is what the paper's figure plotted: method_table/results/final.json at commit 028081b3 (mixed seed counts,
two task variants a side on e46 and e53, steering per key on the two e54 pairs and pooled elsewhere). "After" is
results/known3.json (seeds 0-2 a side, one variant a side, steering pooled on all nine). Per-key steering after the
re-emit is read from the current final.json, at its own seeds. Writes figures/before_after.png (+ .pdf, .csv) and
results/before_after.json, the gaps and calls the analysis page cites.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.figures import FULL_WIDTH, figure, save

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments/known_3seeds"
OLD_COMMIT = "028081b3"
FINAL = "experiments/method_table/results/final.json"
REUSE, NEW = "#0072B2", "#D55E00"
# known3 method -> the old final.json method of the same reading
OLD_METHOD = {"share": "share", "steer": "steering", "prior": "prior"}
# readings whose form is the same before and after (size changed unit, top-neuron share changed k, steering per key on e54)
SAME_FORM = {"share": "share", "steer": "steering", "erank": "erank", "neuron": "neuron", "prior": "prior", "k_half": "k_half"}


def old_final() -> dict[tuple[str, str], Any]:
    text = subprocess.run(["git", "show", f"{OLD_COMMIT}:{FINAL}"], cwd=REPO, check=True, capture_output=True, text=True).stdout
    return {(c["pair"], c["method"]): c for c in json.loads(text)["cells"]}


def side(values: list[float] | None) -> list[float]:
    return [v for v in values or [] if v is not None]


def read(reuse: list[float] | None, new: list[float] | None) -> dict[str, Any] | None:
    """Gap of the side means and its call: apart when every seed of one side lies beyond every seed of the other."""
    r, n = side(reuse), side(new)
    if not r or not n:
        return None
    lo, hi = min(r) - max(n), max(r) - min(n)
    return {"gap": fmean(r) - fmean(n), "lo": lo, "hi": hi, "n": [len(r), len(n)], "apart": lo > 0 or hi < 0}


def call(steer: dict[str, Any] | None, share: dict[str, Any]) -> str:
    if steer is None:
        return "not read"
    if not steer["apart"]:
        return "no call"
    return "same as the share" if (steer["gap"] > 0) == (share["gap"] > 0) else "opposite to the share"


def outside_old_range(known3: dict[tuple[str, str], Any], old: dict[tuple[str, str], Any]) -> dict[str, Any]:
    """Each task side whose 3-seed mean now lies outside the seed range the figure showed before, on an unchanged form."""
    out = {}
    for (pair, m), nc in known3.items():
        oc = old.get((pair, SAME_FORM.get(m, "")))
        if oc is None or "perkey" in (oc.get("form") or ""):
            continue
        for s in ("reuse", "new"):
            after, before = side(nc[s]), side(oc[s])
            eps = 1e-9 * max(map(abs, before))  # a value repeated per seed averages back to itself only to rounding
            if after and before and not min(before) - eps <= fmean(after) <= max(before) + eps:
                out[f"{pair}/{m}/{s}"] = {"after_mean": fmean(after), "before_min": min(before), "before_max": max(before),
                                         "before_n": len(before)}  # fmt: skip
    return out


def summary() -> dict[str, Any]:
    known3 = {(c["pair"], c["method"]): c for c in json.loads((HERE / "results/known3.json").read_text())["cells"]}
    now = {(c["pair"], c["method"]): c for c in json.loads((REPO / FINAL).read_text())["cells"]}
    old = old_final()
    pairs: dict[str, Any] = {}
    for pair in dict.fromkeys(p for p, _ in known3):
        rec: dict[str, Any] = {"name": now[(pair, "share")]["pair_name"]}
        for m, om in OLD_METHOD.items():
            oc, nc = old[(pair, om)], known3[(pair, m)]
            rec[m] = {"before": read(oc["reuse"], oc["new"]) | {"form": oc.get("form", "")},
                      "after": read(nc["reuse"], nc["new"]), "seeds_before": [side(oc["reuse"]), side(oc["new"])],
                      "seeds_after": [side(nc["reuse"]), side(nc["new"])]}  # fmt: skip
            rec[m]["move"] = rec[m]["after"]["gap"] - rec[m]["before"]["gap"]
        pk = now[(pair, "steering_per_key")]
        rec["per_key_now"] = read(pk["reuse"], pk["new"])
        share = rec["share"]["after"]
        rec["calls"] = {"pooled_after": call(rec["steer"]["after"], share), "per_key_now": call(rec["per_key_now"], share),
                        "before": old[(pair, "steering")]["against_share"]}  # fmt: skip
        pairs[pair] = rec
    count = {k: sorted(p for p, r in pairs.items() if r["calls"][k] == "opposite to the share") for k in ("pooled_after", "per_key_now", "before")}
    return {"pairs": pairs, "reversals": {k: {"n": len(v), "of": len(pairs), "pairs": v} for k, v in count.items()},
            "share_gap_positive": sum(r["share"]["after"]["gap"] > 0 for r in pairs.values()),
            "share_apart": sum(r["share"]["after"]["apart"] for r in pairs.values()), "old_commit": OLD_COMMIT,
            "outside_old_range": outside_old_range(known3, old)}  # fmt: skip


def panel(ax: Any, pairs: dict[str, Any], m: str, xlabel: str, rows: list[dict[str, Any]]) -> None:
    for i, (pair, rec) in enumerate(pairs.items()):
        for when, dy, filled in (("before", -0.17, False), ("after", 0.17, True)):
            for s, colour in zip((0, 1), (REUSE, NEW), strict=True):
                xs = rec[m][f"seeds_{when}"][s]
                ax.scatter(xs, [i + dy] * len(xs), s=20, marker="o", linewidths=1.0, edgecolors=colour,
                           facecolors=colour if filled else "none")  # fmt: skip
                rows += [{"panel": m, "pair": pair, "when": when, "side": ("reuse", "new")[s], "value": x} for x in xs]
    ax.set_yticks(range(len(pairs)), [r["name"] for r in pairs.values()], fontsize=7)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_xlabel(xlabel)


def main() -> None:
    out = summary()
    (HERE / "results/before_after.json").write_text(json.dumps(out, indent=1) + "\n")
    fig, (a, b) = figure(size=(FULL_WIDTH[0], 5.0), ncols=2)
    rows: list[dict[str, Any]] = []
    panel(a, out["pairs"], "share", "share of the gain in log-odds", rows)
    panel(b, out["pairs"], "steer", "share of the gain in log-odds", rows)
    a.set_title("Re-gating share, trained update", fontsize=8)
    b.set_title("Steering share, pooled\n(before: per key on the two half-known pairs)", fontsize=8)
    b.set_yticklabels([])
    for colour, label, filled in ((REUSE, "reuse-side seed", True), (NEW, "new-side seed", True),
                                  ("#777777", "before (upper row)", False),
                                  ("#777777", "after (lower row)", True)):  # fmt: skip
        b.scatter([], [], s=20, edgecolors=colour, facecolors=colour if filled else "none", label=label)
    b.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.03, 0.0), frameon=False)
    fig.suptitle("Three seeds and one form a side left every share gap in place; steering moved where its form changed", fontsize=9)
    save(fig, HERE / "figures/before_after.png", n="before: 2 to 8 seeds a side; after: 3 seeds a side; 67M model, nine built pairs",
         data=rows)  # fmt: skip
    print(json.dumps(out["reversals"]), out["share_gap_positive"], out["share_apart"])


if __name__ == "__main__":
    main()
