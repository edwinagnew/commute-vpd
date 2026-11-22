"""P2.60 figures: the lens on E254/E255's prior-matched seeds beside the re-gating share, and the lens's partial
Spearman with E_b given KL on P2.56's procedural states beside the other proxies'. Reads saved JSON only.

Run from the repo root: python3 experiments/adl/plot_p260.py
"""

import json
from pathlib import Path

from commute import figures as F

AMEND2 = Path("experiments/adl/results/amend2/summary.json")
PM = Path("experiments/prior_matched/results")
OUT = Path("experiments/adl/figures")
LENS = F.Style("activation difference lens", "#332288", "d")
FLOOR = 0.0054  # the card's smallest gap: largest |D| over E265's random updates


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def prior_matched() -> None:
    a = load(AMEND2)["a"]
    r1, r2 = load(PM / "a/summary.json")["rows"], load(PM / "b/summary.json")["rows"]
    lens = {
        "copy, 10 seeds": a["e254_cipher"]["copy"],
        "matched cipher 1, 10 seeds": a["e254_cipher"]["cipher"],
        "matched cipher 2, 3 seeds": a["e255_cipher"]["cipher"],
        "random updates, 9": list(a["randoms"].values()),
    }
    share = {
        "copy, 10 seeds": [x["r"] for x in r1["reused"]],
        "matched cipher 1, 10 seeds": [x["r"] for x in r1["new"]],
        "matched cipher 2, 3 seeds": [x["r"] for x in r2["new"]],
        "random updates, 9": [],
    }
    fig, (left, right) = F.figure(F.FULL_WIDTH, ncols=2)
    for i, values in enumerate(lens.values()):
        left.scatter(values, [i] * len(values), color=LENS.color, marker=LENS.marker, s=24, alpha=0.85, linewidths=0)
    left.set_yticks(range(len(lens)), list(lens))
    left.invert_yaxis()
    left.set_xlabel("lens score D: share of the gain lost")
    for x, name in ((-FLOOR, "-floor"), (FLOOR, "+floor")):
        F.reference(left, x, name)
    left.set_xticks([0, 0.01, 0.02])
    left.set_title("Lens: copy and cipher seeds overlap", fontsize=9, pad=14)
    F.dots(right, share, method="regating", xlabel="re-gating share r")
    right.set_yticklabels([])
    right.set_title("Re-gating share: every copy seed apart", fontsize=9, pad=14)
    fig.suptitle("At matched answer prior, the lens does not split copy from cipher; the share does")
    rows = [{"reading": "lens", "group": g, "value": v} for g, vs in lens.items() for v in vs]
    rows += [{"reading": "r", "group": g, "value": v} for g, vs in share.items() for v in vs]
    F.save(fig, OUT / "adl_prior_matched.png", n="67M model; 10 copy and 10 matched-cipher seeds (E254), 3 second-cipher "
           "seeds (E255); the share is P2.57's reading of the same fine-tunes; random updates have no gain for r", data=rows)


PROXIES = {
    "adl": "activation difference lens",
    "share": "re-gating share",
    "untrained": "untrained-decomposition share",
    "steering": "steering share",
    "prior": "original model's prior",
    "erank": "update's effective rank",
    "neuron_kmatched": "neuron version, k matched",
    "neuron": "neuron version",
}


def procedural() -> None:
    table = load(AMEND2)["b"]["tables"]["trained"]
    fig, ax = F.figure((6.8, 3.6))
    rows = []
    for i, (key, name) in enumerate(PROXIES.items()):
        t = table[key]
        style = LENS if key == "adl" else F.METHODS["regating"]
        color = style.color if key == "adl" else "#777777"
        ax.scatter([t["raw"]], [i], facecolors="none", edgecolors=color, marker="o", s=30, label="raw" if i == 1 else None)
        ax.scatter([t["partial_given_kl"]], [i], color=color, marker=LENS.marker if key == "adl" else "o", s=34,
                   label="given KL" if i == 1 else None)
        rows.append({"proxy": key, "raw": t["raw"], "partial_given_kl": t["partial_given_kl"], "n": t["n"]})
    ax.set_yticks(range(len(PROXIES)), [f"{n}, n={table[k]['n']}" for k, n in PROXIES.items()])
    ax.get_yticklabels()[0].set_color(LENS.color)
    ax.invert_yaxis()
    for x in (-0.3, 0.3):
        F.reference(ax, x, f"{x:+.1f}")
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman with definition (b)'s write share")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Procedural fine-tunes: the lens runs against definition (b)", fontsize=9, pad=14)
    F.save(fig, OUT / "adl_procedural_partial.png", n="P2.56's trained RL, SFT and guarded-RL states on three tasks; "
           "states where a proxy is undefined are dropped, so n differs by proxy", data=rows)


if __name__ == "__main__":
    prior_matched()
    procedural()
