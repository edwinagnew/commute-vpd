"""E228's one figure: each fine-tune's own KL-form share against its three KL-matched random draws (left), and the
gain on the trained answer that r needs, against the same draws (right). One row per task side, in pair order. From
results/states/*.json (no GPU):

    ~/commute-agents/fleet/cpu-run python3 experiments/random_kl_share/plot.py   -> figures/random_share.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from commute import figures

HERE = Path(__file__).resolve().parent
STATES = HERE / "results" / "states"
SCORE = HERE / "results" / "score.json"

# Task side (state name without its seed) -> row label, in the order of the pairs; reuse side first in each pair.
ROWS = {
    "e62/copy": "copy at a new marker, reuse",
    "e62/lookup": "lookup, new",
    "e46/ident32": "copy maps, reuse",
    "e46/ident32b": "copy maps b, reuse",
    "e46/rand32": "random maps, new",
    "e46/rand32b": "random maps b, new",
    "e53/lowercase": "lowercase, reuse",
    "e53/cipher": "cipher, new",
    "e53/broad_cipher": "broad cipher, new",
    "e54/elicit": "half-known, reuse",
    "e54/cipher": "cipher equal size, new",
    "e54/cipher_own": "cipher own steps, new",
    "c3/rank1": "rank-1 step 10, reuse",
    "c3/copy": "copy step 10, reuse",
    "c3/cipher": "cipher step 10, new",
    "e58/rank1": "rank-1 96 keys, reuse",
    "e58/cipher": "cipher 96 keys, new",
    "e58/copy": "copy 96 keys, reuse",
    "e58/prior_cipher": "prior-matched cipher, new",
    "e58/elicit": "elicited map, twin real",
    "e58/shuffle": "its shuffled twin",
    "e58/class_map": "class map, twin real",
    "e58/class_shuffled": "its shuffled twin ",
    "case_label/real": "case label, twin real",
    "case_label/twin": "its memorising twin",
    "fresh/real": "repeat label, twin real",
    "fresh/twin": "its memorising twin ",
}


def num(x: object) -> float:
    return float(x) if isinstance(x, int | float) else math.nan


def main() -> None:
    recs = [json.loads(p.read_text()) for p in sorted(STATES.glob("*.json"))]
    band = json.loads(SCORE.read_text())["bands"]["kl_share"]["band"]
    side = {r["state"]: r["state"].rsplit("_s", 1)[0] for r in recs}
    if missing := set(side.values()) - set(ROWS):
        raise SystemExit(f"plot: no row label for {sorted(missing)}")
    groups = {key: [r for r in recs if side[r["state"]] == key] for key in ROWS}
    data = [{"state": r["state"], "series": name, "kl_share": num(d["kl_share"]), "gain": num(d["gain"])}
            for r in recs for name, d in (("fine-tune", r["own"]), *((f"random {s}", d) for s, d in r["draws"].items()))]

    fig, (left, right) = figures.figure((figures.FULL_WIDTH[0], 8.0), ncols=2)
    for ax, key, xlabel in ((left, "kl_share", "re-gating share, KL form"),
                            (right, "gain", "gain on the trained answer, log-odds")):
        own = {ROWS[k]: [num(r["own"][key]) for r in g] for k, g in groups.items()}
        rand = {ROWS[k]: [num(d[key]) for r in g for d in r["draws"].values()] for k, g in groups.items()}
        figures.dots(ax, rand, method="random", xlabel=xlabel, offset=0.18)
        figures.dots(ax, own, method="regating", xlabel=xlabel, offset=-0.18)
    right.set_yticklabels([])
    figures.reference(right, 1.0, "r's floor: gain 1")
    left.annotate("matched change the split carries\nand the gates cannot see: 0 on all 88",
                  (0.02, 0.01), xycoords="axes fraction", fontsize=7, color="#555555")
    left.scatter([], [], **_style("regating"), label="fine-tune, its own reading")
    left.scatter([], [], **_style("random"), label="random change, KL-matched")
    left.legend(loc="upper left", fontsize=7, frameon=False)
    left.set_title("Random changes read 0.97-0.99, as high as\nmost fine-tunes", fontsize=9, pad=12)
    right.set_title("No random change lifts the trained answer\nby 1 log-odds, so r has no random reading", fontsize=9, pad=12)
    figures.save(fig, HERE / "figures" / "random_share.png", data=data,
                 n=f"88 fine-tunes (2-8 seeds per side) of nine built pairs and four twins; 3 random draws each, "
                   f"scaled to the fine-tune's KL on its own rows; split-half band of the KL-form share {band:.4f}")


def _style(method: str) -> dict[str, object]:
    s = figures.METHODS[method]
    return {"color": s.color, "marker": s.marker, "s": 22}


if __name__ == "__main__":
    main()
