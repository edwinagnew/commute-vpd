"""E243 run 2's figure, from saved results only (CPU): the planted head's specific part at block 2's input against its
squared activation change per token, one point per crosscoder seed, with the planted bar and two fine-tunes' own sizes.

    ~/commute-agents/fleet/cpu-run python3 experiments/crosscoder_proper/plot2.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LADDER = json.loads((RES / "run2/ladder.json").read_text())
FINE_TUNES = {"elicit": "e54__elicit_s0.A.c0", "cipher": "e54__cipher_s0.A.c0"}  # run 1's pair, sizes at block 2


def item(rung: str, seed: str) -> dict:
    """Rung 3 is run 1's head at scale 1 (not rerun); the rest are run 2's."""
    path = RES / "run1/items" / f"head.A.c{seed}.json" if rung == "3" else RES / "run2/items" / f"head-r{rung}.A.c{seed}.json"
    return json.loads(path.read_text())


def energy(key: str) -> float:
    return float(json.loads((RES / f"rehearse/items/{key}.json").read_text())["energy"])


def main() -> None:
    fig, ax = figure((6.0, 3.6))
    rows = []
    for rung, row in LADDER["rungs"].items():
        done = {s: v for s, v in row["seeds"].items() if v is not None}
        loss = float(item(rung, next(iter(done)))["spliced"]["clean_1"])
        ax.scatter([row["size"]] * len(done), list(done.values()), color="#009E73", s=28, alpha=0.85, linewidths=0)
        ax.text(row["size"], 0.585, f"Pile loss\n{loss:.1f}", ha="center", va="top", fontsize=6.5, color="#555555")
        rows += [{"rung": int(rung), "size": row["size"], "seed": int(s), "specific": v, "pile_loss": loss}
                 for s, v in done.items()]  # fmt: skip
    bar = LADDER["bar"]
    ax.axhline(bar, color="#777777", linewidth=0.8, linestyle="--")
    ax.text(0.6, bar, f"planted bar, {bar}", fontsize=7, color="#555555", va="bottom")
    for (name, key), y in zip(FINE_TUNES.items(), (0.30, 0.22), strict=True):
        x = energy(key)
        ax.axvline(x, color="#0072B2" if name == "elicit" else "#D55E00", linewidth=0.8, linestyle=":")
        ax.annotate(f"{name}'s own\nsize, {x:.2f}", (x, y), xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                    color="#555555", va="center")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.5, 1e5)
    ax.set_ylim(-0.02, 0.6)
    ax.set_xlabel("planted head's squared activation change per token at block 2's input")
    ax.set_ylabel("share of the change carried\nby fine-tune-specific latents")
    ax.set_title("The planted head's share rises with its size and levels off under the bar", fontsize=9)
    save(fig, HERE / "figures/size_ladder.png",
         n="Each point is one crosscoder seed. Rungs 1, 2 and 5: seed 0; rung 4: seeds 0-1; rung 3: run 1's 3 seeds. "
           "Pile loss is the scaled model's; the original model's is 2.7.",
         data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
