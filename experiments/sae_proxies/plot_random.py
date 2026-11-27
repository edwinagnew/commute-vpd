"""E318 (P2.88): per task, the fine-tunes' error and reconstruction shares beside Fig 3's random updates' shares.

    ~/commute-agents/fleet/cpu-run python3 experiments/sae_proxies/plot_random.py

Reads results/results_random.json (E318, the random draws) and results/results_complement.json (E317, the fine-tunes'
error share, equal to E314's, and their reconstruction share). Writes figures/random_ref.png (+ .pdf, .csv): one column
per task that has an ok draw (E302 reached its 1.00-1.10 log-odds band); in each, the three fine-tune seeds and,
just right, the ok random draws. Capped draws have no share and are not plotted. Per-task values only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, METHODS, figure, save

HERE = Path(__file__).resolve().parent
TRAINED, RANDOM = METHODS["built_new"], METHODS["random"]
OFFSET = 0.18  # the random draws sit this far right of their task's column


def rows() -> list[dict[str, Any]]:
    draws = json.loads((HERE / "results/results_random.json").read_text())["draws"].values()
    states = json.loads((HERE / "results/results_complement.json").read_text())["states"]
    ok = [d for d in draws if d["status"] == "ok"]
    tasks = sorted({d["task"] for d in ok})
    out = [{"task": s["task"], "kind": "fine-tune", "state": n, "draw_seed": None, "gain": s["gain"],
            "error_share": s["error_share"], "recon_share": s["recon_share"]}
           for n, s in states.items() if s["task"] in tasks]  # fmt: skip
    out += [{"task": d["task"], "kind": "random", "state": d["state"], "draw_seed": d["draw_seed"], "gain": d["gain"],
             "error_share": d["error_share"], "recon_share": d["recon_share"]} for d in ok]  # fmt: skip
    return out


def panel(ax: Any, data: list[dict[str, Any]], tasks: list[str], key: str, ylabel: str) -> None:
    for kind, style, dx, label in (("fine-tune", TRAINED, 0.0, "fine-tune, 3 seeds (task built new)"),
                                   ("random", RANDOM, OFFSET, "random update, matched on gain (~1 log-odds)")):  # fmt: skip
        d = [r for r in data if r["kind"] == kind]
        ax.scatter([tasks.index(r["task"]) + dx for r in d], [r[key] for r in d], color=style.color, marker=style.marker,
                   s=22, alpha=0.8, linewidths=0, label=label)  # fmt: skip
    ax.axhline(1, color="#555555", linewidth=0.8, linestyle="--", label="share = 1 (the patch restores the whole gain)")
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set_xticks([i + OFFSET / 2 for i in range(len(tasks))], tasks, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.6)


def main() -> None:
    data = rows()
    tasks = sorted({r["task"] for r in data})
    fig, (a, b) = figure((FULL_WIDTH[0], 3.6), ncols=2)
    panel(a, data, tasks, "error_share", "error share")
    a.set_title("(a) error share: fine-tune vs random", loc="left")
    panel(b, data, tasks, "recon_share", "reconstruction share")
    b.set_title("(b) reconstruction share: fine-tune vs random", loc="left")
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncols=3, fontsize=7, frameon=False)
    save(fig, HERE / "figures/random_ref.png", n="the 4 tasks with ok random draws, all built new; 3 fine-tune seeds per task; "
         "27 ok draws (6 to 9 per task) of E302's 144, rebuilt from seed and scale; fine-tune gains 10.9 to 20.8 log-odds, "
         "random gains 1.01 to 1.09; 8 sites patched at once", data=data)  # fmt: skip


if __name__ == "__main__":
    main()
