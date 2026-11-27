"""E314 (P2.85): the two SAE proxies per task and seed beside each state's matched random update.

    ~/commute-agents/fleet/cpu-run python experiments/sae_proxies/plot.py

Reads results/results.json. Writes figures/proxies.png (+ .pdf, .csv): (a) error share and (b) the explained-variance
gap (Kissane et al.'s per-token form, mean over 13 sites) against the task's re-gating share (seed mean), seed dots per
task, the random reference just right of each task; (c) the error-energy fraction of each state against its random
reference's. Colour: the side the task was built as (reuse = the pair's first task, new = second). Per-task values only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.figures import FULL_WIDTH, METHODS, figure, save

HERE = Path(__file__).resolve().parent
SIDE = {"reuse": METHODS["built_reuse"], "new": METHODS["built_new"]}
RANDOM = METHODS["random"]
OFFSET = 0.012  # the random reference sits this far right of its task's x


def num(v: Any) -> float:
    """A saved value as a float; None and the string 'nan' are nan."""
    return math.nan if v is None else float(v)


def rows(states: dict[str, Any]) -> list[dict[str, Any]]:
    x = {}
    for s in states.values():
        x.setdefault(s["task"], []).append(s["regating_share"])
    return [{"state": n, "task": s["task"], "seed": s["seed"], "side": s["side"], "regating_share": s["regating_share"],
             "task_regating_mean": fmean(x[s["task"]]), "gain": s["gain"], "random_gain": s["random"]["gain"],
             "error_share": num(s["error_share"]), "random_error_share": num(s["random"]["error_share"]),
             "ev_gap_mean": num(s["ev_gap_mean"]), "random_ev_gap_mean": num(s["random"]["ev_gap_mean"]),
             "fvu_gap_mean": num(s["fvu_gap_mean"]), "random_fvu_gap_mean": num(s["random"]["fvu_gap_mean"]),
             "error_energy_frac": num(s["error_energy_frac"]), "random_error_energy_frac": num(s["random"]["error_energy_frac"])}
            for n, s in states.items()]  # fmt: skip


def against_share(ax: Any, data: list[dict[str, Any]], key: str, ylabel: str) -> None:
    for side, st in SIDE.items():
        d = [r for r in data if r["side"] == side]
        ax.scatter([r["task_regating_mean"] for r in d], [r[key] for r in d], color=st.color, marker=st.marker, s=18, alpha=0.85, linewidths=0, label=st.label)
    ax.scatter([r["task_regating_mean"] + OFFSET for r in data], [r[f"random_{key}"] for r in data], color=RANDOM.color, marker=RANDOM.marker,
               s=14, alpha=0.6, linewidths=0, label=RANDOM.label)  # fmt: skip
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set_xlabel("re-gating share (task mean)")
    ax.set_ylabel(ylabel)


def main() -> None:
    res = json.loads((HERE / "results/results.json").read_text())
    data = rows(res["states"])
    fig, (a, b, c) = figure((FULL_WIDTH[0], 3.6), ncols=3)
    against_share(a, data, "error_share", "error share")
    a.set_title("(a) error share", loc="left")
    against_share(b, data, "ev_gap_mean", "EV gap, base - fine-tuned")
    b.set_title("(b) reconstruction gap", loc="left")
    for side, st in SIDE.items():
        d = [r for r in data if r["side"] == side]
        c.scatter([r["random_error_energy_frac"] for r in d], [r["error_energy_frac"] for r in d], color=st.color, marker=st.marker, s=18, alpha=0.85, linewidths=0)
    c.plot([0, 1], [0, 1], color="#999999", linewidth=0.6)
    c.set_xlabel("error energy, random update")
    c.set_ylabel("error energy, trained")
    c.set_title("(c) error-energy fraction", loc="left")
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncols=3, fontsize=7, frameon=False)
    save(fig, HERE / "figures/proxies.png", n="16 tasks x 3 seeds; one random update per state (Gaussian per matrix at the state's norm); "
         "random error shares shown only where |gain| >= 1 log-odds; solid line in (c): trained = random", data=data)  # fmt: skip


def complement() -> None:
    """E317: error share, reconstruction share and their sum per task against the re-gating share (figures/complement.png)."""
    path = HERE / "results/results_complement.json"
    if not path.is_file():
        return
    states = json.loads(path.read_text())["states"]
    x = {}
    for s in states.values():
        x.setdefault(s["task"], []).append(s["regating_share"])
    keys = ("error_share", "recon_share", "sum_share")
    data = [{"state": n, "task": s["task"], "seed": s["seed"], "side": s["side"], "regating_share": s["regating_share"],
             "task_regating_mean": fmean(x[s["task"]]), "gain": s["gain"], "random_gain": s["random"]["gain"]}
            | {k: num(s[k]) for k in keys} | {f"random_{k}": num(s["random"][k]) for k in keys} for n, s in states.items()]  # fmt: skip
    fig, axes = figure((FULL_WIDTH[0], 3.6), ncols=3)
    for ax, k, title in zip(axes, keys, ("(a) error share", "(b) reconstruction share", "(c) sum of the two"), strict=True):
        against_share(ax, data, k, k.replace("_", " "))
        ax.set_title(title, loc="left")
    axes[2].axhline(1, color="#555555", linewidth=0.8, label="sum = 1 (the two parts add)")
    handles = [h for ax in (axes[0], axes[2]) for h in ax.get_legend_handles_labels()[0]]
    labels = [lab for ax in (axes[0], axes[2]) for lab in ax.get_legend_handles_labels()[1]]
    keep = dict(zip(labels, handles, strict=True))
    fig.legend(list(keep.values()), list(keep), loc="outside upper center", ncols=4, fontsize=7, frameon=False)
    save(fig, HERE / "figures/complement.png", n="16 tasks x 3 seeds; random shares shown only where |gain| >= 1 log-odds; "
         "error share hat(o_cur) + e_FT, reconstruction share hat(o_FT) + e(o_cur), 8 sites at once", data=data)  # fmt: skip


if __name__ == "__main__":
    main()
    complement()
