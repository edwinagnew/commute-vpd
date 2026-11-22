"""E237's one-figure summary (no GPU), from results/summary.json, with E244's baselines (the neuron shares and the
crosscoder, read on the same items by experiments/key_grid_baselines) as four more panels: every reading as one panel
over the same five tasks, ordered by the base model's prior of the trained answer (lowest first), seeds as dots at the
common update size, the random update of that size as a grey band. The readings not taken on these five tasks get a
panel saying so.

    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/plot_columns.py
    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/plot_columns.py --against experiments/proxies_vs_r/results/r_a_vpd.json \
        --out experiments/proxies_vs_r/figures/key_tasks_vs_r_a_vpd.png   # each reading against R under wording A (P2.51)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from commute.figures import METHODS, column_dots, figure, save

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "results" / "summary.json").read_text())
BASE = HERE.parent / "key_grid_baselines" / "results" / "run"  # E244's results on E237's items
NEURON = json.loads((BASE / "neuron_0_1.json").read_text())["items"]
SIDES = {r["item"]: r for p in sorted((BASE / "crosscoder" / "sides").glob("*.json")) for r in [json.loads(p.read_text())]}
SHORT = {"cipher": "cipher", "prior_cipher": "prior-\nmatched\ncipher", "copy": "copy", "elicit": "mid", "rank1": "top"}
PANELS = [
    ("r", "re-gating share, log-odds", "regating"),
    ("kl_share", "re-gating share, KL", "regating"),
    ("prior", "log10 prior of trained answer", "prior"),
    ("size", "update size ||dW|| / ||W0||", "size"),
    ("erank", "effective rank of the update", "spectrum"),
    ("R_head_mlp", "R at heads and MLPs", "neuron"),
    ("R_mlp_attn", "R at MLP and attention", "neuron"),
    ("R_block", "R at blocks", "neuron"),
    ("R_topk", "R at top neurons, p97", "neuron_half"),
    ("R_vpd", "R at VPD subcomponents", "neuron_half"),
    ("top_k64", "top-neuron share, k 64", "neuron"),
    ("top_kmatched", "top-neuron share, k matched", "neuron"),
    ("r_n", "neuron share, whole width", "neuron"),
    ("crosscoder_d", "crosscoder: shared minus\nfine-tune-specific part", "steering"),  # steering is not drawn here
]
# E244's readings of one item record: the neuron file's items, or the crosscoder's side files, keyed by item name.
BASELINES: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]] = {
    "top_k64": (NEURON, lambda r: r["top"]["k64"]["share"]),
    "top_kmatched": (NEURON, lambda r: r["top"]["kmatched"]["share"]),
    "r_n": (NEURON, lambda r: r["whole"]["r_n"]),
    "crosscoder_d": (SIDES, lambda r: r["d"]),
}
NOT_READ = "Not read on these five tasks:\n\n- steering share\n- the causal test at VPD\n  with only the importance\n  values refit (S)"
N = ("3 seeds per task (dots), every task scaled to copy's update size; grey band: 3 random updates per task, Gaussian "
     "at each seed's per-matrix norms, same size, each draw a grey x. r and the three neuron shares have no band: a random "
     "update this small moves the answer by under 1 log-odds, where they are undefined. The prior is one number per task. "
     "Crosscoder: copy's lowest dot is seed 0, the item that also repeated the earlier saved crosscoder reading.")


def num(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def baseline(name: str, order: list[str]) -> tuple[dict, dict, dict]:
    """Seeds, random band (min to max of the draws, None where undefined) and draws per task, from E244's files."""
    recs, get = BASELINES[name]
    seeds = {t: [num(get(recs[f"e58/{t}_s{s}_scaled"])) for s in range(3)] for t in order}
    rand = {t: [num(get(recs[f"random_e58/{t}_s{s}"])) for s in range(3)] for t in order}
    bands = {t: (min(v), max(v)) if (v := [x for x in rand[t] if x is not None]) else None for t in order}
    return seeds, bands, rand


def columns(name: str, order: list[str]) -> tuple[dict, dict | None, dict]:
    if name in BASELINES:
        return baseline(name, order)
    if name == "prior":
        return {t: [SUMMARY["log10_prior"][t]] for t in order}, None, {}
    row = SUMMARY["table"][name]
    return {t: row[t]["seeds"] for t in order}, {t: row[t]["band"] for t in order}, {t: row[t]["random"] for t in order}


def panel(ax, name: str, ylabel: str, method: str) -> list[dict]:
    order = SUMMARY["x_order_by_prior"]
    groups, bands, rand = columns(name, order)
    column_dots(ax, groups, method=method, ylabel=ylabel, bands=bands)
    if bands:  # the draws themselves, so a band thinner than a line still shows
        draws = [(i, v) for i, t in enumerate(order) for v in rand[t] if v is not None]
        r = METHODS["random"]
        ax.scatter([i + 0.18 for i, _ in draws], [v for _, v in draws], color=r.color, marker=r.marker, s=12, linewidths=0)
    ax.set_xticks(range(len(order)), [SHORT[t] for t in order], fontsize=6)
    rho = SUMMARY["spearman_vs_prior"].get(name)
    ax.set_title(ylabel + ("" if rho is None else f"\nrank order vs prior {rho:.1f}"), fontsize=7)
    ax.yaxis.label.set_size(6)
    ax.tick_params(axis="y", labelsize=6)
    band = {t: (bands or {}).get(t) or [None, None] for t in order}
    return [{"reading": name, "task": t, "seed": i, "value": v, "band_lo": band[t][0], "band_hi": band[t][1]}
            for t in order for i, v in enumerate(groups[t])]  # fmt: skip


def against(r_path: Path, out: Path) -> None:
    """The same readings with each task placed at R's task mean on the x-axis instead of in prior order: R from an
    experiments/proxies_vs_r JSON (its "key" section, the mean over seeds 0-2), e.g. results/r_a_vpd.json, R under
    wording A at VPD. The R_* panels of the original are left out: they are other R readings, not proxies."""
    rd = json.loads(r_path.read_text())
    rx, order = rd["key"], SUMMARY["x_order_by_prior"]
    panels = [p for p in PANELS if not p[0].startswith("R_")]
    words = {"r_a_vpd": "R under wording A at VPD", "r_a_sae": "R under wording A at the SAE"}.get(rd["name"], rd["name"])
    fig, axes = figure(size=(8.5, 8.0), nrows=3, ncols=3)
    data: list[dict] = []
    for ax, (name, ylabel, method) in zip(axes.flat, panels, strict=False):
        groups, _, rand = columns(name, order)
        m, r =METHODS[method], METHODS["random"]
        for t in order:
            ys = [v for v in groups[t] if v is not None]
            ax.scatter([rx[t]] * len(ys), ys, color=m.color, marker=m.marker, s=16)
            ax.scatter([rx[t]] * len([v for v in rand.get(t, []) if v is not None]), [v for v in rand.get(t, []) if v is not None],
                       color=r.color, marker=r.marker, s=10, linewidths=0)  # fmt: skip
            if ys:
                ax.annotate(SHORT[t].replace("\n", " "), (rx[t], max(ys)), fontsize=5, xytext=(0, 3), textcoords="offset points",
                            ha={"rank1": "right", "elicit": "left"}.get(t, "center"))  # fmt: skip
            data += [{"reading": name, "task": t, "R": rx[t], "seed": i, "value": v} for i, v in enumerate(groups[t])]
        ax.set_title(ylabel, fontsize=7)
        ax.set_xlabel(f"{words}, task mean", fontsize=6)
        ax.set_ylabel(ylabel, fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Every reading on the five key tasks at one update size, against {words} (x: each task's mean over three "
                 "seeds);\ndots are seeds, grey x the size-matched random updates where the reading is defined", fontsize=8)  # fmt: skip
    save(fig, out, n=N, data=data)


def main() -> None:
    fig, axes = figure(size=(10.0, 10.5), nrows=4, ncols=4)
    flat, data = list(axes.flat), []
    for ax, (name, ylabel, method) in zip(flat, PANELS, strict=False):
        data += panel(ax, name, ylabel, method)
    flat[1].legend(fontsize=6, frameon=False, loc="lower right")
    for ax in flat[len(PANELS):]:
        ax.axis("off")
    flat[len(PANELS)].text(0.0, 1.0, NOT_READ, va="top", fontsize=8, transform=flat[len(PANELS)].transAxes)
    fig.suptitle("Every reading on the five key tasks at one update size, tasks ordered by the base model's prior (lowest "
                 "first):\nthe log-odds share splits the ciphers from the rest; the KL share sits in its random band; every R "
                 "sits below its random band;\nthe neuron shares set copy apart; the crosscoder sets no task apart and "
                 "reads mostly below its random band", fontsize=8)
    save(fig, HERE / "figures" / "columns.png", n=N, data=data)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--against", help="an R JSON with a 'key' section; draws each reading against it instead (see against())")
    ap.add_argument("--out", help="figure path for --against")
    a = ap.parse_args()
    if a.against:
        against(Path(a.against), Path(a.out))
    else:
        main()
