"""The research log's three figures for E24, redrawn from saved results with plain labels (no model).

All read results/regating_small.json: the emoticon states under "states", the ' which' then ' was' states under
"extra.states", E18's readings of the large states under "refs.big". Bars are grouped by task; the large Adam update
is dark blue, the small ones green. One training seed per state.

  log_switching.png  per state: the switching share r, the part of the fine-tune's gain on the target that comes from
                     frozen Gamma switching subcomponents on when it reads the updated model; E18's reading of each
                     large state as a black tick.
  log_fixed.png      per state: the fixed change's share, R_shared = (fixed change's recovery) / (own values'
                     recovery), with the 0.7 and 0.3 lines written beforehand; grey dots: the fixed change moved to
                     random subcomponents (random_mean / own values' recovery, mean of 20 draws).
  log_trigger.png    per state, as a fraction of the switching effect (not divided by anything): each input's own
                     updated values at the trigger (rho.trig, the denominator of every share) and the fixed change of
                     a fine-tune to ' http' at the same trigger (rho.cross).

Usage (repo root):  python experiments/regating_small/plot_log.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from plot import HERE, save

INK, MUTED, BLUE, GREEN, ORANGE = "#333333", "#9e9e9e", "#3b6fb6", "#1baf7a", "#eb6834"
TASK = {"emoticon": "emoticon (' :)' then 'o')", "r1_t534_v369": "' which' then ' was'"}
X = (0.0, 1.0, 2.0, 3.6, 4.6)  # a gap between the two tasks

Getter = Callable[[dict[str, Any]], float]


def rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """States in run order: emoticon large, SGD 0.01, Adam 3e-6; then ' which' then ' was' large and SGD 0.03."""
    return list(out["states"].values()) + list(out["extra"]["states"].values())


def is_large(v: dict[str, Any]) -> bool:
    return v["optimizer"] == "adam" and v["lr"] == 3e-5


def tick(v: dict[str, Any]) -> str:
    opt = "Adam" if v["optimizer"] == "adam" else "SGD"
    size = v["heldout"]["rel_norm_max"]
    kind = "large" if is_large(v) else ("small" if size < 0.003 else "smaller")
    rate = f"{v['lr']:g}".replace("e-0", "e-")
    return f"{opt} {rate}\n{kind}\nsize {size:.4f}"


def axes(out: dict[str, Any], ylabel: str, ylim: tuple[float, float], title: str) -> Any:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(X, [tick(v) for v in rows(out)], fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    for x, name in ((1.0, TASK["emoticon"]), (4.1, TASK["r1_t534_v369"])):
        ax.text(x, ylim[1], name, ha="center", va="top", fontsize=9, color=INK, fontweight="bold")
    ax.axvline(2.8, color=MUTED, lw=0.6, ls="--")
    return fig, ax


def bars(ax: Any, out: dict[str, Any], get: Getter, width: float = 0.55, dx: float = 0.0,
         colors: tuple[str, str] = (BLUE, GREEN)) -> None:
    for x, v in zip(X, rows(out), strict=True):
        y = get(v)
        ax.bar(x + dx, y, width, color=colors[0] if is_large(v) else colors[1], zorder=2)
        ax.text(x + dx, y + (0.02 if y >= 0 else -0.06), f"{y:.2f}", ha="center", fontsize=8, color=INK)


def legend_patches(ax: Any, items: tuple[tuple[str, str], ...], ncol: int = 2) -> None:
    for lab, c in items:
        ax.bar(0, 0, color=c, label=lab)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=ncol)


def fig_switching(out: dict[str, Any]) -> None:
    fig, ax = axes(out, "switching share: part of the fine-tune's gain that comes\nfrom Γ switching subcomponents on",
                   (0, 1.08), "Switching share: about the same for the two small emoticon updates, higher for the smaller\n' which' then ' was' update (one seed each)")
    bars(ax, out, lambda v: v["r"])
    for x, v in zip(X, rows(out), strict=True):
        if is_large(v):
            ref = out["refs"]["big"][v["task"]]["r"]
            ax.plot([x - 0.32, x + 0.32], [ref, ref], color=INK, lw=2, zorder=3,
                    label="E18's reading of the same large update" if x == 0 else None)
    legend_patches(ax, (("the large Adam update", BLUE), ("a smaller update", GREEN)))
    save(fig, "log_switching.png")


def fig_fixed(out: dict[str, Any]) -> None:
    fig, ax = axes(out, "share of what each input's own updated values give back\n(0 = original values, 1 = own "
                   "updated values)", (-0.15, 1.08),
                   "One fixed change gives back about the same share for small and large updates (one seed each):\nabove 0.7 on emoticon, "
                   "between the lines on ' which' then ' was'")
    lines = ((0.7, ":", "0.7 or more: a fixed pattern"), (0.3, "--", "0.3 or less: needs each input's own values"))
    for y, ls, txt in lines:
        ax.axhline(y, color=INK, ls=ls, lw=0.9, label=txt)
    bars(ax, out, lambda v: v["R_shared"])
    ax.scatter(X, [v["random_mean"] / v["rho"]["trig"] for v in rows(out)], color=MUTED, s=28, zorder=3,
               label="the fixed change moved to random subcomponents")
    legend_patches(ax, (("the large Adam update", BLUE), ("a smaller update", GREEN)))
    save(fig, "log_fixed.png")


def fig_trigger(out: dict[str, Any]) -> None:
    fig, ax = axes(out, "fraction of the switching effect given back\n(0 = original values, 1 = updated values "
                   "everywhere)", (-0.42, 1.12),
                   "Each input's own values at the trigger give back most of the effect;\n"
                   "the fixed change of a fine-tune to another target works against it")
    bars(ax, out, lambda v: v["rho"]["trig"], width=0.36, dx=-0.19, colors=(BLUE, BLUE))
    bars(ax, out, lambda v: v["rho"]["cross"], width=0.36, dx=0.19, colors=(ORANGE, ORANGE))
    legend_patches(ax, (("each input's own updated values at the trigger (the denominator of every share)", BLUE),
                        ("the fixed change of a fine-tune to ' http', same trigger", ORANGE)), ncol=1)
    save(fig, "log_trigger.png")


def main() -> None:
    out = json.loads((HERE / "results" / "regating_small.json").read_text())
    fig_switching(out)
    fig_fixed(out)
    fig_trigger(out)
    print(f"wrote {HERE / 'figures'}/log_*.png")


if __name__ == "__main__":
    main()
