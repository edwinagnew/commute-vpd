"""The research log's four figures for E18, redrawn from saved results with plain labels (no model).

Every recovery is a share of what each input's own updated importance values give back at the trigger (0 = the
original model's values, 1 = the input's own updated values). Bars are the mean of the two seeds, dots each seed.

  log_fixed.png     per task: another input's updated values (results/regating_leak_amend3.json, mismatch); the
                    updated model's average values minus the original model's average values (regating_leak.json,
                    (mean - mean_W0) / trig); the fixed change added to each input's original values, clipped
                    (regating_leak.json, shared / trig, the filed test); the change on random subcomponents
                    (random_mean / trig); a different target's change (cross / trig).
  log_where.png     per task, the change kept only on some subcomponents, not clipped (regating_leak_amend2.json):
                    the whole change, the whole change shuffled among each matrix's alive subcomponents (mean of 20
                    draws), the bin turned down by more than 0.1, the bins from -0.1 to 0.3 summed, the bin above 0.3.
  log_leftover.png  per task (regating_leak_amend4.json): the input's own updated values with the fixed change removed,
                    another input's such leftover (mean of 5 reshufflings), and the original model's average values
                    with no change; line at 0.15, the most the own leftover could give back under part (c).
  log_natural.png   per task (regating_leak.json, natural): overlap (Jaccard) of the switched-on set with the
                    subcomponents the original model switches on before the target token in Pile text, against
                    random sets of the same size.

Usage (repo root):  python experiments/regating_leak/plot_log.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import HERE, save

RES = HERE / "results"
TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
NAME = {"emoticon": "emoticon\n(' :)' then 'o')", "r1_t534_v369": "' which' then ' was'",
        "r2_t359_v816": "' we' then ' just'"}
INK, MUTED, BLUE, GREEN, ORANGE, RED, PURPLE = "#333333", "#bdbdbd", "#3b6fb6", "#1baf7a", "#eb6834", "#c0392b", "#9b59b6"
SEEDS = ("0", "1")
YLABEL = "share of what each input's own updated values give back\n(0 = original values, 1 = own updated values)"

Getter = Callable[[dict[str, Any]], float]


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(TASKS)), [NAME[n] for n in TASKS], fontsize=9)


def grouped(ax: Any, data: dict[str, Any], bars: tuple[tuple[str, str, Getter], ...], label: bool = True) -> None:
    """Bars per task at the seed mean, one dot per seed; data["tasks"][task]["seeds"][seed] is passed to each getter."""
    w = 0.8 / len(bars)
    for b, (lab, c, get) in enumerate(bars):
        for i, n in enumerate(TASKS):
            ys = [get(data["tasks"][n]["seeds"][k]) for k in SEEDS]
            x = i + (b - (len(bars) - 1) / 2) * w
            m = float(np.mean(ys))
            ax.bar(x, m, w * 0.9, color=c, label=lab if i == 0 else None, zorder=0.4)
            ax.plot([x] * len(ys), ys, "o", color=INK, ms=3)
            if label:
                ax.text(x, m + (0.02 if m >= 0 else -0.06), f"{m:.2f}", ha="center", fontsize=7)


def fig_fixed(base: dict[str, Any], a3: dict[str, Any]) -> None:
    joined = {"tasks": {n: {"seeds": {k: {"b": base["tasks"][n]["seeds"][k], "a3": a3["tasks"][n]["seeds"][k]}
                                      for k in SEEDS}} for n in TASKS}}

    def rho(key: str) -> Getter:
        return lambda s: s["b"]["rho"][key] / s["b"]["rho"]["trig"]

    bars: tuple[tuple[str, str, Getter], ...] = (
        ("another input's updated values", BLUE, lambda s: s["a3"]["mismatch"]["R"]),
        ("the average pattern: the updated model's average values, less what the original's average gives", "#7fb2de",
         lambda s: rho("mean")(s) - rho("mean_W0")(s)),
        ("the fixed change added to each input's original values (the test written beforehand)", GREEN, rho("shared")),
        ("the fixed change moved to random subcomponents", MUTED, lambda s: s["b"]["random_mean"] / s["b"]["rho"]["trig"]),
        ("the fixed change of a fine-tune to a different target", ORANGE, rho("cross")),
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for y, txt in ((0.7, "0.7: pass line for the fixed change"), (0.3, "0.3: leak line")):
        ax.axhline(y, color=INK, ls=":", lw=0.9)
        ax.text(2.48, y + 0.01, txt, ha="right", va="bottom", fontsize=7.5, color=INK)
    grouped(ax, joined, bars)
    style(ax)
    ax.set_ylim(-0.4, 1.12)
    ax.set_ylabel(YLABEL, fontsize=8.5)
    ax.set_title("Another input's values and the average pattern give back most of the effect; the fixed change "
                 "clears 0.7 on emoticon only", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    save(fig, "log_fixed.png")


MIDDLE = ("(-0.1, -0.01]", "(-0.01, 0.01]", "(0.01, 0.03]", "(0.03, 0.1]", "(0.1, 0.3]")


def fig_where(a2: dict[str, Any]) -> None:
    def shuffled(s: dict[str, Any]) -> float:
        d = s["perm"]["noclip"]["R_draws"]
        return sum(d) / len(d)

    bars: tuple[tuple[str, str, Getter], ...] = (
        ("the whole fixed change", GREEN, lambda s: s["perm"]["noclip"]["R_unshuffled"]),
        ("the whole fixed change, shuffled among each matrix's subcomponents", MUTED, shuffled),
        ("only where the update turns importance down by more than 0.1", PURPLE, lambda s: s["bins"]["(-inf, -0.1]"]["R"]),
        ("only the small changes, -0.1 to 0.3 (summed)", "#f2b134", lambda s: sum(s["bins"][b]["R"] for b in MIDDLE)),
        ("only the switched-on set, raised by more than 0.3", BLUE, lambda s: s["bins"]["(0.3, inf]"]["R"]),
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    grouped(ax, a2, bars)
    style(ax)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel(YLABEL, fontsize=8.5)
    ax.set_title("The fixed change works only on its own subcomponents; the ones it turns down carry a share "
                 "comparable to the switched-on set", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    save(fig, "log_where.png")


def fig_leftover(a4: dict[str, Any]) -> None:
    bars: tuple[tuple[str, str, Getter], ...] = (
        ("each input's own updated values with the fixed change removed", RED, lambda s: s["R"]["resid"]),
        ("another input's leftover, the same way", "#e89a8f", lambda s: s["R"]["resid_donor"]),
        ("the original model's average values, no change at all", MUTED, lambda s: s["R"]["mean_W0"]),
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.15, color=INK, ls=":", lw=0.9,
               label="0.15: the most the leftover could give back if the fixed change were enough on its own")
    grouped(ax, a4, bars)
    style(ax)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel(YLABEL, fontsize=8.5)
    ax.set_title("The leftover gives back more than 0.15, but another input's leftover gives nearly as much",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    save(fig, "log_leftover.png")


def fig_natural(base: dict[str, Any]) -> None:
    nat = base["natural"]
    data = {"tasks": {n: {"seeds": {k: nat[n][f"s{k}"] for k in SEEDS}} for n in TASKS}}
    bars: tuple[tuple[str, str, Getter], ...] = (
        ("the switched-on set", BLUE, lambda s: s["J"]),
        ("random sets of the same size (mean of draws)", MUTED, lambda s: s["random_mean"]),
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    grouped(ax, data, bars)
    for i, n in enumerate(TASKS):
        note = f"{nat[n]['n_positions']} Pile positions"
        if not nat[n]["scorable"]:
            note += ",\nunder the 200 needed"
            m = float(np.mean([nat[n][f"s{k}"]["J"] for k in SEEDS]))
            ax.bar(i - 0.2, m, 0.36, facecolor="white", edgecolor=BLUE, hatch="//", zorder=0.5,
                   label="switched-on set, reported but not scored (too few positions)")
        ax.text(i, 0.335, note, ha="center", fontsize=7.5, color=INK)
    style(ax)
    ax.set_ylim(0, 0.37)
    ax.set_ylabel("overlap with the subcomponents the original model\nswitches on before the target in Pile text",
                  fontsize=8.5)
    ax.set_title("The switched-on set overlaps the model's usual subcomponents for the token far above chance",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    save(fig, "log_natural.png")


def main() -> None:
    base = load("regating_leak.json")
    fig_fixed(base, load("regating_leak_amend3.json"))
    fig_where(load("regating_leak_amend2.json"))
    fig_leftover(load("regating_leak_amend4.json"))
    fig_natural(base)


if __name__ == "__main__":
    main()
