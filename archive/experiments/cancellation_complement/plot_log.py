"""The research log's four figures for E20, redrawn from saved results with plain labels (no model).

  log_excess.png  per updated model, the adversary's extra damage over the original model's, split into the part
                  from removing the alive original subcomponents, the part from their changes (with how the two
                  combine), and the rest (dead subcomponents and Δ, alone and combined with the alive removal).
                  Reads results/complement.json.
  log_offset.png  per model, how much less the adversary's whole mask costs than removing the alive block and the
                  dead-and-Δ block one at a time, summed: the offset between the two removals (Amendment 1).
                  Reads results/complement_amend1.json.
  log_steps.png   the adversary's damage against its number of steps at the original model and at ' which' then
                  ' was', under a fixed step of 0.1 (Amendment 2, seed 0's random start) and under the VPD paper's
                  rule, step size 2 / steps (Amendment 3). Reads results/complement_amend2_strong.json and
                  results/complement_amend3.json.
  log_budget.png  the adversarial ratio of five models against the number of steps, the paper's rule, with the 1.5
                  fail line; solid divided by the original model's seed 0, thin dotted by its seed 1 (Amendment 3).
                  Reads results/complement_amend3.json.

Usage (repo root):  python experiments/cancellation_complement/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import HERE, ORDER, save

RES = HERE / "results"
INK, MUTED = "#222222", "#9a9a9a"
BLUE, ORANGE, GREY, RED = "#3b6fb6", "#eb6834", "#7f7f7f", "#c0392b"
WORDS = {"adjoint emoticon": "adjoint split, emoticon fine-tune", "ct emoticon": "component training, emoticon",
         "ct r1_t588_v452": "component training, ' will' then ' have'",
         "ct r2_t359_v816": "component training, ' we' then ' just'",
         "ct r0_t352_v310": "component training, ' it' then ' is'",
         "ct r1_t534_v369": "component training, ' which' then ' was'",
         "ct r2_t275_v2667": "component training, ' in' then ' Fig'",
         "ls emoticon": "least-squares split, emoticon fine-tune"}
FAILS = ("ct r0_t352_v310", "ct r1_t534_v369", "ct r2_t275_v2667", "ls emoticon")


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def label(n: str) -> str:
    return f"{WORDS[n]} ({'fails' if n in FAILS else 'passes'})"


def style(ax: Any) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_excess(out: dict[str, Any]) -> None:
    """Excess X = O + (F_adv + I) + R per state, each part the state minus W0 (complement.json's symbols)."""
    w0, st = out["W0"], out["states"]
    rows = list(ORDER)[::-1]
    parts = (("removing the alive original subcomponents", BLUE,
              lambda r: r["F_orig"] - w0["F_orig"]),
             ("removing their changes as well (what E17 measured)", ORANGE,
              lambda r: r["F_both"] - r["F_orig"]),
             ("the rest: dead subcomponents and Δ, alone and with the alive removal", GREY,
              lambda r: (r["F_all"] - r["F_both"]) - (w0["F_all"] - w0["F_orig"])))
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pos, neg = np.zeros(len(rows)), np.zeros(len(rows))
    for lab, c, f in parts:
        v = np.array([f(st[n]) for n in rows])
        ax.barh(y, v, left=np.where(v >= 0, pos, neg), color=c, height=0.62, label=lab)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    for i, n in enumerate(rows):
        x = st[n]["F_all"] - w0["F_all"]
        ax.text(pos[i] + 0.04, i, f"{x:.2f}", va="center", fontsize=8.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [label(n) for n in rows], fontsize=8.5)
    ax.set_xlabel("adversary's damage at the updated model minus at the original model (KL, nats)")
    ax.set_title("Little of the extra damage comes from removing the original subcomponents;\n"
                 "most falls in the rest of the mask (on ' in' then ' Fig' about a fifth is on the originals)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    save(fig, "log_excess.png")


def fig_offset(a1: dict[str, Any]) -> None:
    """Offset = F_both + F_rest - F_all: removing each block alone, summed, minus removing both at once."""
    rows = ["W0", *list(ORDER)][::-1]
    rec = {"W0": a1["W0"], **a1["states"]}
    v = np.array([rec[n]["F_both"] + rec[n]["F_rest"] - rec[n]["F_all"] for n in rows])
    colors = [INK if n == "W0" else (RED if n in FAILS else MUTED) for n in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(y, v, color=colors, height=0.62)
    for i, x in enumerate(v):
        ax.text(x + (0.03 if x >= 0 else -0.03), i, f"{x:.2f}", va="center", ha="left" if x >= 0 else "right",
                fontsize=8.5)
    ax.axvline(0, color=INK, lw=1)
    names = ["original model" if n == "W0" else label(n) for n in rows]
    ax.set_yticks(y, names, fontsize=8.5)
    ax.set_xlim(-2.2, 1.3)
    ax.set_xlabel("cost of removing the alive block alone + the dead block and Δ alone − both at once (KL, nats)")
    ax.set_title("At the original model the two removals offset each other by 0.90 nats;\n"
                 "on the failing models most or all of it is gone, and on ' in' then ' Fig' both at once cost more than the sum", fontsize=10)
    style(ax)
    save(fig, "log_offset.png")


def fig_steps(a2: dict[str, Any], a3: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for m, name, c in (("W0", "original model", INK), ("ct r1_t534_v369", "' which' then ' was' (fails)", RED)):
        fx = a2["kl"][m]["seed0"]
        sx = sorted(int(k) for k in fx)
        ax.plot(sx, [fx[str(k)] for k in sx], ls="--", marker="o", color=c, mfc="white",
                label=f"{name}, fixed step of 0.1")
        pr = a3["kl"][m]["0"]
        px = sorted(int(k) for k in pr)
        ax.plot(px, [pr[str(k)] for k in px], ls="-", marker="o", color=c, label=f"{name}, paper's rule (step 2 / steps)")
        ax.text(px[-1] * 1.04, pr[str(px[-1])], f"{pr[str(px[-1])]:.2f}", va="center", fontsize=8.5, color=c)
        ax.text(sx[-1] * 1.04, fx[str(sx[-1])], f"{fx[str(sx[-1])]:.1f}", va="center", fontsize=8.5, color=c)
    ax.axvline(20, color=MUTED, ls=":", lw=1)
    ax.text(21, 40, "20 steps: where the project\nreads every adversarial verdict", fontsize=8, color=GREY, va="top")
    ax.set_yscale("log")
    ax.set_xlim(0, 185)
    ax.set_xlabel("number of steps the adversary takes")
    ax.set_ylabel("adversary's damage (KL, nats, log scale)")
    ax.set_title("On these two models, run longer with a fixed step, the damage keeps climbing;\n"
                 "with the paper's step rule it barely grows from 20 to 160 steps", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style(ax)
    save(fig, "log_steps.png")


def fig_budget(a3: dict[str, Any]) -> None:
    kl, steps = a3["kl"], (20, 40, 80, 160)
    models = ("ct r2_t275_v2667", "ct r1_t534_v369", "ls emoticon", "ct r0_t352_v310", "adjoint emoticon")
    colors = dict(zip(models, ("#8e44ad", RED, "#2c7fb8", ORANGE, GREY), strict=True))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for n in models:
        for seed, ls, lw in (("0", "-", 1.8), ("1", ":", 1.0)):
            r = [kl[n]["0"][str(s)] / kl["W0"][seed][str(s)] for s in steps]
            ax.plot(steps, r, ls=ls, lw=lw, marker="o" if seed == "0" else None, color=colors[n])
        ax.text(170, kl[n]["0"]["160"] / kl["W0"]["0"]["160"], WORDS[n], va="center", fontsize=8, color=colors[n])
    ax.axhline(1.5, color=RED, ls="--", lw=1)
    ax.text(21, 1.53, "fail line, 1.5", color=RED, fontsize=8, va="bottom")
    ax.axhline(1, color=INK, lw=0.8)
    ax.plot([], [], color=INK, marker="o", label="divided by the original model's seed 0 (as at 20 steps)")
    ax.plot([], [], color=INK, ls=":", lw=1, label="divided by its seed 1 instead")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_yticks([0.8, 1, 1.5, 2, 3, 5, 7], ["0.8", "1", "1.5", "2", "3", "5", "7"])
    ax.minorticks_off()
    ax.set_xlim(17, 400)
    ax.set_xlabel("number of steps the adversary takes (step size 2 / steps)")
    ax.set_ylabel("adversarial ratio: damage at the model\n÷ damage at the original model (log)")
    ax.set_title("Every model stays on the same side of the fail line from 20 to 160 steps;\n"
                 "a second seed at the original model lowers every ratio, ' it' then ' is' to just above the line", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    style(ax)
    save(fig, "log_budget.png")


def main() -> None:
    fig_excess(load("complement.json"))
    fig_offset(load("complement_amend1.json"))
    a3 = load("complement_amend3.json")
    fig_steps(load("complement_amend2_strong.json"), a3)
    fig_budget(a3)


if __name__ == "__main__":
    main()
