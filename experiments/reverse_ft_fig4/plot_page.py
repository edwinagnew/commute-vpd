"""E312 (P2.84) figures for the analysis page reports/analyses/2026-11-25_analysis_p2-84-reverse-ft.md. Reads
results/summary.json only (no GPU, no model).

  figures/page_per_task.png   each task's three seeds: half-life (top) and D (bottom), state beside its rotated null,
                              tasks in Fig 4's built pairs (side built to reuse, then side built new), the planted
                              wrapper and the D-matched scaled ciphers as labelled reference lines
  figures/page_size.png       half-life against D for every item: states by side, nulls, gain-only, planted pair

    ~/commute-agents/fleet/cpu-run python3 experiments/reverse_ft_fig4/plot_page.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.figures import FULL_WIDTH, METHODS, figure, save

HERE = Path(__file__).resolve().parent
# Fig 4's built pairs (experiments/known_3seeds/results/known3.json .cells[].states): reuse side, then new side.
ORDER = [("c3/rank1", "reuse"), ("c3/copy", "reuse"), ("c3/cipher", "new"),
         ("e46/ident32", "reuse"), ("e46/rand32", "new"),
         ("e53/lowercase", "reuse"), ("e53/cipher", "new"),
         ("e54/elicit", "reuse"), ("e54/cipher", "new"), ("e54/cipher_own", "new"),
         ("e58/rank1", "reuse"), ("e58/copy", "reuse"), ("e58/cipher", "new"), ("e58/prior_cipher", "new"),
         ("e62/copy", "reuse"), ("e62/lookup", "new")]  # fmt: skip
SIDE = {"reuse": METHODS["built_reuse"], "new": METHODS["built_new"]}
NULL = METHODS["random"]
GAIN = "#009E73"


def rows(S: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for task, side in ORDER:
        for name, v in S["groups"][task]["states"].items():
            for kind in ("state", "null"):
                r = v[kind]
                out.append({"task": task, "side": side, "item": name, "kind": kind, "D": r["D"], "t_half": r["t_half"]})
    return out


def per_task(S: dict[str, Any], data: list[dict[str, Any]]) -> None:
    fig, (top, bot) = figure((9.0, 6.0), nrows=2)
    for i, (task, side) in enumerate(ORDER):
        for kind, dx, st in (("state", -0.12, SIDE[side]), ("null", 0.12, NULL)):
            pts = [r for r in data if r["task"] == task and r["kind"] == kind]
            for ax, key in ((top, "t_half"), (bot, "D")):
                ax.scatter([i + dx] * len(pts), [r[key] for r in pts], color=st.color, marker=st.marker, s=22, alpha=0.85, linewidths=0)
    wrap = S["planted"]["wrapper"]["t_half"]
    partners = [r["t_half"] for r in S["planted"]["partners"].values()]
    top.axhline(wrap, color="#555555", lw=0.8, ls="--", label="planted wrapper (positive control)")
    top.axhline(sum(partners) / len(partners), color="#555555", lw=0.8, ls=":", label="cipher scaled to the wrapper's D, mean of 3")
    for st, lab in ((SIDE["reuse"], "state, task built to reuse"), (SIDE["new"], "state, task built new"), (NULL, "rotated null, same per-matrix norm")):
        top.scatter([], [], color=st.color, marker=st.marker, s=22, label=lab)
    for ax in (top, bot):
        ax.set_yscale("log")
        ax.set_xticks(range(len(ORDER)), [t for t, _ in ORDER], rotation=55, ha="right", fontsize=7)
    top.set_xticklabels([])
    top.set_ylabel("steps to undo half of D")
    bot.set_ylabel("D = KL(original || fine-tuned)\nat the read position, nats")
    top.legend(fontsize=6.5, loc="upper right", ncols=2, frameon=False)
    top.set_title("Every task's fine-tune is undone by half within 17 steps except the text cipher;\n"
                  "the rotated nulls start 10-250 times closer to the original and mostly halve faster", fontsize=9)
    save(fig, HERE / "figures" / "page_per_task.png", n="16 tasks x 3 seeds; each state and its null reverse fine-tuned once; "
         "y axes log; null matched on per-matrix norm, not on D", data=data)  # fmt: skip


def size(S: dict[str, Any], data: list[dict[str, Any]]) -> None:
    fig, ax = figure(FULL_WIDTH)
    for side, st in SIDE.items():
        pts = [r for r in data if r["kind"] == "state" and r["side"] == side]
        ax.scatter([r["D"] for r in pts], [r["t_half"] for r in pts], color=st.color, marker=st.marker, s=22, alpha=0.85, linewidths=0,
                   label=f"state, task built {'to reuse' if side == 'reuse' else 'new'}")  # fmt: skip
    nulls = [r for r in data if r["kind"] == "null"]
    ax.scatter([r["D"] for r in nulls], [r["t_half"] for r in nulls], color=NULL.color, marker=NULL.marker, s=22, alpha=0.85, linewidths=0,
               label="rotated null, same per-matrix norm")  # fmt: skip
    gains = list(S["gain"].values())
    ax.scatter([r["D"] for r in gains], [r["t_half"] for r in gains], color=GAIN, marker="^", s=22, alpha=0.85, linewidths=0,
               label="gain-only fine-tune (a wrapper by construction)")  # fmt: skip
    w = S["planted"]["wrapper"]
    ax.scatter([w["D"]], [w["t_half"]], color="black", marker="*", s=70, label="planted wrapper")
    ps = list(S["planted"]["partners"].values())
    ax.scatter([r["D"] for r in ps], [r["t_half"] for r in ps], facecolors="none", edgecolors="black", marker="o", s=30,
               label="cipher scaled to the wrapper's D")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D = KL(original || fine-tuned) at the read position, nats")
    ax.set_ylabel("steps to undo half of D")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_title("The half-life does not follow D across tasks:\nthe nulls sit 10-250 times lower in D and mostly halve faster", fontsize=9)
    data2 = data + [{"task": "gain", "side": "", "item": k, "kind": "gain", "D": r["D"], "t_half": r["t_half"]} for k, r in S["gain"].items()]
    save(fig, HERE / "figures" / "page_size.png", n="48 states, 48 nulls, 15 gain-only states, 1 planted wrapper, 3 scaled ciphers; one "
         "reverse fine-tune each", data=data2)  # fmt: skip


def main() -> None:
    S = json.loads((HERE / "results" / "summary.json").read_text())
    data = rows(S)
    per_task(S, data)
    size(S, data)
    print("wrote figures/page_per_task.png, figures/page_size.png")


if __name__ == "__main__":
    main()
