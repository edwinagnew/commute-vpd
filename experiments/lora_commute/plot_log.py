"""The research log's four figures for E7, redrawn from saved results with plain labels (no model).

  log_faithful.png    the Pile adversarial ratio of every state (scored step), as bars from 1 (the unchanged
                      decomposition), with the pass line (1 + 3 x the unchanged decomposition's seed spread) and the
                      1.5 fail line.
  log_outside.png     the outside share of each update (LoRA ranks 1-32 at the scored step, fine-tune at step 10)
                      beside its random-update null.
  log_left_in_delta.png  the share of the trigger gain the least-squares split leaves in Δ, per LoRA rank and for the
                      fine-tune.
  log_routes.png      the route split of each mapped state: direct and re-switching shares only (the remainder mixes
                      Δ with the masked-off share, which can be negative, so it is not drawn).

Reads results/lora_commute.json only (three component-training arms were rebuilt from the run's log, see the report).
Usage (repo root):  python experiments/lora_commute/plot_log.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import arms, load

FIG = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREY, INK, RED, GREEN = "#2a6fb0", "#d9822b", "#9a9a9a", "#333333", "#c0392b", "#2e8b57"
RANKS = (1, 4, 8, 32)


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)


def fig_faithful(out: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    bars = [(f"LoRA rank {r}, least-squares split", rows[f"lift_{r}"], BLUE) for r in RANKS]
    bars += [("LoRA rank 8, do-nothing", rows["LoRA_8 nothing"], BLUE)]
    bars += [(f"component training, one matrix, rank {r}", rows[f"pb_r{r}"], ORANGE) for r in RANKS]
    bars += [("component training, one matrix, full rank", rows["pb_full"], ORANGE),
             ("component training, all 24 matrices", rows["pb_all"], GREEN),
             ("fine-tune, do-nothing", rows["ft nothing"], GREY),
             ("fine-tune, least-squares split", rows["ft lift"], GREY)]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    y = list(range(len(bars)))[::-1]
    for yi, (lab, row, col) in zip(y, bars, strict=True):
        v = row["pile"]["pgd_kl"]
        ax.barh(yi, v - 1, left=1, color=col, height=0.7, alpha=0.55 if "do-nothing" in lab else 0.95)
        ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=8)
    passline = 1 + 3 * out["W0"]["pgd_floor"]
    ax.axvline(passline, color=GREEN, lw=1, ls="--")
    ax.axvline(1.5, color=RED, lw=1, ls="--")
    ax.text(passline + 0.03, len(bars) + 0.35, f"pass line {passline:.2f}", color=GREEN, fontsize=8, ha="left")
    ax.text(1.53, len(bars) - 0.35, "fail line 1.5", color=RED, fontsize=8, ha="left")
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8.5)
    ax.set_xlim(1, 7.9)
    ax.set_ylim(-0.6, len(bars) + 0.8)
    ax.set_xlabel("adversarial ratio on Pile text (1 = as robust as the unchanged decomposition)")
    ax.set_title("On the one matrix every state is over the pass line;\nacross all 24 matrices component training sits just inside it", fontsize=10)
    style(ax)
    save(fig, "log_faithful.png")


def fig_outside(out: dict[str, Any]) -> None:
    S = out["sizes"]
    labs, vals, nulls = [], [], []
    for r in RANKS:
        a = out["lora"][str(r)]
        ix = a["steps"][str(a["scored_step"] or S["lora_ckpts"][-1])]["inexpressible"]
        labs.append(f"LoRA\nrank {r}")
        vals.append(ix["pooled"])
        nulls.append(ix["pooled_null"])
    ix = out["ft"]["steps"][str(S["ft_ckpts"][0])]["inexpressible"]
    labs.append("fine-tune\n(all 24 matrices)")
    vals.append(ix["pooled"])
    nulls.append(ix["pooled_null"])
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    w = 0.38
    for i, (v, n) in enumerate(zip(vals, nulls, strict=True)):
        col = GREY if i == len(vals) - 1 else BLUE
        ax.bar(i - w / 2, v, w, color=col, label="the update" if i == 0 else None)
        ax.bar(i + w / 2, n, w, color="white", edgecolor=INK, hatch="//",
               label="a random update on the same matrices" if i == 0 else None)
        ax.text(i - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, n + 0.01, f"{n:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labs)), labs, fontsize=8.5)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("outside share: part of the update\nno change of the pieces can produce")
    ax.set_title("The LoRA is no more readable than a random update;\nthe fine-tune sits well below its own random baseline", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "log_outside.png")


def fig_left_in_delta(rows: dict[str, dict[str, Any]]) -> None:
    labs = [f"LoRA\nrank {r}" for r in RANKS] + ["fine-tune"]
    vals = [rows[f"lift_{r}"]["routes"]["shares"]["e_delta"] for r in RANKS]
    vals.append(rows["ft lift"]["routes"]["shares"]["e_delta"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, v in enumerate(vals):
        ax.bar(i, v, 0.6, color=GREY if i == len(vals) - 1 else BLUE)
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labs)), labs, fontsize=8.5)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("share of the trigger gain\nleft in Δ after the split")
    ax.set_title("The split leaves 0.33 to 0.43 of the LoRA's gain in Δ,\n0.09 of the fine-tune's", fontsize=10)
    style(ax)
    save(fig, "log_left_in_delta.png")


def fig_routes(rows: dict[str, dict[str, Any]]) -> None:
    arms_ = [(f"LoRA rank {r}, split", f"lift_{r}") for r in RANKS]
    arms_ += [(f"component training, one matrix, rank {r}", f"pb_r{r}") for r in RANKS]
    arms_ += [("component training, one matrix, full rank", "pb_full"),
              ("component training, all 24 matrices", "pb_all"),
              ("fine-tune, split", "ft lift")]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = list(range(len(arms_)))[::-1]
    for i, (yi, (lab, k)) in enumerate(zip(y, arms_, strict=True)):
        sh = rows[k]["routes"]["shares"]
        d, r = sh["d"], sh["r"]
        first = i == 0
        ax.barh(yi, d, color=INK, height=0.7, label="moved pieces themselves (direct share)" if first else None)
        ax.barh(yi, r, left=d, color=BLUE, alpha=0.6, height=0.7,
                label="Γ switching pieces on differently (re-switching share)" if first else None)
        ax.text(d + 0.005, yi, f"{d:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [a[0] for a in arms_], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the trigger gain on held-out emoticon text")
    ax.set_title("Whatever the rank or the map, the moved pieces carry a tenth of the gain at most;\nΓ switching pieces on differently carries about half or more", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
    style(ax)
    save(fig, "log_routes.png")


def main() -> None:
    out = load(smoke=False)
    rows = arms(out)
    fig_faithful(out, rows)
    fig_outside(out)
    fig_left_in_delta(rows)
    fig_routes(rows)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
