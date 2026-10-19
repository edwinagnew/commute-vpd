"""Figures for lora_matched from results/lora_matched.json (or the smoke file with --smoke).

  per_matrix.png   one question: where does each update's energy sit, and how much of it can the subcomponents read?
                   Top: each matrix's share of the update's input-weighted energy, LoRA (scored step) beside ft (step 10).
                   Bottom: the inexpressible share per matrix for both, with the isotropic null as a grey tick.
  headline.png     one question: does the matched LoRA lift like the fine-tune? Left: e_delta of the four arms (LoRA and
                   ft, do-nothing and lifted) against the filed lines. Right: their Pile PGD ratios against the filed
                   lines, the W0 floor, and round 2's (L40) and round 4's (4090) references for the ft lift.

Usage (repo root):  uv run python experiments/lora_matched/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"LoRA": "#2a78d6", "ft": "#eb6834"}  # categorical slots 1 and 2 (validated), fixed order
GREY = "#7f7f7f"


def load(smoke: bool) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"lora_matched{'_smoke' if smoke else ''}.json").read_text())


def records(out: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """(LoRA seed 0 at its scored step, ft at step 10, the LoRA's step label)."""
    a = out["lora_s0"]
    step = str(a["scored_step"] or max(int(s) for s in a["steps"]))
    return a["steps"][step], out["ft"]["measured"], step


def per_matrix(out: dict[str, Any], path: Path) -> None:
    lora, ft, step = records(out)
    pl, pf = lora["inexpressible"]["per_matrix"], ft["inexpressible"]["per_matrix"]
    keys = sorted(pf, key=lambda k: -pf[k]["energy_share"])
    x = list(range(len(keys)))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    w = 0.38
    for off, (name, per) in ((-w / 2, ("LoRA", pl)), (w / 2, ("ft", pf))):
        top.bar([i + off for i in x], [per[k]["energy_share"] for k in keys], width=w, color=COLORS[name], label=name, linewidth=0)
        bot.scatter(x, [per[k]["omega_act"] for k in keys], s=36, color=COLORS[name], label=name, zorder=3)
    bot.scatter(x, [pf[k]["null_act"] for k in keys], marker="_", s=120, color=GREY, label="isotropic null", zorder=2)
    top.set_ylabel("share of the update's energy")
    bot.set_ylabel("inexpressible share")
    bot.set_ylim(-0.02, 1.0)
    bot.set_xticks(x, [k.replace("mlp.", "").replace("attn.", "") for k in keys], rotation=60, ha="right", fontsize=8)
    top.legend(frameon=False)
    bot.legend(frameon=False, loc="upper right")
    for ax in (top, bot):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    fig.suptitle(f"Where does each update's energy sit, and how much of it can the subcomponents read? (LoRA step {step}, ft step {out['ft']['step']})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def headline(out: dict[str, Any], path: Path) -> None:
    lora, ft, step = records(out)
    arms = [("LoRA nothing", "LoRA", lora["nothing"]), (f"LoRA lift (step {step})", "LoRA", lora["lift"]),
            ("ft nothing", "ft", ft["nothing"]), (f"ft lift (step {out['ft']['step']})", "ft", ft["lift"])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = list(range(len(arms)))
    labels = [a[0] for a in arms]
    colors = [COLORS[a[1]] for a in arms]
    left.bar(x, [a[2]["routes"]["shares"]["e_delta"] for a in arms], color=colors, width=0.6, linewidth=0)
    for y, txt in ((0.2, "M2 null (<= 0.2)"), (0.3, "M2 pass (>= 0.3)")):
        left.axhline(y, color=GREY, linewidth=1, linestyle="--")
        left.text(len(arms) - 0.5, y, txt, ha="right", va="bottom", fontsize=8, color=GREY)
    left.set_ylabel("e_delta (trigger gain outside the subcomponents)")
    left.set_title("How much of the behaviour stays in Delta?", fontsize=10)
    if all("pile" in a[2] for a in arms):
        right.bar(x, [a[2]["pile"]["pgd_kl"] for a in arms], color=colors, width=0.6, linewidth=0)
        fl = out["W0"]["pgd_floor"]
        lines = [(1 + 3 * fl, "W0 + 3 floors"), (2, "C2m null (<= 2)"), (5, "C2m pass (>= 5)"),
                 (out["ref"]["lift_ft_pgd_ratio"], "ft lift, round 2 (L40)"), (out["ref"]["round4_4090"]["lift_ft_pgd_ratio"], "ft lift, round 4 (4090)")]
        for y, txt in lines:
            right.axhline(y, color=GREY, linewidth=1, linestyle="--")
            right.text(-0.4, y, txt, ha="left", va="bottom", fontsize=8, color=GREY)
        right.set_ylabel("Pile PGD KL, ratio to W0")
        right.set_title("Is the lifted state adversarially faithful?", fontsize=10)
    for ax in (left, right):
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does the matched LoRA lift like the fine-tune?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = load(args.smoke)
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    sfx = "_smoke" if args.smoke else ""
    per_matrix(out, figs / f"per_matrix{sfx}.png")
    headline(out, figs / f"headline{sfx}.png")
    print(f"wrote {figs}")


if __name__ == "__main__":
    main()
