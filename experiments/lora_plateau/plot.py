"""Figures for lora_plateau from results/lora_plateau.json (or the smoke file with --smoke).

  trajectory.png   one question: does the LoRA's read side ever become readable? Against the LoRA step (log): the pooled
                   outside_share share / its isotropic null (P1 lines and the fine-tune's 0.51), A's pooled overlap with
                   the alive read span against a random read side's, and |cos| of A with its initialisation; held-out
                   P(o) and the training loss as the plateau check.
  pile.png         one question: does the lifted LoRA state survive the Pile adversary at the step-matched points?
                   Lift and do-nothing Pile PGD ratios at the Pile steps against the W0 floor band, the P2/P3 lines
                   and round 5's fine-tune lift; e_delta of the lift beside.

Usage (repo root):  uv run python experiments/lora_plateau/plot.py [--smoke]
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
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#7f7f7f"


def load(smoke: bool) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"lora_plateau{'_smoke' if smoke else ''}.json").read_text())


def _line(ax: Any, y: float, txt: str, x: float = 0.0) -> None:
    ax.axhline(y, color=GREY, linewidth=1, linestyle="--")
    ax.text(x, y, txt, ha="left", va="bottom", fontsize=8, color=GREY)


def trajectory(out: dict[str, Any], path: Path) -> None:
    ck = out["checkpoints"]
    steps = sorted(int(s) for s in ck)
    ref = out["ref"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    r, ov, co, po = axes
    r.plot(steps, [ck[str(s)]["ratio"] for s in steps], "o-", color=BLUE)
    for y, txt in ((0.8, "P1 pass (>= 0.8)"), (0.6, "P1 null (<= 0.6)"), (ref["ft_lift_step10"]["inexpressible_ratio"], "fine-tune, step 10")):
        _line(r, y, txt, steps[0])
    r.set_ylim(0, 1.1)
    r.set_title("inexpressible share / isotropic null (pooled)", fontsize=10)
    ov.plot(steps, [ck[str(s)]["read_side"]["pooled_in_alive"] for s in steps], "o-", color=BLUE, label="A in the alive read span")
    ov.plot(steps, [ck[str(s)]["read_side"]["pooled_random"] for s in steps], "--", color=GREY, label="random read side")
    ov.set_ylim(0, 1)
    ov.legend(frameon=False, fontsize=8)
    ov.set_title("share of ||delta-W||^2 read inside the alive span", fontsize=10)
    co.plot(steps, [ck[str(s)]["cos_init"]["mean"] for s in steps], "o-", color=BLUE)
    co.set_ylim(0, 1.02)
    co.set_title("mean |cos| of A with its initialisation", fontsize=10)
    po.plot(steps, [ck[str(s)]["eval"]["p_o"] for s in steps], "o-", color=BLUE, label="held-out P(o)")
    po.plot([x["step"] for x in out["train_log"]], [x["loss"] for x in out["train_log"]], color=ORANGE, linewidth=0.8, label="training loss")
    po.set_ylim(0, max(1.05, max(x["loss"] for x in out["train_log"]) * 1.05))
    po.legend(frameon=False, fontsize=8)
    po.set_title(f"learned and at a plateau? (plateau ratio {out['plateau_ratio']:.2f})", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("LoRA step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.6)
    fig.suptitle("Does the LoRA's read side ever become readable by the released subcomponents?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pile(out: dict[str, Any], path: Path) -> None:
    ps = out["pile_steps"]
    steps = sorted(int(s) for s in ps)
    if not all("pile" in ps[str(s)]["lift"] for s in steps):
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    x = list(range(len(steps)))
    w = 0.38
    left.bar([i - w / 2 for i in x], [ps[str(s)]["nothing"]["pile"]["pgd_kl"] for s in steps], width=w, color=GREY, label="do-nothing", linewidth=0)
    left.bar([i + w / 2 for i in x], [ps[str(s)]["lift"]["pile"]["pgd_kl"] for s in steps], width=w, color=BLUE, label="lift", linewidth=0)
    fl = out["W0"]["pgd_floor"]
    for y, txt in ((1 + 3 * fl, "W0 + 3 floors"), (2, "P2/P3 pass (<= 2, with e_delta <= 0.2)"), (5, "P2/P3 null (>= 5)"),
                   (out["ref"]["ft_lift_step10"]["pgd_ratio"], "fine-tune lift, step 10 (round 5)")):
        _line(left, y, txt, -0.4)
    left.set_xticks(x, [f"step {s}" for s in steps])
    left.set_ylabel("Pile PGD KL, ratio to W0")
    left.legend(frameon=False, loc="center left")
    left.set_title("Is the lifted LoRA state adversarially faithful?", fontsize=10)
    right.bar(x, [ps[str(s)]["lift"]["routes"]["shares"]["e_delta"] for s in steps], color=BLUE, width=0.6, linewidth=0)
    _line(right, 0.2, "pass needs e_delta <= 0.2", -0.4)
    right.set_xticks(x, [f"step {s}" for s in steps])
    right.set_ylabel("e_delta of the lift")
    right.set_title("How much of the behaviour stays in Delta?", fontsize=10)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does the lifted LoRA state survive the adversary at the step-matched points?")
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
    trajectory(out, figs / f"trajectory{sfx}.png")
    pile(out, figs / f"pile{sfx}.png")
    print(f"wrote {figs}")


if __name__ == "__main__":
    main()
