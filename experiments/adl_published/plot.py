"""E284 (cell P2.67): the activation difference lens's positive controls, as published, on the same axes.

Left: the authors' toolkit on the paper's three Qwen3-1.7B organisms, the rise in cross-entropy on each organism's
training documents when the fine-tuned model's activation along the mean difference is set to the base model's, per
position, beside the mean random-difference baseline and filed rule (a)'s bar. Right: the same quantity on the 67M's
filed control (E53 lowercase_s0), averaged over positions 0-4 as the stage-0 rule reads it, beside the 64 random
differences' same averages.

    ~/commute-agents/fleet/cpu-run python3 experiments/adl_published/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as F

HERE = Path(__file__).parent
ORGS = {"cake_bake": "cake baking", "kansas_abortion": "Kansas abortion vote", "fda_approval": "FDA approval"}


def main() -> None:
    ctrl = json.loads((HERE / "results/control.json").read_text())
    s0 = json.loads((HERE / "results/stage0.json").read_text())["control"]
    fig, (left, right) = F.figure(F.FULL_WIDTH, ncols=2)
    data, pos = [], ctrl["rule"]["positions"]
    for org, name in ORGS.items():
        y = ctrl["per_organism"]["default"][org]["train"]["incr_ce"]
        left.plot(pos, y, marker="o", markersize=3, linewidth=1, alpha=0.6, label=name)
        data += [{"panel": "qwen", "series": org, "position": p, "incr_ce": v} for p, v in zip(pos, y)]
    mean = ctrl["mean_curves"]["train_default"]
    left.plot(pos, mean, color="black", linewidth=2, label="mean of the three")
    rnd = ctrl["mean_curves"]["train_random_diff_mean"]
    left.plot(pos, rnd, color=F.METHODS["random"].color, linestyle=":", linewidth=1.5, label="random differences, mean")
    bar = ctrl["a"]["bar"]
    left.axhline(bar, color="#777777", linewidth=0.8, linestyle="--", label="rule (a) bar: random mean + 2 sd")
    data += [{"panel": "qwen", "series": "mean", "position": p, "incr_ce": v} for p, v in zip(pos, mean)]
    data += [{"panel": "qwen", "series": "random_diff_mean", "position": p, "incr_ce": v} for p, v in zip(pos, rnd)]
    left.set_xlabel("token position j")
    left.set_ylabel("rise in cross-entropy on training data, nats")
    left.set_title("Qwen3-1.7B, the paper's organisms: passes", fontsize=9)
    left.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.07))

    rand = s0["random_diff_like_mean"]
    right.scatter([0] * len(rand), rand, color=F.METHODS["random"].color, marker="X", s=18, alpha=0.6, linewidths=0,
                  label="64 random differences")  # fmt: skip
    right.scatter([0], [s0["lens_like_mean"]], color="black", marker="o", s=40, zorder=3, label="mean difference (the lens)")
    right.axhline(s0["random_diff_like_max"], color="#777777", linewidth=0.8, linestyle="--", label="pass bar: largest random")
    right.set_xticks([0], ["lowercase fine-tune"])
    right.set_xlim(-1, 1)
    right.set_ylabel("rise in cross-entropy on training data, nats\n(mean over positions 0-4)")
    right.set_title("67M, the filed control: fails", fontsize=9)
    right.legend(fontsize=7, frameon=False, loc="center right")
    curves = [max(c["train"]["incr_ce"]) for c in ctrl["per_organism"]["default"].values()]
    top = 1.08 * max(s0["random_diff_like_max"], *curves)
    for ax in (left, right):
        ax.set_ylim(-0.002, top)  # one scale for both panels: the 67M's largest random difference is the tallest point
    data += [{"panel": "67M", "series": "random_diff", "k": k, "incr_ce_mean_0_4": v} for k, v in enumerate(rand)]
    data += [{"panel": "67M", "series": "lens", "incr_ce_mean_0_4": s0["lens_like_mean"]}]
    fig.suptitle("The lens's causal effect clears its random baseline on the paper's models, not on the 67M control")
    F.save(fig, HERE / "figures/controls.png",
           n="Qwen: 3 organisms, up to 1,000 training documents each, layer 13; 67M: 256 training windows, block 1 output",
           data=data)  # fmt: skip


if __name__ == "__main__":
    main()
