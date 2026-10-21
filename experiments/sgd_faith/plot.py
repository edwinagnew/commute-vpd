"""Figure for sgd_faith: the SGD state beside the Adam reference on the three readings the predictions use, each panel
with its pass line: (a) Pile ratios to W0, (b) E4's trigger-local ratio under the state's and W0's gates, (c) the route
split's shares at the trigger.

Usage (repo root): uv run python experiments/sgd_faith/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
ARMS = (("adam", "Adam, lr 1e-5", "C1"), ("sgd", "plain SGD, lr 0.01", "C0"))
PILE = (("stochastic_kl", "stochastic"), ("pgd_kl", "adversarial"), ("ci_masked_kl", "CI-masked"))
SHARES = (("d", "changed pieces,\nold gates"), ("r", "re-gating"), ("i", "interaction"), ("e_gate", "ungated\npieces"),
          ("e_delta", "outside\ncomponents"))


def bars(ax: Axes, out: dict, labels: list[str], values: dict[str, list[float]]) -> None:
    w = 0.38
    for n, (arm, label, colour) in enumerate(ARMS):
        if arm in out:
            ax.bar([i + (n - 0.5) * w for i in range(len(labels))], values[arm], w, color=colour, label=label)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    suffix = "_smoke" if ap.parse_args().smoke else ""
    out = json.loads((HERE / "results" / f"sgd_faith{suffix}.json").read_text())
    arms = [a for a, _, _ in ARMS if a in out]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4))

    bars(a, out, [n for _, n in PILE], {x: [out[x]["measure"]["pile"][k] for k, _ in PILE] for x in arms})
    f = out["W0"]["pgd_floor"]
    a.axhline(1.0, color="grey", lw=0.8, label="W0 (1 by definition)")
    a.axhline(1.5, color="k", ls=":", label=f"pass line 1.5 (W0 adversarial floor {f:.2f})")
    a.set(ylabel="Pile KL, ratio to W0", title="(a) Pile faithfulness")
    a.legend(fontsize=7)

    bars(b, out, ["state's gates", "W0's gates"],
         {x: [out[x]["trigger"]["state_gated"], out[x]["trigger"]["w0_gated"]] for x in arms})
    b.axhline(0.8, color="k", ls=":", label="0.8: carried (state's gates)")
    b.axhline(0.5, color="k", ls="--", lw=0.8, label="0.5: not carried (W0's gates)")
    b.set(ylabel="CI-masked P(o) / target P(o) at the trigger", ylim=(0, 1.05), title="(b) Trigger, masked model")
    b.legend(fontsize=7)

    bars(c, out, [n for _, n in SHARES], {x: [out[x]["measure"]["routes"]["shares"][k] for k, _ in SHARES] for x in arms})
    c.axhline(0.1, color="k", ls=":", label="0.1: the log's 'a tenth or less'")
    c.axhline(0.0, color="grey", lw=0.8)
    c.set(ylabel="share of the log-odds gain at the trigger", title="(c) Route split")
    c.legend(fontsize=7)

    fig.suptitle("Saved SGD emoticon state against the Adam reference, same job, same card"
                 + (" (smoke sizes)" if suffix else ""))
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"faithfulness{suffix}.png", dpi=130)


if __name__ == "__main__":
    main()
