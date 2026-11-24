"""P2.75 run 1's figure: each arm's specific part (spec 3.6) under both forms of nu, against the gate's common bar.

    python experiments/diff_sae/plot.py [--tag run1]     -> figures/<tag>_specific.png (+ .pdf, .csv)

An arm with no reading (a stopped fit, or a fit not yet landed) gets its slot and the reason, not a point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
# (item key, label); None splits the gate arms at site A from the descriptive ones.
ARMS = (
    ("head-r1.A.c0", "head, fine-\ntune size,\nseed 0"),
    ("head-r1.A.c1", "head, fine-\ntune size,\nseed 1"),
    ("random_d0-r1.A.c0", "matched\nGaussian,\ndraw 0"),
    ("random_d1-r1.A.c0", "matched\nGaussian,\ndraw 1"),
    ("head-r1.A.c0.k33", "head,\nseed 0,\nk 33 fit"),
    None,
    ("plant.A.c0", "exact\nplanted\nchange"),
    ("head-r5.A.c0", "head at\n100x\nsize"),
    ("head.B.c0", "block-3\nhead,\nsite B"),
    ("gauss.B.c0", "Gaussian,\nsite B"),
)
FORMS = (("lit", "nu from encoding h_F (code as published)", "#0072B2", "o"),
         ("D", "nu from encoding D (post's words)", "#D55E00", "s"))  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    a = ap.parse_args()
    res = HERE / "results" / a.tag
    gate = json.loads((res / "gate.json").read_text())
    fig, ax = figure((7.5, 3.8))
    xs, labels, rows, x, split = [], [], [], 0.0, 0.0
    for arm in ARMS:
        if arm is None:
            ax.axvline(x - 0.5, color="#999999", linewidth=0.8)
            ax.annotate("descriptive, not gated", (x - 0.4, 0.97), fontsize=7, color="#555555")
            split = x - 0.5
            continue
        key, label = arm
        p = res / "items" / f"{key}.json"
        d = json.loads(p.read_text()) if p.is_file() else {}
        if isinstance(d.get("read"), dict):
            first = not rows
            for j, (form, name, colour, marker) in enumerate(FORMS):
                v = d["read"][form]["published_0.3"]["specific"]
                ax.scatter([x + 0.14 * (2 * j - 1)], [v], color=colour, marker=marker, s=30, label=name if first else None)
                rows.append({"arm": key, "form": form, "specific": v, "bar": gate["K1"] * 0.5})
        else:
            why = {None: "k-33 fit:\npending", "dead share at the end": "dead share\n> 0.3, not read"}.get(
                d.get("stopped"), "non-finite\nloss, not read")
            ax.annotate(why, (x, 0.05), ha="center", fontsize=6.5, color="#555555")
        xs.append(x)
        labels.append(label)
        x += 1
    bar = gate["D_published"]["bar"]
    # drawn over the gate's arms only: the site-B arms are descriptive and have no bar
    ax.plot([-0.5, split], [bar, bar], color="k", linewidth=0.8, linestyle="--", label=f"gate bar, half the head's reference capacity: {bar:.2f}")
    ax.set_xlim(-0.5, x - 0.5)
    ax.set_xticks(xs, labels, fontsize=6.5)
    ax.set_ylabel("specific part of the change\n(nu <= 0.3, as published)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("No arm at a fine-tune's size reaches the bar; the Gaussians read over the head under nu from D", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, HERE / "figures" / f"{a.tag}_specific.png", n="one SAE fit per point (two seeds on the head), 1,536 latents at k 100, k 33 where marked; "
         "share of the change's energy on each side's score read windows", data=rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
