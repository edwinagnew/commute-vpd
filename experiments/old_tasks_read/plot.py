"""E292 step 3 (P2.74): how hard each task is for the 67M, old near-realistic tasks beside E272's realistic panel, on
the same four axes, one point per state (task and seed):

  prior          W0's mean log-odds of the target on the read rows (old: score.py `difficulty.prior`; panel: its read
                 file's `prior`, in its scored unit: log-odds, or the label contrast for a label task)
  gain           the log-odds gain at the scored state (old: step 10; panel: the wide set's kl_hi pick)
  steps to half  the first step whose gain is at least half the scored state's gain (old: the retrain's per-step curve,
                 NT.steps_to_half, saved-state tasks have none; panel: the first save whose held gain reaches half the
                 pick's, saves at steps 1-5, 8, 10, 11, 21, 43, ..., so a coarse upper bound)
  Pile loss      nats per token on Pile val rows 0-63, W0 minus W0 + dW (old: score.py; panel: results/e272_pile.json)

    ~/commute-agents/fleet/cpu-run python3 experiments/old_tasks_read/plot.py   (after score.py)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
RESULTS, FIG = HERE / "results", HERE / "figures"
E272 = EXPERIMENTS / "convergence_panel" / "results"
AXES = (("prior", "base log-odds of the target"), ("gain", "gain reached (log-odds)"), ("steps_to_half", "steps to half the gain"),
        ("pile_loss_change", "Pile loss change (nats/token)"))  # fmt: skip


def old_rows() -> list[dict[str, Any]]:
    diff = json.loads((RESULTS / "summary.json").read_text())["difficulty"]
    return [{"set": "old", "task": d["task"], "state": s} | {k: d.get(k) for k, _ in AXES} for s, d in diff.items()]


def half_step(saves: list[dict[str, Any]], step: int, gain: float) -> int | None:
    pts = sorted((s["step"], s["held_gain"]) for s in saves if "held_gain" in s and s["step"] <= step)
    return next((st for st, g in pts if g >= gain / 2), None) if gain > 0 else None


def panel_rows() -> list[dict[str, Any]]:
    sel = json.loads((E272 / "wide" / "selection.json").read_text())["kl"]["hi"]["tasks"]
    pile = json.loads((RESULTS / "e272_pile.json").read_text())["states"]
    out = []
    for task, by in sel.items():
        rd = json.loads((E272 / "wide" / "read" / f"{task}.json").read_text())["states"]
        tr = json.loads((E272 / "train" / f"{task}.json").read_text())["seeds"]
        for seed, s in by.items():
            r = rd.get(s["file"], {})
            saves = [x for x in tr.get(seed, []) if isinstance(x, dict) and "step" in x and "held_gain" in x]
            out.append({"set": "realistic", "task": task, "state": f"{task}_s{seed}", "prior": r.get("prior"), "gain": r.get("gain"),
                        "steps_to_half": half_step(saves, s["step"], s.get("held_gain", math.nan)),
                        "pile_loss_change": pile.get(s["file"], {}).get("pile_loss_change")})  # fmt: skip
    return out


def main() -> None:
    rows = old_rows() + panel_rows()
    FIG.mkdir(exist_ok=True)
    with (FIG / "difficulty.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "task", "state", *(k for k, _ in AXES)])
        w.writeheader()
        w.writerows(rows)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (k, label) in zip(axs, AXES, strict=True):
        for x, name, colour in ((0, "old", "tab:blue"), (1, "realistic", "tab:orange")):
            ys = [r[k] for r in rows if r["set"] == name and isinstance(r[k], int | float) and math.isfinite(r[k])]
            xs = [x + 0.18 * ((i * 7919) % 11 / 10 - 0.5) for i in range(len(ys))]
            ax.scatter(xs, ys, s=14, alpha=0.7, color=colour)
            if ys:
                ax.hlines(sorted(ys)[len(ys) // 2], x - 0.25, x + 0.25, color="k", lw=1.5)
            n_none = sum(1 for r in rows if r["set"] == name) - len(ys)
            if n_none:
                ax.text(x, 1.0, f"{n_none} not logged", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
        ax.set_xticks([0, 1], ["old tasks", "realistic panel"])
        ax.set_title(label, fontsize=10)
    fig.suptitle("How hard each task is for the 67M: one point per state, line at the median", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"difficulty.{ext}", dpi=150)
    for name in ("old", "realistic"):
        for k, _ in AXES:
            ys = sorted(r[k] for r in rows if r["set"] == name and isinstance(r[k], int | float) and math.isfinite(r[k]))
            print(f"{name} {k}: n {len(ys)}, median {ys[len(ys) // 2] if ys else None}, range {(ys[0], ys[-1]) if ys else None}")
    print(f"wrote {FIG / 'difficulty.png'}, .pdf, .csv: {len(rows)} states")


if __name__ == "__main__":
    main()
