"""E190's figures, from results/stage0.json and stage2.json only (no model). Panels drawn by E189's plot.py (imported).

jobs.png   columns: retention, then gain kept under top-k per block at each k. Top row: per built pair the reuse-minus-new
           gap (dot), its seed-spread band (grey) and the random updates' gap (cross). Bottom row: each side's mean (filled
           reuse, open new) beside its random update (cross) and, for gain kept, its random-k floor (plus).
w0_mask.png  no update: per k, each task's target log-odds under the mask as a share of W0's unmasked (one dot per task);
           the Pile loss the mask adds is in the row label.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_jobs_k/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
EP = load_experiment("neuron_jobs", "plot")  # records, gaps, sides, ROWS
KS = (240, 768, 1536)


def view(score: dict[str, Any], k: int) -> dict[str, Any]:
    return {p: {"retention": s["retention"], "kept": s[f"kept_k{k}"], "kept_random": s[f"kept_random_k{k}"]} for p, s in score.items()}


def records(score: dict[str, Any]) -> list[dict[str, Any]]:
    out = [r for r in EP.records(view(score, KS[0])) if r["reading"] == "retention"]
    for k in KS:
        out += [{**r, "k": k} for r in EP.records(view(score, k)) if r["reading"] == "kept"]
    return out


def jobs(score: dict[str, Any], topk: dict[str, Any], figures: Path) -> None:
    recs = records(score)
    cols = [("retention", None)] + [("kept", k) for k in KS]
    fig, axes = FG.figure((11.0, 7.0), ncols=len(cols), nrows=2)
    for j, (reading, k) in enumerate(cols):
        rows = [r for r in recs if r["reading"] == reading and r.get("k") == k]
        EP.gaps(axes[0][j], rows, reading)
        EP.sides(axes[1][j], rows, reading)
        if k is not None:
            ok = "passes" if topk[f"k{k}"]["gap_over_0.4"] else "fails"
            axes[0][j].set_title(f"k {k} (top-k check {ok})", fontsize=8, pad=14)
            for t in axes[1][j].get_legend().get_texts():  # E189's sides() names its k-240 floor
                t.set_text(t.get_text().replace("random 240 neurons", f"random {k} neurons"))
            for ax, what in ((axes[0][j], "reuse side minus new side"), (axes[1][j], "side mean")):
                ax.set_xlabel(f"share of the gain kept under top-{k}\nper block, {what}")
        if j:
            for ax in axes[:, j]:
                ax.set_yticklabels([])
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.label.set_fontsize(7)
    calls = {c: sum(r["call"] in ("reuse side higher", "new side higher") for r in recs if (r["reading"], r.get("k")) == c) for c in cols}
    fig.suptitle(f"Neuron readings separate the built sides on {calls[cols[0]]} of {len(EP.ROWS)} pairs by retention; by gain kept "
                 + ", ".join(f"{calls[('kept', k)]} at k {k}" for k in KS), fontsize=8)  # fmt: skip
    FG.save(fig, figures / "jobs.png", n="2 to 8 saved fine-tune seeds per side; 2,000 Pile windows of 512 tokens; read rows as the "
            "share's, the read position; 4 blocks of 3072 neurons, 32 per block for retention; one random update per side", data=recs)  # fmt: skip


def w0_mask(w0: dict[str, Any], figures: Path) -> None:
    rows = {f"k {k} (Pile +{w0[f'k{k}']['pile_loss_added']:.2f} nats)": [1 + v["diff_over_T"] for v in w0[f"k{k}"]["tasks"].values()] for k in KS}
    data = [{"k": k, "task": t, **v, "pile_loss_added": w0[f"k{k}"]["pile_loss_added"]} for k in KS for t, v in w0[f"k{k}"]["tasks"].items()]
    fig, ax = FG.figure()
    FG.dots(ax, rows, method="neuron", xlabel="W0's target log-odds under the top-k mask,\nas a share of unmasked (one dot per task)")
    FG.reference(ax, 1.0, "unmasked")
    ax.set_title("What the top-k mask alone does to W0", fontsize=8)
    FG.save(fig, figures / "w0_mask.png", n=f"{len(data) // len(KS)} tasks' read rows at the read position; Pile val rows 4048-4304", data=data)


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    s0 = json.loads((results / "stage0.json").read_text())
    jobs(json.loads((results / "stage2.json").read_text())["score"], s0["C0_topk"], figures)
    w0_mask(s0["W0_masked"], figures)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
