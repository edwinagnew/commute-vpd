"""E190's figure, from results/stage0.json and stage2.json only (no model). Panels drawn by E189's plot.py (imported).

jobs.png
  top left    retention of the moved neurons' Pile jobs: per built pair the reuse-minus-new gap (dot), its seed-spread band
              (grey) and the random updates' gap (cross);
  top right   each side's mean retention (filled reuse, open new) against its random update of matched size (cross);
  bottom      gain kept under top-k per block at k 240, 768, 1536: each side's mean, its random update, and the random-k
              floor (plus). Each title gives the Pile loss the mask alone adds to W0 and the sparsity check's result
              (stage 0 C0_topk: an update placed on the most-kept neurons keeps more than a random-k mask by over 0.4).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/neuron_jobs_k/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
EP = load_experiment("neuron_jobs", "plot")  # records, gaps, sides, ROWS
KS = (240, 768, 1536)
XLIM = (-0.5, 1.5)  # gain kept: 0 = W0, 1 = the whole gain


def nan_back(x: Any) -> Any:
    """The run's JSON writes nan as the string "nan" (open_pair.jsonable): read it back as a float."""
    if isinstance(x, dict):
        return {k: nan_back(v) for k, v in x.items()}
    if isinstance(x, list):
        return [nan_back(v) for v in x]
    return math.nan if x == "nan" else x


def load(path: Path) -> dict[str, Any]:
    return nan_back(json.loads(path.read_text()))


def view(score: dict[str, Any], k: int) -> dict[str, Any]:
    return {p: {"retention": s["retention"], "kept": s[f"kept_k{k}"], "kept_random": s[f"kept_random_k{k}"]} for p, s in score.items()}


def records(score: dict[str, Any]) -> list[dict[str, Any]]:
    out = [r for r in EP.records(view(score, KS[0])) if r["reading"] == "retention"]
    for k in KS:
        out += [{**r, "k": k} for r in EP.records(view(score, k)) if r["reading"] == "kept"]
    return out


def retention_row(axes: Any, ret: list[dict[str, Any]]) -> None:
    EP.gaps(axes[0], ret, "retention")
    EP.sides(axes[1], ret, "retention")
    axes[1].get_legend().remove()  # one legend for the figure, in the empty panel
    axes[1].set_yticklabels([])
    n_ret = sum(r["call"] != "no call" for r in ret)
    axes[0].set_title(f"Retention gap: beyond seed range\non {n_ret} of {len(ret)} pairs", fontsize=8, pad=14)
    axes[1].set_title("Retention: every side below\nits random update", fontsize=8, pad=14)


def kept_panel(ax: Any, rows: list[dict[str, Any]], k: int, loss: float, ok: bool) -> None:
    """Side means, random update and random-k floor of gain kept at k; x cut to XLIM, off-scale points counted in a note."""
    EP.sides(ax, rows, "kept")
    ax.get_legend().remove()
    ax.set_xlim(*XLIM)
    xs = [r[c] for r in rows for c in ("reuse_mean", "new_mean", "reuse_random_update", "new_random_update", "reuse_random_k", "new_random_k")]
    off = sum(not XLIM[0] <= x <= XLIM[1] for x in xs if not math.isnan(x))
    ax.annotate(f"{off} off scale", (0.98, 1.0), xycoords="axes fraction", ha="right", va="bottom", fontsize=6, color="#555555")
    ax.set_xlabel(f"share of the gain kept under top-{k} per block,\nside mean")
    ax.set_title(f"k {k} of 3072: the mask alone adds {loss:.2f} nats\nof Pile loss to W0; sparsity check {'passed' if ok else 'FAILED'}",
                 fontsize=8, color="#222222" if ok else "#B00000", pad=12)  # fmt: skip


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    s0, score = load(results / "stage0.json"), load(results / "stage2.json")["score"]
    recs = records(score)
    fig, axes = FG.figure((10.0, 7.8), ncols=3, nrows=2)
    retention_row(axes[0], [r for r in recs if r["reading"] == "retention"])
    failed = [k for k in KS if not s0["C0_topk"][f"k{k}"]["gap_over_0.4"]]
    for j, k in enumerate(KS):
        kept_panel(axes[1][j], [r for r in recs if r.get("k") == k], k, s0["W0_masked"][f"k{k}"]["pile_loss_added"], k not in failed)
        if j:
            axes[1][j].set_yticklabels([])
    handles, labels = axes[1][0].get_legend_handles_labels()
    axes[0][2].set_axis_off()
    axes[0][2].legend(handles, [t.replace("random 240 neurons", "random k neurons (floor)") for t in labels], loc="center", fontsize=7,
                      title="dot: reuse-minus-new gap; grey bar:\nseed-spread band", title_fontsize=7)  # fmt: skip
    axes[0][2].text(0.5, 0.1, "Gain kept, x cut to -0.5..1.5: the points off\nscale are random updates, whose gain is under\n"
                    "0.05 log-odds, so their ratio is noise", ha="center", fontsize=7, color="#555555", transform=axes[0][2].transAxes)  # fmt: skip
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.label.set_fontsize(7)
    head = "sparsity check failed at every k" if len(failed) == len(KS) else f"sparsity check failed at k {failed}"
    fig.suptitle("Neurons a fine-tune moves keep their Pile jobs less than a random update's do; "
                 f"gain kept under top-k is not read ({head})", fontsize=9)  # fmt: skip
    FG.save(fig, figures / "jobs.png", n="2 to 8 saved fine-tune seeds per side; 2,000 Pile windows of 512 tokens (retention), 256 "
            "held-out rows (Pile loss); read rows as the share's, the read position; 32 of 3072 neurons per block for retention; "
            "one random update per side", data=recs)  # fmt: skip


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
