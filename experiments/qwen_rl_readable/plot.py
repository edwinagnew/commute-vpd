"""E163 run 1 figure: output change against held-out pass rate for each R recipe, with W0, the eligibility bar, the
2x/5x/10x targets, and N and F at step 100. Reads results/*.json and selection.json; writes figures/run1.png."""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import figure, reference, save

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RECIPES = {  # pod: (label, colour, marker)
    "control": ("RL, peak 1e-6", "#0072B2", "o"),
    "r3e-6": ("RL, peak 3e-6", "#E69F00", "s"),  # its step labels go below the point (step 10 sits on the 1e-6 step 25)
    "r1e-5": ("RL, peak 1e-5", "#D55E00", "^"),
}
NULLS = {"N": ("shuffled-reward null, step 100", "#555555", "X"), "F": ("own-samples fine-tune, step 100", "#009E73", "D")}


def gsm_reads(pod: str) -> list[dict]:
    reads = json.loads((RES / f"{pod}.json").read_text())["reads"]
    return [{"pod": pod, "step": int(s), "multiple": r["multiple"], "pass": r["pass"], "fits": r["fits"]}
            for s, r in sorted(reads.items(), key=lambda kv: int(kv[0])) if r.get("pass") is not None]


def main() -> None:
    sel = json.loads((RES / "selection.json").read_text())
    bar, fits_bar, w0 = sel["bar"], sel["fits_bar"], sel["w0_pass"]
    fig, ax = figure((7.5, 4.8))
    data: list[dict] = []
    for pod, (label, colour, marker) in RECIPES.items():
        rows = gsm_reads(pod)
        data += rows
        ax.plot([r["multiple"] for r in rows], [r["pass"] for r in rows], color=colour, linewidth=1, label=label)
        for r in rows:
            held = r["fits"] >= fits_bar
            ax.scatter(r["multiple"], r["pass"], color=colour, marker=marker, s=36, zorder=3,
                       facecolors=colour if held else "white", edgecolors=colour)
            ax.annotate(str(r["step"]), (r["multiple"], r["pass"]), xytext=(4, -10 if pod == "r3e-6" else 3), textcoords="offset points",
                        fontsize=7, color=colour)
    for pod, (label, colour, marker) in NULLS.items():
        (r,) = gsm_reads(pod)
        data.append(r)
        ax.scatter(r["multiple"], r["pass"], color=colour, marker=marker, s=44, zorder=3, label=label)
    ax.axhline(w0, color="#999999", linestyle=":", linewidth=1, label=f"unchanged model's pass rate ({w0:.3f})")
    ax.axhline(bar, color="#333333", linestyle="--", linewidth=1, label=f"bar: RL peak 1e-6 at step 100, less 0.03 ({bar:.3f})")
    ax.scatter([], [], marker="o", facecolors="white", edgecolors="#333333", label="hollow: too few answers fit the window")
    for k in (2, 5, 10):
        reference(ax, k, f"{k}x")
    ax.set_xlim(0, 10.6)
    ax.set_xlabel("output change on 128 maths rows, as a multiple of the decomposition's masked error")
    ax.set_ylabel("held-out GSM8K pass rate")
    ax.set_title("No RL recipe reached 5x with its pass rate and answer lengths held", fontsize=10)
    ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.31))
    save(fig, HERE / "figures" / "run1.png", data=data,
         n="one seed (draw seed 69); pass over the 1319 GSM8K test problems whose answers fit the window; "
           "numbers beside points are training steps; masked error 0.0932 nats per token")


if __name__ == "__main__":
    main()
