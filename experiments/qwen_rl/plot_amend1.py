"""E69 Amendment 1's figures and the unfiled readings beside the filed scores; no model, no GPU.

Reads results/amend1_score.json (the filed scores and the saved steps) and the per-problem answers. Writes:
  results/amend1_reading.json  step 100's rows, which the score has only for seed 1's null; leads, gains, gaps between seeds,
                               answer forms, openings, lengths and the sampled reading per seed: read, not filed
  figures/leads.png            RL minus the null per seed, grader and way of treating long answers
  figures/saved_steps.png      whole-answer gain from W0 at steps 10, 20, 50, 100, both arms, both seeds, three graders

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_rl/plot_amend1.py
"""

from __future__ import annotations

import collections
import json
import math
import statistics
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from relaxed import CI, HERE, RESULTS, ROOT, SEED1, Rec, ci, lead, paired, read, strict

SEED2 = "lr1e-6w10_s70"
SEEDS = {"seed 1": SEED1, "seed 2": SEED2}
SETS = ("gsm8k", "gsm1k")
FIG = HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
COL = {"R": "#2a78d6", "N": "#eb6834", "R-N": "#6250d6", "W0": "#52514e"}
ARM = {"R": "RL", "N": "shuffled-reward null"}
Key = Callable[[Rec], bool]
Rows = dict[int, Rec]


def lenient(r: Rec) -> bool:
    return bool(r["lenient"])


def capped_strict(r: Rec) -> bool:
    return bool(r["capped_strict"])


def capped_lenient(r: Rec) -> bool:
    return bool(r["capped_lenient"])


def capped_relaxed(r: Rec) -> bool:
    return ci({"response": r["capped_response"], "gold": r["gold"]})


GRADERS: dict[str, Key] = {"strict": strict, "relaxed": ci, "lenient": lenient}
CAPPED: dict[str, Key] = {"capped_strict": capped_strict, "capped_relaxed": capped_relaxed, "capped_lenient": capped_lenient}


def whole(a: Rows, b: Rows, key: Key) -> dict[str, Any]:
    return paired([int(key(a[i])) - int(key(b[i])) for i in b if i in a])


def gain(x: Rows, w0: Rows, key: Key) -> dict[str, Any]:
    return paired([int(key(x[i])) - int(key(w0[i])) for i in w0 if i in x and x[i]["fits"] and w0[i]["fits"]])


def arm_rows(x: Rows, w0: Rows) -> dict[str, Any]:
    out = {f"whole_{g}": whole(x, w0, k) for g, k in GRADERS.items()}
    out |= {f"drop_{g}": gain(x, w0, k) for g, k in GRADERS.items()}
    return out | {g: whole(x, w0, k) for g, k in CAPPED.items()}


def lead_rows(r: Rows, n: Rows, w0: Rows) -> dict[str, Any]:
    out = {f"drop_{g}": lead(w0, r, n, k) for g, k in GRADERS.items()}
    out |= {f"whole_{g}": whole(r, n, k) for g, k in GRADERS.items()}
    return out | {g: whole(r, n, k) for g, k in CAPPED.items()}


def opening(r: Rec) -> str:
    return " ".join(r["response"].split()[:2])


def forms(x: Rows) -> dict[str, Any]:
    """What the answers look like; the first three counts are over answers that fit the window."""
    fit = [r for r in x.values() if r["fits"]]
    return {"k": len(x), "fits": len(fit), "strict_line": sum(r["strict_pred"] is not None for r in fit),
            "relaxed_form": sum(bool(CI.findall(r["response"])) for r in fit),
            "right_when_fits_relaxed": sum(ci(r) for r in fit) / len(fit),
            "strict_differs_from_relaxed": sum(strict(r) != ci(r) for r in fit),
            "relaxed_differs_from_lenient": sum(ci(r) != lenient(r) for r in fit),
            "strict_differs_from_relaxed_all": sum(strict(r) != ci(r) for r in x.values()),
            "lenient_right_relaxed_wrong_all": sum(lenient(r) and not ci(r) for r in x.values()),
            "mean_tokens": statistics.mean(r["response_tokens"] for r in x.values()),
            "truncated": sum(r["truncated"] for r in x.values()),
            "openings": collections.Counter(opening(r) for r in x.values()).most_common(3)}


def seed_gaps(w0: Rows, sides: list[tuple[Rows, Rows]]) -> dict[str, Any]:
    """Seed 2's lead minus seed 1's, problem by problem at fixed weights: over all problems, and over those all five states finish."""
    (r1, n1), (r2, n2) = sides
    five = [i for i in w0 if all(x[i]["fits"] for x in (w0, r1, n1, r2, n2))]

    def gap(key: Key, pool: list[int]) -> dict[str, Any]:
        return paired([int(key(r2[i])) - int(key(n2[i])) - int(key(r1[i])) + int(key(n1[i])) for i in pool])

    out = {f"five_{g}": gap(k, five) for g, k in GRADERS.items()}
    out |= {f"whole_{g}": gap(k, list(w0)) for g, k in GRADERS.items()}
    return out | {g: gap(k, list(w0)) for g, k in CAPPED.items()}


def kept(s: dict[str, Any]) -> float:
    """Sampled answers lenient-right and inside the window, as a share of all: the reward with the format ignored."""
    return s["lenient_when_inside"] * s["inside"]


def reward(s: dict[str, Any]) -> float:
    return s["reward"]


def span_diff(x: dict[str, Any], y: dict[str, Any], f: Callable[[dict[str, Any]], float], steps: list[str]) -> dict[str, Any]:
    """Mean over steps of x minus y, SE over steps."""
    d = [f(x[t]) - f(y[t]) for t in steps]
    return {"n": len(d), "diff": statistics.mean(d), "se": statistics.stdev(d) / math.sqrt(len(d))}


def sampled() -> dict[str, Any]:
    """Per span, seed by seed, as plot.py's tables: SE over steps. Seed 2 has no F; the unchanged model there is seed 1's F,
    same prompts by step, other draws, so its zero line is not paired on draws as seed 1's is."""
    roll = {name: json.loads((RESULTS / folder / "rollouts_summary.json").read_text()) for name, folder in SEEDS.items()}
    w0 = roll["seed 1"]["F"]["steps"]
    out: dict[str, Any] = {}
    for name, r in roll.items():
        out[name] = {}
        for lo, hi in ((1, 10), (11, 20), (21, 50), (51, 75), (76, 95), (96, 100), (51, 100)):
            steps = [str(t) for t in range(lo, hi + 1)]

            row: dict[str, Any] = {}
            for k, x, y in (("R-W0", r["R"]["steps"], w0), ("N-W0", r["N"]["steps"], w0), ("R-N", r["R"]["steps"], r["N"]["steps"])):
                row[k] = {"reward": span_diff(x, y, reward, steps), "kept": span_diff(x, y, kept, steps)}
            for a in "RN":
                sp = r[a]["spans"].get(f"{lo}-{hi}")
                if sp:
                    row[a] = {k: sp[k] for k in ("line_when_inside", "lenient_when_inside", "inside")} | {"firsts": sp["firsts"]}
            out[name][f"{lo}-{hi}"] = row
    return out


def reading() -> dict[str, Any]:
    out: dict[str, Any] = {"note": "Read, not filed: predictions_amend1.json files only S0, S1, S1r, S2, B1, E_R10, A1, A2."}
    for s in SETS:
        w0 = read(ROOT / "acc" / f"W0_{s}.jsonl")
        sides = []
        row: dict[str, Any] = {"W0": forms(w0)}
        for name, folder in SEEDS.items():
            r, n = read(ROOT / folder / "acc" / f"R_{s}.jsonl"), read(ROOT / folder / "acc" / f"N_{s}.jsonl")
            sides.append((r, n))
            row[name] = {"lead": lead_rows(r, n, w0), "R": arm_rows(r, w0), "N": arm_rows(n, w0),
                         "forms": {"R": forms(r), "N": forms(n)}}
        row["seed2_minus_seed1"] = seed_gaps(w0, sides)
        out[s] = row
    out["saved_forms"] = {name: {f"{a}{t}": forms(read(ROOT / folder / "acc" / f"{a}{t}_gsm8k.jsonl")) for a in "RN" for t in (10, 20, 50)}
                          for name, folder in SEEDS.items()}
    out["sampled"] = sampled()
    return out


def tidy(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)


def fig_leads(read_: dict[str, Any]) -> None:
    pools = (("drop", "answers all three\nstates finish (filed)"), ("whole", "whole answers\n(read afterwards)"), ("capped", "past the window counts\nwrong (read afterwards)"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, grader, title in zip(axes, ("strict", "relaxed"), ("Strict grader", "Format-relaxed grader"), strict=True):
        for j, seed in enumerate(SEEDS):
            cols = [read_["gsm8k"][seed]["lead"][f"drop_{grader}"], read_["gsm8k"][seed]["lead"][f"whole_{grader}"], read_["gsm8k"][seed]["lead"][f"capped_{grader}"]]
            x = [i + (j - 0.5) * 0.2 for i in range(3)]
            y, e = [100 * c["diff"] for c in cols], [100 * c["se"] for c in cols]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=9, capsize=3, lw=1.6, color=COL["R-N"], mec=COL["R-N"], mew=1.8, mfc=COL["R-N"] if j == 0 else "white", label=seed)
            for a, v, er in zip(x, y, e, strict=True):
                ax.annotate(f"{v:+.2f}", (a, v + er + 0.15), ha="center", color=INK, fontsize=9)
        ax.axhline(0, color=INK, lw=1, label="RL level with the null")
        ax.set_xticks(range(3), [p[1] for p in pools], fontsize=9)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.6, 11)
        ax.set_title(title, color=INK, fontsize=11)
        tidy(ax)
    axes[0].set_ylabel("RL minus null, same problems, points", color=INK)
    axes[0].legend(frameon=False, loc="upper left", ncol=3)
    fig.suptitle("Does RL lead the null on both seeds, under each way of treating long answers?", color=INK, fontsize=12)
    fig.text(0.01, 0.005, "GSM8K test. Bars: one standard error over problems at fixed weights.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(FIG / "leads.png", dpi=150)
    plt.close(fig)


def fig_saved(score: dict[str, Any], read_: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    steps = (10, 20, 50, 100)
    panels = (("strict", "Strict: unchanged model right on 75.8%"), ("relaxed", "Format-relaxed: unchanged model right on 78.5%"), ("lenient", "Lenient: unchanged model right on 78.9%"))
    for ax, (grader, title) in zip(axes, panels, strict=True):
        for a, arm in enumerate("RN"):
            for b, (name, folder, ls, face) in enumerate((("seed 1", SEED1, "-", COL[arm]), ("seed 2", SEED2, "--", "white"))):
                x = [i + (2 * a + b - 1.5) * 0.035 for i in range(4)]
                saved = score["saved"]["states"][folder]
                pts = [saved[f"{arm}{t}"][f"whole_{grader}"] for t in steps[:3]] + [read_["gsm8k"][name][arm][f"whole_{grader}"]]
                ax.errorbar(x, [100 * p["diff"] for p in pts], yerr=[100 * p["se"] for p in pts], fmt="o", ls=ls, lw=1.8, ms=7, capsize=2.5,
                            color=COL[arm], mfc=face, mec=COL[arm], label=f"{ARM[arm]}, {name}")
        ax.axhline(0, color=COL["W0"], lw=1.4, label="unchanged model: the zero line")
        ax.set_ylim(-4, 13.5)
        ax.set_xticks(range(4), [str(t) for t in steps])
        ax.set_xlim(-0.3, 3.3)
        ax.set_xlabel("saved step (evenly spaced, not to scale)", color=INK)
        ax.set_title(title, color=INK, fontsize=11)
        tidy(ax)
    axes[0].set_ylabel("gain from the unchanged model, points", color=INK)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Does accuracy climb steadily across saved steps, in RL and in the null?", color=INK)
    fig.text(0.01, 0.005, "GSM8K test, all 1,319 problems on whole answers. Bars: one standard error over problems at fixed weights.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(FIG / "saved_steps.png", dpi=150)
    plt.close(fig)


def main() -> None:
    score = json.loads((RESULTS / "amend1_score.json").read_text())
    out = reading()
    (RESULTS / "amend1_reading.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    fig_leads(out)
    fig_saved(score, out)


if __name__ == "__main__":
    main()
