"""E222 (P2.41 stage 3) figures, from run_stage3.py's per-state files and E220's KL in them; no model.

  stage3_gain.png         the gain on the read rows per behaviour and arm, seeds as dots, beside the 1 log-odds floor
                          and each behaviour's do-nothing Gaussian update
  stage3_gain_vs_kl.png   the same gains against E220's KL from W0 on Pile text: lower's twelve states fall on one line
Also writes results/stage3_gain_vs_kl.json: the least-squares line of gain on KL over lower's twelve states and its r.

    ~/commute-agents/fleet/cpu-run python3 experiments/rl_67m/plot_stage3.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from commute import figures
from commute.local_faith import MIN_GAIN

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ROWS = [("repeat", "rl"), ("repeat", "own"), ("repeat", "demo"), ("repeat", "shuffled"),
        ("lower", "rl"), ("lower", "own"), ("lower", "demo"), ("lower", "shuffled"), ("french_en", "demo")]  # fmt: skip
JOB = {"repeat": "1", "lower": "2", "french_en": "anchor"}
ARM = {"rl": ("RL", "#0072B2", "o"), "own": ("own samples", "#009E73", "^"), "demo": ("demonstrations", "#E69F00", "D"),
       "shuffled": ("shuffled reward", "#CC79A7", "v")}  # fmt: skip
LABEL = {"repeat": "repeat", "lower": "lowercase", "french_en": "French anchor"}


def states() -> list[dict]:
    out = []
    for b, arm in ROWS:
        for s in (0, 1, 2):
            r = json.loads((RES / "stage3" / f"{b}_{arm}_s{s}.json").read_text())
            out.append({"behaviour": b, "arm": arm, "seed": s, "gain": r["gain"]["lo"], "kl": r["e220"]["kl_saved"],
                        "rate_0": r["e220"]["rate_0"], "rate_stop": r["e220"]["rate_stop"]})  # fmt: skip
    return out


def gain_figure(recs: list[dict]) -> None:
    fig, ax = figures.figure(figures.FULL_WIDTH)
    names = [f"{LABEL[b]}, {ARM[a][0]}" for b, a in ROWS]
    for i, (b, a) in enumerate(ROWS):
        _, color, marker = ARM[a]
        xs = [r["gain"] for r in recs if (r["behaviour"], r["arm"]) == (b, a)]
        ax.scatter(xs, [i] * len(xs), color=color, marker=marker, s=26, alpha=0.9, linewidths=0)
    g = {b: json.loads((RES / f"stage3_{j}.json").read_text())["gauss"]["gain"]["lo"] for b, j in JOB.items()}
    gx = [g[b] for b, _ in ROWS]
    st = figures.METHODS["random"]
    ax.scatter(gx, [i + 0.3 for i in range(len(ROWS))], color=st.color, marker=st.marker, s=16,
               label="do-nothing: Gaussian update,\nsame size (under each row)")
    ax.set_yticks(range(len(ROWS)), names)
    ax.invert_yaxis()
    figures.reference(ax, MIN_GAIN, "floor, gain 1")
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.legend(loc="center right", fontsize=7, frameon=False)
    ax.set_xlabel("gain: change in log-odds of the text's own next token on the 200 read rows")
    ax.set_title("Two arms clear the floor: RL on repeat and the French anchor")
    figures.save(fig, HERE / "figures" / "stage3_gain.png", n="3 seeds per arm (dots), 200 read rows per behaviour, "
                 "every update at size 0.0069", data=recs + [{"behaviour": b, "arm": "gaussian", "gain": v} for b, v in g.items()])


def fit(recs: list[dict]) -> dict:
    lo = [r for r in recs if r["behaviour"] == "lower"]
    x, y = [r["kl"] for r in lo], [r["gain"] for r in lo]
    slope, icpt = statistics.linear_regression(x, y)
    out = {"what": "least-squares gain = slope * kl + intercept over lower's 12 states (rl, own, demo, shuffled x 3 seeds)",
           "slope": slope, "intercept": icpt, "r": statistics.correlation(x, y), "n": len(lo)}  # fmt: skip
    (RES / "stage3_gain_vs_kl.json").write_text(json.dumps(out, indent=1))
    return out


def kl_figure(recs: list[dict], line: dict) -> None:
    fig, ax = figures.figure(figures.FULL_WIDTH)
    shape = {"repeat": "o", "lower": "s", "french_en": "*"}
    for r in recs:
        ax.scatter(r["kl"], r["gain"], color=ARM[r["arm"]][1], marker=shape[r["behaviour"]], s=30, alpha=0.9, linewidths=0)
    for b, m in shape.items():
        ax.scatter([], [], color="#444444", marker=m, label=LABEL[b])
    for name, color, _ in ARM.values():
        ax.scatter([], [], color=color, marker="o", label=name)
    xs = [0.0, 4.2]
    ax.plot(xs, [line["slope"] * v + line["intercept"] for v in xs], color="#777777", linewidth=0.8, label="line through lowercase's 12 states")
    ax.axhline(MIN_GAIN, color="#777777", linewidth=0.8, linestyle="--", label="floor, gain 1")
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    ax.set_xlabel("KL from the original model on Pile text, nats per token (read in E220)")
    ax.set_ylabel("gain on the read rows, log-odds")
    ax.set_title("On lowercase the gain falls as the output drifts; on repeat RL rises above the drift")
    figures.save(fig, HERE / "figures" / "stage3_gain_vs_kl.png", n="27 states: 3 seeds per arm, 200 read rows each",
                 data=recs)


def main() -> None:
    recs = states()
    gain_figure(recs)
    kl_figure(recs, fit(recs))
    print(json.dumps(json.loads((RES / "stage3_gain_vs_kl.json").read_text())))


if __name__ == "__main__":
    main()
