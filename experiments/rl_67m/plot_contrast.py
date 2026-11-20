"""E223 (P2.41 run 5) figures, from run_contrast.py's per-state files, E222's per-state files and the ctl's E53 reading;
no model.

  contrast_gain.png       the gain per behaviour and arm in the two-form contrast beside E222's gain in log-odds of the
                          text's own next token, the 1 log-odds floor and the do-nothing Gaussian update
  contrast_share.png      the re-gating share per readable seed, the rotated-untrained share beside, E53's lowercase
                          contrast share and E222's French anchor (log-odds) as references
  contrast_numerator.png  the share's numerator against its gain, both in the contrast, every state: where a state sits
                          against the line "share 1" and the line "numerator 0"
Also writes results/contrast_numerator_vs_kl.json: Pearson r of the share's numerator on E220's KL over lower's 12 states.

    ~/commute-agents/fleet/cpu-run python3 experiments/rl_67m/plot_contrast.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from commute import figures
from commute.jobs import load_experiment
from commute.local_faith import MIN_GAIN

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
P3 = load_experiment("rl_67m", "plot_stage3")  # E222's arm colours and behaviour labels
ARM, LABEL = P3.ARM, P3.LABEL
ROWS = [(b, a) for b in ("repeat", "lower") for a in ("rl", "own", "demo", "shuffled")]
UNTRAINED = ("#555555", "x")


def num(x: object) -> float:
    return float("nan") if x in (None, "nan") else float(x)  # type: ignore[arg-type]


def states() -> list[dict]:
    out = []
    for b, arm in ROWS:
        for s in (0, 1, 2):
            c = json.loads((RES / "contrast" / f"{b}_{arm}_s{s}.json").read_text())
            lo = json.loads((RES / "stage3" / f"{b}_{arm}_s{s}.json").read_text())
            out.append({"behaviour": b, "arm": arm, "seed": s, "gain": c["gain"]["contrast"], "gain_lo_e222": lo["gain"]["lo"],
                        "share": num(c["share"]["share"]["contrast"]), "numerator": c["share"]["numerator"]["contrast"],
                        "share_gain": c["share"]["gain"]["contrast"], "rotated": num(c["untrained"]["share"]["contrast"]["rotated"]),
                        "kl": c["e220"]["kl_saved"]})  # fmt: skip
    return out


def gain_figure(recs: list[dict]) -> None:
    fig, ax = figures.figure(figures.FULL_WIDTH)
    for i, (b, a) in enumerate(ROWS):
        _, color, marker = ARM[a]
        rs = [r for r in recs if (r["behaviour"], r["arm"]) == (b, a)]
        ax.scatter([r["gain"] for r in rs], [i] * 3, color=color, marker=marker, s=28, linewidths=0)
        ax.scatter([r["gain_lo_e222"] for r in rs], [i + 0.28] * 3, facecolors="none", edgecolors=color, marker=marker, s=22)
    g = {b: json.loads((RES / f"contrast_{j}.json").read_text())["gauss"]["gain"]["contrast"] for b, j in (("repeat", "1"), ("lower", "2"))}
    st = figures.METHODS["random"]
    ax.scatter([g[b] for b, _ in ROWS], [i - 0.28 for i in range(len(ROWS))], color=st.color, marker=st.marker, s=14,
               label="do-nothing: Gaussian update, same size (contrast)")
    ax.scatter([], [], color="#444444", marker="o", s=28, label="filled: gain in the two-form contrast (this run)")
    ax.scatter([], [], facecolors="none", edgecolors="#444444", marker="o", s=22, label="hollow: gain in log-odds of the text's token (E222)")
    ax.set_yticks(range(len(ROWS)), [f"{LABEL[b]}, {ARM[a][0]}" for b, a in ROWS])
    ax.invert_yaxis()
    figures.reference(ax, MIN_GAIN, "floor, gain 1")
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_xlabel("gain on the 200 read rows: updated model minus original")
    ax.set_title("Scored as a contrast, RL's lowercase gain is positive")
    figures.save(fig, HERE / "figures" / "contrast_gain.png", n="3 seeds per arm, 200 read rows per behaviour, every update "
                 "at size 0.0069; E222 read other rows: the text's own token, copying rows on repeat", data=recs + [{"behaviour": b, "arm": "gaussian", "gain": v} for b, v in g.items()])


def share_figure(recs: list[dict]) -> None:
    fig, ax = figures.figure(figures.FULL_WIDTH)
    names = []
    for i, (b, a) in enumerate(ROWS):
        rs = [r for r in recs if (r["behaviour"], r["arm"]) == (b, a) and not math.isnan(r["share"])]
        _, color, marker = ARM[a]
        ax.scatter([r["share"] for r in rs], [i] * len(rs), color=color, marker=marker, s=28, linewidths=0)
        ax.scatter([r["rotated"] for r in rs], [i + 0.28] * len(rs), color=UNTRAINED[0], marker=UNTRAINED[1], s=16)
        names.append(f"{LABEL[b]}, {ARM[a][0]} ({len(rs)} of 3 readable)")
    anchor = [json.loads((RES / "stage3" / f"french_en_demo_s{s}.json").read_text())["share"]["share"]["lo"] for s in (0, 1, 2)]
    ax.scatter(anchor, [len(ROWS)] * 3, color=ARM["demo"][1], marker="*", s=40, linewidths=0)
    names.append("French anchor, demonstrations\n(log-odds, E222: other unit)")
    e53 = json.loads((RES / "contrast_ctl.json").read_text())["e53 lowercase_s0 contrast (beside)"]["share"]["share"]["contrast"]
    figures.reference(ax, e53, f"E53 lowercase, contrast: {e53:.2f}")
    ax.scatter([], [], color=UNTRAINED[0], marker=UNTRAINED[1], s=16, label="same ratio on W0's rotated SVD (untrained null)")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    ax.set_xlim(-1.45, 1.45)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_xlabel("re-gating share in the two-form contrast: 0 none of the gain, 1 all of it")
    ax.set_title("On lowercase, RL's share is negative")
    figures.save(fig, HERE / "figures" / "contrast_share.png", n="readable seeds only (gain at least 1), 200 read rows per "
                 "behaviour; repeat's own, demonstration and shuffled arms have no readable seed", data=recs)


def numerator_figure(recs: list[dict]) -> None:
    fig, ax = figures.figure(figures.FULL_WIDTH)
    for r in recs:
        _, color, marker = ARM[r["arm"]]
        filled = r["behaviour"] == "lower"
        ax.scatter(r["share_gain"], r["numerator"], marker=marker, s=30, linewidths=0.9, facecolors=color if filled else "none", edgecolors=color)
    for name, color, marker in ARM.values():
        ax.scatter([], [], color=color, marker=marker, label=name)
    ax.scatter([], [], facecolors="none", edgecolors="#444444", label="hollow: repeat; filled: lowercase")
    ax.plot([-2, 10], [-2, 10], color="#777777", linewidth=0.8, label="share 1: all of the gain through changed CI values")
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    figures.reference(ax, MIN_GAIN, "floor, gain 1")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_xlabel("gain in the contrast, T1 - T0 (the share's denominator)")
    ax.set_ylabel("numerator, L(C0,G1) - L(C0,G0)")
    ax.set_title("Lowercase RL: new CI values alone favour capitals")
    figures.save(fig, HERE / "figures" / "contrast_numerator.png", n="24 states, 3 seeds per arm, 200 read rows each; both "
                 "axes the mean over rows in log p(a) - log p(b)", data=recs)


def kl_fit(recs: list[dict]) -> dict:
    lo = [r for r in recs if r["behaviour"] == "lower"]
    rl = [r for r in lo if r["arm"] in ("rl", "shuffled")]
    out = {"what": "Pearson r of the share's numerator (contrast) on E220's KL over lower's states",
           "r_all_12": statistics.correlation([r["kl"] for r in lo], [r["numerator"] for r in lo]),
           "r_rl_and_shuffled_6": statistics.correlation([r["kl"] for r in rl], [r["numerator"] for r in rl]),
           "r_gain_all_12": statistics.correlation([r["kl"] for r in lo], [r["gain"] for r in lo])}  # fmt: skip
    (RES / "contrast_numerator_vs_kl.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    recs = states()
    gain_figure(recs)
    share_figure(recs)
    numerator_figure(recs)
    print(json.dumps(kl_fit(recs)))


if __name__ == "__main__":
    main()
