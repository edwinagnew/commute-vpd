"""Figures for E53 (performed_anchor), read from results/read.json only; nothing is recomputed from the model.
Plan: reports/plans/2026-10-29_plan_performed-anchor.md; analysis: reports/analyses/2026-10-29_analysis_e53-performed-anchor.md.

  ratio.png   the stand-in ratio (masked gain / subcomponent gain) of the four step-300 fine-tunes with its 90% interval
              over windows, the gate gain over the subcomponent gain beside it, the filed leans 0.7 and 0.5 and the
              toy fine-tunes' readings (E46: copies 0.87-0.97, random maps 0.36-0.41) as bands
  gain.png    the ratio against the full model's gain G for the cipher at steps 10, 100, 300 and lowercase at step 300,
              with each task's gain floor and the cipher's random delta-W, which sits under its floor (unread)
  nulls.png   where the nulls landed: G against the off-target KL for the fine-tunes, the KL-matched random delta-W and
              the G-matched hand writes (both at the alpha cap)
  relay.png   the relay rho against G, same rows as gain.png, with E50's toy readings (copies, random maps, hand writes)

Usage (repo root): ~/commute-agents/fleet/cpu-run <clone>/.venv/bin/python experiments/performed_anchor/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results" / "read.json").read_text())
P = json.loads((HERE / "results" / "predictions.json").read_text())
ROWS, S, TH = R["rows"], R["score"], P["thresholds"]
FIG = HERE / "figures"
LOW_C, CIP_C, NULL_C, WRITE_C, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#2e9e5b", "#0b0b0b", "#8a8984"
COL = {"lowercase": LOW_C, "cipher": CIP_C}
NAME = {"lowercase": "lowercase (performed)", "cipher": "cipher (new)"}
MARK = {0: "o", 1: "s"}
# E46 Amendment 3 / E50 toy references (reports/analyses/2026-10-29_analysis_e46-amendment3.md, ..._e50-restore-anchors.md)
TOY_RATIO = {"copy fine-tunes (E46)": (0.872, 0.968), "random-map fine-tunes (E46)": (0.356, 0.414)}
TOY_RELAY = {"copy fine-tunes (E50)": (0.757, 0.781), "random-map fine-tunes (E50)": (0.664, 0.730),
             "hand writes (E50, low control)": (-0.32, -0.16)}


def ratio(k: str) -> float:
    return ROWS[k]["masked_gain"] / ROWS[k]["piece_gain"]


def finish(ax: plt.Axes, path: str, title: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    ax.figure.savefig(FIG / path, dpi=130)
    plt.close(ax.figure)


def toy_bands(ax: plt.Axes, bands: dict[str, tuple[float, float]], x: float) -> None:
    for name, (lo, hi) in bands.items():
        ax.axhspan(lo, hi, color=MUTED, alpha=0.18, lw=0)
        ax.text(x, (lo + hi) / 2, name, va="center", ha="right", fontsize=8, color=INK)


def fig_ratio() -> None:
    _fig, ax = plt.subplots(figsize=(8.5, 5))
    toy_bands(ax, TOY_RATIO, 4.45)
    for y, lab in ((TH["lean_lowercase_ratio_min"], "lean for lowercase: at least 0.7"),
                   (TH["lean_cipher_ratio_max"], "lean for the cipher: at most 0.5")):
        ax.axhline(y, color=INK, ls="--", lw=0.8)
        ax.text(-0.45, y + 0.012, lab, fontsize=8)
    for i, (task, seed) in enumerate((t, s) for t in ("lowercase", "cipher") for s in (0, 1)):
        k = f"{task}/s{seed}/step300"
        r, lo_hi = ratio(k), ROWS[k]["boot"]["ratio_ci90"]
        ax.errorbar(i, r, yerr=[[r - lo_hi[0]], [lo_hi[1] - r]], fmt=MARK[seed], color=COL[task], ms=9, capsize=4,
                    label="stand-in ratio, 90% interval over windows" if i == 0 else None)
        gU = ROWS[k]["g"] / ROWS[k]["piece_gain"]
        ax.plot(i + 0.18, gU, MARK[seed], mfc="white", mec=COL[task], ms=8,
                label="gate gain over subcomponent gain" if i == 0 else None)
    ax.annotate("", xy=(2.5, S["ratio"]["cipher/s0/step300"]), xytext=(2.5, S["ratio"]["lowercase/s0/step300"]),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.8})
    ax.text(2.55, 0.6, f"worst-pair gap {S['P1']['gap']:.2f}\n(bar {TH['gap']})", fontsize=8)
    ax.set_xticks(range(4), ["lowercase\nseed 0", "lowercase\nseed 1", "cipher\nseed 0", "cipher\nseed 1"])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("stand-in ratio: masked gain / subcomponent gain")
    ax.text(-0.45, 0.03, "zero update and do-nothing: undefined (subcomponent gain exactly 0)", fontsize=8, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), fontsize=8, frameon=False)
    finish(ax, "ratio.png", "Does the stand-in ratio read the performed task above the new one? (step 300, two seeds)")


def by_gain(ax: plt.Axes, value: str) -> None:
    """Cipher steps 10/100/300 as a line per seed, lowercase step 300 as points; x = full-model gain G."""
    for seed in (0, 1):
        ks = [f"cipher/s{seed}/step{n}" for n in (10, 100, 300)]
        xs = [ROWS[k]["G"] for k in ks]
        ys = [ratio(k) if value == "ratio" else ROWS[k]["relay"]["rho"] for k in ks]
        ax.plot(xs, ys, "-" + MARK[seed], color=CIP_C, ms=7, label=f"cipher, seed {seed}, steps 10, 100, 300")
        for x, y, n in zip(xs, ys, (10, 100, 300), strict=True):
            ax.annotate(f"step {n}", (x, y), textcoords="offset points", xytext=(4, -11 if seed else 6), fontsize=7)
        k = f"lowercase/s{seed}/step300"
        y = ratio(k) if value == "ratio" else ROWS[k]["relay"]["rho"]
        ax.plot(ROWS[k]["G"], y, MARK[seed], color=LOW_C, ms=8, label=f"lowercase, seed {seed}, step 300")


def fig_gain() -> None:
    _fig, ax = plt.subplots(figsize=(8.5, 5))
    by_gain(ax, "ratio")
    rnd = ROWS["cipher/random"]
    lo, hi = rnd["boot"]["ratio_ci90"]
    r = ratio("cipher/random")
    ax.errorbar(rnd["G"], r, yerr=[[r - lo], [hi - r]], fmt="D", mfc="white", mec=NULL_C, ecolor=NULL_C, capsize=3,
                label="cipher random δW, under its gain floor: unread")
    for task, yy in (("lowercase", 0.06), ("cipher", 0.12)):
        f = S["bar"][f"{task}/s0/step300"]
        ax.axvline(f, color=COL[task], ls=":", lw=1)
        ax.text(f + 0.04, yy, f"{task} gain floor {f:.2f}", fontsize=8, color=COL[task])
    for y in (TH["lean_lowercase_ratio_min"], TH["lean_cipher_ratio_max"]):
        ax.axhline(y, color=INK, ls="--", lw=0.8)
    ax.text(4.1, 0.71, "lean ≥ 0.7", fontsize=8, ha="right")
    ax.text(4.1, 0.51, "lean ≤ 0.5", fontsize=8, ha="right")
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("full-model gain G at the scored positions, nats of log-odds (0 = the original model)")
    ax.set_ylabel("stand-in ratio")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    finish(ax, "gain.png", "Is the cipher's low ratio only a matter of gain size?")


def fig_nulls() -> None:
    _fig, ax = plt.subplots(figsize=(8.5, 5))
    for task in ("lowercase", "cipher"):
        for seed in (0, 1):
            k = f"{task}/s{seed}/step300"
            ax.plot(ROWS[k]["kl_other"], ROWS[k]["G"], MARK[seed], color=COL[task], ms=8,
                    label=f"{NAME[task]} fine-tune, seed {seed}")
        rk, wk = ROWS[f"{task}/random"], ROWS[f"{task}/write"]
        ax.plot(rk["kl_other"], rk["G"], "D", mfc="white", mec=COL[task], ms=8, label=f"{task}: random δW, KL-matched")
        ax.plot(wk["kl_other"], wk["G"], "X", color=COL[task], mec=WRITE_C, ms=10,
                label=f"{task}: hand write at the α cap of {TH['alpha_max']}")
        ax.axhline(wk["G_target"], color=COL[task], ls=":", lw=1)
        ax.text(60, wk["G_target"] + 0.4, f"hand write's target G, {task}", fontsize=8, ha="right", color=COL[task])
    ax.axhline(0, color=INK, lw=0.8)
    ax.text(60, 0.4, "original model / zero update", fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 70)
    ax.set_xlabel("off-target KL from the original model, nats (mean over the 126 other window positions)")
    ax.set_ylabel("full-model gain G, nats of log-odds")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    finish(ax, "nulls.png", "Did the nulls land where the plan needed them?")


def fig_relay() -> None:
    _fig, ax = plt.subplots(figsize=(8.5, 5))
    toy_bands(ax, TOY_RELAY, 4.15)
    by_gain(ax, "relay")
    ax.axhline(0, color=INK, lw=0.8)
    ax.text(0.05, 0.02, "0: the update's own write carries all the gain", fontsize=8)
    ax.set_xlim(0, 4.2)
    ax.set_ylim(-0.4, 1.0)
    ax.set_xlabel("full-model gain G at the scored positions, nats of log-odds")
    ax.set_ylabel("relay: share of G taken back when every alive\nsubcomponent reads its original input")
    ax.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.33))
    finish(ax, "relay.png", "Does the relay read the performed task above the new one?")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_ratio()
    fig_gain()
    fig_nulls()
    fig_relay()


if __name__ == "__main__":
    main()
