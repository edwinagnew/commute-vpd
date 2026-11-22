"""P2.57's figure (no GPU): every prior-matched pair of a reused and a new task, one colour per pair. x is the pair's
prior gap (M's log10 P(answer), reused minus new); y is each reading's gap (mean over the reused seeds minus mean over
the new), in units of that reading's smallest gap that matters (E237's band: the larger 3-seed range of copy and the
prior-matched cipher; k = 3 takes run A's, read at its seeds 0-2). One marker per reading; the bar is the seed-spread
interval (lowest reused minus highest new, to highest reused minus lowest new). A reading that follows the prior sits
inside the grey band (|y| <= 1) at these small x; one that follows the mechanism sits outside it.

Pairs, each drawn when its files exist: E237's copy against the prior-matched cipher and E250's low-prior copy against
the cipher (3 seeds, P2.54's saved readings, `prior_mech_2x2/train.state_row`), run A (10 seeds), run B (copy's 10
against the second cipher's 3), the cued copy against the cipher (experiments/cued_copy/results/table.json).

    ~/commute-agents/fleet/cpu-run python3 experiments/prior_matched/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.figures import figure, save
from commute.jobs import EXPERIMENTS, load_experiment

W: Any = load_experiment("prior_matched", "world")  # READINGS, band; P2.54's state_row through W.T2
T2, CC = W.T2, W.CC
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "prior_matched_pairs.png"
KHEAD = EXPERIMENTS / "prior_mech_2x2" / "results" / "khead"
COLOUR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # the reference palette's slots 1-5, fixed order
MARK = dict(zip(W.READINGS, ["o", "s", "D", "^", "v", "<", "P", "X"], strict=True))
LABEL = {"head_share": "copy head (2,4) off", "k3_share": "top 3 copy heads off", "r": "re-gating share r", "erank": "effective rank",
         "steering_pooled": "steering, pooled", "steering_perkey": "steering, per key", "neuron_k64": "neurons, k = 64",
         "neuron_kmatched": "neurons, k matched"}  # fmt: skip
Rows = list[dict[str, float]]


def saved_rows(task: str) -> Rows:
    """P2.54's table rows for an E147 task or the low copy at seeds 0-2, with khead's k = 3 share once run 3 has run."""
    a = W.read_json(T2.OUT / "a.json")
    states = [f"{T2.TASK}_s{s}" if task == "lowcopy" else T2.R96.state_name(task, s) for s in range(3)]
    return [T2.state_row(task, st, a) | CC.khead_of(KHEAD / f"{W.R1.fname(st)}.json") for st in states]


def prior_gap(rows_a: Rows, rows_b: Rows) -> float:
    return rows_a[0]["log10_prior"] - rows_b[0]["log10_prior"]


def pairs() -> dict[str, tuple[float, Rows, Rows]]:
    """name -> (prior gap, reused rows, new rows) for every pair whose files exist."""
    cp, pc, ci = saved_rows("copy"), saved_rows("prior_cipher"), saved_rows("cipher")
    out = {"copy vs prior-matched cipher, 3 seeds (E237)": (prior_gap(cp, pc), cp, pc),
           "low-prior copy vs cipher, 3 seeds (E250)": (prior_gap(saved_rows("lowcopy"), ci), saved_rows("lowcopy"), ci)}  # fmt: skip
    for run, name in (("a", "copy vs prior-matched cipher, 10 seeds (run A)"), ("b", "copy vs second matched cipher (run B)")):
        if (f := W.RESULTS / run / "summary.json").is_file():
            s = W.read_json(f)
            out[name] = (s["prior"]["gap"], [{k: T2.num(v) for k, v in r.items()} for r in s["rows"]["reused"]],
                         [{k: T2.num(v) for k, v in r.items()} for r in s["rows"]["new"]])  # fmt: skip
    if (f := EXPERIMENTS / "cued_copy" / "results" / "table.json").is_file():
        rows = W.read_json(f)["rows"]
        a, b = ([{k: T2.num(v) for k, v in r.items()} for r in rows[t]] for t in ("cuedcopy", "cipher"))
        out["cued copy vs cipher (cued copy run)"] = (prior_gap(a, b), a, b)
    return out


def bands(ps: dict[str, tuple[float, Rows, Rows]]) -> dict[str, float]:
    _, a, b = ps["copy vs prior-matched cipher, 3 seeds (E237)"]
    out = {p: W.band([r[p] for r in a], [r[p] for r in b]) for p in W.READINGS}
    if (f := W.RESULTS / "a" / "summary.json").is_file():
        out["k3_share"] = T2.num(W.read_json(f)["calls"]["k3_share"]["band"])
    return out


def main() -> None:
    ps, data = pairs(), []
    bd = bands(ps)
    fig, ax = figure((8.6, 4.6))
    ax.axhspan(-1, 1, color="0.88", lw=0, zorder=0, label="inside the smallest gap that matters")
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    for j, (name, (x, a, b)) in enumerate(ps.items()):
        for i, p in enumerate(W.READINGS):
            va, vb = [r[p] for r in a], [r[p] for r in b]
            if not all(math.isfinite(v) for v in (*va, *vb, bd[p])) or bd[p] <= 0:
                continue
            g, lo, hi = fmean(va) - fmean(vb), min(va) - max(vb), max(va) - min(vb)
            xi = x + (i - 3.5) * 0.006
            ax.errorbar(xi, g / bd[p], yerr=[[(g - lo) / bd[p]], [(hi - g) / bd[p]]], fmt=MARK[p], ms=6, color=COLOUR[j], ecolor=COLOUR[j],
                        elinewidth=1, capsize=0, mec="white", mew=0.6, zorder=3)  # fmt: skip
            data.append({"pair": name, "reading": p, "prior_gap": x, "gap": g, "lo": lo, "hi": hi, "band": bd[p], "n_reused": len(va), "n_new": len(vb)})
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(-0.05, 0.4)
    ax.set_xlabel("prior gap, reused minus new (decades of M's P(answer))")
    ax.set_ylabel("reading's gap, reused minus new\n(in units of its smallest gap that matters)")
    ax.set_title("Each cheap reading's gap between a reused and a new task with matched priors", fontsize=9)
    pair_keys = [ax.plot([], [], "o", color=COLOUR[j], label=n)[0] for j, n in enumerate(ps)]
    read_keys = [ax.plot([], [], MARK[p], color="0.3", ls="", label=LABEL[p])[0] for p in W.READINGS]
    fig.legend(handles=pair_keys, fontsize=6.5, frameon=False, loc="outside right upper", title="pair (colour)", title_fontsize=7)
    fig.legend(handles=read_keys, fontsize=6.5, frameon=False, loc="outside right lower", title="reading (marker)", title_fontsize=7)
    print(save(fig, OUT, n="seeds per side as listed in the .csv (n_reused, n_new); one marker per reading and pair", data=data))
    print(json.dumps({"bands": bd, "pairs": {n: v[0] for n, v in ps.items()}}, indent=1))


if __name__ == "__main__":
    main()
