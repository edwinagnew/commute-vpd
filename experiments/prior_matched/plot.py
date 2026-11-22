"""P2.57's figure (no GPU): every prior-matched pair of a reused and a new task, one colour per pair. x is the pair's
prior gap (M's log10 P(answer), reused minus new); y is each reading's gap (mean over the reused seeds minus mean over
the new), in units of that reading's smallest gap that matters (E237's band: the larger 3-seed range of copy and the
prior-matched cipher at seeds 0-2). One marker per reading; the bar is the seed-spread
interval (lowest reused minus highest new, to highest reused minus lowest new). A reading that follows the prior sits
inside the grey band (|y| <= 1) at these small x; one that follows the mechanism sits outside it.

Pairs, each drawn when its files exist: E237's copy against the prior-matched cipher and E250's low-prior copy against
the cipher (3 seeds, P2.54's saved readings, `prior_mech_2x2/train.state_row`), run A (10 seeds), run B (copy's 10
against the second cipher's 3), the cued copy against the cipher (experiments/cued_copy/results/table.json). E237's
seeds are run A's seeds 0-2 (max |diff| 0), drawn hollow. The right panel is the reference: copy against the cipher,
not prior-matched (1.6 decades apart, the same table), where a reading that follows the mechanism reads as far out.

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
        out["cued copy vs cipher, 3 seeds (E253)"] = (prior_gap(a, b), a, b)
    return out


def unmatched() -> tuple[str, tuple[float, Rows, Rows]]:
    """The reference pair: E237's copy against E147's cipher, whose priors are not matched."""
    rows = W.read_json(EXPERIMENTS / "cued_copy" / "results" / "table.json")["rows"]
    a, b = ([{k: T2.num(v) for k, v in r.items()} for r in rows[t]] for t in ("copy", "cipher"))
    return "copy vs cipher, priors not matched, 3 seeds", (prior_gap(a, b), a, b)


def bands(ps: dict[str, tuple[float, Rows, Rows]]) -> dict[str, float]:
    _, a, b = ps["copy vs prior-matched cipher, 3 seeds (E237)"]
    return {p: W.band([r[p] for r in a], [r[p] for r in b]) for p in W.READINGS}


def draw(ax: Any, name: str, x: float, a: Rows, b: Rows, bd: dict[str, float], colour: str, hollow: bool) -> list[dict[str, Any]]:
    """One pair: each reading's gap of means over its band at x (spread sideways by reading), the seed-spread bar."""
    data = []
    for i, p in enumerate(W.READINGS):
        va, vb = [r[p] for r in a], [r[p] for r in b]
        if not all(math.isfinite(v) for v in (*va, *vb, bd[p])) or bd[p] <= 0:
            continue
        g, lo, hi = fmean(va) - fmean(vb), min(va) - max(vb), max(va) - min(vb)
        xi = x + (i - 3.5) * 0.006 + (0.003 if hollow else 0)
        ax.errorbar(xi, g / bd[p], yerr=[[(g - lo) / bd[p]], [(hi - g) / bd[p]]], fmt=MARK[p], ms=6, color=colour, ecolor=colour,
                    mfc="white" if hollow else colour, mec=colour if hollow else "white", mew=0.8, elinewidth=1, capsize=0, zorder=3)  # fmt: skip
        data.append({"pair": name, "reading": p, "prior_gap": x, "gap": g, "lo": lo, "hi": hi, "band": bd[p], "gap_in_bands": g / bd[p],
                     "n_reused": len(va), "n_new": len(vb)})  # fmt: skip
    return data


def main() -> None:
    ps, (ref_name, (ref_x, ref_a, ref_b)) = pairs(), unmatched()
    bd = bands(ps)
    fig, (ax, rx) = figure((10.0, 5.0), ncols=2)
    ax.get_gridspec().set_width_ratios([4.5, 1])
    rx.sharey(ax)
    for a in (ax, rx):
        a.axhspan(-1, 1, color="0.88", lw=0, zorder=0, label="within the 3-seed spread\n(the smallest gap that matters)")
        a.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.set_yscale("symlog", linthresh=1)
    data = []
    for j, (name, (x, a, b)) in enumerate(ps.items()):
        data += draw(ax, name, x, a, b, bd, COLOUR[j], hollow=name.endswith("(E237)"))
    data += draw(rx, ref_name, ref_x, ref_a, ref_b, bd, "0.35", hollow=False)
    rx.tick_params(labelleft=False)
    ax.set_xlim(-0.05, 0.25)
    rx.set_xlim(ref_x - 0.035, ref_x + 0.035)
    rx.set_xticks([round(ref_x, 2)])
    ax.set_xlabel("prior gap, reused minus new: decades of the base model's P(answer)")
    rx.set_xlabel("reference pair,\npriors not matched")
    ax.set_ylabel("reading's gap, reused minus new,\nin units of its smallest gap that matters")
    ax.set_title("Does each cheap reading still split a copy task from a cipher when their priors are matched?", fontsize=9.5)
    pair_keys = [ax.plot([], [], "o", color=COLOUR[j], mfc="white" if n.endswith("(E237)") else COLOUR[j], ls="", label=n)[0]
                 for j, n in enumerate(ps)]  # fmt: skip
    pair_keys.append(rx.plot([], [], "o", color="0.35", ls="", label=ref_name)[0])
    band_key = ax.get_legend_handles_labels()[0][0]
    read_keys = [ax.plot([], [], MARK[p], color="0.3", ls="", label=LABEL[p])[0] for p in W.READINGS]
    head = lambda t: ax.plot([], [], ls="", label=t)[0]  # a legend sub-heading: an entry with no marker
    fig.legend(handles=[head("pair (colour)"), *pair_keys, band_key, head(" "), head("reading (marker)"), *read_keys], fontsize=6.5,
               frameon=False, loc="outside right upper")  # fmt: skip
    n = ("Seeds per side: 3, except run A 10 against 10 and run B 10 against 3. Bar: lowest to highest difference over seed pairs. "
         "The do-nothing and the random updates read no value on any reading but effective rank.")
    print(save(fig, OUT, n=n, data=data))
    print(json.dumps({"bands": bd, "pairs": {n: v[0] for n, v in ps.items()}, "reference": ref_x}, indent=1))


if __name__ == "__main__":
    main()
