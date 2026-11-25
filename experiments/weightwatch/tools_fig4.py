"""P2.80 page figure: every model-diffing tool read on Fig 4's 48 states (16 tasks x 3 seeds), beside the three cheap
readings Fig 4 plots. One panel per reading; x = the 16 tasks, built to reuse | built new; three seed dots per task;
each reading's own null drawn on its panel. Descriptive: no bar, no verdict.

Reads (all committed results; nothing is recomputed):
  Fig 4 states, display names, cheap readings, neurons  experiments/known_3seeds/results/known3.json (cells)
  re-gating share null (random rotation of each matrix's SVD)  share_untrained stage1 .states.<s>.rotated, or
      known_3seeds stage2 / refit_96key stage1 .states.<s>.e136.rotated
  steering null (mean of 8 random directions)  holes_fill stage1 .states.<s>.steering.pooled.random_mean, or
      known_3seeds stage2 / refit_96key stage1 .states.<s>.holes_fill.steering.pooled.random_mean, or (e62)
      steering_copy_lookup stage1 .states.<s>.steering.pooled.random.mean_lo
  CMAP heads (E288 items/, E305 p2-80/items/)  .compared.whole_logodds_share; null = mean of the 10 random head sets
      of the circuit's size, .cmap.random<k>.share.logodds; circuit size = len(.circuit)
  WeightWatch S (E306)  results/p2-80/summary.json .rows.<s>.S, .haar_range, .sham_S
  Crosscoder d (E203, seed 0 only)  experiments/crosscoder/results/run2/sides/<g>__<t>_s0.json .d; random update
      run2/stage1_random.json .random.d
Writes experiments/weightwatch/figures/tools_fig4.{png,pdf,csv} and results/p2-80/tools_fig4.json.

    ~/commute-agents/fleet/cpu-run python3 experiments/weightwatch/tools_fig4.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from commute import figures as FG

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments"
OUT_FIG = EXP / "weightwatch/figures/tools_fig4.png"
OUT_JSON = EXP / "weightwatch/results/p2-80/tools_fig4.json"
# Task names per pair side (reuse, new), as make_known_scorecard.py names them (paper Table tab:pairs).
TASKS = {"e62_copy_lookup": ("Copy at a new marker", "Lookup"), "e46_copy_random": ("Copy key", "Random map"),
         "e53_lowercase_cipher": ("Lowercase text", "Ciphered text"),
         "e54_matched": ("Runner-up guess", "Cipher (early step)"),
         "e54_own": ("Runner-up guess", "Cipher (late step)"), "open_pair_c3": ("Top guess", "Cipher (24 keys)"),
         "open_pair_c3_copy": ("Copy key (24 keys)", "Cipher (24 keys)"),
         "e58_rank1_96": ("Top guess (96 keys)", "Cipher (96 keys)"),
         "e58_copy_prior_cipher": ("Copy key (96 keys)", "Prior-matched cipher")}
SIDE_COLOUR = {"reuse": "#0072B2", "new": "#D55E00"}
NULL_GREY = "#999999"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def fig4_states() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, float]]]:
    """(side, task) -> {pair, states}; reading -> state -> value, for the four known3 readings used here."""
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    vals: dict[str, dict[str, float]] = {k: {} for k in ("share", "steer", "erank", "neuron")}
    for c in load(EXP / "known_3seeds/results/known3.json")["cells"]:
        if c["method"] not in vals:
            continue
        for i, side in enumerate(("reuse", "new")):
            key = (side, TASKS[c["pair"]][i])
            tasks.setdefault(key, {"pair": c["pair"], "states": c["states"][side]})
            assert tasks[key]["states"] == c["states"][side], key
            for s, v in zip(c["states"][side], c[side]):
                assert s not in vals[c["method"]] or vals[c["method"]][s] == v, (c["method"], s)
                vals[c["method"]][s] = v
    assert len(tasks) == 16 and all(len(v) == 48 for v in vals.values())
    return tasks, vals


def first(*options: Any) -> float:
    return next(float(o) for o in options if o is not None)


def cheap_nulls(states: list[str]) -> dict[str, dict[str, float]]:
    su = load(EXP / "share_untrained/results/stage1.json")["states"]
    k2 = load(EXP / "known_3seeds/results/stage2.json")["states"]
    rf = load(EXP / "refit_96key/results/stage1.json")["states"]
    hf = load(EXP / "holes_fill/results/stage1.json")["states"]
    sc = load(EXP / "steering_copy_lookup/results/stage1.json")["states"]
    share, steer = {}, {}
    for s in states:
        share[s] = first(su.get(s, {}).get("rotated"), k2.get(s, {}).get("e136", {}).get("rotated"),
                         rf.get(s, {}).get("e136", {}).get("rotated"))
        short = s.split("/")[1]
        steer[s] = first(hf.get(s, {}).get("steering", {}).get("pooled", {}).get("random_mean"),
                         k2.get(s, {}).get("holes_fill", {}).get("steering", {}).get("pooled", {}).get("random_mean"),
                         rf.get(s, {}).get("holes_fill", {}).get("steering", {}).get("pooled", {}).get("random_mean"),
                         sc.get(short, {}).get("steering", {}).get("pooled", {}).get("random", {}).get("mean_lo")
                         if s.startswith("e62/") else None)
    return {"share": share, "steer": steer}


def cmap(tasks: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per state: the head share, its random-head-set null and the circuit size, from the item whose role is the
    first Fig 4 pair naming that task (c3 cipher: open_pair_c3, as built; e58 cipher: e58_rank1_96's scale)."""
    items = [load(p) for d in ("items", "p2-80/items") for p in sorted((EXP / "cmap/results" / d).glob("*_x*.json"))]
    out: dict[str, dict[str, float]] = {"value": {}, "null": {}, "heads": {}, "c": {}, "empty": {}}
    for (side, _), t in tasks.items():
        role = f"{t['pair']}:{side}"
        for s in t["states"]:
            (it,) = [i for i in items if i["state"] == s and role in i["roles"]]
            rnd = [v["share"]["logodds"] for k, v in it["cmap"].items() if k.startswith("random") and v]
            out["heads"][s] = len(it["circuit"])
            if it["circuit"]:  # an empty circuit has no CMAP reading: drawn hollow at its 0, left out of every mean
                out["value"][s] = it["compared"]["whole_logodds_share"]
            else:
                out["empty"][s] = it["compared"]["whole_logodds_share"]
            out["c"][s] = it["c"]
            if rnd:
                out["null"][s] = statistics.fmean(rnd)
    return out


def weightwatch(states: list[str]) -> dict[str, dict[str, Any]]:
    rows = load(EXP / "weightwatch/results/p2-80/summary.json")["rows"]
    return {"value": {s: rows[s]["S"] for s in states}, "sham": {s: rows[s]["sham_S"] for s in states},
            "haar": {s: rows[s]["haar_range"] for s in states}}


def crosscoder(states: list[str]) -> dict[str, dict[str, float]]:
    sides = EXP / "crosscoder/results/run2/sides"
    out = {}
    for s in states:
        p = sides / (s.replace("/", "__") + ".json")
        if s.endswith("_s0"):
            side = load(p)
            assert side["c"] == 1 and side["state"] == s
            out[s] = side["d"]
    return {"value": out}


def span(xs: list[float]) -> list[float]:
    return [min(xs), max(xs)]


def main() -> None:
    tasks, vals = fig4_states()
    order = [(side, t) for side in ("reuse", "new") for t in sorted(t for s2, t in tasks if s2 == side)]
    states = [s for k in order for s in tasks[k]["states"]]
    nulls = cheap_nulls(states)
    cm, ww, cc = cmap(tasks), weightwatch(states), crosscoder(states)
    cc_random = load(EXP / "crosscoder/results/run2/stage1_random.json")["random"]["d"]
    ww_gauss = load(EXP / "weightwatch/results/p2-80/stage0.json")["gaussian"]["mean"]
    # (key, title, y label, values, per-state null for a band, line null (value, label))
    panels = [
        ("share", "Re-gating share (Fig 4)", "share", vals["share"], nulls["share"], None),
        ("steer", "Steering share (Fig 4)", "share", vals["steer"], nulls["steer"], None),
        ("erank", "Effective rank (Fig 4); no null", "effective rank", vals["erank"], None, None),
        ("cmap", "CMAP at heads; new sides at matched size", "log-odds share", cm["value"], cm["null"], None),
        ("neuron", "CMAP at neurons, our adaptation; no null", "share", vals["neuron"], None, None),
        ("ww", "WeightWatch: our score S, sign-blind", "S (1 = isotropic)", ww["value"], None, (ww_gauss, "Gaussian update, 20 draws")),
        ("cc", "Crosscoder, seed 0; unvalidated, control failed", "d = shared - specific", cc["value"], None,
         (cc_random, "one random update")),
    ]
    band_label = {"share": "random rotation of the SVD", "steer": "random directions", "cmap": "random head sets"}
    fig, axes = FG.figure((7.5, 10.5), ncols=2, nrows=4)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"order": [t for _, t in order], "states": {f"{s}|{t}": tasks[(s, t)]["states"]
                                                                       for s, t in order}}
    for ax, (key, title, ylab, v, band, line) in zip(axes.flat, panels):
        for side in ("reuse", "new"):
            groups = {t: ([v[s] for s in tasks[(sd, t)]["states"] if s in v] if sd == side else [])
                      for sd, t in order}
            FG.column_dots(ax, groups, method=f"built_{side}", ylabel=ylab)
        if key == "cmap":
            for i, k in enumerate(order):
                e = [cm["empty"][s] for s in tasks[k]["states"] if s in cm["empty"]]
                ax.scatter([i] * len(e), e, facecolors="none", edgecolors=SIDE_COLOUR[k[0]], s=22, linewidths=0.9,
                           label="no circuit found (left out)" if e and not any(
                               x.get_label() == "no circuit found (left out)" for x in ax.collections) else None)
        if band is not None:
            labelled = False
            for i, k in enumerate(order):
                b = [band[s] for s in tasks[k]["states"] if s in band]
                if b:
                    ax.fill_between([i - 0.3, i + 0.3], min(b), max(b), color=NULL_GREY, edgecolor=NULL_GREY,
                                    alpha=0.45, linewidth=1.0, label=None if labelled else f"null: {band_label[key]}")
                    labelled = True
        if key == "ww":
            for i, k in enumerate(order):
                h = [ww["haar"][s] for s in tasks[k]["states"]]
                ax.fill_between([i - 0.3, i + 0.3], min(r[0] for r in h), max(r[1] for r in h), color=NULL_GREY,
                                alpha=0.35, linewidth=0, label="null: same-spectrum rotation" if i == 0 else None)
                ax.scatter([i] * 3, [ww["sham"][s] for s in tasks[k]["states"]], marker="x", color="#555555", s=12,
                           linewidths=0.8, label="null: orientation sham" if i == 0 else None)
        if line is not None:
            ax.axhline(line[0], color="#555555", linestyle="--", linewidth=0.8, label=f"null: {line[1]}")
        ax.axvline(7.5, color="#BBBBBB", linewidth=0.8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.label.set_size(7.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, loc="best", frameon=False)
        rec: dict[str, Any] = {"per_task": {}}
        for side, t in order:
            xs = [v[s] for s in tasks[(side, t)]["states"] if s in v]
            if xs:
                rec["per_task"][t] = {"side": side, "values": xs, "mean": statistics.fmean(xs)}
            for s in tasks[(side, t)]["states"]:
                records.append({"reading": key, "side": side, "task": t, "state": s, "value": v.get(s),
                                "null": None if band is None else band.get(s)})
        for side in ("reuse", "new"):
            means = [r["mean"] for r in rec["per_task"].values() if r["side"] == side]
            per_state = [x for r in rec["per_task"].values() if r["side"] == side for x in r["values"]]
            rec[side] = {"task_mean_range": span(means), "state_range": span(per_state), "n_states": len(per_state)}
        if band is not None:
            rec["null_state_range"] = span([band[s] for s in states if s in band])
            rec["null_n_states"] = sum(s in band for s in states)
        if line is not None:
            rec["null_line"] = line[0]
        summary[key] = rec
    summary["cmap"]["heads"] = cm["heads"]
    summary["cmap"]["c"] = cm["c"]
    summary["cmap"]["empty"] = sorted(cm["empty"])
    summary["cmap"]["heads_task_mean"] = {t: statistics.fmean(h) for sd, t in order
                                          if (h := [cm["heads"][s] for s in tasks[(sd, t)]["states"] if cm["heads"][s]])}
    summary["ww"]["sham_state_range"] = span(list(ww["sham"].values()))
    summary["ww"]["haar_range"] = [min(r[0] for r in ww["haar"].values()), max(r[1] for r in ww["haar"].values())]
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0, 1, "Not read on these states:\n\nActivation difference lens: its 67M\ncontrol fails, so no reading.\n\n"
            "Diff-SAE, Delta-Crosscoder: no\npassing positive control at the\nread site; not run.\n\n"
            "CMAP at heads reads each new side\nat its pair's matched size; hollow:\nseeds with no circuit, left out.\n"
            "The crosscoder has seed 0 only.", va="top", fontsize=7.5, transform=ax.transAxes)
    fig.suptitle("What each tool reads on the 48 states of Fig 4, task by task", fontsize=10)
    FG.save(fig, OUT_FIG, n="16 tasks x 3 seeds (crosscoder: seed 0, 16 states). Blue: built to reuse; orange: built new. "
            "Grey: each reading's own null.", data=records)
    OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {OUT_FIG} and {OUT_JSON}")


if __name__ == "__main__":
    main()
