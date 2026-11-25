"""E159's figure and reading (cell P1.15), from the results JSONs only (no model): per reading kind (one site, the best of
five sites; two sites, the best of ten pairs in order), the rescaled rewiring share down the ladder (last number odd, sum
of two, sum of three) beside the linear and MLP probes of the plain 67M for each rung's target. The first two rungs are
E157's rescaled readings (sum_parity_beside/results/readings.json: the saved control, and E157's sum-of-two arm), the third
this run's, rescaled between this run's planted ceiling and floor on the three-number windows (E157's `rescale`). Each
rung's probes are read on the windows its fine-tune was read on. Writes results/readings.json by the filed rule
(predictions.json, "outcome_rule"). Graded reading only: no label, no cut-off.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/difficulty_ladder/plot.py     (after the job)
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from commute import figures as FG
from commute.jobs import load_experiment

P: Any = load_experiment("sum_parity_beside", "plot")  # E157: best, rescale, KINDS, EPS
HERE = Path(__file__).resolve().parent
E157 = HERE.parent / "sum_parity_beside" / "results" / "readings.json"
SEEDS = (0, 1)
RUNGS = {"odd": ("last number odd", "pairs"), "sum2": ("sum of two", "pairs"), "sum3": ("sum of three", "triples")}
PROBES = {"linear": ("linear probe", "#777777", "s", "--"), "mlp": ("MLP probe", "#D55E00", "^", ":")}
NOTE = "1 at the ceiling, 0 at the floor; probes: (accuracy - 0.5) / 0.5"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def mean(vals: list[float | None]) -> float | None:
    v = [h for h in vals if h is not None]
    return sum(v) / len(v) if v else None


def steps(vals: list[float], bar: float) -> list[str]:
    """Each step down the ladder: 'fall' if the value drops by at least `bar`, 'rise' if it grows by at least `bar`, else 'level'."""
    return ["fall" if a - b >= bar - P.EPS else "rise" if b - a >= bar - P.EPS else "level" for a, b in itertools.pairwise(vals)]


def outcome(share: list[float], lin: list[float], mlp: list[float], step_bar: float, probe_bar: float) -> dict[str, Any]:
    """The filed reading of one kind, from the rescaled share (seed mean) and each probe's best held-out accuracy per rung."""
    S, L, N = steps(share, step_bar), steps(lin, probe_bar), steps(mlp, probe_bar)
    shape = "out of order" if "rise" in S else "falls" if all(s == "fall" for s in S) else "flat" if all(s == "level" for s in S) else "partial"
    track = ("tracks both probes (they agree here, so the ladder does not tell them apart)" if S == L == N else
             "tracks the MLP probe, not the linear one" if S == N else "mirrors the linear probe only" if S == L else "tracks neither probe")
    rival = "a" if shape == "falls" and S == N else "b" if shape in ("flat", "out of order") or (S == L and S != N) else "neither"
    return {"share_steps": S, "linear_steps": L, "mlp_steps": N, "shape": shape, "tracking": track, "rival": rival}


def probe_bar(probes: dict[str, Any], th: dict[str, Any]) -> float:
    """max(probe_bar_min, 2 x the null's sd), the sd the root mean square distance from 0.5 of every shuffled-label probe's
    held-out accuracy (both layouts, every target, site and probe)."""
    nulls = [(r[k]["shuffled_held_out"] - 0.5) ** 2 for lay in ("pairs", "triples") for t in probes.get(lay, {}).values()
             for r in t["sites"].values() for k in ("linear", "mlp")]
    return max(th["probe_bar_min"], 2 * (sum(nulls) / len(nulls)) ** 0.5) if nulls else th["probe_bar_min"]


def status(res: Path) -> tuple[str | None, list[dict[str, Any]]]:
    """A stop's message, or None; and the sum-of-three records per seed."""
    for name in ("repeat", "probes", "target3", "ceiling", "floor"):
        if load(res / f"{name}.json").get("stopped"):
            return f"stopped at {name}: {load(res / f'{name}.json')['stopped']}", []
    return None, [load(res / f"sum3_s{s}.json") for s in SEEDS]


def settled(outs: list[dict[str, Any]], done: str) -> bool:
    s0_unlearned = bool(outs[0].get("no_reading")) and "rung" not in outs[0]
    return all(bool(o.get("no_reading")) or bool(o.get(done)) or (s == 1 and not o and s0_unlearned) for s, o in zip(SEEDS, outs, strict=True))


def readings(res: Path, th: dict[str, Any], e157: dict[str, Any]) -> dict[str, Any]:
    stop, outs = status(res)
    probes, ceil, floor, rep = load(res / "probes.json"), load(res / "ceiling.json"), load(res / "floor.json"), load(res / "repeat.json")
    probe_max = {r: probes.get(lay, {}).get(r, {}).get("max") for r, (_, lay) in RUNGS.items()}
    bar = probe_bar(probes, th)
    kinds: dict[str, Any] = {}
    for kind, done in (("one_site", "h_done"), ("two_sites", "b_done")):
        hc, hf = P.best(ceil, kind), P.best(floor, kind)
        raw3 = [None if o.get("no_reading") or not o.get(done) else P.best(o, kind) for o in outs]
        old = e157["kinds"][kind]["rescaled"]
        rescaled = {"odd": old["control"], "sum2": old["new"], "sum3": [P.rescale(h, hc, hf, th) for h in raw3]}
        share = [mean(rescaled[r]) for r in RUNGS]
        k: dict[str, Any] = {"ceiling": hc, "floor": hf, "span": None if hc is None or hf is None else hc - hf, "raw_sum3": raw3,
                             "rescaled": rescaled, "share": share}
        complete = bool(rep.get("passed")) and all(probe_max.values()) and bool(ceil.get(done) and floor.get(done)) and settled(outs, done)
        if stop:
            k["reading"] = stop
        elif not complete:
            k["reading"] = "incomplete: resubmit the job"
        elif all(h is None for h in raw3):
            k["reading"] = "no reading: no seed of the sum-of-three fine-tune was learned and read"
        elif share[2] is None:
            k["reading"] = "not rescaled: the planted ceiling minus floor on the three-number windows is under scale_min"
        else:
            pm = [probe_max[r] for r in RUNGS]
            k["reading"] = outcome([s for s in share if s is not None], [p["linear"] for p in pm if p], [p["mlp"] for p in pm if p], th["step_bar"], bar)
        kinds[kind] = k
    return {"probe_bar": bar, "probe_max": probe_max, "repeat": {"H": rep.get("H"), "H_e157": rep.get("H_e157"), "passed": rep.get("passed")},
            "ceiling_in2": ceil.get("H", {}).get("in2", {}).get("H"), "sum3_no_reading": [o.get("no_reading") for o in outs], "kinds": kinds}


def title(r: Any) -> str:
    if isinstance(r, str):
        return r
    return f"share {r['shape']}; {r['tracking']}"


def main(res: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    th = json.loads((res / "predictions.json").read_text())["thresholds"]
    read = readings(res, th, load(E157))
    (res / "readings.json").write_text(json.dumps(read, indent=1))
    print(json.dumps(read, indent=1))
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    style = FG.METHODS["regating"]
    data: list[dict[str, Any]] = []
    for ax, (kind, (_, _, name)) in zip(axes, P.KINDS.items(), strict=True):
        k = read["kinds"][kind]
        for i, r in enumerate(RUNGS):
            pts = [h for h in k["rescaled"][r] if h is not None]
            ax.scatter([i] * len(pts), pts, color=style.color, marker=style.marker, s=22, alpha=0.85, linewidths=0)
            data += [{"kind": kind, "rung": r, "series": "share", "seed": s, "value": h} for s, h in enumerate(k["rescaled"][r])]
        xs = [i for i, v in enumerate(k["share"]) if v is not None]
        ax.plot(xs, [k["share"][i] for i in xs], color=style.color, linewidth=1, label="rewiring share (dots: seeds)")
        for p, (label, color, marker, ls) in PROBES.items():
            vals = [(i, (read["probe_max"][r][p] - 0.5) / 0.5) for i, r in enumerate(RUNGS) if read["probe_max"][r]]
            ax.plot([v[0] for v in vals], [v[1] for v in vals], color=color, marker=marker, linestyle=ls, linewidth=1, label=label)
            data += [{"kind": kind, "rung": list(RUNGS)[i], "series": p, "seed": "", "value": v} for i, v in vals]
        ax.set_xticks(range(len(RUNGS)), [v[0] for v in RUNGS.values()])
        ax.set_ylim(-0.25, 1.1)
        ax.set_xlabel("rung of the ladder (target bit)")
        ax.set_title(f"{name}\n{title(k['reading'])}", fontsize=7, loc="left", wrap=True)
    axes[0].set_ylabel(f"rescaled reading\n({NOTE})", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Does the rewiring share fall down the difficulty ladder where the MLP probe falls?", fontsize=9)
    FG.save(fig, out / "difficulty_ladder.png", n="2 seeds per fine-tune (first two rungs: E154's and E157's saved readings, "
            "rescaled between E157's planted ceiling and floor; third rung between this run's, on three-number windows); probes "
            "of the plain 67M, best of four sites, last position, on held-out number pairs", data=data)


if __name__ == "__main__":
    main()
