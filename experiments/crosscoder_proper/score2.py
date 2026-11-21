"""P2.49 run 2's CPU jobs (no GPU, no model): the stage-0 check, the ladder (which rungs passed, the first rung X that
passed on every seed, and the keys to run next) and stage 2's reading at X. Rules: results/run2/predictions.json
"rules"; items: results/<tag>/items/<key>.json from run2.py; the native rung's items are run 1's.

  check0    the base against itself through the scaled path (self.A.c0)    -> results/<tag>/stage0.json (exit 1 on a miss)
  ladder    per rung: every seed's specific part and passed/failed/unbuilt/open; X; the head's spliced-loss drop where
            stage 2 runs (HOLD when it misses the bar); next keys -> results/<tag>/ladder.json
  read      at X: the random draws (G1, G2), each task against the random draws, the two built pairs, the stated range
            -> results/<tag>/summary.json; --x I reads rung I (the rehearsal's only use)

    ~/commute-agents/fleet/cpu-run python3 experiments/crosscoder_proper/score2.py ladder
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

R2: Any = load_experiment("crosscoder_proper", "run2")
S1: Any = load_experiment("crosscoder_proper", "score")  # run 1's item reader, pair rule
HERE = Path(__file__).resolve().parent
P2 = json.loads(R2.PRED.read_text())
CFG = R2.settings(P2)
TH1 = json.loads(R2.RUN.PRED.read_text())["thresholds"]


def reading(res: Path, key: str) -> float | str | None:
    """An item's specific part; 'unbuilt: <check>' when it stopped at any check (the size match, FVE, any early
    check: predictions.json rule L1), None when it has not run."""
    p = res / "items" / f"{key}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    if d.get("done"):
        return float(d["read"]["specific"])
    return f"unbuilt: {d['stopped']}" if d.get("stopped") else None


def head_key(rung: int, seed: int) -> str:
    return f"head.{CFG.site}.c{seed}" if rung == CFG.native else f"head-r{rung}.{CFG.site}.c{seed}"


def rung_row(res: Path, native: Path, i: int, bar: float) -> dict[str, Any]:
    vals = {c: reading(native if i == CFG.native else res, head_key(i, c)) for c in CFG.seeds}
    v = list(vals.values())
    if any(isinstance(x, str) for x in v):
        status = "unbuilt"  # a seed stopped (rule L1): the rung is skipped and reported
    elif any(isinstance(x, float) and x < bar for x in v):
        status = "failed"  # one seed under the bar fails the rung (every-seed rule)
    elif all(isinstance(x, float) for x in v):
        status = "passed"
    else:
        status = "open"
    return {"size": CFG.rungs[i], "times_ref": CFG.rungs[i] / CFG.ref_energy, "seeds": vals, "status": status}


def spliced_drop(d: dict[str, Any], self_: dict[str, Any] | None) -> list[float]:
    """Per model, this run's self crosscoder's recovered fraction minus the item's; nan without self or when
    crosscoder.spliced returned nan (ablated <= clean: a broken model), which the item JSON holds as "nan" (run.py
    jsonable): float() reads both."""
    rec = lambda x, m: float(x["spliced"][f"recovered_{m}"])
    return [rec(self_, m) - rec(d, m) if self_ else math.nan for m in (0, 1)]


def misses(drops: list[float], th: dict[str, Any]) -> bool:
    return not all(x <= th["spliced_recovered_drop"] for x in drops)  # a nan misses: a broken model is unread


def head_spliced(res: Path, native: Path, rung: int, th: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """The head's spliced-loss drop at a rung, per finished seed, against the bar stage-2 items are read under."""
    self_, out = S1.item(res, f"self.{CFG.site}.c0"), {}
    for c in CFG.seeds:
        if d := S1.item(native if rung == CFG.native else res, head_key(rung, c)):
            out[c] = {"drop": (dr := spliced_drop(d, self_)), "misses": misses(dr, th)}
    return out


def next_ladder(rows: dict[int, dict[str, Any]], live: list[int], cand: int | None) -> list[str]:
    """After round 1 and before X: seed 0 of every live rung under the candidate (they decide whether it is X) and the
    candidate's missing seeds; with no candidate, seed 0 of the lowest live rung."""
    s0 = CFG.seeds[0]
    if not live:
        return []
    if cand is None:
        return [head_key(live[0], s0)]
    return [head_key(i, s0) for i in live if i < cand] + [head_key(cand, c) for c, v in rows[cand]["seeds"].items() if v is None]


def ladder(res: Path, native: Path, bar: float, th: dict[str, Any]) -> dict[str, Any]:
    """X = the lowest rung that passed on every seed with every rung under it failed or unbuilt. The candidate = the
    lowest live (not failed or unbuilt) rung with a finished seed. Stage 2 runs at X, or at the candidate on the pods
    left (of 4), unless the head misses the spliced-loss bar there (HOLD: tell the lead)."""
    rows = {i: rung_row(res, native, i, bar) for i in range(1, len(CFG.rungs))}
    live = [i for i, r in rows.items() if r["status"] not in ("failed", "unbuilt")]
    cand = next((i for i in live if any(isinstance(v, float) for v in rows[i]["seeds"].values())), None)
    x = live[0] if live and rows[live[0]]["status"] == "passed" else None
    first = [head_key(i, c) for i, c in CFG.round1 if rows[i]["seeds"][c] is None]  # round 1, all at once
    nxt = first or ([] if x is not None else next_ladder(rows, live, cand))
    at = x if x is not None else cand
    head = head_spliced(res, native, at, th) if at is not None else {}
    hold = any(h["misses"] for h in head.values())
    todo = [k for k in CFG.stage2(at) if reading(res, k) is None] if at is not None else []  # a stopped item is not relisted
    s2 = [] if hold or first else todo if x is not None else todo[: max(0, 4 - len(nxt))]
    status = ("none: no rung passed; stage 2 does not run" if not live else f"X = rung {x}" if x is not None
              else f"candidate rung {cand}" if cand is not None else f"searching from rung {live[0]}")  # fmt: skip
    if hold:
        status = f"stage 2 HOLD: head misses spliced bar at {'X' if x is not None else 'candidate'} (rung {at}): tell the lead"
    return {"rungs": rows, "bar": bar, "X": x, "candidate": cand, "head_spliced_at": at, "head_spliced": head, "status": status,
            "unbuilt": [i for i, r in rows.items() if r["status"] == "unbuilt"], "next_ladder": nxt, "next_stage2": s2}  # fmt: skip


def side_values(res: Path, keys: list[str], unread: set[str]) -> list[float]:
    return [math.nan if (v := reading(res, k)) is None or isinstance(v, str) or k in unread else v for k in keys]


def spliced_unread(res: Path, keys: list[str], th: dict[str, Any]) -> set[str]:
    """Items whose spliced-loss recovered fraction falls more than run 1's drop under this run's self crosscoder, or
    reads nan (a broken model)."""
    self_ = S1.item(res, f"self.{CFG.site}.c0")
    return {k for k in keys if (d := S1.item(res, k)) and misses(spliced_drop(d, self_), th)}


def against(task: list[float], rnd: list[float]) -> str:
    """Rule 'task against random'; a tie when every draw reads under run 1's random floor is 'degenerate' (rule G3)."""
    if any(map(math.isnan, task + rnd)):
        return "not read"
    if min(task) > max(rnd):
        return "above random"
    if max(task) < min(rnd):
        return "under random"
    return "degenerate" if max(rnd) < CFG.random_floor else "ties random"


def read(res: Path, x: int, th: dict[str, Any]) -> dict[str, Any]:
    keys = CFG.stage2(x)
    unread = spliced_unread(res, keys, th)
    rnd = side_values(res, keys[: CFG.draws], unread)
    key = lambda t, s: f"e58__{t}_s{s}-r{x}.{CFG.site}.c0"
    tasks = {t: side_values(res, [key(t, s) for s in CFG.seeds], unread) for t in CFG.tasks}
    g1 = not any(map(math.isnan, rnd)) and max(rnd) < CFG.planted_part
    reuse = [v for r, _ in CFG.pairs for v in tasks[r]]
    g2 = not any(map(math.isnan, rnd + reuse)) and min(rnd) > max(reuse)
    pairs = {f"{r} vs {n}": S1.call(tasks[r], tasks[n]) if g1 and g2 else f"not applicable at rung {x}" for r, n in CFG.pairs}
    counts = {k: (S1.item(res, k) or {}).get("read", {}).get("counts", {}).get("ft_specific") for k in keys}
    own = json.loads((res / "sizes.json").read_text())["own_at_scale_1"] if (res / "sizes.json").is_file() else {}
    calls = {t: against(v, rnd) if g1 else "not read (G1 failed: a draw missing or reading new)" for t, v in tasks.items()}
    return {"X": x, "size": CFG.rungs[x], "times_ref": CFG.rungs[x] / CFG.ref_energy, "G1_random_not_new": g1, "G2_random_above_reuse": g2,
            "random": rnd, "tasks": {t: {"specific": v, "against_random": calls[t]} for t, v in tasks.items()},
            "ties_random": sum(c == "ties random" for c in calls.values()), "random_floor": CFG.random_floor,
            "pairs": pairs, "counts": counts, "spliced_unread": sorted(unread),
            "times_own_size": {k: CFG.rungs[x] / v["energy"] for k, v in own.items() if v["energy"] > 0}}  # fmt: skip


def check0(res: Path) -> int:
    d = S1.item(res, f"self.{CFG.site}.c0")
    ok = d is not None
    S1.write(res / "stage0.json", {"experiment": P2["experiment"], "self": ok, "checks": d["checks"] if d else None,
                                   "read": {k: d["read"].get(k) for k in ("fve_base", "fve_ft", "dead", "max_rn_dev_alive", "counts")} if d else None})  # fmt: skip
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("check0", "ladder", "read"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--x", type=int, default=None)
    a = ap.parse_args()
    size = "rehearsal" if a.rehearse else "full"
    res, th = HERE / "results" / (a.tag or ("rehearse2" if a.rehearse else "run2")), TH1[size]
    bar = CFG.planted_part if size == "full" else th["planted_part"]  # the rehearsal keeps run 1's relaxed bar
    if a.job == "check0":
        return check0(res)
    lad = ladder(res, HERE / "results" / CFG.native_tags[size], bar, th)
    if a.job == "ladder":
        S1.write(res / "ladder.json", lad)
        return 0
    x = a.x if a.x is not None else lad["X"]
    if x is None:
        print(f"no X: {lad['status']}")
        return 1
    S1.write(res / "summary.json", read(res, x, th) | {"ladder": lad})
    return 0


if __name__ == "__main__":
    sys.exit(main())
