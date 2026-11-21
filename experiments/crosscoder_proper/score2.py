"""P2.49 run 2's CPU jobs (no GPU, no model): the stage-0 check, the ladder (which rungs passed, the first rung X that
passed on every seed, and the keys to run next) and stage 2's reading at X. Rules: results/run2/predictions.json
"rules"; items: results/<tag>/items/<key>.json from run2.py; the native rung's items are run 1's.

  check0    the base against itself through the scaled path (self.A.c0)    -> results/<tag>/stage0.json (exit 1 on a miss)
  ladder    per rung: every seed's specific part and pass/fail/open; X; next keys -> results/<tag>/ladder.json
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

from commute.jobs import config_from_settings, load_experiment

R2: Any = load_experiment("crosscoder_proper", "run2")
S1: Any = load_experiment("crosscoder_proper", "score")  # run 1's item reader, pair rule, spliced-loss bar
HERE = Path(__file__).resolve().parent
P2 = json.loads(R2.PRED.read_text())
S = {k: tuple(map(tuple, v)) if k == "pairs" else tuple(v) if isinstance(v, list) else v for k, v in P2["settings"].items()}
CFG = config_from_settings(R2.Config, S)
TH1 = json.loads(R2.RUN.PRED.read_text())["thresholds"]


def reading(res: Path, key: str) -> float | str | None:
    """An item's specific part; 'unbuilt' when its size could not be matched, 'stopped: <check>' at any other stop,
    None when it has not run."""
    p = res / "items" / f"{key}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    if d.get("done"):
        return float(d["read"]["specific"])
    stop = d.get("stopped")
    return "unbuilt" if stop == "change matched to the rung's size" else f"stopped: {stop}" if stop else None


def head_key(rung: int, seed: int) -> str:
    return f"head.{CFG.site}.c{seed}" if rung == CFG.native else f"head-r{rung}.{CFG.site}.c{seed}"


def rung_row(res: Path, native: Path, i: int, bar: float) -> dict[str, Any]:
    vals = {c: reading(native if i == CFG.native else res, head_key(i, c)) for c in CFG.seeds}
    v = list(vals.values())
    if any(isinstance(x, str) and x.startswith("stopped") for x in v):
        status = "stopped"
    elif "unbuilt" in v:
        status = "unbuilt"
    elif any(isinstance(x, float) and x < bar for x in v):
        status = "failed"  # one seed under the bar fails the rung (every-seed rule)
    elif all(isinstance(x, float) for x in v):
        status = "passed"
    else:
        status = "open"
    return {"size": CFG.rungs[i], "times_ref": CFG.rungs[i] / CFG.ref_energy, "seeds": vals, "status": status}


def ladder(res: Path, native: Path, bar: float) -> dict[str, Any]:
    """X = the lowest rung that passed on every seed with every rung under it failed or unbuilt. Next: seed 0 of every
    rung not yet run, then the missing seeds of the lowest open rung (the candidate) and stage 2 at the candidate."""
    rows = {i: rung_row(res, native, i, bar) for i in range(1, len(CFG.rungs))}
    first_seed = [head_key(i, CFG.seeds[0]) for i, r in rows.items() if r["seeds"][CFG.seeds[0]] is None]
    x, cand, status = None, None, "searching"
    for i, r in rows.items():
        if r["status"] in ("failed", "unbuilt"):
            continue
        if r["status"] == "stopped":
            status = f"stopped at rung {i}: tell the lead"
        elif r["status"] == "passed":
            x, status = i, f"X = rung {i}"
        elif r["seeds"][CFG.seeds[0]] is not None:
            cand = i
        break
    if all(r["status"] in ("failed", "unbuilt") for r in rows.values()):
        status = "none: no rung passed; stage 2 does not run"
    todo = lambda i: [k for k in CFG.stage2(i) if not isinstance(reading(res, k), float)]
    if first_seed:  # round 1 first: seed 0 at every rung, all at once
        nxt, s2 = first_seed, []
    elif x is not None:
        nxt, s2 = [], todo(x)
    elif cand is not None:  # the candidate's other seeds, and stage 2 at the candidate on the pods left (of 4)
        nxt = [head_key(cand, c) for c in CFG.seeds if rows[cand]["seeds"][c] is None]
        s2 = todo(cand)[: max(0, 4 - len(nxt))]
    else:
        nxt, s2 = [], []
    return {"rungs": rows, "bar": bar, "X": x, "candidate": cand, "status": status, "next_ladder": nxt, "next_stage2": s2}


def side_values(res: Path, keys: list[str], unread: set[str]) -> list[float]:
    return [math.nan if (v := reading(res, k)) is None or isinstance(v, str) or k in unread else v for k in keys]


def spliced_unread(res: Path, keys: list[str], th: dict[str, Any]) -> set[str]:
    """Items whose spliced-loss recovered fraction falls more than run 1's drop under this run's self crosscoder."""
    self_ = S1.item(res, f"self.{CFG.site}.c0")
    out = set()
    for k in keys:
        d = S1.item(res, k)
        if d and self_ and any(d["spliced"][f"recovered_{m}"] < self_["spliced"][f"recovered_{m}"] - th["spliced_recovered_drop"] for m in (0, 1)):
            out.add(k)
    return out


def against(task: list[float], rnd: list[float]) -> str:
    if any(map(math.isnan, task + rnd)):
        return "not read"
    return "above random" if min(task) > max(rnd) else "under random" if max(task) < min(rnd) else "ties random"


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
    return {"X": x, "size": CFG.rungs[x], "times_ref": CFG.rungs[x] / CFG.ref_energy, "G1_random_not_new": g1, "G2_random_above_reuse": g2,
            "random": rnd, "tasks": {t: {"specific": v, "against_random": against(v, rnd) if g1 else "not read (G1 failed: a draw missing or reading new)"} for t, v in tasks.items()},
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
    lad = ladder(res, HERE / "results" / CFG.native_tags[size], bar)
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
