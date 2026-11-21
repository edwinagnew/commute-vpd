"""E235's CPU jobs (no GPU, no model): the per-pod item lists and inputs, stage 0's and stage 1's filed rules, the
two-per-card timing gain and the rehearsal's resume check. Rules: results/predictions.json "rules"; item files:
results/<tag>/items/<key>.json from run.py.

  list STAGE N   per-pod item keys (run.py's own `stage_items`, sharded as `pod` shards them) and every input outside
                 git a pod of that stage reads, with sizes -> results/<tag>/inputs_<STAGE>_<N>.json
  check0         S0a and S0b per site -> results/<tag>/stage0.json (exit 1 when a site fails)
  check1         G1, G2, the spliced-loss bar, ranges over crosscoder and fine-tune seeds, the primary pair call and the
                 run-2 decision -> results/<tag>/stage1.json
  timingcheck    timing_solo, timing_p0, timing_p1 -> the throughput of two crosscoders on one card against one -> timing.json
  resumecheck    (rehearsal) step-matched losses of self.A.c0 resumed (tag rehearse) against uninterrupted (rehearse_A),
                 floor rehearse_A against rehearse_B -> results/rehearse/resume_check.json (exit 1 on a miss)

    ~/commute-agents/fleet/cpu-run python experiments/crosscoder_proper/score.py check0 --tag run1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.jobs import load_experiment

RUN: Any = load_experiment("crosscoder_proper", "run")
HERE = Path(__file__).resolve().parent
P = json.loads(RUN.PRED.read_text())
CFG = RUN.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"]["run"].items()})
REPO = HERE.parents[1]


def item(res: Path, key: str) -> dict[str, Any] | None:
    p = res / "items" / f"{key}.json"
    d = json.loads(p.read_text()) if p.is_file() else None
    return d if d and d.get("done") else None


def write(path: Path, d: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1, default=str))
    print(json.dumps(d, indent=1, default=str)[:6000])


def size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def inputs(stage: str, n: int, res: Path, cc: dict[str, Any]) -> None:
    """Per pod: its item keys and the files outside git it reads (on top of the code), with sizes."""
    _, _, inv = RUN.SM.pairs()
    sets = REPO / "artifacts" / "sets"  # artifacts/ is the cache root locally and on a pod (env.CACHE_DIR)
    n_pile = math.ceil(cc["tokens"] / cc["epochs"] / 2 / RUN.ROW / cc["chunk_rows"]) * cc["chunk_rows"]
    common = [REPO / "artifacts/vpd", sets / f"train_{cc['pile_start']}_{n_pile}.npy", sets / f"train_{cc['pool_start']}_{cc['pool_rows']}.npy",
              sets / "val_0_8192.npy"]  # fmt: skip
    pods = []
    for i in range(n):
        keys = [t.key for t in RUN.stage_items(stage, CFG)[i::n]]
        files: set[Path] = set(common)
        for t in RUN.stage_items(stage, CFG)[i::n]:
            if t.what == "head":
                files |= {RUN.GRAFTS / f"{CFG.heads[t.site]}.pt", sets / "train_0_32768.npy"}
                continue
            name = CFG.control_task if t.what in ("self", "direction") else CFG.gauss_of if t.what == "gauss" else t.what.replace("__", "/")
            task = inv["states"][name]["task"] if name in inv["states"] else name
            if name in inv["states"]:
                files.add(REPO / inv["states"][name]["file"])
            if "path" in inv["tasks"][task]:
                files.add(REPO / inv["tasks"][task]["path"])
        listed = {str(f.relative_to(REPO)): size(f) for f in sorted(files)}
        pods.append({"pod": i, "items": keys, "n_items": len(keys), "inputs": listed, "mb": round(sum(listed.values()) / 1e6)})
    write(res / f"inputs_{stage}_{n}.json", {"stage": stage, "pods": n, "items": sum(p["n_items"] for p in pods), "per_pod": pods})


def passes_planted(h: dict[str, Any], th: dict[str, Any]) -> bool:
    """The planted bar: the lead latent fine-tune-specific (when filed) and the specific part at least planted_part."""
    lead_ok = bool(h["read"].get("lead", {}).get("specific")) or not th["planted_lead_specific"]
    return lead_ok and h["read"]["specific"] >= th["planted_part"]


def check0(res: Path, th: dict[str, Any]) -> int:
    sites = {}
    for s in CFG.sites:
        self_ = item(res, f"self.{s}.c0")
        heads = {c: item(res, f"head.{s}.c{c}") for c in CFG.seeds}
        per = {c: None if h is None else {"lead": h["read"].get("lead"), "specific": h["read"]["specific"],
                                          "count": h["read"]["counts"]["ft_specific"], "pass": passes_planted(h, th)}
               for c, h in heads.items()}  # fmt: skip
        s0a, s0b = self_ is not None, all(v is not None and v["pass"] for v in per.values())
        keep = ("fve_base", "fve_ft", "dead", "max_rn_dev_alive", "counts")
        sites[s] = {"S0a_self": s0a, "S0b_head": per, "S0b": s0b, "pass": s0a and s0b,
                    "self": None if self_ is None else {k: self_["read"].get(k) for k in keep}}  # fmt: skip
    write(res / "stage0.json", {"experiment": P["experiment"], "sites": sites})
    return 0 if all(v["pass"] for v in sites.values()) else 1


def call(reuse: list[float], new: list[float]) -> str:
    """The every-seed pair rule (predictions.json 'pair call')."""
    if len(reuse) < 2 or len(new) < 2 or any(math.isnan(v) for v in reuse + new):
        return "not read"
    return "reuse/new as built" if min(new) > max(reuse) else "reversed" if max(new) < min(reuse) else "no call"


def key(name: str, site: str, c: int) -> str:
    return f"{name.replace('/', '__')}.{site}.c{c}"


def value(res: Path, k: str, f: str = "specific", unread: tuple[str, ...] = ()) -> float:
    """A reading of item k ('energy' from the item, the rest from its read); nan if missing or unread."""
    d = item(res, k)
    if d is None or k in unread:
        return math.nan
    return float(d["energy"] if f == "energy" else d["read"][f])


def spliced_misses(res: Path, site: str, th: dict[str, Any]) -> tuple[str, ...]:
    """Stage-1 items at `site` whose spliced-loss recovered fraction falls more than the drop under the self crosscoder's."""
    self_ = item(res, f"self.{site}.c0")
    out = []
    for t in RUN.stage_items("1", CFG):
        d = item(res, t.key)
        if t.site == site and d is not None and self_ is not None:
            bar = [self_["spliced"][f"recovered_{m}"] - th["spliced_recovered_drop"] for m in (0, 1)]
            if any(d["spliced"][f"recovered_{m}"] < bar[m] for m in (0, 1)):
                out.append(t.key)
    return tuple(out)


def ranges(res: Path, site: str, unread: tuple[str, ...]) -> dict[str, Any]:
    """Each reading over the crosscoder seeds (fine-tune seed 0) and over the fine-tune seeds (crosscoder seed 0)."""
    out = {}
    for f in ("specific", "d_authors", "d_alt", "energy"):
        for side, names in (("reuse", CFG.reuse), ("new", CFG.new)):
            cc = [value(res, key(names[0], site, c), f, unread) for c in CFG.seeds]
            ft = [value(res, key(n, site, 0), f, unread) for n in names]
            out[f"{f} {side}"] = {"crosscoder_seeds": cc, "finetune_seeds": ft, "cc_range": max(cc) - min(cc), "ft_range": max(ft) - min(ft)}
    return out


def site_read(res: Path, site: str, ok0: bool, th: dict[str, Any]) -> dict[str, Any]:
    unread = spliced_misses(res, site, th)
    gauss = [item(res, f"gauss.{site}.c{c}") for c in CFG.seeds]
    g_spec = [value(res, f"gauss.{site}.c{c}", unread=unread) for c in CFG.seeds]
    reuse_all = [value(res, key(n, site, 0), unread=unread) for n in CFG.reuse]
    reuse_all += [value(res, key(CFG.reuse[0], site, c), unread=unread) for c in CFG.seeds[1:]]
    g1 = all(g is not None for g in gauss) and not any(passes_planted(g, th) for g in gauss if g is not None)
    g2 = not any(math.isnan(v) for v in g_spec + reuse_all) and min(g_spec) > max(reuse_all)
    rv = [value(res, key(n, site, 0), unread=unread) for n in CFG.reuse]
    nv = [value(res, key(n, site, 0), unread=unread) for n in CFG.new]
    gap = math.nan if any(map(math.isnan, rv + nv)) else fmean(nv) - fmean(rv)
    return {"S0": ok0, "G1": g1, "G2": g2, "gauss_specific": g_spec, "reuse_values": reuse_all,
            "gauss_match": [g.get("gauss_match") if g else None for g in gauss], "spliced_unread": list(unread),
            "ranges": ranges(res, site, unread), "grafts_readable": ok0 and g1,
            "pair": {"reuse": rv, "new": nv, "gap": gap, "call": call(rv, nv) if ok0 and g2 else "not applicable at this update size"}}  # fmt: skip


def check1(res: Path, th: dict[str, Any]) -> None:
    s0 = json.loads((res / "stage0.json").read_text())["sites"]
    sites = {s: site_read(res, s, bool(s0[s]["pass"]), th) for s in CFG.sites}
    pairs_a, grafts_b = sites["A"]["S0"] and sites["A"]["G2"], sites["B"]["grafts_readable"]
    run2 = "as planned" if pairs_a and grafts_b else "grafts at B only" if grafts_b else "pairs at A only (ask the user)" if pairs_a else "not run"
    sweep = {}
    for k in ["head.A.c0", *(f"head.A.c0.k{k}" for k in CFG.k_sweep)]:
        d = item(res, k)
        sweep[k] = None if d is None else {f: d["read"][f] for f in ("specific", "fve_base", "fve_ft", "l0", "dead")}
    dr = item(res, "direction.A.c0")
    direction = None if dr is None else {k: dr["read"].get(k) for k in ("lead", "specific", "cos_planted_best", "counts")}
    write(res / "stage1.json", {"experiment": P["experiment"], "sites": sites, "run2": run2, "k_sweep": sweep, "direction": direction})


def timingcheck(res: Path) -> None:
    t = {k: json.loads((res / f"timing_{k}.json").read_text()) for k in ("solo", "p0", "p1")}
    per = {k: v["train_minutes"] + v["stream_minutes"] for k, v in t.items()}
    gain = 2 * per["solo"] / fmean([per["p0"], per["p1"]])
    write(res / "timing.json", {"minutes_per_crosscoder": per, "gain_two_per_card": gain, "two_per_card": gain >= 1.5, "timings": t})


def resumecheck(results: Path) -> int:
    """Pass when every post-resume loss difference is within twice the floor two uninterrupted runs give, plus 1e-6."""
    c = {t: {r["step"]: r["loss"] for r in json.loads((results / t / "items" / "self.A.c0.json").read_text())["curve"]}
         for t in ("rehearse_A", "rehearse_B", "rehearse")}  # fmt: skip
    at = json.loads((results / "rehearse" / "items" / "self.A.c0.json").read_text())["resumed_at"]
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-12)
    A = c["rehearse_A"]
    floor = max(rel(c["rehearse_B"][s], A[s]) for s in A if s in c["rehearse_B"])
    after = {s: rel(c["rehearse"][s], A[s]) for s in A if s in c["rehearse"] and s >= at}
    ok = bool(after) and at > 0 and max(after.values()) <= 2 * floor + 1e-6
    write(results / "rehearse" / "resume_check.json", {"resumed_at": at, "floor_max_rel": floor, "pass": ok, "steps_compared": len(after),
                                                       "resumed_max_rel": max(after.values()) if after else math.nan, "diffs": after})  # fmt: skip
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("list", "check0", "check1", "timingcheck", "resumecheck"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--rehearse", action="store_true")
    a = ap.parse_args()
    size = "rehearsal" if a.rehearse else "full"
    res, th = HERE / "results" / a.tag, P["thresholds"][size]
    if a.job == "list":
        inputs(a.args[0], int(a.args[1]), res, P["settings"]["cc"][size])
    elif a.job == "check0":
        return check0(res, th)
    elif a.job == "check1":
        check1(res, th)
    elif a.job == "timingcheck":
        timingcheck(res)
    else:
        return resumecheck(HERE / "results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
