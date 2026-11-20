"""P2.46's score and hand-off, no GPU (run.py trains and reads; this reads what it wrote). Rules: results/predictions.json.

  score       per pair and side the reach ratios at matched output change (primary) and matched gain, the share check,
              and the rules P1, P1b, P3 over the nine built pairs and the four twins -> results/score.json
  handoff N   results/handoff.json, the states for definition-lead in E136's inventory format, and the N sync lists
              results/manifest_<I>_of_<N>.txt (the inputs outside git each shard reads)

    python experiments/gain_only/score.py handoff 3      # before the pods: the sync lists
    python experiments/gain_only/score.py score          # after every shard is downloaded
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Any

from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN

R: Any = load_experiment("gain_only", "run")
E228 = R.E228


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) else math.nan  # nan is written as the string "nan"


def at(reads: dict[str, Any], key: str) -> dict[str, Any] | None:
    """The read of the state kept for `key` ("kl:<ref>" or "gain:<ref>"), or None."""
    return next((v for v in reads["states"].values() if key in v["for"]), None)


def side_reach(ref: str, task: str, out: Path, seeds: tuple[int, ...], kind: str) -> dict[str, Any]:
    """A reference's reach: the median over seeds of the kept state's gain (read from its file) over the reference's mean
    gain; refused when the reference gain is under MIN_GAIN or any seed is unread or not finite."""
    paths = [(out / "items" / f"{R.fname(task)}_s{s}.json", out / "reads" / f"{R.fname(task)}_s{s}.json") for s in seeds]
    if not all(a.is_file() and b.is_file() for a, b in paths):
        return {"reach": math.nan, "status": "unread"}
    items, reads = [json.loads(a.read_text()) for a, _ in paths], [json.loads(b.read_text()) for _, b in paths]
    if not all(x.get("done") for x in reads):
        return {"reach": math.nan, "status": "unread"}
    g_ref = items[0]["refs"][ref]["gain"]
    got = [at(x, f"{kind}:{ref}") for x in reads]
    vals = [num(s["gain"]) / g_ref if s else math.nan for s in got]
    shares = [num(s["kl_share"]) for s in got if s]
    if not g_ref >= MIN_GAIN or not all(math.isfinite(v) for v in vals):
        return {"reach": math.nan, "status": "refused", "ref_gain": g_ref, "seeds": vals}
    return {"reach": median(vals), "seeds": vals, "ref_gain": g_ref, "kl_share": median(shares) if shares else math.nan, "status": "ok"}


def pair_call(reuse: list[float], new: list[float], bar: float) -> str:
    """The filed per-pair call on the sides' reach (each side the mean over its references); a reach of bar reaches."""
    if not reuse or not new or not all(math.isfinite(v) for v in reuse + new):
        return "not read"
    a, b = fmean(reuse) >= bar, fmean(new) >= bar
    return {(True, False): "as filed", (False, True): "reversed", (True, True): "both reach", (False, False): "neither reaches"}[(a, b)]


def share_call(mine: list[float], full_reuse: float, band: float) -> str:
    """P3: every gain-only side's KL-matched kl_share at least the reuse side's full fine-tune kl_share minus band."""
    if not mine or not all(map(math.isfinite, [*mine, full_reuse])):
        return "not read"
    return "as filed" if min(mine) >= full_reuse - band else "below"


def group_rule(calls: list[str], min_read: int, misses: int) -> dict[str, Any]:
    n_read, n_ok = sum(c != "not read" for c in calls), sum(c == "as filed" for c in calls)
    verdict = "no call" if n_read < min_read else "holds" if n_ok >= n_read - misses else "fails"
    return {"verdict": verdict, "as_filed": n_ok, "read": n_read}


def score(cfg: Any, th: dict[str, Any], out: Path) -> dict[str, Any]:
    sides, refs, inv = R.catalogue()
    task_of = {ref: inv["states"][s[0]]["task"] for ref, s in refs.items()}
    full = {s: num(json.loads(p.read_text())["own"]["kl_share"]) for s in E228.every_state(E228.catalogue()[0]) if (p := E228.STATES / f"{E228.fname(s)}.json").is_file()}
    res: dict[str, Any] = {"experiment": R.EXPERIMENT, "pairs": {}, "rules": {}}
    for p, sd in sides.items():
        rp: dict[str, Any] = {k: {ref: {m: side_reach(ref, task_of[ref], out, cfg.seeds, m) for m in ("kl", "gain")} for ref in sd[k]} for k in ("reuse", "new")}
        res["pairs"][p] = rp
        for m in ("kl", "gain"):
            reach = {k: [rp[k][ref][m]["reach"] for ref in sd[k]] for k in ("reuse", "new")}
            rp[f"call_{m}"] = pair_call(reach["reuse"], reach["new"], th["reach_bar"])
        reuse_full = [full[s] for ref in sd["reuse"] for s in refs[ref] if s in full]
        rp["full_reuse_kl_share"] = fmean(reuse_full) if reuse_full else math.nan
        rp["share_call"] = share_call([rp[k][ref]["kl"].get("kl_share", math.nan) for k in ("reuse", "new") for ref in sd[k]], rp["full_reuse_kl_share"], th["share_band"])
    for group, ps in (("built", E228.SM.BUILT), ("twins", tuple(p for p in sides if p not in E228.SM.BUILT))):
        for rule, key in (("P1", "call_kl"), ("P1b", "call_gain"), ("P3", "share_call")):
            calls = [res["pairs"][p][key] for p in ps]
            res["rules"][f"{rule}_{group}"] = group_rule(calls, th["min_read"][group], th["misses_allowed"][group]) | {"calls": dict(zip(ps, calls, strict=True))}
            R.log(f"{rule} {group}: {json.dumps({k: v for k, v in res['rules'][f'{rule}_{group}'].items() if k != 'calls'})}")
    R.dump(out / "score.json", res)
    return res


def handoff(cfg: Any, out: Path, n: int) -> None:
    """results/handoff.json: every exported state in E136's inventory format (task = its reference's task, so the read
    rows are the same), with what it was kept for, its sha256 and reads; and per shard the inputs outside git (for sync)."""
    sides, refs, inv = R.catalogue()
    states: dict[str, Any] = {}
    for p in sorted((out / "reads").glob("*.json")):
        rd = json.loads(p.read_text())
        for step, v in rd["states"].items():
            states[f"gain_only/{R.fname(rd['task'])}_s{rd['seed']}_step{step}"] = {"task": rd["task"], "file": v["file"], "sha256": v["sha256"], "for": v["for"],
                                                                                  "gain": v["gain"], "kl": v["kl"]}  # fmt: skip
    doc = {"experiment": R.EXPERIMENT, "format": "torch.save dict: matrix name -> bf16 tensor dW (W = W0 + dW) over all 24 decomposed matrices, "
           "zero except h.<l>.mlp.down_proj (W0 diag(g - 1) on its columns) and h.<l>.attn.o_proj (on its 128-column head blocks); "
           "the format of experiments/refit_96key/results/large/dW_*.pt. The gains themselves: results/large/gains/<task>_s<seed>.pt "
           "(name -> {step, gain, kl, reached, gains: {mlp (4, 3072), head (4, 6)}}).", "pairs": sides,
           "tasks": {t: inv["tasks"][t] for t in {v["task"] for v in states.values()}}, "states": states}  # fmt: skip
    R.dump(out / "handoff.json", doc)
    todo, ctl = R.items(refs, inv, cfg), inv["states"][R.CONTROL_STATE]
    for i in range(n):
        mine = R.shard(todo, i, n)
        files = {inv["states"][s]["file"] for _, _, rs in mine for ref in rs for s in refs[ref]} | {ctl["file"], inv["tasks"][ctl["task"]]["path"]}
        files |= {inv["tasks"][t]["path"] for t, _, _ in mine if "path" in inv["tasks"][t]}
        shared = ["artifacts/vpd/", "artifacts/sets/", str(E228.ALIVE.relative_to(EXPERIMENTS.parent))]
        mb = sum((EXPERIMENTS.parent / f).stat().st_size for f in files) / 2**20
        head = f"# P2.46 shard {i} of {n}: {len(mine)} items, tasks {sorted({t for t, _, _ in mine})}; {mb:.0f} MB of states and windows"
        (out / f"manifest_{i}_of_{n}.txt").write_text("\n".join([head, *shared, *sorted(files)]) + "\n")
    R.log(f"handoff: {len(states)} states; {n} sync lists")


def main(argv: list[str]) -> int:
    require_filed(R.PRED)
    cfg, th = R.load_config()
    out = R.HERE / "results"
    if argv == ["score"]:
        score(cfg, th, out)
    else:
        assert len(argv) == 2 and argv[0] == "handoff", f"usage: score.py score | handoff N, got {argv}"
        handoff(cfg, out, int(argv[1]))
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
