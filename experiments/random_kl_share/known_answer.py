"""E228's known-answer stage (check 1, cell P2.44): the KL-form share and r (E228's one pass, run.py `readings`) and
neurons for half the gain (k_half, E98's reader as E227 called it) on E214's known-answer items, each on its windows fed
WHOLE (`realistic_reads.TargetRead`; Gamma is bidirectional over the row), scored by R's rule (every reuse reading above
every new one by more than the split-half band). It corrects E227 (experiments/known_answer_baselines), whose windows
were altered; E227's effective rank stands and is not recomputed. Plan reports/plans/2026-11-20_plan_p2-44-random-kl-
matched.md, "Known-answer stage"; settings, expected values and rules in results/predictions_known_answer.json (the
run refuses unless it is committed unchanged). Nothing is trained.

Targets: the new edits' written answer at the cue windows' last position (E214's y per cue); the reuse edits lower the
copy answer (E227), so per row the token whose log-probability the edit raises most at copy24's position `half`.

One job, in order, each item's JSON written as it ends (a rerun skips it):
  0  do-nothing (zero change, E214's cue windows): every K difference, r's numerator and gain, the KL and the neuron
     cells exactly 0, both shares nan, k_half not read; exit 1 on a miss
  1  new_few, new_thin, x2, x10, rebuilt (moved: "does not apply", it appends neurons): gate, then the readings on all
     rows and two halves -> results/known_answer/<item>.json; the new items' neuron reading must reproduce E227's saved
     one (no Gamma, causal, same target), exit 1 if not
  2  score (also alone, no GPU) -> results/known_answer/score.json

    PRED_SHA=<sha> python experiments/random_kl_share/known_answer.py 2>&1 | tee -a experiments/random_kl_share/results/known_answer.log
    python experiments/random_kl_share/known_answer.py score | manifest
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from commute import realistic_reads as RR
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

RS: Any = load_experiment("random_kl_share", "run")  # E228: Splitter, readings, kl_of, log, dump, stop, num
KA: Any = load_experiment("known_answer_baselines", "run")  # E227: e214_world, weight_change, gate, CUE_ITEMS
PRED = RS.RESULTS / "predictions_known_answer.json"
OUT = RS.RESULTS / "known_answer"
E227 = EXPERIMENTS / "known_answer_baselines" / "results" / "items"
KEYS = ("kl_share", "r", "k_half")
log, dump, stop, num = RS.log, RS.dump, RS.stop, RS.num
INPUTS = ("artifacts/vpd/", "artifacts/sets/train_0_32768.npy", "artifacts/sets/val_0_256.npy",
          "experiments/transport_test/results/large/alive.pt", "experiments/open_pair/results/large/amend1/windows_copy_trained.pt",
          "experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt")  # fmt: skip


@dataclass(frozen=True)
class Config:  # filed in predictions_known_answer.json "settings"
    reuse: tuple[str, ...]
    new: tuple[str, ...]
    absent: tuple[str, ...]  # items no reading applies to
    split_seed: int
    split_tol: float

    def __post_init__(self) -> None:
        assert set(self.reuse) == {"x2", "x10", "rebuilt"} and set(self.new) == {"new_few", "new_thin"} and self.absent == ("moved",)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    return config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}), pred["thresholds"]


# ---- one item -------------------------------------------------------------------------------------------------------


def item_read(w: Any, item: str) -> tuple[Any, RR.TargetRead]:
    """F and its windows as E214 fed them, whole: cue items' last position, target y per cue; copy24 items' `half`,
    target the task table's answer (replaced for the reuse items by `most_raised`)."""
    F, toks, _, piece, _ = w.make(item)
    assert piece is None
    if item in KA.CUE_ITEMS:
        t, y = toks.cpu(), torch.tensor(w.cues.y)[w.cues.cue]
        return F, RR.TargetRead(item, t, y, t.shape[1] - 1, t, y)
    h, table = (d := torch.load(KA.KL.P.COPY24, weights_only=False))["heldout"], d["table"]
    assert torch.equal(h.tokens, toks.cpu())
    return F, RR.target_read(item, h.tokens, table, h.half)


@torch.no_grad()
def most_raised(net: Any, r: RR.TargetRead, dW: dict[str, Tensor], batch: int) -> RR.TargetRead:
    """The same read with each row's target the token whose log-probability M + dW raises most over M at r.pos."""
    out = []
    for s in range(0, len(r.toks), batch):
        t = r.toks[s : s + batch]
        p0 = net.run(t)[0][:, r.pos].float().log_softmax(-1)
        with net.at(dW):
            p1 = net.run(t)[0][:, r.pos].float().log_softmax(-1)
        out.append((p1 - p0).argmax(-1).cpu())
    y = torch.cat(out)
    return RR.TargetRead(r.name, r.toks, y, r.pos, r.toks, y)


def neurons(net: Any, r: RR.Read, dW: dict[str, Tensor], S: dict[str, Any]) -> dict[str, Any]:
    kh = RR.k_half_of(net, r, dW, S)["lo"]
    return {k: kh[k] for k in ("k_half", "r_half", "r_half_num", "G", "numerators")}


def read_all(net: Any, sp: Any, r: RR.TargetRead, dW: dict[str, Tensor], S: dict[str, Any], seed: int) -> dict[str, Any]:
    perm = torch.randperm(len(r.toks), generator=torch.Generator().manual_seed(seed))
    h = len(perm) // 2
    rec = {"pass": RS.readings(net, sp, r, dW, perm), "kl": RS.kl_of(net, r, dW, S["batch"])}
    rec["neurons"] = {n: neurons(net, rr, dW, S) for n, rr in (("all", r), ("half1", RR.subset(r, perm[:h])), ("half2", RR.subset(r, perm[h:])))}
    return rec


def zero_check(w: Any, sp: Any, S: dict[str, Any], cfg: Config) -> None:
    path = OUT / "do_nothing.json"
    F, r = item_read(w, "do_nothing")
    zero, _ = KA.weight_change(w.net, w.M, F)
    z = read_all(w.net, sp, r, zero, S, cfg.split_seed) | {"gate": KA.gate(w.net, F, r, zero)}
    p, K, n = z["pass"], z["pass"]["K"], z["neurons"]["all"]
    exact = {"numerator": p["numerator"], "gain": p["gain"], "kl": z["kl"], "K00-K0": K["K00"] - K["K0"], "K01-K0": K["K01"] - K["K0"],
             "K10-K00": K["K10"] - K["K00"], "K11-K01": K["K11"] - K["K01"], "r_half_num": n["r_half_num"], "G": n["G"], "gate": z["gate"]}  # fmt: skip
    z |= {"experiment": RS.EXPERIMENT, "item": "do_nothing", "exact": exact}
    log(f"[do_nothing] {json.dumps(RS.jsonable(z))}")
    bad = {k: v for k, v in exact.items() if v != 0}
    bad |= {k: v for k, v in (("kl_share", p["kl_share"]), ("r", p["r"])) if not math.isnan(v)}
    bad |= {"k_half": n["k_half"]} if n["k_half"] is not None else {}
    if bad:
        stop(path, z, f"the do-nothing does not read 0 / nan: {bad}")
    dump(path, z | {"done": True})
    log("STAGE do_nothing DONE")


def reproduces(item: str, rec: dict[str, Any], tol: float) -> str:
    """'' if the new item's neuron reading equals E227's saved one (k_half equal, r_half_num within tol), else why not."""
    old, got = json.loads((E227 / f"{item}.json").read_text())["rows"]["all"], rec["neurons"]["all"]
    ok = old["k_half"] == got["k_half"] and abs(old["r_half_num"] - got["r_half_num"]) <= tol
    return "" if ok else f"{item}: neurons {got['k_half']}, {got['r_half_num']} against E227's {old['k_half']}, {old['r_half_num']}"


def read_item(w: Any, sp: Any, item: str, cfg: Config, th: dict[str, Any], S: dict[str, Any]) -> None:
    path = OUT / f"{item}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return log(f"{item}: done before, skipped")
    t0 = time.time()
    F, r = item_read(w, item)
    dW, why = KA.weight_change(w.net, w.M, F)
    rec: dict[str, Any] = {"experiment": RS.EXPERIMENT, "item": item, "n_rows": len(r.toks), "pos": r.pos, "seq_len": r.toks.shape[1]}
    if dW is None:
        return dump(path, rec | {"reading": "does not apply", "why": why, "done": True})
    if not (g := KA.gate(w.net, F, r, dW)) <= th["gate_tol"]:
        stop(path, rec | {"gate": g}, f"{item}: M plus its weight change does not give F's logits ({g:.3g})")
    if item in cfg.reuse:
        r = most_raised(w.net, r, dW, S["batch"])
        rec["targets"] = torch.bincount(r.tgt).nonzero().flatten().tolist()
    rec |= {"reading": "read", "gate": g} | read_all(w.net, sp, r, dW, S, cfg.split_seed)
    log(f"[{item}] {json.dumps(RS.jsonable({k: v for k, v in rec.items() if k != 'neurons'}))}")
    log(f"[{item} neurons] " + json.dumps(RS.jsonable({n: {k: v for k, v in x.items() if k != 'numerators'} for n, x in rec["neurons"].items()})))
    if item in cfg.new and (miss := reproduces(item, rec, th["neuron_reproduce_tol"])):
        stop(path, rec, miss)
    dump(path, rec | {"done": True, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30})
    log(f"ITEM {item} DONE in {time.time() - t0:.0f} s")
    del F
    torch.cuda.empty_cache()


# ---- score (no GPU) -------------------------------------------------------------------------------------------------


def value(rec: dict[str, Any], key: str, rows: str = "all") -> float:
    """Oriented so higher = more reuse: the shares as read, k_half as -log2 (fewer neurons = reuse); nan unreadable."""
    if rec.get("reading") != "read":
        return math.nan
    if key == "k_half":
        k = rec["neurons"][rows]["k_half"]
        return -math.log2(k) if k else math.nan
    p = rec["pass"] if rows == "all" else rec["pass"]["halves"][int(rows[-1]) - 1]
    return num(p[key])


def band_of(gaps: list[float], band_min: float) -> dict[str, Any]:
    """R's band: the largest finite split-half gap over the items; band_min when none is finite or all are 0."""
    g = [x for x in gaps if math.isfinite(x)]
    return {"band": max(g), "source": "largest split-half gap"} if g and max(g) > 0 else {"band": band_min, "source": "band_min"}


def order(v: dict[str, float], reuse: tuple[str, ...], new: tuple[str, ...], band: float) -> dict[str, Any]:
    """R's rule over the reuse x new pairs: a pair is ordered when reuse - new > band, strictly; 'reuse above new' when
    every pair is read and ordered, 'silent' when no pair is read, else 'mixed: n of m pairs ordered'."""
    pairs = [(a, b) for a in reuse for b in new]
    read = [(a, b) for a, b in pairs if math.isfinite(v[a]) and math.isfinite(v[b])]
    n = sum(v[a] - v[b] > band for a, b in read)
    if not read:
        return {"verdict": "silent", "values": v}
    unread = f", {len(pairs) - len(read)} unread" if len(read) < len(pairs) else ""
    return {"verdict": "reuse above new" if n == len(pairs) else f"mixed: {n} of {len(pairs)} pairs ordered{unread}", "values": v, "band": band}


def score(cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    recs = {i: json.loads((OUT / f"{i}.json").read_text()) for i in (*cfg.reuse, *cfg.new)}
    out: dict[str, Any] = {"experiment": RS.EXPERIMENT, "absent": {i: "does not apply" for i in cfg.absent}, "rules": {}}
    for key in KEYS:
        b = band_of([abs(value(r, key, "half1") - value(r, key, "half2")) for r in recs.values()], th["band_min"][key])
        out["rules"][key] = order({i: value(r, key) for i, r in recs.items()}, cfg.reuse, cfg.new, b["band"]) | {"band_source": b["source"]}
        log(f"score {key}: {json.dumps(RS.jsonable(out['rules'][key]))}")
    dump(OUT / "score.json", out)
    return out


def main(argv: list[str]) -> int:
    require_filed(PRED)
    cfg, th = load_config()
    if argv in (["score"], ["manifest"]):
        _ = score(cfg, th) if argv == ["score"] else (OUT / "manifest.txt").write_text("\n".join(("# E228 known-answer stage", *INPUTS)) + "\n")
        return 0
    assert not argv, f"usage: known_answer.py [score | manifest], got {argv}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E211 and E214
    S = json.loads((EXPERIMENTS / "neuron_r" / "results" / "predictions.json").read_text())["settings"]
    log(f"{RS.EXPERIMENT} known-answer start; settings {asdict(cfg)}; thresholds {th}")
    w = KA.e214_world()
    KA.KL.RESULTS = OUT / "e214_build"  # E214's build record goes here, not to E227's folder
    KA.KL.RESULTS.mkdir(parents=True, exist_ok=True)
    _ = w.edits  # build first: a detector swap rebuilds the cue windows (E227 stage 0)
    sp = RS.Splitter(w.net, cfg.split_tol)
    zero_check(w, sp, S, cfg)
    for item in (*cfg.new, *cfg.reuse, *cfg.absent):
        read_item(w, sp, item, cfg, th, S)
    score(cfg, th)
    log(f"{RS.EXPERIMENT} known-answer DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
