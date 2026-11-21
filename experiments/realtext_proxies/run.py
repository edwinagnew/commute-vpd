"""P2.53 (compare-lead): the three proxies still standing, read on the four real-text pairs' saved fine-tunes at each
pair's scored (larger) matched size, three seeds a side. Plan reports/plans/2026-11-21_plan_p2-53-realtext-proxies.md;
settings, bars and filed rules in results/predictions.json (the run refuses unless it is committed unchanged). Nothing
is trained: every fine-tune is a saved state of E211 (P2.38 run 2) or E233 (P2.48).

The pairs, the side built as needing what the original model holds first (built labels, run 1 and E233's premise):
  label      L1 topic label   vs L3 seeded coin label, 512 news windows, size 0.01    (E211)
  lowercase  T1 Pile lowercase vs T4 reversed word order, 500 held windows, size 0.0175 (E211)
  language   T2 French Wikipedia vs T3 Yoruba Wikipedia, 500 held windows, size 0.0175 (E211)
  british    British spelling vs its scrambled twin, 500 held occurrences, size 0.0175  (E233)

The readings, each by the code that defined it (imported, not copied):
  r      E228's `readings` (experiments/random_kl_share/run.py): (L01 - L00) / (T1 - T0) in log-odds of the scored token,
         nan under a gain of 1; its kl_share rides along unreported
  S      P1.45's `cells` and `score` (experiments/gating_vpd/run.py): 1 - KL(F||H) / KL(F||M) at the read position
  prior  E133's `keydep.geo_rate` (experiments/prior_vs_share/run.py stage 1) of W0's log P(scored token): one per task
Controls, first: the do-nothing (a zero change on each pair's first rows: r's cells and K's exactly 0, S's KL(F||M)
and K01 - f under kl_floor); then, before each fine-tune, a Gaussian change at that fine-tune's per-matrix norms
(`finetune.random_direction`), read for r (expected unreadable, gain under 1: a stop if not) and S.

Jobs:
  pod I [--rehearse]  the pairs settings.pods[I] names; --rehearse reads only seed 0 of each pair's first side, at full
                      read sizes, then exits (the resume test: the same line again skips what is written)
  score               no GPU: per pair and proxy the scorecard's seed call against the built labels -> results/table.json

    PRED_SHA=<sha> python experiments/realtext_proxies/run.py pod 0 2>&1 | tee -a experiments/realtext_proxies/results/pod0.log
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import keydep as KD
from commute import old_tasks as OT
from commute import realistic as R
from commute import realistic_reads as RR
from commute import rule_retention as RT
from commute import steering as ST
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

E228: Any = load_experiment("random_kl_share", "run")  # readings, Splitter
GV: Any = load_experiment("gating_vpd", "run")  # P1.45's cells, score, halves
R2: Any = load_experiment("realistic_tasks", "run2")  # E211: task_data (the label rows), Config, PRED
BS: Any = load_experiment("british_style", "run")  # E233: reads, Config, PRED
HERE = Path(__file__).resolve().parent
RESULTS, PRED, REPO = HERE / "results", HERE / "results" / "predictions.json", EXPERIMENTS.parent
E211, E233 = EXPERIMENTS / "realistic_tasks" / "results", EXPERIMENTS / "british_style" / "results"
TEXTS = E211 / "large" / "run3" / "texts.pt"  # E212's prep: E211's read rows of every task, int32
PAIRS = {"label": ("L1", "L3"), "lowercase": ("T1", "T4"), "language": ("T2", "T3"), "british": ("real", "twin")}
SIZE = {"label": "0.01", "lowercase": "0.0175", "language": "0.0175", "british": "0.0175"}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    pods: tuple[tuple[str, ...], ...]
    seeds: tuple[int, ...]
    random_seed: int  # the draw before state k of a pod has seed random_seed + 10 * pod + k
    split_tol: float
    split_seed: int
    s_batch: int

    def __post_init__(self) -> None:
        assert sorted(p for pod in self.pods for p in pod) == sorted(PAIRS) and self.seeds == (0, 1, 2)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


# ---- the saved states and their rows --------------------------------------------------------------------------------


def saved(pair: str, task: str, seed: int) -> tuple[Path, float]:
    """A state's file and its saved r (E211's share in log-odds, E233's share; nan where its gain was under 1)."""
    if pair == "british":
        t = json.loads((E233 / "scored" / "states" / f"{task}_s{seed}.json").read_text())["targets"][SIZE[pair]]
        return REPO / t["file"], float(t["share"])
    t = json.loads((E211 / f"run2_{task}.json").read_text())["states"][f"s{seed}_{SIZE[pair]}"]
    return REPO / t["file"], float(t["share"]["share"]["lo"])


def rows(net: ST.Net, pair: str) -> dict[str, RR.Read]:
    """Each side's read rows as its source read them: the label pair by E211's own loader (it holds the labels), the
    text pairs from E212's copy of E211's rows, the British pair from E233's windows."""
    if pair == "british":
        C = config_from_settings(BS.Config, json.loads(BS.PRED.read_text())["settings"])
        return BS.reads(torch.load(E233 / "large" / "data.pt", weights_only=False), C.half)
    C2 = config_from_settings(R2.Config, json.loads(R2.PRED.read_text())["settings"])
    texts = torch.load(TEXTS, weights_only=False)["reads"]
    out = {}
    for t in PAIRS[pair]:
        if pair == "label":
            out[t] = R2.task_data(net, C2, t)[0]
            assert torch.equal(out[t].toks, texts[t].long()), f"{t}: E211's label rows are not E212's copy"
        else:
            w = texts[t].long()
            out[t] = RR.Read(t, w, w[:, C2.half + 1], C2.half, w, w[:, C2.half + 1])
    return out


# ---- the three readings ---------------------------------------------------------------------------------------------


@torch.no_grad()
def prior(net: ST.Net, r: RR.Read) -> float:
    lp = torch.cat([ST.logp_at(net.run(r.toks[s : s + 64])[0], r.pos, r.tgt[s : s + 64]) for s in range(0, len(r.toks), 64)])
    return KD.geo_rate(lp)


def s_read(net: ST.Net, M: RT.Model, r: RR.Read, dW: dict[str, Tensor], cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """P1.45's S of W0 + dW on r's rows at r's read position, with its two halves."""
    F = RT.Model(OT.with_delta(net, {f"{k}.weight": v for k, v in dW.items()}).net)
    c = GV.cells(net.model, F, M, r.toks, slice(r.pos, r.pos + 1), cfg.s_batch)
    del F
    torch.cuda.empty_cache()
    return GV.score(c, th["kl_floor"]) | {"halves": [GV.score(c, th["kl_floor"], h) for h in GV.halves(len(r.toks), cfg.split_seed)]}


def both(net: ST.Net, sp: Any, M: RT.Model, r: RR.Read, dW: dict[str, Tensor], cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    rec = E228.readings(net, sp, r, dW) | {"S": s_read(net, M, r, dW, cfg, th), "size": R.update_size(dW, net.W0)}
    return rec | {"seconds": time.time() - t0}


def brief(rec: dict[str, Any]) -> str:
    s = rec["S"]
    return f"r {rec['r']:.4f} (gain {rec['gain']:.4f}), S {s['S']}, S_c {s['S_c']}, kl_fm {s['means']['kl_fm']:.4g}, f {s['means']['f']:.4g}, {rec['seconds']:.1f} s"


# ---- the pod job ----------------------------------------------------------------------------------------------------


class Out:
    def __init__(self, i: int) -> None:
        self.path = RESULTS / f"pod{i}.json"
        self.d: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {"pod": i, "zero": {}, "prior": {}, "states": {}}
        self.d.pop("stopped", None)

    def put(self, where: str, key: str, rec: Any) -> None:
        self.d[where][key] = rec
        write_stage(self.path, jsonable(self.d))

    def stop(self, why: str) -> None:
        self.d["stopped"] = why
        write_stage(self.path, jsonable(self.d))
        log(f"STOP: {why}")
        hard_exit(1)


def zero(o: Out, net: ST.Net, sp: Any, M: RT.Model, pair: str, r: RR.Read, cfg: Config, th: dict[str, Any]) -> None:
    """The do-nothing on a pair's first rows: r's numerator, gain and E57's K differences exactly 0; S's KL(F||M) and
    K01 - f under kl_floor, f positive (P1.45's stage-0 checks, its filed floor)."""
    if pair in o.d["zero"]:
        return
    z = both(net, sp, M, r, {k: torch.zeros_like(w).cpu() for k, w in net.W0.items()}, cfg, th)
    K, m = z["K"], z["S"]["means"]
    exact = {"numerator": z["numerator"], "gain": z["gain"], "K00-K0": K["K00"] - K["K0"], "K01-K0": K["K01"] - K["K0"],
             "K10-K00": K["K10"] - K["K00"], "K11-K01": K["K11"] - K["K01"]}  # fmt: skip
    miss = [f"{k} {v}" for k, v in exact.items() if v != 0]
    miss += [f"KL(F||M) {m['kl_fm']}"] if not m["kl_fm"] <= th["kl_floor"] else []
    miss += [f"K01 - f {m['K01'] - m['f']}"] if not abs(m["K01"] - m["f"]) <= th["kl_floor"] else []
    miss += [f"floor f {m['f']}"] if not (math.isfinite(m["f"]) and m["f"] > 0) else []
    o.put("zero", pair, z | {"exact": exact, "miss": miss})
    log(f"[zero {pair}] exact {json.dumps(jsonable(exact))}; {brief(z)}; {'MISS ' + '; '.join(miss) if miss else 'checks hold'}")
    if miss:
        o.stop(f"do-nothing on {pair}: " + "; ".join(miss))


def one_state(o: Out, net: ST.Net, sp: Any, M: RT.Model, pair: str, task: str, r: RR.Read, seed: int, draw: int, cfg: Config, th: dict[str, Any]) -> None:
    name = f"{pair}/{task}_s{seed}"
    if o.d["states"].get(name, {}).get("done"):
        log(f"{name}: done before, skipped")
        return
    path, saved_r = saved(pair, task, seed)
    dW = {k: v.float() for k, v in torch.load(path, weights_only=False).items()}
    rnd = both(net, sp, M, r, FT.random_direction(dW, draw), cfg, th) | {"draw_seed": draw}
    log(f"[{name} random {draw}] {brief(rnd)}")
    rec = {"pair": pair, "task": task, "seed": seed, "file": str(path.relative_to(REPO)), "saved_r": saved_r, "random": rnd}
    if not abs(rnd["gain"]) < th["random_gain_max"]:
        o.put("states", name, rec)
        o.stop(f"{name}: a random change at its norms moves the scored token by {rnd['gain']:.3f} log-odds")
    own = both(net, sp, M, r, dW, cfg, th)
    log(f"[{name} own] {brief(own)}; saved r {saved_r:.4f}")
    rec |= {"own": own, "done": True}
    o.put("states", name, rec)
    same = (math.isnan(saved_r) and math.isnan(own["r"])) or abs(own["r"] - saved_r) <= th["reproduce_tol"]
    if not same:
        o.stop(f"{name}: r {own['r']:.4f} does not reproduce its saved {saved_r:.4f}")


def pod_job(cfg: Config, th: dict[str, Any], i: int, rehearse: bool) -> None:
    net, o = ST.Net("cuda"), Out(i)
    net.net.requires_grad_(False)
    M, sp = RT.Model(net.net), E228.Splitter(net, cfg.split_tol)
    k = 0
    for pair in cfg.pods[i]:
        rd = rows(net, pair)
        zero(o, net, sp, M, pair, rd[PAIRS[pair][0]], cfg, th)
        for task in PAIRS[pair]:
            if task not in o.d["prior"]:
                o.put("prior", task, {"pair": pair, "prior": prior(net, rd[task]), "n_rows": len(rd[task].toks), "pos": rd[task].pos})
                log(f"[prior {task}] {json.dumps(jsonable(o.d['prior'][task]))}")
        log(f"STAGE 0 {pair} DONE")
        for task in PAIRS[pair][:1] if rehearse else PAIRS[pair]:
            for seed in cfg.seeds[:1] if rehearse else cfg.seeds:
                one_state(o, net, sp, M, pair, task, rd[task], seed, cfg.random_seed + 10 * i + k, cfg, th)
                k += 1
        log(f"PAIR {pair} DONE")
    o.d["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
    write_stage(o.path, jsonable(o.d | {"done": not rehearse}))
    log(f"P2.53 pod {i}{' rehearse' if rehearse else ''} DONE, peak GPU {o.d['peak_gpu_gb']:.2f} GB")


# ---- score (no GPU) -------------------------------------------------------------------------------------------------


def num(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


R_OF, S_OF, GAIN_OF = (lambda x: num(x["r"])), (lambda x: num(x["S"]["S"])), (lambda x: num(x["gain"]))


def vals(st: dict[str, Any], pair: str, tasks: tuple[str, ...], seeds: tuple[int, ...], f: Any, arm: str = "own") -> list[float | None]:
    return [f(st[f"{pair}/{t}_s{s}"][arm]) for t in tasks for s in seeds]


def score(cfg: Config) -> dict[str, Any]:
    """Per pair and proxy: the scorecard's seed call (`realistic.seed_call`) of the built-reuse side against the other,
    read against the built direction "+" (`realistic.filed`); S's random control: the largest S of the pair's random
    changes against the smallest S of its fine-tunes; the prior one value a side (log10)."""
    st: dict[str, Any] = {}
    pri: dict[str, Any] = {}
    for i in range(len(cfg.pods)):
        d = json.loads((RESULTS / f"pod{i}.json").read_text())
        st |= d["states"]
        pri |= d["prior"]
    out: dict[str, Any] = {"pairs": {}}
    for pair, (a, b) in PAIRS.items():
        calls = {
            "r": R.seed_call("r", vals(st, pair, (a,), cfg.seeds, R_OF), vals(st, pair, (b,), cfg.seeds, R_OF)),
            "S": R.seed_call("S", vals(st, pair, (a,), cfg.seeds, S_OF), vals(st, pair, (b,), cfg.seeds, S_OF)),
            "prior": R.seed_call("prior", [math.log10(pri[a]["prior"])], [math.log10(pri[b]["prior"])]),
        }
        for c in calls.values():
            c["filed"] = R.filed(c, "+")
        own_S = [x for x in vals(st, pair, (a, b), cfg.seeds, S_OF) if x is not None]
        rnd_S = [x for x in vals(st, pair, (a, b), cfg.seeds, S_OF, "random") if x is not None]
        rnd_gain = [x for x in vals(st, pair, (a, b), cfg.seeds, GAIN_OF, "random") if x is not None]
        out["pairs"][pair] = {"calls": calls, "S_random_max": max(rnd_S, default=None), "S_own_min": min(own_S, default=None),
                              "S_random_below_all": bool(rnd_S and own_S and max(rnd_S) < min(own_S)), "random_gain_absmax": max(map(abs, rnd_gain), default=None)}  # fmt: skip
    out["orders"] = {p: [pair for pair, v in out["pairs"].items() if v["calls"][p]["filed"] == "hit"] for p in ("r", "S", "prior")}
    out["readable"] = {p: [pair for pair, v in out["pairs"].items() if v["calls"][p]["filed"] != "unreadable"] for p in ("r", "S", "prior")}
    write_stage(RESULTS / "table.json", jsonable(out))
    log(json.dumps(jsonable(out["orders"])))
    return out


def main(argv: list[str]) -> int:
    cfg, th = load_config()
    if argv[1:2] == ["score"]:
        score(cfg)
        return 0
    assert argv[1] == "pod" and len(argv) in (3, 4), __doc__
    require_filed(PRED)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E228 and E241
    log(f"P2.53 pod {argv[2]} start; settings {cfg}; thresholds {th}; card {torch.cuda.get_device_name()}")
    pod_job(cfg, th, int(argv[2]), argv[3:] == ["--rehearse"])
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main(sys.argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
