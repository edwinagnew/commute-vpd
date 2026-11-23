"""E### (cell P2.64; user via writeup-lead, 2026-11-22, aim (a): "if we can detect reuse/new on realistic fine-tunes"):
the write share E_b over the needed set K of the frozen definition (theory/2026-11-22_definition_frozen.md, section 2)
on the saved 67M real-text fine-tunes of P2.38 run 2 (E211), each beside its matched random update (E246's draw at the
fine-tune's per-matrix norms). Descriptive: read all at once, no verdict. Plan
reports/plans/2026-11-23_plan_realtext-eb-over-k.md; settings and filed expected values in results/predictions.json,
which the run refuses unless committed unchanged.

The pairs, the side built to need what M holds first (E211's tasks, experiments/realistic_tasks/run.py and run2.py):
  label      L1 topic label (AG News)  vs L3 seeded coin label, the same 512 news windows of 35 tokens, size 0.01
  lowercase  T1 Pile text lowercased    vs T4 each sentence's words reversed, 500 windows each,  size 0.0175
  language   T2 French Wikipedia        vs T3 Yoruba Wikipedia,                500 windows each,  size 0.0175
D is the task's read rows as E211 and E246 read them (E212's copy, experiments/realistic_tasks/results/large/run3/
texts.pt), cut after the read position (label: the last of 35; text: 63 of 127, so D's windows are 64 tokens), and the
designated output is F's argmax token at that position. Per item: F = M + the saved delta-W (sha256 checked against
E211's record) or E246's Gaussian draw (its size checked against E246's record); the featurization of
`commute.kstar_vpd`; one write pass (`commute.stand_in.write_change`); K by `commute.needed.needed_set` (E268's
sequence: derangements x examples pairs, every subcomponent and remainder of the 24 matrices); E_b over K =
`stand_in.explained_share` over K and changed (1 when that set is empty: flagged degenerate). results/items/<item>.json
holds the readings; results/large/items_<item>.npz num, den, dd, changed, in_K, first_flip, flips. A rerun skips
written items. The cheap readings (E211's share, E246's r, S and M's prior) are not recomputed: `summary` copies them.

Stages; each GPU job runs stage 0 first (skipped once results/stage0_<tag>.json says done); any miss stops the job:
    0        on L1's D and T1's D: (a) the do-nothing (F = M): no changed variable, every dd exactly 0, E_b over K
             exactly 1; (b) the identity interchange (x' = x) on the task's seed-0 fine-tune: no argmax change; its
             largest logit deviation is the floor for fragile positions, its rate the time estimate per item
    items    the 36 items (18 fine-tunes, 18 matched randoms); --shard I N takes every N-th fine-tune from the I-th
             with its random; --only ITEM ... reads just those (the rehearsal)
    summary  CPU: every item JSON with the saved cheap readings into results/summary.json

    python experiments/realtext_eb/run.py --stage items --only L1_s0 T1_s0 --tag rh    (rehearsal, cluster debug)
    PRED_SHA=<sha> python experiments/realtext_eb/run.py --stage items --shard 0 4 --tag p0    (a pod)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
from torch import Tensor

from commute import finetune as FT
from commute import kstar_vpd as KV
from commute import needed as IX
from commute import old_tasks as OT
from commute import realistic as RL
from commute import rule_retention as R
from commute import stand_in as SI
from commute import vpd
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    require_filed,
    write_stage,
)
from commute.steering import Net

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", EXPERIMENTS.parent
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E211 = EXPERIMENTS / "realistic_tasks" / "results"
E246 = EXPERIMENTS / "realtext_proxies" / "results"
TEXTS = E211 / "large" / "run3" / "texts.pt"
PAIRS = {"label": ("L1", "L3"), "lowercase": ("T1", "T4"), "language": ("T2", "T3")}  # built-reuse side first
SIZE = {"label": "0.01", "lowercase": "0.0175", "language": "0.0175"}  # the scored size, as E246
PAIR_OF = {t: p for p, ts in PAIRS.items() for t in ts}
LAST = slice(-1, None)
CHECKS = ("split_fp64", "F_from_pieces", "M_from_pieces")  # E226's per-matrix featurization guards


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stop(path: Path, out: dict[str, Any], why: str) -> NoReturn:
    write_stage(path, out | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    seeds: tuple[int, ...]
    label_pos: int  # the label rows' read position (their last)
    text_pos: int  # the text windows' read position; D is each window cut after it
    derangements: int  # P
    examples: int  # windows tested per derangement
    seed: int  # the derangements' seed
    rows: int  # cases per forward with one read position
    logit_rows: int  # cases x read positions per forward, at most
    cache_batch: int  # windows per batch of F's cached run and of the write pass
    ident_vars: int  # subcomponents per matrix in the identity control, beside its remainder
    split_tol: float
    split_fp64_tol: float
    F_from_pieces_tol: float
    M_from_pieces_tol: float
    random_size_rel_tol: float

    def __post_init__(self) -> None:
        assert self.seeds == (0, 1, 2) and self.label_pos > 0 and self.text_pos > 0
        assert self.derangements >= 1 and self.examples >= 1 and self.rows >= 1 and self.logit_rows >= self.rows and self.ident_vars >= 1


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


# ---- items and their saved inputs -----------------------------------------------------------------------------------


def states(cfg: Config) -> list[str]:
    return [f"{t}_s{s}" for ts in PAIRS.values() for t in ts for s in cfg.seeds]


def parse(item: str) -> tuple[str, int, bool]:
    """'T2_s1_random' -> ('T2', 1, True)."""
    task, seed, *rest = item.split("_")
    assert task in PAIR_OF and seed[0] == "s" and rest in ([], ["random"]), item
    return task, int(seed[1:]), rest == ["random"]


def e246() -> dict[str, Any]:
    """E246's records of the three pairs: {"states": {"<pair>/<task>_s<seed>": ...}, "prior": {task: ...}}."""
    out: dict[str, Any] = {"states": {}, "prior": {}}
    for i in (0, 1):
        d = json.loads((E246 / f"pod{i}.json").read_text())
        out["states"] |= {k: v for k, v in d["states"].items() if k.split("/")[0] in PAIRS}
        out["prior"] |= {k: v for k, v in d["prior"].items() if k in PAIR_OF}
    return out


def e211_state(task: str, seed: int) -> dict[str, Any]:
    return json.loads((E211 / f"run2_{task}.json").read_text())["states"][f"s{seed}_{SIZE[PAIR_OF[task]]}"]


def update(item: str, rec246: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """The item's delta-W keyed by module (fp32, CPU) and its record: the saved E211 state, or E246's Gaussian draw at
    that state's per-matrix norms (`finetune.random_direction`, E246's seed)."""
    task, seed, rnd = parse(item)
    st = e211_state(task, seed)
    raw = (REPO / st["file"]).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    dW = {k: v.float() for k, v in torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu").items()}
    facts: dict[str, Any] = {"file": st["file"], "sha256": sha, "sha256_e211": st["sha256"]}
    if rnd:
        r = rec246["states"][f"{PAIR_OF[task]}/{task}_s{seed}"]["random"]
        dW = FT.random_direction(dW, r["draw_seed"])
        facts |= {"draw_seed": r["draw_seed"], "size_e246": r["size"]}
    return dW, facts


class World:
    """M (the 67M at W0), the decomposition, the split's bases, D per task and E246's records."""

    def __init__(self, cfg: Config) -> None:
        self.cfg, self.net = cfg, Net("cuda")
        self.net.net.requires_grad_(False)
        self.M = R.Model(self.net.net)
        self.uv, self.delta = vpd.component_weights(self.net.model), vpd.weight_deltas(self.net.model)
        self.bases = KV.bases(self.uv, cfg.split_tol)
        self.texts = torch.load(TEXTS, weights_only=False)["reads"]
        self.rec246 = e246()

    def windows(self, task: str) -> Tensor:
        """D: the task's read rows cut after the read position, which must be the one E246 read."""
        pos = self.cfg.label_pos if task.startswith("L") else self.cfg.text_pos
        w = self.texts[task].long()
        assert w.shape[1] > pos and pos == self.rec246["prior"][task]["pos"], (task, tuple(w.shape), pos)
        return w[:, : pos + 1].cuda()

    def model(self, dW: dict[str, Tensor]) -> tuple[R.Model, KV.Feat]:
        assert sorted(dW) == sorted(self.net.W0), "delta-W must cover the 24 decomposed matrices"
        F = R.Model(OT.with_delta(self.net, {f"{k}.weight": v for k, v in dW.items()}).net)
        return F, KV.featurize(self.M, F, self.uv, self.delta, self.bases)


def changed_mask(feat: KV.Feat) -> Tensor:
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return ch


def guards(w: World, item: str, dW: dict[str, Tensor], facts: dict[str, Any], feat: KV.Feat) -> tuple[dict[str, Any], list[str]]:
    """The update's size and the featurization's worst per-matrix checks, and every miss: a state not E211's saved
    file, a random not E246's draw, a featurization outside E226's tolerances."""
    cfg, size = w.cfg, RL.update_size(dW, w.net.W0)
    worst = {k: max(c[k] for c in feat.checks.values()) for k in CHECKS}
    miss = [f"{item}: sha256 {facts['sha256']} is not E211's {facts['sha256_e211']}"] if facts["sha256"] != facts["sha256_e211"] else []
    if "size_e246" in facts and abs(size - facts["size_e246"]) > cfg.random_size_rel_tol * facts["size_e246"]:
        miss.append(f"{item}: size {size!r} is not E246's draw's {facts['size_e246']!r}")
    miss += [f"{item}: {k} {v:.3g} over {getattr(cfg, k + '_tol')}" for k, v in worst.items() if v > getattr(cfg, k + "_tol")]
    return {"size": size, "featurization": worst}, miss


# ---- one item -------------------------------------------------------------------------------------------------------


def read_item(w: World, item: str, floor: float) -> None:
    path, cfg = RESULTS / "items" / f"{item}.json", w.cfg
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{item}: done before, skipped")
        return
    t0, (task, seed, rnd) = time.time(), parse(item)
    dW, facts = update(item, w.rec246)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "task": task, "pair": PAIR_OF[task], "seed": seed, "random": rnd} | facts
    F, feat = w.model(dW)
    g, miss = guards(w, item, dW, facts, feat)
    out |= g
    if miss:
        stop(path, out, "; ".join(miss))
    toks, ch = w.windows(task), changed_mask(feat)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.cache_batch)))
    log(f"{item}: {len(toks)} windows of {toks.shape[1]} tokens, {feat.n} variables, {int(ch.sum())} changed, size {g['size']:.6f}")
    nd = IX.needed_set(F, feat, toks, LAST, cfg.derangements, cfg.examples, cfg.seed, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
    K = nd.K
    out |= {"n_windows": len(toks), "window_len": toks.shape[1], "n_vars": feat.n, "n_changed": int(ch.sum()), "n_pairs": nd.n_pairs,
            "derangements": cfg.derangements, "examples": nd.m, "min_margin": float(nd.run.margin.min()),
            "n_fragile_positions": int((nd.run.margin <= floor).sum()), "fragile_floor": floor,
            "E_b_K": SI.explained_share(num, den, dd, K & ch), "size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()),
            "degenerate": not bool((K & ch).any()),
            "by_derangement": [{"size_K": int((K & (nd.first_flip <= p)).sum()),
                                "E_b_K": SI.explained_share(num, den, dd, K & (nd.first_flip <= p) & ch)} for p in range(cfg.derangements)]}  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"items_{item}.npz", num=num.numpy(), den=den.numpy(), dd=dd.numpy(), changed=ch.numpy(),
                        in_K=K.numpy(), first_flip=nd.first_flip.numpy(), flips=nd.scan.flips.numpy())  # fmt: skip
    out |= {"seconds": time.time() - t0, "cases_per_s": nd.scan.rows / nd.scan.seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    write_stage(path, out)
    log(f"{item}: E_b over K {out['E_b_K']:.6f}{' (degenerate: K has no changed member)' if out['degenerate'] else ''}; |K| {out['size_K']} "
        f"({out['n_changed_K']} changed); {out['n_fragile_positions']} fragile positions; {out['seconds']:.0f} s, peak GPU {out['peak_gpu_gb']:.1f} GB")  # fmt: skip
    del F, nd
    torch.cuda.empty_cache()


# ---- stage 0, summary, main -----------------------------------------------------------------------------------------


def stage0_task(w: World, task: str) -> dict[str, Any]:
    cfg, toks = w.cfg, w.windows(task)
    F, feat = w.model({k: torch.zeros_like(v).cpu() for k, v in w.net.W0.items()})
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.cache_batch)))
    dn = {"n_changed": len(feat.changed), "dd_max": float(dd.abs().max()),
          "E_b_K": SI.explained_share(num, den, dd, torch.ones(feat.n, dtype=torch.bool) & changed_mask(feat))}  # any K: K and changed is empty
    del F
    item = f"{task}_s0"
    dW, facts = update(item, w.rec246)
    F, feat = w.model(dW)
    _, miss = guards(w, item, dW, facts, feat)
    idn = IX.identity_scan(F, feat, toks, LAST, cfg.examples, cfg.ident_vars, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
    del F
    torch.cuda.empty_cache()
    miss += [f"{task} identity: {idn['flips']} argmax changes"] if idn["flips"] else []
    miss += [f"{task} do-nothing: {k} {dn[k]}" for k, want in (("n_changed", 0), ("dd_max", 0.0), ("E_b_K", 1.0)) if dn[k] != want]
    est = idn["n_vars"] * cfg.derangements * min(cfg.examples, len(toks)) / idn["cases_per_s"]
    return {"do_nothing": dn, "identity": idn, "est_seconds_per_item": est, "miss": miss}


def stage0(w: World, tag: str) -> float:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return float(done["floor"])
    t0 = time.time()
    out: dict[str, Any] = {"experiment": EXPERIMENT, "card": torch.cuda.get_device_name()} | {t: stage0_task(w, t) for t in ("L1", "T1")}
    miss = out["L1"]["miss"] + out["T1"]["miss"]
    out |= {"floor": max(out[t]["identity"]["max_dev"] for t in ("L1", "T1")), "seconds": time.time() - t0}
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    write_stage(path, out | {"done": True})
    log(f"STAGE 0 DONE: est. {out['L1']['est_seconds_per_item']:.0f} s per label item, {out['T1']['est_seconds_per_item']:.0f} s per text item")
    return float(out["floor"])


def summary() -> None:
    """Every item JSON, each with the saved cheap readings of its update: E246's r and S (own or random) and E211's
    share in log-odds (fine-tunes only), and M's prior per task (E246's geometric mean of P(scored token) at W0)."""
    rec = e246()
    every = {}
    for p in sorted((RESULTS / "items").glob("*.json")):
        r = json.loads(p.read_text())
        arm = rec["states"][f"{r['pair']}/{r['task']}_s{r['seed']}"]["random" if r["random"] else "own"]
        share = None if r["random"] else e211_state(r["task"], r["seed"])["share"]["share"]["lo"]
        every[r["item"]] = r | {"saved": {"r_e246": arm["r"], "S_e246": arm["S"]["S"], "share_e211": share}}
    s0 = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))]
    write_stage(RESULTS / "summary.json", {"experiment": EXPERIMENT, "stage0": s0, "prior": {t: v["prior"] for t, v in rec["prior"].items()}, "items": every})
    log(f"summary: {len(every)} items, {len(s0)} stage-0 files")


def main(stage: str, shard: tuple[int, int], only: list[str], tag: str) -> int:
    if stage == "summary":
        summary()
        return 0
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226, E257, E268
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, only {only}, tag {tag}; settings {asdict(cfg)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    w = World(cfg)
    floor = stage0(w, tag)
    if stage == "0":
        return 0
    every = [i for s in states(cfg) for i in (s, f"{s}_random")]
    todo = [i for s in states(cfg)[shard[0] :: shard[1]] for i in (s, f"{s}_random")] if not only else only
    assert set(todo) <= set(every), sorted(set(todo) - set(every))
    log(f"stage items: {len(todo)} of {len(every)} items {todo}")
    for k, item in enumerate(todo):
        read_item(w, item, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "items", "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th fine-tune from the I-th, each with its random")
    ap.add_argument("--only", nargs="+", default=[], help="read just these items (the rehearsal)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    if args.stage != "summary":
        require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.only, args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
