"""E252 (cell P2.55) run 1: the k-head label. Runs 1-2 switched off one head, (2,4); here M's top k heads by E62's
prefix score (k = 1, 2, 3, 5) are mean-ablated together (`copying.ablated`, E62's saved means) in M and in each saved
state F, and the share of F's gain lost is read on the state's own read rows with run 1's formula (`copying.share_lost`
via run 1's `share_of`). The `exclude` heads ((1,1), M's previous-token head, which feeds every task) are kept out of
the ranking and the null; one arm switches off (2,4) and them together ("pair"), to show that set is not copy-specific.
The null: for each k, k heads drawn from the rest outside the top 5 (every k-set when there are at most n_draws of
them, else n_draws distinct seeded sets). A null head is flagged "prefix" when its prefix score is at or over E62's
chance (the causal-uniform pattern's score); draws holding one are reported apart and kept out of the null band.
Nothing is trained.
Plan reports/plans/2026-11-21_plan_prior-mech-2x2.md (run 3); rules results/predictions_khead.json (refuses unless committed).

  control  M's copying (E52's second-copy top-1) with the top k off and with each null draw off; at k = 1 it must pass
           E62's control_a and repeat E62's 0.0095 (exit 1). Then the k = 1 share on every state runs 1-2 read, against
           its saved (2,4) share, within floor_mult x the floor (two batch sizes on floor_states, measured here; exit 1)
  items    E62's 6 states, E237's 15 key states, E250's 3 low-copy states and their 3 random updates  -> results/khead/<item>.json
  summary  task means per k, the null band per k, the filed rules, the 2x2 call at the best k          -> results/khead/summary.json

    sbatch --comment=E252 cluster/prior_mech_2x2_khead.sbatch
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)
from commute.open_pair import jsonable

T2: Any = load_experiment("prior_mech_2x2", "train")  # run 2: World, run 1's Reader with the low copy in its inventory
R1, E228 = T2.R1, T2.E228
RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT = RESULTS / "predictions_khead.json", RESULTS / "khead"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
STAGE_A = json.loads((R1.E62 / "run.json").read_text())["stage_a"]  # E62's prefix and previous-token scores, chance
NOT_VIA = ("e58/cipher", "e58/prior_cipher", "e58/rank1", "e58/elicit")
log = R1.log
Head = tuple[int, int]
Rows = dict[str, Tensor]


@dataclass(frozen=True)
class Config:  # filed in predictions_khead.json "settings"
    ks: tuple[int, ...]
    n_top: int  # the null is drawn outside M's top n_top heads
    n_draws: int
    draw_seed: int
    exclude: tuple[list[int], ...]  # kept out of the ranking and the null; switched off with (2,4) in the "pair" arm
    floor_batch: int  # the second batch size of the floor (run 1 read at batch 64)
    floor_states: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.ks == tuple(sorted(set(self.ks))) and {1, 3, 5} <= set(self.ks) and self.ks[-1] <= self.n_top, self.ks
        assert self.n_draws >= 10 and self.floor_batch != 64 and self.floor_states


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


class Heads:
    """E62's ranking of M's heads by prefix score with `exclude` left out, the top-k sets, the k-matched null, its
    flagged heads, and the pair arm ((2,4) with `exclude`)."""

    def __init__(self, cfg: Config, a: dict[str, Any] = STAGE_A) -> None:
        scores, ex = torch.tensor(a["scores"]), set[Head]((int(b), int(h)) for b, h in cfg.exclude)
        for h in ex:
            scores[h] = -math.inf  # last in the ranking, and dropped from the pool below
        top = set(CP.by_score(scores)[: cfg.n_top])
        self.pool = [h for h in CP.by_score(scores) if h not in top | ex]
        self.flag = {"prefix": {h for h in self.pool if float(scores[h]) >= a["chance"]}}
        self.top = {k: CP.top_k(scores, k) for k in cfg.ks}
        self.pair: list[Head] = sorted(set(self.top[1] or []) | ex)
        self.null = {k: CP.k_draws(self.pool, k, cfg.n_draws, cfg.draw_seed + k) for k in cfg.ks}
        assert self.top[1] == [tuple(R1.load_config()[0].head)], self.top[1]  # k = 1 is runs 1-2's head
        assert all(not set(d) & top for v in self.null.values() for d in v)  # no draw meets the top set, so none equals it

    def flags(self, d: list[Head]) -> list[str]:
        return [f for f, s in self.flag.items() if s & set(d)]

    def sets(self) -> dict[str, list[Head]]:
        out: dict[str, list[Head]] = {"on": [], "pair": self.pair} | {f"top{k}": h for k, h in self.top.items() if h is not None}
        return out | {f"null{k}_{i}": d for k, v in self.null.items() for i, d in enumerate(v)}


class Job:
    def __init__(self, cfg: Config) -> None:
        self.cfg, self.heads = cfg, Heads(cfg)
        self.w = T2.World(T2.load_config()[0])
        self.rd = self.w.rd
        self.rd.c["lowcopy"] = json.loads((T2.OUT / "b.json").read_text())["c"]  # as run 2's main
        self.m: dict[str, Rows] = {}

    def item(self, item: str) -> tuple[Any, dict[str, Tensor], str]:
        """Read rows, the update as read (saved times c, or run 2's random direction times c), the M cache key."""
        state = item.removeprefix("random/")
        r, dW = E228.state_read(state, self.rd.inv)[0], self.w.dw(item)
        self.w.dws.clear()  # one update in memory at a time
        return r, dW, self.rd.task(state)

    def rows(self, r: Any, dW: dict[str, Tensor] | None, sets: dict[str, list[Head]]) -> Rows:
        return {name: self.rd.lo(r, dW, hs) for name, hs in sets.items()}

    def of_M(self, task: str, r: Any) -> Rows:
        if task not in self.m:
            self.m[task] = self.rows(r, None, self.heads.sets())
        return self.m[task]


def top1(job: Job, heads: list[Head]) -> float:
    with CP.ablated(job.rd.net, heads, job.rd.means):
        return CP.induction(job.rd.net, job.rd.cfg.induction)["top1_second"]


def control_m(job: Job, th: dict[str, Any]) -> dict[str, Any]:
    """M's copying with the top k off, and with each null draw off; k = 1 must pass control_a and repeat E62."""
    H = job.heads
    out = {"top1_base": top1(job, []), "top1_top": {k: top1(job, h) if h else math.nan for k, h in H.top.items()},
           "top1_null": {k: [{"heads": d, "flags": H.flags(d), "top1": top1(job, d)} for d in v] for k, v in H.null.items()},
           "e62_curve": {k: STAGE_A["curve_top1_by_k"][k - 1] for k in H.top}, "flagged": {f: sorted(s) for f, s in H.flag.items()},
           "pair": H.pair, "top1_pair": top1(job, H.pair)}  # fmt: skip
    clean1 = [x["top1"] for x in out["top1_null"][1] if not x["flags"]]
    out["status"] = CP.control_a(out["top1_base"], out["top1_top"][1], clean1, th)
    out["k1_dev"] = abs(out["top1_top"][1] - STAGE_A["top1_H"])
    log(f"[control M] top-1 {out['top1_base']:.4f}; top k off {out['top1_top']} (E62's curve {out['e62_curve']}); pair {H.pair} off {out['top1_pair']:.4f}; clean k=1 null median "
        f"{statistics.median(clean1):.4f}; flagged {out['flagged']} -> {out['status']}, k=1 dev from E62 {out['k1_dev']:.4f}")  # fmt: skip
    if out["status"] != "pass" or not out["k1_dev"] <= th["top1_ref_tol"]:
        stop(OUT / "control.json", out, f"control M: {out['status']}, k=1 dev {out['k1_dev']}", log)
    return out


def saved_share(item: str) -> float:
    """Run 1's (2,4) share for its states, run 2's for the low copy."""
    f = f"{R1.fname(item)}.json"
    return json.loads(((T2.OUT / "head" / f) if "lowcopy" in item else (RESULTS / "stage1" / f)).read_text())["share_lost"]


def k1_share(job: Job, item: str, batch: int = 64) -> float:
    r, dW, _ = job.item(item)
    old, sets = job.rd.cfg, {"on": [], "top1": job.heads.top[1] or []}  # top[1] is (2,4), asserted in Heads
    job.rd.cfg = dataclasses.replace(old, batch=batch)
    try:
        return R1.share_of(job.rows(r, dW, sets), job.rows(r, None, sets), "top1")[2]
    finally:
        job.rd.cfg = old


def control_k1(job: Job, th: dict[str, Any], items: list[str], out: dict[str, Any]) -> None:
    """The k = 1 share repeats the saved (2,4) share on every state runs 1-2 read, within floor_mult x the floor: the
    largest gap between batch 64 and floor_batch on floor_states (a tie passes; a floor of 0 asks for exact equality)."""
    gaps = {s: abs(k1_share(job, s) - k1_share(job, s, job.cfg.floor_batch)) for s in job.cfg.floor_states}
    out["floor"], out["floor_gaps"] = max(gaps.values()), gaps
    out["k1_dev"] = {s: abs(k1_share(job, s) - saved_share(s)) for s in items if not s.startswith("random/")}
    worst = max(out["k1_dev"].values())
    log(f"[control k=1] floor {out['floor']:.3e} ({gaps}); largest |share - saved (2,4) share| {worst:.3e} over {len(out['k1_dev'])} states")
    if not worst <= th["floor_mult"] * out["floor"]:
        stop(OUT / "control.json", out, f"k=1 shares off the saved ones by {worst:.3e} > {th['floor_mult']} x floor {out['floor']:.3e}", log)
    write_stage(OUT / "control.json", jsonable(out | {"status": "pass"}))


def at_k(job: Job, f: Rows, m: Rows, k: int, th: dict[str, Any]) -> dict[str, Any]:
    if job.heads.top[k] is None:
        return {"heads": None, "share_lost": math.nan, "label": "tie", "null_clean_median": math.nan}
    G, G_off, share = R1.share_of(f, m, f"top{k}")
    null = [{"heads": d, "flags": job.heads.flags(d), "share": R1.share_of(f, m, f"null{k}_{i}")[2]} for i, d in enumerate(job.heads.null[k])]
    med = lambda xs: statistics.median(xs) if xs else math.nan
    fin = [x for x in null if CP.finite(x["share"])]
    sd = CP.share_sd(f["on"] - m["on"], f[f"top{k}"] - m[f"top{k}"], job.rd.cfg.n_boot, job.rd.cfg.boot_seed)
    return {"heads": job.heads.top[k], "G_off": G_off, "share_lost": share, "share_sd": sd, "label": CP.share_label(G, G_off, th), "null": null,
            "null_clean_median": med([x["share"] for x in fin if not x["flags"]]),
            "null_flagged_median": {fl: med([x["share"] for x in fin if fl in x["flags"]]) for fl in job.heads.flag}}  # fmt: skip


def read_item(job: Job, item: str, th: dict[str, Any]) -> dict[str, Any]:
    path = OUT / f"{R1.fname(item)}.json"
    if path.is_file():
        return json.loads(path.read_text())
    t0, (r, dW, task) = time.time(), job.item(item)
    m, f = job.of_M(task, r), job.rows(r, dW, job.heads.sets())
    rec = {"experiment": EXPERIMENT, "item": item, "task": ("random/" if item.startswith("random/") else "") + task, "c": job.rd.scale(item.removeprefix("random/")),
           "n_rows": len(r.toks), "G": float((f["on"] - m["on"]).mean()), "by_k": {k: at_k(job, f, m, k, th) for k in job.cfg.ks}}  # fmt: skip
    G, G_off, share = R1.share_of(f, m, "pair")
    rec["pair"] = {"heads": job.heads.pair, "G_off": G_off, "share_lost": share, "label": CP.share_label(G, G_off, th), "null_clean_median": math.nan}
    write_stage(path, jsonable(rec | {"seconds": time.time() - t0}))
    log(f"[{item}] G {rec['G']:.4f}; pair {job.heads.pair} share {share:.4f}; " + "; ".join(f"k={k} share {v['share_lost']:.4f} ({v['label']}), clean null median {v['null_clean_median']:.4f}, "
        f"flagged {v.get('null_flagged_median')}" for k, v in rec["by_k"].items()) + f"; {time.time() - t0:.0f} s")  # fmt: skip
    torch.cuda.empty_cache()
    return json.loads(path.read_text())  # the written form (string keys, nan as "nan"), as a rerun reads it


# ---- summary and the filed rules -------------------------------------------------------------------------------------


def view(recs: list[dict[str, Any]], k: int | None) -> dict[str, list[dict[str, Any]]]:
    """Per task, run 1's record shape at k (the pair arm for None): share_lost, label, the clean null median (as run
    1's random_median; nan for the pair)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        v = r["pair"] if k is None else r["by_k"][str(k)]
        by.setdefault(r["task"], []).append({"share_lost": T2.num(v["share_lost"]), "label": v["label"], "random_median": T2.num(v["null_clean_median"])})
    return by


def at_one_k(recs: list[dict[str, Any]], k: int, th: dict[str, Any]) -> dict[str, Any]:
    """Task means (nan unless every seed is readable), the null band nu (the largest |clean null median| over readable
    states), each corner's label as run 2's, and run 2's mechanism / prior call on the k-head share."""
    by = view(recs, k)
    mean = {t: R1.task_mean(v) for t, v in by.items()}
    meds = [abs(x["random_median"]) for t, v in by.items() if not t.startswith("random/") for x in v if x["label"] in R1.READABLE]
    nu = max(meds) if meds and all(math.isfinite(x) for x in meds) else math.nan  # a readable state with no clean draw: not read
    over = lambda t: all(x["share_lost"] > x["random_median"] for x in by[t])
    labels = {"copy": R1.verdict(mean["e58/copy"] >= th["share_new_max"] and over("e58/copy"), mean["e58/copy"]),
              "lowcopy": R1.verdict(mean["e58/lowcopy"] >= th["share_new_max"] and over("e58/lowcopy"), mean["e58/lowcopy"]),
              "ciphers": R1.verdict(max(mean["e58/cipher"], mean["e58/prior_cipher"]) <= th["share_new_max"], mean["e58/cipher"], mean["e58/prior_cipher"])}  # fmt: skip
    vals = {c: [x["share_lost"] for x in by[f"e58/{c}"]] for c in T2.CORNERS}
    eff = T2.effects(vals, max(max(v) - min(v) for v in vals.values()))
    eff["follows"] = eff["follows"] if all(v == "holds" for v in labels.values()) else "not_read"
    return {"task_mean_share": mean, "null_band": nu, "corners_labelled": labels, "two_by_two": eff}


def rules(ks: tuple[int, ...], per: dict[int, dict[str, Any]], recs: list[dict[str, Any]], th: dict[str, Any]) -> dict[str, Any]:
    m3, m5 = per[3]["task_mean_share"], per[5]["task_mean_share"]
    pair = {t: R1.task_mean(v) for t, v in view(recs, None).items()}
    ci = (pair["e58/cipher"], pair["e58/prior_cipher"])
    lo, hi = th["copy_k3_band"]
    e62 = {k: per[k]["task_mean_share"]["e62/copy"] for k in ks}
    best = min((k for k in ks if math.isfinite(e62[k])), key=lambda k: (-e62[k], k), default=None)  # ties: the smaller k
    rand = [r["G"] for r in recs if r["task"].startswith("random/")]
    return {"K1_copy_and_lowcopy_k3_in_band": R1.verdict(all(lo <= m3[t] <= hi for t in ("e58/copy", "e58/lowcopy")), m3["e58/copy"], m3["e58/lowcopy"]),
            "K2_e62_copy_k3_in_band": R1.verdict(th["e62_copy_k3_band"][0] <= m3["e62/copy"] <= th["e62_copy_k3_band"][1], m3["e62/copy"]),
            "K3_not_via_under_bar_every_k": R1.verdict(all(per[k]["task_mean_share"][t] <= th["not_via_max"] for k in ks for t in NOT_VIA),
                                                       *(per[k]["task_mean_share"][t] for k in ks for t in NOT_VIA)),
            "K4_null_band_under_bar_every_k": R1.verdict(all(per[k]["null_band"] <= th["null_max"] for k in ks), *(per[k]["null_band"] for k in ks)),
            "K5_copy_k5_not_under_stop": R1.verdict(m5["e58/copy"] >= th["copy_k5_min"], m5["e58/copy"]),
            "K6_random_updates_under_gain_gate": R1.verdict(all(g < th["G_min"] for g in rand), *rand),
            "pair_task_mean_share": pair, "K7_pair_costs_the_ciphers": R1.verdict(min(ci) > th["not_via_max"], *ci),
            "best_k": best, "two_by_two_at_best_k": per[best]["two_by_two"]["follows"] if best is not None else "not_read"}  # fmt: skip


def summary(cfg: Config, th: dict[str, Any], recs: list[dict[str, Any]]) -> None:
    per = {k: at_one_k(recs, k, th) for k in cfg.ks}
    out = {"experiment": EXPERIMENT, "by_k": per, "rules": rules(cfg.ks, per, recs, th)}
    write_stage(OUT / "summary.json", jsonable(out))
    for k, v in per.items():
        log(f"[summary k={k}] task means {v['task_mean_share']}; null band {v['null_band']:.4f}; corners {v['corners_labelled']}; 2x2 {v['two_by_two']}")
    log(f"[summary] rules {out['rules']}")


def main() -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as runs 1-2 read
    log(f"{EXPERIMENT} run 3 start; settings {asdict(cfg)}; thresholds {th}")
    OUT.mkdir(parents=True, exist_ok=True)
    job = Job(cfg)
    log(f"[heads] top-k {job.heads.top}; null pool {job.heads.pool}; draws per k {[len(v) for v in job.heads.null.values()]}")
    c1, c2 = job.rd.cfg, T2.load_config()[0]
    items = c1.e62_states() + c1.key_states() + c2.new_states() + [f"random/{s}" for s in c2.new_states()]
    control_k1(job, th, items, control_m(job, th))
    summary(cfg, th, [read_item(job, it, th) for it in items])
    log(f"{EXPERIMENT} run 3 DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
