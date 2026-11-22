"""E253 (cell P2.55) run 2: the low-prior copy built as its own task. E62's copy map ("A B ... A M" -> B, read at the
marker M; B drawn per window, train and read B pools disjoint, so only copying from the context solves it), with B
drawn from rarer tokens than E62's so that M's prior on the answer sits near the cipher's (log10 -5.86), not E62's
-3.50. Three filed B bands of frequency ranks are tried in order; on each, the windows of each split are ordered by M's
P(B) and the longest lowest-first run whose geometric mean is at or under the cipher's prior is kept (E250's
`lowest_run`, `prior_mech_2x2/train.py`; a selection inside this task only); the first band that clears the construction check is
used. Fine-tuned with E62's call (`keydep.finetune`, step 30, seeds 0-2), scaled to E237's common size; beside it one
Gaussian random update per seed at its per-matrix norms times the same c. Read with the earlier runs' readers, by
import: run 1's (2,4) share (`prior_mech_2x2/run.py`), the k-head arms (`khead.py`), r, size, erank, neuron and
steering shares (`train.py`'s World over holes_fill's Job; per-key steering keyed on A, the token before the read).
Plan reports/plans/2026-11-21_plan_cued-copy.md; rules results/predictions.json (refuses unless committed).

  0  controls, alone: run 1's control (M's copying falls with (2,4) off; the zero update reads 0), the prior path and
     holes_fill's readers repeat their saved readings (run 2's `controls`)                      -> results/control.json
  a  the bands, the selection, the construction check (exit 1 on a miss); the zero update on the new rows reads 0
                                                                                                -> results/a.json
  b  three fine-tunes; E12's rule at step 100 (exit 1 on a miss); sizes, c                      -> results/b.json
  c  per new state and random update: (2,4) share, k-head arms, r, size, erank, neuron and steering shares, the gain on
     B-absent windows                                                                           -> results/items/<item>.json
  table  the 2x2 with the cued copy, the row-selected low copy beside it, the filed rules      -> results/table.json

    sbatch --comment=E253 cluster/cued_copy.sbatch
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import torch
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import fair_steering as FS
from commute import finetune as FT
from commute import keydep as KD
from commute import steering as ST
from commute.data import LAST, Windows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)
from commute.open_pair import jsonable

T2: Any = load_experiment("prior_mech_2x2", "train")  # run 2: World (run 1's Reader, holes_fill's Job), lowest_run, the table
KH: Any = load_experiment("prior_mech_2x2", "khead")  # the k-head label: Heads, Job, read_item
R1, HF, E228 = T2.R1, T2.HF, T2.E228
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, LARGE, ITEMS = RESULTS / "predictions.json", RESULTS / "large", RESULTS / "items"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
TASK = "e58/cuedcopy"  # E147's naming, so run 1's Reader scales it by c["cuedcopy"]
WIN_FILE = LARGE / "windows.pt"
N = 2 * AN.HALF + 1  # the window the model sees; the target is a last column (data.LAST)
log = R1.log


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    seeds: tuple[int, ...]
    step: int  # the read step (E237's)
    learn_step: int  # E12's rule is read here
    n_keys: int
    key_ranks: tuple[int, ...]
    key_seed: int
    bands: tuple[list[int], ...]  # B's frequency ranks, tried in this order
    heldout_every: int
    n_train_pool: int
    n_read_pool: int
    window_seed: int
    select_bar: float  # the kept rows' geometric-mean P(B), at or under (the cipher's prior)
    prior_bar: float  # the construction check on the read rows
    n_train_cap: int
    n_read_cap: int
    n_train_min: int
    n_read_min: int
    n_keys_min: int
    min_train_per_key: int
    random_seed: int
    absent_seed: int

    def __post_init__(self) -> None:
        assert self.seeds == (0, 1, 2) and self.step == 30 and self.learn_step == 100 and self.step in KD.E12_CHECKS
        assert self.select_bar <= self.prior_bar < 1 and self.n_train_min <= self.n_train_cap and self.n_read_min <= self.n_read_cap
        assert all(lo < hi for lo, hi in self.bands) and len(self.key_ranks) == 2 and self.n_keys_min <= self.n_keys

    def states(self) -> list[str]:
        return [f"{TASK}_s{s}" for s in self.seeds]

    def items(self) -> list[str]:
        return self.states() + [f"random/{s}" for s in self.states()]


def load() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) and k != "bands" else v for k, v in pred["settings"].items()}
    s["bands"] = tuple(pred["settings"]["bands"])
    return config_from_settings(Config, s), pred["thresholds"]


def read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return str(p.relative_to(EXPERIMENTS.parent))


class CueTask(HF.Task):
    """holes_fill's Task with per-key steering keyed on A (the token before the read position), not on the marker."""

    def idx(self, toks: Tensor) -> Tensor:
        return FS.key_index(toks[:, self.read.half - 1], self.pool)


class World(T2.World):
    """Run 2's world with the cued copy's task and states in the inventory."""

    def __init__(self, cc: Config) -> None:
        super().__init__(T2.load_config()[0])
        self.cc, inv = cc, self.rd.inv
        inv["tasks"][TASK] = {"format": "windows_pt", "path": rel(WIN_FILE), "arm": "copy", "batch": 64}
        for st in cc.states():
            inv["states"][st] = {"task": TASK, "file": rel(LARGE / f"dW_{st.split('_')[-1]}.pt")}
            inv["states"][f"random/{st}"] = {"task": TASK, "file": None}

    def dw(self, item: str) -> dict[str, Tensor]:
        if TASK not in item:
            return super().dw(item)
        if item not in self.dws:
            state = item.removeprefix("random/")
            d = {k: v.float() for k, v in torch.load(EXPERIMENTS.parent / self.rd.inv["states"][state]["file"], weights_only=False).items()}
            if item.startswith("random/"):
                d = FT.random_direction(d, self.cc.random_seed + self.cc.states().index(state))
            self.dws[item] = {k: v * self.rd.scale(state) for k, v in d.items()}
        return self.dws[item]


class KJob(KH.Job):
    """khead's Job on this world (its own __init__ would build run 2's world a second time)."""

    def __init__(self, w: World) -> None:
        self.cfg, self.w, self.rd, self.m = KH.load_config()[0], w, w.rd, {}
        self.heads = KH.Heads(self.cfg)


# ---- stage a: the bands, the selection, the construction check -------------------------------------------------------


def lp_B(w: World, win: Windows, b: Tensor, dW: dict[str, Tensor] | None = None, readout: Any = ST.logp_at) -> Tensor:
    return CP.read_lo(w.net, CP.with_target(win, b).tokens, AN.HALF, N, dW, readout)


def geo(lp: Tensor) -> float:
    return math.exp(float(lp.mean())) if len(lp) else math.nan


def keys_of(cc: Config, ranked: list[int], marker: int) -> list[int]:
    pool = [t for t in ranked[cc.key_ranks[0] : cc.key_ranks[1]] if t != marker]
    return [pool[int(i)] for i in torch.randperm(len(pool), generator=torch.Generator().manual_seed(cc.key_seed))[: cc.n_keys]]


def band(w: World, rows: Tensor, marker: int, keys: list[int], bs: list[int]) -> dict[str, Any]:
    """One band's pools (E62's `cue_windows`, rows split as E62's), M's log P(B) on each, the selection, the check."""
    cc, held = w.cc, torch.arange(rows.shape[0]) % w.cc.heldout_every == 0
    tr, b_tr, _ = CP.cue_windows(rows[~held], marker, keys, bs[::2], AN.HALF, cc.n_train_pool, cc.window_seed)
    rd, b_rd, p_rd = CP.cue_windows(rows[held], marker, keys, bs[1::2], AN.HALF, cc.n_read_pool, cc.window_seed + 2)
    lp_tr, lp_rd = lp_B(w, tr, b_tr), lp_B(w, rd, b_rd)
    m_tr, m_rd = T2.lowest_run(lp_tr, cc.select_bar, cc.n_train_cap), T2.lowest_run(lp_rd, cc.select_bar, cc.n_read_cap)
    a_tr, a_rd = tr.tokens[:, AN.HALF - 1], rd.tokens[:, AN.HALF - 1]
    fed = [k for k in keys if int((a_tr[m_tr] == k).sum()) >= cc.min_train_per_key]  # every read key gets a train window
    m_rd = m_rd & torch.isin(a_rd, torch.tensor(fed or [-1]))
    rec = {"n_b": len(bs), "geo_pool": {"train": geo(lp_tr), "read": geo(lp_rd)}, "n_kept": {"train": int(m_tr.sum()), "read": int(m_rd.sum())},
           "geo_kept": {"train": geo(lp_tr[m_tr]), "read": geo(lp_rd[m_rd])}, "n_keys_read": int(a_rd[m_rd].unique().numel()), "n_keys_fed": len(fed)}  # fmt: skip
    rec["pass"] = (rec["n_kept"]["train"] >= cc.n_train_min and rec["n_kept"]["read"] >= cc.n_read_min and rec["n_keys_read"] >= cc.n_keys_min
                   and rec["geo_kept"]["read"] <= cc.prior_bar)  # fmt: skip  (nan, an empty pool, fails)
    rec["windows"] = {"train": tr.subset(m_tr), "read": rd.subset(m_rd), "target": {"copy": {"train": b_tr[m_tr], "read": b_rd[m_rd]}},
                      "p_read": p_rd[m_rd], "bs_read": bs[1::2], "keys": fed, "marker": marker}  # fmt: skip
    return rec


def zero_check(w: World, win: dict[str, Any]) -> float:
    """The zero update on the new read rows: |F - M| of the target's log-odds, must be 0 exactly."""
    _, d = E228.state_read(w.rd.cfg.key_states()[0], w.rd.inv)
    zero = {k: torch.zeros_like(v) for k, v in d.items()}
    m = lp_B(w, win["read"], win["target"]["copy"]["read"], None, ST.logodds_at)
    return float((lp_B(w, win["read"], win["target"]["copy"]["read"], zero, ST.logodds_at) - m).abs().max())


def stage_a(w: World) -> None:
    path = RESULTS / "a.json"
    if path.is_file() and "stopped" not in read_json(path):
        return log("stage a done before")
    rows, _, marker, ranked, _ = KD.vocab(KD.Ctx(w.net, {}, {}, {}, LARGE))  # every ranked token is intact (CPU check, plan)
    keys = keys_of(w.cc, ranked, marker)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "marker": marker, "keys": keys, "bands": {}}
    for lo, hi in w.cc.bands:
        rec = band(w, rows, marker, keys, [t for t in ranked[lo:hi] if t not in keys and t != marker])
        win = rec.pop("windows")
        out["bands"][f"{lo}-{hi}"] = rec
        log(f"[a band {lo}-{hi}] {json.dumps(rec)}")
        if rec["pass"]:
            LARGE.mkdir(parents=True, exist_ok=True)
            torch.save(win | {"band": [lo, hi]}, WIN_FILE)
            out |= {"band": [lo, hi], "prior": rec["geo_kept"]["read"], "log10_prior": math.log10(rec["geo_kept"]["read"]),
                    "zero_max_abs": zero_check(w, win), "windows_sha256": KD.sha256(WIN_FILE)}  # fmt: skip
            log(f"[a] band {lo}-{hi}: read prior {out['prior']:.4e} (log10 {out['log10_prior']:.3f}); zero update max |F - M| {out['zero_max_abs']}")
            if out["zero_max_abs"] != 0:
                stop(path, out, f"zero update reads {out['zero_max_abs']} on the new rows", log)
            return write_stage(path, jsonable(out))
    stop(path, out, "construction: no band clears the check", log)


# ---- stage b: the fine-tunes ---------------------------------------------------------------------------------------


def stage_b(w: World, th: dict[str, Any]) -> None:
    path, cc = RESULTS / "b.json", w.cc
    out: dict[str, Any] = read_json(path) if path.is_file() else {"experiment": EXPERIMENT, "states": {}}
    win = torch.load(WIN_FILE, weights_only=False)
    t = win["target"]["copy"]
    data = {"table": LAST, "train": CP.with_target(win["train"], t["train"]), "heldout": CP.with_target(win["read"], t["read"]), "ci_len": N}
    for s, st in zip(cc.seeds, cc.states(), strict=True):
        if st in out["states"]:
            continue
        ft = KD.finetune(KD.Ctx(w.net, {}, {}, {}, LARGE), data, s, (cc.step, cc.learn_step))
        d, f = ft.deltas[cc.step], EXPERIMENTS.parent / w.rd.inv["states"][st]["file"]
        torch.save({k: v.bfloat16() for k, v in d.items()}, f)
        size = T2.E205.sizes({f"{k}.weight": v.bfloat16().float() for k, v in d.items()}, w.net.W0)["whole"]
        out["states"][st] = {"p_heldout": {k: ft.evals[k]["p_o"] for k in (cc.step, cc.learn_step)}, "size": size, "sha256": KD.sha256(f)}
        log(f"[b {st}] held-out P(B) {out['states'][st]['p_heldout']}; size {size:.6g}")
        write_stage(path, jsonable(out))
        if not out["states"][st]["p_heldout"][cc.learn_step] >= th["p_learn"]:
            stop(path, out, f"{st} does not learn: P(B) {out['states'][st]['p_heldout'][cc.learn_step]:.4f} at step {cc.learn_step}", log)
    out["c"] = read_json(R1.E237_C.parent / "stage0.json")["target"] / fmean(v["size"] for v in out["states"].values())
    log(f"[b] c {out['c']:.4f}")
    write_stage(path, jsonable(out | {"done": True}))


# ---- stage c: the readings -----------------------------------------------------------------------------------------


def install_task(w: World) -> None:
    """holes_fill's Task for the new rows, its train windows (target last) and its pool of A keys."""
    win, base = torch.load(WIN_FILE, weights_only=False), HF.load_task(TASK, w.rd.inv["tasks"][TASK])
    train = CP.with_target(win["train"], win["target"]["copy"]["train"]).tokens
    pool = torch.tensor(sorted(win["keys"]))
    assert bool(torch.isin(base.read.toks[:, AN.HALF - 1], pool).all())  # by construction (band's `fed`)
    t = CueTask(base.read, train, pool)
    t.lo0 = HF.lo_at(w.net, t, t.read.toks)
    w.hf.tasks[TASK] = t


def absent_gain(w: World, item: str) -> float:
    """The update's mean log-odds gain for B on read windows with B overwritten (E62's `absent_b`): ~0 if it copies."""
    win = torch.load(WIN_FILE, weights_only=False)
    b = win["target"]["copy"]["read"]
    toks = CP.absent_b(win["read"].tokens, win["p_read"], b, win["bs_read"], w.cc.absent_seed)
    ab = Windows(toks, win["read"].g, win["read"].row, win["read"].pos, AN.HALF)
    return float((lp_B(w, ab, b, w.dw(item), ST.logodds_at) - lp_B(w, ab, b, None, ST.logodds_at)).mean())


def stage_c(w: World, th1: dict[str, Any], thk: dict[str, Any]) -> None:
    R1.OUT, KH.OUT = ITEMS / "head", ITEMS / "khead"  # the imported readers write their per-item files here
    for d in (R1.OUT, KH.OUT):
        d.mkdir(parents=True, exist_ok=True)
    install_task(w)
    job = KJob(w)
    for i, item in enumerate(w.cc.items()):
        if (path := ITEMS / f"{R1.fname(item)}.json").is_file():
            continue
        t0, rec = time.time(), {"experiment": EXPERIMENT, "item": item}
        if not item.startswith("random/"):
            rec["head"] = R1.read_state(w.rd, item, th1)
        rec["khead"] = KH.read_item(job, item, thk)
        rec |= T2.shares(w, item, i) | T2.weights(w, item) | {"absent_gain": absent_gain(w, item)}
        write_stage(path, jsonable(rec | {"seconds": time.time() - t0}))
        log(f"[c {item}] " + json.dumps(jsonable({k: v for k, v in rec.items() if k not in ("head", "khead", "experiment")})) + f"; {time.time() - t0:.0f} s")
        torch.cuda.empty_cache()


# ---- the table and the filed rules -----------------------------------------------------------------------------------

PROXIES = ("k3_share", "pair_share", *T2.PROXIES)
SLOT = "lowcopy"  # run 2's `effects` names the low-prior reused corner so; here the cued copy fills it


def khead_of(f: Path) -> dict[str, float]:
    k = read_json(f) if f.is_file() else {"by_k": {"3": {}}, "pair": {}}
    return {"k3_share": T2.num(k["by_k"]["3"].get("share_lost")), "pair_share": T2.num(k["pair"].get("share_lost"))}


def cued_row(state: str, a: dict[str, Any]) -> dict[str, float]:
    c = read_json(ITEMS / f"{R1.fname(state)}.json")
    v = {"head_share": c["head"]["share_lost"], "log10_prior": a["log10_prior"], **{k: c[k] for k in ("r", "size", "erank")}}
    v |= {f"steering_{k}": x for k, x in c["steering"].items()} | {f"neuron_{k}": x for k, x in c["neuron"].items()}
    return {p: T2.num(v.get(p)) for p in T2.PROXIES} | khead_of(ITEMS / "khead" / f"{R1.fname(state)}.json")


def rows_of(cc: Config) -> dict[str, list[dict[str, float]]]:
    """Seed rows per corner: the cued copy from this run; the key corners and the row-selected low copy from run 2's
    table rows (`train.state_row`), with the k-head arms from khead's files when run 3 has run (nan otherwise)."""
    a, a2, kd = read_json(RESULTS / "a.json"), read_json(T2.OUT / "a.json"), T2.RESULTS / "khead"
    out = {"cuedcopy": [cued_row(s, a) for s in cc.states()]}
    for t in ("copy", "prior_cipher", "cipher", "lowcopy"):
        ss = [f"{T2.TASK}_s{s}" if t == "lowcopy" else R1.R96.state_name(t, s) for s in cc.seeds]
        out[t] = [T2.state_row(t, s, a2) | khead_of(kd / f"{R1.fname(s)}.json") for s in ss]
    return out


def label(cc: Config, th: dict[str, Any]) -> dict[str, Any]:
    """Q0: the cued copy runs through the copy heads: (2,4) mean >= share_new_max with every seed over its median random
    head (run 1's verdict), and the k = 3 mean >= k3_min; under stop_k3 at k = 3 it is not labelled reused."""
    recs = [read_json(ITEMS / f"{R1.fname(s)}.json") for s in cc.states()]
    h, k3 = [r["head"] for r in recs], [r["khead"]["by_k"]["3"] for r in recs]
    ok1, ok3 = all(r["label"] in R1.READABLE for r in h), all(r["label"] in R1.READABLE for r in k3)
    m1 = fmean(r["share_lost"] for r in h) if ok1 else math.nan
    m3 = fmean(T2.num(r["share_lost"]) for r in k3) if ok3 else math.nan
    over = ok1 and all(T2.num(r["share_lost"]) > T2.num(r["random_median"]) for r in h)
    return {"k1_mean": m1, "k3_mean": m3, "Q0a_k1": R1.verdict(m1 >= th["share_new_max"] and over, m1), "Q0b_k3": R1.verdict(m3 >= th["k3_min"], m3),
            "stop_not_reused": R1.verdict(m3 < th["stop_k3"], m3)}  # fmt: skip


def table(cc: Config, th: dict[str, Any]) -> None:
    rows, s1 = rows_of(cc), read_json(T2.S1 / "summary.json")
    lab = label(cc, th)
    readable = lab["Q0a_k1"] == "holds" and lab["stop_not_reused"] == "misses" and all(
        s1["rules"][k] == "holds" for k in ("P1_copy_uses_head", "P2_ciphers_do_not"))  # fmt: skip
    out: dict[str, Any] = {"experiment": EXPERIMENT, "rows": rows, "label": lab, "corners_readable": readable, "proxies": {}, "side_lowcopy": {}}
    for p in PROXIES:
        vals = {"copy": [r[p] for r in rows["copy"]], "prior_cipher": [r[p] for r in rows["prior_cipher"]],
                SLOT: [r[p] for r in rows["cuedcopy"]], "cipher": [r[p] for r in rows["cipher"]]}  # fmt: skip
        e = T2.effects(vals, max(max(v) - min(v) for v in vals.values()))  # band: the largest seed range over the corners
        e["corner_mean"]["cuedcopy"] = e["corner_mean"].pop(SLOT)
        out["proxies"][p] = e | {"follows": e["follows"] if readable or p == "log10_prior" else "not_read"}
        out["side_lowcopy"][p] = fmean(r[p] for r in rows["lowcopy"])
    rnd = [read_json(ITEMS / f"{R1.fname('random/' + s)}.json") for s in cc.states()]
    new = [read_json(ITEMS / f"{R1.fname(s)}.json") for s in cc.states()]
    out["random"] = {"hf_gain": [r["hf_gain"] for r in rnd], "erank": [r["erank"] for r in rnd]}
    out["absent_gain_over_G"] = [r["absent_gain"] / r["head"]["G"] if r["head"]["G"] > 0 else math.nan for r in new]  # a zero gain: not read
    pr = out["proxies"]
    out["rules"] = {**{k: lab[k] for k in ("Q0a_k1", "Q0b_k3", "stop_not_reused")}, "Q1_r_follows": pr["r"]["follows"],
                    "Q2_prior_follows": pr["log10_prior"]["follows"], "Q3_steering": {k: pr[k]["follows"] for k in ("steering_pooled", "steering_perkey")},
                    "Q4_neurons": {k: pr[k]["follows"] for k in ("neuron_k64", "neuron_kmatched")}, "Q5_erank": pr["erank"]["follows"],
                    "C_absent": R1.verdict(all(abs(x) <= th["absent_max"] for x in out["absent_gain_over_G"]), *out["absent_gain_over_G"])}  # fmt: skip
    write_stage(RESULTS / "table.json", jsonable(out))
    for p, e in pr.items():
        log(f"[table {p}] corners {e['corner_mean']}; side low copy {out['side_lowcopy'][p]:.4f}; mechanism {e['mechanism_effect']:.4f}, prior {e['prior_effect']:.4f}, band {e['band']:.4f} -> {e['follows']}")
    log(f"[table] label {lab}; random {out['random']}; absent gain / G {out['absent_gain_over_G']}; rules {out['rules']}; median random hf gain {median(out['random']['hf_gain']):.4f}")


def main() -> int:
    cc, th = load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as every saved reading
    log(f"{EXPERIMENT} cued copy start; settings {asdict(cc)}; thresholds {th}")
    for need in (T2.S1 / "summary.json", T2.OUT / "table.json"):
        if not need.is_file():
            log(f"STOP: {need} missing (runs 1 and 2 of P2.54 must have finished)")
            return 1
    ITEMS.mkdir(parents=True, exist_ok=True)
    w, (th1, thk) = World(cc), (R1.load_config()[1], KH.load_config()[1])
    R1.OUT = RESULTS  # run 1's control writes results/control.json
    R1.control(w.rd, th1)
    ctl: dict[str, Any] = {"experiment": EXPERIMENT}
    if miss := T2.controls(w, T2.load_config()[1], ctl):
        stop(RESULTS / "control2.json", ctl, "controls: " + "; ".join(miss), log)
    write_stage(RESULTS / "control2.json", jsonable(ctl))
    stage_a(w)
    stage_b(w, th)
    w.rd.c["cuedcopy"] = read_json(RESULTS / "b.json")["c"]
    stage_c(w, th1, thk)
    table(cc, th)
    log(f"{EXPERIMENT} cued copy DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
