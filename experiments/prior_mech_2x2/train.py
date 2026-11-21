"""E_TBD2 (cell P2.54) run 2: the low-prior copy, then every proxy on the four corners of prior x mechanism.

The low-prior reused corner: E147's key copy task ("M x -> x", same 96 keys, marker, table and recipe) on its
lowest-prior windows only. A pool of windows is drawn by E58's rule (`kv.windows` on E12's Pile rows, split by row)
from a new seed; on each split the windows are ordered by M's P(x) at the read, the longest lowest-first run whose
geometric mean is at or under `prior_bar` is found and, over E147's count of rows, thinned evenly to it (`lowest_run`).
Fine-tuned with E147's call (`keydep.finetune`, step 30, seeds 0-2), saved, and scaled to E237's common size like
every key state. Beside it, one
Gaussian random update per seed at its per-matrix norms times the same c (E237's `finetune.random_direction`). Run 1
(`run.py`, its control and its head switch-off) must have finished first; this run reads its labels. Plan
reports/plans/2026-11-21_plan_prior-mech-2x2.md; rules results/predictions_stage2.json (refuses unless committed).

  a  controls, alone: the prior path repeats E133's copy prior; holes_fill's readers repeat E147's saved neuron and
     steering shares on e58/copy_s0 (c = 1); then the selection and M's prior on the kept read rows (exit 1 on a
     miss, or when the kept rows or keys are too few or the prior is over the bar)                                    -> results/stage2/a.json
  b  three fine-tunes, sizes, c                                                               -> results/stage2/b.json
  c  head switch-off (run 1's read_state) on the three new states; r, size, erank, neuron and steering shares on the
     three new states and their random updates; neuron and steering shares on E237's 15 key states at E237's c
     (on the new task, the neuron and steering shares read only the keys with a train window and their read rows,
     `keys_with_train`; counts in each item's "hf_rows")                                     -> results/stage2/<item>.json
  table  the corners (seed means), the two effects per proxy, the filed rules                -> results/stage2/table.json
Every scored number is also printed. A rerun skips what is written.

    sbatch --comment=E_TBD2 cluster/prior_mech_2x2_stage2.sbatch
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from torch import Tensor

from commute import finetune as FT
from commute import keydep as KD
from commute import steering as ST
from commute import update_spectrum as US
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

R1: Any = load_experiment("prior_mech_2x2", "run")  # run 1: Reader, read_state, the head share's thresholds
HF: Any = load_experiment("holes_fill", "run")  # Job.state: TopK neuron shares and fair steering
PVS: Any = load_experiment("prior_vs_share", "run")  # forward_prior: E133's prior path
E205: Any = load_experiment("held_model_size_matched", "run")  # sizes: E187's ||dW|| / ||W0||
E228, R96 = R1.E228, R1.R96
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT, LARGE = RESULTS / "predictions_stage2.json", RESULTS / "stage2", RESULTS / "large"
S1 = RESULTS / "stage1"  # run 1's results
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
X = EXPERIMENTS
TASK = "e58/lowcopy"  # named in E147's form so run 1's Reader scales it by c like a key state
WIN_FILE = LARGE / "windows_lowcopy.pt"
log = R1.log


@dataclass(frozen=True)
class Config:  # filed in predictions_stage2.json "settings"
    source_task: str  # E147's map the low-prior windows are cut from
    seeds: tuple[int, ...]
    prior_bar: float  # the kept rows' geometric-mean P(x) at the read, at or under
    n_pool: int  # windows drawn by E58's rule before the selection
    pool_seed: int
    n_train_min: int
    n_read_min: int
    n_keys_min: int
    step: int
    random_seed: int
    split_tol: float
    batch: int

    def __post_init__(self) -> None:
        assert self.source_task == "copy" and self.seeds == (0, 1, 2) and self.step == 30 and 0 < self.prior_bar < 1

    def new_states(self) -> list[str]:
        return [f"{TASK}_s{s}" for s in self.seeds]

    def items(self, key_states: list[str]) -> list[str]:
        """Stage c's order: the new states, their random updates, then the key states."""
        return self.new_states() + [f"random/{s}" for s in self.new_states()] + key_states


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def source_spec() -> dict[str, Any]:
    return read_json(R1.E147_INV)["maps"]["copy"]


# ---- the world: run 1's Reader with the new task in its inventory, holes_fill's readers, the split -----------------


class World:
    def __init__(self, cfg: Config) -> None:
        self.cfg, self.rd = cfg, R1.Reader(R1.load_config()[0])
        self.net, self.dws = self.rd.net, dict[str, dict[str, Tensor]]()
        inv = self.rd.inv
        inv["tasks"][TASK] = {"format": "kv", "path": str(WIN_FILE.relative_to(X.parent)), "batch": 64}
        for s in cfg.seeds:
            st = f"{TASK}_s{s}"
            inv["states"][st] = {"task": TASK, "file": str((LARGE / f"dW_lowcopy_s{s}.pt").relative_to(X.parent))}
            inv["states"][f"random/{st}"] = {"task": TASK, "file": None}
        hf = read_json(X / "holes_fill" / "results" / "predictions.json")
        self.hf = HF.Job(self.net, inv, hf["settings"], hf["thresholds"])
        k_m = read_json(X / "share_untrained" / "results" / "stage0.json")["k_m"]  # as E187 and holes_fill's stage 0 set them
        self.hf.ks = {"k64": {l: int(hf["settings"]["k_fixed"]) for l in HF.FS.LAYERS}, "kmatched": {l: int(k_m[key]) for l, key in enumerate(HF.DOWN)}}
        self.hf.dW = self.dw  # every holes_fill read takes this run's (scaled) update
        self.sp: Any = None

    def dw(self, item: str) -> dict[str, Tensor]:
        """The item's update as read: a saved state times its task's c, or a new state's random direction times c."""
        if item not in self.dws:
            state = item.removeprefix("random/")
            d = {k: v.float() for k, v in torch.load(X.parent / self.rd.inv["states"][state]["file"], weights_only=False).items()}
            if item.startswith("random/"):
                d = FT.random_direction(d, self.cfg.random_seed + self.cfg.new_states().index(state))
            c = self.rd.scale(state)
            self.dws[item] = {k: v * c for k, v in d.items()}
        return self.dws[item]


# ---- stage a: controls, the selection ------------------------------------------------------------------------------


def controls(w: World, th: dict[str, Any], out: dict[str, Any]) -> list[str]:
    ctl = read_json(X / "prior_vs_share" / "results" / "predictions.json")["stage0"]["controls"][0]
    p = PVS.forward_prior(w.net, {"windows": {"format": "kv", "path": ctl["windows"]}, "name": "copy", "built": "control"})
    out["prior_repeat"] = {"got": p, "saved": ctl["expected"], "rel": p / ctl["expected"] - 1}
    rec, _ = w.hf.state("e58/copy_s0", 0)  # copy's c is 1: E147's saved reading must repeat
    ref = read_json(X / "refit_96key" / "results" / "stage1.json")["states"]["e58/copy_s0"]["holes_fill"]
    got = {f"neuron {k}": (rec["neuron"][k]["share"], ref["neuron"][k]["share"]) for k in rec["neuron"]}
    got |= {f"steering {k}": (rec["steering"][k]["share"], ref["steering"][k]["share"]) for k in rec["steering"]}
    out["hf_repeat"] = got
    log(f"[a] prior repeat {out['prior_repeat']}; holes_fill repeat on e58/copy_s0 {got}")
    miss = [f"prior rel {out['prior_repeat']['rel']:.2e}"] if not abs(out["prior_repeat"]["rel"]) <= th["prior_repro_rel"] else []
    return miss + [f"{k} {a} vs {b}" for k, (a, b) in got.items() if not abs(a - b) <= th["share_tol"]]


def run_length(sorted_lp: Tensor, bar: float) -> int:
    """The longest prefix of ascending log P whose mean is at or under log(bar); 0 if even the lowest row is over it."""
    ok = sorted_lp.cumsum(0) / torch.arange(1, len(sorted_lp) + 1) <= math.log(bar)
    return int(ok.nonzero().max()) + 1 if bool(ok.any()) else 0


def lowest_run(lp: Tensor, bar: float, cap: int) -> Tensor:
    """Mask of the longest lowest-first run of rows (stable sort) whose mean log P is at or under log(bar). Over `cap`
    rows, the run is thinned evenly in sorted order to `cap` rows (so the kept prior stays near the bar rather than
    far under it), and the thinned rows' own longest run is kept, so their mean is at or under the bar too."""
    order = torch.sort(lp.double(), stable=True)
    idx = order.indices[: run_length(order.values, bar)]
    if len(idx) > cap:
        idx = idx[torch.linspace(0, len(idx) - 1, cap).round().long()]
        idx = idx[: run_length(lp.double()[idx], bar)]
    mask = torch.zeros(len(lp), dtype=torch.bool)
    mask[idx] = True
    return mask


def logp(w: World, toks: Tensor, table: Tensor, half: int) -> Tensor:
    b = w.cfg.batch
    return torch.cat([ST.logp_at(w.net.run(toks[s : s + b])[0], half, table[toks[s : s + b, half]]) for s in range(0, len(toks), b)])


def pool(cfg: Config, src: dict[str, Any]) -> dict[str, Any]:
    """E58's window rule (`kv.windows` on E12's Pile rows, `data.split_windows` by row) on E147's 96 copy keys, at
    n_pool windows from a new seed: the rows the selection draws from."""
    from commute import anchors as AN
    from commute import kv
    from commute.data import pile_rows, split_windows

    task = src["task"]
    w = kv.windows(pile_rows("train", 0, AN.ROWS, log), task.marker, tuple(task.keys), AN.HALF, cfg.n_pool, cfg.pool_seed)
    train, heldout = split_windows(w)
    return {"train": train, "heldout": heldout}


def select(w: World, src: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for part, win in pool(w.cfg, src).items():
        lp = logp(w, win.tokens, src["table"], win.half)
        m = lowest_run(lp, w.cfg.prior_bar, len(src[part]))  # at most E147's own count per split
        kept[part] = win.subset(m)
        out[part] = {"n_pool": len(lp), "n_kept": int(m.sum()), "n_keys": int(kept[part].tokens[:, win.half].unique().numel()),
                     "geo_prior_pool": math.exp(float(lp.mean())), "geo_prior_kept": math.exp(float(lp[m].mean())) if bool(m.any()) else math.nan}  # fmt: skip
        log(f"[a {part}] kept {out[part]['n_kept']} of {out[part]['n_pool']} windows on {out[part]['n_keys']} keys; geo prior {out[part]['geo_prior_pool']:.3e} -> {out[part]['geo_prior_kept']:.3e} (bar {w.cfg.prior_bar:.3e})")
    return kept


def stage_a(w: World, th: dict[str, Any]) -> None:
    """The controls, then the construction check before any training: the kept read rows' prior at or under the bar
    (1.5 decades under key copy's), enough rows and keys."""
    path = OUT / "a.json"
    if path.is_file() and "stopped" not in read_json(path):
        log("stage a done before")
        return
    out: dict[str, Any] = {"experiment": EXPERIMENT}
    if miss := controls(w, th, out):
        stop(path, out, "stage a controls: " + "; ".join(miss), log)
    src = torch.load(X.parent / source_spec()["windows"], weights_only=False)
    kept = select(w, src, out)
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({"train": kept["train"], "heldout": kept["heldout"], "task": src["task"], "table": src["table"]}, WIN_FILE)
    out["prior"] = PVS.forward_prior(w.net, {"windows": {"format": "kv", "path": str(WIN_FILE.relative_to(X.parent))}, "name": TASK, "built": "lowcopy"})
    out["log10_prior"] = math.log10(out["prior"])
    log(f"[a] the new task's prior on its read rows (E133's path) {out['prior']:.4e}, log10 {out['log10_prior']:.3f} (bar {math.log10(w.cfg.prior_bar):.3f})")
    few = out["train"]["n_kept"] < w.cfg.n_train_min or out["heldout"]["n_kept"] < w.cfg.n_read_min or out["train"]["n_keys"] < w.cfg.n_keys_min
    if few or not out["prior"] <= w.cfg.prior_bar:
        stop(path, out, f"stage a: kept {out['train']['n_kept']} train / {out['heldout']['n_kept']} read rows on {out['train']['n_keys']} keys, prior {out['prior']:.3e}", log)
    write_stage(path, jsonable(out))


# ---- stage b: the fine-tunes ---------------------------------------------------------------------------------------


def stage_b(w: World) -> None:
    path = OUT / "b.json"
    out: dict[str, Any] = read_json(path) if path.is_file() else {"experiment": EXPERIMENT, "states": {}}
    win = torch.load(WIN_FILE, weights_only=False)
    for s, st in zip(w.cfg.seeds, w.cfg.new_states(), strict=True):
        if st in out["states"]:
            continue
        ft = KD.finetune(KD.Ctx(w.net, {}, {}, {}, LARGE), win, s, (w.cfg.step,))  # keydep.finetune reads only ctx.net
        d = ft.deltas[w.cfg.step]
        f = X.parent / w.rd.inv["states"][st]["file"]
        torch.save({k: v.bfloat16() for k, v in d.items()}, f)
        size = E205.sizes({f"{k}.weight": v.bfloat16().float() for k, v in d.items()}, w.net.W0)["whole"]
        out["states"][st] = {"p_heldout": ft.evals[w.cfg.step]["p_o"], "size": size, "sha256": KD.sha256(f)}
        log(f"[b {st}] step {w.cfg.step}: held-out P(x) {out['states'][st]['p_heldout']:.4f}; size {size:.6g}")
        write_stage(path, out)
    target = read_json(R1.E237_C.parent / "stage0.json")["target"]
    out["c"] = target / fmean(v["size"] for v in out["states"].values())
    log(f"[b] c {out['c']:.4f} (E237's common size {target:.6g})")
    write_stage(path, out | {"done": True})


# ---- stage c: the readings -----------------------------------------------------------------------------------------


def keys_with_train(t: Any) -> Any:
    """holes_fill's Task for the new task, its key pool cut to the keys that have a train window (per-key steering fits
    one vector per pool key, fair_steering.vectors) and its read rows to those keys (key_index needs every read key in
    the pool). Only holes_fill's readers (neuron and steering shares, their gain) read this; the head switch-off, r and
    the prior read all the task's read rows."""
    t.pool = t.pool[torch.isin(t.pool, t.train[:, t.read.half])]
    keep = torch.isin(t.read.toks[:, t.read.half], t.pool)
    t.read = dataclasses.replace(t.read, toks=t.read.toks[keep])
    return t


def lowcopy_task(w: World) -> dict[str, int]:
    """Installs the cut Task in holes_fill's cache (its random updates share it); returns the counts it reads on."""
    t = keys_with_train(HF.load_task(TASK, w.rd.inv["tasks"][TASK]))
    t.lo0 = HF.lo_at(w.net, t, t.read.toks)
    w.hf.tasks[TASK] = t
    n = {"n_keys": len(t.pool), "n_read_rows": len(t.read.toks), "n_train_rows": len(t.train)}
    log(f"[c] steering and neuron reads on {TASK}: {n['n_keys']} keys with a train window, {n['n_read_rows']} read rows")
    return n


def shares(w: World, item: str, i: int) -> dict[str, Any]:
    rec, _ = w.hf.state(item, i)
    return {"hf_gain": rec["gain"], "neuron": {k: v["share"] for k, v in rec["neuron"].items()},
            "steering": {k: v["share"] for k, v in rec.get("steering", {}).items()}}  # fmt: skip


def weights(w: World, item: str) -> dict[str, Any]:
    """r (log-odds form), size and erank of a new item; the key states' are E237's, saved (stage1.json 'scaled')."""
    state, dW = item.removeprefix("random/"), w.dw(item)
    r, _ = E228.state_read(state, w.rd.inv)
    if w.sp is None:
        w.sp = E228.Splitter(w.net, w.cfg.split_tol)
    z = E228.readings(w.net, w.sp, r, dW)
    size = E205.sizes({f"{k}.weight": v for k, v in dW.items()}, w.net.W0)["whole"]
    return {"r": z["r"], "r_gain": z["gain"], "size": size, "erank": US.state(dW)["erank"]}


def stage_c(w: World, th1: dict[str, Any], items: list[str]) -> None:
    R1.OUT = OUT / "head"  # run 1's read_state writes the new states' head switch-off here, not into run 1's results
    R1.OUT.mkdir(parents=True, exist_ok=True)
    hf_rows = lowcopy_task(w)
    for i, item in enumerate(items):
        path = OUT / f"{R1.fname(item)}.json"
        if path.is_file():
            continue
        t0, rec = time.time(), {"experiment": EXPERIMENT, "item": item}
        if item.startswith(f"{TASK}_s"):
            rec["head"] = R1.read_state(w.rd, item, th1)  # writes <item>.json; overwritten below with the whole record
        rec |= shares(w, item, i) | (weights(w, item) | {"hf_rows": hf_rows} if TASK in item else {})
        write_stage(path, jsonable(rec | {"seconds": time.time() - t0}))
        log(f"[c {item}] " + json.dumps({k: v for k, v in rec.items() if k not in ("head", "experiment")}) + f"; {time.time() - t0:.0f} s")
        torch.cuda.empty_cache()


# ---- the table and the filed rules -----------------------------------------------------------------------------------

CORNERS = {"copy": ("high", "reused"), "prior_cipher": ("high", "new"), "lowcopy": ("low", "reused"), "cipher": ("low", "new")}
EXTRA = ("rank1", "elicit")  # read on the same footing, outside the 2x2
PROXIES = ("head_share", "r", "log10_prior", "size", "erank", "steering_pooled", "steering_perkey", "neuron_k64", "neuron_kmatched")


def num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) else math.nan


def state_row(task: str, state: str, a: dict[str, Any]) -> dict[str, float]:
    """One state's proxies: stage c's file; for a key state, r, size and erank from E237 (scaled) and the head share
    from run 1; the prior is the task's (E133's saved, E237's summary; the new task's from stage a)."""
    c = read_json(OUT / f"{R1.fname(state)}.json")
    if task == "lowcopy":
        w = {k: c[k] for k in ("r", "size", "erank")} | {"head_share": c["head"]["share_lost"], "log10_prior": a["log10_prior"]}
    else:
        e = read_json(R1.E237_C.parent / "stage1.json")["states"][state]["scaled"]
        w = {k: e[k] for k in ("r", "size", "erank")} | {"head_share": read_json(S1 / f"{R1.fname(state)}.json")["share_lost"],
                                                       "log10_prior": read_json(R1.E237_C.parent / "summary.json")["log10_prior"][task]}  # fmt: skip
    sh = {f"steering_{k}": v for k, v in c["steering"].items()} | {f"neuron_{k}": v for k, v in c["neuron"].items()}
    return {p: num((w | sh).get(p)) for p in PROXIES}


def effects(vals: dict[str, list[float]], band: float) -> dict[str, Any]:
    """Seed means per corner; the mechanism effect ((reused) - (new)) / 2 and the prior effect ((high) - (low)) / 2;
    "mechanism" / "prior" when that effect is the larger in size and over the band, "no_call" otherwise (a tie too)."""
    m = {t: fmean(v) for t, v in vals.items()}
    mech = (m["copy"] + m["lowcopy"] - m["prior_cipher"] - m["cipher"]) / 2
    prior = (m["copy"] + m["prior_cipher"] - m["lowcopy"] - m["cipher"]) / 2
    if not all(math.isfinite(x) for x in (mech, prior, band)):
        call = "not_read"
    else:
        call = "mechanism" if abs(mech) > abs(prior) and abs(mech) > band else "prior" if abs(prior) > abs(mech) and abs(prior) > band else "no_call"
    return {"corner_mean": m, "mechanism_effect": mech, "prior_effect": prior, "band": band, "follows": call}


def table(cfg: Config, th: dict[str, Any]) -> None:
    a, s1 = read_json(OUT / "a.json"), read_json(S1 / "summary.json")
    states = {t: [f"{TASK}_s{s}" if t == "lowcopy" else R96.state_name(t, s) for s in cfg.seeds] for t in (*CORNERS, *EXTRA)}
    rows = {t: [state_row(t, s, a) for s in ss] for t, ss in states.items()}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "rows": rows, "proxies": {}}
    for p in PROXIES:
        vals = {t: [r[p] for r in rows[t]] for t in CORNERS}
        band = max(max(v) - min(v) for v in vals.values())  # the largest seed range over the four corners, this job
        out["proxies"][p] = effects(vals, band)
    out["random"] = {s: read_json(OUT / f"{R1.fname('random/' + s)}.json") for s in cfg.new_states()}
    low = [read_json(OUT / f"{R1.fname(s)}.json")["head"] for s in cfg.new_states()]
    ok = all(r["label"] in R1.READABLE for r in low)
    low_mean = fmean(r["share_lost"] for r in low) if ok else math.nan
    over = ok and all(r["share_lost"] > r["random_median"] for r in low)
    labels = {"copy": s1["rules"]["P1_copy_uses_head"], "ciphers": s1["rules"]["P2_ciphers_do_not"],
              "lowcopy": R1.verdict(low_mean >= th["share_new_max"] and over, low_mean)}  # fmt: skip
    readable = all(v == "holds" for v in labels.values())
    pr = out["proxies"]
    out["rules"] = {"Q0_lowcopy_uses_head": labels["lowcopy"], "corners_labelled": labels,
                    "Q1_r_follows_mechanism": pr["r"]["follows"] if readable else "not_read",
                    "Q2_prior_follows_prior": pr["log10_prior"]["follows"],
                    "Q3_steering_follows": {k: pr[k]["follows"] if readable else "not_read" for k in ("steering_pooled", "steering_perkey")},
                    "Q4_neurons_follow": {k: pr[k]["follows"] if readable else "not_read" for k in ("neuron_k64", "neuron_kmatched")}}  # fmt: skip
    write_stage(OUT / "table.json", jsonable(out))
    for p, e in pr.items():
        log(f"[table {p}] corners {e['corner_mean']}; mechanism effect {e['mechanism_effect']:.4f}, prior effect {e['prior_effect']:.4f}, band {e['band']:.4f} -> {e['follows']}")
    log(f"[table] rules {out['rules']}")


def main() -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as every saved reading
    log(f"{EXPERIMENT} run 2 start; settings {asdict(cfg)}; thresholds {th}")
    if not (s := S1 / "summary.json").is_file():
        log(f"STOP: run 1 has not finished ({s} missing)")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    w = World(cfg)
    stage_a(w, th)
    stage_b(w)
    w.rd.c["lowcopy"] = read_json(OUT / "b.json")["c"]
    stage_c(w, R1.load_config()[1], cfg.items(w.rd.cfg.key_states()))
    table(cfg, th)
    log(f"{EXPERIMENT} run 2 DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
