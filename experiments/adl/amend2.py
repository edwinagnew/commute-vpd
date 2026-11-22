"""E267 (cell P2.60), amendment 2 of the activation difference lens (run.py, E264/E265): the same D, settings, control
and bars, read on two new item sets. Settings, rules and expected values in results/amend2/predictions.json (refuses
unless committed unchanged); everything is written under results/amend2/ through run.py's `use("amend2")`.

  (a) P2.57's prior-matched states: E254's 10 copy and 10 prior-matched cipher seeds and its 6 matched random updates,
      E255's 3 second-cipher seeds on copy's keys and their 3 random updates. Each is built as prior_matched/world.py
      `World.dw` builds it (the saved update times the run's c; a random is `finetune.random_direction` of the unscaled
      update at the run's random_seed + the state's index in its order, times c) and read on its run's task (E147's kv
      windows; holes_fill's reader, as run.py reads every item).
  (b) P2.56's procedural states: E256's rl, sft and random arms and E260's rlg and randg, 3 seeds of sequence, brackets
      and addition, as saved. Each is read on its own task: E256's held-out items (procedural_rl `build_items`, the
      first n_held), the answer in a last column the model never sees, read at the cue (E256's `CueRead` layout) in
      holes_fill's log-odds. A random's D is over its matched state's gain, as run.py's.

  run      stage 0 first (run.py's stage0 on amend1's control, which must also repeat amend1's D and gain within
           repeat_rel; a miss exits 1), then every item; a rerun skips written items   -> results/amend2/stage0.json, items/
  list     CPU, no model: every item, the files it loads (each checked to exist), the procedural held-out items built
  summary  CPU, no model: (a) the seed-level calls, (b) Spearman and partial Spearman given KL with E_b beside every
           other proxy's, on the same states                                            -> results/amend2/summary.json

    PRED_SHA=<sha> python experiments/adl/amend2.py run
    python experiments/adl/amend2.py list
    python experiments/adl/amend2.py summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any

import torch

from commute import finetune as FT
from commute import seed_rank as SR
from commute import vpd
from commute.data import LAST, cached_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.realistic import update_size

R: Any = load_experiment("adl", "run")  # the lens: World, read_item, stage0, Item, use, log, dump
RA: Any = load_experiment("prior_matched", "run_a")  # E254: load_config (states, items, random_seed), OUT, W (world.py)
RB: Any = load_experiment("prior_matched", "run_b")  # E255: load_config, OUT, TASK
PR: Any = load_experiment("procedural_rl", "run")  # E256/E260: Config, build_items, NT, RT, RESULTS, READS
KC: Any = load_experiment("kl_control", "run")  # E262: proxy_table (the partial Spearman given KL), PROCEDURAL
R.use("amend2")
HF, W = R.HF, RA.W
REPO = EXPERIMENTS.parent
A1_STAGE0 = R.RESULTS / "amend1" / "stage0.json"
KC_SUMMARY = EXPERIMENTS / "kl_control" / "results" / "summary.json"
PRIOR = ("e254", "e255")
log, read = R.log, R.read


@dataclass(frozen=True)
class Config(R.Config):  # amend1's settings and the procedural arms
    trained_arms: tuple[str, ...]
    random_arms: tuple[str, ...]
    collapsed: tuple[str, ...]  # trained states left out of the second set of (b)

    def __post_init__(self) -> None:
        super().__post_init__()
        for f in ("trained_arms", "random_arms", "collapsed"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        assert set(self.trained_arms) | set(self.random_arms) == set(PR.ARMS) and set(self.random_arms) == set(PR.RANDOM)


# ---- the items --------------------------------------------------------------------------------------------------------


def prior_runs() -> list[tuple[str, Path, list[str], list[str], int, dict[str, float]]]:
    """Per run: its code, results folder, state order (the random seed's index), items read (no do-nothing),
    random_seed and c per task (train.json)."""
    ca, cb = RA.load_config()[0], RB.load_config()[0]
    c_b = read(RB.OUT / "train.json")["c"]
    return [("e254", RA.OUT, [st for t in ca.tasks for st in ca.states(t)], ca.items(), ca.random_seed, read(RA.OUT / "train.json")["c"]),
            ("e255", RB.OUT, cb.states(), cb.states() + [f"random/{st}" for st in cb.states()], cb.random_seed, {RB.TASK: c_b})]  # fmt: skip


def task_of(state: str) -> str:
    return state.removeprefix("e58/").rsplit("_s", 1)[0]  # e58/pa_copy_s3 -> pa_copy


def prior_items() -> list[Any]:
    out = []
    for code, d, order, names, seed0, c in prior_runs():
        for n in names:
            st, rnd = n.removeprefix("random/"), n.startswith("random/")
            t = task_of(st)
            size = read(d / "items" / f"{W.R1.fname(n)}.json")["size"]
            side = "copy" if t == "pa_copy" else "cipher"
            out.append(R.Item(f"{code}/{n}", code, st, c[t], rnd, t, side, seed0 + order.index(st) if rnd else -1, size))
    return out


def prior_inventory() -> dict[str, dict[str, Any]]:
    """E254/E255's tasks and states as prior_matched/world.py `add_task` / `add_state` enter them."""
    windows = {f"pa_{t}": W.E147_INV["maps"][t]["windows"] for t in RA.load_config()[0].tasks}
    windows[RB.TASK] = read(RB.OUT / "a.json")["chosen"]["windows"]
    tasks = {f"e58/{t}": {"format": "kv", "path": p, "batch": 64} for t, p in windows.items()}
    states = {it.state: {"task": f"e58/{it.pair}", "file": W.rel(W.LARGE / f"dW_{it.state.removeprefix('e58/')}.pt")} for it in prior_items()}
    return {"tasks": tasks, "states": states}


def proc_states() -> dict[str, dict[str, Any]]:
    """Every saved procedural state, by name (sequence_rl_s0): its train record (arm, seed, file, sha256, matched_to)."""
    out: dict[str, dict[str, Any]] = {}
    for task in PR.RT.PROCEDURAL:
        for f in (f"train_{task}.json", f"train_guarded_{task}.json"):
            out |= {n: s | {"task": task} for n, s in read(PR.RESULTS / f)["states"].items()}
    return out


def proc_items(cfg: Config) -> list[Any]:
    """Random arms first (the do-nothing reference, as E256 read them), then the trained arms."""
    st = proc_states()
    assert {s["arm"] for s in st.values()} == set(cfg.trained_arms) | set(cfg.random_arms), {s["arm"] for s in st.values()}
    order = sorted(st, key=lambda n: (st[n]["arm"] not in cfg.random_arms, n))
    return [R.Item(f"p256/{n}", "proc", n, 1.0, st[n]["arm"] in cfg.random_arms, st[n]["task"], st[n]["arm"],
                   size_want=read(PR.READS / f"{n}.json")["size"]) for n in order]  # fmt: skip


def all_items(cfg: Config) -> list[Any]:
    return prior_items() + proc_items(cfg)


def source(it: Any) -> Path:
    """The record the item's source run wrote: its size and gain."""
    if it.group == "proc":
        return PR.READS / f"{it.state}.json"
    return (RA.OUT if it.group == "e254" else RB.OUT) / "items" / f"{W.R1.fname(it.name.split('/', 1)[1])}.json"


def saved_gain(it: Any) -> float:
    rec = read(source(it))
    return rec["gain"]["lo"] if it.group == "proc" else rec["hf_gain"]


def held_items() -> tuple[dict[str, Any], int]:
    """E256's held-out items per task (procedural_rl Ctx.__init__ and job_read: build_items, the first n_held) and the
    prompt length (the cue is its last position)."""
    C = config_from_settings(PR.Config, read(PR.PRED)["settings"])
    tok = vpd.tokenizer()
    v = PR.NT.Vocab([tok.decode([i]) for i in range(len(tok))], tok.encode)
    items = PR.build_items(v, C, cached_rows("train", 0, C.pile_train_n), cached_rows("val", C.val_start, C.val_n))
    return {t: d["held"].take(slice(0, C.n_held)) for t, d in items.items()}, C.prompt_len


# ---- the world ----------------------------------------------------------------------------------------------------------


class World(R.World):
    """run.py's world with E254/E255's tasks and states in the reader's inventory and the procedural tasks beside it."""

    def __init__(self, cfg: Config, th: dict[str, Any]) -> None:
        super().__init__(cfg, th)
        inv = prior_inventory()
        self.hf.inv["tasks"] |= inv["tasks"]
        self.hf.inv["states"] |= inv["states"]
        self.proc, (self.held, self.prompt_len), self.proc_tasks = proc_states(), held_items(), dict[str, Any]()

    def task(self, state: str) -> Any:
        if state not in self.proc:
            return super().task(state)
        name = self.proc[state]["task"]
        if name not in self.proc_tasks:
            h, L = self.held[name], self.prompt_len
            rd = SimpleNamespace(toks=torch.cat([h.toks, h.true[:, None]], 1), token=LAST, half=L - 1, ci_len=L)
            t = HF.Task(rd, None, None)
            t.lo0 = HF.lo_at(self.net, t, t.read.toks)
            self.proc_tasks[name] = t
        return self.proc_tasks[name]

    def load_proc(self, name: str) -> dict[str, torch.Tensor]:
        s = self.proc[name]["state"]
        path = REPO / s["file"]
        if (sha := hashlib.sha256(path.read_bytes()).hexdigest()) != s["sha256"]:
            raise SystemExit(f"{path} has sha256 {sha}, its train record {s['sha256']}")
        return {k: v.float() for k, v in torch.load(path, weights_only=False).items()}

    def change(self, it: Any) -> tuple[Any, Any, float]:
        if it.group == "proc":
            dW = self.load_proc(it.state)
            ref = self.load_proc(self.proc[it.state]["matched_to"]) if it.random else None
            size = update_size(dW, self.net.W0)
        elif it.group in PRIOR:
            d = self.hf.dW(it.state)
            dW = {k: v * it.c for k, v in (FT.random_direction(d, it.seed) if it.random else d).items()}
            ref = {k: v * it.c for k, v in d.items()} if it.random else None
            size = R.KG.E205.sizes({f"{k}.weight": v for k, v in dW.items()}, self.net.W0)["whole"]
        else:
            return super().change(it)
        if not abs(size / it.size_want - 1) <= self.th["size_rel"]:
            raise SystemExit(f"{it.name}: size {size} is not its source's {it.size_want} ({source(it)}): not that item")
        return dW, ref, size


# ---- the job -----------------------------------------------------------------------------------------------------------


def stage0(w: World, P: dict[str, Any]) -> bool:
    """run.py's stage 0 on amend1's control, then its D and gain against amend1's (repeat_amend1)."""
    if R.STAGE0.is_file() and read(R.STAGE0).get("passed") is True:
        log("stage 0 passed before")
        return True
    ok, ref = R.stage0(w, P["repeat_gain"]), P["repeat_amend1"]
    out = read(R.STAGE0)
    rep = {k: {"got": out["control"][k], "amend1": ref[k], "rel": abs(out["control"][k] / ref[k] - 1)} for k in ("D", "gain")}
    miss = [f"control {k} {v['got']} is not amend1's {v['amend1']} (rel {v['rel']:.3g})" for k, v in rep.items() if not v["rel"] <= ref["rel_tol"]]
    out |= {"repeat_amend1": rep, "misses": out["misses"] + miss, "passed": ok and not miss}
    R.dump(R.STAGE0, out)
    log(f"[stage0 repeat of amend1] {json.dumps(rep)}; misses {miss}; PASSED {out['passed']}")
    return out["passed"]


def run_items(w: World, cfg: Config) -> None:
    todo = [it for it in all_items(cfg) if not (R.ITEMS / f"{R.fname(it.name)}.json").is_file()]
    for k, it in enumerate(todo):
        rec = R.read_item(w, it)
        rec["gain_saved"] = saved_gain(it)
        R.write_item(rec)
        log(f"ITEM {k + 1} of {len(todo)} DONE; gain {rec['gain']:.6f} vs its source's {rec['gain_saved']:.6f}")


def listing(cfg: Config) -> None:
    """Every item and the files it loads; exits naming any that is missing. Builds the procedural held-out items."""
    inv, proc, missing = prior_inventory(), proc_states(), list[str]()
    for it in all_items(cfg):
        if it.group == "proc":
            files = [proc[it.state]["state"]["file"]] + ([proc[proc[it.state]["matched_to"]]["state"]["file"]] if it.random else [])
        else:
            files = [inv["states"][it.state]["file"], inv["tasks"][f"e58/{it.pair}"]["path"]]
        files.append(str(source(it).relative_to(REPO)))
        missing += [f for f in files if not (REPO / f).is_file()]
        print(f"{it.name:42s} c {it.c:.6f} random {it.random!s:5s} seed {it.seed:7d} size {it.size_want:.6g} loads {files}")
    pts = {q["state"] for q in read(KC.PROCEDURAL)["points"]}
    no_eb = [n for n in proc if n not in pts]
    held, L = held_items()
    print(f"held-out items per task {({t: tuple(h.toks.shape) for t, h in held.items()})}, cue at {L - 1}; "
          f"distinct answers {({t: int(h.true.unique().numel()) for t, h in held.items()})}")  # fmt: skip
    n = {g: sum(it.group == g for it in all_items(cfg)) for g in (*PRIOR, "proc")}
    print(f"{sum(n.values())} items {n}; files missing {missing}; procedural states without E_b {no_eb}")
    if missing or no_eb:
        raise SystemExit(1)


# ---- the summary (CPU) -------------------------------------------------------------------------------------------------


def separation(copy: list[float], cipher: list[float], th: dict[str, Any]) -> dict[str, Any]:
    """The filed a_separation rule: cipher minus copy, exact Mann-Whitney of cipher over copy."""
    if len(copy) < 2 or len(cipher) < 2 or not all(math.isfinite(x) for x in copy + cipher):
        return {"call": "not read", "copy": copy, "cipher": cipher}
    gap, (u, p) = fmean(cipher) - fmean(copy), SR.mann_whitney(cipher, copy)
    a, g = p <= th["alpha"], th["smallest_gap"]
    call = "cipher above, apart" if a and gap > g else "copy above, apart" if a and gap < -g else "overlap"
    return {"call": call, "gap": gap, "sign": math.copysign(1, gap) if gap else 0.0, "U": u, "p": p, "copy": copy, "cipher": cipher,
            "range_copy": [min(copy), max(copy)], "range_cipher": [min(cipher), max(cipher)]}  # fmt: skip


def part_a(items: list[Any], D: Any, th: dict[str, Any]) -> dict[str, Any]:
    side = lambda g, s: [D(it.name) for it in items if it.group == g and it.side == s and not it.random]
    rnd = {it.name: D(it.name) for it in items if it.group in PRIOR and it.random}
    fin = [abs(x) for x in rnd.values() if math.isfinite(x)]
    return {"e254_cipher": separation(side("e254", "copy"), side("e254", "cipher"), th),
            "e255_cipher": separation(side("e254", "copy"), side("e255", "cipher"), th),
            "randoms": rnd, "random_abs_max": max(fin) if fin else math.nan, "smallest_gap": th["smallest_gap"]}  # fmt: skip


def label(rho: float | None, bar: float) -> str:
    return "not read" if rho is None else ">= bar (rival a)" if rho >= bar else "|rho| < bar (rival b)" if abs(rho) < bar else "<= -bar"


def close(a: float | None, b: float | None) -> bool:
    """The same call on the same states, up to the summation order of a correlation (row order may differ)."""
    return a == b if a is None or b is None else abs(a - b) <= 1e-12


def part_b(cfg: Config, items: list[Any], D: Any, th: dict[str, Any]) -> dict[str, Any]:
    pts = {q["state"]: q for q in read(KC.PROCEDURAL)["points"]}
    rows = [pts[it.state] | {"adl": D(it.name), "collapsed": it.state in cfg.collapsed} for it in items if it.group == "proc"]
    trained = [q for q in rows if q["arm"] in cfg.trained_arms]
    proxies = ["adl"] + read(KC.RESULTS / "predictions.json")["settings"]["proxies"]
    tab = {"trained": KC.proxy_table(trained, proxies), "trained_no_collapsed": KC.proxy_table([q for q in trained if not q["collapsed"]], proxies)}
    kc = read(KC_SUMMARY)["A_procedural"]["trained_no_repeat"]  # the same 27 states, read by E262
    same = {p: close(tab["trained"][p]["partial_given_kl"], kc[p]["partial_given_kl"]) for p in proxies[1:]}
    return {"tables": tab, "kl_control_trained_no_repeat": kc, "other_proxies_repeat_kl_control": same,
            "adl_partial": {k: label(t["adl"]["partial_given_kl"], th["partial_bar"]) for k, t in tab.items()},
            "points": [{k: q[k] for k in ("state", "task", "arm", "adl", "R_b", "kl", "collapsed")} for q in rows]}  # fmt: skip


def checks(items: list[Any], recs: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """The reported gain repeats (predictions stage0.reported_checks): items outside their bar."""
    out = {}
    for it in items:
        if it.name in recs:
            g, s = recs[it.name]["gain"], recs[it.name]["gain_saved"]
            bar = th["reader_tol"] if it.group in PRIOR else th["proc_gain_rel"] * max(1.0, abs(s))
            out[it.name] = {"gain": g, "saved": s, "diff": g - s, "bar": bar}
    return {"outside": {n: v for n, v in out.items() if not abs(v["diff"]) <= v["bar"]}, "max_abs_diff": max((abs(v["diff"]) for v in out.values()), default=math.nan)}


def summary(cfg: Config, th: dict[str, Any]) -> None:
    recs = {r["name"]: r for r in (read(p) for p in R.ITEMS.glob("*.json"))}
    items = all_items(cfg)
    D = lambda n: recs[n]["D"] if n in recs else math.nan
    out = {"experiment": R.EXPERIMENT, "missing": [it.name for it in items if it.name not in recs], "gain_checks": checks(items, recs, th),
           "stage0_passed": read(R.STAGE0).get("passed") if R.STAGE0.is_file() else None,
           "a": part_a(items, D, th), "b": part_b(cfg, items, D, th)}  # fmt: skip
    R.dump(R.OUT / "summary.json", out)
    a, b = out["a"], out["b"]
    for k in ("e254_cipher", "e255_cipher"):
        log(f"[a {k}] {a[k]['call']}; gap {a[k].get('gap', math.nan):.5f} p {a[k].get('p', math.nan):.3g}; copy {a[k]['copy']}; cipher {a[k]['cipher']}")
    log(f"[a randoms] |D| max {a['random_abs_max']:.5f} (smallest gap {th['smallest_gap']})")
    for k, t in b["tables"].items():
        log(f"[b {k}] n {t['n_states']}; " + "; ".join(f"{p} raw {t[p]['raw']} partial {t[p]['partial_given_kl']} (n {t[p]['n']})" for p in t if isinstance(t[p], dict)))
    log(f"[b] adl partial {b['adl_partial']}; other proxies repeat E262: {b['other_proxies_repeat_kl_control']}; missing {out['missing']}; "
        f"gain checks outside {list(out['gain_checks']['outside'])}")  # fmt: skip


def main(stage: str) -> int:
    P = read(R.PRED)
    cfg, th = config_from_settings(Config, P["settings"]), P["thresholds"]
    if stage == "list":
        listing(cfg)
        return 0
    require_filed(R.PRED)
    if stage == "summary":
        summary(cfg, th)
        return 0
    if not re.fullmatch(r"E\d+", R.EXPERIMENT):
        raise SystemExit(f"predictions.json experiment is {R.EXPERIMENT!r}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as run.py
    log(f"{R.EXPERIMENT} amend2 run; GPU {torch.cuda.get_device_name()}; settings {asdict(cfg)}; thresholds {th}")
    w = World(cfg, th)
    if not stage0(w, P):
        return 1
    run_items(w, cfg)
    log(f"FINISHED amend2 run; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "list", "summary"])
    code = 1
    try:
        code = main(ap.parse_args().stage)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, flush=True)
    hard_exit(code)
