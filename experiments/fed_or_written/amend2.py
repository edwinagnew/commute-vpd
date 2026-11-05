"""E100 Amendment 2, job A: the sum of single reverts against a written reference on each state's own site, and two checks
on what "fed" reads. No training. Plan: reports/plans/2026-11-05_plan_e100-amend2.md; it adopts Fable's advice as written,
reports/reviews/2026-11-05_review_e100-next-fable-advice.md. Rules, bars and leans: results/predictions_amend2.json
(committed first; the run refuses otherwise; never edited) and `commute.fed_rules` (Amendment 2 sections). The plant:
plants.py. E100's run.py, amend1.py and their files are read, never written.

  stage 0, alone: copy-24 seed 0's sum re-read (0.39511) and the check's 4-MLP plant on its site (0.6388), each within
           1e-3 (`~/commute-data/scratch/mechanism-detection-lead/e100-check/spread_plant.json`). Exit 1 on a miss.
  stage 1: every row of table_amend1.json (E100's 35 rows and Amendment 1's 20): a plant with the state's own per-part
           norm profile at its gain on its site, W its sum, rho = the state's sum / W. Bars from the copy and written rows;
           the site-relative rule is kept only if its gap is at least the fixed bars' 0.22, else the fixed bars stand.
           Each row's reading under the kept rule, the fixed reading beside. Writes `kept_rule`, which job B needs.
  stage 2: (a) E62's in-context copy seed 0 (the control) and emoticon's two seeds with E62's prefix-scored copying heads
           at their natural-text means (`copying.ablated`): the gain kept, and emoticon's windows with the target earlier;
           (b) E99's class map and shuffled twin, seeds 0-1: the sum, W, rho and the reading on their own sites.
results/amend2.json is written after every state; a resubmit re-runs stage 0 and skips every state already there.
"STAGE <n> DONE" is printed when a stage is complete; every number is printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import copying as CP
from commute import fed_rules as FR
from commute import locator as L
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256

JB = load_experiment("locator_table", "job")
RUN = load_experiment("fed_or_written", "run")
A1 = load_experiment("fed_or_written", "amend1")
PL = load_experiment("fed_or_written", "plants")
sites = JB.sites
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT, TABLE = RESULTS / "predictions_amend2.json", RESULTS / "amend2.json", RESULTS / "table_amend1.json"
E99_RESULTS = sites.EXP / "definition_verify" / "results"
CLASS_WINDOWS = sites.EXP / "open_pair" / "results" / "large" / "amend7"
HEADS_FILE = sites.EXP / "induction_cue" / "results" / "large" / "heads.pt"
dump, say, load, stop = JB.dump, JB.say, JB.load, JB.stop


class Sites:
    """W0's reads (no part outputs: the sum and the plant need none) and the plant's writes on one site at a time."""

    def __init__(self, J: Any) -> None:
        self.J, self.key = J, ""
        self.base: L.Base | None = None
        self.X: Any = None  # plants.Writes

    def at(self, key: str) -> tuple[L.Base, Any]:
        if key != self.key:
            self.base = self.X = None
            torch.cuda.empty_cache()
            s = sites.site(self.J.F, key)
            self.base = L.Base(s, L.read(self.J.net, s), {}, {}, {})
            self.X = PL.kv_writes(self.J.net, sites.kv(self.J.F, sites.KV[key])[1], s.batch) if key in sites.KV else PL.site_writes(self.J.net, s)
            self.key = key
        assert self.base is not None and self.X is not None
        return self.base, self.X


def row_state(J: Any, fresh: dict[str, tuple[str, int, Path]], task: str, seed: int) -> tuple[str, ST.DW]:
    """(site key, dW) of a table row: E99's poles as E100 read them; E97's tasks; Amendment 1's fresh seeds."""
    if task in RUN.POLES:
        return "e62_copy", RUN.e99_dW(*RUN.e99_states(J, task)[seed])
    T = sites.TASKS[task]
    return T.site, J.F.state(T.states[seed] if seed in T.states else fresh[f"{task} s{seed}"][2])


def reference(J: Any, S: Sites, key: str, dW: ST.DW, G: float, s: float) -> dict[str, Any]:
    """The state's profile-matched plant on its site, and rho."""
    base, X = S.at(key)
    return against(J, base, X, dW, G, s)


def against(J: Any, base: L.Base, X: Any, dW: ST.DW, G: float, s: float) -> dict[str, Any]:
    """The plant matched to the state's per-part norm profile and gain, W, its status and rho."""
    sh = PL.part_shares(dW)
    p = PL.plant(J.net, base, X, PL.profile(X, sh), G, J.th)
    return {"plant": p, "W": p["W"], "status": p["status"], "rho": FR.rho(s, p["W"], p["status"]), "state_shares": sh,
            "state_mlp_share": sum(v for n, v in sh.items() if n.startswith("mlp"))}  # fmt: skip


# ----------------------------------------------------------------------------- stages


def stage0(J: Any, S: Sites, out: dict[str, Any]) -> None:
    th = J.th
    base, X = S.at("kv_copy24")
    r = PL.read_sum(J.net, base, J.F.state(sites.TASKS["T13 copy-24"].states[0]))
    p = PL.plant(J.net, base, X, {f"mlp{b}": 1.0 for b in range(L.BLOCKS)}, r["G"], th)
    rec = {"sum": r["reverts_sum"], "G": r["G"], "plant": p}
    rec["misses"] = [f"{n} {v} off {ref} by more than {th['stage0_tol']}" for n, v, ref in
                     (("sum", rec["sum"], th["stage0_sum_ref"]), ("4-MLP plant W", p["W"], th["stage0_plant_ref"]))
                     if not abs(v - ref) <= th["stage0_tol"]]  # fmt: skip
    rec["status"] = "miss" if rec["misses"] else "pass"
    out["stage0"] = rec
    dump(OUT, out)
    say("stage0", {k: v for k, v in rec.items() if k != "plant"} | {"plant": {k: p[k] for k in ("alpha", "G", "match_status", "W")}})
    if rec["misses"]:
        stop(OUT, out, f"stage 0: {rec['misses']}")


def stage1(J: Any, S: Sites, out: dict[str, Any]) -> None:
    th, rows = J.th, load(TABLE)["table"]
    fresh = A1.fresh(J.F)
    done = out.setdefault("stage1", {}).setdefault("rows", {})
    order = sorted(rows, key=lambda r: (row_site(r["task"]) in J.P["settings"]["slow_sites"], row_site(r["task"]), r["task"], r["seed"]))
    for r in order:
        name = f"{r['task']} s{r['seed']}"
        if name in done:
            continue
        key, dW = row_state(J, fresh, r["task"], r["seed"])
        rec = {"task": r["task"], "seed": r["seed"], "site": key, "G": r["G"], "sum": r["reverts_sum"], "fresh": r["fresh"]}
        rec |= reference(J, S, key, dW, r["G"], r["reverts_sum"])
        done[name] = rec
        dump(OUT, out)
        say(f"row {name}", brief(rec))
    pool = lambda tasks: [v["rho"] for v in done.values() if v["task"] in tasks]
    b = FR.site_bars(pool(J.P["settings"]["copy_rows"]), pool(J.P["settings"]["written_rows"]), th["fixed_gap"])
    out["stage1"]["bars"] = b
    for v in done.values():
        v |= FR.rho_reading(v["G"], v["sum"], v["status"], v["rho"], b, th)
        v["reading_fixed"] = FR.sum_reading(v["G"], v["sum"], th["fed_bar"], th["written_bar"], th["min_gain"])
    out["kept_rule"] = b["rule"]
    dump(OUT, out)
    say("stage1 bars", b)
    for n, v in done.items():
        log(f"[reading {n}] sum {v['sum']:.4f} W {v['W']:.4f} rho {v['rho']:.4f} ({v['status']}) -> {v['reading']} [{v['basis']}]; fixed {v['reading_fixed']}")


def row_site(task: str) -> str:
    return "e62_copy" if task in RUN.POLES else sites.TASKS[task].site


def brief(rec: dict[str, Any]) -> dict[str, Any]:
    p = rec["plant"]
    return {k: rec[k] for k in ("site", "G", "sum", "W", "rho", "status", "state_mlp_share") if k in rec} | {
        "plant": {k: p[k] for k in ("alpha", "G", "match_status", "seconds") if k in p}}


def heads_off(J: Any, S: Sites, key: str, dW: ST.DW, heads: list[CP.Head], means: torch.Tensor) -> dict[str, Any]:
    """G with the heads off (the state and W0 both at the heads' means), and G intact, on the site."""
    base, _ = S.at(key)
    with CP.ablated(J.net, heads, means):
        off = (L.read(J.net, base.site, dW) - L.read(J.net, base.site))[:, 0].mean()
    on = (L.read(J.net, base.site, dW) - base.lo0)[:, 0].mean()
    return {"G_on": float(on), "G_off": float(off), "kept": FR.kept_share(float(on), float(off))}


def stage2_copying(J: Any, S: Sites, out: dict[str, Any]) -> None:
    th, st = J.th, J.P["settings"]["copying"]
    rec: dict[str, Any] = {"heads_sha256": sha256(HEADS_FILE)}
    out.setdefault("stage2", {})["copying"] = rec
    hp = torch.load(HEADS_FILE, weights_only=False)
    heads, means = [tuple(h) for h in st["heads"]], hp["means"].to(J.net.device)
    task, seed = st["control"]
    T = sites.TASKS[task]
    rec["control"] = heads_off(J, S, T.site, J.F.state(T.states[seed]), heads, means)
    say(f"copying control {task} s{seed}", rec["control"])
    T = sites.TASKS["T1 emoticon"]
    base, _ = S.at(T.site)
    s = base.site
    earlier = (s.toks[:, : int(s.pos[0]) + 1] == s.tgt[:, None]).any(1)  # one read per window, at `half`
    rec["frac_earlier"] = float(earlier.double().mean())
    rec["n_windows"] = len(s.toks)
    for seed in (0, 1):
        r = heads_off(J, S, T.site, J.F.state(T.states[seed]), heads, means)
        r["reading"] = FR.copying_reading(r["kept"], rec["frac_earlier"], rec["control"]["kept"], th)
        rec[f"T1 emoticon s{seed}"] = r
        say(f"copying T1 emoticon s{seed}", r | {"frac_earlier": rec["frac_earlier"]})
    rec["reading"] = FR.task_reading([rec[f"T1 emoticon s{s}"]["reading"] for s in (0, 1)])
    dump(OUT, out)


def class_state(name: str, rec: dict[str, Any]) -> ST.DW:
    path = Path(rec["e99_large"]).expanduser() / rec["trained"][name]["file"]
    if sha256(path) != rec["trained"][name]["sha256"]:
        raise ValueError(f"{path}: sha256 differs from E99's stage4_class.json")
    return {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}


def stage2_class(J: Any, out: dict[str, Any]) -> None:
    """E99's class map and twin on their own sites (held-out windows of the trained keys, E90's kv_site)."""
    th, e99 = J.th, load(E99_RESULTS / "stage4_class.json")
    e99["e99_large"] = J.P["settings"]["e99_large"]
    done = out["stage2"].setdefault("class", {})
    b = out["stage1"]["bars"]
    for m in ("class_map", "class_shuffled"):
        path = CLASS_WINDOWS / f"windows_{m}_trained.pt"
        if sha256(path) != e99["sha256"][path.name]:
            raise ValueError(f"{path}: sha256 differs from E99's record")
        s, w = sites.kv(J.F, path)
        base = L.Base(s, L.read(J.net, s), {}, {}, {})
        X = PL.kv_writes(J.net, w, s.batch)
        for seed in (0, 1):
            name = f"{m}_s{seed}"
            if name in done:
                continue
            dW = class_state(name, e99)
            r = PL.read_sum(J.net, base, dW)
            rec = {"G": r["G"], "sum": r["reverts_sum"], "own": r["own"]} | against(J, base, X, dW, r["G"], r["reverts_sum"])
            rec |= FR.rho_reading(rec["G"], rec["sum"], rec["status"], rec["rho"], b, th)
            done[name] = rec
            dump(OUT, out)
            say(f"class {name}", brief(rec) | {"reading": rec["reading"], "basis": rec["basis"]})
        del X
        torch.cuda.empty_cache()


def outcome(P: dict[str, Any], out: dict[str, Any]) -> None:
    b, c, cl = out["stage1"]["bars"], out["stage2"]["copying"], out["stage2"]["class"]
    leans = P["expected"]["leans"]
    got = {"gap_at_least_fixed": b["rule"] == FR.SITE_RELATIVE,
           "emoticon_keeps_half": all(c[f"T1 emoticon s{s}"]["kept"] > P["thresholds"]["kept_max"] for s in (0, 1)),
           "class_both_written": all(v["reading"] == FR.WRITTEN for v in cl.values())}  # fmt: skip
    out["outcome"] = {"kept_rule": b["rule"], "gap": b["gap"], "emoticon": c["reading"], "class": {k: v["reading"] for k, v in cl.items()},
                      "leans": {k: {"lean": leans[k], "held": got[k]} for k in got}}  # fmt: skip
    dump(OUT, out)
    say("outcome", out["outcome"])


# ----------------------------------------------------------------------------- main


def timed(k: int, f: Any, *a: Any) -> None:
    t0 = time.monotonic()
    f(*a)
    log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
    print(f"STAGE {k} DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    P = filed()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E100
    net = ST.Net(args.device)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {head} predictions {sha256(PRED)}")
    J = JB.Job(net, sites.Files(), P, {})
    S = Sites(J)
    out = load(OUT)
    out |= {"git_head": head, "predictions_sha256": sha256(PRED)}
    timed(0, stage0, J, S, out)
    timed(1, stage1, J, S, out)
    timed(2, lambda: (stage2_copying(J, S, out), stage2_class(J, out)))
    outcome(P, out)
    out["files"] = J.F.seen
    dump(OUT, out)
    hard_exit(0)


def filed() -> dict[str, Any]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(PRED.read_text())


if __name__ == "__main__":
    main()
