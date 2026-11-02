"""readout64 (E64): does some block before the last hold E54's half-known values, specifically to the key, as a linear
readout trained on the Pile over the pair's 64 values; and does removing that direction cost the elicit fine-tunes more
of their gain than the matched ciphers? Plan: reports/plans/2026-11-02_plan_baseline-next3-draft.md, option 1 and its
"Amendment before filing (2026-11-02)". Every setting, rule and threshold: results/predictions.json (filed before the
run, read here, never edited). Rules and readout code: src/commute/readout.py.

  stage 0, alone, cheapest first; every miss exits 1 before any removal on a fine-tuned state is scored:
    inputs     windows and the twelve saved states' sha256; W0's rates, keys, values and every state's gain reproduce E54/E56
    model      W0's own possessed set M, elicit and cipher, on the 97 held-out windows, then on the fit half.
               Stops if elicit |M| < 8 or cipher |M| > 2
    positions  Pile positions per value; a value under 200 is unfit. Stops if too many are unfit
    readouts   activations cached, one readout per block trained in seconds, saved; held-out fit recorded (a miss does not
               stop: predictions.json -> stage0.fit)
    possession per block on all windows and the fit half; P1 recorded; l*; the construction check P3 (stops)
    controls   W0's drop at l* over M's windows (stops); the do-nothing (random direction) share on every state (stops)
  then the removal share A per state at l* and at the last block, P2, the branch             -> results/run.json
run.json is written after each step and every scored number is also printed. The target model only, fp32, TF32 off.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import torch

from commute import keydep as KD
from commute import readout as RO
from commute import steering as ST
from commute.data import pile_rows
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E54, E56 = HERE.parent / "keydependent_elicit" / "results", HERE.parent / "keydependent_patch" / "results"
TASKS = ("elicit", "cipher")
log = KD.log


def dump(out: dict[str, Any]) -> None:
    (RESULTS / "run.json").write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout


def stop(out: dict[str, Any], msg: str) -> NoReturn:
    out["stopped"] = msg
    dump(out)
    sys.exit(msg)


def task_of(tasks: dict[str, ST.Task], n: str) -> ST.Task:
    return tasks["elicit" if n.startswith("elicit") else "cipher"]


def keys_of(t: ST.Task, sl: slice = slice(None)) -> list[int]:
    return t.toks[sl, t.half].tolist()


def pools(t: ST.Task) -> dict[str, slice]:
    return {"all": slice(None), "fit": t.part("fit")}


def say_reading(tag: str, r: dict[str, Any], keys: tuple[int, ...]) -> None:
    ratio = [None if r["log_ratio"][k] is None else round(math.exp(r["log_ratio"][k]), 2) for k in keys]
    log(f"[{tag}] possessed {len(r['possessed'])} {r['possessed']} medians {[r['medians'][k] for k in keys]} P ratios {ratio}")


# ----------------------------------------------------------------------------- stage 0


def stage0_model(net: ST.Net, tasks: dict[str, ST.Task], vals: torch.Tensor, vidx: dict[str, dict[int, int]],
                 th: dict[str, Any], out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """W0's own possessed set M per task and pool: the reference for every possession below."""
    ref: dict[str, dict[str, Any]] = {}
    for n, t in tasks.items():
        lg = net.run(t.toks)[0][:, t.half][:, vals.to(net.device)]
        ref[n] = {p: RO.key_reading(lg[sl], keys_of(t, sl), t.keys, vidx[n], th) for p, sl in pools(t).items()}
        for p, r in ref[n].items():
            say_reading(f"W0 {n} {p}", r, t.keys)
        log(f"[W0 {n}] keys at rank > {th['old_cipher_above']} of 64: {RO.above_count(ref[n]['all']['medians'], th['old_cipher_above'])} (the plan's old cipher clause, reported)")
    out["W0"] = ref
    return ref


def stage0_positions(rows: torch.Tensor, tasks: dict[str, ST.Task], vals: torch.Tensor, ref: list[int], S: dict[str, Any],
                     th: dict[str, Any], out: dict[str, Any]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], dict[str, list[int]]]:
    """Windows per value on the fit rows (at most `cap`) and the held rows (at most `held_cap`); which keys are unfit."""
    half, nf = tasks["elicit"].half, S["fit_row_count"]
    tr, tr_y, counts = RO.next_token_windows(rows[:nf], vals, half, S["cap"], S["positions_seed"])
    he, he_y, held = RO.next_token_windows(rows[nf:], vals, half, S["held_cap"], S["held_seed"])
    n_of = dict(zip(vals.tolist(), counts, strict=True))
    unfit = {n: [k for k in t.keys if n_of[int(t.table[k])] < th["unfit_below"]] for n, t in tasks.items()}
    out["stage0"]["positions"] = {"per_value": n_of, "held_per_value": dict(zip(vals.tolist(), held, strict=True)), "unfit_keys": unfit,
                                  "fit_windows": len(tr), "held_windows": len(he)}  # fmt: skip
    for n, t in tasks.items():
        log(f"[stage0 positions {n}] per value {[n_of[int(t.table[k])] for k in t.keys]} unfit keys {unfit[n]} "
            f"values {[int(t.table[k]) for k in unfit[n]]}")
    log(f"[stage0 positions] {len(tr)} fit windows, {len(he)} held-out; held per value {held}")
    if msg := RO.unfit_stop(unfit, ref, th["max_unfit"], th["frac_num"], th["frac_den"]):
        stop(out, msg)
    return (tr, tr_y), (he, he_y), unfit


def w0_at_training_prior(net: ST.Net, tasks: dict[str, ST.Task], vals: torch.Tensor, vidx: dict[str, dict[int, int]], th: dict[str, Any],
                         cap: int, out: dict[str, Any]) -> None:
    """Reported, never scored: W0's possessed set after moving its logits to the readout's training prior (a value over
    the cap loses log(count / cap)), so that a miss of P3 can be read: the prior, or the readout."""
    n_of = out["stage0"]["positions"]["per_value"]
    shift = torch.tensor([min(0.0, math.log(cap / max(n_of[v], 1))) for v in vals.tolist()], device=net.device, dtype=torch.float32)
    out["W0_training_prior"] = {}
    for n, t in tasks.items():
        lg = net.run(t.toks)[0][:, t.half][:, vals.to(net.device)] + shift
        out["W0_training_prior"][n] = RO.key_reading(lg, keys_of(t), t.keys, vidx[n], th)
        say_reading(f"W0 at the training prior {n} all, reported", out["W0_training_prior"][n], t.keys)


def stage0_readouts(net: ST.Net, fit: tuple[torch.Tensor, torch.Tensor], held: tuple[torch.Tensor, torch.Tensor], vals: torch.Tensor,
                    half: int, P: dict[str, Any], large: Path, out: dict[str, Any]) -> RO.Probes:
    """Cache, train and save; the fit: the last block's held-out 64-way loss <= W0's own restricted loss + fit_margin."""
    th, S = P["thresholds"], P["settings"]
    assert bool((fit[0][:, half + 1] == vals[fit[1]]).all()) and bool((held[0][:, half + 1] == vals[held[1]]).all())
    h, w0 = RO.cache64(net, fit[0], vals, half)
    y = fit[1].to(h.device)
    log(f"[stage0 readouts] cached {h.shape}; W0's 64-way loss on the fit windows {RO.xent64(w0[None], y)[0]:.4f}; chance {math.log(len(vals)):.4f}")
    try:
        probes = RO.train_readouts64(h, y, w0, S["readout_seed"], th["early_checks"], log)
    except RO.ReadoutStop as e:
        stop(out, str(e))
    torch.save(probes.state_dict(), large / "readouts.pt")
    with torch.no_grad():
        hh, wh = RO.cache64(net, held[0], vals, half)
        yh = held[1].to(hh.device)
        rec = {"readout": RO.xent64(probes(hh), yh), "model": RO.xent64(wh[None], yh)[0], "held_in": RO.xent64(probes(h), y),
               "model_held_in": RO.xent64(w0[None], y)[0], "sha256": KD.sha256(large / "readouts.pt")}
    rec["over_model"] = rec["readout"][RO.LAST] - rec["model"]
    rec["pass"] = rec["over_model"] <= th["fit_margin"]
    out["stage0"]["fit"] = rec
    log(f"[stage0 fit] held-out loss per block {rec['readout']} model {rec['model']:.6f} over_model {rec['over_model']:.6f} pass {rec['pass']}; "
        f"held-in {rec['held_in']} model {rec['model_held_in']:.6f}")
    return probes


def possession(net: ST.Net, probes: RO.Probes, tasks: dict[str, ST.Task], ref: dict[str, dict[str, Any]], vidx: dict[str, dict[int, int]],
               unfit: dict[str, list[int]], th: dict[str, Any], out: dict[str, Any]) -> int | None:
    """The readouts through the same rule, per block and pool, against M; P1 on all windows, l* on the fit half, P3."""
    num, den, rec = th["frac_num"], th["frac_den"], {}
    for n, t in tasks.items():
        lg, rec[n] = RO.readout_logits64(net, probes, t), {}
        for p, sl in pools(t).items():
            rec[n][p] = []
            for b in range(len(RO.RESID)):
                r = RO.key_reading(lg[b, sl], keys_of(t, sl), t.keys, vidx[n], th, unfit[n])
                M = ref["elicit"][p]["possessed"]
                r["verdict"] = RO.relative_possession(r["possessed"], M, num, den) if n == "elicit" else RO.cipher_possession(r["possessed"], len(M), num, den)
                rec[n][p].append(r)
                say_reading(f"readout {n} {p} block {b}", r, t.keys)
                log(f"[possession {n} {p} block {b}] {json.dumps(r['verdict'])}")
    flags = lambda n, p: [r["verdict"]["possessed"] for r in rec[n][p]]
    last = lambda n: rec[n]["all"][RO.LAST]
    out["possession"] = rec
    out["P1"] = RO.p1(flags("elicit", "all"), flags("cipher", "all"))
    out["P3"] = RO.construction(last("elicit")["verdict"], last("cipher")["verdict"], ref["cipher"]["all"]["possessed"], th["max_cipher_M"])
    out["l_star"] = RO.earliest(flags("elicit", "fit"))
    for k in ("P1", "P3"):
        log(f"[score {k}] {json.dumps(out[k])}")
    log(f"[old clause, reported] last-block cipher keys at rank > {th['old_cipher_above']} of 64: {RO.above_count(last('cipher')['medians'], th['old_cipher_above'])}")
    log(f"[l*] {out['l_star']} (earliest block before the last possessing the elicit map on the fit half)")
    return out["l_star"]


def idx_of(t: ST.Task, vals: torch.Tensor) -> torch.Tensor:
    """(N,) each window's target as an index into the readout's outputs."""
    return (t.tgt(t.toks)[:, None] == vals).nonzero()[:, 1]


def say(tag: str, r: dict[str, float]) -> None:
    keys = ("A_lo", "A_lo_raw", "A_nats", "G_logodds", "G_score", "lo_drop", "lo_drop_W0", "nats_lost", "median_drop", "P_tuned_mean", "P_tuned_geo")
    log(f"[{tag}] " + " ".join(f"{k} {r[k]:.6g}" for k in keys))


def controls(net: ST.Net, probes: RO.Probes, tasks: dict[str, ST.Task], dws: dict[str, ST.DW], block: int, vals: torch.Tensor,
             ref: dict[str, dict[str, Any]], P: dict[str, Any], out: dict[str, Any]) -> None:
    """W0's drop under the removal over the windows of M's keys (the scored control; all windows and the cipher beside
    it), the random direction on W0, each target row's geometry, and the do-nothing share on every state."""
    S, th, d = P["settings"], P["thresholds"], probes.w.shape[1]
    w0: dict[str, Any] = {}
    for n, t in tasks.items():
        sc, U = t.part("score"), RO.target_dirs(probes, block, idx_of(t, vals))
        drops = RO.w0_drops(net, t, block, U)
        w0[f"{n}_M"] = RO.w0_control(drops, keys_of(t, sc), ref["elicit"]["all"]["possessed"], th["w0_min_drop"])
        w0[f"{n}_all"] = RO.w0_control(drops, keys_of(t, sc), t.keys, th["w0_min_drop"])
        rand = RO.w0_drops(net, t, block, RO.random_dirs(t.tgt(t.toks), d, S["random_seed"]))
        w0[f"{n}_random"] = RO.w0_control(rand, keys_of(t, sc), t.keys, th["w0_min_drop"])
        log(f"[control W0 {n} block {block}] over M's keys {json.dumps(w0[f'{n}_M'])} all windows {json.dumps(w0[f'{n}_all'])} random {json.dumps(w0[f'{n}_random'])}")
    w0["pass"] = w0["elicit_M"]["pass"]
    out["W0_control"] = w0
    out["rows"] = {f"block{b}": RO.column_geometry(probes, b, list(range(len(vals)))) for b in sorted({block, RO.LAST})}
    for k, g in out["rows"].items():
        log(f"[rows {k}] cos_mean {[round(v['cos_mean'], 3) for v in g.values()]} norm {[round(v['norm'], 3) for v in g.values()]}")
    rand_states = {}
    for n, dW in dws.items():
        t = task_of(tasks, n)
        rand_states[n] = RO.ablation(net, t, dW, block, RO.random_dirs(t.tgt(t.toks), d, S["random_seed"]), th["floor_min"])
        say(f"do-nothing {n} block {block}", rand_states[n])
    out["random"] = {**rand_states, **RO.do_nothing({n: r["A_lo"] for n, r in rand_states.items()}, th["random_tol"])}
    log(f"[control] W0 pass {w0['pass']}; do-nothing max |A_rand| {out['random']['max_abs']} pass {out['random']['pass']}")


def score(net: ST.Net, probes: RO.Probes, tasks: dict[str, ST.Task], dws: dict[str, ST.DW], block: int, vals: torch.Tensor,
          void: str | None, P: dict[str, Any], out: dict[str, Any]) -> None:
    """A per state at `block`, and the same removal at the last block on every state (reported; expected ~1)."""
    th, A = P["thresholds"], {}
    out["removal"] = {"block": block, "scored": void is None, "states": {}}
    out["last_block"] = {"block": RO.LAST, "scored": False, "states": {}}
    for n, dW in dws.items():
        t = task_of(tasks, n)
        for rec, b in ((out["removal"], block), (out["last_block"], RO.LAST)):
            rec["states"][n] = RO.ablation(net, t, dW, b, RO.target_dirs(probes, b, idx_of(t, vals)), th["floor_min"])
            say(f"removal {n} block {b}", rec["states"][n])
        A[n] = out["removal"]["states"][n]["A_lo"]
        dump(out)
    out["P2"] = RO.p2(A, list(RO.SEEDS), list(RO.OWN_SEEDS), th["min_gap"], th["own_tol"], void)
    out["branch"] = RO.branch64(out["P1"], out["P2"], out["stage0"]["fit"]["pass"])
    log(f"[score P2] {json.dumps(jsonable(out['P2']))}")
    log(f"[branch] {out['branch']}")


# ----------------------------------------------------------------------------- main


def run(net: ST.Net, tasks: dict[str, ST.Task], P: dict[str, Any], large: Path) -> None:
    S, th = P["settings"], P["thresholds"]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").strip(), "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
                           "stage0": {"inputs": {}}}
    dws = RO.check_inputs(net, tasks, E54, E56, P["stage0"]["thresholds"], out["stage0"]["inputs"], log)
    dump(out)
    if not out["stage0"]["inputs"]["pass"]:
        stop(out, "[stage0 inputs] FAIL: stop and tell the user")
    vals = RO.pair_values(tasks)
    vidx = {n: {k: vals.tolist().index(int(t.table[k])) for k in t.keys} for n, t in tasks.items()}
    ref = stage0_model(net, tasks, vals, vidx, th, out)
    dump(out)
    Me, Mc = ref["elicit"]["all"]["possessed"], ref["cipher"]["all"]["possessed"]
    if msg := RO.model_stop(len(Me), len(Mc), th["min_M"], th["max_cipher_M"]):
        stop(out, msg)
    rows = pile_rows(*S["train_rows"], log=log)
    fit, held, unfit = stage0_positions(rows, tasks, vals, Me, S, th, out)
    w0_at_training_prior(net, tasks, vals, vidx, th, S["cap"], out)
    dump(out)
    probes = stage0_readouts(net, fit, held, vals, tasks["elicit"].half, P, large, out)
    dump(out)
    if not out["stage0"]["fit"]["pass"]:  # recorded, not a stop: predictions.json -> stage0.fit
        log("[stage0 fit] over the filed margin: recorded; a P1 elicit-clause or P2 miss reads unreadable")
    l_star = possession(net, probes, tasks, ref, vidx, unfit, th, out)
    dump(out)
    if out["P3"]["status"] != "pass":
        stop(out, "[stage0 P3] FAIL: the last-block readout does not reproduce the model's possessed set, or the cipher is possessed; stop and tell the user")
    block = RO.LAST if l_star is None else l_star
    controls(net, probes, tasks, dws, block, vals, ref, P, out)
    dump(out)
    if not out["random"]["pass"]:
        stop(out, "[stage0 do-nothing] FAIL: a random direction moves a share over the filed bound; stop and tell the user")
    if l_star is not None and not out["W0_control"]["pass"]:
        out["P2"] = RO.p2({}, list(RO.SEEDS), list(RO.OWN_SEEDS), 0, 0, "W0's positive control failed")
        out["branch"] = RO.branch64(out["P1"], out["P2"], out["stage0"]["fit"]["pass"])
        stop(out, "[stage0 W0 control] FAIL: the direction is decodable but W0 does not use it; P2 void; Jain's reverse fine-tune is a later ask; tell the user")
    out["stage0"]["pass"] = True
    log("[stage0] passed")
    score(net, probes, tasks, dws, block, vals, None if l_star is not None else "no block before the last possesses (fit half)", P, out)
    dump(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="where the readout weights go")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    rel = str(PRED.relative_to(REPO))
    if not git("ls-files", rel).strip() or git("diff", "HEAD", "--", rel):
        sys.exit(f"{rel} is not committed unchanged: commit it first")
    P = json.loads(PRED.read_text())
    S = P["settings"]
    assert (S["iters"], S["lr"], tuple(S["decay_at"]), S["decay"], S["batch"]) == (RO.ITERS, RO.LR, RO.DECAY_AT, RO.DECAY, RO.BATCH)
    assert (S["window"], list(S["sites"])) == (RO.WIN, list(RO.RESID)), "predictions.json disagrees with the code"
    a.large.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net(a.device)
    tasks = {n: ST.load_task(net, E54 / "large" / f"windows_{n}.pt", n, n) for n in TASKS}
    run(net, tasks, P, a.large.resolve())


if __name__ == "__main__":
    main()
