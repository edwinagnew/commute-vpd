"""jain_readout (E59): does the original 67M model hold E54's elicit map (and not the cipher) in a Jain et al. Def 1
linear readout trained on Pile text, before the last block; and does each fine-tune's gain run through that readout's
direction? Plan: reports/plans/2026-10-30_plan_baseline-next2-draft.md (approved by the user). Every setting, rule and
threshold: results/predictions.json (filed before the run, read here, never edited); readout and ablation code in
src/commute/readout.py.

  run.py   stage 0 (any miss exits 1, before any ablation of a fine-tuned state is scored):
             inputs: windows and the twelve saved weight changes' sha256 against E54/E56; W0's rates, keys and values
               reproduce E54's stage1.json; every state's gain G reproduces stage2.json / job2.json
             readouts: one per block, trained on Pile (saved to <large>/readouts.pt); held-out Pile loss per block and
               the model's own; the last block's within the filed margin of the model's
             possession per block and task (all held-out windows; fit half for l*); P1 recorded; P3 (stop on a miss)
             W0's positive control at l* (median log P drop of the elicit target); the do-nothing (random direction)
               share on every state
           then the ablation share A per state at l*, P2 and the decision tree          -> results/run.json
Results are written after every step; every scored number is also printed. The target model only, fp32, TF32 off.
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

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E54, E56 = HERE.parent / "keydependent_elicit" / "results", HERE.parent / "keydependent_patch" / "results"
SEEDS, OWN = ["s0", "s1", "s2", "s3", "s4"], ["s0", "s1"]  # cipher_own s2-s4 were not saved by E56
log = KD.log


def dump(out: dict[str, Any]) -> None:
    fin = lambda o: (str(o) if isinstance(o, float) and not math.isfinite(o) else
                     {str(k): fin(v) for k, v in o.items()} if isinstance(o, dict) else
                     [fin(v) for v in o] if isinstance(o, list | tuple) else o)  # fmt: skip
    (RESULTS / "run.json").write_text(json.dumps(fin(out), indent=1, sort_keys=True))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout


def stop(out: dict[str, Any], msg: str) -> NoReturn:
    out["stopped"] = msg
    dump(out)
    sys.exit(msg)


def task_of(tasks: dict[str, ST.Task], n: str) -> ST.Task:
    return tasks["elicit" if n.startswith("elicit") else "cipher"]


def refs() -> dict[str, dict[str, Any]]:
    """The twelve saved states: path, filed sha256 and gain G (E54 stage2.json; E56 job2.json)."""
    s2, j2 = json.loads((E54 / "stage2.json").read_text()), json.loads((E56 / "job2.json").read_text())
    out = {f"{k}_{s}": (E54, s2) for k in ("elicit", "cipher", "cipher_own") for s in OWN}
    out |= {f"{k}_{s}": (E56, j2) for k in ("elicit", "cipher") for s in SEEDS[2:]}
    return {n: {"path": d / "large" / f"{n}.pt", "sha256": j["finetunes"][n]["sha256"], "G": j["states"][n]["G"]}
            for n, (d, j) in out.items()}  # fmt: skip


# ----------------------------------------------------------------------------- stage 0


def stage0_inputs(net: ST.Net, tasks: dict[str, ST.Task], th: dict[str, float], out: dict[str, Any]) -> dict[str, ST.DW]:
    s1, rec, dws = json.loads((E54 / "stage1.json").read_text()), out["stage0"]["inputs"], {}
    for n, t in tasks.items():
        rate = KD.geo_rate(t.lp0)
        rec[n] = {"sha256": KD.sha256(E54 / "large" / f"windows_{n}.pt"), "rate": rate, "rate_rel": rate / s1[n]["rate"] - 1,
                  "keys_values": list(t.keys) == s1[n]["keys"] and [int(t.table[x]) for x in t.keys] == s1[n]["values"]}
        rec[n]["ok"] = bool(rec[n]["sha256"] == s1["sha256"][n] and rec[n]["keys_values"] and abs(rec[n]["rate_rel"]) <= th["rate_rel_tol"])
        log(f"[stage0 {n}] W0 rate {rate:.10e} rel {rec[n]['rate_rel']:.1e} ok {rec[n]['ok']}")
    for n, r in refs().items():
        dws[n] = torch.load(r["path"], weights_only=True)
        g = ST.gains(net, task_of(tasks, n), dws[n])
        rec[n] = {**g, "sha256": KD.sha256(r["path"]), "dG": g["G"] - r["G"]}
        rec[n]["ok"] = bool(rec[n]["sha256"] == r["sha256"] and abs(rec[n]["dG"]) <= th["reproduce_tol"])
        log(f"[stage0 {n}] G {g['G']:.12g} G_score {g['G_score']:.12g} dG {rec[n]['dG']:.1e} sha {rec[n]['sha256'][:12]} ok {rec[n]['ok']}")
    rec["pass"] = all(v["ok"] for v in rec.values() if isinstance(v, dict))
    return dws


def stage0_readouts(net: ST.Net, S: dict[str, Any], th: dict[str, float], large: Path, out: dict[str, Any]) -> RO.Probes:
    train = pile_rows(*S["train_rows"], log=log)
    try:
        probes = RO.train_readouts(net, train, S["readout_seed"], th["readout_max_minutes"], log)
    except RO.ReadoutStop as e:  # review L2: the reason goes into run.json
        stop(out, str(e))
    torch.save(probes.state_dict(), large / "readouts.pt")
    loss = RO.held_out_loss(net, probes, pile_rows(*S["held_rows"], log=log), S["held_windows"], S["held_seed"])
    loss["gap_last"] = loss["readout"][RO.LAST] - loss["model"]
    out["stage0"]["readouts"] = {**loss, "sha256": KD.sha256(large / "readouts.pt"), "pass": loss["gap_last"] <= th["loss_gap_max"]}
    log(f"[stage0 readouts] held-out loss per block {loss['readout']} model {loss['model']:.6f} gap_last {loss['gap_last']:.6f} "
        f"pass {out['stage0']['readouts']['pass']}")
    return probes


def possession(net: ST.Net, probes: RO.Probes, tasks: dict[str, ST.Task], P: dict[str, Any], out: dict[str, Any]) -> int | None:
    """Ranks, per-key medians and possession per block on all held-out windows and on the fit half; P1 and P3; l*."""
    th, rec = P["thresholds"], {}
    for n, t in tasks.items():
        ranks, keys = RO.target_ranks(net, probes, t), t.toks[:, t.half]
        rec[n] = {}
        for part, sl in (("all", slice(None)), ("fit", t.part("fit"))):
            med = [RO.key_medians(ranks[b, sl].tolist(), keys[sl].tolist(), list(t.keys)) for b in range(len(RO.RESID))]
            poss = [RO.possession(m, th["possess_rank"], th["possess_keys"], len(t.keys)) for m in med]
            rec[n][part] = {"medians": [{str(k): v for k, v in m.items()} for m in med], "possession": poss}
            for b in range(len(RO.RESID)):
                log(f"[possession {n} {part} block {b}] count {poss[b]['count']}/{poss[b]['covered']} possessed {poss[b]['possessed']} "
                    f"medians {[med[b][k] for k in t.keys]}")
        rec[n]["ranks"] = ranks.tolist()
    flags = lambda n, part: [p["possessed"] for p in rec[n][part]["possession"]]
    last = lambda n: {int(k): v for k, v in rec[n]["all"]["medians"][RO.LAST].items()}
    out["possession"] = rec
    out["P1"] = RO.p1(flags("elicit", "all"), flags("cipher", "all"))
    out["P3"] = RO.p3(last("elicit"), last("cipher"), tuple(th["p3_band"]), th["possess_keys"], th["p3_cipher_above"])
    out["l_star"] = RO.earliest(flags("elicit", "fit"))
    for k in ("P1", "P3"):
        log(f"[score {k}] {json.dumps(out[k])}")
    log(f"[l*] {out['l_star']} (earliest block before the last possessing the elicit map on the fit half)")
    return out["l_star"]


def say(tag: str, r: dict[str, float]) -> None:
    keys = ("A_lo", "A_lo_raw", "A_nats", "G_logodds", "G_score", "lo_drop", "lo_drop_W0", "nats_lost", "median_drop", "P_tuned_mean", "P_tuned_geo")
    log(f"[{tag}] " + " ".join(f"{k} {r[k]:.6g}" for k in keys))


def controls(net: ST.Net, probes: RO.Probes, tasks: dict[str, ST.Task], dws: dict[str, ST.DW], block: int,
             P: dict[str, Any], out: dict[str, Any]) -> None:
    """W0's positive control (target and random direction, both tasks), the random-direction share on every state,
    and each target value's readout column geometry at the ablation block and the last block (review L1)."""
    S, th, d = P["settings"], P["thresholds"], probes.w.shape[1]
    w0: dict[str, Any] = {}
    for n, t in tasks.items():
        tgt = t.tgt(t.toks)
        for kind, U in (("target", RO.target_dirs(probes, block, tgt)), ("random", RO.random_dirs(tgt, d, S["random_seed"]))):
            w0[f"{n}_{kind}"] = RO.ablation(net, t, None, block, U, th["floor_min"])
            say(f"control W0 {n} {kind} block {block}", w0[f"{n}_{kind}"])
    w0["pass"] = w0["elicit_target"]["median_drop"] >= th["w0_min_drop"]
    out["columns"] = {f"{n}_block{b}": RO.column_geometry(probes, b, sorted({int(t.table[x]) for x in t.keys}))
                      for n, t in tasks.items() for b in sorted({block, RO.LAST})}  # fmt: skip
    for k, g in out["columns"].items():
        log(f"[columns {k}] cos_mean {[round(v['cos_mean'], 4) for v in g.values()]} norm {[round(v['norm'], 4) for v in g.values()]}")
    rand = {}
    for n, dW in dws.items():
        t = task_of(tasks, n)
        rand[n] = RO.ablation(net, t, dW, block, RO.random_dirs(t.tgt(t.toks), d, S["random_seed"]), th["floor_min"])
        say(f"do-nothing {n} block {block}", rand[n])
    out["W0_control"], out["random"] = w0, {**rand, **RO.do_nothing({n: r["A_lo"] for n, r in rand.items()}, th["random_tol"])}
    log(f"[control] W0 pass {w0['pass']}; do-nothing max |A_rand (A_lo)| {out['random']['max_abs']} pass {out['random']['pass']}")


def score(net: ST.Net, probes: RO.Probes, tasks: dict[str, ST.Task], dws: dict[str, ST.DW], block: int,
          void: str | None, P: dict[str, Any], out: dict[str, Any]) -> None:
    """A per state at `block` (P2 reads A_lo); the same removal at the last block on every state, reported (review M2:
    the positive control that A can read high, expected ~1)."""
    th, A = P["thresholds"], {}
    out["ablation"] = {"block": block, "scored": void is None, "states": {}}
    out["last_block"] = {"block": RO.LAST, "scored": False, "states": {}}
    for n, dW in dws.items():
        t = task_of(tasks, n)
        for rec_, b in ((out["ablation"], block), (out["last_block"], RO.LAST)):
            rec_["states"][n] = RO.ablation(net, t, dW, b, RO.target_dirs(probes, b, t.tgt(t.toks)), th["floor_min"])
            say(f"ablation {n} block {b}", rec_["states"][n])
        A[n] = out["ablation"]["states"][n]["A_lo"]
        dump(out)
    out["P2"] = RO.p2(A, SEEDS, OWN, th["min_gap"], th["own_tol"], void)
    out["branch"] = RO.branch(out["P1"], out["P2"])
    log(f"[score P2] {json.dumps(out['P2'], default=str)}")
    log(f"[branch] {out['branch']}")


# ----------------------------------------------------------------------------- main


def run(net: ST.Net, tasks: dict[str, ST.Task], P: dict[str, Any], large: Path) -> None:
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").strip(), "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
                           "stage0": {"inputs": {}}}
    dws = stage0_inputs(net, tasks, P["stage0"]["thresholds"], out)
    dump(out)
    if not out["stage0"]["inputs"]["pass"]:
        stop(out, "[stage0 inputs] FAIL: stop and tell the user")
    probes = stage0_readouts(net, P["settings"], P["thresholds"], large, out)
    dump(out)
    if not out["stage0"]["readouts"]["pass"]:
        stop(out, "[stage0 readouts] FAIL: the last block's readout is over the filed margin from the model's loss; tell the user")
    l_star = possession(net, probes, tasks, P, out)
    dump(out)
    if out["P3"]["status"] != "pass":
        stop(out, "[stage0 P3] FAIL: the last-block readout does not reproduce the construction; stop and tell the user")
    block = RO.LAST if l_star is None else l_star
    controls(net, probes, tasks, dws, block, P, out)
    dump(out)
    if not out["random"]["pass"]:
        stop(out, "[stage0 do-nothing] FAIL: a random direction moves a share over the filed bound; stop and tell the user")
    if l_star is not None and not out["W0_control"]["pass"]:
        out["P2"], out["branch"] = RO.p2({}, SEEDS, OWN, 0, 0, "W0's positive control failed"), "W0 control failed: P2 void"
        stop(out, "[stage0 W0 control] FAIL: the direction is decodable but W0 does not use it; P2 void; tell the user")
    score(net, probes, tasks, dws, block, None if l_star is not None else "no block before the last possesses (fit half)", P, out)
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
    assert (S["window"], S["first_position"], list(S["sites"])) == (RO.WIN, RO.POS0, list(RO.RESID)), "predictions.json disagrees with the code"
    a.large.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net(a.device)
    tasks = {n: ST.load_task(net, E54 / "large" / f"windows_{n}.pt", n, n) for n in ("elicit", "cipher")}
    run(net, tasks, P, a.large.resolve())


if __name__ == "__main__":
    main()
