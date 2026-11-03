"""E70 Amendment 1: remove the direction the count moves the residual along, not the ridge's weights. Plan:
reports/plans/2026-11-03_plan_e70-existing-new-counting.md, "Amendment 1". Rules, expected values and edge cases:
results/predictions_e70_a1.json (the run refuses unless it is committed unchanged). Trains nothing: LOW, the probes, E70's
results and the 20 states at 10,000 steps are read as saved; a missing one exits with its name.

Stages, each written to results/e70_a1.json as it ends (states one by one), every scored number in the log:
  control  LOW, 'C a 40': the same rows and model as E70 (else stop), 20 random directions, then the pattern a_b; the drop
           and the accuracy after removal. R1 is computed here and acted on after `why`.
  why      LOW only: a ridge refit on fresh rows, read on the held rows, after removing a_b and after removing w; variance
           share of h along a_b, w and random; cos(a_b, w). R3 misses: stop. R1 no teeth: stop, branch 'closed'.
  states   The 20 states: random shares, then the shares under a_b, with accuracy after removal.
  verdict  R2 by block (e70.r2_table), then the tree (removal.one_direction_branch).
A stop writes results/e70_a1_stop.json and exits 1. Exit 3 at the deadline: resubmit, it resumes from e70_a1.json. Changed
constants exit 2. Usage: cluster/e70_a1.sbatch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
from c1b import LOW, PRED_C1B, RunC, all_runs, committed
from e70 import OUT as E70_OUT
from e70 import (
    PRED_E70,
    PROBES,
    Dirs,
    Pair,
    answers,
    count_of,
    held_rows,
    load_dirs,
    r2_table,
    read,
    removed,
)
from run import EXIT_RESUME, FT_SAVES, RESULTS, device, load_model, log, write_json
from torch import Tensor

from commute import pcfg, removal
from commute.interchange import logits_at_art, run_at_art
from commute.jobs import hard_exit
from commute.mingpt import GPT
from commute.pcfg import A
from commute.readout import shares

PRED, OUT, STOP = RESULTS / "predictions_e70_a1.json", RESULTS / "e70_a1.json", RESULTS / "e70_a1_stop.json"
KIND, LAST = "pattern", FT_SAVES[-1]


def stop(res: dict[str, Any], why: str) -> NoReturn:
    res["stop"] = why
    write_json(OUT, res)
    write_json(STOP, {"why": why, "date": time.strftime("%Y-%m-%d %H:%M")})
    log(f"STOP: {why}; see {STOP.name} (exit 1)")
    hard_exit(1)


def require(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        log(f"missing saved files, nothing retrained (exit 1): {missing}")
        hard_exit(1)


def accuracy(logits: Tensor, tgt: Tensor) -> float:
    return float((logits.argmax(-1) == tgt.to(logits.device)).double().mean())


def label(seq: np.ndarray) -> Tensor:
    return torch.from_numpy(count_of(seq, A).astype(np.float64)).to(device())


def patterns(res: dict[str, Any]) -> list[Tensor]:
    return [torch.tensor(x, dtype=torch.float32, device=device())[:, None] for x in res["directions"]]


def same_as_e70(res: dict[str, Any], e70: dict[str, Any], low: GPT, seq: np.ndarray, dirs: Dirs, k: dict[str, Any]) -> None:
    """Rows and model: discrete readings that rounding cannot move by more than one row."""
    reading = removal.ridge_reading(run_at_art(low, seq)[1], dirs.ridge_w, dirs.ridge_b)
    c = count_of(seq, A)
    acc, floor = [float((r == c).mean()) for r in reading], removal.majority(c)
    ref = e70["probes"]["ridge"]
    res["identity"] = {"ridge_acc": acc, "ridge_acc_e70": ref["acc_a"], "floor": floor, "floor_e70": ref["floor_a"]}
    log(f"identity: {res['identity']}")
    if floor != ref["floor_a"] or any(abs(x - y) > 1 / k["n_held"] for x, y in zip(acc, ref["acc_a"], strict=True)):
        stop(res, f"LOW or the rows differ from E70's: {res['identity']}")


def stage_control(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    e70 = json.loads(E70_OUT.read_text())
    low, seq, dirs = load_model(LOW), held_rows(k).a, load_dirs(k)
    same_as_e70(res, e70, low, seq, dirs, k)
    tgt, resid = answers(seq), run_at_art(low, seq)[1]
    first = logits_at_art(low, seq)
    clean = read(first, tgt)[1]

    def go(b: int, u: Tensor) -> tuple[float, float]:
        logits = removed(low, seq, resid, b, u)
        return float((clean - read(logits, tgt)[1]).mean()), accuracy(logits, tgt)

    n = low.cfg.n_layer
    rand = [[go(b, u) for u in dirs.random["ridge"][b]] for b in range(n)]
    drops = [[x[0] for x in row] for row in rand]
    ref = e70["control"]["ridge"]
    diff = [float(np.abs(np.array(drops[b]) - np.array(ref["random"][b])).max()) for b in range(n)]
    out: dict[str, Any] = {"random": drops, "band": [removal.band_of(x) for x in drops], "band_e70": ref["band"], "random_max_diff_e70": diff,
                           "acc_clean": accuracy(first, tgt), "acc_random_mean": [float(np.mean([x[1] for x in row])) for row in rand]}  # fmt: skip
    res["control"] = {KIND: out}
    log(f"control LOW random: {json.dumps(out)}")
    if any(d > k["reproduce_frac"] * band for d, band in zip(diff, ref["band"], strict=True)):
        stop(res, f"E70's random drops did not reproduce: max difference per block {diff}, E70's band {ref['band']}")
    write_json(OUT, res)

    fit = pcfg.task_batch(np.random.default_rng(k["fit_key"]), k["n_fit"], "C", A)[0]
    basis = [removal.pattern_basis(h, label(fit)) for h in run_at_art(low, fit)[1]]
    res["directions"] = [u[:, 0].tolist() for u in basis]
    pat, ridge = [go(b, basis[b]) for b in range(n)], [go(b, dirs.probe["ridge"][b]) for b in range(n)]
    out |= {"drop": [x[0] for x in pat], "acc": [x[1] for x in pat], "drop_ridge": [x[0] for x in ridge], "acc_ridge": [x[1] for x in ridge], "drop_ridge_e70": ref["drop"]}
    res["r1"] = removal.r1({KIND: out["drop"]}, {KIND: out["band"]})
    for b in range(n):
        log(f"control LOW block {b + 1}: pattern drop {out['drop'][b]:.4f} band {out['band'][b]:.4f} acc {out['acc'][b]:.4f} (clean {out['acc_clean']:.4f}, random {out['acc_random_mean'][b]:.4f}); "
            f"ridge drop {out['drop_ridge'][b]:.4f} (E70 {ref['drop'][b]:.4f}) acc {out['acc_ridge'][b]:.4f}; random vs E70 {diff[b]:.2e}")  # fmt: skip
    log(f"R1: {res['r1']}")
    return 0


def stage_why(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    low, seq, dirs, basis = load_model(LOW), held_rows(k).a, load_dirs(k), patterns(res)
    fresh = pcfg.task_batch(np.random.default_rng(k["refit_key"]), k["n_refit"], "C", A)[0]
    fit_h, read_h, y, c = run_at_art(low, fresh)[1], run_at_art(low, seq)[1], label(fresh), count_of(seq, A)

    def refit(b: int, u: Tensor) -> float:
        w, b0 = removal.fit_ridge(removal.remove(fit_h[b], u), y, k["ridge_lambda"])
        return float((removal.ridge_reading(removal.remove(read_h[b], u)[None], w[None], b0[None])[0] == c).mean())

    def share(b: int, u: Tensor) -> float:
        return float((read_h[b] @ u).var() / read_h[b].var(0).sum())

    n, ridge, before, floor = low.cfg.n_layer, dirs.probe["ridge"], res["identity"]["ridge_acc"], res["identity"]["floor"]
    out = {"refit_after_pattern": [refit(b, basis[b]) for b in range(n)], "refit_after_ridge": [refit(b, ridge[b]) for b in range(n)], "before": before, "floor": floor,
           "var_share_pattern": [share(b, basis[b]) for b in range(n)], "var_share_ridge": [share(b, ridge[b]) for b in range(n)],
           "var_share_random": [float(np.mean([share(b, u) for u in dirs.random["ridge"][b]])) for b in range(n)],
           "cos_pattern_ridge": [float(basis[b][:, 0] @ ridge[b][:, 0]) for b in range(n)]}  # fmt: skip
    out["r3_erased"] = [removal.erased(out["refit_after_pattern"][b], floor, before[b]) for b in range(n)]
    res["why"] = out
    for b in range(n):
        log(f"why block {b + 1}: refit after pattern {out['refit_after_pattern'][b]:.4f}, after ridge {out['refit_after_ridge'][b]:.4f}, before {before[b]:.4f}, floor {floor:.4f}, erased {out['r3_erased'][b]}; "
            f"variance share pattern {out['var_share_pattern'][b]:.5f} ridge {out['var_share_ridge'][b]:.5f} random {out['var_share_random'][b]:.5f}; cos {out['cos_pattern_ridge'][b]:.4f}")  # fmt: skip
    if not all(out["r3_erased"][1:]):
        stop(res, f"R3: the refit still reads the count after removing the pattern at blocks 2-6: {out['refit_after_pattern']}")
    if res["r1"]["stop"]:
        res["branch"] = removal.one_direction_branch(res["r1"]["teeth"][KIND], [])
        log(f"BRANCH: {res['branch']}")
        stop(res, "LOW shows no loss above band at any block: the one-direction test closes, P and N not read")
    return 0


def low_reads(low: GPT, seq_by_task: dict[str, np.ndarray], dirs: Dirs, basis: list[Tensor]) -> dict[Any, dict[str, Pair]]:
    """LOW on the state rows, both targets (same texts, one forward): clean, per block under a_b (draw -1) and each draw."""
    tgts, seq = {t: answers(s) for t, s in seq_by_task.items()}, seq_by_task["N"]
    resid = run_at_art(low, seq)[1]
    both = lambda logits: {t: read(logits, g) for t, g in tgts.items()}
    out: dict[Any, dict[str, Pair]] = {"clean": both(logits_at_art(low, seq))}
    for b, u in enumerate(basis):
        out[b, -1] = both(removed(low, seq, resid, b, u))
        for i, r in enumerate(dirs.random["ridge"][b]):
            out[b, i] = both(removed(low, seq, resid, b, r))
    return out


def read_state(r: RunC, seq: np.ndarray, lows: dict[Any, dict[str, Pair]], dirs: Dirs, basis: list[Tensor], floor: float) -> dict[str, Any]:
    model, tgt = load_model(r.path(LAST)), answers(seq)
    first, resid = logits_at_art(model, seq), run_at_art(model, seq)[1]
    st = read(first, tgt)

    def share(b: int, i: int, logits: Tensor) -> dict[str, float]:
        return shares(lows["clean"][r.task], lows[b, i][r.task], st, read(logits, tgt), floor)

    n = len(basis)
    rand = [[share(b, i, removed(model, seq, resid, b, u))["A_lo"] for i, u in enumerate(dirs.random["ridge"][b])] for b in range(n)]
    probes = []
    for b in range(n):
        logits = removed(model, seq, resid, b, basis[b])
        probes.append(share(b, -1, logits) | {"acc": accuracy(logits, tgt)})
    return {"random": {KIND: rand}, "probe": {KIND: probes}, "acc_clean": accuracy(first, tgt)}


def stage_states(res: dict[str, Any], k: dict[str, Any], deadline: float) -> int:
    runs = all_runs(k)
    require([r.path(LAST) for r in runs])
    done = res.setdefault("states", {})
    todo = [r for r in runs if r.sid(LAST) not in done]
    if not todo:
        return 0
    held, dirs, basis = held_rows(k), load_dirs(k), patterns(res)
    lows = low_reads(load_model(LOW), held.b, dirs, basis)
    for r in todo:
        if time.monotonic() > deadline:
            log(f"states: deadline with {r.sid(LAST)} next; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        t = time.monotonic()
        done[r.sid(LAST)] = read_state(r, held.b[r.task], lows, dirs, basis, k["floor_min"]) | {"seconds": time.monotonic() - t}
        write_json(OUT, res)
        log(f"{r.sid(LAST)} ({r.name}): {json.dumps(done[r.sid(LAST)])}")
    return 0


def spread(res: dict[str, Any], names: list[str], b: int, q: str) -> list[float]:
    """Mean, min, max over the states' finite values; all nan when there is none."""
    x = [v for s in names if math.isfinite(v := res["states"][s]["probe"][KIND][b][q])]
    return [float(np.mean(x)), min(x), max(x)] if x else [math.nan] * 3


def stage_verdict(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    ft = json.loads(E70_OUT.read_text())["finetunes"]
    ten = {s: m for s, m in ft.items() if m["step"] == LAST}  # no 1K state is read: r2_table finds that side empty
    table = r2_table(res | {"finetunes": ten}, k, KIND)
    names = {t: [s for s, m in ten.items() if m["task"] == t] for t in "PN"}
    res["summary"] = [{f"{t}_{q}": spread(res, names[t], b, q) for t in "PN" for q in ("A_lo", "A_lo_raw", "A_nats", "acc")} for b in range(len(table["rows"]))]
    res["r2"], res["branch"] = table, removal.one_direction_branch(res["r1"]["teeth"][KIND], table["rows"])
    for row, summary in zip(table["rows"], res["summary"], strict=True):
        log(f"R2 block {row['block']}: {row['call']}; gap {row['gap']:.4f} reverse {row['reverse_gap']:.4f} band {row['band']:.4f} n P {row['n_P']} N {row['n_N']} nan P {row['nan_P']} N {row['nan_N']}; "
            f"mean min max {json.dumps(summary)}")  # fmt: skip
    log(f"R2 over all blocks, teeth or not: {table['verdict']}; holds at {table['holds_at']}, reversal at {table['reversal_at']}, no call at {table['no_call_at']}")
    if table["reversal_at"]:
        log(f"REVERSAL: N above P by the band at blocks {table['reversal_at']}. Announce it.")
    log(f"BRANCH: {res['branch']}")
    res["complete"] = True
    return 0


STAGES = {"control": stage_control, "why": stage_why, "states": stage_states, "verdict": stage_verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    deadline = time.monotonic() + 60 * args.deadline_min
    filed = json.loads(PRED.read_text()) if PRED.exists() else {}
    if not filed or str(filed.get("status", "")).startswith("draft") or not committed(PRED):
        log(f"file and commit the predictions first (a draft or an uncommitted change is refused): {PRED}")
        hard_exit(1)
    if STOP.exists():
        log(f"a stop fired ({STOP.read_text().strip()}); nothing runs until the lead clears it (exit 1)")
        hard_exit(1)
    require([LOW, PROBES, E70_OUT])
    k = json.loads(PRED_C1B.read_text())["constants"] | json.loads(PRED_E70.read_text())["constants"] | filed["constants"]
    res: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {"constants": k, "done": []}
    if res["constants"] != k:
        log("e70_a1.json was written under other constants: move it aside deliberately before rerunning (exit 2)")
        hard_exit(2)
    device()  # GPU only: fail before any work
    for name, stage in STAGES.items():
        if name in res["done"]:
            continue
        t = time.monotonic()
        rc = stage(res, k, deadline)
        log(f"stage {name}: {time.monotonic() - t:.0f} s (exit {rc})")
        if rc:
            hard_exit(rc)
        res["done"].append(name)
        write_json(OUT, res)
    log("E70 Amendment 1 complete")
    hard_exit(0)


if __name__ == "__main__":
    main()
