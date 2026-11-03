"""E70 Amendment 2, the last: does the largest direction with no covariance with the count cost what a_b cost? Plan:
reports/plans/2026-11-03_plan_e70-existing-new-counting.md, "Amendment 2". Rules, expected values and edge cases:
results/predictions_e70_a2.json (the run refuses unless it is committed unchanged). Trains nothing: LOW, the probes, E70's and
Amendment 1's results and the 20 states at 10,000 steps are read as saved; a missing one exits with its name.

v_b = the top principal component of LOW's residual after a_b is removed, on a_b's own 20,000 fit rows. a_b is read from
results/e70_a1.json. v_b is not matched on size: every per-block log line states both variance shares.

Stages, each written to results/e70_a2.json as it ends (states one by one), every scored number in the log:
  control  LOW, 'C a 40': R0 (same rows, model and random drops as E70), then v_b's drop and accuracy. No teeth stops nothing.
  why      LOW only: covariance with the count, on fit and held rows (R4); share of v_b between count classes; a ridge refit
           after removing v_b (R3').
  states   The 20 states: random shares, then the shares under v_b, with accuracy after removal.
  verdict  R2 by block (e70.r2_table), the tree (removal.no_count_branch), block 6, share per unit of variance share.
Only apparatus misses stop: results/e70_a2_stop.json and exit 1. Exit 3 at the deadline: resubmit, it resumes from e70_a2.json.
Changed constants exit 2. Usage: cluster/e70_a2.sbatch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, NoReturn

import numpy as np
from c1b import LOW, PRED_C1B, all_runs, committed
from e70 import OUT as E70_OUT
from e70 import (
    PRED_E70,
    PROBES,
    Dirs,
    answers,
    count_of,
    held_rows,
    load_dirs,
    r2_table,
    read,
    removed,
)
from e70_a1 import KIND as A1_KIND
from e70_a1 import OUT as A1_OUT
from e70_a1 import PRED as PRED_A1
from e70_a1 import accuracy, label, low_reads, patterns, read_state, require
from run import EXIT_RESUME, FT_SAVES, RESULTS, device, load_model, log, write_json
from torch import Tensor

from commute import pcfg, removal
from commute.interchange import logits_at_art, run_at_art
from commute.jobs import hard_exit
from commute.pcfg import A, B

PRED, OUT, STOP = RESULTS / "predictions_e70_a2.json", RESULTS / "e70_a2.json", RESULTS / "e70_a2_stop.json"
KIND, LAST = "no_count", FT_SAVES[-1]


def stop(res: dict[str, Any], why: str) -> NoReturn:
    res["stop"] = why
    write_json(OUT, res)
    write_json(STOP, {"why": why, "date": time.strftime("%Y-%m-%d %H:%M")})
    log(f"STOP: {why}; see {STOP.name} (exit 1)")
    hard_exit(1)


def amendment_1() -> dict[str, Any]:
    return json.loads(A1_OUT.read_text())


def sizes(res: dict[str, Any], b: int) -> str:
    """The clause on every per-block log line: v_b is not matched on size."""
    c = res["control"][KIND]
    return f"variance share v_b {c['var_share_v'][b]:.4f} a_b {c['var_share_a'][b]:.4f}"


def same_as_e70(res: dict[str, Any], e70: dict[str, Any], resid: Tensor, seq: np.ndarray, dirs: Dirs, k: dict[str, Any]) -> None:
    """R0(a): rows and model. Discrete readings that rounding cannot move by more than one row."""
    c = count_of(seq, A)
    acc = [float((r == c).mean()) for r in removal.ridge_reading(resid, dirs.ridge_w, dirs.ridge_b)]
    floor, ref = removal.majority(c), e70["probes"]["ridge"]
    res["identity"] = {"ridge_acc": acc, "ridge_acc_e70": ref["acc_a"], "floor": floor, "floor_e70": ref["floor_a"]}
    log(f"identity: {res['identity']}")
    if not removal.same_reading(acc, ref["acc_a"], floor, ref["floor_a"], k["n_held"]):
        stop(res, f"R0: LOW or the rows differ from E70's: {res['identity']}")


def stage_control(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    e70, a1 = json.loads(E70_OUT.read_text()), amendment_1()
    if not a1.get("complete") or a1["r2"]["holds_at"] != k["a1_held_at"]:
        stop(res, f"e70_a1.json is not the filed Amendment 1 result: holds at {a1.get('r2', {}).get('holds_at')}, filed {k['a1_held_at']}")
    low, seq, dirs = load_model(LOW), held_rows(k).a, load_dirs(k)
    tgt, resid = answers(seq), run_at_art(low, seq)[1]
    same_as_e70(res, e70, resid, seq, dirs, k)

    first = logits_at_art(low, seq)
    clean = read(first, tgt)[1]

    def go(b: int, u: Tensor) -> tuple[float, float]:
        logits = removed(low, seq, resid, b, u)
        return float((clean - read(logits, tgt)[1]).mean()), accuracy(logits, tgt)

    n, ref = low.cfg.n_layer, e70["control"]["ridge"]
    rand = [[go(b, u) for u in dirs.random["ridge"][b]] for b in range(n)]
    drops = [[x[0] for x in row] for row in rand]
    diff = [float(np.abs(np.array(drops[b]) - np.array(ref["random"][b])).max()) for b in range(n)]
    out: dict[str, Any] = {"random": drops, "band": [removal.band_of(x) for x in drops], "band_e70": ref["band"], "random_max_diff_e70": diff,
                           "acc_clean": accuracy(first, tgt), "acc_random_mean": [float(np.mean([x[1] for x in row])) for row in rand]}  # fmt: skip
    res["control"] = {KIND: out}
    log(f"control LOW random: {json.dumps(out)}")
    if not removal.reproduced(diff, ref["band"], k["reproduce_frac"]):
        stop(res, f"R0: E70's random drops did not reproduce: max difference per block {diff}, E70's band {ref['band']}")
    write_json(OUT, res)

    fit = pcfg.task_batch(np.random.default_rng(k["fit_key"]), k["n_fit"], "C", A)[0]
    a_b = patterns(a1)
    basis = [removal.top_pc_after(h, a) for h, a in zip(run_at_art(low, fit)[1], a_b, strict=True)]
    res["directions"] = [u[:, 0].tolist() for u in basis]
    v, a = [go(b, basis[b]) for b in range(n)], [go(b, a_b[b]) for b in range(n)]
    share = lambda u: [removal.variance_share(resid[b], u[b]) for b in range(n)]
    out |= {"drop": [x[0] for x in v], "acc": [x[1] for x in v], "drop_a_b": [x[0] for x in a], "acc_a_b": [x[1] for x in a], "drop_a_b_a1": a1["control"][A1_KIND]["drop"],
            "var_share_v": share(basis), "var_share_a": share(a_b),
            "var_share_random": [float(np.mean([removal.variance_share(resid[b], u) for u in dirs.random["ridge"][b]])) for b in range(n)]}  # fmt: skip
    res["teeth"] = removal.r1({KIND: out["drop"]}, {KIND: out["band"]})["teeth"][KIND]  # read, stops nothing
    for b in range(n):
        log(f"control LOW block {b + 1}: v_b drop {out['drop'][b]:.4f} band {out['band'][b]:.4f} teeth {res['teeth'][b]} acc {out['acc'][b]:.4f} (clean {out['acc_clean']:.4f}, random {out['acc_random_mean'][b]:.4f}); "
            f"a_b drop {out['drop_a_b'][b]:.4f} (Amendment 1 {out['drop_a_b_a1'][b]:.4f}) acc {out['acc_a_b'][b]:.4f}; {sizes(res, b)}, random {out['var_share_random'][b]:.5f}; random vs E70 {diff[b]:.2e}")  # fmt: skip
    return 0


def stage_why(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    low, held, dirs, v, a = load_model(LOW), held_rows(k).a, load_dirs(k), patterns(res), patterns(amendment_1())
    fit = pcfg.task_batch(np.random.default_rng(k["fit_key"]), k["n_fit"], "C", A)[0]
    fresh = pcfg.task_batch(np.random.default_rng(k["refit_key"]), k["n_refit"], "C", A)[0]
    h_fit, h_fresh, h_held = run_at_art(low, fit)[1], run_at_art(low, fresh)[1], run_at_art(low, held)[1]
    y_fit, y_fresh, y_held, c, c_b = label(fit), label(fresh), label(held), count_of(held, A), count_of(held, B)
    n, before, floor = low.cfg.n_layer, res["identity"]["ridge_acc"], res["identity"]["floor"]

    def along(u: list[Tensor], labels: np.ndarray) -> list[float]:
        return [removal.class_share((h_held[b] @ u[b])[:, 0].double().cpu().numpy(), labels) for b in range(n)]

    out = {"cov_fit_v": [removal.abs_cov(h_fit[b], v[b], y_fit) for b in range(n)], "cov_fit_a": [removal.abs_cov(h_fit[b], a[b], y_fit) for b in range(n)],
           "cov_held_v": [removal.abs_cov(h_held[b], v[b], y_held) for b in range(n)], "cov_held_a": [removal.abs_cov(h_held[b], a[b], y_held) for b in range(n)],
           "cos_v_a": [float(v[b][:, 0].double() @ a[b][:, 0].double()) for b in range(n)],
           "class_a_v": along(v, c), "class_a_a": along(a, c), "class_b_v": along(v, c_b), "class_b_a": along(a, c_b),
           "class_a_random": [float(np.mean([removal.class_share((h_held[b] @ u)[:, 0].double().cpu().numpy(), c) for u in dirs.random["ridge"][b]])) for b in range(n)],
           "refit_after_v": [removal.refit_accuracy(h_fresh[b], y_fresh, h_held[b], c, v[b], k["ridge_lambda"]) for b in range(n)], "before": before, "floor": floor}  # fmt: skip
    out["r3_prime_readable"] = [removal.above_midpoint(out["refit_after_v"][b], floor, before[b]) for b in range(n)]
    out["r4_cov_fit_ok"] = [out["cov_fit_v"][b] <= k["cov_tol"] * out["cov_fit_a"][b] for b in range(n)]
    res["why"] = out
    for b in range(n):
        log(f"why block {b + 1}: {sizes(res, b)}; |cov with count| fit v_b {out['cov_fit_v'][b]:.3e} a_b {out['cov_fit_a'][b]:.3e}, held v_b {out['cov_held_v'][b]:.3e} a_b {out['cov_held_a'][b]:.3e}; cos {out['cos_v_a'][b]:.2e}; "
            f"class share count(a) v_b {out['class_a_v'][b]:.4f} a_b {out['class_a_a'][b]:.4f} random {out['class_a_random'][b]:.4f}, count(b) v_b {out['class_b_v'][b]:.4f} a_b {out['class_b_a'][b]:.4f}; "
            f"refit after v_b {out['refit_after_v'][b]:.4f}, before {before[b]:.4f}, floor {floor:.4f}, readable {out['r3_prime_readable'][b]}")  # fmt: skip
    if not all(out["r4_cov_fit_ok"]):
        stop(res, f"R4: v_b has covariance with the count on its own fit rows: {out['cov_fit_v']} against a_b's {out['cov_fit_a']}")
    if not all(out["r3_prime_readable"][1:]):
        stop(res, f"R3': the refit no longer reads the count after removing v_b at blocks 2-6, so v_b carries it and is no null: {out['refit_after_v']}")
    return 0


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
        row = read_state(r, held.b[r.task], lows, dirs, basis, k["floor_min"])  # Amendment 1's reader, filed under its slot name
        done[r.sid(LAST)] = {"random": {KIND: row["random"][A1_KIND]}, "probe": {KIND: row["probe"][A1_KIND]}, "acc_clean": row["acc_clean"], "seconds": time.monotonic() - t}
        write_json(OUT, res)
        log(f"{r.sid(LAST)} ({r.name}): {json.dumps(done[r.sid(LAST)])}")
    return 0


def spread(res: dict[str, Any], names: list[str], b: int, q: str) -> list[float]:
    """Mean, min, max over the states' finite values; all nan when there is none."""
    x = [v for s in names if math.isfinite(v := res["states"][s]["probe"][KIND][b][q])]
    return [float(np.mean(x)), min(x), max(x)] if x else [math.nan] * 3


def per_variance(share: list[float], var: list[float]) -> list[float]:
    return [s / w if w > 0 else math.nan for s, w in zip(share, var, strict=True)]


def stage_verdict(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    a1, c = amendment_1(), res["control"][KIND]
    ft = json.loads(E70_OUT.read_text())["finetunes"]
    ten = {s: m for s, m in ft.items() if m["step"] == LAST}  # no 1K state is read: r2_table finds that side empty
    table = r2_table(res | {"finetunes": ten}, k, KIND)
    names = {t: [s for s, m in ten.items() if m["task"] == t] for t in "PN"}
    n, rows = len(table["rows"]), table["rows"]
    res["summary"] = [{f"{t}_{q}": spread(res, names[t], b, q) for t in "PN" for q in ("A_lo", "A_lo_raw", "A_nats", "acc")} for b in range(n)]
    mean = lambda summary, t: [summary[b][f"{t}_A_lo"][0] for b in range(n)]
    p_a, p_v, n_a, n_v = mean(a1["summary"], "P"), mean(res["summary"], "P"), mean(a1["summary"], "N"), mean(res["summary"], "N")
    res["r2"], res["branch"] = table, removal.no_count_branch(rows, p_a, p_v, res["why"]["class_a_v"], k["a1_held_at"], k["carries_frac"])
    for row, summary in zip(rows, res["summary"], strict=True):
        b = row["block"] - 1
        log(f"R2 block {row['block']}: {row['call']}; gap {row['gap']:.4f} reverse {row['reverse_gap']:.4f} band {row['band']:.4f} n P {row['n_P']} N {row['n_N']} nan P {row['nan_P']} N {row['nan_N']}; "
            f"P mean a_b {p_a[b]:.4f} v_b {p_v[b]:.4f}; N mean a_b {n_a[b]:.4f} v_b {n_v[b]:.4f}; {sizes(res, b)}; mean min max {json.dumps(summary)}")  # fmt: skip
    log(f"R2 over all blocks: {table['verdict']}; holds at {table['holds_at']}, reversal at {table['reversal_at']}, no call at {table['no_call_at']}")
    res["block_6"] = {"call": rows[-1]["call"], "a1_call": a1["r2"]["rows"][-1]["call"], "N_minus_P_means_v": n_v[-1] - p_v[-1], "N_minus_P_means_a": n_a[-1] - p_a[-1], "reverse_gap": rows[-1]["reverse_gap"]}
    log(f"block 6, read apart and unable to change Amendment 1's filed branch: {json.dumps(res['block_6'])}; {sizes(res, n - 1)}")
    res["share_per_variance_descriptive"] = {"P_v": per_variance(p_v, c["var_share_v"]), "N_v": per_variance(n_v, c["var_share_v"]), "P_a": per_variance(p_a, c["var_share_a"]), "N_a": per_variance(n_a, c["var_share_a"])}
    log(f"share lost per unit of variance share, descriptive, unscored: {json.dumps(res['share_per_variance_descriptive'])}")
    log(f"BRANCH: {res['branch']}; v_b is not matched on size")
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
    require([LOW, PROBES, E70_OUT, A1_OUT])
    k = {}
    for path in (PRED_C1B, PRED_E70, PRED_A1):
        k |= json.loads(path.read_text())["constants"]
    k |= filed["constants"]
    res: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {"constants": k, "done": []}
    if res["constants"] != k:
        log("e70_a2.json was written under other constants: move it aside deliberately before rerunning (exit 2)")
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
    log("E70 Amendment 2 complete")
    hard_exit(0)


if __name__ == "__main__":
    main()
