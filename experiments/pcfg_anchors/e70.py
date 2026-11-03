"""E70: the existing/new definition read in full on the counting model. Plan:
reports/plans/2026-11-03_plan_e70-existing-new-counting.md. Rules, expected values, edge cases and departures:
results/predictions_e70.json (the run refuses unless it is committed unchanged). Pure rules: src/commute/removal.py.

Stages, each written to results/e70.json as it ends (fine-tunes and states one by one), in the plan's order:
  low        LOW pretrained (results/large/a1/pt_LOW.pt); count_a against a1_LOW.json. Off by > reproduce_tol: stop.
  probes     ridge (one direction, the headline) and softmax (centred columns, rank r) by block -> large/e70/probes.pt.
  control    on LOW, 'C a 40': the drop of count(a) under random directions, then the probes; R1. No teeth: stop.
  finetunes  all 20 C1b runs (c1b.finetune; states at 1K and 10K); task accuracy against c1b.json, stops nothing.
  states     per state, per block: random shares, the probes' shares (readout.shares), g o f.
  verdict    bands, R2 per block and per probe, the table.
A stop writes results/e70_stop.json and exits 1. Exit 3 at the deadline: resubmit, it resumes from e70.json and the saved
files. Changed constants against e70.json exit 2. Every scored number goes to the log. Usage: cluster/e70.sbatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from amend3 import texts_of
from c1b import LOW, PRED_C1B, RunC, all_runs, committed, finetune, task_rows
from c1b import OUT as C1B_OUT
from run import (
    A1,
    EVAL_SEED,
    EXIT_RESUME,
    FT_SAVES,
    LARGE,
    RESULTS,
    a1_row,
    behaviour,
    device,
    eval_seqs,
    load_model,
    log,
    pretrain,
    write_json,
)
from torch import Tensor

from commute import pcfg, probe, removal
from commute.interchange import logits_at_art, run_at_art
from commute.mingpt import GPT
from commute.pcfg import A, B
from commute.readout import shares
from commute.steering import logodds_at, logp_at

PRED_E70, OUT, STOP = RESULTS / "predictions_e70.json", RESULTS / "e70.json", RESULTS / "e70_stop.json"
PROBES = LARGE / "e70" / "probes.pt"
KINDS = ("ridge", "softmax")
Pair = tuple[Tensor, Tensor]  # (log P, log-odds) per row


@dataclass(frozen=True)
class Held:
    a: np.ndarray  # 'C a 40', answer count(a): LOW's control and the probes' accuracy
    b: dict[str, np.ndarray]  # 'C b 40' task rows, "P" and "N": same texts, the answer token is the state's target


@dataclass(frozen=True)
class Dirs:
    probe: dict[str, list[Tensor]]  # kind -> per block, the basis (d, r)
    random: dict[str, list[list[Tensor]]]  # kind -> per block, per draw
    ridge_w: Tensor  # (n_layer, d), float64
    ridge_b: Tensor  # (n_layer,)


def held_rows(k: dict[str, Any]) -> Held:
    rng = lambda key: np.random.default_rng(k[key])
    return Held(pcfg.task_batch(rng("held_a_key"), k["n_held"], "C", A)[0], {t: task_rows(t, rng("held_b_key"), k["n_held"]) for t in "PN"})


def count_of(seq: np.ndarray, op: int) -> np.ndarray:
    return pcfg.count(texts_of(seq), op)


def answers(seq: np.ndarray) -> Tensor:
    return torch.from_numpy(seq[:, pcfg.ANS].astype(np.int64))


def read(logits: Tensor, tgt: Tensor) -> Pair:
    return logp_at(logits[:, None], 0, tgt), logodds_at(logits[:, None], 0, tgt)


def removed(model: GPT, seq: np.ndarray, resid: Tensor, b: int, u: Tensor) -> Tensor:
    """Logits with the basis u removed at `=` after block index b (0-based); resid is the model's own, clean, on seq."""
    return logits_at_art(model, seq, b + 1, removal.remove(resid[b], u))


def stop(res: dict[str, Any], why: str) -> None:
    res["stop"] = why
    write_json(OUT, res)
    write_json(STOP, {"why": why, "date": time.strftime("%Y-%m-%d %H:%M")})
    log(f"STOP: {why}; see {STOP.name} (exit 1)")
    sys.exit(1)


def load_dirs(k: dict[str, Any]) -> Dirs:
    p = torch.load(PROBES, map_location=device(), weights_only=False)
    d, ranks = p["ridge_w"].shape[1], {kind: p["U"][kind][0].shape[1] for kind in KINDS}
    rand = {kind: [[removal.random_basis(d, ranks[kind], [k["random_key"], ranks[kind], b, i]).to(device()) for i in range(k["n_draws"])]
                   for b in range(len(p["ridge_w"]))] for kind in KINDS}  # fmt: skip
    return Dirs(p["U"], rand, p["ridge_w"], p["ridge_b"])


# ---- stages ----


def stage_low(res: dict[str, Any], k: dict[str, Any], deadline: float) -> int:
    model, step, ms = pretrain("LOW", A1, LOW, deadline=deadline)
    if step < A1.iters:
        log(f"LOW: stopped at step {step} of {A1.iters}; resubmit to resume (exit {EXIT_RESUME})")
        return EXIT_RESUME
    beh, ref = behaviour(model, A1.eval_n), a1_row("LOW")
    a = cast(float, beh["count_a"])
    res["low"] = {"count_a": a, "count_b": beh["count_b"], "count_a_a1": ref["count_a"], "count_b_a1": ref["count_b"], "ms_per_step": ms}
    log(f"LOW: {res['low']}")
    if abs(a - ref["count_a"]) > k["reproduce_tol"]:
        stop(res, f"LOW count_a {a} is off a1_LOW.json's {ref['count_a']} by more than {k['reproduce_tol']}")
    return 0


def fit_probes(low: GPT, k: dict[str, Any]) -> dict[str, Any]:
    """Both probes by block, saved: {"U": kind -> per block basis, "ridge_w", "ridge_b", "values", "soft"}."""
    fit = pcfg.task_batch(np.random.default_rng(k["fit_key"]), k["n_fit"], "C", A)[0]
    y = torch.from_numpy(count_of(fit, A).astype(np.float64)).to(device())
    pairs = [removal.fit_ridge(r, y, k["ridge_lambda"]) for r in run_at_art(low, fit)[1]]
    w, b = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
    soft = probe.fit_probes({"LOW": low}, A, k["probe_seed"])["LOW"]
    held = probe.held_out_rows(A, k["probe_seed"], EVAL_SEED)
    values = removal.seen_values((held[:, pcfg.ANS] - pcfg.NUM0).astype(np.int64), k["min_seen"])
    cols = soft.w.detach()[:, :, pcfg.NUM0 + torch.tensor(values, device=device())]
    U = {"ridge": [removal.unit_basis(x) for x in w], "softmax": [removal.centred_basis(cols[i]) for i in range(len(w))]}
    p = {"U": U, "ridge_w": w, "ridge_b": b, "values": values, "soft": soft, "held": held}
    PROBES.parent.mkdir(parents=True, exist_ok=True)
    torch.save({q: v for q, v in p.items() if q in ("U", "ridge_w", "ridge_b", "values")}, PROBES)
    return p


def stage_probes(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    low, held = load_model(LOW), held_rows(k)
    p = fit_probes(low, k)
    n, seq_b = low.cfg.n_layer, held.b["N"]
    ra, rb = run_at_art(low, held.a)[1], run_at_art(low, seq_b)[1]
    ca, cb = count_of(held.a, A), count_of(seq_b, A)
    acc = lambda reading, c: [float((reading[i] == c).mean()) for i in range(n)]
    ridge = {"acc_a": acc(removal.ridge_reading(ra, p["ridge_w"], p["ridge_b"]), ca), "acc_b": acc(removal.ridge_reading(rb, p["ridge_w"], p["ridge_b"]), cb),
             "floor_a": removal.majority(ca), "floor_b": removal.majority(cb)}  # fmt: skip
    tok = lambda c: torch.from_numpy(pcfg.NUM0 + c.astype(np.int64)).to(device())
    sa = probe.accuracies(p["soft"], run_at_art(low, p["held"])[1], answers(p["held"]).to(device()))
    sb = probe.accuracies(p["soft"], rb, tok(cb))
    a1c = json.loads((RESULTS / "a1c_probe.json").read_text())["probes"]["raw"]["count_a"][k["probe_seed"]]["LOW"]
    soft = {"acc_a": sa[:n].tolist(), "shuffled_a": sa[n:].tolist(), "acc_a_a1c": a1c["acc"], "shuffled_a_a1c": a1c["shuffled"],
            "floor_a": removal.majority((p["held"][:, pcfg.ANS] - pcfg.NUM0).astype(np.int64)), "acc_b": sb[:n].tolist(),
            "shuffled_b": sb[n:].tolist(), "floor_b": removal.majority(cb)}  # fmt: skip
    corr = float(np.corrcoef(cb, count_of(seq_b, B))[0, 1])
    res["probes"] = {"ridge": ridge, "softmax": soft, "values": p["values"], "rank": len(p["values"]) - 1, "corr_count_a_count_b_on_C_b_40": corr}
    for kind, row in (("ridge", ridge), ("softmax", soft)):
        for i in range(n):
            log(f"probe {kind} block {i + 1}: C a 40 acc {row['acc_a'][i]:.4f} (floor {row['floor_a']:.4f}); C b 40 acc {row['acc_b'][i]:.4f} (floor {row['floor_b']:.4f})")
    log(f"probes: {json.dumps(res['probes'])}")
    return 0


def stage_control(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    low, seq, dirs = load_model(LOW), held_rows(k).a, load_dirs(k)
    tgt, resid = answers(seq), run_at_art(low, seq)[1]
    clean = read(logits_at_art(low, seq), tgt)[1]
    drop = lambda b, u: float((clean - read(removed(low, seq, resid, b, u), tgt)[1]).mean())
    n, out = low.cfg.n_layer, {}
    for kind in KINDS:
        rand = [[drop(b, u) for u in dirs.random[kind][b]] for b in range(n)]
        out[kind] = {"random": rand, "band": [removal.band_of(x) for x in rand]}
    for kind in KINDS:
        out[kind]["drop"] = [drop(b, dirs.probe[kind][b]) for b in range(n)]
        log(f"control LOW {kind}: band {np.round(out[kind]['band'], 4).tolist()} drop {np.round(out[kind]['drop'], 4).tolist()}; random {json.dumps(out[kind]['random'])}")
    res["control"], res["r1"] = out, removal.r1({q: out[q]["drop"] for q in KINDS}, {q: out[q]["band"] for q in KINDS})
    log(f"R1: {res['r1']}")
    if res["r1"]["stop"]:
        stop(res, "the probe's drop is within the band at every block for both probes: the removal has no teeth")
    return 0


def task_acc(model: GPT, task: str, seq: np.ndarray) -> float:
    want = count_of(seq, A) + 1 if task == "P" else count_of(seq, B)
    return float((run_at_art(model, seq)[0] == pcfg.NUM0 + want).mean())


def stage_finetunes(res: dict[str, Any], k: dict[str, Any], deadline: float) -> int:
    rows, ref = eval_seqs(B, A1.eval_n), json.loads(C1B_OUT.read_text())["checks"]
    done = res.setdefault("finetunes", {})
    for r in all_runs(k):
        if all(r.sid(s) in done for s in FT_SAVES):
            continue
        t = time.monotonic()
        if t > deadline or not finetune(r, deadline):
            log(f"{r.name}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        for s in FT_SAVES:
            row = {"run": r.name, "task": r.task, "lr": r.lr, "seed": r.seed, "step": s, "task_acc": task_acc(load_model(r.path(s)), r.task, rows),
                   "task_acc_c1b": ref[r.sid(s)]["task_acc"][r.task], "seconds": time.monotonic() - t}  # fmt: skip
            done[r.sid(s)] = row
            log(f"{r.sid(s)}: {row}")
        write_json(OUT, res)
        r.path("train").unlink(missing_ok=True)
    return 0


def low_reads(low: GPT, b_rows: dict[str, np.ndarray], dirs: Dirs) -> dict[Any, dict[str, Pair]]:
    """LOW's clean and removed reads on the state rows, for both targets (same inputs, so one forward for two)."""
    tgts, seq = {t: answers(s) for t, s in b_rows.items()}, b_rows["N"]
    resid = run_at_art(low, seq)[1]
    both = lambda logits: {t: read(logits, g) for t, g in tgts.items()}
    out: dict[Any, dict[str, Pair]] = {"clean": both(logits_at_art(low, seq))}
    for kind in KINDS:
        for b, draws in enumerate(dirs.random[kind]):
            out[kind, b, -1] = both(removed(low, seq, resid, b, dirs.probe[kind][b]))
            for i, u in enumerate(draws):
                out[kind, b, i] = both(removed(low, seq, resid, b, u))
    return out


def read_state(r: RunC, step: int, seq: np.ndarray, lows: dict[Any, dict[str, Pair]], dirs: Dirs, reading: np.ndarray, floor: float) -> dict[str, Any]:  # fmt: skip
    """Random shares first (A_lo of each draw), then the probes' shares in full, then g o f, per block."""
    model, tgt = load_model(r.path(step)), answers(seq)
    st = read(logits_at_art(model, seq), tgt)
    pred, resid = run_at_art(model, seq)
    n = len(resid)

    def share(kind: str, b: int, i: int, u: Tensor) -> dict[str, float]:
        return shares(lows["clean"][r.task], lows[kind, b, i][r.task], st, read(removed(model, seq, resid, b, u), tgt), floor)

    rand = {kind: [[share(kind, b, i, u)["A_lo"] for i, u in enumerate(dirs.random[kind][b])] for b in range(n)] for kind in KINDS}
    probes = {kind: [share(kind, b, -1, dirs.probe[kind][b]) for b in range(n)] for kind in KINDS}
    return {"random": rand, "probe": probes, "gf": [removal.lookup_accuracy(reading[b], pred) for b in range(n)]}


def stage_states(res: dict[str, Any], k: dict[str, Any], deadline: float) -> int:
    done, todo = res.setdefault("states", {}), [(r, s) for r in all_runs(k) for s in FT_SAVES if r.sid(s) not in res.get("states", {})]
    if not todo:
        return 0
    low, held, dirs = load_model(LOW), held_rows(k), load_dirs(k)
    lows = low_reads(low, held.b, dirs)
    reading = removal.ridge_reading(run_at_art(low, held.b["N"])[1], dirs.ridge_w, dirs.ridge_b)
    for r, s in todo:
        if time.monotonic() > deadline:
            log(f"states: deadline with {r.sid(s)} next; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        t = time.monotonic()
        row = read_state(r, s, held.b[r.task], lows, dirs, reading, k["floor_min"])
        done[r.sid(s)] = row | {"seconds": time.monotonic() - t}
        write_json(OUT, res)
        log(f"{r.sid(s)} ({r.name}, step {s}): {json.dumps(done[r.sid(s)])}")
    return 0


def side(res: dict[str, Any], task: str, step: int) -> list[str]:
    return [s for s, m in res["finetunes"].items() if m["task"] == task and m["step"] == step]


def r2_table(res: dict[str, Any], k: dict[str, Any], kind: str) -> dict[str, Any]:
    """R2 by block: every 10K state sets the band; states below task_acc_min are left out of R2 and listed."""
    st, ft, last = res["states"], res["finetunes"], FT_SAVES[-1]
    learned = lambda task: [s for s in side(res, task, last) if ft[s]["task_acc"] >= k["task_acc_min"]]
    rows = []
    for b in range(len(res["control"][kind]["band"])):
        band = removal.max_band([removal.band_of(st[s]["random"][kind][b]) for s in (*side(res, "P", last), *side(res, "N", last))])
        val = lambda names, q="A_lo", b=b: {s: st[s]["probe"][kind][b][q] for s in names}
        row = removal.r2_block(list(val(learned("P")).values()), list(val(learned("N")).values()), band)
        rows.append(row | {"block": b + 1, "shares": {f"{t}_{step}": val(side(res, t, step)) for t in "PN" for step in FT_SAVES},
                           "shares_raw": {f"{t}_{last}": val(side(res, t, last), "A_lo_raw") for t in "PN"}})  # fmt: skip
    unlearned = [s for s, m in ft.items() if m["step"] == last and m["task_acc"] < k["task_acc_min"]]
    return {"rows": rows, "not_yet_learned_10K": unlearned} | removal.r2_verdict(rows)


def stage_verdict(res: dict[str, Any], k: dict[str, Any], _: float) -> int:
    res["r2"] = {kind: r2_table(res, k, kind) for kind in KINDS}
    for kind, table in res["r2"].items():
        for row in table["rows"]:
            log(f"R2 {kind} block {row['block']}: {row['call']}; gap {row['gap']:.4f} reverse {row['reverse_gap']:.4f} band {row['band']:.4f} "
                f"n P {row['n_P']} N {row['n_N']} nan P {row['nan_P']} N {row['nan_N']}; shares {json.dumps(row['shares'])}")  # fmt: skip
        log(f"R2 {kind}: {table['verdict']}; holds at {table['holds_at']}, no call at {table['no_call_at']}; not yet learned at 10K {table['not_yet_learned_10K']}")
        if table["reversal_at"]:
            log(f"REVERSAL ({kind}): N above P by the band at blocks {table['reversal_at']}. Announce it.")
    res["gf_by_block"] = {f"{t}_{step}": {s: [g["acc"] for g in res["states"][s]["gf"]] for s in side(res, t, step)} for t in "PN" for step in FT_SAVES}
    log(f"g o f: {json.dumps(res['gf_by_block'])}")
    res["complete"] = True
    return 0


STAGES = {"low": stage_low, "probes": stage_probes, "control": stage_control, "finetunes": stage_finetunes, "states": stage_states, "verdict": stage_verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    deadline = time.monotonic() + 60 * args.deadline_min
    filed = json.loads(PRED_E70.read_text()) if PRED_E70.exists() else {}
    if not filed or str(filed.get("status", "")).startswith("draft") or not committed(PRED_E70):
        sys.exit(f"file and commit the predictions first (a draft or an uncommitted change is refused): {PRED_E70}")
    if STOP.exists():
        sys.exit(f"a stop row fired ({STOP.read_text().strip()}); nothing runs until the lead clears it (exit 1)")
    k = json.loads(PRED_C1B.read_text())["constants"] | filed["constants"]
    res: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {"constants": k, "done": []}
    if res["constants"] != k:
        log("e70.json was written under other constants: move it aside deliberately before rerunning (exit 2)")
        sys.exit(2)
    device()  # GPU only: fail before any work
    for name, stage in STAGES.items():
        if name in res["done"]:
            continue
        t = time.monotonic()
        rc = stage(res, k, deadline)
        log(f"stage {name}: {time.monotonic() - t:.0f} s (exit {rc})")
        if rc:
            sys.exit(rc)
        res["done"].append(name)
        write_json(OUT, res)
    log("E70 complete")


if __name__ == "__main__":
    main()
