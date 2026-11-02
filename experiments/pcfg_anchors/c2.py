"""E55 Stage C2: VPD on the C1b truth set. Plan: reports/plans/2026-10-31_plan_e55-c2-vpd.md. Config:
c2_config.yaml (param-decomp's ss_llama_simple_mlp-2L recipe; C from the released 67M decomposition's C per unit rank,
14,592 subcomponents). Predictions: results/predictions_c2.json. The $0 stages, never scored, are in c2_rehearse.py.

One decomposition of LOW on LOW's own pretraining rows (pcfg.pretrain_batch, prior LOW; 258-token inputs), trained by
`commute.decompose` (param-decomp's step, with saves and resume). Each C1b state is then read by the route split's
re-gating share r and the stand-in ratio (`commute.pcfg_vpd`) on natural 'C b 40' rows labelled with the state's task.

Stages (repo root, GPU). Each refuses (exit 2) while predictions_c2.json is a draft or a number it reads is not filed:
  decompose  the full decomposition, resumed from results/large/c2/decomp_train.pt; decomp_log.jsonl grows as it runs. A
             missed early check (decomposition_stops.early_checks, {"<step>": {"<eval key>": {"max": x, "min": y}}}) saves
             and exits 1. Exit 3 at the deadline (resubmit by hand), exit 0 with results/large/c2/decomp_final.pt.
  read       from decomp_final.pt. Controls first and alone: the alive set (saved, so every job reads on the same set),
             the zero update (over its filed bounds: exit 1, apparatus), the two ID states. Then the 14 calibration
             states and results/c2_cutoffs.json, for the lead to commit.
  score      refuses (exit 2) unless c2_cutoffs.json is committed and equals the cut-offs recomputed from c2.json. The
             21 scored states and the five unlearned N states (not scored); labels and scores.
read and score rewrite results/c2.json after every state, skip the states already in it and exit 3 when the slowest state
so far would overrun the deadline (resubmit by hand). Every number goes to the log as well.

Usage: python experiments/pcfg_anchors/c2.py {decompose,read,score} [--deadline-min M]  (cluster/pcfg_c2.sbatch,
cluster/pcfg_c2_normal.sbatch)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from amend3 import texts_of
from amend5 import prompt_seqs
from c1b import committed, id_paths, id_prompt, lr_key, score_one, task_rows
from run import (
    EVAL_SEED,
    EXIT_RESUME,
    FT_SAVES,
    LARGE,
    RESULTS,
    device,
    load_model,
    log,
    write_json,
)
from torch import Tensor

from commute import decompose as D
from commute import pcfg, pcfg_vpd, revival, vpd
from commute.anchors import lift_context
from commute.pcfg import A
from commute.pcfg_train import Batches

HERE = Path(__file__).resolve().parent
CONFIG, PRED_C2 = HERE / "c2_config.yaml", RESULTS / "predictions_c2.json"
LOW_PATH, C1B_JSON, C1B_CUTOFFS = LARGE / "a1" / "pt_LOW.pt", RESULTS / "c1b.json", RESULTS / "c1b_cutoffs.json"
OUT, CUTOFFS, DECOMP_DIR = RESULTS / "c2.json", RESULTS / "c2_cutoffs.json", LARGE / "c2"
TRAIN_CKPT, FINAL, ALIVE = DECOMP_DIR / "decomp_train.pt", DECOMP_DIR / "decomp_final.pt", DECOMP_DIR / "alive.pt"
DECOMP_SEED, EVAL_DECOMP_SEED, ALIVE_SEED, READ_TAIL = 200, 201, 202, 60  # data streams; READ_TAIL: [EVAL_SEED, 60]
CHUNK = 50  # batches per generated chunk
READINGS, ZERO, PARTS = ("r", "ratio"), "zero_update", ("controls", "calibration", "scored", "unscored")
STATE_SECONDS = 60.0  # per-state reading time assumed until one is measured (the rehearsal's 256 windows took ~4 s; 1,024 here)


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the rehearsal and the scored run."""

    steps: int  # config.steps (the schedules' length)
    stop_first: int  # 0, or the step where run A stops and is resumed from its file (the rehearsal)
    save_every: int
    log_every: int
    alive_rows: int  # pretraining rows for the alive set (mean causal importance > 1e-6)
    read_windows: int  # natural 'C b 40' windows per reading
    read_chunk: int  # windows per route-cell chunk (host memory: ~40 MB of gates per window at 38K subcomponents)
    read_batch: int

    def __post_init__(self) -> None:
        assert self.steps % self.save_every == 0 and self.save_every % self.log_every == 0, self
        assert self.stop_first == 0 or (0 < self.stop_first < self.steps and self.stop_first % self.log_every == 0), self
        assert self.read_windows % self.read_chunk == 0 and self.read_chunk % self.read_batch == 0, self


REHEARSAL = Sizes(steps=400, stop_first=200, save_every=200, log_every=50, alive_rows=512, read_windows=256, read_chunk=64, read_batch=32)
FULL = Sizes(steps=400_000, stop_first=0, save_every=5_000, log_every=200, alive_rows=8192, read_windows=1024, read_chunk=64, read_batch=32)


def inputs(rows: np.ndarray) -> np.ndarray:
    return rows[:, :-1]  # 258 tokens: the minGPT's block (SEQ - 1)


def pretrain_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    return inputs(pcfg.pretrain_batch(rng, n, pcfg.PRIORS["LOW"]))


def data(cfg: Any) -> tuple[Batches, Any]:
    """Training batches by step and an endless iterator of eval batches (fresh streams, LOW's pretraining rows)."""
    train = Batches(pretrain_rows, DECOMP_SEED, cfg.batch_size, CHUNK, device())
    ev = Batches(pretrain_rows, EVAL_DECOMP_SEED, cfg.eval_batch_size, CHUNK, device())

    def eval_iter() -> Any:
        i = 0
        while True:
            yield ev.get(i)
            i += 1

    return train, eval_iter()


def fresh(cfg: Any) -> D.Decomp:
    return D.build(load_model(LOW_PATH), cfg, (cfg.batch_size, pcfg.SEQ - 1), device())


# ---- one state's reading ----


def state_path(sid: str) -> Path:
    p = Path(json.loads(C1B_JSON.read_text())["meta"][sid]["path"])
    return LARGE / p.relative_to(p.parents[1])  # rebased on this clone's results/large


def answers(task: str, rows: np.ndarray) -> np.ndarray:
    """Token ids of each row's answer under the task: count(a) + 1 (P), count(b) (N) or count(a) (the ID states)."""
    t = texts_of(rows)
    n_a, n_b = pcfg.count(t, A), pcfg.count(t, pcfg.B)
    return pcfg.NUM0 + {"P": n_a + 1, "N": n_b, "ID": n_a}[task]


def alive_set(model: Any, n_rows: int) -> dict[str, Tensor]:
    batches = [torch.from_numpy(pretrain_rows(np.random.default_rng([ALIVE_SEED, i]), 64).astype(np.int64)) for i in range(n_rows // 64)]
    return {k: v.cpu().bool() for k, v in vpd.alive_masks(vpd.mean_ci(model, batches)).items()}


@dataclass(frozen=True)
class Reader:
    """What every state's reading shares: the decomposition, LOW, the split's bases and the natural rows."""

    model: Any
    low: Any
    keys: list[str]
    lift: dict[str, Any]
    comps0: dict[str, tuple[Tensor, Tensor]]
    W0: dict[str, Tensor]
    rows: np.ndarray
    sz: Sizes
    ratio_floor: float


def reader(model: Any, alive: dict[str, Tensor], sz: Sizes, ratio_floor: float) -> Reader:
    rows = task_rows("N", np.random.default_rng([EVAL_SEED, READ_TAIL]), sz.read_windows)
    return Reader(model, load_model(LOW_PATH), vpd.module_keys(model), lift_context(model, alive), vpd.component_weights(model),
                  vpd.target_weights(model), rows, sz, ratio_floor)  # fmt: skip


def read_one(rd: Reader, st: Any, task: str) -> dict[str, Any]:
    """Both readings, the matrix share, the outside norm share and T0 - LOW of state `st`, on the natural rows answered
    with its task; the ID states read the same texts under their own prompt."""
    t = time.perf_counter()
    rows = prompt_seqs(texts_of(rd.rows), id_prompt()) if task == "ID" else rd.rows
    w = pcfg_vpd.windows(rows, answers(task, rows))
    dW = pcfg_vpd.matrix_update(st, rd.low, rd.keys)
    cells = pcfg_vpd.cells_of(rd.model, rd.lift, rd.comps0, rd.W0, dW, w, rd.sz.read_chunk, rd.sz.read_batch)
    lo_low = pcfg_vpd.answer_log_odds(rd.low, w)
    full = pcfg_vpd.answer_log_odds(st, w) - lo_low
    sh = pcfg_vpd.shares(cells, rd.ratio_floor)
    return {"task": task, "cells": cells, **sh, "full_gain": full,
            "matrix_share": sh["gain"] / full if abs(full) >= pcfg_vpd.GAIN_MIN else float("nan"),
            "outside_norm_share": pcfg_vpd.outside_norm_share(st, rd.low, rd.keys), "T0_minus_LOW": cells["T0"] - lo_low,
            "seconds": time.perf_counter() - t, **D.memory()}  # fmt: skip


def xs(rec: dict[str, Any], share_min: float) -> dict[str, float | None]:
    """The readings a label may use: None (void) when nan, or when the 24 matrices carry under share_min of the state's
    gain; an undefined share (nan: the state's own gain is under gain_min) is void too."""
    carried = rec["matrix_share"] >= share_min
    return {q: rec[q] if carried and math.isfinite(rec[q]) else None for q in READINGS}


def do_nothing_misses(rec: dict[str, Any], gain_min: float, t0_max: float) -> list[str]:
    """predictions_c2.json do_nothing, on the zero update's record; a nan misses."""
    c = rec["cells"]
    rows = (("T1 - T0", c["T1"] - c["T0"], gain_min), ("L11 - L00", c["L11"] - c["L00"], gain_min), ("T0 - LOW", rec["T0_minus_LOW"], t0_max))
    return [f"|{name}| = {abs(v)} over {bar}" for name, v, bar in rows if not abs(v) <= bar]


# ---- what read and score share ----


def filed() -> dict[str, Any] | None:
    """The filed numbers read and score use; None, with the reason logged, while predictions_c2.json is a draft or any of
    them is still text."""
    pred = json.loads(PRED_C2.read_text())
    k, id_min = pred["constants"], pred["positive_controls"].get("ID_min", {})
    out = {"ratio_floor": k["ratio_floor"], "gain_min": k["gain_min"], "matrix_share_min": k["matrix_share_min"],
           "T0_vs_LOW_max": pred["do_nothing"].get("T0_vs_LOW_max"), "ID_min_r": id_min.get("r"), "ID_min_ratio": id_min.get("ratio")}  # fmt: skip
    unfiled = [n for n, v in out.items() if not isinstance(v, int | float)]
    if str(pred.get("status", "")).startswith("DRAFT") or unfiled:
        log(f"{PRED_C2.name} is a draft or lacks numbers {unfiled}: file it first (exit 2)")
        return None
    assert out["gain_min"] == pcfg_vpd.GAIN_MIN, out
    return out | {"high_is_existing": k["high_is_existing"]}


@dataclass(frozen=True)
class C1bStates:
    """C1b's state sets, unchanged: labels, recipe and update size per state."""

    meta: dict[str, dict[str, Any]]
    nuisance: dict[str, dict[str, float]]
    calibration: dict[str, str]
    scored: dict[str, str]
    unlearned: list[str]  # read and labelled, not scored: not in the truth


def c1b_states() -> C1bStates:
    c1b = json.loads(C1B_JSON.read_text())
    meta, truth, cal = c1b["meta"], c1b["truth"], json.loads(C1B_CUTOFFS.read_text())["truth"]
    scored = {s: v for s, v in truth.items() if s not in cal}
    unlearned = [s for s in meta if s not in truth]
    assert (len(cal), len(scored), len(unlearned)) == (14, 21, 5), (len(cal), len(scored), len(unlearned))
    assert all(truth[s] == v for s, v in cal.items()) and all(meta[s]["seed"] >= 2 for s in scored)
    assert all((meta[s]["task"], meta[s]["lr"], meta[s]["step"]) == ("N", 1e-5, 1000) for s in unlearned), unlearned
    return C1bStates(meta, c1b["nuisance"], cal, scored, unlearned)


def state_table(c: C1bStates) -> dict[str, tuple[str, Path]]:
    """Each readable state's task and file. The zero update is LOW read as a state, on task P (as the rehearsal)."""
    return {ZERO: ("P", LOW_PATH), **{s: ("ID", p) for s, p in id_paths().items()}, **{s: (c.meta[s]["task"], state_path(s)) for s in c.meta}}


def open_results(k: dict[str, Any]) -> dict[str, Any] | None:
    """c2.json, or a new one; None when it was written under other filed numbers (never dropped by the code)."""
    res = json.loads(OUT.read_text()) if OUT.exists() else {"constants": k, **{p: {} for p in PARTS}}
    if res["constants"] != k:
        log(f"{OUT.name} was written under other filed numbers: move it aside deliberately before rerunning (exit 2)")
        return None
    return res


def open_reader(k: dict[str, Any], res: dict[str, Any]) -> Reader:
    """The final decomposition (FULL's sizes), its alive set from ALIVE if an earlier job saved it, and the split's bases."""
    cfg = D.load_config(CONFIG, steps=FULL.steps)
    model, step = D.load_final(FINAL, load_model(LOW_PATH), cfg, device())
    assert step == FULL.steps, f"{FINAL} is at step {step}"
    if not ALIVE.exists():
        tmp = ALIVE.with_suffix(".tmp")
        torch.save(alive_set(model, FULL.alive_rows), tmp)
        tmp.replace(ALIVE)
    alive = torch.load(ALIVE, map_location="cpu", weights_only=True)
    res["alive"] = {m: int(v.sum()) for m, v in alive.items()}
    log(f"alive total {sum(res['alive'].values())} of {sum(model.module_to_c.values())}: {res['alive']}")
    write_json(OUT, res)
    return reader(model, alive, FULL, k["ratio_floor"])


def read_part(res: dict[str, Any], part: str, sids: list[str], rd: Reader, table: dict[str, tuple[str, Path]], deadline: float) -> int:
    """Read the states of `sids` not yet in res[part]; c2.json is rewritten after each. EXIT_RESUME when the slowest state
    so far would overrun the deadline; a state is not started that cannot finish."""
    for sid in sids:
        if sid in res[part]:
            continue
        took = [r["seconds"] for p in PARTS for r in res[p].values()]
        if time.monotonic() + max(took, default=STATE_SECONDS) > deadline:
            log(f"{part}: deadline before {sid}; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        task, path = table[sid]
        rec = read_one(rd, load_model(path), task)
        res[part][sid] = rec | {"x": xs(rec, res["constants"]["matrix_share_min"])}
        log(f"{part} {sid}: {res[part][sid]}")
        write_json(OUT, res)
    return 0


# ---- read: controls, calibration, cut-offs ----


def controls(res: dict[str, Any], rd: Reader, table: dict[str, tuple[str, Path]], deadline: float) -> int:
    """The zero update alone: over its bounds is exit 1, on every resubmission too. Then the ID states, against their
    filed bars: a reading neither state reaches has no positive control (reported, the run continues)."""
    k = res["constants"]
    rc = read_part(res, "controls", [ZERO], rd, table, deadline)
    if rc:
        return rc
    misses = do_nothing_misses(res["controls"][ZERO], k["gain_min"], k["T0_vs_LOW_max"])
    if misses:
        res["stop"] = misses
        write_json(OUT, res)
        log(f"STOP, do-nothing control missed (apparatus): {misses} (exit 1)")
        return 1
    ids = list(id_paths())
    rc = read_part(res, "controls", ids, rd, table, deadline)
    if rc:
        return rc
    xid = [res["controls"][s]["x"] for s in ids]
    res["positive_control"] = {q: any(x[q] is not None and x[q] >= k[f"ID_min_{q}"] for x in xid) for q in READINGS}
    log(f"controls: positive control per reading {res['positive_control']} (report only); ID readings {dict(zip(ids, xid, strict=True))}")
    write_json(OUT, res)
    return 0


def compute_cutoffs(res: dict[str, Any], truth: dict[str, str]) -> dict[str, Any]:
    cal, high = res["calibration"], res["constants"]["high_is_existing"]
    return {q: revival.cutoff({s: cal[s]["x"][q] for s in truth}, truth, high[q]) for q in READINGS}


def stage_read(deadline: float) -> int:
    k = filed()
    res = open_results(k) if k is not None else None
    if k is None or res is None:
        return 2
    if not FINAL.exists():
        log(f"{FINAL} is missing: the decompose stage comes first (exit 2)")
        return 2
    c = c1b_states()
    rd, table = open_reader(k, res), state_table(c)
    rc = controls(res, rd, table, deadline) or read_part(res, "calibration", list(c.calibration), rd, table, deadline)
    if rc:
        return rc
    cut = json.loads(json.dumps(compute_cutoffs(res, c.calibration)))
    log(f"cut-offs: {cut}")
    if CUTOFFS.exists():
        assert json.loads(CUTOFFS.read_text())["cutoffs"] == cut, f"{CUTOFFS.name} differs from the cut-offs computed now"
        log(f"{CUTOFFS.name} exists and matches")
        return 0
    write_json(CUTOFFS, {"date": time.strftime("%Y-%m-%d"), "status": "computed by c2.py read; commit this file before the score job", "cutoffs": cut,
                         "readings": {q: {s: res["calibration"][s]["x"][q] for s in c.calibration} for q in READINGS}, "truth": c.calibration})  # fmt: skip
    return 0


# ---- score ----


def within_cell_labels(c: C1bStates, states: list[str]) -> dict[str, str]:
    """predictions_c2.json rules.within_cell_update_size (plot_c1b.cell_labeller's rule): per recipe cell (lr, steps), small
    update = existing, the cut-off at the midpoint of the gap between the calibration seeds' classes; a cell holding one
    class in calibration labels every state that class."""
    cell = lambda s: (c.meta[s]["lr"], c.meta[s]["step"])
    size = lambda s: c.nuisance[s]["update_size"]
    lab = {}
    for s in states:
        cal = [t for t in c.calibration if cell(t) == cell(s)]
        assert cal, cell(s)
        ex, nw = ([size(t) for t in cal if c.calibration[t] == v] for v in ("existing", "new"))
        if not ex or not nw:
            lab[s] = "existing" if ex else "new"
            continue
        assert max(ex) < min(nw), cell(s)
        lab[s] = "existing" if size(s) <= (max(ex) + min(nw)) / 2 else "new"
    return lab


def scores(res: dict[str, Any], c: C1bStates) -> dict[str, Any]:
    """C1b's rule on each reading (all states, margin 2; per run, the 10K saves, margin 1; by cell, described), and the
    within-cell update-size labeller beside it, on all states and on each reading's own non-void states (a tie is a tie)."""
    last = {s: v for s, v in c.scored.items() if c.meta[s]["step"] == FT_SAVES[-1]}
    within = within_cell_labels(c, list(c.scored))
    nu = {"nuisance": c.nuisance}
    out: dict[str, Any] = {"within_cell_update_size": {"labels": within, "all": revival.score(within, c.scored), "per_run": revival.score(within, last)}}
    for q in READINGS:
        lab = {s: res["scored"][s]["labels"][q] for s in c.scored}
        cells: dict[str, dict[str, str]] = {}
        for s, v in c.scored.items():
            cells.setdefault(f"{c.meta[s]['task']}_{lr_key(c.meta[s]['lr'])}_{c.meta[s]['step']}", {})[s] = v
        out[q] = {"all": score_one(nu, lab, c.scored, revival.MARGIN_ALL), "per_run": score_one(nu, lab, last, revival.MARGIN_RUN),
                  "by_cell": {n: {f: v for f, v in revival.score(lab, t).items() if f in ("n", "agree", "voided")} for n, t in sorted(cells.items())},
                  "within_cell_agree_on_same_states": revival.score(lab, c.scored, {"within": within})["nuisance_agree"]["within"]}  # fmt: skip
    return out


def stage_score(deadline: float) -> int:
    k = filed()
    res = open_results(k) if k is not None else None
    if k is None or res is None:
        return 2
    if not CUTOFFS.exists() or not committed(CUTOFFS):
        log(f"{CUTOFFS.name} is missing or not committed unmodified: commit the read job's file first (exit 2)")
        return 2
    c = c1b_states()
    cut = json.loads(CUTOFFS.read_text())
    assert cut["cutoffs"] == json.loads(json.dumps(compute_cutoffs(res, c.calibration))), f"{CUTOFFS.name} differs from c2.json's calibration"
    res["cutoffs"] = cut
    rd, table = open_reader(k, res), state_table(c)
    rc = read_part(res, "scored", list(c.scored), rd, table, deadline) or read_part(res, "unscored", c.unlearned, rd, table, deadline)
    if rc:
        return rc
    for part in ("controls", "scored", "unscored"):
        for sid, row in res[part].items():
            row["labels"] = {q: revival.label_by_cutoff(row["x"][q], cut["cutoffs"][q]) for q in READINGS}
            log(f"{part} {sid}: x {row['x']}; labels {row['labels']}; truth {c.scored.get(sid, 'not scored')}")
    res["update_size"] = {s: c.nuisance[s]["update_size"] for s in (*c.scored, *c.unlearned)}
    res["scores"] = scores(res, c)
    res["complete"] = True
    write_json(OUT, res)
    log(f"update sizes: {res['update_size']}")
    log(f"scores: {json.dumps(res['scores'])}")
    return 0


# ---- the full decomposition ----


def stage_decompose(deadline_min: float) -> int:
    pred = json.loads(PRED_C2.read_text())
    if pred.get("status", "").startswith("DRAFT"):
        log(f"{PRED_C2.name} is a draft: the full decomposition waits for the filed predictions (exit 2)")
        return 2
    deadline = time.monotonic() + 60 * deadline_min
    cfg = D.load_config(CONFIG, steps=FULL.steps)
    checks = pred["decomposition_stops"]["early_checks"]
    problems = D.early_check_problems(checks, cfg.eval_freq)
    if problems:
        log(f"decomposition_stops.early_checks cannot work: {problems} (exit 2)")
        return 2
    torch.cuda.reset_peak_memory_stats()
    d = fresh(cfg)
    train, ev = data(cfg)
    plan = D.Plan(FULL.steps, FULL.save_every, FULL.log_every, deadline)
    step, _ = D.run(d, train.get, TRAIN_CKPT, plan, ev, log=log, on_eval=D.early_stop(checks, d, TRAIN_CKPT, log),
                    on_record=D.jsonl_appender(DECOMP_DIR / "decomp_log.jsonl"))  # fmt: skip
    train.close()
    if step < FULL.steps:
        log(f"decompose: step {step} of {FULL.steps}; resubmit to resume (exit {EXIT_RESUME})")
        return EXIT_RESUME
    D.save_final(FINAL, d, step)
    log(f"decompose: finished at step {step}: {FINAL}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("decompose", "read", "score"))
    ap.add_argument("--deadline-min", type=float, default=22.0)
    a = ap.parse_args()
    assert PRED_C2.exists(), f"{PRED_C2} is missing: predictions are filed before any run"
    log(f"c2 {a.stage} on {torch.cuda.get_device_name()}")
    deadline = time.monotonic() + 60 * a.deadline_min
    stages: dict[str, Callable[[], int]] = {"decompose": lambda: stage_decompose(a.deadline_min), "read": lambda: stage_read(deadline),
                                            "score": lambda: stage_score(deadline)}  # fmt: skip
    sys.exit(stages[a.stage]())


if __name__ == "__main__":
    main()
