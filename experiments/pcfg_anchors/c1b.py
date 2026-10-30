"""E55 Stage C1b: the non-VPD instruments, cut-offs calibrated on held-out seeds, on a truth set the recipe cannot read.
Plan: reports/plans/2026-10-30_plan_e55-c1b-refined-instruments.md. Rules, thresholds, expected values, edge cases and
departures: results/predictions_c1b.json (read, never edited; the run refuses a draft). Instruments and rules:
src/commute/revival.py.

Truth set. Every run fine-tunes LOW with Stage A's recipe (answer-only loss, AdamW, batch 96, 20% warm-up then cosine
over 10K steps; saves at 1K and 10K in results/large/c1b/<key>_<step>.pt, resumed from <key>_train.pt) on natural
'C b 40' rows from Stage A's stream default_rng([FT_SEED0 + seed, chunk]). Task P answers count(a) + 1 (reuses count a by
construction), task N count(b) (Stage A's N recipe: new). Grid: lr in constants.lrs x tasks x seeds (calibration_seeds,
scored_seeds); constants.existing_runs are Stage A's arms already trained on exactly that recipe (states in
results/large/a2/). Each state is read by Stage A's checks (run.read_state) and its task accuracy on 4,000 fresh rows,
and placed: task accuracy < task_acc_min -> "not yet learned"; checks agree with construction (P: count a through,
count b not through; N: count a not through) -> "truth" (P existing, N new); else "disagreement".

Instruments, per state, against LOW (each reading x; 'existing' on the side the filed cut-off names):
  prune    best over K of (top-K revival accuracy - mean of 20 random K-unit draws) on 'C a 40' (C1's readings);
  reverse  steps to 95% count a at constants.reverse_lr for every state, cap reverse_cap inside a reverse_schedule
           schedule, over scratch's (NEVER) steps; low = existing;
  steer    E52's gain share S of the mean residual shift at `=` (C1's readings) on Stage A's qualifying 'C b 40' rows;
  probe    selectivity of a linear probe on LOW's residual at `=` predicting the state's own answers on natural
           'C b 40' rows, minus a shuffled-label probe, best block (revival.possession).

Stages (each refuses while results/c1b_stop.json exists; a stop row writes it and exits 1):
  controls   model-free checks; LOW reproduces Stage A; pruning control; scratch; probe controls on LOW; steering and
             probe on the ID states (Amendment 5, 2K). Stop rows: LOW off by > reproduce_tol, scratch <= 1K or capped.
  train      (needs controls) each run trained and its states checked -> c1b.json after each run; at the end the stop
             rows: task P fails (< task_acc_min on >= p_fail_stop of the P 10K states), >= disagreement_stop
             construction-check disagreements among scored-seed states.
  calibrate  (needs every state checked) the instruments on the calibration-seed truth states -> c1b.json after each;
             then the cut-offs (revival.cutoff) -> results/c1b_cutoffs.json, to be committed before score.
  score      (refuses unless c1b_cutoffs.json is committed, unmodified, and equal to the cut-offs recomputed from
             c1b.json) the instruments on the scored-seed truth states, labels and scores after each; C1's states
             re-labelled from c1.json's readings, described.
Exit 3 at the deadline (a reverse fine-tune cut by it is rerun); resubmit the same stage to resume. Every reading and
score is printed to the log.

Usage (repo root, GPU): python experiments/pcfg_anchors/c1b.py {controls,train,calibrate,score}
(cluster/pcfg_anchors.sbatch c1b STAGE)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from amend3 import texts_of
from amend5 import A5_DIR, PRED5, prompt_seqs
from amend5 import OUT as OUT5
from c1 import (
    N_RFT_EVAL,
    PRUNE_SEED,
    REV_SEED,
    Data,
    build_data,
    rft_batches,
    rows_ok,
    summarise_prune,
)
from run import (
    A1,
    A2_DIR,
    ARMS,
    EVAL_SEED,
    EXIT_RESUME,
    FT_BATCH,
    FT_ITERS,
    FT_SAVES,
    FT_SEED0,
    LARGE,
    PRED,
    RESULTS,
    WEIGHT_DECAY,
    A2Eval,
    Arm,
    a1_row,
    a2_eval_data,
    device,
    eval_seqs,
    filed_bars,
    load_model,
    log,
    read_state,
    state_key,
    state_path,
    write_json,
)

from commute import pcfg, revival
from commute.interchange import make_pairs, patch_shares, run_at_art
from commute.mingpt import GPT
from commute.pcfg import A, B
from commute.pcfg_train import Batches, Run, Schedule, answer_loss, train

PRED_C1B = RESULTS / "predictions_c1b.json"
OUT, STOP, CUTOFFS = RESULTS / "c1b.json", RESULTS / "c1b_stop.json", RESULTS / "c1b_cutoffs.json"
C1B_DIR, LOW = LARGE / "c1b", LARGE / "a1" / "pt_LOW.pt"
CHUNK, PROBE_SEED, PROBE_HELD = 100, 0, [EVAL_SEED, 41]  # probe: one stream and one held-out set for every state
INSTRUMENTS = ("prune", "reverse", "steer", "probe")
HIGH_IS_EXISTING = {"prune": True, "reverse": False, "steer": True, "probe": True}
RECIPE = {"iters": FT_ITERS, "batch": FT_BATCH, "saves": list(FT_SAVES), "prune_k": list(revival.PRUNE_K),
          "n_random": revival.N_RANDOM, "eval_every": revival.EVAL_EVERY, "n_rft_eval": N_RFT_EVAL,
          "seeds": [PRUNE_SEED, REV_SEED, PROBE_SEED, *PROBE_HELD]}  # fmt: skip  (compared with c1b.json's on resume)


@dataclass(frozen=True)
class RunC:
    task: str  # "P" or "N"
    lr: float
    seed: int
    a2: str | None  # the Stage A arm already trained on this recipe (constants.existing_runs), else None

    @property
    def name(self) -> str:
        return f"{self.task}_{lr_key(self.lr)}_s{self.seed}"

    def key(self) -> str:
        if self.a2 is not None:
            return state_key(self.a2, ARMS[self.a2])
        recipe = {"task": self.task, "lr": self.lr, "seed": self.seed, "prior": "LOW", "iters": FT_ITERS, "batch": FT_BATCH}
        return f"{self.name}-{hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:8]}"

    def sid(self, step: int) -> str:
        return f"a2/{self.a2}/{step}" if self.a2 is not None else f"c1b/{self.name}/{step}"

    def path(self, step: int | str) -> Path:
        return state_path(self.key(), step, A2_DIR if self.a2 is not None else C1B_DIR)


def lr_key(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")  # 1e-4, as the plan writes it


def all_runs(k: dict[str, Any]) -> list[RunC]:
    """Calibration seeds first (calibrate reads them first), then scored; P before N; lrs in constants order."""
    return [RunC(t, lr, s, k["existing_runs"].get(f"{t}_{lr_key(lr)}_s{s}"))
            for s in (*k["calibration_seeds"], *k["scored_seeds"]) for t in ("P", "N") for lr in k["lrs"]]  # fmt: skip


def task_rows(task: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """Natural 'C b 40' rows (Stage A's N stream: task_batch at C_Tr = 0); task P relabels them count(a) + 1."""
    seq = pcfg.task_batch(rng, n, "C", B, c_rate=0.0)[0]
    if task == "P":
        seq[:, pcfg.ANS] = pcfg.NUM0 + pcfg.count(texts_of(seq), A) + 1
    return seq


def count_a_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """Natural 'C b 40' rows answered count(a): the probe's positive control on LOW."""
    seq = task_rows("N", rng, n)
    seq[:, pcfg.ANS] = pcfg.NUM0 + pcfg.count(texts_of(seq), A)
    return seq


def natural_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    return task_rows("N", rng, n)


def finetune(r: RunC, deadline: float) -> bool:
    """run.finetune's recipe and saves with this run's task rows; False when the deadline stopped it."""
    model = load_model(LOW)
    torch.manual_seed(FT_SEED0 + r.seed)
    spec = Run(Schedule(FT_ITERS, r.lr), WEIGHT_DECAY, save_every=1000, log_every=500, loss=answer_loss)
    for it in FT_SAVES:
        if r.path(it).exists():
            continue
        batches = Batches(lambda rng, n: task_rows(r.task, rng, n), FT_SEED0 + r.seed, FT_BATCH, CHUNK, device())
        step, _ = train(model, batches, spec, r.path("train"), stop_at=it, deadline=deadline, log=lambda m: log(f"{r.name} {m}"))
        if step < it:
            return False
        tmp = r.path(it).with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "step": step}, tmp)
        tmp.replace(r.path(it))
    return True


def update_size(model: GPT, low: GPT) -> float:
    """||W - W0||_F / ||W0||_F over every parameter (the update-size nuisance labeller)."""
    p0 = dict(low.named_parameters())
    num = sum(float((p - p0[n]).double().pow(2).sum()) for n, p in model.named_parameters())
    return (num / sum(float(p.double().pow(2).sum()) for p in p0.values())) ** 0.5


def check_state(model: GPT, low: GPT, ev: A2Eval, task_eval: np.ndarray, bars: dict[str, Any]) -> dict[str, Any]:
    """Stage A's checks (run.read_state), task accuracy of P and N on the same 4,000 fresh natural rows, update size."""
    out: dict[str, Any] = dict(read_state(model, ev, bars))
    pred, _ = run_at_art(model, task_eval)
    t = texts_of(task_eval)
    out["task_acc"] = {"P": float((pred == pcfg.NUM0 + pcfg.count(t, A) + 1).mean()), "N": float((pred == pcfg.NUM0 + pcfg.count(t, B)).mean())}
    out["update_size"] = update_size(model, low)
    return out


def placement(task: str, chk: dict[str, Any], k: dict[str, Any]) -> str:
    if chk["task_acc"][task] < k["task_acc_min"]:
        return "not yet learned"
    a, b = chk["check_count_a"]["label"], chk["check_count_b"]["label"]
    agrees = (a == "through" and b != "through") if task == "P" else a == "not through"
    return "truth" if agrees else "disagreement"


def id_paths() -> dict[str, Path]:
    arms = json.loads(OUT5.read_text())["arms"]
    return {f"amend5/{a}/2000": state_path(arms[a]["state_key"], 2000, A5_DIR) for a in ("ID_s0", "ID_s1")}


def model_free_checks(k: dict[str, Any], data: Data) -> dict[str, bool]:
    runs = all_runs(k)
    p0, n0 = (task_rows(t, np.random.default_rng([FT_SEED0, 0]), FT_BATCH * CHUNK) for t in ("P", "N"))
    tp = texts_of(p0)
    return {"rows_P_chunk0": bool((p0[:, pcfg.ANS] - pcfg.NUM0 == pcfg.count(tp, A) + 1).all()) and rows_ok(n0, B),
            "P_N_same_seed_share_texts": bool((tp == texts_of(n0)).all()),
            "existing_runs_are_the_N_recipe": all(ARMS[r.a2] == Arm("LOW", r.lr, 0.0, r.seed) for r in runs if r.a2 is not None),
            "existing_runs_listed": len([r for r in runs if r.a2 is not None]) == len(k["existing_runs"]),
            "existing_state_files": all(r.path(s).exists() for r in runs if r.a2 is not None for s in FT_SAVES),
            "id_state_files": all(p.exists() for p in id_paths().values()),
            "seeds_disjoint": not set(k["calibration_seeds"]) & set(k["scored_seeds"]),
            "rows_score_eval_steer": rows_ok(data.score, A) and rows_ok(data.eval, A) and rows_ok(data.steer, B),
            "plan_bar_table": [revival.pass_bar(n, n // 2) for n in (24, 21, 18, 16)] == [17, 15, 13, 12]}  # fmt: skip


def stop(res: dict[str, Any], why: list[str]) -> None:
    res["stop"] = why
    write_json(OUT, res)
    write_json(STOP, {"why": why, "date": time.strftime("%Y-%m-%d %H:%M")})
    log(f"STOP: {why}; see {STOP.name} (exit 1)")
    sys.exit(1)


# ---- controls ----


def id_prompt() -> tuple[int, int]:
    fam, op = json.loads(PRED5.read_text())["constants"]["id_prompt"]
    return pcfg.FAMILIES.index(fam), pcfg.TERMINALS.index(op)


def controls(res: dict[str, Any], k: dict[str, Any], data: Data, bars: dict[str, Any], deadline: float) -> int:
    low, out = load_model(LOW), {}
    pairs0 = make_pairs(np.random.default_rng([EVAL_SEED, 30, 0, A]), A)  # a1c's sample set 0
    la = bars["l_star_a"]
    out["LOW"] = {"B_a": revival.accuracy(low, data.eval), "B_a_stage_a": a1_row("LOW")["count_a"],
                  "P_a": float(patch_shares(low, pairs0, A)[la - 1]),
                  "P_a_stage_a": json.loads((RESULTS / "a1c_patch.json").read_text())["patch"]["P_a_positive"][0][la - 1]}  # fmt: skip
    log(f"control LOW reproduces Stage A: {out['LOW']}")
    if max(abs(out["LOW"]["B_a"] - out["LOW"]["B_a_stage_a"]), abs(out["LOW"]["P_a"] - out["LOW"]["P_a_stage_a"])) > k["reproduce_tol"]:
        stop(res | {"controls": out}, ["LOW does not reproduce Stage A's B_a / P_a (apparatus)"])
    torch.manual_seed(REV_SEED)
    sc = revival.reverse_finetune(load_model(LARGE / "a1" / "pt_NEVER.pt"), rft_batches(), k["reverse_lr"], WEIGHT_DECAY,
                                  data.eval[:N_RFT_EVAL], deadline, k["reverse_cap"], k["reverse_schedule"])  # fmt: skip
    if sc["end"] == "deadline":
        return EXIT_RESUME
    out["scratch"] = sc
    log(f"control scratch (NEVER, lr {k['reverse_lr']:.0e}): steps {sc['steps']} ({sc['end']}); curve {sc['curve']}")
    if sc["steps"] is None or sc["steps"] <= revival.SCRATCH_VOID_MAX:
        stop(res | {"controls": out}, [f"scratch steps {sc['steps']} ({sc['end']}): reverse fine-tuning void"])
    out["prune_LOW_importance"] = revival.prune_readings(low, data.score, data.eval, PRUNE_SEED, sign=-1.0)
    out["prune_control_pass"] = revival.prune_control(out["prune_LOW_importance"])
    log(f"control pruning (LOW, sign -1) pass {out['prune_control_pass']}: {summarise_prune(out['prune_LOW_importance'])}")
    t = time.monotonic()
    out["probe_LOW"] = {"count_a": revival.possession(low, count_a_rows, PROBE_SEED, PROBE_HELD),
                        "count_b": revival.possession(low, natural_rows, PROBE_SEED, PROBE_HELD),
                        "own_answers": revival.possession(low, natural_rows, PROBE_SEED, PROBE_HELD, low)}  # fmt: skip
    out["probe_seconds"] = (time.monotonic() - t) / 3
    for name, p in out["probe_LOW"].items():
        log(f"control probe LOW {name}: selectivity {p['x']:.3f} at block {p['block']}; acc {np.round(p['acc'], 3).tolist()} shuffled {np.round(p['shuffled'], 3).tolist()}")
    log(f"probe: {out['probe_seconds']:.0f} s each")
    rows_id = prompt_seqs(texts_of(data.steer), id_prompt())
    out["ID"] = {}
    for sid, path in id_paths().items():
        st = load_model(path)
        out["ID"][sid] = {"steer": revival.steering(low, st, rows_id), "probe": revival.possession(low, natural_rows, PROBE_SEED, PROBE_HELD, st)}
        log(f"control {sid}: steer S {out['ID'][sid]['steer']['S']:.3f} (G {out['ID'][sid]['steer']['G_score']:.3f}) on its own prompt; probe {out['ID'][sid]['probe']['x']:.3f}")
    out["void"] = {"prune": not out["prune_control_pass"],
                   "probe": out["probe_LOW"]["count_a"]["x"] < k["probe_void_low_count_a"] or all(v["probe"]["x"] < k["probe_void_id"] for v in out["ID"].values())}  # fmt: skip
    out["steer_positive_control"] = any(v["steer"]["S"] >= k["steer_pc_min"] for v in out["ID"].values())
    log(f"controls: instruments void {out['void']}; steering positive control {out['steer_positive_control']} (report only)")
    res["controls"] = out
    write_json(OUT, res)
    return 0


# ---- train and checks ----


def train_stage(res: dict[str, Any], k: dict[str, Any], bars: dict[str, Any], deadline: float) -> int:
    low, ev, task_eval = load_model(LOW), a2_eval_data(), eval_seqs(B, A1.eval_n)
    C1B_DIR.mkdir(parents=True, exist_ok=True)
    for r in all_runs(k):
        if all(r.sid(s) in res["checks"] for s in FT_SAVES):
            continue
        if time.monotonic() > deadline or not finetune(r, deadline):
            log(f"{r.name}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        for s in FT_SAVES:
            chk = check_state(load_model(r.path(s)), low, ev, task_eval, bars)
            sid, place = r.sid(s), placement(r.task, chk, k)
            res["checks"][sid] = chk
            res["placement"][sid] = place
            res["nuisance"][sid] = {"lr": r.lr, "steps": s, "update_size": chk["update_size"]}
            res["meta"][sid] = {"run": r.name, "task": r.task, "lr": r.lr, "seed": r.seed, "step": s, "path": str(r.path(s))}
            if place == "truth":
                res["truth"][sid] = "existing" if r.task == "P" else "new"
            ca, cb = chk["check_count_a"], chk["check_count_b"]
            log(f"{sid} ({r.name}): {place}; task acc P {chk['task_acc']['P']:.3f} N {chk['task_acc']['N']:.3f}; count a B {ca['B']:.3f} "
                f"P {ca['P']:.3f} {ca['label']}; count b B {cb['B']:.3f} P {cb['P']:.3f} {cb['label']}; update size {chk['update_size']:.4f}")  # fmt: skip
        write_json(OUT, res)
        r.path("train").unlink(missing_ok=True)
    return train_stops(res, k)


def train_stops(res: dict[str, Any], k: dict[str, Any]) -> int:
    meta, chk = res["meta"], res["checks"]
    p_miss = [s for s in chk if meta[s]["task"] == "P" and meta[s]["step"] == FT_SAVES[-1] and chk[s]["task_acc"]["P"] < k["task_acc_min"]]
    dis = [s for s, p in res["placement"].items() if p == "disagreement" and meta[s]["seed"] in k["scored_seeds"]]
    res["train_stops"] = {"P_10K_below_task_acc_min": p_miss, "scored_seed_disagreements": dis,
                          "all_disagreements": [s for s, p in res["placement"].items() if p == "disagreement"],
                          "recipe_ceiling_full": ceilings(res, {s: v for s, v in res["truth"].items() if meta[s]["seed"] in k["scored_seeds"]})}  # fmt: skip
    log(f"train stop rows: {res['train_stops']}")
    why = [f"task P fails: {len(p_miss)} P 10K states below {k['task_acc_min']}"] * (len(p_miss) >= k["p_fail_stop"])
    why += [f"{len(dis)} construction-check disagreements among scored-seed states"] * (len(dis) >= k["disagreement_stop"])
    if why:
        stop(res, why)
    write_json(OUT, res)
    return 0


# ---- instrument readings ----


def read_instruments(path: Path, k: dict[str, Any], data: Data, deadline: float) -> dict[str, Any] | None:
    """The four readings of one state; None when the deadline cut its reverse fine-tune (run last: it trains in place)."""
    tuned, low = load_model(path), load_model(LOW)
    out: dict[str, Any] = {"steer": revival.steering(low, tuned, data.steer),
                           "probe": revival.possession(low, natural_rows, PROBE_SEED, PROBE_HELD, tuned),
                           "prune": revival.prune_readings(tuned, data.score, data.eval, PRUNE_SEED)}  # fmt: skip
    torch.manual_seed(REV_SEED)
    out["reverse"] = revival.reverse_finetune(tuned, rft_batches(), k["reverse_lr"], WEIGHT_DECAY, data.eval[:N_RFT_EVAL],
                                              deadline, k["reverse_cap"], k["reverse_schedule"])  # fmt: skip
    return None if out["reverse"]["end"] == "deadline" else out


def prune_x(p: dict[str, Any], ctl: dict[str, Any]) -> float | None:
    """Best over K of top-K accuracy - mean of the random draws; None if the pruning control failed or count a is
    already >= TARGET (nothing to revive)."""
    if ctl["void"]["prune"] or p["acc0"] >= revival.TARGET:
        return None
    return max(revival.revival_z(p[str(kk)]["top"], p[str(kk)]["random"], p["n_eval"])[0] for kk in revival.PRUNE_K)


def reverse_x(steps: int | None, ctl: dict[str, Any], k: dict[str, Any]) -> float | None:
    """Steps to TARGET (a capped run counts as the cap) over scratch's; None if the state starts at TARGET."""
    return None if steps == 0 else (k["reverse_cap"] if steps is None else steps) / ctl["scratch"]["steps"]


def steer_x(s: dict[str, Any]) -> float | None:
    return None if revival.steer_label(s) == "void" else float(s["S"])


def xs(row: dict[str, Any], ctl: dict[str, Any], k: dict[str, Any]) -> dict[str, float | None]:
    """Each instrument's reading x, None when void (the probe: when its controls failed)."""
    return {"prune": prune_x(row["prune"], ctl), "reverse": reverse_x(row["reverse"]["steps"], ctl, k),
            "steer": steer_x(row["steer"]), "probe": None if ctl["void"]["probe"] else float(row["probe"]["x"])}  # fmt: skip


def read_states(res: dict[str, Any], part: str, sids: list[str], k: dict[str, Any], data: Data, deadline: float,
                after: Any = None) -> int:  # fmt: skip
    for sid in sids:
        if sid in res[part]:
            continue
        row = None if time.monotonic() > deadline else read_instruments(Path(res["meta"][sid]["path"]), k, data, deadline)
        if row is None:
            log(f"{sid}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        res[part][sid] = row | {"x": xs(row, res["controls"], k), "truth": res["truth"][sid]}
        if after is not None:
            after(res)
        st = row["steer"]
        log(f"{part} {sid} ({res['truth'][sid]}): x {res[part][sid]['x']}; reverse steps {row['reverse']['steps']} ({row['reverse']['end']}); "
            f"steer S {st['S']:.3f} block {st['block']} G {st['G_score']:.3f}{' (gain < 0.01: flag)' if st['G_score'] < 0.01 else ''}; "
            f"probe block {row['probe']['block']}; prune {summarise_prune(row['prune'])}" + (f"; labels {res[part][sid]['labels']}" if "labels" in res[part][sid] else ""))  # fmt: skip
        write_json(OUT, res)
    return 0


def truth_at(res: dict[str, Any], seeds: list[int]) -> list[str]:
    return [s for s in res["truth"] if res["meta"][s]["seed"] in seeds]


def compute_cutoffs(res: dict[str, Any]) -> dict[str, Any]:
    cal = res["calibration"]
    return {i: revival.cutoff({s: r["x"][i] for s, r in cal.items()}, {s: r["truth"] for s, r in cal.items()}, HIGH_IS_EXISTING[i])
            for i in INSTRUMENTS}  # fmt: skip


def calibrate(res: dict[str, Any], k: dict[str, Any], data: Data, deadline: float) -> int:
    rc = read_states(res, "calibration", truth_at(res, k["calibration_seeds"]), k, data, deadline)
    if rc:
        return rc
    cut = compute_cutoffs(res)
    for i, c in cut.items():
        log(f"cut-off {i}: {c}")
    file = {"date": time.strftime("%Y-%m-%d"), "status": "computed by c1b.py calibrate; commit this file before the score job",
            "cutoffs": cut, "readings": {i: {s: r["x"][i] for s, r in res["calibration"].items()} for i in INSTRUMENTS},
            "truth": {s: r["truth"] for s, r in res["calibration"].items()}}  # fmt: skip
    if CUTOFFS.exists():
        assert json.loads(CUTOFFS.read_text())["cutoffs"] == json.loads(json.dumps(cut)), f"{CUTOFFS.name} differs from the cut-offs computed now"
        log(f"{CUTOFFS.name} exists and matches")
    else:
        write_json(CUTOFFS, file)
    return 0


# ---- scores ----


def ceilings(res: dict[str, Any], truth: dict[str, str]) -> dict[str, Any]:
    """Each recipe-only labeller's hindsight ceiling on these states (labels dropped from the summary)."""
    out = {n: revival.ceiling({s: res["nuisance"][s][n] for s in truth}, truth) for n in ("lr", "steps", "update_size")}
    return {n: {q: v for q, v in c.items() if q != "labels"} for n, c in out.items()} | {"n": len(truth)}


def score_one(res: dict[str, Any], lab: dict[str, str], truth: dict[str, str], margin: int) -> dict[str, Any]:
    """revival.score against the three recipe-only labellers at their hindsight best on the instrument's own non-void
    states (the ceiling n_c is the best of the three)."""
    used = {s: v for s, v in truth.items() if lab.get(s) in ("existing", "new")}
    ceil = {n: revival.ceiling({s: res["nuisance"][s][n] for s in used}, used) for n in ("lr", "steps", "update_size")}
    sc = revival.score(lab, truth, {n: c["labels"] for n, c in ceil.items()}, margin)
    return sc | {"n_c": max((c["agree"] for c in ceil.values()), default=0),
                 "ceiling": {n: {q: v for q, v in c.items() if q != "labels"} for n, c in ceil.items()}}  # fmt: skip


def scores(res: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    cut, meta = res["cutoffs"]["cutoffs"], res["meta"]
    truth = {s: res["truth"][s] for s in truth_at(res, k["scored_seeds"])}
    last = {s: v for s, v in truth.items() if meta[s]["step"] == FT_SAVES[-1]}
    out: dict[str, Any] = {}
    for i in INSTRUMENTS:
        lab = {s: revival.label_by_cutoff(r["x"][i], cut[i]) for s, r in res["scored"].items()}
        cells: dict[str, dict[str, str]] = {}
        for s, v in truth.items():
            cells.setdefault(f"{meta[s]['task']}_{lr_key(meta[s]['lr'])}_{meta[s]['step']}", {})[s] = v
        out[i] = {"all": score_one(res, lab, truth, revival.MARGIN_ALL), "per_run": score_one(res, lab, last, revival.MARGIN_RUN),
                  "by_cell": {c: {q: v for q, v in revival.score(lab, t).items() if q in ("n", "agree", "voided")} for c, t in sorted(cells.items())}}  # fmt: skip
    return out


def committed(path: Path) -> bool:
    """The file is tracked by git and has no uncommitted change."""
    git = lambda *a: subprocess.run(["git", *a], check=False, cwd=RESULTS, capture_output=True).returncode == 0
    return git("ls-files", "--error-unmatch", path.name) and git("diff", "--quiet", "HEAD", "--", path.name)


def c1_described(res: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    """C1's states re-labelled with the new cut-offs from c1.json's readings (no new reading): prune and steer as read;
    reverse only where C1 read it at reverse_lr (its first reverse_cap steps are this recipe's), else void; probe not
    read. Agreement with C1's scored truth, described only."""
    c1 = json.loads((RESULTS / "c1.json").read_text())
    cut, ctl, out = res["cutoffs"]["cutoffs"], res["controls"], {}
    for sid, row in c1["states"].items():
        x: dict[str, float | None] = dict.fromkeys(INSTRUMENTS)
        x["steer"] = steer_x(row["steer"])
        if "prune" in row:  # not steering only
            r = row["reverse"]
            x["prune"] = prune_x(row["prune"], ctl)
            if abs(r["lr"] - k["reverse_lr"]) < 1e-12:
                x["reverse"] = reverse_x(None if r["steps"] is None or r["steps"] > k["reverse_cap"] else r["steps"], ctl, k)
        out[sid] = {"x": x, "labels": {i: revival.label_by_cutoff(x[i], cut[i]) for i in INSTRUMENTS}}
    truth = c1["truth"]["scored"]
    agree = {i: revival.score({s: v["labels"][i] for s, v in out.items()}, truth) for i in INSTRUMENTS}
    return {"states": out, "agreement": {i: {q: a[q] for q in ("n", "agree", "p_value")} for i, a in agree.items()}}


def score_stage(res: dict[str, Any], k: dict[str, Any], data: Data, deadline: float) -> int:
    if not CUTOFFS.exists() or not committed(CUTOFFS):
        log(f"{CUTOFFS.name} is missing or not committed unmodified: commit the calibrate job's file first (exit 2)")
        return 2
    filed = json.loads(CUTOFFS.read_text())
    assert filed["cutoffs"] == json.loads(json.dumps(compute_cutoffs(res))), f"{CUTOFFS.name} differs from c1b.json's calibration"
    res["cutoffs"] = filed

    def relabel(r: dict[str, Any]) -> None:
        for row in r["scored"].values():
            row["labels"] = {i: revival.label_by_cutoff(row["x"][i], filed["cutoffs"][i]) for i in INSTRUMENTS}
        r["scores"] = scores(r, k)

    rc = read_states(res, "scored", truth_at(res, k["scored_seeds"]), k, data, deadline, relabel)
    if rc:
        return rc
    relabel(res)
    res["c1_described"] = c1_described(res, k)
    res["complete"] = True
    write_json(OUT, res)
    log(f"scores: {json.dumps(res['scores'])}")
    log(f"C1's states re-labelled (described): {json.dumps(res['c1_described']['agreement'])}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("controls", "train", "calibrate", "score"))
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    deadline = time.monotonic() + 60 * args.deadline_min
    if not PRED_C1B.exists() or str(json.loads(PRED_C1B.read_text()).get("status", "")).startswith("draft"):
        sys.exit(f"file the predictions first (a draft is refused): {PRED_C1B}")
    if STOP.exists():
        sys.exit(f"a stop row fired ({STOP.read_text().strip()}); nothing runs until the lead clears it (exit 1)")
    k = json.loads(PRED_C1B.read_text())["constants"]
    bars = filed_bars(json.loads(PRED.read_text())["a2_bars"], json.loads((RESULTS / "a1c_patch.json").read_text())["a2_bars"])
    assert bars is not None, "Stage A's bars are filed"
    data = build_data()
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    keep = prev.get("recipe") == RECIPE and prev.get("constants") == k
    if prev and not keep:
        log("c1b.json was written under another recipe or constants: its readings are dropped")
    res: dict[str, Any] = {"recipe": RECIPE, "constants": k, "model_free": model_free_checks(k, data)}
    for part in ("checks", "placement", "truth", "nuisance", "meta", "calibration", "scored"):
        res[part] = prev.get(part, {}) if keep else {}
    for part in ("controls", "train_stops"):
        res[part] = prev.get(part) if keep else None
    log(f"model-free checks {res['model_free']}")
    if not all(res["model_free"].values()):
        stop(res, [f"model-free check missed: {[c for c, v in res['model_free'].items() if not v]}"])
    trained = res["train_stops"] is not None and len(res["checks"]) == len(all_runs(k)) * len(FT_SAVES)
    ready = {"controls": res["controls"] is None, "train": res["controls"] is not None, "calibrate": trained,
             "score": trained and len(res["calibration"]) == len(truth_at(res, k["calibration_seeds"]))}  # fmt: skip
    if not ready[args.stage]:
        log(f"{args.stage}: not ready (controls run once, then train -> calibrate -> score; see c1b.json; exit 2)")
        sys.exit(2)
    stages = {"controls": lambda: controls(res, k, data, bars, deadline), "train": lambda: train_stage(res, k, bars, deadline),
              "calibrate": lambda: calibrate(res, k, data, deadline), "score": lambda: score_stage(res, k, data, deadline)}  # fmt: skip
    sys.exit(stages[args.stage]())


if __name__ == "__main__":
    main()
