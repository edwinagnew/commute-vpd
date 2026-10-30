"""E55 Stage C1: non-VPD instruments scored against the reuse ground truth. Plan: reports/plans/2026-10-29_plan_reuse-stage1.md
(Stage C, C1). Truth set, rules, pass bars and departures: results/predictions_c1.json (read, never edited; the run
refuses a draft). Instruments: src/commute/revival.py.

States: those listed in predictions_c1.json constants (truth_scored, amend5b_states, steering_only, class_B), ids
'<src>/<arm>/<step>' with src 'a2' (Stage A, results/a2_<arm>.json, states in results/large/a2/), 'amend5'
(results/amend5.json, results/large/amend5/) or 'amend5b' (results/amend5b.json, results/large/amend5b/; each placed
in the scored set or class B by its own class, truth_sets). Per state, against its base (LOW, or HIGH for R):
  pruning revival (not steering_only): count-a accuracy on Stage A's 4,000 'C a 40' rows after removing the top-K units
      by grad . w of the count-a loss, K in {5, 15}, against 20 random K-unit draws;
  reverse fine-tuning (not steering_only): steps to 95% count a (1,000 of those rows, read every 50 steps), cap 10K,
      Stage A's fine-tune recipe at lr = the state's fine-tune lr / LR_DIVISOR; scratch = NEVER, same recipe and lr;
  steering: on Stage A's 1,000 qualifying 'C b 40' rows, the gain share of the mean tuned-minus-base residual at `=`.
Labels by revival.py's filed rules; scores (binomial bar at the actual n and a margin over the better nuisance
labeller, on every state and on one vote per training run at its 1K state), agreement per fine-tune lr (described) and
the class-B readings, rewritten with every state.

Order: model-free controls (a miss writes results/c1_control_miss.json, exit 1); the instruments' controls, alone
(--controls-only; pruning on LOW with the sign flipped must lower count a beyond the random draws, else pruning is
void; scratch at each lr must take > 1K steps, else that lr is void); then the states, results/c1.json after each.
The states run refuses (exit 2) until c1.json holds the controls. Exit 3 at the deadline (a reverse fine-tune cut
by it is discarded and rerun); a resubmit resumes. Readings are kept across a changed truth set; a changed recipe drops
them.

Usage (repo root, GPU): python experiments/pcfg_anchors/c1.py [--controls-only]  (cluster/pcfg_anchors.sbatch c1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from amend3 import texts_of
from amend5 import A5_DIR
from amend5 import OUT as OUT5
from amend5b import A5B_DIR, OUT5B
from run import (
    A1,
    EVAL_SEED,
    EXIT_RESUME,
    FT_BATCH,
    LARGE,
    RESULTS,
    WEIGHT_DECAY,
    device,
    eval_seqs,
    load_model,
    log,
    state_path,
    write_json,
)

from commute import pcfg, revival
from commute.interchange import N_B, counter_seqs, qualifying_texts
from commute.pcfg import A, B
from commute.pcfg_train import Batches

PRED_C1 = RESULTS / "predictions_c1.json"
OUT, MISS = RESULTS / "c1.json", RESULTS / "c1_control_miss.json"
LR_DIVISOR = 10  # plan: reverse fine-tuning at the fine-tune lr / 10
N_SCORE, N_RFT_EVAL, CHUNK = 1000, 1000, 100  # rows scoring grad . w; rows read during reverse fine-tuning; batches per chunk
SCORE_TAIL, PRUNE_SEED, REV_SEED = 80, 301, 300  # streams: grad . w rows, random draws, reverse fine-tuning data and dropout
RECIPE = {"prune_k": list(revival.PRUNE_K), "n_random": revival.N_RANDOM, "target": revival.TARGET, "cap": revival.CAP,
          "eval_every": revival.EVAL_EVERY, "lr_divisor": LR_DIVISOR, "n_score": N_SCORE, "n_rft_eval": N_RFT_EVAL,
          "seeds": [SCORE_TAIL, PRUNE_SEED, REV_SEED], "batch": FT_BATCH}  # fmt: skip  (JSON types: compared with c1.json's copy on resume)
INSTRUMENTS = ("prune", "reverse", "steer")


@dataclass(frozen=True)
class State:
    id: str
    src: str  # "a2" or "amend5"
    arm: str
    step: int
    path: Path
    base: str  # prior of the model it was fine-tuned from
    ft_lr: float
    file_truth: str  # "existing", "new" or "B", from the checks' result files


@dataclass(frozen=True)
class Data:
    score: np.ndarray  # 'C a 40' rows for grad . w
    eval: np.ndarray  # Stage A's 4,000 'C a 40' rows (count-a accuracy)
    steer: np.ndarray  # Stage A's 1,000 qualifying 'C b 40' rows


@dataclass(frozen=True)
class Truth:
    scored: dict[str, str]  # state -> "existing" / "new": pruning and reverse fine-tuning
    steer_only: dict[str, str]  # R: steering only
    class_B: list[str]  # reported, not scored

    def steer(self) -> dict[str, str]:
        return self.scored | self.steer_only


def truth_from(labels: dict[str, str], prior: str, amend5_class: str | None) -> str:
    """The checks' reading of a state: HIGH-based (R) by count b, LOW-based by count a, through -> existing, not
    through -> new, else B. An Amendment 5 or 5b state is existing only as class 'reuses', and new as class 'new' too."""
    lab = labels["b"] if prior == "HIGH" else labels["a"]
    if amend5_class is not None:
        if amend5_class == "reuses":
            return "existing"
        return "new" if amend5_class == "new" or lab == "not through" else "B"
    return {"through": "existing", "not through": "new"}.get(lab, "B")


def load_state(sid: str) -> State:
    src, arm, step = sid.split("/")
    if src == "a2":
        f = json.loads((RESULTS / f"a2_{arm}.json").read_text())
        key, prior, lr, root, cls = f["state_key"], f["arm"]["prior"], f["arm"]["lr"], LARGE / "a2", None
    else:
        out, root = {"amend5": (OUT5, A5_DIR), "amend5b": (OUT5B, A5B_DIR)}[src]
        f = json.loads(out.read_text())["arms"][arm]
        key, prior, lr = f["state_key"], "LOW", f["arm"]["lr"]
        cls = f["states"][step]["class"]
    s = f["states"][step]
    labels = {t: s[f"check_count_{t}"]["label"] for t in "ab"}
    return State(sid, src, arm, int(step), state_path(key, step, root), prior, float(lr), truth_from(labels, prior, cls))


def amend5b_ready(sid: str) -> bool:
    """amend5b.json exists and holds the state's arm (Amendment 5b has run it)."""
    return OUT5B.exists() and sid.split("/")[1] in json.loads(OUT5B.read_text()).get("arms", {})


def all_states(k: dict[str, Any]) -> tuple[dict[str, State], list[str]]:
    """Every listed state, in run order: truth_scored, amend5b_states, steering_only, class_B; and the amend5b states
    not yet run (left out)."""
    ids = [*k["truth_scored"], *k["amend5b_states"], *k["steering_only"], *k["class_B"]]
    assert len(ids) == len(set(ids)), "a state is listed twice"
    missing = [s for s in k["amend5b_states"] if not amend5b_ready(s)]
    return {sid: load_state(sid) for sid in ids if sid not in missing}, missing


def truth_sets(k: dict[str, Any], states: dict[str, State]) -> Truth:
    """The listed truth, plus each run amend5b state where its own file puts it: existing or new into the scored
    set, B into class B (lead, 2026-10-30)."""
    b5 = {s: states[s].file_truth for s in k["amend5b_states"] if s in states}
    return Truth(k["truth_scored"] | {s: v for s, v in b5.items() if v != "B"}, dict(k["steering_only"]),
                 [*k["class_B"], *(s for s, v in b5.items() if v == "B")])  # fmt: skip


def build_data() -> Data:
    score = pcfg.task_batch(np.random.default_rng([EVAL_SEED, SCORE_TAIL]), N_SCORE, "C", A)[0]
    steer = counter_seqs(qualifying_texts(np.random.default_rng([EVAL_SEED, 51]), N_B), B)  # = run.a2_eval_data's b_texts
    return Data(score, eval_seqs(A, A1.eval_n), steer)


def rft_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """Reverse fine-tuning rows: natural 'C a 40', answer count(a)."""
    return pcfg.task_batch(rng, n, "C", A)[0]


def rft_batches() -> Batches:
    return Batches(rft_rows, REV_SEED, FT_BATCH, CHUNK, device())


def lr_key(lr: float) -> str:
    return f"{lr:.0e}"


def rows_ok(seq: np.ndarray, op: int) -> bool:
    """'C <op> 40' rows whose answer is the count of op in the window."""
    t = texts_of(seq)
    head = (seq[:, 1] == pcfg.FAM0 + pcfg.FAMILIES.index("C")) & (seq[:, 2] == pcfg.TERM0 + op) & (seq[:, 3] == pcfg.NUM0 + pcfg.WINDOW)
    return bool((head & (seq[:, pcfg.ANS] - pcfg.NUM0 == pcfg.count(t, op))).all())


def nuisance_labels(states: dict[str, State]) -> dict[str, dict[str, str]]:
    """Model-free labellers that read only how a state was trained: 'short' (fine-tuned <= 1K steps -> existing) and
    'slow' (fine-tune lr <= 1e-5 -> existing). Scored beside every instrument: an instrument that only tracks how far
    a state moved could pass through them."""
    return {"short": {s: "existing" if st.step <= 1000 else "new" for s, st in states.items()},
            "slow": {s: "existing" if st.ft_lr <= 1e-5 else "new" for s, st in states.items()}}  # fmt: skip


def model_free_checks(k: dict[str, Any], states: dict[str, State], missing: list[str], data: Data) -> tuple[dict[str, bool], dict[str, Any]]:  # fmt: skip
    """truth_matches_files: the listed labels against the checks' files (amend5b states are placed by their file, so
    only listed ones are compared); amend5b_run: every amend5b state is in amend5b.json (a dry run before Amendment 5b
    shows it false)."""
    want = {**k["truth_scored"], **k["steering_only"], **dict.fromkeys(k["class_B"], "B")}
    wrong = {s: (want[s], st.file_truth) for s, st in states.items() if s in want and want[s] != st.file_truth}
    lrs = {lr_key(st.ft_lr / LR_DIVISOR) for s, st in states.items() if s not in k["steering_only"]}
    scratch = {lr_key(lr) for lr in k["scratch_lrs"]}
    chunk0 = rft_rows(np.random.default_rng([REV_SEED, 0]), FT_BATCH * CHUNK)  # Batches' chunk 0
    checks = {"truth_matches_files": not wrong, "amend5b_run": not missing,
              "state_files_exist": all(st.path.exists() for st in states.values()),
              "steering_only_HIGH": all(states[s].base == "HIGH" for s in k["steering_only"]),
              "scored_and_B_from_LOW": all(st.base == "LOW" for s, st in states.items() if s not in k["steering_only"]),
              "every_reverse_lr_has_scratch": lrs <= scratch,
              "rows_score": rows_ok(data.score, A), "rows_eval": rows_ok(data.eval, A), "rows_rft_chunk0": rows_ok(chunk0, A),
              "rows_steer_qualifying": rows_ok(data.steer, B) and bool((pcfg.count(texts_of(data.steer), B) != pcfg.count(texts_of(data.steer), A) + 1).all()),
              "plan_bar_table": [revival.pass_bar(n, 0) for n in (8, 12, 14, 16)] == [7, 10, 11, 12]}  # fmt: skip
    info = {"truth_mismatches": wrong, "amend5b_not_run": missing, "reverse_lrs": sorted(lrs), "missing_files": [str(st.path) for st in states.values() if not st.path.exists()],
            "n_rows": {"score": len(data.score), "eval": len(data.eval), "rft_eval": N_RFT_EVAL, "steer": len(data.steer)}}  # fmt: skip
    return checks, info


def controls(k: dict[str, Any], data: Data, deadline: float) -> dict[str, Any] | None:
    """Pruning on LOW with the sign flipped (removing the units count a needs) and, described, in Alg. 3's direction;
    scratch (NEVER) at each lr. None when the deadline cut a scratch run."""
    low = load_model(LARGE / "a1" / "pt_LOW.pt")
    out: dict[str, Any] = {"prune_LOW_importance": revival.prune_readings(low, data.score, data.eval, PRUNE_SEED, sign=-1.0),
                           "prune_LOW_revival": revival.prune_readings(low, data.score, data.eval, PRUNE_SEED), "scratch": {}}  # fmt: skip
    out["prune_control_pass"] = revival.prune_control(out["prune_LOW_importance"])
    log(f"pruning control (LOW, sign -1) pass {out['prune_control_pass']}: {summarise_prune(out['prune_LOW_importance'])}")
    for lr in k["scratch_lrs"]:
        torch.manual_seed(REV_SEED)
        r = revival.reverse_finetune(load_model(LARGE / "a1" / "pt_NEVER.pt"), rft_batches(), lr, WEIGHT_DECAY, data.eval[:N_RFT_EVAL], deadline)
        if r["end"] == "deadline":
            return None
        r["void"] = revival.scratch_value(r["steps"]) <= revival.SCRATCH_VOID_MAX
        out["scratch"][lr_key(lr)] = r
        log(f"scratch lr {lr:.0e}: steps {r['steps']} ({r['end']}), void {r['void']}, last read {r['curve'][-1]}")
    return out


def summarise_prune(r: dict[str, Any]) -> str:
    return f"acc0 {r['acc0']:.3f}; " + "; ".join(f"K {k}: top {r[str(k)]['top']:.3f} random {np.mean(r[str(k)]['random']):.3f} +- {np.std(r[str(k)]['random'], ddof=1):.3f}" for k in revival.PRUNE_K)


def read_one(st: State, k: dict[str, Any], data: Data, deadline: float) -> dict[str, Any] | None:
    """The state's readings; None when the deadline cut its reverse fine-tune."""
    tuned, base = load_model(st.path), load_model(LARGE / "a1" / f"pt_{st.base}.pt")
    out: dict[str, Any] = {"steer": revival.steering(base, tuned, data.steer)}
    if st.id not in k["steering_only"]:
        out["prune"] = revival.prune_readings(tuned, data.score, data.eval, PRUNE_SEED)
        torch.manual_seed(REV_SEED)
        out["reverse"] = revival.reverse_finetune(tuned, rft_batches(), st.ft_lr / LR_DIVISOR, WEIGHT_DECAY, data.eval[:N_RFT_EVAL], deadline)
        if out["reverse"]["end"] == "deadline":
            return None
    return out


def labels_of(row: dict[str, Any], ctl: dict[str, Any]) -> dict[str, str]:
    lab = {"steer": revival.steer_label(row["steer"])}
    if "prune" in row:
        lab["prune"] = revival.prune_label(row["prune"]) if ctl["prune_control_pass"] else "void"
        r = row["reverse"]
        lab["reverse"] = revival.reverse_label(r["steps"], r["end"], ctl["scratch"][lr_key(r["lr"])]["steps"])
    return lab


def scores(T: Truth, states: dict[str, State], labels: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Per instrument (and per nuisance labeller): the score on every state (binomial bar at the actual n, and agreement
    >= the better nuisance labeller's + MARGIN_ALL on the same states) and on one vote per training run (its 1K state;
    + MARGIN_SEED); per fine-tune lr stratum (1e-4, 1e-5), described; class B's label counts."""
    truth = {"prune": T.scored, "reverse": T.scored, "steer": T.steer()}
    nuis = nuisance_labels(states)
    per = {i: {s: lab[i] for s, lab in labels.items() if i in lab} for i in INSTRUMENTS}
    per |= {f"nuisance_{n}": lab for n, lab in nuis.items()}
    out: dict[str, Any] = {}
    for name, lab in per.items():
        t = truth.get(name, T.scored)
        t1k = {s: v for s, v in t.items() if states[s].step == 1000}
        strata = {lr_key(lr): {s: v for s, v in t.items() if lr_key(states[s].ft_lr) == lr_key(lr)} for lr in (1e-4, 1e-5)}
        b = [lab.get(s, "missing") for s in T.class_B]
        out[name] = {"all": revival.score(lab, t, nuis, revival.MARGIN_ALL),
                     "per_seed": revival.score(lab, t1k, nuis, revival.MARGIN_SEED),
                     "by_ft_lr": {k: {q: v for q, v in revival.score(lab, st).items() if q in ("n", "agree", "voided")} for k, st in strata.items()},
                     "class_B": {v: b.count(v) for v in sorted(set(b))}}  # fmt: skip
    return out


def stop(res: dict[str, Any]) -> None:
    write_json(MISS, res)
    log(f"STOP: a model-free control missed: {res['model_free']}; see {MISS.name} (exit 1)")
    sys.exit(1)


def run_states(res: dict[str, Any], k: dict[str, Any], T: Truth, states: dict[str, State], data: Data, deadline: float) -> int:
    for sid, st in states.items():
        if sid in res["states"]:
            continue
        row = None if time.monotonic() > deadline else read_one(st, k, data, deadline)
        if row is None:
            log(f"{sid}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        res["states"][sid] = row | {"labels": labels_of(row, res["controls"]), "truth": st.file_truth}
        res["scores"] = scores(T, states, {s: r["labels"] for s, r in res["states"].items()})
        res["complete"] = len(res["states"]) == len(states)
        rev = row.get("reverse", {})
        log(f"{sid} ({st.file_truth}): labels {res['states'][sid]['labels']}; steer S {row['steer']['S']:.3f} at block "
            f"{row['steer']['block']} (G {row['steer']['G_score']:.3f}); reverse steps {rev.get('steps')} ({rev.get('end')})"
            + (f"; prune {summarise_prune(row['prune'])}" if "prune" in row else ""))  # fmt: skip
        write_json(OUT, res)
    res["scores"] = scores(T, states, {s: r["labels"] for s, r in res["states"].items()})  # also after a changed truth set
    res["complete"] = True
    write_json(OUT, res)
    log(f"scores: {json.dumps(res['scores'])}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls-only", action="store_true", help="the model-free and instrument controls only")
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    deadline = time.monotonic() + 60 * args.deadline_min
    if not PRED_C1.exists() or str(json.loads(PRED_C1.read_text()).get("status", "")).startswith("draft"):
        sys.exit(f"file the predictions first (a draft is refused): {PRED_C1}")
    k = json.loads(PRED_C1.read_text())["constants"]
    (states, missing), data = all_states(k), build_data()
    T = truth_sets(k, states)
    checks, info = model_free_checks(k, states, missing, data)
    log(f"model-free controls {checks}; {info}")
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    keep = prev.get("recipe") == RECIPE
    res = {"constants": k, "recipe": RECIPE, "model_free": checks, "model_free_info": info,
           "truth": {"scored": T.scored, "steering_only": T.steer_only, "class_B": T.class_B},
           "controls": prev.get("controls") if keep else None, "states": prev.get("states", {}) if keep else {}}  # fmt: skip
    if not all(checks.values()):
        stop(res)
    if res["controls"] is None and args.controls_only:
        res["controls"] = controls(k, data, deadline)
        if res["controls"] is None:
            log(f"controls: deadline; resubmit to resume (exit {EXIT_RESUME})")
            sys.exit(EXIT_RESUME)
        write_json(OUT, res)
    if res["controls"] is None:
        log("no instrument controls in c1.json: run the controls job first (--controls-only; exit 2)")
        sys.exit(2)
    sys.exit(0 if args.controls_only else run_states(res, k, T, states, data, deadline))


if __name__ == "__main__":
    main()
