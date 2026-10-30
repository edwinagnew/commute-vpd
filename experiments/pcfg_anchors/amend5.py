"""E55 Amendment 5: reuse anchors fine-tuned on natural text. Plan: reports/plans/2026-10-30_plan_e55-amend5-natural-anchors.md.
Rules, thresholds and expected values: results/predictions_amend5.json (read, never edited; the run refuses a draft).

Arms, all from LOW with Stage A's recipe (run.finetune: answer-only loss, Adam, weight decay 0.1, batch 96, the
10K-iteration schedule, so 500 to 2K steps sit in its warm-up as Stage A's 1K states did), saved at 500, 1K and 2K in
results/large/amend5/<key>_<step>.pt (the key hashes the recipe) and resumed from <key>_train.pt:
  WN (lr 1e-5, seeds 0-3), WNf (lr 1e-4, seed 0): 'C b 40' answered with count(b), on natural texts rejection-sampled
      for count(b) = count(a) + 1, so LOW's count_a + 1 gives every label.
  ID (lr 1e-5, seeds 0, 1): count(a) on natural texts under constants.id_prompt ('T a 40': the T family,
      which answered a terminal in pretraining and never took an operand), so the fine-tune must route that prompt to
      LOW's existing count of a.
Training data: chunk c from default_rng([run.FT_SEED0 + seed, c, TAIL[task]]), Stage A's stream with a new tail.

Readings per state. WN, WNf: run.read_state (Stage A's B and P on its natural qualifying rows, labelled by the filed
bars); the shortcut check on fresh training-distribution rows (share answered 14 beside the share labelled 14);
Amendment 4's off-label swap-built read (described only); the state class. ID: per count, B = share answering the
count (g = identity) at the ID prompt on Stage A's qualifying texts, P on Stage A's pairs for that count at the ID
prompt, labelled by the filed bars (count a scored, count b described).

Order: model-free controls; W0 controls on LOW (incl. LOW not counting a at the ID prompt) and LOW's readings as the do-nothing reference (a miss writes
results/amend5_control_miss.json and exits 1); then the arms in ARMS5's order, results/amend5.json after each.
--controls-only stops after the W0 controls. Exit 3 at the deadline; a resubmit resumes.

Usage (repo root, GPU): python experiments/pcfg_anchors/amend5.py [--controls-only]  (cluster/pcfg_anchors.sbatch amend5)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from amend3 import texts_of
from amend4 import PRED4, off_label_rows, pair_ok, pairs_of, reading_through
from run import (
    A1,
    EVAL_SEED,
    EXIT_RESUME,
    FT_BATCH,
    FT_ITERS,
    FT_SEED0,
    LARGE,
    PRED,
    RESULTS,
    TRAIN_SEED,
    WEIGHT_DECAY,
    A2Eval,
    a1_row,
    a2_eval_data,
    device,
    eval_seqs,
    filed_bars,
    load_model,
    log,
    read_state,
    state_path,
    write_json,
)

from commute import pcfg
from commute.interchange import (
    MIN_PAIRS,
    N_B,
    N_PAIRS,
    Pairs,
    behaviour_share,
    counter_seqs,
    label,
    make_pairs,
    patch_shares,
    patch_shares_seqs,
    run_at_art,
)
from commute.mingpt import GPT
from commute.pcfg import A, B
from commute.pcfg_train import Batches, Run, Schedule, answer_loss, train

PRED5 = RESULTS / "predictions_amend5.json"
OUT, MISS = RESULTS / "amend5.json", RESULTS / "amend5_control_miss.json"
A5_DIR = LARGE / "amend5"
SAVES, CHUNK = (500, 1000, 2000), 100  # CHUNK: run.finetune's batches per drawn chunk
TAIL = {"WN": 70, "ID": 71}  # training streams
EVAL_TAIL = {"WN": 72, "ID": 73}  # fresh training-distribution rows: default_rng([EVAL_SEED, tail]), A1.eval_n rows
PRETRAIN_CHUNKS = 10  # chunks of LOW's own pretraining stream (run.pretrain's batches) searched for the ID prompt
TAG = {A: "a", B: "b"}
Prompt = tuple[int, int]  # (family index, terminal code)


@dataclass(frozen=True)
class Arm5:
    task: str  # "WN" or "ID"
    lr: float
    seed: int


ARMS5 = {"ID_s0": Arm5("ID", 1e-5, 0), **{f"WN_s{s}": Arm5("WN", 1e-5, s) for s in range(4)},
         "WNf_s0": Arm5("WN", 1e-4, 0), "ID_s1": Arm5("ID", 1e-5, 1)}  # fmt: skip  (run order)


@dataclass(frozen=True)
class Eval5:
    stage_a: A2Eval  # Stage A's qualifying texts and pairs (run.a2_eval_data)
    prompt: Prompt  # the ID prompt
    rows: dict[str, np.ndarray]  # task -> fresh training-distribution rows
    off: tuple[np.ndarray, dict[int, Pairs]]  # Amendment 4's off-label texts and pairs within them


def prompt_seqs(texts: np.ndarray, prompt: Prompt) -> np.ndarray:
    """'<family> <operand> <O'>' sequences; the operand slot holds the terminal even for T (assemble writes NULL there).
    For family C this is interchange.counter_seqs. The answer column is assemble's, which no read uses."""
    seq = pcfg.assemble(np.full(len(texts), prompt[0]), np.full(len(texts), prompt[1]), texts)
    seq[:, 2] = pcfg.TERM0 + prompt[1]
    return seq


def wn_texts(rng: np.random.Generator, n: int) -> np.ndarray:
    """n natural texts with count(b) = count(a) + 1 in the window, in stream order (about one text in five)."""
    out, got = [], 0
    while got < n:
        t = pcfg.sample_texts(rng, 6 * (n - got) + 16)
        t = t[pcfg.count(t, B) == pcfg.count(t, A) + 1][: n - got]
        out.append(t)
        got += len(t)
    return np.concatenate(out)


def task_rows(task: str, rng: np.random.Generator, n: int, prompt: Prompt) -> np.ndarray:
    """WN: 'C b 40' on wn_texts, answer count(b). ID: the ID prompt on natural texts, answer count(a)."""
    if task == "WN":
        return counter_seqs(wn_texts(rng, n), B)
    texts = pcfg.sample_texts(rng, n)
    seq = prompt_seqs(texts, prompt)
    seq[:, pcfg.ANS] = pcfg.NUM0 + pcfg.count(texts, A)
    return seq


def train_chunk(arm: Arm5, c: int, prompt: Prompt) -> np.ndarray:
    return task_rows(arm.task, np.random.default_rng([FT_SEED0 + arm.seed, c, TAIL[arm.task]]), FT_BATCH * CHUNK, prompt)


class TailBatches(Batches):
    """run.finetune's batches, chunk c drawn by train_chunk."""

    def __init__(self, arm: Arm5, prompt: Prompt) -> None:
        super().__init__(lambda _rng, _n: np.empty(0), FT_SEED0 + arm.seed, FT_BATCH, CHUNK, device())
        self.arm, self.prompt = arm, prompt

    def _draw(self, c: int) -> np.ndarray:
        return train_chunk(self.arm, c, self.prompt)


def key5(name: str, arm: Arm5, prompt: Prompt) -> str:
    recipe = asdict(arm) | {"iters": FT_ITERS, "batch": FT_BATCH, "tail": TAIL[arm.task],
                            "prompt": list(prompt) if arm.task == "ID" else None}  # fmt: skip
    return f"{name}-{hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:8]}"


def finetune5(key: str, arm: Arm5, prompt: Prompt, deadline: float) -> bool:
    """run.finetune with this amendment's data and saves; False when the deadline stopped it."""
    model = load_model(LARGE / "a1" / "pt_LOW.pt")
    torch.manual_seed(FT_SEED0 + arm.seed)
    run = Run(Schedule(FT_ITERS, arm.lr), WEIGHT_DECAY, save_every=SAVES[0], log_every=100, loss=answer_loss)
    for it in SAVES:
        if state_path(key, it, A5_DIR).exists():
            continue
        step, _ = train(model, TailBatches(arm, prompt), run, state_path(key, "train", A5_DIR), stop_at=it,
                        deadline=deadline, log=lambda m: log(f"{key} {m}"))  # fmt: skip
        if step < it:
            return False
        tmp = state_path(key, it, A5_DIR).with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "step": step}, tmp)
        tmp.replace(state_path(key, it, A5_DIR))
    return True


def hist(x: np.ndarray) -> dict[str, int]:
    v, c = np.unique(x, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(v, c, strict=True)}


def share_answering(model: GPT, seq: np.ndarray, value: np.ndarray) -> float:
    out, _ = run_at_art(model, seq)
    return float((out == pcfg.NUM0 + value).mean())


def prompt_check(model: GPT, data: A2Eval, bars: dict[str, Any], prompt: Prompt) -> dict[str, Any]:
    """ID's checks at its prompt, g = identity, per count: B on Stage A's qualifying texts, P on Stage A's pairs for that
    count at the filed l*, labelled by the filed bars."""
    out: dict[str, Any] = {}
    for op, t in TAG.items():
        pairs = data.pairs[op]
        assert pairs is not None, "model-free control: Stage A's pairs are built"
        p_all = patch_shares_seqs(model, prompt_seqs(pairs.x, prompt), prompt_seqs(pairs.donor, prompt))
        b_ = share_answering(model, prompt_seqs(data.b_texts, prompt), pcfg.count(data.b_texts, op))
        p_ = float(p_all[bars[f"l_star_{t}"] - 1])
        out[f"check_count_{t}"] = {"B": b_, "P": p_, "P_by_block": p_all.tolist(), "n_pairs": len(pairs.x),
                                   "label": label(b_, p_, bars[f"B_{t}"], bars[f"P_{t}"])}  # fmt: skip
    return out


def train_dist(model: GPT, seq: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    """On fresh training-distribution rows: accuracy, share answered shortcut_label beside the share labelled it."""
    out, _ = run_at_art(model, seq)
    ans, lab, s = out - pcfg.NUM0, seq[:, pcfg.ANS].astype(np.int64) - pcfg.NUM0, k["shortcut_label"]
    return {"acc": float((ans == lab).mean()), "answered_shortcut": float((ans == s).mean()),
            "labelled_shortcut": float((lab == s).mean()), "answer_hist": hist(ans), "n": len(seq)}  # fmt: skip


def off_label_read(model: GPT, off: tuple[np.ndarray, dict[int, Pairs]], bars: dict[str, Any]) -> dict[str, Any]:
    """Amendment 4's Part 2 read, described only: per count B ('C b 40') on the first N_B off-label rows and P at l*
    on pairs within them, Stage A's label; amend4.reading_through."""
    r: dict[str, Any] = {}
    for op, t in TAG.items():
        b_ = behaviour_share(model, off[0][:N_B], op)
        p_ = float(patch_shares(model, off[1][op], B)[bars[f"l_star_{t}"] - 1])
        r[t] = {"B": b_, "P": p_, "label": label(b_, p_, bars[f"B_{t}"], bars[f"P_{t}"])}
    return r | {"reading": reading_through(r["a"]["label"], r["b"]["label"])}


def state_class(a: str, b: str, answered_shortcut: float, k: dict[str, Any]) -> str:
    """'reuses' iff count a 'through' and count b not 'through'; 'new' the reverse; 'both' iff both; otherwise
    'shortcut' iff answered_shortcut >= shortcut_min (inclusive), else 'neither'."""
    if a == b == "through":
        return "both"
    if a == "through" or b == "through":
        return "reuses" if a == "through" else "new"
    return "shortcut" if answered_shortcut >= k["shortcut_min"] else "neither"


def read_wn(model: GPT, ev: Eval5, bars: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(read_state(model, ev.stage_a, bars))
    out["train_dist"] = train_dist(model, ev.rows["WN"], k)
    out["off_label"] = off_label_read(model, ev.off, bars)
    a, b = out["check_count_a"]["label"], out["check_count_b"]["label"]
    out["class"] = state_class(a, b, out["train_dist"]["answered_shortcut"], k)
    out["class_strict_reuses"] = a == "through" and b == "not through"  # the plan's wording, described
    return out


def read_id(model: GPT, ev: Eval5, bars: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    out = prompt_check(model, ev.stage_a, bars, ev.prompt) | {"train_dist": train_dist(model, ev.rows["ID"], k)}
    out["class"] = state_class(out["check_count_a"]["label"], out["check_count_b"]["label"], 0.0, k)
    return out


def build(k: dict[str, Any], k4: dict[str, Any]) -> Eval5:
    """Every row and pair set the run reads (model-free, deterministic)."""
    fam, op = k["id_prompt"]
    prompt = (pcfg.FAMILIES.index(fam), pcfg.TERMINALS.index(op))
    rows = {t: task_rows(t, np.random.default_rng([EVAL_SEED, EVAL_TAIL[t]]), A1.eval_n, prompt) for t in TAIL}
    off = off_label_rows(k4)
    off_pairs = {o: pairs_of(np.random.default_rng([EVAL_SEED, k4["off_seed_tail"], o]), off, off[0], o, k4["n_pairs"])[0] for o in TAG}  # amend4.build's
    return Eval5(a2_eval_data(), prompt, rows, (off[0], off_pairs))


def rows_ok(seq: np.ndarray, task: str, prompt: Prompt) -> bool:
    """The prompt slots and the label rule: WN 'C b 40' with count b = count a + 1 = answer; ID its prompt, answer
    count a."""
    t = texts_of(seq)
    a, b, ans = pcfg.count(t, A), pcfg.count(t, B), seq[:, pcfg.ANS] - pcfg.NUM0
    fam, op = (pcfg.FAMILIES.index("C"), B) if task == "WN" else prompt
    head = (seq[:, 1] == pcfg.FAM0 + fam) & (seq[:, 2] == pcfg.TERM0 + op) & (seq[:, 3] == pcfg.NUM0 + pcfg.WINDOW)
    rule = (b == a + 1) & (ans == b) if task == "WN" else ans == a
    return bool((head & rule).all())


def prompt_in_pretraining(prompt: Prompt) -> dict[str, Any]:
    """Described: is the ID prompt absent from LOW's pretraining by construction (operand prior 0, or family T whose
    operand slot is NULL), and from the first PRETRAIN_CHUNKS chunks of run.pretrain's own stream for LOW?"""
    fam, op = prompt
    structural = fam == pcfg.FAMILIES.index("T") or pcfg.PRIORS["LOW"][op] == 0
    hits, n = 0, 0
    for c in range(PRETRAIN_CHUNKS):
        seq = pcfg.pretrain_batch(np.random.default_rng([TRAIN_SEED, c]), A1.batch * A1.chunk, pcfg.PRIORS["LOW"])
        hits += int(((seq[:, 1] == pcfg.FAM0 + fam) & (seq[:, 2] == pcfg.TERM0 + op)).sum())
        n += len(seq)
    return {"structural": structural, "pretrain_rows_searched": n, "hits": hits}


def model_free_checks(ev: Eval5) -> tuple[dict[str, bool], dict[str, Any]]:
    chunks = {name: train_chunk(arm, 0, ev.prompt) for name, arm in ARMS5.items()}
    checks = {f"{name}_chunk0_rows": rows_ok(seq, ARMS5[name].task, ev.prompt) for name, seq in chunks.items()}
    checks |= {f"eval_rows_{t}": rows_ok(seq, t, ev.prompt) for t, seq in ev.rows.items()}
    d = ev.stage_a
    checks["stage_a_b_texts_qualifying"] = len(d.b_texts) == N_B and bool((pcfg.count(d.b_texts, B) != pcfg.count(d.b_texts, A) + 1).all())
    for op, t in TAG.items():
        checks[f"stage_a_pairs_{t}_full"] = pair_ok(d.pairs[op], op, N_PAIRS)
        checks[f"off_pairs_{t}"] = pair_ok(ev.off[1][op], op, MIN_PAIRS)
    info = {"prompt": prompt_in_pretraining(ev.prompt), "id_prompt_tokens": [int(pcfg.FAM0 + ev.prompt[0]), int(pcfg.TERM0 + ev.prompt[1])],
            "label_hist_chunk0": {n: hist(s[:, pcfg.ANS] - pcfg.NUM0) for n, s in chunks.items()},
            "label_hist_eval": {t: hist(s[:, pcfg.ANS] - pcfg.NUM0) for t, s in ev.rows.items()},
            "n_rows": {"chunk0": FT_BATCH * CHUNK, "eval": A1.eval_n, "b_texts": len(d.b_texts), "off": len(ev.off[0])},
            "n_pairs": {f"{w}_{t}": 0 if p is None else len(p.x) for w, ps in (("stage_a", d.pairs), ("off", ev.off[1])) for op, t in TAG.items() for p in [ps[op]]}}  # fmt: skip
    return checks, info


def w0_controls(low: GPT, ev: Eval5, bars: dict[str, Any], k: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    """LOW reproduces its Stage A count-a references (B_a's p_plus, a1_LOW.json count_a, on the same 4,000 rows; P_a's
    positive reference, sample set 0's pairs at l*_a, a1c_patch.json) within reproduce_tol, through this file's
    prompt reads; LOW's answer on the WN training rows ('C b 40', chunk 0 of each seed) + 1 is the label on >=
    coverage_min; LOW's B_a with g = identity at the ID prompt is <= id_w0_max, so ID must route the prompt to the count.
    LOW's readings as every arm reads them are the do-nothing reference; LOW at 'C c 40' is read once, described."""
    ca = (pcfg.FAMILIES.index("C"), A)
    seq_a = eval_seqs(A, A1.eval_n)
    b_a = share_answering(low, prompt_seqs(texts_of(seq_a), ca), pcfg.count(texts_of(seq_a), A))
    pairs0 = make_pairs(np.random.default_rng([EVAL_SEED, 30, 0, A]), A)
    assert pairs0 is not None
    p_a = float(patch_shares_seqs(low, prompt_seqs(pairs0.x, ca), prompt_seqs(pairs0.donor, ca))[bars["l_star_a"] - 1])
    ref_b = a1_row("LOW")["count_a"]
    ref_p = json.loads((RESULTS / "a1c_patch.json").read_text())["patch"]["P_a_positive"][0][bars["l_star_a"] - 1]
    wn = np.concatenate([train_chunk(arm, 0, ev.prompt) for n, arm in ARMS5.items() if arm.task == "WN" and n != "WNf_s0"])
    out, _ = run_at_art(low, wn)
    cover = float((out + 1 == wn[:, pcfg.ANS]).mean())
    ref = read_wn(low, ev, bars, k) | {"id_prompt": read_id(low, ev, bars, k)}
    ref["C_c_40"] = prompt_check(low, ev.stage_a, bars, (pcfg.FAMILIES.index("C"), pcfg.C))
    checks = {"LOW_B_a_reproduces": abs(b_a - ref_b) <= k["reproduce_tol"],
              "LOW_P_a_reproduces": abs(p_a - ref_p) <= k["reproduce_tol"], "LOW_coverage": cover >= k["coverage_min"],
              "id_prompt_not_count_a_at_W0": ref["id_prompt"]["check_count_a"]["B"] <= k["id_w0_max"]}  # fmt: skip
    a4 = json.loads((RESULTS / "amend4.json").read_text())["part2_off_label"]["LOW"]
    ref["off_label_reproduces_amend4"] = all(abs(ref["off_label"][t][q] - a4[t][q]) <= k["reproduce_tol"] for t in "ab" for q in "BP")
    info = {"B_a": b_a, "B_a_stage_a": ref_b, "P_a": p_a, "P_a_stage_a": ref_p, "coverage": cover, "coverage_n": len(wn)}
    log(f"W0 controls {checks} {info}")
    return checks, {"controls": info, "reference": ref}


def summary(name: str, it: int, r: dict[str, Any]) -> None:
    ca, cb = r["check_count_a"], r["check_count_b"]
    log(f"{name} {it}: class {r['class']}; a B {ca['B']:.3f} P {ca['P']:.3f} {ca['label']}; b B {cb['B']:.3f} "
        f"P {cb['P']:.3f} {cb['label']}; train acc {r['train_dist']['acc']:.3f} answered "
        f"{r['train_dist']['answered_shortcut']:.3f} labelled {r['train_dist']['labelled_shortcut']:.3f}")  # fmt: skip


def stop(res: dict[str, Any]) -> None:
    write_json(MISS, res)
    log(f"STOP: a control missed (apparatus): {res['controls']}; see {MISS.name} (exit 1)")
    sys.exit(1)


def run_arms(res: dict[str, Any], ev: Eval5, bars: dict[str, Any], k: dict[str, Any], deadline: float) -> int:
    for name, arm in ARMS5.items():
        if name in res["arms"]:
            log(f"{name}: in {OUT.name}; skipped")
            continue
        key = key5(name, arm, ev.prompt)
        if time.monotonic() > deadline or not finetune5(key, arm, ev.prompt, deadline):
            log(f"{name}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        read = read_wn if arm.task == "WN" else read_id
        states = {str(it): read(load_model(state_path(key, it, A5_DIR)), ev, bars, k) for it in SAVES}
        for it, r in states.items():
            summary(name, int(it), r)
        res["arms"][name] = {"arm": asdict(arm), "state_key": key, "saves": list(SAVES), "states": states}
        write_json(OUT, res)
        state_path(key, "train", A5_DIR).unlink(missing_ok=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls-only", action="store_true", help="stop after the model-free and W0 controls")
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    deadline = time.monotonic() + 60 * args.deadline_min
    if not PRED5.exists() or str(json.loads(PRED5.read_text()).get("status", "")).startswith("draft"):
        sys.exit(f"file the predictions first (a draft is refused): {PRED5}")
    k, k4 = json.loads(PRED5.read_text())["constants"], json.loads(PRED4.read_text())["constants"]
    bars = filed_bars(json.loads(PRED.read_text())["a2_bars"], json.loads((RESULTS / "a1c_patch.json").read_text())["a2_bars"])
    assert bars is not None, "Stage A's bars are filed"
    A5_DIR.mkdir(parents=True, exist_ok=True)
    ev = build(k, k4)
    checks, info = model_free_checks(ev)
    log(f"model-free controls {checks}; {info['prompt']}; n_pairs {info['n_pairs']}")
    if not all(checks.values()):
        stop({"controls": checks, "model_free": info})
    w0_checks, w0 = w0_controls(load_model(LARGE / "a1" / "pt_LOW.pt"), ev, bars, k)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    res = {"constants": k, "controls": checks | w0_checks, "model_free": info, "W0": w0,
           "arms": prev.get("arms", {}) if prev.get("constants") == k else {}}  # fmt: skip
    if not all(w0_checks.values()):
        stop(res)
    write_json(OUT, res)
    sys.exit(0 if args.controls_only else run_arms(res, ev, bars, k, deadline))


if __name__ == "__main__":
    main()
