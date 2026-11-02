"""E55 Amendment 1 ($0 diagnostic): which answer rule do the all-spurious W fine-tunes follow on natural rows, and when
do they leave count(a) + 1? Why: reports/analyses/2026-10-29_analysis_e55-stage-a.md ("What next" 2, 3) and the review
reports/reviews/2026-10-29_review_e55-stage-a-analysis.md (finding 2); code review
reports/reviews/2026-10-29_review_e55-amend1-code.md. Every rule, threshold and expected value:
results/predictions_amend1.json (second filing, before the run; read here, never edited).

Per state, 8,000 fixed natural texts asked 'C b 40' (a, b, c = window counts, a + b + c = 40, d = b - a - 1). Four rules
are scored on the rows where each differs from the other three: a + 1 (the spurious rule), b, a (LOW's own answer) and
mid, the midpoint of a + 1 and b rounded towards a + 1. Since a + b + c = 40, mid is always floor or ceil of (41 - c) / 2,
so a rule of count c alone differs from it only in rounding. The rounding read tells them apart: on rows with d odd,
mid rounds down where d > 0 and up where d < 0; a fixed rounding (count c alone, or (a + b + 1) / 2 rounded one way)
rounds the same way in both halves. Per-row answers go to results/large/amend1/answers/.

Sections, cheapest first; each writes results/amend1.json when it finishes and is skipped when already there:
  controls    the rules against the grammar; synthetic answerers through the scoring code (model-free); N1 10K on
              rule b, LOW on 'C a 40' and 'C b 40' on rule a; LOW's read_state (W0 = the do-nothing reference). A miss
              writes results/amend1_control_miss.json, not amend1.json, and exits 1: a rerun re-reads the controls.
  saved       W1, W1s1, W2 at 1K and 10K (results/large/a2/)
  trajectory  W1 and W1s1 re-run with a2's recipe and state keys, states at traj_saves in results/large/amend1/, each
              read with run.py read_state (B, P for count a and b) and the rule read; max |dw| against a2's 1K and 10K
              states. One run per seed sampled at six steps: not independent anchors, never in Stage C's n. Exit 3 at
              the deadline (resubmit resumes).

Usage (repo root, GPU): python experiments/pcfg_anchors/amend1.py [--deadline-min 22]  (cluster/pcfg_anchors.sbatch amend1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
import torch
from run import (
    A2_DIR,
    ARMS,
    EVAL_SEED,
    EXIT_RESUME,
    LARGE,
    PRED,
    RESULTS,
    A2Eval,
    a1_models,
    a2_eval_data,
    filed_bars,
    finetune,
    load_model,
    log,
    read_state,
    state_key,
    state_path,
    write_json,
)

from commute import pcfg
from commute.interchange import counter_seqs, run_at_art
from commute.mingpt import GPT
from commute.pcfg import A, B, C

PRED1 = RESULTS / "predictions_amend1.json"
OUT, MISS = RESULTS / "amend1.json", RESULTS / "amend1_control_miss.json"
TRAJ = LARGE / "amend1"
W_ARMS, TRAJ_ARMS, SAVED = ("W1", "W1s1", "W2"), ("W1", "W1s1"), (1000, 10_000)
PRIMARY = ("a_plus_1", "b", "mid", "a")
ROUNDING = {("floor", "ceil"): "towards a+1", ("ceil", "floor"): "towards b", ("floor", "floor"): "fixed floor",
            ("ceil", "ceil"): "fixed ceil"}  # (upper half d > 0, lower half d < 0) -> rounding  # fmt: skip
MID_ROUTE = {"towards a+1": "mid (uses the sign of b - a - 1, i.e. counts a and b)",
             "towards b": "midpoint rounded towards b (uses the sign of b - a - 1, i.e. counts a and b)",
             "fixed floor": "fixed rounding, floor ((a+b+1)/2 or count c alone; cannot separate them)",
             "fixed ceil": "fixed rounding, ceil ((a+b+1)/2 or count c alone; cannot separate them)"}  # fmt: skip
SYNTHETIC = {"mid": MID_ROUTE["towards a+1"], "towards_b": MID_ROUTE["towards b"], "c_floor": MID_ROUTE["fixed floor"],
             "c_ceil": MID_ROUTE["fixed ceil"], "a_plus_1": "a_plus_1", "constant_14": "mixed"}  # fmt: skip
NOT_ANCHORS = "trajectory states: one run per seed sampled at six steps; not independent anchors, never in Stage C's n"


def counts(texts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = (pcfg.count(texts, op).astype(np.int64) for op in (A, B, C))
    assert ((a + b + c) == pcfg.WINDOW).all(), "the window holds only a, b, c"
    return a, b, c


def rule_answers(texts: np.ndarray) -> dict[str, np.ndarray]:
    """Each rule's answer (a count) per text. mid = a + 1 + trunc(d / 2) (towards a + 1); towards_b = a + 1 + d -
    trunc(d / 2); c_floor, c_ceil = (41 - c) / 2 rounded down, up. Only the four PRIMARY rules are scored as rules."""
    a, b, c = counts(texts)
    d = b - a - 1
    half = np.sign(d) * (np.abs(d) // 2)
    return {"a_plus_1": a + 1, "b": b, "mid": a + 1 + half, "a": a, "towards_b": a + 1 + d - half,
            "c_floor": (41 - c) // 2, "c_ceil": (42 - c) // 2, "d": d}  # fmt: skip


def distinct(rule: str, ans: dict[str, np.ndarray]) -> np.ndarray:
    """Rows where `rule`'s answer differs from every other primary rule's."""
    return np.logical_and.reduce([ans[rule] != ans[r] for r in PRIMARY if r != rule])


def formula_check(rng: np.random.Generator) -> dict[str, bool]:
    """Every rule but a gives the label on spurious rows; mid's rounding on known cases; mid is c_floor or c_ceil."""
    s = rule_answers(pcfg.spurious_texts(rng, 2000))
    out = {f"{r}_is_label_on_spurious": bool((s[r] == s["b"]).all()) for r in ("a_plus_1", "mid", "towards_b", "c_floor", "c_ceil")}
    texts = np.full((4, pcfg.TEXT), C, np.int8)  # windows with a = 13 and b = a + 2, a, a - 1, a + 3
    for i, b in enumerate((15, 13, 12, 16)):
        w = texts[i, -pcfg.WINDOW :]
        w[:13], w[13 : 13 + b] = A, B
    out["mid_rounds_towards_a_plus_1"] = rule_answers(texts)["mid"].tolist() == [14, 14, 13, 15]
    n = rule_answers(pcfg.sample_texts(rng, 4000))
    odd = n["d"] % 2 == 1
    out["mid_is_c_floor_or_c_ceil"] = bool(((n["mid"] == n["c_floor"]) | (n["mid"] == n["c_ceil"])).all())
    out["mid_floor_up_ceil_down"] = bool((n["mid"] == np.where(n["d"] > 0, n["c_floor"], n["c_ceil"]))[odd].all())
    return out


def eval_texts(k: dict[str, Any]) -> np.ndarray:
    return pcfg.sample_texts(np.random.default_rng([EVAL_SEED, 70]), k["n_rows"])


def const_floor(values: np.ndarray) -> float:
    """Share of these rows a constant answer scores at best: the most common value's share (edge case E9)."""
    return float(np.unique(values, return_counts=True)[1].max() / len(values)) if len(values) else float("nan")


def bar_read(pred: np.ndarray, target: np.ndarray, rows: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    """Share of `rows` answering `target`, its constant floor f, hi = f + hi (1 - f) and the label: 'follows' iff share >
    hi, 'does not follow' iff share < lo; strict; under min_rows rows -> nan -> 'neither' (E1, E3)."""
    n = int(rows.sum())
    share = float((pred[rows] == target[rows]).mean()) if n >= k["min_rows"] else float("nan")
    f = const_floor(target[rows])
    hi = f + k["hi"] * (1 - f)
    lab = "follows" if share > hi else "does not follow" if share < k["lo"] else "neither"
    return {"S": share, "n": n, "const_floor": f, "hi": hi, "label": lab}


def rounding_read(pred: np.ndarray, ans: dict[str, np.ndarray], k: dict[str, Any]) -> dict[str, Any]:
    """On rows with d odd and both roundings unlike a + 1, b and a, per half (d > 0, d < 0): the c_floor and c_ceil
    reads; the half follows the one labelled 'follows' (they differ on every row). Rounding by ROUNDING, else 'mixed';
    a half under min_rows -> 'void'."""
    clear = np.logical_and.reduce([ans[c] != ans[r] for c in ("c_floor", "c_ceil") for r in ("a_plus_1", "b", "a")])
    out: dict[str, Any] = {}
    for half, side in (("upper", ans["d"] > 0), ("lower", ans["d"] < 0)):
        rows = clear & side & (ans["d"] % 2 == 1)
        out[half] = {r: bar_read(pred, ans[f"c_{r}"], rows, k) for r in ("floor", "ceil")}
        out[half]["follows"] = next((r for r in ("floor", "ceil") if out[half][r]["label"] == "follows"), None)
    void = any(out[h]["floor"]["n"] < k["min_rows"] for h in ("upper", "lower"))
    out["rounding"] = "void" if void else ROUNDING.get((out["upper"]["follows"], out["lower"]["follows"]), "mixed")
    return out


def route(rules: dict[str, Any], rounding: str) -> str:
    """The one primary rule that follows, mid resolved by the rounding read (MID_ROUTE). With none following, a decisive
    rounding read names the route (a midpoint rounded towards b fails mid's bar on mid's own rows); else 'mixed'."""
    f = [r for r in PRIMARY if rules[r]["label"] == "follows"]
    if len(f) > 1:
        return "ambiguous: " + "+".join(f)
    if f and f[0] != "mid":
        return f[0]
    if f:
        return MID_ROUTE.get(rounding, f"mid-like, rounding {rounding}")
    return MID_ROUTE.get(rounding, "mixed")


def score(pred: np.ndarray, texts: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    """Model-free: per primary rule its overall share and its distinct-row read; the rounding read; route; other; the
    (b - a, answer - (a + 1)) table, offsets clipped to [-7, 7], non-number answers in their own column."""
    ans = rule_answers(texts)
    row: dict[str, Any] = {r: {"overall": float((pred == ans[r]).mean())} | bar_read(pred, ans[r], distinct(r, ans), k)
                           for r in PRIMARY}  # fmt: skip
    row["rounding"] = rounding_read(pred, ans, k)
    row["route"] = route(row, row["rounding"]["rounding"])
    row["other"] = float(np.logical_and.reduce([pred != ans[r] for r in PRIMARY]).mean())
    d = np.clip(ans["b"] - ans["a"], -7, 7)
    o = np.where(pred < 0, "non-number", np.clip(pred - ans["a_plus_1"], -7, 7).astype(str))
    keys, n = np.unique(np.stack([d.astype(str), o]), axis=1, return_counts=True)
    row["table_b_minus_a__answer_minus_a_plus_1"] = {f"{x},{y}": int(c) for (x, y), c in zip(keys.T, n, strict=True)}
    return row


def synthetic_controls(texts: np.ndarray, k: dict[str, Any]) -> dict[str, bool]:
    """Each synthetic answerer must read its own route through `score` (SYNTHETIC)."""
    ans = rule_answers(texts)
    preds = {r: ans[r] for r in ("mid", "towards_b", "c_floor", "c_ceil", "a_plus_1")} | {"constant_14": np.full(len(texts), 14)}
    got = {name: score(p, texts, k)["route"] for name, p in preds.items()}
    log(f"synthetic answerers' routes: {got}")
    return {f"synthetic_{name}": got[name] == want for name, want in SYNTHETIC.items()}


def rule_read(model: GPT, texts: np.ndarray, tag: str, k: dict[str, Any], op: int = B) -> dict[str, Any]:
    """`score` on the model's answers to '<op> 40'; answers saved to results/large/amend1/answers/<tag>.npy."""
    out_tok, _ = run_at_art(model, counter_seqs(texts, op))
    pred = out_tok.astype(np.int64) - pcfg.NUM0
    (TRAJ / "answers").mkdir(parents=True, exist_ok=True)
    np.save(TRAJ / "answers" / f"{tag}.npy", pred.astype(np.int16))
    row = score(pred, texts, k) | {"prefix": "C a 40" if op == A else "C b 40", "n_rows": len(texts)}
    rd = row["rounding"]
    halves = "; ".join(f"{h} {r} S {rd[h][r]['S']:.4f} n {rd[h][r]['n']} f {rd[h][r]['const_floor']:.4f} hi {rd[h][r]['hi']:.4f} {rd[h][r]['label']}"
                       for h in ("upper", "lower") for r in ("floor", "ceil"))  # fmt: skip
    rules = ", ".join(f"{r} overall {row[r]['overall']:.4f} S {row[r]['S']:.4f} n {row[r]['n']} f {row[r]['const_floor']:.4f} "
                      f"hi {row[r]['hi']:.4f} {row[r]['label']}" for r in PRIMARY)  # fmt: skip
    log(f"{tag}: route {row['route']}; other {row['other']:.4f}; {rules}; rounding {rd['rounding']}: {halves}")
    return row


def section_controls(models: dict[str, GPT], texts: np.ndarray, data: A2Eval, bars: dict, k: dict[str, Any]) -> dict:  # type: ignore[type-arg]
    TRAJ.mkdir(parents=True, exist_ok=True)
    np.save(TRAJ / "counts.npy", np.stack(counts(texts)).astype(np.int8))  # rows (a, b, c) of the answers files
    checks = formula_check(np.random.default_rng([EVAL_SEED, 71])) | synthetic_controls(texts, k)
    n1 = rule_read(load_model(state_path(state_key("N1", ARMS["N1"]), 10_000)), texts, "N1_10000", k)
    low_a = rule_read(models["LOW"], texts, "LOW_C_a_40", k, op=A)
    low_b = rule_read(models["LOW"], texts, "LOW_C_b_40", k)
    w0 = read_state(models["LOW"], data, bars)
    lo = k["control_min"]
    checks |= {"N1_10K_overall_b": n1["b"]["overall"] >= lo, "N1_10K_S_b": n1["b"]["S"] >= lo,
               "LOW_C_a_40_overall_a": low_a["a"]["overall"] >= lo, "W0_LOW_C_b_40_overall_a": low_b["a"]["overall"] >= lo}  # fmt: skip
    log(f"controls {checks}; W0 read_state count a {w0['check_count_a']}, count b {w0['check_count_b']}")
    return {"checks": checks, "N1_10000": n1, "LOW_C_a_40": low_a, "W0": {"rule_read": low_b, "read_state": w0}}


def section_saved(texts: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in W_ARMS:
        a2 = json.loads((RESULTS / f"a2_{name}.json").read_text())["states"]
        for it in SAVED:
            labels = {f"A2_label_count_{t}": a2[str(it)][f"check_count_{t}"]["label"] for t in ("a", "b")}
            log(f"{name}_{it} A2 labels: {labels}")
            out[f"{name}_{it}"] = rule_read(load_model(state_path(state_key(name, ARMS[name]), it)), texts, f"{name}_{it}", k) | labels
    return out


def weight_gap(key: str, it: int) -> float:
    """max |dw| over every parameter between the re-run's state and a2's at step `it` (finding 3)."""
    new, old = (torch.load(state_path(key, it, root), map_location="cpu", weights_only=False)["model"] for root in (TRAJ, A2_DIR))
    return max(float((new[n].float() - old[n].float()).abs().max()) for n in new)


def reproduction(name: str, key: str, states: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    """|re-run - a2| <= reproduce_tol for B and P of count a and count b at 1K and 10K (nan -> miss, E8), and max |dw|
    against a2's states with its 'same run' verdict (<= same_run_max_dw at both)."""
    a2 = json.loads((RESULTS / f"a2_{name}.json").read_text())["states"]
    get = lambda s, it, t, c: s[str(it)][f"check_count_{t}"][c]
    within = {f"{it}_{t}_{c}": bool(abs(get(states, it, t, c) - get(a2, it, t, c)) <= k["reproduce_tol"])
              for it in SAVED for t in ("a", "b") for c in ("B", "P")}  # fmt: skip
    dw = {str(it): weight_gap(key, it) for it in SAVED}
    return {"B_P_within_tol": within, "max_abs_dw": dw, "same_run": all(v <= k["same_run_max_dw"] for v in dw.values())}


def section_trajectory(name: str, texts: np.ndarray, data: A2Eval, bars: dict, k: dict[str, Any],  # type: ignore[type-arg]
                       deadline: float) -> dict[str, Any] | None:  # fmt: skip
    """None when the deadline stopped the fine-tune (resubmit resumes from <TRAJ>/<key>_train.pt)."""
    key, saves = state_key(name, ARMS[name]), tuple(k["traj_saves"])
    if not finetune(key, ARMS[name], deadline, saves=saves, root=TRAJ):
        return None
    states = {}
    for it in saves:
        model = load_model(state_path(key, it, TRAJ))
        states[str(it)] = read_state(model, data, bars) | {"rule_read": rule_read(model, texts, f"traj_{name}_{it}", k)}
        log(f"traj {name} {it}: count a {states[str(it)]['check_count_a']}; count b {states[str(it)]['check_count_b']}")
    state_path(key, "train", TRAJ).unlink(missing_ok=True)
    rep = reproduction(name, key, states, k)
    log(f"traj {name} against a2 at 1K and 10K: {rep}")
    return {"state_key": key, "note": NOT_ANCHORS, "states": states, "reproduces_a2": rep}


def run(deadline: float) -> int:
    k = json.loads(PRED1.read_text())["constants"]
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {"note": NOT_ANCHORS, "constants": k}
    bars = filed_bars(json.loads(PRED.read_text())["a2_bars"], json.loads((RESULTS / "a1c_patch.json").read_text())["a2_bars"])
    assert bars is not None, "predictions.json a2_bars not filed"
    models, data, texts = a1_models(), a2_eval_data(), eval_texts(k)
    if "controls" not in out:
        ctrl = section_controls(models, texts, data, bars, k)
        if not all(ctrl["checks"].values()):
            write_json(MISS, ctrl)
            log(f"STOP: a control missed (apparatus): {ctrl['checks']}; see {MISS.name} (exit 1)")
            return 1
        out["controls"] = ctrl
        write_json(OUT, out)
    if "saved" not in out:
        out["saved"] = section_saved(texts, k)
        write_json(OUT, out)
    for name in TRAJ_ARMS:
        if name in out.setdefault("trajectory", {}):
            continue
        row = None if time.monotonic() > deadline else section_trajectory(name, texts, data, bars, k, deadline)
        if row is None:
            log(f"trajectory {name}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        out["trajectory"][name] = row
        write_json(OUT, out)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    if not (PRED.exists() and PRED1.exists()):
        sys.exit(f"file the predictions first: {PRED1}")
    sys.exit(run(time.monotonic() + 60 * args.deadline_min))


if __name__ == "__main__":
    main()
