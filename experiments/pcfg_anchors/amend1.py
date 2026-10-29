"""E55 Amendment 1 ($0 diagnostic): which answer rule do the all-spurious W fine-tunes follow on natural rows, and when
do they leave count(a) + 1? Why: reports/analyses/2026-10-29_analysis_e55-stage-a.md ("What next" 2, 3) and the review
reports/reviews/2026-10-29_review_e55-stage-a-analysis.md (finding 2). Every rule, threshold and expected value:
results/predictions_amend1.json (filed before the run, read here, never edited).

Per state, 8,000 fixed natural texts asked 'C b 40'; each answer is compared with six rules (a, b, c = window counts):
a + 1 (the spurious rule), b, the midpoint of a + 1 and b rounded towards a + 1, a (LOW's own answer) and the two
roundings of (41 - c) / 2 (count c alone). Each rule is scored on the rows where it differs from the others; the
per-row answers go to results/large/amend1/answers/.

Sections, cheapest first; each writes results/amend1.json when it finishes and is skipped when already there:
  controls    the rules against the grammar; N1 10K on rule b, LOW on 'C a 40' and 'C b 40' on rule a (exit 1 on a miss);
              LOW's read_state (W0 = the do-nothing reference)
  saved       W1, W1s1, W2 at 1K and 10K (results/large/a2/)
  trajectory  W1 and W1s1 re-run with a2's recipe and state keys, states at TRAJ_SAVES in results/large/amend1/, each
              read with run.py read_state (B, P for count a and b) and the rule read. One run per seed sampled at six
              steps: these states are not independent anchors and never enter Stage C's n. Exit 3 at the deadline.

Usage (repo root, GPU): python experiments/pcfg_anchors/amend1.py [--deadline-min 22]  (cluster/pcfg_anchors.sbatch amend1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
from run import (
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
OUT = RESULTS / "amend1.json"
TRAJ = LARGE / "amend1"
W_ARMS, TRAJ_ARMS, SAVED = ("W1", "W1s1", "W2"), ("W1", "W1s1"), (1000, 10_000)
PRIMARY = ("a_plus_1", "b", "mid", "a")
RULES = (*PRIMARY, "c_floor", "c_ceil")
NOT_ANCHORS = "trajectory states: one run per seed sampled at six steps; not independent anchors, never in Stage C's n"


def counts(texts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = (pcfg.count(texts, op).astype(np.int64) for op in (A, B, C))
    assert ((a + b + c) == pcfg.WINDOW).all(), "the window holds only a, b, c"
    return a, b, c


def rule_answers(texts: np.ndarray) -> dict[str, np.ndarray]:
    """Each rule's answer (a count) per text; mid = a + 1 + trunc((b - a - 1) / 2), rounding towards a + 1."""
    a, b, c = counts(texts)
    d = b - a - 1
    mid = a + 1 + np.sign(d) * (np.abs(d) // 2)
    return {"a_plus_1": a + 1, "b": b, "mid": mid, "a": a, "c_floor": (41 - c) // 2, "c_ceil": (42 - c) // 2}


def distinct(rule: str, ans: dict[str, np.ndarray]) -> np.ndarray:
    """Rows where `rule`'s answer differs from every other primary rule's."""
    return np.logical_and.reduce([ans[rule] != ans[r] for r in PRIMARY if r != rule])


def formula_check(rng: np.random.Generator) -> dict[str, bool]:
    """Every rule but a gives the label on spurious rows (count(b) = count(a) + 1); mid's rounding on known cases."""
    s = rule_answers(pcfg.spurious_texts(rng, 2000))
    out = {f"{r}_is_label_on_spurious": bool((s[r] == s["b"]).all()) for r in RULES if r != "a"}
    texts = np.full((4, pcfg.TEXT), C, np.int8)  # windows with a = 13 and b = a + 2, a, a - 1, a + 3
    for i, b in enumerate((15, 13, 12, 16)):
        w = texts[i, -pcfg.WINDOW :]
        w[:13], w[13 : 13 + b] = A, B
    out["mid_rounds_towards_a_plus_1"] = rule_answers(texts)["mid"].tolist() == [14, 14, 13, 15]
    return out


def eval_texts(k: dict[str, Any]) -> np.ndarray:
    return pcfg.sample_texts(np.random.default_rng([EVAL_SEED, 70]), k["n_rows"])


def const_floor(values: np.ndarray) -> float:
    """Share of these rows a constant answer scores at best: the most common value's share (edge case E9)."""
    return float(np.unique(values, return_counts=True)[1].max() / len(values)) if len(values) else float("nan")


def follows(share: float, floor: float, k: dict[str, Any]) -> str:
    """'follows' iff share > floor + hi (1 - floor); 'does not follow' iff share < lo. Strict; nan -> 'neither' (E1, E3)."""
    return "follows" if share > floor + k["hi"] * (1 - floor) else "does not follow" if share < k["lo"] else "neither"


def route(labels: dict[str, str]) -> str:
    f = [r for r in PRIMARY if labels[r] == "follows"]
    return f[0] if len(f) == 1 else "mixed" if not f else "ambiguous: " + "+".join(f)


def rule_read(model: GPT, texts: np.ndarray, tag: str, k: dict[str, Any], op: int = B) -> dict[str, Any]:
    """Overall and distinct-row shares per rule, other, route and the (b - a, answer - (a + 1)) table; the per-row
    answers are saved to results/large/amend1/answers/<tag>.npy."""
    out_tok, _ = run_at_art(model, counter_seqs(texts, op))
    pred = out_tok.astype(np.int64) - pcfg.NUM0
    (TRAJ / "answers").mkdir(parents=True, exist_ok=True)
    np.save(TRAJ / "answers" / f"{tag}.npy", pred.astype(np.int16))
    ans = rule_answers(texts)
    row: dict[str, Any] = {"prefix": "C a 40" if op == A else "C b 40", "n_rows": len(texts)}
    for r in RULES:
        m = distinct(r, ans)
        share = float((pred[m] == ans[r][m]).mean()) if m.sum() >= k["min_rows"] else float("nan")
        floor = const_floor(ans[r][m])
        row[r] = {"overall": float((pred == ans[r]).mean()), "S": share, "n_distinct": int(m.sum()),
                  "const_floor": floor, "label": follows(share, floor, k)}  # fmt: skip
    row["other"] = float(np.logical_and.reduce([pred != ans[r] for r in RULES]).mean())
    row["route"] = route({r: row[r]["label"] for r in PRIMARY})
    row["c_check"] = [r for r in ("c_floor", "c_ceil") if row[r]["label"] == "follows"]
    d, o = np.clip(ans["b"] - ans["a"], -7, 7), np.clip(pred - ans["a_plus_1"], -7, 7)
    keys, n = np.unique(np.stack([d, o]), axis=1, return_counts=True)
    row["table_b_minus_a__answer_minus_a_plus_1"] = {f"{x},{y}": int(c) for (x, y), c in zip(keys.T, n, strict=True)}
    log(f"{tag}: route {row['route']}, c_check {row['c_check']}, other {row['other']:.4f}, "
        + ", ".join(f"{r} overall {row[r]['overall']:.4f} S {row[r]['S']:.4f} (n {row[r]['n_distinct']})" for r in RULES))  # fmt: skip
    return row


def section_controls(models: dict[str, GPT], texts: np.ndarray, data: A2Eval, bars: dict, k: dict[str, Any]) -> dict:  # type: ignore[type-arg]
    TRAJ.mkdir(parents=True, exist_ok=True)
    np.save(TRAJ / "counts.npy", np.stack(counts(texts)).astype(np.int8))  # rows (a, b, c) of the answers files
    fc = formula_check(np.random.default_rng([EVAL_SEED, 71]))
    n1 = rule_read(load_model(state_path(state_key("N1", ARMS["N1"]), 10_000)), texts, "N1_10000", k)
    low_a = rule_read(models["LOW"], texts, "LOW_C_a_40", k, op=A)
    low_b = rule_read(models["LOW"], texts, "LOW_C_b_40", k)
    w0 = read_state(models["LOW"], data, bars)
    lo = k["control_min"]
    checks = fc | {"N1_10K_overall_b": n1["b"]["overall"] >= lo, "N1_10K_S_b": n1["b"]["S"] >= lo,
                   "LOW_C_a_40_overall_a": low_a["a"]["overall"] >= lo, "W0_LOW_C_b_40_overall_a": low_b["a"]["overall"] >= lo}  # fmt: skip
    log(f"controls {checks}; W0 read_state count a {w0['check_count_a']}, count b {w0['check_count_b']}")
    return {"checks": checks, "N1_10000": n1, "LOW_C_a_40": low_a, "W0": {"rule_read": low_b, "read_state": w0}}


def section_saved(texts: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in W_ARMS:
        a2 = json.loads((RESULTS / f"a2_{name}.json").read_text())["states"]
        for it in SAVED:
            row = rule_read(load_model(state_path(state_key(name, ARMS[name]), it)), texts, f"{name}_{it}", k)
            out[f"{name}_{it}"] = row | {"A2_label_count_a": a2[str(it)]["check_count_a"]["label"],
                                         "A2_label_count_b": a2[str(it)]["check_count_b"]["label"]}  # fmt: skip
    return out


def reproduction(name: str, states: dict[str, Any], tol: float) -> dict[str, bool]:
    """|re-run - a2| <= tol for B and P of count a and count b at 1K and 10K; nan -> miss (E8)."""
    a2 = json.loads((RESULTS / f"a2_{name}.json").read_text())["states"]
    get = lambda s, it, t, c: s[str(it)][f"check_count_{t}"][c]
    return {f"{it}_{t}_{c}": bool(abs(get(states, it, t, c) - get(a2, it, t, c)) <= tol)
            for it in SAVED for t in ("a", "b") for c in ("B", "P")}  # fmt: skip


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
    rep = reproduction(name, states, k["reproduce_tol"])
    log(f"traj {name} reproduces a2 at 1K and 10K: {rep}")
    return {"state_key": key, "note": NOT_ANCHORS, "states": states, "reproduces_a2": rep}


def run(deadline: float) -> int:
    k = json.loads(PRED1.read_text())["constants"]
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {"note": NOT_ANCHORS, "constants": k}
    bars = filed_bars(json.loads(PRED.read_text())["a2_bars"], json.loads((RESULTS / "a1c_patch.json").read_text())["a2_bars"])
    assert bars is not None, "predictions.json a2_bars not filed"
    models, data, texts = a1_models(), a2_eval_data(), eval_texts(k)
    if "controls" not in out:
        out["controls"] = section_controls(models, texts, data, bars, k)
        write_json(OUT, out)
    if not all(out["controls"]["checks"].values()):
        log(f"STOP: a control missed (apparatus): {out['controls']['checks']} (exit 1)")
        return 1
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
