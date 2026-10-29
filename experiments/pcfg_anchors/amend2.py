"""E55 Amendment 2 ($0 confirmatory test of a post-hoc rule): do the slow all-spurious fine-tunes W1 and W1s1 at 10K
answer the 'third' rule on fresh natural rows? Why: reports/analyses/2026-10-29_analysis_e55-stage-a.md (Amendment 1,
"What does the slow fine-tune answer below count(a) + 1?"): after the run, the writer found that W1 and W1s1 at 10K
answer third = round((2 (a + 1) + b) / 3) on ~98% of amend1's 8,000 rows. Found by looking, so it is tested here on
rows never looked at. Every rule, threshold and expected value: results/predictions_amend2.json (read, never edited).

Five rules (amend1's a + 1, b, mid, a, plus third) are scored on fresh natural rows (a new seed; no text shared with
amend1's rows, checked), each on the rows where it differs from the other four, with amend1's bar against the best
constant answer (bar_read, E9). third is also read pairwise, on the rows where it differs from each other rule. The
c read: the best share any rule of count c alone can reach (the per-c modal answer, an upper bound) against
the shares of third and mid. Model-free answers go through the same code first, as controls.

Sections, cheapest first: controls (formula, synthetic answerers, rows disjoint, N1 10K on b, LOW = W0 on a), then
the saved W states (results/large/a2/), then amend1's trajectory states (results/large/amend1/; one run per seed
sampled six times, context, not anchors). A control miss writes results/amend2_control_miss.json and exits 1;
otherwise results/amend2.json. Per-row answers: results/large/amend2/answers/<state>.npy, rows in counts.npy.

Usage (repo root, GPU, no training): python experiments/pcfg_anchors/amend2.py  (cluster/pcfg_anchors.sbatch amend2)
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from amend1 import (
    PRED1,
    SAVED,
    TRAJ,
    W_ARMS,
    bar_read,
    counts,
    eval_texts,
    rule_answers,
)
from run import (
    ARMS,
    EVAL_SEED,
    LARGE,
    RESULTS,
    a1_models,
    load_model,
    log,
    state_key,
    state_path,
    write_json,
)

from commute import pcfg
from commute.interchange import counter_seqs, run_at_art
from commute.mingpt import GPT
from commute.pcfg import B

PRED2 = RESULTS / "predictions_amend2.json"
OUT, MISS = RESULTS / "amend2.json", RESULTS / "amend2_control_miss.json"
ANS_DIR = LARGE / "amend2" / "answers"
RULES = ("a_plus_1", "b", "mid", "a", "third")
NOT_ANCHORS = "trajectory states: one run per seed sampled at six steps; context, not independent anchors"


def fresh_texts(k: dict[str, Any]) -> np.ndarray:
    return pcfg.sample_texts(np.random.default_rng([EVAL_SEED, k["rows_seed_tail"]]), k["n_rows"])


def rules(texts: np.ndarray) -> dict[str, np.ndarray]:
    """amend1's rules plus third = nearest integer to (2 (a + 1) + b) / 3 = a + 1 + d / 3. d / 3 has fractional part 0,
    1/3 or 2/3, so no .5 tie exists, and the nearest integer is a + 1 + floor((d + 1) / 3) (integer arithmetic)."""
    ans = rule_answers(texts)
    return ans | {"third": ans["a_plus_1"] + (ans["d"] + 1) // 3}


def distinct(rule: str, ans: dict[str, np.ndarray]) -> np.ndarray:
    """Rows where `rule`'s answer differs from every other rule in RULES."""
    return np.logical_and.reduce([ans[rule] != ans[r] for r in RULES if r != rule])


def c_read(pred: np.ndarray, ans: dict[str, np.ndarray], c: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    """c_best = the share of rows a rule of count c alone can match at best: in each count(c) group, the share of the
    state's most common answer there (the exact per-c modal answer, fitted on these same rows: an upper bound for any
    rule of c alone). For third and mid: 'beats count c alone' iff overall share - c_best > c_margin (strict)."""
    c_best = sum(int(np.unique(pred[c == v], return_counts=True)[1].max()) for v in np.unique(c)) / len(pred)
    out: dict[str, Any] = {"c_best": c_best}
    for r in ("third", "mid"):
        gap = float((pred == ans[r]).mean()) - c_best
        out[r] = {"minus_c_best": gap, "label": "beats count c alone" if gap > k["c_margin"] else "not shown"}
    return out


def score(pred: np.ndarray, ans: dict[str, np.ndarray], k: dict[str, Any]) -> dict[str, Any]:
    """Per rule: overall share and its distinct-row read; third pairwise against each other rule; route (the one rule
    labelled 'follows'; none -> 'mixed'; several -> 'ambiguous: r1+r2'); other (equal to no rule)."""
    row: dict[str, Any] = {r: {"overall": float((pred == ans[r]).mean())} | bar_read(pred, ans[r], distinct(r, ans), k)
                           for r in RULES}  # fmt: skip
    row["third_vs"] = {r: bar_read(pred, ans["third"], ans["third"] != ans[r], k) for r in RULES if r != "third"}
    f = [r for r in RULES if row[r]["label"] == "follows"]
    row["route"] = f[0] if len(f) == 1 else "ambiguous: " + "+".join(f) if f else "mixed"
    row["other"] = float(np.logical_and.reduce([pred != ans[r] for r in RULES]).mean())
    return row


def read(pred: np.ndarray, texts: np.ndarray, tag: str, k: dict[str, Any]) -> dict[str, Any]:
    """`score` and `c_read` on one answer vector, every scored number logged."""
    ans = rules(texts)
    _, _, c = counts(texts)
    row = score(pred, ans, k) | {"c_read": c_read(pred, ans, c, k), "n_rows": len(texts)}
    fmt = lambda x: f"S {x['S']:.4f} n {x['n']} f {x['const_floor']:.4f} hi {x['hi']:.4f} {x['label']}"
    per = "; ".join(f"{r} overall {row[r]['overall']:.4f} {fmt(row[r])}" for r in RULES)
    vs = "; ".join(f"vs {r} {fmt(v)}" for r, v in row["third_vs"].items())
    cr = row["c_read"]
    log(f"{tag}: route {row['route']}; other {row['other']:.4f}; {per}; third pairwise: {vs}; "
        f"c read c_best {cr['c_best']:.4f}, third - c_best {cr['third']['minus_c_best']:.4f} {cr['third']['label']}, "
        f"mid - c_best {cr['mid']['minus_c_best']:.4f} {cr['mid']['label']}")  # fmt: skip
    return row


def model_read(model: GPT, texts: np.ndarray, tag: str, k: dict[str, Any]) -> dict[str, Any]:
    out_tok, _ = run_at_art(model, counter_seqs(texts, B))
    pred = out_tok.astype(np.int64) - pcfg.NUM0
    np.save(ANS_DIR / f"{tag}.npy", pred.astype(np.int16))
    return read(pred, texts, tag, k)


def formula_check(texts: np.ndarray, k: dict[str, Any]) -> dict[str, bool]:
    """third's integer form equals rint of the fraction; third is the label on spurious rows; no text of amend1's."""
    ans, sp = rules(texts), rules(pcfg.spurious_texts(np.random.default_rng([EVAL_SEED, k["rows_seed_tail"], 2]), 2000))
    exact = np.rint((2 * ans["a_plus_1"] + ans["b"]) / 3).astype(np.int64)
    old = {t.tobytes() for t in eval_texts(k["amend1"])}
    shared = sum(t.tobytes() in old for t in texts)
    log(f"fresh rows: {len(texts)}, texts shared with amend1's rows: {shared}")
    return {"third_is_rint": bool((ans["third"] == exact).all()), "third_is_label_on_spurious": bool((sp["third"] == sp["b"]).all()),
            "rows_disjoint_from_amend1": shared == 0}  # fmt: skip


def synthetic_controls(texts: np.ndarray, k: dict[str, Any]) -> dict[str, bool]:
    """Each model-free answerer must read its own route, third and mid beat count c alone on their own answerers, not on c_floor (count c
    alone), a constant or non-numbers."""
    ans, n = rules(texts), len(texts)
    preds = {r: ans[r] for r in RULES} | {"c_floor": ans["c_floor"], "constant_13": np.full(n, 13),
                                         "constant_14": np.full(n, 14), "non_number": np.full(n, -1)}  # fmt: skip
    got = {name: read(p, texts, f"synthetic_{name}", k) for name, p in preds.items()}
    want = {r: r for r in RULES} | {"constant_13": "mixed", "constant_14": "mixed", "non_number": "mixed"}
    beats = {("third", "third"): True, ("mid", "mid"): True} | {(s, r): False for s in ("c_floor", "constant_14", "non_number")
                                                               for r in ("third", "mid")}  # fmt: skip
    return {f"synthetic_{s}_route": got[s]["route"] == w for s, w in want.items()} | {
        f"synthetic_{s}_c_read_{r}": (got[s]["c_read"][r]["label"] != "not shown") == b for (s, r), b in beats.items()}  # fmt: skip


def section_controls(models: dict[str, GPT], texts: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    checks = formula_check(texts, k) | synthetic_controls(texts, k)
    n1 = model_read(load_model(state_path(state_key("N1", ARMS["N1"]), 10_000)), texts, "N1_10000", k)
    w0 = model_read(models["LOW"], texts, "LOW_C_b_40", k)
    lo = k["control_min"]
    checks |= {"N1_10K_overall_b": n1["b"]["overall"] >= lo, "N1_10K_S_b": n1["b"]["S"] >= lo,
               "W0_LOW_overall_a": w0["a"]["overall"] >= lo}  # fmt: skip
    log(f"controls {checks}")
    return {"checks": checks, "N1_10000": n1, "W0_LOW_C_b_40": w0}


def section_states(texts: np.ndarray, k: dict[str, Any]) -> dict[str, Any]:
    saved = {f"{n}_{it}": state_path(state_key(n, ARMS[n]), it) for n in W_ARMS for it in SAVED}
    traj = {f"traj_{n}_{it}": state_path(state_key(n, ARMS[n]), it, TRAJ) for n in ("W1", "W1s1") for it in k["traj_saves"]}
    return {"saved": {t: model_read(load_model(p), texts, t, k) for t, p in saved.items()},
            "trajectory": {"note": NOT_ANCHORS} | {t: model_read(load_model(p), texts, t, k) for t, p in traj.items()}}


def main() -> None:
    if not PRED2.exists():
        sys.exit(f"file the predictions first: {PRED2}")
    k = json.loads(PRED2.read_text())["constants"]
    a1 = json.loads(PRED1.read_text())["constants"]
    k |= {"amend1": a1, "traj_saves": a1["traj_saves"]}
    texts = fresh_texts(k)
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ANS_DIR.parent / "counts.npy", np.stack(counts(texts)).astype(np.int8))  # rows (a, b, c) of the answers
    ctrl = section_controls(a1_models(), texts, k)
    if not all(ctrl["checks"].values()):
        write_json(MISS, ctrl)
        log(f"STOP: a control missed (apparatus): {ctrl['checks']}; see {MISS.name} (exit 1)")
        sys.exit(1)
    write_json(OUT, {"constants": k, "controls": ctrl} | section_states(texts, k))


if __name__ == "__main__":
    main()
