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

Two row sets. fresh: 20,000 natural rows (new texts, but in the (a, b) cells amend1's rows already held, so a
within-cell replication). unseen: rows built by a/c swaps in the window with b - a in +-4..+-8, keeping only (a, b)
cells absent from amend1's rows (the off-distribution read). Every read is also given per (a, b) cell (cell_read).

Sections, cheapest first: model-free controls (formula, synthetic answerers, rows disjoint, unseen rows valid), then
N1 10K on b and LOW = W0 on a on both sets (fresh: stop controls; unseen: the cells where both hold are the cells
the W states are scored on), then the saved W states (results/large/a2/), then amend1's trajectory states
(results/large/amend1/; one run per seed sampled six times, context, not anchors), each on both sets. A control
miss writes results/amend2_control_miss.json and exits 1; otherwise results/amend2.json. Per-row answers:
results/large/amend2/answers/<set>_<state>.npy; rows in <set>_counts.npy (W states on unseen: unseen_valid_counts.npy).

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


def unseen_texts(k: dict[str, Any]) -> np.ndarray:
    """Off-distribution rows: natural texts whose window count(a) is set to count(b) - delta, for each delta in
    unseen_deltas, by a/c swaps only (pcfg._set_window_count, the step spurious_texts uses to build the fine-tuning
    rows): count(b), the b positions and every token outside the window are untouched, so a + b + c = 40 still. Rows
    that cannot be built (target a < 0, or fewer c than a's to add) and rows whose (a, b) cell occurs in amend1's
    8,000 rows are dropped."""
    rng = np.random.default_rng([EVAL_SEED, k["unseen_seed_tail"]])
    a1, b1, _ = counts(eval_texts(k["amend1"]))
    seen = set(zip(a1.tolist(), b1.tolist(), strict=True))
    out = []
    for delta in k["unseen_deltas"]:
        t = pcfg.sample_texts(rng, k["unseen_per_delta"])
        a, b, c = counts(t)
        tgt = b - delta
        new = np.array([(x, y) not in seen for x, y in zip(tgt.tolist(), b.tolist(), strict=True)])
        ok = (tgt >= 0) & (tgt - a <= c) & new
        out.append(pcfg._set_window_count(rng, t[ok], pcfg.A, tgt[ok]))  # pyright: ignore[reportPrivateUsage]
    return np.concatenate(out)


def unseen_check(texts: np.ndarray, k: dict[str, Any]) -> dict[str, bool]:
    """The built rows are valid inputs: the natural length, terminal tokens only, a + b + c = 40 (counts asserts),
    b - a in unseen_deltas, and no (a, b) cell of amend1's rows."""
    a, b, _ = counts(texts)
    a1, b1, _ = counts(eval_texts(k["amend1"]))
    seen = set(zip(a1.tolist(), b1.tolist(), strict=True))
    return {"unseen_shape": texts.shape[1] == pcfg.TEXT, "unseen_terminals_only": bool(np.isin(texts, (0, 1, 2)).all()),
            "unseen_deltas": bool(np.isin(b - a, k["unseen_deltas"]).all()),
            "unseen_cells_not_in_amend1": not seen & set(zip(a.tolist(), b.tolist(), strict=True))}  # fmt: skip


def cell_read(pred: np.ndarray, ans: dict[str, np.ndarray], k: dict[str, Any]) -> dict[str, Any]:
    """Every rule answer is fixed within an (a, b) cell, so cells, not rows, are the units (review finding 2). Per
    cell: n and each rule's share. For third and mid, over the cells where the rule is distinct with >= min_cell_rows
    rows: how many the rule answers on more than half the rows (strict); k = ceil(cell_frac * cells); 'follows in
    cells' iff cells >= min_cells and majority >= k, 'void' under min_cells, else 'not'; and the sd of the pooled
    distinct share over n_boot resamples of those cells (seeded)."""
    key = ans["a"] * 64 + ans["b"]
    masks = {int(v): key == v for v in np.unique(key)}
    out: dict[str, Any] = {"cells": {f"{v // 64},{v % 64}": {"n": int(m.sum())} | {r: float((pred[m] == ans[r][m]).mean())
                                     for r in RULES} for v, m in masks.items()}}  # fmt: skip
    rng = np.random.default_rng([EVAL_SEED, k["unseen_seed_tail"], 1])
    for r in ("third", "mid"):
        dist = distinct(r, ans)
        own = [m for m in masks.values() if dist[m][0] and m.sum() >= k["min_cell_rows"]]
        hits = np.array([(pred[m] == ans[r][m]).sum() for m in own])
        ns = np.array([m.sum() for m in own])
        need = int(np.ceil(k["cell_frac"] * len(own)))
        major = int((2 * hits > ns).sum())
        lab = "void" if len(own) < k["min_cells"] else "follows in cells" if major >= need else "not"
        draws = rng.integers(0, len(own), (k["n_boot"], len(own))) if own else np.zeros((0, 0), np.int64)
        sd = float(np.std(hits[draws].sum(1) / ns[draws].sum(1))) if own else float("nan")
        out[r] = {"cells": len(own), "majority": major, "k": need, "label": lab, "pooled_boot_sd": sd}
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
    row = score(pred, ans, k) | {"c_read": c_read(pred, ans, c, k), "cell_read": cell_read(pred, ans, k), "n_rows": len(texts)}
    fmt = lambda x: f"S {x['S']:.4f} n {x['n']} f {x['const_floor']:.4f} hi {x['hi']:.4f} {x['label']}"
    per = "; ".join(f"{r} overall {row[r]['overall']:.4f} {fmt(row[r])}" for r in RULES)
    vs = "; ".join(f"vs {r} {fmt(v)}" for r, v in row["third_vs"].items())
    cr = row["c_read"]
    log(f"{tag}: route {row['route']}; other {row['other']:.4f}; {per}; third pairwise: {vs}; "
        f"c read c_best {cr['c_best']:.4f}, third - c_best {cr['third']['minus_c_best']:.4f} {cr['third']['label']}, "
        f"mid - c_best {cr['mid']['minus_c_best']:.4f} {cr['mid']['label']}")  # fmt: skip
    cl = row["cell_read"]
    cells = "; ".join(f"({c}) n {v['n']} " + " ".join(f"{r} {v[r]:.3f}" for r in RULES) for c, v in cl["cells"].items())
    log(f"{tag} cells: " + "; ".join(f"{r} majority {cl[r]['majority']} of {cl[r]['cells']} (k {cl[r]['k']}) {cl[r]['label']}, "
                                     f"pooled S cell-bootstrap sd {cl[r]['pooled_boot_sd']:.4f}" for r in ("third", "mid"))
        + f"; per cell: {cells}")  # fmt: skip
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
    """Each model-free answerer must read its own route; third and mid beat count c alone on their own answerers,
    not on c_floor (count c alone), a constant or non-numbers."""
    ans, n = rules(texts), len(texts)
    preds = {r: ans[r] for r in RULES} | {"c_floor": ans["c_floor"], "constant_13": np.full(n, 13),
                                         "constant_14": np.full(n, 14), "non_number": np.full(n, -1)}  # fmt: skip
    got = {name: read(p, texts, f"synthetic_{name}", k) for name, p in preds.items()}
    want = {r: r for r in RULES} | {"constant_13": "mixed", "constant_14": "mixed", "non_number": "mixed"}
    beats = {("third", "third"): True, ("mid", "mid"): True} | {(s, r): False for s in ("c_floor", "constant_14", "non_number")
                                                               for r in ("third", "mid")}  # fmt: skip
    return {f"synthetic_{s}_route": got[s]["route"] == w for s, w in want.items()} | {
        f"synthetic_{s}_c_read_{r}": (got[s]["c_read"][r]["label"] != "not shown") == b for (s, r), b in beats.items()}  # fmt: skip


def cell_controls(texts: np.ndarray, k: dict[str, Any]) -> dict[str, bool]:
    """On the unseen rows: pure third and pure mid each read 'follows' pooled and 'follows in cells' for their own rule
    and not for the other; constant 14 reads neither for either."""
    ans, n = rules(texts), len(texts)
    got = {s: read(p, texts, f"synthetic_unseen_{s}", k) for s, p in (("third", ans["third"]), ("mid", ans["mid"]),
                                                                       ("constant_14", np.full(n, 14)))}  # fmt: skip
    out = {}
    for s in got:
        for r in ("third", "mid"):
            want = s == r
            out[f"unseen_synthetic_{s}_{r}_pooled"] = (got[s][r]["label"] == "follows") == want
            out[f"unseen_synthetic_{s}_{r}_cells"] = (got[s]["cell_read"][r]["label"] == "follows in cells") == want
    return out


def model_free_controls(rows: dict[str, np.ndarray], k: dict[str, Any]) -> dict[str, bool]:
    fresh, unseen = rows["fresh"], rows["unseen"]
    return formula_check(fresh, k) | synthetic_controls(fresh, k) | unseen_check(unseen, k) | cell_controls(unseen, k)


def model_controls(rows: dict[str, np.ndarray], k: dict[str, Any]) -> dict[str, Any]:
    """N1 10K (rule b) and LOW = W0 (rule a) on both row sets. The fresh-row reads are stop controls; on the unseen
    rows they decide, cell by cell, which rows the W states are scored on (valid_rows)."""
    models = {"N1_10000": load_model(state_path(state_key("N1", ARMS["N1"]), 10_000)), "LOW_C_b_40": a1_models()["LOW"]}
    reads = {f"{s}_{t}": model_read(m, rows[s], f"{s}_{t}", k) for s in rows for t, m in models.items()}
    lo = k["control_min"]
    checks = {"fresh_N1_10K_overall_b": reads["fresh_N1_10000"]["b"]["overall"] >= lo,
              "fresh_N1_10K_S_b": reads["fresh_N1_10000"]["b"]["S"] >= lo,
              "fresh_W0_LOW_overall_a": reads["fresh_LOW_C_b_40"]["a"]["overall"] >= lo}  # fmt: skip
    return {"checks": checks, "reads": reads}


def valid_rows(reads: dict[str, Any], texts: np.ndarray, k: dict[str, Any]) -> np.ndarray:
    """Unseen rows in the cells where both controls hold: LOW answers a and N1 10K answers b on >= control_min of the
    cell's rows (the model can count a and count b there)."""
    a, b, _ = counts(texts)
    low, n1 = (reads[f"unseen_{t}"]["cell_read"]["cells"] for t in ("LOW_C_b_40", "N1_10000"))
    ok = {c for c in low if low[c]["a"] >= k["control_min"] and n1[c]["b"] >= k["control_min"]}
    return np.array([f"{x},{y}" in ok for x, y in zip(a.tolist(), b.tolist(), strict=True)])


def section_states(texts: np.ndarray, rows: str, k: dict[str, Any]) -> dict[str, Any]:
    saved = {f"{n}_{it}": state_path(state_key(n, ARMS[n]), it) for n in W_ARMS for it in SAVED}
    traj = {f"traj_{n}_{it}": state_path(state_key(n, ARMS[n]), it, TRAJ) for n in ("W1", "W1s1") for it in k["traj_saves"]}
    return {"saved": {t: model_read(load_model(p), texts, f"{rows}_{t}", k) for t, p in saved.items()},
            "trajectory": {"note": NOT_ANCHORS} | {t: model_read(load_model(p), texts, f"{rows}_{t}", k) for t, p in traj.items()}}  # fmt: skip


def stop(ctrl: dict[str, Any]) -> None:
    write_json(MISS, ctrl)
    log(f"STOP: a control missed (apparatus): {ctrl['checks']}; see {MISS.name} (exit 1)")
    sys.exit(1)


def save_counts(name: str, texts: np.ndarray) -> None:
    np.save(ANS_DIR.parent / f"{name}_counts.npy", np.stack(counts(texts)).astype(np.int8))


def main() -> None:
    if not PRED2.exists():
        sys.exit(f"file the predictions first: {PRED2}")
    k = json.loads(PRED2.read_text())["constants"]
    a1 = json.loads(PRED1.read_text())["constants"]
    k |= {"amend1": a1, "traj_saves": a1["traj_saves"]}
    rows = {"fresh": fresh_texts(k), "unseen": unseen_texts(k)}
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    for s, t in rows.items():  # rows (a, b, c) of answers/<s>_<control>.npy
        save_counts(s, t)
    log(f"rows: fresh {len(rows['fresh'])}, unseen {len(rows['unseen'])}")
    ctrl: dict[str, Any] = {"checks": model_free_controls(rows, k)}
    log(f"model-free controls {ctrl['checks']}")
    if not all(ctrl["checks"].values()):
        stop(ctrl)
    mc = model_controls(rows, k)
    keep = valid_rows(mc["reads"], rows["unseen"], k)
    ctrl = {"checks": ctrl["checks"] | mc["checks"]} | mc["reads"]
    ctrl["unseen_valid"] = {"rows": int(keep.sum()), "of": len(keep)}
    log(f"model controls {mc['checks']}; unseen rows in cells where LOW counts a and N1 counts b: {ctrl['unseen_valid']}")
    if not all(ctrl["checks"].values()):
        stop(ctrl)
    rows["unseen"] = rows["unseen"][keep]
    save_counts("unseen_valid", rows["unseen"])  # rows of answers/unseen_<W state>.npy
    if keep.sum() < k["min_rows"]:  # too few valid unseen rows: the U predictions are void (E18)
        log(f"unseen rows void: {int(keep.sum())} valid rows < {k['min_rows']}")
        del rows["unseen"]
    states = {s: section_states(t, s, k) for s, t in rows.items()}
    write_json(OUT, {"constants": k, "controls": ctrl} | states)


if __name__ == "__main__":
    main()
