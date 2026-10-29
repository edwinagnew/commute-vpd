"""E55 Amendment 3 ($0 diagnostic, no training): are the models broken by swap-built text, or by rare (a, b) pairs?
Why: reports/reviews/2026-10-29_review_e55-amend2-analysis.md (findings 1-3, 8). On Amendment 2's rows built by a/c
swaps in the window, N1 fails to count b (untouched by the swap) more as the swap grows, and LOW fails 341 of 341 rows
at (17, 13), a pair whose counts are all in range. Every rule, threshold and expected value:
results/predictions_amend3.json (read here, never edited).

Part 1, natural rows by pair. Natural texts (a new seed) are rejection-sampled into the filed (count a, count b)
pairs, up to max_per_pair rows each, at most max_texts texts drawn. Per pair: LOW on 'C a 40' and 'C b 40' (both
answer a), N1 10K (b), and the W states (amend1's four rules plus third, amend2.rules). Natural (17, 13) separates
"swap-built text" from "rare pair"; the new pairs test the third rule outside the cells it was found on.

Part 2, swap size in Stage A's reads. Stage A's B and P checks, B nulls and positive references read natural texts
only (interchange.qualifying_texts, no swap; checked: no 'c c c' trigram, which the grammar cannot make and the swaps
can). The swap-built rows Stage A reads are its C_Te = 1 test rows (run.eval_seqs(B, 4000, 1.0), every row from
pcfg.spurious_texts) and the fine-tuning rows, built the same way. The C_Te = 1 rows are rebuilt with count a before
the swap recorded (the swap step is wrapped: same rng calls, same rows; checked). Per read: share right by |swap|, and
the slope of right on |swap| within each count b (swap_read; on these rows count a = b - 1, so b fixes the cell).

Sections, cheapest first: model-free controls; model controls (LOW and N1 on the reference pairs and on the C_Te = 1
rows with no swap); Part 1; Part 2. A control miss writes results/amend3_control_miss.json and exits 1; otherwise
results/amend3.json. Per-row answers: results/large/amend3/answers/<rows>_<read>.npy, rows' counts beside them.

Usage (repo root, GPU, no training): python experiments/pcfg_anchors/amend3.py  (cluster/pcfg_anchors.sbatch amend3)
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from amend1 import PRED1, eval_texts
from amend2 import PRED2, RULES, rules
from amend2 import fresh_texts as amend2_fresh
from run import (
    A1,
    ARMS,
    EVAL_SEED,
    FT_BATCH,
    FT_SAVES,
    FT_SEED0,
    LARGE,
    RESULTS,
    a1_models,
    eval_seqs,
    load_model,
    log,
    state_key,
    state_path,
    write_json,
)

from commute import pcfg
from commute.interchange import N_B, counter_seqs, qualifying_texts, run_at_art
from commute.mingpt import GPT
from commute.pcfg import A, B, C

PRED3 = RESULTS / "predictions_amend3.json"
OUT, MISS = RESULTS / "amend3.json", RESULTS / "amend3_control_miss.json"
ANS_DIR = LARGE / "amend3" / "answers"
W_STATES = tuple(f"{n}_{it}" for n in ("W1", "W1s1", "W2") for it in FT_SAVES)
Rows = tuple[np.ndarray, np.ndarray]  # pairs: (texts, pair index); C_Te = 1: (texts, count a before the swap)


def pair_rows(k: dict[str, Any]) -> Rows:
    """Natural texts in the filed pairs: chunks of `chunk` texts from default_rng([EVAL_SEED, pairs_seed_tail]) until
    every pair holds max_per_pair rows or max_texts are drawn (deterministic); per pair the first rows in stream order."""
    pairs = [tuple(p) for p in k["pairs"]]
    rng, t0, drawn = np.random.default_rng([EVAL_SEED, k["pairs_seed_tail"]]), time.monotonic(), 0
    keep: dict[tuple[int, ...], list[np.ndarray]] = {p: [] for p in pairs}
    have = dict.fromkeys(pairs, 0)
    while drawn < k["max_texts"] and min(have.values()) < k["max_per_pair"]:
        t = pcfg.sample_texts(rng, k["chunk"])
        drawn += len(t)
        a, b = pcfg.count(t, A), pcfg.count(t, B)
        for p in pairs:
            got = t[(a == p[0]) & (b == p[1])][: k["max_per_pair"] - have[p]]
            keep[p].append(got)
            have[p] += len(got)
    log(f"pair rows: {drawn} texts drawn in {time.monotonic() - t0:.0f} s; rows per pair {have}")
    pid = np.concatenate([np.full(have[p], i) for i, p in enumerate(pairs)])
    return np.concatenate([x for p in pairs for x in keep[p]]), pid


@contextmanager
def swap_recorder() -> Iterator[list[np.ndarray]]:
    """While open, pcfg._set_window_count records count(op) before each call (same rng calls, so the same rows)."""
    orig, pre = pcfg._set_window_count, []  # pyright: ignore[reportPrivateUsage]

    def record(rng: np.random.Generator, texts: np.ndarray, op: int, target: np.ndarray) -> np.ndarray:
        pre.append(pcfg.count(texts, op).astype(np.int64))
        return orig(rng, texts, op, target)

    pcfg._set_window_count = record  # pyright: ignore[reportPrivateUsage]
    try:
        yield pre
    finally:
        pcfg._set_window_count = orig  # pyright: ignore[reportPrivateUsage]


def texts_of(seq: np.ndarray) -> np.ndarray:
    return (seq[:, pcfg.TEXT0 : pcfg.TEXT0 + pcfg.TEXT] - pcfg.TERM0).astype(np.int8)


def swapped_rows(make: Any) -> Rows:
    """Texts of make() (every row through the swap step once, in order) and count a before the swap."""
    with swap_recorder() as pre:
        texts = texts_of(make())
    a_pre = np.concatenate(pre)
    assert len(a_pre) == len(texts), "every row passes the swap step once"
    return texts, a_pre


def swap_size(rows: Rows) -> np.ndarray:
    return np.abs(pcfg.count(rows[0], A) - rows[1])


def swap_hist(swap: np.ndarray) -> dict[str, int]:
    v, n = np.unique(swap, return_counts=True)
    return {str(x): int(y) for x, y in zip(v, n, strict=True)}


def ccc_rows(texts: np.ndarray) -> int:
    """Rows whose window holds the trigram c c c (no grammar leaf is a lone c, so natural text never does)."""
    w = texts[:, -pcfg.WINDOW :] == C
    return int((w[:, :-2] & w[:, 1:-1] & w[:, 2:]).any(1).sum())


def model_free_checks(k: dict[str, Any], pr: Rows, te1: Rows) -> dict[str, bool]:
    """Rows are what the filing says (the te1 label histogram equals Stage A's saved one, a2_N1.json); the swap read
    labels known answerers as filed (synthetic)."""
    (pt, pid), tt = pr, te1[0]
    old = {t.tobytes() for t in eval_texts(json.loads(PRED1.read_text())["constants"])}
    old |= {t.tobytes() for t in amend2_fresh(json.loads(PRED2.read_text())["constants"])}
    ta, tb, n = pcfg.count(tt, A), pcfg.count(tt, B), len(tt)
    noisy = np.where(np.random.default_rng([EVAL_SEED, k["pairs_seed_tail"], 1]).random(n) < 0.1, -1, tb)
    synth = {"b": (tb, "not shown"), "a_pre_plus_1": (te1[1] + 1, "depends"), "constant_13": (np.full(n, 13), "not shown"),
             "b_10pct_wrong": (noisy, "not shown")}  # fmt: skip
    got = {s: swap_read(p == tb, swap_size(te1), tb, k, f"synthetic {s}")["label"] for s, (p, _) in synth.items()}
    return {"pairs_counts_match": bool((np.array(k["pairs"])[pid] == np.stack([pcfg.count(pt, A), pcfg.count(pt, B)], 1)).all()),
            "pairs_natural_no_ccc": ccc_rows(pt) == 0, "pairs_disjoint_from_amend1_amend2": not any(t.tobytes() in old for t in pt),
            "te1_wrapper_changes_no_row": bool((texts_of(eval_seqs(B, A1.eval_n, 1.0)) == tt).all()),
            "te1_label_hist_is_stage_a_N1_count_b_te1_hist": swap_hist(tb) == json.loads((RESULTS / "a2_N1.json").read_text())["states"]["10000"]["count_b_te1"]["hist"],
            "te1_label_is_a_plus_1": bool((tb == ta + 1).all()), "te1_rows_hold_ccc": ccc_rows(tt) > 0,
            "b_check_rows_no_ccc": ccc_rows(qualifying_texts(np.random.default_rng([EVAL_SEED, 51]), N_B)) == 0} | {
            f"synthetic_{s}_{w.replace(' ', '_')}": got[s] == w for s, (_, w) in synth.items()}  # fmt: skip


def answers(model: GPT, texts: np.ndarray, op: int, tag: str) -> np.ndarray:
    out_tok, _ = run_at_art(model, counter_seqs(texts, op))
    pred = out_tok.astype(np.int64) - pcfg.NUM0
    np.save(ANS_DIR / f"{tag}.npy", pred.astype(np.int16))
    return pred


def pair_read(pred: np.ndarray, pr: Rows, k: dict[str, Any], tag: str) -> dict[str, Any]:
    """Per filed pair: n, the share answering each rule, the most common answer and its share (None / nan if empty)."""
    (texts, pid), out = pr, {}
    ans = rules(texts)
    for i, (pa, pb) in enumerate(k["pairs"]):
        m = pid == i
        row: dict[str, Any] = {"n": int(m.sum()), "mode": None, "mode_share": float("nan")} | {r: float("nan") for r in RULES}
        if m.any():
            v, c = np.unique(pred[m], return_counts=True)
            row |= {"mode": int(v[c.argmax()]), "mode_share": float(c.max() / m.sum())} | {r: float((pred[m] == ans[r][m]).mean()) for r in RULES}
        out[f"{pa},{pb}"] = row
    log(f"{tag} by pair: " + "; ".join(f"({p}) n {r['n']} mode {r['mode']} ({r['mode_share']:.3f}) "
                                       + " ".join(f"{x} {r[x]:.3f}" for x in RULES) for p, r in out.items()))  # fmt: skip
    return out


def third_on_new(cells: dict[str, Any], low: dict[str, Any], n1: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    """Over the filed new pairs with >= min_pair_rows rows where LOW ('C a 40') answers a and N1 10K answers b on >=
    control_min of the rows (valid): majority = pairs answered third on more than half the rows (strict). 'void'
    under min_pairs valid pairs; 'holds' iff majority >= ceil(hold_frac * pairs); 'fails' iff majority <=
    floor(fail_frac * pairs); else 'mixed'."""
    new = [f"{a},{b}" for a, b in k["new_pairs"]]
    own = [p for p in new if cells[p]["n"] >= k["min_pair_rows"] and low[p]["a"] >= k["control_min"] and n1[p]["b"] >= k["control_min"]]
    major, m = sum(cells[p]["third"] > 0.5 for p in own), len(own)
    lab = ("void" if m < k["min_pairs"] else "holds" if major >= np.ceil(k["hold_frac"] * m)
           else "fails" if major <= np.floor(k["fail_frac"] * m) else "mixed")  # fmt: skip
    return {"valid_pairs": own, "majority": int(major), "label": lab, "modes": {p: cells[p]["mode"] for p in new}}


def swap_read(hit: np.ndarray, swap: np.ndarray, b: np.ndarray, k: dict[str, Any], tag: str) -> dict[str, Any]:
    """Share right per |swap| (clipped at swap_clip) and per (b, |swap|) cell with >= min_cell_rows rows; the slope of
    right (0/1) on |swap| within count b (both demeaned per b) and its HC0 sd; effect = -slope * swap_span (the drop
    over a swap of swap_span). 'depends' iff effect > max(drop_min, 3 sd_effect); 'rises' iff -effect > the same;
    else 'not shown'; no spread of |swap| within any b -> 'void'."""
    y, s = hit.astype(float), swap.astype(float)
    for v in np.unique(b):
        m = b == v
        y[m], s[m] = y[m] - y[m].mean(), s[m] - s[m].mean()
    sxx, span = float((s**2).sum()), k["swap_span"]
    beta = float((s * y).sum() / sxx) if sxx else float("nan")
    sd = span * float(np.sqrt(((s * (y - beta * s)) ** 2).sum()) / sxx) if sxx else float("nan")
    eff, bar = -beta * span, max(k["drop_min"], 3 * sd)
    lab = "void" if not sxx else "depends" if eff > bar else "rises" if -eff > bar else "not shown"
    clip = np.minimum(swap, k["swap_clip"])
    by = {str(v): float(hit[clip == v].mean()) for v in np.unique(clip)}
    cells = {f"{x},{v}": float(hit[(b == x) & (swap == v)].mean()) for x in np.unique(b) for v in np.unique(swap)
             if ((b == x) & (swap == v)).sum() >= k["min_cell_rows"]}  # fmt: skip
    row = {"overall": float(hit.mean()), "by_swap": by, "by_b_swap": cells, "effect": eff, "sd_effect": sd, "label": lab}
    log(f"{tag}: overall {row['overall']:.4f}; by |swap| {by}; effect {eff:.4f} sd {sd:.4f} {lab}; by (b, |swap|) {cells}")
    return row


def te1_read(pred: np.ndarray, te1: Rows, offset: int, tag: str, k: dict[str, Any]) -> dict[str, Any]:
    """swap_read with right = count a after the swap + offset (1: the label b; 0: LOW's own answer), and the share
    answering count a before the swap + offset, by |swap|."""
    a, swap = pcfg.count(te1[0], A), swap_size(te1)
    row = swap_read(pred == a + offset, swap, pcfg.count(te1[0], B), k, f"te1 {tag}")
    pre, clip = pred == te1[1] + offset, np.minimum(swap, k["swap_clip"])
    row["pre_swap_share_by_swap"] = {str(v): float(pre[clip == v].mean()) for v in np.unique(clip)}
    lower, b = te1[1] > a, pcfg.count(te1[0], B)  # direction: a -> c (lowering a) or c -> a (raising a)
    row["by_b_direction_swap"] = {f"{x},{'lower' if lo else 'raise'},{v}": float((pred == a + offset)[m].mean())
                                  for x in np.unique(b) for lo in (True, False) for v in np.unique(swap[swap > 0])
                                  if (m := (b == x) & (lower == lo) & (swap == v)).sum() >= k["min_cell_rows"]}  # fmt: skip
    log(f"te1 {tag}: share answering count a before the swap + {offset}, by |swap|: {row['pre_swap_share_by_swap']}; "
        f"right by (b, direction, |swap|): {row['by_b_direction_swap']}")
    return row


CONTROL_READS = {"LOW_C_a_40": ("LOW", A, 0), "LOW_C_b_40": ("LOW", B, 0), "N1_10000": ("N1_10000", B, 1)}  # (model, op, offset)


def model_controls(models: dict[str, GPT], pr: Rows, te1: Rows, k: dict[str, Any]) -> dict[str, Any]:
    """CONTROL_READS on the reference pairs pooled and on the C_Te = 1 rows with no swap (natural texts that happen to
    have b = a + 1): share answering a + offset (LOW: a; N1: a + 1 = b), each >= control_min."""
    ref = np.isin(pr[1], [i for i, p in enumerate(k["pairs"]) if p in k["reference_pairs"]])
    zero, pa = swap_size(te1) == 0, pcfg.count(pr[0], A)
    shares = {}
    for tag, (m, op, off) in CONTROL_READS.items():
        right = pcfg.count(pr[0], B) if off else pa
        shares[f"ref_pairs_{tag}"] = float((answers(models[m], pr[0], op, f"ctrl_pairs_{tag}")[ref] == right[ref]).mean())
        shares[f"te1_no_swap_{tag}"] = float((answers(models[m], te1[0], op, f"ctrl_te1_{tag}")[zero] == te1[1][zero] + off).mean())
    log(f"model controls (>= {k['control_min']}): {shares}")
    return {"shares": shares, "checks": {s: v >= k["control_min"] for s, v in shares.items()}}


def part1(models: dict[str, GPT], pr: Rows, k: dict[str, Any]) -> dict[str, Any]:
    reads = {t: (m, op) for t, (m, op, _) in CONTROL_READS.items()} | {s: (s, B) for s in W_STATES}
    out = {t: pair_read(answers(models[m], pr[0], op, f"pairs_{t}"), pr, k, f"pairs {t}") for t, (m, op) in reads.items()}
    out["third_on_new_pairs"] = {s: third_on_new(out[s], out["LOW_C_a_40"], out["N1_10000"], k) for s in W_STATES}
    log(f"third on new pairs: {out['third_on_new_pairs']}")
    return out


def part2(models: dict[str, GPT], te1: Rows, k: dict[str, Any]) -> dict[str, Any]:
    """The three bases and every A2 state on the C_Te = 1 rows; the |swap| histogram of the first fine-tuning chunk of
    the all-spurious seed-0 stream (W1, W2: run.finetune's Batches chunk 0), model-free."""
    train = swapped_rows(lambda: pcfg.task_batch(np.random.default_rng([FT_SEED0, 0]), FT_BATCH * 100, "C", B, c_rate=1.0)[0])
    hist = {"te1": swap_hist(swap_size(te1)), "train_seed0_chunk0": swap_hist(swap_size(train))}
    log(f"|swap| histograms: {hist}")
    reads = {"LOW_C_a_40": ("LOW", A, 0), "LOW_C_b_40": ("LOW", B, 0), "HIGH_C_a_40": ("HIGH", A, 0),
             "HIGH_C_b_40": ("HIGH", B, 1), "NEVER_C_b_40": ("NEVER", B, 1)} | {f"{n}_{it}": (f"{n}_{it}", B, 1) for n in ARMS for it in FT_SAVES}  # fmt: skip
    out = {t: te1_read(answers(models[m], te1[0], op, f"te1_{t}"), te1, off, t, k) for t, (m, op, off) in reads.items()}
    return {"swap_histograms": hist, "reads": out}


def load_models() -> dict[str, GPT]:
    """The three bases, and every A2 state as '<arm>_<step>'."""
    return a1_models() | {f"{n}_{it}": load_model(state_path(state_key(n, ARMS[n]), it)) for n in ARMS for it in FT_SAVES}


def stop(ctrl: dict[str, Any]) -> None:
    write_json(MISS, ctrl)
    log(f"STOP: a control missed (apparatus): {ctrl['checks']}; see {MISS.name} (exit 1)")
    sys.exit(1)


def main() -> None:
    if not PRED3.exists():
        sys.exit(f"file the predictions first: {PRED3}")
    k = json.loads(PRED3.read_text())["constants"]
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    pr, te1 = pair_rows(k), swapped_rows(lambda: eval_seqs(B, A1.eval_n, 1.0))
    np.save(ANS_DIR.parent / "pairs_counts.npy", np.stack([pcfg.count(pr[0], x) for x in (A, B, C)]).astype(np.int8))
    np.save(ANS_DIR.parent / "te1_counts.npy", np.stack([pcfg.count(te1[0], A), pcfg.count(te1[0], B), te1[1]]).astype(np.int8))
    ctrl: dict[str, Any] = {"checks": model_free_checks(k, pr, te1)}
    log(f"model-free controls {ctrl['checks']}")
    if not all(ctrl["checks"].values()):
        stop(ctrl)
    models = load_models()
    mc = model_controls(models, pr, te1, k)
    ctrl = {"checks": ctrl["checks"] | mc["checks"], "shares": mc["shares"]}
    if not all(ctrl["checks"].values()):
        stop(ctrl)
    write_json(OUT, {"constants": k, "controls": ctrl, "pairs": part1(models, pr, k), "swap": part2(models, te1, k)})


if __name__ == "__main__":
    main()
