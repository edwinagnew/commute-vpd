"""E55 Amendment 4 ($0, no training): on swap-built rows, do the early "reuse the old count of a" states (W1 and W1s1
at 1K, M3 at 1K) answer through their count of a or their count of b? Why: Amendment 3, "What next" item 2
(reports/analyses/2026-10-29_analysis_e55-stage-a.md). On swap-built rows count b = count a + 1, so answers cannot
tell the two apart. Rules, thresholds and expected values: results/predictions_amend4.json (read, never edited; the
run refuses a draft).

Part 1 (the lead's design). Recipients: Stage A's C_Te = 1 test rows (amend3.swapped_rows) with |swap| >= 1. Donors:
natural qualifying texts (count b != count a + 1), a new seed. Count-a patch: a donor with the recipient's count b and
another count a; count-b patch: the recipient's count a and another count b. Stage A's patch (whole residual at `=`
after block l, 'C b 40'): P = share of pairs whose patched output equals the donor's and differs from the
recipient's. At the scored block also the shares changed, = donor's count a + 1 and = donor's count b, and P by |swap|.
The same patch on Stage A's natural pairs (run.a2_eval_data's seeds) is each state's reference and a stop control: it
must reproduce a2_<arm>.json's P.

Part 2 (proposed by the code writer, the lead decides whether it is scored). Off-label swap-built rows: natural texts
whose count a is moved by 1 or 2 by the same swap step (pcfg._set_window_count), kept where count a + 1 != count b and
count a != count b. B (share answering count b; share answering count a + 1) and P per count on pairs within this pool
(interchange.pair_index), so recipient and donor are both swap-built and differ in one count, as in Stage A.

Sections, cheapest first: model-free controls (rows, pairs); model controls (LOW's count-a patch and every state's
natural P, as Stage A); Part 1; Part 2. A control miss writes results/amend4_control_miss.json and exits 1; otherwise
results/amend4.json. Per-pair outputs at the scored block: results/large/amend4/outputs/<tag>.npy.

Usage (repo root, GPU, no training): python experiments/pcfg_anchors/amend4.py  (cluster/pcfg_anchors.sbatch amend4)
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from amend3 import ccc_rows, swap_size, swapped_rows, texts_of
from run import (
    A1,
    ARMS,
    EVAL_SEED,
    LARGE,
    PRED,
    RESULTS,
    a1_models,
    eval_seqs,
    filed_bars,
    load_model,
    log,
    state_key,
    state_path,
    write_json,
)

from commute import pcfg
from commute.interchange import (
    MIN_PAIRS,
    N_B,
    Pairs,
    behaviour_share,
    counter_seqs,
    label,
    make_pairs,
    pair_index,
    qualifying_texts,
    run_at_art,
)
from commute.mingpt import GPT
from commute.pcfg import A, B, C

PRED4 = RESULTS / "predictions_amend4.json"
OUT, MISS = RESULTS / "amend4.json", RESULTS / "amend4_control_miss.json"
OUT_DIR = LARGE / "amend4" / "outputs"
SCORED = ("W1_1000", "W1s1_1000", "M3_1000")
STATES = (*SCORED, "N1_1000", "W1_10000")  # N1 1K: positive control for b; W1 10K: described
TAG = {A: "a", B: "b"}
Rows = tuple[np.ndarray, np.ndarray]  # (texts, |swap|)
PairSet = tuple[Pairs, np.ndarray]  # (pairs, |swap| of each recipient)


def recipients(te1: Rows) -> Rows:
    """Stage A's C_Te = 1 rows (as amend3 rebuilt them: texts, count a before the swap) with |swap| >= 1."""
    s = swap_size(te1)
    return te1[0][s >= 1], s[s >= 1]


def off_label_rows(k: dict[str, Any]) -> Rows:
    """k['off_pool'] natural texts from default_rng([EVAL_SEED, off_seed_tail]) with count a moved by a step drawn
    from k['off_steps'] (a -> c or c -> a in the window), kept where the target is reachable and count a + 1 != b,
    count a != b; in stream order."""
    rng, out, got = np.random.default_rng([EVAL_SEED, k["off_seed_tail"]]), [], 0
    while got < k["off_pool"]:
        t = pcfg.sample_texts(rng, 2 * k["off_pool"])
        a, b, c = (pcfg.count(t, x) for x in (A, B, C))
        tgt = a + rng.choice(k["off_steps"], len(t))
        keep = (tgt >= 0) & (c >= tgt - a) & (tgt + 1 != b) & (tgt != b)
        t, tgt = t[keep][: k["off_pool"] - got], tgt[keep][: k["off_pool"] - got]
        out.append((pcfg._set_window_count(rng, t.copy(), A, tgt), np.abs(tgt - pcfg.count(t, A))))  # pyright: ignore[reportPrivateUsage]
        got += len(t)
    return np.concatenate([x for x, _ in out]), np.concatenate([s for _, s in out])


def pairs_of(rng: np.random.Generator, rows: Rows, pool: np.ndarray, f_op: int, n: int) -> PairSet:
    xi, di = pair_index(rng, rows[0], pool, f_op, n)
    return Pairs(rows[0][xi], pool[di]), rows[1][xi]


def build(k: dict[str, Any]) -> dict[str, Any]:
    """Every row and pair set the run reads (model-free, deterministic)."""
    te1 = swapped_rows(lambda: eval_seqs(B, A1.eval_n, 1.0))
    rec, off = recipients(te1), off_label_rows(k)
    donors = qualifying_texts(np.random.default_rng([EVAL_SEED, k["donor_seed_tail"]]), k["donor_pool"])
    n = k["n_pairs"]
    return {"te1": te1[0], "rec": rec, "donors": donors, "off": off,
            "swap": {op: pairs_of(np.random.default_rng([EVAL_SEED, k["donor_seed_tail"], op]), rec, donors, op, n) for op in (A, B)},
            "natural": {op: make_pairs(np.random.default_rng([EVAL_SEED, 50, op]), op) for op in (A, B)},  # a2_eval_data's
            "off_pairs": {op: pairs_of(np.random.default_rng([EVAL_SEED, k["off_seed_tail"], op]), off, off[0], op, n) for op in (A, B)}}  # fmt: skip


def pair_ok(p: Pairs | None, f_op: int, n_min: int) -> bool:
    """>= n_min pairs, each with the same count of the other operand and a different count of f_op."""
    if p is None or len(p.x) < n_min:
        return False
    o = B if f_op == A else A
    return bool((pcfg.count(p.x, o) == pcfg.count(p.donor, o)).all() and (pcfg.count(p.x, f_op) != pcfg.count(p.donor, f_op)).all())


def model_free_checks(k: dict[str, Any], d: dict[str, Any]) -> dict[str, bool]:
    rec, off, dn = d["rec"], d["off"][0], d["donors"]
    hist = json.loads((RESULTS / "amend3.json").read_text())["swap"]["swap_histograms"]["te1"]
    oa, ob = pcfg.count(off, A), pcfg.count(off, B)
    out = {"recipients_n_is_amend3_swapped_rows": len(rec[0]) == sum(v for s, v in hist.items() if s != "0"),
           "recipients_label_is_a_plus_1": bool((pcfg.count(rec[0], B) == pcfg.count(rec[0], A) + 1).all()),
           "te1_wrapper_changes_no_row": bool((texts_of(eval_seqs(B, A1.eval_n, 1.0)) == d["te1"]).all()),
           "donors_natural_no_ccc": ccc_rows(dn) == 0,
           "donors_qualifying": bool((pcfg.count(dn, B) != pcfg.count(dn, A) + 1).all()),
           "off_label_off_both_answers": bool(((oa + 1 != ob) & (oa != ob)).all()),
           "off_swap_in_steps": bool(np.isin(d["off"][1], np.abs(k["off_steps"])).all()),
           "off_n": len(off) == k["off_pool"]}  # fmt: skip
    for op in (A, B):
        out[f"swap_pairs_{TAG[op]}"] = pair_ok(d["swap"][op][0], op, MIN_PAIRS)
        out[f"natural_pairs_{TAG[op]}_full"] = pair_ok(d["natural"][op], op, k["n_pairs"])
        out[f"off_pairs_{TAG[op]}"] = pair_ok(d["off_pairs"][op][0], op, MIN_PAIRS)
    return out


def patch_read(model: GPT, pair_set: PairSet | Pairs, f_op: int, blk: int, prefix: int, k: dict[str, Any], tag: str) -> dict[str, Any]:  # fmt: skip
    """Stage A's P at every block (interchange.patch_shares' formula); at block blk the shares changed, = donor's
    count a + 1, = donor's count b, and (with recipients' |swap|) P by |swap| clipped at swap_clip."""
    p, swap = pair_set if isinstance(pair_set, tuple) else (pair_set, None)
    x, dseq = counter_seqs(p.x, prefix), counter_seqs(p.donor, prefix)
    out_x, _ = run_at_art(model, x)
    out_d, res_d = run_at_art(model, dseq)
    by_block, at = [], np.empty(0)
    for b_ in range(1, len(model.blocks) + 1):
        out_p, _ = run_at_art(model, x, block=b_, patch=res_d[b_ - 1])
        by_block.append(float(((out_p == out_d) & (out_d != out_x)).mean()))
        at = out_p if b_ == blk else at
    np.save(OUT_DIR / f"{tag}.npy", np.stack([out_x, out_d, at]).astype(np.int16))
    da, db = pcfg.NUM0 + pcfg.count(p.donor, A) + 1, pcfg.NUM0 + pcfg.count(p.donor, B)
    row: dict[str, Any] = {"f": TAG[f_op], "block": blk, "n_pairs": len(p.x), "P": by_block[blk - 1], "P_by_block": by_block,
                           "changed": float((at != out_x).mean()), "eq_donor_a_plus_1": float((at == da).mean()),
                           "eq_donor_b": float((at == db).mean()), "donor_out_differs": float((out_d != out_x).mean())}  # fmt: skip
    if swap is not None:
        hit, clip = (at == out_d) & (out_d != out_x), np.minimum(swap, k["swap_clip"])
        row["P_by_swap"] = {str(v): [float(hit[clip == v].mean()), int((clip == v).sum())] for v in np.unique(clip)}
    log(f"{tag}: " + " ".join(f"{q} {v}" for q, v in row.items() if q != "P_by_block") + f" by block {np.round(by_block, 3).tolist()}")
    return row


def reading_p(pa: float, pb: float, bars: dict[str, Any]) -> str:
    """Part 1: 'both' iff P_a > hi_Pa and P_b > hi_Pb; 'reads b' iff P_b > hi_Pb and P_a < lo_Pa; 'reads a' iff P_a >
    hi_Pa and P_b < lo_Pb; else 'neither' (strict; a nan fails every test -> 'neither')."""
    a_hi, b_hi = pa > bars["P_a"]["hi"], pb > bars["P_b"]["hi"]
    if a_hi and b_hi:
        return "both"
    if b_hi and pa < bars["P_a"]["lo"]:
        return "reads b"
    if a_hi and pb < bars["P_b"]["lo"]:
        return "reads a"
    return "neither"


def reading_through(la: str, lb: str) -> str:
    """Part 2, from Stage A's per-count labels: 'reads b' iff b 'through' and a 'not through'; 'reads a' the reverse;
    'both' iff both 'through'; else 'neither'."""
    if la == lb == "through":
        return "both"
    return {("not through", "through"): "reads b", ("through", "not through"): "reads a"}.get((la, lb), "neither")


def natural_controls(models: dict[str, GPT], d: dict[str, Any], bars: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """LOW's count-a patch on Stage A's natural pairs ('C a 40') >= low_pa_min; every state's natural P at l*
    ('C b 40') within reproduce_tol of a2_<arm>.json; N1 1K's natural P_b > hi_Pb."""
    blk = {A: bars["l_star_a"], B: bars["l_star_b"]}
    low = patch_read(models["LOW"], d["natural"][A], A, blk[A], A, k, "natural_LOW_C_a_40_a")
    reads, checks = {"LOW_C_a_40": {"a": low}}, {"LOW_natural_P_a": low["P"] >= k["low_pa_min"]}
    for s in STATES:
        arm, it = s.split("_")
        a2 = json.loads((RESULTS / f"a2_{arm}.json").read_text())["states"][it]
        reads[s] = {TAG[op]: patch_read(models[s], d["natural"][op], op, blk[op], B, k, f"natural_{s}_{TAG[op]}") for op in (A, B)}
        for t in ("a", "b"):
            checks[f"{s}_natural_P_{t}_reproduces_a2"] = abs(reads[s][t]["P"] - a2[f"check_count_{t}"]["P"]) <= k["reproduce_tol"]
    checks["N1_1000_natural_P_b_above_hi"] = reads["N1_1000"]["b"]["P"] > bars["P_b"]["hi"]
    log(f"natural controls: {checks}")
    return {"reads": reads, "checks": checks}


def part1(models: dict[str, GPT], d: dict[str, Any], bars: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    blk = {A: bars["l_star_a"], B: bars["l_star_b"]}
    out: dict[str, Any] = {}
    for s in (*STATES, "LOW"):
        r = {TAG[op]: patch_read(models[s], d["swap"][op], op, blk[op], B, k, f"swap_{s}_{TAG[op]}") for op in (A, B)}
        out[s] = r | {"reading": reading_p(r["a"]["P"], r["b"]["P"], bars)}
    valid = out["N1_1000"]["reading"] == "reads b"
    out["valid"], out["readings"] = valid, {s: out[s]["reading"] if valid else "void" for s in SCORED}
    log(f"part 1 readings (valid {valid}): { {s: out[s]['reading'] for s in (*STATES, 'LOW')} }")
    return out


def part2(models: dict[str, GPT], d: dict[str, Any], bars: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    blk, texts = {A: bars["l_star_a"], B: bars["l_star_b"]}, d["off"][0][:N_B]
    out: dict[str, Any] = {}
    for s in (*STATES, "LOW"):
        r: dict[str, Any] = {}
        for op in (A, B):
            t = TAG[op]
            r[t] = patch_read(models[s], d["off_pairs"][op], op, blk[op], B, k, f"off_{s}_{t}")
            r[t]["B"] = behaviour_share(models[s], texts, op)
            r[t]["label"] = label(r[t]["B"], r[t]["P"], bars[f"B_{t}"], bars[f"P_{t}"])
        out[s] = r | {"reading": reading_through(r["a"]["label"], r["b"]["label"])}
        log(f"off {s}: B_a {r['a']['B']:.4f} B_b {r['b']['B']:.4f} labels {r['a']['label']}, {r['b']['label']} -> {out[s]['reading']}")
    valid = out["N1_1000"]["reading"] == "reads b"
    out["valid"], out["readings"] = valid, {s: out[s]["reading"] if valid else "void" for s in SCORED}
    return out


def load_models() -> dict[str, GPT]:
    def state(s: str) -> GPT:
        arm, it = s.split("_")
        return load_model(state_path(state_key(arm, ARMS[arm]), int(it)))

    return {"LOW": a1_models()["LOW"]} | {s: state(s) for s in STATES}


def stop(ctrl: dict[str, Any]) -> None:
    write_json(MISS, ctrl)
    log(f"STOP: a control missed (apparatus): {ctrl['checks']}; see {MISS.name} (exit 1)")
    sys.exit(1)


def main() -> None:
    if not PRED4.exists() or str(json.loads(PRED4.read_text()).get("status", "")).startswith("draft"):
        sys.exit(f"file the predictions first (a draft is refused): {PRED4}")
    k = json.loads(PRED4.read_text())["constants"]
    bars = filed_bars(json.loads(PRED.read_text())["a2_bars"], json.loads((RESULTS / "a1c_patch.json").read_text())["a2_bars"])
    assert bars is not None and bars["l_star_a"] == bars["l_star_b"] == k["scored_block"], "Stage A's filed l* is the scored block"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = build(k)
    ctrl: dict[str, Any] = {"checks": model_free_checks(k, d)}
    log(f"model-free controls {ctrl['checks']}")
    if not all(ctrl["checks"].values()):
        stop(ctrl)
    models = load_models()
    nat = natural_controls(models, d, bars, k)
    ctrl = {"checks": ctrl["checks"] | nat["checks"]}
    if not all(ctrl["checks"].values()):
        stop(ctrl | {"natural": nat["reads"]})
    write_json(OUT, {"constants": k, "controls": ctrl, "natural": nat["reads"], "part1_swap_recipients": part1(models, d, bars, k),
                     "part2_off_label": part2(models, d, bars, k)})  # fmt: skip


if __name__ == "__main__":
    main()
