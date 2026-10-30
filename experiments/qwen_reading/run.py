"""E57 (qwen_reading), Q3: reading what each saved Qwen3-1.7B update does to the decomposition. Plan:
reports/plans/2026-10-30_plan_q3-reading-the-update.md (sections 4, 6, 7); predictions and first-minutes values in
results/predictions.json (the run refuses to start without it, or while results/stopped.json exists). Shared rules:
commute.qwen_reading; the job (model, decomposition, weight sets, cells, Σ, split) in lab.py beside this file.

Stages, one cluster L40 job each, cheapest first; each writes results/<stage>.json (C: C_<state>.json per state) as
it finishes and resumes from what is written; every number also goes to results/run.log. A missed early check writes
results/stopped.json and exits 1. Large files (gate counts, think samples) go to results/large/<stage>/.
  A   controls alone: W0 as the fine-tuned model (4 cells = K0, |S| = 0), E40 step 10's first 16 rows against
      Amendment 2, do-nothing through the state path (share 1); then W0's split-half floor, the plain and prompted
      answer sets (the prompt positive control, S_prompt), and row 4 (a)-(b) on the four δW and the Gaussian null.
  B1  W0's think-on answers to GSM8K test 0-127 (model card sampling, 2048 new tokens): large/B1/think.jsonl.
  B2  the reference sets T (think), L (length-matched no-think), C (FineWeb), longest row first.
  C0  the two random controls on the maths rows: the Gaussian δW (KL-matched to E40) and Amendment 2's in-span random
      change: four cells, S. A share >= 0.8 means row 1 has no teeth (card before Stage D).
  C   E40 step 10, gold step 100, anchor 2 seeds 0 and 1 step 100: four cells on the maths and own rows, S,
      placement. After C0, B2 and each state of C: results/summary.json (teeth verdict, row-3 overlaps).
Usage (repo root, L40): python experiments/qwen_reading/run.py --stage A|B1|B2|C0|C [--decomp DIR] [--e40 DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from lab import (
    A2,
    CFG,
    DEV,
    DW_SUMS,
    E45,
    F64,
    LARGE,
    MATHS,
    REF,
    RESULTS,
    STATES,
    WEB,
    Lab,
    null_seed,
    pinned,
)
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_sets, split
from commute import qwen_reading as qr
from commute.qwen_anchors import candidate, user
from commute.qwen_maps import random_pieces
from commute.qwen_sft import sha256
from commute.qwen_vpd import KINDS, SITES, importances, parse_site

THINK = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}  # the Qwen3 model card's thinking mode
CONTROLS = ("gauss", "inspan")


def save(name: str, t: dict[str, Tensor]) -> None:
    p = LARGE / f"{name}.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in t.items()}, str(p))


def on(name: str) -> Tensor:
    t = load_file(str(LARGE / f"{name}.safetensors"))
    return qr.gated(t["active"], int(t["n_pos"]))


def total(rows: dict[str, list[float]], n: int | None = None) -> dict[str, float]:
    return {k: sum(v[:n]) for k, v in rows.items()}


class Text:
    """The tokenizer, GSM8K test 0-127, and the plain per-problem rows unpacked from E40's maths rows."""

    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
        self.think_end = int(self.tok.convert_tokens_to_ids("</think>"))
        self.sets = qwen_sets.load()
        self.test = self.sets["test"][: CFG.problems + 1]  # one more: where the last problem's response ends
        flat = [t for r in json.loads(MATHS.read_text())["rows"] for t in r]
        prompts = [self.ids(qwen_eval.prompt(self.tok, it["question"])) for it in self.test]
        self.plain = qr.unpack(flat, prompts)[: CFG.problems]
        self.n_prompt = [len(p) for p in prompts]
        assert len(self.plain) == CFG.problems, f"{len(self.plain)} problems in the maths rows"

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def think_prompt(self, i: int) -> str:
        msgs = [{"role": "user", "content": user(self.test[i]["question"])}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def keep(self, ids: list[int]) -> Tensor:
        """The answer positions: after the row's last </think>."""
        k = torch.zeros(len(ids), dtype=torch.bool, device=DEV)
        k[qr.after(ids, self.think_end) :] = True
        return k


def count_rows(lab: Lab, rows: list[list[int]], keeps: list[Tensor | None], tag: str) -> qr.Counts:
    """Gate counts over rows, longest first (the counts do not depend on the order), so the GPU rule after the
    first row reads this set's longest."""
    c = lab.counts()
    torch.cuda.reset_peak_memory_stats()
    pairs = sorted(zip(rows, keeps, strict=True), key=lambda p: -len(p[0]))
    for i, (r, k) in enumerate(pairs):
        qr.gate_counts(lab.model, torch.tensor([r], device=DEV), lab.gamma, lab.comps, c, k)
        if i == 0:
            lab.log(f"[{tag}] first row: {len(r)} tokens, {lab.peaks()}")
            lab.gpu_rule(f"{tag}, {len(r)} tokens")
    lab.log(f"[{tag}] {len(rows)} rows, {c.n_pos} positions, {int(c.on().sum())} gated on, {lab.peaks()}")
    return c

# ----------------------------------------------------------------------------- A


def stage_A(lab: Lab) -> None:
    out = lab.get("A")
    if "controls" not in out:
        out["controls"] = a_controls(lab)
        lab.put("A", out)
    if "sets" not in out:
        out["sets"] = a_sets(lab)
        lab.put("A", out)
    if "row4" not in out:
        out["row4"] = a_row4(lab)
        lab.put("A", out)


def a_controls(lab: Lab) -> dict[str, Any]:
    """Section 6.1, in order, each a stop: W0 as FT, E40's 16 rows, do-nothing; the GPU rule in every pass."""
    rows, n16, n = lab.maths(), CFG.n16, CFG.n16 * CFG.row_len
    lab.set_weights("FT", None)
    c0, c1 = lab.counts(CFG.n_rows), lab.counts()
    w = lab.cells(rows, c0=c0, c1=c1, tag="W0 as FT")
    z = total(lab.cells(rows[:n16], lab.change(None, "zero"), tag="W0, zero split, 16 rows"))
    w0, k0_16, s_w0 = total(w), total(w, n16)["K0"], int((c1.on() & ~c0.on()).sum())
    lab.check("A_W0", all(qr.close(w0[k], w0["K0"], CFG.id_tol) for k in ("K00", "K01"))
              and all(qr.close(z[k], k0_16, CFG.id_tol) for k in ("K10", "K11")) and s_w0 == 0,
              {"K0": w0["K0"], "K00": w0["K00"], "K01": w0["K01"], "K10_16": z["K10"], "K11_16": z["K11"],
               "K0_16": k0_16, "S": s_w0})
    save("A/W0_maths", c0.tensors())
    dW = lab.load_dw("e40")
    lab.set_weights("FT", dW)
    e = total(lab.cells(rows[:n16], tag="E40 step 10, 16 rows"))
    lab.check("A_E40", qr.close(e["K0"] / n, REF["K0_16"], CFG.rep_tol)
              and qr.close(e["K00"] / n, REF["K00_16"], CFG.rep_tol),
              {"K0_16": e["K0"] / n, "K00_16": e["K00"] / n, "ref": [REF["K0_16"], REF["K00_16"]]})
    d = total(lab.cells(rows[:n16], lab.change(dW, None), tag="do-nothing, 16 rows"))
    share = qr.shares(e | d, n)["regating"]
    lab.check("A_do_nothing", qr.ok(share) and abs(share - 1) <= CFG.id_tol, {"share": share, **d})
    lab.set_weights("FT", None)
    return {"W0": w0, "W0_zero16": z, "S_W0": s_w0, "E40_16": e, "dn_16": d, "dn_share": share,
            "means_W0": {k: v / rows.numel() for k, v in w0.items()}}


def a_sets(lab: Lab) -> dict[str, Any]:
    """The split-half floor; the gate-count check; the plain answer set and the prompted one (S_prompt)."""
    st = load_file(str(LARGE / "A/W0_maths.safetensors"))
    half, n_pos = CFG.n_rows // 2, CFG.n_rows // 2 * CFG.row_len
    a, b = (qr.gated(st["rows"][sl].long().sum(0), n_pos) for sl in (slice(0, half), slice(half, None)))
    floor = qr.split_floor(a, b)
    tx = Text()
    first, k = torch.tensor([tx.plain[0]], device=DEV), tx.keep(tx.plain[0])
    ref, got = lab.counts(), lab.counts()
    _, g = importances(lab.model, first, lab.gamma, lab.comps)
    ref.add_row(g, k)
    del g
    qr.gate_counts(lab.model, first, lab.gamma, lab.comps, got, k)
    gap = float((got.gsum - ref.gsum).abs().max() / ref.gsum.abs().max().clamp(min=1e-30))
    lab.check("A_gate_counts", bool((got.active == ref.active).all()) and got.n_pos == ref.n_pos and gap <= 1e-6,
              {"gsum_rel_gap": gap, "n_pos": [got.n_pos, ref.n_pos]})
    plain = count_rows(lab, tx.plain, [tx.keep(r) for r in tx.plain], "plain")
    fmt = [tx.ids(candidate(tx.tok, "format", tx.test[i]["question"], ())) + r[tx.n_prompt[i] :]
           for i, r in enumerate(tx.plain)]
    prompted = count_rows(lab, fmt, [tx.keep(r) for r in fmt], "prompted")
    save("A/plain", plain.tensors())
    save("A/prompt", prompted.tensors())
    save("A/layout", {"site_of": lab.lay.site_of})
    s_prompt = int((prompted.on() & ~plain.on()).sum())
    return {"split_half": floor, "min_size": qr.min_size(floor), "S_prompt": s_prompt,
            "S_prompt_clears": s_prompt >= qr.min_size(floor), "plain_on": int(plain.on().sum()),
            "plain_positions": plain.n_pos, "prompt_positions": prompted.n_pos}


def a_row4(lab: Lab) -> dict[str, Any]:
    """Row 4 (a)-(b): share of ‖δW‖² by layer and kind, effective rank per matrix, for the four δW and the Gaussian
    null (split.random_like of E40's step 10, seeds 0-3, seeded per site). First records E45's sha256 for Stage C."""
    DW_SUMS.write_text("".join(f"{sha256(lab.dw_path(n))}  {n}\n" for n in STATES if n != "e40"))
    sig, out, e40 = lab.sigma(), {}, lab.load_dw("e40")
    for name in [*STATES, *(f"null{k}" for k in CFG.null_seeds)]:
        dW = lab.load_dw(name) if name in STATES and name != "e40" else e40
        e, er = {}, {}
        for s in SITES:
            M = dW[s].to(DEV, F64)
            if name.startswith("null"):
                M = split.random_like(M, sig[s].to(DEV), null_seed(int(name[4:]), s))
            e[s], er[s] = float(M.pow(2).sum()), qr.effective_rank(torch.linalg.svdvals(M.float()))
            del M
        E = sum(e.values())
        out[name] = {"layer": [sum(v for s, v in e.items() if parse_site(s)[0] == i) / E for i in range(28)],
                     "kind": {k: sum(v for s, v in e.items() if parse_site(s)[1] == k) / E for k in KINDS},
                     "erank": er, "energy": E}
        lab.log(f"[row4] {name}: kinds {json.dumps(out[name]['kind'])}, median erank "
                f"{sorted(er.values())[len(er) // 2]:.1f}")
    return out

# ----------------------------------------------------------------------------- B1, B2


def stage_B1(lab: Lab) -> None:
    tx = Text()
    tok, gen = qwen_eval.load_model()
    sink = LARGE / "B1" / "think.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(x) for x in sink.read_text().splitlines()] if sink.is_file() else []
    for b, s in enumerate(range(0, CFG.problems, CFG.think_batch)):
        idx = [i for i in range(s, s + CFG.think_batch) if i not in {r["idx"] for r in recs}]
        if idx:
            torch.manual_seed(b)
            outs = qwen_eval.sample(tok, gen, [tx.think_prompt(i) for i in idx], 1, THINK, CFG.think_new)
            new = [{"idx": i, "ids": ids, "truncated": cut, "closed": tx.think_end in ids}
                   for i, (ids, cut) in zip(idx, outs, strict=True)]
            with sink.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
            recs += new
            lab.log(f"[B1] batch {b}: {len(new)} answers, {lab.peaks()}")
        drop = sum(not r["closed"] for r in recs) / len(recs)
        lab.check("B1", drop <= CFG.drop_max, {"after_batch": b, "dropped_share": drop, "n": len(recs)})
    lens = [len(tx.ids(tx.think_prompt(r["idx"]))) + len(r["ids"]) for r in recs]
    lab.put("B1", {"n": len(recs), "dropped": sum(not r["closed"] for r in recs), "max_len": max(lens),
                   "mean_len": sum(lens) / len(lens), "truncated": sum(r["truncated"] for r in recs)})


def stage_B2(lab: Lab) -> None:
    tx = Text()
    assert lab.alive is not None  # the live set is read from U and V before U leaves the GPU
    lab.comps.U = {k: v.cpu() for k, v in lab.comps.U.items()}  # Γ's gates read V only: 9.7 GB freed for ~2,300-token rows
    recs = sorted((json.loads(x) for x in (LARGE / "B1" / "think.jsonl").read_text().splitlines()),
                  key=lambda r: r["idx"])
    kept = [r for r in recs if r["closed"]]
    T = [tx.ids(tx.think_prompt(r["idx"])) + r["ids"] for r in kept]
    L = [qr.length_matched(tx.plain[r["idx"]], t, tx.plain, r["idx"], tx.think_end)
         for r, t in zip(kept, T, strict=True)]
    web = json.loads(WEB.read_text())["rows"][: CFG.n_rows]
    counts = {"T": count_rows(lab, T, [tx.keep(t) for t in T], "T think"),
              "L": count_rows(lab, L, [tx.keep(x) for x in L], "L length-matched"),
              "C": count_rows(lab, web, [None] * len(web), "C fineweb")}
    plain, sets = on("A/plain"), {}
    for name, c in counts.items():
        save(f"B2/{name}", c.tensors())
        sets[name] = c.on() & ~plain
    sizes = {k: int(v.sum()) for k, v in sets.items()}
    both = int((sets["T"] & sets["L"]).sum())
    lab.put("B2", {"rows": len(kept), "dropped": len(recs) - len(kept), "longest": max(len(t) for t in T),
                   "longest_L": max(len(x) for x in L), "sizes": sizes,
                   "T_and_L": both, "T_like_L": both >= 0.8 * sizes["T"],
                   "L_aligned": sum(qr.after(a, tx.think_end) == qr.after(t, tx.think_end)
                                    for a, t in zip(L, T, strict=True))})
    summarize(lab)

# ----------------------------------------------------------------------------- C0, C


def read_state(lab: Lab, name: str, dW: dict[str, Tensor], own: Tensor | None, nulls: bool) -> dict[str, Any]:
    """Row 1's cells on the maths rows (and the own rows), row 2's S, and (nulls) row 4 (c)'s placement."""
    rows = {"maths": lab.maths()} | ({} if own is None else {"own": own})
    lab.set_weights("FT", dW)
    c0, c1, o0 = lab.counts(), lab.counts(CFG.n_rows), lab.counts()
    cells = {"maths": lab.cells(rows["maths"], c0=c0, c1=c1, tag=f"{name} maths")}
    w0 = load_file(str(LARGE / "A/W0_maths.safetensors"))["active"]
    lab.check(f"C_g0_{name}", bool((c0.active.cpu() == w0).all()),
              {"g0_counts_differ_from_A": int((c0.active.cpu() != w0).sum())})
    if name == "e40":
        m, n = total(cells["maths"]), rows["maths"].numel()
        lab.check("C_E40", qr.close(m["K00"] / n, REF["K00"], CFG.rep_tol)
                  and qr.close(m["K0"] / n, REF["K0"], CFG.rep_tol),
                  {"K00": m["K00"] / n, "K0": m["K0"] / n, "ref": [REF["K00"], REF["K0"]]})
    if own is not None:
        cells["own"] = lab.cells(own, c0=o0, tag=f"{name} own")
    sp = lab.split(dW, nulls=nulls, halves=nulls and name == "e40")
    ch = lab.change(dW, sp.pop("X"))
    for k, r in rows.items():
        cells[k] |= lab.cells(r, ch, tag=f"{name} {k}, split")
    del ch
    save(f"C/{name}_g0", c0.tensors())
    save(f"C/{name}_g1", c1.tensors())
    S = c1.on() & ~c0.on()
    rec: dict[str, Any] = {"rows": cells, "S": int(S.sum()), "S_by_kind": {k: int(S[lab.lay.kind_mask(k)].sum())
                                                                          for k in KINDS},
                           "stored_error": sp["stored_error"], "split_s": sp["s"]}
    for k, r in cells.items():
        n = len(r["K0"]) * CFG.row_len
        rec[k] = qr.whole(r, CFG.row_len) | {
            "boot95": qr.boot_share(r, qr.resample(len(r["K0"]), CFG.boot), CFG.row_len),
            "means": {c: v / n for c, v in total(r).items()}}
    if nulls:
        rec["placement"] = placement(lab, sp["pe"], o0.on() if own is not None else c0.on())
    return rec


def placement(lab: Lab, pe: dict[str, Tensor], g_on: Tensor) -> dict[str, Any]:
    null = [qr.placement(pe[f"null{k}"], g_on) for k in CFG.null_seeds]
    base = sum(null) / len(null)
    share = qr.placement(pe["ls"], g_on)
    mlp = lab.lay.kind_mask("gate", "up", "down")
    out = {"share": share, "null": null, "excess": share - base, "gated_on": int(g_on.sum()),
           "mlp_share": qr.div(float(pe["ls"][mlp].sum()), float(pe["ls"].sum()))}
    if "half" in pe:  # the positive control: E40 split onto the often-on half alone
        h = qr.placement(pe["half"], g_on) - base
        out["often_on_half"] = {"excess": h, "expected": 1 - base, "sees": qr.ok(h) and qr.r9(h) >= qr.r9((1 - base) / 2)}
    return out


def inspan(lab: Lab) -> dict[str, Tensor]:
    """E44 Amendment 2's random change at a = 17.78: its Σ-matched Gaussian pieces (per-site scale from amend2/S.json
    step 10) on the live columns, as the weight change dσ(dU, dV) + dU dVᵀ (fp32, host)."""
    rec = json.loads((A2 / "S.json").read_text())["10"]["per_site"]
    out = {}
    for s in SITES:
        U_A, V_A = lab.alive.uv(s, DEV, F64)
        RU, RV = (x.double() for x in random_pieces(tuple(U_A.shape), tuple(V_A.shape), 0, s, DEV))
        dU, dV = rec[s]["a"] * CFG.inspan_a * RU, rec[s]["a"] * CFG.inspan_a * RV
        out[s] = pinned((split.dsigma(dU, dV, U_A, V_A) + dU @ dV.T).float().cpu())
    return out


def stage_C0(lab: Lab) -> None:
    out, r16 = lab.get("C0"), lab.maths(CFG.n16)
    e40 = lab.load_dw("e40")
    lab.set_weights("FT", e40)
    target = lab.kl_rows("FT", r16)
    lab.set_weights("state", inspan(lab))  # the in-span change is checked here and rebuilt after gauss is read
    kin = lab.kl_rows("state", r16)
    lab.drop_set("state")
    lab.check("C0", qr.close(target, REF["kl_e40_16"], CFG.rep_tol) and qr.close(kin, REF["kl_inspan_16"],
              CFG.rep_tol), {"kl_e40_16": target, "kl_inspan_16": kin})
    out |= {"kl_e40_16": target, "kl_inspan_16": kin}
    if "gauss" not in out:
        lab.drop_set("FT")
        sig = lab.sigma()
        G = {s: pinned(split.random_like(e40[s].to(DEV, F64), sig[s].to(DEV), null_seed(0, s)).float().cpu())
             for s in SITES}
        del e40

        def kl_at(b: float) -> float:
            lab.set_weights("state", G, b)
            x = lab.kl_rows("state", r16)
            lab.log(f"[C0 search] b {b:.6g}: KL(W0 ‖ Gaussian) {x:.9g}, target {target:.9g}")
            return x

        out["search"] = qr.scale_search(kl_at, target, CFG.kl_match, CFG.kl_evals)
        lab.drop_set("state")
        for g in G.values():
            g.mul_(out["search"]["b"])  # in place: no second copy
        lab.log(f"[C0 search] done, {lab.peaks()}")
        out["gauss"] = read_state(lab, "gauss", G, None, nulls=False) | {"matched": out["search"]["matched"]}
        lab.put("C0", out)
        del G
        lab.drop_set("FT")
    else:
        del e40
    if "inspan" not in out:
        out["inspan"] = read_state(lab, "inspan", inspan(lab), None, nulls=False) | {"matched": True}
        lab.put("C0", out)
    out["no_teeth"] = any(out[n]["maths"]["status"] == "scored" and qr.r9(out[n]["maths"]["regating"]) >= CFG.high
                          for n in CONTROLS)
    lab.put("C0", out)
    summarize(lab)


def stage_C(lab: Lab) -> None:
    if not all(n in lab.get("C0") for n in CONTROLS):
        sys.exit("refusing to start C: stage C0 has not finished (the random controls run alone first)")
    tx, longest = None, 0.0
    for name, st in STATES.items():
        if (RESULTS / f"C_{name}.json").is_file():
            continue
        lab.wall(longest)
        t0 = time.time()
        dW, own = lab.load_dw(name), None
        if st.own is not None:
            tx = tx or Text()
            rows = qr.own_rows(tx.tok, st.own[0], st.own[1], tx.sets["train"])
            lab.set_weights("FT", dW)
            x = lab.kl_rows("FT", torch.tensor(rows))
            want = json.loads((E45 / f"{st.e_record}.json").read_text())["states"]["100"]["kl"]["own"]
            lab.check(f"C_E45_{name}", qr.close(x, want, CFG.rep_tol), {"kl_own": x, "E45": want})
            own = torch.tensor([r[: CFG.row_len] for r in rows])
        lab.put(f"C_{name}", read_state(lab, name, dW, own, nulls=True))
        del dW
        longest = max(longest, time.time() - t0)
        summarize(lab)

# ----------------------------------------------------------------------------- the summary


def summarize(lab: Lab) -> None:
    """results/summary.json from whatever is written: the teeth verdict (C0 and C_e40); row 3's overlaps of every
    state's S with P, T, L, C against the site-stratified draws; the anchor-2 seed pair (the overlap's control)."""
    out: dict[str, Any] = {}
    c0, e40 = lab.get("C0"), lab.get("C_e40")
    if e40 and all(n in c0 for n in CONTROLS):
        ctrl = {n: {"rows": c0[n]["rows"]["maths"], "matched": c0[n]["matched"]} for n in CONTROLS}
        out["teeth"] = qr.teeth(e40["rows"]["maths"], ctrl, qr.resample(CFG.n_rows, CFG.boot), CFG.row_len,
                               CFG.high)
    A = lab.get("A").get("sets")
    if A:
        site_of = load_file(str(LARGE / "A/layout.safetensors"))["site_of"]
        plain, pool = on("A/plain"), ~on("A/W0_maths")  # the draws' pool: live, not gated on under g0
        refs = {"P": on("A/prompt") & ~plain}
        if (LARGE / "B2/T.safetensors").is_file():
            refs |= {k: on(f"B2/{k}") & ~plain for k in ("T", "L", "C")}
        S = {n: on(f"C/{n}_g1") & ~on(f"C/{n}_g0") for n in [*STATES, *CONTROLS]
             if (LARGE / f"C/{n}_g1.safetensors").is_file()}
        out |= {"min_size": A["min_size"], "ref_sizes": {k: int(v.sum()) for k, v in refs.items()}}
        out["overlap"] = {n: {k: qr.overlap(s, x, site_of, len(SITES), pool, CFG.draws) for k, x in refs.items()}
                          | {"size": int(s.sum()), "scored": int(s.sum()) >= A["min_size"]} for n, s in S.items()}
        if "a2long_s0" in S and "a2long_s1" in S:
            ov = qr.overlap(S["a2long_s0"], S["a2long_s1"], site_of, len(SITES), pool, CFG.draws)
            out["seed_pair"] = ov | {"above_99": qr.above_99(ov)}
    lab.put("summary", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("A", "B1", "B2", "C0", "C"))
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    a = ap.parse_args()
    lab = Lab(a.stage, a.e40, a.decomp)
    with torch.no_grad():
        {"A": stage_A, "B1": stage_B1, "B2": stage_B2, "C0": stage_C0, "C": stage_C}[a.stage](lab)
    lab.log(f"[{a.stage} done] {lab.peaks()}")


if __name__ == "__main__":
    main()
