"""open_pair Amendment 2 (E58, 2026-10-30): the elicit, rank-1 and copy maps at four times the keys (128: 96 trained,
32 held out; 64 / 16 if E54's key supply is short), seed 0, E54's recipe; held-out carry x, r and the stand-in ratio on
each, read as Amendment 1 reads them. Does a rule start to carry with more keys, and do r and the ratio move with it?
Plan: reports/plans/2026-10-30_plan_e58-open-pair.md, section "Amendment 2". Rules, settings and thresholds:
results/predictions_amend2.json (filed and committed before the run; the run refuses otherwise; never edited).

  K  (keys; exit 1 on a miss): E54's selection (`keydep.vocab`, `keydep.select`, E54's settings with k = 128) must give
     E54's marker, E54's count of qualified keys and, drawn at k = 32, E54's 32 keys and values exactly. K = 128 if the
     draw gives 128 keys with distinct values, else 64 if it gives 64, else stop. Key split `open_pair.split_keys`
     (K / 4 held out, seed 0).
  W  (windows; exit 1 on a miss): E54's window rule at E54's density (500 x K / 32 windows, seed 0, row split); the
     three tables (elicit = E54's value rule, rank-1 = W0's argmax after "M key", copy = the key), cut by key; the zero
     update reads 0 exactly on all six split tasks.
  E  (the control, alone, first): the elicit fine-tune at its scored step (exit 1 if it never learns), its label;
     exit 1 unless `open_pair.elicit_control` reads
     "pass" or "below" (too small, unreadable or x > 0.05 stops the job before any other map). Then its r and ratio.
  M  rank-1, then copy: fine-tune, read at elicit's scored step (the common step; each map's own step reported), label,
     x_novel, r, ratio, S (exit 1 on a block-0 shift over 1e-5 nats).
  score: `open_pair.score_amend2` -> results/amend2.json. Every scored number is also printed.
Writes <large>/amend2/windows_{elicit,rank1,copy}{,_trained,_new}.pt (~1-3 MB each) and {elicit,rank1,copy}_s0.pt
(~113 MB each).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from amend1 import (  # E58 Amendment 1's helpers, imported unchanged
    git,
    load_split,
    x_novel,
)

from commute import anchors as AN
from commute import elicit as EL
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute.data import split_windows
from commute.heldout_keys import PARTS, behaviour, gain, say_label, split_task, vpd_read

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_amend2.json", RESULTS / "amend2.json"
E54_STAGE1 = HERE.parent / "keydependent_elicit" / "results" / "stage1.json"
MAPS = ("elicit", "rank1", "copy")  # the control first
log = KD.log


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


# ----------------------------------------------------------------------------- K: the key supply


def stage_keys(ctx: KD.Ctx, out: dict[str, Any]) -> tuple[torch.Tensor, int, list[int], dict[int, Any]]:
    """E54's selection at k = 128, checked against E54's stage 1; the K keys (128 or 64, E54's draw order)."""
    S, e54 = ctx.P["settings"], json.loads(E54_STAGE1.read_text())
    rows, plain, marker, ranked, _ = KD.vocab(ctx)
    cands = ranked[S["key_ranks"][0] : S["key_ranks"][1]]
    keys, qual = KD.select(ctx, rows, plain, marker, cands)
    k32 = EL.draw_keys(qual, cands, 32, S["key_draw_seed"])
    rec = out["keys"] = {"marker": marker, "n_qualified": len(qual), "n_draw_at_128": len(keys),
                         "n_distinct_values": len({q["value"] for q in qual.values()}),
                         "qualified_sha256": hashlib.sha256(json.dumps(sorted((x, q["value"]) for x, q in qual.items())).encode()).hexdigest(),
                         "e54_reproduced": k32 == e54["elicit"]["keys"] and [qual[x]["value"] for x in k32] == e54["elicit"]["values"]}
    log(f"[keys] marker {marker} (E54 {e54['marker']}) qualified {len(qual)} (E54 {e54['n_qualified']}) distinct values "
        f"{rec['n_distinct_values']} drawn at k=128: {len(keys)}; E54's 32 keys and values reproduced: {rec['e54_reproduced']}")
    if marker != e54["marker"] or len(qual) != e54["n_qualified"] or not rec["e54_reproduced"]:
        stop(out, "E54's key selection does not reproduce on this card")
    K = next((k for k in (S["k"], S["k_fallback"]) if len(EL.draw_keys(qual, cands, k, S["key_draw_seed"])) == k), 0)
    if not K:
        stop(out, f"fewer than {S['k_fallback']} qualified keys with distinct values")
    rec["K"] = K
    keys = EL.draw_keys(qual, cands, K, S["key_draw_seed"])
    rec["keys"], rec["held"] = keys, OP.split_keys(keys, K // 4, S["split_seed"])[1]
    dump(out)
    return rows, marker, keys, qual


# ----------------------------------------------------------------------------- W: windows and the three maps


def build_maps(ctx: KD.Ctx, rows: torch.Tensor, marker: int, keys: list[int], qual: dict[int, Any], d: Path,
               out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """E54's windows rule at E54's density on the K keys; the elicit, rank-1 and copy tables; each file cut by key."""
    S, K, V = ctx.P["settings"], len(keys), int(ctx.net.net.lm_head.weight.shape[0])
    trained, held = OP.split_keys(keys, K // 4, S["split_seed"])
    w = kv.windows(rows, marker, tuple(keys), AN.HALF, S["windows_per_32_keys"] * K // 32, S["task_windows_seed"])
    train, heldout = split_windows(w)
    dist = kv.key_distributions(lambda x: ctx.net.run(x)[0], train, ctx.net.device)
    maps = {"elicit": kv.KVTask("elicit", marker, tuple(keys), tuple(qual[x]["value"] for x in keys)),
            "rank1": kv.KVTask("rank1", marker, tuple(keys), tuple(int(dist[x].argmax()) for x in keys)),
            "copy": kv.identity("copy", marker, tuple(keys))}
    rec = out["maps"] = {}
    for n, m in maps.items():
        torch.save({"train": train, "heldout": heldout, "task": m, "table": m.table(V)}, d / f"windows_{n}.pt")
        val = dict(zip(m.keys, m.values, strict=True))
        rec[n] = {"values": [val[x] for x in keys], "n_distinct": len(set(m.values)),
                  "n_novel": OP.n_novel([val[x] for x in trained], [val[x] for x in held]),
                  "sha256": {"full": KD.sha256(d / f"windows_{n}.pt"), **split_task(d / f"windows_{n}.pt", trained, held, d, n)}}
    rec["n_windows"] = {"train": len(train), "heldout": len(heldout)}
    tasks = {n: load_split(ctx, d, n) for n in MAPS}
    zero = {k: torch.zeros_like(x) for k, x in ctx.net.W0.items()}
    for n in MAPS:
        rec[n]["zero_max_abs"] = max(float(gain(ctx.net, tasks[n][p], zero).abs().max()) for p in PARTS)
        rec[n]["w0_geo_p"] = {p: KD.geo_rate(tasks[n][p].lp0) for p in PARTS}
        rec[n]["n_windows"] = {p: len(tasks[n][p].toks) for p in PARTS}
        log(f"[maps {n}] distinct {rec[n]['n_distinct']} novel held-out {rec[n]['n_novel']} windows {rec[n]['n_windows']} "
            f"W0 geo-mean P(value) {rec[n]['w0_geo_p']} zero-update max |gain| {rec[n]['zero_max_abs']}")
    dump(out)
    if any(rec[n]["zero_max_abs"] != 0.0 for n in MAPS):
        stop(out, "the zero update does not read 0 on a split task")
    return tasks


# ----------------------------------------------------------------------------- E, then the maps


def one_map(ctx: KD.Ctx, n: str, t: dict[str, Any], d: Path, out: dict[str, Any], step: int | None) -> int:
    """Fine-tune one map (E54's recipe, every E12 checkpoint kept) and read it at `step`, the common step every map is
    read at; `step` None (elicit, first) sets it to elicit's own scored step and stops if elicit does not learn (review
    of a23c3ea: maps read at their own steps would mix update size into the key count). Each map's own scored step is
    reported. Returns the step read."""
    th, seed = ctx.P["thresholds"], ctx.P["settings"]["seed"]
    ft = KD.finetune(ctx, torch.load(d / f"windows_{n}_trained.pt", weights_only=False), seed, KD.E12_CHECKS)
    own, learned = KD.scored_step(ft.evals)
    p = {k: e["p_o"] for k, e in ft.evals.items()}
    log(f"[ft {n}] own scored step {own} learned {learned} p_heldout {p}")
    if step is None and not learned:
        stop(out, "the elicit map did not reach held-out P >= 0.5 by step 300")
    step = own if step is None else step
    dW = ft.deltas[step]
    torch.save(dW, d / f"{n}_s{seed}.pt")
    b = behaviour(ctx.net, t, dW, th)
    meta = {"step": step, "own_step": own, "own_learned": learned, "norm": EL.summed_norm(dW), "p_heldout": p,
            "norms": {k: EL.summed_norm(v) for k, v in ft.deltas.items()}, "sha256": KD.sha256(d / f"{n}_s{seed}.pt")}
    rec = out["states"][n] = {"ft": meta, **b, "x_novel": x_novel(b, t)}
    log(f"[ft {n}] read at the common step {step} ||dW|| {meta['norm']:.4f} (norms by step {meta['norms']})")
    say_label(n, b)
    log(f"[x_novel {n}] {rec['x_novel']:.4f}")
    dump(out)
    if n == "elicit":
        rec["control"] = OP.elicit_control(b["x"], b["label"], th)
        log(f"[control] elicit x {b['x']:.4f} G_spec {b['G_spec']:.4f} (gate {th['G_min']:.3f}) -> {rec['control']}")
        if rec["control"] not in ("pass", "below"):
            stop(out, f"the control reads {rec['control']}")
    rec.update(vpd_read(ctx, t["trained"], dW, KD.base_of(ctx, t["trained"])))
    log(f"[read {n}] r {rec['r']:.4f} ratio {rec['ratio']:.4f} S {rec['S']:.4f} (block {rec['layer']}, block-0 {rec['S_block0']:.1e})")
    dump(out)
    if abs(rec["S_block0"]) * abs(rec["G_score_logP"]) > th["block0_nats"]:
        stop(out, f"{n}: block-0 shift over {th['block0_nats']} nats")
    return step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="this run writes to <large>/amend2")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P = json.loads(PRED.read_text())
    assert P["settings"]["half"] == AN.HALF, "predictions settings disagree with the code"
    d = a.large.resolve() / "amend2"
    d.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (HERE / "amend2.py", HERE / "amend1.py", *(REPO / "src" / "commute" / f for f in ("open_pair.py", "heldout_keys.py", "keydep.py")))]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "states": {}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    rows, marker, keys, qual = stage_keys(ctx, out)
    tasks = build_maps(ctx, rows, marker, keys, qual, d, out)
    step: int | None = None  # elicit (the control) sets the common step; rank-1 and copy are read there
    for n in MAPS:
        step = one_map(ctx, n, tasks[n], d, out, step)
    out["score"] = OP.score_amend2(out["states"], out["maps"]["rank1"]["n_novel"], P["reference_24_keys"], P["thresholds"])
    dump(out)
    log(f"[score] {json.dumps(OP.jsonable(out['score']))}")


if __name__ == "__main__":
    main()
