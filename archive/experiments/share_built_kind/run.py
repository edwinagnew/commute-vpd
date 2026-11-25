"""E200 share_built_kind (cell P2.33): the 67M re-gating share on E139's fine-tunes, whose kind of change is known by
construction. Plan reports/plans/2026-11-17_plan_share-built-kind.md; card experiments/share_built_kind/card.txt; rules,
thresholds and expected values results/predictions.json (the run refuses unless it is committed unchanged).

The share is the do-nothing route split's r (`local_faith.route_cells`, `masked_lo`, unchanged): r = (L01 - L00) / (T1 - T0),
T the target model's log-odds of a window's target token at its last position (T0 at W0, T1 at the fine-tune), L the
CI-masked model (W0's subcomponents, Delta out) under Gamma's gates from W0's forward (L00) or from the fine-tune's forward
(L01). E139's fine-tunes changed every weight, so the fine-tune is loaded whole for T1 and Gamma's gates, and the masked
forward for L01 runs with all of W0's weights; for a change to the 24 decomposed matrices only this is route_split's r.
r is read only where T1 - T0 >= local_faith.MIN_GAIN (1 log-odds), else nan.
Windows and target, per pair (E137's change F = M + its piece, `VR.piece_target`): the first n_read windows of the pair's
read half (`VR.pair_windows`, not trained on) where F's top next token differs from M's; the target is F's top token,
appended as a last column (`data.LAST`, ci_len 8: no forward and no Gamma sees it).

  stage 0  the identity control, alone: a fine-tune at zero steps on each pair's windows. gain, L01 - L00 and the largest
           gate change must be 0 (within th.identity_tol); a miss is exit 1 and nothing else runs
  stage 1  the positive control, alone: E46's ident32_s0 (the scorecard's e46_copy_random reuse side, seed 0) built as a
           whole state W0 + dW and read through the same loading path on its own windows and table; r must equal the saved
           0.80244 within th.positive_tol, else exit 1 before any E139 state
  stage 2  the 12 saved states: Always on 300 steps and 3000 steps (amendment 2), Table, XOR (reference only)
  stage 3  Cue, 3 seeds, fine-tuned now with E139's recipe (`VR.finetune`, whose step-10 check lost its wrong clause on
           2026-11-17), saved to results/large/cue_s<seed>.pt, then read
  stage 4  the filed rules (results/rules.json)
Results after every state in results/stage<n>.json; every number printed; `STAGE n DONE` after each stage. A rerun resumes.
    sbatch --comment='E200 share built kind' cluster/share_built_kind.sbatch
"""

from __future__ import annotations

import json
import math
import os
import time
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.data import LAST
from commute.hidden_maps import chunks, run
from commute.jobs import hard_exit, require_filed
from commute.local_faith import MIN_GAIN, masked_lo, route_cells
from commute.old_tasks import clone_net
from commute.steering import Net

EXPERIMENT = "E200"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
E139 = HERE.parent / "reuse_variable_finetune" / "results"
R1 = HERE.parent / "reuse_variable_test" / "results"
E46 = HERE.parent / "regating_table" / "results"  # the positive control's state, windows and saved share (stage0b.json)
SCORECARD = HERE.parent / "method_table" / "results" / "final.json"
SEEDS = (0, 1, 2)
HALF = 7  # the last position of an 8-token window; the target sits in a ninth column
_t0 = time.monotonic()

# state name -> (group, E137 pair, path); Cue's are written by stage 2
SAVED = {
    **{f"always_on_s{s}": ("Always on, 300 steps", "Always on", E139 / "large" / f"always_on_s{s}.pt") for s in SEEDS},
    **{f"amend2_always_on_s{s}_step3000": ("Always on, 3000 steps", "Always on", E139 / "large" / f"amend2_always_on_s{s}_step3000.pt") for s in SEEDS},
    **{f"table_s{s}": ("Table", "Table", E139 / "large" / f"table_s{s}.pt") for s in SEEDS},
    **{f"xor_s{s}": ("XOR", "XOR", E139 / "large" / f"xor_s{s}.pt") for s in SEEDS},
}
CUE = {f"cue_s{s}": ("Cue", "Cue", RESULTS / "large" / f"cue_s{s}.pt") for s in SEEDS}


@dataclass(frozen=True)
class Config(VR.Sizes):
    n_read: int = 2048  # windows per pair for the share
    steps: int = 300  # E139's Config, unchanged
    lr: float = 3e-5
    warmup: int = 30
    batch: int = 1024
    check_n: int = 4096
    early: tuple[int, ...] = (1, 2, 3, 5, 10, 30, 100, 300)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.window == HALF + 1 and self.n_read <= self.D // 2 and self.check_n <= self.D // 2


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _t0:7.0f}s] {msg}", flush=True)


def dump(path: Path, out: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    os.replace(tmp, path)


def target_rows(net: Net, F: VR.Target, read: Tensor, cfg: Config) -> tuple[Tensor, dict[str, Any]]:
    """The first cfg.n_read windows of `read` where F's top next token differs from M's, each with F's top token as a ninth
    column; and how many windows were scanned for them."""
    picked, n_seen = [], 0
    for t in chunks(read, cfg.read_batch):
        yf, ym = F(t)[0][:, -1].argmax(-1).cpu(), run(net, t)[0][:, -1].argmax(-1).cpu()
        keep = yf != ym
        picked.append(torch.cat([t.cpu()[keep], yf[keep, None]], 1))
        n_seen += t.shape[0]
        if sum(p.shape[0] for p in picked) >= cfg.n_read:
            break
    rows = torch.cat(picked)[: cfg.n_read]
    assert rows.shape == (cfg.n_read, cfg.window + 1), rows.shape
    return rows, {"scanned": n_seen, "changed_share": sum(p.shape[0] for p in picked) / n_seen}


@contextmanager
def loaded(net: Net, sd: dict[str, Tensor], w0: dict[str, Tensor]) -> Iterator[None]:
    """Every weight of the target model set to `sd` (a fine-tune's `state_dict`), W0's restored on exit."""
    net.net.load_state_dict(sd)
    try:
        yield
    finally:
        net.net.load_state_dict(w0)


def sizes(sd: dict[str, Tensor], w0: dict[str, Tensor], decomposed: set[str]) -> dict[str, float]:
    """||delta|| over the 24 decomposed matrices and over every other weight (embeddings, norms, unembedding): for an audit."""
    sq = {"decomposed": 0.0, "other": 0.0}
    for k, v in w0.items():
        sq["decomposed" if k in decomposed else "other"] += float((sd[k].to(v.device) - v).double().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sq.items()}


def read_state(net: Net, sd: dict[str, Tensor], w0: dict[str, Tensor], rows: Tensor, base: dict[str, Any],
               half: int = HALF, token: int | Tensor = LAST, ci_len: int | None = HALF + 1) -> dict[str, Any]:
    """T1 and g1 from the whole fine-tune; L01 from W0's masked model under g1; the share r (nan below MIN_GAIN). `base` is
    `route_cells` at W0 on the same rows, half, token and ci_len (batch 64 throughout)."""
    with loaded(net, sd, w0):
        cur = route_cells(net.model, rows, half, token, ci_len=ci_len)
    L01, _ = masked_lo(net.model, rows, half, token, cur["gates"], ci_len=ci_len)
    gain, num = cur["T"] - base["T"], L01 - base["L"]
    max_dg = max(float((g1[k] - g0[k]).abs().max()) for g0, g1 in zip(base["gates"], cur["gates"], strict=True) for k in g0)
    return {"r": num / gain if gain >= MIN_GAIN else float("nan"), "gain": gain, "num": num, "max_dg": max_dg,
            "cells": {"T0": base["T"], "T1": cur["T"], "L00": base["L"], "L01": L01, "L11_own": cur["L"]}}


def stage0(net: Net, w0: dict[str, Tensor], rows: dict[str, Tensor], base: dict[str, Any], scan: dict[str, Any], th: dict[str, Any]) -> None:
    path = RESULTS / "stage0.json"
    out: dict[str, Any] = {"pairs": {}, "passed": False}
    ident = clone_net(net).net.state_dict()  # a fine-tune at zero steps
    for pair in VR.PAIRS:
        rec = read_state(net, ident, w0, rows[pair], base[pair]) | scan[pair]
        out["pairs"][pair] = rec
        log(f"identity {pair}: gain {rec['gain']:.3e}, L01-L00 {rec['num']:.3e}, max gate change {rec['max_dg']:.3e}, "
            f"T0 {rec['cells']['T0']:.4f}, L00 {rec['cells']['L00']:.4f}, windows changed {rec['changed_share']:.4f}")
        if max(abs(rec["gain"]), abs(rec["num"]), rec["max_dg"]) > th["identity_tol"]:
            out["stopped"] = f"{pair}: the identity reads off zero"
            dump(path, out)
            log(f"STOPPED: {out['stopped']}")
            raise SystemExit(1)
    out["passed"] = True
    dump(path, out)
    print("STAGE 0 DONE", flush=True)


def stage1(net: Net, w0: dict[str, Tensor], th: dict[str, Any]) -> None:
    """E46's ident32_s0 (dW on the 24 matrices) as a whole state, read as every E139 state is, against its scorecard share:
    final.json `.cells[] | select(.method=="share" and .pair=="e46_copy_random") | .reuse[0]` (E136 reproduced it exactly)."""
    path = RESULTS / "stage1.json"
    cell = next(c for c in json.loads(SCORECARD.read_text())["cells"] if c["method"] == "share" and c["pair"] == "e46_copy_random")
    saved = cell["reuse"][0]
    w = torch.load(E46 / "large" / "windows_ident32.pt", weights_only=False)
    rows, half, table = w["heldout"].tokens, w["heldout"].half, w["table"]
    dW = torch.load(E46 / "large" / "ident32_s0.pt", weights_only=False)
    sd = dict(w0) | {f"{k}.weight": w0[f"{k}.weight"] + v.float().to(w0[f"{k}.weight"].device) for k, v in dW.items()}
    assert len(dW) == 24 and all(f"{k}.weight" in w0 for k in dW)
    base = route_cells(net.model, rows, half, table, ci_len=None)
    rec = read_state(net, sd, w0, rows, base, half, table, None) | {"saved_r": saved, "state": "regating_table ident32_s0"}
    rec["diff"] = rec["r"] - saved
    log(f"positive control ident32_s0: r {rec['r']:.6f} against saved {saved:.6f} (diff {rec['diff']:.2e}), gain {rec['gain']:.4f}")
    rec["passed"] = abs(rec["diff"]) <= th["positive_tol"]  # nan fails
    dump(path, rec)
    if not rec["passed"]:
        log("STOPPED: the positive control does not reproduce its saved share")
        raise SystemExit(1)
    print("STAGE 1 DONE", flush=True)


def read_into(path: Path, name: str, group: str, pair: str, sd: dict[str, Tensor], ctx: dict[str, Any], extra: dict[str, Any]) -> None:
    out = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    rec = {"group": group, "pair": pair, **read_state(ctx["net"], sd, ctx["w0"], ctx["rows"][pair], ctx["base"][pair]),
           "size": sizes(sd, ctx["w0"], ctx["decomposed"]), **extra}
    out["states"][name] = rec
    dump(path, out)
    log(f"{name} ({group}): r {rec['r']:.4f}, gain {rec['gain']:.4f}, L01-L00 {rec['num']:.4f}, max gate change {rec['max_dg']:.4f}, "
        f"||dW|| decomposed {rec['size']['decomposed']:.4f}, other {rec['size']['other']:.4f}")


def done(path: Path) -> set[str]:
    return set(json.loads(path.read_text())["states"]) if path.is_file() else set()


def stage2(ctx: dict[str, Any]) -> None:
    path = RESULTS / "stage2.json"
    for name, (group, pair, src) in SAVED.items():
        if name not in done(path):
            read_into(path, name, group, pair, torch.load(src, map_location=ctx["net"].device), ctx, {"source": str(src.relative_to(HERE.parent))})
    print("STAGE 2 DONE", flush=True)


def stage3(ctx: dict[str, Any], cfg: Config, sets: VR.Sets, task: VR.Task, p: VR.Pieces) -> None:
    """Cue, E139's recipe and seeds (300_000 + crc32(item) % 65536); step 10 read beside E139's own step-10 record."""
    path, net = RESULTS / "stage3.json", ctx["net"]
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    fit, read = VR.pair_windows(cfg, "Cue", sets, task)
    F, sch = VR.piece_target(net, p, "Cue"), VR.Schedule(cfg.steps, cfg.lr, cfg.warmup, cfg.early)
    for name, (group, pair, state) in CUE.items():
        if name in done(path):
            continue
        old = json.loads((E139 / f"{name}.json").read_text())["train"]["early"]
        if state.is_file():
            ft, rec = VR.load_state(net, state, clone_net), json.loads((RESULTS / f"{name}_train.json").read_text())
        else:
            ft = clone_net(net)
            rec = VR.finetune(net, ft, F, fit, read[: cfg.check_n], sch, cfg.batch, 300_000 + zlib.crc32(name.encode()) % 65_536, log)
            e10, o10 = next(e for e in rec["early"] if e["step"] == 10), next(e for e in old if e["step"] == 10)
            log(f"{name} step 10: KL(F||ft) {e10['kl_to_F']:.5f} (E139 {o10['kl_to_F']:.5f}), start KL(F||M) {rec['kl_F_M']:.5f}")
            dump(RESULTS / f"{name}_train.json", rec)
            if rec["stopped"]:
                out = json.loads(path.read_text()) if path.is_file() else {"states": {}}
                out["states"][name] = {"group": group, "pair": pair, "stopped": rec["stopped"], "r": float("nan")}
                dump(path, out)
                log(f"{name}: STOPPED in training ({rec['stopped']}); no state, no reading")
                continue
            VR.save_state(ft, state)
        kl_end = rec["early"][-1]["kl_to_F"]
        read_into(path, name, group, pair, ft.net.state_dict(), ctx, {"kl_to_F_end": kl_end, "kl_F_M": rec["kl_F_M"]})
        del ft
        torch.cuda.empty_cache()
    print("STAGE 3 DONE", flush=True)


def verdicts(shares: dict[str, list[float]], th: dict[str, Any]) -> dict[str, Any]:
    """The filed rule: Always on's readable states pooled (both recipes) against each rewiring group's; d = difference of
    means; bar = 2 * spread * sqrt(1/n_a + 1/n_b); 'higher' if d >= bar, 'lower' if d <= -bar, else 'within the bar';
    'unreadable' if either side has fewer than min_readable finite shares. The prediction passes only on 'higher'."""
    ao = [x for x in shares.get("Always on", []) if math.isfinite(x)]
    out: dict[str, Any] = {}
    for other, key in (("Table", "P1"), ("Cue", "P2")):
        b = [x for x in shares.get(other, []) if math.isfinite(x)]
        if min(len(ao), len(b)) < th["min_readable"]:
            out[key] = {"vs": other, "call": "unreadable", "n": [len(ao), len(b)], "result": "unreadable"}
            continue
        d = round(sum(ao) / len(ao) - sum(b) / len(b), 9)  # rounded: a tie at the bar is not decided by float rounding
        bar = round(2 * th["spread"] * math.sqrt(1 / len(ao) + 1 / len(b)), 9)
        call = "higher" if d >= bar else "lower" if d <= -bar else "within the bar"
        out[key] = {"vs": other, "d": d, "bar": bar, "n": [len(ao), len(b)], "call": call, "result": "pass" if call == "higher" else "miss"}
    return out


def stage4(th: dict[str, Any]) -> None:
    states = {**json.loads((RESULTS / "stage2.json").read_text())["states"], **json.loads((RESULTS / "stage3.json").read_text())["states"]}
    shares: dict[str, list[float]] = {}
    for rec in states.values():
        shares.setdefault(rec["pair"], []).append(rec["r"])
    out = {"shares_by_pair": shares, "predictions": verdicts(shares, th)}
    dump(RESULTS / "rules.json", out)
    for key, v in out["predictions"].items():
        log(f"{key} Always on vs {v['vs']}: {v['call']} ({v['result']}), d {v.get('d', float('nan')):.4f}, bar {v.get('bar', float('nan')):.4f}, n {v['n']}")
    print("STAGE 4 DONE", flush=True)


def main() -> int:
    th = json.loads(PRED.read_text())["thresholds"]
    cfg, net = Config(), Net("cuda")
    net.net.requires_grad_(False)
    from commute import vpd

    tok = vpd.tokenizer()
    sets = VR.make_sets(cfg, tok, log)
    task = VR.make_task(sets, tok, int(net.net.lm_head.weight.shape[0]), cfg)  # type: ignore[union-attr]
    if task.cue != json.loads((R1 / "stage0.json").read_text())["pieces"]["cue_token"]:
        log(f"the cue {task.cue} is not R1's: the rows or the vocabulary differ")
        return 1
    p = VR.load_pieces(R1 / "large" / "pieces.pt", net.device, task)
    w0 = {k: v.clone() for k, v in net.net.state_dict().items()}
    rows, base, scan = {}, {}, {}
    for pair in VR.PAIRS:
        rows[pair], scan[pair] = target_rows(net, VR.piece_target(net, p, pair), VR.pair_windows(cfg, pair, sets, task)[1], cfg)
        base[pair] = route_cells(net.model, rows[pair], HALF, LAST, ci_len=HALF + 1)
    log(f"{EXPERIMENT}: rows ready, cue {task.cue}")
    RESULTS.mkdir(exist_ok=True)
    stage0(net, w0, rows, base, scan, th)
    stage1(net, w0, th)
    ctx = {"net": net, "w0": w0, "rows": rows, "base": base, "decomposed": {f"{k}.weight" for k in net.W0}}
    assert len(ctx["decomposed"] & set(w0)) == 24, "the decomposed matrices' names are not the state's"
    stage2(ctx)
    stage3(ctx, cfg, sets, task, p)
    stage4(th)
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
