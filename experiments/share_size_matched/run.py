"""E204 share_size_matched (cell P2.34): does the 67M re-gating share follow the built kind of change or its size? Plan
reports/plans/2026-11-17_plan_share-size-matched.md; card card.txt; rules, thresholds and expected values
results/predictions.json (the run refuses unless it is committed unchanged).

F_big is E137's Always on edit (`VR.piece_edit`: Q b added at block 2's input, last position, f = 1) with b scaled by one
factor alpha: the same fixed push, larger. alpha is set by a scan of F_big against M only, on E202's window rule
(`run2.big_gain_rows`: the first 2048 read-half windows where F's top token differs from M's and F raises its log-odds by
at least 1), so that F_big's mean gain on its kept windows matches Table's F there (19.13). The 67M is then fine-tuned to
imitate F_big with E139's recipe and Always on seeds (`VR.finetune`, 300 steps), and each state is read exactly as E202 read
its states (`run.read_into`, same share, floor and target) on F_big's windows. Table's three readings are E202's
(results/run2/stage3.json), not read again: same code, same windows (Table's F is unchanged), same saved states.

  set-up   E202's Always on windows (alpha 1), with the scan's first record
  stage 0  the identity control on them, alone (E202's stage0); a miss is exit 1
  stage 1  the positive control, alone (E200's stage1: ident32_s0, r 0.80244 within 0.001); a miss is exit 1
  stage 2  the alpha scan: alpha 1 must repeat E202's Always on windows; the grid until F_big's gain reaches Table's, then
           bisection in log alpha; stops on no crossing, a poor match, too few windows, or a failed kind check
  stage 3  per seed: fine-tune (VR.finetune's step-10 check), save, E139's leftover distances, read; a seed whose gain is
           under th.reach_share of Table's fine-tunes' mean gain stops the job (the filed "not reached" stop)
  stage 4  the filed rule (E200's `verdicts`, unchanged) on the three new shares against E202's Table shares
Results in results/stage<n>.json after every record; every number printed; `STAGE n DONE` after each stage. A rerun resumes
stage 3 from the saved states.
    sbatch --comment='E204 share size matched' cluster/share_size_matched.sbatch
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.data import LAST
from commute.hidden_maps import chunks, run
from commute.identity import log_odds
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import route_cells
from commute.old_tasks import clone_net
from commute.steering import Net

E202 = load_experiment("share_built_kind", "run2")  # the window rule and the identity stage
E200 = E202.E200  # the read, the positive control, the rule, Config (E139's recipe)
E139 = load_experiment("reuse_variable_finetune", "run")  # the leftover distances
EXPERIMENT = "E204"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
RUN2 = E202.OUT  # experiments/share_built_kind/results/run2, read only
E202.OUT = E200.RESULTS = RESULTS  # their stage 0 and stage 1 write here, not over E202's
AO, BIG = "Always on", "Always on, matched size"
SEEDS = (0, 1, 2)
HALF = E200.HALF
log, dump = E200.log, E200.dump


def kind_check(net: Net, p: VR.Pieces, alpha: float, rows: Tensor, batch: int) -> dict[str, float]:
    """On F_big's kept windows: the share whose target the unit push (alpha 1) also raises over M (log-odds); the share held by
    the most common target; the push's norm over the mean norm of M's stream at block 2's input, last position."""
    if rows.shape[0] == 0:
        return {"raised_by_unit": math.nan, "top_target_share": math.nan, "push_ratio": math.nan}
    F1, raised, norms = VR.piece_target(net, p, AO), [], []
    for t in chunks(rows, batch):
        toks = t[:, :-1]
        l1, (lm, seen) = F1(toks)[0][:, -1], run(net, toks, sites=("in2",))
        y = t[:, -1].to(l1.device)
        raised.append((log_odds(l1, y) - log_odds(lm[:, -1], y) > 0).cpu())
        norms.append(seen["in2"][:, -1].float().norm(dim=-1).cpu())
    return {"raised_by_unit": float(torch.cat(raised).float().mean()),
            "top_target_share": float(torch.bincount(rows[:, -1]).max()) / rows.shape[0],
            "push_ratio": alpha * float(p.b.norm()) / float(torch.cat(norms).mean())}


def scan_point(net: Net, p: VR.Pieces, alpha: float, read: Tensor, cfg: Any) -> tuple[Tensor, dict[str, Any]]:
    """F_big at this alpha: E202's windows and counts, F_big's mean gain on them, and the kind check."""
    F = VR.piece_target(net, replace(p, b=alpha * p.b), AO)
    rows, counts = E202.big_gain_rows(net, F, read, cfg)
    rec = {"alpha": alpha, **counts, **kind_check(net, p, alpha, rows, cfg.read_batch)}
    log(f"alpha {alpha:.4f}: {rec['n_rows']} windows ({rec['changed']} changed of {rec['scanned']} scanned), F_big gain "
        f"{rec['F_gain_mean']:.4f}, raised by the unit push {rec['raised_by_unit']:.4f}, top target share "
        f"{rec['top_target_share']:.4f}, push / stream norm {rec['push_ratio']:.4f}")
    return rows, rec


def stage2(net: Net, p: VR.Pieces, read: Tensor, first: tuple[Tensor, dict[str, Any]], th: dict[str, Any], cfg: Any) -> tuple[float, Tensor]:
    """The alpha scan against M only. Returns alpha and F_big's windows."""
    path = RESULTS / "stage2.json"
    e202 = json.loads((RUN2 / "stage2.json").read_text())["pairs"]
    target = e202["Table"]["F_gain_mean"]
    out: dict[str, Any] = {"target_gain": target, "scan": [first[1]], "passed": False}
    pts: dict[float, tuple[Tensor, dict[str, Any]]] = {1.0: first}

    def stop(msg: str) -> NoReturn:
        out["stopped"] = msg
        dump(path, out)
        log(f"STOPPED: {msg}")
        raise SystemExit(1)

    def gain_at(a: float) -> float:
        pts[a] = scan_point(net, p, a, read, cfg)
        out["scan"].append(pts[a][1])
        dump(path, out)
        return pts[a][1]["F_gain_mean"]

    def best() -> float:
        return min(pts, key=lambda a: abs(pts[a][1]["F_gain_mean"] - target) if math.isfinite(pts[a][1]["F_gain_mean"]) else math.inf)

    one, ref = first[1], e202[AO]
    if abs(one["n_rows"] - ref["n_rows"]) > th["repro_rows"] or not abs(one["F_gain_mean"] - ref["F_gain_mean"]) <= th["repro_gain"]:
        stop(f"alpha 1 does not repeat E202's Always on windows ({one['n_rows']} against {ref['n_rows']}, gain "
             f"{one['F_gain_mean']:.4f} against {ref['F_gain_mean']:.4f})")
    lo, hi = 1.0, None
    for a in th["alpha_grid"][1:]:
        if gain_at(a) >= target:
            hi = a
            break
        lo = a
    if hi is None:
        stop(f"no alpha up to {th['alpha_grid'][-1]} reaches Table's gain {target:.4f}: the push saturates")
    for _ in range(th["bisect_max"]):
        if abs(pts[best()][1]["F_gain_mean"] - target) <= th["match_tol"]:
            break
        mid = math.sqrt(lo * hi)
        if gain_at(mid) >= target:
            hi = mid
        else:
            lo = mid
    alpha = best()
    rows, rec = pts[alpha]
    out |= {"alpha": alpha, "chosen": rec}
    log(f"alpha chosen {alpha:.4f}: F_big gain {rec['F_gain_mean']:.4f} against Table's {target:.4f}, {rec['n_rows']} windows")
    top_bar = max(th["top_share_max"], one["top_target_share"])
    if not abs(rec["F_gain_mean"] - target) <= th["match_stop"]:
        stop(f"the best alpha misses Table's gain by more than {th['match_stop']}")
    if rec["n_rows"] < th["min_windows"]:
        stop(f"fewer than {th['min_windows']} windows at the chosen alpha")
    if not rec["top_target_share"] <= top_bar:
        stop(f"kind check: one target holds {rec['top_target_share']:.4f} of the windows (bar {top_bar:.4f}): the output is near constant")
    if not rec["raised_by_unit"] >= th["raised_min"]:
        stop(f"kind check: the unit push raises only {rec['raised_by_unit']:.4f} of F_big's targets (bar {th['raised_min']})")
    out["passed"] = True
    dump(path, out)
    torch.save(rows, RESULTS / "large" / "rows_big.pt")
    print("STAGE 2 DONE", flush=True)
    return alpha, rows


def stage3(ctx: dict[str, Any], p: VR.Pieces, alpha: float, cfg: Any, th: dict[str, Any], sets: VR.Sets, task: VR.Task) -> None:
    """Per seed: E139's recipe toward F_big with E139's Always on batch draws (300_000 + crc32('always_on_s<seed>') % 65536)."""
    path, net = RESULTS / "stage3.json", ctx["net"]
    fit, read = VR.pair_windows(cfg, AO, sets, task)
    F, sch = VR.piece_target(net, replace(p, b=alpha * p.b), AO), VR.Schedule(cfg.steps, cfg.lr, cfg.warmup, cfg.early)
    table_gains = [s["gain"] for s in json.loads((RUN2 / "stage3.json").read_text())["states"].values() if s["group"] == "Table"]
    reach = th["reach_share"] * sum(table_gains) / len(table_gains)
    for s in SEEDS:
        name = f"always_on_big_s{s}"
        state, tpath = RESULTS / "large" / f"{name}.pt", RESULTS / f"{name}_train.json"
        if name not in E200.done(path):
            if state.is_file():
                ft, rec = VR.load_state(net, state, clone_net), json.loads(tpath.read_text())
            else:
                ft = clone_net(net)
                rec = VR.finetune(net, ft, F, fit, read[: cfg.check_n], sch, cfg.batch, 300_000 + zlib.crc32(f"always_on_s{s}".encode()) % 65_536, log)
                dump(tpath, rec)
                e10 = next(e for e in rec["early"] if e["step"] == 10)
                log(f"{name} step 10: KL(F_big||ft) {e10['kl_to_F']:.5f} against start KL(F_big||M) {rec['kl_F_M']:.5f}")
                if rec["stopped"]:
                    log(f"STOPPED: {name} in training ({rec['stopped']}); no state, no reading")
                    raise SystemExit(1)
                VR.save_state(ft, state)
            dist = E139.distances(net, F, ft, read, cfg)
            log(f"{name} leftover on the read half: TV(ft, F_big) {dist['tv_to_F']:.5f}, KL(F_big||ft) {dist['kl_to_F']:.5f}, "
                f"gap TV(ft, M) {dist['gap']:.5f}, F_big's gap TV(F_big, M) {dist['target_gap']:.5f}")
            E200.read_into(path, name, BIG, BIG, ft.net.state_dict(), ctx,
                           {"alpha": alpha, "leftover": dist, "kl_F_M": rec["kl_F_M"], "kl_to_F_end": rec["early"][-1]["kl_to_F"]})
            del ft
            torch.cuda.empty_cache()
        got = json.loads(path.read_text())["states"][name]
        log(f"{name}: gain {got['gain']:.4f} against the reach bar {reach:.4f} ({th['reach_share']} of Table's fine-tunes' mean gain)")
        if not got["gain"] >= reach:
            log(f"STOPPED: {name} does not reach the large change; the size-matched comparison is not read")
            raise SystemExit(1)
    print("STAGE 3 DONE", flush=True)


def rivals(big: list[float], table: list[float], th: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """E200's rule (`verdicts` P1, its "Always on" side the matched size here); one call decides both rivals: K passes on
    'higher', S on 'within the bar' or 'lower'; 'unreadable' is neither."""
    call = E200.verdicts({"Always on": big, "Table": table}, th)["P1"]
    c = call["call"]
    return call, {"K": c if c == "unreadable" else "pass" if c == "higher" else "miss",
                  "S": c if c == "unreadable" else "miss" if c == "higher" else "pass"}


def stage4(th: dict[str, Any]) -> None:
    big = [s["r"] for s in json.loads((RESULTS / "stage3.json").read_text())["states"].values()]
    table = [s["r"] for s in json.loads((RUN2 / "stage3.json").read_text())["states"].values() if s["group"] == "Table"]
    call, preds = rivals(big, table, th)
    out = {"shares": {BIG: big, "Table (E202)": table}, "call": call, "predictions": preds}
    dump(RESULTS / "rules.json", out)
    log(f"matched Always on vs Table: {call['call']}, d {call.get('d', math.nan):.4f}, bar {call.get('bar', math.nan):.4f}, n {call['n']}; "
        f"K (kind) {preds['K']}, S (size) {preds['S']}")
    print("STAGE 4 DONE", flush=True)


def main() -> int:
    th = json.loads(PRED.read_text())["thresholds"]
    cfg, net = E200.Config(), Net("cuda")
    net.net.requires_grad_(False)
    from commute import vpd

    tok = vpd.tokenizer()
    sets = VR.make_sets(cfg, tok, log)
    task = VR.make_task(sets, tok, int(net.net.lm_head.weight.shape[0]), cfg)  # type: ignore[union-attr]
    if task.cue != json.loads((E200.R1 / "stage0.json").read_text())["pieces"]["cue_token"]:
        log(f"the cue {task.cue} is not R1's: the rows or the vocabulary differ")
        return 1
    p = VR.load_pieces(E200.R1 / "large" / "pieces.pt", net.device, task)
    w0 = {k: v.clone() for k, v in net.net.state_dict().items()}
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    read = VR.pair_windows(cfg, AO, sets, task)[1]
    first = scan_point(net, p, 1.0, read, cfg)
    log(f"{EXPERIMENT}: rows ready, cue {task.cue}, |b| {float(p.b.norm()):.4f}")
    E202.stage0(net, w0, {AO: first[0]}, {AO: route_cells(net.model, first[0], HALF, LAST, ci_len=HALF + 1)}, th)
    E200.stage1(net, w0, th)
    alpha, rows = stage2(net, p, read, first, th, cfg)
    ctx = {"net": net, "w0": w0, "rows": {BIG: rows}, "base": {BIG: route_cells(net.model, rows, HALF, LAST, ci_len=HALF + 1)},
           "decomposed": {f"{k}.weight" for k in net.W0}}
    assert len(ctx["decomposed"] & set(w0)) == 24, "the decomposed matrices' names are not the state's"
    stage3(ctx, p, alpha, cfg, th, sets, task)
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
