"""E207, the second (last) run of cell P2.34: the size-matched test done the other way round. Plan
reports/plans/2026-11-17_plan_share-size-matched.md, section "Second run"; card card_run2.txt; rules, thresholds and
expected values results/run2/predictions.json (the run refuses unless it is committed unchanged). run.py (E204) stays as it ran.

E204 could not scale Always on up to Table's gain (the push saturated near 16 log-odds). So Table is scaled down instead:
F_small is E137's Table edit (`VR.piece_edit`: C z + c added at block 2's input, last position, z block 0's input there) with
C and c multiplied by one factor beta < 1. The map is the same full-rank function of the input, only smaller, so F_small is
still built rewiring. beta is set by a scan of F_small against M only, on E202's window rule (`run2.big_gain_rows`: the first
2048 read-half windows where F's top token differs from M's and F raises its log-odds by at least 1), so that F_small's mean
gain on its kept windows matches Always on's F there (1.8337). The 67M is then fine-tuned to imitate F_small with E139's
recipe and Table's batch draws (`VR.finetune`, 300 steps), and each state is read exactly as E202 read its states
(`run.read_into`) on F_small's windows. Always on's six readings are E202's (results/run2/stage3.json), not read again.

  set-up   E202's Table windows (beta 1), with the scan's first record
  stage 0  the identity control on them, alone (E202's stage0); a miss is exit 1
  stage 1  the positive control, alone (E200's stage1: ident32_s0, r 0.80244 within 0.001); a miss is exit 1
  stage 2  the beta scan: beta 1 must repeat E202's Table windows; halving beta until F_small's gain falls to Always on's,
           then bisection in log beta; stops on no crossing, a match worse than 0.2, under 32 windows, or the kind check
  stage 3  per seed: fine-tune (VR.finetune's step-10 check), save, E139's leftover distances, read; a seed whose gain is
           under local_faith.MIN_GAIN (1 log-odds) stops the job (the filed reach stop: no reading)
  stage 4  the filed rule (E200's `verdicts`, unchanged) on E202's six Always on shares against the three new shares
Results in results/run2/stage<n>.json after every record; every number printed; `STAGE n DONE` after each stage. A rerun
resumes stage 3 from the saved states.
    python experiments/share_size_matched/run2.py
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
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN, route_cells
from commute.old_tasks import clone_net
from commute.steering import Net

E202 = load_experiment("share_built_kind", "run2")  # the window rule and the identity stage
E200 = E202.E200  # the read, the positive control, the rule, Config (E139's recipe)
E139 = load_experiment("reuse_variable_finetune", "run")  # the leftover distances
EXPERIMENT = "E207"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "run2"
LARGE = HERE / "results" / "large"
PRED = OUT / "predictions.json"
RUN2 = E202.OUT  # experiments/share_built_kind/results/run2, read only
E202.OUT = E200.RESULTS = OUT  # their stage 0 and stage 1 write here, not over E202's
AO, TABLE, SMALL = "Always on", "Table", "Table, matched size"
SEEDS = (0, 1, 2)
HALF = E200.HALF
log, dump = E200.log, E200.dump


def scaled(p: VR.Pieces, beta: float) -> VR.Pieces:
    return replace(p, C=beta * p.C, c=beta * p.c)


def answer_share(rows: Tensor, task: VR.Task) -> float:
    """The kind check: the share of kept windows whose target (ninth column) is the Table answer of the last input token
    (`task.table`, -1 off the 32 keys, so a window without a key never counts)."""
    if rows.shape[0] == 0:
        return math.nan
    return float((rows[:, -1] == task.table[rows[:, -2]]).float().mean())


def scan_point(net: Net, p: VR.Pieces, beta: float, read: Tensor, cfg: Any) -> tuple[Tensor, dict[str, Any]]:
    """F_small at this beta: E202's windows and counts, F_small's mean gain on them, and the share that are Table answers."""
    rows, counts = E202.big_gain_rows(net, VR.piece_target(net, scaled(p, beta), TABLE), read, cfg)
    rec = {"beta": beta, **counts, "answer_share": answer_share(rows, p.task)}
    log(f"beta {beta:.5f}: {rec['n_rows']} windows ({rec['changed']} changed of {rec['scanned']} scanned), F_small gain "
        f"{rec['F_gain_mean']:.4f}, Table answers {rec['answer_share']:.4f}")
    return rows, rec


def stage2(net: Net, p: VR.Pieces, read: Tensor, first: tuple[Tensor, dict[str, Any]], th: dict[str, Any], cfg: Any) -> tuple[float, Tensor]:
    """The beta scan against M only. lo is a beta whose gain is under the target, hi one at or over it. Returns beta and the
    windows."""
    path = OUT / "stage2.json"
    e202 = json.loads((RUN2 / "stage2.json").read_text())["pairs"]
    target = e202[AO]["F_gain_mean"]
    out: dict[str, Any] = {"target_gain": target, "scan": [first[1]], "passed": False}
    pts: dict[float, tuple[Tensor, dict[str, Any]]] = {1.0: first}

    def stop(msg: str) -> NoReturn:
        out["stopped"] = msg
        dump(path, out)
        log(f"STOPPED: {msg}")
        raise SystemExit(1)

    def gain_at(b: float) -> float:
        pts[b] = scan_point(net, p, b, read, cfg)
        out["scan"].append(pts[b][1])
        dump(path, out)
        return pts[b][1]["F_gain_mean"]

    def miss(b: float) -> float:
        g = pts[b][1]["F_gain_mean"]
        return abs(g - target) if math.isfinite(g) else math.inf

    one, ref = first[1], e202[TABLE]
    if abs(one["n_rows"] - ref["n_rows"]) > th["repro_rows"] or not abs(one["F_gain_mean"] - ref["F_gain_mean"]) <= th["repro_gain"]:
        stop(f"beta 1 does not repeat E202's Table windows ({one['n_rows']} against {ref['n_rows']}, gain "
             f"{one['F_gain_mean']:.4f} against {ref['F_gain_mean']:.4f})")
    lo, hi = None, 1.0
    for b in th["beta_grid"][1:]:
        if not gain_at(b) >= target:  # nan (no window kept) counts as under
            lo = b
            break
        hi = b
    if lo is None:
        stop(f"no beta down to {th['beta_grid'][-1]} brings F_small's gain down to Always on's {target:.4f}")
    for _ in range(th["bisect_max"]):
        if miss(min(pts, key=miss)) <= th["match_tol"]:
            break
        mid = math.sqrt(lo * hi)
        if gain_at(mid) >= target:
            hi = mid
        else:
            lo = mid
    beta = min(pts, key=miss)
    rows, rec = pts[beta]
    out |= {"beta": beta, "chosen": rec}
    log(f"beta chosen {beta:.5f}: F_small gain {rec['F_gain_mean']:.4f} against Always on's {target:.4f}, {rec['n_rows']} windows, "
        f"Table answers {rec['answer_share']:.4f} (beta 1: {one['answer_share']:.4f})")
    if not miss(beta) <= th["match_tol"]:
        stop(f"the best beta misses Always on's gain by more than {th['match_tol']}")
    if rec["n_rows"] < th["min_windows"]:
        stop(f"fewer than {th['min_windows']} windows at the chosen beta")
    if not rec["answer_share"] >= th["answer_min"]:
        stop(f"kind check: only {rec['answer_share']:.4f} of F_small's targets are Table answers (bar {th['answer_min']})")
    out["passed"] = True
    dump(path, out)
    torch.save(rows, LARGE / "rows_small_table.pt")
    print("STAGE 2 DONE", flush=True)
    return beta, rows


def stage3(ctx: dict[str, Any], p: VR.Pieces, beta: float, cfg: Any, sets: VR.Sets) -> None:
    """Per seed: E139's recipe toward F_small with E139's Table batch draws (300_000 + crc32('table_s<seed>') % 65536)."""
    path, net = OUT / "stage3.json", ctx["net"]
    fit, read = VR.pair_windows(cfg, TABLE, sets, p.task)
    F, sch = VR.piece_target(net, scaled(p, beta), TABLE), VR.Schedule(cfg.steps, cfg.lr, cfg.warmup, cfg.early)
    for s in SEEDS:
        name = f"table_small_s{s}"
        state, tpath = LARGE / f"{name}.pt", OUT / f"{name}_train.json"
        if name not in E200.done(path):
            if state.is_file():
                ft, rec = VR.load_state(net, state, clone_net), json.loads(tpath.read_text())
            else:
                ft = clone_net(net)
                rec = VR.finetune(net, ft, F, fit, read[: cfg.check_n], sch, cfg.batch, 300_000 + zlib.crc32(f"table_s{s}".encode()) % 65_536, log)
                dump(tpath, rec)
                e10 = next(e for e in rec["early"] if e["step"] == 10)
                log(f"{name} step 10: KL(F_small||ft) {e10['kl_to_F']:.5f} against start KL(F_small||M) {rec['kl_F_M']:.5f}")
                if rec["stopped"]:
                    log(f"STOPPED: {name} in training ({rec['stopped']}); no state, no reading")
                    raise SystemExit(1)
                VR.save_state(ft, state)
            dist = E139.distances(net, F, ft, read, cfg)
            log(f"{name} leftover on the read half: TV(ft, F_small) {dist['tv_to_F']:.5f}, KL(F_small||ft) {dist['kl_to_F']:.5f}, "
                f"gap TV(ft, M) {dist['gap']:.5f}, F_small's gap TV(F_small, M) {dist['target_gap']:.5f}")
            E200.read_into(path, name, SMALL, SMALL, ft.net.state_dict(), ctx,
                           {"beta": beta, "leftover": dist, "kl_F_M": rec["kl_F_M"], "kl_to_F_end": rec["early"][-1]["kl_to_F"]})
            del ft
            torch.cuda.empty_cache()
        got = json.loads(path.read_text())["states"][name]
        log(f"{name}: gain {got['gain']:.4f} against the reach bar {MIN_GAIN}")
        if not got["gain"] >= MIN_GAIN:
            log(f"STOPPED: {name} does not reach the small change (the filed reach stop); no reading")
            raise SystemExit(1)
    print("STAGE 3 DONE", flush=True)


def rivals(ao: list[float], small: list[float], th: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """E200's rule (`verdicts` P1, its "Table" side the small Table here): d = mean(Always on) - mean(small Table). One call
    decides both rivals: K passes on 'higher', S on 'within the bar' or 'lower'; 'unreadable' is neither."""
    call = E200.verdicts({AO: ao, TABLE: small}, th)["P1"]
    c = call["call"]
    return call, {"K": c if c == "unreadable" else "pass" if c == "higher" else "miss",
                  "S": c if c == "unreadable" else "miss" if c == "higher" else "pass"}


def stage4(th: dict[str, Any]) -> None:
    small = [s["r"] for s in json.loads((OUT / "stage3.json").read_text())["states"].values()]
    ao = [s["r"] for s in json.loads((RUN2 / "stage3.json").read_text())["states"].values() if s["pair"] == AO]
    call, preds = rivals(ao, small, th)
    out = {"shares": {f"{AO} (E202)": ao, SMALL: small}, "call": call, "predictions": preds}
    dump(OUT / "rules.json", out)
    log(f"Always on vs small Table: {call['call']}, d {call.get('d', math.nan):.4f}, bar {call.get('bar', math.nan):.4f}, n {call['n']}; "
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
    OUT.mkdir(exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    read = VR.pair_windows(cfg, TABLE, sets, task)[1]
    first = scan_point(net, p, 1.0, read, cfg)
    log(f"{EXPERIMENT}: rows ready, cue {task.cue}")
    E202.stage0(net, w0, {TABLE: first[0]}, {TABLE: route_cells(net.model, first[0], HALF, LAST, ci_len=HALF + 1)}, th)
    E200.stage1(net, w0, th)
    beta, rows = stage2(net, p, read, first, th, cfg)
    ctx = {"net": net, "w0": w0, "rows": {SMALL: rows}, "base": {SMALL: route_cells(net.model, rows, HALF, LAST, ci_len=HALF + 1)},
           "decomposed": {f"{k}.weight" for k in net.W0}}
    assert len(ctx["decomposed"] & set(w0)) == 24, "the decomposed matrices' names are not the state's"
    stage3(ctx, p, beta, cfg, sets)
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
