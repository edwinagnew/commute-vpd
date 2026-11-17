"""E202, the second run of cell P2.33: E200's share re-read on big-gain windows (user, 2026-11-17). Plan
reports/plans/2026-11-17_plan_share-built-kind.md, addendum "Second run"; card card_run2.txt; rules, thresholds and expected
values results/run2/predictions.json (the run refuses unless it is committed unchanged).

What changes from E200 (run.py, which stays as it ran): the windows only. Per pair, the first n_read (2048) windows of the
pair's read half where F's top next token differs from M's (E200's rule) AND F raises that token's log-odds by at least
local_faith.MIN_GAIN over M's log-odds of the same token (`log_odds`, the quantity of the share's T cells). The selection
reads F and M only, before any fine-tune; it scans the whole read half (100,000 windows) at most, which is the cap. The
share, its gain floor, the target (F's top token in a ninth column), the 15 states and the rule are E200's functions.

  set-up   the windows, per pair, with their counts; a pair under th.min_windows windows gets no rows
  stage 0  the identity control, alone, on each pair's new windows (E200's tolerance); a miss is exit 1
  stage 1  the positive control, alone (E200's stage1 unchanged: ident32_s0 on its own windows, r 0.80244 within 0.001)
  stage 2  the window counts; exit 1 if Always on, Cue or Table has fewer than th.min_windows
  stage 3  the 15 states: E139's 12 saved states and E200's three Cue states (results/large/cue_s*.pt), read only
  stage 4  E200's filed rule (`verdicts`) on the finite shares, results/run2/rules.json
Every number printed; `STAGE n DONE` after each stage. A rerun resumes stage 3 after the last state read.
    sbatch --comment='E202 share built kind big-gain windows' cluster/share_built_kind_run2.sbatch
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.data import LAST
from commute.hidden_maps import chunks, run
from commute.identity import log_odds
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN, route_cells
from commute.old_tasks import clone_net
from commute.steering import Net

E200 = load_experiment("share_built_kind", "run")
EXPERIMENT = "E202"
OUT = E200.RESULTS / "run2"
PRED = OUT / "predictions.json"
SCORED = ("Always on", "Cue", "Table")
STATES = {**E200.SAVED, **E200.CUE}  # 15 states; E200.CUE points at E200's results/large/cue_s*.pt
E200.RESULTS = OUT  # E200's stage1 writes its stage1.json here, not over E200's
log, dump = E200.log, E200.dump


def big_gain_rows(net: Net, F: VR.Target, read: Tensor, cfg: Any) -> tuple[Tensor, dict[str, Any]]:
    """The first cfg.n_read windows of `read` where F's top token differs from M's and F's log-odds of it exceeds M's by at
    least MIN_GAIN, each with F's top token as a ninth column; the counts, and F's mean gain on the rows kept."""
    picked, gains, n_seen, n_changed = [], [], 0, 0
    for t in chunks(read, cfg.read_batch):
        lf, lm = F(t)[0][:, -1], run(net, t)[0][:, -1]
        yf = lf.argmax(-1)
        gain = log_odds(lf, yf) - log_odds(lm, yf)
        changed = yf != lm.argmax(-1)
        keep = (changed & (gain >= MIN_GAIN)).cpu()
        picked.append(torch.cat([t.cpu()[keep], yf.cpu()[keep, None]], 1))
        gains.append(gain.double().cpu()[keep])
        n_seen, n_changed = n_seen + t.shape[0], n_changed + int(changed.sum())
        if sum(p.shape[0] for p in picked) >= cfg.n_read:
            break
    rows, g = torch.cat(picked)[: cfg.n_read], torch.cat(gains)[: cfg.n_read]
    return rows, {"pool": read.shape[0], "scanned": n_seen, "changed": n_changed, "qualifying_in_scanned": sum(p.shape[0] for p in picked),
                  "n_rows": rows.shape[0], "F_gain_mean": float(g.mean()) if len(g) else float("nan")}


def stage0(net: Net, w0: dict[str, Tensor], rows: dict[str, Tensor], base: dict[str, Any], th: dict[str, Any]) -> None:
    out: dict[str, Any] = {"pairs": {}, "passed": False}
    ident = clone_net(net).net.state_dict()  # a fine-tune at zero steps
    for pair, r in rows.items():
        rec = out["pairs"][pair] = E200.read_state(net, ident, w0, r, base[pair])
        log(f"identity {pair}: gain {rec['gain']:.3e}, L01-L00 {rec['num']:.3e}, max gate change {rec['max_dg']:.3e}, "
            f"T0 {rec['cells']['T0']:.4f}, L00 {rec['cells']['L00']:.4f}, windows {r.shape[0]}")
        if max(abs(rec["gain"]), abs(rec["num"]), rec["max_dg"]) > th["identity_tol"]:
            out["stopped"] = f"{pair}: the identity reads off zero"
            dump(OUT / "stage0.json", out)
            log(f"STOPPED: {out['stopped']}")
            raise SystemExit(1)
    out["passed"] = True
    dump(OUT / "stage0.json", out)
    print("STAGE 0 DONE", flush=True)


def stage2(counts: dict[str, Any], th: dict[str, Any]) -> None:
    short = [p for p in SCORED if counts[p]["n_rows"] < th["min_windows"]]
    dump(OUT / "stage2.json", {"pairs": counts, "min_windows": th["min_windows"], "short": short})
    for pair, c in counts.items():
        log(f"windows {pair}: {c['n_rows']} kept ({c['qualifying_in_scanned']} qualifying, {c['changed']} changed, of {c['scanned']} "
            f"scanned in a pool of {c['pool']}), F's mean gain on them {c['F_gain_mean']:.4f}")
    if short:
        log(f"STOPPED: fewer than {th['min_windows']} big-gain windows for {short}")
        raise SystemExit(1)
    print("STAGE 2 DONE", flush=True)


def stage3(ctx: dict[str, Any]) -> None:
    path = OUT / "stage3.json"
    for name, (group, pair, src) in STATES.items():
        if name in E200.done(path):
            continue
        if pair not in ctx["rows"]:  # XOR only (stage 2 stops on a scored pair): reference, not read
            out = json.loads(path.read_text()) if path.is_file() else {"states": {}}
            out["states"][name] = {"group": group, "pair": pair, "r": float("nan"), "not_read": "too few big-gain windows"}
            dump(path, out)
            log(f"{name} ({group}): not read, too few big-gain windows")
            continue
        sd = torch.load(src, map_location=ctx["net"].device)
        E200.read_into(path, name, group, pair, sd, ctx, {"source": str(src.relative_to(E200.HERE.parent))})
        del sd
    print("STAGE 3 DONE", flush=True)


def stage4(th: dict[str, Any]) -> None:
    shares: dict[str, list[float]] = {}
    for rec in json.loads((OUT / "stage3.json").read_text())["states"].values():
        shares.setdefault(rec["pair"], []).append(rec["r"])
    out = {"shares_by_pair": shares, "predictions": E200.verdicts(shares, th)}
    dump(OUT / "rules.json", out)
    for key, v in out["predictions"].items():
        log(f"{key} Always on vs {v['vs']}: {v['call']} ({v['result']}), d {v.get('d', float('nan')):.4f}, bar {v.get('bar', float('nan')):.4f}, n {v['n']}")
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
    rows, base, counts = {}, {}, {}
    for pair in VR.PAIRS:
        r, counts[pair] = big_gain_rows(net, VR.piece_target(net, p, pair), VR.pair_windows(cfg, pair, sets, task)[1], cfg)
        if r.shape[0] >= th["min_windows"]:
            rows[pair], base[pair] = r, route_cells(net.model, r, E200.HALF, LAST, ci_len=E200.HALF + 1)
        log(f"{pair}: {r.shape[0]} big-gain windows of {counts[pair]['scanned']} scanned")
    log(f"{EXPERIMENT}: rows ready, cue {task.cue}")
    OUT.mkdir(exist_ok=True)
    stage0(net, w0, rows, base, th)
    E200.stage1(net, w0, th)
    stage2(counts, th)
    ctx = {"net": net, "w0": w0, "rows": rows, "base": base, "decomposed": {f"{k}.weight" for k in net.W0}}
    assert len(ctx["decomposed"] & set(w0)) == 24, "the decomposed matrices' names are not the state's"
    stage3(ctx)
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
