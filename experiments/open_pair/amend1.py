"""open_pair Amendment 1 (E58, 2026-10-30): two $0 checks on how to read E58, one job, each control first.
Plan: reports/plans/2026-10-30_plan_e58-open-pair.md, section "Amendment 1". Rules, settings and thresholds:
results/predictions_amend1.json (filed and committed before the run; the run refuses otherwise; never edited).

  A  (the rank-k readout on E58's six saved states; exit 1 at the first control miss): first the zero update, which
     must read exactly 0 everywhere; then per state, sha256 of the state
     and of its task's windows equal run.json's; at the held-out keys' windows, the log-odds gain (tuned minus W0) of
     W0's top-20 tokens at the window's key, of the window's own value and of the trained keys' values. Its control,
     checked on every state before the next: the own-value gain equals run.json's H, and the trained-value push equals
     the value run.json's H, H_spec and H_other imply (`open_pair.implied_push`).
  C0 (the label's positive control, built): the rank-1 map (each key's value = W0's most probable next token after
     "M key", over E54's training windows) and the copy map (value = key) on E54's windows and E58's key split; the
     zero update reads G = H = 0 exactly on every split task (exit 1 otherwise); W0's geometric-mean P(value); the
     rank-1 map's distinct values and held-out keys with a value no trained key has.
  C1 per map (rank-1 then copy), seeds 0-2: E54's recipe (`keydep.elicit_ft`) on the 24 trained keys' windows; E58's
     label (value-specific gain, E58's bars and gate); the readout; r, ratio and S (exit 1 on a block-0 shift).
  score: `open_pair.score_amend1` -> results/amend1.json. Every scored number is also printed.
Writes <large>/amend1/windows_{rank1,copy}{,_trained,_new}.pt and <large>/amend1/{rank1,copy}_s{0,1,2}.pt (~113 MB each).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute import steering as ST
from commute.heldout_keys import PARTS, behaviour, gain, say_label, split_task, vpd_read
from commute.identity import log_odds

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, RUN, OUT = RESULTS / "predictions_amend1.json", RESULTS / "run.json", RESULTS / "amend1.json"
E54 = HERE.parent / "keydependent_elicit" / "results"
E58_STATES = [f"{k}_s{s}" for k in ("elicit", "cipher") for s in (0, 1, 2)]
MAPS = ("rank1", "copy")
log = KD.log
Tasks = dict[str, dict[str, ST.Task]]


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the user")


def load_split(ctx: KD.Ctx, d: Path, name: str) -> dict[str, ST.Task]:
    return {p: ST.load_task(ctx.net, d / f"windows_{name}_{p}.pt", f"{name}_{p}", name) for p in PARTS}


# ----------------------------------------------------------------------------- the rank-k readout


def readout(net: ST.Net, t: ST.Task, dW: ST.DW, trained_vals: list[int], th: dict[str, Any]) -> dict[str, Any]:
    """At each held-out-key window: the gain (log-odds, tuned minus W0, at the answer position) of W0's top-k tokens at
    the window's key (ranked by W0's mean next-token distribution over that key's windows), of the window's own value,
    T_w = the mean over the trained keys' distinct values other than the own value, and R_w = the mean over the top-k
    tokens that are neither (W0's other preferences). d = T_w - R_w; the verdict is `open_pair.push_verdict`."""
    keys, own, k = t.toks[:, t.half], t.tgt(t.toks), th["top_k"]
    L0 = net.run(t.toks)[0][:, t.half].float().cpu()
    with net.at(dW):
        L1 = net.run(t.toks)[0][:, t.half].float().cpu()
    p0 = torch.softmax(L0.double(), -1)
    at_key = {x: p0[keys == x].mean(0) for x in keys.unique().tolist()}
    top = torch.stack([at_key[x].topk(k).indices for x in keys.tolist()])
    own_rank = torch.tensor([1 + int((at_key[x] > at_key[x][v]).sum()) for x, v in zip(keys.tolist(), own.tolist(), strict=True)])
    g = lambda tok: (log_odds(L1, tok) - log_odds(L0, tok)).double()
    R = torch.stack([g(top[:, j]) for j in range(k)], 1)
    tv = torch.tensor(sorted(set(trained_vals)))
    keep = (tv[None, :] != own[:, None]).double()
    T_w = (torch.stack([g(int(v)) for v in tv], 1) * keep).sum(1) / keep.sum(1)
    peer = (~torch.isin(top, tv) & (top != own[:, None])).double()
    R_w = (R * peer).sum(1) / peer.sum(1)
    d, nb, sd = T_w - R_w, th["n_boot"], th["boot_seed"]
    rec = {"own": float(g(own).mean()), "T": float(T_w.mean()), "R_peer": float(R_w.mean()), "d": float(d.mean()),
           "s_d": OP.boot_sd_mean(d, nb, sd), "s_R": OP.boot_sd_mean(R_w, nb, sd), "rank_gain": R.mean(0).tolist(),
           "own_rank_median": float(own_rank.double().median()), "n_peer_mean": float(peer.sum(1).mean()), "n": len(d)}
    rec["verdict"] = OP.push_verdict(rec["d"], rec["s_d"], rec["R_peer"], rec["s_R"], th["k_sd"])
    return rec


def say_readout(n: str, r: dict[str, Any]) -> None:
    log(f"[readout {n}] own {r['own']:.4f} T {r['T']:.4f} R_peer {r['R_peer']:.4f} d {r['d']:.4f} (3 s_d {3 * r['s_d']:.4f}, "
        f"3 s_R {3 * r['s_R']:.4f}) -> {r['verdict']} | own rank median {r['own_rank_median']:.0f} | gain by W0 rank 1-20 "
        f"{[round(v, 3) for v in r['rank_gain']]}")


def stage_a(ctx: KD.Ctx, e58: dict[str, Any], large: Path, out: dict[str, Any]) -> None:
    """The readout on E58's six states; each state's control (files, own gain = H, T = the implied push) before the next."""
    th, tol, rec = ctx.P["thresholds"], ctx.P["thresholds"]["identity_tol"], out["readout"]
    tasks = {k: load_split(ctx, large, k) for k in ("elicit", "cipher")}
    tv = {k: [int(t["trained"].table[x]) for x in t["trained"].keys] for k, t in tasks.items()}
    z = readout(ctx.net, tasks["elicit"]["new"], {k: torch.zeros_like(w) for k, w in ctx.net.W0.items()}, tv["elicit"], th)
    out["readout_zero"] = {**z, "pass": z["own"] == z["T"] == z["R_peer"] == 0.0 and max(map(abs, z["rank_gain"])) == 0.0}
    say_readout("zero update", z)
    if not out["readout_zero"]["pass"]:
        stop(out, "the readout of the zero update is not exactly 0")
    for n in E58_STATES:
        kind, st = n.split("_")[0], e58["states"][n]
        files = {p: KD.sha256(large / f"windows_{kind}_{p}.pt") for p in PARTS}
        if KD.sha256(large / f"{n}.pt") != st["ft"]["sha256"] or files != e58["windows_sha256"][kind]:
            stop(out, f"{n}: a saved file is not the one E58's run recorded")
        r = readout(ctx.net, tasks[kind]["new"], torch.load(large / f"{n}.pt", weights_only=True), tv[kind], th)
        r["H_e58"], r["T_implied"] = st["H"], OP.implied_push(st["H"], st["H_spec"], st["H_other"], th["n_pool"], th["n_held"])
        r["control"] = abs(r["own"] - r["H_e58"]) <= tol and abs(r["T"] - r["T_implied"]) <= tol
        rec[n] = r
        say_readout(n, r)
        log(f"[readout {n} control] own {r['own']:.6f} vs E58 H {r['H_e58']:.6f}; T {r['T']:.6f} vs implied {r['T_implied']:.6f} (tol {tol}) -> {r['control']}")
        dump(out)
        if not r["control"]:
            stop(out, f"{n}: the readout does not reproduce E58's H or the implied trained-value push")


# ----------------------------------------------------------------------------- the label's positive control


def build_maps(ctx: KD.Ctx, trained: list[int], held: list[int], d: Path) -> dict[str, Any]:
    """The rank-1 and copy maps on E54's elicit windows (same windows, marker and keys; only the table differs), split by
    E58's keys; their values and sha256."""
    src = torch.load(E54 / "large" / "windows_elicit.pt", weights_only=False)
    task, V = src["task"], int(src["table"].shape[0])
    dist = kv.key_distributions(lambda x: ctx.net.run(x)[0], src["train"], ctx.net.device)
    maps = {"rank1": kv.KVTask("rank1", task.marker, task.keys, tuple(int(dist[x].argmax()) for x in task.keys)),
            "copy": kv.identity("copy", task.marker, task.keys)}
    rec: dict[str, Any] = {}
    for n, m in maps.items():
        torch.save({**src, "task": m, "table": m.table(V)}, d / f"windows_{n}.pt")
        val = dict(zip(m.keys, m.values, strict=True))
        rec[n] = {"values": [val[x] for x in m.keys], "sha256": {"full": KD.sha256(d / f"windows_{n}.pt"), **split_task(d / f"windows_{n}.pt", trained, held, d, n)},
                  "n_distinct": len(set(m.values)), "n_novel": OP.n_novel([val[x] for x in trained], [val[x] for x in held]),
                  "n_value_is_key": sum(val[x] == x for x in m.keys)}
    return rec


def stage_c0(ctx: KD.Ctx, trained: list[int], held: list[int], d: Path, out: dict[str, Any]) -> Tasks:
    rec = out["c0"] = build_maps(ctx, trained, held, d)
    tasks = {n: load_split(ctx, d, n) for n in MAPS}
    zero = {k: torch.zeros_like(w) for k, w in ctx.net.W0.items()}
    for n in MAPS:
        rec[n]["zero_max_abs"] = max(float(gain(ctx.net, tasks[n][p], zero).abs().max()) for p in PARTS)
        rec[n]["w0_geo_p"] = {p: KD.geo_rate(tasks[n][p].lp0) for p in PARTS}
        log(f"[c0 {n}] values {rec[n]['values']} distinct {rec[n]['n_distinct']} novel held-out {rec[n]['n_novel']} value=key {rec[n]['n_value_is_key']} "
            f"| W0 geo-mean P(value) {rec[n]['w0_geo_p']} | zero-update max |gain| {rec[n]['zero_max_abs']}")
    dump(out)
    if any(rec[n]["zero_max_abs"] != 0.0 for n in MAPS):
        stop(out, "the zero update does not read 0 on a control task")
    return tasks


def x_novel(b: dict[str, Any], t: dict[str, ST.Task]) -> float:
    """Reported: the key-mean H_spec over held-out keys whose value no trained key has, over G_spec (nan if none)."""
    tv = {int(t["trained"].table[x]) for x in t["trained"].keys}
    hs = [b["H_spec_per_key"][str(x)] for x in t["new"].keys if int(t["new"].table[x]) not in tv]
    return sum(hs) / len(hs) / b["G_spec"] if hs and b["G_spec"] else float("nan")


def stage_c1(ctx: KD.Ctx, tasks: Tasks, d: Path, out: dict[str, Any]) -> None:
    th, st = ctx.P["thresholds"], out["control"]
    for n in MAPS:
        ws = torch.load(d / f"windows_{n}_trained.pt", weights_only=False)
        tv = [int(tasks[n]["trained"].table[x]) for x in tasks[n]["trained"].keys]
        base = KD.base_of(ctx, tasks[n]["trained"])
        for s in ctx.P["settings"]["seeds"]:
            name = f"{n}_s{s}"
            dW, meta = KD.elicit_ft(ctx, ws, s)
            torch.save(dW, d / f"{name}.pt")
            b = behaviour(ctx.net, tasks[n], dW, th)
            st[name] = {"ft": {**meta, "sha256": KD.sha256(d / f"{name}.pt")}, **b, "x_novel": x_novel(b, tasks[n])}
            log(f"[ft {name}] step {meta['step']} learned {meta['learned']} ||dW|| {meta['norm']:.4f} p_heldout {meta['p_heldout']}")
            say_label(name, b)
            st[name]["readout"] = readout(ctx.net, tasks[n]["new"], dW, tv, th)
            say_readout(name, st[name]["readout"])
            st[name].update(vpd_read(ctx, tasks[n]["trained"], dW, base))
            log(f"[read {name}] x_novel {st[name]['x_novel']:.4f} r {st[name]['r']:.4f} ratio {st[name]['ratio']:.4f} S {st[name]['S']:.4f} "
                f"(block {st[name]['layer']}, block-0 {st[name]['S_block0']:.1e})")
            dump(out)
            if abs(st[name]["S_block0"]) * abs(st[name]["G_score_logP"]) > th["block0_nats"]:
                stop(out, f"{name}: block-0 shift over {th['block0_nats']} nats")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="E58's windows and states; this run writes to <large>/amend1")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P, e58 = json.loads(PRED.read_text()), json.loads(RUN.read_text())
    S = P["settings"]
    large = a.large.resolve()
    d = large / "amend1"
    d.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (HERE / "amend1.py", REPO / "src" / "commute" / "open_pair.py", REPO / "src" / "commute" / "heldout_keys.py")]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "run_json_sha256": KD.sha256(RUN), "readout": {}, "control": {}}
    trained, held = e58["key_split"]["trained"], e58["key_split"]["held"]
    if (trained, held) != (S["trained_keys"], S["held_keys"]):
        stop(out, "E58's key split is not the filed one")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    stage_a(ctx, e58, large, out)
    tasks = stage_c0(ctx, trained, held, d, out)
    stage_c1(ctx, tasks, d, out)
    out["score"] = OP.score_amend1(out["readout"], out["control"], out["c0"]["rank1"]["n_novel"], P["thresholds"], S["seeds"])
    dump(out)
    log(f"[score] {json.dumps(OP.jsonable(out['score']))}")


if __name__ == "__main__":
    main()
