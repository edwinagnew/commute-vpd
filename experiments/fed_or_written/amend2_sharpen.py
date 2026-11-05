"""E100 Amendment 2, job B: the strong-bigram "sharpen" map against its shuffled twin, two seeds each, read by the sum of
single reverts under the rule job A kept. Plan: reports/plans/2026-11-05_plan_e100-amend2.md; Fable's advice, section 2 and 4:
reports/reviews/2026-11-05_review_e100-next-fable-advice.md. Rules and leans: results/predictions_amend2.json (committed
first; the run refuses otherwise) and `commute.fed_rules`. Refuses to start unless results/amend2.json records a kept rule.

  stage 0, alone (W0): E54's candidate keys (word-like Pile ranks 100-1099), W0's top-1 next token after "M key" on E54's
           selection windows; keys whose top-1 (not the key) has P >= 0.1, 32 drawn as E54 draws (`elicit.draw_keys`, values
           distinct); E54's 500 task windows on them and their row split; on the train windows the keys still strong, the
           first 24 kept (`fed_rules.strong_keys`). Stop under 24. Sharpen table = W0's top-1 there (E58 Amendment 1's
           rank-1 recipe, `experiments/open_pair/amend1.py` build_maps); twin = the same values permuted so no key keeps
           its own (`fed_rules.twin_values`). The zero update must read G = 0 exactly on both sites.
  stage 1: sharpen seeds 0-1, E54's recipe (`keydep.finetune`, every step 1-30 kept) at E12's scored step (`keydep.scored_step`).
  stage 2: twin seeds 0-1, the same recipe, read at the step in 1-30 whose ||dW|| (`elicit.summed_norm`) is nearest the same
           seed's sharpen state's (E54's cipher match, `elicit.matched_step`); stop unless within 5%.
  Each state: step-1 gain on its site finite and positive (else stop); saved in bf16 to results/large/dW_<map>_s<seed>.pt
  and read as saved: G, the 28 single reverts and their sum, the profile-matched plant (plants.py), rho, the reading.
  outcome: `fed_rules.pair_outcome`. results/amend2_sharpen.json is written after every state; a resubmit skips stage 0
  when its windows files match their recorded sha256, and every state already read. "STAGE <n> DONE" per stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import anchors as AN
from commute import elicit as EL
from commute import fed_rules as FR
from commute import keydep as KD
from commute import kv
from commute import locator as L
from commute import steering as ST
from commute.data import split_windows
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256

JB = load_experiment("locator_table", "job")
PL = load_experiment("fed_or_written", "plants")
sites = JB.sites
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE = RESULTS / "large"
PRED, OUT_A, OUT = RESULTS / "predictions_amend2.json", RESULTS / "amend2.json", RESULTS / "amend2_sharpen.json"
MAPS = ("sharpen", "twin")
dump, say, load, stop = JB.dump, JB.say, JB.load, JB.stop


def windows_path(m: str) -> Path:
    return LARGE / f"windows_{m}_trained.pt"


# ----------------------------------------------------------------------------- stage 0: the keys, the two tables


def top1(net: ST.Net, w: Any) -> tuple[dict[int, int], dict[int, float]]:
    """W0's top-1 next token after each key and its probability, from the mean distribution over the key's windows."""
    dist = kv.key_distributions(lambda x: net.run(x)[0], w, net.device)
    return {k: int(d.argmax()) for k, d in dist.items()}, {k: float(d.max()) for k, d in dist.items()}


def build(ctx: KD.Ctx, out: dict[str, Any]) -> None:
    S, net = ctx.P["settings"]["sharpen"], ctx.net
    rows, _plain, marker, ranked, tok = KD.vocab(ctx)
    cands = ranked[S["key_ranks"][0] : S["key_ranks"][1]]
    train_rows = rows[torch.arange(rows.shape[0]) % 5 != 0]
    sel = kv.windows(train_rows, marker, tuple(cands), AN.HALF, S["selection_windows_per_key"] * len(cands), S["selection_seed"])
    v, p = top1(net, sel)
    qual = {x: {"value": v[x]} for x in cands if v[x] != x and p[x] >= S["p_min"]}
    drawn = EL.draw_keys(qual, cands, S["k_drawn"], S["key_draw_seed"])
    w = kv.windows(rows, marker, tuple(drawn), AN.HALF, S["task_windows"], S["task_windows_seed"])
    train, held = split_windows(w)
    tv, tp = top1(net, train)
    keys = FR.strong_keys(drawn, tv, tp, S["p_min"], S["k"])
    rec = out["stage0"] = {"marker": marker, "n_candidates": len(cands), "n_qualified": len(qual), "n_drawn": len(drawn), "n_strong": len(keys),
                           "drawn": {str(x): {"key": tok.decode([x]), "top1": tok.decode([tv[x]]), "p": tp[x]} for x in drawn}}  # fmt: skip
    say("stage0 keys", {k: rec[k] for k in ("marker", "n_candidates", "n_qualified", "n_drawn", "n_strong")})
    if len(keys) < S["k"]:
        stop(OUT, out, f"{len(keys)} strong keys, under {S['k']}: no sharpen pair; hand the column over as the check worded it")
    values = [tv[x] for x in keys]
    twin = FR.twin_values(values, S["twin_seed"])
    mask = lambda ws: torch.isin(ws.tokens[:, ws.half], torch.tensor(keys))
    V = int(net.net.lm_head.weight.shape[0])
    dist = kv.key_distributions(lambda x: net.run(x)[0], train, net.device)
    for m, vals in zip(MAPS, (values, twin), strict=True):
        task = kv.KVTask(m, marker, tuple(keys), tuple(vals))
        torch.save({"train": train.subset(mask(train)), "heldout": held.subset(mask(held)), "task": task, "table": task.table(V)}, windows_path(m))
        p_val = [float(dist[x][y]) for x, y in zip(keys, vals, strict=True)]
        rec[m] = {"values": vals, "sha256": sha256(windows_path(m)), "p_value_train": p_val,
                  "p_value_geo": float(torch.tensor(p_val).log().mean().exp()), "n_distinct": len(set(vals))}  # fmt: skip
        log(f"[stage0 {m}] keys {[tok.decode([x]) for x in keys]} values {[tok.decode([y]) for y in vals]} "
            f"P(value | M key) geo-mean over keys {rec[m]['p_value_geo']:.3e}")
    rec["keys"] = keys
    dump(OUT, out)


def stage0(J: Any, ctx: KD.Ctx, out: dict[str, Any]) -> dict[str, tuple[L.Base, Any, dict[str, Any]]]:
    rec = out.get("stage0", {})
    if rec.get("status") != "pass" or any(not windows_path(m).is_file() or sha256(windows_path(m)) != rec[m]["sha256"] for m in MAPS):
        out.pop("states", None)  # new windows: nothing read before them stands
        build(ctx, out)
        rec = out["stage0"]
    got = {}
    zero = {k: torch.zeros_like(v).cpu() for k, v in J.net.W0.items()}
    for m in MAPS:
        s, w = sites.kv(J.F, windows_path(m))
        base = L.Base(s, L.read(J.net, s), {}, {}, {})
        rec[m]["zero_G"] = float((L.read(J.net, s, zero) - base.lo0)[:, 0].abs().max())
        rec[m]["n_heldout"] = len(s.toks)
        got[m] = (base, PL.kv_writes(J.net, w, s.batch), w)
    rec["status"] = "pass" if all(rec[m]["zero_G"] == 0.0 for m in MAPS) else "miss"
    dump(OUT, out)
    say("stage0 zero update", {m: rec[m]["zero_G"] for m in MAPS})
    if rec["status"] != "pass":
        stop(OUT, out, "the zero update does not read G = 0 on a site")
    return got


# ----------------------------------------------------------------------------- stages 1-2: the fine-tunes


def gain(J: Any, base: L.Base, dW: ST.DW) -> float:
    return float((L.read(J.net, base.site, dW) - base.lo0)[:, 0].mean())


def train(J: Any, ctx: KD.Ctx, m: str, seed: int, base: L.Base, out: dict[str, Any]) -> tuple[ST.DW, dict[str, Any]]:
    """E54's recipe; the sharpen at E12's scored step, the twin at the sharpen's ||dW||. Stops on a missed early check."""
    w = torch.load(windows_path(m), weights_only=False)
    ft = KD.finetune(ctx, w, seed, KD.CIPHER_CHECKS)
    norms = {k: EL.summed_norm(ft.deltas[k]) for k in range(1, 31)}
    meta: dict[str, Any] = {"G_step1": gain(J, base, ft.deltas[1]), "p_heldout": {k: e["p_o"] for k, e in ft.evals.items()}}
    if m == "sharpen":
        meta["step"], meta["learned"] = KD.scored_step(ft.evals)
    else:
        target = out["states"][f"sharpen_s{seed}"]["norm"]
        meta |= {"step": EL.matched_step(norms, target), "norm_target": target}
        meta["norm_matched"] = FR.norm_matched(norms[meta["step"]], target, J.th["norm_tol"])
    dW = ft.deltas[meta["step"]]
    meta |= {"norm": EL.summed_norm(dW), "frobenius": sum(float(v.double().pow(2).sum()) for v in dW.values()) ** 0.5}
    say(f"ft {m}_s{seed}", meta)
    if not FR.step1_ok(meta["G_step1"]):
        stop(OUT, out, f"{m}_s{seed}: step-1 gain {meta['G_step1']} is not finite and positive")
    if m == "twin" and not meta["norm_matched"]:
        stop(OUT, out, f"twin_s{seed}: ||dW|| {meta['norm']:.4f} not within {J.th['norm_tol']} of the sharpen's {meta['norm_target']:.4f}")
    return dW, meta


def state(J: Any, ctx: KD.Ctx, m: str, seed: int, got: dict[str, Any], out: dict[str, Any]) -> None:
    name = f"{m}_s{seed}"
    if name in out.setdefault("states", {}):
        return
    base, X, _ = got[m]
    dW, meta = train(J, ctx, m, seed, base, out)
    path = LARGE / f"dW_{name}.pt"
    torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
    meta |= {"file": path.name, "sha256": sha256(path)}
    dW = {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}  # read as saved
    r = PL.read_sum(J.net, base, dW)
    sh = PL.part_shares(dW)
    p = PL.plant(J.net, base, X, PL.profile(X, sh), r["G"], J.th)
    rec = {"ft": meta, "G": r["G"], "sum": r["reverts_sum"], "own": r["own"], "plant": p, "W": p["W"], "status": p["status"],
           "rho": FR.rho(r["reverts_sum"], p["W"], p["status"]), "state_shares": sh}  # fmt: skip
    rec |= FR.rho_reading(rec["G"], rec["sum"], rec["status"], rec["rho"], out["bars"], J.th)
    rec["reading_fixed"] = FR.sum_reading(rec["G"], rec["sum"], J.th["fed_bar"], J.th["written_bar"], J.th["min_gain"])
    out["states"][name] = rec
    dump(OUT, out)
    log(f"[state {name}] G {rec['G']:.4f} sum {rec['sum']:.4f} W {rec['W']:.4f} rho {rec['rho']:.4f} ({rec['status']}, plant alpha "
        f"{p['alpha']:.4f} {p['match_status']}) -> {rec['reading']} [{rec['basis']}]; fixed {rec['reading_fixed']}")


def outcome(J: Any, out: dict[str, Any]) -> None:
    st = out["states"]
    reads = {m: [st[f"{m}_s{s}"]["reading"] for s in J.P["settings"]["sharpen"]["seeds"]] for m in MAPS}
    W = [st[f"sharpen_s{s}"]["W"] for s in J.P["settings"]["sharpen"]["seeds"]]
    out["outcome"] = {"readings": reads, "rule": out["bars"]["rule"],
                      "pair": FR.pair_outcome(reads["sharpen"], reads["twin"], W, out["bars"]["rule"], J.th["w_unexplained"])}  # fmt: skip
    dump(OUT, out)
    say("outcome", out["outcome"])


# ----------------------------------------------------------------------------- main


def filed() -> dict[str, Any]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    a = load(OUT_A)
    if a.get("kept_rule") not in (FR.SITE_RELATIVE, FR.FIXED):
        sys.exit(f"{OUT_A.relative_to(REPO)} records no kept rule: job A's stage 1 first")
    return json.loads(PRED.read_text())


def timed(k: int, f: Any, *a: Any) -> Any:
    t0 = time.monotonic()
    got = f(*a)
    log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
    print(f"STAGE {k} DONE", flush=True)
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    P = filed()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # reads as E100; the fine-tune keeps E12's bf16 autocast
    LARGE.mkdir(parents=True, exist_ok=True)
    net = ST.Net(args.device)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {head} predictions {sha256(PRED)}")
    J = JB.Job(net, sites.Files(), P, {})
    ctx = KD.Ctx(net, {}, {}, P, LARGE)  # the fine-tune and the vocabulary read only ctx.net and ctx.P
    out = load(OUT)
    a = load(OUT_A)
    out |= {"git_head": head, "predictions_sha256": sha256(PRED), "bars": a["stage1"]["bars"], "amend2_sha256": sha256(OUT_A)}
    got = timed(0, stage0, J, ctx, out)
    seeds = P["settings"]["sharpen"]["seeds"]
    for k, m in ((1, "sharpen"), (2, "twin")):
        timed(k, lambda m=m: [state(J, ctx, m, s, got, out) for s in seeds])
    outcome(J, out)
    out["files"] = J.F.seen
    dump(OUT, out)
    hard_exit(0)


if __name__ == "__main__":
    main()
