"""E96 stage 2: does a part of the 67M carry a fine-tune's gain on a task W0 already performs? E53's lowercase against its
cipher, step 300, batch orders 0 and 1; no training. Plan: reports/plans/2026-11-05_plan_e96-locate-mechanism.md, stage 1's
filed tree outcome "stage 2 by the sweep alone". Rules and expected values: results/predictions_stage2.json (committed
first; the run refuses otherwise; never edited), `commute.locate`.

  stage 0, alone: the states' sha256; each state's gain on E53's own bf16 read equals read.json's within E53's tolerance; the
    zero update reads 0; both precision paths reported, W0's own capability measured. Exit 1 on a miss.
  gradient check, on E62's copy_s0 and copy_s2: why head salience failed, told apart. Its true falls and single steps must
    re-read stage 1's sweep.json and nominators.json (the apparatus's positive control). Exit 1 on a miss. Nominates nothing.
  sweep, deciding: every head alone and each state's greedy pair, W0 and four states, W0's capability under each.
  beside, deciding nothing: head salience, MLP-unit salience at K = 320 against 20 random sets, the ridge probe.
  scorer, last: arithmetic over the saved files.

Writes results/stage2_{stage0,gradient,sweep,shares,beside,score}.json as each stage ends, and stage2_mlp_scores.pt; every
number is also printed. Stage 1's files are read, never written. Refuses if stage2_stage0.json exists: this run deletes nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run import Reads, dump, probe, salience, sweep_state, units_off
from torch import Tensor

from commute import copying as CP
from commute import locate as L
from commute import performed
from commute import steering as ST
from commute.data import pile_rows
from commute.finetune import logits_of
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256
from commute.open_pair import jsonable
from commute.revival_kv import zero_units

E53 = load_experiment("performed_anchor", "run")
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
E62 = HERE.parent / "induction_cue" / "results"
PRED = RESULTS / "predictions_stage2.json"
STAGE0, GRADIENT, SWEEP, SHARES, BESIDE, SCORE = (RESULTS / f"stage2_{f}.json" for f in ("stage0", "gradient", "sweep", "shares", "beside", "score"))
TASKS, SEEDS, STEP = ("lowercase", "cipher"), (0, 1), 300
LOW = [f"lowercase_s{s}" for s in SEEDS]


def say(tag: str, rec: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(rec))}")


# ----------------------------------------------------------------------------- stage 0


@torch.no_grad()
def bf16_read(R: Reads, w: Tensor, dW: ST.DW | None, batch: int) -> Tensor:
    """Per window, E53's own read: the log-odds of the true next token at `half` under bf16 autocast (`finetune.logits_of`,
    as `performed_anchor` `full_read`; `cells` agrees to the last digit, read.json's T0_full and T0)."""
    with R.net.at(dW):
        return torch.cat([ST.logodds_at(logits_of(R.net.net, w[s : s + batch, : R.n].to(R.net.device)), R.half, w[s : s + batch, -1]) for s in range(0, len(w), batch)])


def stage0(R: Reads, b: Tensor, th: dict[str, Any]) -> float:
    """Writes stage2_stage0.json whether or not it passes; a miss ends the job with exit 1. The sweep reads the fp32 path, as
    stage 1's control did, and E53's stored gain is the bf16 path's, so both are reported and the fp32 gain is the
    share's denominator. Returns W0's capability intact."""
    stored = json.loads((E53.RESULTS / "read.json").read_text())["rows"]
    batch = E53.CFG.batch
    zero = {k: torch.zeros_like(x).cpu() for k, x in R.net.W0.items()}
    rec: dict[str, Any] = {"states": {}, "zero": {}, "stored_file": "experiments/performed_anchor/results/read.json rows <task>/s<seed>/step300, G"}
    base = {m: bf16_read(R, w, None, batch) for m, w in R.W.items()}
    for name, dW in R.dws.items():
        m, s = name.split("_")
        G16, G32 = float((bf16_read(R, R.W[m], dW, batch) - base[m]).mean()), float((R.lo(R.W[m], dW) - R.lo(R.W[m])).mean())
        G_stored = stored[E53.key(m, int(s[1:]), STEP)]["G"]
        rec["states"][name] = {"G_bf16": G16, "G_stored": G_stored, "diff": abs(G16 - G_stored), "G_fp32": G32, "fp32_minus_bf16": G32 - G16}
    for m, w in R.W.items():
        rec["zero"][f"{m}_bf16"] = float((bf16_read(R, w, zero, batch) - base[m]).abs().max())
        rec["zero"][f"{m}_fp32"] = float((R.lo(w, zero) - R.lo(w)).abs().max())
    gap = float(L.form_gap(R.net, R.W["lowercase"], R.half, R.n, b).mean())
    rec["w0_capability"] = {"intact": gap, "reading_possible": gap > 0, "a0_reference_other_positions": json.loads(E53.A0.read_text())["lowercase"]["mean"]}
    rec["status"] = L.gain_apparatus(rec["states"], rec["zero"], th)
    dump(STAGE0, rec)
    say("stage0", rec)
    if rec["status"] != "pass":
        log(f"STOP: stage 0: {rec['status']}: tell the lead")
        hard_exit(1)
    return gap


# ----------------------------------------------------------------------------- the gradient check


def falls(R: Reads, w: Tensor, head: CP.Head, dW: ST.DW | None) -> dict[str, Any]:
    return {"true_fall": float((R.lo(w, dW) - R.lo(w, dW, [head])).mean()),
            "path_fall": -L.head_path(R.net, w, R.half, R.n, R.means, head, dW),
            "single_fall": -float(L.head_salience(R.net, w, R.half, R.n, R.means, dW)[head])}  # fmt: skip


def gradient_check(R: Reads, heads: list[list[int]], th: dict[str, Any]) -> None:
    """Each state and W0 and their difference, which is what salience ranks. The summary reads the difference's cases."""
    swept, noms = (json.loads(p.read_text()) for p in (RESULTS / "sweep.json", RESULTS / "nominators.json"))
    cases: dict[str, Any] = {}
    for name, dW in R.dws.items():
        for a, c in heads:
            head: CP.Head = (a, c)
            w, h = R.W["copy"], CP.head_name(head)
            state, w0 = falls(R, w, head, dW), falls(R, w, head, None)
            net: dict[str, Any] = {k: state[k] - w0[k] for k in state}
            ran = swept["states"][name]
            case = {"state": state, "W0": w0, "net": net, "stage1_net_fall": ran["G"] - ran["heads"][h]["G_abl"], "stage1_salience": noms["salience"][name][h]}
            for rec in (state, w0, net):
                rec["outcome"] = L.path_outcome(rec["true_fall"], rec["path_fall"], rec["single_fall"], th["path_tol"])
            cases[f"{name} {h}"] = case
            say(f"gradient {name} {h}", case)
    bad = [k for k, c in cases.items() if not (abs(c["net"]["true_fall"] - c["stage1_net_fall"]) <= th["reread_tol"] and abs(c["net"]["single_fall"] - c["stage1_salience"]) <= th["reread_tol"])]
    rec = {"cases": cases, "outcome": L.path_summary([c["net"]["outcome"] for c in cases.values()]), "does_not_reread_stage1": bad}
    dump(GRADIENT, rec)
    log(f"[gradient outcome] {rec['outcome']}")
    if bad:
        log(f"STOP: gradient check: {bad} do not re-read stage 1: tell the lead")
        hard_exit(1)


# ----------------------------------------------------------------------------- the sweep


def share_se(R: Reads, name: str, heads: list[CP.Head]) -> float:
    """Standard error over windows of the share, the noise floor of a share at this gain. nan with no gain."""
    m, dW = name.split("_")[0], R.dws[name]
    G = float((R.intact(name) - R.w0_off(m, [])).mean())
    per = (R.lo(R.W[m], dW, heads) - R.w0_off(m, heads)) / G if G > 0 else torch.full((1,), math.nan)
    return float(per.std() / math.sqrt(len(per)))


def as_heads(pair: str) -> list[CP.Head]:
    return [(int(a), int(b)) for a, b in (x.split(",") for x in pair.split("+"))]


def same_pair(pair: str) -> str:
    """One name for a pair, whichever head the greedy step took first."""
    return "+".join(sorted(pair.split("+")))


def sweep(R: Reads, b: Tensor, gap: float) -> None:
    w, gap_off = R.W["lowercase"], {}
    for h in R.heads:
        with CP.ablated(R.net, [h], R.means):
            gap_off[CP.head_name(h)] = float(L.form_gap(R.net, w, R.half, R.n, b).mean())
    rec: dict[str, Any] = {"w0": {"lo": {m: float(R.w0_off(m, []).mean()) for m in R.W}, "gap": gap, "gap_off": gap_off,
                                  "lo_off": {m: {CP.head_name(h): float(R.w0_off(m, [h]).mean()) for h in R.heads} for m in R.W}},
                           "states": {}, "pairs": {}}  # fmt: skip
    say("sweep W0", rec["w0"])
    dump(SWEEP, rec)
    for name in R.dws:
        st = sweep_state(R, name)
        if "best_pair" in st:
            top = st["best_pair"].split("+")[0]
            st["top_share_se"] = share_se(R, name, as_heads(top))
        rec["states"][name] = st
        say(f"sweep {name}", st)
        dump(SWEEP, rec)
    for pair in sorted({same_pair(r["best_pair"]) for r in rec["states"].values() if "best_pair" in r}):
        hs = as_heads(pair)
        with CP.ablated(R.net, hs, R.means):
            pair_gap = float(L.form_gap(R.net, w, R.half, R.n, b).mean())
        rec["pairs"][pair] = {"gap_off": pair_gap, "lo_off": {m: float(R.w0_off(m, hs).mean()) for m in R.W}, "states": {n: R.part(n, hs) for n in R.dws}}
        say(f"sweep pair {pair}", rec["pairs"][pair])
        dump(SWEEP, rec)
    small = {"heads": {CP.head_name(h): {n: rec["states"][n]["heads"][CP.head_name(h)]["share"] for n in R.dws} for h in R.heads},
             "pairs": {p: {n: r["states"][n]["share"] for n in R.dws} for p, r in rec["pairs"].items()},
             "G": {n: rec["states"][n]["G"] for n in R.dws}, "top_share_se": {n: rec["states"][n].get("top_share_se") for n in R.dws}}  # fmt: skip
    dump(SHARES, small)
    say("shares", small)


# ----------------------------------------------------------------------------- beside, deciding nothing


def mlp_units(R: Reads, b: Tensor, C: dict[str, Any]) -> dict[str, Any]:
    """Stage 1's reads (`run.units_off`, the same draws), lowercase's states only, with the same seed's cipher as control."""
    K, total = C["mlp_k"], ST.N_LAYERS * ST.D_MLP
    rng = np.random.default_rng([C["random_seed"], K])
    draws = [sorted(rng.choice(total, K, replace=False).tolist()) for _ in range(C["n_random"])]
    cache: dict[tuple[str, tuple[int, ...]], Tensor] = {}
    w, odd = R.W["lowercase"], b[1::2]
    gap_intact = float(L.form_gap(R.net, w[1::2], R.half, R.n, odd).mean())
    out: dict[str, Any] = {}
    scores = {}
    for name in LOW:
        scores[name] = L.unit_scores(R.net, w[0::2], R.half, R.n, R.dws[name])
        top = sorted(torch.topk(scores[name], K).indices.tolist())
        out[name] = {**units_off(R, name, top, cache), "units": top, "control": units_off(R, f"cipher_{name.split('_')[1]}", top, cache),
                     "random_shares": [units_off(R, name, d, cache)["share"] for d in draws], "gap_intact": gap_intact,
                     "gap_off": float(L.form_gap(R.net, w[1::2], R.half, R.n, odd, zero_units(top)).mean())}  # fmt: skip
        say(f"mlp {name}", {k: v for k, v in out[name].items() if k != "units"})
    torch.save(torch.stack([scores[n] for n in LOW]).float(), RESULTS / "stage2_mlp_scores.pt")
    return out


def beside(R: Reads, b: Tensor, C: dict[str, Any]) -> None:
    low = dataclasses.replace(R, W={"lowercase": R.W["lowercase"]}, dws={n: R.dws[n] for n in LOW}, off0={})
    rec: dict[str, Any] = {"salience": salience(low), "probe": probe(low)}
    say("beside salience", rec["salience"])
    say("beside probe", rec["probe"])
    dump(BESIDE, rec)
    rec["mlp"] = mlp_units(R, b, C)
    dump(BESIDE, rec)


# ----------------------------------------------------------------------------- the scorer


def scorer(th: dict[str, Any]) -> None:
    swept, near = (L.from_saved(json.loads(p.read_text())) for p in (SWEEP, BESIDE))
    rec = L.score_stage2(swept, near, th)
    dump(SCORE, rec)
    say("score", rec)
    log(f"[outcome] {rec['outcome']}")
    for claim in rec["claims"]:
        log(f"[claim] {claim}")


# ----------------------------------------------------------------------------- load and main


def committed(path: Path) -> bool:
    rel = str(path.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    return path.is_file() and bool(git("ls-files", rel).stdout.strip()) and not git("diff", "--quiet", "HEAD", "--", rel).returncode


def e53_windows(name: str, pi: dict[int, int]) -> tuple[Tensor, Tensor]:
    """E53's scored windows of `name` (`performed_anchor` `Run.scored`'s draw), the true next token as a last column, and
    each window's other form (the capitalised form on lowercase, x on the cipher)."""
    C = E53.CFG
    text, at = E53.task_text(name, pile_rows("val", 0, 256), pi)
    w, idx = performed.windows_at(text, at, C.half, C.n_windows, C.window_seed)
    assert torch.equal(w[:, C.half + 1], at.a[idx]), "a window's next token is not its scored form"
    return torch.cat([w, w[:, C.half + 1 : C.half + 2]], 1), at.b[idx]


def load(a: argparse.Namespace, S: dict[str, Any]) -> tuple[Reads, Reads, Tensor]:
    """The reads on E53's windows and on E62's copy windows, on one model and stage 1's means. Every file checked first."""
    means_file = RESULTS / "means.pt"
    if sha256(means_file) != S["means_sha256"]:
        sys.exit("STOP: stage 0: means.pt is not stage 1's: tell the lead")
    run62 = json.loads((E62 / "run.json").read_text())
    large = a.large.resolve()
    for f, sha in {"windows": run62["windows_sha256"], **{n: run62["states"][n]["sha256"] for n in S["gradient_states"]}}.items():
        if sha256(large / f"{f}.pt") != sha:
            sys.exit(f"STOP: stage 0: {f}.pt is not the one run.json recorded: tell the lead")
    state = E53.state_loader(json.loads((E53.RESULTS / "finetune.json").read_text()), [E53.key(t, s, STEP) for t in TASKS for s in SEEDS])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as stage 1
    net = ST.Net(a.device)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {git} predictions {sha256(PRED)}")
    means = torch.load(means_file, weights_only=True, map_location="cpu").to(net.device)
    low, b = e53_windows("lowercase", {})
    cip, _ = e53_windows("cipher", E53.e53_pi())
    dws = {f"{t}_s{s}": state(E53.key(t, s, STEP)) for t in TASKS for s in SEEDS}
    v = torch.load(large / "windows.pt", weights_only=False)
    half62 = json.loads((E62 / "predictions.json").read_text())["settings"]["half"]
    copy = CP.with_target(v["read"], v["target"]["copy"]["read"]).tokens
    dw62 = {n: torch.load(large / f"{n}.pt", weights_only=True, map_location="cpu") for n in S["gradient_states"]}
    return Reads(net, E53.CFG.half, means, {"lowercase": low, "cipher": cip}, dws, {}), Reads(net, half62, means, {"copy": copy}, dw62, {}), b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--large", type=Path, default=E62 / "large")
    a = ap.parse_args()
    if not committed(PRED):
        sys.exit(f"{PRED} is not committed unchanged: file and commit it before the run")
    if STAGE0.exists():
        sys.exit(f"{STAGE0} exists: move the results aside, this run deletes nothing")
    P = json.loads(PRED.read_text())
    th, S = P["thresholds"], P["settings"]
    R, R62, b = load(a, S)
    gap = stage0(R, b, th)
    gradient_check(R62, S["gradient_heads"], th)
    sweep(R, b, gap)
    beside(R, b, S)
    scorer(th)
    hard_exit(0)


if __name__ == "__main__":
    main()
