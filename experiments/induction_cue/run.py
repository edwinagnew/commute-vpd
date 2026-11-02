"""induction_cue (E62), the copying-heads test. The 67M copies in context (after "A B ... A" it says B). Teach it to copy at
a new marker M ("A B ... A M" -> B) beside a lookup at the same marker and windows ("A M" -> a fixed word of A). Does the
copy map run through the model's existing copying heads, does the lookup not, and does the route split's r follow that
label? Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md. Every setting and rule: results/predictions.json (filed
and committed first; the run refuses otherwise; never edited). Rules: `commute.copying`.

  (a) W0 alone: prefix-matching score of the 24 heads on repeated random tokens; H = the heads over the bar; W0's
      second-copy top-1 (E52's control) falls under H's mean ablation and holds under random heads'. Exit 1 on a miss.
  (b) W0's prior for B at the marker; the lookup's words drawn, per key, from every word-like token outside the keys, Bs and
      marker and outside that key's own windows, so that their prior is within `prior_match_max` of B's
      (`copying.pick_words`; admissible counts per key printed first, Amendment 1); the zero update reads 0; then the data.LAST path equals the
      lookup-table path on E54's saved elicit task (fine-tune and r), the fine-tune's tolerance a factor times the old
      path's run-to-run difference. Exit 1 on a prior over the stop bar, no matched word set or any miss.
  (c) Lookup then copy map, seeds 0-2, E58's recipe (`keydep.finetune`); each pair read at `copying.read_step`, chosen on
      the specific gains (printed with the raw, B-absent and pair-broken gains at every kept step); states saved; the 24
      heads' prefix scores under each saved state are printed and stored (reported, not a rule).
  (d) The label: share of each state's gain lost under H, net of W0's change; the random-head floor; the two
      cheap-solution controls (copy) and the specific share (lookup).
  (e) r on every state with `local_faith.route_split`, then `copying.score`.
Results go to results/run.json after every stage; every scored number is also printed. <large>/: windows.pt, heads.pt,
{copy,lookup}_s{0,1,2}.pt (113 MB each).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import elicit as EL
from commute import keydep as KD
from commute import steering as ST
from commute.data import LAST
from commute.heldout_keys import others
from commute.local_faith import route_cells
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions.json", RESULTS / "run.json"
E54 = HERE.parent / "keydependent_elicit" / "results" / "large"
MAPS = ("lookup", "copy")  # run order: the map that should not use the heads first
log = KD.log


@dataclass(frozen=True)
class Cfg:
    half: int
    seeds: list[int]
    windows: dict[str, Any]  # `copying.build_cue`'s sizes and seeds
    score: dict[str, Any]  # the prefix-matching rows
    induction: dict[str, Any]  # E52's control
    control_seed: int
    n_random: int
    random_seed: int
    mean_spans: int
    mean_seed: int
    read_steps: list[int]
    n_boot: int

    def __post_init__(self) -> None:
        w = self.windows
        assert self.half == AN.HALF and set(self.read_steps) <= set(KD.E12_CHECKS), "predictions.json disagrees with the code"
        assert w["pool_ranks"] == [AN.VALUE_RANK0, AN.VALUE_RANK0 + AN.POOL] and w["n_keys"] < AN.POOL, "predictions.json disagrees with the code"

    @property
    def n(self) -> int:
        return 2 * self.half + 1


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


def stop(out: dict[str, Any], why: str) -> NoReturn:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


def build(ctx: KD.Ctx, c: Cfg, out: dict[str, Any]) -> CP.Cue:
    rows, _, marker, ranked, tok = KD.vocab(ctx)
    cue = CP.build_cue(rows, marker, ranked, lambda t: tok.encode(tok.decode([t])) == [t], c.half, c.windows)
    out["pools"] = {**cue.info, "marker": [marker, tok.decode([marker])]}
    log(f"[pools] {json.dumps(out['pools'])}")
    save_windows(ctx, cue, out)
    return cue


def save_windows(ctx: KD.Ctx, cue: CP.Cue, out: dict[str, Any]) -> None:
    """windows.pt again once stage (b) has fixed the lookup's words (the targets of the lookup are in it)."""
    torch.save({"train": cue.train, "read": cue.read, "target": cue.target, "p_read": cue.p_read, "pools": out["pools"]}, ctx.large / "windows.pt")
    out["windows_sha256"] = KD.sha256(ctx.large / "windows.pt")


# ----------------------------------------------------------------------------- (a)


def score_rows(c: Cfg, cue: CP.Cue) -> Tensor:
    """The prefix-matching score's rows (repeated random tokens), the same in stage (a) and under every fine-tuned state."""
    S = c.score
    pool = cue.ranked[S["rank_from"] : int(len(cue.ranked) * (1 - S["rank_drop_last"]))]
    return CP.repeated_random(pool, S["length"], S["repeats"], S["sequences"], S["sos"], S["seed"])


def stage_a(ctx: KD.Ctx, c: Cfg, cue: CP.Cue, th: dict[str, Any], out: dict[str, Any]) -> tuple[list[CP.Head], list[list[CP.Head]], Tensor]:
    net = ctx.net
    toks = score_rows(c, cue)
    pats, err = CP.attention_patterns(net, toks)
    if err > th["pattern_tol"]:
        stop(out, f"the recomputed attention pattern is off o_proj's input by {err}: cannot read the heads")
    scores = torch.stack([CP.stripe_score(p, CP.stripe_mask(toks)) for p in pats]).cpu()
    literal = torch.stack([CP.stripe_score(p, CP.stripe_mask(toks, -1)) for p in pats]).cpu()
    chance = float(CP.stripe_score(CP.uniform_attention(toks.shape[1]), CP.stripe_mask(toks))[0])
    bar = th["bar_multiple"] * chance
    H = CP.over_bar(scores, bar)
    spans = CP.natural_spans(cue.mean_rows, c.half, c.mean_spans, c.mean_seed)
    prev = torch.stack([CP.previous_token(p) for p in CP.attention_patterns(net, spans[:64])[0]]).cpu()
    means = CP.head_means(net, spans)
    torch.save({"scores": scores, "means": means.cpu(), "H": H}, ctx.large / "heads.pt")
    base = CP.induction(net, c.induction)["top1_second"]
    curve = []
    ranked = CP.by_score(scores)
    for k in range(1, min(len(ranked), 12) + 1):  # reported: what an amended bar would need
        with CP.ablated(net, ranked[:k], means):
            curve.append(CP.induction(net, c.induction)["top1_second"])
    rec: dict[str, Any] = {"pattern_err": err, "scores": scores.tolist(), "literal_offset_scores": literal.tolist(), "previous_token": prev.tolist(), "chance": chance, "bar": bar, "H": H,
                           "top1_base": base, "by_score_order": ranked[:12], "curve_top1_by_k": curve, "status_heads": CP.heads_status(H, scores.numel())}
    out["stage_a"] = rec
    log(f"[a] chance {chance:.4f} bar {bar:.4f} H {H} pattern err {err:.2e} top-1 base {base:.4f} (E52 {th['top1_ref']:.4f}) status {rec['status_heads']}")
    log(f"[a] scores by block {scores.tolist()}")
    log(f"[a] literal-offset scores {literal.tolist()} previous-token {prev.tolist()}")
    log(f"[a] top-1 with the top k heads ablated {curve}")
    dump(out)
    if rec["status_heads"] != "ok":
        stop(out, f"stage (a): the set of heads over the bar is {rec['status_heads']} ({len(H)} of {scores.numel()})")
    with CP.ablated(net, H, means):
        rec["top1_H"] = CP.induction(net, c.induction)["top1_second"]
    draws = CP.random_others(H, scores.shape[0], scores.shape[1], c.n_random, c.random_seed)
    rec["random_draws"], rec["top1_random"] = draws, []
    for d in draws:
        with CP.ablated(net, d, means):
            rec["top1_random"].append(CP.induction(net, c.induction)["top1_second"])
    rec["status"] = CP.control_a(base, rec["top1_H"], rec["top1_random"], th)
    log(f"[a] top-1 under H {rec['top1_H']:.4f} (bar {th['top1_H_max']}) under random heads {rec['top1_random']} (median bar {th['top1_random_min']}) draws {draws} -> {rec['status']}")
    dump(out)
    if rec["status"] != "pass":
        stop(out, f"stage (a): {rec['status']}")
    return H, draws, means


def prefix_scores_under(net: ST.Net, toks: Tensor, dW: ST.DW) -> Tensor:
    """(blocks, heads): stage (a)'s prefix-matching score of every head with the update `dW` added to W0."""
    with net.at(dW):
        pats, _ = CP.attention_patterns(net, toks)
    mask = CP.stripe_mask(toks)
    return torch.stack([CP.stripe_score(p, mask) for p in pats]).cpu()


# ----------------------------------------------------------------------------- (b)


def stage_b(ctx: KD.Ctx, c: Cfg, cue: CP.Cue, H: list[CP.Head], means: Tensor, th: dict[str, Any], out: dict[str, Any]) -> str:
    """W0's prior for B, then the lookup's words drawn to match it (the prior is measured here, so the words are too),
    then the zero update. Stops: B's prior over prior_stop; no word set within prior_match_max of it; a zero update not 0."""
    net, n = ctx.net, c.n
    zero = {k: torch.zeros_like(x) for k, x in net.W0.items()}
    geo = lambda w: KD.geo_rate(CP.read_lo(net, w, c.half, n, readout=ST.logp_at))
    ctl = cue.controls(c.half, c.control_seed)
    rec: dict[str, Any] = {"prior_copy": geo(cue.read_windows("copy")), "prior_b_absent": geo(ctl["absent"]), "prior_pair_broken": geo(ctl["broken"]), "zero_max_abs": {}}
    rec["band"] = CP.prior_band(rec["prior_copy"], th)
    out["stage_b"] = rec
    log(f"[b] W0 prior: B at the marker {rec['prior_copy']:.3e} -> {rec['band']} | B absent {rec['prior_b_absent']:.3e} | pair broken {rec['prior_pair_broken']:.3e}")
    dump(out)
    if rec["band"] == "stop":
        stop(out, f"stage (b): W0's prior for B {rec['prior_copy']} is over {th['prior_stop']}")
    lp = CP.word_logp(net, cue.read.tokens, c.half, n, cue.word_pool)
    at, target_lp = cue.read.tokens[:, c.half - 1], math.log(rec["prior_copy"])
    own = CP.own_tokens(cue.keys, torch.cat([cue.train.tokens, cue.read.tokens]), c.half)
    adm = CP.admissible(cue.keys, cue.word_pool, at, lp, target_lp, th["prior_match_max"], own)
    rec["n_candidates"], rec["admissible_per_key"] = len(cue.word_pool), {str(k): len(v) for k, v in adm.items()}
    rec["candidate_log10_prior_histogram"] = CP.prior_histogram(lp)
    log(f"[b] candidates {len(cue.word_pool)}; admissible per key (min {min(rec['admissible_per_key'].values())}) {json.dumps(rec['admissible_per_key'])}")
    log(f"[b] candidates by mean log10 P over the read windows (bin lower edge: count) {json.dumps(rec['candidate_log10_prior_histogram'])}")
    dump(out)
    try:
        words = CP.pick_words(cue.keys, cue.word_pool, at, lp, target_lp, th["prior_match_max"], c.windows["pool_seed"], own)
    except AssertionError as e:
        stop(out, f"stage (b): no word set from the {len(cue.word_pool)} candidates puts the lookup's prior within a factor {th['prior_match_max']} of B's {rec['prior_copy']:.3e} ({e})")
    cue.set_words(words)
    out["pools"]["words"] = cue.info["words"]
    save_windows(ctx, cue, out)
    rec["prior_lookup"] = geo(cue.read_windows("lookup"))
    rec["prior_ratio"] = CP.prior_ratio(rec["prior_copy"], rec["prior_lookup"])
    log(f"[b] priors: B {rec['prior_copy']:.3e} lookup word {rec['prior_lookup']:.3e} ratio {rec['prior_ratio']:.3f} (bar {th['prior_match_max']})")
    dump(out)
    if not rec["prior_ratio"] <= th["prior_match_max"]:
        stop(out, f"stage (b): the lookup's prior {rec['prior_lookup']:.3e} is not within a factor {th['prior_match_max']} of B's {rec['prior_copy']:.3e}")
    for m in MAPS:
        w = cue.read_windows(m)
        with CP.ablated(net, H, means):
            ab = float((CP.read_lo(net, w, c.half, n, zero) - CP.read_lo(net, w, c.half, n)).abs().max())
        rec["zero_max_abs"][m] = max(ab, float((CP.read_lo(net, w, c.half, n, zero) - CP.read_lo(net, w, c.half, n)).abs().max()))
    log(f"[b] zero update {rec['zero_max_abs']}")
    dump(out)
    if any(v != 0.0 for v in rec["zero_max_abs"].values()):
        stop(out, "stage (b): a zero update does not read 0")
    return rec["band"]


def path_control(ctx: KD.Ctx, c: Cfg, th: dict[str, Any], out: dict[str, Any]) -> None:
    """E54's elicit task through both target forms: the lookup table against data.LAST, a fine-tune and the route split."""
    for f, sha in th["e54_sha256"].items():
        if KD.sha256(E54 / f"{f}.pt") != sha:
            stop(out, f"E54's {f} is not the one its run recorded")
    w = torch.load(E54 / "windows_elicit.pt", weights_only=False)
    t = ST.load_task(ctx.net, E54 / "windows_elicit.pt", "elicit", "elicit")
    col = lambda x: CP.with_target(x, w["table"][x.tokens[:, c.half]])
    new = {"table": LAST, "train": col(w["train"]), "heldout": col(w["heldout"]), "ci_len": c.n}
    ft_old, ft_new, ft_rep = KD.finetune(ctx, w, 0, (30,)), KD.finetune(ctx, new, 0, (30,)), KD.finetune(ctx, w, 0, (30,))
    gap = lambda a, b: max(float((a.deltas[30][k] - b.deltas[30][k]).abs().max()) for k in a.deltas[30])
    ft_diff, ft_repeat = gap(ft_old, ft_new), gap(ft_old, ft_rep)  # the old path twice is the measured floor of the tolerance
    dW = torch.load(E54 / "elicit_s0.pt", weights_only=True)
    base_old, base_rep = KD.base_of(ctx, t), KD.base_of(ctx, t)
    toks = col(w["heldout"]).tokens
    base_new = route_cells(ctx.net.model, toks, c.half, LAST, KD.BATCH, c.n)
    old, rep = KD.route_r(ctx, t, dW, base_old), KD.route_r(ctx, t, dW, base_rep)
    now = KD.route_windows(ctx, toks, c.half, LAST, dW, base_new, c.n)
    rec = {"ft_max_abs_diff": ft_diff, "ft_repeat_max_abs_diff": ft_repeat, "ft_tol": th["ft_tol_factor"] * ft_repeat, "r_old": old["r"], "r_repeat": rep["r"], "r_new": now["r"], "r_ref": th["r_ref"],
           "cell_diff_new": max(abs(old["cells"][k] - now["cells"][k]) for k in old["cells"]), "cell_diff_repeat": max(abs(old["cells"][k] - rep["cells"][k]) for k in old["cells"])}
    rec["pass"] = bool(ft_diff <= rec["ft_tol"] and rec["cell_diff_new"] <= th["path_tol"] and abs(old["r"] - th["r_ref"]) <= th["r_tol"])
    out["path_control"] = rec
    log(f"[path] {json.dumps(rec)}")
    dump(out)
    if not rec["pass"]:
        stop(out, "the per-window target path differs from the lookup-table path")


# ----------------------------------------------------------------------------- (c), (d), (e)


def gain_row(ctx: KD.Ctx, c: Cfg, cue: CP.Cue, m: str, dW: ST.DW, ref: dict[str, Any]) -> dict[str, float]:
    """The gains (mean log-odds change on the read windows) at one kept step. Copy map: raw, with B absent, with the pair
    broken, and specific = raw - with B absent. Lookup: raw, the mean over the other words, and specific = raw - others
    (as E58 defines it, `heldout_keys.specific`). The read step is chosen on the specific gain."""
    mean = lambda w, base, ro=ST.logodds_at: float((CP.read_lo(ctx.net, w, c.half, c.n, dW, ro) - base).mean())
    w = cue.read_windows(m)
    raw = mean(w, ref["base"][m])
    if m == "copy":
        absent, broken = (mean(ref["ctl"][k], ref["base_ctl"][k]) for k in ("absent", "broken"))
        return {"raw": raw, "absent": absent, "broken": broken, "specific": raw - absent}
    others_gain = mean(w, ref["base_other"], ref["other"])
    return {"raw": raw, "others": others_gain, "specific": raw - others_gain}


def stage_c(ctx: KD.Ctx, c: Cfg, cue: CP.Cue, th: dict[str, Any], out: dict[str, Any]) -> dict[str, ST.DW]:
    dws: dict[str, ST.DW] = {}
    toks = score_rows(c, cue)
    ctl = cue.controls(c.half, c.control_seed)
    other = others(list(cue.words.values()))
    ref = {"base": {m: CP.read_lo(ctx.net, cue.read_windows(m), c.half, c.n) for m in MAPS}, "ctl": ctl,
           "base_ctl": {k: CP.read_lo(ctx.net, v, c.half, c.n) for k, v in ctl.items()}, "other": other,
           "base_other": CP.read_lo(ctx.net, cue.read_windows("lookup"), c.half, c.n, None, other)}
    for s in c.seeds:
        fts, gains = {}, {}
        for m in MAPS:
            w = {"table": LAST, "train": CP.with_target(cue.train, cue.target[m]["train"]), "heldout": CP.with_target(cue.read, cue.target[m]["read"]), "ci_len": c.n}
            fts[m] = KD.finetune(ctx, w, s, KD.E12_CHECKS)
            gains[m] = {st: gain_row(ctx, c, cue, m, fts[m].deltas[st], ref) for st in c.read_steps}
            log(f"[c {m}_s{s}] gains by step {json.dumps(gains[m])} (gate {th['G_min']} on the specific gain) p_read {({k: round(e['p_o'], 4) for k, e in fts[m].evals.items()})}")
        step = CP.read_step({st: g["specific"] for st, g in gains["copy"].items()}, {st: g["specific"] for st, g in gains["lookup"].items()}, th["G_min"])
        for m in MAPS:
            n, dW = f"{m}_s{s}", fts[m].deltas[step]
            dws[n] = dW
            torch.save(dW, ctx.large / f"{n}.pt")
            out["states"][n] = {"step": step, "gains_by_step": gains[m], "p_read": fts[m].evals[step]["p_o"], "kl_other": fts[m].evals[step]["kl_other"], "norm": EL.summed_norm(dW), "sha256": KD.sha256(ctx.large / f"{n}.pt")}
            log(f"[c {n}] read at step {step} ||dW|| {out['states'][n]['norm']:.4f}")
            prefix = prefix_scores_under(ctx.net, toks, dW)  # reported, not a rule: tells recruiting new heads from a wrong label
            out["states"][n]["prefix_scores"] = prefix.tolist()
            log(f"[c {n}] prefix scores under the state (W0's in stage a; bar {out['stage_a']['bar']:.4f}) {json.dumps(prefix.tolist())}")
        dump(out)
    return dws


def stage_d(ctx: KD.Ctx, c: Cfg, cue: CP.Cue, dws: dict[str, ST.DW], H: list[CP.Head], draws: list[list[CP.Head]], means: Tensor, th: dict[str, Any], out: dict[str, Any]) -> None:
    net, n = ctx.net, c.n
    lo = lambda w, dW=None, ro=ST.logodds_at: CP.read_lo(net, w, c.half, n, dW, ro)

    def lo_abl(w: Tensor, heads: list[CP.Head], dW: ST.DW | None = None) -> Tensor:
        with CP.ablated(net, heads, means):
            return lo(w, dW)

    W = {m: cue.read_windows(m) for m in MAPS}
    ctl = cue.controls(c.half, c.control_seed)
    lo0 = {m: lo(W[m]) for m in MAPS}
    lo0_H = {m: lo_abl(W[m], H) for m in MAPS}
    lo0_r = {m: [lo_abl(W[m], d) for d in draws] for m in MAPS}
    lo0_ctl = {k: lo(v) for k, v in ctl.items()}
    other = others(list(cue.words.values()))
    other0 = lo(W["lookup"], None, other)
    for name, dW in dws.items():
        m = name.split("_")[0]
        d, d_abl = lo(W[m], dW) - lo0[m], lo_abl(W[m], H, dW) - lo0_H[m]
        G, G_abl = float(d.mean()), float(d_abl.mean())
        rand = [CP.share_lost(G, float((lo_abl(W[m], dr, dW) - lo0_r[m][i]).mean())) for i, dr in enumerate(draws)]
        rec = {"G": G, "G_abl": G_abl, "share": CP.share_lost(G, G_abl), "share_sd": CP.share_sd(d, d_abl, c.n_boot, 0), "label": CP.share_label(G, G_abl, th),
               "random_shares": rand, "random_median": statistics.median(rand)}
        if m == "copy":
            rec.update({k + "_ratio": float((lo(v, dW) - lo0_ctl[k]).mean()) / G if G > 0 else math.nan for k, v in ctl.items()})
        else:
            rec["spec_frac"] = (G - float((lo(W[m], dW, other) - other0).mean())) / G if G > 0 else math.nan
        out["states"][name].update(rec)
        log(f"[d {name}] {json.dumps(jsonable(rec))}")
        dump(out)


def stage_e(ctx: KD.Ctx, c: Cfg, cue: CP.Cue, dws: dict[str, ST.DW], th: dict[str, Any], out: dict[str, Any]) -> None:
    base = {m: route_cells(ctx.net.model, cue.read_windows(m), c.half, LAST, KD.BATCH, c.n) for m in MAPS}
    zero = {k: torch.zeros_like(x).cpu() for k, x in ctx.net.W0.items()}
    z = KD.route_windows(ctx, cue.read_windows("copy"), c.half, LAST, zero, base["copy"], c.n)
    out["zero_route"] = {"gain": z["G_logodds"], "L10_minus_L00": z["cells"]["L10"] - z["cells"]["L00"], "L01_minus_L00": z["cells"]["L01"] - z["cells"]["L00"]}
    log(f"[e zero update] {json.dumps(out['zero_route'])}")
    dump(out)
    if abs(z["G_logodds"]) > th["zero_route_tol"]:
        stop(out, "stage (e): the zero update's route gain is not 0")
    for name, dW in dws.items():
        rd = KD.route_windows(ctx, cue.read_windows(name.split("_")[0]), c.half, LAST, dW, base[name.split("_")[0]], c.n)
        out["states"][name].update({"r": rd["r"], "G_route": rd["G_logodds"], "cells": rd["cells"]})
        log(f"[e {name}] r {rd['r']:.4f} gain {rd['G_logodds']:.4f} cells {json.dumps(rd['cells'])}")
        dump(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P = json.loads(PRED.read_text())
    c, th = Cfg(**P["settings"]), P["thresholds"]
    large = a.large.resolve()
    large.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (HERE / "run.py", *(REPO / "src" / "commute" / f for f in ("copying.py", "keydep.py", "finetune.py", "data.py", "local_faith.py")))]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code), "predictions_sha256": KD.sha256(PRED), "states": {}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    out["host"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    log(f"[host] {out['host']}")
    cue = build(ctx, c, out)
    H, draws, means = stage_a(ctx, c, cue, th, out)
    band = stage_b(ctx, c, cue, H, means, th, out)
    path_control(ctx, c, th, out)
    dws = stage_c(ctx, c, cue, th, out)
    stage_d(ctx, c, cue, dws, H, draws, means, th, out)
    stage_e(ctx, c, cue, dws, th, out)
    out["score"] = CP.score(out["states"], band, th, c.seeds, out["stage_b"]["prior_ratio"])
    dump(out)
    log(f"[score] {json.dumps(jsonable(out['score']))}")
    log(f"[prior beside r] B {out['stage_b']['prior_copy']:.3e} word {out['stage_b']['prior_lookup']:.3e}; {({n: (round(s['r'], 4), s['label'], round(s['norm'], 3)) for n, s in out['states'].items()})}")


if __name__ == "__main__":
    main()
