"""open_pair Amendment 3 (E58, 2026-10-30, after job 46383): Amendment 2 with its control corrected. The no-rule map is
the cipher at 128 keys (Fable, reports/plans/2026-10-30_advice_e58a2-control-FABLE.md); elicit is a scored map; every
scored map is read at the common step (E12's rule on elicit, as filed) and at step 30; Fable's checks 1, 3, 4 and 5 are
reported lines. Plan: reports/plans/2026-10-30_plan_e58-open-pair.md, section "Amendment 3". Rules, settings and
thresholds: results/predictions_amend3.json (filed and committed before the run; the run refuses otherwise).

  K, W  amend2's key selection and maps (elicit, rank-1, copy), unchanged, into <large>/amend3; then the cipher
        (`kv.bijection`, E54's pool and gate) and the shuffled elicit map (check 3) on the same windows; zero update 0.
  T     the cipher and elicit fine-tunes (training only); the common step = elicit's E12 scored step (exit 1 if none).
  C     the control, read first and alone: the cipher's label at the common step (at its own E12 step only if too small
        there); exit 1 unless `open_pair.control3` reads "pass" or "below". The norm-matched cipher and its r reported.
  E     elicit at the common step and at step 30 (label, r, ratio, S); check 1 (x at every checkpoint, and E58's 24-key
        elicit_s0 re-trained, same seed); check 4 (elicit's update read on the rank-1 and copy held-out windows);
        check 5 (per-key H_spec against W0's P(value) and key-specificity, Spearman).
  S     check 3, the shuffled elicit map at the common step; exit 1 if its x > 0.05 (the label carries on the pool).
  M     rank-1, then copy, at the common step and at step 30 (exit 1 on a block-0 shift over 1e-5 nats).
  score `open_pair.score_amend3` -> results/amend3.json. Every number is also printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import amend2
import torch
from amend1 import git, load_split, x_novel
from amend2 import build_maps, stage_keys

from commute import anchors as AN
from commute import elicit as EL
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute.data import Windows
from commute.finetune import FinetuneResult, logits_of
from commute.heldout_keys import PARTS, behaviour, gain, say_label, split_task, vpd_read

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_amend3.json", RESULTS / "amend3.json"
E58_RUN = RESULTS / "run.json"
SCORED = ("elicit", "rank1", "copy")
log = KD.log
amend2.OUT = OUT  # amend2's dump and stop write this run's file, never amend2.json


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def add_maps(ctx: KD.Ctx, keys: list[int], d: Path, out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The cipher (E54's rule: kv.bijection from word-like ranks 1000-2999, seed 0, W0 P(value | M key) < 1e-4) and the
    shuffled elicit map (trained keys take the next trained key's elicit value in a seeded cycle; held-out keys keep
    theirs) on amend2's windows; cut by key; the zero update must read 0."""
    S, V = ctx.P["settings"], int(ctx.net.net.lm_head.weight.shape[0])
    src = torch.load(d / "windows_elicit.pt", weights_only=False)
    trained, held = OP.split_keys(keys, len(keys) // 4, S["split_seed"])
    tr, ho = src["train"], src["heldout"]
    both = Windows(torch.cat([tr.tokens, ho.tokens]), torch.cat([tr.g, ho.g]), torch.cat([tr.row, ho.row]), torch.cat([tr.pos, ho.pos]), tr.half)
    dist = kv.key_distributions(lambda x: logits_of(ctx.net.net, x), both, ctx.net.device)
    ranked = KD.vocab(ctx)[3]
    pool = ranked[AN.VALUE_RANK0 : AN.VALUE_RANK0 + AN.POOL]
    ev = dict(zip(src["task"].keys, src["task"].values, strict=True))
    cyc = [trained[int(i)] for i in torch.randperm(len(trained), generator=torch.Generator().manual_seed(S["shuffle_seed"]))]
    sh = {**ev, **{k: ev[cyc[(i + 1) % len(cyc)]] for i, k in enumerate(cyc)}}
    maps = {"cipher": kv.bijection("cipher", src["task"].marker, tuple(keys), pool, seed=0, ok=lambda k, v: float(dist[k][v]) < AN.GATE),
            "shuffle": kv.KVTask("shuffle", src["task"].marker, tuple(keys), tuple(sh[k] for k in keys))}
    tasks = {}
    for n, m in maps.items():
        torch.save({**src, "task": m, "table": m.table(V)}, d / f"windows_{n}.pt")
        val = dict(zip(m.keys, m.values, strict=True))
        sha = split_task(d / f"windows_{n}.pt", trained, held, d, n)
        tasks[n] = load_split(ctx, d, n)
        zero = {k: torch.zeros_like(x) for k, x in ctx.net.W0.items()}
        out["maps"][n] = {"sha256": sha, "values": [val[x] for x in keys], "n_distinct": len(set(m.values)), "n_fixed_points": sum(val[k] == ev[k] for k in trained),
                          "zero_max_abs": max(float(gain(ctx.net, tasks[n][p], zero).abs().max()) for p in PARTS),
                          "w0_geo_p": {p: KD.geo_rate(tasks[n][p].lp0) for p in PARTS}}
        log(f"[maps {n}] {json.dumps({k: v for k, v in out['maps'][n].items() if k != 'values'})}")
    dump(out)
    if any(out["maps"][n]["zero_max_abs"] != 0.0 for n in maps) or out["maps"]["shuffle"]["n_fixed_points"]:
        stop(out, "a zero update is not 0, or the shuffle left a trained key its own value")
    return tasks


def train(ctx: KD.Ctx, d: Path, n: str, checks: tuple[int, ...]) -> FinetuneResult:
    ft = KD.finetune(ctx, torch.load(d / f"windows_{n}_trained.pt", weights_only=False), ctx.P["settings"]["seed"], checks)
    log(f"[ft {n}] p_heldout {({k: round(e['p_o'], 4) for k, e in ft.evals.items()})} own E12 step {KD.scored_step(ft.evals)}")
    return ft


def read(ctx: KD.Ctx, name: str, t: dict[str, Any], ft: FinetuneResult, step: int, out: dict[str, Any], vpd: bool,
         save: Path | None = None) -> dict[str, Any]:
    """The label (and x_novel) of one checkpoint; with `vpd`, r, ratio and S beside it (block-0 stop)."""
    th, dW = ctx.P["thresholds"], ft.deltas[step]
    rec = {"step": step, "norm": EL.summed_norm(dW), **behaviour(ctx.net, t, dW, th)}
    rec["x_novel"] = x_novel(rec, t)
    if save is not None:
        torch.save(dW, save)
        rec["sha256"] = KD.sha256(save)
    say_label(f"{name} @ {step}", rec)
    if vpd:
        rec.update(vpd_read(ctx, t["trained"], dW, KD.base_of(ctx, t["trained"])))
        log(f"[read {name} @ {step}] r {rec['r']:.4f} ratio {rec['ratio']:.4f} S {rec['S']:.4f} block-0 {rec['S_block0']:.1e}")
        if abs(rec["S_block0"]) * abs(rec["G_score_logP"]) > th["block0_nats"]:
            stop(out, f"{name} @ {step}: block-0 shift over {th['block0_nats']} nats")
    out["reads"][f"{name}@{step}"] = rec
    dump(out)
    return rec


def control(ctx: KD.Ctx, tasks: dict[str, Any], ft_c: FinetuneResult, ft_e: FinetuneResult, step: int, out: dict[str, Any]) -> None:
    th = ctx.P["thresholds"]
    c = read(ctx, "cipher", tasks["cipher"], ft_c, step, out, vpd=False)
    own, learned = KD.scored_step(ft_c.evals)
    at_own = None
    if c["label"] == "too_small" and learned:
        o = read(ctx, "cipher", tasks["cipher"], ft_c, own, out, vpd=False)
        at_own = OP.elicit_control(o["x"], o["label"], th)
    status = out["control"] = OP.control3(OP.elicit_control(c["x"], c["label"], th), at_own)
    log(f"[control] cipher @ {step}: x {c['x']:.4f} label {c['label']}; own step {own} (read: {at_own is not None}) -> {status}")
    if status not in ("pass", "below"):
        stop(out, f"the cipher control reads {status}")
    m = EL.matched_step({k: EL.summed_norm(ft_c.deltas[k]) for k in range(1, 31)}, EL.summed_norm(ft_e.deltas[step]))
    read(ctx, "cipher", tasks["cipher"], ft_c, m, out, vpd=True)  # norm-matched: reported, with its r as the drift line
    out["cipher_matched_step"] = m


def elicit_checks(ctx: KD.Ctx, tasks: dict[str, Any], ft_e: FinetuneResult, step: int, qual: dict[int, Any], large: Path, out: dict[str, Any]) -> None:
    """Checks 1, 4 and 5 (reported)."""
    th, ch = ctx.P["thresholds"], out["checks"]
    xs = {s: (out["reads"].get(f"elicit@{s}") or read(ctx, "elicit", tasks["elicit"], ft_e, s, out, vpd=False))["x"] for s in KD.E12_CHECKS}
    e58 = load_split(ctx, large, "elicit")
    ft24 = train(ctx, large, "elicit", KD.E12_CHECKS)
    xs24 = {s: read(ctx, "elicit24", e58, ft24, s, out, vpd=False)["x"] for s in KD.E12_CHECKS}
    ch["1"] = {"x": xs, "x_24keys": xs24, "trajectory": OP.trajectory(xs, step, tuple(s for s in KD.E12_CHECKS if s > step), th),
               "x_24keys_step10_vs_E58": [xs24[10], json.loads(E58_RUN.read_text())["states"]["elicit_s0"]["x"]]}
    dW = ft_e.deltas[step]
    ch["4"] = {m: {k: v for k, v in behaviour(ctx.net, tasks[m], dW, th).items() if k in ("H", "H_spec", "x", "H_other")} for m in ("rank1", "copy")}
    per = out["reads"][f"elicit@{step}"]["H_spec_per_key"]
    held = [int(k) for k in per]
    hs = [per[str(k)] for k in held]
    ch["5"] = {"rho_p": OP.spearman(hs, [qual[k]["p"] for k in held]),
               "rho_specificity": OP.spearman(hs, [qual[k]["p"] / qual[k]["median_other"] for k in held])}
    log(f"[checks 1, 4, 5] {json.dumps(OP.jsonable(ch))}")
    dump(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="E58's split windows; this run writes to <large>/amend3")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P = json.loads(PRED.read_text())
    th = P["thresholds"]
    large = a.large.resolve()
    d = large / "amend3"
    d.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (HERE / "amend3.py", HERE / "amend2.py", HERE / "amend1.py", *(REPO / "src" / "commute" / f for f in ("open_pair.py", "heldout_keys.py", "keydep.py")))]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "reads": {}, "checks": {}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    rows, marker, keys, qual = stage_keys(ctx, out)
    tasks = {**build_maps(ctx, rows, marker, keys, qual, d, out), **add_maps(ctx, keys, d, out)}
    ft_c, ft_e = train(ctx, d, "cipher", KD.CIPHER_CHECKS), train(ctx, d, "elicit", KD.E12_CHECKS)
    step, learned = KD.scored_step(ft_e.evals)
    out["common_step"] = step
    if not learned:
        stop(out, "elicit reaches held-out P >= 0.5 at no E12 checkpoint")
    control(ctx, tasks, ft_c, ft_e, step, out)
    del ft_c
    st, st30 = {}, {}
    st["elicit"] = read(ctx, "elicit", tasks["elicit"], ft_e, step, out, vpd=True, save=d / f"elicit_s0_step{step}.pt")
    st30["elicit"] = st["elicit"] if step == 30 else read(ctx, "elicit", tasks["elicit"], ft_e, 30, out, vpd=True, save=d / "elicit_s0_step30.pt")
    a2 = json.loads((RESULTS / "amend2.json").read_text())["states"]["elicit"]
    out["checks"]["reproduce_46383"] = {"same_sha256": st["elicit"]["sha256"] == a2["ft"]["sha256"], "x": [st["elicit"]["x"], a2["x"]]}
    elicit_checks(ctx, tasks, ft_e, step, qual, large, out)
    sh = read(ctx, "shuffle", tasks["shuffle"], train(ctx, d, "shuffle", KD.E12_CHECKS), step, out, vpd=False)
    out["checks"]["3"] = OP.elicit_control(sh["x"], sh["label"], th)
    log(f"[check 3] shuffled elicit x {sh['x']:.4f} -> {out['checks']['3']}")
    if out["checks"]["3"] == "fail":
        stop(out, "the shuffled elicit map carries (x > 0.05): the label reads the pool at 128 values")
    for n in ("rank1", "copy"):
        ft = train(ctx, d, n, KD.E12_CHECKS)
        st[n] = read(ctx, n, tasks[n], ft, step, out, vpd=True, save=d / f"{n}_s0_step{step}.pt")
        st30[n] = st[n] if step == 30 else read(ctx, n, tasks[n], ft, 30, out, vpd=True, save=d / f"{n}_s0_step30.pt")
    out["score"] = OP.score_amend3(st, st30, out["control"], out["maps"]["rank1"]["n_novel"], P["reference_24_keys"], th)
    dump(out)
    log(f"[score] {json.dumps(OP.jsonable(out['score']))}")


if __name__ == "__main__":
    main()
