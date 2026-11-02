"""open_pair (E58): E54's elicit map and its matched cipher fine-tuned on 24 of their 32 keys (E54's recipe, seeds 0-2),
each state labelled by behaviour alone (does the fine-tune carry to the 8 keys it never saw?) with the reuse line's
bounded-g rule, then read by the route split's r, the stand-in ratio and steering's S on the trained keys.
Plan: reports/plans/2026-10-30_plan_e58-open-pair.md. Every setting, rule and threshold: results/predictions.json
(filed and committed before the run; the run refuses otherwise; never edited).

  stage 0, alone (exit 1 on a miss): sha256 of E54's windows files, elicit_s0.pt and cipher_s0.pt; the key split; the
           zero update reads G = H = 0 exactly on all four split tasks; E54's elicit_s0 on E54's own 32-key task
           reproduces r, the stand-in ratio and S, and block 0 reads 0; E54's cipher_s0's value-specific gain is at
           least half its raw gain; the G_spec gate from E54's two seed-0 states (amendment 2026-10-30, after job 46259).
  stage 1: per seed, elicit then the cipher at the elicit update's norm (E54's `keydep` fine-tunes on the trained keys'
           windows); after each, G_spec, H_spec (own value's gain minus the mean gain of the pool's other 31 values) and
           the label, raw G and H beside. Exit 1 if elicit_s0's G_spec is under the gate, or if any cipher reads
           "reuses" (the label's negative control).
  stage 2: r, ratio, S on every state (trained keys' held-out windows); exit 1 on a block-0 shift over 1e-5 nats.
  score:   `open_pair.score` -> results/run.json. Every scored number is also printed.
Writes <large>/windows_{elicit,cipher}_{trained,new}.pt and <large>/{elicit,cipher}_s{0,1,2}.pt (~113 MB each).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from commute import anchors as AN
from commute import keydep as KD
from commute import open_pair as OP
from commute import steering as ST
from commute.heldout_keys import (
    PARTS,
    behaviour,
    gain,
    say_label,
    specific,
    split_task,
    vpd_read,
)

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E54 = HERE.parent / "keydependent_elicit" / "results"
KINDS = ("elicit", "cipher")
log = KD.log
Tasks = dict[str, dict[str, ST.Task]]


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(out: dict[str, Any]) -> None:
    (RESULTS / "run.json").write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the user")


# ----------------------------------------------------------------------------- stages
# (the split tasks, the label and the readings moved to `commute.heldout_keys` on 2026-10-30, after this run, for E58
# Amendment 1; moved, not changed)


def spec_split(ctx: KD.Ctx, kind: str) -> dict[str, Any]:
    """E54's saved <kind>_s0 on its own 32 keys: raw G, the pool's other-value gain, G_spec and G_spec / G
    (amendment 2026-10-30, after job 46259)."""
    t = ST.load_task(ctx.net, E54 / "large" / f"windows_{kind}.pt", kind, kind)
    raw, spec = specific(ctx.net, t, torch.load(E54 / "large" / f"{kind}_s0.pt", weights_only=True), [int(t.table[k]) for k in t.keys])
    G, Gs = float(raw.mean()), float(spec.mean())
    return {"G": G, "other_gain": G - Gs, "G_spec": Gs, "frac": Gs / G if G else math.nan}


def stage0(ctx: KD.Ctx, tasks: Tasks, out: dict[str, Any]) -> float:
    """The controls; returns the G_spec gate."""
    c = ctx.P["stage0"]
    zero = {k: torch.zeros_like(w) for k, w in ctx.net.W0.items()}
    rec: dict[str, Any] = {"zero_max_abs": {f"{n}_{p}": float(gain(ctx.net, tasks[n][p], zero).abs().max()) for n in KINDS for p in PARTS}}
    t = ST.load_task(ctx.net, E54 / "large" / "windows_elicit.pt", "elicit", "elicit")
    rd = vpd_read(ctx, t, torch.load(E54 / "large" / "elicit_s0.pt", weights_only=True), KD.base_of(ctx, t))
    rec["e54_elicit_s0"] = {k: rd[k] for k in ("r", "ratio", "S", "S_block0", "layer")}
    rec["checks"] = {"zero": all(v == 0.0 for v in rec["zero_max_abs"].values()),
                     "r": abs(rd["r"] - c["r_ref"]) <= c["r_tol"], "ratio": abs(rd["ratio"] - c["ratio_ref"]) <= c["ratio_tol"],
                     "S": abs(rd["S"] - c["S_ref"]) <= c["S_tol"], "block0": abs(rd["S_block0"]) <= c["block0_tol"]}
    e, ci = rec["e54_elicit_s0_spec"], rec["e54_cipher_s0_spec"] = spec_split(ctx, "elicit"), spec_split(ctx, "cipher")
    rec["checks"]["cipher_specific"] = bool(ci["frac"] >= c["spec_frac_min"])
    gate = rec["G_min_spec"] = OP.spec_gate(c["G_min_cap"], c["G_ref_raw"], [e["G_spec"], ci["G_spec"]])
    rec["pass"] = all(rec["checks"].values()) and math.isfinite(gate)
    out["stage0"] = rec
    log(f"[stage0] zero-update max |gain| {rec['zero_max_abs']} | E54 elicit_s0 r {rd['r']:.4f} (ref {c['r_ref']:.4f}) ratio "
        f"{rd['ratio']:.4f} (ref {c['ratio_ref']:.4f}) S {rd['S']:.4f} (ref {c['S_ref']:.4f}) S_block0 {rd['S_block0']:.1e} | "
        f"specific gain / raw G: E54 cipher_s0 {ci['G_spec']:.4f} / {ci['G']:.4f} = {ci['frac']:.4f} (bar {c['spec_frac_min']}), "
        f"elicit_s0 {e['G_spec']:.4f} / {e['G']:.4f} = {e['frac']:.4f} (reported) | G_spec gate {gate:.4f} | checks {rec['checks']}")
    dump(out)
    if not rec["pass"]:
        stop(out, "stage 0 missed")
    return gate


def stage1(ctx: KD.Ctx, tasks: Tasks, out: dict[str, Any], gate: float) -> dict[str, ST.DW]:
    th, seeds, st = {**ctx.P["thresholds"], "G_min": gate}, ctx.P["settings"]["seeds"], out["states"]
    ws = {n: torch.load(ctx.large / f"windows_{n}_trained.pt", weights_only=False) for n in KINDS}
    dws: dict[str, ST.DW] = {}

    def keep(n: str, dW: ST.DW, meta: dict[str, Any]) -> None:
        dws[n] = dW
        torch.save(dW, ctx.large / f"{n}.pt")
        st[n] = {"ft": {**meta, "sha256": KD.sha256(ctx.large / f"{n}.pt")}, **behaviour(ctx.net, tasks[n.split("_")[0]], dW, th)}
        log(f"[ft {n}] step {meta['step']} ||dW|| {meta['norm']:.4f}" + (f" at_bound {meta['at_bound']}" if "at_bound" in meta else ""))
        say_label(n, st[n])
        dump(out)

    for s in seeds:
        e, c = f"elicit_s{s}", f"cipher_s{s}"
        dW, meta = KD.elicit_ft(ctx, ws["elicit"], s)
        keep(e, dW, meta)
        if s == seeds[0] and not st[e]["G_spec"] >= th["G_min"]:
            stop(out, f"{e} G_spec {st[e]['G_spec']:.3f} < {th['G_min']:.3f}: E54's recipe on 24 keys is too small to label (plan: raise the step once, as an amendment)")
        dW, _, meta_c, _ = KD.cipher_ft(ctx, ws["cipher"], s, meta["norm"])
        keep(c, dW, meta_c)
        if st[c]["label"] == "reuses":
            stop(out, f"{c} reads 'reuses': the label's negative control failed, the rule is broken")
    return dws


def stage2(ctx: KD.Ctx, tasks: Tasks, dws: dict[str, ST.DW], out: dict[str, Any]) -> None:
    bases = {n: KD.base_of(ctx, tasks[n]["trained"]) for n in KINDS}
    for n, dW in dws.items():
        kind = n.split("_")[0]
        st = out["states"][n]
        st.update(vpd_read(ctx, tasks[kind]["trained"], dW, bases[kind]))
        log(f"[read {n}] r {st['r']:.4f} ratio {st['ratio']:.4f} S {st['S']:.4f} (block {st['layer']}, block-0 {st['S_block0']:.1e}) "
            f"G_logodds_route {st['G_logodds_route']:.4f} vs G {st['G']:.4f}")
        dump(out)
        if abs(st["S_block0"]) * abs(st["G_score_logP"]) > 1e-5:  # E54's stop, in nats
            stop(out, f"{n}: block-0 shift over 1e-5 nats")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="where this run's windows and states go")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    rel = str(PRED.relative_to(REPO))
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode or not git("ls-files", rel):
        sys.exit(f"{rel} is not committed unchanged: commit it before the run")
    P = json.loads(PRED.read_text())
    S = P["settings"]
    assert S["half"] == AN.HALF and S["route_batch"] == KD.BATCH, "predictions.json settings disagree with the code"
    large = a.large.resolve()
    large.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (HERE / "run.py", REPO / "src" / "commute" / "open_pair.py")]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "states": {}}
    sha = {n: KD.sha256(E54 / "large" / f"{n}.pt") for n in P["stage0"]["sha256"]}
    out["e54_sha256"] = sha
    if sha != P["stage0"]["sha256"]:
        stop(out, f"an E54 file is not the one its run recorded: {sha}")
    keys = json.loads((E54 / "stage1.json").read_text())["elicit"]["keys"]
    trained, held = OP.split_keys(keys, S["n_held"], S["split_seed"])
    out["key_split"] = {"trained": trained, "held": held}
    if (trained, held) != (S["trained_keys"], S["held_keys"]):
        stop(out, "the key split is not the filed one")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    out["windows_sha256"] = {n: split_task(E54 / "large" / f"windows_{n}.pt", trained, held, large, n) for n in KINDS}
    tasks = {n: {p: ST.load_task(ctx.net, large / f"windows_{n}_{p}.pt", f"{n}_{p}", n) for p in PARTS} for n in KINDS}
    out["n_windows"] = {f"{n}_{p}": len(tasks[n][p].toks) for n in KINDS for p in PARTS}
    log(f"[split] trained {trained} held {held} windows {out['n_windows']}")
    gate = stage0(ctx, tasks, out)
    dws = stage1(ctx, tasks, out, gate)
    stage2(ctx, tasks, dws, out)
    out["score"] = OP.score(out["states"], P["thresholds"], S["seeds"])
    dump(out)
    log(f"[score] {json.dumps(_finite(out['score']))}")


if __name__ == "__main__":
    main()
