"""open_pair Amendment 6 (E58, 2026-10-30, after the review of the re-gating synthesis): two maps that break the tie
between "own guess" and "high W0 prior". Fable, reports/plans/2026-10-30_advice_regating-baseline-FABLE.md, (b)1 and (c).
Plan: reports/plans/2026-10-30_plan_e58-open-pair.md, section "Amendment 6". Rules, settings and thresholds:
results/predictions_amend6.json (filed and committed first; the run refuses otherwise).

  W0  (the W0 control, first; exit 1 on a miss) Amendment 3's elicit windows (sha256 against amend3.json) and its key split;
      W0's geometric-mean P(token | M key) per key (`kv.key_distributions(geometric=True)`, fp32). Two tables on those
      windows: the prior-matched cipher (`kv.bijection` from E58's cipher pool, each key's value at W0 geo-P in the band
      and past rank `min_rank`, the keys and elicit words excluded) and the first-letter rule (one word per first letter
      of the key, from the same pool, W0 geo-P under `letter_max_p` after every key of that letter). Each cut by key
      (`heldout_keys.split_task`); the zero update reads 0 exactly; each map's W0 prior of its trained values (keydep.geo_rate,
      the number saved as amend3.json maps.<map>.w0_geo_p) must sit in its band.
  C   (the harness control, alone) E58's 128-key cipher, seed 0, re-trained and read at its gate step from amend4.json;
      exit 1 unless r reproduces amend4.json's within `reproduce_tol`.
  PM  seeds 0-2: fine-tune (every step 1-30 kept), label reads from step 10 until G_spec reaches the gate
      (`amend4.over_gate`), then label, x, r, ratio at step 30 and at that step.
  FL  seeds 0-2, the same, except r is read at a step only when x there passes the entry condition (`open_pair.carries`).
  score `open_pair.score_amend6` -> results/amend6.json, with each map's W0 prior beside its r and E58's five maps' priors
      and step-30 r from amend3/4/5.json. Every number is also printed. No weights saved; windows in <large>/amend6.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import amend2
import amend3
import amend4
import torch
from amend1 import git, load_split
from amend3 import read
from amend4 import over_gate, stop, train

from commute import anchors as AN
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute.data import Windows
from commute.heldout_keys import PARTS, gain, split_task

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_amend6.json", RESULTS / "amend6.json"
MAPS = ("prior_cipher", "first_letter")  # the scored map first
log = KD.log
amend2.OUT = amend3.OUT = amend4.OUT = OUT  # their dump and stop write this run's file


def draw(name: str, marker: int, keys: list[int], pools: dict[str, list[int]], ok: Any) -> tuple[kv.KVTask, str] | None:
    """`kv.bijection` (seed 0) from the first pool that serves every key; None if none does."""
    for pool_name, pool in pools.items():
        try:
            return kv.bijection(name, marker, tuple(keys), pool, seed=0, ok=ok), pool_name
        except AssertionError:
            log(f"[maps {name}] pool {pool_name} ran out of admissible values")
    return None


def tables(ctx: KD.Ctx, src: dict[str, Any], keys: list[int], elicit: list[int], out: dict[str, Any]) -> list[kv.KVTask]:
    """The prior-matched cipher and the first-letter rule on the keys (see the module docstring)."""
    S = ctx.P["settings"]
    tr, ho = src["train"], src["heldout"]
    both = Windows(torch.cat([tr.tokens, ho.tokens]), torch.cat([tr.g, ho.g]), torch.cat([tr.row, ho.row]), torch.cat([tr.pos, ho.pos]), tr.half)
    geo = kv.key_distributions(lambda x: ctx.net.run(x)[0], both, ctx.net.device, geometric=True)
    _, _, _, ranked, tok = KD.vocab(ctx)
    banned = set(keys) | set(elicit)
    pools = {"cipher_pool": [v for v in ranked[AN.VALUE_RANK0 : AN.VALUE_RANK0 + AN.POOL] if v not in banned],
             "ranked_from_1000": [v for v in ranked[AN.VALUE_RANK0 :] if v not in banned]}
    (lo, hi), kth = S["band"], {k: float(geo[k].topk(S["min_rank"]).values[-1]) for k in keys}
    pm = draw("prior_cipher", src["task"].marker, keys, pools, lambda k, v: lo <= float(geo[k][v]) <= hi and float(geo[k][v]) < kth[k])
    letter = {k: tok.decode([k]).strip()[:1].lower() for k in keys}
    group: dict[str, list[int]] = {}
    for k in keys:
        group.setdefault(letter[k], []).append(k)
    rep = {g[0]: g for g in group.values()}  # one representative key per letter draws the letter's word
    fl = draw("letters", src["task"].marker, list(rep), pools, lambda r, v: all(float(geo[k][v]) < S["letter_max_p"] for k in rep[r]))
    if pm is None or fl is None:
        stop(out, "a pool ran out of admissible values")
    word = {letter[r]: v for r, v in zip(fl[0].keys, fl[0].values, strict=True)}
    out["tables"] = {"prior_cipher_pool": pm[1], "letter_pool": fl[1], "letters": {L: len(g) for L, g in sorted(group.items())},
                     "letter_of": {str(k): letter[k] for k in keys}}
    return [pm[0], kv.KVTask("first_letter", src["task"].marker, tuple(keys), tuple(word[letter[k]] for k in keys))]


def save_map(ctx: KD.Ctx, src: dict[str, Any], m: kv.KVTask, trained: list[int], held: list[int], d: Path, out: dict[str, Any]) -> dict[str, Any]:
    """Amendment 3's windows with `m`'s table, cut by key; its record (zero update, W0 prior of its values)."""
    V = int(ctx.net.net.lm_head.weight.shape[0])
    torch.save({**src, "task": m, "table": m.table(V)}, d / f"windows_{m.name}.pt")
    sha = split_task(d / f"windows_{m.name}.pt", trained, held, d, m.name)
    t = load_split(ctx, d, m.name)
    val = dict(zip(m.keys, m.values, strict=True))
    zero = {k: torch.zeros_like(x) for k, x in ctx.net.W0.items()}
    letters_trained = {out["tables"]["letter_of"][str(k)] for k in trained}
    out["maps"][m.name] = {"sha256": sha, "values": [val[k] for k in m.keys], "n_distinct": len(set(m.values)),
                           "n_held_value_trained": sum(val[k] in {val[x] for x in trained} for k in held),
                           "n_held_letter_trained": sum(out["tables"]["letter_of"][str(k)] in letters_trained for k in held),
                           "zero_max_abs": max(float(gain(ctx.net, t[p], zero).abs().max()) for p in PARTS),
                           "w0_geo_p": {p: KD.geo_rate(t[p].lp0) for p in PARTS}}
    log(f"[maps {m.name}] {json.dumps({k: v for k, v in out['maps'][m.name].items() if k != 'values'})}")
    return t


def w0_control(ctx: KD.Ctx, a3: dict[str, Any], d3: Path, d: Path, out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    S, th = ctx.P["settings"], ctx.P["thresholds"]
    if KD.sha256(d3 / "windows_elicit.pt") != a3["maps"]["elicit"]["sha256"]["full"]:
        stop(out, "Amendment 3's elicit windows changed")
    keys = a3["keys"]["keys"]
    trained, held = OP.split_keys(keys, len(keys) // 4, S["split_seed"])
    if held != a3["keys"]["held"]:
        stop(out, "the key split differs from Amendment 3's")
    src = torch.load(d3 / "windows_elicit.pt", weights_only=False)
    out["maps"] = {}
    tasks = {m.name: save_map(ctx, src, m, trained, held, d, out) for m in tables(ctx, src, keys, a3["maps"]["elicit"]["values"], out)}
    p = {n: out["maps"][n]["w0_geo_p"]["trained"] for n in MAPS}
    ok = (all(out["maps"][n]["zero_max_abs"] == 0.0 for n in MAPS) and S["band"][0] <= p["prior_cipher"] <= S["band"][1]
          and p["first_letter"] < S["letter_max_p"])
    out["w0_control"] = {"pass": ok, "prior_trained": p, "copy_prior": th["copy_prior"]}
    log(f"[W0 control] {json.dumps(out['w0_control'])}")
    amend3.dump(out)
    if not ok:
        stop(out, "the W0 control missed (a zero update not 0, or a prior outside its band)")
    return tasks


def harness_control(ctx: KD.Ctx, a3: dict[str, Any], a4: dict[str, Any], d3: Path, out: dict[str, Any]) -> None:
    """E58's cipher seed 0 at its gate step, re-trained here; r must equal amend4.json's within reproduce_tol."""
    if any(KD.sha256(d3 / f"windows_cipher_{p}.pt") != a3["maps"]["cipher"]["sha256"][p] for p in PARTS):
        stop(out, "Amendment 3's cipher windows changed")
    step = a4["control"]["0"]["step"]
    rd = read(ctx, "cipher_s0", load_split(ctx, d3, "cipher"), train(ctx, d3, "cipher", 0, KD.CIPHER_CHECKS), step, out, vpd=True)
    ref = a4["reads"][f"cipher_s0@{step}"]["r"]
    out["harness_control"] = {"step": step, "r": rd["r"], "r_amend4": ref, "pass": OP.comparable([(rd["r"], ref)], ctx.P["thresholds"]["reproduce_tol"])}
    log(f"[harness control] {json.dumps(out['harness_control'])}")
    amend3.dump(out)
    if not out["harness_control"]["pass"]:
        stop(out, "E58's cipher seed 0 does not reproduce amend4.json's r")


def one_seed(ctx: KD.Ctx, n: str, t: dict[str, Any], d: Path, seed: int, out: dict[str, Any]) -> None:
    run, th = f"{n}_s{seed}", ctx.P["thresholds"]
    ft = train(ctx, d, n, seed, KD.CIPHER_CHECKS)
    at = out["gate_step"][run] = over_gate(ctx, run, t, ft, out)
    for s in sorted({30, at if at is not None else 30}):
        if n == "first_letter":  # the entry condition first, r only if it holds
            rd = out["reads"].get(f"{run}@{s}") or read(ctx, run, t, ft, s, out, vpd=False)
            if not OP.carries(rd, th):
                log(f"[entry {run} @ {s}] x {rd['x']:.4f} label {rd['label']}: not over {th['x_entry']}; r not read")
                continue
        read(ctx, run, t, ft, s, out, vpd=True)


def references(a3: dict[str, Any], a4: dict[str, Any], a5: dict[str, Any]) -> dict[str, Any]:
    """E58's five maps: W0 prior of the trained values (amend3.json) and step-30 r, seeds 0-2 (amend3/4/5.json)."""
    r = lambda f, k: (f["reads"].get(k) or {}).get("r")
    rs = {"cipher": [r(a4, f"cipher_s{s}@30") for s in (0, 1, 2)], "shuffle": [r(a4, "shuffle_s0@30"), r(a5, "shuffle_s1@30"), r(a5, "shuffle_s2@30")],
          **{n: [r(a3, f"{n}@30"), r(a4, f"{n}_s1@30"), r(a4, f"{n}_s2@30")] for n in ("elicit", "rank1", "copy")}}
    return {n: {"w0_geo_p_trained": a3["maps"][n]["w0_geo_p"]["trained"], "r30": v} for n, v in rs.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="reads <large>/amend3; writes <large>/amend6 (windows, ~5 MB)")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P = json.loads(PRED.read_text())
    a3, a4, a5 = (json.loads((RESULTS / f"amend{i}.json").read_text()) for i in (3, 4, 5))
    d3, d = a.large.resolve() / "amend3", a.large.resolve() / "amend6"
    d.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (*(HERE / f"amend{i}.py" for i in range(1, 7)), *(REPO / "src" / "commute" / f for f in ("open_pair.py", "heldout_keys.py", "keydep.py", "kv.py")))]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "reads": {}, "gate_step": {}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    out["host"] = {"hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    log(f"[host] {out['host']}")
    tasks = w0_control(ctx, a3, d3, d, out)
    harness_control(ctx, a3, a4, d3, out)
    seeds = P["settings"]["seeds"]
    for n in MAPS:
        for seed in seeds:
            one_seed(ctx, n, tasks[n], d, seed, out)
    at30 = lambda n: [out["reads"].get(f"{n}_s{s}@30") for s in seeds]
    at_gate = lambda n: [out["reads"].get(f"{n}_s{s}@{out['gate_step'][f'{n}_s{s}']}") for s in seeds]
    out["score"] = OP.score_amend6(at30("prior_cipher"), at30("first_letter"), P["thresholds"])
    out["at_gate_step"] = OP.score_amend6(at_gate("prior_cipher"), at_gate("first_letter"), P["thresholds"])  # reported
    ref = references(a3, a4, a5)
    out["prior_beside_r"] = {**{n: {"w0_geo_p_trained": out["maps"][n]["w0_geo_p"]["trained"], "r30": out["score"][k]["r"]}
                                for n, k in zip(MAPS, ("PM", "FL"), strict=True)}, **ref}
    mean = {n: [r for r in v["r30"] if OP.finite(r)] for n, v in out["prior_beside_r"].items()}
    kept = [n for n, v in mean.items() if v]
    out["rho_r_logprior"] = OP.spearman([sum(mean[n]) / len(mean[n]) for n in kept], [out["prior_beside_r"][n]["w0_geo_p_trained"] for n in kept])
    amend3.dump(out)
    log(f"[score] {json.dumps(OP.jsonable(out['score']))}")
    log(f"[at gate step, reported] {json.dumps(OP.jsonable(out['at_gate_step']))}")
    log(f"[prior beside r] {json.dumps(OP.jsonable(out['prior_beside_r']))} Spearman(r, prior) over maps {out['rho_r_logprior']:.3f}")


if __name__ == "__main__":
    main()
