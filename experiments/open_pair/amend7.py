"""open_pair Amendment 7 (E58, 2026-10-30, after Amendment 6's first-letter rule did not carry): a new rule that carries
to unseen keys, its shuffled twin, and a cipher at elicit's prior. Fable, reports/plans/2026-10-30_advice_regating-carry-cell-FABLE.md,
step 1. Plan: reports/plans/2026-10-30_plan_e58-open-pair.md, section "Amendment 7". Rules, settings and thresholds:
results/predictions_amend7.json (filed and committed first; the run refuses otherwise).

  W0  (the W0 control, first; exit 1 on a miss) Amendment 3's elicit windows (sha256) and key split; W0's geometric-mean
      P(token | M key) per key (fp32). Three tables on those windows: the class map (class = the sign pair of the key's
      scores on the two leading principal components of W0's input embeddings of the 128 keys, `open_pair.pc_classes`;
      one word per class from E58's cipher pool, W0 prior under `class_max_p` after every one of the 128 keys); its twin
      (the same four words by `open_pair.shuffled_classes`, same class sizes, no structure); the elicit-prior cipher
      (`kv.bijection`, prior in `ep_band` and past rank `ep_min_rank`; skipped, not stopped, if both pools run out).
      Each cut by key; the zero update reads 0 exactly on every map and on the cross-class null; class-map and twin
      priors under `class_max_p`; every class holds at least `min_class_trained` trained keys in both maps.
  C   (the harness control, alone) E58's cipher seed 0 at its gate step: `amend6.harness_control`.
  T   (the twin, the carry control) seeds 0-2: fine-tune, label reads from step 10 to the gate, then x, x_null
      (`open_pair.cross_class_null`) and r at step 30 and the gate step; x and x_null at step 100. Exit 1 unless
      `open_pair.twin_clean` reads "clean" at step 30.
  CL  seeds 0-2, the same, except r is read at a step only when `open_pair.carries` holds there.
  EP  seeds 0-2 (when built): label, x and r at step 30 and the gate step.
  score `open_pair.score_amend7` -> results/amend7.json, each map's W0 prior beside its r and E58's priors and r. Every
      number is also printed. No weights saved; windows in <large>/amend7.
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
import amend6
import torch
from amend1 import git
from amend3 import read
from amend4 import over_gate, stop, train
from amend6 import draw, harness_control, references, save_map

from commute import anchors as AN
from commute import keydep as KD
from commute import kv
from commute import open_pair as OP
from commute import steering as ST
from commute.data import Windows
from commute.heldout_keys import pool_of, specific

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_amend7.json", RESULTS / "amend7.json"
TWIN, CMAP, EP = "class_shuffled", "class_map", "elicit_prior_cipher"  # run order: the twin (the control) first
log = KD.log
amend2.OUT = amend3.OUT = amend4.OUT = amend6.OUT = OUT  # their dump and stop write this run's file


def tables(ctx: KD.Ctx, src: dict[str, Any], keys: list[int], trained: list[int], elicit: list[int],
           out: dict[str, Any]) -> tuple[list[tuple[kv.KVTask, str]], dict[str, dict[int, int]]]:
    """The class map, its twin and the elicit-prior cipher (see the module docstring); each map's class per key."""
    S, marker = ctx.P["settings"], src["task"].marker
    tr, ho = src["train"], src["heldout"]
    both = Windows(torch.cat([tr.tokens, ho.tokens]), torch.cat([tr.g, ho.g]), torch.cat([tr.row, ho.row]), torch.cat([tr.pos, ho.pos]), tr.half)
    geo = kv.key_distributions(lambda x: ctx.net.run(x)[0], both, ctx.net.device, geometric=True)
    _, _, _, ranked, tok = KD.vocab(ctx)
    banned = set(keys) | set(elicit)
    pools = {"cipher_pool": [v for v in ranked[AN.VALUE_RANK0 : AN.VALUE_RANK0 + AN.POOL] if v not in banned],
             "ranked_from_1000": [v for v in ranked[AN.VALUE_RANK0 :] if v not in banned]}
    pc = OP.pc_classes(ctx.net.net.wte.weight[keys])
    cls = {CMAP: dict(zip(keys, pc, strict=True)), TWIN: dict(zip(keys, OP.shuffled_classes(pc, S["twin_seed"]), strict=True))}
    n_tr = {n: [sum(c[k] == j for k in trained) for j in range(4)] for n, c in cls.items()}
    out["tables"] = {"n_trained": n_tr, "n_all": [pc.count(j) for j in range(4)], "twin_agree": sum(cls[TWIN][k] == cls[CMAP][k] for k in keys),
                     "class_of": {n: {str(k): j for k, j in c.items()} for n, c in cls.items()}}
    if min(min(v) for v in n_tr.values()) < S["min_class_trained"]:
        stop(out, f"a class has fewer than {S['min_class_trained']} trained keys: {n_tr}")
    top = torch.stack([geo[k] for k in keys]).max(0).values  # each token's largest prior over the 128 keys
    rep = [next(k for k in keys if cls[CMAP][k] == j) for j in range(4)]  # one key per class draws the class's word
    got = draw("class_words", marker, rep, pools, lambda _k, v: float(top[v]) < S["class_max_p"])
    if got is None:
        stop(out, "no admissible class word left in either pool")
    word = dict(enumerate(got[0].values))
    (lo, hi), kth = S["ep_band"], {k: float(geo[k].topk(S["ep_min_rank"]).values[-1]) for k in keys}
    ep = draw(EP, marker, keys, pools, lambda k, v: lo <= float(geo[k][v]) <= hi and float(geo[k][v]) < kth[k])
    out["tables"].update({"word_of": {str(j): [v, tok.decode([v])] for j, v in word.items()}, "ep_built": ep is not None})
    maps = [(kv.KVTask(n, marker, tuple(keys), tuple(word[cls[n][k]] for k in keys)), got[1]) for n in (TWIN, CMAP)]
    return [*maps, *([ep] if ep is not None else [])], cls


def const_tasks(ctx: KD.Ctx, m: kv.KVTask, held: list[int], d: Path) -> dict[int, ST.Task]:
    """Per class, the held-out keys' windows with every target set to that class's word (for `with_null`)."""
    w = torch.load(d / f"windows_{m.name}_new.pt", weights_only=False)
    word = dict(zip(m.keys, m.values, strict=True))
    out = {}
    for j, v in enumerate(dict.fromkeys(word[k] for k in sorted(m.keys))):
        t = kv.KVTask(f"{m.name}_const{j}_new", m.marker, tuple(held), (v,) * len(held))
        torch.save({**w, "task": t, "table": t.table(int(w["table"].shape[0]))}, d / f"windows_{t.name}.pt")
        out[v] = ST.load_task(ctx.net, d / f"windows_{t.name}.pt", t.name, m.name)
    return out


def words_at(t: ST.Task) -> list[int]:
    """Per window of a task, its target word (the window's class, named by its word)."""
    return [int(t.table[k]) for k in t.toks[:, t.half].tolist()]


def with_null(ctx: KD.Ctx, t: dict[str, Any], const: dict[int, ST.Task], dW: ST.DW, rd: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """x_null = `open_pair.cross_class_null` over the held-out windows, over G_spec (classes named by their word). Its
    own term recomputes H_spec from the same gains: stop unless it agrees within null_self_tol."""
    s = {v: specific(ctx.net, c, dW, pool_of(t))[1].tolist() for v, c in const.items()}
    own, null = OP.cross_class_null(s, words_at(t["new"]))
    if not abs(own - rd["H_spec"]) <= ctx.P["thresholds"]["null_self_tol"]:
        stop(out, f"the cross-class null's own term {own} differs from H_spec {rd['H_spec']}")
    xn = null / rd["G_spec"] if rd["G_spec"] != 0 else float("nan")
    return {"x_null": xn, "x_minus_null": rd["x"] - xn}


def w0_control(ctx: KD.Ctx, a3: dict[str, Any], d3: Path, d: Path, out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    S = ctx.P["settings"]
    if KD.sha256(d3 / "windows_elicit.pt") != a3["maps"]["elicit"]["sha256"]["full"]:
        stop(out, "Amendment 3's elicit windows changed")
    keys = a3["keys"]["keys"]
    trained, held = OP.split_keys(keys, len(keys) // 4, S["split_seed"])
    if held != a3["keys"]["held"]:
        stop(out, "the key split differs from Amendment 3's")
    src = torch.load(d3 / "windows_elicit.pt", weights_only=False)
    out["maps"] = {}
    made, cls = tables(ctx, src, keys, trained, a3["maps"]["elicit"]["values"], out)
    tasks = {m.name: save_map(ctx, src, m, pool, trained, held, d, {"tables": {"letter_of": {str(k): c for k, c in cls[CMAP].items()}},
                                                                    "maps": out["maps"]}) for m, pool in made}
    zero = {k: torch.zeros_like(x) for k, x in ctx.net.W0.items()}
    for m, _ in made[:2]:
        tasks[m.name]["const"] = const_tasks(ctx, m, held, d)
        t = tasks[m.name]
        s = {v: specific(ctx.net, c, zero, pool_of(t))[1].tolist() for v, c in t["const"].items()}
        out["maps"][m.name]["zero_null"] = list(OP.cross_class_null(s, words_at(t["new"])))
    p = {n: out["maps"][n]["w0_geo_p"]["trained"] for n in out["maps"]}
    ok = (all(v["zero_max_abs"] == 0.0 for v in out["maps"].values()) and all(out["maps"][n]["zero_null"] == [0.0, 0.0] for n in (TWIN, CMAP))
          and p[TWIN] < S["class_max_p"] and p[CMAP] < S["class_max_p"])
    ep_in_band = EP in p and S["ep_band"][0] <= p[EP] <= S["ep_band"][1]
    out["w0_control"] = {"pass": ok, "prior_trained": p, "ep_in_band": ep_in_band, "n_held": {n: [sum(c[k] == j for k in held) for j in range(4)] for n, c in cls.items()}}
    log(f"[tables] {json.dumps({k: v for k, v in out['tables'].items() if k != 'class_of'})}")
    log(f"[W0 control] {json.dumps(out['w0_control'])}")
    amend3.dump(out)
    if not ok:
        stop(out, "the W0 control missed (a zero update not 0, or a class-map or twin prior over class_max_p)")
    return tasks


def rule_seed(ctx: KD.Ctx, n: str, t: dict[str, Any], d: Path, seed: int, out: dict[str, Any]) -> None:
    """One seed of the twin or the class map: x and x_null at 30, the gate step and 100; r at 30 and the gate step
    (the twin always, the class map only where it `carries`)."""
    run = f"{n}_s{seed}"
    ft = train(ctx, d, n, seed, KD.CIPHER_CHECKS)
    at = out["gate_step"][run] = over_gate(ctx, run, t, ft, out)
    for s in sorted({30, at if at is not None else 30, 100}):
        rd = out["reads"].get(f"{run}@{s}") or read(ctx, run, t, ft, s, out, vpd=False)
        null = with_null(ctx, t, t["const"], ft.deltas[s], rd, out)
        rd.update(null)
        log(f"[entry {run} @ {s}] x {rd['x']:.4f} x_null {null['x_null']:.4f} x - x_null {null['x_minus_null']:.4f} label {rd['label']}")
        if s != 100 and (n == TWIN or OP.carries(rd, ctx.P["thresholds"])):
            read(ctx, run, t, ft, s, out, vpd=True).update(null)
        amend3.dump(out)


def ep_seed(ctx: KD.Ctx, t: dict[str, Any], d: Path, seed: int, out: dict[str, Any]) -> None:
    run = f"{EP}_s{seed}"
    ft = train(ctx, d, EP, seed, KD.CIPHER_CHECKS)
    at = out["gate_step"][run] = over_gate(ctx, run, t, ft, out)
    for s in sorted({30, at if at is not None else 30}):
        read(ctx, run, t, ft, s, out, vpd=True)


def score(out: dict[str, Any], P: dict[str, Any], a3: dict[str, Any], a4: dict[str, Any], a5: dict[str, Any], a6: dict[str, Any]) -> None:
    seeds, th = P["settings"]["seeds"], P["thresholds"]
    at = lambda n, step: [out["reads"].get(f"{n}_s{s}@{step(n, s)}") for s in seeds]
    s30, gate = (lambda n, s: 30), (lambda n, s: out["gate_step"].get(f"{n}_s{s}"))
    ep = EP in out["maps"]
    out["score"] = OP.score_amend7(at(TWIN, s30), at(CMAP, s30), at(EP, s30) if ep else None, th)
    out["score"]["EP"]["scored"] = ep and out["maps"][EP]["pool"] == "cipher_pool" and out["w0_control"]["ep_in_band"]
    out["at_gate_step"] = OP.score_amend7(at(TWIN, gate), at(CMAP, gate), at(EP, gate) if ep else None, th)  # reported
    out["at_step_100"] = {n: [(rd or {}).get("x_minus_null") for rd in at(n, lambda n, s: 100)] for n in (TWIN, CMAP)}  # reported
    r30 = lambda n: [(rd or {}).get("r") for rd in at(n, s30)]
    out["prior_beside_r"] = {**{n: {"w0_geo_p_trained": out["maps"][n]["w0_geo_p"]["trained"], "r30": r30(n)} for n in out["maps"]},
                             "prior_cipher": a6["prior_beside_r"]["prior_cipher"], **references(a3, a4, a5)}
    amend3.dump(out)
    log(f"[score] {json.dumps(OP.jsonable(out['score']))}")
    log(f"[at gate step, reported] {json.dumps(OP.jsonable(out['at_gate_step']))}")
    log(f"[x - x_null at step 100, reported] {json.dumps(OP.jsonable(out['at_step_100']))}")
    log(f"[prior beside r] {json.dumps(OP.jsonable(out['prior_beside_r']))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="reads <large>/amend3; writes <large>/amend7 (windows, ~10 MB)")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P = json.loads(PRED.read_text())
    a3, a4, a5, a6 = (json.loads((RESULTS / f"amend{i}.json").read_text()) for i in (3, 4, 5, 6))
    d3, d = a.large.resolve() / "amend3", a.large.resolve() / "amend7"
    d.mkdir(parents=True, exist_ok=True)
    code = [str(p.relative_to(REPO)) for p in (*(HERE / f"amend{i}.py" for i in range(1, 8)), *(REPO / "src" / "commute" / f for f in ("open_pair.py", "heldout_keys.py", "keydep.py", "kv.py")))]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "reads": {}, "gate_step": {}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    out["host"] = {"hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    log(f"[host] {out['host']}")
    tasks = w0_control(ctx, a3, d3, d, out)
    harness_control(ctx, a3, a4, d3, out)
    seeds = P["settings"]["seeds"]
    for seed in seeds:
        rule_seed(ctx, TWIN, tasks[TWIN], d, seed, out)
    twin = OP.twin_clean([out["reads"].get(f"{TWIN}_s{s}@30") for s in seeds], P["thresholds"])
    log(f"[twin control] {twin}")
    if twin != "clean":
        stop(out, f"the shuffled-class twin reads {twin} at step 30")
    for seed in seeds:
        rule_seed(ctx, CMAP, tasks[CMAP], d, seed, out)
    for seed in seeds if EP in tasks else []:
        ep_seed(ctx, tasks[EP], d, seed, out)
    score(out, P, a3, a4, a5, a6)


if __name__ == "__main__":
    main()
