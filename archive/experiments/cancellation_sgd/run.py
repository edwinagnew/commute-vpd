"""cancellation_sgd (E23): where does the adversary's extra damage sit on the component-training failures that persist
after E21 (SGD at step 1000, the key-to-value SGD states, r2 at step 100), beside their passing twins (Adam 1e-5 at step
1000, SGD r0 at step 100), and do the step-1000 verdicts hold at larger adversary budgets? Plan:
reports/plans/2026-10-22_plan_small-update-reruns.md -> E23 (approved class 1, the user's instruction 2026-10-22).

States: E21's saved bf16 offsets (sgd_ladder_long, sgd_kv; cluster NFS), rebuilt by `saved_states.offsets_state`.
Stage A (`revert --shard N`), per shard, cheapest and control first, results dumped after every model:
  W0       the Pile harness (256 rows, 20 steps, adversary seeds 0, 1, 2): faith_score.w0_check, the same-job floor f
  G0       W0 through `excess.measure` (128 rows, seed 0): E20 amendment 1's W0 numbers reproduced; then seed 1 (the
           null decomposition: W0's own revert noise)
  C1       per state: the harness's adversarial ratio reproduces E21's verdict and is within 25% of E21's ratio
  parts    per state: `excess.measure` (E17's three tests, E20's split with amendment 1); shard 0 holds the control
           (SGD r0 step 100), measured first, which must show no excess (P0) or the shard stops
Stage B (`budget --shard N`): E20 amendment 3's sweep, `gamma.pgd_sources` with n sign steps of 2 / n, n in BUDGETS,
seed 0 (W0 also seed 1), fp32 `cancel.masked_kl`; gate C3: 20 steps reproduce stage A's F_all within 1e-3 relative.
  run.py --predictions        file results/predictions.json (once)
  run.py revert --shard 0|1   -> results/revert_shard{N}.json, results/large/revert_shard{N}.log, sources in large/sources/
  run.py budget --shard 0|1   -> results/budget_shard{N}.json (needs revert_shard0.json)
  run.py score                -> results/cancellation_sgd.json (JSON only)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
EXP = HERE.parent
ROUND1 = EXP / "transport_test" / "results" / "large"
E20_A1 = EXP / "cancellation_complement" / "results" / "complement_amend1.json"
LONG, KV = EXP / "sgd_ladder_long" / "results", EXP / "sgd_kv" / "results"
R0, R1, R2 = "r0_t352_v310", "r1_t534_v369", "r2_t275_v2667"
# name -> (offsets file, where E21's reading is: ("long", task, arm, step) or ("kv", task, arm))
STATES: dict[str, tuple[Path, tuple[Any, ...]]] = {
    "sgd r0 @100": (LONG / "large" / R0 / "sgd_0.03_step100.pt", ("long", R0, "sgd_0.03", 100)),
    "sgd r0 @1000": (LONG / "large" / R0 / "sgd_0.03_step1000.pt", ("long", R0, "sgd_0.03", 1000)),
    "sgd r1 @1000": (LONG / "large" / R1 / "sgd_0.03_step1000.pt", ("long", R1, "sgd_0.03", 1000)),
    "adam r0 @1000": (LONG / "large" / R0 / "adam_1e-05_step1000.pt", ("long", R0, "adam_1e-05", 1000)),
    "adam r1 @1000": (LONG / "large" / R1 / "adam_1e-05_step1000.pt", ("long", R1, "adam_1e-05", 1000)),
    "sgd rand32": (KV / "large" / "rand32" / "sgd_0.01.pt", ("kv", "rand32", "sgd_0.01")),
    "sgd ident32": (KV / "large" / "ident32" / "sgd_0.003.pt", ("kv", "ident32", "sgd_0.003")),
    "sgd r2 @100": (LONG / "large" / R2 / "sgd_0.1_step100.pt", ("long", R2, "sgd_0.1", 100)),
}
CONTROL = "sgd r0 @100"
FAILING = ("sgd r0 @1000", "sgd r1 @1000", "sgd rand32", "sgd ident32", "sgd r2 @100")
SGD_1000, ADAM_1000 = ("sgd r0 @1000", "sgd r1 @1000"), ("adam r0 @1000", "adam r1 @1000")
A_SHARDS = ((CONTROL, *SGD_1000, *ADAM_1000), ("sgd rand32", "sgd ident32", "sgd r2 @100"))
BUDGETS = (20, 40, 80, 160)
# stage B: (model, seed, budgets); shard 1 re-reads W0 seed 0 at 20 steps only (gate C3), its ratios use shard 0's W0
B_SHARDS = ((("W0", 0, BUDGETS), *((s, 0, BUDGETS) for s in SGD_1000)),
            (("W0", 0, (20,)), ("W0", 1, BUDGETS), *((s, 0, BUDGETS) for s in ADAM_1000)))
ROWS, BATCH, PILE_ROWS, PILE_SEEDS = 128, 8, 256, (0, 1, 2)
G0_TOL, C1_TOL, C3_TOL, FULL_TOL = 1e-3, 0.25, 1e-3, 1e-4  # relative, relative, relative, nats
STAY_FAILING, STAY_PASSING, STABLE = 1.5, 1.241, 1.5  # amendment 3's thresholds (P9, P10, P11)
SHARE, DENSE_SHARE, ADV_SHARE = 0.5, 0.05, 0.2  # E20's and E17's readings, as filed below


PREDICTIONS: dict[str, Any] = {
    "design": "reports/plans/2026-10-22_plan_small-update-reruns.md -> E23 (approved class 1: GPU passes of existing "
              "measurements on saved states; the user's instruction 2026-10-22); parents E17, E20 with amendments 1 and 3",
    "states": {n: f"{p.relative_to(EXP.parent)} (E21: {ref})" for n, (p, ref) in STATES.items()},
    "setup": {
        "rebuild": "saved_states.offsets_state: U2 = U + dU, V2 = V + dV (bf16 offsets), delta-W = U2 V2^T - U V^T; the "
                   "rebuilt state differs from the trained one by the offsets' bf16 rounding (gate C1 reads its effect)",
        "stage_A": f"Pile val rows 0..{ROWS} in batches of {BATCH}, every KL fp32; per model its own adversary "
                   "(gamma.pgd_sources, 20 sign steps of 0.1, seed 0); commute.excess.measure: E17's coherence, F_full, "
                   "F_half (3 halves), F_adv, F_perm (3); E20's F_all, F_orig, F_both and amendment 1's F_dead, F_dense, "
                   "F_rest and the fixed-level arms at this job's W0 levels. Parts (excess.parts): X = F_all(s) - "
                   "F_all(W0) = O + F_adv + I + R, R = D + K + P + J exactly; reading by amendment 1's rule",
        "stage_B": f"E20 amendment 3's recipe: gamma.pgd_sources with n sign steps of 2 / n for n in {list(BUDGETS)}, "
                   "each budget its own run from the seed's random start, seed 0 (W0 also seed 1), fp32 cancel.masked_kl "
                   "on the same 128 rows; r(s, n) = KL(s, n) / KL(W0 seed 0, n)",
        "floor": f"the Pile harness at W0 ({PILE_ROWS} rows, 20 steps, adversary seeds {list(PILE_SEEDS)}) in each stage-A "
                 "job: f = max/min - 1 of W0's PGD KL (0.209 on the L40); band = f x F_all(W0, 128 rows), about 0.14 nats",
        "noise": "stage A: W0 under adversary seed 1 decomposed against seed 0 (null_parts: W0's own revert noise, one "
                 "draw) and the band; stage B: W0 seed 0 vs seed 1 per budget (amendment 3: 0.66 vs 0.71 at 20, 0.71 vs "
                 "0.86 at 160); one adversary seed per state",
        "expected_at_W0": "stage A seed 0 = E20 amendment 1's W0: F_all 0.6604, F_orig 0.4868, F_dead 1.0701, F_dense "
                          "0.0068, F_rest 1.0754, F_adv = F_full = 0, F_both = F_orig; stage B: 0.660 at 20 steps rising "
                          "to about 0.71 at 160 (amendment 3 shard 0 was running; not read here beyond 20 steps)",
        "expected_states": "C1: E21's ratios (256 rows): control 0.92 pass; SGD r0, r1 @1000 2.82, 3.00 fail; Adam r0, r1 "
                           "@1000 1.07, 0.95 pass; rand32 4.16, ident32 2.37, r2 @100 5.20 fail. At 128 rows X of the "
                           "failing states about (ratio - 1) x 0.66 nats; passing states and the control |X| < band",
        "blind_to": "Pile rows only, never the trigger; one adversary family (sign steps, one source per subcomponent "
                    "over all rows) and one seed per state; KL is not additive (I, P, J absorb it); the parts say where "
                    "the adversary's mask costs more, not why; no budget past 160",
        "positive_control": f"{CONTROL} (passes E21 at 0.92, the same task and optimiser as the step-1000 failure): its "
                            "excess X, R and J lie within the band (P0); stage B's: the Adam twins stay passing (P4)",
    },
    "gates": [
        {"id": "W0_pile", "rule": "faith_score.w0_check: W0's seed-0 PGD KL within 10% of 0.69, CI-masked within 2% of "
                                  "0.348, floor <= 0.25; per stage-A shard, first", "if_fail": "stop the shard"},
        {"id": "G0_W0_reproduces_E20", "rule": f"W0 seed 0: F_all, F_orig, F_dead, F_dense, F_rest within {G0_TOL} relative "
                                               "of E20 amendment 1's W0; F_adv, F_full <= 1e-6; |F_both - F_orig| <= 1e-6; "
                                               "|F_hooks_all - F_all| <= 1e-4; closure G1 ok", "if_fail": "stop the shard"},
        {"id": "C1_reproduces_E21", "rule": f"per state: the harness's adversarial ratio (256 rows, seed 0, this job's W0) "
                                            "gives E21's verdict under faith_score.verdict with this job's floor, and is "
                                            f"within {C1_TOL:.0%} of E21's ratio", "if_fail": "state skipped; the control: stop"},
        {"id": "G1_hooks", "rule": f"per model: E20's closure (hooks at zero <= {1e-6}, at the adversary's fractions <= "
                                   f"{1e-4} nats on the first batch) and E17's identity |F_full - kl_to_w0| <= {FULL_TOL}",
         "if_fail": "stop the shard"},
        {"id": "C3_budget_reproduces", "rule": f"stage B: each model's seed-0 KL at 20 steps within {C3_TOL} relative of "
                                               "stage A's F_all (W0: shard 0's)", "if_fail": "stop the shard"},
    ],
    "predictions": [
        {"id": "P0_control_no_excess", "arm": CONTROL, "scored": True,
         "pass": "|X|, |R| and |J| each <= band (f x F_all(W0)), absolute nats; measured first in shard 0",
         "null": "a passing state shows an excess: the decomposition reads something other than the verdict",
         "predicted": "pass: E21's ratio 0.92 puts X near -0.05 nats; E20's passing states had R and J under 0.12",
         "if_fail": "shard 0 stops; P1, P2 unscored"},
        {"id": "P1_E20_account", "arm": list(FAILING), "scored": True, "kind": "two-sided question",
         "rule": f"per failing state, E20's account holds iff X > band, R >= {SHARE} X, amendment 1's reading is '4' "
                 f"(J >= 0.5 R) and |D| <= {DENSE_SHARE} X (Delta' negligible)",
         "decision": "holds on >= 4 of 5 -> E20's account (the excess is the joint removal of alive and dead blocks, not "
                     "the originals, the changes or Delta') generalises to SGD and the key-to-value failures; on <= 1 -> "
                     "it is particular to the Adam step-10 states; 2-3 -> mixed, per state",
         "lean": "holds, low confidence: E20 read '4' on all eight states, passing ones included"},
        {"id": "P2_E17_readings", "arm": list(FAILING), "scored": True, "kind": "two-sided question",
         "rule": f"per failing state, E17's readings hold iff kappa >= 1 (null within 0.1 of 1), h <= 0.5 and F_adv <= "
                 f"{ADV_SHARE} X",
         "decision": "holds on >= 4 of 5 -> no cancellation among the changes on these failures either; <= 1 -> the SGD "
                     "failures are cancellation-like, unlike Adam's",
         "lean": "holds, low confidence: SGD's component training is near its adjoint split (cosine 0.957), whose kappa "
                 "was 2.4 and h 0.32 in E17"},
        {"id": "P3_sgd_stays_failing", "arm": list(SGD_1000), "scored": True,
         "pass": f"r(s, n) >= {STAY_FAILING} at every n in {list(BUDGETS[1:])} on both SGD step-1000 states (absolute ratio; "
                 "amendment 3's P9 twin)",
         "null": "the SGD ratios fall toward Adam's as the search gets stronger: the late SGD fail is a budget artefact",
         "predicted": "pass, moderate confidence: amendment 2 kept r1 t534 (Adam step 10) at 2.3 to 2.7 times W0 to 100 steps"},
        {"id": "P4_adam_stays_passing", "arm": list(ADAM_1000), "scored": True,
         "pass": f"r(s, n) <= {STAY_PASSING} at every n in {list(BUDGETS[1:])} on both Adam step-1000 states (absolute ratio; "
                 "amendment 3's P10 twin; the positive control of stage B)",
         "null": "a passing state starts failing under a stronger search",
         "predicted": "pass, moderate confidence: E21 read 1.07 and 0.95 at 20 steps"},
        {"id": "P5_ratio_stable", "arm": [*SGD_1000, *ADAM_1000], "scored": True,
         "pass": f"max over n of r(s, n) / min over n <= {STABLE} on >= 3 of the 4 step-1000 states (relative; P11 twin)",
         "null": "a verdict depends on the budget it was read at",
         "predicted": "pass, low confidence"},
    ],
    "reported_not_scored": "every part on all states, the Adam twins' parts beside the SGD failures; null_parts; levels; "
                           "kappa, h, F_adv / F_perm per state; KL per model and budget; W0 seed spread per budget; "
                           "r(s, n) for every budget",
}


def log_to(path: Path) -> Any:
    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(line + "\n")
    return log


def _default(o: Any) -> Any:
    import torch

    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(type(o))


def _clean(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True, default=_default))


def e21_ratio(ref: tuple[Any, ...]) -> tuple[float, float]:
    """E21's adversarial ratio of a state and its job's W0 floor, from the result JSONs (in git)."""
    if ref[0] == "long":
        d = json.loads((LONG / f"sgd_ladder_long_{ref[1]}.json").read_text())
        return d["states"][ref[2]][str(ref[3])]["pile"]["pgd_kl"], d["W0"]["pgd_floor"]
    d = json.loads((KV / "sgd_kv.json").read_text())
    return d["tasks"][ref[1]]["arms"][ref[2]]["pile"]["pgd_kl"], d["pgd_floor"][ref[1]]


# ----------------------------------------------------------------------------- the model and one state


def load(device: str, log: Any, pile: bool) -> dict[str, Any]:
    from commute import component_training, saved_states, vpd
    from commute.data import batches, pile_rows
    from commute.faithfulness import pile_blocks

    model = vpd.load_component_model(device)
    alive = saved_states.load_alive(ROUND1)
    uv = component_training.alive_uv(model, alive)
    return {"model": model, "alive": alive, "uv": uv, "uv_cpu": {k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()},
            "W0": vpd.target_weights(model), "batches": batches(pile_rows("val", 0, ROWS, log), BATCH),
            "blocks": pile_blocks(PILE_ROWS, log) if pile else []}


def at_state(ctx: dict[str, Any], update: Any) -> Any:
    """The model set to a state (update None: W0) for a `with` block."""
    from commute.component_training import components_with
    from commute.faithfulness import model_state

    if update is None:
        return model_state(ctx["model"])
    dW, uv2 = update
    return model_state(ctx["model"], target={k: ctx["W0"][k] + dW[k].to(ctx["W0"][k].device) for k in ctx["W0"]},
                       components=components_with(ctx["model"], ctx["alive"], uv2))


def state(ctx: dict[str, Any], name: str) -> Any:
    from commute import saved_states

    return saved_states.offsets_state(STATES[name][0], ctx["uv_cpu"])


def short(rec: dict[str, Any]) -> str:
    keys = ("pgd_kl", "F_all", "F_orig", "F_adv", "F_both", "F_dead", "F_dense", "F_rest", "F_full", "kl_to_w0")
    return " ".join(f"{k} {rec[k]:.5f}" for k in keys if k in rec)


# ----------------------------------------------------------------------------- stage A


def gate_g0(w0: dict[str, Any]) -> dict[str, Any]:
    """W0 seed 0 reproduces E20 amendment 1's W0 (same rows, adversary and code path) and its exact identities."""
    ref = json.loads(E20_A1.read_text())["W0"]
    rel = {k: abs(w0[k] - ref[k]) / abs(ref[k]) for k in ("F_all", "F_orig", "F_dead", "F_dense", "F_rest")}
    ok = (w0["G1"]["ok"] and all(v <= G0_TOL for v in rel.values()) and w0["F_adv"] <= 1e-6 and w0["F_full"] <= 1e-6
          and abs(w0["F_both"] - w0["F_orig"]) <= 1e-6 and abs(w0["F_hooks_all"] - w0["F_all"]) <= 1e-4)
    return {"rel": rel, "ok": ok}


def gate_c1(ratio: float, f: float, ref: tuple[Any, ...]) -> dict[str, Any]:
    from commute.faith_score import verdict

    r21, f21 = e21_ratio(ref)
    rec = {"ratio": ratio, "verdict": verdict(ratio, f), "e21_ratio": r21, "e21_verdict": verdict(r21, f21),
           "rel": ratio / r21 - 1}
    rec["ok"] = rec["verdict"] == rec["e21_verdict"] and abs(rec["rel"]) <= C1_TOL
    return rec


def p0(d: dict[str, Any], band: float) -> dict[str, Any]:
    """The positive control: the passing state shows no excess, and none in R or J, beyond f x F_all(W0)."""
    return {"band": band, **{k: d[k] for k in ("X", "R", "J")},
            "verdict": "pass" if all(abs(d[k]) <= band for k in ("X", "R", "J")) else "fail"}


def stage_revert(device: str, shard: int) -> None:
    import torch

    from commute import excess, faith_score
    from commute.faithfulness import pile_faithfulness

    assert PRED.is_file(), "file predictions first: run.py --predictions"
    out_path, log = RESULTS / f"revert_shard{shard}.json", log_to(LARGE / f"revert_shard{shard}.log")
    t0 = time.time()
    ctx = load(device, log, pile=True)
    m = ctx["model"]
    pile = {s: pile_faithfulness(m, ctx["blocks"], s, 20, log) for s in PILE_SEEDS}
    out: dict[str, Any] = {"W0_pile": faith_score.w0_check(pile), "states": {}, "C1": {}, "parts": {}}
    f, w0_pgd = out["W0_pile"]["pgd_floor"], pile[0]["pgd_kl"]
    log(f"[W0 pile] floor {f:.4f} seed-0 pgd {w0_pgd:.4f} pass {out['W0_pile']['pass']} ({time.time() - t0:.0f} s)")
    dump(out_path, out)
    if not out["W0_pile"]["pass"]:
        sys.exit("W0 is not at its known Pile values: nothing run")
    (LARGE / "sources").mkdir(parents=True, exist_ok=True)
    w0, src0 = excess.measure(m, ctx["batches"], ctx["alive"], ctx["uv"], ctx["W0"], None, 0, None)
    out["W0"], out["G0"] = w0, gate_g0(w0)
    torch.save(src0, LARGE / "sources" / f"shard{shard}_W0_s0.pt")
    log(f"[W0 s0] {short(w0)} | G1 {w0['G1']} | G0 {out['G0']} ({time.time() - t0:.0f} s)")
    dump(out_path, out)
    if not out["G0"]["ok"]:
        sys.exit(f"G0 failed: W0 does not reproduce E20 amendment 1: {out['G0']}")
    out["W0_s1"], _ = excess.measure(m, ctx["batches"], ctx["alive"], ctx["uv"], ctx["W0"], None, 1, src0)
    out["null_parts"] = excess.parts(out["W0_s1"], w0)
    log(f"[W0 s1] {short(out['W0_s1'])} | null parts {json.dumps({k: round(v, 4) for k, v in out['null_parts'].items() if isinstance(v, float)})}")
    dump(out_path, out)
    band = f * w0["F_all"]
    for name in A_SHARDS[shard]:
        upd = state(ctx, name)
        with at_state(ctx, upd):
            ratio = pile_faithfulness(m, ctx["blocks"], 0, 20, log)["pgd_kl"] / w0_pgd
        out["C1"][name] = gate_c1(ratio, f, STATES[name][1])
        log(f"[{name}] C1 {out['C1'][name]} ({time.time() - t0:.0f} s)")
        dump(out_path, out)
        if not out["C1"][name]["ok"]:
            if name == CONTROL:
                sys.exit(f"C1 failed on the control {name}")
            continue
        rec, src = excess.measure(m, ctx["batches"], ctx["alive"], ctx["uv"], ctx["W0"], upd, 0, src0)
        rec["G1"]["full_minus_kl_to_w0"] = rec.get("F_full", float("nan")) - rec["kl_to_w0"]
        rec["G1"]["ok"] = rec["G1"]["ok"] and abs(rec["G1"]["full_minus_kl_to_w0"]) <= FULL_TOL
        out["states"][name] = rec
        torch.save(src, LARGE / "sources" / f"{name.replace(' ', '_').replace('@', 'step')}.pt")
        dump(out_path, out)
        if not rec["G1"]["ok"]:
            sys.exit(f"G1 failed on {name}: {rec['G1']}")
        d = out["parts"][name] = excess.parts(rec, w0)
        c = rec["coherence"]
        log(f"[{name}] {short(rec)} | kappa {c['pooled']:.3f} (null {c['pooled_null']:.3f}) h {d['h']:.3f} "
            f"F_adv/perm {d['adv_over_perm']:.2f}")
        log(f"[{name}] X {d['X']:.4f} = O {d['O']:.4f} + F_adv {d['F_adv']:.4f} + I {d['I']:.4f} + R {d['R']:.4f}; "
            f"R = D {d['D']:.4f} + K {d['K']:.4f} + P {d['P']:.4f} + J {d['J']:.4f} | reading {d['reading']} "
            f"({time.time() - t0:.0f} s)")
        if name == CONTROL:
            out["P0"] = p0(d, band)
            log(f"[P0] {out['P0']}")
        dump(out_path, out)
        if name == CONTROL and out["P0"]["verdict"] != "pass":
            sys.exit(f"P0 failed: the passing control shows an excess: {out['P0']}")
    out["seconds"] = time.time() - t0
    dump(out_path, out)
    log(f"done in {out['seconds']:.0f} s -> {out_path}")


# ----------------------------------------------------------------------------- stage B


def stage_budget(device: str, shard: int) -> None:
    import torch

    from commute import cancel, gamma

    a0 = json.loads((RESULTS / "revert_shard0.json").read_text())
    assert a0.get("P0", {}).get("verdict") == "pass" and all(s in a0["states"] for s in (*SGD_1000, *ADAM_1000)), \
        "stage A shard 0 must finish with its control passing first"
    out_path, log = RESULTS / f"budget_shard{shard}.json", log_to(LARGE / f"budget_shard{shard}.log")
    t0 = time.time()
    ctx = load(device, log, pile=False)
    out: dict[str, Any] = {"kl": {}, "C3": {}}
    (LARGE / "sources_budget").mkdir(parents=True, exist_ok=True)
    for name, seed, budgets in B_SHARDS[shard]:
        upd = None if name == "W0" else state(ctx, name)
        out["kl"].setdefault(name, {})[str(seed)] = {}
        with at_state(ctx, upd):
            for n in budgets:
                src, _ = gamma.pgd_sources(ctx["model"], ctx["batches"], n, step_size=2 / n, seed=seed)
                kl = cancel.masked_kl(ctx["model"], ctx["batches"], src)
                torch.save(src, LARGE / "sources_budget" / f"{name.replace(' ', '_').replace('@', 'step')}_s{seed}_{n}.pt")
                out["kl"][name][str(seed)][str(n)] = kl
                gate = None
                if (n, seed) == (20, 0):
                    ref = (a0["W0"] if name == "W0" else a0["states"][name])["F_all"]
                    gate = out["C3"][name] = {"ref": ref, "rel": abs(kl - ref) / ref, "ok": abs(kl - ref) <= C3_TOL * ref}
                dump(out_path, out)
                log(f"[{name}] seed {seed} {n} steps of {2 / n:.4f}: KL (fp32) {kl:.5f}" + (f" | C3 {gate}" if gate else "")
                    + f" ({time.time() - t0:.0f} s)")
                if gate is not None and not gate["ok"]:
                    sys.exit(f"C3 failed on {name}: {gate}")
    out["seconds"] = time.time() - t0
    dump(out_path, out)
    log(f"done in {out['seconds']:.0f} s -> {out_path}")


# ----------------------------------------------------------------------------- scoring (JSON only)


def e20_account(d: dict[str, Any], band: float) -> bool:
    """E20's account on one state: an excess, most of it in the rest R, R mostly the joint term J, Delta' negligible."""
    return d["X"] > band and d["R"] >= SHARE * d["X"] and d["reading"] == "4" and abs(d["D"]) <= DENSE_SHARE * d["X"]


def e17_reading(rec: dict[str, Any], d: dict[str, Any]) -> bool:
    """E17's readings on one state: the changes add up (kappa >= 1, null within 0.1 of 1), a random half removes less
    than half (h <= 0.5), and the changes' own revert explains little of the excess (F_adv <= 0.2 X)."""
    c = rec["coherence"]
    return abs(c["pooled_null"] - 1) <= 0.1 and c["pooled"] >= 1 and d["h"] <= 0.5 and d["F_adv"] <= ADV_SHARE * d["X"]


def count(per: dict[str, bool], hi: int, lo: int) -> dict[str, Any]:
    n = sum(per.values())
    return {"per_state": per, "n": n, "of": len(per),
            "reading": "holds" if n >= hi else "does not hold" if n <= lo else "mixed"}


def stage_score() -> None:
    a = [json.loads((RESULTS / f"revert_shard{i}.json").read_text()) for i in range(len(A_SHARDS))]
    parts = {n: p for s in a for n, p in s["parts"].items()}
    recs = {n: r for s in a for n, r in s["states"].items()}
    band = {n: s["W0_pile"]["pgd_floor"] * s["W0"]["F_all"] for s in a for n in s["parts"]}
    got = [n for n in FAILING if n in parts]
    sc: dict[str, Any] = {
        "G0": [s["G0"]["ok"] for s in a], "W0_pile": [s["W0_pile"]["pass"] for s in a],
        "C1": {n: r["ok"] for s in a for n, r in s["C1"].items()}, "G1": {n: r["G1"]["ok"] for n, r in recs.items()},
        "P0_control_no_excess": a[0].get("P0", {"verdict": "not run"}),
        "P1_E20_account": count({n: e20_account(parts[n], band[n]) for n in got}, 4, 1) if len(got) == len(FAILING)
        else {"reading": "not run", "missing": [n for n in FAILING if n not in got]},
        "P2_E17_readings": count({n: e17_reading(recs[n], parts[n]) for n in got}, 4, 1) if len(got) == len(FAILING)
        else {"reading": "not run", "missing": [n for n in FAILING if n not in got]},
    }
    out: dict[str, Any] = {"score": sc, "parts": parts, "null_parts": [s["null_parts"] for s in a]}
    b = [RESULTS / f"budget_shard{i}.json" for i in range(len(B_SHARDS))]
    if all(p.is_file() for p in b):
        kl: dict[str, Any] = {}
        for p in b:
            for name, seeds in json.loads(p.read_text())["kl"].items():
                for seed, v in seeds.items():
                    kl.setdefault(name, {}).setdefault(seed, {}).update(v)
        w0 = kl["W0"]["0"]
        r = {m: {n: kl[m]["0"][n] / w0[n] for n in w0} for m in kl if m != "W0"}
        later = [str(n) for n in BUDGETS[1:]]
        spread = {m: max(r[m].values()) / min(r[m].values()) for m in (*SGD_1000, *ADAM_1000)}
        sc["P3_sgd_stays_failing"] = "pass" if all(r[m][n] >= STAY_FAILING for m in SGD_1000 for n in later) else "fail"
        sc["P4_adam_stays_passing"] = "pass" if all(r[m][n] <= STAY_PASSING for m in ADAM_1000 for n in later) else "fail"
        sc["P5_ratio_stable"] = {"verdict": "pass" if sum(v <= STABLE for v in spread.values()) >= 3 else "fail",
                                 "max_over_min": spread}
        sc["W0_seed_spread"] = {n: kl["W0"]["1"][n] / w0[n] for n in w0}
        out.update(kl=kl, ratio=r)
    dump(RESULTS / "cancellation_sgd.json", out)
    print(json.dumps(_clean(sc), indent=1, default=_default))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["revert", "budget", "score"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; predictions are not edited after filing (add a dated report instead)")
        dump(PRED, {**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"filed {PRED}")
    elif args.stage == "revert":
        stage_revert(args.device, args.shard)
    elif args.stage == "budget":
        stage_budget(args.device, args.shard)
    elif args.stage == "score":
        stage_score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
