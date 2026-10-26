"""new_vs_existing (round 7): what in the subcomponents tells a new mechanism from an existing one?

Plan: reports/plans/2026-10-20_plan_new-vs-existing-signatures.md. Four key -> value tasks in one format (src/commute/kv.py):
two "existing" (M x -> x: the model has copying machinery) and two "new" (M x -> pi(x), a random bijection), each a
full-weight fine-tune of the 24 matrices at two seeds, scored at its first checkpoint with held-out P(target) >= P_LEARN.
Per fine-tuned state (src/commute/signatures.py, harness.py, local_faith.py, recruit.py, ladder.py):
  1  outside_share share of delta-W over its isotropic null (and a random delta-W of the same norms as the null's check);
  2  route split of the Euclidean-lifted state (e_delta, r, e_gate; d, i) and of the do-nothing state;
  3  relevance at W0 of T, the TOP_K subcomponents the lift moves most: ablate them and CI-matched random sets at the
     trigger, read the target's log-odds drop and the KL; the same in the fine-tuned (do-nothing) model;
  4  T's mean CI at the trigger at W0 and under the trained activations, the lift energy on switched-off subcomponents,
     T's on-rate over Pile at W0, and its overlap with the "repeat the current token" machinery;
  5  the switched-on and switched-off sets at the trigger (frozen Gamma; |dg| > DG_THR), their sizes and overlaps;
  6  Gamma coverage of the task gradient at W0 (the ladder's number); across seeds the Jaccard of T and of S_on.
Stages (repo root, project venv, GPU):
  --predictions  file results/predictions.json (refuses to overwrite)
  build          Pile rows -> marker, keys, values, windows, baselines -> results/tasks.json, results/large/windows_<task>.pt
  run            fine-tunes and measurements -> results/new_vs_existing.json (dumped after every seed), scored at the end
  all            build, run
  --smoke        any stage at the SMOKE sizes; writes *_smoke.json
Inputs: experiments/transport_test/results/large/alive.pt.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

HALF = 20
FT_LR = 3e-5  # the ladder's recipe
P_LEARN = 0.5  # a fine-tune has learned at the first checkpoint whose held-out P(target) reaches this
GATE = 1e-4  # a random entry's baseline P(value | M key) must be below this ("a behaviour the model lacks")
IDENT_MAX = 0.5  # the identity task needs a gain: baseline P(key | M key) at most this
LIFT_TOL = 1e-6
DG_THR = 0.3  # switched on / off at the trigger: mean g moves by more than this (the recruit test's rule)
G_ON = 0.5  # "on": g above this

PREDICTIONS: dict[str, Any] = {
    "filed": "2026-10-20",
    "plan": "reports/plans/2026-10-20_plan_new-vs-existing-signatures.md",
    "classes": {"existing": ["ident32", "ident32b"], "new": ["rand32", "rand32b"]},
    "scored_step_rule": "per fine-tune, the first checkpoint (10, 30, 100, 300) with held-out P(target) >= 0.5; else the "
                        "last checkpoint, flagged 'did not learn'. The scored step and ||delta-W||/||W0|| are recorded "
                        "beside every number (a step difference between the classes is a confound to check, not a match)",
    "separation_rule": "a number separates the classes when the two classes' ranges (one value per fine-tuned state: 4 per "
                       "class; for the across-seed numbers one per task: 2 per class) do not overlap AND the gap between "
                       "the ranges exceeds the larger within-class range; the predicted direction must hold",
    "G1_all_learn": {"rule": "every fine-tune (4 tasks x 2 seeds) reaches held-out P(target) >= 0.5 within 300 steps; a state "
                             "that does not is dropped from every class range and the round is flagged", "predicted": "pass"},
    "G2_lift_exact": {"rule": "the lifted state's target model has the fine-tune's held-out P(target) to 1e-3 and its route "
                              "shares sum to 1 to 1e-6, on every state", "predicted": "pass"},
    "G3_random_null": {"rule": "a Gaussian delta-W with the fine-tune's per-matrix norms reads pooled outside_share share / "
                               "isotropic null within [0.9, 1.1] on every state", "predicted": "pass"},
    "P1_inexpressible": {
        "number": "pooled input-weighted outside_share share of the fine-tune's delta-W / its isotropic null (harness.outside_share_of)",
        "rule": "separates, existing lower (expected existing <= 0.7, new >= 0.85)",
        "predicted": "pass",
        "blind_to": "whether the expressible part does anything (round 2: the outside_share part alone also partly carries the behaviour)",
        "control": "G3's random delta-W reads 1"},
    "P2_residual_share": {
        "number": "e_delta of the Euclidean-lifted state: the share of the held-out log-odds gain at the trigger that stays in Delta",
        "rule": "does NOT separate (the key-value round read 0.13 identity, 0.18 random, 0.13 one-pair random)",
        "predicted": "pass (no separation)",
        "blind_to": "everything away from the trigger",
        "control": "do-nothing reads e_delta = 1 by construction"},
    "P3_relevance_W0": {
        "number": "k_lo = |mean drop of W0's target log-odds at the trigger when T (the 100 subcomponents the lift moves most) is "
                  "ablated| / mean |drop| over 5 random sets matched per matrix on size and on the decile of CI at the trigger",
        "rule": "separates, existing higher (expected existing k_lo >= 3, new within 0.5-2). Indicative: depends on |T| = 100 "
                "and the matching, both fixed here",
        "predicted": "pass",
        "blind_to": "for the new tasks W0's target probability is ~1e-6, so k_lo near 1 is also what no signal reads; the "
                    "random sets' spread is the floor. Reported beside: the KL version (k_kl), and the same ablations in the "
                    "fine-tuned model (does T carry the trained behaviour)",
        "control": "the random sets give 1 by construction"},
    "P4_route": {
        "number": "r (re-switching share) and e_gate (masked-off share) of the lifted state",
        "rule": "both separate: r existing higher (expected >= 0.6 vs <= 0.35), e_gate new higher (expected >= 0.45 vs <= 0.25). "
                "The positive control of the round: if this does not separate on the new seeds and tasks, the round has no power",
        "predicted": "pass",
        "blind_to": "the update itself; it reads how frozen Gamma routes the trained model at the trigger",
        "control": "the key-value round's 0.80 / 0.06 (identity) and 0.27 / 0.52 (random)"},
    "P5_seed_consistency": {
        "number": "Jaccard of T between the two fine-tune seeds of a task (2 values per class)",
        "rule": "separates, existing higher (expected existing >= 0.5); Jaccard of S_on reported beside",
        "predicted": "pass",
        "blind_to": "what produces the behaviour: it compares which subcomponents moved most (the recruit test's lesson)",
        "control": "two random 100-sets read 0.005"},
    "reported_not_scored": ["T's mean CI at the trigger at W0 and under the trained activations (existing higher expected)",
                            "Gamma coverage of the task gradient at W0 (existing higher expected; the ladder found it does not predict d)",
                            "lift energy on subcomponents switched off at the trigger in the trained model (new higher expected)",
                            "T's and S_on's on-rate over Pile at W0 against the alive mean (new tasks' T rarer expected)",
                            ("identity tasks: Jaccard of S_on and T with the top-100 'repeat the current token' subcomponents on Pile "
                             "(>= 0.2 expected against ~0.01 for a random set)"),
                            "sizes of S_on and S_off and mean |dg| at the trigger", "do-nothing route split (r only; the G3-style identity check)"],
}


@dataclass(frozen=True)
class Task:
    name: str
    cls: str  # "existing" | "new"
    key_rank0: int  # keys: word-like tokens by frequency rank, from this rank
    rule: str  # "identity" | "bijection"
    bij_seed: int


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run, in one place."""

    rows: int  # Pile train rows: token frequencies, the marker, and the windows
    n_windows: int  # per task, split 4:1 train:held-out by source row
    K: int
    value_rank0: int  # the value pool starts at this rank (mid-frequency words pass the baseline gate)
    pool: int
    tasks: tuple[Task, ...]
    steps: int
    checkpoints: tuple[int, ...]
    seeds: tuple[int, ...]
    top_k: int  # |T|
    n_random: int  # CI-matched random sets per state
    pile_rows: int  # Pile val rows for the on-rate and the repeat contrast
    suffix: str

    def __post_init__(self) -> None:
        assert max(self.checkpoints) == self.steps and self.K <= self.pool
        assert all(t.key_rank0 + self.K <= self.value_rank0 for t in self.tasks)
        assert sorted(t.cls for t in self.tasks) == ["existing", "existing", "new", "new"]


FULL = Sizes(rows=32768, n_windows=500, K=32, value_rank0=1000, pool=2000,
             tasks=(Task("ident32", "existing", 100, "identity", 0), Task("ident32b", "existing", 132, "identity", 0),
                    Task("rand32", "new", 100, "bijection", 0), Task("rand32b", "new", 132, "bijection", 1)),
             steps=300, checkpoints=(10, 30, 100, 300), seeds=(0, 1), top_k=100, n_random=5, pile_rows=64, suffix="")
SMOKE = Sizes(rows=64, n_windows=16, K=2, value_rank0=20, pool=40,
              tasks=(Task("ident2", "existing", 5, "identity", 0), Task("ident2b", "existing", 7, "identity", 0),
                     Task("rand2", "new", 5, "bijection", 0), Task("rand2b", "new", 7, "bijection", 1)),
              steps=2, checkpoints=(1, 2), seeds=(0, 1), top_k=10, n_random=2, pile_rows=2, suffix="_smoke")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, Tensor):
        return o.tolist()
    return float(o)


def file_predictions() -> None:
    assert not PRED.is_file(), f"{PRED} exists; predictions are never edited after filing (add a dated report)"
    dump(PRED.name, PREDICTIONS)
    log(f"filed {PRED}")


def require_predictions() -> dict[str, Any]:
    assert PRED.is_file(), f"file the predictions first ({PRED})"
    return json.loads(PRED.read_text())


# ----------------------------------------------------------------------------- shared pieces


def load_all(device: str) -> dict[str, Any]:
    from commute import component_training, split, vpd

    model = vpd.load_component_model(device)
    alive = {k: v.cpu().bool() for k, v in torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"].items()}
    uv = component_training.alive_uv(model, alive)
    uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()}
    W0 = vpd.target_weights(model)
    t0 = time.time()
    bases = split.bases(uv_cpu, LIFT_TOL)
    log(f"lift bases in {time.time() - t0:.0f} s")
    return {"device": device, "model": model, "W0": W0, "comps0": vpd.component_weights(model), "alive": alive, "uv": uv,
            "uv_cpu": uv_cpu, "bases": bases,
            "non_saturated": [k for k in W0 if bases[k].r_u < W0[k].shape[0] and bases[k].r_v < W0[k].shape[1]]}


def logits_fn(ctx: dict[str, Any]) -> Any:
    from commute.finetune import logits_of

    return lambda toks: logits_of(ctx["model"].target_model, toks)


def scored_step(p_at: dict[int, float]) -> tuple[int, bool]:
    """The first checkpoint with held-out P(target) >= P_LEARN; else the last, with learned = False."""
    for step in sorted(p_at):
        if p_at[step] >= P_LEARN:
            return step, True
    return max(p_at), False


def sel_json(sel: dict[str, Tensor]) -> list[list[Any]]:
    return [[k, int(c)] for k, v in sel.items() for c in v.tolist()]


# ----------------------------------------------------------------------------- stage: build


def stage_build(S: Sizes, device: str) -> None:
    from commute import kv, ladder, vpd
    from commute.data import pile_rows, split_windows

    require_predictions()
    ctx = load_all(device)
    tok = vpd.tokenizer()
    lf = logits_fn(ctx)
    rows = pile_rows("train", 0, S.rows, log)
    with torch.no_grad():
        vocab = lf(rows[:1].to(device)).shape[-1]
    plain = ladder.plain_mask(tok.decode, vocab)
    counts = ladder.count_tokens(rows, HALF)
    marker = kv.pick_marker(counts, tok.decode, vocab)
    ranked = [t for t, _ in sorted(counts.items(), key=lambda tc: -tc[1]) if t < vocab and bool(plain[t])]
    pool = ranked[S.value_rank0 : S.value_rank0 + S.pool]
    out: dict[str, Any] = {"n_rows": S.rows, "marker": marker, "marker_str": tok.decode([marker]),
                           "marker_count": counts.get(marker, 0), "vocab": vocab, "tasks": {}}
    log(f"build: marker {marker} {out['marker_str']!r}; {len(ranked)} word-like tokens ranked")
    LARGE.mkdir(parents=True, exist_ok=True)
    for t in S.tasks:
        keys = tuple(ranked[t.key_rank0 : t.key_rank0 + S.K])
        w = kv.windows(rows, marker, keys, HALF, S.n_windows, seed=0)  # same contexts for tasks on the same keys
        dist = kv.key_distributions(lf, w, device)
        ident = {k: float(dist[k][k]) for k in keys}
        assert max(ident.values()) <= IDENT_MAX, f"identity baseline {max(ident.values()):.3f} > {IDENT_MAX}: raise key_rank0"
        task = (kv.identity(t.name, marker, keys) if t.rule == "identity"
                else kv.bijection(t.name, marker, keys, pool, seed=t.bij_seed, ok=lambda k, v, d=dist: float(d[k][v]) < GATE))
        baseline = {k: float(dist[k][v]) for k, v in zip(task.keys, task.values)}
        train, held = split_windows(w)
        assert len(held) >= 1 and len(train) >= 1, (len(train), len(held))
        rec = {"cls": t.cls, "rule": t.rule, "K": S.K, "keys": list(task.keys), "values": list(task.values),
               "entries": [f"{tok.decode([k])!r} -> {tok.decode([v])!r}" for k, v in zip(task.keys, task.values)],
               "baseline": baseline, "baseline_max": max(baseline.values()), "identity_baseline": ident,
               "n_train": len(train), "n_heldout": len(held),
               "heldout_keys_missing": sorted(set(task.keys) - set(held.tokens[:, HALF].tolist()))}
        torch.save({"train": train, "heldout": held, "task": task, "table": task.table(vocab)}, LARGE / f"windows_{t.name}.pt")
        out["tasks"][t.name] = rec
        log(f"build: {t.name} ({t.cls}, {t.rule}): baseline max {rec['baseline_max']:.2e}, identity max {max(ident.values()):.3f}, "
            f"train {len(train)} held-out {len(held)}; e.g. {rec['entries'][:3]}")
    dump(f"tasks{S.suffix}.json", out)


# ----------------------------------------------------------------------------- stage: run


def task_context(ctx: dict[str, Any], w: dict[str, Any]) -> dict[str, Any]:
    """Everything at W0 that every seed of a task shares: the route-split base cells and gates, W0's trigger logits and
    mean CI at the trigger, the input second moments over the train windows, and Gamma coverage of the task gradient."""
    from commute import ladder, split
    from commute import signatures as sg
    from commute.data import batches
    from commute.local_faith import route_cells

    model, train, held, table = ctx["model"], w["train"], w["heldout"], w["table"]
    base = route_cells(model, held.tokens, held.half, table)
    sigma = {k: v.cpu() for k, v in split.input_second_moments(model, batches(train.tokens, 32), list(ctx["W0"])).items()}
    G = ladder.trigger_gradient(model, train.tokens, HALF, table)
    gb = ladder.trigger_gbar(model, ctx["alive"], train.tokens, HALF)
    cov = ladder.coverage_from(G, ctx["uv_cpu"], gb["gbar"])
    return {"base": base, "ref0": sg.trigger_logits(model, held.tokens, held.half), "g0": sg.trigger_g(base["gates"], held.half),
            "sigma": sigma, "coverage": {"coverage": cov["coverage"], "l0_trigger": gb["l0_trigger"], "max_gbar": gb["max_gbar"]}}


def relevance(ctx: dict[str, Any], tgt: dict[str, Tensor], ref: Tensor, held: Any, table: Tensor, T: dict[str, Tensor],
              rands: list[dict[str, Tensor]]) -> dict[str, Any]:
    """Ablate T and each random set from the released components in a model with target weights `tgt` (W0, or the
    fine-tune's do-nothing state); `ref` = that model's unablated trigger logits. k_lo and k_kl are T's effect over the
    random sets' mean."""
    from commute import signatures as sg
    from commute.faithfulness import model_state

    def one(sel: dict[str, Tensor]) -> dict[str, float]:
        with model_state(ctx["model"], target=sg.ablated_weights(tgt, ctx["comps0"], sel)):
            return sg.trigger_stats(ref, sg.trigger_logits(ctx["model"], held.tokens, held.half), held.tokens, held.half, table)

    t, rs = one(T), [one(r) for r in rands]
    mean_abs_drop = sum(abs(r["drop"]) for r in rs) / len(rs)
    mean_kl = sum(r["kl"] for r in rs) / len(rs)
    return {"T": t, "random": rs, "k_lo": abs(t["drop"]) / mean_abs_drop if mean_abs_drop else float("inf"),
            "k_kl": t["kl"] / mean_kl if mean_kl else float("inf")}


def measure_state(ctx: dict[str, Any], tc: dict[str, Any], w: dict[str, Any], dW: dict[str, Tensor], p_o: float, seed: int,
                  S: Sizes, pile: dict[str, Any]) -> dict[str, Any]:
    """Numbers 1-5 of the docstring for one fine-tuned state."""
    from commute import component_training, recruit, split
    from commute import signatures as sg
    from commute.faithfulness import model_state
    from commute.finetune import random_direction
    from commute.harness import outside_share_of
    from commute.local_faith import route_cells, route_split

    model, W0, alive, held, table = ctx["model"], ctx["W0"], ctx["alive"], w["heldout"], w["table"]
    rec: dict[str, Any] = {}
    inex = outside_share_of(dW, ctx["bases"], tc["sigma"], ctx["non_saturated"])
    rnd = outside_share_of(random_direction(dW, seed), ctx["bases"], tc["sigma"], ctx["non_saturated"])
    rec["inexpressible"] = {"pooled": inex["pooled"], "null": inex["pooled_null"], "ratio": inex["pooled"] / inex["pooled_null"],
                            "random_ratio": rnd["pooled"] / rnd["pooled_null"], "per_matrix": inex["per_matrix"]}
    tgt = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
    with model_state(model, target=tgt):
        cur = route_cells(model, held.tokens, held.half, table)
        rec["routes_nothing"] = route_split(model, held.tokens, held.half, table, tc["base"], ctx["comps0"])
        ref_ft = sg.trigger_logits(model, held.tokens, held.half)
    st = split.split_state(dW, ctx["uv_cpu"], ctx["bases"])
    comps = component_training.components_with(model, alive, {k: (v[1], v[2]) for k, v in st.items()})
    with model_state(model, target=tgt, components=comps):
        rec["routes_lift"] = route_split(model, held.tokens, held.half, table, tc["base"], ctx["comps0"])
    rl = rec["routes_lift"]
    rec["G2"] = {"p_o_diff": abs(rl["p_cells"]["T1"] - p_o), "share_sum": rl["share_sum"]}
    rec["G2"]["ok"] = rec["G2"]["p_o_diff"] <= 1e-3 and abs(rl["share_sum"] - 1) <= 1e-6
    # where the update landed, and the gates
    energy = sg.split_energy(dW, ctx["uv_cpu"], ctx["bases"], alive)
    T = sg.top_by(energy, alive, S.top_k)
    g1 = sg.trigger_g(cur["gates"], held.half)
    dg = recruit.trigger_dg(cur["gates"], tc["base"]["gates"], held.half, alive)
    S_on, S_off = recruit.switched(dg, DG_THR, True), recruit.switched(dg, DG_THR, False)
    off_now = {k: ((g1[k] < G_ON) & alive[k]).nonzero().flatten() for k in alive}
    repeat_top = sg.top_by(pile["repeat_contrast"], alive, S.top_k)
    rec["T"] = {"members": sel_json(T), "energy_share": sg.share_in(energy, T), "g0_mean": sg.mean_over(tc["g0"], T),
                "g1_mean": sg.mean_over(g1, T), "energy_share_off_at_trigger": sg.share_in(energy, off_now),
                "pile_on_rate_mean": sg.mean_over(pile["on_rate"], T), "jaccard_repeat_top": recruit.jaccard(T, repeat_top),
                "jaccard_S_on": recruit.jaccard(T, S_on)}
    rec["S_on"] = {"members": sel_json(S_on), "size": recruit.size(S_on), "size_off": recruit.size(S_off),
                   "mean_abs_dg": sg.mean_over({k: v.abs() for k, v in dg.items()}, {k: alive[k].nonzero().flatten() for k in alive}),
                   "pile_on_rate_mean": sg.mean_over(pile["on_rate"], S_on), "jaccard_repeat_top": recruit.jaccard(S_on, repeat_top)}
    rands = [sg.matched_random(T, alive, tc["g0"], seed=1000 * seed + i) for i in range(S.n_random)]
    rec["relevance_W0"] = relevance(ctx, W0, tc["ref0"], held, table, T, rands)
    rec["relevance_ft"] = relevance(ctx, tgt, ref_ft, held, table, T, rands)
    s = rl["shares"]
    log(f"  outside_share/null {rec['inexpressible']['ratio']:.3f} (random {rec['inexpressible']['random_ratio']:.3f}) | lift "
        f"d {s['d']:.3f} r {s['r']:.3f} e_gate {s['e_gate']:.3f} e_delta {s['e_delta']:.3f} | G2 {rec['G2']['ok']} | T: energy "
        f"{rec['T']['energy_share']:.3f} g0 {rec['T']['g0_mean']:.3f} g1 {rec['T']['g1_mean']:.3f} off-share "
        f"{rec['T']['energy_share_off_at_trigger']:.3f} | S_on {rec['S_on']['size']} S_off {rec['S_on']['size_off']} | relevance W0 "
        f"k_lo {rec['relevance_W0']['k_lo']:.2f} (drop {rec['relevance_W0']['T']['drop']:.3f}) k_kl {rec['relevance_W0']['k_kl']:.2f}"
        f" | ft k_lo {rec['relevance_ft']['k_lo']:.2f}")
    return rec


def run_task(ctx: dict[str, Any], t: Task, w: dict[str, Any], S: Sizes, pile: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    from commute import recruit
    from commute.finetune import ArmAConfig, finetune_arm_a

    train, held, table = w["train"], w["heldout"], w["table"]
    t0 = time.time()
    tc = task_context(ctx, w)
    rec: dict[str, Any] = {"cls": t.cls, "rule": t.rule, "coverage": tc["coverage"], "baseline_lo_trigger": tc["base"]["T"],
                           "n_train": len(train), "n_heldout": len(held), "seeds": {}}
    log(f"task {t.name} ({t.cls}): coverage {tc['coverage']['coverage']:.3f}, W0 trigger log-odds {tc['base']['T']:.2f}")
    (LARGE / t.name).mkdir(parents=True, exist_ok=True)
    for seed in S.seeds:
        cfg = ArmAConfig(o_token=table, lr=FT_LR, steps=S.steps, checkpoints=S.checkpoints, seed=seed)
        ft = finetune_arm_a(ctx["model"], train, held, cfg, ctx["device"], log)
        step, learned = scored_step({s: e["p_o"] for s, e in ft.evals.items()})
        dW = ft.deltas[step]
        torch.save(dW, LARGE / t.name / f"dW_ft_seed{seed}_step{step}.pt")
        ev = ft.evals[step]
        log(f" seed {seed}: learned {learned} at step {step}, P {ev['p_o']:.3f}, rel norm {ev['rel_norm_max']:.4f}, kl_other {ev['kl_other']:.3f}")
        rec["seeds"][str(seed)] = {"scored_step": step, "learned": learned, "p_o": ev["p_o"], "rel_norm_max": ev["rel_norm_max"],
                                   "kl_other": ev["kl_other"], "evals": ft.evals,
                                   **measure_state(ctx, tc, w, dW, ev["p_o"], seed, S, pile)}
        out["tasks"][t.name] = rec
        dump(f"new_vs_existing{S.suffix}.json", out)
    sets = [rec["seeds"][str(s)] for s in S.seeds]
    T_sets = [{k: torch.tensor([c for m, c in r["T"]["members"] if m == k], dtype=torch.long) for k in ctx["alive"]} for r in sets]
    on_sets = [{k: torch.tensor([c for m, c in r["S_on"]["members"] if m == k], dtype=torch.long) for k in ctx["alive"]} for r in sets]
    rec["across_seeds"] = {"jaccard_T": recruit.jaccard(T_sets[0], T_sets[1]), "jaccard_S_on": recruit.jaccard(on_sets[0], on_sets[1]),
                           "jaccard_random_T": recruit.jaccard(recruit.random_like(T_sets[0], ctx["alive"], 0),
                                                               recruit.random_like(T_sets[0], ctx["alive"], 1))}
    rec["seconds"] = time.time() - t0
    log(f"task {t.name}: Jaccard across seeds T {rec['across_seeds']['jaccard_T']:.3f} S_on {rec['across_seeds']['jaccard_S_on']:.3f} "
        f"(random {rec['across_seeds']['jaccard_random_T']:.4f}) | {rec['seconds']:.0f} s")
    return rec


def separation(existing: list[float], new: list[float], higher: str) -> dict[str, Any]:
    """The filed rule: non-overlapping class ranges with a gap above the larger within-class range, in the predicted
    direction (`higher` = the class predicted to read higher)."""
    if not existing or not new:
        return {"existing": existing, "new": new, "separates": None, "direction_ok": None}
    lo_e, hi_e, lo_n, hi_n = min(existing), max(existing), min(new), max(new)
    gap = lo_n - hi_e if higher == "new" else lo_e - hi_n
    within = max(hi_e - lo_e, hi_n - lo_n)
    return {"existing": existing, "new": new, "gap": gap, "within": within, "direction_ok": gap > 0, "separates": gap > within}


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules (PREDICTIONS / results/predictions.json)."""
    recs = out["tasks"]
    states = [(n, s, r) for n, t in recs.items() for s, r in t["seeds"].items()]
    res: dict[str, Any] = {}
    res["G1"] = {"status": "pass" if all(r["learned"] for _, _, r in states) else "FAIL",
                 "steps": {f"{n}/{s}": r["scored_step"] for n, s, r in states}}
    res["G2"] = {"status": "pass" if all(r["G2"]["ok"] for _, _, r in states) else "FAIL", "worst_p_o_diff": max(r["G2"]["p_o_diff"] for _, _, r in states)}
    rr = [r["inexpressible"]["random_ratio"] for _, _, r in states]
    res["G3"] = {"status": "pass" if all(0.9 <= x <= 1.1 for x in rr) else "FAIL", "random_ratios": rr}

    def per_state(cls: str, f: Any) -> list[float]:
        return [f(r) for n, _, r in states if recs[n]["cls"] == cls and r["learned"]]

    def per_task(cls: str, f: Any) -> list[float]:
        return [f(t) for t in recs.values() if t["cls"] == cls and all(r["learned"] for r in t["seeds"].values())]

    def verdict(sep: dict[str, Any], predicted: bool) -> str:
        if sep["separates"] is None:
            return "unscorable (G1)"
        return "pass" if sep["separates"] == predicted and (sep["direction_ok"] or not predicted) else "FAIL"

    p1 = separation(*(per_state(c, lambda r: r["inexpressible"]["ratio"]) for c in ("existing", "new")), higher="new")
    p2 = separation(*(per_state(c, lambda r: r["routes_lift"]["shares"]["e_delta"]) for c in ("existing", "new")), higher="new")
    p3 = separation(*(per_state(c, lambda r: r["relevance_W0"]["k_lo"]) for c in ("existing", "new")), higher="existing")
    p4r = separation(*(per_state(c, lambda r: r["routes_lift"]["shares"]["r"]) for c in ("existing", "new")), higher="existing")
    p4e = separation(*(per_state(c, lambda r: r["routes_lift"]["shares"]["e_gate"]) for c in ("existing", "new")), higher="new")
    p5 = separation(*(per_task(c, lambda t: t["across_seeds"]["jaccard_T"]) for c in ("existing", "new")), higher="existing")
    res["P1"] = {"status": verdict(p1, True), **p1}
    res["P2"] = {"status": verdict(p2, False), **p2}
    res["P3"] = {"status": verdict(p3, True), **p3, "k_kl": {c: per_state(c, lambda r: r["relevance_W0"]["k_kl"]) for c in ("existing", "new")}}
    res["P4"] = {"status": "pass" if verdict(p4r, True) == "pass" and verdict(p4e, True) == "pass" else "FAIL", "r": p4r, "e_gate": p4e}
    res["P5"] = {"status": verdict(p5, True), **p5, "jaccard_S_on": {c: per_task(c, lambda t: t["across_seeds"]["jaccard_S_on"]) for c in ("existing", "new")}}
    return res


def stage_run(S: Sizes, device: str) -> None:
    from commute import signatures as sg
    from commute.data import batches, pile_rows

    require_predictions()
    tasks_file = RESULTS / f"tasks{S.suffix}.json"
    assert tasks_file.is_file(), f"run `build` first ({tasks_file})"
    tasks = json.loads(tasks_file.read_text())
    ctx = load_all(device)
    pile = sg.pile_gate_stats(ctx["model"], batches(pile_rows("val", 0, S.pile_rows, log), 8), ctx["alive"], G_ON)
    alive_all = {k: v.nonzero().flatten() for k, v in ctx["alive"].items()}
    out: dict[str, Any] = {"tasks": {}, "task_table": tasks,
                           "pile": {"n_repeat": pile["n_repeat"], "n_positions": pile["n_positions"],
                                    "alive_on_rate_mean": sg.mean_over(pile["on_rate"], alive_all)}}
    log(f"pile at W0: {pile['n_repeat']} repeat positions of {pile['n_positions']}; alive mean on-rate {out['pile']['alive_on_rate_mean']:.4f}")
    for t in S.tasks:
        w = torch.load(LARGE / f"windows_{t.name}.pt", weights_only=False)
        out["tasks"][t.name] = run_task(ctx, t, w, S, pile, out) | {"task": tasks["tasks"][t.name]}
        dump(f"new_vs_existing{S.suffix}.json", out)
    out["score"] = score(out)
    dump(f"new_vs_existing{S.suffix}.json", out)
    log(f"score: {json.dumps(out['score'], indent=1)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["build", "run", "all"])
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    assert args.stage, "a stage or --predictions"
    S = SMOKE if args.smoke else FULL
    for stage in {"build": [stage_build], "run": [stage_run], "all": [stage_build, stage_run]}[args.stage]:
        stage(S, args.device)


if __name__ == "__main__":
    main()
