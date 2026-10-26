"""Identity test: after post-training inside the decomposition (pull-back) or a lift from outside, are the
subcomponents that carry the update still the same subcomponents?

Plan and design check: reports/plans/2026-10-15_plan_identity-test.md (test 1, confirmed by the user 2026-10-15).
Identity claims: theory/component_identity.md (same contexts, same action; Gamma is frozen and reads the raw
pre-weight activations, so its own profile can only drift through upstream weights and is reported, not scored).

Stages:
  --predictions   file results/predictions.json (refuses to overwrite)
  all             GPU: regenerate the states, attribution, identity measures -> results/identity.json
  all --smoke     GPU, minutes: two states, one matrix, five subcomponents; checks every path runs

Usage (repo root):  uv run python experiments/identity_test/run.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute.attribution import (
    ablation_effect,
    gamma_profiles,
    heldout_log_odds,
    read_acts,
    revert_one,
    reverted_weights,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"
LIFT = HERE.parent / "lift_test" / "results"

PB_LR = 1e-5  # amendment 1's selected pull-back learning rate
PB_STEPS = (10, 30)
LIFT_STEPS = (2, 3, 10)  # W fine-tune steps matched by max ||dW|| / ||W0|| to pull-back steps 10, 30; and step 10
K_SEL = 20  # subcomponents selected per state by revert-one effect
K_POS = 64  # top positions per subcomponent for the write-side (ablation) signature
K_JAC = 100
REVERT_K = (1, 3, 10, 20, 30, 100, 300, 1000)
BANK_ROWS = (512, 32)  # Pile val rows 512..543, not used by any earlier stage
SIBLINGS = (2359, 1672, 2623, 3290, 3327, 3382)  # h.2.mlp.down_proj, VPD paper's emoticon subcomponent and siblings
SIB_KEY = "h.2.mlp.down_proj"

PREDICTIONS: dict[str, Any] = {
    "about": (
        "identity_test: for the round-2 updates of Goodfire's 67M Pile model and its VPD decomposition s-55ea3f9b (Gamma "
        "frozen, Delta fixed), are the subcomponents that carry the emoticon update still the same subcomponents? "
        "Filed before any identity measurement was run. Round 2 showed the pull-back (lr 1e-5) is faithful at steps 10 "
        "and 30 and the Euclidean lift is PGD-fragile (reports/analyses/2026-10-14_analysis_lift-test.md)."
    ),
    "setup": {
        "states": {
            "pullback_s10, pullback_s30": "Adam on alive (U, V) offsets, lr 1e-5, arm A's loss, windows, batch, clip and "
                                          "seed (round 2 P3'), regenerated; held-out P(o) checked against pullback2.json",
            "lift_s2, lift_s3, lift_s10": "Euclidean lift (tol 1e-6) of the arm A W fine-tune rerun (lr 3e-5, seed 0) at "
                                          "steps 2, 3, 10, regenerated and checked against pullback.json. Steps 2 and 3 "
                                          "match pull-back steps 10 and 30 by max ||dW||/||W0|| (0.0053 vs 0.0051; "
                                          "0.0065 vs 0.0066). Step 10 is where the lift's PGD ratio was 5.4",
        },
        "revert_one": "for every alive subcomponent j, the drop in held-out mean log-odds of `o` at the trigger position "
                      "(73 windows, fp32) when j alone is reverted to its W0 value (u2 v2^T -> u v^T), weights otherwise "
                      "at the state. gain = log-odds(state) - log-odds(W0)",
        "selected": "the K_SEL = 20 subcomponents with the largest revert-one drop, per state",
        "null_group": "for each selected subcomponent, one random alive subcomponent of the same matrix with |revert-one| "
                      "below that matrix's median (seed 0): drift from upstream changes only",
        "bank": "Pile val rows 512..543 (16 384 positions) plus the 73 held-out windows (2 993 positions)",
        "read_signature": "x . v over the bank, x the matrix's input under W0 (before, v) or the state (after, v2). "
                          "r_self = |Pearson(after_j, before_j)|; rank of j among all alive before-profiles of the same "
                          "matrix by |Pearson| with after_j; top-100 Jaccard by |x . v|",
        "write_signature": "ablation effect: mean over j's top-64 Pile-bank positions (by |x . v| at W0) of "
                           "log-softmax(full) - log-softmax(j ablated), a vocabulary vector; before at W0 with (u, v), "
                           "after at the state with (u2, v2), same positions. write_cos = cosine(before, after). "
                           "floor_j = cosine of the W0 effect over even-ranked vs odd-ranked positions (split half)",
        "gamma_profile": "reported only: |Pearson| of j's g over the Pile bank before vs after (upstream drift only, since "
                         "Gamma reads raw activations), and |Pearson(g_j, |x . v_j|)| before and after",
        "gauge_free_change": "|cos(u, u2)|, |cos(v, v2)|, ||u2|| ||v2|| / (||u|| ||v||), ||u2 v2^T - u v^T|| / ||u v^T||",
    },
    "predictions": [
        {
            "id": "1_identity_kept",
            "scored": True,
            "claim": "The subcomponents that carry the pull-back's update are still the same subcomponents.",
            "measure": "state pullback_s30, its 20 selected subcomponents. A subcomponent keeps identity if r_self >= 0.9 "
                       "AND rank == 1 AND write_cos >= floor_j - 0.1",
            "pass": ">= 18 of 20 keep identity",
            "null": "<= 10 of 20 keep identity",
            "prediction": "pass",
        },
        {
            "id": "2_update_concentrated",
            "scored": True,
            "claim": "The pull-back's behavioural update is carried by a few subcomponents.",
            "measure": "state pullback_s30: reverting the 10 subcomponents with the largest revert-one drop together; "
                       "fraction of gain removed = (log-odds(state) - log-odds(reverted)) / gain",
            "pass": "fraction removed >= 0.5",
            "null": "fraction removed < 0.2 (diffuse)",
            "prediction": "fail (Adam spreads the update over all alive U, V; round 1 found 10 random subcomponents "
                          "as good as the top 10)",
        },
    ],
    "reported_not_scored": [
        "the same measures for pullback_s10 and the three lift states (pull-back vs lift at matched norm)",
        "whether 2359 or a sibling is among the 10 largest revert-one drops, and its change statistics",
        "fraction removed for k in (1, 3, 10, 20, 30, 100, 300, 1000) and for all alive subcomponents",
    ],
    "filed": None,
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(name: str, obj: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    return p


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; do not edit a prediction after filing (add a dated report instead)")
    PREDICTIONS["filed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump("predictions.json", PREDICTIONS)
    log(f"filed {PRED}")


# ----------------------------------------------------------------------------- states


def pullback_states(model: Any, UV: dict[str, Any], win: dict[str, Any], o_token: int, device: str,
                    W0: dict[str, torch.Tensor]) -> tuple[dict[int, Any], list[dict[str, float]]]:
    """Round 2's selected pull-back (lr 1e-5), states (U2, V2) at PB_STEPS."""
    import copy

    from commute import component_training, vpd
    from commute.finetune import ArmAConfig, arm_a_loss, eval_arm_a

    cfg = ArmAConfig(o_token=o_token)
    net = model.target_model
    ref = copy.deepcopy(net).eval().requires_grad_(False)
    off = component_training.Offsets(UV)
    params = list(off.parameters())
    opt = torch.optim.Adam(params, lr=PB_LR, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg.seed)
    handles = component_training.hooks({k: vpd.target_linear(model, k) for k in UV}, off)
    states, curve = {}, []
    try:
        for step in range(1, max(PB_STEPS) + 1):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, len(win["train"]), (cfg.batch,), generator=gen)
            loss, _ = arm_a_loss(net, ref, win["train"].tokens[idx].to(device), cfg, win["train"].half)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if step in (1, *PB_STEPS):
                ev = eval_arm_a(net, ref, win["heldout"], o_token, device)
                rel = max(float(off.delta_w(k).norm() / W0[k].norm()) for k in UV)
                curve.append({"step": step, "heldout_p_o": ev["p_o"], "rel_norm_max": rel})
                log(f"[pullback] step {step}: held-out P(o) {ev['p_o']:.4f} rel_norm_max {rel:.5f}")
            if step in PB_STEPS:
                states[step] = {k: off.components(k) for k in UV}
    finally:
        for h in handles:
            h.remove()
    return states, curve


def lift_states(model: Any, UV: dict[str, Any], win: dict[str, Any], o_token: int, device: str,
                keys: list[str]) -> tuple[dict[int, Any], dict[int, Any]]:
    """W fine-tune rerun (arm A, checkpoints 1..10), then the Euclidean lift of dW at LIFT_STEPS."""
    from commute import split
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=o_token, steps=max(LIFT_STEPS), checkpoints=(1, 2, 3, 5, 10))
    res = finetune_arm_a(model, win["train"], win["heldout"], cfg, device, log)
    states = {}
    t0 = time.time()
    for k in keys:
        U_A, V_A = UV[k][0].double().cpu(), UV[k][1].double().cpu()
        b = split.basis(U_A, V_A, 1e-6)
        for s in LIFT_STEPS:
            U2, V2, _ = split.split_with_delta_change(res.deltas[s][k].double(), b, U_A, V_A)
            states.setdefault(s, {})[k] = (U2.float().to(device), V2.float().to(device))
    log(f"[lift] bases and lifts in {time.time() - t0:.0f} s")
    return states, {s: {"dW": res.deltas[s], **res.evals[s]} for s in LIFT_STEPS}


# ----------------------------------------------------------------------------- attribution


# ----------------------------------------------------------------------------- main stage


def stage_all(args: argparse.Namespace) -> None:
    from commute import vpd
    from commute.data import batches, pile_rows
    from commute.faithfulness import model_state
    from commute.identity import change_stats, cosine, profile_retrieval, topk_jaccard

    if not PRED.is_file():
        sys.exit("file predictions first: run.py --predictions")
    smoke = args.smoke
    dev = args.device
    model = vpd.load_component_model(dev)
    W0 = vpd.target_weights(model)
    alive = torch.load(ROUND1 / "alive.pt", weights_only=False)["alive"]
    win = torch.load(ROUND1 / "windows.pt", weights_only=False)
    o_token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    keys = vpd.module_keys(model)
    UV = {k: (model.components[k].U.detach()[alive[k]].T.float().contiguous(),
              model.components[k].V.detach()[:, alive[k]].float().contiguous()) for k in keys}
    out: dict[str, Any] = {"smoke": smoke, "states": {}}
    name = "identity_smoke.json" if smoke else "identity.json"

    # --- states, with reproduction checks against round 2
    pb, pb_curve = pullback_states(model, UV, win, o_token, dev, W0)
    ref2 = {c["step"]: c for c in json.loads((LIFT / "pullback2.json").read_text())["runs"]["1e-05"]["curve"]}
    out["pullback_repro"] = {c["step"]: {"p_o": c["heldout_p_o"] - ref2[c["step"]]["heldout_p_o"],
                                         "rel_norm_max": c["rel_norm_max"] - ref2[c["step"]]["rel_norm_max"]} for c in pb_curve}
    lf, lf_info = lift_states(model, UV, win, o_token, dev, keys)
    ref1 = json.loads((LIFT / "pullback.json").read_text())["rerun"]["evals"]
    out["lift_repro"] = {s: {"p_o": lf_info[s]["p_o"] - ref1[str(s)]["p_o"],
                             "rel_norm_max": lf_info[s]["rel_norm_max"] - ref1[str(s)]["rel_norm_max"]} for s in LIFT_STEPS}
    log(f"reproduction: pull-back {out['pullback_repro']}; W rerun {out['lift_repro']}")
    states: dict[str, dict[str, Any]] = {}
    for s in PB_STEPS:
        uv2 = pb[s]
        states[f"pullback_s{s}"] = {"uv": uv2, "W": {k: W0[k] + (uv2[k][0] @ uv2[k][1].T - UV[k][0] @ UV[k][1].T) for k in keys}}
    for s in LIFT_STEPS:
        states[f"lift_s{s}"] = {"uv": lf[s], "W": {k: W0[k] + lf_info[s]["dW"][k].to(dev) for k in keys}}
    if smoke:
        states = {n: states[n] for n in ("pullback_s10", "lift_s2")}
    att_keys = [SIB_KEY] if smoke else keys
    limit = 64 if smoke else None
    k_sel = 5 if smoke else K_SEL

    # --- banks
    pile = pile_rows("val", *BANK_ROWS, log)
    pile_b = [b.to(dev) for b in batches(pile, 8)]
    wtoks = win["heldout"].tokens.to(dev)
    half = win["heldout"].half
    n_pile = pile.numel()
    before_acts = read_acts(model, UV, pile_b + [wtoks], keys)
    lo_W0 = heldout_log_odds(model, wtoks, half, o_token)
    sib_pos = {c: int(alive[SIB_KEY][:c].sum()) for c in SIBLINGS if bool(alive[SIB_KEY][c])}
    eff_cache: dict[tuple[str, int], Any] = {}  # (matrix, j) -> (positions, before effect, floor)

    def before_effect(k: str, j: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        if (k, j) not in eff_cache:
            pos = before_acts[k][:n_pile, j].abs().topk(K_POS).indices.cpu()
            e = ablation_effect(model, k, j, UV[k][0], UV[k][1], pile, pos)
            eff_cache[(k, j)] = (pos, e.mean(0), cosine(e[0::2].mean(0), e[1::2].mean(0)))
        return eff_cache[(k, j)]

    gen = torch.Generator().manual_seed(0)
    for sname, S in states.items():
        t0 = time.time()
        st, W = S["uv"], S["W"]
        rec: dict[str, Any] = {}
        with model_state(model, target=W):
            lo = heldout_log_odds(model, wtoks, half, o_token)
            gain = lo - lo_W0
            drop = revert_one(model, UV, st, wtoks, half, o_token, lo, att_keys, limit=limit)
        flat = torch.cat([drop[k] for k in att_keys])
        owner = [(k, j) for k in att_keys for j in range(len(drop[k]))]
        order = flat.argsort(descending=True)
        top = [owner[i] for i in order[:max(REVERT_K)].tolist()]
        rec["log_odds"] = {"W0": lo_W0, "state": lo, "gain": gain}
        rec["revert_one_top50"] = [{"key": f"{k}:{int(alive[k].nonzero()[j])}", "drop": float(drop[k][j])} for k, j in top[:50]]
        rec["revert_one_sum"] = float(flat.sum())
        cum = {}
        for kk in (*REVERT_K, "all"):
            chosen = owner if kk == "all" else top[:kk]
            sel: dict[str, list[int]] = {}
            for k, j in chosen:
                sel.setdefault(k, []).append(j)
            Wr = reverted_weights(W, UV, st, {k: torch.tensor(v) for k, v in sel.items()})
            with model_state(model, target=Wr):
                cum[str(kk)] = (lo - heldout_log_odds(model, wtoks, half, o_token)) / gain
        rec["fraction_removed"] = cum
        k1, j1 = top[0]
        rec["check_top1_hook_vs_weights"] = {"hook": float(drop[k1][j1]), "weights": cum["1"] * gain}
        log(f"[{sname}] check: top-1 revert drop by hook {float(drop[k1][j1]):.5f}, by weights {cum['1'] * gain:.5f}")
        log(f"[{sname}] gain {gain:.3f}; fraction removed by top-k revert {json.dumps({k: round(v, 3) for k, v in cum.items()})}")
        if sname.startswith("pullback") and not smoke:
            assert abs(cum["all"] - 1) < 1e-2, f"reverting every subcomponent should give W0 back: {cum['all']}"

        # change statistics, all alive
        cs = {k: change_stats(UV[k][0], UV[k][1], st[k][0], st[k][1]) for k in keys}
        rel_all = torch.cat([cs[k]["rel_change"] for k in keys]).float()
        rec["rel_change_quantiles"] = {q: float(rel_all.quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)}

        # groups: selected, siblings, null
        selected = top[:k_sel]
        groups = {"selected": selected, "siblings": [(SIB_KEY, p) for p in sib_pos.values()], "null": []}
        for k, j in selected:
            med = drop[k].abs().median()
            pool = (drop[k].abs() <= med).nonzero().flatten()
            groups["null"].append((k, int(pool[torch.randint(len(pool), (1,), generator=gen)])))
        with model_state(model, target=W):
            after_acts = read_acts(model, {k: (st[k][0], st[k][1]) for k in keys}, pile_b + [wtoks], keys)
            need = {}
            for g in groups.values():
                for k, j in g:
                    need.setdefault(k, set()).add(j)
            need_t = {k: torch.tensor(sorted(v)) for k, v in need.items()}
            need_full = {k: alive[k].nonzero().flatten()[v] for k, v in need_t.items()}
            g_after = gamma_profiles(model, pile_b, need_full)
        g_before = gamma_profiles(model, pile_b, need_full)

        def measure(k: str, j: int) -> dict[str, Any]:
            pr = profile_retrieval(before_acts[k], after_acts[k], torch.tensor([j], device=before_acts[k].device))
            pos, e_before, floor = before_effect(k, j)
            with model_state(model, target=W):
                e_after = ablation_effect(model, k, j, st[k][0], st[k][1], pile, pos).mean(0)
            col = int((need_t[k] == j).nonzero())
            gb, ga = g_before[k][:, col], g_after[k][:, col]
            ab, aa = before_acts[k][:n_pile, j].abs(), after_acts[k][:n_pile, j].abs()
            r_pile = profile_retrieval(before_acts[k][:n_pile], after_acts[k][:n_pile], torch.tensor([j], device=ab.device))
            r_win = profile_retrieval(before_acts[k][n_pile:], after_acts[k][n_pile:], torch.tensor([j], device=ab.device))
            m = {
                "key": f"{k}:{int(alive[k].nonzero()[j])}", "revert_drop": float(drop[k][j]) if k in drop else None,
                **{s: float(v[j]) for s, v in cs[k].items()},
                "r_self": float(pr["r_self"][0]), "rank": int(pr["rank"][0]), "r_other_max": float(pr["r_other_max"][0]),
                "r_self_pile": float(r_pile["r_self"][0]), "r_self_windows": float(r_win["r_self"][0]),
                "jaccard_top100": topk_jaccard(before_acts[k][:, j], after_acts[k][:, j], K_JAC),
                "write_cos": cosine(e_before, e_after), "write_floor": floor,
                "write_norm_ratio": float(e_after.norm() / e_before.norm()),
                "gamma_r": float(profile_retrieval(gb[:, None], ga[:, None], torch.tensor([0]))["r_self"][0]),
                "gamma_vs_act_before": float(profile_retrieval(gb[:, None], ab[:, None], torch.tensor([0]))["r_self"][0]),
                "gamma_vs_act_after": float(profile_retrieval(ga[:, None], aa[:, None], torch.tensor([0]))["r_self"][0]),
            }
            m["keeps_identity"] = m["r_self"] >= 0.9 and m["rank"] == 1 and m["write_cos"] >= m["write_floor"] - 0.1
            return m

        rec["groups"] = {g: [measure(k, j) for k, j in lst] for g, lst in groups.items()}
        rec["n_keep_selected"] = sum(m["keeps_identity"] for m in rec["groups"]["selected"])
        rec["sibling_in_top10"] = any(k == SIB_KEY and j in sib_pos.values() for k, j in top[:10])
        out["states"][sname] = rec
        LARGE.mkdir(parents=True, exist_ok=True)
        torch.save(drop, LARGE / f"revert_one_{sname}.pt")
        log(f"[{sname}] keep identity {rec['n_keep_selected']}/{len(selected)}; sibling in top 10: {rec['sibling_in_top10']}; "
            f"{time.time() - t0:.0f} s")
        for m in rec["groups"]["selected"][:5]:
            log(f"   {m['key']}: drop {m['revert_drop']:.3f} rel_change {m['rel_change']:.4f} r_self {m['r_self']:.3f} "
                f"rank {m['rank']} jac {m['jaccard_top100']:.2f} write_cos {m['write_cos']:.3f} (floor {m['write_floor']:.3f})")
        dump(name, out)

    if "pullback_s30" in out["states"]:
        r = out["states"]["pullback_s30"]
        out["score"] = {
            "1_identity_kept": {"n_keep": r["n_keep_selected"], "pass": r["n_keep_selected"] >= 18,
                                "null": r["n_keep_selected"] <= 10},
            "2_update_concentrated": {"fraction_removed_top10": r["fraction_removed"]["10"],
                                      "pass": r["fraction_removed"]["10"] >= 0.5, "null": r["fraction_removed"]["10"] < 0.2},
        }
        log(f"score {json.dumps(out['score'])}")
    dump(name, out)
    log("DONE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("stage", nargs="?", choices=["all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        file_predictions()
        return
    if args.stage is None:
        ap.error("give a stage or --predictions")
    stage_all(args)


if __name__ == "__main__":
    main()
