"""E309 (cell P2.79) stage "scans": the OLD code's side of the readings no earlier reproduction exercised, to
results/old_c_<tag>.json (tags A and B: the old code run twice, its own floor), keyed as new_c.py keys them:

  needed.scan (VPD)       x2 (E268's world): E268's identity control (x' = x, 3 subcomponents and the remainder per
                          matrix, 64 windows), then every variable against one derangement of 2 windows (2 pairs)
  needed_sae.scan         e58/copy_s0_scaled (E263's item, E245's SAEs): the identity control as E298 filed it (per
                          site the first 3 latents active on the windows and the error term), then every variable
                          (the moved-latent shortcut) against one derangement of 2 windows
  SAE write pass          the same item, `stand_in.write_change` on `sae_feat.sites` (E263's `read`): E_b over E245's K*,
                          and with the 8 error terms left out (sae_eb score.py `latents_only`'s selection)
  neurons, no remainder   copy24_s0 (E277's trained world), E277's fresh `write_pass`: E_b over its K with the four
                          layer remainders left out (`remainder_columns`' selection, j == H + NEUR)
  per-key steering        E135's copy_s0 per-key vector (`kind_record` on vectors["perkey"])
Scan readings per item: needed_set_size (|K|), scan_flips (pairs that flipped, summed), scan_first_digest (an exact
integer digest of first_pair) and scan_max_dev. Identity: flips 0 expected; max_dev is the scan path's fp32 floor.

Runs only against the OLD src: from the root of the e309-old worktree (62e4ca8d), this file and old_c.sbatch copied in.
  --dry  CPU: settings, saved values and the inputs that need no model; stops before the model.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from old import Out, npz

from commute import needed as IX
from commute import needed_sae as NS
from commute import sae_feat as SF
from commute import stand_in as SI
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment

N_PAIRS, SEED = 2, 268  # one derangement of 2 windows; E268's seed
SAE_ITEM = "e58/copy_s0_scaled"


def settings(exp: str) -> dict[str, Any]:
    return json.loads((EXPERIMENTS / exp / "results" / "predictions.json").read_text())["settings"]


def digest(first: torch.Tensor) -> int:
    """An exact integer digest of Scan.first_pair: sum over variables of (first_pair + 2) x (index + 1)."""
    return int(((first.long() + 2) * torch.arange(1, len(first) + 1)).sum())


def put_scan(o: Out, item: str, sc: Any, **s: Any) -> None:
    o.put(f"needed_set_size/{item}", int((sc.first_pair >= 0).sum()), None, **s)
    o.put(f"scan_flips/{item}", int(sc.flips.sum()), None, **s)
    o.put(f"scan_first_digest/{item}", digest(sc.first_pair), None, **s)
    o.put(f"scan_max_dev/{item}", sc.max_dev, None, cases=sc.rows, **s)


def vpd_scan(o: Out) -> None:
    KN: Any = load_experiment("kstar_needed", "run")
    cfg, w = KN.load_config(), KN.world()
    r = KN.identity(w, cfg, None)
    o.put("scan_flips/x2@vpd_identity", r["flips"], None, cases=r["cases"])
    o.put("scan_max_dev/x2@vpd_identity", r["max_dev"], None, cases=r["cases"])
    F, feat, toks, at = KN.featurized(w, "x2")
    run = IX.cache(F, feat, toks, at, cfg.cache_batch)
    ex, pt = IX.derangements(len(toks), 1, N_PAIRS, SEED)
    sc = IX.scan(F, feat, run, at, ex, pt, KN.batch_rows(cfg, run.top.shape[1]), print)
    put_scan(o, "x2@vpd_P1m2", sc, n_vars=feat.n, windows=tuple(toks.shape))


def sae_world() -> tuple[Any, Any, Any]:
    E251: Any = load_experiment("sae_r_a_random", "run")
    cfg, _ = E251.E245.load_config("scored")
    return E251.World(E251.E242.load_config()[0]), E251.E245.load_saes(cfg), cfg


def sae(o: Out) -> None:
    w, saes, cfg = sae_world()
    F, M, toks, at, _ = w.make(SAE_ITEM)
    for N in (F, M):  # as sae_eb `read`
        N.net.requires_grad_(False)
        for e in N.extra.values():
            e.requires_grad_(False)
    ss = SF.sites(F, M, saes, toks.shape[1])
    n = SF.n_vars(ss)
    duck: list[Any] = list(ss)  # as sae_eb `read`: write_change is typed for VPDSite
    num, den, dd = SI.write_change(F.net, duck, n, list(toks.split(cfg.gain_batch)))
    changed = torch.zeros(n, dtype=torch.bool)
    changed[sorted(SF.changed_vars(ss))] = True
    K = npz("experiments/sae_r_a/results/large/items_e58-copy_s0_scaled.npz")["main_in_K_polished"].bool()
    rem = torch.zeros(n, dtype=torch.bool)
    rem[[s.rem for s in ss]] = True
    summ = ("experiments/sae_eb/results/summary.json", ["rows", {"item": SAE_ITEM}])
    o.put(f"E_b_sae/{SAE_ITEM}@fresh", SI.explained_share(num, den, dd, K & changed), (summ[0], [*summ[1], "E_b_sae"]), batch=cfg.gain_batch)
    o.put(f"E_b_sae/{SAE_ITEM}@fresh_no_leftover", SI.explained_share(num, den, dd, K & changed & ~rem), (summ[0], [*summ[1], "E_b_sae_latents_only"]), batch=cfg.gain_batch)
    s = settings("fig2_standard")
    run = NS.cache(F, ss, toks, at, s["cache_batch"])
    rows = IX.rows_per_forward(s["rows"], s["logit_rows"], run.top.shape[1])
    ex = torch.arange(min(s["examples"], len(toks))).to(run.top.device)
    only = {j: torch.cat([NS.Swap(q, run.outs[j]).active(ex, s["ident_vars"], s["sae_chunk"]).cpu(), torch.tensor([q.rem - q.off])]) for j, q in enumerate(ss)}
    sc = NS.scan(F, ss, run, at, ex, ex, rows, s["sae_err_rows"], s["sae_chunk"], print, only)
    o.put(f"scan_flips/{SAE_ITEM}@sae_identity", int(sc.flips.sum()), None, cases=sc.rows)
    o.put(f"scan_max_dev/{SAE_ITEM}@sae_identity", sc.max_dev, None, cases=sc.rows)
    ex, pt = IX.derangements(len(toks), 1, N_PAIRS, SEED)
    sc = NS.scan(F, ss, run, at, ex, pt, rows, s["sae_err_rows"], s["sae_chunk"], print)
    put_scan(o, f"{SAE_ITEM}@sae_P1m2", sc, n_vars=n, windows=tuple(toks.shape))


def neurons_no_rem(o: Out) -> None:
    from commute import needed_neurons as NN

    NR: Any = load_experiment("needed_neurons", "run")
    cfg, c229 = NR.load_config(), NR.load_e229()
    feat = NR.E229.feat_of(c229, NR.E229.load_pct(c229), cfg.band)
    F, M, toks, _, _ = NR.Trained(cfg, c229).make("copy24_s0")
    num, den, dd, ch, _ = NR.write_pass(F, M, toks, feat, cfg.cache_batch)
    K = npz("experiments/needed_neurons/results/large/items_copy24_s0.npz")["in_K"].bool()
    rem = torch.arange(len(ch)) % (NN.H + NN.NEUR + 1) == NN.H + NN.NEUR
    o.put("E_b_neurons/copy24_s0@fresh_no_leftover", SI.explained_share(num, den, dd, K & ch & ~rem),
          ("experiments/needed_neurons/results/summary.json", ["items", "copy24_s0", "E_b_K_no_rem"]), batch=cfg.cache_batch)  # fmt: skip


def steering_perkey(o: Out) -> None:
    from commute import steering as ST

    net = ST.Net("cuda")
    SC: Any = load_experiment("steering_copy_lookup", "run")
    s = json.loads(SC.PRED.read_text())["settings"]
    c, run62, large = SC.Cfg(s["half"], s["n"], tuple(s["strengths"]), s["n_random"], s["random_seed"]), json.loads((SC.E62_DIR / "run.json").read_text()), (SC.E62_DIR / "large").resolve()
    p, _ = SC.load_pair(c, large, run62)
    states = SC.load_states(large, run62)
    w0 = SC.baselines(net, p)
    G = SC.gains(net, p, states, w0)
    vs = SC.vectors(net, p, states["copy_s0"])
    rec = SC.kind_record(net, p, c, "copy", "lookup", vs["perkey"], w0, G["copy_s0"], G["lookup_s0"], torch.Generator().manual_seed(c.random_seed))
    o.put("steering_perkey/e62/copy_s0", rec["share_lo"], ("experiments/steering_copy_lookup/results/stage1.json", ["states", "copy_s0", "steering", "perkey", "share_lo"]),
          gain=G["copy_s0"]["lo"], key_pos="half - 1", dtype="fp32", floor="none (G != 0)", setting=[rec["layer"], rec["strength"]])  # fmt: skip


GPU = {"steering_perkey": steering_perkey, "neurons_no_rem": neurons_no_rem, "vpd_scan": vpd_scan, "sae": sae}


def dry() -> None:
    """The settings and saved values every group reads; no model."""
    for exp in ("kstar_needed", "fig2_standard", "needed_neurons", "steering_copy_lookup", "sae_r_a"):
        print(exp, {k: v for k, v in settings(exp).items() if not isinstance(v, (list, dict)) and not k.endswith("source")} if exp != "fig2_standard" else
              {k: settings(exp)[k] for k in ("examples", "ident_vars", "rows", "logit_rows", "cache_batch", "sae_err_rows", "sae_chunk")})  # fmt: skip
    print("K* e58", tuple(npz("experiments/sae_r_a/results/large/items_e58-copy_s0_scaled.npz")["main_in_K_polished"].shape),
          "K copy24", tuple(npz("experiments/needed_neurons/results/large/items_copy24_s0.npz")["in_K"].shape))  # fmt: skip
    print("derangement", [t.tolist() for t in IX.derangements(75, 1, N_PAIRS, SEED)])
    print("OLD_C dry: inputs read (no model)", flush=True)


def main(tag: str, is_dry: bool) -> int:
    if is_dry:
        dry()
        return 0
    o = Out(f"c_{tag}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every old run
    o.d["_meta"] = {"card": torch.cuda.get_device_name(), "torch": torch.__version__}
    for n, f in GPU.items():
        o.run(n, f)
        torch.cuda.empty_cache()
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"OLD_C {tag} DONE; errors: {o.errors or 'none'}", flush=True)
    return 1 if o.errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="names the run: results/old_c_<tag>.json (A, B)")
    ap.add_argument("--dry", action="store_true", help="CPU: read settings and saved inputs, stop before the model")
    a = ap.parse_args()
    hard_exit(main(a.tag, a.dry))
