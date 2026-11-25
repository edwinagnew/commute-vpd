"""E309 (cell P2.79): the reading refactor's old-code reproduction harness. Each reading the plan names
(reports/plans/2026-11-25_plan_reading-refactor.md, "States to check"; appendix A1-A11) is computed here THROUGH THE
OLD CODE PATH, the saved run's own functions reached by `load_experiment` exactly as that run called them, and written
to results/old_<tag>.json keyed `<reading>/<state>[@<variant>]`: the value, the saved value it should repeat (file and
jq path), and the settings the old path used. Two runs (tags A and B) on one card give the old code's own spread, the
floor the new functions are held to (compare.py). Nothing here is a new measurement; nothing is trained.

  --part data  CPU (`fleet/cpu-run --mem 16G`): readings from saved arrays and saved updates (E_b from saved sums, size,
               erank, summed norm, the log10 of a saved prior). W0 is `vpd.checkpoint_weights` (no model is built); the
               gpu part checks that it equals the live W0 every old run read
  --part gpu   one L40 (old.sbatch): every reading that needs the model, cheapest first, the W0 check first
A reading that raises is written as {"error": ...} and the rest still run; the job then exits 1.

    ~/commute-agents/fleet/cpu-run --mem 16G python3 experiments/reading_refactor/old.py --tag A --part data
    sbatch --comment=E309 experiments/reading_refactor/old.sbatch A
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import elicit as EL
from commute import realistic as RZ
from commute import stand_in as SI
from commute import update_spectrum as US
from commute import vpd
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", EXPERIMENTS.parent
E62_COPY = "experiments/induction_cue/results/large/copy_s0.pt"
SEQ_RL = "experiments/procedural_rl/results/large/states/sequence_rl_s0.pt"
LOWER_128 = "experiments/convergence_panel/results/large/states/lower_s0_step128.pt"
PANEL_READ = "experiments/convergence_panel/results/wide/read/lower.json"
Saved = tuple[str, list[Any]] | None  # (repo file, key path; a {field: value} element selects from a list)


def jq(keys: list[Any]) -> str:
    out = "."
    for k in keys:
        if isinstance(k, dict):
            ((f, v),) = k.items()
            out += f"[]|select(.{f}=={json.dumps(v)})|."
        else:
            out += f"[{json.dumps(k)}]"
    return out.removesuffix("|.")


def lookup(saved: Saved) -> Any:
    if saved is None:
        return None
    file, keys = saved
    x = json.loads((REPO / file).read_text())
    for k in keys:
        x = next(e for e in x if all(e.get(f) == v for f, v in k.items())) if isinstance(k, dict) else x[k]
    return x


class Out:
    """results/old_<tag>.json, rewritten after every reading (a part run later adds to what an earlier part wrote)."""

    def __init__(self, tag: str) -> None:
        self.path = RESULTS / f"old_{tag}.json"
        self.d: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.errors: list[str] = []

    def put(self, key: str, value: Any, saved: Saved, **settings: Any) -> None:
        rec = {"value": value, "saved": lookup(saved), "saved_file": saved and saved[0], "saved_jq": saved and jq(saved[1]), "settings": settings}
        self.d[key] = rec
        self.write()
        print(f"[{key}] {value!r} (saved {rec['saved']!r})", flush=True)

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        tmp.replace(self.path)

    def run(self, name: str, f: Callable[[Out], None]) -> None:
        t0 = time.time()
        try:
            f(self)
        except Exception:  # noqa: BLE001 (one reading's failure must not lose the others: recorded, the job exits 1)
            self.errors.append(name)
            self.d[f"_error/{name}"] = traceback.format_exc()
            self.write()
            print(f"[{name}] ERROR\n{traceback.format_exc()}", flush=True)
        self.d.setdefault("_seconds", {})[name] = time.time() - t0
        self.write()


def dw(path: str) -> dict[str, torch.Tensor]:
    """A saved update as every old reader took it: the file's tensors, cast to fp32."""
    return {k: v.float() for k, v in torch.load(REPO / path, weights_only=False, map_location="cpu").items()}


def npz(path: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in np.load(REPO / path).items()}


# ----------------------------------------------------------------------------- data part (CPU)


def eb_saved_sums(o: Out) -> None:
    """E_b = stand_in.explained_share(num, den, dd, K & changed) from each run's saved sums and saved K, as it ran."""
    KN: Any = load_experiment("kstar_needed", "run")
    for item in ("x2", "new_few"):  # E268: E247's sums, K from E268's scan
        a, K = KN.saved(item, "built"), npz(f"experiments/kstar_needed/results/large/items_{item}.npz")["in_K"].bool()
        o.put(f"E_b_vpd/{item}", SI.explained_share(a["num"], a["den"], a["dd"], K & a["changed"]), (f"experiments/kstar_needed/results/items/{item}.json", ["E_b_K"]),
              sums=a["src"], K="experiments/kstar_needed in_K", sel="K & changed", dtype="float64", remainders="kept")  # fmt: skip
    a = KN.saved("e58/copy_s0", "items")  # E275: E257's sums, K from E275's scan
    K = npz("experiments/claim2_needed/results/large/items_e58-copy_s0.npz")["in_K"].bool()
    o.put("E_b_vpd/e58/copy_s0", SI.explained_share(a["num"], a["den"], a["dd"], K & a["changed"]), ("experiments/claim2_needed/results/items/e58-copy_s0.json", ["E_b_K"]),
          sums=a["src"], K="experiments/claim2_needed in_K", sel="K & changed", dtype="float64", remainders="kept")  # fmt: skip
    for item in ("x2", "copy24_s0"):  # E277: its own sums and K
        z = npz(f"experiments/needed_neurons/results/large/items_{item}.npz")
        o.put(f"E_b_neurons/{item}", SI.explained_share(z["num"], z["den"], z["dd"], z["in_K"].bool() & z["changed"].bool()),
              (f"experiments/needed_neurons/results/items/{item}.json", ["E_b_K"]), sums="needed_neurons npz", K="needed_neurons in_K", sel="K & changed",
              dtype="float64", remainders="kept")  # fmt: skip
    z, k = npz("experiments/sae_eb/results/large/items_e58-copy_s0_scaled.npz"), npz("experiments/sae_r_a/results/large/items_e58-copy_s0_scaled.npz")
    sel = k["changed"].bool() & k["main_in_K_polished"].bool()  # E263: SAE sums over E245's K* (not K)
    o.put("E_b_sae/e58/copy_s0_scaled", SI.explained_share(z["num"], z["den"], z["dd"], sel), ("experiments/sae_eb/results/summary.json", ["rows", {"item": "e58/copy_s0_scaled"}, "E_b_sae"]),
          sums="sae_eb npz", K="sae_r_a main_in_K_polished (K*)", sel="K* & changed", dtype="float64", remainders="kept")  # fmt: skip


def spectra_and_sizes(o: Out) -> None:
    W0 = vpd.checkpoint_weights()[0]
    o.put("erank/e62/copy_s0", US.state(US.load(REPO / E62_COPY))["erank"], ("experiments/update_svd/results/results.json", ["pairs", "e62_copy_lookup", "reuse", "copy_s0", "erank"]),
          path="update_svd: US.state(US.load(file)), fp64")  # fmt: skip
    for name, f, saved in (("sequence_rl_s0", SEQ_RL, ("experiments/procedural_rl/results/read/sequence_rl_s0.json", [])),
                           ("lower_s0_step128", LOWER_128, (PANEL_READ, ["states", LOWER_128]))):  # fmt: skip
        d = dw(f)
        o.put(f"erank/{name}", US.state(d)["erank"], (saved[0], [*saved[1], "erank"]), path="RR.Reader.state / convergence_panel readings: US.state(fp32 dW)")
        o.put(f"size_rel/{name}", RZ.update_size(d, W0), (saved[0], [*saved[1], "size"]), path="realistic.update_size(dW, W0)", W0="checkpoint_weights")
    d = dw("experiments/induction_cue/results/large/copy_s0.pt")  # size_matched Readers.size on holes_fill's Job.dW (fp32)
    w0n = math.sqrt(sum(float(w.double().pow(2).sum()) for w in W0.values()))
    o.put("size_rel/e62/copy_s0", math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values())) / w0n,
          ("experiments/size_matched/results/m67.json", ["pairs", "e62_copy_lookup", "sizes", "reuse", 0]), path="size_matched Readers.size (sqrt sum / W0 norm)",
          W0="checkpoint_weights")  # fmt: skip
    raw = torch.load(REPO / E62_COPY, weights_only=True, map_location="cpu")  # E135 load_states: the file's dtype
    o.put("summed_norm/e62/copy_s0", EL.summed_norm(raw), ("experiments/induction_cue/results/run.json", ["states", "copy_s0", "norm"]), path="elicit.summed_norm (scorecard size)")


def saved_priors(o: Out) -> None:
    """The scorecard prior is log10 of a geometric-mean P; e62's P is copied from E62's run.json (no forward)."""
    for name in ("e62_copy", "e46_ident32"):
        p = lookup(("experiments/prior_vs_share/results/stage1.json", [name, "prior"]))
        o.put(f"prior_log10_geo/{name}@saved_P", math.log10(p), ("experiments/prior_vs_share/results/stage2.json", ["points", {"name": name}, "log10_prior"]),
              path="prior_vs_share point(): math.log10(stage1 prior)")  # fmt: skip


# ----------------------------------------------------------------------------- gpu part


class Gpu:
    """The model once, as every old run built it (`steering.Net`, tf32 off), and the old modules on demand."""

    def __init__(self) -> None:
        from commute import steering as ST

        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.net = ST.Net("cuda")


def w0_check(g: Gpu, o: Out) -> None:
    ck = vpd.checkpoint_weights()[0]
    same = all(torch.equal(g.net.W0[k].cpu(), ck[k]) for k in g.net.W0) and sorted(ck) == sorted(g.net.W0)
    o.d["_meta"] = {"card": torch.cuda.get_device_name(), "torch": torch.__version__, "W0_live_equals_checkpoint": same}
    o.write()


def share_e136(g: Gpu, o: Out) -> None:
    """E136 `Job.trained`: route_cells at W0, then route_split on W0 + dW, released components, the inventory's batch."""
    from commute.faithfulness import model_state
    from commute.local_faith import route_cells, route_split

    WIN: Any = load_experiment("share_untrained", "windows")
    inv = json.loads((REPO / "experiments/share_untrained/results/inventory.json").read_text())
    comps0 = vpd.component_weights(g.net.model)
    for state in ("c3/cipher_s0", "e53/lowercase_s0"):
        task = inv["states"][state]["task"]
        r, d = WIN.load_read(task, inv["tasks"][task]), dw(inv["states"][state]["file"])
        base = route_cells(g.net.model, r.toks, r.half, r.token, r.batch, r.ci_len)
        with model_state(g.net.model, target={k: w + d[k].to(w.device) for k, w in g.net.W0.items()}):
            rs = route_split(g.net.model, r.toks, r.half, r.token, base, comps0, r.batch, r.ci_len)
        o.put(f"share/{state}@b{r.batch}", rs["shares"]["r"], ("experiments/share_untrained/results/stage1.json", ["states", state, "trained"]),
              batch=r.batch, rows=f"E136 {task} ({len(r.toks)})", unit="lo", dtype="bf16 autocast", floor="|gain| >= 1e-6", gain=rs["gain"])  # fmt: skip


def share_e211(g: Gpu, o: Out) -> None:
    """E211's ctl: realistic_reads.share_of on E53's scored windows (masked_reads batch 64), floor MIN_GAIN one-sided."""
    from commute import realistic_reads as RR

    R2: Any = load_experiment("realistic_tasks", "run2")
    rec = RR.share_of(g.net, R2.e53_read("lowercase"), dw("experiments/performed_anchor/results/large/dW_lowercase_s0_step300.pt"))
    o.put("share/e53/lowercase_s0@b64", rec["share"]["lo"], ("experiments/realistic_tasks/results/run2_ctl.json", ["reproduce e53/lowercase_s0", "share", "got"]),
          batch=64, rows="E53 scored windows (500)", unit="lo", dtype="bf16 autocast", floor="gain >= 1 (MIN_GAIN)", gain=rec["gain"]["lo"])  # fmt: skip


def share_e140(g: Gpu, o: Out) -> None:
    """real_vs_twin stage 2: masked_reads.share_cells on the first n_read fit windows of case real, MR.share per unit."""
    from commute import masked_reads as MR

    RVT: Any = load_experiment("real_vs_twin", "run")
    P = json.loads(RVT.PRED.read_text())
    p = RVT.load_pair("case")
    toks, y = p.part("real", "fit", P["settings"]["n_read"])
    cells = MR.share_cells(g.net.model, g.net.W0, RVT.state(p, "real", 0, {}), toks, y, p.ab, RVT.HALF)
    for u in ("lo", "contrast"):
        v = MR.share(cells["L01"][u] - cells["L00"][u], cells["T1"][u] - cells["T0"][u], P["thresholds"]["gain_floor"])
        o.put(f"share/case_real_s0@{u}", v, ("experiments/real_vs_twin/results/stage2.json", ["states", "case_real_s0", "share", u]),
              batch=MR.BATCH, rows=f"case_label fit[:{len(toks)}]", unit=u, dtype="bf16 autocast", floor=f"gain >= {P['thresholds']['gain_floor']}")  # fmt: skip


def steering_e135(g: Gpu, o: Out) -> None:
    """E135 stage 1, copy_s0's pooled vector: fit half (even windows) selection, score half read, its own fp32 copies."""
    SC: Any = load_experiment("steering_copy_lookup", "run")
    s = json.loads(SC.PRED.read_text())["settings"]
    c, run62, large = SC.Cfg(s["half"], s["n"], tuple(s["strengths"]), s["n_random"], s["random_seed"]), json.loads((SC.E62_DIR / "run.json").read_text()), (SC.E62_DIR / "large").resolve()
    p, _ = SC.load_pair(c, large, run62)
    states = SC.load_states(large, run62)
    w0 = SC.baselines(g.net, p)
    G = SC.gains(g.net, p, states, w0)
    vs = SC.vectors(g.net, p, states["copy_s0"])
    rec = SC.kind_record(g.net, p, c, "copy", "lookup", vs["pooled"], w0, G["copy_s0"], G["lookup_s0"], torch.Generator().manual_seed(c.random_seed))
    o.put("steering_pooled/e62/copy_s0", rec["share_lo"], ("experiments/steering_copy_lookup/results/stage1.json", ["states", "copy_s0", "steering", "pooled", "share_lo"]),
          select_rows="fit half (100)", read_rows="score half (100)", key_pos="half", unit="lo", dtype="fp32", floor="none (G != 0)", setting=[rec["layer"], rec["strength"]])  # fmt: skip


def steering_e143(g: Gpu, o: Out) -> None:
    """E143 `Job.state`: pooled vector from every training window, selection on the first n_select, read on E136's rows."""
    HF: Any = load_experiment("holes_fill", "run")
    P = json.loads(HF.PRED.read_text())
    inv = json.loads((HF.E136 / "inventory.json").read_text())
    job, name = HF.Job(g.net, inv, P["settings"], P["thresholds"]), "e46/ident32_s0"
    pairs = {q["id"]: q for q in inv["pairs"] if q["id"] in HF.PAIRS}
    i = list(dict.fromkeys(s for q in HF.PAIRS for s in pairs[q]["reuse"] + pairs[q]["new"])).index(name)
    t, d = job.task(inv["states"][name]["task"]), job.dW(name)
    gain = float((HF.lo_at(g.net, t, t.read.toks, d) - t.lo0).mean())
    assert t.train is not None and t.pool is not None
    vs = HF.FS.vectors(g.net, t.x(t.train), t.idx(t.train), len(t.pool), d, t.read.half)
    rec = job.kind(t, vs["pooled"], gain, torch.Generator().manual_seed(P["settings"]["random_seed"] + i))
    o.put(f"steering_pooled/{name}", rec["share"], ("experiments/holes_fill/results/stage1.json", ["states", name, "steering", "pooled", "share"]),
          select_rows=f"train[:{P['settings']['n_select']}]", read_rows=f"E136 held ({len(t.read.toks)})", key_pos="read position", unit="lo",
          dtype="fp32", floor=f"gain >= {P['thresholds']['gain_floor']}", setting=rec["setting"])  # fmt: skip


def kl_procedural(g: Gpu, o: Out) -> None:
    """E256's kl_saved (rl_67m `kl_from_base`: KL(W0 || F), every position, fp32 per position) and the other direction
    (rule_retention.kl: KL(F || W0) per window, fp64, every position) on the same rows; the second has no saved value."""
    from commute import rule_retention as R
    from commute.data import cached_rows

    PRL: Any = load_experiment("procedural_rl", "run")
    s = json.loads(PRL.PRED.read_text())["settings"]
    rows, d = cached_rows("val", 0, s["kl_rows"]), dw(SEQ_RL)
    o.put("kl_base_to_ft/sequence_rl_s0", PRL.S1.kl_from_base(g.net, rows, d, s["kl_batch"]), ("experiments/procedural_rl/results/train_sequence.json", ["states", "sequence_rl_s0", "kl_saved"]),
          batch=s["kl_batch"], rows=f"cached val 0..{s['kl_rows']}", positions="all", dtype="fp32 per position, fp64 sum")  # fmt: skip
    per: list[torch.Tensor] = []
    for b in rows.split(s["kl_batch"]):
        l0 = g.net.run(b)[0]
        with g.net.at(d):
            per.append(R.kl(g.net.run(b)[0], l0).cpu())
    o.put("kl_ft_to_base/sequence_rl_s0", float(torch.cat(per).mean()), None, batch=s["kl_batch"], rows=f"cached val 0..{s['kl_rows']}", positions="all", dtype="fp64")


def panel(g: Gpu, o: Out) -> None:
    """E272 `readings` on lower_s0_step128 (its wide-set read): every panel reading at once."""
    CP: Any = load_experiment("convergence_panel", "run")
    C, _ = CP.load()
    rd = CP.reader(g.net, C, CP.TK.build(g.net, C, "lower").read)
    rec = CP.readings(rd, CP.load_dW(LOWER_128), CP.rows_kl(C), CP.rows_adl(C), C, 0)
    at = ["states", LOWER_128]
    common = {"rows": f"lower read ({len(rd.r.toks)})", "unit": rd.r.scored}
    for key, name, extra in (("share", "share", {"batch": 64, "dtype": "bf16 autocast", "floor": "gain >= 1"}), ("steering", "steering_pooled", {"select_rows": f"fit[:{C.n_select}]", "dtype": "fp32", "floor": "gain >= 1"}),
                             ("kl", "kl_base_to_ft", {"rows": f"cached val 0..{C.kl_rows}", "positions": "all", "batch": 8}), ("prior", "prior_mean_lo", {"path": "steered(None) at W0"}),
                             ("gain", "gain", {}), ("size", "size_rel", {"W0": "live"}), ("erank", "erank", {})):  # fmt: skip
        o.put(f"{name}/lower_s0_step128", rec[key], (PANEL_READ, [*at, key]), **(common | extra))
    o.put("adl_D/lower_s0_step128", rec["adl"]["D"], (PANEL_READ, [*at, "adl", "D"]), **common, pile_rows=f"val 0..{C.adl_rows}[:, :{C.adl_k}]", batch=C.adl_batch, floor="gain >= 1")


def prior_forward(g: Gpu, o: Out) -> None:
    """E133 `forward_prior` (keydep.geo_rate of W0's log P on a kv task's held-out windows, one full batch)."""
    PVS: Any = load_experiment("prior_vs_share", "run")
    row = next(m for m in json.loads((REPO / "experiments/prior_vs_share/results/inventory.json").read_text())["maps"] if m["name"] == "e46_ident32")
    o.put("prior_geo/e46_ident32", PVS.forward_prior(g.net, row), ("experiments/prior_vs_share/results/stage1.json", ["e46_ident32", "prior"]),
          rows="regating_table windows_ident32 heldout", batch="all at once", dtype="fp32", unit="P (geometric mean)")  # fmt: skip


def eb_fresh(g: Gpu, o: Out) -> None:
    """Fresh write passes: x2 at VPD (E257's rb_states write_pass, E268's world) and at neurons (E277's write_pass), and
    copy24_s0 at neurons (E277's trained world), each over its saved K; then copy24_s0's neuron K scan (E277's)."""
    from commute import needed as IX
    from commute import needed_neurons as NN

    KN: Any = load_experiment("kstar_needed", "run")
    NR: Any = load_experiment("needed_neurons", "run")
    cfg, c229 = NR.load_config(), NR.load_e229()
    feat = NR.E229.feat_of(c229, NR.E229.load_pct(c229), cfg.band)
    w = KN.world()
    a, _ = KN.RB.write_pass(w, "x2")
    K = npz("experiments/kstar_needed/results/large/items_x2.npz")["in_K"].bool()
    o.put("E_b_vpd/x2@fresh", SI.explained_share(a["num"], a["den"], a["dd"], K & a["changed"]), ("experiments/kstar_needed/results/items/x2.json", ["E_b_K"]),
          sums="rb_states write_pass (fresh)", K="kstar_needed in_K", sel="K & changed", dtype="fp32 forward, fp64 sums", batch=KN.E242.load_config()[0].gain_batch)  # fmt: skip
    F, M, toks, _, _ = w.make("x2")
    num, den, dd, ch, _ = NR.write_pass(F, M, toks, feat, cfg.cache_batch)
    z = npz("experiments/needed_neurons/results/large/items_x2.npz")
    o.put("E_b_neurons/x2@fresh", SI.explained_share(num, den, dd, z["in_K"].bool() & ch), ("experiments/needed_neurons/results/items/x2.json", ["E_b_K"]),
          sums="needed_neurons write_pass (fresh)", K="needed_neurons in_K", sel="K & changed", batch=cfg.cache_batch, dd_max_dev=float((dd - z["dd"]).abs().max()))  # fmt: skip
    del F, M, w
    torch.cuda.empty_cache()
    tw = NR.Trained(cfg, c229)
    F, M, toks, at, _ = tw.make("copy24_s0")
    num, den, dd, ch, _ = NR.write_pass(F, M, toks, feat, cfg.cache_batch)
    z = npz("experiments/needed_neurons/results/large/items_copy24_s0.npz")
    o.put("E_b_neurons/copy24_s0@fresh", SI.explained_share(num, den, dd, z["in_K"].bool() & ch), ("experiments/needed_neurons/results/items/copy24_s0.json", ["E_b_K"]),
          sums="needed_neurons write_pass (fresh)", K="needed_neurons in_K", sel="K & changed", batch=cfg.cache_batch, dd_max_dev=float((dd - z["dd"]).abs().max()))  # fmt: skip
    run = NN.cache(F.net, toks, at, cfg.cache_batch)
    m = min(cfg.examples, len(toks))
    ex, pt = IX.derangements(len(toks), cfg.derangements, m, cfg.seed)
    sc = NN.scan(F.net, feat, run, at, ex, pt, NR.rows_for(cfg, run.top.shape[1]), print)
    Kn = sc.first_pair >= 0
    o.put("needed_set_size/copy24_s0@neurons", int(Kn.sum()), ("experiments/needed_neurons/results/items/copy24_s0.json", ["size_K"]),
          derangements=cfg.derangements, examples=m, seed=cfg.seed, mask_equals_saved=bool(torch.equal(Kn, z["in_K"].bool())))  # fmt: skip


DATA = {"eb_saved_sums": eb_saved_sums, "spectra_and_sizes": spectra_and_sizes, "saved_priors": saved_priors}
GPU = {"share_e136": share_e136, "share_e211": share_e211, "share_e140": share_e140, "prior_forward": prior_forward, "kl_procedural": kl_procedural,
       "steering_e135": steering_e135, "steering_e143": steering_e143, "panel": panel, "eb_fresh": eb_fresh}  # fmt: skip


def main(tag: str, part: str, only: list[str]) -> int:
    o = Out(tag)
    if part == "data":
        for n, f in DATA.items():
            if not only or n in only:
                o.run(n, f)
    else:
        g = Gpu()
        w0_check(g, o)
        for n, gf in GPU.items():
            if not only or n in only:
                o.run(n, lambda o, gf=gf: gf(g, o))
        print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"OLD {tag} {part} DONE; errors: {o.errors or 'none'}", flush=True)
    return 1 if o.errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="names the run: results/old_<tag>.json (A, B)")
    ap.add_argument("--part", choices=("data", "gpu"), required=True)
    ap.add_argument("--only", nargs="*", default=[], help="run only these groups (names in DATA / GPU)")
    a = ap.parse_args()
    hard_exit(main(a.tag, a.part, a.only))
