"""E301's circuits and built edits, by the existing builders (run.py drives the stages; plan
reports/plans/2026-11-24_plan_p1-63-more-circuits.md).

Windows (copy24's format: a key -> value table read at one position `half`, so E195's attribution applies unchanged):
    induction   21 distinct tokens of ids induction_pool shown twice, the last cut (41 tokens), read at 40 (seq[19]);
                seq[19] -> seq[20] drawn from a table of induction_pairs pairs; seeds (table, training, held-out)
    bigram      41-token Pile spans with a key at 20; a key is a token whose next token in the count rows is one token in
                at least bigram_min_share of at least bigram_min_count occurrences; its value that token
Builders, each called unchanged: C and x2/x10 by experiments/rule_retention/run.py (`attribution`, `rescaled`, on a
`Data` built here); rebuilt by experiments/rule_retention_fixed/run.py (`zeroed`, `fit`, `trained`); new_few/new_thin by
`commute.known_edits.build_swapping` on E214's World's cue windows (experiments/kstar_levels/run.py) at slot_layer = C's
layer; randoms by experiments/needed_neurons/run.py `matched_random` (E290's rule).
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from commute import known_edits as KE
from commute import old_tasks as OT
from commute import rule_retention as R
from commute.data import pile_rows
from commute.jobs import EXPERIMENTS, config_from_settings, load_experiment

RU: Any = load_experiment("rule_retention", "run")  # E193: Data, cut, attribution, rescaled, mean_kl, clone
E195: Any = load_experiment("rule_retention_fixed", "run")  # zeroed, trained, fit
KL: Any = load_experiment("kstar_levels", "run")  # E214: Config, World (M, cues), NOT_BUILT
NNR: Any = load_experiment("needed_neurons", "run")  # E277: matched_random (E290's rule)
VOCAB = 50277


def kl_world() -> Any:
    """E214's World on its filed settings and thresholds: the 67M (`.net`, `.M`) and the cue windows (`.cues`)."""
    pred = json.loads((EXPERIMENTS / "kstar_levels" / "results" / "predictions.json").read_text())
    s = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    w = KL.World(config_from_settings(KL.Config, s), pred["thresholds"])
    w.net.net.requires_grad_(False)
    return w


# ---- windows ----------------------------------------------------------------------------------------------------------


def induction(cfg: Any) -> dict[str, Any]:
    lo, hi = cfg.induction_pool
    pk = torch.arange(lo, hi)[torch.randperm(hi - lo, generator=torch.Generator().manual_seed(cfg.induction_seeds[0]))]
    n = cfg.induction_pairs
    keys, vals, rest = pk[:n], pk[n : 2 * n], pk[2 * n :]
    table = torch.full((VOCAB,), -1, dtype=torch.long)
    table[keys] = vals
    L = (cfg.window + 1) // 2

    def draw(m: int, seed: int) -> Tensor:
        g = torch.Generator().manual_seed(seed)
        s = torch.stack([rest[torch.randperm(len(rest), generator=g)[:L]] for _ in range(m)])
        j = torch.randint(0, n, (m,), generator=g)
        s[:, L - 2], s[:, L - 1] = keys[j], vals[j]
        return torch.cat([s, s], 1)[:, :-1]

    return {"train": draw(cfg.n_train, cfg.induction_seeds[1]), "held": draw(cfg.n_held, cfg.induction_seeds[2]), "table": table}


def bigram(cfg: Any) -> dict[str, Any]:
    rows = pile_rows(*cfg.bigram_rows["count"])
    a, b = rows[:, :-1].flatten(), rows[:, 1:].flatten()
    pair, cnt = torch.unique(a * VOCAB + b, return_counts=True)
    ka, kb = pair // VOCAB, pair % VOCAB
    tot = torch.zeros(VOCAB, dtype=torch.long).index_add_(0, ka, cnt)
    best = torch.zeros(VOCAB, dtype=torch.long).scatter_reduce(0, ka, cnt, "amax")
    mode = torch.full((VOCAB,), -1, dtype=torch.long)
    for i in torch.argsort(cnt, descending=True).tolist():  # the first pair seen per key in count order is its mode
        if mode[int(ka[i])] < 0:
            mode[int(ka[i])] = int(kb[i])
    share = best.double() / tot.clamp(min=1)
    keys = ((tot >= cfg.bigram_min_count) & (share >= cfg.bigram_min_share) & (mode != torch.arange(VOCAB))).nonzero().squeeze(1)
    table = torch.full((VOCAB,), -1, dtype=torch.long)
    table[keys] = mode[keys]
    h = cfg.half["bigram"]

    def spans(part: str, n: int, seed: int) -> Tensor:
        src = pile_rows(*cfg.bigram_rows[part])
        pos = (table[src] >= 0).nonzero()
        pos = pos[(pos[:, 1] >= h) & (pos[:, 1] <= src.shape[1] - (cfg.window - h))]
        pick = pos[torch.randperm(len(pos), generator=torch.Generator().manual_seed(seed))[:n]]
        assert len(pick) == n, f"bigram {part}: {len(pos)} key positions, {n} wanted"
        return torch.stack([src[r, p - h : p - h + cfg.window] for r, p in pick.tolist()])

    return {"train": spans("train", cfg.n_train, cfg.bigram_seeds[0]), "held": spans("held", cfg.n_held, cfg.bigram_seeds[1]),
            "table": table, "n_keys": len(keys)}  # fmt: skip


WINDOWS = {"induction": induction, "bigram": bigram}


@torch.no_grad()
def top1(M: R.Model, toks: Tensor, half: int, table: Tensor, batch: int = 64) -> float:
    """M's top-1 at the read position against the table's answer."""
    hit = [M.logits(t)[:, half].argmax(-1) == table.to(t.device)[t[:, half]] for t in toks.split(batch)]
    return float(torch.cat(hit).double().mean())


def data(cfg: Any, win: dict[str, Any], half: int, device: str) -> Any:
    """E193's Data on this circuit's windows; the Pile halves cut as rule_retention/run.py load_data cuts them."""
    gen = torch.Generator().manual_seed(cfg.data_seed)
    pt = RU.cut(pile_rows("train", 0, 4096), cfg.pile_train, cfg.window, gen)
    ph = RU.cut(pile_rows("val", 0, 256), cfg.pile_held, cfg.window, gen)
    return RU.Data(win["train"].to(device), win["held"].to(device), win["table"].to(device), half, pt.to(device), ph.to(device))


# ---- the edits --------------------------------------------------------------------------------------------------------


def circuit_C(M: R.Model, d: Any, layer: int, k: int, batch: int) -> tuple[list[int], dict[str, Any]]:
    A = RU.attribution(M, d, batch)
    C = sorted(A[layer].abs().topk(k).indices.tolist())
    return C, {"C": C, "attribution_C": A[layer, C].tolist(), "signed_sum": float(A[layer, C].sum()), "mass": A.abs().sum(1).tolist()}


def fit_ns(cfg: Any, layer: int, C: list[int], trained: list[int], seed: int) -> SimpleNamespace:
    keys = ("lr", "warmup", "batch", "steps", "eval_batch", "log_every", "check_step", "target")
    return SimpleNamespace(**{k: getattr(cfg, k) for k in keys}, fit_seed=seed, layer=layer, C=tuple(C), trained=tuple(trained))


def rebuilt(M: R.Model, d: Any, ns: SimpleNamespace) -> tuple[R.Model, list[dict[str, float]]]:
    """C zeroed, the trained layers' MLP fitted to KL(x<target> || F) by E195's fit (unchanged)."""
    T, F = RU.rescaled(M, ns.layer, list(ns.C), ns.target), E195.zeroed(M, ns)
    curve = E195.fit(F, T, d, ns)
    return F, curve


def changed(F: R.Model, M: R.Model) -> dict[str, Tensor]:
    """F's parameters that differ from M's, by name, on CPU (what is saved of a built edit)."""
    mp = dict(M.net.named_parameters())
    return {k: p.detach().cpu().clone() for k, p in F.net.named_parameters() if not torch.equal(p, mp[k])}


def load(net: Any, params: dict[str, Tensor]) -> R.Model:
    """M with the saved parameters copied in."""
    F = R.Model(OT.clone_net(net).net)
    with torch.no_grad():
        for k, v in params.items():
            F.net.get_parameter(k).copy_(v)
    return F


def new_edits(w: Any, slot_layer: int) -> tuple[dict[str, R.Model], dict[str, Any], Any]:
    """E214's new_few and new_thin at `slot_layer` on its cue windows, E214's settings otherwise; reuse_* dropped."""
    cfg = dataclasses.replace(w.cfg, slot_layer=slot_layer)
    pile = RU.cut(pile_rows("val", 0, 256), cfg.eval_batch, cfg.window, torch.Generator().manual_seed(cfg.data_seed))

    def make_cues(skip: tuple[int, ...]) -> Any:
        w.skip = skip
        w.__dict__.pop("cues", None)
        return w.cues

    ns = SimpleNamespace(**dataclasses.asdict(cfg), a_prime_copy_max=w.th["a_prime_copy_max"])
    edits, rec, cues = KE.build_swapping(w.M, make_cues, pile, ns, RU.log)
    rec["misses"] = [m for m in rec["misses"] if not m.startswith(KL.NOT_BUILT)]
    return {k: v for k, v in edits.items() if k not in KL.NOT_BUILT}, rec, cues


def random_of(F: R.Model, M: R.Model, net: Any, seed: int) -> tuple[R.Model, dict[str, Any]]:
    """E290's random: a Gaussian per tensor F changed, at that tensor's Frobenius norm of F - M."""
    dw, rec = NNR.matched_random(F.net, M.net, seed)
    return R.Model(OT.with_delta(net, dw).net), rec


def norm_check(F: R.Model, Fr: R.Model, M: R.Model, tol: float) -> list[str]:
    """The random changes exactly F's tensors, each at F's norm within tol (relative)."""
    a, b = changed(F, M), changed(Fr, M)
    mp = {k: v.detach().cpu() for k, v in M.net.named_parameters()}
    if set(a) != set(b):
        return [f"random changes {sorted(b)} not {sorted(a)}"]
    rel = {k: abs(float((b[k] - mp[k]).double().norm() / (a[k] - mp[k]).double().norm()) - 1) for k in a}
    return [f"{k} norm off by {v:.3g}" for k, v in rel.items() if v > tol]
