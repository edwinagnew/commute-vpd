"""E206 (cell P1.28): grafts at other sites, read by the held-model definition (plan
reports/plans/2026-11-18_plan_graft-sites.md; card card.md; settings, their sources, the build checks and the filed rules
in results/predictions.json, which the run refuses to start without, committed unchanged). Descriptive: no cut-offs and
no labels; the rules are comparisons between survivals.

M (the 67M) is frozen. Each graft is fresh machinery added to a copy of M and trained alone (`commute.graft`), then read
with E197's held model (`read` of experiments/held_model_retention/run.py). Six grafts, by role:
    last_lowest   a head on the last layer at the offset k whose largest M-head share is lowest (E201's rule; a rebuild
                  of E201's graft with a new seed)
    neurons       n_neurons MLP neurons on the last layer, trained to output a fixed random derangement of the current
                  token over the n_lookup most frequent tokens (every position whose token is in that set)
    early_lowest  a head on layer `early_layer` at the same k
    last_rank<r>  a head on the last layer at the offset ranked r by largest M-head share (rank 1 = lowest), r in `ranks`
Stages:
    0  controls, alone, stopping on a miss: E197's do-nothing check; zero-write grafts (a head on the early layer and the
       neurons on the last, output weights zero) read through the held model, which must give H = M with the graft units
       unpaired; the attention-pattern check; on M the head shares over offsets, which must repeat E201's on the same
       windows within share_tol; the offsets by rank; the lookup set, its derangement, M's accuracy on it and M's mean
       log-probability of it against n_null other derangements.
    per graft, in the order above, each writing results/grafts/<role>_<part>.json when it ends and skipped once it has:
       build   train (one attempt), save results/large/<role>.pt
       checks  held-out accuracy of F and M; the graft's own mechanism (a head's attention at k from the last position;
               the neurons' write alone, read through the unembedding, picking the target among the lookup set); F with
               the graft's output zeroed against M. A miss: reported, not read; the next graft runs.
       read    the three readings, the graft units' pairing, the graft-site sizes; reading 3's site map saved (bf16)
    summary.json: S per graft and reading, and the filed rules evaluated.

    python experiments/graft_sites/run.py 2>&1 | tee -a experiments/graft_sites/results/run.log
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from commute import graft as G
from commute import held_model as HM
from commute import rule_retention as R
from commute.jobs import hard_exit, load_experiment, require_filed

HR: Any = load_experiment("held_model_retention", "run")  # E197: read, misses, identity, CHECKS
GN: Any = load_experiment("graft_negative", "run")  # E201: data (the same windows), its stage0.json
P = HR.P
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, LARGE, GRAFTS = RESULTS / "predictions.json", RESULTS / "large", RESULTS / "grafts"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop = P.log, P.dump, P.stop
chunks = G.chunks


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"; the data fields are E201's, so the windows are E201's
    offsets: tuple[int, int]
    window: int
    train_rows: int
    val_rows: int
    n_probe: int
    probe_seed: int
    n_acc: int
    acc_seed: int
    n_fit: int
    fit_seed: int
    n_read: int
    read_seed: int
    sum_batch: int
    eval_batch: int
    last_layer: int
    early_layer: int
    ranks: tuple[int, ...]
    n_neurons: int
    n_lookup: int
    lookup_seed: int
    n_null: int
    graft_seed: int
    recipe: G.Recipe

    def __post_init__(self) -> None:
        lo, hi = self.offsets
        assert 0 < lo < hi and hi + 2 < self.window and 0 <= self.early_layer < self.last_layer
        assert all(1 < r <= hi - lo + 1 for r in self.ranks) and len(set(self.ranks)) == len(self.ranks)


@dataclass(frozen=True)
class Spec:
    role: str
    kind: str  # "head" or "neurons"
    layer: int
    k: int  # a head's offset; 0 for the neurons
    seed: int  # the graft's init; its training draws use seed + 1

    @property
    def site(self) -> HM.Site:
        return ("attn" if self.kind == "head" else "mlp", self.layer)


# ---- the two tasks -------------------------------------------------------------------------------------------------


@dataclass
class Shift:
    """A head's task: at the last position, the token k back."""

    k: int

    def loss(self, F: R.Model, toks: Tensor) -> Tensor:
        return nn.functional.cross_entropy(F.logits(toks)[:, -1], toks[:, -1 - self.k])

    @torch.no_grad()
    def accuracy(self, N: R.Model, toks: Tensor, batch: int) -> float:
        return sum(int((N.logits(t)[:, -1].argmax(-1) == t[:, -1 - self.k]).sum()) for t in chunks(toks, batch)) / toks.shape[0]

    def mechanism(self, F: R.Model, g: nn.Module, layer: int, toks: Tensor, batch: int) -> float:
        assert isinstance(g, G.Head)
        return G.head_attention(F, g, layer, toks, self.k, batch)


@dataclass
class Lookup:
    """The neurons' task: at every position whose token is in `tokens`, the token sigma(x) = tokens[perm[i]] for
    x = tokens[i], perm a derangement. `target` maps every vocabulary id to its target, -1 outside the set."""

    tokens: Tensor
    perm: Tensor
    target: Tensor

    def loss(self, F: R.Model, toks: Tensor) -> Tensor:
        y = self.target[toks]
        return nn.functional.cross_entropy(F.logits(toks)[y >= 0], y[y >= 0])

    @torch.no_grad()
    def accuracy(self, N: R.Model, toks: Tensor, batch: int) -> float:
        hit, n = 0, 0
        for t in chunks(toks, batch):
            y = self.target[t]
            hit, n = hit + int((N.logits(t)[y >= 0].argmax(-1) == y[y >= 0]).sum()), n + int((y >= 0).sum())
        return hit / n

    @torch.no_grad()
    def mechanism(self, F: R.Model, g: nn.Module, layer: int, toks: Tensor, batch: int) -> float:
        """The share of set positions where the neurons' write alone, read through the final norm's weight and the
        unembedding (a positive per-position scale does not change the argmax), ranks the target first in the set."""
        U = (F.net.lm_head.weight[self.tokens] * F.net.ln_f.weight).T
        hit, n = 0, 0
        for t in chunks(toks, batch):
            y, res = self.target[t], R.m_inputs(F, t)[layer][1]
            w = g(F.net.h[layer].rms_2(res))[y >= 0]
            hit, n = hit + int((self.tokens[(w @ U).argmax(-1)] == y[y >= 0]).sum()), n + int((y >= 0).sum())
        return hit / n


def derangement(n: int, gen: torch.Generator) -> Tensor:
    while bool(((p := torch.randperm(n, generator=gen)) == torch.arange(n)).any()):
        pass
    return p


def lookup(train: Tensor, cfg: Config, vocab: int, dev: str) -> Lookup:
    """The n_lookup most frequent tokens of the graft's training rows (ties to the lower id) and a seeded derangement."""
    counts = torch.bincount(train.flatten(), minlength=vocab)
    tokens = torch.argsort(counts, descending=True, stable=True)[: cfg.n_lookup]
    perm = derangement(cfg.n_lookup, torch.Generator().manual_seed(cfg.lookup_seed))
    target = torch.full((vocab,), -1, dtype=torch.long)
    target[tokens] = tokens[perm]
    return Lookup(tokens.to(dev), perm.to(dev), target.to(dev))


@torch.no_grad()
def null_rank(M: R.Model, toks: Tensor, lk: Lookup, cfg: Config) -> dict[str, Any]:
    """M's mean log-probability of sigma(x) at the set positions, against n_null other derangements of the same set."""
    index = torch.full_like(lk.target, -1)
    index[lk.tokens] = torch.arange(len(lk.tokens), device=index.device)
    lp, ix = [], []
    for t in chunks(toks, cfg.eval_batch):
        m = lk.target[t] >= 0
        lp.append(M.logits(t).log_softmax(-1)[m][:, lk.tokens].double())
        ix.append(index[t][m])
    LP, I = torch.cat(lp), torch.cat(ix)

    def score(perm: Tensor) -> float:
        return float(LP[torch.arange(len(I), device=I.device), perm[I]].mean())

    gen = torch.Generator().manual_seed(cfg.lookup_seed + 1)
    ours, null = score(lk.perm), torch.tensor([score(derangement(len(lk.tokens), gen).to(I.device)) for _ in range(cfg.n_null)])
    return {"M_logp": ours, "null_mean": float(null.mean()), "null_sd": float(null.std()), "null_share_at_or_above": float((null >= ours).double().mean()), "positions": len(I)}


# ---- stage 0 -------------------------------------------------------------------------------------------------------


def zero_grafts(M: R.Model, cfg: Config) -> tuple[R.Model, Any]:
    """M with a head on the early layer and the neurons on the last, output weights zero; planted: every M unit with
    itself at gain 1, every graft unit unpaired (it writes nothing)."""
    F, planted = P.clone(M), HR.identity(M)
    for g, site, n in ((G.attach_head(F, cfg.early_layer, cfg.graft_seed, cfg.recipe), ("attn", cfg.early_layer), 1),
                       (G.attach_neurons(F, cfg.last_layer, cfg.n_neurons, cfg.graft_seed, cfg.recipe), ("mlp", cfg.last_layer), cfg.n_neurons)):  # fmt: skip
        with torch.no_grad():
            G.out_param(g.requires_grad_(False)).zero_()
        p, gain = planted[site]
        planted[site] = (torch.cat([p, torch.full((n,), -1)]), torch.cat([gain, torch.full((n,), float("nan"), dtype=torch.float64)]))
    return F, planted


def stage0(M: R.Model, d: Any, lk: Lookup, cfg: Config, th: dict[str, Any], want: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"experiment": EXPERIMENT}
    bad: list[str] = []
    for name, (F, planted) in (("do_nothing", HR.CHECKS["do_nothing"](M, cfg)), ("zero_grafts", zero_grafts(M, cfg))):
        fit, rd = HR.read(F, M, d.fit, d.read, cfg, th, name)
        rd["misses"] = HR.misses(rd, fit, planted, want[name], th)
        log(f"{name}: {'checks hold' if not rd['misses'] else 'MISSES ' + '; '.join(rd['misses'])}")
        out[name], bad = rd, bad + [f"{name}: {m}" for m in rd["misses"]]
        del F, fit
    sh, err = G.shares(M, d.probe, cfg.offsets, cfg.eval_batch)
    top = sh.flatten(0, 1).max(0).values
    ks = list(range(cfg.offsets[0], cfg.offsets[1] + 1))
    e201 = json.loads((GN.RESULTS / "stage0.json").read_text())["max_share_by_offset"]
    order = [ks[i] for i in torch.argsort(top, stable=True).tolist()]  # rank 1 = the lowest largest share; ties to the smaller offset
    out |= {"shares": sh.tolist(), "max_share_by_offset": dict(zip(ks, top.tolist(), strict=True)), "pattern_error": err,
            "e201_share_off": max(abs(float(top[i]) - e201[str(k)]) for i, k in enumerate(ks)), "order": order,
            "k_lowest": order[0], "k_by_rank": {str(r): order[r - 1] for r in cfg.ranks}}  # fmt: skip
    log(f"pattern check: max relative error {err:.3g} (tol {th['pattern_tol']}); largest M head share by offset {out['max_share_by_offset']}; "
        f"off E201's by {out['e201_share_off']:.3g} (tol {th['share_tol']}); offsets by rank {order}; k lowest {order[0]}, by rank {out['k_by_rank']}")
    bad += [f"pattern error {err:.3g}"] if not err <= th["pattern_tol"] else []
    bad += [f"shares off E201's by {out['e201_share_off']:.3g}"] if not out["e201_share_off"] <= th["share_tol"] else []
    out["lookup"] = {"tokens": lk.tokens.tolist(), "perm": lk.perm.tolist(), "acc_M": lk.accuracy(M, d.acc, cfg.eval_batch), "null": null_rank(M, d.acc, lk, cfg)}
    log(f"lookup: M accuracy {out['lookup']['acc_M']:.4f}; M mean log-probability of the derangement {out['lookup']['null']}")
    if bad:
        stop(RESULTS / "stage0.json", out, "stage 0: " + "; ".join(bad))
    return out | {"done": True}


# ---- one graft -----------------------------------------------------------------------------------------------------


def specs(cfg: Config, s0: dict[str, Any]) -> list[Spec]:
    k, last, early = s0["k_lowest"], cfg.last_layer, cfg.early_layer
    roles = [("last_lowest", "head", last, k), ("neurons", "neurons", last, 0), ("early_lowest", "head", early, k)]
    roles += [(f"last_rank{r}", "head", last, s0["k_by_rank"][str(r)]) for r in cfg.ranks]
    return [Spec(role, kind, l, kk, cfg.graft_seed + 10 * (i + 1)) for i, (role, kind, l, kk) in enumerate(roles)]


def attach(M: R.Model, s: Spec, cfg: Config) -> tuple[R.Model, nn.Module]:
    F = P.clone(M)
    g = G.attach_head(F, s.layer, s.seed, cfg.recipe) if s.kind == "head" else G.attach_neurons(F, s.layer, cfg.n_neurons, s.seed, cfg.recipe)
    return F, g


def build(M: R.Model, s: Spec, task: Any, d: Any, cfg: Config) -> dict[str, Any]:
    F, g = attach(M, s, cfg)
    out = G.train(g, lambda t: task.loss(F, t), d.train, cfg.window, cfg.recipe, s.seed + 1, lambda: task.accuracy(F, d.acc, cfg.eval_batch), log)
    out |= {"spec": asdict(s), "sha256": None if out["failed"] else G.save(g, LARGE / f"{s.role}.pt")}
    log(f"{s.role}: build {'FAILED ' + out['failed'] if out['failed'] else 'saved results/large/' + s.role + '.pt sha256 ' + out['sha256']}")
    return out


def checks(M: R.Model, F: R.Model, g: nn.Module, s: Spec, task: Any, d: Any, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    out = {"acc_F": task.accuracy(F, d.acc, cfg.eval_batch), "acc_M": task.accuracy(M, d.acc, cfg.eval_batch),
           "mechanism": task.mechanism(F, g, s.layer, d.probe, cfg.eval_batch), "zeroed_logit_error": G.zeroed_error(F, M, g, d.probe, cfg.eval_batch)}  # fmt: skip
    out["misses"] = [f"{k} {out[k]:.4g}" for k, ok in (("acc_F", out["acc_F"] >= th["acc_min"]), ("mechanism", out["mechanism"] >= th["mech_min"]),
                                                     ("zeroed_logit_error", out["zeroed_logit_error"] <= th["logit_tol"])) if not ok]  # fmt: skip
    log(f"{s.role} build checks: held-out accuracy F {out['acc_F']:.4f}, M {out['acc_M']:.4f}; mechanism {out['mechanism']:.4f}; "
        f"graft zeroed vs M logit error {out['zeroed_logit_error']:.3g}; {'MISSES ' + '; '.join(out['misses']) + ' (not read)' if out['misses'] else 'hold'}")
    return out


def units(fit: HM.Fit, M: R.Model, site: HM.Site, th: dict[str, Any]) -> dict[str, Any]:
    """Readings 1 and 2: whether every M unit's F counterpart is paired with it at gain 1 (every site), and how many of
    the graft's units the assignment paired (an MLP graft under reading 2 is inside the block unit: none of its own)."""
    out: dict[str, Any] = {}
    for r, ps in fit.pairs.items():
        same, dev, n_site = True, 0.0, 0
        for s, p in ps.items():
            WM = HM.out_weight(M, s)
            n = WM.shape[1] // (HM.UNIT[r][s[0]] or WM.shape[1])
            same &= torch.equal(p.partner[:n], torch.arange(n))
            dev = max(dev, float((p.gain[:n] - 1).abs().max()))
            n_site = n if s == site else n_site
        new = ps[site].partner[n_site:]
        out[r] = {"graft_units": len(new), "graft_paired": int((new >= 0).sum()), "others_identity": same, "others_max_gain_dev": dev,
                  "S_zero_by_arithmetic": same and not bool((new >= 0).any()) and dev <= th["gain_tol"]}  # fmt: skip
    return out


def reading(M: R.Model, F: R.Model, g: nn.Module, s: Spec, d: Any, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    fit, rd = HR.read(F, M, d.fit, d.read, cfg, th, s.role)
    key = f"{s.site[0]}{s.layer}"
    out = {"graft": rd, "units": units(fit, M, s.site, th), "sizes": G.site_sizes(M, g, s.layer, d.fit, cfg.sum_batch)}
    z = out["sizes"]
    out["graft_unfitted_share"] = rd["readings"]["3"]["residual"][key] * z["F_site"] / z["graft"]
    for r in ("1", "2"):
        u, S = out["units"][r], rd["readings"][r]["S"]
        u["S_as_expected"] = u["S_zero_by_arithmetic"] and S is not None and abs(S) <= th["s_tol"]
        log(f"{s.role} reading {r} units: {u}")
    log(f"{s.role} graft site {key}: |graft write|^2 {z['graft']:.6g}, |F site write|^2 {z['F_site']:.6g}; reading 3 S {rd['readings']['3']['S']}, "
        f"share of the graft's own write left unfitted {out['graft_unfitted_share']:.6g}")
    torch.save(fit.maps["3"][s.site].to(torch.bfloat16).cpu(), LARGE / f"map3_{s.role}.pt")
    return out


# ---- driving -------------------------------------------------------------------------------------------------------


def staged(path: Path, make: Any) -> dict[str, Any]:
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before, read")
        return json.loads(path.read_text())
    out = make() | {"experiment": EXPERIMENT, "done": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, out)
    log(f"{path.stem.upper()} DONE")
    return out


def one(M: R.Model, s: Spec, task: Any, d: Any, cfg: Config, th: dict[str, Any]) -> None:
    """One graft: build, checks, and the reading when the checks hold."""
    if staged(GRAFTS / f"{s.role}_build.json", lambda: build(M, s, task, d, cfg))["failed"]:
        return
    F, g = attach(M, s, cfg)
    G.load(g, LARGE / f"{s.role}.pt")
    if not staged(GRAFTS / f"{s.role}_checks.json", lambda: checks(M, F, g, s, task, d, cfg, th))["misses"]:
        staged(GRAFTS / f"{s.role}_read.json", lambda: reading(M, F, g, s, d, cfg, th))


def summary(pred: dict[str, Any], s0: dict[str, Any]) -> dict[str, Any]:
    """S per graft and reading (None: not built, missed a build check, or KL(F || M) at the floor) and the filed rules."""
    S: dict[str, dict[str, float | None]] = {}
    for role in pred["rules"]["roles"]:
        f = GRAFTS / f"{role}_read.json"
        S[role] = {r: json.loads(f.read_text())["graft"]["readings"][r]["S"] for r in HM.READINGS} if f.is_file() else dict.fromkeys(HM.READINGS)
    s3 = {k: v["3"] for k, v in S.items()}
    rules = pred["rules"]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "S": S, "k": {"lowest": s0["k_lowest"], **s0["k_by_rank"]}, "rules": {}}
    for role, v in s3.items():
        out["rules"][f"R1 {role}"] = "no reading" if v is None else "holds" if v < rules["R1"]["old_task_floor"] else "missed"
    for name, (hi, lo) in (("R2", rules["R2"]["pair"]), ("R3", rules["R3"]["pair"])):
        a, b = s3[hi], s3[lo]
        out["rules"][name] = "no reading" if a is None or b is None else "holds" if a > b else "missed"
    return out


def main() -> int:
    pred = json.loads(PRED.read_text())
    st: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items() if k != "recipe"}
    rec: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"]["recipe"].items()}
    cfg, th = Config(**st, recipe=G.Recipe(**rec)), pred["thresholds"]
    from commute.steering import Net

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    log(f"{EXPERIMENT} (P1.28) start; settings {asdict(cfg)}; thresholds {th}")
    net = Net("cuda")
    net.net.requires_grad_(False)
    M = R.Model(net.net)
    d = GN.data(cfg, "cuda")
    lk = lookup(d.train, cfg, int(net.net.lm_head.weight.shape[0]), "cuda")  # type: ignore[union-attr]
    s0 = staged(RESULTS / "stage0.json", lambda: stage0(M, d, lk, cfg, th, pred["code_checks"]))
    for s in specs(cfg, s0):
        one(M, s, Shift(s.k) if s.kind == "head" else lk, d, cfg, th)
        torch.cuda.empty_cache()
    out = summary(pred, s0)
    dump(RESULTS / "summary.json", out)
    log(f"S per graft and reading: {out['S']}")
    log(f"rules: {out['rules']}")
    log(f"{EXPERIMENT} DONE")
    return 0


def cli() -> None:
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()
