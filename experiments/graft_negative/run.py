"""E201 (cell P1.26, run 2): the grafted-head negative case, built and read with E197's held model (plan
reports/plans/2026-11-18_plan_held-model-retention.md, "The negative case", "Settled" answer 6, appendix recipe; card
card.md; settings, their sources and every expected value in results/predictions.json, which the run refuses to start
without, committed unchanged). Descriptive: no bars, no verdicts.

M (the 67M) is frozen. F is M with one fresh attention head appended to the last layer: it reads that layer's rms_1
output with M's rotary positions, and its 128-wide output joins M's six heads' outputs before the output matrix, which is
M's o_proj with the graft's 768 x 128 beside it (`Grafted`). So `commute.held_model` (unchanged) sees seven F heads at
that site, and F's site write is M's attention write plus the graft's.
    0  E197's do-nothing check on this run's fit and read windows (E197's own functions), stopping on a miss; a check that
       the attention patterns computed here give back each M head's output; on M, each of the 24 heads' mean attention at
       each offset of `offsets`, over the queries whose key at every offset is past position 1; k = the offset whose
       largest head share is lowest, s = that share.
    1  the build: the graft (seed graft_seed) trained alone, cross-entropy at the last position against the token k back,
       the 67M's own pretraining recipe (predictions.json "setting_sources"), `steps` steps of `batch` windows, one
       attempt. Saves results/large/graft.pt (1.5 MB).
    2  build checks, none using the held model: held-out accuracy of F and of M; the graft's attention at offset k from
       the last position; each M head's share at k inside F against stage 0's; F with the graft's output zeroed against M.
       Any miss stops the job before stage 3.
    3  E197's three readings of F (`read` of experiments/held_model_retention/run.py), plus at the graft's site: the
       graft's partner in readings 1 and 2, whether every other unit is paired with itself at gain 1, the write sizes, and
       the share of the graft's own write that reading 3's map leaves unfitted. Saves reading 3's map there (bf16).
Each stage writes results/stage<n>.json when it ends and is skipped on a rerun once that file says done. The log:

    python experiments/graft_negative/run.py 2>&1 | tee -a experiments/graft_negative/results/run.log
    (on a pod: E120_PRED_SHA=<sha of the predictions file> python ...)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from commute import held_model as HM
from commute import rule_retention as R
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.variable_read import cut

HR: Any = load_experiment("held_model_retention", "run")  # E197: read, misses, CHECKS
P = HR.P  # P1.24's run module: chunks, clone, log, dump, stop
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, LARGE = RESULTS / "predictions.json", RESULTS / "large"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop, chunks = P.log, P.dump, P.stop, P.chunks
D = R.D_HEAD


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    layer: int  # the graft's layer
    offsets: tuple[int, int]  # the offsets measured, inclusive
    window: int
    train_rows: int  # Pile train rows 0..train_rows: the first half for the graft's training, the second for the fit
    val_rows: int  # Pile val rows 0..val_rows: the first half for probe and accuracy windows, the second for read windows
    n_probe: int
    probe_seed: int
    n_acc: int
    acc_seed: int
    graft_seed: int
    init_std: float  # q, k, v
    out_std: float  # o
    lr: float
    lr_min_frac: float
    warmup: int
    steps: int
    batch: int
    beta1: float
    beta2: float
    weight_decay: float
    grad_clip: float
    train_seed: int
    acc_at: tuple[int, ...]  # steps at which held-out accuracy is logged
    n_fit: int
    fit_seed: int
    n_read: int
    read_seed: int
    sum_batch: int
    eval_batch: int

    def __post_init__(self) -> None:
        lo, hi = self.offsets
        assert 0 < lo < hi and hi + 2 < self.window and 0 < self.warmup < self.steps and self.acc_at[-1] == self.steps
        assert abs(self.out_std - self.init_std / math.sqrt(2 * 4)) < 1e-12  # the 67M's residual-scaled init, 4 layers

    @property
    def queries(self) -> slice:  # query positions whose key at the largest offset is past the two sink positions
        return slice(self.offsets[1] + 2, self.window)


# ---- the graft and the grafted attention ---------------------------------------------------------------------------


def pattern(attn: Any, h: Tensor, wq: Tensor, wk: Tensor, n: int) -> Tensor:
    """(B, n, T, T) attention of n heads with query and key weights wq, wk on normed input h: the 67M's manual path
    (model_config.yaml flash_attention false) with `attn`'s rotary tables."""
    B, T, _ = h.shape
    q, k = ((h @ w.T).view(B, T, n, D).transpose(1, 2) for w in (wq, wk))
    pos = torch.arange(T, device=h.device)[None]
    q, k = attn.apply_rotary_pos_emb(q, k, attn.rotary_cos[pos].to(q.dtype), attn.rotary_sin[pos].to(q.dtype))
    s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
    return s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=h.device).tril(), float("-inf")).softmax(-1)


class Graft(nn.Module):
    """One head: q, k, v of 128 x 768 and o of 768 x 128, the 67M's own init (normal, sd init_std; o at out_std)."""

    base: Any

    def __init__(self, base: Any, cfg: Config) -> None:
        super().__init__()
        gen, d = torch.Generator().manual_seed(cfg.graft_seed), base.o_proj.weight.shape[0]
        self.wq, self.wk, self.wv = (nn.Parameter(torch.randn(D, d, generator=gen) * cfg.init_std) for _ in range(3))
        self.wo = nn.Parameter(torch.randn(d, D, generator=gen) * cfg.out_std)
        object.__setattr__(self, "base", base)  # M's attention, for its rotary tables only; not a submodule

    def pattern(self, h: Tensor) -> Tensor:
        return pattern(self.base, h, self.wq, self.wk, 1)[:, 0]

    def z(self, h: Tensor) -> Tensor:
        return self.pattern(h) @ (h @ self.wv.T)


class Joined(nn.Module):
    """F's output matrix at the grafted layer: M's o_proj on the first 768 inputs, the graft's o on the last 128."""

    graft: Graft

    def __init__(self, m_o: nn.Linear, graft: Graft) -> None:
        super().__init__()
        self.m_o = m_o
        object.__setattr__(self, "graft", graft)

    @property
    def weight(self) -> Tensor:  # (768, 896), what held_model.out_weight reads
        return torch.cat([self.m_o.weight, self.graft.wo], 1)

    def forward(self, z: Tensor) -> Tensor:
        n = self.m_o.weight.shape[1]
        return self.m_o(z[..., :n]) + z[..., n:] @ self.graft.wo.T


class Grafted(nn.Module):
    """M's six heads (`base`, M's module, frozen) and the graft, joined before the output matrix."""

    def __init__(self, base: Any, graft: Graft) -> None:
        super().__init__()
        self.base, self.graft, self.o_proj = base, graft, Joined(base.o_proj, graft)

    def forward(self, h: Tensor) -> Tensor:
        seen: list[Tensor] = []
        hook = self.base.o_proj.register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
        try:
            self.base(h)
        finally:
            hook.remove()
        return self.o_proj(torch.cat([seen[0], self.graft.z(h)], -1))


def grafted(M: R.Model, cfg: Config) -> tuple[R.Model, Graft]:
    F = P.clone(M)
    base = F.net.h[cfg.layer].attn
    g = Graft(base, cfg).to(base.o_proj.weight.device)
    F.net.h[cfg.layer].attn = Grafted(base, g)
    return F, g


# ---- measurements --------------------------------------------------------------------------------------------------


def m_attn(block: Any) -> Any:
    return getattr(block.attn, "base", block.attn)


@torch.no_grad()
def shares(N: R.Model, toks: Tensor, cfg: Config) -> tuple[Tensor, float]:
    """(layers, 6, offsets) each M head's mean attention at each offset inside N, and the max relative error of the six
    heads' outputs rebuilt from these patterns against the module's own (the pattern code's check)."""
    lo, hi = cfg.offsets
    tot, err = torch.zeros(len(N.net.h), R.N_HEAD, hi - lo + 1, dtype=torch.float64), 0.0
    for t in chunks(toks, cfg.eval_batch):
        for l, (x, _) in enumerate(R.m_inputs(N, t)):
            a, b = m_attn(N.net.h[l]), N.net.h[l]
            h = b.rms_1(x)
            att = pattern(a, h, a.q_proj.weight, a.k_proj.weight, R.N_HEAD)
            v = (h @ a.v_proj.weight.T).view(*h.shape[:2], R.N_HEAD, D).transpose(1, 2)
            z, z0 = (att @ v).transpose(1, 2).flatten(2), HM.z(b, x)[..., : R.N_HEAD * D]
            err = max(err, float((z - z0).abs().max() / z0.abs().max()))
            q = torch.arange(cfg.window, device=t.device)[cfg.queries]
            for i, d in enumerate(range(lo, hi + 1)):
                tot[l, :, i] += att[:, :, q, q - d].double().sum((0, 2)).cpu()
    return tot / (toks.shape[0] * len(range(cfg.window)[cfg.queries])), err


@torch.no_grad()
def accuracy(N: R.Model, toks: Tensor, k: int, cfg: Config) -> float:
    hit = sum(int((N.logits(t)[:, -1].argmax(-1) == t[:, -1 - k]).sum()) for t in chunks(toks, cfg.eval_batch))
    return hit / toks.shape[0]


@torch.no_grad()
def graft_attention(F: R.Model, g: Graft, toks: Tensor, k: int, cfg: Config) -> dict[str, float]:
    """The graft's mean attention at offset k: from the last position, and over `cfg.queries`."""
    last, mid, q = 0.0, 0.0, torch.arange(cfg.window, device=toks.device)[cfg.queries]
    for t in chunks(toks, cfg.eval_batch):
        x = R.m_inputs(F, t)[cfg.layer][0]
        att = g.pattern(F.net.h[cfg.layer].rms_1(x))
        last += float(att[:, -1, -1 - k].sum())
        mid += float(att[:, q, q - k].mean(1).sum())
    return {"last": last / toks.shape[0], "queries": mid / toks.shape[0]}


@torch.no_grad()
def site_sizes(M: R.Model, g: Graft, toks: Tensor, cfg: Config) -> dict[str, Any]:
    """Over the fit positions, on M's input to the grafted layer: |graft write|^2, each M head's |write|^2, <graft, head>,
    and |F's attention write|^2 (the denominator of held_model.residual there)."""
    n_g, n_F, n_h, cross = 0.0, 0.0, torch.zeros(R.N_HEAD, dtype=torch.float64), torch.zeros(R.N_HEAD, dtype=torch.float64)
    for t in chunks(toks, cfg.sum_batch):
        x = R.m_inputs(M, t)[cfg.layer][0]
        b = M.net.h[cfg.layer]
        fh = R.heads(b, x).double()  # (B, T, 6, 768)
        fg = (g.z(b.rms_1(x)) @ g.wo.T).double()
        n_g += float(fg.pow(2).sum())
        n_F += float((fh.sum(2) + fg).pow(2).sum())
        n_h += fh.pow(2).sum((0, 1, 3)).cpu()
        cross += torch.einsum("btd,bthd->h", fg, fh).cpu()
    return {"graft": n_g, "F_site": n_F, "M_heads": n_h.tolist(), "graft_dot_M_heads": cross.tolist()}


# ---- stages --------------------------------------------------------------------------------------------------------


@dataclass
class Data:
    train: Tensor  # graft training rows
    fit: Tensor  # fit windows (reading)
    probe: Tensor
    acc: Tensor
    read: Tensor


def data(cfg: Config, dev: str) -> Data:
    tr, va = pile_rows("train", 0, cfg.train_rows, log=log), pile_rows("val", 0, cfg.val_rows, log=log)
    ht, hv = cfg.train_rows // 2, cfg.val_rows // 2
    g = lambda s: torch.Generator().manual_seed(s)
    return Data(tr[:ht], cut(tr[ht:], cfg.n_fit, cfg.window, g(cfg.fit_seed)).to(dev), cut(va[:hv], cfg.n_probe, cfg.window, g(cfg.probe_seed)).to(dev),
                cut(va[:hv], cfg.n_acc, cfg.window, g(cfg.acc_seed)).to(dev), cut(va[hv:], cfg.n_read, cfg.window, g(cfg.read_seed)).to(dev))  # fmt: skip


def stage0(M: R.Model, d: Data, cfg: Config, th: dict[str, Any], want: dict[str, Any]) -> dict[str, Any]:
    F, planted = HR.CHECKS["do_nothing"](M, cfg)
    fit, rd = HR.read(F, M, d.fit, d.read, cfg, th, "do_nothing")
    rd["misses"] = HR.misses(rd, fit, planted, want, th)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "do_nothing": rd}
    sh, err = shares(M, d.probe, cfg)
    top = sh.flatten(0, 1).max(0).values
    i = int(top.argmin())
    out |= {"shares": sh.tolist(), "max_share_by_offset": dict(zip(range(cfg.offsets[0], cfg.offsets[1] + 1), top.tolist(), strict=True)),
            "k": cfg.offsets[0] + i, "s": float(top[i]), "pattern_error": err}  # fmt: skip
    log(f"do_nothing: {'checks hold' if not rd['misses'] else 'MISSES ' + '; '.join(rd['misses'])}")
    log(f"pattern check: max relative error of the heads' outputs rebuilt from the patterns {err:.3g} (tol {th['pattern_tol']})")
    log(f"largest M head share by offset: {out['max_share_by_offset']}; k = {out['k']}, s = {out['s']:.6g}")
    if rd["misses"] or not err <= th["pattern_tol"]:
        stop(RESULTS / "stage0.json", out, "stage 0: " + "; ".join(rd["misses"] + ([f"pattern error {err:.3g}"] if err > th["pattern_tol"] else [])))
    return out | {"done": True}


def rate(t: int, cfg: Config) -> float:
    """The 67M's pretraining schedule (param-decomp pretrain/train.py get_lr), t from 0, over cfg.steps."""
    lo = cfg.lr * cfg.lr_min_frac
    if t < cfg.warmup:
        return cfg.lr * (t + 1) / cfg.warmup
    return lo + 0.5 * (1.0 + math.cos(math.pi * (t - cfg.warmup) / (cfg.steps - cfg.warmup))) * (cfg.lr - lo)


def stage1(M: R.Model, d: Data, cfg: Config, k: int) -> dict[str, Any]:
    F, g = grafted(M, cfg)
    opt = torch.optim.AdamW(g.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    gen, curve, acc = torch.Generator().manual_seed(cfg.train_seed), list[float](), dict[int, float]()
    for t in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = rate(t, cfg)
        toks = cut(d.train, cfg.batch, cfg.window, gen).to(d.fit.device)
        loss = nn.functional.cross_entropy(F.logits(toks)[:, -1], toks[:, -1 - k])
        curve.append(float(loss.detach()))
        if not math.isfinite(curve[-1]):
            stop(RESULTS / "stage1.json", {"experiment": EXPERIMENT, "curve": curve}, f"stage 1: loss {curve[-1]} at step {t + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()), cfg.grad_clip)
        opt.step()
        if t + 1 in cfg.acc_at or t == 0 or (t + 1) % 250 == 0:
            line = f"step {t + 1}: loss {curve[-1]:.4f} (mean of last 250 {sum(curve[-250:]) / len(curve[-250:]):.4f}), rate {rate(t, cfg):.3g}"
            if t + 1 in cfg.acc_at:
                acc[t + 1] = accuracy(F, d.acc, k, cfg)
                line += f", held-out accuracy {acc[t + 1]:.4f}"
            log(line)
    g.requires_grad_(False)
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in g.named_parameters()}, LARGE / "graft.pt")
    sha = hashlib.sha256((LARGE / "graft.pt").read_bytes()).hexdigest()
    log(f"graft saved: results/large/graft.pt sha256 {sha}")
    return {"experiment": EXPERIMENT, "k": k, "curve": curve, "accuracy_at": acc, "graft_sha256": sha, "done": True}


def load_graft(M: R.Model, cfg: Config) -> tuple[R.Model, Graft]:
    F, g = grafted(M, cfg)
    state = torch.load(LARGE / "graft.pt", weights_only=True)
    with torch.no_grad():
        for n, p in g.named_parameters():
            p.copy_(state[n])
    return F, g.requires_grad_(False)


@torch.no_grad()
def stage2(M: R.Model, F: R.Model, g: Graft, d: Data, cfg: Config, th: dict[str, Any], s0: dict[str, Any]) -> dict[str, Any]:
    k, s = s0["k"], s0["s"]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "k": k, "acc_F": accuracy(F, d.acc, k, cfg), "acc_M": accuracy(M, d.acc, k, cfg)}
    out["graft_attention"] = graft_attention(F, g, d.probe, k, cfg)
    sh = shares(F, d.probe, cfg)[0][..., k - cfg.offsets[0]]
    sh0 = torch.tensor(s0["shares"], dtype=torch.float64)[..., k - cfg.offsets[0]]
    out["M_share_at_k_in_F"], out["M_share_off"] = float(sh.max()), float((sh - sh0).abs().max())
    wo = g.wo.clone()
    g.wo.zero_()
    out["no_graft_logit_error"] = max(HM.logit_error(F.logits(t), M.logits(t)) for t in chunks(d.probe, cfg.eval_batch))
    g.wo.copy_(wo)
    misses = [] if out["acc_F"] >= th["acc_min"] else [f"accuracy {out['acc_F']:.4f}"]
    misses += [] if out["graft_attention"]["last"] >= th["attn_min"] else [f"graft attention at k {out['graft_attention']['last']:.4f}"]
    misses += [] if out["M_share_off"] <= th["share_tol"] else [f"M head share at k off stage 0's by {out['M_share_off']:.3g}"]
    misses += [] if out["no_graft_logit_error"] <= th["logit_tol"] else [f"no-graft logit error {out['no_graft_logit_error']:.3g}"]
    log(f"build checks: held-out accuracy F {out['acc_F']:.4f}, M {out['acc_M']:.4f}; graft attention at k from the last position "
        f"{out['graft_attention']['last']:.4f} (over the probe queries {out['graft_attention']['queries']:.4f}); largest M head share at k "
        f"in F {out['M_share_at_k_in_F']:.6g} (s {s:.6g}, off by {out['M_share_off']:.3g}); graft zeroed vs M logit error {out['no_graft_logit_error']:.3g}")
    if misses:
        stop(RESULTS / "stage2.json", out | {"misses": misses}, "stage 2, the graft did not build: " + "; ".join(misses))
    return out | {"misses": [], "done": True}


def unit_report(fit: HM.Fit, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """Readings 1 and 2: the graft's partner, gain and error; whether every other F unit is paired with itself at gain 1."""
    out: dict[str, Any] = {}
    for r, ps in fit.pairs.items():
        dev, same = 0.0, True
        for site, p in ps.items():
            n = R.N_HEAD if site == ("attn", cfg.layer) else len(p.partner)
            same &= torch.equal(p.partner[:n], torch.arange(n))
            dev = max(dev, float((p.gain[:n] - 1).abs().max()))
        p = ps[("attn", cfg.layer)]
        out[r] = {"graft_partner": int(p.partner[-1]), "graft_gain": float(p.gain[-1]), "graft_err": float(p.err[-1]),
                  "others_identity": same, "others_max_gain_dev": dev,
                  "S_zero_by_arithmetic": same and int(p.partner[-1]) == -1 and dev <= th["gain_tol"]}  # fmt: skip
    return out


def stage3(M: R.Model, F: R.Model, g: Graft, d: Data, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    fit, rd = HR.read(F, M, d.fit, d.read, cfg, th, "graft")
    site = f"attn{cfg.layer}"
    out: dict[str, Any] = {"experiment": EXPERIMENT, "graft": rd, "units": unit_report(fit, cfg, th), "sizes": site_sizes(M, g, d.fit, cfg)}
    z = out["sizes"]
    out["graft_unfitted_share"] = rd["readings"]["3"]["residual"][site] * z["F_site"] / z["graft"]
    out["graft_within_weakest_head"] = z["graft"] <= min(z["M_heads"])
    for r in ("1", "2"):
        u, S = out["units"][r], rd["readings"][r]["S"]
        u["S_as_expected"] = u["S_zero_by_arithmetic"] and S is not None and abs(S) <= th["s_tol"]
        log(f"graft reading {r}: graft partner {u['graft_partner']} (-1 unpaired), gain {u['graft_gain']:.4g}; other units identity "
            f"{u['others_identity']}, max |gain - 1| {u['others_max_gain_dev']:.3g}; S {S}, as arithmetic expects (0) {u['S_as_expected']}")
    log(f"graft site sizes over the fit positions: |graft write|^2 {z['graft']:.6g}, M heads {[f'{v:.4g}' for v in z['M_heads']]}, "
        f"<graft, head> {[f'{v:.4g}' for v in z['graft_dot_M_heads']]}; graft within the weakest head {out['graft_within_weakest_head']}")
    log(f"graft reading 3: S {rd['readings']['3']['S']}; site residual {rd['readings']['3']['residual'][site]:.6g}, "
        f"share of the graft's own write left unfitted {out['graft_unfitted_share']:.6g}")
    torch.save(fit.maps["3"][("attn", cfg.layer)].to(torch.bfloat16).cpu(), LARGE / f"map3_{site}.pt")
    return out | {"done": True}


def staged(n: int, make: Any) -> dict[str, Any]:
    path = RESULTS / f"stage{n}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"stage {n}: done before, read from {path.name}")
        return json.loads(path.read_text())
    out = make()
    dump(path, out)
    log(f"STAGE {n} DONE")
    return out


def main() -> int:
    pred = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg = Config(**settings)
    th = pred["thresholds"]
    from commute.steering import Net

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    log(f"{EXPERIMENT} (P1.26, run 2) start; settings {asdict(cfg)}; thresholds {th}")
    net = Net("cuda")
    net.net.requires_grad_(False)
    M = R.Model(net.net)
    d = data(cfg, "cuda")
    s0 = staged(0, lambda: stage0(M, d, cfg, th, pred["code_checks"]["do_nothing"]))
    staged(1, lambda: stage1(M, d, cfg, s0["k"]))
    F, g = load_graft(M, cfg)
    staged(2, lambda: stage2(M, F, g, d, cfg, th, s0))
    staged(3, lambda: stage3(M, F, g, d, cfg, th))
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
