"""E90's model-side pieces, shared by stage1.py and stage2.py: the sites, the reads under a removal, the capabilities on natural
Pile inputs, the possession readout, the hand write, the saved records. GPU only; settings from results/predictions.json."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import keydep as KD
from commute import perturb as PT
from commute import readout as RO
from commute import removal as RM
from commute import steering as ST
from commute.jobs import hard_exit
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, EXP, REPO = HERE / "results", HERE.parent, HERE.parents[1]
LARGE, PRED, STOP = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stop.json"
AMEND = RESULTS / "predictions_amend1.json"
log = KD.log
Reads = dict[Any, tuple[Tensor, Tensor]]  # "clean", or (block, draw; -1 the basis), to (log P, log-odds) per window


def dump(path: Path, res: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(res), indent=1, sort_keys=True))


def stop(path: Path, res: dict[str, Any], why: str) -> NoReturn:
    res["stopped"] = why
    dump(path, res)
    dump(STOP, {"why": why})
    log(f"STOP: {why}; see {STOP.name} (exit 1)")
    hard_exit(1)


def filed() -> dict[str, Any]:
    """The predictions with Amendment 1's under "amend1", each refused unless committed unchanged; a stop file also refuses, until
    the lead clears it."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False).stdout
    for path in (PRED, AMEND):
        rel = str(path.relative_to(REPO))
        if not git("ls-files", rel).strip() or git("diff", "HEAD", "--", rel):
            log(f"{rel} is not committed unchanged: commit it first (exit 1)")
            hard_exit(1)
    if STOP.exists():
        log(f"a stop fired ({STOP.read_text().strip()}); nothing runs until the lead clears it (exit 1)")
        hard_exit(1)
    return {**json.loads(PRED.read_text()), "amend1": json.loads(AMEND.read_text())}


# ---- saved records ----


def records() -> dict[Path, str]:
    """Every state and windows file with a saved sha256, by path."""
    load = lambda p: json.loads((EXP / p).read_text())
    out: dict[Path, str] = {}
    table, pair, ctl, cue = load("regating_table/results/stage0a.json")["groups"], load("open_pair/results/run.json"), load("open_pair/results/amend1.json"), load("induction_cue/results/run.json")
    large = lambda e: EXP / e / "results" / "large"
    for n in ("ident32", "rand32"):
        out[large("regating_table") / f"windows_{n}.pt"] = table["e12/build"]["sha256"][n]
        out |= {large("regating_table") / f"{n}_s{s}.pt": table[f"e12/{n}"]["states"][f"s{s}"]["sha256"] for s in (0, 1)}
    for n in ("cipher", "elicit"):
        out[large("open_pair") / f"windows_{n}_trained.pt"] = pair["windows_sha256"][n]["trained"]
        out |= {large("open_pair") / f"{n}_s{s}.pt": pair["states"][f"{n}_s{s}"]["ft"]["sha256"] for s in range(3)}
    for n in ("rank1", "copy"):
        out[large("open_pair") / "amend1" / f"windows_{n}_trained.pt"] = ctl["c0"][n]["sha256"]["trained"]
        out |= {large("open_pair") / "amend1" / f"{n}_s{s}.pt": ctl["control"][f"{n}_s{s}"]["ft"]["sha256"] for s in range(3)}
    out[large("induction_cue") / "windows.pt"] = cue["windows_sha256"]
    out |= {large("induction_cue") / f"{k}.pt": v["sha256"] for k, v in cue["states"].items()}
    return out


def checked(path: Path, rec: dict[Path, str], seen: dict[str, str]) -> Path:
    """Recorded in `seen`; a file with a record that differs raises. A file with no record is only recorded."""
    sha = KD.sha256(path)
    seen[str(path.relative_to(EXP))] = sha
    if path in rec and rec[path] != sha:
        raise ValueError(f"{path}: sha256 {sha} differs from the saved record {rec[path]}")
    return path


# ---- sites and reads ----


@dataclass(frozen=True)
class Site:
    """Windows, each window's target at index `half`. With `alt` (each window's other form) the second read is the pairwise
    log P(tgt) - log P(alt), E53's score, in place of the log-odds."""

    toks: Tensor
    tgt: Tensor
    half: int
    alt: Tensor | None = None


def kv_site(path: Path) -> tuple[Site, Any]:
    """A kv windows file's held-out windows, and the file's dict (train windows, task)."""
    w = torch.load(path, weights_only=False)
    held = w["heldout"]
    return Site(held.tokens, w["table"][held.tokens[:, held.half]], held.half), w


def reads(net: ST.Net, s: Site, dW: ST.DW | None = None, edit: ST.Edit | None = None, batch: int = 256) -> tuple[Tensor, Tensor]:
    """(log P, log-odds) of the target at `half`, per window; with `s.alt`, (log P, log P(target) - log P(alt))."""
    lp, lo = [], []
    with net.at(dW):
        for i in range(0, len(s.toks), batch):
            logits, t = net.run(s.toks[i : i + batch], edit)[0], s.tgt[i : i + batch]
            lp.append(ST.logp_at(logits, s.half, t))
            lo.append(ST.logodds_at(logits, s.half, t) if s.alt is None else lp[-1] - ST.logp_at(logits, s.half, s.alt[i : i + batch]))
    return torch.cat(lp), torch.cat(lo)


def removal_edit(block: int, half: int, u: Tensor) -> ST.Edit:
    """Remove the basis u from block `block`'s output residual at index `half` only."""

    def f(x: Tensor) -> Tensor:
        x = x.clone()
        x[:, half] = RM.remove(x[:, half], u.to(x.device, x.dtype))
        return x

    return {RO.RESID[block]: f}


def read_all(net: ST.Net, s: Site, dW: ST.DW | None, bases: dict[int, Tensor], randoms: dict[int, list[Tensor]]) -> Reads:
    """A state on one site: clean, under each block's basis (draw -1) and each random basis."""
    out: Reads = {"clean": reads(net, s, dW)}
    for b, u in bases.items():
        out[b, -1] = reads(net, s, dW, removal_edit(b, s.half, u))
        for i, r in enumerate(randoms[b]):
            out[b, i] = reads(net, s, dW, removal_edit(b, s.half, r))
    return out


def state_row(w0: Reads, st: Reads, blocks: Iterable[int], n_random: int, floor: float) -> dict[str, Any]:
    """Per block: the share under the basis, the random draws' A_lo and this state's band. W0's reads are on the same windows."""
    out: dict[str, Any] = {"G_logodds": float((st["clean"][1] - w0["clean"][1]).mean())}
    for b in blocks:
        draws = [RO.shares(w0["clean"], w0[b, i], st["clean"], st[b, i], floor)["A_lo"] for i in range(n_random)]
        out[str(b)] = {**RO.shares(w0["clean"], w0[b, -1], st["clean"], st[b, -1], floor), "random": draws, "band": RM.band_of(draws)}
    return out


def band_over(rows: Iterable[dict[str, Any]], b: int) -> float:
    return RM.max_band([r[str(b)]["band"] for r in rows])


def say_row(tag: str, r: dict[str, Any], blocks: Iterable[int]) -> None:
    for b in blocks:
        log(f"[{tag} block {b}] A_lo {r[str(b)]['A_lo']:.4f} band {r[str(b)]['band']:.4f} G_logodds {r['G_logodds']:.4f} lo_drop {r[str(b)]['lo_drop']:.4f} lo_drop_W0 {r[str(b)]['lo_drop_W0']:.4f}")


# ---- capabilities on natural inputs ----


@dataclass
class Capability:
    """`fit_h` (blocks, M, d) W0's residual at index `half` on the fit rows with `y` the value index; `read` on the held rows."""

    name: str
    fit_h: Tensor
    y: Tensor
    read: Site
    rank: int

    def bases(self, blocks: Iterable[int]) -> dict[int, Tensor]:
        """The value form: per-value mean residuals, centred, orthonormal."""
        seen = self.y.unique()
        return {b: RM.centred_basis(torch.stack([self.fit_h[b][self.y == v].mean(0) for v in seen], 1)) for b in blocks}


def random_bases(bases: dict[int, Tensor], seed: int, tag: int, n: int) -> dict[int, list[Tensor]]:
    return {b: [RM.random_basis(u.shape[0], u.shape[1], [seed, tag, b, i]) for i in range(n)] for b, u in bases.items()}


def capped_held(s: Site, n: int, seed: int) -> Site:
    i = torch.randperm(len(s.toks), generator=torch.Generator().manual_seed(seed))[:n]
    return Site(s.toks[i], s.tgt[i], s.half, None if s.alt is None else s.alt[i])


def identity_capability(net: ST.Net, name: str, rows: Tensor, keys: Tensor, half: int, S: dict[str, Any], n_read: int) -> Capability:
    """Natural occurrences of `keys` at index `half`; `next_token_windows` at half - 1 reads the token at `half`."""
    nf = S["pile"]["fit_row_count"]
    toks, y, counts = RO.next_token_windows(rows[:nf], keys, half - 1, S["pile"]["cap"], S["pile"]["positions_seed"])
    held, hy, _ = RO.next_token_windows(rows[nf:], keys, half - 1, S["pile"]["held_cap"], S["pile"]["held_seed"])
    log(f"[{name}] fit windows {len(toks)}, held {len(held)}; per key {counts}")
    h = RO.cache64(net, toks, keys, half)[0]
    read = capped_held(Site(held, keys[hy], half), n_read, S["control_seed"])
    return Capability(name, h, y.to(h.device), read, len(keys) - 1)


def own_top1(net: ST.Net, s: Site, batch: int = 256) -> Site:
    """Amendment 1's control read for token identity: `s`'s windows, each with W0's own top-1 next token as the target."""
    top = [net.run(s.toks[i : i + batch])[0][:, s.half].argmax(-1).cpu() for i in range(0, len(s.toks), batch)]
    return Site(s.toks, torch.cat(top), s.half)


def leaning_capability(net: ST.Net, name: str, rows: Tensor, trigger: int, target: int, half: int, S: dict[str, Any], n_read: int) -> Capability:
    """R7: natural occurrences of the trigger; y is the scalar, W0's log-odds of the target there. Its basis is
    `removal.pattern_basis`, not the value form."""
    nf, keys, tgt = S["pile"]["fit_row_count"], torch.tensor([trigger]), torch.tensor([target])
    toks = RO.next_token_windows(rows[:nf], keys, half - 1, S["pile"]["cap"], S["pile"]["positions_seed"])[0]
    held = RO.next_token_windows(rows[nf:], keys, half - 1, S["pile"]["cap"], S["pile"]["held_seed"])[0]
    log(f"[{name}] fit windows {len(toks)}, held {len(held)}")
    h = RO.cache64(net, toks, tgt, half)[0]
    lean = reads(net, Site(toks, tgt.expand(len(toks)), half))[1].to(h.device)
    return Capability(name, h, lean, capped_held(Site(held, tgt.expand(len(held)), half), n_read, S["control_seed"]), 1)


def scan_guess(net: ST.Net, rows: Tensor, vals: Tensor, half: int, cap: int, n: int, seed: int) -> tuple[Tensor, dict[int, int]]:
    """Candidates: W0's argmax read at indices half..WIN-1 of n random spans, at most 2 x cap per value, each re-cut with the
    position at `half`. Returns the (M, WIN) windows and the raw count per value."""
    gen, batch = torch.Generator().manual_seed(seed), 256
    L, rs, ps, seen = rows.shape[1], [], [], torch.zeros(len(vals), dtype=torch.long)
    lookup = torch.full((net.net.lm_head.weight.shape[0],), -1, device=net.device)
    lookup[vals.to(net.device)] = torch.arange(len(vals), device=net.device)
    r = torch.randint(0, rows.shape[0], (n,), generator=gen)
    s = torch.randint(0, L - RO.WIN - half + 1, (n,), generator=gen)
    for i in range(0, n, batch):
        toks = rows[r[i : i + batch][:, None], s[i : i + batch][:, None] + torch.arange(RO.WIN)]
        v = lookup[net.run(toks)[0][:, half:].argmax(-1)].cpu()
        for row, col in (v >= 0).nonzero().tolist():
            k = int(v[row, col])
            seen[k] += 1
            if seen[k] <= 2 * cap:
                rs.append(int(r[i + row]))
                ps.append(int(s[i + row]) + col)  # the re-cut window starts at the span's start + col
    start = torch.tensor(ps)
    return rows[torch.tensor(rs)[:, None], start[:, None] + torch.arange(RO.WIN)], dict(zip(vals.tolist(), seen.tolist(), strict=True))


def keep_guess(net: ST.Net, toks: Tensor, vals: Tensor, half: int, cap: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The re-cut windows whose argmax at `half` is still one of the values, at most `cap` per value: (windows, residuals
    (blocks, M, d), W0's logits over the values (M, m), value index (M,))."""
    keep, hs, lg, ys = [], [], [], []
    v = vals.to(net.device)
    for i in range(0, len(toks), 256):
        logits, h = RO.resid(net, toks[i : i + 256])
        a = logits[:, half].argmax(-1)
        m = (a[:, None] == v).any(1)
        keep.append(m.cpu())
        hs.append(h[:, :, half][:, m].clone())
        lg.append(logits[:, half][m][:, v])
        ys.append((a[m][:, None] == v).long().argmax(1))
    y = torch.cat(ys)
    i = RO.capped_idx(y, cap)
    return toks[torch.cat(keep).nonzero()[:, 0]][i.cpu()], torch.cat(hs, 1)[:, i], torch.cat(lg)[i], y[i]


def fit_values(y: Tensor, m: int, unfit_below: int) -> Tensor:
    """The value indices with at least `unfit_below` fit windows."""
    return (torch.bincount(y, minlength=m) >= unfit_below).nonzero()[:, 0]


def value_capability(net: ST.Net, name: str, vals: Tensor, cand: tuple[Tensor, Tensor], half: int, caps: tuple[int, int], unfit_below: int) -> tuple[Capability, Tensor, Tensor, Tensor, Tensor, list[int]]:
    """W0's guess on the fit and held candidates (`cand`), the unfit values out. Returns the capability (read: every held
    window), the fit logits, the held residuals, the held logits and the held value index, all over the fit values, and the
    windows per value before the unfit values leave."""
    _, h, w0, y = keep_guess(net, cand[0], vals, half, caps[0])
    ht, hh, hw, hy = keep_guess(net, cand[1], vals, half, caps[1])
    fit = fit_values(y, len(vals), unfit_below)
    new = torch.full((len(vals),), -1, device=y.device)
    new[fit] = torch.arange(len(fit), device=y.device)
    f, hf = new[y] >= 0, new[hy] >= 0
    read = Site(ht[hf.cpu()], vals[hy[hf].cpu()], half)
    c = Capability(name, h[:, f], new[y[f]], read, len(fit) - 1)
    return c, w0[f][:, fit], hh[:, hf], hw[hf][:, fit], new[hy[hf]], torch.bincount(y, minlength=len(vals)).tolist()


# ---- the possession readout ----


@torch.no_grad()
def accuracy(probes: RO.Probes, h: Tensor, y: Tensor) -> list[float]:
    return (probes(h).argmax(-1) == y).double().mean(1).tolist()


def train_pair(h: Tensor, y: Tensor, w0: Tensor, R: dict[str, Any], say: Callable[[str], None]) -> tuple[RO.Probes, RO.Probes]:
    """The readout under E64's early checks (raises ReadoutStop) and its shuffled-label twin, with no bars."""
    checks = RO.early_checks64(R["early_checks"], y)
    say(f"[readout] entropy {RO.entropy_of(y):.4f}, early bars {checks['bars']}")
    probes = RO.train_readouts64(h, y, w0, R["readout_seed"], checks, say)
    yy = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(R["shuffle_seed"])).to(y.device)]
    twin = RO.train_readouts64(h, yy, w0, R["readout_seed"], {**checks, "bars": {}}, lambda m: say(f"[twin] {m}"))
    return probes, twin


# ---- the hand write ----


def zero_except(net: ST.Net, key: str, x: Tensor) -> ST.DW:
    assert key in net.W0, key
    return {k: x.to(w.device, w.dtype) if k == key else torch.zeros_like(w) for k, w in net.W0.items()}


def table_inputs(net: ST.Net, train: Tensor, half: int, keys: list[int], values: list[int], key_at: int | None = None) -> tuple[Tensor, Tensor]:
    """A hand write's (A, U) for a key table: the keys' mean inputs (grouped at `key_at`, default `half`), the answers' rows."""
    lm = net.net.lm_head.weight.detach().float()
    return PT.key_inputs(net.model, train, half, keys, key_at), lm[torch.tensor(values, device=lm.device)].T


def hand_write(net: ST.Net, A: Tensor, U: Tensor, s: Site, G: float, rel_tol: float) -> tuple[ST.DW | None, dict[str, Any], Tensor]:
    """X = U pinv(A) into `perturb.WRITE_KEY`, alpha matched to the gain G on `s` (`reads`' second number), as regating_table's
    `write_row`. Returns the state (None unless the match is "ok"), its record and X."""
    lm = net.net.lm_head.weight.detach().float()
    X = PT.key_write(A, U)
    u = lm[s.tgt.to(lm.device)] if s.alt is None else lm[s.tgt.to(lm.device)] - lm[s.alt.to(lm.device)]
    fit = PT.write_fit(X, A, PT.window_inputs(net.model, s.toks, s.half), u)
    if G <= 0:
        return None, {"status": "no_target", "fit": fit, "G_target": G}, X
    lo0 = reads(net, s)[1]
    state = lambda a: zero_except(net, PT.WRITE_KEY, a * X)
    m = PT.match(lambda a: float((reads(net, s, state(a))[1] - lo0).mean()), G, PT.ALPHA_MAX, rel_tol)
    status = {"ok": "ok", "cap": "unreachable", "no_match": "no_match"}[m["status"]]
    rec = {"status": status, "alpha": m["x"], "G_write": m["f"], "G_target": G, "fit": fit, "norm": float(m["x"] * X.norm()), "cond_A": fit["cond_A"], "rel_err_mean": fit["rel_err_mean"]}
    return state(m["x"]) if status == "ok" else None, rec, X


def variance_share(h: Tensor, u: Tensor) -> float:
    """The share of the residual's variance in the basis (rows h (n, d), orthonormal u (d, r))."""
    x = h - h.mean(0)
    return float((x @ u.to(x.device, x.dtype)).pow(2).sum() / x.pow(2).sum())


