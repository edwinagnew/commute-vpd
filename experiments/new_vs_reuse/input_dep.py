"""E90 stage 1 (iv): the input-dependent ladder rung, lowercase and emoticon. Plan: reports/plans/2026-11-05_plan_e90-input-dependent.md,
whose settings are the constants below and whose "Lead's edit" overrides the rest. Rules: results/predictions.json I0-I3, rules.py.
Each task reads like `stage1.r7`: its control alone, then the state, then its spared control; a table that does not keep the
old target word trains and reads nothing. Writes, per task, results/large/{readout,windows,dW}_<task>*.pt and X_<task>.pt.

Departures from the plan's text, each forced by what is on disk or in the code:
- The fine-tune's target is `data.LAST` (a last column per window, `copying.with_target`), not a (V,) table: `ArmAConfig(o_token=table)`
  reads the table at the token at index 20, which is the trigger on every window here, so it would train one constant target.
- The emoticon's trigger is the commonest token at index 20 of the old windows, ' :' (243 of 365 train, 52 of 73 held; ten
  tokens in all), not "':', asserted equal on all windows": the assertion is false on disk and ':' itself is 8 of 365.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import apparatus as AP
import numpy as np
import reading as RD
import rules
import torch
from apparatus import EXP, LARGE, Site, dump, log
from reading import Basis, Job
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute import lowercase as LC
from commute import performed as PF
from commute import perturb as PT
from commute import readout as RO
from commute import steering as ST
from commute import vpd
from commute.data import LAST, Windows, pile_rows
from commute.finetune import ArmAConfig, finetune_arm_a
from commute.row_copy import lowercase_rows

HALF, PREV, SEED = 20, 19, 0
N_KEYS, MIN_WINDOWS, KEEP_SHARE, SCAN, N_TRAIN, N_HELD = 16, 40, 0.5, 400, 80, 20  # SCAN: windows read per key, the writer's choice
N_COMMON, UNFIT_BELOW, CAPS, FT_STEPS = 7, 50, (4000, 500), 10
SITE_CAP, XC_CAPS, XC_ROWS, XC_FIT_ROWS, PUSH_ROWS = 500, (2000, 250), 256, 224, 512
EMOTICON_BY_S = 20 * 60
EMOTICON = EXP / "transport_test/results/large/windows.pt"
O_TARGET = 80
START = time.monotonic()


@dataclass
class Table:
    """A trigger table, keys in frequency order; `train` and `held` are W0-kept 41-token windows, each with its key's answer."""

    keys: list[int]
    answers: list[int]
    train: Tensor
    train_answer: Tensor
    held: Tensor
    held_answer: Tensor
    per_key: list[dict[str, Any]]


def record(j: Job, task: str, **rec: Any) -> None:
    j.res.setdefault("input_dependent", {}).setdefault(task, {}).update(rec)
    log(f"[{task}] {json.dumps(rec)}")
    dump(j.out, j.res)


def w0_top1(net: Any, toks: Tensor, batch: int = 256) -> Tensor:
    return torch.cat([net.run(toks[i : i + batch])[0][:, HALF].argmax(-1).cpu() for i in range(0, len(toks), batch)])


def common_values(top1: Tensor, target: int) -> Tensor:
    """V: the old target and W0's commonest top-1s at the natural trigger, ascending."""
    v, n = top1.unique(return_counts=True)
    return torch.tensor(sorted({target, *v[n.argsort(descending=True, stable=True)[:N_COMMON]].tolist()}))


def trigger_table(net: Any, rows: Tensor, t: int, values: list[int], seed: int) -> Table:
    """The commonest previous tokens with enough windows, each answered by W0's modal top-1 on its train windows; a key leaves when
    under KEEP_SHARE of its windows hold that answer, when the answer is not in `values`, or when a half is left empty."""
    toks = RO.next_token_windows(rows, torch.tensor([t]), PREV, 10**9, seed)[0]
    prev = toks[:, PREV]
    tk, n = prev.unique(return_counts=True)
    order = n.argsort(descending=True, stable=True)
    keys = [int(tk[i]) for i in order if n[i] >= MIN_WINDOWS][:N_KEYS]
    none = torch.empty(0, dtype=torch.long)
    out = Table([], [], none, none, none, none, [])
    for k in keys:
        pool = toks[prev == k][:SCAN]
        a, cut = w0_top1(net, pool), len(pool) * N_TRAIN // (N_TRAIN + N_HELD)
        answer = rules.mode(a[:cut].numpy())
        ok = a == answer
        train, held = pool[:cut][ok[:cut]][:N_TRAIN], pool[cut:][ok[cut:]][:N_HELD]
        share = float(ok.double().mean())
        kept = share >= KEEP_SHARE and answer in values and len(train) > 0 and len(held) > 0
        out.per_key.append({"key": k, "n": int((prev == k).sum()), "answer": answer, "kept_share": share, "n_train": len(train), "n_held": len(held), "kept": kept})
        if kept:
            out.keys.append(k)
            out.answers.append(answer)
            out.train, out.held = torch.cat([out.train, train]), torch.cat([out.held, held])
            out.train_answer = torch.cat([out.train_answer, torch.full((len(train),), answer)])
            out.held_answer = torch.cat([out.held_answer, torch.full((len(held),), answer)])
    return out


def no_version(j: Job, task: str, cand: Tensor, top1: Tensor, target: int) -> dict[str, Any]:
    """The lead's edit: nothing trained or read; the row prints the windows at the trigger where W0's top-1 is the target."""
    hit, tok = cand[top1 == target], vpd.tokenizer()
    shown = [tok.decode(w[: HALF + 1].tolist()) for w in hit[:50]]
    for line in shown:
        log(f"[{task}] W0's top-1 is the target at: {line!r}")
    return rules.gap(rules.NO_VERSION, windows_top1_target=len(hit), of_windows=len(cand), shown=shown)


def table_state(j: Job, task: str, tb: Table) -> ST.DW:
    """Arm A, 10 steps, each window's own answer as the target (module docstring); the state saved."""
    col = lambda toks, a: CP.with_target(Windows(toks, torch.zeros(len(toks)), torch.zeros(len(toks)), torch.zeros(len(toks)), HALF), a)
    cfg = ArmAConfig(o_token=LAST, steps=FT_STEPS, checkpoints=(FT_STEPS,), seed=SEED, ci_len=RO.WIN)
    ft = finetune_arm_a(j.net.model, col(tb.train, tb.train_answer), col(tb.held, tb.held_answer), cfg, j.net.device, log)
    return save_state(j, task, ft.deltas[FT_STEPS], ft.evals[FT_STEPS])


def save_state(j: Job, task: str, dW: ST.DW, evals: dict[str, Any]) -> ST.DW:
    path = LARGE / f"dW_{task}_s{SEED}.pt"
    torch.save(dW, path)
    record(j, task, finetune=evals, dW_sha256=KD.sha256(path))
    return dW


def scored(rule: Any, on: dict[str, bool], block: int, state: dict[str, Any], control: dict[str, Any] | None) -> dict[str, Any]:
    """The state against its spared control at b*, with the band over both."""
    b = str(block)
    band = AP.band_over([x for x in (state, control) if x], block)
    return {**rule(on[b], state[b]["A_lo"], control[b]["A_lo"] if control else math.nan, band, control is not None), "block": block}


def controls_first(j: Job, c: AP.Capability, f: Basis, block: int | None) -> dict[str, bool]:
    """C's control alone, and the do-nothing state on it (I0, written under `R1`, C's name) when a block is possessed."""
    on = RD.control(j, c, f)
    if block is not None:
        RD.do_nothing(j, c, f, on, block)
    return on


def read_table(j: Job, task: str, tb: Table, f: Basis, on: dict[str, bool], block: int) -> dict[str, Any]:
    s = Site(tb.held, tb.held_answer, HALF)
    w0 = AP.read_all(j.net, s, None, f.bases, f.randoms)
    state = RD.read_state(j, f"{task}_s{SEED}", s, table_state(j, task, tb), w0, f)
    write = RD.read_write(j, task, s, tb.train, (tb.keys, tb.answers), state["G_logodds"], w0, f, PREV)
    return scored(rules.r3, on, block, state, write)


def trigger_row(j: Job, task: str, tag: int, t: int, target: int, rows0: Tensor, rows: Tensor) -> dict[str, Any]:
    """I1, I2: the table at the trigger, its admissibility, then the readout and control alone, then `read_table`."""
    P, nf = j.S["pile"], j.S["pile"]["fit_row_count"]
    key = torch.tensor([t])
    cand = RO.next_token_windows(rows[:nf], key, PREV, CAPS[0], P["positions_seed"])[0]
    hcand = RO.next_token_windows(rows[nf:], key, PREV, CAPS[1], P["held_seed"])[0]
    top1 = w0_top1(j.net, cand)
    vals = common_values(top1, target)
    c, w0, hh, hw, hy, per = AP.value_capability(j.net, f"W0's guess, {task}", vals, (cand, hcand), HALF, CAPS, UNFIT_BELOW)
    fit = [v for v, n in zip(vals.tolist(), per, strict=True) if n >= UNFIT_BELOW]
    tb = trigger_table(j.net, rows0, t, fit, P["positions_seed"])
    adm = rules.parity(np.array(tb.answers), np.array(tb.answers))
    record(j, task, trigger=t, target=target, values=vals.tolist(), fit_per_value=per, fit_values=fit, per_key=tb.per_key, keys=tb.keys, answers=tb.answers, admissibility=adm)
    if not rules.keeps_target(target, fit, tb.answers):
        return no_version(j, task, cand, top1, target)
    if not adm["admissible"]:
        return rules.gap(rules.NO_C, admissibility=adm["label"])
    torch.save({"train": tb.train, "train_answer": tb.train_answer, "held": tb.held, "held_answer": tb.held_answer, "keys": tb.keys, "answers": tb.answers}, LARGE / f"windows_{task}.pt")
    readout = RD.possession(j, c, w0, (hh, hw, hy), LARGE / f"readout_{task}.pt", fit)
    record(j, task, readout=readout)
    block, f = readout["b_star"], RD.value_form(j, c, tag)
    on = controls_first(j, c, f, block)
    if block is None:
        return rules.r3(None, math.nan, math.nan, math.nan)
    if not on[str(block)]:
        return rules.r3(False, math.nan, math.nan, math.nan)
    return read_table(j, task, tb, f, on, block)


def emoticon_row(j: Job, tag: int, rows0: Tensor, rows: Tensor) -> dict[str, Any]:
    """I2. The plan's drop order: past 20 min the emoticon moves to the stage-2 job. The trigger: module docstring."""
    if time.monotonic() - START > EMOTICON_BY_S:
        return rules.gap("past 20 min: not read here, moves to the stage-2 job")
    w = torch.load(AP.checked(EMOTICON, j.rec, j.res["sha256"]), weights_only=False)
    v, n = torch.cat([w["train"].tokens, w["heldout"].tokens])[:, HALF].unique(return_counts=True)
    record(j, "emoticon", old_windows_at_20=dict(zip(v.tolist(), n.tolist(), strict=True)))
    return trigger_row(j, "emoticon", tag, int(v[n.argmax()]), O_TARGET, rows0, rows)


# ---- lowercase, two-sided ----


def capital_tokens(tok: Any, vocab: int) -> Tensor:
    """Per token id, a capital word start; the rows hold ids past the tokenizer's length (the model's vocabulary is larger)."""
    out = torch.zeros(vocab, dtype=torch.bool)
    out[: len(tok)] = torch.tensor([bool(PF.CAPITAL.match(s)) for s in tok.convert_ids_to_tokens(list(range(len(tok))))])
    return out


def case_sites(rows: Tensor, tok: Any, caps: Tensor, n: int, seed: int) -> tuple[Site, Site, int]:
    """The case positions of `rows`, as (lowercased windows, cased windows, cased windows dropped), at most n of each. The answer
    is the context's form, the other form `alt`. A cased window with no capital word start at 0-20 drops: its context reads lowercased."""
    cp = PF.case_pairs(rows, tok)
    wl, il = PF.windows_at(cp.lower, cp.in_lower, HALF, len(cp.in_lower), seed)
    wc, ic = PF.windows_at(rows, cp.in_cased, HALF, len(cp.in_cased), seed)
    keep = caps[wc[:, : HALF + 1]].any(1)
    lower = Site(wl, cp.in_lower.a[il], HALF, cp.in_lower.b[il])
    cased = Site(wc[keep], cp.in_cased.b[ic][keep], HALF, cp.in_cased.a[ic][keep])
    return AP.capped_held(lower, n, seed), AP.capped_held(cased, n, seed), int((~keep).sum())


def both_cases(net: Any, lower: Site, cased: Site) -> tuple[Site, Tensor, Tensor, Tensor]:
    """The two classes as one site (lowercased first), the class (1 = cased), the residuals and W0's two logits (lower, Capital)."""
    join = lambda a, b: torch.cat([a, b])
    assert lower.alt is not None and cased.alt is not None
    s = Site(join(lower.toks, cased.toks), join(lower.tgt, cased.tgt), HALF, join(lower.alt, cased.alt))
    y = join(torch.zeros(len(lower.toks), dtype=torch.long), torch.ones(len(cased.toks), dtype=torch.long))
    h = RO.cache64(net, s.toks, torch.tensor([0]), HALF)[0]
    d = AP.reads(net, s)[1]
    return s, y.to(h.device), h, torch.stack([torch.zeros_like(d), (2 * y - 1) * d], 1).float().to(h.device)


def lowercase_state(j: Job, rows0: Tensor, tok: Any) -> ST.DW:
    """Arm B on rows 0-2399 in order, even rows lowercased, odd cased: Adam 3e-5, batch 8, 300 steps, clip 1. The state saved."""
    train = rows0[: LC.STEPS * LC.BATCH].clone()
    train[::2] = lowercase_rows(train[::2], tok, len(train[::2]))
    low = LC.Lower(train, lowercase_rows(pile_rows("val", 0, 256, log=log), tok, 64), None, math.nan)
    return save_state(j, "lowercase", LC.arm_b_states(j.net.model, j.net.device, low, "adam", SEED, (LC.STEPS,), log)[LC.STEPS], {})


def push_inputs(j: Job, rows0: Tensor, tok: Any, s: Site) -> tuple[Tensor, Tensor]:
    """K = 1: A the mean block-1 MLP input over lowercased train windows; U the mean over the site's word pairs of (u_lower - u_Capital)."""
    cp = PF.case_pairs(rows0[: PUSH_ROWS : 2], tok)
    train = PF.windows_at(cp.lower, cp.in_lower, HALF, SITE_CAP, j.S["pile"]["positions_seed"])[0]
    lm = j.net.net.lm_head.weight.detach().float()
    assert s.alt is not None
    return PT.window_inputs(j.net.model, train, HALF).mean(0, keepdim=True).T, (lm[s.tgt.to(lm.device)] - lm[s.alt.to(lm.device)]).mean(0, keepdim=True).T


def read_lowercase(j: Job, rows0: Tensor, tok: Any, sites: tuple[Site, Site], f: Basis, on: dict[str, bool], block: int) -> dict[str, Any]:
    """The state on the lowercased half against the push at matched gain; the cased half read for the state's gain alone."""
    lower, cased = sites
    w0 = AP.read_all(j.net, lower, None, f.bases, f.randoms)
    dW = lowercase_state(j, rows0, tok)
    state = RD.read_state(j, f"lowercase_s{SEED}", lower, dW, w0, f)
    G_cased = float((AP.reads(j.net, cased, dW)[1] - AP.reads(j.net, cased)[1]).mean())
    log(f"[lowercase] G lowercased {state['G_logodds']:.4f}, G cased {G_cased:.4f} (under 0 names the constant push)")
    push = RD.read_matched(j, "lowercase", lower, push_inputs(j, rows0, tok, lower), state["G_logodds"], w0, f)
    return {**scored(rules.i3, on, block, state, push), "G_lowercased": state["G_logodds"], "G_cased": G_cased}


def case_row(j: Job, tag: int, rows0: Tensor, rows: Tensor) -> dict[str, Any]:
    """I3: C = the context's case on rows 32768-33023 and their lowercased twins, read at rows 4096-4351's case positions."""
    P, tok = j.S["pile"], vpd.tokenizer()
    caps = capital_tokens(tok, j.net.net.lm_head.weight.shape[0])
    lf, cf, nf = case_sites(rows[:XC_FIT_ROWS], tok, caps, XC_CAPS[0], P["positions_seed"])
    lh, ch, nh = case_sites(rows[XC_FIT_ROWS:XC_ROWS], tok, caps, XC_CAPS[1], P["held_seed"])
    lower, cased, nr = case_sites(rows0[4096:4352], tok, caps, SITE_CAP, P["positions_seed"])
    site, ans = torch.cat([lower.toks, cased.toks]), torch.cat([torch.zeros(len(lower.toks)), torch.ones(len(cased.toks))]).long()
    adm = rules.parity(caps[site[:, : HALF + 1]].any(1).long().numpy(), ans.numpy())
    record(j, "lowercase", windows={"fit": [len(lf.toks), len(cf.toks)], "held": [len(lh.toks), len(ch.toks)], "read": [len(lower.toks), len(cased.toks)]}, cased_dropped={"fit": nf, "held": nh, "read": nr}, admissibility=adm)
    if not adm["admissible"]:
        return rules.gap(rules.NO_C, admissibility=adm["label"])
    torch.save({"lower": vars(lower), "cased": vars(cased)}, LARGE / "windows_lowercase.pt")
    _, y, h, w0 = both_cases(j.net, lf, cf)
    held, hy, hh, hw = both_cases(j.net, lh, ch)
    c = AP.Capability("the context's case", h, y, held, 1)
    readout = RD.possession(j, c, w0, (hh, hw, hy), LARGE / "readout_lowercase.pt", [0, 1])
    record(j, "lowercase", readout=readout)
    block, f = readout["b_star"], RD.value_form(j, c, tag)
    on = controls_first(j, c, f, block)
    if block is None:
        return rules.r3(None, math.nan, math.nan, math.nan)
    if not on[str(block)]:
        return rules.i3(False, math.nan, math.nan, math.nan)
    return read_lowercase(j, rows0, tok, (lower, cased), f, on, block)


def unstopped(row: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """A readout early-check miss on this row: the row is unreadable, a gap, and the job goes on (no stop file). Those bars are
    unmeasured on 2 and 8 classes."""
    try:
        return row()
    except RO.ReadoutStop as e:
        log(f"[readout miss, the row reads {rules.NO_READOUT}; the job goes on] {e}")
        return rules.gap(rules.NO_READOUT, readout_miss=str(e))


def tasks(j: Job, rows: Tensor, trigger: int, target: int) -> list[tuple[str, Any]]:
    """The plan's order: ladder, lowercase, emoticon. `rows` are E64's, X_C; rows 0-32767 give the tables and the read site."""
    rows0 = pile_rows("train", 0, 32768, log=log)
    return [("I1", lambda: unstopped(lambda: trigger_row(j, "ladder", 5, trigger, target, rows0, rows))), ("I3", lambda: unstopped(lambda: case_row(j, 6, rows0, rows))),
            ("I2", lambda: unstopped(lambda: emoticon_row(j, 7, rows0, rows)))]  # fmt: skip
