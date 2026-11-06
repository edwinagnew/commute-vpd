"""E116's tasks and readouts: each task's states, value set S, fit windows (P2's fit, P3) and held-out windows (P1, P2's
read), and each readout C's values, key slot, excluded types and scoring windows. Paths and sha256 records come from E97's
`locator_table/sites.py` (`TASKS`, `Files`); every file read is checked. Plan: reports/plans/2026-11-06_plan_e116-definition-table.md,
appendix §1 and §4, and the lead's decisions of 2026-11-06 (predictions.json "lead_decisions_2026-11-06"). GPU job only
(the E53 sites tokenize Pile rows)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.data import pile_rows
from commute.jobs import load_experiment
from commute.performed import windows_at

SITES = load_experiment("locator_table", "sites")
E53 = SITES.E53
HALF = 20  # the 41-token windows' read index (E54's, E62's, E112's)


@dataclass(frozen=True)
class Site:
    """Windows (N, T), read at index `half`; `tgt` (N,) the task's answer token there."""

    toks: Tensor
    half: int
    tgt: Tensor


@dataclass(frozen=True)
class Task:
    name: str
    row: str  # E97's / E100's row name, for the locator column
    states: dict[int, Path]
    values: Tensor  # S, the task's value set (tokens)
    keys: Tensor  # every key of the task, trained and held-out (empty for the text ciphers)
    fit: Site
    held: Site | None
    p1_na: str | None  # why P1 cannot be applied; None when it can
    g: str  # "identity" (the answer is the readout's own value) or "table" (lead's decision 1, 2026-11-06)


@dataclass(frozen=True)
class Readout:
    name: str
    kind: str  # "copy", "guess" or "identity"
    values: Tensor  # the classes (tokens)
    slot: int  # the key slot relative to the read index: 0 (key tables, text ciphers) or -1 (E62) (lead's decision 3)
    exclude: Tensor  # types left out of the Pile positions at the key slot
    score: Site  # the task windows P0 is scored on
    own: str
    twins: tuple[str, ...]
    locator_block: int | None = None  # E97's located block before the last (E62 copy: heads (2,3)+(2,4))


def uniq(x: Tensor | list[int]) -> Tensor:
    return torch.tensor(list(dict.fromkeys(torch.as_tensor(x).tolist())), dtype=torch.long)


def add_records(F: Any) -> None:
    """broad_cipher's saved sha256 (E53 Amendment 1b), beside the records `sites.records` reads."""
    ft = json.loads((E53.RESULTS / "finetune_amend1b.json").read_text())["states"]
    F.rec |= {E53.LARGE / v["file"]: v["sha256"] for v in ft.values()}


def kv_site(w: dict[str, Any]) -> Site:
    t = w["heldout"].tokens
    return Site(t, w["heldout"].half, w["table"][t[:, w["heldout"].half]])


def kv_task(F: Any, row: str, name: str, trained: Path, new: Path | None, g: str) -> Task:
    """A key table: fit = the trained keys' held-out-row windows (E90's `kv_site`); held = the held-out keys' windows."""
    w = torch.load(F.checked(trained), weights_only=False)
    vals, keys = list(w["task"].values), list(w["task"].keys)
    held, na = None, "no held-out keys: the file's held-out split holds the same keys (appendix §2)"
    if new is not None:
        n = torch.load(F.checked(new), weights_only=False)
        held, na, vals, keys = kv_site(n), None, vals + list(n["task"].values), keys + list(n["task"].keys)
    return Task(name, row, SITES.TASKS[row].states, uniq(vals), uniq(keys), kv_site(w), held, na, g)


def e62_task(F: Any, row: str, m: str) -> tuple[Task, int]:
    """E62's map m on its 800 train windows (fit) and 200 read windows (held; Bs never trained); and the marker."""
    v = torch.load(F.checked(SITES.CUE / "large" / "windows.pt"), weights_only=False)
    tg = v["target"][m]
    fit, held = Site(v["train"].tokens, HALF, tg["train"]), Site(v["read"].tokens, HALF, tg["read"])
    na = None if m == "copy" else "a lookup defines no answer for an unseen key (appendix §2)"
    g = "identity" if m == "copy" else "table"
    return Task(f"e62_{m}", row, SITES.TASKS[row].states, uniq(torch.cat([tg["train"], tg["read"]])), uniq(v["pools"]["keys"]), fit, held, na, g), int(
        v["pools"]["marker"][0])  # fmt: skip


def e53_task(task: str, row: str, pi: dict[int, int], S: Any) -> Task:
    """A text cipher on E53's rows, E53's 127-token windows at positions whose target is a table value: fit = up to
    e53_fit_windows (seeded) of the ciphered train rows 0-2399, held = every window of the 64 ciphered held rows."""
    h = S.e53_half

    def site(rows: Tensor, n: int) -> Site:
        text, at = E53.task_text(task, rows, pi)
        w, idx = windows_at(text, at, h, n, S.e53_window_seed)
        return Site(w, h, at.a[idx])

    fit = site(pile_rows("train", 0, 4096)[: E53.CFG.n_train], S.e53_fit_windows)
    held = site(pile_rows("val", 0, 256)[: E53.CFG.n_held], 10**9)
    states = {s: E53.LARGE / f"dW_{task}_s{s}_step{SITES.E53_STEP}.pt" for s in (0, 1)}
    na = "the table defines no target for an unseen word (appendix §2)"
    return Task(task, row, states, uniq(sorted(set(pi.values()))), uniq([]), fit, held, na, "table")


def build(F: Any, S: Any) -> tuple[dict[str, Task], list[Readout]]:
    """The ten tasks read (26 states) and their readouts (appendix §4)."""
    pair, reg = SITES.PAIR, SITES.REG
    a1 = pair / "amend1"
    T = {t.name: t for t in [
        kv_task(F, "T4 ident32", "ident32", reg / "windows_ident32.pt", None, "identity"),
        kv_task(F, "T6 rand32", "rand32", reg / "windows_rand32.pt", None, "table"),
        kv_task(F, "T13 copy-24", "copy24", a1 / "windows_copy_trained.pt", a1 / "windows_copy_new.pt", "identity"),
        kv_task(F, "T12 rank-1", "rank1", a1 / "windows_rank1_trained.pt", a1 / "windows_rank1_new.pt", "identity"),
        kv_task(F, "T11 elicit", "elicit", pair / "windows_elicit_trained.pt", pair / "windows_elicit_new.pt", "table"),
        kv_task(F, "T11 cipher", "cipher24", pair / "windows_cipher_trained.pt", pair / "windows_cipher_new.pt", "table"),
    ]}  # fmt: skip
    (copy, marker), (lookup, _) = e62_task(F, "T5 copy", "copy"), e62_task(F, "T5 lookup", "lookup")
    T |= {copy.name: copy, lookup.name: lookup}
    maps = {"cipher": E53.e53_pi(), "broad_cipher": E53.a1_maps()["broad_cipher"]}
    T |= {n: e53_task(n, "T8 cipher" if n == "cipher" else "broad cipher", maps[n], S) for n in maps}
    assert copy.held is not None and lookup.held is not None
    none = uniq([])

    def guess(n: str, twins: tuple[str, ...]) -> Readout:
        held = T[n].held
        assert held is not None
        return Readout(f"guess_{n}", "guess", T[n].values, 0, T[n].keys, held, n, twins)

    def ident(n: str, twins: tuple[str, ...], slot: int = 0, score: Site | None = None) -> Readout:
        return Readout(f"identity_{n}", "identity", T[n].keys, slot, none, score or T[n].fit, n, twins)

    R = [Readout("copy", "copy", uniq(copy.held.tgt), -1, torch.cat([copy.keys, torch.tensor([marker])]), copy.held, "e62_copy",
                 ("e62_lookup",), S.copy_locator_block),
         guess("rank1", ("cipher24",)), guess("elicit", ("cipher24",)), guess("cipher", ()), guess("broad_cipher", ()),
         ident("ident32", ("rand32",)), ident("rand32", ("ident32",)), ident("copy24", ("cipher24", "rank1", "elicit")),
         ident("cipher24", ("rank1", "elicit")), ident("e62_lookup", ("e62_copy",), -1, lookup.held)]  # fmt: skip
    return T, R
