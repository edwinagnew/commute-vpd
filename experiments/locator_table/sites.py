"""E97's tasks: each task's site (inputs and the positions read), its saved states and its control, every file checked
against the sha256 its experiment saved. GPU job only (the E53 sites tokenize Pile rows on load).

Sites: E62's 200 read windows (41 tokens, read at 20); E53's lowercase and cipher on whole val rows 0-255 at every scored
position, A0's read (`performed.log_odds`, the E96 check's F2; read here in fp32), and beside on E53's own 127-token windows
(E96's site); the 24-key and 32-key tables' held-out windows (E90's `kv_site`); the trigger tasks' held-out windows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from commute import locator as L
from commute import performed
from commute import regating as G
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import load_experiment
from commute.keydep import sha256

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
AP = load_experiment("new_vs_reuse", "apparatus")
E53 = load_experiment("performed_anchor", "run")
CUE, E96 = EXP / "induction_cue" / "results", EXP / "locate_mechanism" / "results"
PAIR, REG = EXP / "open_pair" / "results" / "large", EXP / "regating_table" / "results" / "large"
RECRUIT = EXP / "recruitment_test" / "results"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
COPY24 = PAIR / "amend1" / "windows_copy_trained.pt"
E53_STEP, E53_ROWS = 300, 256


@dataclass(frozen=True)
class Task:
    row: str  # the table's task name
    e90: str  # E90's task name, whose C the table copies
    site: str  # a key of `site`
    states: dict[int, Path]  # seed -> state file
    control: str  # the control task: the same seed's state, on its own site
    constant: bool = False  # a constant answer: beside only, verdict "does not apply" (decisions.md line 121)
    in_table: bool = True


def _seeds(d: Path, stem: str, seeds: range) -> dict[int, Path]:
    return {s: d / f"{stem}_s{s}.pt" for s in seeds}


def _e53(task: str) -> dict[int, Path]:
    return {s: E53.LARGE / f"dW_{task}_s{s}_step{E53_STEP}.pt" for s in (0, 1)}


def _trigger(t: str, row: str) -> list[Task]:
    d = RECRUIT / "large" / "states"
    null = Task(f"{row} null", row, t, {0: d / f"{t}_null.pt", 1: d / f"{t}_null.pt"}, row, in_table=False)
    return [Task(row, "T1 emoticon" if t == "emoticon" else "T2", t, {s: d / f"{t}_s{s}.pt" for s in (0, 1)}, null.row, True), null]


TASKS: dict[str, Task] = {t.row: t for t in [
    Task("T5 copy", "T5 copy", "e62_copy", _seeds(CUE / "large", "copy", range(3)), "T5 lookup"),
    Task("T5 lookup", "T5 lookup", "e62_lookup", _seeds(CUE / "large", "lookup", range(3)), "T5 copy"),
    Task("T7 lowercase", "T7 lowercase", "e53_lowercase", _e53("lowercase"), "T8 cipher"),
    Task("T8 cipher", "T8 cipher", "e53_cipher", _e53("cipher"), "T7 lowercase"),
    Task("T12 rank-1", "T12 rank-1", "kv_rank1", _seeds(PAIR / "amend1", "rank1", range(3)), "T11 cipher"),
    Task("T11 elicit", "T11 elicit", "kv_elicit", _seeds(PAIR, "elicit", range(3)), "T11 cipher"),
    Task("T13 copy-24", "T13 copy-24", "kv_copy24", _seeds(PAIR / "amend1", "copy", range(3)), "T11 cipher"),
    Task("T11 cipher", "T11 cipher", "kv_cipher24", _seeds(PAIR, "cipher", range(3)), "T12 rank-1"),
    Task("T4 ident32", "T4 ident32", "kv_ident32", _seeds(REG, "ident32", range(2)), "T6 rand32"),
    Task("T6 rand32", "T6 rand32", "kv_rand32", _seeds(REG, "rand32", range(2)), "T4 ident32"),
    *_trigger("emoticon", "T1 emoticon"), *_trigger("r1_t534_v369", "T2 r1_t534_v369"), *_trigger("r2_t359_v816", "T2 r2_t359_v816"),
]}  # fmt: skip
STAGES = {1: ["T5 copy", "T5 lookup"], 2: ["T7 lowercase", "T8 cipher"],
          3: ["T12 rank-1", "T11 elicit", "T13 copy-24", "T11 cipher", "T4 ident32", "T6 rand32", "T1 emoticon", "T2 r1_t534_v369", "T2 r2_t359_v816"]}  # fmt: skip
KV = {"kv_rank1": PAIR / "amend1" / "windows_rank1_trained.pt", "kv_elicit": PAIR / "windows_elicit_trained.pt", "kv_copy24": COPY24,
      "kv_cipher24": PAIR / "windows_cipher_trained.pt", "kv_ident32": REG / "windows_ident32.pt", "kv_rand32": REG / "windows_rand32.pt"}  # fmt: skip


def records() -> dict[Path, str]:
    """Every saved sha256 this job's inputs have: E90's (E62, the tables), E53's fine-tune record, E38's states."""
    rec: dict[Path, str] = dict(AP.records())
    ft = json.loads((E53.RESULTS / "finetune.json").read_text())["states"]
    rec |= {E53.LARGE / v["file"]: v["sha256"] for v in ft.values()}
    st = json.loads((RECRUIT / "amend2_stage0.json").read_text())["states"]
    rec |= {RECRUIT / "large" / "states" / f"{k.replace('/', '_')}.pt": v["sha256"] for k, v in st.items()}
    return rec


@dataclass
class Files:
    """Each input's sha256, kept in `seen`; one that differs from its saved record raises."""

    rec: dict[Path, str] = field(default_factory=records)
    seen: dict[str, str] = field(default_factory=dict)

    def checked(self, path: Path) -> Path:
        sha = sha256(path)
        self.seen[str(path.relative_to(EXP))] = sha
        if path in self.rec and self.rec[path] != sha:
            raise ValueError(f"{path}: sha256 {sha} differs from the saved record {self.rec[path]}")
        return path

    def state(self, path: Path) -> ST.DW:
        """fp32 on CPU; E53's are saved in bf16."""
        if not path.is_file():
            raise FileNotFoundError(path)
        return {k: v.float() for k, v in torch.load(self.checked(path), weights_only=True, map_location="cpu").items()}


def e62(F: Files, m: str) -> L.Site:
    v = torch.load(F.checked(CUE / "large" / "windows.pt"), weights_only=False)
    half = json.loads((CUE / "predictions.json").read_text())["settings"]["half"]
    w = v["read"]
    return L.windows(w.tokens[:, : 2 * half + 1], v["target"][m]["read"], half, w.row)


def _e53_text(task: str) -> tuple[torch.Tensor, Any]:
    return E53.task_text(task, pile_rows("val", 0, E53_ROWS), E53.e53_pi() if task == "cipher" else {})


def e53_rows(task: str) -> L.Site:
    """Whole rows at every scored position: the read at pos - 1, the other form b (A0's `performed.log_odds`)."""
    text, at = _e53_text(task)
    return L.Site(text, at.row, at.pos - 1, at.a, at.row, at.b)


def e53_windows(task: str) -> L.Site:
    """E53's own 500 windows of 127 tokens, read at 63 (`performed_anchor` `Run.scored`'s draw, as E96's stage 2)."""
    C = E53.CFG
    text, at = _e53_text(task)
    w, idx = performed.windows_at(text, at, C.half, C.n_windows, C.window_seed)
    assert torch.equal(w[:, C.half + 1], at.a[idx]), "a window's next token is not its scored form"
    return L.windows(w, at.a[idx], C.half, at.row[idx], at.b[idx])


def kv(F: Files, path: Path) -> tuple[L.Site, dict[str, Any]]:
    """A key table's held-out windows (E90's `kv_site`), and the file (its train windows and table, for the plants)."""
    w = torch.load(F.checked(path), weights_only=False)
    held = w["heldout"]
    return L.windows(held.tokens, w["table"][held.tokens[:, held.half]], held.half, held.row), w


def trigger(F: Files, name: str) -> L.Site:
    """A constant-answer task's held-out windows (`regating.load_windows`), its one target at every window."""
    F.checked(TRANSPORT / "large" / "windows.pt" if name == "emoticon" else LADDER / f"windows_{name}.pt")
    _, held, target = G.load_windows(name, TRANSPORT, LADDER)
    return L.windows(held.tokens, torch.full((len(held.tokens),), int(target)), held.half, held.row)


def site(F: Files, key: str) -> L.Site:
    if key.startswith("e62_"):
        return e62(F, key[4:])
    if key.startswith("e53w_"):
        return e53_windows(key[5:])
    if key.startswith("e53_"):
        return e53_rows(key[4:])
    if key in KV:
        return kv(F, KV[key])[0]
    return trigger(F, key)
