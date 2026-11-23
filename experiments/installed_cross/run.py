"""E271 (cell P2.65), the installed-skill cross. For each of three dissimilar tasks the released 67M (M) is at floor on,
a base M_T does T only behind a cue token (uncued items and Pile text distilled to M); every base (and M) is then
fine-tuned uncued on every task, three seeds, each stopped at one held-out gain per task. Reuse = the diagonal (a base
fine-tuned on its own installed task), new = the off-diagonal, by training history. Plan
reports/plans/2026-11-22_plan_p2-65-installed-cross.md; settings, bars and expected values results/predictions.json
(the run refuses unless it is committed unchanged). Shared steps: `commute.installed`; readers E211's
(`realistic_reads.Reader` via experiments/realistic_tasks/run2.py), ADL `commute.adl`, KL rl_67m's `kl_from_base`.

The controls stage, first and alone, decomposition-free (jobs 1-4); the readings after (job 5); decompositions in a
later file once job 4 passes (plan, "Stage D").
  1 ctl            the three tasks' items (saved, sha256), a zero update reads as M, M at floor on addition and cipher,
                   the cue's own effect on M per task (reported)
  2 install TASK   three distillation weights, each with an early check every 100 steps (a leak at step 500 ends it);
                   the lock on the stop rows (cued gain over M cued, uncued within 0.2 decades of M, Pile KL under the KL
                   of M's own fine-tune at G*); the strongest weight whose lock holds is kept; exit 1 if none
  3 cross BASE TASK  one cell, 3 seeds (BASE in TASKS + ("M",)): fine-tunes to G*_task (half the cued gain); steps to G*,
                   Pile KL; for a moved base the transplant onto M (gain kept on M over gain on the base)
  4 controls       (no GPU, cpu-run) controls (2) steps and (3) transplant against the filed bars, the cards of the cross
                   jobs alike -> results/controls.json; exit 1 on a miss, which stops the readings (the miss is the result)
  5 read BASE TASK refuses unless controls.json passed: per state the readings at matched gain, its matched Gaussian,
                   and the state rescaled to its task's median size (matched size); a Gaussian gaining >= MIN_GAIN exits 1
Amended 2026-11-22 after the gate-5 review (reports/reviews/2026-11-22_review_e271-installed-cross.md): predictions.json
"amendment_2026-11-22" lists every change.

    sbatch --comment='E271 rehearse: ctl' cluster/installed_cross.sbatch ctl   (the rest: the sbatch file)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import adl as ADL
from commute import finetune as FT
from commute import installed as IN
from commute import realistic as R
from commute import realistic_reads as RR
from commute import rl_tasks as RT
from commute import steering as ST
from commute import vpd
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable
from commute.rl_tasks_2191 import Vocab

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, LARGE, READS = RESULTS / "predictions.json", RESULTS / "large", RESULTS / "read"
SC = load_experiment("installed_cross", "score")  # the cell's tasks by the filed rule (amendment 2026-11-22d)
TASKS = SC.TASKS  # empty until every candidate's install has finished
BASES = (*TASKS, "M")
S1 = load_experiment("rl_67m", "run_stage1")  # kl_from_base
R2 = load_experiment("realistic_tasks", "run2")  # E211's reader, Out, Stopped, brief


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    L: int
    n_train: int
    n_held: int  # held[:n_stop] are the stop rows, held[n_stop:] the read rows
    n_stop: int
    shots: int
    n_keys: int
    key_ids: tuple[int, int]
    value_ids: tuple[int, int]
    yo_pool: int
    sst2_min_tokens: int  # SST-2 sentences of at least this many tokens (E209's 8; amendment 2026-11-22d)
    parity_range: tuple[int, int]  # parity's numbers: [lo, hi) whose " n" is one token (amendment 2026-11-22f)
    parity_held_frac: float  # the share of those numbers held out (held items use only these)
    eos_gap: int  # Yoruba: no end-of-text in a window's last eos_gap tokens before the answer (amendment 2026-11-22)
    cue: int
    seed: int
    lambdas: tuple[float, ...]
    install_lr: float
    install_steps: int
    install_sweep: dict[str, list[list[float]]]  # per task [(ans_w, lambda), ...], weakest first; others (1, lambdas)
    install_steps_by_task: dict[str, int]  # a task listed here trains this many steps (amendment 2026-11-22c)
    install_every: int
    ft_lr: float
    ft_steps: int
    batch: int
    pile_batch: int
    pile_len: int
    grad_clip: float
    seeds: tuple[int, ...]
    kl_rows: int
    kl_batch: int
    adl_k: int
    adl_rows: int
    adl_batch: int

    def __post_init__(self) -> None:
        for f in fields(self):
            if isinstance(v := getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(v))
        assert self.n_stop < self.n_held <= self.yo_pool and self.n_held - self.n_stop >= 256 and self.L >= 32
        assert self.key_ids[1] <= self.value_ids[0] and self.lambdas == tuple(sorted(self.lambdas)) and self.seeds == (0, 1, 2)
        assert self.ft_steps >= 5 * self.install_every and self.pile_len <= vpd.SEQ_LEN


class Ctx:
    def __init__(self, gpu: bool) -> None:
        require_filed(PRED)
        self.P = json.loads(PRED.read_text())
        self.C, self.th, self.exp = config_from_settings(Config, self.P["settings"]), self.P["thresholds"], self.P["experiment"]
        if gpu:
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as every saved reading
            self.net = ST.Net("cuda")
            self.M = {k: w.clone() for k, w in self.net.W0.items()}
            self.kl_rows = cached_rows("val", 0, self.C.kl_rows)

    def tasks(self) -> dict[str, IN.Task]:
        assert (got := sha(LARGE / "tasks.pt")) == (want := done("ctl")["data"]["sha256"]), f"tasks.pt sha256 {got}, ctl recorded {want}"
        d = torch.load(LARGE / "tasks.pt", weights_only=False)
        return {n: IN.Task(n, d[n]["train"], d[n]["held"], d[n]["dense"], d[n].get("cue_pos", 0)) for n in SC.CANDIDATES if n in d}
        # (a tasks.pt from before amendment f has no parity; its other tasks load as before)

    def base(self, name: str) -> ST.DW | None:
        return None if name == "M" else load_dw(done(f"install_{name}")["kept"]["file"], done(f"install_{name}")["kept"]["sha256"])


def done(name: str) -> dict[str, Any]:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_dw(name: str, dW: ST.DW) -> dict[str, Any]:
    (LARGE / "states").mkdir(parents=True, exist_ok=True)
    p = LARGE / "states" / f"{name}.pt"
    torch.save({k: v.to(torch.bfloat16) for k, v in dW.items()}, p)
    return {"file": str(p.relative_to(REPO)), "sha256": sha(p)}


def load_dw(file: str, want: str) -> ST.DW:
    p = REPO / file
    assert (got := sha(p)) == want, f"{p}: sha256 {got}, recorded {want}"
    return {k: v.float() for k, v in torch.load(p, weights_only=False).items()}


def out(x: Ctx, job: str) -> Any:
    """E211's Out (put, write, stop), built without touching CUDA, so the CPU job (controls) runs on cpu-run (review
    2026-11-22 finding 1; run2.py's Out.__init__ calls torch.cuda.get_device_name)."""
    o = R2.Out.__new__(R2.Out)
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    o.path, o.d = RESULTS / f"{job}.json", {"job": job, "card": card, "states": {}, "experiment": x.exp}
    return o


# ----------------------------------------------------------------------------- job 1: ctl


BUILT = ("addition", "cipher", "yoruba", "sst2", "cipherlocal", "parity")  # every task tasks.pt holds; the cipher is kept, unused
LABELS = {"sst2": R.LABEL_WORDS["L2"], "parity": IN.PARITY_WORDS}  # label tasks: ctl reads M's two-label accuracy


def build_tasks(x: Ctx, keep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every task's items; a task already in `keep` (the existing tasks.pt) is kept as it is, so an install made on it
    stays valid (amendment 2026-11-22d); only the missing ones are built."""
    C, tok = x.C, vpd.tokenizer()
    v = Vocab([tok.decode([i]) for i in range(len(tok))], tok.encode)
    train_rows, val_rows = cached_rows("train", 0, C.n_train), cached_rows("val", 64, C.n_held)
    seq = lambda it: torch.cat([it.toks, it.true[:, None]], 1)
    table = lambda: IN.cipher_table(v.texts, C.n_keys, C.key_ids, C.value_ids, C.seed)

    def cipher(pos: int) -> dict[str, Any]:
        tb = table()
        return {"train": IN.cipher_items(train_rows, v, C.L, C.n_train, tb, C.seed), "held": IN.cipher_items(val_rows, v, C.L, C.n_held, tb, C.seed + 1),
                "dense": False, "table": tb, "cue_pos": pos}  # fmt: skip

    def yoruba() -> dict[str, Any]:
        train_yo, pool = R.pack((tok.encode(t) for t in R.wiki_texts("yo")), tok.eos_token_id, (C.n_train, C.yo_pool), C.L + 1)
        return {"train": train_yo, "held": yoruba_held(x, v, pool, tok.eos_token_id), "dense": True}

    make = {"addition": lambda: {"train": seq(RT.addition_items(train_rows, v, C.L, C.n_train, C.shots, C.seed)),
                                 "held": seq(RT.addition_items(val_rows, v, C.L, C.n_held, C.shots, C.seed + 1)), "dense": False},
            "cipher": lambda: cipher(0), "yoruba": yoruba, "sst2": lambda: sst2_task(C, tok),
            "cipherlocal": lambda: cipher(C.L - 3), "parity": lambda: parity_task(C, v, train_rows, val_rows)}  # fmt: skip  (cipherlocal: the cue over the "\n\n" just before the key)
    return {t: keep[t] if t in keep else make[t]() for t in BUILT}


def parity_task(C: Config, v: Vocab, train_rows: Tensor, val_rows: Tensor) -> dict[str, Any]:
    """Parity (built here, amendment 2026-11-22f): fill, "\\n\\n", a one-token number and " is", answer " even" / " odd";
    train and held numbers disjoint (seeded), so the held rows read the rule, not the numbers seen in training."""
    train_n, held_n = IN.parity_numbers(v.encode, *C.parity_range, C.parity_held_frac, C.seed)
    return {"train": IN.parity_items(train_rows, v, C.L, C.n_train, train_n, C.seed), "held": IN.parity_items(val_rows, v, C.L, C.n_held, held_n, C.seed + 1),
            "dense": False, "numbers": {"train": len(train_n), "held": len(held_n)}}  # fmt: skip


def sst2_task(C: Config, tok: Any) -> dict[str, Any]:
    """SST-2 (P2.38's L2): a sentence's first tokens left-padded with end-of-text to L - len(" Topic:"), the marker, and
    the label token (' positive' for 1, ' negative' for 0) as the answer; train from SST-2 train in a seeded order,
    held from its validation split, sentences of at least sst2_min_tokens tokens (E209's)."""
    labels = tuple(tok.encode(w) for w in R.LABEL_WORDS["L2"])
    assert all(len(t) == 1 for t in labels), labels
    n = C.L - len(tok.encode(R.MARKER))
    out: dict[str, Any] = {"dense": False}
    for part, split, count in (("train", "train", C.n_train), ("held", "validation", C.n_held)):
        texts, y = R.sst2(split, tok.encode, C.sst2_min_tokens)
        idx = torch.randperm(len(texts), generator=torch.Generator().manual_seed(C.seed))[:count]
        assert len(idx) == count, f"SST-2 {split}: {len(idx)} sentences, {count} needed"
        out[part] = IN.label_items(R.label_windows([texts[i] for i in idx.tolist()], tok.encode, n, tok.eos_token_id), y[idx], (labels[0][0], labels[1][0]))
    return out


def yoruba_held(x: Ctx, v: Vocab, pool: Tensor, eos: int) -> Tensor:
    """E211's hard-position rule on windows (review 2026-11-22 finding 3): the answer starts a word (a space then a
    letter, `realistic.word_starts`' rule on the decoded piece) and no end-of-text sits in the last eos_gap tokens; of
    those, the n_held M predicts worst; then a seeded permutation, so the stop rows and the read rows are one draw."""
    C = x.C
    start = torch.tensor([len(t) > 1 and t[0] == " " and t[1].isalpha() for t in v.texts])
    ok = start[pool[:, -1]] & ~(pool[:, -C.eos_gap - 1 :] == eos).any(1)
    cand = pool[ok]
    assert len(cand) >= C.n_held, f"{len(cand)} Yoruba windows pass the word-start and end-of-text rule, {C.n_held} needed"
    lo, _ = IN.answer_lo(x.net.net, cand)
    hard = cand[torch.sort(lo, stable=True).indices[: C.n_held]]
    return hard[torch.randperm(C.n_held, generator=torch.Generator().manual_seed(C.seed))]


def item_sha(d: dict[str, Any]) -> str:
    return hashlib.sha256(b"".join(d[s].numpy().tobytes() for s in ("train", "held"))).hexdigest()


def job_ctl(x: Ctx, o: Any) -> None:
    LARGE.mkdir(parents=True, exist_ok=True)
    old_file = LARGE / "tasks.pt"
    old = torch.load(old_file, weights_only=False) if old_file.is_file() else {}
    o.put("kept_items", {t: item_sha(v) for t, v in old.items()})  # addition's install was made on these exact items
    d = build_tasks(x, old)
    torch.save(d, LARGE / "tasks.pt")
    if changed := [t for t in old if item_sha(d[t]) != item_sha(old[t])]:
        raise o.stop(f"items changed for {changed}: their installs no longer match")
    every = torch.cat([d[t][s][:, :-1].flatten() for t in BUILT for s in ("train", "held")])
    o.put("data", {"sha256": sha(LARGE / "tasks.pt"), "cue_in_data": int((every == x.C.cue).sum()), "cipher_keys": len(d["cipher"]["table"]),
                   "items_sha256": {t: item_sha(d[t]) for t in BUILT},
                   "shapes": {t: {s: list(d[t][s].shape) for s in ("train", "held")} for t in BUILT}})  # fmt: skip
    if o.d["data"]["cue_in_data"] or len(d["cipher"]["table"]) != x.C.n_keys:
        raise o.stop("the cue token occurs in the items, or the cipher has too few keys")
    zero = {k: torch.zeros_like(w) for k, w in x.net.W0.items()}
    h = d["addition"]["held"]
    a = IN.answer_lo(x.net.net, h)[0]
    with x.net.at(zero):
        b = IN.answer_lo(x.net.net, h)[0]
    o.put("zero", z := {"kl": S1.kl_from_base(x.net, x.kl_rows, zero, x.C.kl_batch), "same": bool(torch.equal(a, b))})
    if not (z["kl"] == 0.0 and z["same"]):
        raise o.stop("a zero update does not read as M")
    floor = {}
    for t in BUILT:
        lo, top = IN.answer_lo(x.net.net, d[t]["held"])
        lc, tc = IN.answer_lo(x.net.net, IN.cued(d[t]["held"], x.C.cue, d[t].get("cue_pos", 0)))
        e = lc - lo  # the cue's own effect on M, per row (review finding 6): reported with its standard error
        floor[t] = {"lo": float(lo.mean()), "rate": float(torch.sigmoid(lo).mean()), "top1": float(top.float().mean()),
                    "cued_lo": float(lc.mean()), "cued_top1": float(tc.float().mean()),
                    "cue_effect": float(e.mean()), "cue_effect_se": float(e.std(unbiased=False)) / len(e) ** 0.5}  # fmt: skip
        if t in LABELS:  # amendment 2026-11-22f: M's accuracy between the two label tokens (chance 0.5), cued and uncued
            ids = [vpd.tokenizer().encode(w)[0] for w in LABELS[t]]
            h = d[t]["held"]
            floor[t] |= {"label_acc": IN.label_acc(x.net.net, h, ids), "cued_label_acc": IN.label_acc(x.net.net, IN.cued(h, x.C.cue, d[t].get("cue_pos", 0)), ids)}
    o.put("floor", floor)
    if bad := [t for t, b in x.th["floor_rate_max"].items() if not floor[t]["rate"] <= b]:
        raise o.stop(f"M is not at floor on {bad}")


# ----------------------------------------------------------------------------- job 2: install TASK


def m_reads(x: Ctx, t: IN.Task) -> tuple[Tensor, Tensor]:
    """M's answer log-odds on the stop rows, cued and uncued: the lock's references (amendment 2026-11-22: the lock and
    G* are taken on the stop rows, review finding 9; cued gain over M cued, finding 6)."""
    rows = t.held[: x.C.n_stop]
    return IN.answer_lo(x.net.net, t.cued(rows, x.C.cue))[0], IN.answer_lo(x.net.net, rows)[0]


def uncued_p(lu: Tensor, lm: Tensor) -> dict[str, float]:
    """Descriptive (amendment 2026-11-22d): the uncued answer's probability, typical (sigmoid of the mean log-odds) and
    mean, at M and at the model, beside the decades the bar reads."""
    sig = lambda t: float(torch.sigmoid(t.mean())), lambda t: float(torch.sigmoid(t).mean())
    return {"M_typical": sig[0](lm), "M_mean": sig[1](lm), "typical": sig[0](lu), "mean": sig[1](lu)}


def lock_reads(x: Ctx, t: IN.Task, m: tuple[Tensor, Tensor]) -> dict[str, Any]:
    """The model as net holds it on the stop rows: cued gain over M cued, top-1 cued, uncued shift over M in decades
    (the bar's reading) and in absolute probability (described)."""
    rows = t.held[: x.C.n_stop]
    lc, tc = IN.answer_lo(x.net.net, t.cued(rows, x.C.cue))
    lu, _ = IN.answer_lo(x.net.net, rows)
    return {"cued_gain": float((lc - m[0]).mean()), "cued_top1": float(tc.float().mean()), "uncued_decades": IN.decades(float((lu - m[1]).mean())),
            "uncued_p": uncued_p(lu, m[1])}  # fmt: skip


def lock(x: Ctx, t: IN.Task, dW: ST.DW, m: tuple[Tensor, Tensor]) -> dict[str, Any]:
    """The lock of a candidate base: lock_reads at the base, its Pile KL and size; (a) cued gain at the task's bar and
    G* = half of it at least g_star_min (review finding 4); (b) the uncued shift within the bar."""
    with x.net.at(dW):
        rec = lock_reads(x, t, m)
    rec |= {"pile_kl": S1.kl_from_base(x.net, x.kl_rows, dW, x.C.kl_batch), "size": R.update_size(dW, x.net.W0)}
    a = rec["cued_gain"] >= x.th["cued_gain_min"][t.name] and 0.5 * rec["cued_gain"] >= x.th["g_star_min"]
    return rec | {"a_cued": a, "b_uncued": abs(rec["uncued_decades"]) <= x.th["uncued_decades_max"]}


def install_check(x: Ctx, t: IN.Task, m: tuple[Tensor, Tensor]) -> Any:
    """Gate 4 at install (review finding 5): every install_every steps the lock reads on the stop rows, logged; at
    early_step a shift over the bar by more than leak_margin_decades ends that weight (the lock leaks)."""
    th = x.th["install_early"]

    def f(step: int) -> dict[str, Any]:
        if step == 0:
            return {"done": False}
        r = lock_reads(x, t, m)
        leak = step == th["step"] and abs(r["uncued_decades"]) > x.th["uncued_decades_max"] + th["leak_margin_decades"]
        if step % (5 * x.C.install_every) == 0 or leak:
            log(f"[install {t.name} step {step}] {r}" + (" LEAK: this weight stopped" if leak else ""))
        return r | {"done": leak, "leak_stopped": leak}

    return f


def finetune(x: Ctx, t: IN.Task, target: float, seed: int) -> dict[str, Any]:
    """The base as net holds it, fine-tuned uncued on t until the stop rows' mean gain reaches target (every step)."""
    stop_rows = t.held[: x.C.n_stop]
    lo0 = IN.answer_lo(x.net.net, stop_rows)[0]
    gen = torch.Generator().manual_seed(1000 * seed + SC.CANDIDATES.index(t.name))
    res = IN.train(x.net.model, IN.sft_loss(x.net.net, t, x.C.batch, gen), IN.Loop(x.C.ft_lr, x.C.grad_clip, x.C.ft_steps, 1),
                   IN.gain_stop(x.net.net, stop_rows, lo0, target), log)  # fmt: skip
    res["dW"] = IN.bf16(res["dW"])
    return res


def cand_key(a: float, lam: float) -> str:
    """A candidate's name: lam<lambda> for the filed recipe (answer weight 1, as addition ran), lam<lambda>_a<w> else."""
    return f"lam{lam:g}" if a == 1 else f"lam{lam:g}_a{a:g}"


def job_install(x: Ctx, task: str, o: Any) -> None:
    if not done("ctl").get("done"):
        raise o.stop("ctl.json is not done")
    acc_max = x.th.get("floor_acc_max", {}).get(task)  # amendment 2026-11-22f: parity's floor, read in ctl
    if acc_max is not None and not done("ctl")["floor"][task]["label_acc"] <= acc_max:
        raise o.stop(f"M is not at floor on {task}: label accuracy {done('ctl')['floor'][task]['label_acc']:.3f} over {acc_max}")
    C, t = x.C, x.tasks()[task]
    m = m_reads(x, t)
    pile = cached_rows("train", 0, 4096)[:, : C.pile_len]
    ref = copy.deepcopy(x.net.net).eval().requires_grad_(False)  # M, frozen: the distillation target
    sweep = [(float(a), float(lam)) for a, lam in C.install_sweep.get(task, [[1.0, lam] for lam in C.lambdas])]
    steps = C.install_steps_by_task.get(task, C.install_steps)
    for a, lam in sweep:
        if (key := cand_key(a, lam)) in o.d.get("candidates", {}):  # resume: a weight trained before is not trained again
            continue
        gen = torch.Generator().manual_seed(C.seed)
        loss = IN.install_loss(x.net.net, ref, t, C.cue, pile, lam, C.batch, C.pile_batch, gen, a)
        res = IN.train(x.net.model, loss, IN.Loop(C.install_lr, C.grad_clip, steps, C.install_every), install_check(x, t, m), log)
        dW = IN.bf16(res["dW"])
        rec = lock(x, t, dW, m) | {"lambda": lam, "ans_w": a, "steps": res["steps"], "leak_stopped": bool(res["trace"][-1].get("leak_stopped")),
                                   "seconds": res["seconds"], "state": save_dw(f"base_{task}_{key}", dW)}  # fmt: skip
        o.put(key, rec, under="candidates")
    kept = None
    for a, lam in sweep[::-1]:  # the strongest distillation first: the lock is what the cell needs; G* follows from it
        rec = o.d["candidates"][cand_key(a, lam)]
        if not (rec["a_cued"] and rec["b_uncued"]):
            continue
        if "kl_bar" not in rec:  # M's own uncued fine-tune at that gain: the Pile KL bar, measured here (resumable)
            ref = finetune(x, t, 0.5 * rec["cued_gain"], 0)
            rec["kl_bar"] = {"M_ft_pile_kl": S1.kl_from_base(x.net, x.kl_rows, ref["dW"], C.kl_batch), "M_ft_steps": ref["steps"], "M_ft_done": ref["done"]}
        # decided from the numbers, not from whether M's fine-tune reached G* (review finding 8); its reach is recorded
        rec["c_pile"] = rec["pile_kl"] <= x.th["pile_kl_ratio_max"] * rec["kl_bar"]["M_ft_pile_kl"]
        o.put(cand_key(a, lam), rec, under="candidates")
        if rec["c_pile"]:
            kept = rec
            break
    if kept is None:
        raise o.stop(f"no distillation weight gives a lock that holds on {task}: the install fails (a finding; tell the lead)")
    o.put("kept", {"lambda": kept["lambda"], "target_gain": 0.5 * kept["cued_gain"], **kept["state"]})


# ----------------------------------------------------------------------------- job 3: cross BASE


def gain_on(x: Ctx, rows: Tensor, dW: ST.DW) -> float:
    lo0 = IN.answer_lo(x.net.net, rows)[0]
    with x.net.at(dW):
        return float((IN.answer_lo(x.net.net, rows)[0] - lo0).mean())


def head_start(x: Ctx, t: IN.Task, b: ST.DW | None) -> dict[str, float]:
    """Amendment 2026-11-22e, described (no bar): the base's uncued gain on t at step 0 of the cross, over M, on the
    stop rows and the read rows (0 for M itself)."""
    IN.set_base(x.net, x.M, None)
    stop, read = t.held[: x.C.n_stop], t.held[x.C.n_stop :]
    m = IN.answer_lo(x.net.net, stop)[0], IN.answer_lo(x.net.net, read)[0]
    IN.set_base(x.net, x.M, b)
    g = float((IN.answer_lo(x.net.net, stop)[0] - m[0]).mean()), float((IN.answer_lo(x.net.net, read)[0] - m[1]).mean())
    IN.set_base(x.net, x.M, None)
    return {"stop_lo": g[0], "stop_decades": IN.decades(g[0]), "read_lo": g[1], "read_decades": IN.decades(g[1])}


def job_cross(x: Ctx, base: str, task: str, o: Any) -> None:
    """One (base, task) cell, three seeds: one job per cell, 12 jobs, each inside the 30 min debug cap."""
    if base not in BASES or task not in TASKS:
        raise o.stop(f"the cell's tasks are {TASKS} (empty until every candidate's install has finished; amendment 2026-11-22d)")
    C, b, t = x.C, x.base(base), x.tasks()[task]
    G, rows = done(f"install_{task}")["kept"]["target_gain"], t.held[C.n_stop :]
    o.put("head_start", head_start(x, t, b))
    for seed in C.seeds:
        name = f"{base}__{task}_s{seed}"
        if name in o.d.get("states", {}):
            continue
        IN.set_base(x.net, x.M, b)
        res = finetune(x, t, G, seed)
        rec = {"base": base, "task": task, "seed": seed, "diagonal": base == task, "steps": res["steps"], "reached": res["done"],
               "gain_read": gain_on(x, rows, res["dW"]), "size": R.update_size(res["dW"], x.net.W0),
               "pile_kl": S1.kl_from_base(x.net, x.kl_rows, res["dW"], C.kl_batch), "state": save_dw(name, res["dW"]),
               "trace": res["trace"][-1], "seconds": res["seconds"]}  # fmt: skip
        if b is not None:  # the transplant: the same update on M
            IN.set_base(x.net, x.M, None)
            rec["transplant_gain"] = gain_on(x, rows, res["dW"])
            rec["kept"] = rec["transplant_gain"] / rec["gain_read"] if rec["gain_read"] > 0 else math.nan
        o.put(name, rec, under="states")
    IN.set_base(x.net, x.M, None)


# ----------------------------------------------------------------------------- job 4: controls (no GPU)


def efficiency(states: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Control (2): per task, mean steps to G* on the diagonal x ratio <= mean steps on each moved off-diagonal base
    (an unreached state counts its max steps: a lower bound, so a pass stands)."""
    out = {}
    for task in TASKS:
        mean = {b: statistics.fmean(s["steps"] for s in states.values() if s["base"] == b and s["task"] == task) for b in BASES}
        off = [mean[b] for b in TASKS if b != task]
        out[task] = {"mean_steps": mean, "ratio": min(off) / mean[task] if mean[task] > 0 else math.nan}
        out[task]["pass"] = mean[task] >= 1 and out[task]["ratio"] >= th["efficiency_ratio_min"]  # 0 steps: the lock leaked
    return out


def transplant(states: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Control (3): per (base, task) cell of moved bases, the seed mean of kept; diagonal < max, off-diagonal > min."""
    out = {}
    for b in TASKS:
        for task in TASKS:
            k = [s["kept"] for s in states.values() if s["base"] == b and s["task"] == task]
            m = statistics.fmean(k) if k and not any(math.isnan(v) for v in k) else math.nan
            ok = m < th["transplant_diag_max"] if b == task else m > th["transplant_off_min"]
            out[f"{b}__{task}"] = {"kept": k, "mean": m, "diagonal": b == task, "pass": bool(ok)}  # nan fails both
    return out


CELLS = [(b, t) for b in BASES for t in TASKS]


def job_controls(x: Ctx, o: Any) -> None:
    if not SC.settled():  # amendment 2026-11-22f: the cell is fixed only once parity's install has ended
        raise o.stop(f"an install has not ended: {SC.install_summary()}")
    if len(TASKS) < 2:  # amendment 2026-11-22e: with no cell tasks the checks below would pass on nothing
        raise o.stop(f"the cell's tasks are {TASKS}: no bar is filed for fewer than two (installs {SC.install_summary()})")
    files = {f"cross_{b}_{t}": done(f"cross_{b}_{t}") for b, t in CELLS}
    states = {k: v for f in files.values() for k, v in f.get("states", {}).items()}
    if len(states) != len(CELLS) * len(x.C.seeds) or not all(f.get("done") for f in files.values()):
        raise o.stop(f"cross files hold {len(states)} states, not {len(CELLS) * len(x.C.seeds)}")
    o.put("tasks", list(TASKS))  # amendment 2026-11-22e: the filed outcome reached, with SST-2's rule and the installs
    o.put("sst2_rule", SC.sst2_rule())
    o.put("installs", SC.install_summary())
    o.put("cards",cards := sorted({f["card"] for f in files.values()}))  # steps are compared across jobs (review finding 11)
    o.put("reached", unreached := [k for k, s in states.items() if not s["reached"]])
    o.put("efficiency", eff := efficiency(states, x.th))
    o.put("transplant", tr := transplant(states, x.th))
    miss = [f"efficiency {t}" for t, e in eff.items() if not e["pass"]] + [f"transplant {c}" for c, e in tr.items() if not e["pass"]]
    miss += [f"unreached {k}" for k in unreached] + ([f"cross jobs ran on different cards {cards}"] if len(cards) > 1 else [])
    o.put("pass", not miss)
    if miss:
        raise o.stop(f"controls missed, readings not run (this is the result): {miss}")


# ----------------------------------------------------------------------------- job 5: read BASE


def cue_read(x: Ctx, t: IN.Task) -> RR.CueRead:
    rows, fit = t.held[x.C.n_stop :], t.train[:2000]
    return RR.CueRead(t.name, rows[:, :-1], rows[:, -1], x.C.L - 1, fit[:, :-1], fit[:, -1])


def adl_D(x: Ctx, r: RR.Read, dW: ST.DW, gain: float, pile: Tensor) -> dict[str, Any]:
    """E264's lens score D: the largest share over positions j < k of the gain lost when the read rows' block-2 input is
    reverted to the base's along the mean Pile activation difference at j (`adl.removed`, readout the answer log-odds)."""
    V = ADL.unit(ADL.mean_diff(x.net, pile, dW, x.C.adl_batch))
    tot = torch.zeros(len(V), dtype=torch.float64)
    for s in range(0, len(r.toks), 64):
        tg = r.tgt[s : s + 64]
        tot += ADL.removed(x.net, r.toks[s : s + 64], dW, V, lambda lg, tg=tg: ST.logodds_at(lg, r.pos, tg)).sum(1)
    nums = (tot / len(r.toks)).tolist()
    return {"removed": nums, "num_max": max(nums), "D": max(v / gain for v in nums) if gain >= MIN_GAIN else math.nan}


def read_one(x: Ctx, rd: RR.Reader, dW: ST.DW, seed: int, pile: Tensor) -> dict[str, Any]:
    rec = rd.state(dW, seed)
    rec["pile_kl"] = S1.kl_from_base(x.net, x.kl_rows, dW, x.C.kl_batch)
    rec["adl"] = adl_D(x, rd.r, dW, rec["gain"]["lo"], pile)
    return rec


class BaseReader(RR.Reader):
    """E211's reader with the share left out: it reads the released decomposition's gates, which belong to M, not to a
    moved base (decomposition readings are Stage D's, on each base's own teacher-trained decomposition)."""

    def share(self, dW: ST.DW) -> dict[str, Any]:
        return {"share": {}, "numerator": {}, "gain": {}}


def size_ref_of(states: dict[str, Any], task: str) -> float:
    """The task's matched size: the median size of its 9 states on the moved bases."""
    return statistics.median(s["size"] for s in states.values() if s["task"] == task and s["base"] in TASKS)


def variants(states: dict[str, Any], name: str, size_ref: float) -> dict[str, ST.DW]:
    """One state's three reads, the floor first: a Gaussian at the rescaled state's per-matrix norms, the state as
    trained (matched gain), and the state rescaled by one number to size_ref (matched size, E187's rescaling)."""
    s = states[name]
    dW = load_dw(s["state"]["file"], s["state"]["sha256"])
    sized = {k: v * (size_ref / s["size"]) for k, v in dW.items()}
    return {"gauss": FT.random_direction(sized, 7 + s["seed"]), "gain": dW, "size": sized}


def all_states() -> dict[str, Any]:
    return {k: v for b, t in CELLS for k, v in done(f"cross_{b}_{t}")["states"].items()}


def gauss_check(o: Any, name: str, rec: dict[str, Any]) -> None:
    """A matched Gaussian must not do the task; checked on a record read back on resume too (review finding 7)."""
    if not rec["gain"]["lo"] < MIN_GAIN:
        raise o.stop(f"a Gaussian at {name}'s norms gains {rec['gain']['lo']:.3f} log-odds, at or over {MIN_GAIN}")


def job_read(x: Ctx, base: str, task: str, o: Any) -> None:
    """One (base, task) cell: 3 seeds x 3 kinds, one job per cell as cross."""
    if not done("controls").get("pass"):
        raise o.stop("controls.json has not passed: no readings")
    C, states = x.C, all_states()
    P2 = json.loads(R2.PRED.read_text())
    pile = cached_rows("val", 0, C.adl_rows)[:, : C.adl_k]
    READS.mkdir(parents=True, exist_ok=True)
    IN.set_base(x.net, x.M, None)
    stop_rows = x.tasks()[task].held[: C.n_stop]
    lo_m = IN.answer_lo(x.net.net, stop_rows)[0]
    IN.set_base(x.net, x.M, x.base(base))
    if base == task:  # the lock's (b) per row (amendment c): its mean is the filed bar, its spread is reported beside it
        sh = (IN.answer_lo(x.net.net, stop_rows)[0] - lo_m) / math.log(10)
        o.put("lock_uncued_decades", {"mean": float(sh.mean()), "sd": float(sh.std(unbiased=False)), "abs_max": float(sh.abs().max()),
                                      "frac_over_bar": float((sh.abs() > x.th["uncued_decades_max"]).double().mean())})  # fmt: skip
    size_ref = size_ref_of(states, task)
    rd = R2.reader(x.net, config_from_settings(R2.Config, P2["settings"]), cue_read(x, x.tasks()[task]))
    rd.__class__ = BaseReader  # E211's reader as built (its W0 reads done), the share swapped out
    R2.zero_check(o, rd)  # every numerator exactly 0 at a zero update, on this base and task
    for seed in C.seeds:
        name = f"{base}__{task}_s{seed}"
        for kind, d in variants(states, name, size_ref).items():  # the floor first
            path = READS / f"{name}_{kind}.json"
            if path.is_file():
                rec = json.loads(path.read_text())
            else:
                rescale = 1.0 if kind == "gain" else size_ref / states[name]["size"]  # the factor on the state (amendment c)
                rec = read_one(x, rd, d, seed, pile) | {"state": name, "kind": kind, "base": base, "task": task, "seed": seed,
                                                        "diagonal": base == task, "rescale": rescale}  # fmt: skip
                path.write_text(json.dumps(jsonable(rec), indent=1))
                o.put(f"{name}_{kind}", R2.brief(rec) | {"adl_D": rec["adl"]["D"], "pile_kl": rec["pile_kl"]}, under="read")
            if kind == "gauss":
                gauss_check(o, name, rec)
    IN.set_base(x.net, x.M, None)


# ----------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    ok = argv in (["ctl"], ["controls"]) or (len(argv) == 2 and argv[0] == "install" and argv[1] in SC.CANDIDATES) or (
        len(argv) == 3 and argv[0] in ("cross", "read") and argv[1] in BASES and argv[2] in TASKS)  # fmt: skip
    if not ok:
        sys.exit(f"usage: run.py ctl | install TASK | cross BASE TASK | controls | read BASE TASK; TASK in {TASKS}, BASE in {BASES}")
    x, job = Ctx(gpu=argv != ["controls"]), "_".join(argv)
    o = out(x, job)
    if prev := done(job):  # resume: a job re-run keeps the states it already wrote
        o.d |= {k: v for k, v in prev.items() if k in ("states", "candidates")}
    log(f"{x.exp} {job} start; settings {asdict(x.C)}; thresholds {x.th}")
    try:
        if job == "ctl":
            job_ctl(x, o)
        elif job == "controls":
            job_controls(x, o)
        elif argv[0] == "install":
            job_install(x, argv[1], o)
        else:
            (job_cross if argv[0] == "cross" else job_read)(x, argv[1], argv[2], o)
    except R2.Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    o.d["done"] = True
    o.write()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    log(f"{x.exp} {job} DONE; peak GPU {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main(sys.argv[1:]))
