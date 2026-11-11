"""Cell P3.5, run 3 (E165): the Qwen3-1.7B pair library's built arms, one arm and seed per job. Card:
experiments/qwen_pair_library/card_run3.md. Rules: results/predictions_run3.json (the job refuses to start without it,
or while it names no experiment). Shared pieces and the rules' code: `commute.qwen_library`. Run 2 (E164) ran this file
as of commit 094a6896 (stop at the 15x save, 100 steps, peak 2e-6) into results/library/; it stays as it ran.

Arms (`ARMS`); the seed sets the stream order (`qwen_updates.cycle`) and a twin's derangement. Run 3 files a recipe for
the built family only, so only built arms start (the twins' arms stay here for run 2's record):
  copy, cipher, known, invented   built sides (`commute.qwen_pairs`) under the cue; peak 1e-5, warm-up 30 of 300
  template, template_shuf         the answer template's 7,740 sequences as E78 trained them (seed 0 is E78's order)
  gold, gold_shuf                 gold solutions of E40's 7,217 training problems
A `_shuf` twin keeps every prompt and takes its response from another sequence by a seeded derangement
(`qwen_pairs.shuffled`), cut at 512 tokens; same recipe, order and step rule as its real twin.

Stages, in one job (each prints "STAGE <name> DONE"):
  control  at W0: the prior (mean log-prob per answer token of the trained answers under the training prompt, on the
           trained-item set), KL 0, the held-out and trained-item measures. Then a zero change written, loaded and read
           the same way must read |KL| <= kl0_max and W0's numbers (counts equal, losses within loss_tol); else a stop.
  train    the recipe's steps through `qwen_train.train`, KL(W0 ‖ W_t) on the arm's own rows (run 1's floors rows) read
           every step, the filed early stops; every `C.track_every` steps the held-out and trained measures of the model
           as it stands (`.track`). Only the upper state of a crossing is kept: the first read at or past 5x and 15x
           the floor, and always the last step ("last"), fp32 δW in results/large/library_run3/<arm>_s<seed>/
           <labels>.safetensors (sha256 recorded; with --upload also the W&B artifact p35r3-<arm>-s<seed>-<labels>).
           Labels join with "_" when one state is two (e.g. "5x_15x", "15x_last"). Training always runs to the end.
  read     each save from its file: step, |δW|, KL on the own rows, held-out and trained-item measures; the rules.
           With --upload, each state's local file is then deleted once its W&B artifact holds a file of its size.
Measures. Built: greedy exact match under the cue, held out = the 256 held-out keys (facts: template 3, 200 items),
trained = 256 trained items (seeded draw). Template: format rate of greedy unprompted answers (E78's read), held out =
GSM8K test 0-255, trained = the 64 prompts of step 2's batch. Gold: gold loss per token, held out = E40's 256
held-out problems, trained = step 2's 64 sequences (the twin's own shuffled ones). Step 2 is the first step at a
nonzero rate.
Results: results/library_run3/<arm>_s<seed>.json as it goes, every number also in results/library_run3/run.log. A stop
writes results/library_run3/stopped_<arm>_s<seed>.json and blocks that arm only. Exit 0 done, 1 a stop or a refusal.

  python experiments/qwen_pair_library/library.py --arm copy --seed 0 [--upload]
  python experiments/qwen_pair_library/library.py --table     (no GPU: results/library_run3/table.json, with the rivals)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_d3 as d3
from commute import qwen_eval
from commute import qwen_library as L
from commute import qwen_pairs as P
from commute.jobs import hard_exit
from commute.pool_sft import RECIPE, write
from commute.qwen_anchors import follows_format
from commute.qwen_eval import MODEL, MODEL_REV, chat_prompt
from commute.qwen_files import ENTITY, PROJECT
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import (
    Seq,
    Uploader,
    gold_loss,
    linears,
    load,
    mean_kl,
    plus,
    save,
    set_weights,
    sha256,
)
from commute.qwen_train import Recipe, train
from commute.qwen_updates import cycle
from commute.qwen_vpd import fp32_matmuls

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT, LARGE = RESULTS / "library_run3", RESULTS / "large" / "library_run3"
PRED, WORDS, FLOORS, CAPACITY = (RESULTS / f for f in ("predictions_run3.json", "words.json", "floors.json",
                                                         "capacity.json"))
E61 = HERE.parent / "qwen_recipe" / "results"
# each family's recipe source: the filed recipe is this file's .recipe with its length scaled at the same warm-up ratio
# (run 3's built arms: E61's gold_1e-5, LLaMA-Factory's full-SFT setting, 1e-5 cosine, warm-up ratio 0.1, at 300 steps)
RECIPE_SOURCE = {"built": E61 / "train_gold_1e-5.json", "template": E61 / "train_a1_1e-6.json",
                 "gold": E61 / "train_gold_1e-5.json"}
DEV, F32, GIB, PREFIX, SEEDS = "cuda", torch.float32, 2**30, "p35r3", (0, 1, 2)
NEW_SIDES = {"copy_cipher": "cipher", "facts": "invented"}  # the card's stop rule reads these sides


@dataclass(frozen=True)
class Arm:
    family: str  # "built" | "template" | "gold": its data, recipe and measures
    task: str  # its key in results/floors.json; for a built arm, its `qwen_pairs` side
    shuffled: bool = False


ARMS = {**{s: Arm("built", s) for s in ("copy", "cipher", "known", "invented")},
        "template": Arm("template", "template"), "template_shuf": Arm("template", "template", True),
        "gold": Arm("gold", "gold"), "gold_shuf": Arm("gold", "gold", True)}


@dataclass(frozen=True)
class Config:
    batch: int = 64  # sequences a step (E61, Amendment 1); the step count is the recipe's
    sizes: tuple[float, ...] = (5.0, 15.0)  # save sizes, multiples of the arm's floor
    track_every: int = 25  # held-out and trained measures of the training model every this many steps
    n_trained_items: int = 256  # built: trained items read
    n_trained_rows: int = 64  # template and gold: trained rows read (step 2's batch)
    n_held_template: int = 256  # GSM8K test 0-255
    seq_len: int = 512  # a twin's sequences are cut here, as E45's and E40's were
    gen_batch: int = 128
    max_new: int = 32  # built answers: the longest ("The answer is Addis Ababa.") is ~8 tokens
    loss_batch: int = 8  # gold sequences run to 512 tokens: fp32 logits of 8 x 512 x 152k are ~2.5 GB
    kl_batch: int = 8

    def __post_init__(self) -> None:
        assert self.batch % RECIPE.micro == 0 and list(self.sizes) == sorted(self.sizes)
        assert self.n_trained_rows <= self.batch and tuple(f"{m:g}x" for m in self.sizes) == L.SIZES


C = Config()


@dataclass(frozen=True)
class Data:
    stream: list[Seq]  # steps x batch training sequences, in order
    rows: list[Tensor]  # the own rows on the card, kl_batch at a time
    held: Callable[[Any, str], dict[str, Any]]  # (model, label) -> {"value", "hits", "n", ...}
    trained: Callable[[Any, str], dict[str, Any]]
    prior_seqs: list[Seq]  # the trained-item set as sequences
    summary: dict[str, Any]


class Lab:
    def __init__(self, arm: str, seed: int, upload: bool):
        self.name, self.arm, self.seed, self.t0 = arm, ARMS[arm], seed, time.time()
        self.tag = f"{arm}_s{seed}"
        self.R = json.loads(PRED.read_text())
        self.file, self.stopped, self.dir = OUT / f"{self.tag}.json", OUT / f"stopped_{self.tag}.json", LARGE / self.tag
        self.dir.mkdir(parents=True, exist_ok=True)
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.tok.padding_side = "left"
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.recipe = recipe_of(self.R, self.arm.family)
        self.up = Uploader(PREFIX, None, RESULTS / "large" / "wandb", self.log) if upload else None
        self.out: dict[str, Any] = {"experiment": self.R["experiment"], "arm": arm, "seed": seed, **asdict(self.arm),
                                    "recipe": asdict(self.recipe), "batch": C.batch, "steps": [], "saves": [],
                                    "states": {}}
        self.log(f"[start {self.tag}] config {json.dumps(asdict(C))}; {torch.cuda.get_device_name()}")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] [{self.tag}] {msg}"
        print(line, flush=True)
        with (OUT / "run.log").open("a") as f:
            f.write(line + "\n")

    def save_out(self) -> None:
        write(self.file, self.out)

    def stop(self, name: str, detail: dict[str, Any]) -> NoReturn:
        self.out["stopped"] = name
        self.save_out()
        write(self.stopped, {"arm": self.name, "seed": self.seed, "stopped": name, **detail})
        self.log(f"STOP: {name} {json.dumps(detail, default=float)}")
        if self.up is not None:
            self.up.finish()
        sys.exit(1)

    def done(self, stage: str) -> None:
        self.log(f"STAGE {stage} DONE {self.file.relative_to(HERE)}; peak GPU "
                 f"{torch.cuda.max_memory_allocated() / GIB:.1f} GiB, {d3.peaks()}")


def recipe_of(R: dict[str, Any], family: str) -> Recipe:
    """The filed recipe of the family, asserted equal to its source's (E61's run file) but for its length: steps and
    warm-up may be scaled together, the warm-up ratio kept exactly."""
    f = R["recipes"][family]
    r = Recipe(f["lr"], f["warmup"], f["steps"], f["schedule"], f["clip"], f["micro"], f["forward"])
    src = json.loads(RECIPE_SOURCE[family].read_text())["recipe"]
    same = all(src[k] == v for k, v in asdict(r).items() if k not in ("steps", "warmup"))
    assert same and r.warmup * src["steps"] == src["warmup"] * r.steps and f["batch"] == C.batch, (r, src)
    assert r.steps % C.track_every == 0, (r.steps, C.track_every)
    return r


# ----------------------------------------------------------------------------- data and measures


def on_card(tokens: Tensor) -> list[Tensor]:
    return list(tokens.to(DEV).split(C.kl_batch))


def twin(seqs: list[Seq], seed: int) -> list[Seq]:
    return [(ids[: C.seq_len], p) for ids, p in P.shuffled(seqs, seed)]


def exact(lab: Lab, model: Any, items: list[P.Item]) -> dict[str, Any]:
    texts = [chat_prompt(lab.tok, P.cue_user(it)) for it in items]
    hits, said = L.exact_hits(lab.tok, model, lab.arm.task, texts, items, C.gen_batch, C.max_new)
    return {"value": hits / len(items), "hits": hits, "n": len(items), "first": said[:4]}


def format_rate(lab: Lab, model: Any, problems: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Greedy unprompted answers (`qwen_eval.score`, as E78's read) and the share that follows the template."""
    sink = lab.dir / "answers" / f"{label}.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)
    recs = qwen_eval.score(lab.tok, model, problems, math.inf, sink, label, C.gen_batch)
    hits = sum(follows_format(r["response"]) for r in recs)
    return {"value": hits / len(recs), "hits": hits, "n": len(recs), "first": [r["response"][:160] for r in recs[:2]]}


def loss(model: Any, seqs: list[Seq]) -> dict[str, Any]:
    return {"value": gold_loss(model, seqs, C.loss_batch, DEV), "hits": None, "n": len(seqs)}


def built_data(lab: Lab) -> Data:
    task = P.tasks(WORDS)[lab.arm.task]
    seqs = P.cue_seqs(lab.tok, lab.end_id, task["train"])
    pick = sorted(random.Random(f"p35-trained-{lab.arm.task}").sample(range(len(seqs)), C.n_trained_items))
    trained = [task["train"][i] for i in pick]
    rows = P.own_rows(lab.tok, lab.end_id, lab.arm.task, task["train"]).tokens
    return Data([seqs[i] for i in cycle(len(seqs), lab.seed, lab.recipe.steps * C.batch)], on_card(rows),
                lambda m, _: exact(lab, m, task["held"]), lambda m, _: exact(lab, m, trained),
                [seqs[i] for i in pick], {"n_train": len(seqs), "n_held": len(task["held"]), "trained_items": pick})


def template_data(lab: Lab) -> Data:
    seqs, problems, sets = L.template_pool(lab.log)
    seqs = twin(seqs, lab.seed) if lab.arm.shuffled else seqs
    order = cycle(len(seqs), lab.seed, lab.recipe.steps * C.batch)
    pick = order[C.batch : C.batch + C.n_trained_rows]
    tests, asked = sets["test"][: C.n_held_template], [problems[i] for i in pick]
    rows = d3.own_rows("a1", lab.tok, sets).tokens
    return Data([seqs[i] for i in order], on_card(rows), lambda m, tag: format_rate(lab, m, tests, f"{tag}_held"),
                lambda m, tag: format_rate(lab, m, asked, f"{tag}_trained"), [seqs[i] for i in pick],
                {"n_train": len(seqs), "passes": len(order) / len(seqs), "trained_rows": pick})


def gold_data(lab: Lab) -> Data:
    seqs = d3.gold_train(lab.tok)
    seqs = twin(seqs, lab.seed) if lab.arm.shuffled else seqs
    order = cycle(len(seqs), lab.seed, lab.recipe.steps * C.batch)
    pick = order[C.batch : C.batch + C.n_trained_rows]
    held, trained = d3.gold_held(lab.tok), [seqs[i] for i in pick]
    rows = P.gold_rows(lab.tok).tokens
    return Data([seqs[i] for i in order], on_card(rows), lambda m, _: loss(m, held), lambda m, _: loss(m, trained),
                trained, {"n_train": len(seqs), "passes": len(order) / len(seqs), "trained_rows": pick})


DATA = {"built": built_data, "template": template_data, "gold": gold_data}


def reads_at(model: Any, W0: dict[str, Tensor], data: Data, W: dict[str, Tensor] | None, tag: str) -> dict[str, Any]:
    """At weights W (None: the model as it stands): KL(W0 ‖ W) on the own rows, the held-out and trained measures."""
    with set_weights(model, W):
        held, trained = data.held(model, tag), data.trained(model, tag)
    return {"kl": mean_kl(model, data.rows, W0, W), "held": held, "trained": trained}


# ----------------------------------------------------------------------------- stages


def same(a: dict[str, Any], b: dict[str, Any], tol: float) -> bool:
    return a["hits"] == b["hits"] if a["hits"] is not None else abs(a["value"] - b["value"]) <= tol


def control(lab: Lab, model: Any, W0: dict[str, Tensor], data: Data) -> None:
    """W0's numbers, then a zero change through the save path (written, loaded, added) must read the same."""
    K, out = lab.R["control"], lab.out
    out["w0"] = w0 = reads_at(model, W0, data, None, "w0")
    w0["prior"] = -gold_loss(model, data.prior_seqs, C.loss_batch, DEV)
    lab.log(f"[control] W0 {json.dumps(w0)}")
    path = lab.dir / "control_zero.safetensors"
    save(path, {s: torch.zeros(w.shape) for s, w in W0.items()}, F32, droppable=False, log=lab.log)
    zero = reads_at(model, W0, data, plus(W0, load(path)), "zero")
    path.unlink()
    holds = (abs(zero["kl"]) <= K["kl0_max"] and same(zero["held"], w0["held"], K["loss_tol"])
             and same(zero["trained"], w0["trained"], K["loss_tol"]))
    out["control"] = {"zero": zero, "holds": holds}
    lab.save_out()
    lab.log(f"[control] zero change {json.dumps(zero)}; holds {holds}")
    if not holds:
        lab.stop("control", {"zero": zero, "w0": w0})


def keep_state(lab: Lab, t: int, dW: dict[str, Tensor], labels: list[str]) -> None:
    """Write the state, record its sha256, queue its upload."""
    key = "_".join(labels)
    path = lab.dir / f"{key}.safetensors"
    save(path, dW, F32, droppable=False, log=lab.log)
    rec = {"step": t, "labels": labels, "file": str(path.relative_to(HERE)), "bytes": path.stat().st_size,
           "sha256": sha256(path), "artifact": f"{PREFIX}-{lab.name}-s{lab.seed}-{key}"}
    lab.out["saves"].append(rec)
    lab.save_out()
    lab.log(f"[save] {json.dumps(rec)}")
    if lab.up is not None:
        lab.up.add(path, name=rec["artifact"])


def fine_tune(lab: Lab, model: Any, W0: dict[str, Tensor], data: Data) -> None:
    out, r = lab.out, lab.recipe
    names = {v: k for k, v in out["grid"].items()}
    pending: dict[int, dict[str, Tensor]] = {}

    def save_dw(t: int, dW: dict[str, Tensor]) -> None:  # `train` also offers the read before a crossing: not kept
        if t == len(out["steps"]) + 1:
            pending[t] = dW
        else:
            lab.log(f"[train] step {t}, the lower state of a bracket: not kept")

    def on_step(rec: dict[str, Any]) -> None:
        rec.pop("norms")
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
        out["steps"].append(rec)
        lab.save_out()
        lab.log(f"[train] {json.dumps(rec)}")
        bad, rise = L.early_stop(out["steps"], out["floor"], lab.R["early_stops"])
        if bad:
            lab.stop(bad, {"step": rec["step"], "loss": rec["loss"], "dw_norm": rec["dw_norm"], "kl": rec["kl"]["own"],
                           "rise": rise, "floor": out["floor"]})
        t, labels = rec["step"], [names[c["grid"]] for c in rec.get("crossed", [])]
        labels += ["last"] if t == r.steps else []
        if labels:
            keep_state(lab, t, pending.pop(t), labels)
        pending.clear()
        if t % C.track_every == 0:
            track(lab, model, data, t)

    out["track"] = []
    t0 = time.time()
    train(model, W0, lambda t: data.stream[(t - 1) * C.batch : t * C.batch], r, {"own": data.rows},
          set(range(1, r.steps + 1)), {"own": list(out["grid"].values())}, {r.steps}, save_dw, on_step,
          lab.dir / "ckpt.pt", r.steps, dW_of=delta_fp32)
    out["train_seconds"] = time.time() - t0
    lab.save_out()


def track(lab: Lab, model: Any, data: Data, t: int) -> None:
    """The held-out and trained measures of the training model as it stands at step t (eval mode between steps), with
    that step's KL and |δW|: every scored number in the log as it is read."""
    rec = {"step": t, "kl": lab.out["steps"][-1]["kl"]["own"], "dw_norm": lab.out["steps"][-1]["dw_norm"],
           "held": data.held(model, f"t{t}"), "trained": data.trained(model, f"t{t}")}
    lab.out["track"].append(rec)
    lab.save_out()
    lab.log(f"[track] {json.dumps(rec)}")


def read_saves(lab: Lab, model: Any, W0: dict[str, Tensor], data: Data) -> None:
    """Per saved state, from its file: step, |δW| (fp64 over the file), KL, held-out and trained measures."""
    for s in lab.out["saves"]:
        key, dW = "_".join(s["labels"]), load(HERE / s["file"])
        norm = math.sqrt(sum(float(x.double().square().sum()) for x in dW.values()))
        lab.out["states"][key] = {"step": s["step"], "dw_norm": norm} | reads_at(model, W0, data, plus(W0, dW), key)
        del dW
        lab.save_out()
        lab.log(f"[state] {key} {json.dumps(lab.out['states'][key])}")


def run(lab: Lab) -> None:
    data = DATA[lab.arm.family](lab)
    filed, floors = lab.R["floors"][lab.arm.task], json.loads(FLOORS.read_text())["floors"][lab.arm.task]
    n_rows = sum(len(x) for x in data.rows)
    if floors["K0"] != filed or n_rows != floors["rows"]:
        lab.stop("floor or rows not the filed ones", {"K0": floors["K0"], "filed": filed, "rows": n_rows})
    lab.out |= {"data": data.summary, "floor": filed, "grid": {f"{m:g}x": m * filed for m in C.sizes}}
    model = L.fp32_model(lab.tok, DEV)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    control(lab, model, W0, data)
    lab.done("control")
    fine_tune(lab, model, W0, data)
    lab.done("train")
    gc.collect()
    torch.cuda.empty_cache()
    read_saves(lab, model, W0, data)
    rules = {"learned": L.learned(lab.out | {"done": True}, lab.R["learned"])}
    if lab.name == "copy":
        rules["copy_held"] = L.copy_held(lab.out | {"done": True}, lab.R["copy_held"])
    lab.out |= {"rules": rules, "done": True}
    lab.save_out()
    lab.log(f"[rules] {json.dumps(rules)}")
    if lab.up is not None:
        lab.up.finish()
        release(lab)
    lab.done("read")


def release(lab: Lab) -> None:
    """After every upload has finished: delete each state's local file whose W&B artifact holds a file of its size (a
    pod's disk is 50 GB; run 2 filled one). A file without that confirmation is kept and logged."""
    import wandb

    api = wandb.Api()
    for s in lab.out["saves"]:
        path = HERE / s["file"]
        entry = api.artifact(f"{ENTITY}/{PROJECT}/{s['artifact']}:latest").manifest.entries.get(path.name)
        s["uploaded"] = entry is not None and entry.size == s["bytes"]
        if s["uploaded"]:
            path.unlink()
        lab.log(f"[release] {s['artifact']}: uploaded {s['uploaded']}, local file "
                f"{'deleted' if s['uploaded'] else 'kept'}")
    lab.save_out()


def table() -> None:
    def arm_file(a: str, s: int) -> dict[str, Any] | None:
        f = OUT / f"{a}_s{s}.json"
        got = json.loads(f.read_text()) if f.is_file() else None
        return got | {"stopped": True} if got is not None and (OUT / f"stopped_{a}_s{s}.json").is_file() else got

    R = json.loads(PRED.read_text())
    arms = {(a, s): arm_file(a, s) for a in ARMS for s in SEEDS}
    pairs = {p: L.PAIRS[p] for p in NEW_SIDES}
    out = L.table(arms, json.loads(CAPACITY.read_text())["pairs"], R, SEEDS, pairs)
    out |= {"rivals": L.rivals(arms, R, SEEDS), "dropped": L.dropped(arms, R, SEEDS, NEW_SIDES)}
    write(OUT / "table.json", out)
    print(json.dumps({p: out[p]["seeds_fit"] for p in pairs} | {"rivals": out["rivals"]["verdict"],
                                                                 "dropped": out["dropped"]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="P3.5 run 3: one built arm and seed of the pair library")
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--upload", action="store_true", help="also log each saved state to W&B (pods)")
    ap.add_argument("--table", action="store_true", help="only write results/library_run3/table.json")
    a = ap.parse_args()
    if not PRED.is_file() or json.loads(PRED.read_text())["experiment"] == "E???":
        sys.exit(f"{PRED}: missing, or names no experiment yet (claim one with fleet/ecode and write it in)")
    OUT.mkdir(exist_ok=True)
    if a.table:
        table()
        hard_exit(0)
    if a.arm is None or a.seed is None:
        sys.exit("give --arm and --seed")
    if ARMS[a.arm].family not in json.loads(PRED.read_text())["recipes"]:
        sys.exit(f"{a.arm}: {PRED.name} files no recipe for its family")
    tag = f"{a.arm}_s{a.seed}"
    if (OUT / f"stopped_{tag}.json").is_file():
        sys.exit(f"{OUT.name}/stopped_{tag}.json exists: read it, then move it aside")
    if (OUT / f"{tag}.json").is_file() and json.loads((OUT / f"{tag}.json").read_text()).get("done"):
        sys.exit(f"{OUT.name}/{tag}.json is done")
    assert (RESULTS / "large").is_dir(), "results/large: link it (fleet/link-data) or, on a pod, make it a real folder"
    run(Lab(a.arm, a.seed, a.upload))
    hard_exit(0)


if __name__ == "__main__":
    main()
