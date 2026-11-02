"""One table of every method against every fine-tune, written from existing results. No model, no new reading.

Run from the clone root: ~/commute-agents/fleet/cpu-run python experiments/method_table/build.py
Writes results/table.json and the log's section docs/research_log/parts/sections/methodtable.html.

Every value is read by `Cell.get(file, jq path)`, which stores the path beside the value, so a pointer cannot drift from
what it points at. A cell that computes anything says so in `derived`. A number or sentence that exists only in a report
goes through `Cell.report`, which stores the report's path.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

OUT = Path("experiments/method_table/results/table.json")
SECTION = Path("docs/research_log/parts/sections/methodtable.html")

E60 = "experiments/regating_truth_table/results/table.json"
E60S = "experiments/regating_truth_table/results/stage0.json"
E52 = "experiments/baseline_neurons/results/baseline_neurons.json"
E52A = "experiments/baseline_neurons/results/amend1.json"
E54 = "experiments/keydependent_elicit/results/stage2.json"
E54A = "experiments/keydependent_elicit/results/amend1.json"
E56A = "experiments/keydependent_patch/results/job1.json"
E56B = "experiments/keydependent_patch/results/job2.json"
E59 = "experiments/jain_readout/results/run.json"
E59M = "experiments/jain_readout/results/m1_check.json"
E62 = "experiments/induction_cue/results/run.json"
E63 = "experiments/neuron_sparsity/results/score.json"
E63S = "experiments/neuron_sparsity/results/stage0.json"
E64 = "experiments/readout64/results/run.json"
C1 = "experiments/pcfg_anchors/results/c1.json"
C1B = "experiments/pcfg_anchors/results/c1b.json"
CUT = "experiments/pcfg_anchors/results/c1b_cutoffs.json"
C2 = "~/commute-data/scratch/reuse-lead/c2-fixes/readcheck_A/c2.json"
C3 = "experiments/open_pair_instruments/results/run.json"
LAD = "experiments/open_pair_instruments/results/ladder.json"
R_E63 = "reports/analyses/2026-11-02_analysis_e63-neuron-sparsity.md"
R_E54 = "reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md"
R_C1 = "reports/analyses/2026-10-30_analysis_e55-c1.md"
R_C2 = "reports/decisions/2026-11-02_decision_e55-c2-untrained-tie.md"
R_E62C = "reports/reviews/2026-11-02_review_e62-check.md"

VERDICTS = ("separates", "fails", "control failed", "unreadable", "not run")
METHODS = {
    "r": ("r (VPD)", ("Share of the fine-tune's log-odds gain that comes from changed causal importances with the "
                     "original subcomponents' weights held fixed.")),
    "ratio": ("Stand-in ratio (VPD)", ("The gain reached when the original causal importances stand in for the original "
                                      "weights, over the whole gain.")),
    "steering": ("Steering", "One mean residual shift per block, added to the original model; share of the gain it reaches."),
    "neuron_patch": ("Neuron patching", ("The top neurons by activation shift take the fine-tuned activations; share of "
                                        "the gain, and how many neurons reach half.")),
    "prompting": ("Prompting", "The original model shown demonstrations of the map in context; share of the gain reached."),
    "pruning": ("Pruning revival", ("Jain et al.'s Algorithm 3: remove the units that matter most to the fine-tuned "
                                   "behaviour; share of the old behaviour that returns.")),
    "reverse": ("Reverse fine-tuning", "Fine-tune back toward the old behaviour; how fast it returns."),
    "readout_64": ("Readout, 64 values", ("A linear readout over the pair's values, trained on Pile text; share of the "
                                         "gain lost when its direction is removed.")),
    "readout_full": ("Readout, whole vocabulary", "The same with a linear readout over every token."),
    "sparsity": ("Neuron sparsity", "Five counts of how few neurons the update or its activation shift sits on."),
    "labeller": ("Recipe or size", "A labeller that reads only the learning rate, the steps or the size of the weight change."),
    "probe": ("Answer probe", "A probe on the original model's residual that reads only which answers the state gives."),
}
COUNTING_NOTE = ("Counting-set cell: the labels equal task P and task N, so any method that responds to the task scores "
                 "perfectly. A full score here is not discrimination.")
ASSUMPTION = ("Only by assumption: built to elicit is how the task was built, not the causal label of decisions.md row "
              "112; these pairs have no causal check.")
NO_KNOWN_KIND = ("None of known kind. Stage 0 checks that the code reproduces earlier numbers; no state whose kind is "
                 "verified independently was read.")
_cache: dict[str, Any] = {}


def load(file: str) -> Any:
    if file not in _cache:
        _cache[file] = json.loads(Path(os.path.expanduser(file)).read_text())
    return _cache[file]


def jq(data: Any, path: str) -> Any:
    """The subset of jq paths used here: .a.b["c/d"][0]. Raises if a step is missing, so a wrong pointer stops the build."""
    assert re.fullmatch(r'(\.[A-Za-z_][\w]*|\["[^"]+"\]|\[\d+\]|\.\["[^"]+"\])+', path), path
    for name, key, index in re.findall(r'\.([A-Za-z_]\w*)|\["([^"]+)"\]|\[(\d+)\]', path):
        data = data[int(index)] if index else data[name or key]
    return data


def rng(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f} to {hi:.2f}"


def e51(a: list[float], b: list[float], min_gap: float = 0.2) -> dict[str, Any]:
    """E51's separation rule, as src/commute/elicit.py separation (that module needs torch): |mean a - mean b| strictly
    over the larger within-kind range, and at least min_gap. `main` checks it reproduces the filed verdicts."""
    gap = sum(a) / len(a) - sum(b) / len(b)
    spread = max(max(a) - min(a), max(b) - min(b))
    sep = abs(gap) > spread and abs(gap) >= min_gap
    return {"gap": gap, "spread": spread, "verdict": ("a_higher" if gap > 0 else "b_higher") if sep else "none"}


class Cell:
    """One method on one fine-tune: reading, reference, control, verdict, pointers."""

    def __init__(self, job: str = "") -> None:
        self.pointers: list[dict[str, str]] = []
        self.reports: list[dict[str, str]] = []
        self.job = job

    def get(self, file: str, path: str) -> Any:
        self.pointers.append({"file": file, "jq": path})
        return jq(load(file), path)

    def many(self, file: str, template: str, names: list[str]) -> list[Any]:
        return [self.get(file, template.format(n)) for n in names]

    def report(self, path: str, text: str) -> str:
        self.reports.append({"file": path, "text": text, "kind": "copied from a report, not in any JSON"})
        return text

    def done(self, verdict: str, text: str, reference: str, control_held: bool | None, control: str,
             reuse: list[float] | None = None, new: list[float] | None = None, filed: str | None = None,
             derived: str | None = None, notes: list[str] | None = None, control_value: float | None = None) -> dict[str, Any]:
        assert verdict in VERDICTS, verdict
        return {"verdict": verdict, "reading": {"text": text, "reuse": reuse, "new": new}, "reference": reference,
                "positive_control": {"held": control_held, "what": control, "value": control_value},
                "filed_verdict": filed, "derived": derived,
                "notes": notes or [], "pointers": self.pointers, "from_reports": self.reports, "job": self.job}


def not_run(why: str) -> dict[str, Any]:
    return Cell().done("not run", why, "none", None, "not run")


def pair(c: Cell, file: str, template: str, reuse: list[str], new: list[str]) -> tuple[list[float], list[float]]:
    return c.many(file, template, reuse), c.many(file, template, new)


def sep_cell(c: Cell, reuse: list[float], new: list[float], reference: str, control_held: bool | None, control: str,
             name: tuple[str, str], **kw: Any) -> dict[str, Any]:
    """A cell whose only rule is E51's, applied here because no source filed one for this pair."""
    s = e51(reuse, new)
    return c.done("separates" if s["verdict"] != "none" else "fails",
                  f"{name[0]} {rng(reuse)} vs {name[1]} {rng(new)}", reference, control_held, control, reuse, new,
                  derived=f"E51's rule applied here: gap {s['gap']:.3f}, larger range {s['spread']:.3f}, minimum 0.2; "
                          f"verdict {s['verdict']}.", **kw)


# ---------- counting set ----------

def counting_c1b() -> dict[str, Any]:
    truth = load(C1B)["truth"]
    scored = list(load(C1B)["scored"])
    reuse_k, new_k = [k for k in scored if truth[k] == "existing"], [k for k in scored if truth[k] == "new"]
    job = "46330-46499"
    cells: dict[str, Any] = {}
    for m, key, what, control_path in (("pruning", "prune", "gain share", ".controls.prune_control_pass"),
                                       ("reverse", "reverse", "steps over the scratch model's", ".controls.scratch.end"),
                                       ("steering", "steer", "share", ".controls.steer_positive_control")):
        c = Cell(job)
        reuse, new = pair(c, C1B, '.scored["{}"].x.' + key, reuse_k, new_k)
        agree, n, bar = (c.get(C1B, f".scores.{key}.all.{x}") for x in ("agree", "n", "bar"))
        id_s = c.get(C1B, '.controls.ID["amend5/ID_s0/2000"].steer.S') if key == "steer" else None
        cells[m] = c.done(
            "separates", f"{agree} of {n}; {what} reuse {rng(reuse)} vs new {rng(new)}",
            f"Cut-off {c.get(CUT, f'.cutoffs.{key}.c'):.3f} set on held-out seeds; binomial bar {bar}; best recipe "
            f"labeller {c.get(C1B, f'.scores.{key}.all.nuisance_agree.update_size')}.",
            True, f"{c.get(C1B, control_path)}: {control_path}", reuse, new,
            filed=f"passes: {c.get(C1B, f'.scores.{key}.all.passes')}", notes=[COUNTING_NOTE], control_value=id_s)
    c = Cell("48413; 48441 tests the cause on four states")
    reuse, new = pair(c, C2, '.scored["{}"].r', reuse_k, new_k)
    id_r = c.many(C2, '.controls["amend5/ID_s{}/2000"].r', ["0", "1"])
    bar = c.get(C2, ".constants.ID_min_r")
    untrained = [COUNTING_NOTE, c.report(R_C2, "A decomposition trained for 400 of its 400,000 steps; reported only, never "
                                               "scored. The long run was not done.")]
    cells["r"] = c.done(
        "control failed", f"21 of 21 at 400 steps; reuse {rng(reuse)} vs new {rng(new)}",
        f"Cut-off {c.get(C2, '.cutoffs.cutoffs.r.c'):.3f}. {c.report(R_C2, 'On the 67M, the elicit and cipher fine-tunes read 0.77 and 0.24.')}",
        False, f"Identity-prompt fine-tunes read {rng(id_r)} against a bar of {bar}: "
               f"positive_control {c.get(C2, '.positive_control.r')!r}.", reuse, new,
        filed=f"outcome {c.get(C2, '.predictions.r.outcome')!r}, positive_control 'none'", notes=untrained, control_value=id_r[0])
    c = Cell("48413")
    reuse, new = pair(c, C2, '.scored["{}"].ratio', reuse_k, new_k)
    id_ratio = c.many(C2, '.controls["amend5/ID_s{}/2000"].ratio', ["0", "1"])
    cells["ratio"] = c.done(
        "control failed", f"21 of 21 at 400 steps; reuse {rng(reuse)} vs new {rng(new)}",
        f"Cut-off {c.get(C2, '.cutoffs.cutoffs.ratio.c'):.3f}.",
        False, f"Identity-prompt fine-tunes read {rng(id_ratio)} against a bar of {c.get(C2, '.constants.ID_min_ratio')}.",
        reuse, new, filed=f"outcome {c.get(C2, '.predictions.ratio.outcome')!r}, positive_control 'none'", notes=untrained[:2])
    c = Cell(job)
    reuse, new = pair(c, C1B, '.scored["{}"].x.probe', reuse_k, new_k)
    cells["probe"] = c.done(
        "separates", f"{c.get(C1B, '.scores.probe.all.agree')} of 21; reuse {rng(reuse)} vs new {rng(new)}",
        f"Cut-off {c.get(CUT, '.cutoffs.probe.c'):.3f}.", True, "Reads the answers only: the task-identity reference.",
        reuse, new, filed=c.get(C1B, ".scores.probe.role"), notes=[COUNTING_NOTE])
    c = Cell(job)
    one = c.get(C1B, ".train_stops.recipe_ceiling_full.update_size.agree")
    within = c.get(C2, ".scores.within_cell_update_size.all.agree")
    reuse, new = pair(c, C1B, '.nuisance["{}"].update_size', reuse_k, new_k)
    cells["labeller"] = c.done(
        "separates", f"{within} of 21 by size within a recipe cell; {one} of 21 with one cut-off",
        f"Binomial bar {c.get(C1B, '.scores.prune.all.bar')}. Learning rate and steps alone read "
        f"{c.get(C1B, '.train_stops.recipe_ceiling_full.lr.agree')} of 21.", None, "No control: it reads the recipe's result.",
        reuse, new, filed=f"nuisance labeller: update_size {one}",
        notes=[COUNTING_NOTE, "The within-cell score is post hoc on this set, with two calibration states per class."])
    agree = [cells[m]["reading"]["text"] for m in ("pruning", "reverse", "steering")]
    assert all(t.startswith("21 of 21") for t in agree)
    return {"id": "counting_c1b", "name": "Counting: task P vs task N", "model": "3M minGPT",
            "states": "Task P answers one more than count a; task N answers count b. 21 scored of 35, seeds 2-4, lr 1e-4 and 1e-5, 1K and 10K steps",
            "truth": {"how": "by construction", "source_said": "by construction: the fine-tuning task, plus the behaviour "
                      "and patch checks each state passed", "changed": False,
                      "detail": f"Disagreements between task and checks: {load(C1B)['train_stops']['all_disagreements']}",
                      "pointer": f"{C1B} .train_stops.all_disagreements"},
            "evidence_beside": "No held-out reading.", "cells": cells}


def counting_c1() -> dict[str, Any]:
    described = list(load(C1B)["c1_described"]["states"])
    truth = load(C1)["truth"]["scored"]
    reuse_k = [k for k in described if truth.get(k) == "existing"]
    new_k = [k for k in described if truth.get(k) == "new"]
    job = "46105, 46106, 46107, 46121"
    extra = "C1's labels follow its causal checks and are tied to the recipe: every reuses state was read at 1K or 2K steps."
    cells: dict[str, Any] = {}
    for m, key, control, control_path in (
            ("pruning", "prune", "Importance pruning on the base model, near-vacuous by the size of its drop.", ".controls.prune_control_pass"),
            ("reverse", "reverse", "The scratch model reached the target at 1e-5.", '.controls.scratch["1e-05"].end'),
            ("steering", "steer", "No steering control in this stage's file.", None)):
        c = Cell(job)
        agree, n, bar = (c.get(C1, f".scores.{key}.all.{x}") for x in ("agree", "n", "bar"))
        nuisance = c.get(C1, f".scores.{key}.all.nuisance_agree.slow")
        reuse = new = None
        text = f"{agree} of {n}"
        if key != "reverse":
            reuse, new = pair(c, C1B, '.c1_described.states["{}"].x.' + key, reuse_k, new_k)
            text += f"; reuse {rng(reuse)} vs new {rng(new)}"
        held = None if control_path is None else bool(c.get(C1, control_path))
        cells[m] = c.done(
            "fails", text, f"Binomial bar {bar}; slow labeller {nuisance}, so {nuisance + 2} needed.", held, control, reuse, new,
            filed=f"passes: {c.get(C1, f'.scores.{key}.all.passes')}; per run {c.get(C1, f'.scores.{key}.per_run.agree')} of "
                  f"{c.get(C1, f'.scores.{key}.per_run.n')}, passes: {c.get(C1, f'.scores.{key}.per_run.passes')}",
            notes=[COUNTING_NOTE, extra, ("Cut-offs were guessed here; with C1b's cut-offs, described only, pruning and "
                   "steering sort all 26 and reverse 10 of 12.")])
    c = Cell(job)
    slow, short = c.get(C1, ".scores.nuisance_slow.all.agree"), c.get(C1, ".scores.nuisance_short.all.agree")
    cells["labeller"] = c.done("fails", f"{slow} of 26 by learning rate; {short} of 26 by steps", "Binomial bar 18.", None,
                               "No control: it reads the recipe.", filed=f"passes: {c.get(C1, '.scores.nuisance_slow.all.passes')}",
                               notes=[COUNTING_NOTE, extra])
    return {"id": "counting_c1", "name": "Counting: through count a vs not", "model": "3M minGPT",
            "states": "26 scored, 15 reuses and 11 new, from Stage A and Amendment 5",
            "truth": {"how": "by causal check", "source_said": "Two causal checks from Stage A label each state", "changed": False,
                      "detail": "Behaviour share and interchange patch.", "pointer": f"{C1} .truth.scored; {R_C1}"},
            "evidence_beside": "No held-out reading.", "cells": cells}


# ---------- 67M pairs labelled only by how they were built ----------

def r_ratio(names: tuple[list[str], list[str]], pole: tuple[str, str], job: str = "46258") -> dict[str, Any]:
    out = {}
    for m, key in (("r", "r"), ("ratio", "ratio")):
        c = Cell(job)
        reuse, new = pair(c, E60, '.runs["{}"].' + key + ".value", *names)
        held = c.get(E60S, ".pass")
        out[m] = sep_cell(c, reuse, new, "Zero update reads 0; no filed cut-off (E60: labels none yet).", None,
                          f"{NO_KNOWN_KIND} Stage 0 pass: {held}.", pole)
    return out


def e46() -> dict[str, Any]:
    copy = [f"ident32{b}_s{s}" for b in ("", "b") for s in (0, 1)]
    rand = [f"rand32{b}_s{s}" for b in ("", "b") for s in (0, 1)]
    cells = r_ratio(([f"e46/{n}" for n in copy], [f"e46/{n}" for n in rand]), ("copy", "random"))
    c = Cell("44095")
    reuse, new = pair(c, E52, '.b3["{}"].S', copy, rand)
    cells["steering"] = c.done(
        "separates", f"copy {rng(reuse)} vs random {rng(new)}", "E51's rule; a learned-nothing fine-tune is the null.",
        None, NO_KNOWN_KIND, reuse, new, filed=f"P3 {c.get(E52, '.score.P3.verdict')}, gap {c.get(E52, '.score.P3.gap'):.3f}",
        notes=["Steering may separate this pair by construction: one input-independent vector against a key-dependent cipher.",
               (f"The learned-nothing fine-tune read a gain of {c.get(E52, '.code_step10.G'):.4f} against a floor of "
               f"{c.get(E52, '.code_step10.floor')}, so the filed null check failed narrowly.")])
    c = Cell("44095")
    reuse, new = pair(c, E52, '.b2["{}"].shares["12288"]', copy, rand)
    k = c.many(E52, '.b2["{}"].k50', copy + rand)
    cells["neuron_patch"] = c.done(
        "fails", f"half the gain only at all {max(k)} neurons; full width copy {rng(reuse)} vs random {rng(new)}",
        "E51's rule on neurons needed and on the idle share.", True, "Identity and full residual patches.", reuse, new,
        filed=f"P1 {c.get(E52, '.score.P1.verdict')}, P2 {c.get(E52, '.score.P2.verdict')}",
        notes=["The full-width share is higher for random: the wrong way."])
    c = Cell("44095; amendment 45898")
    reuse, new = pair(c, E52, '.b4.shares["{}"]["16"]', copy, rand)
    a_reuse, a_new = pair(c, E52A, '.shares["{}"]["31"]', copy, rand)
    cells["prompting"] = c.done(
        "fails", f"16 demonstrations: copy {rng(reuse)} vs random {rng(new)}; 31 beside the query: {rng(a_reuse)} vs {rng(a_new)}",
        "Share 0 with no demonstrations; bar 0.5 at 31.", True,
        f"Induction control top-1 {c.get(E52A, '.stage0.induction.top1_second'):.2f} on repeated random text.", a_reuse, a_new,
        filed=f"P4 {c.get(E52, '.score.P4.verdict')}; amendment {c.get(E52A, '.score.verdict')}, band {c.get(E52A, '.score.band')!r}",
        notes=["Top-1 is 0 on every window at every n."])
    cells["sparsity"] = sparsity("T3_B", "ident32 vs rand32 only")
    return {"id": "e46_copy_random", "name": "Copy maps vs random maps", "model": "67M",
            "states": "8, E12's recipe, two tables each, seeds 0-1",
            "truth": {"how": "only by assumption", "source_said": "built to reuse / construction labels (E60); 'a mechanism claim only, never behaviourally verified' (E52 amendment)",
                      "changed": True, "detail": ASSUMPTION},
            "evidence_beside": "No held-out reading.", "cells": cells}


def e53() -> dict[str, Any]:
    low = ["lowercase_s0", "lowercase_s1"]
    new_k = ["cipher_s0", "cipher_s1", "broad_cipher_s0", "broad_cipher_s1"]
    cells = r_ratio(([f"e53/{n}" for n in low], [f"e53/{n}" for n in new_k]), ("lowercase", "ciphers"))
    c = Cell("46258")
    reuse, new = pair(c, E60, '.runs["e53/{}"].S.value', low, new_k)
    cells["steering"] = sep_cell(c, reuse, new, "Zero update reads 0; no filed cut-off. Lowercase is under E60's unfiled 0.5.",
                                 None, NO_KNOWN_KIND, ("lowercase", "ciphers"),
                                 notes=["E53's fit and scoring halves are neighbouring windows of the same documents, so S may be biased up."])
    cells["sparsity"] = sparsity("T3_C", "lowercase vs broad cipher only")
    return {"id": "e53_lowercase_cipher", "name": "Lowercase vs ciphers", "model": "67M", "states": "6, step 300, seeds 0-1",
            "truth": {"how": "only by assumption", "source_said": "built to reuse / construction labels (E60)", "changed": True, "detail": ASSUMPTION},
            "evidence_beside": "No held-out reading.", "cells": cells}


def sparsity(test: str, scope: str) -> dict[str, Any]:
    c = Cell("48527")
    numbers = ["W1", "W1g", "W2", "A1", "C1"]
    passed = [n for n in numbers if c.get(E63, f".rows.{n}.{test}.pass")]
    assert not any(c.get(E63, f".rows.{n}.answers_yes") for n in numbers)
    caveat = c.report(R_E63, "All four tests compare one task with one task. A false yes runs at 6-9% per number and 25-37% over "
                             "five before references, and the test has little power, covers few tasks, and neither reading yet "
                             "has a causal check.")
    return c.done(
        "fails", f"this test passed by {', '.join(passed) or 'none'}, of the five numbers; {scope}",
        "Each number against a rotated, shuffled, Gaussian or random-order reference inside the test.",
        None, f"{NO_KNOWN_KIND} Stage 0 pass: {c.get(E63S, '.pass')}.", filed=c.get(E63, ".rows.branch"),
        notes=["Low power: " + caveat, "A number answers yes only if it passes all four tests; a pass here is on one task pair."])


def e54_matched() -> dict[str, Any]:
    job = "45963, 45973, 45974; 46170, 46171"
    cells: dict[str, Any] = {}
    c = Cell(job)
    reuse, new = c.get(E56B, ".P2.r.a"), c.get(E56B, ".P2.r.b")
    cells["r"] = c.done(
        "separates", f"elicit {rng(reuse)} vs cipher {rng(new)}, five seeds", "E51's rule; zero update reads 0.", None, NO_KNOWN_KIND,
        reuse, new, filed=f"{c.get(E56B, '.P2.r.verdict')}, gap {c.get(E56B, '.P2.r.gap'):.3f}, range {c.get(E56B, '.P2.r.spread'):.3f}",
        notes=[c.report(R_E54, "Stage 0 checks that the code reproduces earlier numbers. It is not a positive control of known kind for either reading.")])
    cells["ratio"] = r_ratio((["e54/elicit_s0", "e54/elicit_s1"], ["e54/cipher_s0", "e54/cipher_s1"]), ("elicit", "cipher"))["ratio"]
    c = Cell(job)
    reuse, new = c.get(E56B, ".P2.S_lo.a"), c.get(E56B, ".P2.S_lo.b")
    cells["steering"] = c.done(
        "fails", f"log-odds share elicit {rng(reuse)} vs cipher {rng(new)}, five seeds", "E51's rule; gap needs at least 0.2.", None,
        NO_KNOWN_KIND, reuse, new,
        filed=f"S_lo verdict {c.get(E56B, '.P2.S_lo.verdict')}; E54's nats S P2 {c.get(E54, '.score.P2.status')}, verdict {c.get(E54, '.score.P2.verdict')}",
        notes=[f"In nats the gap is {c.get(E54, '.score.P2.gap'):.2f}, under the 0.2 minimum."])
    c = Cell(job)
    reuse, new = c.get(E56B, ".P1.N16.a"), c.get(E56B, ".P1.N16.b")
    cells["neuron_patch"] = c.done(
        "fails", f"top 16 neurons elicit {rng(reuse)} vs cipher {rng(new)}; neurons for half the gain "
                 f"{c.get(E56A, '.states.elicit_s0.k50')} vs {c.get(E56A, '.states.cipher_s0.k50')}",
        "E51's rule on a filed grid; zero means the top neurons carry nothing.", True, "Identity and full residual patches, full-width rise.", reuse, new,
        filed=f"N16 {c.get(E56B, '.P1.N16.verdict')}, k50 index {c.get(E56B, '.P1.k50_index.verdict')}; filed null, status {c.get(E56B, '.P1.status')}",
        notes=[(f"The full-width share separates at matched size ({c.get(E56A, '.states.elicit_s0.b2_shares[\"12288\"]'):.2f} vs "
               f"{c.get(E56A, '.states.cipher_s0.b2_shares[\"12288\"]'):.2f} on seed 0) and not at the cipher's own step; adopted reading: it follows progress.")])
    c = Cell("48412; 48453")
    counts = [c.get(E59, f".possession.elicit.all.possession[{b}].count") for b in range(4)]
    cipher = [c.get(E59, f".possession.cipher.all.possession[{b}].count") for b in range(4)]
    cells["readout_full"] = c.done(
        "unreadable", f"keys held, elicit {counts}, cipher {cipher}, of 32",
        "24 keys needed. " + c.report("experiments/TRIED.md", "W0's own logits through the same rules read elicit 16 in band, possession 17/32, cipher 0 (job 48453)."), False,
        f"Construction check fails: {c.get(E59, '.P3.elicit_in_band')} keys in band, 24 needed.",
        filed=f"P1 {c.get(E59, '.P1.status')}, closed at its hard cap as unreadable; {c.get(E59, '.stopped')}",
        notes=["Stopped before the do-nothing, the control on the original model and the ablations.",
               "The model itself misses this bar, so a perfect readout could not have met it."])
    cells["readout_64"] = readout64_matched()
    cells["sparsity"] = sparsity("T1", "five seeds at matched size")
    c = Cell(job)
    reuse, new = c.many(E54, ".finetunes.{}.norm", ["elicit_s0", "elicit_s1"]), c.many(E54, ".finetunes.{}.norm", ["cipher_s0", "cipher_s1"])
    cells["labeller"] = sep_cell(c, reuse, new, "Same lr; the cipher is read at the step matching the elicit size.", None,
                                 "No control: size is matched by design.", ("elicit", "cipher"),
                                 notes=["Size cannot separate this row: it was matched."])
    cells["prompting"] = not_run("E54 did not run it: E52's amendment found no working prompt format.")
    return {"id": "e54_matched", "name": "Half-known vs cipher, equal size", "model": "67M",
            "states": "10, 32 keys, seeds 0-4, elicit step 10, cipher step 8-9",
            "truth": {"how": "only by assumption", "source_said": "construction labels (E60); 'built to elicit, no causal label' (E64)", "changed": True, "detail": ASSUMPTION},
            "evidence_beside": "No held-out reading on this pair; E58's held-out reading on a 24-key version sits with the rank-1 row.", "cells": cells}


def readout64_matched() -> dict[str, Any]:
    c = Cell("48611")
    reuse, new = c.get(E64, ".P2.separation.a"), c.get(E64, ".P2.separation.b")
    return c.done(
        "unreadable", f"gain lost, elicit {rng(reuse)} vs cipher {rng(new)}",
        c.report("experiments/TRIED.md", "Random direction at most 0.0032.") + " A gap of +0.2 was predicted.", False,
        f"Fit bar missed (pass: {c.get(E64, '.stage0.fit.pass')}); control on the original model held (pass: {c.get(E64, '.W0_control.elicit_M.pass')}).",
        reuse, new, filed=c.get(E64, ".branch"),
        notes=[f"P1 {c.get(E64, '.P1.status')}, P2 {c.get(E64, '.P2.status')}, P3 {c.get(E64, '.P3.status')}. The gap's sign is reversed.",
               "A next-token count table with no model nearly matches the holding pattern: provisional, post hoc."])


def e54_own() -> dict[str, Any]:
    seeds = range(5)
    job = "45973, 45974; 46170, 46171"
    def read(c: Cell, kind: str, field: str) -> list[float]:
        return [c.get(E56A if s < 2 else E56B, f'.states.{kind}_s{s}.{field}') for s in seeds]
    cells: dict[str, Any] = {}
    c = Cell(job)
    cells["r"] = sep_cell(c, read(c, "elicit", "r"), read(c, "cipher_own", "r"), "Zero update reads 0; no filed rule on this comparison.", None, NO_KNOWN_KIND,
                          ("elicit", "cipher"), filed=f"P3: matched and own-step cipher agree within 0.1, status {c.get(E56B, '.P3.status')}")
    c = Cell("46070")
    reuse, new = c.get(E54A, '.score["P3-own"].a'), c.get(E54A, '.score["P3-own"].b')
    cells["steering"] = c.done(
        "separates", f"elicit {rng(reuse)} vs cipher {rng(new)}, two seeds", "E51's rule; gap needs at least 0.2.", None, NO_KNOWN_KIND, reuse, new,
        filed=f"P3-own {c.get(E54A, '.score[\"P3-own\"].verdict')}, status {c.get(E54A, '.score[\"P3-own\"].status')}; 'not a test'",
        notes=["The gap comes from the cipher learned further along its own trajectory, so it is not a test of kind."])
    c = Cell(job)
    cells["neuron_patch"] = sep_cell(c, [c.get(E56A if s < 2 else E56B, f'.states.elicit_s{s}.b2_shares["12288"]') for s in seeds],
                                     [c.get(E56A if s < 2 else E56B, f'.states.cipher_own_s{s}.b2_shares["12288"]') for s in seeds],
                                     "Full-width share of the gain.", True, "Identity and full residual patches.", ("elicit", "cipher"),
                                     notes=["Neurons for half the gain read 4096 on both."])
    c = Cell("48611")
    own = [c.get(E64, ".P2.own_diff.s0"), c.get(E64, ".P2.own_diff.s1")]
    cells["readout_64"] = c.done("unreadable", f"own-step and matched cipher differ by {rng(own)}", "Within 0.1 predicted.", False, "Fit bar missed.",
                                 filed=c.get(E64, ".branch"))
    cells["readout_full"] = not_run("E59 stopped before any ablation.")
    cells["sparsity"] = sparsity("T2", "seeds 0-1, equal progress")
    c = Cell("45973, 45974")
    cells["labeller"] = sep_cell(c, c.many(E54, ".finetunes.{}.norm", ["elicit_s0", "elicit_s1"]), c.many(E54, ".finetunes.{}.norm", ["cipher_own_s0", "cipher_own_s1"]),
                                 "Same lr, different steps: not a within-cell reading.", None, "No control: it reads the recipe's result.", ("elicit", "cipher"),
                                 notes=["The cipher is the larger update by design here."])
    cells["ratio"] = not_run("E60's table holds the matched cipher only.")
    return {"id": "e54_own", "name": "Half-known vs cipher, own steps", "model": "67M",
            "states": "10, seeds 0-4, cipher step 30", "truth": {"how": "only by assumption", "source_said": "as the row above", "changed": False, "detail": ASSUMPTION},
            "evidence_beside": "No held-out reading.", "cells": cells}


def open_pair() -> dict[str, Any]:
    job = "46808, 46815, 46840"
    cells: dict[str, Any] = {}
    for m, key, name, zero in (("r", "r", "r", "Zero update reads 0."), ("ratio", "ratio", "ratio", "Zero update reads 0."),
                               ("pruning", "R", "share revived", "Minus 20 random removals."),
                               ("reverse", "t_half", "half-life in steps", "Floor 1 step."),
                               ("steering", "S", "share", "Zero update reads 0."),
                               ("labeller", "norm", "update size", "Sizes matched by design."),
                               ("probe", "q", "probe margin", "Cipher should read about 0.")):
        c = Cell(job)
        rd = f".score.readings.{key}"
        reuse, new = c.get(C3, f".columns.{key}.rank1"), c.get(C3, f".columns.{key}.cipher")
        gap, noise, status = c.get(C3, rd + ".gap"), c.get(C3, rd + ".noise"), c.get(C3, rd + ".status")
        ok = status == "ok"
        verdict = "separates" if ok else "control failed"
        text = f"rank-1 {rng(reuse)} vs cipher {rng(new)}; gap {gap:+.2f}, bar {3 * noise:.2f}"
        filed = f"status {status!r}"
        if key in load(C3)["score"]["verdicts"]:
            filed += f"; verdict {c.get(C3, f'.score.verdicts.{key}')!r}"
        control = f"Poles must differ by 3 seed sd: {'held' if ok else 'failed'}."
        control_held: bool | None = ok
        planted = None
        notes = []
        if key == "R":
            planted = c.get(C3, ".stage0.planted.R")
        if key in ("R", "t_half"):
            control += f" A planted 13-unit edit passed: {c.get(C3, '.stage0.planted.prune_pass.pass')}, {c.get(C3, '.stage0.planted.reverse_pass')}."
            notes.append("As ported to the 67M, this method failed its positive control, so the counting set's 21 of 21 does not carry over.")
        if key in ("r", "ratio"):
            notes.append(f"Elicit's position between the poles is {c.get(C3, rd + '.p'):.2f}; the filed reading is that this follows W0's distance and prior.")
        if key == "S":
            notes.append(f"Steering treats rank-1 like the cipher here and elicit nearer copy: elicit {c.get(C3, rd + '.means.elicit'):.2f}, copy {c.get(C3, rd + '.means.copy'):.2f}.")
        if key == "norm":
            verdict, control_held, control = "fails", None, "No control: size is matched by design."
            notes.append("Size cannot separate this row: it was matched.")
        if key == "q":
            notes.append(f"Filed probe reference missed: all {c.get(C3, '.score.probe_ref.all')}, rank-1 at least elicit {c.get(C3, '.score.probe_ref.rank1_ge_elicit')}.")
        cells[m] = c.done(verdict, text, zero, control_held, control, reuse, new, filed=filed, notes=notes,
                          derived=f"Bar is 3 x the filed noise, {noise:.3f}.", control_value=planted)
    cells["neuron_patch"] = not_run("E58's maps were never patched.")
    cells["sparsity"] = not_run("E63 read E54, E46 and E53 states only.")
    x = jq(load(C3), ".score.readings.x.means")
    return {"id": "open_pair_c3", "name": "Rank-1 vs cipher, step 10", "model": "67M", "states": "16 of C3's 32, seeds 0-7, 24 trained keys",
            "truth": {"how": "only by assumption", "source_said": "'old by construction' and 'new by construction' (C3, its ladder)", "changed": True, "detail": ASSUMPTION},
            "evidence_beside": f"Held-out carry, evidence only: rank-1 {x['rank1']:.2f}, cipher {x['cipher']:.2f}, elicit {x['elicit']:.2f}, copy {x['copy']:.2f} ({C3} .score.readings.x.means).",
            "cells": cells}


def ladder() -> dict[str, Any]:
    cells: dict[str, Any] = {}
    steps = ["10", "30", "100", "300"]
    for m, key in (("pruning", "R"), ("reverse", "t_half")):
        c = Cell("46954")
        gaps = [c.get(LAD, f'.score_ladder.per_step["{s}"].{key}.gap') for s in steps]
        status = [c.get(LAD, f'.score_ladder.per_step["{s}"].{key}.status') for s in steps]
        note = ["As ported to the 67M, this method failed its positive control at every update size E58's recipe makes.",
                "The cipher is only half-learned at step 10, so the one filed-direction gap coincides with it."]
        if key == "R":
            note.append(f"Pruning more units did not pass either: status {c.get(LAD, '.score_kreread.R.status')!r} at the best K up to 320.")
        cells[m] = c.done("control failed", "rank-1 minus cipher at steps 10, 30, 100, 300: " + ", ".join(f"{g:+.2f}" for g in gaps),
                          "Poles must differ by 3 seed sd, with a floor.", False, f"Passed at no step: first passing step {c.get(LAD, f'.score_ladder.pass_step.{key}')}.",
                          filed="; ".join(f"step {s} {st!r}" for s, st in zip(steps, status)), notes=note)
    for m, why in (("r", "VPD at steps 30 to 300 was not read."), ("ratio", "VPD at steps 30 to 300 was not read."),
                   ("labeller", "Sizes are saved in ladder.json; no labeller was placed on them.")):
        cells[m] = not_run(why)
    return {"id": "open_pair_ladder", "name": "Rank-1 vs cipher, steps 10-300", "model": "67M", "states": "9 per step, three seeds, plus C3's 32 re-read",
            "truth": {"how": "only by assumption", "source_said": "'old by construction' and 'new by construction'", "changed": True, "detail": ASSUMPTION},
            "evidence_beside": "No held-out reading in this stage.", "cells": cells}


def e62() -> dict[str, Any]:
    copy, lookup = [f"copy_s{s}" for s in range(3)], [f"lookup_s{s}" for s in range(3)]
    job = "48541"
    c = Cell(job)
    reuse, new = c.get(E62, ".score.r_copy"), c.get(E62, ".score.r_lookup")
    cells: dict[str, Any] = {"r": c.done(
        "unreadable", f"unscored: copy {rng(reuse)} vs lookup {rng(new)}", f"Zero update gain {c.get(E62, '.zero_route.gain')}.",
        None, f"Path control pass: {c.get(E62, '.path_control.pass')}. No known-kind state: the only causal label here is the one that left r unscored.",
        reuse, new, filed=f"r_scored {c.get(E62, '.score.r_scored')}; verdict {c.get(E62, '.score.verdict')!r}",
        notes=["Three readings of the gap stay live: mechanism, context and update size.",
               c.report("experiments/TRIED.md", "From U0 the copy map reads 0.92, so the excess over 1 is a low L00.")])}
    c = Cell(job)
    reuse, new = pair(c, E62, ".states.{}.norm", copy, lookup)
    cells["labeller"] = sep_cell(c, reuse, new, "Same recipe cell, same step.", None, "No control: it reads the recipe's result.", ("copy", "lookup"),
                                 notes=[c.report(R_E62C, "Same recipe cell, a factor 1.19 apart, with no overlap: the size separates the maps as cleanly as r does.")])
    data = load(E62)["states"]
    return {"id": "e62_copy_lookup", "name": "Copy at a new marker vs lookup", "model": "67M", "states": "6, seeds 0-2, step 30",
            "truth": {"how": "by causal check", "source_said": "share of gain lost with W0's copying head off; verdict as filed 'copy_not_elicitation'", "changed": False,
                      "pointer": f"{E62} .states.<state>.label, .states.<state>.share, .score.verdict",
                      "detail": "Labels as filed, seeds 0-2: copy " + ", ".join(data[k]["label"] for k in copy) + "; lookup " + ", ".join(data[k]["label"] for k in lookup)
                                + ". Share lost: copy " + rng([data[k]["share"] for k in copy]) + " against a bar of 0.5, lookup " + rng([data[k]["share"] for k in lookup]) + "."},
            "evidence_beside": "No held-out reading.", "cells": cells}


# ---------- assembling ----------

def check_e51() -> None:
    """The local copy of the rule must reproduce the verdicts it was filed with."""
    for file, path in ((E56B, ".P2.r"), (E56B, ".P2.S_lo"), (E54, ".score.P1"), (E54, ".score.P2")):
        filed = jq(load(file), path)
        assert e51(filed["a"], filed["b"])["verdict"] == filed["verdict"], path


def fill(row: dict[str, Any]) -> dict[str, Any]:
    for m in METHODS:
        row["cells"].setdefault(m, not_run("No source reads this method on this fine-tune."))
    row["cells"] = {m: row["cells"][m] for m in METHODS}
    return row


def write_section(table: dict[str, Any]) -> None:
    cls = {"separates": "y", "fails": "n", "control failed": "p", "unreadable": "q", "not run": "u"}
    groups = {"by construction": [], "by causal check": [], "only by assumption": []}
    for row in table["rows"]:
        groups[row["truth"]["how"]].append(row)
    head = "".join(f'<th title="{html.escape(METHODS[m][1])}">{METHODS[m][0]}</th>' for m in METHODS)
    body = []
    for how, rows in groups.items():
        body.append(f'<tr class="grp"><td colspan="{len(METHODS) + 1}">Truth set {how}</td></tr>')
        for row in rows:
            tds = [f'<td title="{html.escape(row["states"] + ". " + row["truth"]["detail"])}">{row["name"]}</td>']
            for m in METHODS:
                cell = row["cells"][m]
                star = "*" if COUNTING_NOTE in cell["notes"] else ""
                short = "" if cell["verdict"] == "not run" else "<br>" + html.escape(short_reading(cell))
                point = cell["pointers"][0] if cell["pointers"] else None
                title = cell["reading"]["text"] + (f" | {point['file']} {point['jq']}" if point else "")
                tds.append(f'<td class="{cls[cell["verdict"]]}" title="{html.escape(title)}">{cell["verdict"]}{star}{short}</td>')
            body.append("<tr>" + "".join(tds) + "</tr>")
    defs = "".join(f"<li><b>{name}.</b> {html.escape(what)}</li>" for name, what in METHODS.values())
    SECTION.write_text(SECTION_TEMPLATE.format(head=head, body="\n".join(body), defs=defs))


def short_reading(cell: dict[str, Any]) -> str:
    reuse, new = cell["reading"]["reuse"], cell["reading"]["new"]
    count = re.match(r"\d+ of \d+", cell["reading"]["text"])
    if count:
        return count.group(0)
    if reuse and new:
        return f"{rng(reuse)} / {rng(new)}"
    return cell["reading"]["text"].split(";")[0][:32]


SECTION_TEMPLATE = """<section id="methodtable">
<div class="sechead"><div class="eyebrow">The benchmark line's question, in one table</div><h2><span class="num"></span>Every method against every fine-tune</h2></div>

<div class="prose">
<p><b>The question.</b> Does VPD tell reuse from new better than cheaper methods do, judged against a truth it did not define? <b>Not yet answerable.</b> No fine-tune has VPD and a cheaper method both read, with their controls held, against a truth set by construction or a causal check. VPD's reading separates every pair it read that is labelled only by how it was built, where the truth is an assumption.</p>
</div>
<figure class="boxed"><img src="fig/methodtable_shares.png" alt="Dot and range plot, one row per method on a fine-tune pair, grouped by how the truth was set. Each row shows the reuse side and the new side on one axis of shares from below zero to above one, with the do-nothing line at zero and a minimum-gap bar of 0.2. Every wide gap under by construction and by causal check carries a star, a failed control, or an unscored reading."><figcaption>Methods whose reading is a share. Reverse fine-tuning, neuron sparsity and the size labeller are on other scales and appear only in the table.</figcaption></figure>
<div class="prose">
<p><b>How to read a cell.</b> The first line is the verdict. The second gives the reuse side, then the new side, or the count labelled right. Hover for the file and path; the full record is <code>experiments/method_table/results/table.json</code>. The analysis is <code>reports/analyses/2026-11-02_analysis_method-table.md</code>.</p>
<p><b>A star marks a counting-set cell.</b> Those labels equal task P and task N, so any method that responds to the task scores perfectly. 21 of 21 there is not discrimination. As ported to the 67M, pruning revival and reverse fine-tuning failed their positive control, so the 21 of 21 does not carry over. VPD's counting reading is from a decomposition trained for 400 of 400,000 steps: it labelled all 21 right untrained, so it enters as failing its control and not as a pass.</p>
<p><b>Three kinds of truth.</b> By construction: the fine-tuning task fixes the answer, and each state passed behaviour and patch checks. By causal check: switching off or patching the mechanism is the label. Only by assumption: built to elicit is how the task was built, not the causal label; these pairs have no causal check.</p>
</div>
<p class="legend cmpkey"><span class="lg"><b class="cv y">separates</b></span><span class="lg"><b class="cv n">fails</b></span><span class="lg"><b class="cv p">control failed</b></span><span class="lg"><b class="cv q">unreadable</b></span><span class="lg"><b class="cv u">not run</b></span></p>
<div class="tblwrap"><table class="cmp">
<thead><tr><th>Fine-tune</th>{head}</tr></thead>
<tbody>
{body}
</tbody></table></div>
<div class="prose">
<p><b>The methods.</b></p>
<ul>{defs}</ul>
<p><b>What stays attached.</b> Neuron sparsity is a null of a low-power rule that compares one task with one task. The 64-value readout is unreadable as filed. The copying-heads fine-tunes leave r unscored as filed, with verdict <code>copy_not_elicitation</code>.</p>
</div>
</section>
"""


def main() -> None:
    check_e51()
    rows = [fill(r()) for r in (counting_c1b, counting_c1, e62, e46, e53, e54_matched, e54_own, open_pair, ladder)]
    head_to_head = [r["id"] for r in rows if r["truth"]["how"] != "only by assumption" and r["cells"]["r"]["verdict"] == "separates"
                    and any(r["cells"][m]["verdict"] == "separates" for m in METHODS if m not in ("r", "ratio", "probe", "labeller"))]
    table = {
        "question": "Does VPD tell reuse from new better than cheaper methods do, judged against a truth it did not define?",
        "label_rule": "decisions.md 2026-11-02: elicitation is mechanistic; the causal check is the label, the held-out reading is evidence beside it. No counting-set label changes.",
        "counting_note": COUNTING_NOTE, "assumption_words": ASSUMPTION,
        "derived_head_to_head": {"rows": head_to_head, "rule": "derived: truth not by assumption, VPD's r separates with its control and a cheap method separates"},
        "markings_changed": [{"row": r["id"], "source_said": r["truth"]["source_said"], "table": r["truth"]["how"]} for r in rows if r["truth"]["changed"]],
        "methods": {m: {"name": n, "definition": d} for m, (n, d) in METHODS.items()}, "verdicts": VERDICTS, "rows": rows,
        "left_out": "Fine-tunes with no method but VPD's or no reading in the named sources: E58 Amendments 2-7, E57, E34, the Qwen line.",
    }
    OUT.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    write_section(table)
    for r in rows:
        print(r["id"], r["truth"]["how"], {m: c["verdict"] for m, c in r["cells"].items() if c["verdict"] != "not run"})
    print("head to head:", head_to_head)


if __name__ == "__main__":
    main()
