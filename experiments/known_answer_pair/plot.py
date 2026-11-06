"""E103's figures from results/{stage_0,train,read,neurons}.json; no model (the tokenizer only, for answer lengths).
The known-answer fine-tune is green, the fictional-author fine-tune orange, the template blue, the Gaussian change grey,
as in E102's figures.

  figures/gain_curve.png     the gain on the real answers at every candidate step against |dW|, with the template's gain,
                             C1's bar, do-nothing, and the fictional states' gains on the same answers
  figures/w0_deficit.png     W0's summed log P of each real answer against its number of tokens: one fixed deficit per
                             answer, or one per token
  figures/recall.png         real-fact multiple-choice share at W0 and after each update, with W0's interval and R1's line
  figures/pair_rules.png     each pair rule's gap, known minus the fictional seeds' mean, in units of its band
  figures/prompt_overlap.png per state, the share of its carrying set inside the prompt's contrast set, on each row set,
                             with the Gaussian change and the random draws' maximum
Amendment 1 (results/amend1_*.json, neurons_amend1*, the per-position files in results/large/amend1_tokens/):
  figures/amend1_positions.png  each matched state's change in log P at the first answer token, the later ones and the
                                end token of its own trained answers, with C1's bar
  figures/amend1_shared.png     each matched state's answer-word gain on both arms' rows: own rows against the other's
  figures/amend1_pair_rules.png the four matched pair rules in units of their bands, E103's arm and the new one
  figures/amend1_overlap.png    carrying-set overlaps: fictional seeds, the two known arms, across arms, the Gaussian
  figures/amend1_recall.png     real-fact share of every state, the new arm added

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/known_answer_pair/plot.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
GREEN, ORANGE, BLUE, GREY, INK, GRID = "#1a9e77", "#eb6834", "#2a78d6", "#7a7a75", "#52514e", "#e4e3df"
TEMPLATE_NORM = 0.4865  # E91's reference, run.py REF_NORM
FICT_NORM = {"fict_matched": 0.4773, "fict_full": 18.11}  # qwen_size_grid/results/size_grid.json arms.anchor2_s0


def js(name: str) -> Any:
    return json.loads((RES / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def gain_curve(train: dict[str, Any], read: dict[str, Any], pred: dict[str, Any]) -> None:
    norm = {r["step"]: r["dw_norm"] for r in train["steps"]}
    steps = sorted(int(t) for t in train["gains"])
    st = read["states"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([norm[t] for t in steps], [train["gains"][str(t)] for t in steps], "-o", color=GREEN, ms=3.5,
            label="known-answer fine-tune, steps 2 to 33")
    ax.plot([norm[train["matched"]["step"]]], [st["real_matched"]["gain"]], "o", color=GREEN, ms=10, mfc="none", mew=2,
            label="known-answer, matched step 33")
    ax.plot([train["steps"][-1]["dw_norm"]], [st["real_full"]["gain"]], "s", color=GREEN, ms=8, label="known-answer, end (step 335)")
    ax.plot([FICT_NORM["fict_matched"]], [st["fict_matched"]["gain"]], "o", color=ORANGE, ms=8, label="fictional-author, matched step 29")
    ax.plot([FICT_NORM["fict_full"]], [st["fict_full"]["gain"]], "s", color=ORANGE, ms=8, label="fictional-author, end (step 1250)")
    C1 = pred["C1_gain"]
    ax.axhline(C1["template_gain"], color=BLUE, ls="--", lw=1.2, label="template fine-tune's gain on its own rows, 0.67")
    ax.axhline(C1["bar"], color=INK, ls=":", lw=1.2, label="control's pass bar, 0.33")
    ax.axhline(0, color=GREY, lw=1)
    ax.text(0.0026, -0.75, "unchanged model and do-nothing, 0", color=GREY, fontsize=8)
    ax.axvline(TEMPLATE_NORM, color=BLUE, lw=0.8, alpha=0.5)
    ax.text(TEMPLATE_NORM * 1.08, 3.2, "template's\nweight change", color=BLUE, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-1.5, 13.5)
    ax.set_xlabel("size of the weight change, Frobenius norm of all linear layers")
    ax.set_ylabel("gain in mean log P of the real answers\n(nats a token, mean over 217 rows)")
    ax.set_title("The gain on known answers is full-size after 10 steps, at a sixth of the matched size", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gain_curve.png", dpi=150)
    plt.close(fig)


def answer_lengths() -> list[int]:
    """Response tokens of each real row's pair 0 (the answer and the end token), as `run.answer_seqs` builds them."""
    from transformers import AutoTokenizer

    from commute import qwen_tofu as tofu
    from commute.qwen_eval import MODEL, MODEL_REV

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    end = int(tok.convert_tokens_to_ids("<|im_end|>"))
    out = []
    for r in tofu.real_rows():
        ids, p = tofu.seq(tok, end, r["question"] + tofu.ASK, r["answer"])
        out.append(len(ids) - p)
    return out


def w0_deficit(s0: dict[str, Any]) -> None:
    n, mean = answer_lengths(), s0["w0_logp"]
    total = [m * k for m, k in zip(mean, n, strict=True)]
    per_token = sum(total) / sum(n)  # pooled: what each token would carry if the deficit were spread evenly
    ks = sorted(set(n))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    jit = [k + 0.25 * ((i * 0.618) % 1 - 0.5) for i, k in enumerate(n)]
    ax.scatter(jit, total, s=10, color=GREEN, alpha=0.45, lw=0, label="one real answer, unchanged model")
    ax.plot(ks, [statistics.mean(t for t, m in zip(total, n, strict=True) if m == k) for k in ks], "-o", color=GREEN, ms=6,
            label="mean at each length")
    ax.plot([0, max(ks) + 0.5], [0, per_token * (max(ks) + 0.5)], "--", color=GREY,
            label=f"if every token carried the pooled mean, {per_token:.1f} nats")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlim(1.4, max(ks) + 0.6)
    ax.set_xlabel("tokens in the answer, the end-of-turn token included")
    ax.set_ylabel("summed log P of the answer, nats")
    ax.set_title("The unchanged model's deficit is about 45 nats per answer, whatever its length", fontsize=10)
    style(ax)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "w0_deficit.png", dpi=150)
    plt.close(fig)


def recall(read: dict[str, Any], pred: dict[str, Any]) -> None:
    w0 = read["W0"]["real"]
    rows = [("unchanged\nmodel", w0, GREY), ("known-answer\nmatched", read["states"]["real_matched"]["real"], GREEN),
            ("known-answer\nend", read["states"]["real_full"]["real"], GREEN),
            ("fictional\nmatched", read["states"]["fict_matched"]["real"], ORANGE),
            ("fictional\nend", read["states"]["fict_full"]["real"], ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(*w0["ci"], color=GREY, alpha=0.15, lw=0, label="unchanged model, 95% interval")
    line = w0["share"] - pred["R1_recall"]["drop"]
    ax.axhline(line, color=INK, ls=":", lw=1.2, label=f"'kept' line, unchanged model minus 0.05, {line:.2f}")
    ax.axhline(0.25, color=GREY, ls="--", lw=1, label="chance, one of four")
    for i, (_, r, c) in enumerate(rows):
        ax.bar(i, r["share"], color=c, width=0.6)
        ax.errorbar(i, r["share"], yerr=[[r["share"] - r["ci"][0]], [r["ci"][1] - r["share"]]], color=INK, capsize=3)
    ax.set_xticks(range(len(rows)), [lab for lab, *_ in rows], fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False, ncol=1)
    ax.set_ylabel("share of 217 real facts picked\nover three wrong options")
    ax.set_title("Training on known facts raises their recall; the fictional fine-tune's end state loses it", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "recall.png", dpi=150)
    plt.close(fig)


RULES = [("matched", "real", "engaged"), ("matched", "tofu", "engaged"), ("matched", "real", "prompt_overlap"),
         ("matched", "tofu", "prompt_overlap"), ("full", "real", "engaged"), ("full", "tofu", "engaged"),
         ("full", "real", "prompt_overlap"), ("full", "tofu", "prompt_overlap")]
WORDS = {"engaged": "engaged share", "prompt_overlap": "prompt overlap", "real": "real rows", "tofu": "TOFU rows"}


def pair_rules(neu: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axhspan(-1, 1, color=GREY, alpha=0.15, lw=0)
    ax.text(7.4, -1.45, "grey band: within the rule's band, level", color=INK, fontsize=8, ha="right")
    ax.axvline(3.5, color=GRID, lw=1.5)
    ax.text(1.5, 5.0, "matched size, scored", ha="center", fontsize=9, color=INK)
    ax.text(5.5, 5.0, "end states, reported only", ha="center", fontsize=9, color=INK)
    for i, (size, rows, k) in enumerate(RULES):
        r = neu["rules"][f"{size}_{rows}_{k}"]
        c = GREEN if r["verdict"] == "known higher" else ORANGE if r["verdict"] == "new higher" else GREY
        ax.plot(i, r["gap"] / r["band"], "o", color=c, ms=9)
    ax.plot([], [], "o", color=GREY, label="level")
    ax.plot([], [], "o", color=GREEN, label="known higher: past the band and over the Gaussian floor")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(RULES)), [f"{WORDS[k]}\n{WORDS[rows]}" for _, rows, k in RULES], fontsize=7.5)
    ax.set_ylim(-1.8, 5.6)
    ax.set_ylabel("known-answer minus fictional mean,\nin units of the rule's band")
    ax.set_title("At matched size no neuron reading separates the known-answer from the fictional fine-tune", fontsize=10)
    style(ax)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.88), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pair_rules.png", dpi=150)
    plt.close(fig)


STATES = [("real_matched", "known\nmatched", GREEN), ("fict_s42_matched", "fictional\nmatched 42", ORANGE),
          ("fict_s142_matched", "fictional\nmatched 142", ORANGE), ("real_full", "known\nend", GREEN),
          ("fict_s42_full", "fictional\nend 42", ORANGE), ("fict_s142_full", "fictional\nend 142", ORANGE)]


def prompt_overlap(neu: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, rows in zip(axes, ("real", "tofu"), strict=True):
        per = neu["per_state"][rows]
        g = per["gauss"]["prompt_overlap"]
        draws = max(max(per[s]["prompt_overlap_draws"]) for s, *_ in STATES)
        for i, (s, _, c) in enumerate(STATES):
            ax.bar(i, per[s]["prompt_overlap"], color=c, width=0.6)
        ax.axhline(g, color=GREY, ls="--", lw=1.2, label=f"Gaussian change, {g:.3f}")
        ax.axhline(draws, color=INK, ls=":", lw=1.2, label=f"random sets, highest draw {draws:.3f}")
        ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        ax.set_xticks(range(len(STATES)), [lab for _, lab, _ in STATES], fontsize=7.5)
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}", fontsize=9)
        style(ax)
    axes[0].set_ylabel("share of the 1,024-neuron carrying set\ninside the system prompt's contrast set")
    fig.suptitle("The two end states differ in prompt overlap; the matched states do not", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "prompt_overlap.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- Amendment 1
# The new known-answer arm (pair 0 answered in a sentence) is purple; E103's known-answer arm stays green.
PURPLE = "#8a4fbf"
TOKENS = RES / "large" / "amend1_tokens"
OWN = [("e103_matched", "bare", "E103 known-answer, matched\non its bare answers", GREEN),
       ("a1_matched", "sentence", "new known-answer, matched\non its sentence answers", PURPLE),
       ("fict_matched", "fict", "fictional-author, matched\non its own answers", ORANGE)]


def positions(state: str, reader: str) -> dict[str, float]:
    """Change from W0 in log P at the answer's first token, its later tokens (row mean) and the end token, mean over rows."""
    now, w0 = (json.loads((TOKENS / f"{s}.json").read_text())[reader] for s in (state, "W0"))
    pairs = list(zip(now, w0, strict=True))
    later = [statistics.mean(a - b for a, b in zip(x[1:-1], y[1:-1], strict=True)) for x, y in pairs if len(x) > 2]
    return {"first": statistics.mean(x[0] - y[0] for x, y in pairs), "later": statistics.mean(later),
            "end": statistics.mean(x[-1] - y[-1] for x, y in pairs)}


def a1_positions(ref: dict[str, Any]) -> None:
    groups = [("first", "first answer token"), ("later", "later answer tokens,\nrow mean"), ("end", "end-of-turn token")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for j, (state, reader, lab, c) in enumerate(OWN):
        p = positions(state, reader)
        ax.bar([i + (j - 1) * 0.27 for i in range(3)], [p[g] for g, _ in groups], width=0.26, color=c, label=lab)
    ax.axhline(0, color=GREY, lw=1)
    ax.text(2.45, -1.6, "unchanged model and do-nothing, 0", color=GREY, fontsize=7.5, ha="right")
    ax.set_ylim(-2.5, 31)
    ax.plot([-0.45, 1.45], [ref["bar"]] * 2, ":", color=INK, lw=1.4)
    ax.text(0.55, ref["bar"] + 3.2, f"control's bar, applied to the mean over\nall answer tokens but the end token, {ref['bar']:.2f}", color=INK, fontsize=7.5)
    ax.set_xticks(range(3), [g for _, g in groups], fontsize=8.5)
    ax.set_ylabel("change from the unchanged model in log P\nat that position, nats (mean over rows)")
    ax.set_title("E103's gain sat on where a bare answer starts and stops; the new arm's end token does not move", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_positions.png", dpi=150)
    plt.close(fig)


def a1_shared(read: dict[str, Any]) -> None:
    st, readers = read["states"], [("sentence", "the new arm's real-fact sentences"), ("fict", "the fictional arm's answers")]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for j, (state, _, lab, c) in enumerate(OWN):
        for i, (r, _) in enumerate(readers):
            v, own = st[state][r]["content"], (state, r) in {("a1_matched", "sentence"), ("fict_matched", "fict")}
            ax.bar(i + (j - 1) * 0.27, v, width=0.26, color=c if own else "none", edgecolor=c, lw=1.6, hatch=None if own else "//",
                   label=lab.split("\n")[0] if i == 0 else None)
            ax.text(i + (j - 1) * 0.27, v + 0.05, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(range(2), [f"read on {w}" for _, w in readers], fontsize=8.5)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("answer-word gain, end token left out\n(nats a token, mean over rows)")
    ax.set_title("At matched size each update gains almost as much on the other arm's rows as on its own", fontsize=10)
    ax.text(1.45, 2.85, "filled: trained on these rows\nhatched: never trained on them", fontsize=7.5, color=INK, ha="right", va="top")
    style(ax)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_shared.png", dpi=150)
    plt.close(fig)


A1_RULES = [("real", "engaged"), ("tofu", "engaged"), ("real", "prompt_overlap"), ("tofu", "prompt_overlap")]


def a1_pair_rules(neu: dict[str, Any], neu1: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhspan(-1, 1, color=GREY, alpha=0.15, lw=0, label="inside the rule's band: level")
    for i, (rows, k) in enumerate(A1_RULES):
        for dx, src, c, lab in ((-0.12, neu, GREEN, "E103 known-answer, matched"), (0.12, neu1, PURPLE, "new known-answer, matched")):
            r = src["rules"][f"matched_{rows}_{k}"]
            ax.plot(i + dx, r["gap"] / r["band"], "o", color=c, ms=9, label=lab if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(4), [f"{WORDS[k]}\n{WORDS[rows]}" for rows, k in A1_RULES], fontsize=8)
    ax.set_ylim(-2, 2.6)
    ax.set_ylabel("known-answer minus the fictional seeds' mean,\nin units of the rule's band")
    ax.set_title("With the answer format matched, all four neuron readings stay inside their bands", fontsize=10)
    style(ax)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_pair_rules.png", dpi=150)
    plt.close(fig)


def carry(path: Path) -> set[int]:
    return set(json.loads(path.read_text())["carry"])


def a1_overlap() -> None:
    n, n1, e102 = RES / "neurons", RES / "neurons_amend1", HERE.parent / "qwen_carrying" / "results" / "states"
    sets = {"real": {"a1": n1 / "a1_matched__real.json", "e103": n / "real_matched__real.json", "f42": n / "fict_s42_matched__real.json",
                     "f142": n / "fict_s142_matched__real.json", "gauss": n / "gauss__real.json"},
            "tofu": {"a1": n1 / "a1_matched__tofu.json", "e103": n / "real_matched__tofu.json", "f42": e102 / "a2_s42_matched.json",
                     "f142": e102 / "a2_s142_matched.json", "gauss": e102 / "gauss_tofu.json"}}
    pairs = [("f42", "f142", "fictional 42\nin fictional 142", ORANGE), ("a1", "e103", "new known\nin E103 known", PURPLE),
             ("a1", "f42", "new known\nin fictional 42", PURPLE), ("e103", "f42", "E103 known\nin fictional 42", GREEN),
             ("a1", "gauss", "new known in\nGaussian change", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (rows, s) in zip(axes, sets.items(), strict=True):
        for i, (a, b, _, c) in enumerate(pairs):
            v = len(carry(s[a]) & carry(s[b])) / 1024
            ax.bar(i, v, color=c, width=0.6)
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.axhline(0.006, color=INK, ls=":", lw=1, label="two random sets, about 0.006")
        ax.set_xticks(range(len(pairs)), [lab for *_, lab, _ in pairs], fontsize=7)
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        style(ax)
    axes[0].set_ylabel("share of the first set's 1,024 neurons\ninside the second set")
    fig.suptitle("The matched carrying sets differ across arms by far more than between fictional seeds; "
                 "no seed null for the known arm", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_overlap.png", dpi=150)
    plt.close(fig)


def a1_recall(read: dict[str, Any], read1: dict[str, Any]) -> None:
    w0, st, sh = read["W0"]["real"], read["states"], read1["share"]
    rows = [("unchanged\nmodel", w0, GREY), ("E103 known\nmatched", st["real_matched"]["real"], GREEN),
            ("new known\nmatched", sh["a1_matched"], PURPLE), ("fictional\nmatched", st["fict_matched"]["real"], ORANGE),
            ("E103 known\nend", st["real_full"]["real"], GREEN), ("new known\nend", sh["a1_end"], PURPLE),
            ("fictional\nend", st["fict_full"]["real"], ORANGE)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(*w0["ci"], color=GREY, alpha=0.15, lw=0, label="unchanged model, 95% interval")
    line = w0["share"] - 0.05
    ax.axhline(line, color=INK, ls=":", lw=1.2, label=f"'kept' line, unchanged model minus 0.05, {line:.2f}")
    ax.axhline(0.25, color=GREY, ls="--", lw=1, label="chance, one of four")
    for i, (_, r, c) in enumerate(rows):
        ax.bar(i, r["share"], color=c, width=0.6)
        ax.errorbar(i, r["share"], yerr=[[r["share"] - r["ci"][0]], [r["ci"][1] - r["share"]]], color=INK, capsize=3)
    ax.set_xticks(range(len(rows)), [lab for lab, *_ in rows], fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.set_ylabel("share of 217 real facts picked\nover three wrong bare options")
    ax.set_title("Trained on sentences, the known-answer end state picks the bare true answer less often", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "amend1_recall.png", dpi=150)
    plt.close(fig)


def amend1(read: dict[str, Any], neu: dict[str, Any]) -> None:
    ref, read1, neu1 = js("amend1_ref.json")["reference"], js("amend1_read.json"), js("neurons_amend1.json")
    a1_positions(ref)
    a1_shared(read1)
    a1_pair_rules(neu, neu1)
    a1_overlap()
    a1_recall(read, read1)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    pred, s0, train, read, neu = js("predictions.json"), js("stage_0.json"), js("train.json"), js("read.json"), js("neurons.json")
    gain_curve(train, read, pred)
    w0_deficit(s0)
    recall(read, pred)
    pair_rules(neu)
    prompt_overlap(neu)
    amend1(read, neu)
    print(f"wrote {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
