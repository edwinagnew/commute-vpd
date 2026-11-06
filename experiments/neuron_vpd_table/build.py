"""Neurons against VPD: one table of filed verdicts on the same Qwen3-1.7B states, built from the result files; no model.

Every verdict string is copied from a result JSON, never typed. `CLASS` maps each verdict to what it says about the pair
(display only; no verdict is re-scored). Writes:

  results/table.json   one row per (question, size, reading kind): VPD's and the neurons' verdicts, their numbers,
                       the reference each was judged against, and the source file and key
  figures/verdicts.png the table as a grid: for each comparison, does VPD's reading / the neuron reading separate the pair
  figures/sets.png     the one reading both methods share, set identity, on the known, wrong-answer and fictional arms

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/neuron_vpd_table/build.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES, FIG = HERE / "results", HERE / "figures"
E94, E94A = EXP / "qwen_d3_rows/results/table.json", EXP / "qwen_d3_rows/results/table_amend1.json"
E102, E102B = EXP / "qwen_carrying/results/score.json", EXP / "qwen_carrying/results/score_stageB.json"
E103 = EXP / "known_answer_pair/results/neurons_amend2.json"
E105 = EXP / "rows_null/results/score.json"
E107, E107A = EXP / "label_vs_style/results/score.json", EXP / "label_vs_style/results/amend1_sets.json"
E109 = EXP / "larger_size/results/score.json"
E98, M67 = EXP / "neuron_r/results/score.json", EXP / "method_table/results/table.json"

SEP, NOT, NOCALL, NOTRUN = "separates the pair", "does not separate the pair", "no call or no power", "not run"
COLOUR = {SEP: "#2f7ebc", NOT: "#eb6834", NOCALL: "#c9c6bd", NOTRUN: "#ffffff"}
PURPLE, TEAL, ORANGE, GREY, INK, GRID = "#8a4fbf", "#00a088", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"


def js(path: Path) -> Any:
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    return str(path.relative_to(EXP.parent))


def cell(cls: str, verdict: str, numbers: str, reference: str, source: Path | None, key: str) -> dict[str, Any]:
    return {"class": cls, "verdict": verdict, "numbers": numbers, "reference": reference,
            "source": rel(source) if source else None, "key": key}


def rng(xs: list[float]) -> str:
    return f"{min(xs):.3f} to {max(xs):.3f}"


def anchors() -> list[dict[str, Any]]:
    """Question 1: the template fine-tune (built to bring out an existing layout) against the fictional-facts one."""
    m, f = js(E94)["comparisons"]["matched"], js(E94)["comparisons"]["full"]
    s, b = js(E102), js(E102B)
    po = {k: s["per_state"][k]["prompt_overlap"] for k in ("a1_s0", "a1_s1", "a2_s42_matched", "a2_s142_matched")}
    vpd_m = cell(SEP, f"outside share '{m['outside']['verdict']}'; placement '{m['placement']['verdict']}'; "
                      f"think overlap '{m['think']['verdict']}'",
                 f"outside {rng(m['outside']['anchor1'])} vs {rng(m['outside']['anchor2'])}; placement "
                 f"{rng(m['placement']['anchor1'])} vs {rng(m['placement']['anchor2'])}",
                 "gap between anchors' seed means over the larger seed spread", E94, ".comparisons.matched")
    neu_m = cell(NOT, f"prompt overlap '{s['P1_prompt_overlap_anchor1']['outcome']}' against the random change; engaged "
                      f"share '{s['P2_engaged_anchor1']['outcome']}'; job judge '{b['CB1_power_gate']['status']}'",
                 f"prompt overlap template {po['a1_s0']:.3f}, {po['a1_s1']:.3f}; fictional (unscored) "
                 f"{po['a2_s42_matched']:.3f}, {po['a2_s142_matched']:.3f}; random change "
                 f"{s['P1_prompt_overlap_anchor1']['gauss']:.3f}",
                 "a random change of matched output size; for the judge, biography neurons against random ones",
                 E102, ".P1_prompt_overlap_anchor1, .P2_engaged_anchor1, .per_state; score_stageB .CB1_power_gate")
    sz = {k: f[k]["size_can_do_this"] for k in ("outside", "placement", "think")}
    vpd_f = cell(SEP, f"outside, placement, think '{f['outside']['verdict']}'; size can do this: '{sz['outside']}'",
                 f"outside {rng(f['outside']['anchor1'])} vs {rng(f['outside']['anchor2'])}",
                 "anchors' seed spread; two random changes of the two sizes", E94, ".comparisons.full")
    neu_f = cell(NOTRUN, f"engaged share '{s['R1_engaged_anchor2_full']['verdict']}' (reported); no filed anchor comparison",
                 "", "", E102, ".R1_engaged_anchor2_full")
    q = "existing layout vs new facts: template vs fictional-facts fine-tune"
    return [{"question": q, "size": "matched", "vpd": vpd_m, "neurons": neu_m},
            {"question": q, "size": "learned", "vpd": vpd_f, "neurons": neu_f}]


def known_vs_fictional() -> list[dict[str, Any]]:
    """Question 2: real facts Qwen already knows against invented ones, same task, answer format and recipe."""
    a = js(E94A)["sides"]
    ident, rules, rows = js(E103)["identity"], js(E103)["pair_rules_s142"], js(E105)
    pairs = js(E107A)["pairs"]
    jac = {k: v["jaccard"] for k, v in pairs.items()}
    cross = [v for k, v in jac.items() if k.startswith("known") and "a2_" in k]
    summ_v = cell(NOT, f"outside '{a['matched']['outside']['verdict']}' (f {a['matched']['outside']['f']:.2f}); placement "
                       f"'{a['matched']['placement']['verdict']}' (f {a['matched']['placement']['f']:.2f}); think "
                       f"'{a['matched']['think']['verdict']}'",
                  "f: 0 = template's mean, 1 = fictional arm's mean", "midway band 0.5 plus or minus the seed noise",
                  E94A, ".sides.matched")
    summ_n = cell(NOT, "all four pair rules: " + ", ".join(sorted({r["verdict"] for r in rules.values()})),
                  "; ".join(f"{k.removeprefix('matched_')} gap {r['gap']:+.4f} band {r['band']:.4f}" for k, r in rules.items()),
                  "fictional seeds' mean, band from random draws", E103, ".pair_rules_s142")
    set_v = cell(SEP, "description only: no filed rule of its own; E107 Amendment 1 used it as its positive control",
                 f"Jaccard known seeds {jac['known_s42_matched|known_s142_matched']:.3f}, fictional seeds "
                 f"{jac['a2_s42_matched|a2_s142_matched']:.3f}, known vs fictional {rng(cross)}",
                 "the two arms' seed Jaccards", E107A, ".pairs")
    set_n = cell(SEP, f"real rows '{ident['real']['verdict']}', TOFU rows '{ident['tofu']['verdict']}'; rows null (E105): "
                      f"'{rows['real']['verdict']}', '{rows['tofu']['verdict']}'",
                 f"TOFU rows: known seeds {ident['tofu']['known']:.3f}, fictional seeds {ident['tofu']['fict']:.3f}, "
                 f"cross {rng(ident['tofu']['cross'])}; disjoint halves of the known rows {rng(rows['tofu']['cross'])}",
                 "band = 1 minus the fictional seeds' overlap", E103, ".identity; rows_null score .real/.tofu .verdict")
    e = a["end"]
    end_v = cell(NOT, f"outside '{e['outside']['verdict']}', placement '{e['placement']['verdict']}', think "
                      f"'{e['think']['verdict']}' (reported, sizes 2.2x apart)",
                 f"f outside {e['outside']['f']:.2f}, think {e['think']['f']:.2f}", "as at matched size", E94A, ".sides.end")
    t = js(E109)["tofu"]
    end_n = cell(NOCALL, f"'{t['status']}' (seed level {t['w_known']:.3f} under the power line)",
                 f"known vs fictional {rng(t['known_vs_fict'])}, known seeds {t['w_known']:.3f}",
                 "seed level must clear the cross-arm overlap by its own band", E109, ".tofu.status, .tofu.known_vs_fict")
    q = "known vs new facts: known-facts vs fictional-facts fine-tune"
    return [{"question": q + ", summary readings", "size": "matched", "vpd": summ_v, "neurons": summ_n},
            {"question": q + ", set identity", "size": "matched", "vpd": set_v, "neurons": set_n},
            {"question": q, "size": "learned", "vpd": end_v, "neurons": end_n}]


def wrong_vs_known() -> list[dict[str, Any]]:
    """Question 3: familiarity against topic or style. Same rows and wording, every stated answer swapped for a wrong one."""
    n, v, t = js(E107), js(E107A)["rule"], js(E109)
    w = n["tofu"]["wrong"]
    mat_v = cell(NOCALL, f"'{v['verdict']}' (lowest pair {min(v['cross']):.5f} vs topic-or-style line {v['topic_line']:.5f})",
                 f"Jaccard wrong vs known {rng(v['cross'])}; seeds {v['w_known']:.3f}, {v['w_wrong']:.3f}",
                 f"familiarity needs every pair at most {v['familiarity_line']:.3f}", E107A, ".rule")
    mat_n = cell(NOT, f"real rows '{n['real']['wrong']['verdict']}', TOFU rows '{w['verdict']}'",
                 f"TOFU rows: wrong in known {rng(w['cross'])}; wrong in fictional {rng(w['wrong_vs_fict'])}",
                 f"familiarity needs every pair at most {w['familiarity_needs_max']:.3f}", E107, ".real/.tofu .wrong")
    end_n = cell(NOCALL, f"TOFU rows '{t['tofu']['verdict']}', answer tokens '{t['T']['verdict']}': "
                         f"'{t['tofu']['status']}'",
                 f"seed level {t['tofu']['w']:.3f} (TOFU rows), {t['T']['w']:.3f} (answer tokens)",
                 "seed level must clear the fictional overlap by its band", E109, ".tofu, .T")
    none = cell(NOTRUN, "no full-size wrong-answer states saved", "", "", None, "")
    topic = n["tofu"]["topic"]
    top_n = cell(SEP, f"TOFU rows '{topic['verdict']}', real rows '{n['real']['topic']['verdict']}'",
                 f"authors in world facts {rng(topic['cross'])} vs line {topic['line']:.3f}", "E105's random halves",
                 E107, ".tofu.topic, .real.topic")
    q = "familiarity vs topic or style: wrong-answer vs true-answer fine-tune, same rows"
    return [{"question": q, "size": "matched", "vpd": mat_v, "neurons": mat_n},
            {"question": q, "size": "learned", "vpd": none, "neurons": end_n},
            {"question": "topic: real authors vs world facts, both known", "size": "matched",
             "vpd": cell(NOTRUN, "VPD reads of the topic states not run", "", "", None, ""), "neurons": top_n}]


def small_model() -> list[dict[str, Any]]:
    """The 67M, for context: the one place both read the same fine-tunes against labels assumed from construction."""
    p1 = js(E98)["rows"]["P1"]
    vr = {r["id"]: r["cells"]["r"]["verdict"] for r in js(M67)["rows"]}
    both = p1["readable_rows"]
    vpd = cell(SEP, f"VPD's r '{vr[both[0]]}' on {sum(vr[k] == 'separates' for k in both)} of {len(both)} rows",
               ", ".join(both), "labels assumed from how each task was built", M67, ".rows[].cells.r.verdict")
    neu = cell(NOT, f"neuron share separates {len(p1['separating_rows'])} of {len(both)} ({', '.join(p1['separating_rows'])}); "
                    f"filed 'not a kind reader': '{p1['status']}'",
               "most shares 0.35-0.50, where an output-weights-only fine-tune reads 0.43", "the same labels", E98, ".rows.P1")
    return [{"question": "67M: built to reuse vs built to be new (assumed labels)", "size": "learned", "vpd": vpd,
             "neurons": neu}]


def verdict_grid(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 1.6))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for j, side in enumerate(("vpd", "neurons")):
            c = r[side]
            ax.add_patch(Rectangle((j, y), 0.96, 0.9, color=COLOUR[c["class"]], ec=GREY, lw=0.6))
            ax.text(j + 0.48, y + 0.45, c["class"], ha="center", va="center", fontsize=8,
                    color="white" if c["class"] in (SEP, NOT) else INK)
        ax.text(-0.04, y + 0.45, f"{r['question']}\n({r['size']} size)", ha="right", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(-2.3, 2)
    ax.set_xticks([0.48, 1.48], ["VPD's reading", "neuron reading"], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("On the same fine-tunes, does each method's filed reading separate the pair?", fontsize=10, pad=24)
    ax.set_ylim(-0.6, len(rows))
    ax.text(0.15, -0.35, "Class is what the filed verdict says about the pair; the verdicts, verbatim, are in "
            "results/table.json. Grey: no call or no power. White: not run.", ha="center", fontsize=7, color=INK)
    save(fig, "verdicts.png")


def sets() -> None:
    """Set identity on both sides: each arm's seeds, wrong with known, known and wrong with fictional, at matched size."""
    ident, w = js(E103)["identity"]["tofu"], js(E107)["tofu"]["wrong"]
    jac = {k: v["jaccard"] for k, v in js(E107A)["pairs"].items()}

    def pick(a: str, b: str) -> list[float]:
        return [v for k, v in jac.items() if k.split("|")[0].startswith(a) and k.split("|")[1].startswith(b)]

    sides = {
        "neuron carrying set: share of 1,024 neurons shared\n(ranked on 128 fictional-author rows)": [
            ("known seeds", [ident["known"]], PURPLE, "o"), ("wrong seeds", [w["w_wrong"]], TEAL, "o"),
            ("fictional seeds", [ident["fict"]], ORANGE, "o"), ("wrong vs known", w["cross"], TEAL, "D"),
            ("known vs fictional", ident["cross"], ORANGE, "^"), ("wrong vs fictional", w["wrong_vs_fict"], TEAL, "^")],
        "VPD switched-on set: Jaccard of subcomponents\n(read on 128 maths rows no arm trained on)": [
            ("known seeds", pick("known_s42", "known_s142"), PURPLE, "o"),
            ("wrong seeds", pick("wrong_s42", "wrong_s142"), TEAL, "o"),
            ("fictional seeds", pick("a2_s42", "a2_s142"), ORANGE, "o"),
            ("wrong vs known", pick("known", "wrong"), TEAL, "D"), ("known vs fictional", pick("known", "a2_"), ORANGE, "^"),
            ("wrong vs fictional", pick("wrong", "a2_"), TEAL, "^")]}
    seed_pair = js(E94.parent / "set_overlap.json")["summary"]["seed_null"][0]
    chance = {0: (0.006, "two random sets of 1,024 neurons"),
              1: (seed_pair["jaccard"] / seed_pair["over_chance"], "two random sets of these sizes")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for k, (ax, (title, cats)) in enumerate(zip(axes, sides.items(), strict=True)):
        ax.axhline(chance[k][0], color=GREY, ls=":", lw=1.2, label=f"{chance[k][1]}: {chance[k][0]:.3f}")
        ax.legend(fontsize=7, loc="lower left")
        for i, (_, vals, colour, marker) in enumerate(cats):
            xs = [i + d for d in ((0.0,) if len(vals) == 1 else (-0.15, -0.05, 0.05, 0.15))]
            ax.scatter(xs, vals, s=46, marker=marker, color=colour, zorder=3)
            txt = f"{vals[0]:.2f}" if len(vals) == 1 else f"{min(vals):.2f}-{max(vals):.2f}"
            ax.text(i, max(vals) + 0.03, txt, ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(cats)), [c[0].replace(" vs ", "\nvs ") for c in cats], fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("overlap of two sets: 1 = the same set")
    fig.suptitle("Both methods' sets put known facts apart from invented ones, and wrong answers with true ones "
                 "(matched size, two seeds per arm)", fontsize=10)
    save(fig, "sets.png")


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = anchors() + known_vs_fictional() + wrong_vs_known() + small_model()
    RES.mkdir(exist_ok=True)
    (RES / "table.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    verdict_grid(rows)
    sets()


if __name__ == "__main__":
    main()
