# Appendix to the general-definition page: survey table, design check, what was not read

*definition-lead, 2026-11-09. Companion to `2026-11-09_plan_general-definition.md`. Full per-paper notes with verbatim quotes
are in `lit_review/definition_survey/{transfer,finetune,causal,mechanism}.md`. PDFs and extracted text are in
`~/commute-data/lit_review_pdfs/survey/<id>/`.*

## Design check

- **Goal.** `GOALS.md` asks whether post-training teaches new capabilities or strengthens existing ones. This page says what
  that question means, for any model, before any measurement.
- **Nearest tried.** The frozen readout wording (decisions rows 81, 87, 164; E116, E112, E113) is close to candidate 1 and read
  on 4 of 17 kinds. The frozen-body page is the rejected nearest neighbour: it labelled the task, not the fine-tune. E66 tried a
  context patch and stopped at its filed control, so no cross-network swap has been read.
- **Patch or step.** A step: the first part of the paper plan. Zero-run.
- **Decisions relied on or departed from.** Would replace rows 81, 87 and 164 if adopted, by a dated row only the user adds. Keeps
  "must look inside the model" (user, 2026-11-09), which rules out Lubana et al. as a candidate.
- **Reads under an adopted definition?** None is adopted, so the last section of the page reports records and gives no verdict.
- **What result would change what we do.** The user's choice decides which runs get proposed.
- **Control first; numbers.** None run. Each candidate's control is named on the page, for the later step.

## Survey

Kind: **F** formal definition, **I** informal, **O** operational measure only, **E** finding only. "Differs?" asks whether two
fine-tunes of one model that gain the same behaviour could get different answers. Page numbers are PDF pages.

### Fine-tuning

| Source | In the authors' words | Mechanism is | Having it is | Kind | Differs? | Read |
|---|---|---|---|---|---|---|
| Jain et al. 2024, 2311.12786 | "possesses a capability C" if for all x in the sub-domain some layer's linear readout equals f_C (p. 3); Definition 2 grades relevance of g applied to f_C (p. 4) | A map readable inside | Linear readout on pretraining data | F, "idealized" | Mixed: C is about M | Sec. 1-5 and notes |
| Prakash et al. 2024, 2402.14811 | "use a different or the same circuit, i.e. with exactly the same group of heads" (p. 5); "implement the same mechanism" (p. 8) | Set of heads plus their roles | Base circuit recovers F's accuracy, roles match | O | Yes: 0.97 against 0.88-0.89 | Main text |
| Lubana et al. 2023, 2211.08422 | "called mechanistically similar if I(θ1) = I(θ2)" (p. 4) | Input attributes a model is invariant to | Same invariance set | F | Yes, against a chosen list | Main text |
| Panigrahi et al. 2023, 2302.06600 | "A grafting of θft in the region γ onto θpre" (p. 2) | Sparse parameters of the change | Not addressed | O | Yes | Main text |
| Lee et al. 2024, 2401.01967 | "not removed, but rather bypassed" (p. 1) | Toxic value vectors | Present and unchanged after DPO | E | Yes | Main text |
| Wang et al. 2025, 2502.11812 | "their edges undergo significant changes" (p. 1) | Top 5% of edges | Circuit at base checkpoint | O, no cutoff | Yes | Main text |
| Chhabra et al. 2025, 2503.01896 | "without introducing novel mechanisms" (p. 4) | Function of a head class | Known IOI circuit | I | Unclear | Main text |
| Merullo et al. 2023, 2310.08744 | "the proportion of shared nodes in the circuit" (p. 28). Two tasks, one model, no fine-tune | Heads with roles | Not applicable | O | No | Main text, parts of appendix |
| Jain et al. 2024b, 2407.10264 | "a component that cannot be computed by W_IT itself" (p. 7) | The weight change's column space | Instruction-tuned weights | E | Yes | Main text |
| Minder et al. 2025, 2504.02922 | "inability to distinguish between truly novel latents" and shifted ones (p. 10) | Crosscoder latent | Decoder and latent scaling thresholds | O | Unclear | Main text |
| Ward et al. 2025, 2507.12638 | "repurposes pre-existing representations to form new behavioral circuits" (p. 1) | Base-model direction | Direction from base steers F, not base | E | Yes | 8 pages |

### Transfer learning

| Source | In the authors' words | Mechanism is | Having it is | Kind | Differs? | Read |
|---|---|---|---|---|---|---|
| Raghu et al. 2020, 1909.09157 | "already contains highly useful features that can mostly be reused as is" (p. 2-3) | Layer activations | Freezing costs nothing; similarity near 1 | I, O | Partly | Main text |
| Raghu et al. 2019, 1902.07208 | None; "any serious feature reuse is restricted to the lowest couple of layers" (p. 7) | Layer representations | Similarity to initialisation | O | Yes | Main text |
| Neyshabur et al. 2020, 2008.11687 | None; "reusing the pre-trained feature hierarchy" (p. 3) | Feature hierarchy | Similarity, distance, basin | E | Mixed | Main text |
| Kumar et al. 2022, 2202.10054 | "feature distortion" (p. 7), distance between extractors (p. 5) | Feature extractor | Close to a true extractor | F, linear toy | Yes | Main text |
| Yosinski et al. 2014, 1411.1792 | "degree of generality of a set of features" (p. 3) | First n layers | Copied layers train as well on B | O | No | Main text |
| Bansal et al. 2021, 2106.07682 | "stitching-connected if they can be stitched to each other at all layers" (p. 7) | Lower-layer representation | Low stitching penalty | F | Yes | Main text |

### Causal abstraction and circuits

| Source | In the authors' words | Mechanism is | Having it is | Kind | Two networks? | Read |
|---|---|---|---|---|---|---|
| Geiger et al. 2021, 2106.02997 | "constructive abstraction" (p. 3, Def. F.7) | High-level causal model | Abstracted under a chosen alignment | F | No | Main text, App. F |
| Geiger et al. 2025, 2301.04709 | Definitions 5, 25, 31, 33, 41, 44, 48; "consistent descriptions of the same causal situation" (p. 14) is of two models at different levels | Causal model | Exact or η-approximate abstraction | F | No | Definitions and limits |
| Geiger et al. 2024, 2303.02536 | Definition 1 (p. 4); interchange accuracy as a graded notion (p. 1) | Variables in subspaces | High interchange accuracy under some rotation | F, O | No | Most |
| Sutter et al. 2025, 2507.08802 | "any neural network can be mapped to any algorithm" (p. 1) | Algorithm | Distributed abstraction | Theorem | No | Main text |
| Makelov et al. 2023, 2311.17030 | "a definition of a 'ground truth' mechanistic explanation is conceptually challenging" (p. 26) | Subspace | Patch plus extra checks | E | No | Parts |
| Beckers, Halpern 2019, 1812.03789 | Definitions 3.12-3.19; "what counts as an inessential difference" (p. 1) | High-level model | τ-abstraction | F | No | Main text |
| Wang et al. 2022, 2211.00593 | Faithfulness, completeness, minimality (p. 9-10) | Subgraph | Passes the criteria, no thresholds | O | No | Parts |
| Mueller et al. 2025, 2408.01416 | "a causal chain from cause X to effect Y" (p. 5) | Task causal graph | Mapped mediators | I | No | Parts |

### What a mechanism is

| Source | In the authors' words | Kind | Read |
|---|---|---|---|
| Machamer et al. 2000 | "entities and activities organized such that they are productive of regular changes" (p. 3) | I | Sec. 1-2, 5.3 |
| Anderson 2010 | A "working" against a "use" (journal p. 252) | I | Sec. 1-4 |
| Saphra, Wiegreffe 2024, 2410.09087 | A causal mechanism "transforms some subset of model variables" (p. 2) | I | p. 1-6 |
| Olah et al. 2020 | "A 'circuit' is a computational subgraph of a neural network" | I | Claims only |
| Lindsey et al. 2024 | Shared and model-specific features by relative decoder norm | O | Model-diffing section |

## Not read

- Appendices and proofs of nearly every paper, listed per paper in the four notes.
- Anderson 2010 Sections 5-7 and commentaries; Olah et al.'s glossary; Saphra and Wiegreffe pp. 7-19.
- About ten catalogue papers known only from project summaries, marked as such in `mechanism.md`; Venhoff et al. 2510.07364, not fetched.
- Lenc and Vedaldi 2015; Chatterji et al. 2020: cited, not fetched.

Quotes used on the page were matched against the extracted text by search. Extraction lost line breaks and some maths symbols; the
figures in the PDFs were not seen.

## Corrections to earlier claims

- `Raghu et al. 2020` is real and says what the rejected page assumed, but it concerns meta-learning, not ordinary fine-tuning.
- The plan's "causal step from Prakash et al." does not exist: that paper has no switch-off in a fine-tune and no shuffled twin.
