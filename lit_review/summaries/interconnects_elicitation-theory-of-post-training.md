# Elicitation, the simplest way to understand post-training

## Metadata
- **Author**: Nathan Lambert (Interconnects AI; Ai2 post-training lead)
- **Date**: 2025-03-10
- **Source**: blog post, https://www.interconnects.ai/p/elicitation-theory-of-post-training
- **Kind**: opinion essay with practitioner evidence; no experiments of its own, nothing inside the model
- **Read**: 2026-11-19 by writeup-lead, from the page text (user's link). Quotes below are from the page.

## Summary
Lambert states the "elicitation interpretation" of post-training: "all we are doing is extracting and amplifying
valuable behaviors in the base model." The base model is "the platform on which post-training is built", and
post-training's gains are large, fast and come early, which he reads as drawing out what pretraining put there rather
than teaching. He positions it against two views: the "prevailing counterargument" that post-training is "teaching
specific skills", supported by multi-million-example datasets, and LIMA's Superficial Alignment Hypothesis, which he
says gets "important intuitions right but for the wrong reasons": post-training is not "superficial", since changing
the data changes performance a lot.

## Key claims and evidence
- **The claim.** "The astute way to view the model learning new behaviors is not that it is learning entirely new
  abilities but rather learning to reinforce behaviors that were already latent in the pretraining dataset."
- **F1 analogy.** Teams arrive with a chassis and engine (pretraining) and gain more over the season from aerodynamics
  and tuning (post-training) than between chassis.
- **Evidence, all behavioural and anecdotal**: OLMoE's post-training average rose from 35 to 48 without touching most of
  pretraining; on OpenMathInstruct 2 (up to 14.3M instructions) "the vast majority of the gains come from the beginning
  of training"; a few thousand RL prompts surpass that MATH score; bigger base models "can absorb far more diverse
  changes"; Amodei's remark that most of Claude 3.6/3.7's gains came in post-training.
- **Concessions.** "you *can* teach new skills to models in post-training" (citing Raghavendra et al., Revisiting the
  Superficial Alignment Hypothesis, arXiv:2410.03717); "The hardest part today is to know when the skills are entirely
  new — it is often safer to assume the model has seen them at pretraining"; "it is far more efficient to elicit
  behaviors from the model than to teach, and eventually, the tide may shift to favor teaching models".

## Relevance to the project
**Score: 7** (framing and citation, not method).
- It is the source of the word: the paper's title is "Mechanistic Evidence for the Elicitation Hypothesis", and
  `elicitation_vs_sharpening.md` traced Oh et al.'s use of "elicitation" to this post but had not read it.
- It is the behavioural statement our (EH) makes mechanistic. Lambert's own caveat is our gap: he says one cannot tell
  "when the skills are entirely new" and defaults to assuming they were latent. Our definition is a way to tell, from
  inside the model, on fine-tunes whose answer is known by construction.
- His "latent behaviours" are behaviours the base model could already produce; he never says which parts of the network
  carry them, so the post gives no criterion our definition must agree with, only the hypothesis to test.
- His distinction from LIMA is useful for the related-work line: LIMA says post-training changes only style and format;
  Lambert says it can change performance a lot while still only drawing out what is there. Our (EH) is closer to
  Lambert's (large gains, existing mechanisms) than to LIMA's (small, surface-level changes).

## For the (EH) citation (paper's open mark under §1, candidates in refs.bib)
Suggested: cite Lambert (2025) for the name and the informal statement, beside `zhou2023lima` for the
superficial-alignment antecedent, and Raghavendra et al. (2024) as the counter-evidence that post-training can teach
new skills. Neither Lambert nor Raghavendra is in refs.bib yet. Not added: the user is editing the paper (2026-11-19).

```bibtex
@misc{lambert2025elicitation,
  author       = {Lambert, Nathan},
  title        = {Elicitation, the simplest way to understand post-training},
  howpublished = {Interconnects (blog)},
  year         = {2025},
  month        = mar,
  url          = {https://www.interconnects.ai/p/elicitation-theory-of-post-training}
}
@article{raghavendra2024revisiting,
  author  = {Raghavendra, Mohit and Nath, Vaskar and Hendryx, Sean},
  title   = {Revisiting the Superficial Alignment Hypothesis},
  journal = {arXiv preprint arXiv:2410.03717},
  year    = {2024}
}
```
(Raghavendra et al.'s author list is from the arXiv record as remembered; check it before use.)

## Further sources it points to, not yet in the lit review
- LIMO, Less is More for Reasoning, arXiv:2502.03387 (LIMA's idea for reasoning).
- s1, Simple test-time scaling, arXiv:2501.19393.
- Lambert's RLHF book (arXiv:2504.12501), which carries the same view at more length; not read.

## Limitations
No measurement of what changes inside the model; the evidence is benchmark curves and practitioner experience, and
"latent" is never defined. It cannot support or refute (EH) as we state it; it motivates it.
