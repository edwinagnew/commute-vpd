# Layer-Wise Evolution of Representations in Fine-Tuned Transformers: Insights from Sparse AutoEncoders

## Metadata
- **Authors**: Suneel Nadipalli
- **Year**: 2025 (arXiv 2502.16722, v1)
- **Source**: arXiv preprint (no venue stated in the file)
- **DOI/URL**: https://arxiv.org/abs/2502.16722 (URL inferred from the arXiv ID in the filename)

## Summary
The paper fine-tunes BERT-base-uncased (and smaller BERT variants) on three text classification tasks (IMDb sentiment, Spotify ratings, news topics). It compares pre-trained and fine-tuned activations by per-layer cosine similarity of mean-pooled hidden states, and by training a separate sparse autoencoder (SAE) on layers 3, 6 and 12 of each checkpoint and inspecting per-token activations of selected features. The authors conclude that early layers keep general features, middle layers are transitional, and late layers specialise to the task.

## Key Findings
- Cosine similarity between pre-trained and fine-tuned mean-pooled activations is near 1.0 in early layers and drops in deeper layers, most steeply beyond layer 10 for bert-base; the News task shows the steepest drop (Sec. 4.2, Figs 1-2).
- Smaller BERT variants (8, 4, 2 layers) show higher overall similarity, which the authors read as shallower models changing less (Sec. 4.2).
- In hand-picked examples, layer 3 features move from sparse, general-word activations (pre-trained) to domain/sentiment words after fine-tuning; layer 12 features show sharper sentiment contrast (Sec. 4.3, Table 1). These are qualitative observations on a few sentences per dataset.
- Reported accuracies (Table 2): fine-tuned IMDb 93%, Spotify 57%, News 93%; the "Pre-Trained" column reads 54%, 12%, 12%. The paper does not define what the pre-trained column measures.

## Methodology
- Fine-tuning: bert-base-uncased, Adam, lr 2e-5, 10 epochs, on each of three datasets truncated to about 25k rows (80/10/10 split) (Sec. 3.2.1, 3.2.2).
- Similarity: hidden states from layers 1-12 are averaged over tokens and then over samples, and cosine similarity is computed between pre-trained and fine-tuned vectors (Sec. 3.2.2).
- SAEs: one SAE per (checkpoint, layer) with a 768 to 1024 encoder and a linear decoder, trained with MSE plus an L1 sparsity penalty (lambda = 1e-3) (Sec. 3.2.3). Pre-trained and fine-tuned SAEs are trained separately.
- Feature inspection: the top 3 features by variance across samples are chosen and their token-level activations are plotted for hand-picked test sentences (Sec. 3.2.4).

## Relevance to Proposal
**Score: 3**

The paper touches strand 2 of the sweep (SAEs as a lens on what fine-tuning changes) and the question of whether fine-tuning reuses or rewrites features, but it does not decompose weights, does not study any update map, and does not look at commuting an update past a decomposition. Its evidence is too weak to inform the "does post-training strengthen or create mechanisms" question. Useful mainly as a cautionary example of how not to measure this.

## Limitations
- Each SAE is trained separately on the pre-trained and fine-tuned activations, so feature index k in one model is not matched to index k in the other. The "same feature, before and after" comparisons in Sec. 4.3 therefore have no stated basis.
- Cosine similarity is computed on averaged activations, which is a coarse measure that may be dominated by shared mean direction (anisotropy); values near 1.0 say little about feature-level change. No seeds, variance, or significance tests are reported.
- SAE quality is reported only as "loss on the order of 1e-4". No L0/sparsity, dead-feature rate, or reconstruction fidelity is given, and the chosen lambda is not justified.
- Feature interpretation rests on a few hand-picked sentences per dataset and is subjective. Nothing checks that the top-variance features are monosemantic.
- Preprocessing lowercases text and removes punctuation (Sec. 3.2.2), yet Sec. 4.3 interprets activations on punctuation and [CLS]/[SEP] tokens as central results. This is internally inconsistent.
- Layer-3 description is inconsistent: Sec. 1 says it captures "broad semantic structures" that remain unchanged, while Sec. 4.3 says it becomes task-specific after fine-tuning.
- Spotify classification report contains an implausible recall value ("0/40"), and the text does not match Fig. 6 clearly. Only one fine-tuning run per dataset is reported.
- Reference [8] is empty in the reference list, so the "Inspired by [8]" basis for the method is unverifiable from the paper.
- No intervention or causal test: the paper does not show that the features it highlights drive the task output.

## Key Quotes
> "early layers primarily retain general representations, middle layers act as a transition between general and task-specific features, and later layers fully specialize in task adaptation." (Abstract)

> "the activation values for each sample were averaged, producing a single representative activation vector for each dataset-layer-size combination." (Sec. 3.2.2)

> "The SAEs were trained using the Adam optimizer with a learning rate of 2*10<sup>-5</sup> over 10 epochs. They demonstrated strong performance, consistently reaching a loss on the order of 10<sup>-4</sup>." (Sec. 3.2.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper, no comments).
