# Feature Evolution and Migration during Vision Transformer Training

## Metadata
- **Authors**: Joonas Järve, Tarun Khajuria, Halil Ibrahim Aysel, Meelis Kull (University of Tartu)
- **Year**: 2026
- **Source**: arXiv:2608.20134 (extended version of a CIKM 2026 paper)
- **DOI/URL**: arXiv:2608.20134; conference DOI https://doi.org/10.1145/3799682.3841045; code at https://github.com/joonasrooben/VIT-Feature-Evolution-and-Migration

## Summary
The paper tracks visual features in a ViT-Tiny trained from scratch on ImageNet-1k and a 10-superclass subset, using a two-dimensional epoch-by-layer map. It trains a sparse autoencoder (SAE) at every checkpoint and layer on CLS-token activations, then matches SAE features across (checkpoint, layer) pairs using two similarity scores (a fixed reference SAE and independently trained SAEs). It defines "feature migration" as a shift in the layer where a feature is most detectable over training.

## Key Findings
- Migration is most frequent early in training and tends toward earlier layers; migration toward deeper layers is rare (peaks around 4% of reference points, Sec. 4.3).
- Deeper layers stabilise earlier and more strongly than shallow layers; features in early layers have longer active lifetimes (Sec. 4.3, Sec. 5).
- Later-epoch reference features contain more class- and superclass-specific features (Table 1: class-specific counts rise from 70 at epoch 1 to 461 at epoch 300), but feature type shows no clear correlation with migration behaviour (Sec. 4.3).
- Label-entropy differences between the two datasets are large; the authors speculate this may be due to the SAE expansion factor (5x) (Sec. 4.3, Sec. 5). They also note the datasets differ in label granularity and test-set size, so dataset scale is not isolated.

## Methodology
- Extract CLS-token activations at each epoch and layer of a ViT-Tiny (300 epochs, 12 layers, 12 heads).
- Train a BatchTopK SAE (K=32, expansion 5, 150 epochs, fixed by compute) per checkpoint-layer pair; each latent is a "candidate feature".
- Similarity between features: (i) fixed-SAE transfer, applying the source SAE's latent to the target representation and computing Spearman correlation; (ii) independent-SAE best-match Spearman correlation. Final score is the maximum of the two.
- Migration metrics: thresholded layer distribution, layer centre-of-mass, signed mean drift, stability (occupancy within one layer), normalised entropy as localisation, and emergence/disappearance epochs. Feature classes use fixed cutoffs (drift > 1 layer for migratory, < 0.5 layer and stability >= 0.7 for stable).
- Feature labels come from a heuristic ImageNet class-distribution rule plus a WordNet hypernym check, with manual inspection.

## Relevance to Proposal
**Score: 5**

This is the closest match found so far to the "training dynamics of features / continual interpretability" strand: it tracks feature identity across checkpoints with SAE-based matching, and its two-view (fixed vs independent SAE) matching scheme is a direct analogue of asking whether a decomposition from one checkpoint still reads out the next. It does not touch parameter decomposition, weight-space updates, or fine-tuning (the authors list fine-tuning as future work), so it cannot test whether a small update can be carried by existing components.

## Limitations
- Only supervised training on ViT-Tiny from scratch, with one seed reported; the authors say results across seeds, datasets, architectures and optimisers are unverified (Sec. 6).
- Analysis is restricted to linearly decodable features; SAE latents may be unstable, split, or fail to be rediscovered, and the paper acknowledges that absence of a match does not guarantee disappearance (Sec. 3.1, Sec. 6).
- SAEs trained for a fixed 150 epochs rather than to convergence, due to compute (Sec. 4.2).
- Manual and heuristic feature labelling; the authors note the classification is partly subjective (Sec. 4.1, Sec. 6).
- Spearman similarity is sensitive to ties at zero activations (Sec. 6).
- Dataset comparison confounds scale with label granularity and test-set size (Sec. 5).

## Key Quotes
> "Our method instead studies an interpretable feature-level projection of the same training trajectory and, therefore, is complementary to this weight-space perspective." (Sec. 5)

> "Therefore, the final method uses the maximum of the two." (Sec. 3.1, on combining fixed-SAE and independent-SAE similarity)

> "Second, the analysis is restricted to linearly decodable features." (Sec. 6)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
