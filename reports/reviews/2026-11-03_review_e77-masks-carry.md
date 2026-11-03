# Review: does the masked forward barely read its input while the masks carry the output?

*2026-11-03, independent check by reading code and per-row sums with `jq`. No model load, no GPU job.*

**Plain words.** I found no bug, and every number I recomputed matches the files. With all matrices masked, the forward responds weakly to small changes in its stream, and a row's masks keep most of that row's top tokens under another row's text, but nothing shows it ignores its tokens. The hook control's gap comes from the final norm, read at two different base points, so the expectation was wrong and the other reads stand.

## Verdict by part

| part of the statement | verdict | why |
|---|---|---|
| A change in the stream barely passes, all 196 matrices masked | **Stands** | Own-output change at blocks 0, 7: amplitude 0.05-0.11. Gaussians (E74): 0.03. |
| "Barely reads its **input**" | **Not shown** | No read changes the tokens by a small amount, and the one large change has no reference (below). |
| The masks hold enough of row A's top tokens | **Stands, as a mean** | Under row B's tokens, KL to clean A is 0.51 nats (u1), 0.57 (u3). |
| "The masks carry the output" as the only carrier | **Not shown** | The sums are means over positions, and a mixture of A and B fits them. |
| It describes the masked forward in general | **Does not stand** | True only near all-routed: with 12 of 196 matrices masked, KL passed is 0.47 (u1, fixed). |
| A bug in hook or read | **None found** | See "Code". |
| Artefact of row pairing | **No** | Rows share no prefix (below). |
| Artefact of KL's tails | **Yes, for follow_B and the swap's label** | follow_A's direction survives; follow_B does not separate anything. |

**Suggested wording.** "With all 196 matrices masked by the original model's masks, Δ off, small changes in the stream barely reach the logits at early blocks. With another row's tokens under them, those masks keep the output within about 0.5 nats of the masks' own row. How much the output still depends on its tokens is not measured."

## Code

Every path goes through one line, `qwen_transmission/run.py:184`, `((x V)·(m − d)) U + d·(x Wᵀ)`.

- m = 1, d = 1 is an exact identity at every site: the k = 0 point reads τ, amplitude and cosine of exactly 1 on both updates and modes.
- m = g, d = 0 reproduces E74: the swap's own-masks K0 is 549.594611 (u1) and 563.681573 (u3), equal to E74's to all printed digits.
- `swap_read` takes masks from row i and tokens from row i + 1, wrapping, and each KL has the filed reference (`amend1.py:63-69`).
- `hook_read`'s two arms do add the same vector at the last matrix (next section).
- The own-output term scales as it should: at the last matrix KL over ε² reads 0.39, 0.50, 0.49 over the fit rounds.

## The hook control's gap

Control: only `layers.27.mlp.down_proj` masked. Read: amplitude 0.59, cosine 0.86, τ 0.63, where the plan expected near 1. It passed the filed threshold of 0.513, so the job went on, but it missed its expected value.

**What follows the matrix.** `Qwen3MLP.forward` returns the matrix's output. `modeling_qwen3.py:276` adds it to the residual, `:421` applies `model.norm`, and `:494` applies `lm_head`. So only one nonlinear step follows, the final RMS norm (`:59-64`): z becomes weight × z / rms(z).

**Algebra.** Call the residual r, the clean output w = x Wᵀ, and the masked output h = ((x V)·g) U. Every other site reads clean, so x and r are the same in all four forwards.

| forward | input of the norm |
|---|---|
| clean reference | r + w |
| clean with the term (`scaled`, m = 1 + ε g, d = 1) | r + w + ε h |
| masked reference (`alone`, ε = 0) | r + h |
| masked with the term (`alone`, ε) | r + h + ε h |

The same ε h is added in both pairs. **The two logit shifts differ only because the norm is read at r + w in one and r + h in the other.** To first order the shift is ε h with its part along z removed, over rms(z). So two things can shrink it: a larger rms, or more of h lying along the base. In the masked pair h is part of the base, so the part left is minus r's part off r + h.

**So the expectation was wrong, not the hook.** "Near 1" holds only if h is close to w, and it is not close: masking this one matrix gives KL 0.045 per position, 67% of the all-masked 0.067. In KL terms the gap between h and w is about 0.3 of h (0.045 against ε = 1's extrapolated 0.49).

**Which of the two: cannot tell from saved numbers.** Three hints, none decisive:
- E74's Gaussian at this matrix, whose direction has no tie to the base, reads amplitude 0.95 on both updates. That argues against the rms and for the part along the base, but it was read with all sites masked.
- Row by row, cosine rises as K0 falls (correlation -0.52, intercept 0.93-0.98 at K0 = 0). Amplitude follows less well (-0.40, intercept 0.69-0.77), and its spread comes mostly from the clean side (correlation with the clean shift's squared norm -0.96).
- τ reads 0.63 where amplitude squared reads 0.35. Amplitude counts all 151,936 logits and KL counts the likely tokens, and the softer masked base also raises KL.

**Does it undermine the other reads?**

| read | effect |
|---|---|
| Swap, displacement | None: neither adds a term like this. |
| Routing | None on its controls. The whole update has no tie to h. |
| Own-output, read 1 | The bar moves. The most this term can read at the last matrix is 0.59-0.68, not 1, and a mid-stream norm can do the same. But blocks 0, 7 on u1 read 5-10 times below that, and Gaussians with no tie to the base also read low, so "fails" stands as a read of low gain. |
| Own-output, block 21 | Amplitude 0.48 sits under the filed 0.5, but block 27's seven matrices read 0.81, so this block is not clearly deaf. Blocks 0, 7, 14 are. |
| Own-output cosines | Clean and masked terms read different inputs, so a low cosine is not loss alone. |

**Smallest test, $0.** One L40, the first 32 rows, about a minute after load (`hook_read` took 20 s). Run the clean forward and the last-matrix-only forward once each, with a pre-hook on `model.norm` saving its input, and a hook on the last matrix saving h. Then, on those saved vectors only, apply the norm and `lm_head`:
1. Rebuild both shifts. They must read 0.59 and 0.86. If not, something other than the norm is involved and the hook is suspect.
2. Change one thing at a time in the masked shift: only the denominator from clean, then only the part removed from clean. Either change to clean should move it towards 1, and the two say which cause holds.
3. Save per position rms(r + h) / rms(r + w) and h's cosine with each base.

## The swap, recomputed

Per position, 32 rows × 256; L is the masked forward with row A's masks on row B's tokens.

| | u1 | u3 |
|---|---|---|
| K0, own masks | 0.067 | 0.069 |
| KL(A ‖ L) | 0.508 | 0.565 |
| KL(B ‖ L) | 12.74 | 12.01 |
| KL(A ‖ B), KL(B ‖ A) | 27.84, 27.80 | 24.76, 24.84 |
| **follow_A, by hand from the sums** | 4158.709 / 228044.004 = **0.01824** | 4630.497 / 202809.790 = 0.02283 |
| **follow_B, by hand** | 104360.116 / 227717.651 = **0.45829** | 98364.811 / 203479.643 = 0.48341 |
| follow_A, follow_B for a uniform output | 0.408, 0.408 | 0.455, 0.454 |

Both match `amend1.json`. The filed label "between" comes out: follow_B on u1 is below its band (0.481-0.519), so both sides are "below"; on u3 it is inside its band (0.471-0.529).

**Pairing.** No pair shares a prefix: on both row sets the first token already differs for every pair, wrapping included. Rows are packed from a stream, and a row starts mid-text. Only 2.5 (u1) and 3.75 (u3) of 256 positions hold the same token. Shared prompt text does recur, but at other offsets, and masks are matched by position. So the blind spot filed in `predictions_amend1.json` does not apply here, and pairing does not matter to the read. Only one pairing was run.

**KL's tails decide follow_B.** KL(B ‖ A) at 27.8 nats is more than twice the 11.93 nats of ln 151,936, the most a uniform output could be from any distribution. So the denominator measures how far below e⁻²⁰ the tail of one row sits at the other row's tokens.
- A uniform output would read 0.41, and L reads 12.74 nats from B against uniform's 11.35. On average L gives B's tokens less than a know-nothing output does.
- The filed rule's "masks" label on u1 needs follow_B above 0.519, or 14.4 nats, 3 nats worse than uniform. So "between" says little. It stands as filed, and its meaning is narrow.
- A softened copy of A would lower follow_B with no access to B at all. A temperature near 1.54 gives KL 0.51 from itself (E74's table, fine-tuned model). It would read follow_B about 0.64 by rough arithmetic. So softening could cover much of the drop from 1 to 0.46. The rest is either B's tokens or a softening that one temperature does not describe.

**What follow_A does show.** Under another row's tokens, cross-entropy of A's output under L is 1.09 nats against 0.58 of A's own entropy. Only masks can bring that in, so the masks do hold A's top tokens. The tokens still matter: KL rises 7.6 times, 0.067 to 0.508. That rise alone is about 130 times the 0.0034 the whole update moves the same forward. The two KLs have different references, so that is an order of size only.

**What the means cannot exclude.** Per-position KL was not saved. A 50:50 mixture of A and B at 56% of positions, with L close to A elsewhere, gives about 0.42 and 12.7 nats, which fits both sums. Top-token agreement would separate it.

## Routing, recomputed from the per-row sums

τ at k = 49 over τ at k = 196:

| update, masks | ratio | label |
|---|---|---|
| u1, fixed | 0.05306 / 0.005499 = 9.65 | no call (band 8.89-11.11) |
| u1, stochastic | 0.21844 / 0.012789 = 17.08 | steep |
| u3, fixed | 0.005472 / 0.000410 = 13.36 | steep |
| u3, stochastic | 0.03181 / 0.001074 = 29.62 | steep |

Overall "mixed", as filed. τ at k = 12 reads 0.47, 0.67, 0.14, 0.32, in the same order. Displacement recomputes too: 0.0078, 0.0089, 0.0005.

**Not asked, but it bears on the claim.** K0 does not fall as fewer sites are masked. Its sum on u1 with fixed masks:

| sites masked per position | 196 | 147 | 98 | 49 | 12 | the last matrix alone |
|---|---|---|---|---|---|---|
| K0 | 550 | 533 | 581 | 872 | 1156 | 371 |

So masking all sites is about half as far from clean as masking 12 random ones per position. That fits a forward that works as one self-consistent network, but also a library that trains 90% of documents all-routed. It does not separate the two.

## What to say about E77 next

Both amendments are used, so any of this is a new E-code, not a third amendment. The filed labels stand.

## Could not check

Anything on a GPU; g's values; per-position KL; which sites hold the information. The algebra above says nothing else lies between the matrix and the logits, but it is from reading, and test step 1 is what confirms it.

## Not run

1. The final-norm test above.
2. The swap's missing reference, in the same job: KL(B ‖ own-masks L on A's tokens) beside KL(B ‖ L). If they are about equal, B's tokens add nothing at B's tokens; if own-masks reads near 28, they add a lot. Add KL(own-masks ‖ L), and top-token agreement of L with A and with B per position.
3. A small input change under fixed masks, one token, read against the clean model's response.
4. Other row pairings; other mask sources (the filed "off the manifold" caveat stays open).
