cell: P1.52
rival_a: the adopted definition (option b, read by write) is featurization-independent, so the reuse built edits keep their VPD order at the SAE latents, which predicts x2, x10 and rebuilt reading E_b at least 0.2 above both new edits and every random update at the SAE featurization
rival_b: E_b measures how well a featurization's parts line up with the update rather than reuse, which predicts the built order lost at the SAE latents: some reuse edit (x2, x10 or rebuilt) within 0.2 of a new edit or a random update in E_b
rung: premise
smallest_gap: 0.2
spread: 0.0014
n: 3
design: unpaired
unit: built edits per side (reuse: x2, x10, rebuilt; against new_few, new_thin and the random updates)
spread_source: experiments/rb_states/results/items
spread_of: proxy: E_b at VPD across the three seeds of each E237 key task (sd about 0.001-0.0014), the same reading at the other featurization on states that differ only by seed
floor: none
stop: the order lost at SAE (rival b) ends the claim that the definition reads the same at any featurization; it goes to definition-lead and writeup-lead as a limit of the definition, with no further run here
sketch: experiments/sae_eb/figures/sketch_card.png
rehearsal: cluster only: one cluster debug pass per saved state (E245's states were read on cluster debug L40s, 1.3 s per pass over D, 14.5 GB), no pod
