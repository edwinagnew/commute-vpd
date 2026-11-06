# D2, day 2 of 13, row 11-06: moved; qwen-rl-lead finished until the last two columns of D2's table land

**Why finished.** D2's answer is written, read independently and in the log. The only thing left is to add two columns
(teacher training, adjoint split) when qwen-decomp-lead's runs fill them.

**What D2 reads so far (provisional).** We read an RL update and an SFT update of Qwen3-1.7B through the same table.
RL reads like the same loop with shuffled rewards on every reading, within seed noise. Fine-tuning on the model's own
correct answers differs from RL only through how big the update is and which weight matrices it sits in. One reading
looked like an exception: the overlap of RL's switched-on subcomponents with a reference set read on answers deep in a
long context. An independent check and a measure of how many subcomponents the two seeds share put it within counting
noise, under two standard deviations, so no third seed is run. Report `reports/analyses/2026-11-06_analysis_d2-sft-vs-rl.md`;
log card D2 (v261).

**Also today.** Card `decision-rl-extra-reads` closed: neither extra run is needed (decisions.md, 2026-11-06). At the
user's word, this lead does not take part in E111. A note to qwen-decomp-lead lists the existing code it can reuse.

**Pending decisions for the user.** None from this lead.

**Money.** $0. No pod, no Slurm jobs of this lead. **Blocked:** none.

**Friction noted:** 571 (cpu-run queue held check.py ~30 min), 573 (`cpu-run python` not found; empty `results/large`
symlink for E104's files in this clone).

**Restart with** `~/commute-agents/fleet/start qwen-rl-lead` when E111 or E93 Amendment 1 lands: rerun
`experiments/qwen_rl_reads/plot_d2.py` with their rows and update the report.

**Not run.** A third seed of RL and the null (not needed). A second seed of own-samples SFT (not needed). A bootstrap
noise model for the overlap folds.
