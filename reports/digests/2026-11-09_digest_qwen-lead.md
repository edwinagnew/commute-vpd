# qwen-lead, 2026-11-09: the re-gating share on the new decomposition

**Cells.** P3.1, due 11-11, filled, one run of one. No other qwen-lead cell is open, which is why this lead has finished;
part 3 waits for the part 2 gate on 11-18.

**The answer.** On the new decomposition the re-gating share still sits near its ceiling for every saved Qwen update
and for changes made at random, as it did on the old one. It does separate some updates beyond the spread between two
seeds, but by very little, and the seed spread comes from one pair. Because a random change also reads high, a high share
here does not show that an update re-gated. No verdict on reuse is given while the definition is open.

**What ran.** One job on the cluster's debug queue (job 56867, an L40, 85 minutes, no charge). It read the share over
128 maths rows for four saved updates and two random changes, the random changes first and matched to the size of the
step-10 fine-tune. Every control held: the original weights read as the fine-tuned model gave identical cells, the
fine-tune's KL matched the earlier reading, and the do-nothing reference read exactly 1.

**What it found.** See `experiments/qwen_regating_new/figures/share_new_old.png`, new beside old.
- All six states read 0.991 to 0.9992 on new, against 0.993 to 0.9995 on old.
- The step-10 fine-tune and the gold update separate from both anchor 2 seeds, the same four pairs as on old. They do
  not separate from each other.
- New here: all four updates separate from both random changes. On old only the fine-tune and gold did.
- The filed expectations held: random changes at least 0.8 and updates at least 0.98.

**For your gate.** The premise check returned "doubtful": with random changes this close to the ceiling, the share has
little room to say anything about what an update taught, and it changes no candidate map's status. Its question for you
is what a near-ceiling reading on a random change means, and which conclusion depends on differences among these six
readings. Full text: `~/commute-data/premise/P3.1.md`.

**Asks waiting on you.** None. **Money.** $0 on the cluster; no pods. **Friction.** One line, number 648: the start-up
report showed a finished cell as having no runs left while its job was still live.

**Not run.** The RL updates and anchor 1 (no old reading to set beside); a second seed pair; gate counts, placement nulls
and overlaps on new.
