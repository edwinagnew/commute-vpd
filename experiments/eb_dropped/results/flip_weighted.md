# Flip-weighted kept share E_flip beside E_b (P1.57, descriptive)

Filing: `flip_weighted_filing.md` (with its amendment). Written by `experiments/eb_dropped/flip_weighted.py`;
per-item details (n, flips, top-5 variables) in `flip_weighted.json`. Held-out items and controls first; E62
copy against lookup is a restatement (the weighting was chosen after seeing it). 'not as expected': outside the
amendment's expected value as `expected()` reads it; 'rests on near-rounding variables': E_flip moves by more
than 10% between dd floors f = 1e-8 and 1e-6.

| group | item | featurization | E_b | E_flip | f=1e-8 | f=1e-6 | f=1e-4 | n | expected | label |
|---|---|---|---|---|---|---|---|---|---|---|
| control | e280_o2 | neurons | 1 | 1 | 1 | 1 | 1 | 1 | exactly 1 |  |
| control | e280_o2 | vpd | 0.1998 | 0.1194 | 0.1194 | 0.1194 | 0.1194 | 872 |  |  |
| control | e280_q2 | neurons | 0.04042 | 0.04042 | 0.04042 | 0.04042 | 0.04042 | 1 |  |  |
| control | e280_q2 | vpd | 0.2489 | 0.1708 | 0.1708 | 0.1708 | 0.1708 | 472 |  |  |
| control | e280_v2 | neurons | 1 | 1 | 1 | 1 | 1 | 1 | exactly 1 |  |
| control | e280_v2 | vpd | 0.1922 | 0.1227 | 0.1227 | 0.1227 | 0.1254 | 871 |  |  |
| control | new_few | vpd | 0.0127 | 0.01419 | 0.01419 | 0.0142 | 0.01174 | 5523 | about 0.01-0.02 |  |
| control | new_thin | vpd | 0.01878 | 0.01354 | 0.01355 | 0.01355 | 0.01523 | 1250 | about 0.01-0.02 |  |
| control | random_kl_241002 | vpd | 0.01056 | 0.007047 | 0.007051 | 0.008223 | 0.01309 | 32225 | about 0.01 | rests on near-rounding variables |
| control | random_size_241001 | vpd | 0.011 | 0.006562 | 0.006573 | 0.007822 | 0.01364 | 38935 | about 0.01 | rests on near-rounding variables |
| control | rebuilt | vpd | 0.6233 | 0.03 | 0.03002 | 0.03588 | 0.1146 | 16900 |  | rests on near-rounding variables |
| control | x10 | neurons | 0.9998 | 0.8198 | 0.8198 | 0.8198 | 0.8032 | 9 | near 1 | not as expected |
| control | x10 | vpd | 0.7613 | 0.01314 | 0.01315 | 0.05312 | 0.2233 | 3585 | >= 0.5 | not as expected, rests on near-rounding variables |
| control | x2 | neurons | 0.9998 | 0.7722 | 0.7722 | 0.7722 | 0.7588 | 9 | near 1 | not as expected |
| control | x2 | vpd | 0.7614 | 0.01204 | 0.01204 | 0.04662 | 0.1817 | 3585 | >= 0.5 | not as expected, rests on near-rounding variables |
| E62 (restatement) | e62-copy_s0 | vpd | 0.03761 | 0.02894 | 0.02894 | 0.03104 | 0.04661 | 17382 |  |  |
| E62 (restatement) | e62-copy_s1 | vpd | 0.03991 | 0.02576 | 0.02576 | 0.02816 | 0.04758 | 30688 |  |  |
| E62 (restatement) | e62-copy_s2 | vpd | 0.03617 | 0.03109 | 0.03109 | 0.03275 | 0.04507 | 12145 |  |  |
| E62 (restatement) | e62-lookup_s0 | vpd | 0.03995 | 0.02931 | 0.02931 | 0.03101 | 0.0465 | 11186 |  |  |
| E62 (restatement) | e62-lookup_s0_scaled_e62_copy_lookup | neurons | 0.01323 | 0.04531 | 0.04531 | 0.04124 | 0.03774 | 4034 |  |  |
| E62 (restatement) | e62-lookup_s0_scaled_e62_copy_lookup | vpd | 0.04239 | 0.02795 | 0.02795 | 0.02961 | 0.04599 | 14994 |  |  |
| E62 (restatement) | e62-lookup_s1 | vpd | 0.04308 | 0.02998 | 0.02998 | 0.03196 | 0.04806 | 15904 |  |  |
| E62 (restatement) | e62-lookup_s1_scaled_e62_copy_lookup | neurons | 0.01451 | 0.0433 | 0.04329 | 0.03951 | 0.04141 | 6451 |  |  |
| E62 (restatement) | e62-lookup_s1_scaled_e62_copy_lookup | vpd | 0.04286 | 0.0277 | 0.0277 | 0.02987 | 0.04835 | 25649 |  |  |
| E62 (restatement) | e62-lookup_s2 | vpd | 0.04287 | 0.03269 | 0.03269 | 0.03428 | 0.04612 | 9271 |  |  |
| E62 (restatement) | e62-lookup_s2_scaled_e62_copy_lookup | neurons | 0.01292 | 0.04793 | 0.04793 | 0.04419 | 0.03941 | 3898 |  |  |
| E62 (restatement) | e62-lookup_s2_scaled_e62_copy_lookup | vpd | 0.03819 | 0.02999 | 0.02999 | 0.03185 | 0.04505 | 12004 |  |  |
| E62 (restatement) | e62__copy_s0 | neurons | 0.01866 | 0.04825 | 0.04825 | 0.04449 | 0.04672 | 4046 |  |  |
| E62 (restatement) | e62__copy_s1 | neurons | 0.01629 | 0.04446 | 0.04446 | 0.03923 | 0.03965 | 5896 |  | rests on near-rounding variables |
| E62 (restatement) | e62__copy_s2 | neurons | 0.01991 | 0.04718 | 0.04718 | 0.04285 | 0.04615 | 4044 |  |  |
| E62 (restatement) | e62__lookup_s0 | neurons | 0.01481 | 0.04661 | 0.04661 | 0.04454 | 0.04063 | 3743 |  |  |
| E62 (restatement) | e62__lookup_s1 | neurons | 0.01629 | 0.04761 | 0.04761 | 0.04538 | 0.04494 | 5106 |  |  |
| E62 (restatement) | e62__lookup_s2 | neurons | 0.01388 | 0.0536 | 0.0536 | 0.05109 | 0.04316 | 3047 |  |  |

Trained pairs at neurons, reuse minus new: none read yet.

