# E118: Qwen VPD rows beside the comparators

| source | row | state | VPD | ‖δW‖/‖W0‖ | W0 rank (first) / gain | steer (random) share | carrying n / share / engaged |
|---|---|---|---|---|---|---|---|
| E93 | a1_s0 | a1_s0 | passes; share 0.0873 | 0.000371 | 21 / 0.666 | 0.185 (-0.0378) | 1024 / 0.911 / 0.496 |
| E93 | a1_s1 | a1_s1 | passes; share 0.0871 | 0.000373 | 21 / 0.666 | 0.182 (-0.0375) | 1024 / 0.912 / 0.5 |
| E93 | a2_s42 | a2_s42_full | passes; share 0.101 | 0.0138 | 1 / 3.77 | 0.0645 (0.048) | 1024 / 0.726 / 0.501 |
| E93 | a2_s142 | a2_s142_full | passes; share 0.101 | 0.0138 | 1 / 3.76 | 0.0671 (0.046) | 1024 / 0.721 / 0.504 |
| E93 | R_s0 | R_s0_100 | fails; share 0.0924 | 0.000268 | 11 / -0.277 * | 0.296 (-0.0381) | 1024 / 0.924 / 0.503 |
| E93 | R_s70 | R_s70_100 | fails; share 0.0943 | 0.000265 | 11 / 0.163 * | 0.914 (0.176) | 1024 / 1.06 / 0.497 |
| E93 | N_s0 | N_s0_100 | fails; share 0.0892 | 0.000282 | 11 / 0.283 * | 0.729 (0.146) | 1024 / 1.07 / 0.497 |
| E93 | N_s70 | N_s70_100 | fails; share 0.0921 | 0.000256 | 11 / 0.595 * | 0.552 (0.0689) | 1024 / 1.04 / 0.503 |
| E93 | F_s0 | F_s0_100 | fails; share 0.0908 | 0.000209 | 11 / 0.0477 * | 0.495 (0.0375) | 1024 / 1.06 / 0.499 |
| E93 | u1 | u1 | passes; share 0.0875 | 0.00258 | 11 / 1.66 * | 0.693 (0.112) | 1024 / 0.838 / 0.49 |
| E93 | u2 | u2 | passes; share 0.0882 | 0.00218 | 11 / 1.67 * | 0.661 (0.0974) | 1024 / 0.877 / 0.492 |
| E94 | a1_s0 | a1_s0 | direct 0.0093, outside 0.123, think 3.66, set 1286 | 0.000371 | 21 / 0.666 | 0.185 (-0.0378) | 1024 / 0.911 / 0.496 |
| E94 | a1_s1 | a1_s1 | direct 0.00908, outside 0.124, think 3.75, set 1320 | 0.000373 | 21 / 0.666 | 0.182 (-0.0375) | 1024 / 0.912 / 0.5 |
| E94 | a2_s42_matched | a2_s42_matched | direct 0.0303, outside 0.0965, think 3.74, set 1847 | 0.000364 | 1 / 2.05 | 0.827 (0.106) | 1024 / 0.983 / 0.501 |
| E94 | a2_s142_matched | a2_s142_matched | direct 0.0323, outside 0.096, think 3.71, set 1937 | 0.000378 | 1 / 2.06 | 0.829 (0.108) | 1024 / 0.984 / 0.499 |
| E94 | a2_s42_full | a2_s42_full | direct 0.0151, outside 0.186, think 3.08, set 7805 | 0.0138 | 1 / 3.77 | 0.0645 (0.048) | 1024 / 0.726 / 0.501 |
| E94 | a2_s142_full | a2_s142_full | direct 0.0106, outside 0.184, think 3.01, set 7987 | 0.0138 | 1 / 3.76 | 0.0671 (0.046) | 1024 / 0.721 / 0.504 |
| E94 | a2_s42_past | a2_s42_past | direct 0.0441, outside 0.175, think 3.11 | 0.0118 | 1 / 3.52 | 0.227 (0.046) | 1024 / 0.719 / 0.502 |
| E94 | a2_s142_past | a2_s142_past | direct 0.0452, outside 0.17, think 3.21 | 0.0112 | 1 / 3.48 | 0.238 (0.0461) | 1024 / 0.781 / 0.504 |
| E94 | gauss_small | gauss_small | direct 0.0278, outside 0.212, think 3.13 | n/a | n/a | n/a | n/a |
| E94 | alive_small | alive_small | direct 0.028, outside 0.000307, think 3.25 | n/a | n/a | n/a | n/a |
| E94 | gauss_big | gauss_big | direct 0.0212, outside 0.291, think 2.95 | n/a | n/a | n/a | n/a |
| E94 | alive_big | alive_big | direct 0.0295, outside 0.00127, think 3.36 | n/a | n/a | n/a | n/a |
| E94 Am1 | known_s42_matched | known_s42_matched | direct 0.0225, outside 0.0985, think 3.65, set 1501 | 0.000366 | 1 / 1.11 | 0.584 (0.121) | 1024 / None / None * |
| E94 Am1 | known_s142_matched | known_s142_matched | direct 0.0236, outside 0.0992, think 3.59, set 1468 | 0.000369 | 1 / 1.11 | 0.598 (0.123) | 1024 / None / None * |
| E94 Am1 | known_s42_end | known_s42_end | direct 0.0157, outside 0.174, think 3.17, set 3757 | 0.00639 | 1 / 1.48 | 0.378 (0.0854) | 1024 / 2.15 / 0.491 |
| E94 Am1 | known_s142_end | known_s142_end | direct 0.0113, outside 0.163, think 3.19, set 4103 | 0.00629 | 1 / 1.5 | 0.393 (0.0967) | 1024 / 0.134 / 0.489 |
| E104 | R_s0_10 | R_s0_10 | outside 0.147, think 2.85 | 5.24e-05 | 11 / -0.198 * | 0.645 (-0.0418) | 1024 / 0.989 / 0.497 |
| E104 | R_s0_100 | R_s0_100 | outside 0.135, think 3.65 | 0.000268 | 11 / -0.277 * | 0.296 (-0.0381) | 1024 / 0.924 / 0.503 |
| E104 | R_s70_10 | R_s70_10 | outside 0.136, think 3.17 | 5.17e-05 | 11 / -0.118 * | 0.688 (-0.0453) | 1024 / 0.98 / 0.499 |
| E104 | R_s70_100 | R_s70_100 | outside 0.158, think 3.21 | 0.000265 | 11 / 0.163 * | 0.914 (0.176) | 1024 / 1.06 / 0.497 |
| E104 | N_s0_10 | N_s0_10 | outside 0.149, think 2.83 | 4.95e-05 | 11 / -0.0694 * | 0.541 (-0.0344) | 1024 / 0.939 / 0.505 |
| E104 | N_s0_100 | N_s0_100 | outside 0.144, think 3.5 | 0.000282 | 11 / 0.283 * | 0.729 (0.146) | 1024 / 1.07 / 0.497 |
| E104 | N_s70_10 | N_s70_10 | outside 0.153, think 3.54 | 4.93e-05 | 11 / 0.137 * | 0.552 (0.0345) | 1024 / 1.03 / 0.504 |
| E104 | N_s70_100 | N_s70_100 | outside 0.157, think 3.33 | 0.000256 | 11 / 0.595 * | 0.552 (0.0689) | 1024 / 1.04 / 0.503 |
| E104 | F_s0_10 | F_s0_10 | outside 0.165, think 2.7 | 4.44e-05 | 11 / 0.0138 * | 0.386 (0.15) | 1024 / 1.03 / 0.498 |
| E104 | F_s0_100 | F_s0_100 | outside 0.189, think 2.53 | 0.000209 | 11 / 0.0477 * | 0.495 (0.0375) | 1024 / 1.06 / 0.499 |
| E104 | S1 | S1 | outside 0.135, think 2.94 | n/a | n/a | n/a | n/a |
| E104 | S2 | S2 | outside 0.135, think 3.52 | n/a | n/a | n/a | n/a |
| E104 | S3 | S3 | outside 0.186, think 3.01 | n/a | n/a | n/a | n/a |
| E104 | a1_s0 | a1_s0 | outside 0.123, think 3.66 | 0.000371 | 21 / 0.666 | 0.185 (-0.0378) | 1024 / 0.911 / 0.496 |
| E104 | a1_s1 | a1_s1 | outside 0.124, think 3.75 | 0.000373 | 21 / 0.666 | 0.182 (-0.0375) | 1024 / 0.912 / 0.5 |
| E104 | a2_s42_full | a2_s42_full | outside 0.186, think 3.08 | 0.0138 | 1 / 3.77 | 0.0645 (0.048) | 1024 / 0.726 / 0.501 |
| E104 | a2_s142_full | a2_s142_full | outside 0.184, think 3.01 | 0.0138 | 1 / 3.76 | 0.0671 (0.046) | 1024 / 0.721 / 0.504 |
| E104 | a2_s42_matched | a2_s42_matched | outside 0.0965, think 3.74 | 0.000364 | 1 / 2.05 | 0.827 (0.106) | 1024 / 0.983 / 0.501 |
| E104 | a2_s142_matched | a2_s142_matched | outside 0.096, think 3.71 | 0.000378 | 1 / 2.06 | 0.829 (0.108) | 1024 / 0.984 / 0.499 |
| E104 Am1 | copy | S1a1 | outside 0.135, think 2.92 | n/a | n/a | n/a | n/a |
| E107 Am1 | known_s42_matched | known_s42_matched | set 1501 | 0.000366 | 1 / 1.11 | 0.584 (0.121) | 1024 / None / None * |
| E107 Am1 | known_s142_matched | known_s142_matched | set 1468 | 0.000369 | 1 / 1.11 | 0.598 (0.123) | 1024 / None / None * |
| E107 Am1 | wrong_s42_matched | wrong_s42_matched | set 1513 | 0.000376 | 1 / 1.37 | 0.626 (0.155) | 1024 / None / None * |
| E107 Am1 | wrong_s142_matched | wrong_s142_matched | set 1474 | 0.000363 | 1 / 1.38 | 0.637 (0.159) | 1024 / None / None * |
| E107 Am1 | a2_s42_matched | a2_s42_matched | set 1847 | 0.000364 | 1 / 2.05 | 0.827 (0.106) | 1024 / 0.983 / 0.501 |
| E107 Am1 | a2_s142_matched | a2_s142_matched | set 1937 | 0.000378 | 1 / 2.06 | 0.829 (0.108) | 1024 / 0.984 / 0.499 |

pending = not computed yet, E118's GPU job writes it; n/a = does not apply (a constructed control, table.json has the reason); undefined = E92's steering rule gives no vector; * = a note in table.json (gold rows for the GSM8K rank; E103/E107 sets without neuron share or engaged share). Every cell's file and key: table.json.
