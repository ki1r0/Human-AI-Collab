# Stage B result — official rule-policy oracle

Initial status: **FAIL**.

- Run: `runs/stage_b/`
- Exit: 0; no renderer, missing-asset, PhysX, NaN, or Python errors.
- Schedule: all 3,320 physics steps / 664 environment steps completed.
- Official best score: 3; required score: 6.
- Result SHA-256: `6323633ad20fb89a45e9a895099b722105d8a70b3d73e9ac66cfc92f8f15844a`.
- Console SHA-256: `e492f466102e62adcfd1b1ea4f0f7b727f3378fe8b6174a22cee00b5996e2e48`.

Score decomposition from the recorded poses:

| Physics step | Score | Satisfied relationships |
|---:|---:|---|
| 330 | 1 | gear 1 on carrier pin 1 |
| 1,175 | 2 | gear 1 on pin 1; gear 3 on pin 0 |
| 3,155 | 3 | prior two pin relations; reducer transiently aligned with gear 4 |

Gear 2 and gear 4 never satisfied their pin tests; the carrier never satisfied the ring test. The score dropped after the transient reducer alignment. Native termination at the schedule bound auto-reset the environment, which is why `final_score` is 0 while the preserved `best_score` is 3.

Post-run source history identified an embodiment/configuration mismatch: the current 0.15 m external/table workspace offset was introduced while R1 Lite was active in May 2026, whereas the R1 public-data/checkpoint-era source and the still-current learned-agent environment use 0.20 m. The next exploratory iteration changes only that offset back to 0.20 m.

## Iteration 1 — 0.20 m R1 workspace

Status: **FAIL, improved to best score 4/6**.

- All three carrier-pin relations scored, versus two in the initial run.
- The fourth point was a transient reducer/gear-4 relation.
- The ring was assigned to the left arm at initial `(x≈0.639, y≈0.187)` but never reached the carrier; its final pre-reset Y remained near `0.171`.
- Result SHA-256: `033292a0a67b46f92f06439e5c442d393a56feff4f89f3e6fb046d6dbe176864`.
- Console SHA-256: `24cac8acee23b79fe39d61cef30fdbe98c3b747a1283b765d4d8e349d84e458a`.

H6 is supported because score improved above 3; the 0.20 m restoration is retained. Stage B remains failed because carrier/ring and middle-gear/ring never scored.

## Iteration 2 — preregistered randomized-start sweep

Status: **FAIL; H7 refuted**.

| Seed | Best score | Final pre-reset score | Ring lift | Result SHA-256 |
|---:|---:|---:|---:|---|
| 17 | 2 | 0 | 0.9011 m to 1.0994 m | `b505d5f151a6695a3e0492a297ed08a2213c5a306b34806984b114f42b53d9f3` |
| 23 | 4 | 0 | 0.9013 m to 1.0993 m | `3e638010dbca67539016cd73363cdfe4710181b731b3f5976eb29027fee89496` |
| 42 | 2 | 0 | 0.9016 m to 1.0991 m | `073c0cfb500a0aaef3599b706bc262680e3aeef24b483df55f81b6239f6fac16` |

All three runs terminated cleanly at the 3,320-step official schedule without timeout or task success. Saved head and wrist RGB frames plus state snapshots prove that every ring was grasped, lifted, transported, and descended over the carrier. Seed 23 entered ring descent with three mounted pin gears, briefly reached score 4, then fell to zero during the commanded 30-degree rotation. Randomization affects early pin success but is not sufficient to resolve assembly.
