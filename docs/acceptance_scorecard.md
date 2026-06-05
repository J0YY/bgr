# Acceptance Scorecard

This scorecard is generated from local result artifacts. It is not an acceptance probability; it quantifies distance to the internal promotion gates.

## Gate Summary

- Grid-margin mechanism clears the internal mechanism threshold: BGR 0.4342 vs uniform 0.3965, delta +0.0377; margin above +0.03 threshold +0.0077.
- Weighted OpenVLA audit is already unable to clear the official-checkpoint gate before the final random row: BGR 367/400, official 367/400, random 273/300 available; final random shift pending; identity BGR 99/100. Official episode margin is +0, short of +10 by 10 episodes; the pending random row is ledger completion, not a path to promotion.
- Proximal-anchor OpenVLA route is in flight, not yet evidence: adaptation jobs BGR 767128/767129/767130 and random 767131/767132/767133 are queued; fixed perturbation jobs official 767134--767138, BGR 767139--767143, and random 767144--767148 must finish before the +10/400 and +0.02 learned-policy gate can be checked.
- Closest independent benchmark screen is `MiniGrid FourRooms official-package` with treatment `bgr_coverage`: delta vs uniform +0.1075 (4/0/0), worst required-baseline delta +0.0464 (3/1/0), ablation delta +0.0071 (2/2/0), radius delta +0.0000 (0/0/4), failure reason(s): final_median_r80-ceiling-saturated.
- Rejected pre-method calibration route(s): `FetchPush-v4 object-goal calibration`, `FetchSlide-v4 object-goal calibration`, `FetchPickAndPlace-v4 object-goal calibration`.

## Independent Benchmark Screens

| Screen | Treatment | Seeds | dRAUC vs uniform (W/L/T) | Worst required baseline dRAUC | Ablation dRAUC | Radius metric | Radius delta | Cleared gates | Screen gate | Failure reason(s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| MiniGrid FourRooms official-package | bgr_coverage | 4 | +0.1075 (4/0/0) | +0.0464 (3/1/0) | +0.0071 (2/2/0) | final_median_r80 | +0.0000 (0/0/4) | 3/4 | fail | final_median_r80-ceiling-saturated |
| MiniGrid FourRooms abs-r10 follow-up | bgr_coverage | 4 | +0.1075 (4/0/0) | +0.0464 (3/1/0) | +0.0071 (2/2/0) | final_abs_r10 | +0.0000 (0/0/4) | 3/4 | fail | final_abs_r10-floor-saturated |
| MiniGrid FourRooms official-package | bgr | 4 | +0.0641 (3/1/0) | +0.0030 (3/1/0) | -0.0362 (1/3/0) | final_median_r80 | +0.0000 (0/0/4) | 2/4 | fail | state-priority-ablation, final_median_r80-ceiling-saturated |
| MiniGrid FourRooms abs-r10 follow-up | bgr | 4 | +0.0641 (3/1/0) | +0.0030 (3/1/0) | -0.0362 (1/3/0) | final_abs_r10 | +0.0000 (0/0/4) | 2/4 | fail | state-priority-ablation, final_abs_r10-floor-saturated |
| FrozenLake8x8 | bgr | 30 | +0.0140 (14/16/0) | -0.0032 (14/16/0) | n/a | final_median_r80 | -0.0125 (12/18/0) | 2/4 | fail | required-baseline, final_median_r80-contradiction |
| MiniGrid LavaGapS7 | bgr_coverage | 4 | -0.0184 (3/1/0) | +0.0183 (1/2/1) | -0.0350 (1/2/1) | final_abs_r10 | +0.0180 (1/1/2) | 2/4 | fail | uniform-gate, state-priority-ablation |
| MiniGrid FourRooms mid2-5 | bgr | 4 | +0.0557 (2/2/0) | -0.0563 (2/2/0) | +0.0411 (2/2/0) | final_median_r80 | -0.0824 (2/2/0) | 1/4 | fail | uniform-gate, required-baseline, final_median_r80-contradiction |
| PointMaze U-Maze clean-shield | bgr_clean_shield | 4 | +0.0247 (2/2/0) | -0.3010 (0/3/1) | +0.1215 (2/1/1) | final_abs_r20 | -0.1333 (1/2/1) | 1/4 | fail | uniform-gate, required-baseline, final_abs_r20-contradiction |
| PointMaze U-Maze clean-shield | bgr_coverage | 4 | -0.0128 (2/2/0) | -0.3385 (0/4/0) | +0.0840 (3/1/0) | final_abs_r20 | -0.1750 (0/2/2) | 1/4 | fail | uniform-gate, required-baseline, final_abs_r20-contradiction |
| MiniGrid FourRooms mid2-5 | bgr_coverage | 4 | -0.0257 (1/3/0) | -0.1377 (0/4/0) | -0.0403 (1/3/0) | final_median_r80 | +0.0617 (2/2/0) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| FetchReach-v4 goal recovery | bgr_coverage | 4 | -0.0438 (0/1/3) | -0.0562 (0/2/2) | -0.0063 (1/1/2) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| FetchReach-v4 goal recovery | bgr | 4 | -0.0500 (0/2/2) | -0.0625 (0/3/1) | -0.0125 (0/1/3) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| MiniGrid LavaCrossing | bgr_coverage | 4 | -0.0618 (0/3/1) | +0.1188 (2/2/0) | -0.0495 (0/3/1) | final_abs_r10 | -0.1305 (0/3/1) | 1/4 | fail | uniform-gate, state-priority-ablation, final_abs_r10-contradiction |
| PointMaze U-Maze clean-shield | bgr | 4 | -0.0795 (1/3/0) | -0.4052 (0/4/0) | +0.0174 (1/3/0) | final_abs_r20 | -0.2333 (0/2/2) | 1/4 | fail | uniform-gate, required-baseline, final_abs_r20-contradiction |
| MiniGrid LavaCrossing | bgr | 4 | -0.1012 (1/3/0) | +0.0794 (3/1/0) | -0.0889 (1/3/0) | final_abs_r10 | -0.2164 (0/3/1) | 1/4 | fail | uniform-gate, state-priority-ablation, final_abs_r10-contradiction |
| MiniGrid FourRooms midband | bgr | 4 | -0.1127 (1/3/0) | -0.2403 (0/4/0) | -0.0749 (1/3/0) | final_median_r80 | +0.0995 (2/2/0) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| FetchReach-v4 hard-budget goal recovery | bgr_coverage | 4 | -0.2188 (1/3/0) | -0.4813 (0/4/0) | -0.0563 (0/3/1) | final_median_r80 | +0.0008 (1/2/1) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| MiniGrid LavaGapS7 | bgr | 4 | -0.0430 (1/3/0) | -0.0063 (2/1/1) | -0.0596 (1/2/1) | final_abs_r10 | -0.0289 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |
| MiniGrid FourRooms midband | bgr_coverage | 4 | -0.0589 (2/2/0) | -0.1864 (0/4/0) | -0.0210 (3/1/0) | final_median_r80 | -0.0748 (0/4/0) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid DoorKey | bgr_coverage | 4 | -0.1538 (0/2/2) | -0.1613 (1/3/0) | -0.0093 (1/2/1) | final_abs_r10 | -0.1569 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |
| FetchReach-v4 hard-budget goal recovery | bgr | 4 | -0.2375 (0/3/1) | -0.5000 (0/4/0) | -0.0750 (1/2/1) | final_median_r80 | -0.0097 (0/3/1) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid DoorKey | bgr | 4 | -0.2697 (0/4/0) | -0.2772 (0/4/0) | -0.1252 (0/4/0) | final_abs_r10 | -0.2503 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |

## Pre-Method Calibrations

| Calibration | Clean | Recovery range | Median r80 | Decision |
| --- | ---: | ---: | ---: | --- |
| FetchPush-v4 object-goal calibration | 0.2500 | 0.2500--0.2500 | 0.1200 | reject-calibration |
| FetchSlide-v4 object-goal calibration | 0.2500 | 0.1250--0.2500 | 0.0720 | reject-calibration |
| FetchPickAndPlace-v4 object-goal calibration | 0.1250 | 0.0000--0.1250 | 0.0660 | reject-calibration |

## Priority Read

- The controlled grid mechanism is above its internal effect threshold, but it is still a constructed mechanism benchmark.
- The independent-benchmark route has not produced a promotable screen: the closest external-package screen with a visible RAUC lead fails because the radius metric is saturated, while later non-saturated screens trail uniform, stronger baselines, or the state-priority/uniform-radius ablation.
- Rejected pre-method calibrations should not be scaled into BGR comparisons until the reset interface and controller first produce clean, non-saturated recovery curves.
- The latest OpenVLA weighted audit is already unable to clear the official-checkpoint gate; the pending random-shift row is ledger completion, not a path to a positive robotics claim.
- The current acceptance-moving learned-policy work is the queued proximal-anchor OpenVLA route; next actions are to poll, sync completed summaries, and apply the preregistered gate before making any paper-positive claim.
- Do not start another same-protocol MiniGrid, classic-control, PointMaze, or FetchReach screen while this is pending; existing screens already show saturated radius checks, stronger-baseline losses, or state-priority-only ablation failures.
