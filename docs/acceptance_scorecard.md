# Acceptance Scorecard

This scorecard is generated from local result artifacts. It is not an acceptance probability; it quantifies distance to the internal promotion gates.

## Gate Summary

- Grid-margin mechanism clears the internal mechanism threshold: BGR 0.4342 vs uniform 0.3965, delta +0.0377; margin above +0.03 threshold +0.0077.
- Weighted OpenVLA audit is already unable to clear the official-checkpoint gate before the final random row: BGR 367/400, official 367/400, random 273/300 available; final random shift pending; identity BGR 99/100. Official episode margin is +0, short of +10 by 10 episodes; the pending random row is ledger completion, not a path to promotion.
- Closest independent benchmark screen is `PointMaze U-Maze clean-shield` with treatment `bgr_clean_shield`: delta vs uniform +0.0247 (2/2/0), worst required-baseline delta -0.3010 (0/3/1), ablation delta +0.1215 (2/1/1), radius delta -0.1333 (1/2/1).
- Rejected pre-method calibration route(s): `FetchPush-v4 object-goal calibration`.

## Independent Benchmark Screens

| Screen | Treatment | Seeds | dRAUC vs uniform (W/L/T) | Worst required baseline dRAUC | Ablation dRAUC | Radius delta | Screen gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PointMaze U-Maze clean-shield | bgr_clean_shield | 4 | +0.0247 (2/2/0) | -0.3010 (0/3/1) | +0.1215 (2/1/1) | -0.1333 (1/2/1) | fail |
| FrozenLake8x8 | bgr | 30 | +0.0140 (14/16/0) | -0.0032 (14/16/0) | n/a | -0.0125 (12/18/0) | fail |
| PointMaze U-Maze clean-shield | bgr_coverage | 4 | -0.0128 (2/2/0) | -0.3385 (0/4/0) | +0.0840 (3/1/0) | -0.1750 (0/2/2) | fail |
| MiniGrid LavaGapS7 | bgr_coverage | 4 | -0.0184 (3/1/0) | +0.0183 (1/2/1) | -0.0350 (1/2/1) | +0.0180 (1/1/2) | fail |
| MiniGrid LavaGapS7 | bgr | 4 | -0.0430 (1/3/0) | -0.0063 (2/1/1) | -0.0596 (1/2/1) | -0.0289 (0/2/2) | fail |
| FetchReach-v4 goal recovery | bgr_coverage | 4 | -0.0438 (0/1/3) | -0.0562 (0/2/2) | -0.0063 (1/1/2) | +0.0000 (0/0/4) | fail |
| FetchReach-v4 goal recovery | bgr | 4 | -0.0500 (0/2/2) | -0.0625 (0/3/1) | -0.0125 (0/1/3) | +0.0000 (0/0/4) | fail |
| MiniGrid FourRooms midband | bgr_coverage | 4 | -0.0589 (2/2/0) | -0.1864 (0/4/0) | -0.0210 (3/1/0) | -0.0748 (0/4/0) | fail |
| MiniGrid LavaCrossing | bgr_coverage | 4 | -0.0618 (0/3/1) | +0.1188 (2/2/0) | -0.0495 (0/3/1) | -0.1305 (0/3/1) | fail |
| PointMaze U-Maze clean-shield | bgr | 4 | -0.0795 (1/3/0) | -0.4052 (0/4/0) | +0.0174 (1/3/0) | -0.2333 (0/2/2) | fail |
| MiniGrid LavaCrossing | bgr | 4 | -0.1012 (1/3/0) | +0.0794 (3/1/0) | -0.0889 (1/3/0) | -0.2164 (0/3/1) | fail |
| MiniGrid FourRooms midband | bgr | 4 | -0.1127 (1/3/0) | -0.2403 (0/4/0) | -0.0749 (1/3/0) | +0.0995 (2/2/0) | fail |
| MiniGrid DoorKey | bgr_coverage | 4 | -0.1538 (0/2/2) | -0.1613 (1/3/0) | -0.0093 (1/2/1) | -0.1569 (0/2/2) | fail |
| MiniGrid DoorKey | bgr | 4 | -0.2697 (0/4/0) | -0.2772 (0/4/0) | -0.1252 (0/4/0) | -0.2503 (0/2/2) | fail |

## Pre-Method Calibrations

| Calibration | Clean | Recovery range | Median r80 | Decision |
| --- | ---: | ---: | ---: | --- |
| FetchPush-v4 object-goal calibration | 0.2500 | 0.2500--0.2500 | 0.1200 | reject-calibration |

## Priority Read

- The controlled grid mechanism is above its internal effect threshold, but it is still a constructed mechanism benchmark.
- The independent-benchmark route has not produced a promotable screen: the closest external-package screens either trail uniform on mean RAUC or lose to the state-priority/uniform-radius ablation.
- Rejected pre-method calibrations should not be scaled into BGR comparisons until the reset interface and controller first produce clean, non-saturated recovery curves.
- The current OpenVLA weighted curriculum is already unable to clear the official-checkpoint gate; the pending random-shift row is ledger completion, not a path to a positive robotics claim.
- The next acceptance-moving work should change the learned-policy intervention or materially strengthen theory/presentation; another same-protocol MiniGrid/classic-control screen is unlikely to move the gate.
