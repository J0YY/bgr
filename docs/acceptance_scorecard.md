# Acceptance Scorecard

This scorecard is generated from local result artifacts. It is not an acceptance probability; it quantifies distance to the internal promotion gates.

## Gate Summary

- Grid-margin mechanism clears the internal mechanism threshold: BGR 0.4342 vs uniform 0.3965, delta +0.0377; margin above +0.03 threshold +0.0077.
- Perturb-only anchored OpenVLA audit does not clear the learned-policy promotion gate: BGR 371/400, official 367/400, random 372/400; identity BGR 99/100, official 99/100, random 99/100; official gap +4 (+0.0100), random gap -1 (-0.0025), clean deficit 0.
- Closest independent benchmark screen is `MiniGrid FourRooms official-package` with treatment `bgr_coverage`: delta vs uniform +0.1075 (4/0/0), worst required-baseline delta +0.0464 (3/1/0), ablation delta +0.0071 (2/2/0), radius delta +0.0000 (0/0/4), failure reason(s): final_median_r80-ceiling-saturated.
- Rejected pre-method calibration route(s): `FetchPush-v4 object-goal calibration`, `FetchPush-v4 far-push object-goal calibration`, `FetchSlide-v4 object-goal calibration`, `FetchPickAndPlace-v4 object-goal calibration`, `Gymnasium-Robotics HandReach-v3 calibration`, `highway-env parking-v0 calibration`.
- Retired calibrated route(s) that cleared pre-method calibration: `Gymnasium MuJoCo Reacher-v5 calibration` clean 0.8333, range 0.5000--0.9167, r80 3.0000, `Gymnasium MuJoCo InvertedPendulum-v5 calibration` clean 1.0000, range 0.0000--1.0000, r80 0.2100, `Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration` clean 1.0000, range 0.0000--1.0000, r80 0.2825, `Gymnasium Box2D LunarLander-v3 calibration` clean 0.9167, range 0.5833--0.9167, r80 0.5300.

## Promotion Deficits

- Independent benchmark: no screen clears the 4/4 promotion screen. The closest screen, `MiniGrid FourRooms official-package` with `bgr_coverage`, clears 3/4 gates and fails on final_median_r80-ceiling-saturated.
- Learned policy: Perturb-only anchored OpenVLA audit does not clear the learned-policy promotion gate: BGR 371/400, official 367/400, random 372/400; identity BGR 99/100, official 99/100, random 99/100; official gap +4 (+0.0100), random gap -1 (-0.0025), clean deficit 0.
- Retired calibrated route(s): `Gymnasium MuJoCo Reacher-v5 calibration`, `Gymnasium MuJoCo InvertedPendulum-v5 calibration`, `Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration`, `Gymnasium Box2D LunarLander-v3 calibration` cleared pre-method calibration, but the corresponding fixed method screen is negative or tied; not active acceptance evidence.
- Active route: no queued learned-policy route is recorded in the local ledgers.

## Independent Benchmark Screens

| Screen | Treatment | Seeds | dRAUC vs uniform (W/L/T) | Worst required baseline dRAUC | Ablation dRAUC | Radius metric | Radius delta | Cleared gates | Screen gate | Failure reason(s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| MiniGrid FourRooms official-package | bgr_coverage | 4 | +0.1075 (4/0/0) | +0.0464 (3/1/0) | +0.0071 (2/2/0) | final_median_r80 | +0.0000 (0/0/4) | 3/4 | fail | final_median_r80-ceiling-saturated |
| MiniGrid FourRooms abs-r10 follow-up | bgr_coverage | 4 | +0.1075 (4/0/0) | +0.0464 (3/1/0) | +0.0071 (2/2/0) | final_abs_r10 | +0.0000 (0/0/4) | 3/4 | fail | final_abs_r10-floor-saturated |
| Gymnasium Box2D LunarLander-v3 | bgr_coverage | 4 | +0.1278 (2/2/0) | +0.0125 (3/1/0) | +0.0340 (3/1/0) | final_median_r80 | -0.0625 (3/1/0) | 2/4 | fail | uniform-gate, final_median_r80-contradiction |
| Gymnasium MuJoCo InvertedDoublePendulum-v5 | bgr | 4 | +0.0799 (1/0/3) | +0.0833 (1/0/3) | +0.0833 (1/0/3) | final_median_r80 | -0.1900 (0/1/3) | 2/4 | fail | uniform-gate, final_median_r80-contradiction |
| MiniGrid FourRooms official-package | bgr | 4 | +0.0641 (3/1/0) | +0.0030 (3/1/0) | -0.0362 (1/3/0) | final_median_r80 | +0.0000 (0/0/4) | 2/4 | fail | state-priority-ablation, final_median_r80-ceiling-saturated |
| MiniGrid FourRooms abs-r10 follow-up | bgr | 4 | +0.0641 (3/1/0) | +0.0030 (3/1/0) | -0.0362 (1/3/0) | final_abs_r10 | +0.0000 (0/0/4) | 2/4 | fail | state-priority-ablation, final_abs_r10-floor-saturated |
| FrozenLake8x8 | bgr | 30 | +0.0140 (14/16/0) | -0.0032 (14/16/0) | n/a | final_median_r80 | -0.0125 (12/18/0) | 2/4 | fail | required-baseline, final_median_r80-contradiction |
| MiniGrid FourRooms max-radius-10 follow-up | bgr_coverage | 4 | +0.0017 (2/2/0) | +0.0400 (4/0/0) | +0.0064 (3/1/0) | final_median_r80 | +0.0000 (0/0/4) | 2/4 | fail | uniform-gate, final_median_r80-ceiling-saturated |
| bsuite Cartpole | bgr_coverage | 4 | -0.0018 (2/2/0) | -0.0135 (2/2/0) | +0.0008 (2/2/0) | final_median_r80 | +0.0042 (1/2/1) | 2/4 | fail | uniform-gate, required-baseline |
| MiniGrid LavaGapS7 | bgr_coverage | 4 | -0.0184 (3/1/0) | +0.0183 (1/2/1) | -0.0350 (1/2/1) | final_abs_r10 | +0.0180 (1/1/2) | 2/4 | fail | uniform-gate, state-priority-ablation |
| MiniGrid FourRooms mid2-5 | bgr | 4 | +0.0557 (2/2/0) | -0.0563 (2/2/0) | +0.0411 (2/2/0) | final_median_r80 | -0.0824 (2/2/0) | 1/4 | fail | uniform-gate, required-baseline, final_median_r80-contradiction |
| bsuite DeepSea | bgr | 4 | +0.0281 (2/1/1) | +0.0141 (1/3/0) | -0.0141 (1/2/1) | final_median_r80 | -0.2125 (0/1/3) | 1/4 | fail | uniform-gate, state-priority-ablation, final_median_r80-contradiction |
| PointMaze U-Maze clean-shield | bgr_clean_shield | 4 | +0.0247 (2/2/0) | -0.3010 (0/3/1) | +0.1215 (2/1/1) | final_abs_r20 | -0.1333 (1/2/1) | 1/4 | fail | uniform-gate, required-baseline, final_abs_r20-contradiction |
| bsuite DeepSea | bgr_coverage | 4 | +0.0000 (1/2/1) | -0.0141 (0/3/1) | -0.0422 (1/3/0) | final_median_r80 | +0.0000 (1/1/2) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| Gymnasium MuJoCo InvertedPendulum-v5 | bgr | 4 | +0.0000 (0/0/4) | +0.0000 (0/0/4) | +0.0000 (0/0/4) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| Gymnasium MuJoCo InvertedPendulum-v5 | bgr_coverage | 4 | +0.0000 (0/0/4) | +0.0000 (0/0/4) | +0.0000 (0/0/4) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| Gymnasium MuJoCo InvertedDoublePendulum-v5 | bgr_coverage | 4 | -0.0035 (0/1/3) | +0.0000 (0/0/4) | +0.0000 (0/0/4) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| PointMaze U-Maze clean-shield | bgr_coverage | 4 | -0.0128 (2/2/0) | -0.3385 (0/4/0) | +0.0840 (3/1/0) | final_abs_r20 | -0.1750 (0/2/2) | 1/4 | fail | uniform-gate, required-baseline, final_abs_r20-contradiction |
| MiniGrid FourRooms mid2-5 | bgr_coverage | 4 | -0.0257 (1/3/0) | -0.1377 (0/4/0) | -0.0403 (1/3/0) | final_median_r80 | +0.0617 (2/2/0) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| FetchReach-v4 goal recovery | bgr_coverage | 4 | -0.0438 (0/1/3) | -0.0562 (0/2/2) | -0.0063 (1/1/2) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| FetchReach-v4 goal recovery | bgr | 4 | -0.0500 (0/2/2) | -0.0625 (0/3/1) | -0.0125 (0/1/3) | final_median_r80 | +0.0000 (0/0/4) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| MiniGrid LavaCrossing | bgr_coverage | 4 | -0.0618 (0/3/1) | +0.1188 (2/2/0) | -0.0495 (0/3/1) | final_abs_r10 | -0.1305 (0/3/1) | 1/4 | fail | uniform-gate, state-priority-ablation, final_abs_r10-contradiction |
| PointMaze U-Maze clean-shield | bgr | 4 | -0.0795 (1/3/0) | -0.4052 (0/4/0) | +0.0174 (1/3/0) | final_abs_r20 | -0.2333 (0/2/2) | 1/4 | fail | uniform-gate, required-baseline, final_abs_r20-contradiction |
| Gymnasium MuJoCo Reacher-v5 | bgr | 12 | -0.0955 (4/8/0) | -0.0366 (6/6/0) | -0.0014 (6/6/0) | final_median_r80 | +0.5938 (3/2/7) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| MiniGrid LavaCrossing | bgr | 4 | -0.1012 (1/3/0) | +0.0794 (3/1/0) | -0.0889 (1/3/0) | final_abs_r10 | -0.2164 (0/3/1) | 1/4 | fail | uniform-gate, state-priority-ablation, final_abs_r10-contradiction |
| MiniGrid FourRooms midband | bgr | 4 | -0.1127 (1/3/0) | -0.2403 (0/4/0) | -0.0749 (1/3/0) | final_median_r80 | +0.0995 (2/2/0) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| Gymnasium MuJoCo Reacher-v5 | bgr_coverage | 12 | -0.1140 (4/8/0) | -0.0551 (6/6/0) | -0.0200 (6/6/0) | final_median_r80 | +0.4271 (3/2/7) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| FetchReach-v4 hard-budget goal recovery | bgr_coverage | 4 | -0.2188 (1/3/0) | -0.4813 (0/4/0) | -0.0563 (0/3/1) | final_median_r80 | +0.0008 (1/2/1) | 1/4 | fail | uniform-gate, required-baseline, state-priority-ablation |
| Gymnasium Box2D LunarLander-v3 | bgr | 4 | +0.0257 (1/3/0) | -0.0896 (1/3/0) | -0.0681 (2/2/0) | final_median_r80 | -0.0950 (3/1/0) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| bsuite MountainCar | bgr_coverage | 4 | +0.0056 (3/1/0) | -0.0867 (0/4/0) | -0.0005 (2/2/0) | final_median_r80 | +0.0000 (0/0/4) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-ceiling-saturated |
| bsuite MountainCar | bgr | 4 | +0.0036 (3/1/0) | -0.0888 (0/4/0) | -0.0026 (1/3/0) | final_median_r80 | +0.0000 (0/0/4) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-ceiling-saturated |
| bsuite Cartpole | bgr | 4 | -0.0113 (0/4/0) | -0.0229 (0/4/0) | -0.0087 (1/3/0) | final_median_r80 | -0.0208 (1/1/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| bsuite Catch 30-seed | bgr_coverage | 30 | -0.0330 (13/17/0) | -0.1224 (4/26/0) | -0.0136 (15/15/0) | final_median_r80 | -0.0625 (6/11/13) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| bsuite Catch 30-seed | bgr | 30 | -0.0337 (14/16/0) | -0.1231 (3/24/3) | -0.0142 (20/10/0) | final_median_r80 | -0.0867 (5/9/16) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid FourRooms max-radius-10 follow-up | bgr | 4 | -0.0422 (1/3/0) | -0.0039 (1/3/0) | -0.0375 (1/3/0) | final_median_r80 | +0.0000 (0/0/4) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-ceiling-saturated |
| MiniGrid LavaGapS7 | bgr | 4 | -0.0430 (1/3/0) | -0.0063 (2/1/1) | -0.0596 (1/2/1) | final_abs_r10 | -0.0289 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |
| MiniGrid FourRooms midband | bgr_coverage | 4 | -0.0589 (2/2/0) | -0.1864 (0/4/0) | -0.0210 (3/1/0) | final_median_r80 | -0.0748 (0/4/0) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid DoorKey | bgr_coverage | 4 | -0.1538 (0/2/2) | -0.1613 (1/3/0) | -0.0093 (1/2/1) | final_abs_r10 | -0.1569 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |
| FetchReach-v4 hard-budget goal recovery | bgr | 4 | -0.2375 (0/3/1) | -0.5000 (0/4/0) | -0.0750 (1/2/1) | final_median_r80 | -0.0097 (0/3/1) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid DoorKey | bgr | 4 | -0.2697 (0/4/0) | -0.2772 (0/4/0) | -0.1252 (0/4/0) | final_abs_r10 | -0.2503 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |

## Pre-Method Calibrations

| Calibration | Clean | Recovery range | Median r80 | Decision |
| --- | ---: | ---: | ---: | --- |
| FetchPush-v4 object-goal calibration | 0.2500 | 0.2500--0.2500 | 0.1200 | reject-calibration |
| FetchPush-v4 far-push object-goal calibration | 0.6250 | 0.6250--0.8750 | 0.1200 | reject-calibration |
| FetchSlide-v4 object-goal calibration | 0.2500 | 0.1250--0.2500 | 0.0720 | reject-calibration |
| FetchPickAndPlace-v4 object-goal calibration | 0.1250 | 0.0000--0.1250 | 0.0660 | reject-calibration |
| Gymnasium-Robotics HandReach-v3 calibration | 0.0000 | 0.0000--0.0000 | 0.2000 | reject-calibration |
| highway-env parking-v0 calibration | 0.3333 | 0.2500--0.5000 | 9.8000 | reject-calibration |
| Gymnasium MuJoCo Reacher-v5 calibration | 0.8333 | 0.5000--0.9167 | 3.0000 | usable-calibration |
| Gymnasium MuJoCo InvertedPendulum-v5 calibration | 1.0000 | 0.0000--1.0000 | 0.2100 | usable-calibration |
| Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration | 1.0000 | 0.0000--1.0000 | 0.2825 | usable-calibration |
| Gymnasium Box2D LunarLander-v3 calibration | 0.9167 | 0.5833--0.9167 | 0.5300 | usable-calibration |

## Priority Read

- The controlled grid mechanism is above its internal effect threshold, but it is still a constructed mechanism benchmark.
- The independent-benchmark route has not produced a promotable screen: early promising screens either fail paired/radius checks or do not survive scale-up, and later non-saturated screens trail uniform, stronger baselines, or the state-priority/uniform-radius ablation.
- The usable Reacher-v5 calibration is pre-method evidence only; the fixed full all-method comparison is now negative and should not be promoted.
- The InvertedPendulum-v5 calibration also cleared pre-method checks, but its fixed 4-seed method screen ties all methods on final RAUC and median-r80; do not scale or promote it.
- Rejected pre-method calibrations should not be scaled into BGR comparisons until the reset interface and controller first produce clean, non-saturated recovery curves.
- The InvertedDoublePendulum-v5 calibration cleared pre-method checks, but its fixed 4-seed method screen collapses clean success; the small BGR RAUC edge is not acceptance evidence.
- Perturb-only anchored OpenVLA audit does not clear the learned-policy promotion gate: BGR 371/400, official 367/400, random 372/400; identity BGR 99/100, official 99/100, random 99/100; official gap +4 (+0.0100), random gap -1 (-0.0025), clean deficit 0.
- The next acceptance-moving work must find a genuinely different independent route, change the learned-policy intervention, or strengthen theory/presentation; retired calibrated routes are scope evidence, not acceptance evidence.
