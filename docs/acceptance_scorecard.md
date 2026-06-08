# Acceptance Scorecard

This scorecard is generated from local result artifacts. It is not an acceptance probability; it quantifies distance to the internal promotion gates.

## Gate Summary

- Grid-margin mechanism clears the internal mechanism threshold: BGR 0.4342 vs uniform 0.3965, delta +0.0377; margin above +0.03 threshold +0.0077.
- Perturb-only anchored OpenVLA audit does not clear the learned-policy promotion gate: BGR 371/400, official 367/400, random 372/400; identity BGR 99/100, official 99/100, random 99/100; official gap +4 (+0.0100), random gap -1 (-0.0025), clean deficit 0.
- Occlusion-bottleneck OpenVLA route is preregistered, not yet evidence: the clean-plus-occlusion TFDS prep, proximal-anchor BGR/random adaptation, and official/BGR/random 10-task x 10-trial perturbation summaries must finish before the +10/400 and +0.02 learned-policy gate can be checked.
- Closest independent benchmark screen is `MiniGrid FourRooms official-package` with treatment `bgr_coverage`: delta vs uniform +0.1075 (4/0/0), worst required-baseline delta +0.0464 (3/1/0), ablation delta +0.0071 (2/2/0), radius delta +0.0000 (0/0/4), failure reason(s): final_median_r80-ceiling-saturated.
- Rejected pre-method calibration route(s): `FetchPush-v4 object-goal calibration`, `FetchPush-v4 far-push object-goal calibration`, `FetchSlide-v4 object-goal calibration`, `FetchPickAndPlace-v4 object-goal calibration`, `Gymnasium-Robotics HandReach-v3 calibration`, `highway-env parking-v0 calibration`, `highway-env highway-fast-v0 lane calibration`.
- Retired calibrated route(s) that cleared pre-method calibration: `MinAtar Breakout calibration` clean 1.0000, range 0.6667--1.0000, r80 0.6000, `MinAtar Asterix calibration` clean 0.8333, range 0.5000--0.8333, r80 5.3333, `Gymnasium MuJoCo Reacher-v5 calibration` clean 0.8333, range 0.5000--0.9167, r80 3.0000, `Gymnasium MuJoCo InvertedPendulum-v5 calibration` clean 1.0000, range 0.0000--1.0000, r80 0.2100, `Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration` clean 1.0000, range 0.0000--1.0000, r80 0.2825, `Gymnasium Box2D LunarLander-v3 calibration` clean 0.9167, range 0.5833--0.9167, r80 0.5300.
- Rejected route scout(s): `sklearn digits margin replay` best BGR 0.8271 vs uniform 0.8123 (W/L/T=2/2/0), `sklearn tabular margin replay (breast_cancer)` best BGR 0.9610 vs uniform 0.9516 (W/L/T=3/1/0), `sklearn tabular margin replay (wine)` best BGR 0.9702 vs uniform 0.9563 (W/L/T=4/0/0), `OpenML margin replay (ionosphere)` best BGR 0.8429 vs uniform 0.8338 (W/L/T=2/2/0), `OpenML margin replay (sonar)` best BGR 0.7667 vs uniform 0.7378 (W/L/T=4/0/0), `OpenML margin replay (spambase)` best BGR 0.8840 vs uniform 0.8602 (W/L/T=3/1/0), `OpenML numeric external fixed target2 30-seed (banknote-authentication)` best BGR 0.7092 vs uniform 0.7864 (W/L/T=4/26/0), `OpenML numeric external fixed target2 30-seed (climate-model-simulation-crashes)` best BGR 0.9019 vs uniform 0.8796 (W/L/T=25/5/0), `OpenML numeric external fixed target2 30-seed (kc1)` best BGR 0.8145 vs uniform 0.8126 (W/L/T=15/14/1), `OpenML numeric external fixed target2 30-seed (mozilla4)` best BGR 0.6761 vs uniform 0.6808 (W/L/T=11/18/1), `OpenML numeric external fixed target2 30-seed (pc1)` best BGR 0.9063 vs uniform 0.8836 (W/L/T=21/9/0), `OpenML numeric external fixed target2 30-seed (wdbc)` best BGR 0.9505 vs uniform 0.9530 (W/L/T=13/16/1), `OpenML numeric external fixed target2 replication 30-seed (banknote-authentication)` best BGR 0.7483 vs uniform 0.8153 (W/L/T=7/23/0), `OpenML numeric external fixed target2 replication 30-seed (climate-model-simulation-crashes)` best BGR 0.9065 vs uniform 0.8773 (W/L/T=26/4/0), `OpenML numeric external fixed target2 replication 30-seed (kc1)` best BGR 0.8181 vs uniform 0.7873 (W/L/T=18/12/0), `OpenML numeric external fixed target2 replication 30-seed (mozilla4)` best BGR 0.6633 vs uniform 0.6754 (W/L/T=14/16/0), `OpenML numeric external fixed target2 replication 30-seed (pc1)` best BGR 0.8918 vs uniform 0.8947 (W/L/T=14/15/1), `OpenML numeric external fixed target2 replication 30-seed (wdbc)` best BGR 0.9545 vs uniform 0.9442 (W/L/T=22/8/0).
- Superseded route scout(s): `OpenML margin replay (diabetes)` already has fixed positive follow-up evidence; no preregistration action remains.
- Positive pre-existing-dataset follow-up(s): `OpenML diabetes margin 30-seed (diabetes)` BGR 0.7062 vs uniform 0.6689 (W/L/T=24/6/0), fixed gap +0.0303, `OpenML diabetes margin replication 30-seed (diabetes)` BGR 0.7056 vs uniform 0.6673 (W/L/T=23/7/0), fixed gap +0.0416, `OpenML numeric external fixed target2 30-seed (blood-transfusion-service-center)` BGR 0.7625 vs uniform 0.6657 (W/L/T=30/0/0), fixed gap +0.0705, `OpenML numeric external fixed target2 30-seed (phoneme)` BGR 0.7228 vs uniform 0.6896 (W/L/T=21/9/0), fixed gap +0.0524, `OpenML numeric external fixed target2 replication 30-seed (blood-transfusion-service-center)` BGR 0.7595 vs uniform 0.6846 (W/L/T=25/5/0), fixed gap +0.0462, `OpenML numeric external fixed target2 replication 30-seed (phoneme)` BGR 0.7124 vs uniform 0.6758 (W/L/T=21/9/0), fixed gap +0.0332.

## Promotion Deficits

- Independent benchmark: no screen clears the 4/4 promotion screen. The closest screen, `MiniGrid FourRooms official-package` with `bgr_coverage`, clears 3/4 gates and fails on final_median_r80-ceiling-saturated.
- Learned policy: Perturb-only anchored OpenVLA audit does not clear the learned-policy promotion gate: BGR 371/400, official 367/400, random 372/400; identity BGR 99/100, official 99/100, random 99/100; official gap +4 (+0.0100), random gap -1 (-0.0025), clean deficit 0.
- Retired calibrated route(s): `MinAtar Breakout calibration`, `MinAtar Asterix calibration`, `Gymnasium MuJoCo Reacher-v5 calibration`, `Gymnasium MuJoCo InvertedPendulum-v5 calibration`, `Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration`, `Gymnasium Box2D LunarLander-v3 calibration` cleared pre-method calibration, but the corresponding fixed method screen is negative or tied; not active acceptance evidence.
- Rejected route scout(s): `sklearn digits margin replay`, `sklearn tabular margin replay (breast_cancer)`, `sklearn tabular margin replay (wine)`, `OpenML margin replay (ionosphere)`, `OpenML margin replay (sonar)`, `OpenML margin replay (spambase)`, `OpenML numeric external fixed target2 30-seed (banknote-authentication)`, `OpenML numeric external fixed target2 30-seed (climate-model-simulation-crashes)`, `OpenML numeric external fixed target2 30-seed (kc1)`, `OpenML numeric external fixed target2 30-seed (mozilla4)`, `OpenML numeric external fixed target2 30-seed (pc1)`, `OpenML numeric external fixed target2 30-seed (wdbc)`, `OpenML numeric external fixed target2 replication 30-seed (banknote-authentication)`, `OpenML numeric external fixed target2 replication 30-seed (climate-model-simulation-crashes)`, `OpenML numeric external fixed target2 replication 30-seed (kc1)`, `OpenML numeric external fixed target2 replication 30-seed (mozilla4)`, `OpenML numeric external fixed target2 replication 30-seed (pc1)`, `OpenML numeric external fixed target2 replication 30-seed (wdbc)` did not clear the +0.03 and 3/4 paired pre-registration screen; not active acceptance evidence.
- Superseded route scout(s): `OpenML margin replay (diabetes)` have fixed positive follow-up evidence; the scout itself is not an active action item.
- Positive pre-existing-dataset follow-up(s): `OpenML diabetes margin 30-seed (diabetes)`, `OpenML diabetes margin replication 30-seed (diabetes)`, `OpenML numeric external fixed target2 30-seed (blood-transfusion-service-center)`, `OpenML numeric external fixed target2 30-seed (phoneme)`, `OpenML numeric external fixed target2 replication 30-seed (blood-transfusion-service-center)`, `OpenML numeric external fixed target2 replication 30-seed (phoneme)` clear the internal 30-seed margin-replay follow-up gate and are incorporated into the paper as supervised margin-replay evidence.
- Active route: Occlusion-bottleneck OpenVLA route is preregistered, not yet evidence: the clean-plus-occlusion TFDS prep, proximal-anchor BGR/random adaptation, and official/BGR/random 10-task x 10-trial perturbation summaries must finish before the +10/400 and +0.02 learned-policy gate can be checked.

## Independent Benchmark Screens

| Screen | Treatment | Seeds | dRAUC vs uniform (W/L/T) | Worst required baseline dRAUC | Ablation dRAUC | Radius metric | Radius delta | Cleared gates | Screen gate | Failure reason(s) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| MiniGrid FourRooms official-package | bgr_coverage | 4 | +0.1075 (4/0/0) | +0.0464 (3/1/0) | +0.0071 (2/2/0) | final_median_r80 | +0.0000 (0/0/4) | 3/4 | fail | final_median_r80-ceiling-saturated |
| MiniGrid FourRooms abs-r10 follow-up | bgr_coverage | 4 | +0.1075 (4/0/0) | +0.0464 (3/1/0) | +0.0071 (2/2/0) | final_abs_r10 | +0.0000 (0/0/4) | 3/4 | fail | final_abs_r10-floor-saturated |
| Gymnasium Box2D LunarLander-v3 | bgr_coverage | 4 | +0.1278 (2/2/0) | +0.0125 (3/1/0) | +0.0340 (3/1/0) | final_median_r80 | -0.0625 (3/1/0) | 2/4 | fail | uniform-gate, final_median_r80-contradiction |
| Gymnasium MuJoCo InvertedDoublePendulum-v5 | bgr | 4 | +0.0799 (1/0/3) | +0.0833 (1/0/3) | +0.0833 (1/0/3) | final_median_r80 | -0.1900 (0/1/3) | 2/4 | fail | uniform-gate, final_median_r80-contradiction |
| MiniGrid FourRooms official-package | bgr | 4 | +0.0641 (3/1/0) | +0.0030 (3/1/0) | -0.0362 (1/3/0) | final_median_r80 | +0.0000 (0/0/4) | 2/4 | fail | state-priority-ablation, final_median_r80-ceiling-saturated |
| MiniGrid FourRooms abs-r10 follow-up | bgr | 4 | +0.0641 (3/1/0) | +0.0030 (3/1/0) | -0.0362 (1/3/0) | final_abs_r10 | +0.0000 (0/0/4) | 2/4 | fail | state-priority-ablation, final_abs_r10-floor-saturated |
| MinAtar Asterix | bgr_coverage | 4 | +0.0172 (1/2/1) | -0.0219 (0/4/0) | +0.0094 (3/1/0) | final_median_r80 | +0.5000 (1/0/3) | 2/4 | fail | uniform-gate, required-baseline |
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
| MinAtar Breakout | bgr | 4 | +0.0000 (0/0/4) | +0.0000 (0/0/4) | +0.0000 (0/0/4) | final_median_r80 | +0.0000 (0/0/4) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-ceiling-saturated |
| MinAtar Breakout | bgr_coverage | 4 | +0.0000 (0/0/4) | +0.0000 (0/0/4) | +0.0000 (0/0/4) | final_median_r80 | +0.0000 (0/0/4) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-ceiling-saturated |
| bsuite Cartpole | bgr | 4 | -0.0113 (0/4/0) | -0.0229 (0/4/0) | -0.0087 (1/3/0) | final_median_r80 | -0.0208 (1/1/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MinAtar Asterix | bgr | 4 | -0.0187 (1/3/0) | -0.0578 (0/4/0) | -0.0266 (1/3/0) | final_median_r80 | -0.1250 (2/2/0) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| bsuite Catch 30-seed | bgr_coverage | 30 | -0.0330 (13/17/0) | -0.1224 (4/26/0) | -0.0136 (15/15/0) | final_median_r80 | -0.0625 (6/11/13) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| bsuite Catch 30-seed | bgr | 30 | -0.0337 (14/16/0) | -0.1231 (3/24/3) | -0.0142 (20/10/0) | final_median_r80 | -0.0867 (5/9/16) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid FourRooms max-radius-10 follow-up | bgr | 4 | -0.0422 (1/3/0) | -0.0039 (1/3/0) | -0.0375 (1/3/0) | final_median_r80 | +0.0000 (0/0/4) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-ceiling-saturated |
| MiniGrid LavaGapS7 | bgr | 4 | -0.0430 (1/3/0) | -0.0063 (2/1/1) | -0.0596 (1/2/1) | final_abs_r10 | -0.0289 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |
| MiniGrid FourRooms midband | bgr_coverage | 4 | -0.0589 (2/2/0) | -0.1864 (0/4/0) | -0.0210 (3/1/0) | final_median_r80 | -0.0748 (0/4/0) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid DoorKey | bgr_coverage | 4 | -0.1538 (0/2/2) | -0.1613 (1/3/0) | -0.0093 (1/2/1) | final_abs_r10 | -0.1569 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |
| FetchReach-v4 hard-budget goal recovery | bgr | 4 | -0.2375 (0/3/1) | -0.5000 (0/4/0) | -0.0750 (1/2/1) | final_median_r80 | -0.0097 (0/3/1) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_median_r80-contradiction |
| MiniGrid DoorKey | bgr | 4 | -0.2697 (0/4/0) | -0.2772 (0/4/0) | -0.1252 (0/4/0) | final_abs_r10 | -0.2503 (0/2/2) | 0/4 | fail | uniform-gate, required-baseline, state-priority-ablation, final_abs_r10-contradiction |

## Route Scouts

| Scout | Best BGR target | BGR RAUC | Uniform RAUC | dRAUC vs uniform (W/L/T) | Best fixed-radius RAUC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sklearn digits margin replay | 1.0000 | 0.8271 | 0.8123 | +0.0148 (2/2/0) | 0.8425 @ 0.8000 | reject-scout |
| sklearn tabular margin replay (breast_cancer) | 2.0000 | 0.9610 | 0.9516 | +0.0094 (3/1/0) | 0.9566 @ 1.5000 | reject-scout |
| sklearn tabular margin replay (wine) | 0.5000 | 0.9702 | 0.9563 | +0.0139 (4/0/0) | 0.9586 @ 0.5000 | reject-scout |
| OpenML margin replay (diabetes) | 2.0000 | 0.7402 | 0.6797 | +0.0605 (4/0/0) | 0.6999 @ 2.0000 | superseded-by-positive-follow-up |
| OpenML margin replay (ionosphere) | 1.5000 | 0.8429 | 0.8338 | +0.0090 (2/2/0) | 0.8416 @ 2.0000 | reject-scout |
| OpenML margin replay (sonar) | 1.5000 | 0.7667 | 0.7378 | +0.0289 (4/0/0) | 0.7701 @ 2.0000 | reject-scout |
| OpenML margin replay (spambase) | 1.0000 | 0.8840 | 0.8602 | +0.0238 (3/1/0) | 0.8431 @ 2.0000 | reject-scout |
| OpenML diabetes margin 30-seed (diabetes) | 2.0000 | 0.7062 | 0.6689 | +0.0373 (24/6/0) | 0.6759 @ 2.0000 | positive-follow-up |
| OpenML diabetes margin replication 30-seed (diabetes) | 2.0000 | 0.7056 | 0.6673 | +0.0383 (23/7/0) | 0.6640 @ 2.0000 | positive-follow-up |
| OpenML numeric external fixed target2 30-seed (banknote-authentication) | 2.0000 | 0.7092 | 0.7864 | -0.0772 (4/26/0) | 0.7118 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 30-seed (blood-transfusion-service-center) | 2.0000 | 0.7625 | 0.6657 | +0.0968 (30/0/0) | 0.6920 @ 2.0000 | positive-follow-up |
| OpenML numeric external fixed target2 30-seed (climate-model-simulation-crashes) | 2.0000 | 0.9019 | 0.8796 | +0.0223 (25/5/0) | 0.8816 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 30-seed (kc1) | 2.0000 | 0.8145 | 0.8126 | +0.0020 (15/14/1) | 0.8106 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 30-seed (mozilla4) | 2.0000 | 0.6761 | 0.6808 | -0.0047 (11/18/1) | 0.6581 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 30-seed (pc1) | 2.0000 | 0.9063 | 0.8836 | +0.0227 (21/9/0) | 0.9128 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 30-seed (phoneme) | 2.0000 | 0.7228 | 0.6896 | +0.0332 (21/9/0) | 0.6704 @ 2.0000 | positive-follow-up |
| OpenML numeric external fixed target2 30-seed (wdbc) | 2.0000 | 0.9505 | 0.9530 | -0.0026 (13/16/1) | 0.9539 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (banknote-authentication) | 2.0000 | 0.7483 | 0.8153 | -0.0669 (7/23/0) | 0.7095 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (blood-transfusion-service-center) | 2.0000 | 0.7595 | 0.6846 | +0.0749 (25/5/0) | 0.7133 @ 2.0000 | positive-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (climate-model-simulation-crashes) | 2.0000 | 0.9065 | 0.8773 | +0.0292 (26/4/0) | 0.8810 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (kc1) | 2.0000 | 0.8181 | 0.7873 | +0.0307 (18/12/0) | 0.8061 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (mozilla4) | 2.0000 | 0.6633 | 0.6754 | -0.0121 (14/16/0) | 0.6368 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (pc1) | 2.0000 | 0.8918 | 0.8947 | -0.0029 (14/15/1) | 0.8966 @ 2.0000 | reject-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (phoneme) | 2.0000 | 0.7124 | 0.6758 | +0.0366 (21/9/0) | 0.6792 @ 2.0000 | positive-follow-up |
| OpenML numeric external fixed target2 replication 30-seed (wdbc) | 2.0000 | 0.9545 | 0.9442 | +0.0102 (22/8/0) | 0.9486 @ 2.0000 | reject-follow-up |

## Pre-Method Calibrations

| Calibration | Clean | Recovery range | Median r80 | Decision |
| --- | ---: | ---: | ---: | --- |
| FetchPush-v4 object-goal calibration | 0.2500 | 0.2500--0.2500 | 0.1200 | reject-calibration |
| FetchPush-v4 far-push object-goal calibration | 0.6250 | 0.6250--0.8750 | 0.1200 | reject-calibration |
| FetchSlide-v4 object-goal calibration | 0.2500 | 0.1250--0.2500 | 0.0720 | reject-calibration |
| FetchPickAndPlace-v4 object-goal calibration | 0.1250 | 0.0000--0.1250 | 0.0660 | reject-calibration |
| Gymnasium-Robotics HandReach-v3 calibration | 0.0000 | 0.0000--0.0000 | 0.2000 | reject-calibration |
| highway-env parking-v0 calibration | 0.3333 | 0.2500--0.5000 | 9.8000 | reject-calibration |
| highway-env highway-fast-v0 lane calibration | 0.6667 | 0.5833--0.6667 | 6.0000 | reject-calibration |
| MinAtar Breakout calibration | 1.0000 | 0.6667--1.0000 | 0.6000 | usable-calibration |
| MinAtar Asterix calibration | 0.8333 | 0.5000--0.8333 | 5.3333 | usable-calibration |
| Gymnasium MuJoCo Reacher-v5 calibration | 0.8333 | 0.5000--0.9167 | 3.0000 | usable-calibration |
| Gymnasium MuJoCo InvertedPendulum-v5 calibration | 1.0000 | 0.0000--1.0000 | 0.2100 | usable-calibration |
| Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration | 1.0000 | 0.0000--1.0000 | 0.2825 | usable-calibration |
| Gymnasium Box2D LunarLander-v3 calibration | 0.9167 | 0.5833--0.9167 | 0.5300 | usable-calibration |

## Priority Read

- The controlled grid mechanism is above its internal effect threshold, but it is still a constructed mechanism benchmark.
- The standard-environment recovery route still has not produced a promotable screen: early promising screens either fail paired/radius checks or do not survive scale-up, and later non-saturated screens trail uniform, stronger baselines, or the state-priority/uniform-radius ablation.
- The usable Reacher-v5 calibration is pre-method evidence only; the fixed full all-method comparison is now negative and should not be promoted.
- The InvertedPendulum-v5 calibration also cleared pre-method checks, but its fixed 4-seed method screen ties all methods on final RAUC and median-r80; do not scale or promote it.
- Rejected pre-method calibrations should not be scaled into BGR comparisons until the reset interface and controller first produce clean, non-saturated recovery curves.
- The InvertedDoublePendulum-v5 calibration cleared pre-method checks, but its fixed 4-seed method screen collapses clean success; the small BGR RAUC edge is not acceptance evidence.
- `OpenML diabetes margin 30-seed (diabetes)`, `OpenML diabetes margin replication 30-seed (diabetes)`, `OpenML numeric external fixed target2 30-seed (blood-transfusion-service-center)`, `OpenML numeric external fixed target2 30-seed (phoneme)`, `OpenML numeric external fixed target2 replication 30-seed (blood-transfusion-service-center)`, `OpenML numeric external fixed target2 replication 30-seed (phoneme)` now give a replicated positive pre-existing-dataset signal, but it must be framed as supervised margin-replay evidence rather than robotics or standard-environment recovery.
- `OpenML margin replay (diabetes)` is superseded by fixed positive OpenML diabetes follow-ups, so it is no longer a pending preregistration item.
- Most pre-existing-dataset route scouts are rejected before preregistration: their best BGR rows stay below the +0.03 screen even when paired signs are favorable, or fixed-radius replay is competitive.
- Perturb-only anchored OpenVLA audit does not clear the learned-policy promotion gate: BGR 371/400, official 367/400, random 372/400; identity BGR 99/100, official 99/100, random 99/100; official gap +4 (+0.0100), random gap -1 (-0.0025), clean deficit 0.
- Current acceptance-moving learned-policy work: Occlusion-bottleneck OpenVLA route is preregistered, not yet evidence: the clean-plus-occlusion TFDS prep, proximal-anchor BGR/random adaptation, and official/BGR/random 10-task x 10-trial perturbation summaries must finish before the +10/400 and +0.02 learned-policy gate can be checked.
- Do not start another same-protocol MiniGrid, classic-control, PointMaze, or FetchReach screen while this is pending; existing screens already show saturated radius checks, stronger-baseline losses, or state-priority-only ablation failures.
