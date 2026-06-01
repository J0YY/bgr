import unittest

import numpy as np

from bgr.envs.grid_recovery import GridMarginRecoveryBenchmark


class GridMarginRecoveryTest(unittest.TestCase):
    def test_training_near_margin_expands_recovery(self):
        rng = np.random.default_rng(0)
        bench = GridMarginRecoveryBenchmark(
            num_tasks=4,
            grid_size=7,
            obstacle_prob=0.12,
            replay_states_per_task=2,
            max_offset=4,
            learning_rate=0.05,
            seed=0,
        )
        replay_idx = 0
        sigma = bench.states[replay_idx].margin
        before = bench.success_prob(replay_idx, sigma + 0.1)
        for _ in range(12):
            bench.train_step(replay_idx, sigma, rng)
        after = bench.success_prob(replay_idx, sigma + 0.1)
        self.assertGreater(after, before)

    def test_feasibility_decays_beyond_local_radius(self):
        bench = GridMarginRecoveryBenchmark(
            num_tasks=4,
            grid_size=7,
            obstacle_prob=0.12,
            replay_states_per_task=2,
            max_offset=4,
            learning_rate=0.05,
            seed=1,
        )
        replay_idx = next(
            idx for idx, state in enumerate(bench.states) if state.feasible_radius < 0.95
        )
        radius = bench.states[replay_idx].feasible_radius
        self.assertEqual(bench.feasibility(replay_idx, radius), 1.0)
        self.assertLess(bench.feasibility(replay_idx, min(1.0, radius + 0.3)), 1.0)
