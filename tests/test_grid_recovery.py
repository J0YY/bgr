import unittest

import numpy as np

from bgr.envs.grid_recovery import GridRecoveryBenchmark, oracle_action


class GridRecoveryTest(unittest.TestCase):
    def test_generates_replayable_states_with_oracle_actions(self):
        bench = GridRecoveryBenchmark(
            num_tasks=4,
            grid_size=7,
            obstacle_prob=0.12,
            replay_states_per_task=3,
            max_offset=3,
            horizon=20,
            learning_rate=1.0,
            seed=0,
        )
        self.assertEqual(len(bench.replay_states), 12)
        for replay in bench.replay_states:
            task = bench.tasks[replay.task_id]
            self.assertIsNotNone(oracle_action(task, replay.position))

    def test_training_oracle_step_improves_clean_rollout(self):
        rng = np.random.default_rng(1)
        bench = GridRecoveryBenchmark(
            num_tasks=2,
            grid_size=7,
            obstacle_prob=0.12,
            replay_states_per_task=2,
            max_offset=3,
            horizon=20,
            learning_rate=1.0,
            seed=1,
        )
        replay_idx = 0
        before = sum(bench.rollout(replay_idx, 0.0, rng) for _ in range(20))
        for _ in range(20):
            bench.train_step(replay_idx, 0.0, rng)
        after = sum(bench.rollout(replay_idx, 0.0, rng) for _ in range(20))
        self.assertGreaterEqual(after, before)
