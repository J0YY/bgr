import unittest

import numpy as np

from bgr.envs.robot_suffix import RobotSuffixRecoveryBenchmark


class RobotSuffixRecoveryTest(unittest.TestCase):
    def test_boundary_training_expands_object_margin(self):
        rng = np.random.default_rng(0)
        bench = RobotSuffixRecoveryBenchmark(num_tasks=4, suffixes_per_task=2, learning_rate=0.05, seed=0)
        state_idx = 0
        sigma = bench.states[state_idx].margin_object
        before = bench.success_prob(state_idx, sigma + 0.1, "object")
        for _ in range(10):
            bench.train_step(state_idx, sigma, rng, "object")
        after = bench.success_prob(state_idx, sigma + 0.1, "object")
        self.assertGreater(after, before)

    def test_object_training_transfers_somewhat_to_ee_offsets(self):
        rng = np.random.default_rng(1)
        bench = RobotSuffixRecoveryBenchmark(num_tasks=4, suffixes_per_task=2, learning_rate=0.05, seed=1)
        state_idx = 0
        sigma = bench.states[state_idx].margin_object
        before = bench.success_prob(state_idx, sigma, "ee")
        for _ in range(8):
            bench.train_step(state_idx, sigma, rng, "object")
        after = bench.success_prob(state_idx, sigma, "ee")
        self.assertGreaterEqual(after, before)
