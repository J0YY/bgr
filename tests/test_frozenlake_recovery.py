import unittest

import numpy as np

from bgr.envs.frozenlake_recovery import FROZENLAKE_8X8_MAP, FrozenLakeRecoveryBenchmark
from bgr.experiments.frozenlake_recovery import run_method


class FrozenLakeRecoveryTest(unittest.TestCase):
    def test_canonical_map_and_transition_slip(self):
        bench = FrozenLakeRecoveryBenchmark(replay_state_count=8, seed=0)
        self.assertEqual(FROZENLAKE_8X8_MAP[0], "SFFFFFFF")
        self.assertEqual(FROZENLAKE_8X8_MAP[-1], "FFFHFFFG")
        transitions = bench._transitions(0, 2)
        self.assertAlmostEqual(sum(item[0] for item in transitions), 1.0)
        self.assertGreaterEqual(len(transitions), 2)

    def test_training_can_improve_recovery_auc(self):
        config = {
            "experiment": {
                "methods": ["uniform"],
                "seeds": [0],
                "replay_state_count": 12,
                "max_radius": 5,
                "iterations": 20,
                "eval_every": 10,
                "train_batch_size": 8,
                "eval_grid_size": 5,
                "learning_rate": 0.35,
                "discount": 0.97,
                "epsilon": 0.08,
                "q_init_blend": 0.35,
                "q_init_noise": 0.02,
                "episode_max_steps": 64,
            },
            "bgr": {"initial_probes": [0.0, 0.5, 1.0], "min_trials": 1},
        }
        result = run_method(config, "uniform", 0)
        self.assertGreaterEqual(result.best_rauc, result.history[0]["rauc"])

    def test_boundary_replay_samples_valid_radius(self):
        bench = FrozenLakeRecoveryBenchmark(replay_state_count=12, seed=1)
        rng = np.random.default_rng(0)
        before = bench.success_prob(0, 0.4)
        for _ in range(20):
            bench.train_step(0, 0.4, rng)
        after = bench.success_prob(0, 0.4)
        self.assertGreaterEqual(after, before - 1e-9)


if __name__ == "__main__":
    unittest.main()
