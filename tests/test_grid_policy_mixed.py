import unittest

from bgr.experiments.grid import run_method


class GridPolicyMixedTest(unittest.TestCase):
    def test_bgr_mixed_smoke(self):
        config = {
            "experiment": {
                "num_tasks": 4,
                "grid_size": 7,
                "obstacle_prob": 0.1,
                "replay_states_per_task": 2,
                "max_offset": 3,
                "horizon": 18,
                "iterations": 2,
                "eval_every": 1,
                "train_batch_size": 4,
                "eval_trials_per_radius": 1,
                "eval_grid_size": 5,
                "alpha": 0.8,
                "learning_rate": 0.3,
                "clean_pretrain_steps": 1,
                "target_margin": 0.35,
            },
            "bgr": {
                "initial_probes": [0.0, 0.5, 1.0],
                "refresh_per_eval": 4,
                "min_trials": 1,
                "radius_noise": 0.05,
            },
        }
        result = run_method(config, "bgr_mixed", seed=0)
        self.assertEqual(result.method, "bgr_mixed")
        self.assertGreaterEqual(result.final_rauc, 0.0)


if __name__ == "__main__":
    unittest.main()
