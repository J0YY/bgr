import unittest

from bgr.experiments.grid_margin import run_method


class GridMarginAblationTest(unittest.TestCase):
    def test_bgr_ablation_variants_smoke(self):
        config = {
            "experiment": {
                "num_tasks": 6,
                "grid_size": 7,
                "obstacle_prob": 0.1,
                "replay_states_per_task": 2,
                "max_offset": 3,
                "iterations": 2,
                "eval_every": 1,
                "train_batch_size": 4,
                "eval_grid_size": 5,
                "alpha": 0.8,
                "learning_rate": 0.02,
                "target_margin": 0.35,
            },
            "bgr": {
                "initial_probes": [0.0, 0.5, 1.0],
                "refresh_per_eval": 4,
                "min_trials": 1,
                "radius_noise": 0.05,
            },
        }
        for method in ["bgr_no_uncertainty", "bgr_no_sharpness", "bgr_uniform_radius"]:
            result = run_method(config, method, seed=0)
            self.assertEqual(result.method, method)
            self.assertGreaterEqual(result.final_rauc, 0.0)


if __name__ == "__main__":
    unittest.main()
