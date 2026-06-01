import unittest

from bgr.experiments.suffix import run_method


class SuffixStrategyTest(unittest.TestCase):
    def test_bgr_suffix_strategy_variants_smoke(self):
        config = {
            "experiment": {
                "num_tasks": 5,
                "suffixes_per_task": 2,
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
        for method in ["bgr_boundary", "bgr_broad", "bgr_hard"]:
            result = run_method(config, method, seed=0)
            self.assertEqual(result.method, method)
            self.assertGreaterEqual(result.final_rauc, 0.0)


if __name__ == "__main__":
    unittest.main()
