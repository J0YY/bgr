import unittest

from bgr.experiments.estimator import run_method


class EstimatorExperimentTest(unittest.TestCase):
    def test_active_estimator_smoke(self):
        config = {
            "experiment": {
                "seeds": [0],
                "methods": ["active"],
                "num_states": 8,
                "probes_per_state": 6,
                "sigma_max": 1.0,
                "true_grid_size": 31,
                "alpha": 0.8,
                "hit_tolerance": 0.1,
            },
            "active": {"initial_probes": [0.0, 0.5, 1.0], "jitter": 0.05},
        }
        result = run_method(config, "active", 0)
        self.assertEqual(result.method, "active")
        self.assertEqual(result.probes_per_state, 6)
        self.assertGreaterEqual(result.boundary_hit_rate, 0.0)
        self.assertLessEqual(result.boundary_hit_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
