import numpy as np
import unittest

from bgr.curve_estimators import IsotonicCurveEstimator


class CurveEstimatorTest(unittest.TestCase):
    def test_isotonic_estimator_returns_nonincreasing_curve(self):
        estimator = IsotonicCurveEstimator(sigma_max=1.0, alpha=0.8)
        estimator.update(0.0, successes=7, trials=10)
        estimator.update(0.5, successes=9, trials=10)
        estimator.update(1.0, successes=1, trials=10)
        estimate = estimator.fit()
        self.assertTrue(np.all(np.diff(estimate.recovery) <= 1e-9))
        self.assertTrue(0.0 <= estimate.r_alpha <= 1.0)

    def test_next_probe_is_in_range(self):
        rng = np.random.default_rng(0)
        estimator = IsotonicCurveEstimator(sigma_max=1.0, alpha=0.8)
        estimator.update(0.0, successes=10, trials=10)
        estimator.update(1.0, successes=0, trials=10)
        for _ in range(20):
            sigma = estimator.next_probe(rng)
            self.assertTrue(0.0 <= sigma <= 1.0)
