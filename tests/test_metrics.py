import numpy as np
import unittest

from bgr.metrics import critical_radius, recovery_auc


class MetricsTest(unittest.TestCase):
    def test_recovery_auc_linear_curve(self):
        sigmas = np.array([0.0, 0.5, 1.0])
        recovery = np.array([1.0, 0.5, 0.0])
        self.assertEqual(recovery_auc(sigmas, recovery, sigma_max=1.0), 0.5)

    def test_critical_radius_interpolates_crossing(self):
        sigmas = np.array([0.0, 0.5, 1.0])
        recovery = np.array([1.0, 0.7, 0.2])
        self.assertTrue(np.isclose(critical_radius(sigmas, recovery, alpha=0.8), 1.0 / 3.0))
